/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include "rt_external.h"
#include "runtime/rt_error_codes.h"
#include "comm.h"
#include <fcntl.h>
#include "llt_hccl_stub.h"
#include "tsd/tsd_client.h"
#define private public
#undef private
#include "hccl_impl.h"
#include <sys/epoll.h>
#include "ascend_hal.h"
// #ifdef __cplusplus
// extern "C" {
// #endif

#define container_of(ptr, type, member) ((type*)((char*)(1 ? (ptr) : &((type*)0)->member) - offsetof(type, member)))

void* event_process(void* p_cn);

#define DEV_MAX 16
#define CONN_MAX 64 // 每个device上的最大socket连接数

#define SG_LIST_MAX 512   // SG_LIST 最大数量
#define DEV_MAX_NODES 16  // 节点间最大的device数量
#define CONN_MAX_NODES 16 // 节点间最大的连接数
#define SERVER_IP_MAX_VALUE 16
#define LINK_LOCAL_ROLE_SERVER 1
#define LINK_LOCAL_ROLE_CLIENT 0
#define RA_MAX_INSTANCES 16
constexpr u32 MAX_MSG_STR_LEN = 2 * 1024;
constexpr u32 SOCKET_VNIC_IP_INFOS_INTERFACE = 55;
constexpr u32 GET_NOTIFY_BA = 14;
#define RA_CHECK_POINTER_NULL_WITH_RET(ptr) \
    do {                                    \
        if ((ptr) == NULL) {                \
            HCCL_ERROR("pointer is NULL!"); \
            return (-EINVAL);               \
        }                                   \
    } while (0)

u64 NON_ZERO_BIT_INDEX(u64 bitmap)
{
    // 找到bitmap中从bit0往bit31数, 第一个为1的bit位置
    u64 bit_index = 0;
    while (bit_index < 64) {
        if (bitmap & ((u64)(1ULL << bit_index))) {
            break;
        }

        bit_index++;
    }

    return bit_index;
}

u32 RESERVED_LOOP_IP(u32 ip_le) { return (((ip_le >> 24) & 0xFF) < 0x10) ? 1 : 0; }

enum conn_role_e { CONN_SERVER = 0, CONN_CLIENT };

enum work_mode_e { MODE_PID_AS_SERVER = 0, MODE_PID_AS_NORMAL };

typedef struct {
    int ref_count;
    pthread_mutex_t mutex;
} ra_instance;

static s32 work_mode = MODE_PID_AS_SERVER; // 工作模式: PID模拟server还是正常
static struct cn_info cn[QP_MAX];          // 暂时不支持多线程*/
// static sal_mutex_t  mutex = NULL;
static std::mutex g_qpMutex;        // 默认构造函数构造此全局mutex
static std::mutex g_sglistMutex;    // 默认构造函数构造此全局mutex
static std::mutex g_socketShmMutex; // 默认构造函数构造此全局mutex
static u32 hccpThreadStatus = 0;    // hccp线程状态（0：close 1：open）
__thread s32 qp_index;
__thread s32 thread_entry_times = 0;

u32 listen_num[DEV_MAX] = {0};
u32 listen_flag[DEV_MAX] = {0}; // 记录当前进程是否已经成功启动监听
sal_thread_t listen_thread[DEV_MAX] = {NULL};
u64 listen_done[DEV_MAX] = {0};
void* hccl_shm_server_ptr[DEV_MAX] = {NULL}; // socket server 相关共享内存名字
void* hccl_shm_client_ptr[DEV_MAX] = {NULL}; // socket client 相关共享内存名字
u32 bind_port = 6382;
u32 listen_num_nodes[DEV_MAX_NODES] = {0};
u32 listen_flag_nodes[DEV_MAX_NODES] = {0}; // 记录当前进程是否已经成功启动监听
sal_thread_t listen_thread_nodes[DEV_MAX_NODES] = {NULL};
u64 listen_done_nodes[DEV_MAX_NODES] = {0};
u32 bind_port_nodes = 6383;
u32 g_client_ip_nodes[DEV_MAX_NODES] = {0};
char g_shm_mpi_name[64] = {0};
u32 g_test_type = 0; // 0为常规测试项，1为MPI测试
u32 bind_check_link_port = 6363;
u32 g_client_ip_check_nodes[DEV_MAX] = {0};
sal_thread_t checkListenThread[DEV_MAX];
struct check_link_socket linkServerCheckSocket[DEV_MAX];
struct check_link_socket linkClientCheckSocket[DEV_MAX];
s32 g_check_listen_fd[DEV_MAX] = {0};
ra_instance g_ref_instances[RA_MAX_INSTANCES] = {{0, PTHREAD_MUTEX_INITIALIZER}};

void ra_set_test_type(u32 type, const char* name)
{
    g_test_type = type;
    sal_strncpy(g_shm_mpi_name, 64 - 1, name, SalStrLen(name));
    g_shm_mpi_name[63] = 0;
}
struct SgList g_sg_list[SG_LIST_MAX] = {0};

struct qp_socket_info {
    u32 server_ip;
    u32 client_ip;
    char tag[128];
};

typedef struct thread_para_t {
    s32 listenfd;
    s32 device_id;
    u32 ipAddr;
    u32 localIp;
    u32 count;
    sal_sem_t sem;
} thread_para;

struct connect_info_t {
    int listenfd;
    int conn_fd; // 不区分client/server, 描述用于通信的fd即可
    u32 client_ip;
    u32 server_ip;
    u32 status;

    // 保存全局变量对应的role, device_id和tag, 在close时使用
    u32 device_id;
    u32 role; // 0==SERVER, 1==CLIENT
    u64 key;  // 记录connetion在map中的key
    char tag[SOCK_CONN_TAG_SIZE + 1];
};

#if 1
/** HCCL的LLT场景, 以进程模拟节点, 线程模拟节点内的dev */
class Connection {
public:
    Connection() {}
    ~Connection() { socket_.clear(); }

    u64 set_conn(const char* tag, struct connect_info_t& conn)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::string tag_string(tag);

        u64 num = socket_[tag_string].size();
        HCCL_DEBUG(
            "tag[%s], current_num[%llu], server_ip[0x%x] client_ip[0x%x]", tag, num, conn.server_ip, conn.client_ip);

        conn.key = num;
        socket_[tag_string][num] = conn;
        bmp_got_[tag_string] |= ((u64)1 << num);

        return num;
    }

    struct connect_info_t* get_conn(const char* tag, u32 server_ip)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::string tag_string(tag);
        u64 bmp_avalaible = bmp_got_[tag_string] ^ bmp_given_[tag_string];

        HCCL_DEBUG(
            "tag[%s], server_ip[%08x], got[0x%016x], given[0x%016x]", tag, server_ip, bmp_got_[tag_string],
            bmp_given_[tag_string]);

        while (bmp_avalaible != 0) {
            u64 index = NON_ZERO_BIT_INDEX(bmp_avalaible);
            if (socket_[tag_string][index].server_ip == server_ip) {
                bmp_given_[tag_string] |= ((u64)1 << index);

                return &(socket_[tag_string][index]);
            }

            bmp_avalaible &= (~((u64)1 << index));
        }

        return NULL;
    }

    void del_conn(const char* tag, const struct connect_info_t& conn)
    {
        std::unique_lock<std::mutex> lock(mutex_);
        std::string tag_string(tag);

        HCCL_INFO("erase connection, tag[%s], key[%llu]", tag, conn.key);

        socket_[tag_string].erase(conn.key);

        // 如果map的size为0, 则删除该tag对应的socket_簇和bitmap_
        if (socket_[tag_string].size() == 0) {
            HCCL_INFO("erase empty map, tag[%s]", tag_string.c_str());

            bmp_got_.erase(tag_string);
            bmp_given_.erase(tag_string);
            socket_.erase(tag_string);
        }
    }

    void clear(const char* tag) { socket_[std::string(tag)].clear(); }

protected:
private:
    std::map<std::string, u64> bmp_got_;
    std::map<std::string, u64> bmp_given_;
    std::map<std::string, std::map<u64, struct connect_info_t>> socket_;
    std::mutex mutex_;
};

// Connection的默认构造函数会在main函数前被调用
static std::array<Connection, DEV_MAX> g_conn_server;
static std::array<Connection, DEV_MAX> g_conn_client;

// Connection的默认构造函数会在main函数前被调用
static std::array<Connection, DEV_MAX> g_conn_server_nodes;
static std::array<Connection, DEV_MAX> g_conn_client_nodes;

class Listener {
public:
    Listener() : listen_num(0), listen_flag(0), listen_thread(nullptr), listen_done(0) {}
    ~Listener() {}

protected:
private:
    u32 listen_num;
    u32 listen_flag;
    sal_thread_t listen_thread;
    u64 listen_done;
};

// static std::array<Listener, DEV_MAX> g_listener;
static std::array<int, DEV_MAX> g_listen_fd = {0, 0, 0, 0, 0, 0, 0, 0};
static std::array<int, DEV_MAX> g_listen_fd_nodes = {0, 0, 0, 0, 0, 0, 0, 0}; // 节点间监听句柄

struct client_info_t {
    u32 client_ip;
    char tag[SOCK_CONN_TAG_SIZE + 1];
};
#endif

char g_shm_name[64] = {0};

void ra_set_shm_name(const char* name)
{
    sal_strncpy(g_shm_name, 64 - 1, name, SalStrLen(name));
    // printf("ra_set_shm_name %s\n", name);
    g_shm_name[63] = 0;
}

int dev_flag[DEV_MAX] = {0};

void setTargetPort(u16 port_nodes_number, u16 port_number)
{
    bind_port_nodes = port_nodes_number;
    bind_port = port_number;
}

/*
 * 其中qp_mode:
 * 0---- 普通QP
 * 1---- gdr QP，wqe tmp & ts post DB
 * 2---- gdr QP，use common sq & ts post DB
 * 桩函数中，只有异步概率，没有下沉概率，不用管qp_mode
 */
int g_qp_mode = 0;
int RaQpCreate(void* rdma_handle, int flag, int qp_mode, void** qpHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    if (GetInfoFromHandle(rdma_handle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle is null");
        return -1;
    };
    s32 qp_cnt = 0;
    {
        /** 此处可能会与并发，加锁 */
        std::unique_lock<std::mutex> lock(g_qpMutex);
        g_qp_mode = qp_mode;
        for (; qp_cnt < QP_MAX; qp_cnt++) {
            /*cn为线程资源，暂不考虑竞争问题*/
            if (!cn[qp_cnt].set_flag) {
                break;
            }
        }
        HCCL_INFO("TMP_han qp_cnt[%d] QP_MAX[%d] qp_mode[%d]", qp_cnt, QP_MAX, g_qp_mode);
        if (qp_cnt == QP_MAX) {
            HCCL_ERROR("no available qp");
            return -1;
        }

        sal_memset(&cn[qp_cnt], sizeof(struct cn_info), 0, sizeof(struct cn_info));
        flag = flag; /*连接类型  桩函数使用共享内存通信，故无视*/

        // cn[qp_cnt].localIp = localIpAddr;
        cn[qp_cnt].local_port = -1;
        cn[qp_cnt].qpn = qp_cnt;
        cn[qp_cnt].dev_id = idx;
        cn[qp_cnt].qpMode = qp_mode;
        // 引用计数自增
        dev_flag[idx] += 1;

        // 启动后台任务,检视对方发动的指令
        char thread_name[128] = {0};
        if (-1
            == snprintf_s(
                thread_name, sizeof(thread_name), SalStrLen("hccl-gdr-stub_thread") + 3 + 10 + 2 + 2 + 1 + 64 + 3,
                //"%s-%10u-%02d-%02d",
                "%s-%d-%02d-%s-%d", "hccl-gdr-stub_thread",
                // localIpAddr,
                idx, cn[qp_cnt].qpn, g_shm_name, dev_flag[idx])) {
            HCCL_ERROR("thread name construct error");
            return -1;
        }

        HCCL_INFO("qp_cnt=%d, thread_name=%s", qp_cnt, thread_name);
        cn[qp_cnt].qp.thread_id = sal_thread_create(thread_name, event_process, &cn[qp_cnt]);
        if (NULL == cn[qp_cnt].qp.thread_id) {
            // 任务启动失败
            HCCL_ERROR("Create Thread failed");
            cn[qp_cnt].qp.thread_id = NULL;
            return -1;
        }

        cn[qp_cnt].set_flag = true;
        cn[qp_cnt].thread_run_flag = true;
        qp_index = qp_cnt; // 记录本次qp索引，后面按照索引查找对应的qp信息

        /** 释放锁 */
    }

    /** 将创建shm的动作放到create_qp里, 创建shm时必须等到对端也创建好, 才能返回
        否则可能对端还没创建好, 本端访问空指针 */

    // 查找TID MAP, 找到自己的server_ip作为shm_name的一部分
    u32 server_ip = 0;
    u32 rankId_server = 0;
    if (GetDevicePlaneId(device_id, rankId_server)
        != HCCL_SUCCESS) { // 如果此device没有设置网络平面，则默认按照deviceID
        rankId_server = device_id;
    }
    char shm_name[128] = {0};
    ++thread_entry_times;
    if (-1
        == snprintf_s(
            shm_name, sizeof(shm_name), SalStrLen("hccl-gdr-shm-stub") + 8 * 3 + 1 + 1 + 1 + 64 + 3,
            "%s-%08x-%08x-%08x-%s-%d", "hccl-gdr-shm-stub", rankId_server, server_ip, thread_entry_times, g_shm_name,
            0)) {
        HCCL_ERROR("shm name construct error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    HCCL_INFO(
        "###shm name[%s], qpcnt:[%d], pid:[%d], tid:[%d], entrytime:[%d]", shm_name, qp_cnt, SalGetPid(), SalGetTid(),
        thread_entry_times);

    struct qp_msg* qp_shm
        = (struct qp_msg*)sal_share_memory_create(shm_name, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));
    if (NULL == qp_shm) {
        HCCL_ERROR("shm allocate error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    // 判断shm是否被打开过, 进而确定该group是否被占用
    // share_mem_t* shm = container_of(ptr, share_mem_t, user_data);
    // HCCL_INFO("shm->ref_cnt=%d", shm->ref_cnt);

    /** 创建访问redis的互斥锁 */
    u64 ref_cnt = __sync_fetch_and_add(&(qp_shm->ref_cnt), 1);
    HCCL_INFO("ref_cnt[%d]", ref_cnt);

    /*shm 只允许两端连接*/
    if (ref_cnt == 0) {
        /** 第一个竞争胜利者 */
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;

        // mali added on Mar.2nd, 2019 for QP connectiion complete
        u64 try_count = 200;
        while (cn[qp_cnt].qp_shm->ref_cnt < 2 && try_count > 0) {
            HCCL_INFO("wait for peer up, qp_shm->ref_cnt[%lu],shm_name:[%s]", cn[qp_cnt].qp_shm->ref_cnt, shm_name);
            SaluSleep(100000);
            try_count--;
        }
        if (try_count == 0) {
            HCCL_ERROR("wait for peer up timeout,shm_name[%s]", shm_name);
            (void)sal_share_memory_destroy(qp_shm);
            (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
            return -1;
        }
    } else if (ref_cnt == 1) {
        /** 后续的访问者 */
        HCCL_INFO("peer ok,shm_name:[%s]", shm_name);
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist,ref_cnt=%lu,shm_name:[%s]", ref_cnt, shm_name);
        (void)__sync_fetch_and_sub(&(qp_shm->ref_cnt), 1);

        (void)sal_share_memory_destroy(qp_shm);
        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }
    *qpHandle = &cn[qp_cnt];
    return 0;
}

int RaGetQpContext(void* qpHandle, void** qp, void** sendCq, void** recvCq) { return 0; }

int RaQpDestroy(void* handle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    int ret = 0;
    if (handle == NULL) {
        HCCL_ERROR("handle is NULL");
        return -1;
    }

    struct cn_info* p_cn = (struct cn_info*)handle;
    if (!p_cn->set_flag) {
        HCCL_ERROR("flag not set");
        return -1;
    }

    if (p_cn->qp.thread_id != NULL) {
        p_cn->thread_run_flag = false;

        while (sal_thread_is_running(p_cn->qp.thread_id)) {
            SaluSleep(1000 * 10); /* 等待 10 ms, 确保线程已经退出 */
        }
    }

    if (p_cn->qp.shm_msg_ptr != NULL) {
        (void)__sync_fetch_and_sub(&(p_cn->qp_shm->ref_cnt), 1);
        share_mem_t* shareMemPtr = (share_mem_t*)((char*)(p_cn->qp_shm) - offsetof(share_mem_t, user_data));

        sal_share_memory_destroy(p_cn->qp.shm_msg_ptr);
    }

    // 目前只用单网卡 :
    HCCL_INFO("set qp handle flag false!!!!!!!!!!!!!!!!!!!!!!!!");
    p_cn->set_flag = false;

    // 销毁设备ID的引用计数
    dev_flag[p_cn->dev_id] = 0;

    return ret;
}

int RaGetTsqpDepth(void* rdev_handle, unsigned int* temp_depth, unsigned int* qp_num)
{
    *temp_depth = 1;
    *qp_num = 1;

    return 0;
}

int RaSetTsqpDepth(void* rdev_handle, unsigned int temp_depth, unsigned int* qp_num) { return 0; }

int RaQpConnectAsync(void* handle, const void* sock_handle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (handle == NULL) {
        HCCL_ERROR("QP handle NULL");
        return -1;
    }

    // 目前节点间的socket在桩函数里就是为空
#if 0
    if (sock_handle == NULL) {
        HCCL_ERROR("socket handle NULL");
        return -1;
    }
#endif

    struct cn_info* p_cn = (struct cn_info*)handle;

    if (!p_cn->set_flag) {
        HCCL_ERROR("not available qp");
        return -1;
    }

    return 0;
}

int RaGetQpStatus(void* handle, int* status)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    *status = 1;

    return 0;
}

int ra_bind(void* handle, u32 ipAddr, u16 port, u64 timeout_ms)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (handle == NULL) {
        return -1;
    }

    struct cn_info* p_cn = (struct cn_info*)handle;

    if (!p_cn->set_flag) {
        HCCL_ERROR("not available qp");
        return -1;
    }

    p_cn->server_ip = ipAddr;
    p_cn->server_port = port; /*使用共享内存通信，不存在端口号冲突问题*/

    char hccl_gdr_stub[128] = {0};
    if (-1
        == snprintf_s(
            hccl_gdr_stub, sizeof(hccl_gdr_stub), SalStrLen("hccl-gdr-stub_shm") + 3 + 10 + 8 + 2 + 1,
            "%s-%10u-%08u-%02d", "hccl-gdr-stub_shm", p_cn->server_ip, p_cn->server_port, p_cn->qpn)) {
        HCCL_ERROR("shm name construct error");
        return -1;
    } else {
        HCCL_INFO("shm name:%s", hccl_gdr_stub);
    }

    void* ptr = sal_share_memory_create(hccl_gdr_stub, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));

    if (NULL == ptr) {
        HCCL_ERROR("shm allocate error");
        return -2;
    }

    // 判断shm是否被打开过, 进而确定该group是否被占用
    share_mem_t* shm = container_of(ptr, share_mem_t, user_data);

    if (shm->ref_cnt <= 2) /*shm 只允许两端连接*/
    {
        HCCL_INFO("shm valid, success!");
        p_cn->qp.local_qp_msg_ptr = (struct qp_msg*)ptr; /*server排在第一位，client排在第二位*/
        p_cn->qp.remote_qp_msg_ptr = (struct qp_msg*)((s8*)ptr + sizeof(struct qp_msg));
        p_cn->qp.shm_msg_ptr = ptr;
        return 0;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist");
        sal_share_memory_destroy(ptr);
        return -2;
    }
}

int ra_connect(void* handle, u32 ipAddr, u16 port, u64 timeout_ms)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (handle == NULL)
        return -1;
    struct cn_info* p_cn = (struct cn_info*)handle;

    if (!p_cn->set_flag) {
        HCCL_ERROR("not available qp");
        return -1;
    }

    p_cn->server_ip = ipAddr;
    p_cn->server_port = port;

    char hccl_gdr_stub[128] = {0};
    if (-1
        == snprintf_s(
            hccl_gdr_stub, sizeof(hccl_gdr_stub), SalStrLen("hccl-gdr-stub_shm") + 3 + 10 + 8 + 2 + 1,
            "%s-%10u-%08u-%02d", "hccl-gdr-stub_shm", ipAddr, port, p_cn->qpn)) {
        HCCL_ERROR("shm name construct error");
        return -1;
    } else {
        HCCL_INFO("shm name:%s", hccl_gdr_stub);
    }

    void* ptr = sal_share_memory_create(hccl_gdr_stub, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));

    if (NULL == ptr) {
        HCCL_ERROR("shm allocate error");
        return -2;
    }

    // 判断shm是否被打开过, 进而确定该group是否被占用
    share_mem_t* shm = container_of(ptr, share_mem_t, user_data);
    if (shm->ref_cnt <= 2) /*shm 只允许两端连接*/
    {
        HCCL_INFO("shm valid, success!");
        p_cn->qp.remote_qp_msg_ptr = (struct qp_msg*)ptr; /*server排在第一位，client排在第二位*/
        p_cn->qp.local_qp_msg_ptr = (struct qp_msg*)((s8*)ptr + sizeof(struct qp_msg));
        p_cn->qp.shm_msg_ptr = ptr;
        return 0;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist");
        sal_share_memory_destroy(ptr);
        return -2;
    }
}

int RaMrReg(void* handle, struct MrInfoT* mrInfo)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return ((handle == NULL) || (mrInfo == NULL)) ? -1 : 0;
}

int RaGetNotifyMrInfo(void* handle, struct MrInfoT* mrInfo)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return ((handle == nullptr) || (mrInfo == nullptr)) ? -1 : 0;
}

int RaSocketSend(const void* handle, const void* data, u64 size, u64* sentSize) /*同步发送*/
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if ((data == NULL) || (size == 0) || (size > QP_MSG_MAX_SIZE)) {
        HCCL_ERROR("invalid parameters");
        return 1;
    }

    if (handle != NULL) {
        struct connect_info_t* connection = (struct connect_info_t*)handle;

        // 虚拟网卡时socket不为空。使用tcp send传输数据。节点间时socket为空，使用共享内存收发数据
        int socketfd = connection->conn_fd;
        HCCL_DEBUG("handle=%p, socketfd=%d\n", handle, socketfd);
        int sendbuf = 0;
        socklen_t sendbufsize = sizeof(sendbuf);
        int ret = getsockopt(socketfd, SOL_SOCKET, SO_SNDBUF, &sendbuf, &sendbufsize);
        if (ret == 0) {
            HCCL_DEBUG("ra_socket_send size is %d handle=%p, socketfd=%d \n", sendbuf, handle, socketfd);
        }

        ret = send(socketfd, data, size, 0);
        if (ret <= 0) {
            HCCL_ERROR("send msg error: %s(errno: %d), ret[%d] socketfd[%u]\n", strerror(errno), errno, ret, socketfd);
            return 1;
        }
        *sentSize = ret;
        HCCL_DEBUG(
            "[DEBUG-send] dev[%u] server[%x] client[%x] role[%u] fd[%d] send[%d Bytes] size [%lu]",
            connection->device_id, connection->server_ip, connection->client_ip, connection->role, connection->conn_fd,
            ret, size);
        return 0;
    }

    /*传入的为socket handle。此桩函数不用socket，这里查找qp handle*/
    struct cn_info* p_cn = (struct cn_info*)&cn[qp_index];
    // HCCL_INFO("#######qp_index[%d]", qp_index);
    if ((p_cn->qp.local_qp_msg_ptr->cnt - p_cn->qp.remote_qp_msg_ptr->rsp_cnt) > 1) {
        HCCL_ERROR(
            "commutation error:local_qp_msg_ptr->cnt=%d, remote_qp_msg_ptr->rsp_cnt=%d", p_cn->qp.local_qp_msg_ptr->cnt,
            p_cn->qp.remote_qp_msg_ptr->rsp_cnt);
        return 1;
    }

    // HCCL_INFO("wait for the last time send complete");

    while (p_cn->qp.local_qp_msg_ptr->cnt != p_cn->qp.remote_qp_msg_ptr->rsp_cnt) {
        /*之前发送数据已被接收*/
        HCCL_DEBUG("waiting for rx buffer OK...");
        SaluSleep(10000);
    }

    // HCCL_INFO("start send gdr mr info");

    p_cn->qp.local_qp_msg_ptr->cmd = QP_CMD_WRITE_MR;
    p_cn->qp.local_qp_msg_ptr->msg.mr_info.len = size;
    HcclResult ret = sal_memcpy(&p_cn->qp.local_qp_msg_ptr->msg.mr_info.data[0], QP_MSG_MAX_SIZE, data, size);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rdma send: sal memcpy error");
        return 1;
    }
    p_cn->qp.local_qp_msg_ptr->cnt++;
    *sentSize = size;
    return 0;
}

int RaTlvInit(struct TlvInitInfo* init_info, unsigned int* buffer_size, void** tlv_handle)
{
    *tlv_handle = (void*)0x12345678;
    return 0;
}

int RaTlvDeinit(void* tlv_handle) { return 0; }

int RaTlvRequest(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg)
{
    return 0;
}

int RaSocketRecv(const void* handle, void* data, u64 size, u64* recvSize) /*非阻塞接受*/
{
    if (hccpThreadStatus == 0) {
        return -1;
    }
    if ((data == NULL) || (size == 0)) {
        HCCL_ERROR("data[%p], size[0x%016x]", data, size);
        return 1;
    }

    if (handle != NULL) {
        struct connect_info_t* connection = (struct connect_info_t*)handle;

        // 虚拟网卡时socket不为空。使用tcp send传输数据。
        s32 size_recieved = 0;
        int socketfd = connection->conn_fd;

        struct timeval timeout = {3, 0};
        setsockopt(socketfd, SOL_SOCKET, SO_RCVTIMEO, (char*)&timeout, sizeof(timeval));

        HCCL_DEBUG("handle=%p, socketfd=%d\n", handle, socketfd);
        int recvbuf = 0;
        socklen_t recvbufsize = sizeof(recvbuf);
        int ret = getsockopt(socketfd, SOL_SOCKET, SO_RCVBUF, &recvbuf, &recvbufsize);
        if (ret == 0) {
            HCCL_DEBUG("ra_socket_recv buf size is %d, handle=%p, socketfd=%d\n", recvbuf, handle, socketfd);
        }
        int new_recvbuf = 10 * recvbuf;
        ret = setsockopt(socketfd, SOL_SOCKET, SO_RCVBUF, &new_recvbuf, sizeof(new_recvbuf));
        if (ret == 0) {
            HCCL_DEBUG("ra_socket_recv new_recvbuf size is %d handle=%p, socketfd=%d\n", new_recvbuf, handle, socketfd);
        }

        size_recieved = recv(socketfd, data + size_recieved, size, 0);
        if (size_recieved < 0) {
            if (errno == EAGAIN) {
                *recvSize = 0;
                return SOCK_EAGAIN;
            } else if (errno != EINTR && errno != EAGAIN) {
                HCCL_ERROR("recv msg error: %s(errno: %d)\n", strerror(errno), errno);
                return 1;
            } else {
                HCCL_ERROR("recv msg error: %s(errno: %d)\n", strerror(errno), errno);
            }
        }
        *recvSize = size_recieved;
        HCCL_ERROR(
            "[DEBUG-recv] dev[%u] server[%x] client[%x] role[%u] fd[%d] recv[%d Bytes] size [%lu]",
            connection->device_id, connection->server_ip, connection->client_ip, connection->role, connection->conn_fd,
            size_recieved, size);
        return 0;
    }
    // 节点间使用共享内存收发数据
    if (size > QP_MSG_MAX_SIZE) {
        HCCL_ERROR("data[%p], size[0x%016x]", data, size);
        return 1;
    }

    HCCL_DEBUG("start receive gdr mr info");

    struct cn_info* p_cn = (struct cn_info*)&cn[qp_index];

    if (p_cn->rev_buff.size == 0) {
        HCCL_DEBUG("there is not available data to receive");
        return SOCK_EAGAIN;
    } else if ((p_cn->rev_buff.size < 0) || (p_cn->rev_buff.size < size)) {
        HCCL_ERROR("data size error");
        return 1;
    } else {
        /*start_pos变化前，背景线程可能会往缓冲里拷贝数据，这样就使用了错误的地址*/
        /*为了避免冲突，使用互斥锁保护*/
        std::unique_lock<std::mutex> lock(g_qpMutex);

        HCCL_INFO("start pos %d,size %d,p_cn->rev_buff.size:%d", p_cn->rev_buff.start_pos, size, p_cn->rev_buff.size);

        HcclResult ret = sal_memcpy(data, size, &p_cn->rev_buff.buff[p_cn->rev_buff.start_pos], size);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("rdma send: sal memcpy error");
            return 1;
        }
        p_cn->rev_buff.start_pos += size;
        p_cn->rev_buff.size -= size;
        if (p_cn->rev_buff.size == 0) {
            p_cn->rev_buff.start_pos = 0;
        }
        *recvSize = size;
        return 0;
    }
}
int RaSendWrlist(
    void* handle, struct SendWrlistData wr[], struct SendWrRsp op_rsp[], unsigned int send_num,
    unsigned int* complete_num)
{
    HCCL_INFO("ra_send_wrlist send_num[%u]", send_num);
    if ((handle == NULL) || (wr == NULL) || (op_rsp == NULL)) {
        HCCL_ERROR("invalid parameters, handle[%p], wr[%p], op_rsp[%p]", handle, wr, op_rsp);
        return -1;
    }
    s32 ret = 0;
    struct SendWr wr_tmp = {0};

    for (u32 j = 0; j < send_num; j++) {
        wr_tmp.bufList = &(wr[j].memList);
        wr_tmp.bufNum = 1; /* 此处list只有一个，设置为1 */
        wr_tmp.dstAddr = (u64)(uintptr_t)(wr[j].dstAddr);
        wr_tmp.op = wr[j].op; /* RDMA_WRITE: 0 */
        wr_tmp.sendFlag = wr[j].sendFlags;
        HCCL_INFO(
            "[cc] ra_send_wrlist dst[%0x] local[%0x] sendNum[%u] len[%u]", wr_tmp.dstAddr, wr[j].memList.addr, send_num,
            wr[j].memList.len);
        ret = RaSendWr(handle, &wr_tmp, op_rsp);
        if (ret != 0) {
            *complete_num = j;
            HCCL_ERROR("RaSendWr failed, idx[%u] ret[%d]", j, ret);
            return ret;
        }
        op_rsp++;
    }
    *complete_num = send_num;
    return 0;
}

int RaSendWrlistExt(
    void* qp_handle, struct SendWrlistDataExt wr[], struct SendWrRsp op_rsp[], unsigned int send_num,
    unsigned int* complete_num)
{
    HCCL_INFO("ra_send_wrlist_ext fake");
    return 0;
}

int RaSendWr(void* handle, struct SendWr* wr, struct SendWrRsp* op_rsp)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 wqe_index = 0;
    if ((handle == NULL) || (wr == NULL) || (op_rsp == NULL)) {
        HCCL_ERROR("invalid parameters, handle[%p], wr[%p], op_rsp[%p]", handle, wr, op_rsp);
        return -1;
    }

    struct cn_info* p_cn = (struct cn_info*)handle;

    s8 wqn = 0;
    int list_index;

    for (; (p_cn->qp.send_mr_mgr.wqe_set[wqn] == true) && (wqn < WQE_MAX); wqn++, wqe_index = wqn) {
        if (wqn > 64) {
            // 未发送完成任务较多，适当延时, 否则有溢出风险
            SaluSleep(100000);
            HCCL_INFO("wqe_index[%d]", wqe_index);
        }
    };

    if (wqe_index == WQE_MAX) {
        HCCL_ERROR("no available memory for new WQE");
        return -1;
    }

    if (g_qp_mode == 1) { // 下沉模式
        op_rsp->wqeTmp.sqIndex = ((struct cn_info*)handle)->qpn;
        op_rsp->wqeTmp.wqeIndex = wqe_index;
    } else if (g_qp_mode == 2 || g_qp_mode == 3 || g_qp_mode == 4) {
        op_rsp->db.dbIndex = wqe_index;
        op_rsp->db.dbInfo = ((struct cn_info*)handle)->qpn;
    } else if (g_qp_mode == 0) {
        HCCL_INFO("Using Common Qp");
        op_rsp->db.dbIndex = wqe_index;
        op_rsp->db.dbInfo = ((struct cn_info*)handle)->qpn;
    } else {
        HCCL_ERROR("error qp mode[%d]", g_qp_mode);
        return -1;
    }

    std::unique_lock<std::mutex> lock(g_sglistMutex);

    for (list_index = 0; list_index < SG_LIST_MAX; list_index++) {
        if (list_index > 256) {
            // 未发送完成任务较多，适当延时, 否则有溢出风险
            SaluSleep(100000);
            HCCL_INFO("bbbb list_index[%d]", list_index);
        }
        if (g_sg_list[list_index].addr == 0)
            break;
    }
    if (list_index < SG_LIST_MAX) {
        g_sg_list[list_index].addr = wr->bufList->addr;
        g_sg_list[list_index].len = wr->bufList->len;
        // HCCL_INFO("g_sg_list[list_index].addr[0x08%x], index[%d]", g_sg_list[list_index].addr, list_index);
        wr->bufList = &g_sg_list[list_index];
    } else {
        HCCL_ERROR("no available list_index for new g_list");
        return -1;
    }

    HcclResult ret = sal_memcpy(&p_cn->qp.send_mr_mgr.wq[wqn], sizeof(struct SendWr), wr, sizeof(struct SendWr));
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rdma send: sal memcpy error");
        return -1;
    }
    p_cn->qp.send_mr_mgr.wqe_set[wqn] = true; /*将本wqe放入缓冲队列*/

    if (g_qp_mode == 0) {
        stream_class::rdma_send(wqe_index, (void*)p_cn);
        HCCL_INFO("rdma_send done");
    }
    return 0;
}

rtError_t rtRDMASend(u32 qpn, u32 wqe_index, rtStream_t stream)
{
    if ((qpn >= QP_MAX) || !cn[qpn].set_flag || (wqe_index >= WQE_MAX) || !cn[qpn].qp.send_mr_mgr.wqe_set[wqe_index]) {
        HCCL_ERROR("invalid parameters");
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if ((cn[qpn].qp.local_qp_msg_ptr->cnt - cn[qpn].qp.remote_qp_msg_ptr->rsp_cnt) > 1) {
        HCCL_ERROR("commutation error ");
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    while (cn[qpn].qp.local_qp_msg_ptr->cnt != cn[qpn].qp.remote_qp_msg_ptr->rsp_cnt) {
        // 上个命令处理完再处理本次的*/
        // HCCL_INFO("waiting for previous cmd OK...");
        SaluSleep(10000);
    }

    return rtRDMASend_stub(wqe_index, &cn[qpn], stream);
}

// 桩函数中，下沉模式与单算子是相同的，因此处加一层封装即可；
rtError_t rtRDMADBSend(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream)
{
    return rtRDMASend((u32)dbinfo, dbindex, stream);
}

int RaMrDereg(void* handle, struct MrInfoT* mrInfo) /*桩函数只负责将数据发送出去，不需要设置mr*/
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return ((handle == NULL) || (mrInfo == NULL)) ? -1 : 0;
}

int RaRegisterMr(const void* handle, struct MrInfoT* mrInfo, void** mrHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    *mrHandle = (void*)0xabcd;
    return ((handle == NULL) || (mrInfo == NULL)) ? -1 : 0;
}

int RaDeregisterMr(const void* handle, void* mrHandle) /*桩函数只负责将数据发送出去，不需要设置mr*/
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return ((handle == NULL) || (mrHandle == NULL)) ? -1 : 0;
}

int ra_get_sq_index(void* handle, u32* qpn)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if ((handle == NULL) || (qpn == NULL)) {
        return -1;
    }

    struct cn_info* p_cn = (struct cn_info*)handle;

    if (!p_cn->set_flag) {
        return -1;
    }
    *qpn = ((struct cn_info*)handle)->qpn;

    return 0;
}

int RaGetNotifyBaseAddr(void* handle, u64* va, u64* size)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    *va = 0;
    return 0;
}

int RaInit(struct RaInitConfig* config)
{
    hccpThreadStatus = 1;
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return 0;
}

int RaDeinit(struct RaInitConfig* config)
{
    // CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return 0;
}

int ra_is_first_used(int ins_id)
{
    CHK_PRT_RET(
        ins_id < 0 || ins_id >= RA_MAX_INSTANCES, HCCL_ERROR("ins_id(%d) must be in [0, %u)", ins_id, RA_MAX_INSTANCES),
        -22);
    int is_first = 0;
    pthread_mutex_lock(&g_ref_instances[ins_id].mutex);
    if (g_ref_instances[ins_id].ref_count == 0) {
        is_first++;
    }

    g_ref_instances[ins_id].ref_count++;
    pthread_mutex_unlock(&g_ref_instances[ins_id].mutex);
    return is_first;
}

int ra_is_last_used(int ins_id)
{
    CHK_PRT_RET(
        ins_id < 0 || ins_id >= RA_MAX_INSTANCES, HCCL_ERROR("ins_id(%d) must be in [0, %u)", ins_id, RA_MAX_INSTANCES),
        -22);
    int is_last = 0;
    pthread_mutex_lock(&g_ref_instances[ins_id].mutex);
    if (g_ref_instances[ins_id].ref_count == 0) {
        HCCL_ERROR("[ra_is_last_used] is called on ins_id %d which has not been used", ins_id);
        pthread_mutex_unlock(&g_ref_instances[ins_id].mutex);
        return -22;
    }

    if (g_ref_instances[ins_id].ref_count == 1) {
        is_last++;
    }

    g_ref_instances[ins_id].ref_count--;
    pthread_mutex_unlock(&g_ref_instances[ins_id].mutex);
    return is_last;
}

// 两个结构体，判断nic 还是vinc, 判断依据是根据传入IP
rdevInfo_t gRdevNicInfo[DEV_MAX] = {0};
rdevInfo_t gRdevVnicInfo[DEV_MAX] = {0};
static std::mutex g_raNicMutex; // 默认构造函数构造此全局mutex

// 目前只设计支持两个网口, 初始化完成，不支持并发访问
int RaSocketInit(int mode, struct rdev rdevInfo, void** socketHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    // dev_id 传的是物理ID，最大为4，DEV_MAX 为 16
    if (rdevInfo.phyId > DEV_MAX) {
        HCCL_ERROR("in ra_socket_init, error dev_id[%d]", rdevInfo.phyId);
        return -1;
    }

    HCCL_INFO(
        "ra_socket_init, rdevInfo.dev_id[%d], rdevInfo.localIp.addr.s_addr[%u]", rdevInfo.phyId,
        rdevInfo.localIp.addr.s_addr);

    std::unique_lock<std::mutex> lock(g_raNicMutex);
    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (rdevInfo.localIp.addr.s_addr < DEV_MAX) {
        u32 dev_id = rdevInfo.phyId;
        gRdevVnicInfo[dev_id].dev_id = rdevInfo.phyId;
        gRdevVnicInfo[dev_id].local_ip = rdevInfo.localIp.addr.s_addr;
        gRdevVnicInfo[dev_id].idx = rdevInfo.phyId;
        *socketHandle = (void*)&gRdevVnicInfo[rdevInfo.phyId];
        return 0;
    } else {
        // 0， 1 存储device 0 的网卡0，1信息; 2， 3 存储device 1 的网卡0，1信息; 以此类推
        u32 idx = rdevInfo.phyId * 2;
        if (gRdevNicInfo[idx].local_ip != 0) {
            idx = rdevInfo.phyId * 2 + 1;
            if (gRdevNicInfo[idx].local_ip != 0) {
                HCCL_ERROR(
                    "ra_socket_init fail, local_ip0[%u], local_ip1[%u]", gRdevNicInfo[rdevInfo.phyId * 2],
                    gRdevNicInfo[rdevInfo.phyId * 2 + 1]);
                return -1;
            }
        }

        gRdevNicInfo[idx].dev_id = rdevInfo.phyId;
        gRdevNicInfo[idx].local_ip = rdevInfo.localIp.addr.s_addr;
        gRdevNicInfo[idx].idx = idx;
        *socketHandle = (void*)&gRdevNicInfo[idx];
        return 0;
    }
    return -1;
}
int RaSocketInitV1(int mode, struct SocketInitInfoT socket_init, void** socketHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    // dev_id 传的是物理ID，最大为4，DEV_MAX 为 16
    if (socket_init.rdevInfo.phyId > DEV_MAX) {
        HCCL_ERROR("in ra_socket_init, error dev_id[%d]", socket_init.rdevInfo.phyId);
        return -1;
    }

    HCCL_INFO(
        "ra_socket_init, rdevInfo.dev_id[%d], rdevInfo.localIp.addr.s_addr[%u]", socket_init.rdevInfo.phyId,
        socket_init.rdevInfo.localIp.addr.s_addr);

    std::unique_lock<std::mutex> lock(g_raNicMutex);
    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (socket_init.rdevInfo.localIp.addr.s_addr < DEV_MAX) {
        u32 dev_id = socket_init.rdevInfo.phyId;
        gRdevVnicInfo[dev_id].dev_id = socket_init.rdevInfo.phyId;
        gRdevVnicInfo[dev_id].local_ip = socket_init.rdevInfo.localIp.addr.s_addr;
        gRdevVnicInfo[dev_id].idx = socket_init.rdevInfo.phyId;
        *socketHandle = (void*)&gRdevVnicInfo[socket_init.rdevInfo.phyId];
        return 0;
    } else {
        // 0， 1 存储device 0 的网卡0，1信息; 2， 3 存储device 1 的网卡0，1信息; 以此类推
        u32 idx = socket_init.rdevInfo.phyId * 2;
        if (gRdevNicInfo[idx].local_ip != 0) {
            idx = socket_init.rdevInfo.phyId * 2 + 1;
            if (gRdevNicInfo[idx].local_ip != 0) {
                HCCL_ERROR(
                    "ra_socket_init fail, local_ip0[%u], local_ip1[%u]", gRdevNicInfo[socket_init.rdevInfo.phyId * 2],
                    gRdevNicInfo[socket_init.rdevInfo.phyId * 2 + 1]);
                return -1;
            }
        }

        gRdevNicInfo[idx].dev_id = socket_init.rdevInfo.phyId;
        gRdevNicInfo[idx].local_ip = socket_init.rdevInfo.localIp.addr.s_addr;
        gRdevNicInfo[idx].idx = idx;
        *socketHandle = (void*)&gRdevNicInfo[idx];
        return 0;
    }
    return -1;
}

// 返回值为0时，success ，其余为fail
int RaSocketDeinit(void* socket_handle)
{
    // CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (!socket_handle) {
        HCCL_WARNING("in ra_socket_deinit, rdma_handle is null");
        return -1;
    }
    rdevInfo_t* socketHandle = (rdevInfo_t*)socket_handle;
    if (socketHandle->dev_id > DEV_MAX) {
        return -1;
    }
    // 虚拟网卡场景
    HCCL_INFO("ra_socket_deinit: %d", socketHandle->local_ip);
    if (socketHandle->local_ip < DEV_MAX) {
        s32 ret = memset_s(&gRdevVnicInfo[socketHandle->dev_id], sizeof(rdevInfo_t), 0, sizeof(rdevInfo_t));
        if (ret != EOK) {
            HCCL_WARNING(
                "memset_s failed. errorno[%d], params: dest[%p], "
                "destMaxSize[%d]",
                ret, &gRdevVnicInfo[socketHandle->dev_id], sizeof(rdevInfo_t));
        }
        return 0;
    } else {
        // Device网卡场景
        u32 idx = socketHandle->idx;
        s32 ret = memset_s(&gRdevNicInfo[idx], sizeof(rdevInfo_t), 0, sizeof(rdevInfo_t));
        if (ret != EOK) {
            HCCL_WARNING(
                "memset_s failed. errorno[%d], params: dest[%p], "
                "destMaxSize[%d]",
                ret, &gRdevNicInfo[idx], sizeof(rdevInfo_t));
        }
        return 0;
    }
    return -1;
}

int RaRdevInit(int mode, unsigned int notify_type, struct rdev rdevInfo, void** rdma_handle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    // rdev_init 只有rdma使用，因此采用pcie时，不初始化
    // 采用rdma时，初始化内容与socket相同，直接return socket的内容；
    // dev_id 最大为8，DEV_MAX 为 16
    if (rdevInfo.phyId > DEV_MAX) {
        HCCL_ERROR("in ra_dev_init, error dev_id[%d]", rdevInfo.phyId);
        return -1;
    }

    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (rdevInfo.localIp.addr.s_addr < DEV_MAX) {
        HCCL_ERROR("unsupported rdev init by Vic, local_ip[%u]", rdevInfo.localIp.addr.s_addr);
        return -1;
    } else {
        int idx = rdevInfo.phyId * 2;
        if (gRdevNicInfo[idx].local_ip != rdevInfo.localIp.addr.s_addr) {
            idx = rdevInfo.phyId * 2 + 1;
            if (gRdevNicInfo[idx].local_ip != rdevInfo.localIp.addr.s_addr) {
                HCCL_ERROR("ra_rdev_init, should init ra rdev after ra socket");
                return -1;
            }
        }
        if ((void*)&gRdevNicInfo[idx] == nullptr) {
            HCCL_ERROR("ra_rdev_init, rdma handle is null");
        }
        *rdma_handle = (void*)&gRdevNicInfo[idx];
        return 0;
    }
    return -1;
}

int RaRdevInitWithBackup(
    struct RdevInitInfo* init_info, struct rdev* rdevInfo, struct rdev* backup_rdevInfo, void** rdma_handle)
{
    int mode = init_info->mode;
    unsigned int notify_type = init_info->notifyType;
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    // rdev_init 只有rdma使用，因此采用pcie时，不初始化
    // 采用rdma时，初始化内容与socket相同，直接return socket的内容；
    // dev_id 最大为8，DEV_MAX 为 16
    if (rdevInfo->phyId > DEV_MAX) {
        HCCL_ERROR("in ra_dev_init, error dev_id[%d]", rdevInfo->phyId);
        return -1;
    }

    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (rdevInfo->localIp.addr.s_addr < DEV_MAX) {
        HCCL_ERROR("unsupported rdev init by Vic, local_ip[%u]", rdevInfo->localIp.addr.s_addr);
        return -1;
    } else {
        int idx = rdevInfo->phyId * 2;
        if (gRdevNicInfo[idx].local_ip != rdevInfo->localIp.addr.s_addr) {
            idx = rdevInfo->phyId * 2 + 1;
            if (gRdevNicInfo[idx].local_ip != rdevInfo->localIp.addr.s_addr) {
                HCCL_ERROR("ra_rdev_init, should init ra rdev after ra socket");
                return -1;
            }
        }
        if ((void*)&gRdevNicInfo[idx] == nullptr) {
            HCCL_ERROR("ra_rdev_init, rdma handle is null");
        }
        *rdma_handle = (void*)&gRdevNicInfo[idx];
        return 0;
    }
    return -1;
}

int RaRdevInitV2(struct RdevInitInfo init_info, struct rdev rdevInfo, void** rdma_handle)
{
    int mode = init_info.mode;
    unsigned int notify_type = init_info.notifyType;
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    // rdev_init 只有rdma使用，因此采用pcie时，不初始化
    // 采用rdma时，初始化内容与socket相同，直接return socket的内容；
    // dev_id 最大为8，DEV_MAX 为 16
    if (rdevInfo.phyId > DEV_MAX) {
        HCCL_ERROR("in ra_dev_init, error dev_id[%d]", rdevInfo.phyId);
        return -1;
    }

    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (rdevInfo.localIp.addr.s_addr < DEV_MAX) {
        HCCL_ERROR("unsupported rdev init by Vic, local_ip[%u]", rdevInfo.localIp.addr.s_addr);
        return -1;
    } else {
        int idx = rdevInfo.phyId * 2;
        if (gRdevNicInfo[idx].local_ip != rdevInfo.localIp.addr.s_addr) {
            idx = rdevInfo.phyId * 2 + 1;
            if (gRdevNicInfo[idx].local_ip != rdevInfo.localIp.addr.s_addr) {
                HCCL_ERROR("ra_rdev_init, should init ra rdev after ra socket");
                return -1;
            }
        }
        if ((void*)&gRdevNicInfo[idx] == nullptr) {
            HCCL_ERROR("ra_rdev_init, rdma handle is null");
        }
        *rdma_handle = (void*)&gRdevNicInfo[idx];
        return 0;
    }
    return -1;
}

int RaRdevDeinit(void* rdma_handle, unsigned int notify_type)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (!rdma_handle) {
        HCCL_WARNING("in ra_rdev_deinit, rdma_handle is null");
        return -1;
    }
    // ra_rdev_deinit 只做参数较验，不做销毁,由socket 销毁
    rdevInfo_t* rdmaHandle = (rdevInfo_t*)rdma_handle;
    HCCL_INFO(
        "BBBBB rdmaHandle->dev_id[%d], rdmaHandle->local_ip[%d], rdmaHandle->idx[%d]", rdmaHandle->dev_id,
        rdmaHandle->local_ip, rdmaHandle->idx);
    if (rdmaHandle->dev_id > DEV_MAX) {
        HCCL_WARNING("in ra_rdev_deinit, device id[%d] ERROR", rdmaHandle->dev_id);
    }

    // 通过IP 判断当前是nic 还是vic, 小于DEV_MAX则是Vnic
    if (rdmaHandle->local_ip < DEV_MAX) {
        HCCL_WARNING("unsupported rdev deinit by Vic, local_ip[%u]", rdmaHandle->local_ip);
    }

    u32 idx = rdmaHandle->idx;
    if (gRdevNicInfo[idx].local_ip != rdmaHandle->local_ip) {
        HCCL_WARNING(
            "in ra_rdev_deinit, rdma_handle error, idx[%u], rdmaHandle_ip[%u], gRdevNicInfo[%u]", idx,
            rdmaHandle->local_ip, gRdevNicInfo[idx].local_ip);
    }
    s32 ret = memset_s(&gRdevNicInfo[idx], sizeof(rdevInfo_t), 0, sizeof(rdevInfo_t));
    if (ret != EOK) {
        HCCL_WARNING(
            "memset_s failed. errorno[%d], params: dest[%p], "
            "destMaxSize[%d]",
            ret, &gRdevNicInfo[idx], sizeof(rdevInfo_t));
    }

    return 0;
}

int GetInfoFromHandle(void* handle, u32& deviceId, u32& localIp, u32& idx)
{
    if (handle == nullptr) {
        HCCL_ERROR("in ra_rdev_deinit, handle is null");
        return -1;
    }
    rdevInfo_t* rdevHandle = (rdevInfo_t*)handle;
    deviceId = ((rdevInfo_t*)handle)->dev_id;
    localIp = ((rdevInfo_t*)handle)->local_ip;
    idx = ((rdevInfo_t*)handle)->idx;
    return 0;
}

int ra_set_work_mode(int mode)
{
    // CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    work_mode = mode;
    return 0;
}

int ra_get_ip_by_dev(u32 dev_id, u32* ipAddr)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    /** HCCL用于节点内交换数据的socket通过host侧的环回IP完成
        IP地址的分配规则为：device0 - 127.0.0.2, device1 - 127.0.0.3
                            device2 - 127.0.0.4, device3 - 127.0.0.5
    */

    /*s32 ip = (0x7F000010 + dev_id);

    STUB_INFO("dev_id[%d] ip[0x%08x]", dev_id, ip);
    *ipAddr = htonl(ip);
    return HCCL_SUCCESS;*/

    return rt_get_dev_ip(0, dev_id, ipAddr);
}
int ra_change_ip_to_dev(s32 ipAddr, u32* dev_id)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 dev;
    u32 ip;
    ip = ntohl(ipAddr);
    ;
    dev = ip & 0x000000FF;
    dev = dev - 0x10;
    *dev_id = dev;
    HCCL_DEBUG("ra_change_ip_to_dev:ip[0x%08x] dev:%d", ipAddr, *dev_id);
    return 0;
}

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __ra_get_dev_ip(s32 chipType, s32 dev_id, u32* ipAddr)
{
    /** HCCL用于节点内交换数据的socket通过host侧的环回IP完成
        IP地址的分配规则为：device0 - 127.0.0.2, device1 - 127.0.0.3
                            device2 - 127.0.0.4, device3 - 127.0.0.5
    */
    s32 pid = 0;
    if (work_mode == MODE_PID_AS_SERVER) {
        pid = SalGetPid();
    } else {
        // nothing todo
    }
    // s32 ip = (0x7F000010 + ((pid&0xfff0)<<8) + dev_id);
    s32 ip = (0x7F000010 + ((pid & 0xffff) << 8) + dev_id);

    HCCL_INFO("ra_get_dev_ip: dev_id[%d] ip[0x%08x]", dev_id, ip);
    *ipAddr = htonl(ip);
    return HCCL_SUCCESS;
}
strong_alias(__ra_get_dev_ip, rt_get_dev_ip);

u32 map_to_loop_ip(u32 ipAddr)
{
    if (g_test_type == 1) {
        return ((ipAddr & 0xFFFF0000) | 0xFE7F);
    } else {
        return ((ipAddr & 0xFFFFFF00) | 0x7F);
    }
}
#ifdef __cplusplus
} // extern "C"
#endif

int batch_connect_inner_nodes(struct SocketConnectInfoT conn, s32 device_id, u32 num)
{
    // 将远端的device 转变成 IP地址
    u32 remote_ip = 0;
    ra_get_ip_by_dev(conn.remoteIp.addr.s_addr, &remote_ip);
    HCCL_INFO("batch_connect_inner_nodes : device_id[%u], server_ip[0x%08x]", device_id, remote_ip);
    if (device_id >= DEV_MAX) {
        HCCL_ERROR("invalid device_id[%d]", device_id);
        return -1;
    }

    struct connect_info_t connection;
    if ((connection.conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }
    s32 on = 1;
    setsockopt(connection.conn_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = remote_ip;
    server_addr.sin_port = htons(bind_port);

    s32 count = 0;
    while (1) {
        if (count > 500) {
            HCCL_ERROR(
                "client batch connect timeout: device_id[%d], server_ip[0x%08x], %s(errno: %d)", device_id, remote_ip,
                strerror(errno), errno);
            return -1;
        }

        if (connect(connection.conn_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            SaluSleep(10000);
            count++;
            HCCL_INFO("connect waiting...,server_ip[0x%08x]", remote_ip);
            continue;
        } else {
            HCCL_INFO("connect success,server_ip[0x%08x]", remote_ip);
            break;
        }
    }

    // 发送client信息
    struct client_info_t send_info;
    u32 size_total = sizeof(struct client_info_t);
    u32 size_sent = 0;
    u32 size_residue = size_total;

    rt_get_dev_ip(0, ((rdevInfo_t*)conn.socketHandle)->idx, &send_info.client_ip);
    memcpy(send_info.tag, conn.tag, sizeof(conn.tag));
    do {
        char* send_addr = (char*)(&send_info) + size_sent;
        int send_ret = 0;

        send_ret = send(connection.conn_fd, send_addr, size_residue, 0);
        if (send_ret < 0) {
            if (errno == EAGAIN || errno == EINTR) {
                continue;
            }

            HCCL_ERROR("send error: %s(errno: %d)", strerror(errno), errno);
            break;
        }
        size_sent += send_ret;
        size_residue = size_total - size_sent;
    } while (size_residue > 0);

    if (size_sent != size_total) {
        // 发送中途出错, 继续accept
        return 0;
    }

    // 全部存储转换过的IP
    connection.client_ip = send_info.client_ip;
    connection.server_ip = remote_ip;
    connection.status = 1; // 已连接
    connection.device_id = device_id;
    connection.role = CONN_CLIENT;
    memcpy(connection.tag, send_info.tag, sizeof(send_info.tag));
    connection.key = g_conn_client[device_id].set_conn(connection.tag, connection);

    HCCL_DEBUG(
        "client connection : tag[%s], device_id[%d], server_ip[%08x], fd[%d], key[%llu]\n", connection.tag, device_id,
        remote_ip, connection.conn_fd, connection.key);
    return 0;
}

// 此处的device_id，实际是idx编号
int batch_connect_between_nodes(struct SocketConnectInfoT conn, s32 device_id, u32 num)
{
    HCCL_INFO("batch_connect_between_nodes : device_id[%u], server_ip[0x%08x]", device_id, conn.remoteIp.addr.s_addr);
    if (device_id >= DEV_MAX_NODES) {
        HCCL_INFO("device_id beyond max, device_id[%d]", device_id);
        return -1;
    }

    u32 deviceId = 0, localIp = 0, idx = 0;
    if (GetInfoFromHandle(conn.socketHandle, deviceId, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle ERROR, socket_handle is null");
        return -1;
    };

    struct connect_info_t connection;
    if ((connection.conn_fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    s32 on = 1;
    u32 server_ip = map_to_loop_ip(conn.remoteIp.addr.s_addr);
    setsockopt(connection.conn_fd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = server_ip;
    server_addr.sin_port = htons(IsUseRealPortAndName() ? conn.port : bind_port_nodes);
    s32 count = 0;
    while (1) {
        if (count > 500) {
            HCCL_ERROR(
                "client batch connect timeout: device_id[%d], server_ip[%08x], %s(errno: %d)", device_id,
                conn.remoteIp.addr.s_addr, strerror(errno), errno);
            return -1;
        }

        if (connect(connection.conn_fd, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            SaluSleep(10000);
            count++;
            HCCL_INFO("connect waiting..., connect server_ip[0x%08x]", server_ip);
            continue;
        } else {
            HCCL_INFO("connect success!, connect server_ip[0x%08x]", server_ip);
            break;
        }
    }

    // 发送client信息
    struct client_info_t send_info;
    u32 size_total = sizeof(struct client_info_t);
    u32 size_sent = 0;
    u32 size_residue = size_total;
    send_info.client_ip = localIp;
    memcpy(send_info.tag, conn.tag, sizeof(conn.tag));
    do {
        char* send_addr = (char*)(&send_info) + size_sent;
        int send_ret = 0;

        send_ret = send(connection.conn_fd, send_addr, size_residue, 0);
        if (send_ret < 0) {
            if (errno == EAGAIN || errno == EINTR) {
                continue;
            }

            HCCL_ERROR("send error: %s(errno: %d)", strerror(errno), errno);
            break;
        }
        size_sent += send_ret;
        size_residue = size_total - size_sent;
    } while (size_residue > 0);

    if (size_sent != size_total) {
        // 发送中途出错, 继续accept
        return 0;
    }
    HCCL_DEBUG("connect success!, send client ip[0x%08x], tag[%s]", send_info.client_ip, send_info.tag);

    connection.client_ip = send_info.client_ip;
    connection.server_ip = conn.remoteIp.addr.s_addr;
    connection.status = 1; // 已连接
    connection.device_id = device_id;
    connection.role = CONN_CLIENT;
    memcpy(connection.tag, send_info.tag, sizeof(send_info.tag));
    connection.key = g_conn_client_nodes[device_id].set_conn(connection.tag, connection);
    return 0;
}

int RaSocketBatchConnect(struct SocketConnectInfoT conn[], u32 num)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    int ret = 0;
    u32 device_id = 0, ipAddr = 0, idx = 0;
    for (u32 connect_cnt = 0; connect_cnt < num; connect_cnt++) {
        if (GetInfoFromHandle(conn[connect_cnt].socketHandle, device_id, ipAddr, idx)) {
            HCCL_ERROR("GetInfoFromHandle ERROR, socket_handle is null");
            return -1;
        };
        HCCL_DEBUG("batch_connect : device_id[%u], server_ip[0x%08x]", device_id, ipAddr);
        if (ipAddr < SERVER_IP_MAX_VALUE) {
            ra_get_ip_by_dev(ipAddr, &ipAddr);
        }
        if ((ipAddr & 0xFF) != 0x7F || RESERVED_LOOP_IP(ipAddr)) {
            // 节点间的batch Connect
            ret |= batch_connect_between_nodes(conn[connect_cnt], idx, num);
            if (ret > 0) {
                HCCL_ERROR("batch connect between nodes  error, conn[%d], idx[%d], num [%d]", connect_cnt, idx, num);
            }
        } else {
            // 节点内的batch Connect
            ret |= batch_connect_inner_nodes(conn[connect_cnt], idx, num);
            if (ret > 0) {
                HCCL_ERROR("batch connect inner nodes  error, conn[%d], idx[%d], num [%d]", connect_cnt, idx, num);
            }
        }
    }

    return 0;
}

/* 目前stub中batch_connect()是同步方式实现的，因此建链失败后abort无需做任何事 */
int RaSocketBatchAbort(struct SocketConnectInfoT conn[], u32 num) { return 0; }

int RaSocketBatchClose(struct SocketCloseInfoT conn[], u32 num)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    int ret = 0;
    for (int i = 0; i < num; i++) {
        if (conn[i].fdHandle != NULL) {
            struct connect_info_t* connection = (struct connect_info_t*)conn[i].fdHandle;
            close(connection->conn_fd);
            u32 device_id = connection->device_id;
            // 节点间关闭连接
            if ((connection->server_ip & 0xFF) != 0x7F || RESERVED_LOOP_IP(connection->server_ip)) {
                // 节点间，需清除共享内存中的数据
                if (connection->role == CONN_SERVER) {
                    // 关闭此socket 连接
                    HCCL_DEBUG(
                        "server batch close..., device_id[%d], num[%u], current[%d], tag[%s], key[%llu]", device_id,
                        num, i + 1, connection->tag, connection->key);
                    g_conn_server_nodes[device_id].del_conn(connection->tag, *connection);
                } else {
                    // 关闭此socket 连接
                    HCCL_DEBUG(
                        "client batch close..., device_id[%d], num[%u], current[%d], tag[%s], key[%llu]", device_id,
                        num, i + 1, connection->tag, connection->key);
                    g_conn_client_nodes[device_id].del_conn(connection->tag, *connection);
                }
            } else { // 节点内关闭连接
                if (connection->role == CONN_SERVER) {
                    HCCL_DEBUG(
                        "server batch close..., device_id[%d], num[%u], current[%d], tag[%s], key[%llu]", device_id,
                        num, i + 1, connection->tag, connection->key);

                    g_conn_server[device_id].del_conn(connection->tag, *connection);
                } else {
                    HCCL_DEBUG(
                        "client batch close..., device_id[%d], num[%u], current[%d], tag[%s], key[%llu]", device_id,
                        num, i + 1, connection->tag, connection->key);

                    g_conn_client[device_id].del_conn(connection->tag, *connection);
                }
            }
            conn[i].fdHandle = NULL;
        }
    }

    return 0;
}

// 每个device会有一份该线程的copy
#define SUPPORT_ASYNC_SOCKET 0
#define SUPPORT_NON_BLOCK_SOCKET 1

void* accept_process_inner_nodes(void* arg)
{
    int acceptfd = 0;
    thread_para* para = (thread_para*)arg;
    s32 device_id = para->device_id;
    int listenfd = para->listenfd;
    u32 server_ip = para->ipAddr;

    // 通知母线程可以返回, 销毁para参数
    listen_flag[device_id] = 1;
    sal_sem_give(para->sem);

    // 异步socket相关
#if SUPPORT_ASYNC_SOCKET
    struct timeval timeout = {2, 0};
    fd_set rfd;
    int nfds;
#endif

    HCCL_INFO(
        "Enter accept_process_inner_nodes process.. device_id[%u], server_ip[0x%08x], listenfd[%u]", device_id,
        server_ip, listenfd);

    // 设置监听的socket为非阻塞
#if SUPPORT_NON_BLOCK_SOCKET
    struct sockaddr client_sockaddr;
    int sin_size = sizeof(struct sockaddr_in);
    int flags = fcntl(listenfd, F_GETFL, 0);
    if (flags < 0) {
        HCCL_ERROR("tid[%d], fcntl get : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
    if (fcntl(listenfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        HCCL_ERROR("tid[%d], fcntl set : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
#endif

    while (1) {
        if (listen_done[device_id]) {
            break;
        }

#if SUPPORT_ASYNC_SOCKET
        FD_ZERO(&rfd);          // 每次循环都要清空集合，否则不能检测描述符变化
        FD_SET(listenfd, &rfd); // 添加描述符
        timeout.tv_sec = 0;
        timeout.tv_usec = 1000; // select函数会不断修改timeout的值，所以每次循环都应该重新赋值
        nfds = select(listenfd + 1, &rfd, (fd_set*)NULL, (fd_set*)NULL, &timeout);
        if (nfds == 0) {
            // HCCL_INFO("select nothing...");
            continue;
        } else if (nfds > 0) {
            // 异步socket
            for (int i = 0; i < nfds; i++) {
#endif
                if ((acceptfd = accept(listenfd, (struct sockaddr*)NULL, (socklen_t*)NULL)) == -1) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        // 只接受没有请求的状态
                        HCCL_DEBUG("accept : %s(errno: %d)", strerror(errno), errno);
                        SaluSleep(100000);
                        continue;
                    }

                    HCCL_ERROR("accept error: %s(errno: %d)", strerror(errno), errno);
                    break;
                }

                HCCL_DEBUG("accept_process_inner_nodes accept a client, let's go!");

                // 接收client消息, client_ip + tag
                struct client_info_t recv_info;
                u32 size_total = sizeof(struct client_info_t);
                u32 size_recv = 0;
                u32 size_residue = size_total;
                do {
                    char* recv_addr = (char*)(&recv_info) + size_recv;

                    u32 recv_ret = recv(acceptfd, recv_addr, size_residue, 0);
                    if (recv_ret < 0) {
                        if (errno == EAGAIN || errno == EINTR) {
                            continue;
                        }

                        HCCL_ERROR("recv error: %s(errno: %d)", strerror(errno), errno);
                        break;
                    }

                    size_recv += recv_ret;
                    size_residue = size_total - size_recv;
                } while (size_residue > 0);

                if (size_recv != size_total) {
                    // 接收中途出错, 继续accept
                    continue;
                }

                // 记录连接信息
                struct connect_info_t connection;
                connection.conn_fd = acceptfd;
                connection.client_ip = recv_info.client_ip;
                connection.server_ip = server_ip;
                connection.status = 1;
                connection.device_id = device_id;
                connection.role = CONN_SERVER;
                memcpy(connection.tag, recv_info.tag, sizeof(recv_info.tag));
                connection.key = g_conn_server[device_id].set_conn(connection.tag, connection);

                HCCL_DEBUG(
                    "server connection : tag[%s], device_id[%d], client_ip[%08x], fd[%d], key[%llu]", connection.tag,
                    device_id, connection.client_ip, connection.conn_fd, connection.key);

#if SUPPORT_ASYNC_SOCKET
            }
        }
#endif
    }

    HCCL_DEBUG(
        "tid[%d] exit, device_id[%d], server_ip[0x%08x], listen_fd[%d]", SalGetTid(), device_id, server_ip, listenfd);

    return NULL;
}

void* accept_process_between_nodes(void* arg)
{
    int acceptfd = 0;
    thread_para* para = (thread_para*)arg;
    s32 device_id = para->device_id;
    int listenfd = para->listenfd;
    u32 server_ip = para->ipAddr;
    void* ptr;

    listen_flag_nodes[device_id] = 1;

    sal_sem_give(para->sem);
#if SUPPORT_ASYNC_SOCKET
    struct timeval timeout = {2, 0};
    fd_set rfd;
    int nfds;
#endif
    HCCL_INFO(
        "Enter accept_process_between_nodes process.. device_id[%u], server_ip[0x%08x], listenfd[%u]", device_id,
        server_ip, listenfd);
#if SUPPORT_NON_BLOCK_SOCKET
    struct sockaddr client_sockaddr;
    int sin_size = sizeof(struct sockaddr_in);
    int flags = fcntl(listenfd, F_GETFL, 0);
    if (flags < 0) {
        HCCL_ERROR("tid[%d], fcntl get : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
    if (fcntl(listenfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        HCCL_ERROR("tid[%d], fcntl set : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
#endif
    while (1) {
        if (listen_done_nodes[device_id]) {
            break;
        }
#if SUPPORT_ASYNC_SOCKET
        FD_ZERO(&rfd);          // 每次循环都要清空集合，否则不能检测描述符变化
        FD_SET(listenfd, &rfd); // 添加描述符
        timeout.tv_sec = 0;
        timeout.tv_usec = 1000; // select函数会不断修改timeout的值，所以每次循环都应该重新赋值
        nfds = select(listenfd + 1, &rfd, (fd_set*)NULL, (fd_set*)NULL, &timeout);
        if (nfds == 0) {
            continue;
        } else if (nfds > 0) {
            for (int i = 0; i < nfds; i++) {
#endif
                if ((acceptfd = accept(listenfd, (struct sockaddr*)NULL, (socklen_t*)NULL)) == -1) {
                    if (errno == EAGAIN || errno == EWOULDBLOCK) {
                        HCCL_DEBUG("accept : %s(errno: %d)", strerror(errno), errno);
                        SaluSleep(100000);
                        continue;
                    }
                    HCCL_ERROR("accept error: %s(errno: %d)", strerror(errno), errno);
                    break;
                }
                HCCL_INFO("accept_process_between_nodes accept_process_between_nodes accept a client, let's go!");
                struct client_info_t recv_info;
                u32 size_total = sizeof(struct client_info_t);
                u32 size_recv = 0;
                u32 size_residue = size_total;
                do {
                    char* recv_addr = (char*)(&recv_info) + size_recv;
                    u32 recv_ret = recv(acceptfd, recv_addr, size_residue, 0);
                    if (recv_ret < 0) {
                        if (errno == EAGAIN || errno == EINTR) {
                            continue;
                        }
                        HCCL_ERROR("recv error: %s(errno: %d)", strerror(errno), errno);
                        break;
                    }
                    size_recv += recv_ret;
                    size_residue = size_total - size_recv;
                } while (size_residue > 0);
                if (size_recv != size_total) {
                    continue;
                }
                HCCL_INFO("listern receive info[0x%08x], tag[%s]", recv_info.client_ip, recv_info.tag);

                // 记录连接信息
                struct connect_info_t connection;
                connection.conn_fd = acceptfd;
                connection.client_ip = recv_info.client_ip;
                connection.server_ip = server_ip;
                connection.status = 1;
                connection.device_id = device_id;
                connection.role = CONN_SERVER;
                memcpy(connection.tag, recv_info.tag, sizeof(recv_info.tag));
                connection.key = g_conn_server_nodes[device_id].set_conn(connection.tag, connection);
                HCCL_INFO(
                    "accept_process_between_nodes, client_ip[%08x], server_ip[%08x], device_id[%d], tag[%s]",
                    connection.client_ip, server_ip, device_id, connection.tag);
#if SUPPORT_ASYNC_SOCKET
            }
        }
#endif
    }
    HCCL_INFO(
        "tid[%d] exit, device_id[%d], server_ip[0x%08x], listen_fd[%d]", SalGetTid(), device_id, server_ip, listenfd);
    return NULL;
}
int ra_listen_start_inner_nodes(struct SocketListenInfoT conn)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    int listenfd;
    struct sockaddr_in servaddr;
    if (GetInfoFromHandle(conn.socketHandle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle is null");
        return -1;
    };
    ra_get_ip_by_dev(localIp, &localIp);
    if (device_id >= DEV_MAX || idx >= DEV_MAX) {
        HCCL_ERROR("error device_id[%d], idx[%d]", device_id, idx);
        // 节点间的监听IP地址, 目前桩函数限定最多两个节点, 暂时不做任何动作
        return -1;
    }
    /** 节点内socket桩函数是基于环回ip实现socket通信
        桩函数的环回IP带着PID, 因此每个进程只起一个监听线程 */

    u32 listen_thread_index = __sync_fetch_and_add(&(listen_num[idx]), 1);
    if (listen_thread_index > 0) {
        // 非首个线程, 等待建链成功
        u32 timeout = 200;
        while (!listen_flag[idx]) {
            SaluSleep(10000);
            timeout--;
        }

        if (timeout == 0) {
            HCCL_ERROR("waiting listen start... timeout");
            return -1;
        }
    } else {
        if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
            return -1;
        }

        s32 on = 1;
        setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));

        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = localIp;
        servaddr.sin_port = htons(bind_port);

        HCCL_INFO("ra_listen_start_inner_nodes,listen ip[0x%08x]", localIp);
        if (bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
            HCCL_ERROR("bind error: %s(errno: %d) idx=%d, ipAddr[0x%08x]", strerror(errno), errno, idx, localIp);
            return -1;
        }

        if (listen(listenfd, 10) == -1) {
            HCCL_ERROR("listen socket error: %s(errno: %d)\n", strerror(errno), errno);
            return -1;
        }

        thread_para para;
        para.device_id = idx;
        para.listenfd = listenfd;
        para.ipAddr = localIp;
        para.sem = sal_sem_create("async_sem", 1, 0, 0);
        if (para.sem == NULL) {
            HCCL_ERROR("thread semaphore failed");
            return -1;
        }

        // 本device对应的数据清0
        listen_done[idx] = 0;
        listen_thread[idx] = sal_thread_create("listen", accept_process_inner_nodes, (void*)&para);
        if (listen_thread[idx] == NULL) {
            HCCL_ERROR("thread create failed");
            sal_sem_destroy(para.sem);
            return -1;
        }

        // 启动子线程的入参用的局部变量, 需要子线程确认使用完毕后, 本函数方可退出
        sal_sem_take(para.sem, SAL_SEM_FOREVER);
        sal_sem_destroy(para.sem);

        g_listen_fd[idx] = listenfd;

        HCCL_INFO(
            "listen start : device_id[%d], ipAddr[%08x], idx[%d], listenfd[%d], listen_thread[%p]", device_id, localIp,
            idx, g_listen_fd[idx], listen_thread[idx]);
    }
    return 0;
}

// 桩函数中，弱化device id的概念，用idx代替，idx = device_id * 2 + nic_port_id
int ra_listen_start_between_nodes(struct SocketListenInfoT conn)
{
    HCCL_INFO("HCCL TEST STUB Port[%u]", conn.port);
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    int listenfd;
    struct sockaddr_in servaddr;

    if (GetInfoFromHandle(conn.socketHandle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle error");
        return -1;
    }

    if (device_id >= DEV_MAX_NODES || idx >= DEV_MAX_NODES) {
        // 未初始化或错误的device_id，防止内存泄漏
        HCCL_ERROR("error device_id[%d], idx[%d]", device_id, idx);
        return -1;
    }

    u32 listen_thread_index = __sync_fetch_and_add(&(listen_num_nodes[idx]), 1);
    if (listen_thread_index > 0) {
        // 非首个线程, 等待建链成功
        u32 timeout = 200;
        while (!listen_flag_nodes[idx]) {
            SaluSleep(10000);
            timeout--;
        }

        if (timeout == 0) {
            HCCL_ERROR("waiting listen start... timeout");
            return -1;
        }
    } else {
        // 首个线程, 启动监听线程
        if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
            HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
            return -1;
        }

        s32 on = 1;
        // 获取server_ip映射后的环回ip
        u32 server_ip = map_to_loop_ip(localIp);

        // 存取自身的Server_ip;
        g_client_ip_nodes[idx] = localIp;

        setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
        memset(&servaddr, 0, sizeof(servaddr));
        servaddr.sin_family = AF_INET;
        servaddr.sin_addr.s_addr = server_ip;
        servaddr.sin_port = htons(IsUseRealPortAndName() ? conn.port : bind_port_nodes);
        HCCL_INFO(
            "bind start: %s(errno: %d) idx=%d, ipAddr[0x%08x],map_ip[0x%08x],port[%u]", strerror(errno), errno, idx,
            localIp, server_ip, ntohs(servaddr.sin_port));
        if (bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
            HCCL_ERROR(
                "bind error: %s(errno: %d) idx=%d, localIp[0x%08x],map_ip[0x%08x],port[%u]", strerror(errno), errno,
                idx, localIp, server_ip, ntohs(servaddr.sin_port));
            return -1;
        }
        if (listen(listenfd, 10) == -1) {
            HCCL_ERROR("listen end socket error: %s(errno: %d)\n", strerror(errno), errno);
            return -1;
        }
        thread_para para;
        para.device_id = idx;
        para.listenfd = listenfd;
        para.ipAddr = localIp;
        para.sem = sal_sem_create("async_sem", 1, 0, 0);
        if (para.sem == NULL) {
            HCCL_ERROR("thread semaphore failed");
            return -1;
        }

        // 本device对应的数据清0
        // g_conn_server[device_id].clear();
        listen_done_nodes[idx] = 0;
        listen_thread_nodes[idx] = sal_thread_create("listen", accept_process_between_nodes, (void*)&para);
        if (listen_thread_nodes[idx] == NULL) {
            HCCL_ERROR("thread create failed");
            sal_sem_destroy(para.sem);
            return -1;
        }

        // 启动子线程的入参用的局部变量, 需要子线程确认使用完毕后, 本函数方可退出
        sal_sem_take(para.sem, SAL_SEM_FOREVER);
        sal_sem_destroy(para.sem);

        g_listen_fd_nodes[idx] = listenfd;

        HCCL_INFO(
            "listen start : idx[%d], ipAddr[%08x], listenfd[%d], listen_thread[%p]", idx, localIp,
            g_listen_fd_nodes[idx], listen_thread_nodes[idx]);
    }
    return 0;
}

int RaSocketListenStart(struct SocketListenInfoT conn[], u32 num)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    int ret = 0;
    u32 deviceId = 0, localIp = 0, idx = 0;
    for (u32 listen_cnt = 0; listen_cnt < num; listen_cnt++) {
        if (GetInfoFromHandle(conn[listen_cnt].socketHandle, deviceId, localIp, idx)) {
            HCCL_ERROR("GetInfoFromHandle error");
            return -1;
        }

        HCCL_INFO("INTER NODE : listen_cnt=%u, device_id=%d, ipAddr=0x%08x", listen_cnt, deviceId, localIp);
        // 节点内的， 不修改原来的流程
        if (localIp < SERVER_IP_MAX_VALUE) {
            ra_get_ip_by_dev(localIp, &localIp);
        }

        if ((localIp & 0xFF) != 0x7F || RESERVED_LOOP_IP(localIp)) {
            // 节点间的通信
            ret |= ra_listen_start_between_nodes(conn[listen_cnt]);
            if (ret > 0) {
                HCCL_ERROR("ra_listen_start_between_nodes error, ret[%d]", ret);
            }
        } else {
            // 节点内的通信
            ret |= ra_listen_start_inner_nodes(conn[listen_cnt]);
            if (ret > 0) {
                HCCL_ERROR("ra_listen_start_inner_nodes error, ret[%d]", ret);
            }
        }
    }
    return ret;
}

int ra_listen_stop_inner_nodes(u32 device_id)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (listen_num[device_id] <= 0) {
        HCCL_ERROR("listen stop failed, has not been listened");
        return -1;
    }
    u32 listen_thread_index = __sync_fetch_and_sub(&(listen_num[device_id]), 1);
    if (listen_thread_index > 1) {
        // 非最后一个停止的线程, 同步等待最后一个线程结束监听
        u32 timeout = 1000;
        while (listen_flag[device_id]) {
            SaluSleep(10000);
            timeout--;
        }

        if (timeout == 0) {
            HCCL_ERROR("listen stop waiting... timeout");
            return -1;
        }
    } else {
        // 销毁listen fd
        if (g_listen_fd[device_id] > 0) {
            listen_done[device_id] = 1;

            HCCL_DEBUG("listen stop, device_id[%d], listen_fd[%d]", device_id, g_listen_fd[device_id]);

            close(g_listen_fd[device_id]);
            u32 count = 0;
            // 累计等1s钟
            while (sal_thread_is_running(listen_thread[device_id]) && count < 100) {
                SaluSleep(1000 * 10); /* 等待 10 ms, 确保线程已经退出 */
                count++;
            }
            (void)sal_thread_destroy(listen_thread[device_id]);
            g_listen_fd[device_id] = 0;
            // 销毁accept fd(上层调用batch_close销毁, 无需在此销毁)

            listen_thread[device_id] = NULL;
            listen_flag[device_id] = 0;
        } else {
            HCCL_WARNING("listen stop device_id[%d], i[0], listenfd[%d] <= 0", device_id, g_listen_fd[device_id]);
        }
    }
    return 0;
}
int ra_listen_stop_between_nodes(u32 device_id)
{
    void* ptr;
    s32 ret;
    struct connect_info_t* connect;
    u32 listen_thread_index = __sync_fetch_and_sub(&(listen_num_nodes[device_id]), 1);
    if (listen_thread_index > 1) {
        return 0;
        // u32 timeout = 1000;
        // while (listen_flag_nodes[device_id]) {
        //     SaluSleep(10000);
        //     timeout--;
        // }
        // if (timeout == 0) {
        //     HCCL_ERROR("listen stop waiting... timeout");
        //     return -1;
        // }
    } else {
        if (g_listen_fd_nodes[device_id] > 0) {
            listen_done_nodes[device_id] = 1;

            close(g_listen_fd_nodes[device_id]);

            SaluSleep(200000);
            u32 count = 0;
            while (sal_thread_is_running(listen_thread_nodes[device_id]) && count < 100) {
                SaluSleep(1000 * 10); /* 等待 10 ms, 确保线程已经退出 */
                count++;
            }
            (void)sal_thread_destroy(listen_thread_nodes[device_id]);
            g_listen_fd_nodes[device_id] = 0;
            // 销毁accept fd(上层调用batch_close销毁, 无需在此销毁)
            listen_thread_nodes[device_id] = NULL;
            listen_flag_nodes[device_id] = 0;
        } else {
            HCCL_WARNING("listen stop device_id[%d], i[0], listenfd[%d] <= 0", device_id, g_listen_fd_nodes[device_id]);
        }
    }

    return 0;
}

int RaSocketListenStop(struct SocketListenInfoT conn[], u32 num)
{
    // CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    int ret = 0;
    u32 device_id = 0, localIp = 0, idx = 0;
    for (u32 listen_cnt = 0; listen_cnt < num; listen_cnt++) {
        if (GetInfoFromHandle(conn[listen_cnt].socketHandle, device_id, localIp, idx)) {
            HCCL_ERROR("GetInfoFromHandle, conn[listen_cnt].socketHandle is null");
            return -1;
        }
        HCCL_INFO(
            "listen stop : listen_cnt=%u, device_id=%d, ipAddr=0x%08x, idx=%u", listen_cnt, device_id, localIp, idx);

        if (device_id >= DEV_MAX || idx >= DEV_MAX) {
            // 未初始化或错误的device_id，防止内存泄漏
            continue;
        }

        if (localIp < SERVER_IP_MAX_VALUE) {
            ra_get_ip_by_dev(localIp, &localIp);
        }
        // 此处有idx 来代替原来的device id的概念
        if ((localIp & 0xFF) != 0x7F || RESERVED_LOOP_IP(localIp)) {
            // 节点间的监听IP地址
            ret |= ra_listen_stop_between_nodes(idx);
            if (ret > 0) {
                HCCL_ERROR("ra_listen_stop_between_nodes error, ret[%d]", ret);
            }
        } else {
            // 节点内监听的IP地址
            ret |= ra_listen_stop_inner_nodes(idx);
            if (ret > 0) {
                HCCL_ERROR("ra_listen_stop_inner_nodes error, ret[%d]", ret);
            }
        }
    }
    return 0;
}

// 节点内获取sockets 信息
int ra_get_sockets_inner_nodes(u32 role, struct SocketInfoT conn[], u32 num, int* count)
{
    HCCL_DEBUG("ra_get_sockets_inner_nodes, server_ip[0x%08x], role[%d]", conn[0].remoteIp.addr.s_addr, role);
    int valid_socket_num = 0;
    u32 server_ip;
    char tag[SOCK_CONN_TAG_SIZE + 1];
    u32 changeDevId;
    for (int i = 0; i < num; i++) {
        // 节点内的socket
        if (CONN_CLIENT == role) {
            // client 端，以conn[i] 为key进行查找
            server_ip = conn[i].remoteIp.addr.s_addr;
            ra_get_ip_by_dev(server_ip, &server_ip);
            memcpy(tag, conn[i].tag, sizeof(conn[i].tag));
            u32 idx = ((rdevInfo_t*)conn[i].socketHandle)->idx;
            struct connect_info_t* connection = g_conn_client[idx].get_conn(tag, server_ip);
            if (NULL != connection) {
                // client在batch_connect阶段已完成ip发送
                conn[i].fdHandle = connection;
                ra_change_ip_to_dev(connection->server_ip, &changeDevId);
                conn[i].remoteIp.addr.s_addr = changeDevId;
                conn[i].status = connection->status;

                HCCL_INFO(
                    "client: handle[%p], accept_fd[%d], server_ip[%08x]", conn[i].fdHandle,
                    ((struct connect_info_t*)conn[i].fdHandle)->conn_fd, conn[i].remoteIp.addr.s_addr);
                valid_socket_num++;
            }
        } else if (CONN_SERVER == role) {
            server_ip = ((rdevInfo_t*)conn[i].socketHandle)->local_ip;
            ra_get_ip_by_dev(server_ip, &server_ip);
            memcpy(tag, conn[i].tag, sizeof(conn[0].tag));
            u32 idx = ((rdevInfo_t*)conn[i].socketHandle)->idx;
            HCCL_DEBUG("ra_get_sockets_inner_nodes: role[%d], conn idx[%u], server ip[0x%x]", role, idx, server_ip);
            struct connect_info_t* connection = g_conn_server[idx].get_conn(tag, server_ip);
            if (NULL != connection) {
                // client在batch_connect阶段已完成ip发送
                conn[i].fdHandle = connection;
                conn[i].remoteIp.addr.s_addr = 0;
                // conn[i].remoteIp.addr.s_addr = connection->server_ip;
                conn[i].status = connection->status;
                HCCL_INFO(
                    "server: handle[%p], accept_fd[%d], remote_ip[%08x]", conn[i].fdHandle,
                    ((struct connect_info_t*)conn[i].fdHandle)->conn_fd, conn[i].remoteIp.addr.s_addr);

                valid_socket_num++;
            }
        } else {
            HCCL_ERROR("Unknown role");
        }
    }
    *count = valid_socket_num;
    return 0;
}

// 节点间获取sockets 信息
int ra_get_sockets_between_nodes(u32 role, struct SocketInfoT conn[], u32 num, int* count)
{
    int valid_socket_num = 0;
    s32 device_id;
    u32 server_ip;
    char tag[SOCK_CONN_TAG_SIZE + 1] = {0};
    u32 changeDevId;
    HCCL_DEBUG("ra_get_sockets_between_nodes, server_ip[0x%08x], role[%d]", conn[0].remoteIp.addr.s_addr, role);
    for (int i = 0; i < num; i++) {
        // 节点间的socket
        if (CONN_CLIENT == role) {
            // client 端，以conn[i] 为key进行查找
            server_ip = conn[i].remoteIp.addr.s_addr;
            memcpy(tag, conn[i].tag, sizeof(conn[i].tag));
            u32 idx = ((rdevInfo_t*)conn[i].socketHandle)->idx;
            struct connect_info_t* connection = g_conn_client_nodes[idx].get_conn(tag, server_ip);
            if (NULL != connection) {
                conn[i].fdHandle = connection;
                conn[i].remoteIp.addr.s_addr = connection->server_ip;
                conn[i].status = connection->status;
                HCCL_INFO(
                    "ra_get_sockets_between_nodes, client: fd_handle[%p], accept_fd[%p], server_ip[%08x]",
                    conn[i].fdHandle, ((struct connect_info_t*)conn[i].fdHandle)->conn_fd,
                    conn[i].remoteIp.addr.s_addr);
                valid_socket_num++;
            } else {
                conn[i].fdHandle = NULL;
                conn[i].remoteIp.addr.s_addr = 0;
                conn[i].status = 1;
            }
        } else if (CONN_SERVER == role) {
            // Server 端，以自已的IP
            server_ip = ((rdevInfo_t*)conn[i].socketHandle)->local_ip;
            memcpy(tag, conn[i].tag, sizeof(conn[i].tag));
            u32 idx = ((rdevInfo_t*)conn[i].socketHandle)->idx;
            struct connect_info_t* connection = g_conn_server_nodes[idx].get_conn(tag, server_ip);
            if (NULL != connection) {
                // client在batch_connect阶段已完成ip发送
                conn[i].fdHandle = connection;
                conn[i].remoteIp.addr.s_addr = connection->client_ip;
                conn[i].status = connection->status;
                HCCL_INFO(
                    "ra_get_sockets_between_nodes bbb, server: handle[%p], accept_fd[%d], server_ip[%08x]",
                    conn[i].fdHandle, ((struct connect_info_t*)conn[i].fdHandle)->conn_fd,
                    conn[i].remoteIp.addr.s_addr);
                valid_socket_num++;
            } else {
                conn[i].fdHandle = NULL;
                conn[i].remoteIp.addr.s_addr = 0;
                conn[i].status = 1;
            }
        } else {
            HCCL_ERROR("Unknown role");
        }
    }
    *count = valid_socket_num;
    return 0;
}

int RaGetSockets(u32 role, struct SocketInfoT conn[], u32 num, u32* connectedNum)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    if (conn == NULL) {
        HCCL_ERROR("conn is NULL");
        return 1;
    }

    int valid_socket_num = 0;
    char tag[SOCK_CONN_TAG_SIZE + 1];
    int ret = 0;
    SaluSleep(1000 * 100);
    if (conn[0].socketHandle == nullptr) {
        HCCL_ERROR("socket handle is null");
        return 1;
    }
    u32 localIp = ((rdevInfo_t*)conn[0].socketHandle)->local_ip;

    HCCL_DEBUG("ra_get_sockets, localIp[0x%08x], role[%d], num", localIp, role, num);
    // 此接口固定使用conn[0]的localIp和tag作为查找参数
    for (s32 loop = 0; loop < num; ++loop) {
        if (localIp < SERVER_IP_MAX_VALUE) {
            ra_get_ip_by_dev(localIp, &localIp);
        }
    }

    // 注,是以conn[0] 的localIp来判断是节点内通信还是节点间
    if (((localIp & 0xFF) != 0x7F || RESERVED_LOOP_IP(localIp))) {
        /** 节点间的socket, 桩函数通过共享内存shm实现,*/
        ret |= ra_get_sockets_between_nodes(role, conn, num, &valid_socket_num);
        if (ret > 0) {
            HCCL_ERROR("ra_get_sockets_between_nodes error, ret[%d]", ret);
        }
    } else {
        // 节点内的通信
        ret |= ra_get_sockets_inner_nodes(role, conn, num, &valid_socket_num);
        if (ret > 0) {
            HCCL_ERROR("ra_get_sockets_inner_nodes error, ret[%d]", ret);
        }
    }
    *connectedNum = valid_socket_num;

    return 0;
}

void* event_process(void* p_cn)
{
    if (p_cn == NULL)
        return NULL;

    struct cn_info* cn = (struct cn_info*)p_cn;
    struct qp_info* qp = &(cn->qp);

    HCCL_INFO("start thread for command process ");
    while (qp->shm_msg_ptr == NULL) /*等待信息内存申请*/
    {
        // HCCL_INFO("wait for command");
        SaluSleep(100 * EVENT_LOOP_INTER_US);
    }

    while (cn->thread_run_flag) {
        SaluSleep(100 * EVENT_LOOP_INTER_US);
        if ((qp->remote_qp_msg_ptr->cnt - qp->local_qp_msg_ptr->rsp_cnt) > 1) {
            HCCL_ERROR("commutation error ");
            return NULL;
        }

        if (qp->remote_qp_msg_ptr->cnt == qp->local_qp_msg_ptr->rsp_cnt) /*no new command*/
        {
            /*no new command nothing to do*/
            // HCCL_INFO("wait for command");
        } else {
            if (qp->remote_qp_msg_ptr->cmd == QP_CMD_WRITE_MR) /*mr相关信息的交换*/
            {
                /*将数据放入缓冲区(因为上层调用是连续发送，连续接收类型的)，考虑和adapt_recv的冲突问题*/
                std::unique_lock<std::mutex> lock(g_qpMutex);

                HCCL_INFO("start pos %d,size %u", cn->rev_buff.start_pos, cn->rev_buff.size);

                HcclResult ret = sal_memcpy(
                    &cn->rev_buff.buff[cn->rev_buff.start_pos + cn->rev_buff.size],
                    cn->qp.remote_qp_msg_ptr->msg.mr_info.len, &cn->qp.remote_qp_msg_ptr->msg.mr_info.data[0],
                    cn->qp.remote_qp_msg_ptr->msg.mr_info.len);
                if (ret != HCCL_SUCCESS) {
                    HCCL_ERROR("rdma send: sal memcpy error");
                    return NULL;
                }

                cn->rev_buff.size += cn->qp.remote_qp_msg_ptr->msg.mr_info.len;
                qp->local_qp_msg_ptr->rsp_cnt++;                        /*接收完毕*/
            } else if (qp->remote_qp_msg_ptr->cmd == QP_CMD_WRITE_DATA) /*write命令*/
            {
                // if (cn->notify_addr ==  qp->remote_qp_msg_ptr->msg.write_info.dst_addr)
                //{
                //     HCCL_DEBUG("write notify address [%p]",(u64*)(qp->remote_qp_msg_ptr->msg.write_info.dst_addr));
                //     (void)__sync_add_and_fetch((u64*)(qp->remote_qp_msg_ptr->msg.write_info.dst_addr), 1);
                // }
                // else

                HCCL_INFO(
                    "qp->remote_qp_msg_ptr[%p] msg.write_info.dst_addr[%p], "
                    "msg.write_info.len[%d],msg.write_info.data[%p], msg.write_info.len[%d]",
                    (qp->remote_qp_msg_ptr), (qp->remote_qp_msg_ptr->msg.write_info.dst_addr),
                    qp->remote_qp_msg_ptr->msg.write_info.len, (qp->remote_qp_msg_ptr->msg.write_info.data),
                    qp->remote_qp_msg_ptr->msg.write_info.len);

                {
                    HcclResult ret = sal_memcpy(
                        qp->remote_qp_msg_ptr->msg.write_info.dst_addr, qp->remote_qp_msg_ptr->msg.write_info.len,
                        qp->remote_qp_msg_ptr->msg.write_info.data, qp->remote_qp_msg_ptr->msg.write_info.len);
                    HCCL_INFO(
                        "[ccres] dada result[%f] dst result[%f] len[%u]",
                        (float)(qp->remote_qp_msg_ptr->msg.write_info.data[0]),
                        *((float*)(qp->remote_qp_msg_ptr->msg.write_info.dst_addr)),
                        qp->remote_qp_msg_ptr->msg.write_info.len);
                    if (ret != HCCL_SUCCESS) {
                        HCCL_ERROR("rdma send: sal memcpy error");
                        return NULL;
                    }
                }

                qp->local_qp_msg_ptr->rsp_cnt++; /*接收完毕*/
            }
        }
    }

    return NULL;
}

u32 map_to_loop_ip_use_in_check_link(u32 ipAddr)
{
    u32 mapIp;
    if (g_test_type == 1) {
        mapIp = ((ipAddr & 0xFFFF0000) | 0xFE7F);
        return mapIp;
    } else {
        mapIp = ((ipAddr & 0xFFFFFF00) | 0x7F);
        return mapIp;
    }
}

void* accept_process_link_check_nodes(void* arg)
{
    int acceptfd = 0;
    thread_para* para = (thread_para*)arg;
    s32 device_id = para->device_id;
    int listenfd = para->listenfd;
    u32 server_ip = para->ipAddr;
    s32 findLoop = 0;
    // 通知母线程可以返回, 销毁para参数
    sal_sem_give(para->sem);

    HCCL_INFO("Enter accept process.. device_id[%u], server_ip[0x%08x], listenfd[%u]", device_id, server_ip, listenfd);

    // 设置监听的socket为非阻塞
#if SUPPORT_NON_BLOCK_SOCKET
    struct sockaddr client_sockaddr;
    int sin_size = sizeof(struct sockaddr_in);
    int flags = fcntl(listenfd, F_GETFL, 0);
    if (flags < 0) {
        HCCL_ERROR("tid[%d], fcntl get : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
    if (fcntl(listenfd, F_SETFL, flags | O_NONBLOCK) < 0) {
        HCCL_ERROR("tid[%d], fcntl set : %s(errno: %d)", SalGetTid(), strerror(errno), errno);
        return NULL;
    }
#endif

    while (1) {
        if ((acceptfd = accept(listenfd, (struct sockaddr*)NULL, (socklen_t*)NULL)) == -1) {
            if (errno == EAGAIN || errno == EWOULDBLOCK) {
                // 只接受没有请求的状态
                // HCCL_INFO("accept : %s(errno: %d)", strerror(errno), errno);
                SaluSleep(100000);
                continue;
            }

            HCCL_ERROR("accept error: %s(errno: %d)", strerror(errno), errno);
            break;
        }

        // 接收client消息, client_ip + tag
        u32 recv_info;
        u32 size_total = sizeof(recv_info);
        u32 size_recv = 0;
        u32 size_residue = size_total;
        do {
            char* recv_addr = (char*)(&recv_info) + size_recv;

            u32 recv_ret = recv(acceptfd, recv_addr, size_residue, 0);
            if (recv_ret < 0) {
                if (errno == EAGAIN || errno == EINTR) {
                    continue;
                }

                HCCL_ERROR("recv error: %s(errno: %d)", strerror(errno), errno);
                break;
            }

            size_recv += recv_ret;
            size_residue = size_total - size_recv;
        } while (size_residue > 0);

        if (size_recv != size_total) {
            // 接收中途出错, 继续accept
            continue;
        }
        /* 接收到消息后发送自己的ip */
        s32 send_ret;
        send_ret = send(
            acceptfd, (char*)(&g_client_ip_check_nodes[device_id]), sizeof(g_client_ip_check_nodes[device_id]), 0);
        if (send_ret < 0) {
            HCCL_ERROR("send error: %s(errno: %d)", strerror(errno), errno);
        }

        // 记录连接信息
        for (findLoop = 0; findLoop < linkServerCheckSocket[device_id].remoteNum; ++findLoop) {
            if (recv_info == linkServerCheckSocket[device_id].remoteLink[findLoop].remoteIpAddr) {
                linkServerCheckSocket[device_id].remoteLink[findLoop].handle = acceptfd;
                linkServerCheckSocket[device_id].remoteLink[findLoop].checkResult = 0;
                break;
            }
        }
        if (findLoop == linkServerCheckSocket[device_id].remoteNum) {
            HCCL_ERROR(
                "server connection error: device_id[%d], receive client_ip[0x%08x] but not find in server remote ip",
                device_id, recv_info);
        } else {
            HCCL_INFO(
                "server connection success: device_id[%d], client_ip[0x%08x], fd[%d]", device_id, recv_info, acceptfd);
        }
    }

    HCCL_DEBUG(
        "tid[%d] exit, device_id[%d], server_ip[0x%08x], listen_fd[%d]", SalGetTid(), device_id, server_ip, listenfd);

    return NULL;
}

int check_server_listen_start(int device_id, unsigned int ip)
{
    int listenfd;
    struct sockaddr_in servaddr;

    if ((listenfd = socket(AF_INET, SOCK_STREAM, 0)) == -1) {
        HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }
    g_check_listen_fd[device_id] = listenfd;

    // 获取server_ip映射后的环回ip
    u32 server_ip = map_to_loop_ip_use_in_check_link(ip);

    s32 on = 1;
    // 存取自身的client_ip;
    setsockopt(listenfd, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    memset(&servaddr, 0, sizeof(servaddr));
    servaddr.sin_family = AF_INET;
    servaddr.sin_addr.s_addr = server_ip;
    servaddr.sin_port = htons(bind_check_link_port);

    if (bind(listenfd, (struct sockaddr*)&servaddr, sizeof(servaddr)) == -1) {
        HCCL_ERROR(
            "bind error: %s(errno: %d) device_id=%d, ipAddr[0x%08x],map_ip[0x%08x]", strerror(errno), errno, device_id,
            ip, server_ip);
        return -1;
    }
    if (listen(listenfd, 10) == -1) {
        HCCL_ERROR("listen socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    thread_para para;
    para.device_id = device_id;
    para.listenfd = listenfd;
    para.ipAddr = server_ip;
    para.sem = sal_sem_create("async_sem", 1, 0, 0);
    if (para.sem == NULL) {
        HCCL_ERROR("thread semaphore failed");
        return -1;
    }
    checkListenThread[device_id] = sal_thread_create("listen", accept_process_link_check_nodes, (void*)&para);
    if (checkListenThread[device_id] == NULL) {
        HCCL_ERROR("thread create failed");
        sal_sem_destroy(para.sem);
        return -1;
    }

    // 启动子线程的入参用的局部变量, 需要子线程确认使用完毕后, 本函数方可退出
    sal_sem_take(para.sem, SAL_SEM_FOREVER);
    sal_sem_destroy(para.sem);

    HCCL_INFO(
        "check_server_listen_start listen start : device_id[%d], ipAddr[%08x], listenfd[%d], listen_thread[%p]",
        device_id, server_ip, g_check_listen_fd[device_id], checkListenThread[device_id]);

    return 0;
}

int batch_connect_check_link_nodes(u32 ipAddr, s32 deviceId)
{
    HCCL_INFO("batch_connect_inner_nodes : device_id[%u], server_ip[0x%08x]", deviceId, ipAddr);

    if (deviceId >= DEV_MAX) {
        HCCL_ERROR("invalid device_id[%d]", deviceId);
        return -1;
    }

    s32 socketHandle;
    if ((socketHandle = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        HCCL_ERROR("create socket error: %s(errno: %d)\n", strerror(errno), errno);
        return -1;
    }

    s32 on = 1;
    setsockopt(socketHandle, SOL_SOCKET, SO_REUSEADDR, &on, sizeof(on));
    u32 server_ip = map_to_loop_ip_use_in_check_link(ipAddr);
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = server_ip;
    server_addr.sin_port = htons(bind_check_link_port);

    s32 count = 0;
    while (1) {
        if (count > 500) {
            HCCL_ERROR(
                "client batch connect timeout: device_id[%d], server_ip[%08x], %s(errno: %d)", deviceId, ipAddr,
                strerror(errno), errno);
            return -1;
        }

        if (connect(socketHandle, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0) {
            SaluSleep(10000);
            count++;

            HCCL_INFO("connect waiting...,server_ip[0x%08x]", ipAddr);
            continue;
        } else {
            HCCL_INFO("connect success,server_ip[0x%08x]", ipAddr);
            break;
        }
    }

    // 发送client信息
    u32 send_info;
    u32 size_total = sizeof(send_info);
    u32 size_sent = 0;
    u32 size_residue = size_total;
    send_info = g_client_ip_check_nodes[deviceId];
    do {
        char* send_addr = (char*)(&send_info) + size_sent;
        int send_ret = 0;

        send_ret = send(socketHandle, send_addr, size_residue, 0);
        if (send_ret < 0) {
            if (errno == EAGAIN || errno == EINTR) {
                continue;
            }

            HCCL_ERROR("send error: %s(errno: %d)", strerror(errno), errno);
            break;
        }
        size_sent += send_ret;
        size_residue = size_total - size_sent;
    } while (size_residue > 0);

    if (size_sent != size_total) {
        // 发送中途出错, 继续accept
        return 0;
    }
    /* 接收对端的ip */
    u32 remoteServerIp;
    u32 recv_ret = recv(socketHandle, (char*)(&remoteServerIp), sizeof(remoteServerIp), 0);
    if (recv_ret < 0) {
        HCCL_INFO("connect recevive failed.error%s(errno: %d)", strerror(errno), errno);
        return -1;
    }
    /* 保持连接状态 */
    s32 loop;
    for (loop = 0; loop < linkClientCheckSocket[deviceId].remoteNum; ++loop) {
        if (linkClientCheckSocket[deviceId].remoteLink[loop].remoteIpAddr == remoteServerIp) {
            linkClientCheckSocket[deviceId].remoteLink[loop].handle = socketHandle;
            linkClientCheckSocket[deviceId].remoteLink[loop].checkResult = 0;
            break;
        }
    }
    if (loop >= linkClientCheckSocket[deviceId].remoteNum) {
        HCCL_ERROR(
            "client connection: device_id[%d], receive server ip[%08x] but not find in server remote ip", deviceId,
            remoteServerIp);
        return -1;
    }
    HCCL_INFO("client connection : device_id[%d], server_ip[%08x], fd[%d],\n", deviceId, ipAddr, socketHandle);
    return 0;
}

int set_link_check_index(u32 device_id, s32 role, s32 index, s32 result, s32 haveGet)
{
    if (role == 0) {
        if (result < 8) {
            linkClientCheckSocket[device_id].remoteLink[index].checkResult = result;
        }
        if (haveGet < 8) {
            linkClientCheckSocket[device_id].remoteLink[index].resultHaveGet = haveGet;
        }
    } else {
        if (result < 8) {
            linkServerCheckSocket[device_id].remoteLink[index].checkResult = result;
        }
        if (haveGet < 8) {
            linkServerCheckSocket[device_id].remoteLink[index].resultHaveGet = haveGet;
        }
    }
}

int force_set_link_check_result(u32 device_id, s32 role, s32 index, s32 result, s32 haveGet)
{
    s32 clientRole;
    s32 severRole;
    s32 setNum;
    s32 setLoop;
    if (role == 0) {
        clientRole = 1;
        severRole = 0;
    } else if (role == 1) {
        clientRole = 0;
        severRole = 1;
    } else {
        clientRole = 1;
        severRole = 1;
    }
    if (clientRole == 1) {
        setNum = linkClientCheckSocket[device_id].remoteNum;
        if (index < 8) {
            if (index < setNum) {
                set_link_check_index(device_id, 0, index, result, haveGet);
            }
            return 0;
        }
        for (setLoop = 0; setLoop < setNum; ++setLoop) {
            set_link_check_index(device_id, 0, setLoop, result, haveGet);
        }
    }
    if (severRole == 1) {
        setNum = linkServerCheckSocket[device_id].remoteNum;
        if (index < 8) {
            if (index < setNum) {
                set_link_check_index(device_id, 1, index, result, haveGet);
            }
            return 0;
        }
        for (setLoop = 0; setLoop < setNum; ++setLoop) {
            set_link_check_index(device_id, 1, setLoop, result, haveGet);
        }
    }

    return 0;
}

int RaSocketSetWhiteListStatus(unsigned int enable) { return 0; }

int RaSocketGetWhiteListStatus(unsigned int* enable)
{
    *enable = 1;
    return 0;
}

int RaSocketWhiteListAdd(void* socket_handle, struct SocketWlistInfoT white_list[], u32 num)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return 0; // 白名单特性主要支持HCCP安全特性，桩函数暂不适配
}

int RaSocketWhiteListDel(void* socket_handle, struct SocketWlistInfoT white_list[], u32 num)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return 0; // 白名单特性主要支持HCCP安全特性，桩函数暂不适配
}

aclError aclrtDeviceEnablePeerAccess(int32_t devIdDes, u32 flag) { return ACL_SUCCESS; }

aclError aclrtDeviceDisablePeerAccess(int32_t devicePhyId) { return ACL_SUCCESS; }

int RaGetIfnum(struct RaGetIfattr* config, unsigned int* num)
{
    if (config->nicPosition == 0) {
        *num = 4;
    } else if (config->nicPosition == 1) {
        *num = 2;
    }
    return DRV_ERROR_NONE;
}

int RaGetIfaddrs(struct RaGetIfattr* config, struct InterfaceInfo ifaddr_infos[], unsigned int* num)
{
    if (config->nicPosition == 0) {
        ifaddr_infos[0].ifaddr.ip.addr.s_addr = 0x100007f;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[0].ifname[0] = 'e';
        ifaddr_infos[0].ifname[1] = 't';
        ifaddr_infos[0].ifname[2] = 'h';
        ifaddr_infos[0].ifname[3] = '0';
        ifaddr_infos[0].ifname[4] = '\0';
        ifaddr_infos[0].family = AF_INET;

        ifaddr_infos[1].ifaddr.ip.addr.s_addr = 0x200007f;
        ifaddr_infos[1].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[1].ifname[0] = 'd';
        ifaddr_infos[1].ifname[1] = 'o';
        ifaddr_infos[1].ifname[2] = 'c';
        ifaddr_infos[1].ifname[3] = 'k';
        ifaddr_infos[1].ifname[4] = 'e';
        ifaddr_infos[1].ifname[5] = 'r';
        ifaddr_infos[1].ifname[6] = '\0';
        ifaddr_infos[1].family = AF_INET;

        ifaddr_infos[2].ifaddr.ip.addr.s_addr = 0x300007f;
        ifaddr_infos[2].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[2].ifname[0] = 'l';
        ifaddr_infos[2].ifname[1] = 'o';
        ifaddr_infos[2].ifname[2] = '\0';
        ifaddr_infos[2].family = AF_INET;

        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[0] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[1] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[2] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[3] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[4] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[5] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[6] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[7] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[8] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[9] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[10] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[11] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[12] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[13] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[14] = 0;
        ifaddr_infos[3].ifaddr.ip.addr6.s6_addr[15] = 1;
        ifaddr_infos[3].ifaddr.mask.s_addr = 0;
        ifaddr_infos[3].ifname[0] = 'l';
        ifaddr_infos[3].ifname[1] = 'o';
        ifaddr_infos[3].ifname[2] = '\0';
        ifaddr_infos[3].family = AF_INET6;
        *num = 4;
    } else if (config->nicPosition == 1) {
        CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), DRV_ERROR_WAIT_TIMEOUT);
        ifaddr_infos[0].ifaddr.ip.addr.s_addr = 0x400007f | (config->phyId << 28);
        ;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[0].ifname[0] = 'e';
        ifaddr_infos[0].ifname[1] = 't';
        ifaddr_infos[0].ifname[2] = 'h';
        ifaddr_infos[0].ifname[3] = '1';
        ifaddr_infos[0].ifname[4] = '\0';
        ifaddr_infos[0].family = AF_INET;

        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[0] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[1] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[2] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[3] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[4] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[5] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[6] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[7] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[8] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[9] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[10] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[11] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[12] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[13] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[14] = 0;
        ifaddr_infos[1].ifaddr.ip.addr6.s6_addr[15] = 1;
        ifaddr_infos[1].ifaddr.mask.s_addr = 0;
        ifaddr_infos[1].ifname[0] = 'e';
        ifaddr_infos[1].ifname[1] = 't';
        ifaddr_infos[1].ifname[1] = 'h';
        ifaddr_infos[1].ifname[1] = '1';
        ifaddr_infos[1].ifname[2] = '\0';
        ifaddr_infos[1].family = AF_INET6;
        *num = 2;
    }
    return DRV_ERROR_NONE;
}

int ra_get_ifaddrs_ipv4(struct RaGetIfattr* config, struct InterfaceInfo ifaddr_infos[], unsigned int* num)
{
    if (config->nicPosition == 0) {
        ifaddr_infos[0].ifaddr.ip.addr.s_addr = 0x100007f;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[0].ifname[0] = 'e';
        ifaddr_infos[0].ifname[1] = 't';
        ifaddr_infos[0].ifname[2] = 'h';
        ifaddr_infos[0].ifname[3] = '0';
        ifaddr_infos[0].ifname[4] = '\0';
        ifaddr_infos[0].family = AF_INET;

        ifaddr_infos[1].ifaddr.ip.addr.s_addr = 0x200007f;
        ifaddr_infos[1].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[1].ifname[0] = 'd';
        ifaddr_infos[1].ifname[1] = 'o';
        ifaddr_infos[1].ifname[2] = 'c';
        ifaddr_infos[1].ifname[3] = 'k';
        ifaddr_infos[1].ifname[4] = 'e';
        ifaddr_infos[1].ifname[5] = 'r';
        ifaddr_infos[1].ifname[6] = '\0';
        ifaddr_infos[1].family = AF_INET;

        ifaddr_infos[2].ifaddr.ip.addr.s_addr = 0x300007f;
        ifaddr_infos[2].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[2].ifname[0] = 'l';
        ifaddr_infos[2].ifname[1] = 'o';
        ifaddr_infos[2].ifname[2] = '\0';
        ifaddr_infos[2].family = AF_INET;

        *num = 3;
    } else if (config->nicPosition == 1) {
        CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), DRV_ERROR_WAIT_TIMEOUT);
        ifaddr_infos[0].ifaddr.ip.addr.s_addr = 0x400007f | (config->phyId << 28);
        ;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0xffff;
        ifaddr_infos[0].ifname[0] = 'e';
        ifaddr_infos[0].ifname[1] = 't';
        ifaddr_infos[0].ifname[2] = 'h';
        ifaddr_infos[0].ifname[3] = '1';
        ifaddr_infos[0].ifname[4] = '\0';
        ifaddr_infos[0].family = AF_INET;
        *num = 1;
    }
    return DRV_ERROR_NONE;
}

int ra_get_ifaddrs_ipv6(struct RaGetIfattr* config, struct InterfaceInfo ifaddr_infos[], unsigned int* num)
{
    if (config->nicPosition == 0) {
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[0] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[1] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[2] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[3] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[4] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[5] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[6] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[7] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[8] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[9] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[10] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[11] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[12] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[13] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[14] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[15] = 1;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0;
        ifaddr_infos[0].ifname[0] = 'l';
        ifaddr_infos[0].ifname[1] = 'o';
        ifaddr_infos[0].ifname[2] = '\0';
        ifaddr_infos[0].family = AF_INET6;
        *num = 1;
    } else if (config->nicPosition == 1) {
        CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), DRV_ERROR_WAIT_TIMEOUT);
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[0] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[1] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[2] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[3] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[4] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[5] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[6] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[7] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[8] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[9] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[10] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[11] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[12] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[13] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[14] = 0;
        ifaddr_infos[0].ifaddr.ip.addr6.s6_addr[15] = 1;
        ifaddr_infos[0].ifaddr.mask.s_addr = 0;
        ifaddr_infos[0].ifname[0] = 'e';
        ifaddr_infos[0].ifname[1] = 't';
        ifaddr_infos[0].ifname[1] = 'h';
        ifaddr_infos[0].ifname[1] = '1';
        ifaddr_infos[0].ifname[2] = '\0';
        ifaddr_infos[0].family = AF_INET6;
        *num = 1;
    }
    return DRV_ERROR_NONE;
}

int RaGetInterfaceVersion(unsigned int phy_id, unsigned int interface_opcode, unsigned int* interface_version)
{
    if (interface_opcode == SOCKET_VNIC_IP_INFOS_INTERFACE) {
        *interface_version = 0;
    } else if (interface_opcode == GET_NOTIFY_BA) {
        *interface_version = 2;
    } else {
        *interface_version = 1;
    }
    return DRV_ERROR_NONE;
}

int RaEpollCtlAdd(const void* fd_handle, RaEpollEvent event) { return 0; }

int RaEpollCtlMod(const void* fd_handle, RaEpollEvent event) { return 0; }

int RaEpollCtlDel(const void* fd_handle) { return 0; }

int RaSocketGetVnicIpInfos(
    unsigned int phy_id, enum IdType type, unsigned int* ids, unsigned int num, struct IpInfo* infos)
{
    if (type == IdType::PHY_ID_VNIC_IP) {
        infos[0].ip.addr.s_addr = 2;
        infos[0].family = AF_INET;
    } else {
        infos[0].ip.addr.s_addr = 0x130007f;
        infos[0].family = AF_INET;
    }
    return 0;
}

int RaRdevGetSupportLite(void* rdma_handle, int* support_lite) { return 0; }

tsd::TSD_StatusT TsdOpen(const uint32_t phyDeviceId, const uint32_t rankSize)
{
    if (rankSize >= 2) {
        hccpThreadStatus = 1;
    }
    return tsd::TSD_OK;
}

rtError_t rtOpenNetService(rtNetServiceOpenArgs* openArgs)
{
    hccpThreadStatus = 1;
    return ACL_RT_SUCCESS;
}

rtError_t rtCloseNetService()
{
    hccpThreadStatus = 0;
    return ACL_RT_SUCCESS;
}

tsd::TSD_StatusT TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs* openArgs) { return tsd::TSD_OK; }

tsd::TSD_StatusT TsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr) { return tsd::TSD_OK; }

tsd::TSD_StatusT
ProcessCloseSubProcList(const uint32_t logicDeviceId, const ProcStatusParam* closeList, const uint32_t listSize)
{
    return tsd::TSD_OK;
}

tsd::TSD_StatusT TsdClose(const uint32_t phyDeviceId)
{
    hccpThreadStatus = 0;
    return tsd::TSD_OK;
}

drvError_t halEschedAttachDevice(unsigned int devId) { return DRV_ERROR_NONE; }

drvError_t halEschedDettachDevice(unsigned int devId) { return DRV_ERROR_NONE; }

drvError_t halEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type) { return DRV_ERROR_NONE; }
drvError_t halEschedCreateGrpEx(unsigned int devId, struct esched_grp_para* grpPara, unsigned int* grpId)
{
    return DRV_ERROR_NONE;
}

drvError_t
halEschedSubscribeEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedQueryInfo(
    unsigned int devId, ESCHED_QUERY_TYPE type, struct esched_input_info* input, struct esched_output_info* output)
{
    return DRV_ERROR_NONE;
}

drvError_t
halEschedWaitEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout, struct event_info* event)
{
    return DRV_ERROR_NONE;
}

drvError_t halEschedRegisterAckFunc(
    unsigned int grpId, EVENT_ID event_id,
    void (*ackFunc)(unsigned int devId, unsigned int subevent_id, char* msg, unsigned int msgLen))
{
    return DRV_ERROR_NONE;
}

drvError_t halGetAPIVersion(int* halAPIVersion)
{
    if (halAPIVersion == nullptr) {
        return DRV_ERROR_INVALID_VALUE;
    }
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevNum(uint32_t* num)
{
    *num = 1;
    return DRV_ERROR_NONE;
}

extern DevType g_stubDevType;
drvError_t halGetChipInfo(uint32_t devId, halChipInfo* chipInfo)
{
    if (g_stubDevType == DevType::DEV_TYPE_910B) {
        static halChipInfo info = {"Ascend", "910B1", "0"};
        *chipInfo = info;
    } else if (g_stubDevType == DevType::DEV_TYPE_310P1 || g_stubDevType == DevType::DEV_TYPE_310P3) {
        static halChipInfo info = {"Ascend", "310P1", "0"};
        *chipInfo = info;
    } else if (g_stubDevType == DevType::DEV_TYPE_910_93) {
        static halChipInfo info = {"Ascend", "910_9391", "0"};
        *chipInfo = info;
    } else if (g_stubDevType == DevType::DEV_TYPE_910) {
        static halChipInfo info = {"Ascend", "910", "0"};
        *chipInfo = info;
    } else {
        static halChipInfo info = {"Ascend", "910B1", "0"};
        *chipInfo = info;
    }
    return DRV_ERROR_NONE;
}

drvError_t halHostRegister(void* srcPtr, UINT64 size, UINT32 flag, UINT32 devid, void** dstPtr)
{
    *dstPtr = srcPtr;
    return DRV_ERROR_NONE;
}

drvError_t halHostUnregister(void* hostPtr, u32 devid) { return DRV_ERROR_NONE; }

drvError_t halHostUnregisterEx(void* hostPtr, u32 devid, u32 flag) { return DRV_ERROR_NONE; }

drvError_t halMemCtl(int type, void* param_value, size_t param_value_size, void* out_value, size_t* out_size_ret)
{
    if (param_value_size >= sizeof(supportFeaturePara)) {
        supportFeaturePara* ptr = static_cast<supportFeaturePara*>(out_value);
        ptr->support_feature = CTRL_SUPPORT_PCIE_BAR_MEM_MASK;
    }
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeRegister(uint32_t devId, struct halSensorNodeCfg* cfg, uint64_t* handle)
{
    *handle = 1;
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUnregister(uint32_t devId, uint64_t handle) { return DRV_ERROR_NONE; }

drvError_t halSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    return DRV_ERROR_NONE;
}

drvError_t drvGetPlatformInfo(uint32_t* info)
{
    *info = 0;
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    if (moduleType == MODULE_TYPE_SYSTEM && infoType == INFO_TYPE_VERSION) {
        *value = 1280; // 芯片类型910B 910_93
    } else {
        *value = 512;
    }

    return DRV_ERROR_NONE;
}

std::map<u32, u32> g_eventRecord;
drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary* event)
{
    HcclEventMsg* msg = reinterpret_cast<HcclEventMsg*>(event->msg);
    g_eventRecord[msg->hcclEventType] = 1;
    return DRV_ERROR_NONE;
}

int halGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen)
{
    if (cmd == GRP_QUERY_GROUPS_OF_PROCESS) {
        GroupQueryOutput* outTmpBuf = reinterpret_cast<GroupQueryOutput*>(outBuff);
        *outLen = sizeof(GrpQueryGroupsOfProcInfo);
        strcpy(outTmpBuf->grpQueryGroupsOfProcInfo[0].groupName, "test");
    } else {
        *outLen = 0;
    }
    return DRV_ERROR_NONE;
}

drvError_t halSdmaCopy(DVdeviceptr dst, size_t dst_size, DVdeviceptr src, size_t len)
{
    (void)memcpy_s((void*)dst, dst_size, (void*)src, len);
    return (drvError_t)(0);
}

drvError_t halSdmaBatchCopy(void* dst[], void* src[], size_t size[], int count)
{
    for (int i = 0; i < count; i++) {
        printf("dst[%llu], srv[%llu], len[%llu], i[%ld]", dst[i], src[i], size[i], i);
        (void)memcpy_s(dst[i], size[i], src[i], size[i]);
    }
    return (drvError_t)(0);
}

HcclResult WaitHalEvent(u32 hcclEventType)
{
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(30);
    while (true) {
        bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
        if (bTimeout) {
            HCCL_ERROR("heterog send recv time out");
            return HCCL_E_PARA;
        }
        auto iter = g_eventRecord.find(hcclEventType);
        if (iter != g_eventRecord.end()) {
            if (iter->second != 0) {
                break;
            }
        }
        SaluSleep(LLT_ONE_MILLISECOND_OF_USLEEP);
    }
    return HCCL_SUCCESS;
}

HcclResult ResetHalEvent(u32 hcclEventType)
{
    auto iter = g_eventRecord.find(hcclEventType);
    if (iter != g_eventRecord.end()) {
        iter->second = 0;
    }
    return HCCL_SUCCESS;
}

HcclResult ClearHalEvent()
{
    g_eventRecord.clear();
    return HCCL_SUCCESS;
}

#define MEMCPY_SIZE_MAX (2 * 1024 * 1024 * 1024UL - 1)

drvError_t drvMemcpy(DVdeviceptr dst, size_t destMax, DVdeviceptr src, size_t ByteCount)
{
    int ret = 0;
    int dstSize, srcSize;
    size_t tmp;

    if (!dst || !src || (ByteCount == 0) || (destMax == 0)) {
        HCCL_ERROR("Invalid args. [dst=%lx, dstmax=%lx, src=%lx, count=%lx]", dst, destMax, src, ByteCount);
        return DRV_ERROR_INVALID_VALUE;
    }

    while ((ByteCount) && (ret == 0)) {
        if (ByteCount <= MEMCPY_SIZE_MAX) {
            dstSize = destMax;
            srcSize = ByteCount;
            ret = memcpy_s((void*)dst, dstSize, (void*)src, srcSize);
            tmp = ByteCount;
        } else {
            ret = memcpy_s((void*)dst, MEMCPY_SIZE_MAX, (void*)src, MEMCPY_SIZE_MAX);
            dst += MEMCPY_SIZE_MAX;
            src += MEMCPY_SIZE_MAX;
            tmp = MEMCPY_SIZE_MAX;
        }

        ByteCount -= tmp;
        destMax -= tmp;
    }

    if (ret) {
        HCCL_ERROR(
            "Copy memory failed. [dst:%lx, dstmax=%lx, src:%lx, count:%lx, ret:%d]", dst, destMax, src, ByteCount, ret);
        return DRV_ERROR_INVALID_HANDLE;
    }

    return DRV_ERROR_NONE;
}

pid_t drvDeviceGetBareTgid(void) { return getpid(); }

int RaCqCreate(void* rdev_handle, struct CqAttr* attr) { return 0; }

int RaCqDestroy(void* rdev_handle, struct CqAttr* attr) { return 0; }

int RaNormalQpDestroy(void* qp_handle)
{
    if (qp_handle == nullptr) {
        return HCCL_E_PTR;
    }
    return 0;
}

int RaGetLbMax(void* rdevHandle, int* lbMax) { return 0; }

int RaSetQpLbValue(void* qpHandle, int lbValue) { return 0; }

int RaSetQpAttrQos(void* qpHandle, struct QosAttr* attr) { return 0; }

int RaSetQpAttrTimeout(void* qpHandle, u32* timeout) { return 0; }

int RaSetQpAttrRetryCnt(void* qpHandle, u32* retry_cnt) { return 0; }

int RaGetCqeErrInfo(unsigned int phy_id, struct CqeErrInfo* info) { return 0; }

int RaRdevGetCqeErrInfoList(void* rdev_handle, struct CqeErrInfo* infolist, u32* num) { return 0; }

int RaGetQpAttr(void* qp_handle, struct QpAttr* attr) { return 0; }

int RaCreateSrq(const void* rdmaHandle, struct SrqAttr* attr) { return 0; }

int RaDestroySrq(const void*, struct SrqAttr*) { return 0; }
int RaCreateEventHandle(int* event_handle)
{
    RA_CHECK_POINTER_NULL_WITH_RET(event_handle);
    // 1024 specify the max fd num, this arg will be ignored since Linux 2.6.8
    *event_handle = epoll_create(1024);
    if (*event_handle < 0) {
        HCCL_ERROR("create event_handle[%d] error", *event_handle);
        return -EINVAL;
    }

    return 0;
}

int RaCtlEventHandle(int event_handle, const void* fd_handle, int opcode, enum RaEpollEvent event)
{
    int ret;
    int fd = -1;
    int tmpEvent;

    if (event_handle < 0) {
        HCCL_ERROR("[ra_ctl_event_handle]event_handle[%d] is invalid", event_handle);
        return -EINVAL;
    }
    RA_CHECK_POINTER_NULL_WITH_RET(fd_handle);
    if (opcode != EPOLL_CTL_ADD && opcode != EPOLL_CTL_DEL && opcode != EPOLL_CTL_MOD) {
        HCCL_ERROR(
            "[ra_ctl_event_handle]opcode[%d] invalid, valid opcode includes {%d, %d, %d}", opcode, EPOLL_CTL_ADD,
            EPOLL_CTL_DEL, EPOLL_CTL_MOD);
        return -EINVAL;
    }

    if (event == RA_EPOLLONESHOT) {
        tmpEvent = EPOLLIN | EPOLLET | EPOLLONESHOT;
    } else if (event == RA_EPOLLIN) {
        tmpEvent = EPOLLIN;
    } else if (event == RA_EPOLLOUT) {
        tmpEvent = EPOLLOUT;
    } else {
        HCCL_ERROR("[ra_ctl_event_handle]unknown event[%d]", event);
        return -EINVAL;
    }

    tmpEvent = (int)((unsigned int)tmpEvent | EPOLLRDHUP);
    fd = ((struct socket_peer_info*)fd_handle)->fd;

    struct epoll_event ev;
    ev.events = tmpEvent;
    ev.data.ptr = (void*)fd_handle;
    ret = epoll_ctl(event_handle, opcode, fd, &ev);
    if (ret) {
        HCCL_WARNING("epoll_ctl for fd %d failed! ret:%d errno:%d op:%d state:%d", fd, ret, errno, opcode, tmpEvent);
    }

    return ret;
}

int RaWaitEventHandle(
    int event_handle, struct SocketEventInfoT* event_infos, int timeout, unsigned int maxevents,
    unsigned int* events_num)
{
    int event_count;

    RA_CHECK_POINTER_NULL_WITH_RET(event_infos);
    RA_CHECK_POINTER_NULL_WITH_RET(events_num);

    event_count = epoll_wait(event_handle, (struct epoll_event*)event_infos, maxevents, timeout);
    if (event_count < 0) {
        HCCL_ERROR("[ra_wait_event_handle]epoll_wait failed, strerror[%s]", strerror(errno));
        return -EIO;
    }

    *events_num = (unsigned int)event_count;
    return 0;
}

int RaDestroyEventHandle(int* event_handle)
{
    RA_CHECK_POINTER_NULL_WITH_RET(event_handle);

    int ret = close(*event_handle);
    *event_handle = -1;
    return ret;
}

int RaTypicalQpCreate(void* rdev_handle, int flag, int qp_mode, struct TypicalQp* qp_info, void** qp_handle)
{
    return 0;
}

int RaTypicalQpModify(void* qp_handle, struct TypicalQp* local_qp_info, struct TypicalQp* remote_qp_info) { return 0; }

int RaTypicalSendWr(void* qp_handle, struct SendWr* wr, struct SendWrRsp* op_rsp) { return 0; }

// int ra_get_device_capability(const void*, struct device_cap_info *dev_cap_info)
// {
//     static struct device_cap_info cap_info = {0};
//     cap_info.max_cqe = 65535;
//     *dev_cap_info = cap_info;
//     return 0;
// }

int RaRdevGetPortStatus(void* rdmaHandle, enum PortStatus* status) { return 0; }

int RaRemapMr(const void* rdmaHandle, struct MemRemapInfo info[], unsigned int num) { return 0; }

int RaGetTlsEnable(struct RaInfo* info, bool* tls_enable) { return 0; }

int RaNormalQpCreate(void* rdev_handle, struct ibv_qp_init_attr* qp_init_attr, void** qp_handle, void** qp)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    if (GetInfoFromHandle(rdev_handle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle is null");
        return -1;
    };
    s32 qp_cnt = 0;
    {
        /** 此处可能会与并发，加锁 */
        std::unique_lock<std::mutex> lock(g_qpMutex);
        for (; qp_cnt < QP_MAX; qp_cnt++) {
            /*cn为线程资源，暂不考虑竞争问题*/
            if (!cn[qp_cnt].set_flag) {
                break;
            }
        }
        HCCL_INFO("TMP_han qp_cnt[%d] QP_MAX[%d] qp_mode[%d]", qp_cnt, QP_MAX, g_qp_mode);
        if (qp_cnt == QP_MAX) {
            HCCL_ERROR("no available qp");
            return -1;
        }
        sal_memset(&cn[qp_cnt], sizeof(struct cn_info), 0, sizeof(struct cn_info));

        cn[qp_cnt].local_port = -1;
        cn[qp_cnt].qpn = qp_cnt;
        cn[qp_cnt].dev_id = idx;
        // 引用计数自增
        dev_flag[idx] += 1;

        // 启动后台任务,检视对方发动的指令
        char thread_name[128] = {0};
        if (-1
            == snprintf_s(
                thread_name, sizeof(thread_name), SalStrLen("hccl-gdr-stub_thread") + 3 + 10 + 2 + 2 + 1 + 64 + 3,
                //"%s-%10u-%02d-%02d",
                "%s-%d-%02d-%s-%d", "hccl-gdr-stub_thread",
                // localIpAddr,
                idx, cn[qp_cnt].qpn, g_shm_name, dev_flag[idx])) {
            HCCL_ERROR("thread name construct error");
            return -1;
        }

        HCCL_INFO("qp_cnt=%d, thread_name=%s", qp_cnt, thread_name);
        cn[qp_cnt].qp.thread_id = sal_thread_create(thread_name, event_process, &cn[qp_cnt]);
        if (NULL == cn[qp_cnt].qp.thread_id) {
            // 任务启动失败
            HCCL_ERROR("Create Thread failed");
            cn[qp_cnt].qp.thread_id = NULL;
            return -1;
        }

        cn[qp_cnt].set_flag = true;
        cn[qp_cnt].thread_run_flag = true;
        qp_index = qp_cnt; // 记录本次qp索引，后面按照索引查找对应的qp信息

        /** 释放锁 */
    }

    /** 将创建shm的动作放到create_qp里, 创建shm时必须等到对端也创建好, 才能返回
        否则可能对端还没创建好, 本端访问空指针 */

    // 查找TID MAP, 找到自己的server_ip作为shm_name的一部分
    u32 server_ip = 0;
    u32 rankId_server = 0;
    if (GetDevicePlaneId(device_id, rankId_server)
        != HCCL_SUCCESS) { // 如果此device没有设置网络平面，则默认按照deviceID
        rankId_server = device_id;
    }
    char shm_name[128] = {0};
    ++thread_entry_times;
    if (-1
        == snprintf_s(
            shm_name, sizeof(shm_name), SalStrLen("hccl-gdr-shm-stub") + 8 * 3 + 1 + 1 + 1 + 64 + 3,
            "%s-%08x-%08x-%08x-%s-%d", "hccl-gdr-shm-stub", rankId_server, server_ip, thread_entry_times, g_shm_name,
            0)) {
        HCCL_ERROR("shm name construct error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    HCCL_INFO(
        "###shm name[%s], qpcnt:[%d], pid:[%d], tid:[%d], entrytime:[%d]", shm_name, qp_cnt, SalGetPid(), SalGetTid(),
        thread_entry_times);

    struct qp_msg* qp_shm
        = (struct qp_msg*)sal_share_memory_create(shm_name, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));
    if (NULL == qp_shm) {
        HCCL_ERROR("shm allocate error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }
    /** 创建访问redis的互斥锁 */
    u64 ref_cnt = __sync_fetch_and_add(&(qp_shm->ref_cnt), 1);
    HCCL_INFO("ref_cnt[%d]", ref_cnt);

    /*shm 只允许两端连接*/
    if (ref_cnt == 0) {
        /** 第一个竞争胜利者 */
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;

        // mali added on Mar.2nd, 2019 for QP connectiion complete
        // u64 try_count = 200;
        // while (cn[qp_cnt].qp_shm->ref_cnt < 2 && try_count > 0) {
        //     HCCL_INFO("wait for peer up, qp_shm->ref_cnt[%lu],shm_name:[%s]", cn[qp_cnt].qp_shm->ref_cnt,shm_name);
        //     SaluSleep(100000);
        //     try_count--;
        // }
        // if (try_count == 0) {
        //     HCCL_ERROR("wait for peer up timeout,shm_name[%s]",shm_name);
        //     (void)sal_share_memory_destroy(qp_shm);
        //     (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        //     return -1;
        // }

    } else if (ref_cnt == 1) {
        /** 后续的访问者 */
        HCCL_INFO("peer ok,shm_name:[%s]", shm_name);
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist,ref_cnt=%lu,shm_name:[%s]", ref_cnt, shm_name);
        (void)__sync_fetch_and_sub(&(qp_shm->ref_cnt), 1);

        (void)sal_share_memory_destroy(qp_shm);
        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    *qp_handle = &cn[qp_cnt];
    return 0;
}

int RaSocketAcceptCreditAdd(struct SocketListenInfoT conn[], unsigned int num, unsigned int creditLimit) { return 0; }

int RaCreateCompChannel(const void* rdma_handle, void** comp_channel)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    *comp_channel = (void*)0xabcd;
    return ((rdma_handle == NULL) || (comp_channel == NULL)) ? -1 : 0;
}

int RaDestroyCompChannel(const void* rdma_handle, void* comp_channel)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    return ((rdma_handle == NULL) || (comp_channel == NULL)) ? -1 : 0;
}

const void (*g_raSetTcpRecvCallbackPtr)(const void* fdHandle);
int RaSetTcpRecvCallback(const void* socket_Handle, const void* callback)
{
    g_raSetTcpRecvCallbackPtr = reinterpret_cast<const void (*)(const void*)>(callback);
    return 0;
}

struct socket_peer_info g_fdHandle;
void TcpRecvDataCallbackFunc()
{
    g_fdHandle.fd = 1;
    g_fdHandle.phy_id = 1;
}

// #ifdef __cplusplus
// } // extern "C"
// #endif

int ibv_get_cq_event_stub(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context)
{
    if (!channel)
        return -1;

    return 0;
}

void ibv_ack_cq_events_stub(struct ibv_cq* cq, unsigned int nevents) {}

void ibv_query_qp_stub(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)
{}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType) { return DRV_ERROR_NONE; }

drvError_t drvDeviceGetPhyIdByIndex(unsigned int deviceLogicId, unsigned int* devicePhyId) { return DRV_ERROR_NONE; }

int RaQpCreateWithAttrs(void* rdma_handle, struct QpExtAttrs* qp_attrs, void** qpHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    if (GetInfoFromHandle(rdma_handle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle is null");
        return -1;
    };
    s32 qp_cnt = 0;
    {
        /** 此处可能会与并发，加锁 */
        std::unique_lock<std::mutex> lock(g_qpMutex);
        g_qp_mode = qp_attrs->qpMode;
        for (; qp_cnt < QP_MAX; qp_cnt++) {
            /*cn为线程资源，暂不考虑竞争问题*/
            if (!cn[qp_cnt].set_flag) {
                break;
            }
        }
        HCCL_INFO("TMP_han qp_cnt[%d] QP_MAX[%d] qp_mode[%d]", qp_cnt, QP_MAX, g_qp_mode);
        if (qp_cnt == QP_MAX) {
            HCCL_ERROR("no available qp");
            return -1;
        }

        sal_memset(&cn[qp_cnt], sizeof(struct cn_info), 0, sizeof(struct cn_info));

        cn[qp_cnt].local_port = -1;
        cn[qp_cnt].qpn = qp_cnt;
        cn[qp_cnt].dev_id = idx;
        cn[qp_cnt].qpMode = qp_attrs->qpMode;
        // 引用计数自增
        dev_flag[idx] += 1;

        // 启动后台任务,检视对方发动的指令
        char thread_name[128] = {0};
        if (-1
            == snprintf_s(
                thread_name, sizeof(thread_name), SalStrLen("hccl-gdr-stub_thread") + 3 + 10 + 2 + 2 + 1 + 64 + 3,
                //"%s-%10u-%02d-%02d",
                "%s-%d-%02d-%s-%d", "hccl-gdr-stub_thread",
                // localIpAddr,
                idx, cn[qp_cnt].qpn, g_shm_name, dev_flag[idx])) {
            HCCL_ERROR("thread name construct error");
            return -1;
        }

        HCCL_INFO("qp_cnt=%d, thread_name=%s", qp_cnt, thread_name);
        cn[qp_cnt].qp.thread_id = sal_thread_create(thread_name, event_process, &cn[qp_cnt]);
        if (NULL == cn[qp_cnt].qp.thread_id) {
            // 任务启动失败
            HCCL_ERROR("Create Thread failed");
            cn[qp_cnt].qp.thread_id = NULL;
            return -1;
        }

        cn[qp_cnt].set_flag = true;
        cn[qp_cnt].thread_run_flag = true;
        qp_index = qp_cnt; // 记录本次qp索引，后面按照索引查找对应的qp信息

        /** 释放锁 */
    }

    /** 将创建shm的动作放到create_qp里, 创建shm时必须等到对端也创建好, 才能返回
        否则可能对端还没创建好, 本端访问空指针 */

    // 查找TID MAP, 找到自己的server_ip作为shm_name的一部分
    u32 server_ip = 0;
    u32 rankId_server = 0;
    if (GetDevicePlaneId(device_id, rankId_server)
        != HCCL_SUCCESS) { // 如果此device没有设置网络平面，则默认按照deviceID
        rankId_server = device_id;
    }
    char shm_name[128] = {0};
    ++thread_entry_times;
    if (-1
        == snprintf_s(
            shm_name, sizeof(shm_name), SalStrLen("hccl-gdr-shm-stub") + 8 * 3 + 1 + 1 + 1 + 64 + 3,
            "%s-%08x-%08x-%08x-%s-%d", "hccl-gdr-shm-stub", rankId_server, server_ip, thread_entry_times, g_shm_name,
            0)) {
        HCCL_ERROR("shm name construct error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    HCCL_INFO(
        "###shm name[%s], qpcnt:[%d], pid:[%d], tid:[%d], entrytime:[%d]", shm_name, qp_cnt, SalGetPid(), SalGetTid(),
        thread_entry_times);

    struct qp_msg* qp_shm
        = (struct qp_msg*)sal_share_memory_create(shm_name, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));
    if (nullptr == qp_shm) {
        HCCL_ERROR("shm allocate error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    /** 创建访问redis的互斥锁 */
    u64 ref_cnt = __sync_fetch_and_add(&(qp_shm->ref_cnt), 1);
    HCCL_INFO("ref_cnt[%d]", ref_cnt);

    /*shm 只允许两端连接*/
    if (ref_cnt == 0) {
        /** 第一个竞争胜利者 */
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;

        // mali added on Mar.2nd, 2019 for QP connectiion complete
        u64 try_count = 200;
        while (cn[qp_cnt].qp_shm->ref_cnt < 2 && try_count > 0) {
            HCCL_INFO("wait for peer up, qp_shm->ref_cnt[%lu],shm_name:[%s]", cn[qp_cnt].qp_shm->ref_cnt, shm_name);
            SaluSleep(100000);
            try_count--;
        }
        if (try_count == 0) {
            HCCL_ERROR("wait for peer up timeout,shm_name[%s]", shm_name);
            (void)sal_share_memory_destroy(qp_shm);
            (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
            return -1;
        }
    } else if (ref_cnt == 1) {
        /** 后续的访问者 */
        HCCL_INFO("peer ok,shm_name:[%s]", shm_name);
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist,ref_cnt=%lu,shm_name:[%s]", ref_cnt, shm_name);
        (void)__sync_fetch_and_sub(&(qp_shm->ref_cnt), 1);

        (void)sal_share_memory_destroy(qp_shm);
        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }
    *qpHandle = &cn[qp_cnt];
    return 0;
}

int RaAiQpCreate(void* rdma_handle, struct QpExtAttrs* qp_attrs, struct AiQpInfo* info, void** qpHandle)
{
    CHK_PRT_RET(hccpThreadStatus == 0, HCCL_ERROR("Hccp thread has not been started"), -1);
    u32 device_id = 0, localIp = 0, idx = 0;
    if (GetInfoFromHandle(rdma_handle, device_id, localIp, idx)) {
        HCCL_ERROR("GetInfoFromHandle error, conn.socketHandle is null");
        return -1;
    };
    s32 qp_cnt = 0;
    {
        /** 此处可能会与并发，加锁 */
        std::unique_lock<std::mutex> lock(g_qpMutex);
        g_qp_mode = qp_attrs->qpMode;
        for (; qp_cnt < QP_MAX; qp_cnt++) {
            /*cn为线程资源，暂不考虑竞争问题*/
            if (!cn[qp_cnt].set_flag) {
                break;
            }
        }
        HCCL_INFO("TMP_han qp_cnt[%d] QP_MAX[%d] qp_mode[%d]", qp_cnt, QP_MAX, g_qp_mode);
        if (qp_cnt == QP_MAX) {
            HCCL_ERROR("no available qp");
            return -1;
        }

        sal_memset(&cn[qp_cnt], sizeof(struct cn_info), 0, sizeof(struct cn_info));

        cn[qp_cnt].local_port = -1;
        cn[qp_cnt].qpn = qp_cnt;
        cn[qp_cnt].dev_id = idx;
        cn[qp_cnt].qpMode = qp_attrs->qpMode;
        // 引用计数自增
        dev_flag[idx] += 1;

        // 启动后台任务,检视对方发动的指令
        char thread_name[128] = {0};
        if (-1
            == snprintf_s(
                thread_name, sizeof(thread_name), SalStrLen("hccl-gdr-stub_thread") + 3 + 10 + 2 + 2 + 1 + 64 + 3,
                //"%s-%10u-%02d-%02d",
                "%s-%d-%02d-%s-%d", "hccl-gdr-stub_thread",
                // localIpAddr,
                idx, cn[qp_cnt].qpn, g_shm_name, dev_flag[idx])) {
            HCCL_ERROR("thread name construct error");
            return -1;
        }

        HCCL_INFO("qp_cnt=%d, thread_name=%s", qp_cnt, thread_name);
        cn[qp_cnt].qp.thread_id = sal_thread_create(thread_name, event_process, &cn[qp_cnt]);
        if (NULL == cn[qp_cnt].qp.thread_id) {
            // 任务启动失败
            HCCL_ERROR("Create Thread failed");
            cn[qp_cnt].qp.thread_id = NULL;
            return -1;
        }

        cn[qp_cnt].set_flag = true;
        cn[qp_cnt].thread_run_flag = true;
        qp_index = qp_cnt; // 记录本次qp索引，后面按照索引查找对应的qp信息

        /** 释放锁 */
    }

    /** 将创建shm的动作放到create_qp里, 创建shm时必须等到对端也创建好, 才能返回
        否则可能对端还没创建好, 本端访问空指针 */

    // 查找TID MAP, 找到自己的server_ip作为shm_name的一部分
    u32 server_ip = 0;
    u32 rankId_server = 0;
    if (GetDevicePlaneId(device_id, rankId_server)
        != HCCL_SUCCESS) { // 如果此device没有设置网络平面，则默认按照deviceID
        rankId_server = device_id;
    }
    char shm_name[128] = {0};
    ++thread_entry_times;
    if (-1
        == snprintf_s(
            shm_name, sizeof(shm_name), SalStrLen("hccl-gdr-shm-stub") + 8 * 3 + 1 + 1 + 1 + 64 + 3,
            "%s-%08x-%08x-%08x-%s-%d", "hccl-gdr-shm-stub", rankId_server, server_ip, thread_entry_times, g_shm_name,
            0)) {
        HCCL_ERROR("shm name construct error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    HCCL_INFO(
        "###shm name[%s], qpcnt:[%d], pid:[%d], tid:[%d], entrytime:[%d]", shm_name, qp_cnt, SalGetPid(), SalGetTid(),
        thread_entry_times);

    struct qp_msg* qp_shm
        = (struct qp_msg*)sal_share_memory_create(shm_name, SHM_CN_QP_MSG_MAX * sizeof(struct qp_msg));
    if (nullptr == qp_shm) {
        HCCL_ERROR("shm allocate error");

        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }

    /** 创建访问redis的互斥锁 */
    u64 ref_cnt = __sync_fetch_and_add(&(qp_shm->ref_cnt), 1);
    HCCL_INFO("ref_cnt[%d]", ref_cnt);

    /*shm 只允许两端连接*/
    if (ref_cnt == 0) {
        /** 第一个竞争胜利者 */
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;

        // mali added on Mar.2nd, 2019 for QP connectiion complete
        u64 try_count = 200;
        while (cn[qp_cnt].qp_shm->ref_cnt < 2 && try_count > 0) {
            HCCL_INFO("wait for peer up, qp_shm->ref_cnt[%lu],shm_name:[%s]", cn[qp_cnt].qp_shm->ref_cnt, shm_name);
            SaluSleep(100000);
            try_count--;
        }
        if (try_count == 0) {
            HCCL_ERROR("wait for peer up timeout,shm_name[%s]", shm_name);
            (void)sal_share_memory_destroy(qp_shm);
            (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
            return -1;
        }
    } else if (ref_cnt == 1) {
        /** 后续的访问者 */
        HCCL_INFO("peer ok,shm_name:[%s]", shm_name);
        cn[qp_cnt].qp.local_qp_msg_ptr = &qp_shm[0]; // server排在第一位，client排在第二位
        cn[qp_cnt].qp.remote_qp_msg_ptr = &qp_shm[1];
        cn[qp_cnt].qp.shm_msg_ptr = qp_shm;
        cn[qp_cnt].qp_shm = qp_shm;
    } else {
        // 被占用则unmap这块shm
        HCCL_ERROR("shm exist,ref_cnt=%lu,shm_name:[%s]", ref_cnt, shm_name);
        (void)__sync_fetch_and_sub(&(qp_shm->ref_cnt), 1);

        (void)sal_share_memory_destroy(qp_shm);
        (void)sal_thread_destroy(cn[qp_cnt].qp.thread_id);
        return -1;
    }
    *qpHandle = &cn[qp_cnt];
    info->aiQpAddr = 0x3000;
    info->sqIndex = 1;
    info->dbIndex = 2;
    return 0;
}

int RaSendWrV2(QpHandle qphandle, struct SendWrV2* wr, struct SendWrRsp* rsp) { return 0; }

int RaSendNormalWrlist(
    QpHandle qphandle, struct WrInfo wr[], struct SendWrRsp op_rsp[], unsigned int send_num, unsigned int* complete_num)
{
    return 0;
}

int RaPollCq(QpHandle qphandle, bool status, unsigned int num, void* ptr) { return 0; }

int RaRecvWrlist(QpHandle handle, struct RecvWrlistData* wr, unsigned int recvNum, unsigned int* completeNum)
{
    return 0;
}

int RaQpBatchModify(RdmaHandle rdmaHandle, QpHandle qpHandle[], unsigned int num, int expectStatus) { return 0; }

int RaQpDestroyWithoutCQ(void* handle) { return 0; }

int RaSendWrVerbs(void* handle, struct SendWrVerbs* wr, struct SendWrRsp* opRsp) { return 0; }

int RaRecvWrVerbs(void* handle, struct RecvWrVerbs* wr) { return 0; }

int RaPollTypicalCq(void* cqHandle, unsigned int num, void* wc) { return 0; }

int RaTypicalCqCreate(void* rdevHandle, unsigned int cqDepth, unsigned int* cqn, void** cqHandle)
{
    *cqn = 0;
    *cqHandle = (void*)0x1;
    return 0;
}

int RaTypicalCqDestroy(void* rdevHandle, unsigned int cqn, void* cqHandle) { return 0; }

int RaQpCreateWithCQWithAttrs(
    void* rdevHandle, struct QpExtAttrs* extAttrs, unsigned int sendCqn, unsigned int recvCqn, void** qpHandle)
{
    *qpHandle = (void*)0x1;
    return 0;
}

HcclResult HcclSocketSendBuff(HcclSocket* obj, const void* data, u64 size)
{
    HCCL_INFO("call HcclSocketSendBuff");
    CHK_RET(hrtRaSocketBlockSend(nullptr, data, size));
    return HCCL_SUCCESS;
}

HcclResult HcclSocketRecvBuff(HcclSocket* obj, void* recvBuf, u32 recvBufLen)
{
    HCCL_INFO("call HcclSocketRecvBuff");
    CHK_RET(hrtRaSocketBlockRecv(nullptr, recvBuf, recvBufLen));
    return HCCL_SUCCESS;
}

HcclResult HcclSocketSendString(HcclSocket* obj, const std::string& sendMsg)
{
    HCCL_INFO("call HcclSocketSendString");
    u32 msgLen = sendMsg.length();
    u8 buff[MAX_MSG_STR_LEN] = {0};
    s32 sRet = strcpy_s(reinterpret_cast<char*>(buff), MAX_MSG_STR_LEN, sendMsg.c_str());
    if (sRet != 0) {
        HCCL_ERROR("[HcclSocket][Send] Block send message length[%u] is illegal", msgLen);
        return HCCL_E_PARA;
    }
    CHK_RET(hrtRaSocketBlockSend(nullptr, buff, MAX_MSG_STR_LEN));
    return HCCL_SUCCESS;
}

HcclResult HcclSocketRecvString(HcclSocket* obj, std::string& recvMsg)
{
    HCCL_INFO("call HcclSocketRecvString");
    recvMsg.clear();
    u8 recvBuf[MAX_MSG_STR_LEN] = {0};
    CHK_RET(hrtRaSocketBlockRecv(nullptr, reinterpret_cast<void*>(recvBuf), MAX_MSG_STR_LEN));
    recvMsg.assign(reinterpret_cast<char*>(recvBuf));
    return HCCL_SUCCESS;
}

int ra_batch_modify_qp(void* rdma_handle, void* qp_handle[], unsigned int num, int expect_status)
{
    HCCL_INFO("call ra_batch_modify_qp");
    return 0;
}

HcclResult MockSendStub(HcclSocket* obj, const void* data, u64 size)
{
    HCCL_INFO("call MockSendStub");
    if (data != nullptr) {
        u8* userData = const_cast<u8*>(reinterpret_cast<const u8*>(data));
        *userData = 1;
    }
    return HCCL_SUCCESS;
}

HcclResult MockRecvStub(HcclSocket* obj, void* data, u32 size)
{
    HCCL_INFO("call MockSendStub");
    if (data != nullptr) {
        u8* userData = reinterpret_cast<u8*>(data);
        *userData = 1;
    }
    return HCCL_SUCCESS;
}

HcclResult MockCreateOneQp(
    TransportIbverbs* obj, s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, bool useAicpu, u32 udpSport)
{
    HCCL_INFO("call MockCreateOneQp");
    return HCCL_SUCCESS;
}

int RaSaveSnapshot(struct RaInfo* info, enum SaveSnapshotAction action)
{
    HCCL_INFO("call %s", __func__);
    return 0;
}

int RaRestoreSnapshot(struct RaInfo* info)
{
    HCCL_INFO("call %s", __func__);
    return 0;
}

void ClearRdevNicInfo()
{
    std::unique_lock<std::mutex> lock(g_raNicMutex);
    for (u32 i = 0; i < DEV_MAX; i++) {
        gRdevNicInfo[i].dev_id = 0;
        gRdevNicInfo[i].local_ip = 0;
        gRdevNicInfo[i].idx = 0;
        gRdevNicInfo[i].pid = 0;
    }
    for (u32 i = 0; i < DEV_MAX; i++) {
        gRdevVnicInfo[i].dev_id = 0;
        gRdevVnicInfo[i].local_ip = 0;
        gRdevVnicInfo[i].idx = 0;
        gRdevVnicInfo[i].pid = 0;
    }
    HCCL_INFO("ClearRdevNicInfo done");
}
