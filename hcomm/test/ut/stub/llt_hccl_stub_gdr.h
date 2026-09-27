/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_HCCL_STUB_GDR_H__
#define __LLT_HCCL_STUB_GDR_H__

#include <string.h>
#include "transport_ibverbs_pub.h"
#include <network/hccp_common.h>
#include "tsd/tsd_client.h"
#include "adapter_hal.h"
#include "rt_external.h"
using namespace hccl;

/*-------------shm info format-------------*/
#define QP_CMD_WRITE_MR 0
#define QP_CMD_WRITE_DATA 1

#define QP_MSG_MAX_SIZE 1024 * 1024 * 50
#define SHM_CN_QP_MSG_MAX 2
#define SHM_MAX_CLIENT_PER_SERVER 16
#define SHM_MAX_SOCKET_PER_CLIENT 16
#define EVENT_LOOP_INTER_US (1)
#define REMOTE_LINK_MAX_NUM 8

struct msg_mr_info {
    u64 len;
    s8 data[QP_MSG_MAX_SIZE];
};

struct msg_w {
    void* dst_addr;
    u64 len;
    u32 op;
    s32 send_flags;
    s8 data[QP_MSG_MAX_SIZE];
};

struct qp_msg /*shm上的内存分布*/
{
    volatile u64 ref_cnt;
    s8 cmd;
    u32 cnt;
    u32 rsp_cnt;
    union {
        struct msg_mr_info mr_info;
        struct msg_w write_info;
    } msg;
};

struct sock_server {
    u64 ref_flag; /** 用于判定sock_server的shm是否创建完成 */

    u32 server_ip;
    u32 client_num;

    u32 client_ip[SHM_MAX_SOCKET_PER_CLIENT];
};

/*-------------local info format-------------*/
#define WQE_MAX 128
#define QP_MAX 32

struct send_wr_mgr {
    s8 wqe_set[WQE_MAX];
    struct SendWr wq[WQE_MAX];
};

struct qp_info {
    sal_thread_t thread_id;
    void* shm_msg_ptr;
    struct qp_msg* local_qp_msg_ptr;  /*pointer to local shm msg */
    struct qp_msg* remote_qp_msg_ptr; /*pointer to remote shm msg */
    struct send_wr_mgr send_mr_mgr;   /*send async wqe 管理*/
};

struct local_rev_buff {
    u32 size;
    u32 start_pos;
    s8 buff[QP_MSG_MAX_SIZE];
};

struct cn_info /*建链只有一条*/
{
    bool set_flag;        /*本qp是否被使用*/
    bool thread_run_flag; /*背景线程是否在运行*/
    s32 qpn;
    s32 dev_id;
    u32 local_ip;
    u16 local_port;
    u32 server_ip;
    u16 server_port;
    struct local_rev_buff rev_buff;
    void* notify_addr;
    struct qp_info qp; /*shm相关，使用shm作为进程间数据交换的通道 */
    int qpMode;
    struct qp_msg* qp_shm;
};
struct check_remote_link {
    u32 remoteIpAddr;
    s32 handle;
    s32 checkResult;
    s32 resultHaveGet;
};

struct check_link_socket {
    s32 deviceID;
    s32 remoteNum;
    struct check_remote_link remoteLink[REMOTE_LINK_MAX_NUM];
};

using rdevInfo_t = struct rdevInfo {
    u32 dev_id;
    u32 local_ip;
    u32 idx; // 双网口时，idx = dev_id * 2  或 idx = dev_id * 2 + 1
    u32 pid; // current pid
};

struct socket_peer_info {
    int phy_id;
    int fd;
    void* socket_handle;
};

extern struct socket_peer_info g_fdHandle;

rtError_t rtRDMASend_stub(u32 wqe_index, struct cn_info* cn, rtStream_t stream);
tsd::TSD_StatusT TsdOpen(const uint32_t phyDeviceId, const uint32_t rankSize);
tsd::TSD_StatusT TsdProcessOpen(const uint32_t logicDeviceId, ProcOpenArgs* openArgs);
tsd::TSD_StatusT TsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr);
tsd::TSD_StatusT
ProcessCloseSubProcList(const uint32_t logicDeviceId, const ProcStatusParam* closeList, const uint32_t listSize);

void TcpRecvDataCallbackFunc();

HcclResult WaitHalEvent(u32 hcclEventType);
HcclResult ResetHalEvent(u32 hcclEventType);
HcclResult ClearHalEvent();
int ibv_get_cq_event_stub(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context);
void ibv_ack_cq_events_stub(struct ibv_cq* cq, unsigned int nevents);
void ibv_query_qp_stub(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr);
drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType);

HcclResult HcclSocketSendBuff(HcclSocket* obj, const void* data, u64 size);
HcclResult HcclSocketRecvBuff(HcclSocket* obj, void* recvBuf, u32 recvBufLen);
HcclResult HcclSocketSendString(HcclSocket* obj, const std::string& sendMsg);
HcclResult HcclSocketRecvString(HcclSocket* obj, std::string& recvMsg);
HcclResult MockSendStub(HcclSocket* obj, const void* data, u64 size);
HcclResult MockRecvStub(HcclSocket* obj, void* data, u32 size);
HcclResult MockCreateOneQp(
    TransportIbverbs* obj, s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, bool useAicpu, u32 udpSport);
void ClearRdevNicInfo();
#endif /* __LLT_HCCL_STUB_GDR_H__ */
