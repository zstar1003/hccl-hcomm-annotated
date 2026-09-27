/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*----------------------------------------------*
 * 外部变量说明                                 *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 外部函数原型说明                             *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 内部函数原型说明                             *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 全局变量                                     *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 模块级变量                                   *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 常量定义                                     *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 宏定义                                       *
 *----------------------------------------------*/

#ifndef __LLT_HCCL_STUB_H__
#define __LLT_HCCL_STUB_H__

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <assert.h> /* for assert  */
#include <errno.h>
#include <sys/time.h> /* 获取时间 */

#include <string>
#include <list>
#include <mutex>
#include <string>
#include <map>
#include <atomic>

using std::list;
using std::string;

// #include <cce/hccl_api.h>
#include <dlog_pub.h>
#include "driver/ascend_hal.h"
#ifdef __cplusplus
extern "C" {
#endif /*__cplusplus*/
#include <driver/dsmi_common_interface.h>
#ifdef __cplusplus
}
#endif /*__cplusplus*/
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub_gdr.h"
#include "llt_hccl_stub_profiling_plugin.h"
#include "task_profiling_pub.h"
#define EVENT_UNIQUE_ID_BYTES (SAL_UNIQUE_ID_BYTES + 11)
#define EVENT_UNIQUE_ID_PREFIX "hccl-event-"

/** 符号别名, 用于libibverbs和librdmacm(版本化) */
#ifdef HAVE_SYMVER_SUPPORT
#define symver(name, api, ver) asm(".symver " #name "," #api "@" #ver)
#define default_symver(name, api) asm(".symver " #name "," #api "@@" DEFAULT_ABI)
#else
#define symver(name, api, ver)
#define default_symver(name, api) extern __typeof(name) api __attribute__((alias(#name)))
#endif /* HAVE_SYMVER_SUPPORT */

/** 强, 弱符号别名, 用于桩函数实现 */
#if !defined(weak_alias)
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)));
#endif

#if !defined(strong_alias)
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)));
#endif

namespace cce {
typedef enum tagCcDataType {
    CC_DATA_FLOAT = 0,            /**< float type */
    CC_DATA_HALF,                 /**< fp16 type */
    CC_DATA_INT8,                 /**< int8 type */
    CC_DATA_INT32,                /**< int32 type */
    CC_DATA_UINT8,                /**< uint8 type */
    CC_DATA_HALF_UINT16_PROPOSAL, /**<mixed type for proposal*/
    CC_DATA_INT16,                /**< int16 type */
    CC_DATA_UINT16,               /**< uint16 type */
    CC_DATA_UINT32,               /**< uint32 type */
    CC_DATA_INT64,                /**< int64 type */
    CC_DATA_UINT64,               /**< uint64 type */
    CC_DATA_DOUBLE,               /**< double type */
    CC_DATA_BOOL,                 /**< bool type */
    CC_DATA_DUAL,                 /**< dual output type */
    CC_DATA_DUAL_SUB_INT8,        /**< dual output int8 type */
    CC_DATA_DUAL_SUB_UINT8,       /**< dual output uint8 type */
    CC_DATA_COMPLEX64,
    CC_DATA_COMPLEX128,
    CC_DATA_QINT8,
    CC_DATA_QINT16,
    CC_DATA_QINT32,
    CC_DATA_QUINT8,
    CC_DATA_QUINT16,
    CC_DATA_RESERVED
} ccDataType_t;

typedef enum tagCcStatus {
    CC_STATUS_SUCCESS = 0,         /**< succ */
    CC_STATUS_NOT_INITIALIZED = 1, /**< not init */
    CC_STATUS_ALLOC_FAILED = 2,    /**< alloc mem failed */
    CC_STATUS_BAD_PARAM = 3,       /**< para check failed */
    CC_STATUS_INTERNAL_ERROR = 4,  /**< internal error */
    CC_STATUS_KERNEL_ERROR = 5,    /**< kernel error */
    CC_STATUS_RUNTIME_ERROR = 6,   /**< runtime error */
    CC_STATUS_NOT_SUPPORTED = 7,   /**< unsupported error */
    CC_STATUS_INVALID_VALUE = 7,   /**< invalid value error for blas*/
    CC_STATUS_RESERVED             /**< just for check */
} ccStatus_t;

typedef enum tagCceCRedOp {
    CCE_RED_OP_SUM = 0,
    CCE_RED_OP_PROD = 1,
    CCE_RED_OP_MAX = 2,
    CCE_RED_OP_Min = 3,
    CCE_RED_OP_RESERVED
} ccReduceOp_t;

#ifndef DAVINCI_LITE
ccStatus_t ccVectorReduce(
    const void* src1, const void* src2, uint64_t count, ccDataType_t datatype, ccReduceOp_t op, rtStream_t streamId,
    const void* dst);
#endif
}; // namespace cce

// 定义系统支持的最大设备数量为64，最大设备编号为63
#define MAX_DEVICE_NUM 63
#define MIN_DEVICE_ID 0
namespace Adx {
enum class DumpType : int32_t { OPERATOR = 0x01, EXCEPTION = 0x02, ARGS_EXCEPTION = 0x03, OP_OVERFLOW = 0x04 };
bool AdumpIsDumpEnable(DumpType type);
void AdumpPrintWorkSpace(
    const void* workSpaceAddr, const size_t dumpWorkSpaceSize, rtStream_t stream, const char* opType, bool enableSync);
}; // namespace Adx

// event 结构体
typedef void* hccl_rt_event_t;

// 定义桩函数共享event事件结构体
typedef struct rt_event_share_info_stub_s {
    s32 record_status;
    char sem_unique_id[SAL_SEM_UNIQUE_ID_BYTES];     // 跨进程共享时的唯一句柄
    char mutex_unique_id[SAL_MUTEX_UNIQUE_ID_BYTES]; // 跨进程共享时的唯一句柄
} rt_event_share_info_stub_t;

// 定义桩函数event事件结构体
typedef struct rt_event_stub_s {
    sal_sem_t sem;
    sal_mutex_t mutex;
    void* event_handler; // 用于调测定位时识别不同的event
    rt_event_share_info_stub_t* event_share_info;
} rt_event_stub_t;

// 定义threadlocal结构体
typedef struct task_info_s {
    u32 streamId;
    u32 taskId;
} task_info_t;

// 定义device memory共享内存申请时名字和新名字的映射关系，用于跨进程访问
typedef struct rt_name_map_stub_s {
    char shm_real_name[SAL_DMEM_UNIQUE_ID_BYTES]; // 设备共享内存创建时的名字
    char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES];   // setMemoryName时新的命名
    s32 mem_size;                                 // 内存大小
    s32 valid_flag;                               // 该条map是否有效，0为无效，非0为有效
    u32 offset;
} rt_name_map_stub_t;

//  线程状态定义
enum {
    THREAD_STATE_INITALING = 0, // 任务正在启动
    THREAD_STATE_STOPED,        // 任务未启动或已经停止
    THREAD_STATE_WORKING,       // 任务正在工作
    THREAD_STATE_MAX
};

enum class QosErrorCode : int {
    QOS_SUCCESS = 0,
    QOS_UNINIT_ERROR,
    QOS_INIT_ERROR,
    QOS_ILLEGAL_PARA,
    QOS_NOT_FOUND,
    QOS_UNSUPPORTED,
    QOS_DSMI_ERROR,
    QOS_NOMATCH_MPAMID,
};

enum class QosStreamType : int {
    STREAM_FORWARD_COMPUTE = 0,                          // 前向计算
    STREAM_BACKWARD_COMPUTE = 1,                         // 后向计算
    STREAM_PARAMETER_UPDATE = 2,                         // 参数更新
    STREAM_GRADUATION_AGGREGATION = 3,                   // 梯度聚合
    STREAM_HCCL_MODEL_LAY_PARALLEL_FEATURE_MAP = 4,      // 模型层内并行Feature Map通讯
    STREAM_HCCL_MODEL_PIPELINE_PARALLEL_FEATURE_MAP = 5, // Pipeline模型并行Feature Map通讯
    STREAM_HCCL_PARAMETER_PREFETCH = 6,                  // 数据并行参数预取
    STREAM_HCCL_FEATURE_MAP_PREFETCH = 7,                // 数据并行Feature Map预取
    STREAM_HCCL_FEATURE_MAP_SHARE = 8,                   // 数据并行Feature Map共享
    STREAM_HCCL_EMBEDDING_READ_WRITE = 9,                // 数据并行Embedding Table读写
    STREAM_DVPP_COMPUTE = 10,                            // DVPP计算
    STREAM_L2CACHE_PREFETCH = 11,                        // L2 CACHE预取
    STREAM_L2CACHE_INV_WRB_FLUSH = 12,                   // L2 CACHE Invalid/Writeback/Flush操作
    STREAM_AIV_H2D_COPY = 13,                            // 使用AIV从HOST搬移数据到DEVICE
    STREAM_OTHERS, // AI取指令, STARS读SQE, STARS写CQE, 同步通信, SMMU查页表
    STREAM_INVALID,
    STREAM_MAX
};

enum class QosEngineType : int { AI, HCCL, AICPU, MEMCPYS, CMO };

struct QosConfig {
    unsigned int mpamId;    // MPAMID ，取值范围： 0~127
    unsigned int bwHigh;    // 带宽的高水线
    unsigned int bwLow;     // 带宽的低水线
    unsigned int qos;       // qos优先级，取值范围：0~7
    unsigned int hardlimit; // 是否使能hardlimit，1：使能，0：不使能
    unsigned int pmg;
    unsigned int ns;
    unsigned int mode; // AIC/AIV/SDMA support, 0--reg, 1--smmu, 2--sqe
};

// 终止任务时, 等待任务主导退出的尝试次数
#define THREAD_STOP_COUNTER 20

// 默认周期,100ms.
#define THREAD_DEFAULT_UPDATE_INTERVAL 100000

// 最小周期10ms, 一个tick
#define THREAD_UPDATE_MIN 10000

#define NOTIFY_MAX 1024 // 每个device支持的最大notify个数
#define NOTIFY_SHM_NAME_LEN 128
#define NOTIFY_TIMEOUT_CNT 20000

// 定义桩函数notify事件结构体
typedef struct rt_notify_shm_s {
    s32 device_id;
    char ipc_notify_shm_name[NOTIFY_SHM_NAME_LEN];

    volatile u64 name_flag;
    volatile u64 ref_cnt;

    u64 occupied_flag[NOTIFY_MAX];
    u64 record_cnt[NOTIFY_MAX];
} rt_shm_notify_t;

typedef struct rt_notify_ipc_name_s {
    char ipc_notify_shm_name[NOTIFY_SHM_NAME_LEN];

    u64 notify_id;
} rt_shm_ipc_name_t;

typedef struct rt_notify_s {
    rt_shm_ipc_name_t* ipc_name_shm;
    rt_shm_notify_t* ipc_notify_shm;

    u64 notify_id;
} rt_notify_t;

int RAND_bytes(char* buf, int num);

// Thread类定义
class thread_class {
private:
    u32 uithread_update_interval; // 单位us. Thread状态机刷新时间.StopCallBack()下发后,最长uithread_update_interval us内
    // thread_handler()应该返回, 否则Thread会被强制终止.
    u32 uithreadstate;     // Thread状态机
    sal_thread_t threadfd; // Thread句柄

public:
    thread_class();          // 构造函数, 填充默认值.
    virtual ~thread_class(); // 析构函数, 释放必要资源.

    s32 stop_thread();                    // 停止任务.
    s32 start_thread();                   // 启动任务.
    s32 start_thread(string thread_name); // 启动任务(带线程名).

    s32 set_new_interval(u32 uinewinterval); // 设定新定时器间隔
    u32 get_current_interval();              // 查询当前定时器间隔

    s32 update_thread_state(u32 uinewstate); // 刷新任务状态机

    virtual s32 thread_handler();    // 任务主处理函数
    virtual s32 pre_stop_handler();  // StopThread触发, 通知ThreadHandler主动退出.
    virtual s32 pre_start_handler(); // StartThread触发, 通知ThreadHandler即将被调用.
};

// 线程函数实现
void* threadfun(void* p);

// 互斥锁保护任务队列
#define MSG_LOCK()        \
    if (stream_task_lock) \
    (void)sal_mutex_take(stream_task_lock, SAL_MUTEX_FOREVER)

#define MSG_UNLOCK()      \
    if (stream_task_lock) \
    (void)sal_mutex_give(stream_task_lock)

// memcpyasync参数结构
typedef struct rt_memcpy_async_s {
    void* dst;
    void* src;
    uint64_t count;
} rt_memcpy_async_t;

// vectorreduce参数结构
typedef struct rt_vector_reduce_s {
    void* src1;
    void* src2;
    uint32_t count_reduce;
    cce::ccDataType_t datatype;
    cce::ccReduceOp_t op;
    void* dst_reduce;
} rt_vector_reduce_t;

// #if 0
// RDMASend 参数结构
typedef struct rt_rdma_send_s {
    u32 wqe_index;
    void* cn;
} rt_rdma_send_t;
// #endif

typedef void (*rtCallback_t)(void* fnData);
typedef struct rt_callback_func_stub_s {
    rtCallback_t func;
    void* para;
    u8 isExecuted;
    bool isBlock;
} rt_callback_func_stub_t;

// stream task消息格式
typedef struct stream_task_s {
    tasktype_e task_type;
    u32 task_id; // added for HWTS stub

    union {
        rt_event_stub_t event;
        rt_memcpy_async_t memcpystruct;
        rt_vector_reduce_t reducestruct;
        rt_notify_t* notify;
        rt_rdma_send_t rdmasend;
        rt_callback_func_stub_t callbackTask;
        u32 usec;
    } stream_para;

    stream_task_s() : task_type(TASK_TYPE_RESERVED), task_id((u32)(-1))
    {
        memset(&stream_para, 0, sizeof(stream_para));
        stream_para.callbackTask.isExecuted = 0;
        stream_para.callbackTask.isBlock = false;
    }
} stream_task_t;

#define M_PROF_KERNEL_TASK_NAME_LEN (63)
typedef struct tag_rt_profile_data_head_s {
    u64 rserved;
} rtProfileDataHead_t;
typedef struct ProfileTaskTrack {
    rtProfileDataHead_t head;
    u64 timeStamp;
    u16 eventName;
    u16 taskType;
    u16 streamId;
    u16 taskId;
    u32 thread;
    u32 deviceId;
    char kernelName[M_PROF_KERNEL_TASK_NAME_LEN];
    u8 persistant : 1;
    u8 reserved : 7;
} rtProfTaskTrack_t;

using atomic_ptr_t = std::shared_ptr<std::atomic<u32>>;

// stream类定义
class stream_class : private thread_class {
private:
    sal_mutex_t stream_task_lock; // 互斥锁保护任务队列

    // Thread 实现代码
    s32 thread_handler();    // 任务主处理函数
    s32 pre_stop_handler();  // StopThread触发, 通知ThreadHandler主动退出.
    s32 pre_start_handler(); // StartThread触发, 通知ThreadHandler即将被调用.

    sal_sem_t thread_trigger; // 唤醒任务的信号量
    void trigger_thread();    //  临时唤醒任务

    sal_sem_t stream_task_done; // stream_task_done的信号量 ,每完成一次任务的执行，释放一次信号量

    rtError_t stream_usleep(u32 usec); // 实际任务执行体:桩函数rtStreamUsleep实现内容，完成usleep 任务
    rtError_t event_record(rtEvent_t event); // 实际任务执行体:桩函数rtEventRecord实现内容，完成event record任务
    rtError_t event_multidev_record(
        rtEvent_t event); // 实际任务执行体:桩函数rtMultiDevEventRecord实现内容，完成片间event record任务
    rtError_t event_wait(rtEvent_t event); // 实际任务执行体:桩函数rtStreamWaitEvent实现内容，完成event wait任务
    rtError_t memcpy_async(
        void* dst, void* src,
        uint64_t count); // 实际任务执行体:桩函数rtMemcpyAsync实现内容，完成asynchronous Memcpy任务
    rtError_t notify_record(
        rt_notify_t* notify); // 实际任务执行体:桩函数rtNotifyRecord实现内容，完成asynchronous NotifyRecord任务
    rtError_t
    notify_wait(rt_notify_t* notify); // 实际任务执行体:桩函数rtNotifyWait实现内容，完成asynchronous NotifyWait任务
    template <typename T>
    cce::ccStatus_t reduce_op(T src1, T src2, T* dst, const cce::ccReduceOp_t op);
    cce::ccStatus_t vector_reduce(
        const void* src1, const void* src2, uint32_t count, const cce::ccDataType_t datatype,
        const cce::ccReduceOp_t op, void* dst); // 实际任务执行体:桩函数ccVectorReduce实现内容，完成规约操作任务

public:
    explicit stream_class(s32 device_id); // 构造函数, 填充默认值
    virtual ~stream_class();              // 析构函数, 释放必要资源

    void HWTSLog(const stream_task_t& task, u64 ts_start, u64 duration);
    u64 TimestampNanosecond();
    s32 push_task(stream_task_t* stream_task); // 任务队列压入task函数
    s32 get_stream_id() const;                 // 获取stream_id
    s32 get_device_id() const;                 // 获取device_id

    rtError_t stream_synchronize(); // stream_task_done检查函数
    void set_stream_enabled(bool enabled);
    s32 current_dev; // 当前任务队列归属设备ID

    static rtError_t
    rdma_send(u32 wqe_index, void* cnn); // 实际任务执行体:桩函数rtRDMASend实现内容，完成asynchronous RDMASend任务

    void ExecuteCallbackFunc();
    list<stream_task_t> stream_task_list; // 任务队列
private:
    static std::atomic<s32> streamIdCounter_;
    static std::atomic<u32> taskIdCounter_;
    static std::map<rtStream_t, int32_t> streamMap_;
    static std::mutex mapMutex_;

    // 每个device上创建的stream计数, 最后一个stream析构时做Flush(文件以device为粒度)
    static std::map<s32, atomic_ptr_t> refCountMap_;
    // static std::map<s32, std::unique_ptr<Msprof::Engine::Reporter> > reporterMapRuntime_;
    // static std::map<s32, std::unique_ptr<Msprof::Engine::Reporter> > reporterMapHWTS_;
    static std::array<std::string, 8> lineFeed_;

    s32 deviceId_;
    s32 streamId_;

    // std::unique_ptr<Msprof::Engine::Reporter> reporterRuntme_;
    // std::unique_ptr<Msprof::Engine::Reporter> reporterHWTS_;
    ProfReporterData dataRuntime_;
    ProfReporterData dataHWTS_;

    bool stream_enabled_;
};
// Add for optimize runtime Stub by l on 2018-01-11 Above

#ifdef __cplusplus
extern "C" {
#endif
void rtSetCommonPidMode(bool state);
extern HcclResult rt_get_dev_ip(s32 chipType, s32 devId, u32* ipAddr);

HcclResult GetSocketRole(std::vector<u32>& userRanks, u32 srcRank, u32 destRank, HcclSocketRole& role);
HcclResult GetIntraRankIPInfo(
    u32 localUserRank, std::vector<u32> userRanks, std::vector<s32> deviceIds, std::vector<HcclIpAddress>& localIPs,
    std::map<u32, std::vector<HcclRankLinkInfo>>& dstServerMap,
    std::map<u32, std::vector<HcclRankLinkInfo>>& dstClientMap);
HcclResult CreateIntraExchanger(
    const std::string& commTag, HcclNetDevCtx portCtx, s32 deviceLogicId, u32 devicePhyId, u32 localUserRank,
    const u32 userRankSize, std::vector<s32> deviceIds, std::vector<u32> userRanks, bool isSupportReuse,
    IntraExchanger& exchanger);
HcclResult ConstructNetDevCtx(
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, NICDeployment nicDeploy, s32 deviceLogicId, u32 devicePhyId,
    NicType nicType, HcclIpAddress localIp);
void DeConstructNetDevCtx(
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, NICDeployment nicDeploy, s32 deviceLogicId, u32 devicePhyId);
#ifdef __cplusplus
} // extern "C"
#endif

#endif /* __LLT_HCCL_STUB_H__ */
