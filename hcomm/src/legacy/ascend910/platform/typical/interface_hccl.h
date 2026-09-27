/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * 该特性代码不涉及开源
 */

#ifndef INTERFACE_HCCL_H
#define INTERFACE_HCCL_H

#include "acl/acl_base.h"
#include "stream_pub.h"
#include "hccl/base.h"

#define GID_LENGTH 16

#ifdef __cplusplus
extern "C" {
#endif

typedef struct AscendQPInfoDef {
    uint32_t qpn;
    uint32_t gidIdx;
    uint8_t gid[GID_LENGTH];
    uint32_t psn;
    uint32_t sq_depth;
    uint32_t rq_depth;
    uint32_t scq_depth;
    uint32_t rcq_depth;
    uint64_t reserved[32];
} AscendQPInfo;

typedef struct AscendQPQosDef {
    uint32_t tc;
    uint32_t sl;
} AscendQPQos;

/**
 * @brief 异构场景RDMA设备初始化接口
 */
HcclResult hcclAscendRdmaInit();

/**
 * @brief 异构场景RDMA设备解初始化接口
 */
HcclResult hcclAscendRdmaDeInit();

/**
 * @brief 异构场景QP创建接口
 * @param ascendQPInfo (out): 本地QP信息。
 */
HcclResult hcclCreateAscendQP(AscendQPInfo* ascendQPInfo);

/**
 * @brief 异构场景QP带参数创建接口
 * @param ascendQPInfo (in & out): 本地QP信息和参数配置。
 */
HcclResult hcclCreateAscendQPWithAttr(AscendQPInfo* ascendQPInfo);

/**
 * @brief 异构场景QP状态迁移
 * @param localQPInfo (in): 本地QP信息。
 * @param remoteQPInfo (in): 远端QP信息。
 */
HcclResult hcclModifyAscendQP(AscendQPInfo* localQPInfo, AscendQPInfo* remoteQPInfo);

/**
 * @brief 异构场景QP状态迁移扩展，支持用户参数配置qpQos，如tc、sl
 * @param localQPInfo (in): 本端QP信息。
 * @param remoteQPInfo (in): 远端QP信息。
 * @param qpQos (in): QP qos信息。
 */
HcclResult hcclModifyAscendQPEx(AscendQPInfo* localQPInfo, AscendQPInfo* remoteQPInfo, AscendQPQos* qpQos);

/**
 * @brief 异构场景QP销毁接口
 * @param ascendQPInfo (in): 本端待销毁QP信息。
 */
HcclResult hcclDestroyAscendQP(AscendQPInfo* ascendQPInfo);

typedef struct AscendMrInfoDef {
    uint64_t addr; // in: starting address of mr
    uint64_t size; // in: size of mr
    uint32_t key;  // out: local addr access key
} AscendMrInfo;

/**
 * @brief 异构场景数据内存申请
 * @param ptr (in): 基地址
 * @param size (in): 长度
 */
HcclResult hcclAllocWindowMem(void** ptr, uint64_t size);

/**
 * @brief 异构场景数据内存释放
 * @param ptr (in): 基地址
 */
HcclResult hcclFreeWindowMem(void* ptr);

/**
 * @brief 异构场景同步内存申请
 * @param ptr (in): 基地址
 */
HcclResult hcclAllocSyncMem(int32_t** ptr);

/**
 * @brief 异构场景同步内存释放
 * @param ptr (in): 基地址
 */
HcclResult hcclFreeSyncMem(int32_t* ptr);

/**
 * @brief 异构场景内存（数据内存和同步内存）注册
 * @param memInfo (in\out): 注册内存信息
 */
HcclResult hcclRegisterMem(AscendMrInfo* memInfo);

/**
 * @brief 异构场景内存（数据内存和同步内存）解注册
 * @param memInfo (in): 注册内存信息
 */
HcclResult hcclDeRegisterMem(AscendMrInfo* memInfo);

typedef struct AscendSendRecvInfoDef {
    AscendQPInfo* localQPinfo;
    AscendMrInfo* localWindowMem;
    AscendMrInfo* remoteWindowMem;
    AscendMrInfo* localSyncMemPrepare;
    AscendMrInfo* localSyncMemDone;
    AscendMrInfo* localSyncMemAck;
    AscendMrInfo* remoteSyncMemPrepare;
    AscendMrInfo* remoteSyncMemDone;
    AscendMrInfo* remoteSyncMemAck;
    uint32_t immData = 0; // 默认为0， 不发送立即数，当imma不为0时，支持按照immData作为立即数发送
} AscendSendRecvInfo;

typedef struct AscendSendRecvLinkInfoDef {
    AscendQPInfo* localQPinfo;
    AscendMrInfo* localSyncMemPrepare;
    AscendMrInfo* localSyncMemDone;
    AscendMrInfo* localSyncMemAck;
    AscendMrInfo* remoteSyncMemPrepare;
    AscendMrInfo* remoteSyncMemDone;
    AscendMrInfo* remoteSyncMemAck;
    uint32_t immData = 0; // 默认为0， 不发送立即数，当imma不为0时，支持按照immData作为立即数发送
    uint32_t wqePerDoorbell; // 下发多少个wr之后敲一次doorbell，必须小于等于300
} AscendSendRecvLinkInfo;

/**
 * @brief 异构场景点对点通信接口-发送
 * @param sendBuf (in): 发送内存基地址
 * @param count (in): 数据个数
 * @param dataType (in): 数据类型
 * @param sendRecvInfo (in): QP\window内存\同步内存信息
 * @param stream (in): 异步执行stream
 */
HcclResult HcclSendByAscendQP(
    void* sendBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream);

/**
 * @brief 异构场景点对点通信接口-接收
 * @param recvBuf (out): 接收内存基地址
 * @param count (in): 数据个数
 * @param dataType (in): 数据类型
 * @param sendRecvInfo (in): QP\window内存\同步内存信息
 * @param stream (in): 异步执行stream
 */
HcclResult HcclRecvByAscendQP(
    void* recvBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream);

struct HcclErrCqeInfo {
    uint32_t status;
    uint32_t qpn;
    struct timeval time;
};

/**
 * @brief 异构场景QP状态迁移接口，并支持按照QP粒度设置RDMA qos，包括sl、tc。
 * @param localQPInfo (in): 本地QP信息。
 * @param remoteQPInfo (in):对端QP信息。
 * @param remoteQPInfo (in):对本端QP配置Qos，包含tc，sl值。tc值有效范围[0，255]，且必须是4的倍数。sl值有效范围[0，7]。
 */
HcclResult HcclGetCqeErrInfoList(struct HcclErrCqeInfo* infoList, uint32_t* num);

/**
 * @brief 按照QP粒度获取RDMA error cqe 信息。
 * @param qpn (in): qp编号
 * @param infoList (out): error cqe 数组指针。
 * @param num (in/out):获取error cqe的个数，不大于128。初始值配置必须大于0。
 */
HcclResult HcclGetCqeErrInfoListByQpn(uint32_t qpn, struct HcclErrCqeInfo* infoList, uint32_t* num);

/**
 * @brief 异构场景单边通信接口，put数据到对端。
 * @param num (in):要发送的数据组数
 * @param count(in):向对端发送的数据的数量
 * @param dataType(in):对端接收数据的类型
 * @param sendRecvInfo: AscendSendRecvLinkInfo指针，包含的信息有本端的qp信息，本端和对端的syncmem信息
 * @param stream：异步执行stream
 */
HcclResult HcclPutByAscendQP(
    void* putBuf, uint64_t count, HcclDataType dataType, AscendSendRecvInfo* sendRecvInfo, aclrtStream stream);

/**
 * @brief 异构场景单边通信接口，批量put数据到对端。
 * @param num (in):要发送的数据组数
 * @param putMRList(in):向对端发送的数据的mr信息的数组指针
 * @param remoteMRList(in):对端接收数据的mr信息的数组指针
 * @param sendRecvLinkInfo:
 * AscendSendRecvLinkInfo指针，包含的信息有本端的qp信息，本端和对端的syncmem信息和要发送的立即数以及敲doorbell的频次
 * @param stream：异步执行stream
 */
HcclResult HcclBatchPutMRByAscendQP(
    unsigned int num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList, AscendSendRecvLinkInfo* sendRecvLinkInfo,
    aclrtStream stream);

HcclResult HcclWaitPutMRByAscendQP(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream);

HcclResult HcclWaitPutMRDoWait(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream);

HcclResult HcclWaitPutMRDoRecord(AscendSendRecvLinkInfo* sendRecvLinkInfo, aclrtStream stream);

HcclResult hcclGetSyncMemRegKey(AscendMrInfo* memInfo);

typedef struct AscendSendLinkInfoDef {
    AscendQPInfo* localQPinfo;          // 本端的QP内存信息
    AscendMrInfo* remoteNotifyValueMem; // 内存值为1的远端内存
    AscendMrInfo* localSyncMemAck;      // 本端的notifywait
    uint32_t wqePerDoorbell;            // 下发多少个wr之后敲一次doorbell，必须小于等于300
} AscendSendLinkInfo;

HcclResult HcclOneSideBatchPutByAscendQP(
    unsigned int num, AscendMrInfo* putMRList, AscendMrInfo* remoteMRList, AscendSendLinkInfo* sendlinkInfo,
    aclrtStream stream);

typedef struct AscendCQInfoDef {
    uint32_t cqn;     // out: cq number
    uint32_t cqDepth; // in & out: cq depth
    uint64_t reserved[32];
} AscendCQInfo;

typedef struct AscendMrAttrDef {
    uint64_t addr; // in: starting address of mr
    uint64_t size; // in: size of mr
    uint32_t lkey; // out: local addr access key
    uint32_t rkey; // out: remote addr access key
} AscendMrAttr;

struct AscendQPCap {
    uint32_t maxSendWr;
    uint32_t maxRecvWr;
    uint32_t maxSendSge;
    uint32_t maxRecvSge;
    uint32_t maxInlineData;
};

enum AscendQpType {
    ASCEND_QPT_RC = 2,
    ASCEND_QPT_UC,
    ASCEND_QPT_UD,
    ASCEND_QPT_XRC,
    ASCEND_QPT_RAW_PACKET = 8,
    ASCEND_QPT_RAW_ETH = 8
};

typedef struct AscendVerbsQPInfoDef {
    uint32_t qpn;              // out
    uint32_t gidIdx;           // out
    uint8_t gid[GID_LENGTH];   // out
    uint32_t psn;              // out
    struct AscendQPCap cap;    // in
    enum AscendQpType qp_type; // in
    int sqSigAll;              // in
    uint64_t reserved[32];
} AscendVerbsQPInfo;

struct AscendSge {
    uint64_t addr; /**< address of buf */
    uint32_t len;  /**< len of buf */
    uint32_t lkey; /**< local addr access key */
};

enum AscendWrOpcode {
    ASCEND_WR_RDMA_WRITE = 0,
    ASCEND_WR_RDMA_WRITE_WITH_IMM = 1,
    ASCEND_WR_SEND = 2,
    ASCEND_WR_SEND_WITH_IMM = 3,
    ASCEND_WR_RDMA_READ = 4
};

enum AscendSendFlags {
    ASCEND_SEND_FENCE = 1 << 0,
    ASCEND_SEND_SIGNALED = 1 << 1,
    ASCEND_SEND_SOLICITED = 1 << 2,
    ASCEND_SEND_INLINE = 1 << 3,
    ASCEND_SEND_IP_CSUM = 1 << 4
};

struct AscendSendWr {
    uint64_t wrId;                  // user assigned work request ID
    struct AscendSendWr* next;      // next work request
    struct AscendSge* sgList;       // list of sg
    int numSge;                     // num of Sge
    enum AscendWrOpcode opcode;     // operation type
    enum AscendSendFlags sendFlags; // flags of send wr
    uint32_t immData;               // immediate data (network byte order)
    union {
        struct {
            uint64_t remoteAddr; // remote virtual address
            uint32_t rkey;       // remote key
        } rdma;
        struct {
            uint64_t remoteAddr; // remote virtual address
            uint64_t compareAdd; // compare value (atomic) / add value (fetch add)
            uint64_t swap;       // swap value (atomic)
            uint32_t rkey;       // remote key
        } atomic;
        struct {
            struct AscendAh* ah; // address handle
            uint32_t remoteQpn;  // remote QP number
            uint32_t remoteQkey; // remote QP key
        } ud;
    } wr;
    uint32_t xrcRemoteSrqNum;
};

struct AscendRecvWr {
    uint64_t wrId;             // user assigned work request ID
    struct AscendRecvWr* next; // next work request
    struct AscendSge* sgList;  // list of sg
    int numSge;                // num of Sge
};

enum AscendWcStatus {
    ASCEND_WC_SUCCESS = 0,
    ASCEND_WC_LOC_LEN_ERR,
    ASCEND_WC_LOC_QP_OP_ERR,
    ASCEND_WC_LOC_EEC_OP_ERR,
    ASCEND_WC_LOC_PROT_ERR,
    ASCEND_WC_WR_FLUSH_ERR,
    ASCEND_WC_MW_BIND_ERR,
    ASCEND_WC_BAD_RESP_ERR,
    ASCEND_WC_LOC_ACCESS_ERR,
    ASCEND_WC_REM_INV_REQ_ERR,
    ASCEND_WC_REM_ACCESS_ERR,
    ASCEND_WC_REM_OP_ERR,
    ASCEND_WC_RETRY_EXC_ERR,
    ASCEND_WC_RNR_RETRY_EXC_ERR,
    ASCEND_WC_LOC_RDD_VIOL_ERR,
    ASCEND_WC_REM_INV_RD_REQ_ERR,
    ASCEND_WC_REM_ABORT_ERR,
    ASCEND_WC_INV_EECN_ERR,
    ASCEND_WC_INV_EEC_STATE_ERR,
    ASCEND_WC_FATAL_ERR,
    ASCEND_WC_RESP_TIMEOUT_ERR,
    ASCEND_WC_GENERAL_ERR = 251
};

enum AscendWcOpcode {
    ASCEND_WC_SEND = 0,
    ASCEND_WC_RDMA_WRITE = 1,
    ASCEND_WC_RDMA_READ = 2,
    ASCEND_WC_COMP_SWAP = 3,
    ASCEND_WC_FETCH_ADD = 4,
    ASCEND_WC_BIND_MW = 5,
    ASCEND_WC_LOCAL_INV = 6,
    ASCEND_WC_TSO = 7,
    ASCEND_WC_RECV = 128,
    ASCEND_WC_RECV_RDMA_WITH_IMM = 129
};

enum AscendWcFlags { ASCEND_WC_GRH = 1 << 0, ASCEND_WC_WITH_IMM = 1 << 1 };

struct AscendWc {
    uint64_t wrId;              // work request ID
    enum AscendWcStatus status; // completion status
    enum AscendWcOpcode opcode; // operation type
    uint32_t vendorErr;         // vendor error syndrome
    uint32_t byteLen;           // number of bytes transferred
    uint32_t qpNum;             // QP number
    unsigned int wcFlags;       // completion flags (see AscendWcFlags)
    union {
        uint32_t immData; // immediate data (network byte order)
        uint32_t invalidated_rkey;
    };
    uint32_t srcQp;       // remote QP number (not available from driver, always 0)
    uint16_t pkeyIndex;   // P_Key index (for GSI QPs)
    uint16_t slid;        // source local identifier
    uint8_t sl;           // service level
    uint8_t dlidPathBits; // destination LID path bits
    u32 version;
};

/**
 * @brief 异构场景RDMA设备初始化接口，禁止起PollCq线程轮询Send CQ，调用者需调用hcclPollAscendCQ来Poll Cq
 */
HcclResult hcclAscendRdmaInitV2();

/**
 * @brief 异构场景内存（数据内存和同步内存）注册
 * @param memInfo (in\out): 注册内存信息
 */
HcclResult hcclRdmaMemRegister(AscendMrAttr* memInfo);

/**
 * @brief 异构场景内存（数据内存和同步内存）解注册
 * @param memInfo (in): 注册内存信息
 */
HcclResult hcclRdmaMemDeRegister(AscendMrAttr* memInfo);

/**
 * @brief 异构场景CQ创建接口，支持用户参数配置cqDepth，若不指定，则分配默认值
 * @param ascendCQInfo (in & out): 本地CQ信息。
 */
HcclResult hcclCreateAscendCQWithAttr(AscendCQInfo* ascendCQInfo);

/**
 * @brief 异构场景指定CQ并指定参数创建QP，若不指定参数，则分配默认值
 * @param ascendSendCQInfo (in): 本地send CQ信息。
 * @param ascendRecvCQInfo (in): 本地recv CQ信息。
 * @param ascendQPInfo (in & out): 本地QP信息。
 */
HcclResult hcclCreateAscendQPWithCQWithAttr(
    AscendCQInfo* ascendSendCQInfo, AscendCQInfo* ascendRecvCQInfo, AscendVerbsQPInfo* ascendQPInfo);

/**
 * @brief 异构场景QP状态迁移（Verbs版）
 * @param localQPInfo (in): 本地QP信息。
 * @param remoteQPInfo (in): 远端QP信息。
 */
HcclResult hcclModifyAscendVerbsQP(AscendVerbsQPInfo* localQPInfo, AscendVerbsQPInfo* remoteQPInfo);

/**
 * @brief 异构场景QP状态迁移扩展（Verbs版），支持用户参数配置qpQos，如tc、sl
 * @param localQPInfo (in): 本端QP信息。
 * @param remoteQPInfo (in): 远端QP信息。
 * @param qpQos (in): QP qos信息。
 */
HcclResult
hcclModifyAscendVerbsQPEx(AscendVerbsQPInfo* localQPInfo, AscendVerbsQPInfo* remoteQPInfo, AscendQPQos* qpQos);

/**
 * @brief 异构场景PollCq
 * @param cqInfo (in): 本地CQ信息。
 * @param num (in): 要poll的cqe数量。
 * @param polledNum (out): poll成功的cqe数量。
 * @param wc (out): 要poll的cqe数组指针。
 */
HcclResult hcclPollAscendCQ(AscendCQInfo* cqInfo, uint32_t num, uint32_t* polledNum, struct AscendWc* wc);

/**
 * @brief 异构场景PostSend
 * @param qpInfo (in): 本地QP信息。
 * @param sendWr (in): 发送WR链表头指针。
 * @param stream (in)：用户传入的stream
 * @param badWr (out): 失败的WR头指针。
 */
HcclResult hcclAscendPostSend(
    AscendVerbsQPInfo* qpInfo, struct AscendSendWr* sendWr, aclrtStream stream, struct AscendSendWr** badWr);

/**
 * @brief 异构场景PostRecv
 * @param qpInfo (in): 本地QP信息。
 * @param recvWr (in): 接收WR链表头指针。
 * @param stream (in)：用户传入的stream
 * @param badWr (out): 失败的WR头指针。
 */
HcclResult hcclAscendPostRecv(
    AscendVerbsQPInfo* qpInfo, struct AscendRecvWr* recvWr, aclrtStream stream, struct AscendRecvWr** badWr);

/**
 * @brief 异构场景QP销毁接口（Verbs版）
 * @param ascendQPInfo (in): 本端待销毁QP信息。
 */
HcclResult hcclDestroyAscendVerbsQP(AscendVerbsQPInfo* ascendQPInfo);

/**
 * @brief 异构场景CQ销毁接口
 * @param ascendCQInfo (in): 待销毁的CQ信息。
 */
HcclResult hcclDestroyAscendCQ(AscendCQInfo* ascendCQInfo);

#ifdef __cplusplus
} // extern "C"
#endif

#endif
