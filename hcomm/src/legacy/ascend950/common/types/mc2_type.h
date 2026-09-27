/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MC2_TYPE_H
#define HCCL_MC2_TYPE_H

#include <string>
#include <unordered_map>
#include <vector>
#include "exception_util.h"
#include "ccu_api_exception.h"
#include "op_type.h"
#include "data_type.h"
#include "reduce_op.h"

namespace Hccl {

constexpr uint32_t MC2_RES_CTX_MAX = 3;
constexpr uint32_t CCU_PARAM_NUM_PER_DIE = 32;
constexpr uint32_t CCU_PARAM_NUM_MAX = CCU_PARAM_NUM_PER_DIE * 2;
constexpr uint32_t CCU_ONE_PARAM_SIZE = 8;
constexpr uint32_t CCU_TASK_NUM_MAX = 64;
constexpr uint32_t MAX_RANK_NUM = 64; // 最大卡数
constexpr uint32_t MAX_OP_NUM = 8;    // MC2最大通信算子数

constexpr uint32_t UNKNOWN_TILING_V1 = 3;                 // 旧版本 MC2 Tiling version = 3
constexpr uint32_t UNKNOWN_TILING_V2 = 100;               // 新版本 MC2 Tiling version = 100
constexpr uint64_t MC2_WORKSPACE_SIZE = 16 * 1024 * 1024; // aic与ccu的交互空间workspace大小为16*1024*1024B

struct HcclCommParamDesc {
    uint64_t version : 4;  // 版本号，当前是1
    uint64_t groupNum : 4; // groupMatmul的输入数量，每个group对应一个输入和一个输出地址
    uint64_t hasFfts : 1;  // 910下是否是ffts融合算子（多一个ffts_addr参数
    uint64_t
        tilingDataPtrOff : 7; // tilingdata指针所在的参数索引, 此处修改为tilingDataPtr的Offset，需要二次索引到tilingData
    uint64_t
        isDyn : 48; // 输入参数是否是动态输入，从IR输入开始计算，不包含前面的参数，is_dyn是一个bitmap，每个bit对应一个IR输入，如果是动态输入则为1，否则是0
};

struct KFCTilingData {
    uint32_t preparePosition; // 新增结构体，用来区分是否是高阶api
    uint32_t sendOff;         // 发送数据地址偏移，count * dataTypeSize
    uint32_t recvOff;         // 接收数据地址偏移, count * dataTypeSize
    uint32_t tailSendOff;     // 尾块发送数据地址偏移，count * dataTypeSize
    uint32_t tailRecvOff;     // 尾块发送数据地址偏移，count * dataTypeSize
    uint64_t sendCnt;         // 整块发送数据个数
    uint32_t recvCnt;         // 整块接收数据个数
    uint32_t tailSendCnt;     // 尾块发送数据个数
    uint32_t tailRecvCnt;     // 尾块接收数据个数
    uint32_t totalCnt;        // 总数据个数
    uint32_t turnNum;         // 总轮次
    uint32_t tailNum;         // 尾块的轮次
    uint32_t stride;          // 跳写间隔
    uint32_t workspaceOff;    // 使用workspace作为recvbuf时的workspace偏移
    uint32_t notifyOff;       // device notify write/read value偏移
    uint16_t notifyBeginCnt;  // notify write value的使用个数
    uint16_t notifyEndCnt;    // notify read value的使用个数
    uint8_t useBufferType;    // 是否使用workspace作为recvbuf
    uint8_t funID;            // function ID
    uint8_t dataType;         // hccl 数据类型
    uint8_t groupNum;         // groupNum
    uint8_t reuseMode;        // tiling调试，填msgCnt，内存优化选择复用的内存块个数
    uint8_t commType;         // 通信类型
    uint8_t reduceOp;         // reduce op type
    uint8_t commOrder;        // 通信顺序，0表示通信在前，1表示通信在后
    uint8_t waitPolicy;       // 等待任务启动的阻塞策略
    // 2、首轮等待，1、每轮等待。KFC根据此标记在主流任务前面加wait，AIC需要按策略发对应record才能触发执行
    uint8_t rspPolicy; // 任务执行结束时的响应策略， 2、最后通知一次，
    // 1、每轮通知一次。KFC根据此标记在主流任务后面加record
    uint8_t exitPolicy; // 退出策略，0，一次通信任务下发完成直接退出；1. 通信任务执行完成退出；2.
    // 等待AIC通知退出(可以多次执行任务)。
    uint8_t commAlg;   // 用于指定具体通信算法。
    uint8_t taskType;  // 用于识别不同任务。参考KfcTaskType定义
    uint8_t debugMode; // 调测模式
    // 1:单独执行CUBE
    // 2:单独执行Vector
    // 4:单独执行AICPU KFC算子
    // 8:KFC等待通信结束
    // 16:KFC统计各阶段耗时
    // 32:调试多ccu任务模式，通过ccuNum告诉ccu是几个ccu任务
    uint8_t stepSize;        // 用于指定通算频率步长
    uint8_t sendArgIndex;    // 发送数据参数索引，对应算子原型的参数顺序
    uint8_t recvArgIndex;    // 接收数据参数索引，对应算子原型的参数顺序
    uint8_t commOutArgIndex; // 通信输出参数索引，对应算子原型的参数顺序
    uint8_t hasCommOut;      // 是否有通信输出

    uint8_t reduceOutputDataType; // 输入datatype类型为hif8、fp8、int8 才生效， 输出只有fp16/fp32/bf16类型
    uint32_t workspaceSendOffset;
    uint32_t workspaceRecvOffset;
    uint32_t ccuNum;
    uint32_t ccuParamNum[CCU_TASK_NUM_MAX];
    uint64_t paramAddr[CCU_PARAM_NUM_MAX * CCU_TASK_NUM_MAX];
    uint64_t paramValue[CCU_PARAM_NUM_MAX * CCU_TASK_NUM_MAX];

    std::string ToString() const
    {
        return StringFormat(
            "sendArgIndex = %u\nrecvArgIndex = %u\nsendOff = %u\nrecvOff = %u\ntailSendOff = %u\ntailRecvOff = "
            "%u\nsendCnt = %u\nrecvCnt = %u\ntailSendCnt = %u\ntailRecvCnt = %u\ntotalCnt = %u\nturnNum = %u\ntailNum "
            "= %u\ndataType = %u\ncommType = %u\ncommOutArgIndex = %u\nnotifyBeginCnt = %u\nnotifyEndCnt = "
            "%u\nuseBufferType = %u\nreduceOp = %u\nreduceOutputDataType = %u\nworkspaceSendOffset = "
            "%lu\nworkspaceRecvOffset = %lu",
            sendArgIndex, recvArgIndex, sendOff, recvOff, tailSendOff, tailRecvOff, sendCnt, recvCnt, tailSendCnt,
            tailRecvCnt, totalCnt, turnNum, tailNum, dataType, commType, commOutArgIndex, notifyBeginCnt, notifyEndCnt,
            useBufferType, reduceOp, reduceOutputDataType, workspaceSendOffset, workspaceRecvOffset);
    }
};

enum HcclBufferType {
    HCCL_BUFFER_TYPE_DEFAULT = 0,
    HCCL_BUFFER_TYPE_OUTPUT,
    HCCL_BUFFER_TYPE_WINDOW_IN,
    HCCL_BUFFER_TYPE_WINDOW_OUT,
    HCCL_BUFFER_TYPE_WORKSPACE,
    HCCL_BUFFER_TYPE_INPUT,
    HCCL_BUFFER_TYPE_COMMOUT,
    HCCL_BUFFER_TYPE_SEND_WORKSPACE,
    HCCL_BUFFER_TYPE_RECV_WORKSPACE,
    HCCL_BUFFER_TYPE_SEND_RECV_WORKSPACE,
    HCCL_BUFFER_TYPE_END,
};

enum AicpuComType {
    HCCL_CMD_INVALID = 0,
    HCCL_CMD_BROADCAST = 1,
    HCCL_CMD_ALLREDUCE,
    HCCL_CMD_REDUCE,
    HCCL_CMD_SEND,
    HCCL_CMD_RECEIVE,
    HCCL_CMD_ALLGATHER,
    HCCL_CMD_REDUCE_SCATTER,
    HCCL_CMD_ALLTOALLV,
    HCCL_CMD_ALLTOALLVC,
    HCCL_CMD_ALLTOALL,
    HCCL_CMD_GATHER,
    HCCL_CMD_SCATTER,
    HCCL_CMD_BATCH_SEND_RECV,
    HCCL_CMD_BATCH_PUT,
    HCCL_CMD_BATCH_GET,
    HCCL_CMD_ALLGATHER_V,
    HCCL_CMD_REDUCE_SCATTER_V,
    HCCL_CMD_BATCH_WRITE,
    HCCL_CMD_HALF_ALLTOALLV = 20,
    HCCL_CMD_ALL,
    HCCL_CMD_RESERVED
};

constexpr OpType MC2_OP_TYPE[]
    = {OpType::INVALID,       OpType::BROADCAST, OpType::ALLREDUCE,     OpType::REDUCE,    OpType::SEND,
       OpType::RECV,          OpType::ALLGATHER, OpType::REDUCESCATTER, OpType::ALLTOALLV, OpType::ALLTOALLVC,
       OpType::ALLTOALL,      OpType::GATHER,    OpType::INVALID,       OpType::INVALID,   OpType::INVALID,
       OpType::INVALID,       OpType::INVALID,   OpType::INVALID,       OpType::INVALID,   OpType::INVALID,
       OpType::HALFALLTOALLV, OpType::INVALID};

inline OpType MC2OpType(AicpuComType comType)
{
    if (comType >= HCCL_CMD_RESERVED || comType <= HCCL_CMD_INVALID) {
        THROW<Hccl::CcuApiException>(StringFormat("Invalid OpType [%u].", comType));
    }
    return MC2_OP_TYPE[comType];
}

constexpr ReduceOp MC2_REDUCE_TYPE[] = {ReduceOp::SUM, ReduceOp::PROD, ReduceOp::MAX, ReduceOp::MIN, ReduceOp::INVALID};

inline ReduceOp MC2ReduceType(HcclReduceOp reduceOp)
{
    if (reduceOp >= (sizeof(MC2_REDUCE_TYPE) / sizeof(MC2_REDUCE_TYPE[0])) || reduceOp < HCCL_REDUCE_SUM) {
        THROW<Hccl::CcuApiException>(StringFormat("Invalid ReduceOp [%u].", reduceOp));
    }
    return MC2_REDUCE_TYPE[reduceOp];
}

constexpr DataType MC2_DATA_TYPE[]
    = {DataType::INT8,    DataType::INT16,   DataType::INT32,   DataType::FP16,     DataType::FP32,
       DataType::INT64,   DataType::UINT64,  DataType::UINT8,   DataType::UINT16,   DataType::UINT32,
       DataType::FP64,    DataType::BFP16,   DataType::INT128,  DataType::BF16_SAT, DataType::HIF8,
       DataType::FP8E4M3, DataType::FP8E5M2, DataType::FP8E8M0, DataType::INVALID};

inline DataType MC2DataType(HcclDataType dataType)
{
    if (dataType >= (sizeof(MC2_DATA_TYPE) / sizeof(MC2_DATA_TYPE[0])) || dataType < HCCL_DATA_TYPE_INT8) {
        THROW<Hccl::CcuApiException>(StringFormat("Invalid DataType [%u].", dataType));
    }
    return MC2_DATA_TYPE[dataType];
}

struct AivAicpuOpParam {
    AicpuComType commType; // 32b
    HcclReduceOp opType;   // 32b
    uint64_t sendBuffer;
    uint64_t recvBuffer;
    uint64_t count;
    uint64_t strideLen;

    // offset 32B
    HcclDataType hcclDataType;
    uint32_t valid; // 检查消息有效性

    // 存地址
    uint64_t sendCnt; // send CKE地址
    uint64_t rcvCnt;  // rcv CKE地址

    uint8_t isLast;        // 是否最后一个下
    uint8_t funID;         // 功能ID，1地址消息；  2开始工作
    uint8_t everyTurnRsp;  // 每轮都需要等待执行结束发送响应，再执行下一轮
    uint8_t everyTurnWait; // 每轮都需要等待work消息再执行
    uint8_t totalTurnCnt;  // 总轮次
    uint8_t res[59];       // 整体消息128字节
};

struct KFCTaskV2 {
    uint64_t inputA;  // A矩阵地址，通信在前时为sendbuffer
    uint64_t outputC; // 输出C矩阵地址
    uint64_t commOut; // 双输出时，通信输出地址
    uint64_t ctxNum;
    uint64_t context[MC2_RES_CTX_MAX]; // HCCL通信context
    uint64_t workSpace;                // 通信结果不直接输出时，放到workspace中
    uint64_t tilingData;               // 通信
};

struct HcclAiRMAWQ {
    u32 jettyId;
    u64 sqVA;
    u32 wqeSize;
    u32 sqDepth;
    u64 headAddr; // AIV无依赖
    u64 tailAddr; // AIV无依赖
    u64 dbAddr;
    u32 tp_id;
    uint8_t rmtEid[16];
    uint32_t rmtObjId; // rmtTokenID
    uint32_t rmtTokenValue;
    uint32_t localTokenId;
};

struct HcclAiRMACQ {
    u32 jfcId;
    u64 cqVA;
    u32 cqeSize;
    u32 cqDepth;
    u64 headAddr;
    u64 tailAddr;
    u64 dbAddr;
};

struct HcclCombinOpParam {
    uint64_t workSpace;                // client和server之间通信的地址
    uint64_t workSpaceSize;            // client和server之间通信的空间大小
    uint32_t rankId;                   // 当前卡rankId
    uint32_t rankDim;                  // 总卡数
    uint64_t winSize;                  // ccu不使用
    uint64_t windowsIn[MAX_RANK_NUM];  // ccu不使用
    uint64_t windowsOut[MAX_RANK_NUM]; // ccu不使用

    // for ccu
    uint64_t xnAddr;  // Xn寄存器其实地址
    uint64_t ckeAddr; // CKE寄存器其实地址
    uint64_t msAddr;  // MS地址，预留
    uint64_t msSize;  // 可写的MS个数，预留

    uint32_t opType[MAX_OP_NUM];
    uint8_t algorithmType[MAX_OP_NUM];

    HcclAiRMAWQ wq[MAX_RANK_NUM];
    HcclAiRMACQ cq[MAX_RANK_NUM];
};

struct Mc2ServerCfg {
    uint32_t version;
    uint8_t debugMode;
    uint8_t sendArgIndex;
    uint8_t recvArgIndex;
    uint8_t commOutArgIndex;
    uint8_t reserved[8];

    std::string ToString() const
    {
        return StringFormat(
            "debugMode = %u\nsendArgIndex = %u\nrecvArgIndex = %u\ncommOutArgIndex = %u\n", debugMode, sendArgIndex,
            recvArgIndex, commOutArgIndex);
    }
};

struct Mc2CommConfig {
    uint8_t skipLocalRankCopy;
    uint8_t skipBufferWindowCopy;
    uint8_t stepSize;
    uint8_t communicationEngine; // 用于标记使用AIV、CCU、AICPU做通信加速器，定义同通信域加速器配置
    char reserved[12];
    char groupName[128];     // 指定通信域
    char algConfig[128];     // 指定算法
    uint32_t opType;         // 算子类型
    uint32_t reduceType;     // reduce类型，sum,max等
    uint32_t dataType;       // 输入数据类型
    uint32_t outputDataType; // 输出数据类型

    std::string ToString() const
    {
        return StringFormat(
            "opType = %u\nreduceType = %u\ndataType = %u\noutputDataType = %u\n", opType, reduceType, dataType,
            outputDataType);
    }
};
struct Mc2Tiling {
    uint32_t version;                // 版本
    uint32_t commConfigNum;          // commConfig的个数，每个通信切片一个hcclConfig
    struct Mc2ServerCfg serverCfg;   // 计算部分tiling
    struct Mc2CommConfig commConfig; // 通信部分tiling，共有Mc2CommConfig个，每个通信切片一个hcclConfig

    std::string ToString()
    {
        auto selfDesc = StringFormat("version = %u\ncommConfigNum = %u\n", version, commConfigNum);
        return selfDesc + serverCfg.ToString() + commConfig.ToString();
    }
};

struct Mc2InitTilingInner {      // 这个必须放到mc2tiling的最前面
    uint32_t version;            // tiling结构体版本号,外部不可配置, 100开始
    uint32_t mc2HcommCnt;        // 通信的次数
    uint32_t offset[MAX_OP_NUM]; // 每个通信的偏移
    uint8_t debugMode;           // 调测模式, 0表示关闭,1表示开启,外部可配置
    uint8_t preparePosition;     // prepare消息发送的位置，0表示device，1表示host,外部可配置
    char reserved[22];
};

struct Mc2CcTilingInner {
    uint8_t skipLocalRankCopy; // 跳过本卡拷贝，在通信结果只需要给MC2内部计算使用或者本卡拷贝由aicore完成时,
    uint8_t skipBufferWindowCopy; // 跳过hbm到window间搬运 0不跳过，1跳过snd-window, 2跳过window-rcv
    uint8_t stepSize;             // 通信步长，粗粒度融合时填0,
    uint8_t version;              // 版本号
    char reserved[8];             // 保留字段
    uint8_t protocol;             // 协议类型 0:ubmemory 1:urma
    uint8_t
        communicationEngine; // 用于标记使用AIV、CCU、AICPU做通信加速器，定义同通信域加速器配置 0:默认ccu 1:ccu 2:aiv
    uint8_t srcDataType; // 输入数据类型
    uint8_t dstDataType; // 输出数据类型
    char groupName[128]; // groupName
    char algConfig[128]; // 算法配置
    uint32_t opType;     // tiling结构体版本号
    uint32_t reduceType; // reduce类型
};

constexpr uint32_t ALG_CONFIG_SIZE = 128;
constexpr uint32_t MAX_OP_NAME_SIZE = 256;
constexpr uint32_t MAX_MEM_TAG_SIZE = 256;
struct HcclOpArgs {
    DataType srcDataType;
    DataType dstDataType;
    ReduceOp reduceType;
    uint64_t count{0};
    char algConfig[ALG_CONFIG_SIZE];
    HcclAccelerator commEngine;
    uint64_t reverse;

    void Init()
    { // if not set value, give a default
        srcDataType = DataType::FP16;
        dstDataType = DataType::FP16;
        reduceType = ReduceOp::SUM;
        count = 0;
    }
};

} // namespace Hccl

#endif // HCCL_MC2_TYPE_H
