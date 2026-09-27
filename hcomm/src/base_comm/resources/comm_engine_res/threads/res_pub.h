/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RES_PUB_H
#define RES_PUB_H
#include <map>
#include <string>
#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "string_util.h"
constexpr u32 INVALID_U32 = UINT32_MAX;
constexpr u64 DFX_INVALID_U64 = UINT64_MAX;
constexpr s32 INVALID_RANKID = INT32_MAX;

// 以下枚举来源于 legacy/ascend950，以 enum : uint8_t 形式独立定义，
// 使 res_pub.h 脱离 legacy 头文件依赖。值与 MAKE_ENUM 原定义一致。
namespace Hccl {

enum OpTypeVal : u8 {
    OP_TYPE_ALLREDUCE = 0,
    OP_TYPE_BROADCAST = 1,
    OP_TYPE_ALLGATHER = 2,
    OP_TYPE_REDUCESCATTER = 3,
    OP_TYPE_SEND = 4,
    OP_TYPE_RECV = 5,
    OP_TYPE_BARRIER = 6,
    OP_TYPE_ALLTOALL = 7,
    OP_TYPE_REDUCE = 8,
    OP_TYPE_GATHER = 9,
    OP_TYPE_SCATTER = 10,
    OP_TYPE_ALLTOALLV = 11,
    OP_TYPE_ALLTOALLVC = 12,
    OP_TYPE_HALFALLTOALLV = 13,
    OP_TYPE_BATCHSENDRECV = 14,
    OP_TYPE_BATCHGET = 15,
    OP_TYPE_BATCHPUT = 16,
    OP_TYPE_ALLGATHERV = 17,
    OP_TYPE_REDUCESCATTERV = 18,
    OP_TYPE_DEBUGCASE = 19,
    OP_TYPE_INVALID = 20,
    OP_TYPE_COUNT
};

enum AlgTypeVal : u8 {
    ALG_TYPE_NOT_SPECIFIED = 0,
    ALG_TYPE_RING = 1,
    ALG_TYPE_MULTI_RING = 2,
    ALG_TYPE_MESH = 3,
    ALG_TYPE_RECURSIVE_HD = 4,
    ALG_TYPE_BINARY_HD = 5,
    ALG_TYPE_PAIR_WISE = 6,
    ALG_TYPE_INVALID_VAL = 7,
    ALG_TYPE_COUNT
};

enum TaskParamTypeVal : u8 {
    TASK_SDMA = 0,
    TASK_RDMA = 1,
    TASK_REDUCE_INLINE = 2,
    TASK_REDUCE_TBE = 3,
    TASK_NOTIFY_RECORD = 4,
    TASK_NOTIFY_WAIT = 5,
    TASK_SEND_NOTIFY = 6,
    TASK_SEND_PAYLOAD = 7,
    TASK_WRITE_WITH_NOTIFY = 8,
    TASK_WRITE_REDUCE_WITH_NOTIFY = 9,
    TASK_CCU = 10,
    TASK_AICPU_KERNEL = 11,
    TASK_AICPU_REDUCE = 12,
    TASK_AIV = 13,
    TASK_UB_INLINE_WRITE = 14,
    TASK_UB_REDUCE_INLINE = 15,
    TASK_UB = 16,
    TASK_DPU_KERNEL = 17,
    TASK_DPU_THREAD_FENCE = 18,
    TASK_DPU_CHANNEL_FENCE = 19,
    TASK_DPU_INLINE_WRITE = 20,
    TASK_DPU_NOTIFY_WAIT = 21,
    TASK_DPU_WRITE_WITH_NOTIFY = 22,
    TASK_PARAM_TYPE_COUNT
};

enum DfxLinkTypeVal : u8 {
    LINK_ONCHIP = 0,
    LINK_HCCS = 1,
    LINK_PCIE = 2,
    LINK_ROCE = 3,
    LINK_SIO = 4,
    LINK_HCCS_SW = 5,
    LINK_STANDARD_ROCE = 6,
    LINK_UB = 7,
    LINK_UBoE = 8,
    LINK_RESERVED = 9,
    LINK_TYPE_COUNT
};

enum DfxWorkflowMode : u8 {
    NEW_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB = 0,
    NEW_WORKFLOW_MODE_OP_BASE = 1,
    NEW_WORKFLOW_MODE_RESERVED = 2
};

enum DfxTransportType : u8 {
    DFX_TRANSPORT_TYPE_SDMA = 0,
    DFX_TRANSPORT_TYPE_RDMA = 1,
    DFX_TRANSPORT_TYPE_LOCAL = 2,
    DFX_TRANSPORT_TYPE_UB = 3,
    DFX_TRANSPORT_TYPE_ROCE = 4
};

constexpr u32 DFX_INVALID_RANKID = 0xFFFFFFFF;

enum DfxTaskRole : u8 { NEW_TASK_ROLE_DST = 0, NEW_TASK_ROLE_SRC = 1 };

inline std::string TaskParamTypeValToStr(u8 val)
{
    static const std::map<TaskParamTypeVal, std::string> taskParamTypeValStrMap = {
        {TaskParamTypeVal::TASK_SDMA, "TASK_SDMA"},
        {TaskParamTypeVal::TASK_RDMA, "TASK_RDMA"},
        {TaskParamTypeVal::TASK_REDUCE_INLINE, "TASK_REDUCE_INLINE"},
        {TaskParamTypeVal::TASK_REDUCE_TBE, "TASK_REDUCE_TBE"},
        {TaskParamTypeVal::TASK_NOTIFY_RECORD, "TASK_NOTIFY_RECORD"},
        {TaskParamTypeVal::TASK_NOTIFY_WAIT, "TASK_NOTIFY_WAIT"},
        {TaskParamTypeVal::TASK_SEND_NOTIFY, "TASK_SEND_NOTIFY"},
        {TaskParamTypeVal::TASK_SEND_PAYLOAD, "TASK_SEND_PAYLOAD"},
        {TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY, "TASK_WRITE_WITH_NOTIFY"},
        {TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY, "TASK_WRITE_REDUCE_WITH_NOTIFY"},
        {TaskParamTypeVal::TASK_CCU, "TASK_CCU"},
        {TaskParamTypeVal::TASK_AICPU_KERNEL, "TASK_AICPU_KERNEL"},
        {TaskParamTypeVal::TASK_AICPU_REDUCE, "TASK_AICPU_REDUCE"},
        {TaskParamTypeVal::TASK_AIV, "TASK_AIV"},
        {TaskParamTypeVal::TASK_UB_INLINE_WRITE, "TASK_UB_INLINE_WRITE"},
        {TaskParamTypeVal::TASK_UB_REDUCE_INLINE, "TASK_UB_REDUCE_INLINE"},
        {TaskParamTypeVal::TASK_UB, "TASK_UB"},
        {TaskParamTypeVal::TASK_DPU_KERNEL, "TASK_DPU_KERNEL"},
        {TaskParamTypeVal::TASK_DPU_THREAD_FENCE, "TASK_DPU_THREAD_FENCE"},
        {TaskParamTypeVal::TASK_DPU_CHANNEL_FENCE, "TASK_DPU_CHANNEL_FENCE"},
        {TaskParamTypeVal::TASK_DPU_INLINE_WRITE, "TASK_DPU_INLINE_WRITE"},
        {TaskParamTypeVal::TASK_DPU_NOTIFY_WAIT, "TASK_DPU_NOTIFY_WAIT"},
        {TaskParamTypeVal::TASK_DPU_WRITE_WITH_NOTIFY, "TASK_DPU_WRITE_WITH_NOTIFY"},
    };
    auto iter = taskParamTypeValStrMap.find(static_cast<TaskParamTypeVal>(val));
    if (iter == taskParamTypeValStrMap.end()) {
        return "TaskParamTypeVal(" + std::to_string(val) + ")";
    }
    return iter->second;
}

inline std::string DfxLinkTypeValToStr(u8 val)
{
    static const std::map<DfxLinkTypeVal, std::string> dfxLinkTypeValStrMap = {
        {DfxLinkTypeVal::LINK_ONCHIP, "LINK_ONCHIP"},
        {DfxLinkTypeVal::LINK_HCCS, "LINK_HCCS"},
        {DfxLinkTypeVal::LINK_PCIE, "LINK_PCIE"},
        {DfxLinkTypeVal::LINK_ROCE, "LINK_ROCE"},
        {DfxLinkTypeVal::LINK_SIO, "LINK_SIO"},
        {DfxLinkTypeVal::LINK_HCCS_SW, "LINK_HCCS_SW"},
        {DfxLinkTypeVal::LINK_STANDARD_ROCE, "LINK_STANDARD_ROCE"},
        {DfxLinkTypeVal::LINK_UB, "LINK_UB"},
        {DfxLinkTypeVal::LINK_UBoE, "LINK_UBoE"},
        {DfxLinkTypeVal::LINK_RESERVED, "LINK_RESERVED"},
    };
    auto iter = dfxLinkTypeValStrMap.find(static_cast<DfxLinkTypeVal>(val));
    if (iter == dfxLinkTypeValStrMap.end()) {
        return "DfxLinkTypeVal(" + std::to_string(val) + ")";
    }
    return iter->second;
}

inline std::string DfxTransportTypeToStr(u8 val)
{
    static const std::map<DfxTransportType, std::string> dfxTransportTypeStrMap = {
        {DfxTransportType::DFX_TRANSPORT_TYPE_SDMA, "DFX_TRANSPORT_TYPE_SDMA"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_RDMA, "DFX_TRANSPORT_TYPE_RDMA"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL, "DFX_TRANSPORT_TYPE_LOCAL"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_UB, "DFX_TRANSPORT_TYPE_UB"},
        {DfxTransportType::DFX_TRANSPORT_TYPE_ROCE, "DFX_TRANSPORT_TYPE_ROCE"},
    };
    auto iter = dfxTransportTypeStrMap.find(static_cast<DfxTransportType>(val));
    if (iter == dfxTransportTypeStrMap.end()) {
        return "DfxTransportType(" + std::to_string(val) + ")";
    }
    return iter->second;
}

} // namespace Hccl

namespace Hccl {

// DFX 环形队列容量
constexpr u32 DFX_TASK_INFO_QUEUE_CAPACITY = 2048 + 128; // 2048 基础容量 + 128 余量
constexpr u32 DFX_OP_INFO_QUEUE_CAPACITY = 1024;

struct DfxDfxOpInfo {
    // 8B 对齐字段（7 × 8B = 56B，offset 0-55）
    void* commHandle{nullptr}; // 通信域句柄，来源于 dfxOpInfo_->comm_，用于从 CollCommAicpu 获取上下文
                               //   cclTag 不单独存储：与 opType 一一对应（均来自 CMD_OP_TYPE_INFO_MAP），
                               //   上报 Msprof 时通过 opTypeHashCache_[opType] 查表转为 GetProfHashId 哈希值
    u64 count{0};   // 发送数据个数，来源于 dfxOpInfo_->op_.dataCount
    u64 srcAddr{0}; // 算子级输入地址，来源于 dfxOpInfo_->op_.newInputMem
    u64 dstAddr{0}; // 算子级输出地址，来源于 dfxOpInfo_->op_.newOutputMem
    u64 srcSize{0}; // 算子级输入大小，来源于 dfxOpInfo_->op_.inputMemSize
    u64 dstSize{0}; // 算子级输出大小，来源于 dfxOpInfo_->op_.outputMemSize
    void* hcclCommDfxLite{
        nullptr}; // HcclCommDfxLite 指针（void* 避免 base_comm 对 coll_communicator_mgr 的编译期依赖）
                  //   上报时 static_cast<HcclCommDfxLite*>(hcclCommDfxLite)->GetChannelRemoteRankId(channelHandle)

    // 4B 对齐字段（3 × 4B = 12B，offset 56-67）
    u32 opIndex{0}; // 算子序号，标识当前是 algTag 数组的第几个，来源于 dfxOpInfo_->opIndex_
    u32 cpuWaitAicpuNotifyId{0}; // Host 等 Device 的 notify ID，来源于 dfxOpInfo_->cpuWaitAicpuNotifyId_
    u32 aicpuWaitCpuNotifyId{0}; // Device 等 Host 的 notify ID，来源于 DfxOpInfo

    // 1B 对齐字段（3 × 1B = 3B，offset 68-70）
    u8 opType{0}; // 算子类型枚举值，来源于 dfxOpInfo_->op_.opType（OpType 底层 uint8_t）
    u8 algType{0}; // 通信算法枚举值，来源于 AlgType 底层 uint8_t（如 RING/MESH 等；当前 Lite 路径固定为 NHR）
                   //   上报 Msprof 时通过 algTypeHashCache_[algType] 查表转为 GetProfHashId 哈希值
    u8 dataType{
        0}; // 数据类型枚举值，来源于 dfxOpInfo_->op_.dataType（HcclDataType 底层 uint8_t）
            //   由 SetCurrDfxOpInfo 从 oldDataType 转换后获取；从 task 级提升为算子级，每个算子只有一个 dataType

    // 固定长度尾部字段（288B，offset 71-358）
    char algTag[288]{0}; // 算子标签字符串，来源于 dfxOpInfo_->algTag_，288 = TAG_MAX_LENGTH(256) + 32 余量
    // 总计 360B：8B字段56B + 4B字段12B + 1B字段3B + algTag 288B + 尾部padding 1B（对齐到8B）

    std::string Describe() const
    {
        return StringFormat(
            "commHandle[%p], count[%llu], srcAddr[0x%llx], dstAddr[0x%llx], srcSize[%llu], dstSize[%llu], "
            "hcclCommDfxLite[%p], opIndex[%u], cpuWaitAicpuNotifyId[%u], aicpuWaitCpuNotifyId[%u], "
            "opType[%u], algType[%u], dataType[%u], algTag[%s]",
            commHandle, static_cast<unsigned long long>(count), static_cast<unsigned long long>(srcAddr),
            static_cast<unsigned long long>(dstAddr), static_cast<unsigned long long>(srcSize),
            static_cast<unsigned long long>(dstSize), hcclCommDfxLite, opIndex, cpuWaitAicpuNotifyId,
            aicpuWaitCpuNotifyId, opType, algType, dataType, algTag);
    }
};

struct DfxTaskParaNotify { // Notify 任务参数（NOTIFY_RECORD/NOTIFY_WAIT）
    u64 sqeAddr;           // SQE 中的偏移地址
    std::string Describe() const { return StringFormat("sqeAddr[0x%llx]", static_cast<unsigned long long>(sqeAddr)); }
};

struct DfxTaskParaDma { // SDMA 任务参数，信息从 SQE 获取
    u64 sqeAddr;        // SQE 中的偏移地址
    std::string Describe() const { return StringFormat("sqeAddr[0x%llx]", static_cast<unsigned long long>(sqeAddr)); }
};

struct DfxTaskParaUbDma {      // UB DMA 任务参数
    u64 sqeAddr;               // SQE 中的偏移地址
    u64 srcAddr;               // 源地址
    u64 dstAddr;               // 目的地址
    u64 size;                  // 数据大小（字节）
    u32 notifyId{INVALID_U32}; // Notify ID，来源于 ParaDMA::notifyID，taskException 使用
                               //   write with notify SQE 中无此字段；cnt notify 不支持跨片
    uint64_t jettyHandle{0};   // taskException 用于 dump jetty context
    uint32_t jettyId{0};       // taskException 用于 dump jetty context
    u32 tpn{0};                // import_jetty返回的tpn，异常时通知管控面用
    std::string Describe() const
    {
        return StringFormat(
            "srcAddr[0x%llx], dstAddr[0x%llx], size[0x%llx], notifyId[0x%x], "
            "jettyHandle[0x%llx], jettyId[%u], tpn[%u]",
            static_cast<unsigned long long>(srcAddr), static_cast<unsigned long long>(dstAddr),
            static_cast<unsigned long long>(size), notifyId, static_cast<unsigned long long>(jettyHandle), jettyId,
            tpn);
    }
};

struct DfxTaskParaReduce {     // UB Reduce 任务参数
    u64 sqeAddr;               // SQE 中的偏移地址
    u64 srcAddr;               // 源地址
    u64 dstAddr;               // 目的地址
    u64 size;                  // 数据大小（字节）
    u32 notifyId{INVALID_U32}; // Notify ID，来源于 ParaReduce::notifyID，taskException 使用
    u8 reduceOp; // Reduce 操作类型枚举值（HcclReduceOp 底层 uint8_t，仅 Reduce 类 task 有效）
    uint64_t jettyHandle{0}; // taskException 用于 dump jetty context
    uint32_t jettyId{0};     // taskException 用于 dump jetty context
    u32 tpn{0};              // import_jetty返回的tpn，异常时通知管控面用
    std::string Describe() const
    {
        return StringFormat(
            "srcAddr[0x%llx], dstAddr[0x%llx], size[0x%llx], notifyId[0x%x], reduceOp[%u], "
            "jettyHandle[0x%llx], jettyId[%u], tpn[%u]",
            static_cast<unsigned long long>(srcAddr), static_cast<unsigned long long>(dstAddr),
            static_cast<unsigned long long>(size), notifyId, reduceOp, static_cast<unsigned long long>(jettyHandle),
            jettyId, tpn);
    }
};

struct DfxTaskParaWriteValue { // P2P WriteValue 任务参数
    u64 sqeAddr;               // SQE 中的偏移地址
    u32 notifyId{INVALID_U32}; // Notify ID，来源于 ParaReduce::notifyID，taskException 使用
    std::string Describe() const
    {
        return StringFormat("sqeAddr[0x%llx], notifyId[0x%x]", static_cast<unsigned long long>(sqeAddr), notifyId);
    }
};

struct DfxTaskInfo {
    // 8B 对齐字段
    u64 dfxOpInfo{DFX_INVALID_U64};     // 算子级上下文指针，指向 AicpuTsThread 缓存的 DfxDfxOpInfo
    u64 channelHandle{DFX_INVALID_U64}; // Channel 句柄，来源于 TaskInfo::channelHandle_
                                        //   获取 remoteRankId；taskException 通过 dfxOpInfo->commHandle 定位通信域
                                        //   （与 DfxTaskInfo::hcclCommDfxLite 职责不重叠）
    // 任务参数（按 taskType 使用其中一个分支，与 TaskParam::taskPara 的 union 模式一致）
    union {
        DfxTaskParaNotify Notify; // NOTIFY_RECORD/NOTIFY_WAIT 使用
        DfxTaskParaDma Dma;       // SDMA 使用
        DfxTaskParaUbDma ubDma;
        DfxTaskParaReduce Reduce; // REDUCE 使用
        DfxTaskParaWriteValue writeValue;
    } taskPara;

    // 4B 对齐字段
    u32 sqId;   // Stream Queue ID
    u32 taskId; // Stream Queue Entry ID

    // 1B 对齐字段
    u8 taskType;      // 任务类型枚举值（TaskParamType 底层 uint8_t）
    u8 linkType;      // 链路类型枚举值（DfxLinkType 底层 uint8_t）
    u8 transportType; // 传输类型：0=SDMA, 1=RDMA, 2=LOCAL（由 remoteRank 推导），还有 UB Transport 写入的 UB
    // 尾部填充 1B 对齐到 64B

    bool IsTaskTypeValid() const { return taskType < static_cast<u8>(TaskParamTypeVal::TASK_PARAM_TYPE_COUNT); }
    std::string Describe() const
    {
        std::string result = StringFormat(
            "DfxTaskInfo: sqId[%u], taskId(sqeId)[%u], taskType[%s], linkType[%s], transportType[%s], "
            "channelHandle[0x%llx], ",
            sqId, taskId, TaskParamTypeValToStr(taskType).c_str(), DfxLinkTypeValToStr(linkType).c_str(),
            DfxTransportTypeToStr(transportType).c_str(), static_cast<unsigned long long>(channelHandle));

        auto type = static_cast<TaskParamTypeVal>(taskType);
        switch (type) {
            case TaskParamTypeVal::TASK_UB:
            case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
            case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
                result += taskPara.ubDma.Describe();
                break;
            case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
            case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            case TaskParamTypeVal::TASK_REDUCE_INLINE:
                result += taskPara.Reduce.Describe();
                break;
            case TaskParamTypeVal::TASK_SDMA:
            case TaskParamTypeVal::TASK_RDMA:
                result += taskPara.Dma.Describe();
                break;
            case TaskParamTypeVal::TASK_NOTIFY_RECORD:
            case TaskParamTypeVal::TASK_NOTIFY_WAIT:
            case TaskParamTypeVal::TASK_SEND_NOTIFY:
                result += taskPara.Notify.Describe();
                break;
            default:
                result += "taskType not supported";
                break;
        }
        return result;
    }
};
} // namespace Hccl
#endif // RES_PUB_H
