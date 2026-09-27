/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_PARAM_H
#define TASK_PARAM_H

#include <vector>
#include <string>
#include <memory>
#include "securec.h"
#include "hccl/base.h"
#include "const_val.h"
#include "enum_factory.h"
#include "string_util.h"
#include "ip_address.h"
#include "op_type.h"

namespace Hccl {

MAKE_ENUM(DmaOp, HCCL_DMA_READ, HCCL_DMA_WRITE, HCCL_DMA_NOTIFY_WAIT)

MAKE_ENUM(AlgType, NOT_SPECIFIED, RING, MULTI_RING, MESH, RECURSIVE_HD, BINARY_HD, PAIR_WISE, INVALID_VAL)

MAKE_ENUM(
    TaskParamType, TASK_SDMA, TASK_RDMA, TASK_REDUCE_INLINE, TASK_REDUCE_TBE, TASK_NOTIFY_RECORD, TASK_NOTIFY_WAIT,
    TASK_SEND_NOTIFY, TASK_SEND_PAYLOAD, TASK_WRITE_WITH_NOTIFY, TASK_WRITE_REDUCE_WITH_NOTIFY, TASK_CCU,
    TASK_AICPU_KERNEL, TASK_AICPU_REDUCE, TASK_AIV, TASK_UB_INLINE_WRITE, TASK_UB_REDUCE_INLINE, TASK_UB,
    TASK_DPU_KERNEL, TASK_DPU_THREAD_FENCE, TASK_DPU_CHANNEL_FENCE, TASK_DPU_INLINE_WRITE, TASK_DPU_NOTIFY_WAIT,
    TASK_DPU_WRITE_WITH_NOTIFY)

MAKE_ENUM(DfxLinkType, ONCHIP, HCCS, PCIE, ROCE, SIO, HCCS_SW, STANDARD_ROCE, UB, UBoE, RESERVED)

MAKE_ENUM(CcuProfilinType, CCU_TASK_PROFILING, CCU_WAITCKE_PROFILING, CCU_LOOPGROUP_PROFILING, CCU_MAP_PROFILING)

constexpr uint16_t CCU_MAX_CHANNEL_NUM = 16;               // 最多16条link
constexpr uint16_t INVALID_CKE_ID = 0xFFFF;                // CKE ID非法值
constexpr uint16_t INVALID_VALUE_CHANNELID = 0xFFFF;       // channel id非法值
constexpr u64 INVALID_VALUE_NOTIFYID = 0xFFFFFFFFFFFFFFFF; // NOTIFY id非法值
constexpr u32 INVALID_VALUE_RANKID = 0xFFFFFFFF;           // rank id非法值
struct CcuProfilingInfo {
    std::string name;  // CCU任务名或微码名
    uint8_t type;      // 枚举，0为Task粒度，1为WaitCKE，2为LoopGroup，3为channelId->RemoteRankId的映射
    uint8_t dieId;     // CCU任务执行的DieId
    uint8_t missionId; // CCU任务执行的MissionId
    uint16_t instrId;
    uint8_t reduceOpType;   // 与HcclReduceOp类型保持一致
    uint8_t inputDataType;  // 与HcclDataType类型保持一致
    uint8_t outputDataType; // 与HcclDataType类型保持一致
    uint64_t dataSize;      // 输入数据大小
    uint32_t ckeId;
    uint32_t mask;
    uint16_t channelId[CCU_MAX_CHANNEL_NUM];     // LoopGroup所包含的搬运指令使用的ChannelId
    uint32_t remoteRankId[CCU_MAX_CHANNEL_NUM];  // LoopGroup所包含的搬运指令的对端
    uint64_t channelHandle[CCU_MAX_CHANNEL_NUM]; // channel handle句柄

    CcuProfilingInfo()
        : name(""),
          type(0),
          dieId(0),
          missionId(0),
          instrId(0),
          reduceOpType(0),
          inputDataType(0),
          outputDataType(0),
          dataSize(0),
          ckeId(0),
          mask(0)
    {
        (void)memset_s(channelId, sizeof(channelId), INVALID_VALUE_CHANNELID, sizeof(channelId));
        for (u32 i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
            remoteRankId[i] = INVALID_VALUE_RANKID;
        }
        for (u32 i = 0; i < CCU_MAX_CHANNEL_NUM; i++) {
            channelHandle[i] = INVALID_VALUE_NOTIFYID;
        }
    }
};
constexpr u32 ADD_LEN = 128;
struct ParaDMA {
    const void* src;
    const void* dst;
    std::size_t size;
    u64 notifyID;
    u32 notifyValue;
    DfxLinkType linkType;
    DmaOp dmaOp;
    Eid locEid{};
    Eid rmtEid{};
    char locAddr[ADD_LEN]{};
    char rmtAddr[ADD_LEN]{};
};

struct ParaReduce {
    const void* src;
    const void* dst;
    std::size_t size;
    u64 notifyID;
    u32 notifyValue;
    DfxLinkType linkType;
    HcclReduceOp reduceOp{HcclReduceOp::HCCL_REDUCE_RESERVED};
    HcclDataType dataType{HcclDataType::HCCL_DATA_TYPE_RESERVED};
    Eid locEid{};
    Eid rmtEid{};
};

struct ParaNotify {
    u64 notifyID;
    u32 value;
};

struct ParaCcu {
    u8 dieId;
    u8 missionId;
    u8 execMissionId;
    u32 instrId;
    u64 executeId;
    u64 ccuKernelHandle{0};
};

struct ParaAiv {
    HcclCMDType cmdType;
    u32 tag;
    u64 count;
    u32 numBlocks;
    u32 rankSize;
    void* flagMem;
    u64 flagMemSize;
    u32 rank;
    u32 sendRecvRemoteRank;
    HcclDataType dataType;
};

struct TaskParam {
    TaskParamType taskType;
    u64 beginTime;
    u64 endTime;
    u64 aicpuTaskId{0};
    uint16_t npuDevId{0};
    bool isMaster{false};
    u32 tid{0};
    union {
        ParaDMA DMA;       // taskType = SDMA/RDMA使用, 包括rtRDMASend写notify
        ParaReduce Reduce; // taskType = inline/CCE Reduce使用
        ParaNotify Notify; // taskType = Notify Record/Wait使用
        ParaCcu Ccu;
        ParaAiv Aiv; // aiv param
    } taskPara;
    std::shared_ptr<std::vector<CcuProfilingInfo>> ccuDetailInfo; // taskType为TASK_CCU时，ParaCcu的补充profiling信息
    std::string Describe() const
    {
        return StringFormat(
                   "TaskParam[taskType[%s] beginTime[%llu] endTime[%llu] aicpuTaskId[%llu]",
                   taskType.Describe().c_str(), beginTime, endTime, aicpuTaskId)
               + DescribeDetail(*this) + "]";
    }

private:
    static std::string DescribeDetail(const TaskParam& param)
    {
        std::string result;
        switch (param.taskType) {
            case TaskParamType::TASK_SDMA:
            case TaskParamType::TASK_RDMA:
            case TaskParamType::TASK_SEND_PAYLOAD:
            case TaskParamType::TASK_UB_INLINE_WRITE:
            case TaskParamType::TASK_UB:
            case TaskParamType::TASK_WRITE_WITH_NOTIFY:
            case TaskParamType::TASK_WRITE_REDUCE_WITH_NOTIFY:
            case TaskParamType::TASK_DPU_INLINE_WRITE:
            case TaskParamType::TASK_DPU_WRITE_WITH_NOTIFY:
                result += StringFormat(
                    " src[%p] dst[%p] size[%zu] notifyID[%llu] dmaOp[%s] linkType[%s]", param.taskPara.DMA.src,
                    param.taskPara.DMA.dst, param.taskPara.DMA.size, param.taskPara.DMA.notifyID,
                    param.taskPara.DMA.dmaOp.Describe().c_str(), param.taskPara.DMA.linkType.Describe().c_str());
                break;
            case TaskParamType::TASK_REDUCE_INLINE:
            case TaskParamType::TASK_UB_REDUCE_INLINE:
            case TaskParamType::TASK_REDUCE_TBE:
                result += StringFormat(
                    " src[%p] dst[%p] size[%zu] notifyID[%llu] reduceOp[%d] dataType[%d] linkType[%s]",
                    param.taskPara.Reduce.src, param.taskPara.Reduce.dst, param.taskPara.Reduce.size,
                    param.taskPara.Reduce.notifyID, static_cast<int>(param.taskPara.Reduce.reduceOp),
                    static_cast<int>(param.taskPara.Reduce.dataType),
                    param.taskPara.Reduce.linkType.Describe().c_str());
                break;
            case TaskParamType::TASK_NOTIFY_RECORD:
            case TaskParamType::TASK_NOTIFY_WAIT:
            case TaskParamType::TASK_SEND_NOTIFY:
            case TaskParamType::TASK_DPU_NOTIFY_WAIT:
            case TaskParamType::TASK_DPU_CHANNEL_FENCE:
                result += StringFormat(
                    " notifyID[%llu] value[%u]", param.taskPara.Notify.notifyID, param.taskPara.Notify.value);
                break;
            case TaskParamType::TASK_AIV:
                result += StringFormat(
                    " cmdType[%d] tag[%u] count[%llu] numBlocks[%u] rankSize[%u]"
                    " rank[%u] remoteRank[%u] dataType[%d]",
                    static_cast<int>(param.taskPara.Aiv.cmdType), param.taskPara.Aiv.tag, param.taskPara.Aiv.count,
                    param.taskPara.Aiv.numBlocks, param.taskPara.Aiv.rankSize, param.taskPara.Aiv.rank,
                    param.taskPara.Aiv.sendRecvRemoteRank, static_cast<int>(param.taskPara.Aiv.dataType));
                break;
            case TaskParamType::TASK_CCU:
                result += StringFormat(
                    " dieId[%u] missionId[%u] execMissionId[%u] instrId[%u] executeId[%llu]", param.taskPara.Ccu.dieId,
                    param.taskPara.Ccu.missionId, param.taskPara.Ccu.execMissionId, param.taskPara.Ccu.instrId,
                    param.taskPara.Ccu.executeId);
                break;
            default:
                break;
        }
        return result;
    }
};

const std::map<HcclCMDType, std::pair<Hccl::OpType, std::string>> CMD_OP_TYPE_INFO_MAP = {
    {HcclCMDType::HCCL_CMD_ALLREDUCE, {Hccl::OpType::ALLREDUCE, "OpType::ALLREDUCE"}},
    {HcclCMDType::HCCL_CMD_ALLGATHER, {Hccl::OpType::ALLGATHER, "OpType::ALLGATHER"}},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER, {Hccl::OpType::REDUCESCATTER, "OpType::REDUCESCATTER"}},
    {HcclCMDType::HCCL_CMD_SEND, {Hccl::OpType::SEND, "OpType::SEND"}},
    {HcclCMDType::HCCL_CMD_RECEIVE, {Hccl::OpType::RECV, "OpType::RECV"}},
    {HcclCMDType::HCCL_CMD_ALLTOALL, {Hccl::OpType::ALLTOALL, "OpType::ALLTOALL"}},
    {HcclCMDType::HCCL_CMD_ALLTOALLV, {Hccl::OpType::ALLTOALLV, "OpType::ALLTOALLV"}},
    {HcclCMDType::HCCL_CMD_BROADCAST, {Hccl::OpType::BROADCAST, "OpType::BROADCAST"}},
    {HcclCMDType::HCCL_CMD_ALLGATHER_V, {Hccl::OpType::ALLGATHERV, "OpType::ALLGATHERV"}},
    {HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V, {Hccl::OpType::REDUCESCATTERV, "OpType::REDUCESCATTERV"}},
    {HcclCMDType::HCCL_CMD_REDUCE, {Hccl::OpType::REDUCE, "OpType::REDUCE"}},
    {HcclCMDType::HCCL_CMD_ALLTOALLVC, {Hccl::OpType::ALLTOALLVC, "OpType::ALLTOALLVC"}},
    {HcclCMDType::HCCL_CMD_SCATTER, {Hccl::OpType::SCATTER, "OpType::SCATTER"}},
    {HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, {Hccl::OpType::BATCHSENDRECV, "OpType::BATCHSENDRECV"}},
    {HcclCMDType::HCCL_CMD_HALF_ALLTOALLV, {Hccl::OpType::HALFALLTOALLV, "OpType::HALFALLTOALLV"}},
    {HcclCMDType::HCCL_CMD_BARRIER, {Hccl::OpType::BARRIER, "OpType::BARRIER"}},
    {HcclCMDType::HCCL_CMD_GATHER, {Hccl::OpType::GATHER, "OpType::GATHER"}},
    {HcclCMDType::HCCL_CMD_BATCH_GET, {Hccl::OpType::BATCHGET, "OpType::BATCHGET"}},
    {HcclCMDType::HCCL_CMD_BATCH_PUT, {Hccl::OpType::BATCHPUT, "OpType::BATCHPUT"}},
};

} // namespace Hccl

#endif
