/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_ALG_COMM_H
#define COLL_ALG_COMM_H

#include <string>
#include <vector>
#include <map>
#include <set>
#include <unordered_set>

#include "hccl_common.h"
#include "hccl_types.h"
#include "transport_pub.h"
#include "stream_pub.h"
#include "local_notify.h"
#include "hccl_trace_info.h"
#include "common.h"
#include "threadManage.h"
#include "template_v1_utils.h"

namespace hccl {
using RankId = u32;

enum class OpMode { OFFLOAD = 0, OPBASE = 1 };

enum class DeviceMode { HOST = 0, AICPU = 1 };

enum class AlgExpansionMode {
    SUPERK_HOST = 0,
    SUPERK_AICPU = 1,
    SUPERK_AIV = 2,
    // SUPERK_CCU = 3,
    SUPERK_RECURSIVE = 4
};

enum class TransportStatus { INIT, READY, STOP };

enum TransportMemType {
    CCL_INPUT = 0,
    CCL_OUTPUT,
    SCRATCH,
    PARAM_INPUT,
    PARAM_OUTPUT,
    AIV_INPUT,
    AIV_OUTPUT,
    USER_MEM,
    RESERVED
};

enum class TransportLinkType : int { RESERVED = -1, HCCS = 0, SIO = 1, RDMA = 2, MAX_NUM };

struct TransportRequest {
    bool isValid = false;
    RankId localUserRank = 0;
    RankId remoteUserRank = 0;
    TransportMemType inputMemType = TransportMemType::RESERVED;
    TransportMemType outputMemType = TransportMemType::RESERVED;
    bool isUsedRdma = false;
    u32 notifyNum = 0;
    TransportLinkType linkType = TransportLinkType::RESERVED;
};

struct SingleSubCommTransport {
    std::vector<TransportRequest> transportRequests;
    std::vector<LINK> links;
    std::vector<TransportStatus> status; // 代表该transport是否ready, stop后为stop, 建链后为ready
    u64 taskNum = 0;
    std::map<u32, u32> userRank2subCommRank;
    std::map<u32, u32> subCommRank2UserRank;
    bool supportDataReceivedAck = false;
    LinkMode linkMode = LinkMode::LINK_DUPLEX_MODE;
    bool enableUseOneDoorbell = false;
    bool needVirtualLink = false;   // for alltoall 多线程性能提升使用
    std::vector<LINK> virtualLinks; // for alltoall 多线程性能提升使用
    bool isZeroCopy = false;
};
using LevelNSubCommTransport = std::vector<SingleSubCommTransport>;
using OpCommTransport = std::vector<LevelNSubCommTransport>;

struct AlgResourceRequest {
    u64 scratchMemSize = 0;
    u32 streamNum = 0;
    u32 notifyNum = 0;
    u64 aivBufferRequest = 0;
    DeviceMode mode = DeviceMode::HOST; // 用于区分是host模式，还是aicpu模式
    OpCommTransport opTransport;
    bool isInGraphCaptureZeroCopy = false;
    void Describe()
    {
        HCCL_DEBUG(
            "[AlgResourceRequest], scratchMemSize[%u], streamNum[%u], notifyNum[%u], aivBufferRequest[%llu], "
            "DeviceMode[%d].",
            scratchMemSize, streamNum, notifyNum, aivBufferRequest, mode);
    };
};

struct AlgResourceResponse {
    DeviceMem cclInputMem;
    DeviceMem cclOutputMem;
    DeviceMem paramInputMem;
    DeviceMem paramOutputMem;
    DeviceMem scratchMem;
    DeviceMem aivInputMem;
    DeviceMem aivOutputMem;
    DeviceMem aivCommInfoMem;
    std::vector<Stream> slaveStreams;
    std::vector<Stream> slaveDevStreams;
    std::vector<std::shared_ptr<LocalNotify>> notifiesMain; // Main Signals, 与Aux成对使用，大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify>> notifiesAux; // Auxiliary Signals, 与Main成对使用, 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify>> notifiesDevMain; // 大小等同于slaveStreams
    std::vector<std::shared_ptr<LocalNotify>> notifiesDevAux;  // 大小等同于slaveStreams
    OpCommTransport opTransportResponse;                       // 默认的Transport资源
    OpCommTransport opTransportResponseBackUp;                 // Transport备资源 (借轨场景使用)
    std::vector<std::shared_ptr<ThreadManage>> threadManage;
};

enum class BatchSendRecvCurMode { SEND = 0, RECV = 1, SEND_RECV = 2, SEND_RECV_RESERVED };

struct OpParam {
    std::string tag = "";
    Stream stream;
    void* inputPtr = nullptr;
    u64 inputSize = 0;
    void* outputPtr = nullptr;
    u64 outputSize = 0;
    HcclReduceOp reduceType = HcclReduceOp::HCCL_REDUCE_RESERVED;
    SyncMode syncMode = SyncMode::DEFAULT_TIMEWAITSYNCMODE;
    RankId root = INVALID_VALUE_RANKID;
    RankId dstRank = 0;
    RankId srcRank = 0;
    bool aicpuUnfoldMode = false;
    uint8_t aicpuCacheEnable = 0;
    bool isCapture = false;
    HcclTraceInfo* opBaseAtraceInfo = nullptr;
    union {
        struct {
            u64 count;
            HcclDataType dataType;
            u64 strideCount;
        } DataDes = {0, HCCL_DATA_TYPE_RESERVED, 0};
        struct {
            void* counts;
            void* displs;
            HcclDataType dataType;
        } VDataDes;
        struct {
            HcclDataType sendType;
            HcclDataType recvType;
            u64 sendCount;
            u64 recvCount;
            void* sendCounts;
            void* recvCounts;
            void* sdispls;
            void* rdispls;
            void* sendCountMatrix;
        } All2AllDataDes;
        struct {
            HcclSendRecvItem* sendRecvItemsPtr;
            u32 itemNum;
            u32 curIterNum;
            BatchSendRecvCurMode curMode;
            u8* isDirectRemoteRank;
        } BatchSendRecvDataDes;
        struct {
            u32 itemNum;
            u32 queueNum;
            u32 queueIdx;
        } BatchWriteDataDes;
    };
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    bool supportZeroCopy = false;
    bool isZeroCopy = false;
    u8 aclGraphZeroCopyEnable = 0;  // 记录和传递外部配置参数aclGraphZeroCopyEnable
    bool supportRoceDirect = false; // AIV场景支持Roce直驱
    bool isNpuDirectRoce = false;   // AIV场景使用Roce直驱标记位
    s32 aivTag = 0;                 // AIV场景使用的软同步标记位
    u32 index = 0;
    bool isInplaceError = false;
    u32 rankSize = 0;
    u32 aivCoreLimit = 0;
    u8 deterministic = 0;
    u32 srTag = 0;
    u32 localGroupRank = 0;
    bool isGroupMode = false;
    bool supportSymmetricMemory = false;
    void* inputSymWindow = nullptr;
    u64 inputOffset = 0;
    void* outputSymWindow = nullptr;
    u64 outputOffset = 0;
    bool needIncreLink = false;

    inline HcclDataType GetDataType() const
    {
        if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
            return VDataDes.dataType;
        }
        return DataDes.dataType;
    }
    inline u64 GetDataCount(RankId rankId) const
    {
        if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
            return static_cast<const u64*>(VDataDes.counts)[rankId];
        }
        return DataDes.count;
    }
    inline u64 GetStrideCount() const
    {
        if (opType == HcclCMDType::HCCL_CMD_ALLGATHER_V || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
            return 0;
        }
        return DataDes.strideCount;
    }
    // 重载<符号，用于map
    bool operator<(const OpParam& other) const noexcept
    {
        switch (opType) {
            // 比较数据类型、数据量、通信域、可用核数、确定性、capture场景
            case HcclCMDType::HCCL_CMD_ALLGATHER:
                return std::tie(opType, DataDes.count, DataDes.dataType, tag, aivCoreLimit, deterministic, isCapture)
                       < std::tie(
                           other.opType, other.DataDes.count, other.DataDes.dataType, other.tag, other.aivCoreLimit,
                           other.deterministic, other.isCapture);
            case HcclCMDType::HCCL_CMD_ALLTOALL:
                return std::tie(
                           opType, All2AllDataDes.sendCount, All2AllDataDes.sendType, All2AllDataDes.recvCount,
                           All2AllDataDes.recvType, tag, aivCoreLimit, deterministic, isCapture)
                       < std::tie(
                           other.opType, other.All2AllDataDes.sendCount, other.All2AllDataDes.sendType,
                           other.All2AllDataDes.recvCount, other.All2AllDataDes.recvType, other.tag, other.aivCoreLimit,
                           other.deterministic, other.isCapture);
            case HcclCMDType::HCCL_CMD_BROADCAST:
                return std::tie(
                           opType, DataDes.count, DataDes.dataType, root, tag, aivCoreLimit, deterministic, isCapture)
                       < std::tie(
                           other.opType, other.DataDes.count, other.DataDes.dataType, other.root, other.tag,
                           other.aivCoreLimit, other.deterministic, other.isCapture);
            case HcclCMDType::HCCL_CMD_ALLREDUCE:
            case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
                return std::tie(
                           opType, DataDes.count, DataDes.dataType, reduceType, tag, aivCoreLimit, deterministic,
                           isCapture)
                       < std::tie(
                           other.opType, other.DataDes.count, other.DataDes.dataType, other.reduceType, other.tag,
                           other.aivCoreLimit, other.deterministic, other.isCapture);
            default:
                break;
        }
        return true;
    }
};

struct AlgDesc {
    bool isZeroCopy = false;
    bool isAivMode = false;
    bool isAivCrossNode = false;
    bool isLastSelect = false;
    s32 deterministic = -1; // -1:invalid，0:disable，1:enable，2:strict
    s32 aivTagNum = 1;
    AlgType algType;
    // executor所支持的各级算法，当vector为空时表示不校验，若外部传入的algType不支持，重定向为vector第一个元素
    // 由于默认算法要从列表里的第一个取，因此使用顺序确定的vector而非set
    std::vector<AlgTypeLevel0> level0SupportedAlgos;
    std::vector<AlgTypeLevel1> level1SupportedAlgos;
    std::vector<AlgTypeLevel2> level2SupportedAlgos;
};

struct ResourceLimit {
    bool ifLimit = false;
    bool ifCompileForAiv = false; // 图编译时选择AIV算法，不运行
    u32 aivCoreLimit = 0;
};

} // namespace hccl
#endif
