/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ALG_TEMPLATE_BASE_PUB_H
#define ALG_TEMPLATE_BASE_PUB_H

#include <cstring>
#include <vector>
#include <memory>
#include <list>
#include "hccl/base.h"
#include "externalinput_pub.h"
#include "mem_device_pub.h"
#include "stream_pub.h"
#include "transport_pub.h"
#include "adapter_pub.h"
#include "dispatcher.h"
#include "local_notify.h"
#include "template_v1_utils.h"
#include "op_context.h"
#include "comm_ahc_pub.h"

namespace hccl {
constexpr s32 HCCL_EXEC_STAGE_NOT_SET = -1;
constexpr s32 HCCL_EXEC_STEP_NOT_SET = -1;
constexpr s32 HCCL_EXEC_PLANE_NOT_SET = -1;
constexpr u64 ZERO_SLICE = 0;
constexpr u32 TWO_RANK_SIZE = 2;
constexpr u32 DMA_REDUCE_TWO_OFFSET = 2;
constexpr u32 DMA_REDUCE_THREE_OFFSET = 3;
constexpr u64 HCCL_CHUNK_SIZE = 1024 * 1024 * 1024; // 1024*1024*1024的size
constexpr u64 HCCL_MIN_PIPELINE_SLICE_ALIGN = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910B = 16384;
constexpr u64 HCCL_MIN_SLICE_ALIGN_910_93 = 16384;
constexpr u64 HCCL_MIN_SLICE_ALIGN_ONCHIP = 512;
constexpr u64 HCCL_MIN_SLICE_ALIGN = 128;
constexpr u64 HCCL_NIC_MAX_NUM = 8;
constexpr u64 DOUBLE_RING_NUM = 2;
constexpr u64 DOUBLE_RING_STREAM_NUM = 3;
constexpr u32 ALIGNED_SUB_RING_INDEX = 0;
constexpr u32 ALIGNED_MAIN_RING_INDEX = 1;

// AnyPath相关，SDMA数据量切分比例
constexpr u32 MAX_SPLIT_VALUE = 100;
constexpr u32 BEST_SPLIT_VALUE_SR = 87;
constexpr u32 BEST_SPLIT_VALUE_DR = 90;
constexpr u64 HCCL_SPLIT_SIZE_INTER_SERVER = 8388608; // 每卡通信量的切分边界

enum TemplateType {
    // 内置template
    TEMPLATE_ALL_GATHER_HD_STAGE = 0,          // AllGatherHDStage
    TEMPLATE_ALL_2_ALL_V_DIRECT_FULL_MESH = 1, // AlltoAllVDirectFullMesh
    TEMPLATE_ALL_REDUCE_REDUCE_BCAST = 2,      // AllReduceReduceBcast
    TEMPLATE_BROADCAST_NHR_V1 = 3,             // BroadcastNHRV1
    TEMPLATE_BROADCAST_NHR = 4,                // BroadcastNHR
    TEMPLATE_BROADCAST_NHR_ONESHOT = 5,        // BroadcastNHROneshot
    TEMPLATE_BROADCAST_NB = 6,                 // BroadcastNB
    TEMPLATE_BROADCAST_NB_BINARY = 7,          // BroadcastNBBinary
    TEMPLATE_BROADCAST_HD = 8,                 // BroadcastHD
    TEMPLATE_BROADCAST_RECURSIVE_HD = 10,      // BcastRecursiveHalvingDoubling
    TEMPLATE_BROADCAST_RING = 11,              // BroadcastRing
    TEMPLATE_BROADCAST_STAR = 12,              // BroadcastStar
    TEMPLATE_ALL_2_ALL_V_FOR310P = 13,         // AlltoAllVFor310P
    TEMPLATE_ALL_2_ALL_V_PAIRWISE = 15,        // AlltoAllVPairwise
    TEMPLATE_ALL_2_ALL_V_STAGED_MESH = 16,     // AlltoAllVStagedMesh
    TEMPLATE_ALL_2_ALL_V_STAGED_PAIRWISE = 17, // AlltoAllVStagedPairwise
    TEMPLATE_REDUCESCATTER_HDSTAGE = 18,       // ReduceScatterHDStage
    TEMPLATE_REDUCESCATTER_LOCAL_REDUCE = 19,  // ReduceScatterLocalReduce
    TEMPLATE_REDUCESCATTER_NB = 20,            // ReduceScatterNB
    TEMPLATE_REDUCESCATTER_NHR = 21,           // ReduceScatterNHR
    TEMPLATE_REDUCESCATTER_NHR_V1 = 22,        // ReduceScatterNHRV1
    TEMPLATE_REDUCESCATTER_PIPELINE = 23,      // ReduceScatterPipeline
    TEMPLATE_REDUCESCATTER_UNIFIED_MARCH = 24, // ReduceScatterUnifiedMarch
    TEMPLATE_REDUCESCATTER_DB_RING_SLC = 25,   // AlignedReduceScatterDoubleRingWithSerialLocalCopy
    TEMPLATE_REDUCESCATTER_DB_RING = 26,       // AlignedReduceScatterDoubleRing
    TEMPLATE_REDUCESCATTER_HD = 27,            // ReduceScatterHalvingDoubling
    TEMPLATE_REDUCESCATTER_MESH_DIRECT = 28,   // ReduceScatterMeshDirect
    TEMPLATE_REDUCESCATTER_MESH_ATOMIC = 29,   // ReduceScatterMeshAtomic
    TEMPLATE_REDUCESCATTER_MESH_MIX_SS = 30,   // ReduceScatterMeshMixSingleStream
    TEMPLATE_REDUCESCATTER_MESH_MIX = 31,      // ReduceScatterMeshMix
    TEMPLATE_REDUCESCATTER_MESH = 32,          // ReduceScatterMesh
    TEMPLATE_REDUCESCATTER_RECURSIVE_HD = 33,  // ReduceScatterRecursiveHalvingDoubling
    TEMPLATE_REDUCESCATTER_RING_DIRECT = 34,   // ReduceScatterRingConcurrentDirect
    TEMPLATE_REDUCESCATTER_RING = 35,          // ReduceScatterRing
    TEMPLATE_ALL_REDUCE_RECURSIVE_HALVING_DOUBLING = 36,
    TEMPLATE_ALL_REDUCE_RING = 37,
    TEMPLATE_REDUCE_RECURSIVE_HALVING_DOUBLING = 38,
    TEMPLATE_REDUCE_RING = 39,
    TEMPLATE_REDUCE_NHR_ONE_SHOT = 40,
    TEMPLATE_ALL_REDUCE_CHUNK_MESH = 41,
    TEMPLATE_ALL_REDUCE_DOUBLING_DIRECT = 42,
    TEMPLATE_ALL_REDUCE_DOUBLING = 43,
    TEMPLATE_ALL_REDUCE_HD_OPTIM = 44,
    TEMPLATE_ALL_REDUCE_LOCAL_REDUCE_BCAST = 45,
    TEMPLATE_ALL_REDUCE_LOCAL_REDUCE = 46,
    TEMPLATE_ALL_REDUCE_MESH_DIRECT_ONESHOT = 47,
    TEMPLATE_ALL_REDUCE_MESH_DIRECT = 48,
    TEMPLATE_ALL_REDUCE_NB = 49,
    TEMPLATE_ALL_REDUCE_NHR_ONESHOT = 50,
    TEMPLATE_ALL_REDUCE_NHR_V1 = 51,
    TEMPLATE_ALL_REDUCE_NHR = 52,
    TEMPLATE_ALL_REDUCE_OPBASE_PIPELINE = 53,
    TEMPLATE_ALIGNED_ALL_GATHER_DOUBLE_RING = 54,        // AlignedAllGatherDoubleRing
    TEMPLATE_ALL_GATHER_HALVING_DOUBLING = 55,           // AllGatherHalvingDoubling
    TEMPLATE_ALL_GATHER_MESH = 56,                       // AllGatherMesh
    TEMPLATE_ALL_GATHER_MESH_ATOMIC = 57,                // AllGatherMeshAtomic
    TEMPLATE_ALL_GATHER_MESH_DIRECT = 58,                // AllGatherMeshDirect
    TEMPLATE_ALL_GATHER_MESH_MIX = 59,                   // AllGatherMeshMix
    TEMPLATE_ALL_GATHER_RECURSIVE_HALVING_DOUBLING = 60, // AllGatherRecursiveHalvingDoubling
    TEMPLATE_ALL_GATHER_RING_CONCURRENT_DIRECT = 61,     // AllGatherRingConcurrentDirect
    TEMPLATE_ALL_GATHER_RING = 62,                       // AllGatherRing
    TEMPLATE_ALL_GATHER_NB = 63,                         // AllGatherNB
    TEMPLATE_ALL_GATHER_NHRV1 = 64,                      // AllGatherNHRV1
    TEMPLATE_ALL_GATHER_NHR = 65,                        // AllGatherNHR
    TEMPLATE_ALL_GATHER_PIPELINE = 66,                   // AllGatherPipeline
    TEMPLATE_ALL_GATHER_UNIFIED_MARCH = 67,              // AllGatherUnifiedMarch

    TEMPLATE_MULTI_ROOT_SCATTER_RING = 68,        // MultiRootScatterRing
    TEMPLATE_SCATTER_DOUBLE_RING_DIRECT = 69,     // ScatterDoubleRingDirect
    TEMPLATE_SCATTER_MESH = 70,                   // ScatterMesh
    TEMPLATE_SCATTER_RING_CONCURRENT_DIRECT = 71, // ScatterRingConcurrentDirect
    TEMPLATE_SCATTER_RING = 72,                   // ScatterRing
    TEMPLATE_SCATTER_NB = 73,                     // ScatterNB
    TEMPLATE_SCATTER_NHR = 74,                    // ScatterNHR
    TEMPLATE_SCATTER_RING_DIRECT = 75,            // ScatterRingDirect

    TEMPLATE_GATHER_MESH = 76, // GatherMesh
    TEMPLATE_GATHER_RING = 77, // GatherRing
    TEMPLATE_GATHER_STAR = 78, // GatherStar

    TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_CCL_ENOUGH = 79, // AlltoallPipelineMeshPairwiseCCLEnough
    TEMPLATE_ALL_2_ALL_PIPELINE_MESH_PAIRWISE_PING_PONG = 80,  // AlltoallPipelineMeshPairwisePingPong

    TEMPLATE_ALL_REDUCE_AHC = 81,
    TEMPLATE_ALL_REDUCE_AHC_BROKE = 82,
    TEMPLATE_ALL_GATHER_AHC = 83,
    TEMPLATE_ALL_GATHER_AHC_BROKE = 84,
    TEMPLATE_REDUCESCATTER_AHC = 85,
    TEMPLATE_REDUCESCATTER_AHC_BROKE = 86,
    TEMPLATE_ALL_GATHER_RING_DIRECT = 87, // AllGatherRingDirect
    TEMPLATE_ALL_GATHER_HCCS_SIO = 88,
    TEMPLATE_REDUCESCATTER_HCCS_SIO = 89,

    TEMPLATE_ALLREDUCE_GRAPH_PIPELINE = 90, // AllReduceGraphPipeline

    TEMPLATE_ALL_GATHER_GRAPH_PIPELINE = 91,    // AllGatherGraphPipeline AG图模式pipeline
    TEMPLATE_REDUCESCATTER_GRAPH_PIPELINE = 92, // ReduceScatterGraphPipeline AG图模式pipeline

    TEMPLATE_ALL_GATHER_SLIM_RING = 93,
    TEMPLATE_REDUCESCATTER_SLIM_RING = 94,

    TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE = 95,         // ReduceScatterPlantLocalReduce RS规约保序单机
    TEMPLATE_REDUCESCATTER_PLANT_LOCAL_REDUCE_COMBINE = 96, // ReduceScatterPlantLocalReduceCombine RS规约保序跨机
    TEMPLATE_REDUCESCATTER_V_PIPELINE = 97,                 // ReduceScatterVPipeline RSV多机Pipeline

    TEMPLATE_ALL_GATHER_V_PIPELINE = 98, // AllGatherV pipeline

    TEMPLATE_ALL_REDUCE_DOUBLING_LOCAL_REDUCE = 99, // AllReduceDoublingLocalReduce AR 910A单机小数据量tbe reduce优化

    TEMPLATE_ALL_2_ALL_V_CONTINUOUS_PIPELINE = 100, // AlltoallvContinuousPipeline

    TEMPLATE_ALL_GATHER_V_GRAPH_PIPELINE = 101, // AllGatherV Graph pipeline
    TEMPLATE_REDUCESCATTER_MULTI_DETERMINISTIC_PIPELINE = 102,
    TEMPLATE_ALL_REDUCE_MULTI_DETERMINISTIC_PIPELINE = 103,
    TEMPLATE_ALL_2_ALL_FULL_MESH_SYMMETRIC_MEMORY = 104,

    TEMPLATE_NATIVE_MAX_NUM, // 内置template最大值

    TEMPLATE_CUSTOM_BEGIN = 1000,  // 用户自定义template起始值
    TEMPLATE_CUSTOM_MAX_NUM = 2000 // 用户自定义template最大值
};

enum class SliceType { SLICE_TYPE_TX, SLICE_TYPE_RX };

using GroupSlicesInfo = std::vector<MemBlockInfo>;

enum class HalvingDoublingType { BINARY_BLOCK_HALVING_DOUBLING, RECURSIVE_HALVING_DOUBLING, RESERVED_ALGORITHM_TYPE };

using SliceType = enum SliceType;

enum class RunStage { RUN_PREPARE, RUN_REDUCE_SCATTER, RUN_ALLGATHER, RUN_ALLREDUCE, RUN_DEFAULT };

struct PrepareData {
    u32 root = INVALID_VALUE_RANKID;
    u32 userRank = INVALID_VALUE_RANKID;
    u32 userRankSize = 0;
    u32 interRank = INVALID_VALUE_RANKID;
    u32 interRankSize = 0;

    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;
    HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED;
    u64 baseOffset = 0;

    DeviceMem inputMem;
    DeviceMem outputMem;
    DeviceMem scratchMem;
    DeviceMem cclInMem;
    DeviceMem cclOutMem;

    Stream stream;
    const std::vector<Stream>* subStreamsPtr = nullptr;
    const std::vector<std::shared_ptr<LocalNotify>>* signalPtr = nullptr;
    const std::vector<std::shared_ptr<LocalNotify>>* signalAuxPtr = nullptr;

    const std::vector<LINK>* linksPtr = nullptr;
    const std::vector<Slice>* slicesPtr = nullptr;
    const std::vector<std::vector<Slice>>* multRingsSlicesPtr = nullptr;
    const std::vector<u32>* nicRankListPtr = nullptr;

    HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_RESERVED;
    HcomCollOpInfo* opInfo = nullptr;
    bool disableDMAReduce = false;
    bool isSuPodAsym = false;
    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;

    const SendRecvInfo* localSendRecvInfoPtr = nullptr;
    const ZCopySendRecvInfo* sendRecvInfoPtr = nullptr;
    u32 devNumInlocalPod = 0;
    u32 rankIdxInPod = 0;
    u64 reduceAttr = 0;

    AlgOpContext algOpContext;

    bool needAlltoallvCache = false; // 用于alltoallv类算子的aicpu cache
};

struct HcclTopoInfo;
class TopoMatcher;
class ExecutorBase {
public:
    explicit ExecutorBase(const HcclDispatcher dispatcher);
    virtual ~ExecutorBase();

    virtual HcclResult RunAsync();
    virtual HcclResult
    RunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links);
    virtual HcclResult RunAsyncStaged(
        const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links, RunStage stage);
    /* 12个参数 */
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED,
        const u32 root = INVALID_VALUE_RANKID, const std::vector<Slice>& slices = std::vector<Slice>(ZERO_SLICE),
        const u64 baseOffset = 0, std::vector<u32> nicRankList = {0, 1, 2, 3, 4, 5, 6, 7},
        const bool disableDMAReduce = false);

    /* 11个参数 */
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType, const Stream& stream,
        const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED, const u32 root = INVALID_VALUE_RANKID,
        const std::vector<Slice>& slices = std::vector<Slice>(ZERO_SLICE), const u64 baseOffset = 0,
        std::vector<u32> nicRankList = {0, 1, 2, 3, 4, 5, 6, 7}, const bool disableDMAReduce = false);

    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const std::vector<std::vector<Slice>>& multRingsSlices,
        const HcclReduceOp reductionOp = HCCL_REDUCE_RESERVED, const u32 root = INVALID_VALUE_RANKID,
        const u64 baseOffset = 0, const bool disableDMAReduce = false);

    virtual HcclResult Prepare(PrepareData& param);

    /* 1个参数 */
    // AllGatherNHR, ScatterNHR
    virtual HcclResult Prepare(bool needSaveRankMap);

    // AHC 扩展参数
    virtual HcclResult Prepare(AHCExtendPreparePara& extendParam);

    // GatherStar
    virtual HcclResult Prepare(u32 userRank);

    /* 2个参数 */
    // ReduceScatterNB, ReduceScatterNHRV1, ReduceScatterRing, ReduceScatterRecursiveHalvingDoubling
    virtual HcclResult Prepare(u64 reduceAttrBitMap, HcomCollOpInfo* opInfo = nullptr);

    // ReduceScatterNHR
    virtual HcclResult Prepare(u64 reduceAttrBitMap, bool needMerge);

    // ReduceScatterMeshMixSingleStream, ReduceScatterMesh
    virtual HcclResult Prepare(u64 reduceAttrBitMap, u32 streamIndex);

    // ScatterMesh
    virtual HcclResult Prepare(u32 interRank, u32 interRankSize);

    /* 3个参数 */
    // for AllGatherHalvingDoubling based on input_scratch_Mem_nicRankList Prepare
    // and should be called soon template AllGatherHalvingDoubling created
    virtual HcclResult Prepare(u32 blockSize, UserMemType hdInputMemType, UserMemType hdOutputMemType);

    /* 4个参数 */
    // ScatterRingDirect
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, const std::vector<u32>& ringsOrders,
        const std::vector<Slice>& userMemInputSlices);

    // AllGatherRingDirect
    virtual HcclResult
    Prepare(HcomCollOpInfo* opInfo, u32 userRank, const std::vector<Slice>& userMemOutputSlices, bool isSdma = true);

    /* 5个参数 */
    // AHC 5个参数，带扩展参数
    virtual HcclResult Prepare(
        u64 totalCount, const std::vector<std::vector<std::vector<u32>>>& subGroups,
        std::map<AHCConcOpType, TemplateType>& ahcAlgOption, bool extendFlag = false,
        AHCExtendPreparePara extendPara = AHCExtendPreparePara());

    /* 6个参数 */
    // AlltoAllVStagedPairwise
    virtual HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode, Stream& mainStream);

    /* 7个参数 */
    virtual HcclResult Prepare(
        u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 userRank, HcomCollOpInfo* opInfo, bool aicpu);

    // AlltoAllVPairWise
    virtual HcclResult Prepare(
        AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, bool isAlltoAllZCopyMode,
        const Stream& stream, HcclWorkflowMode workMode, std::map<u32, std::vector<u64>>& rankSendDisplsMap,
        std::map<u32, std::vector<u64>>& rankRecvDisplsMap);

    // AlignedAllGatherDoubleRing
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        std::vector<std::shared_ptr<LocalNotify>>& mainSignals, std::vector<std::shared_ptr<LocalNotify>>& subSignals,
        const std::vector<std::vector<u32>>& ringsOrders,
        const std::vector<std::vector<Slice>>& userMemOutputSlicesOfDoubleRing);

    // AllGatherMeshAtomic, AllgatherMeshDirect, AllGatherMesh, AllGatherMeshMix, GatherMesh
    virtual HcclResult Prepare(
        std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 userRank = INVALID_VALUE_RANKID,
        HcomCollOpInfo* opInfo = nullptr, u32 interRank = INVALID_VALUE_RANKID, u32 interRankSize = 0);

    /* 8个参数 */
    virtual HcclResult Prepare(
        u64 reduceAttrBitMap, std::vector<Stream>& meshStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize, u32 userRank,
        HcomCollOpInfo* opInfo);

    // AlltoAllVStagedPairwise
    virtual HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, DeviceMem& scratchInputMem, DeviceMem& scratchOutputMem,
        StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode,
        Stream& mainStream);

    // AllGatherRingConcurrentDirect ScatterRingConcurrentDirect
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<u32>& ringsOrder,
        const std::vector<Slice>& userMemSlices, bool isSdma = true);

    /* 9个参数 */
    // scatterDoubleRingDirect
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, const u32 userRank, const u32 subRingRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<std::vector<u32>>& ringsOrders,
        const std::vector<std::vector<Slice>>& multiRingSlices,
        const std::vector<std::vector<Slice>>& userMemInputSlices);

    // ReduceScatterRingConcurrentDirect
    virtual HcclResult Prepare(
        const u64 reduceAttrBitMap, const HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<u32>& ringsOrder,
        const std::vector<Slice>& userMemInputSlices, bool isSdma = true);

    // AlltoAllVPairWise
    virtual HcclResult Prepare(
        AlltoAllVBufferInfo& sendBuffer, AlltoAllVBufferInfo& recvBuffer, DeviceMem& scratchInputMem,
        DeviceMem& scratchOutputMem, bool isAlltoAllZCopyMode, const Stream& stream, HcclWorkflowMode workMode,
        std::map<u32, std::vector<u64>>& rankSendDisplsMap, std::map<u32, std::vector<u64>>& rankRecvDisplsMap);

    /* 10个参数 */
    virtual HcclResult Prepare(
        const HcomCollOpInfo* opInfo, DeviceMem& cclBufferA, DeviceMem& cclBufferB, const u64 count,
        const SubCommInfo& level1CommInfo, const SubCommInfo& level0CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub);

    // AlltoAllPipelineMeshPairwiseCCLEnough, AlltoAllPipelineMeshPairwisePingPong
    virtual HcclResult Prepare(
        u32 userRank, A2aPipelineMemory A2aPipelineMemory, const SubCommInfo& level0CommInfo,
        const SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
        std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
        std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo,
        HcclWorkflowMode workMode = HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE);

    // AlltoAllVSatgedMesh
    virtual HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, StageAlltoAllVAddrInfo& sendAddrInfo,
        StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode, u32 userRank, Stream& mainStream,
        std::vector<Stream>& subStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain);

    // ReduceScatterPlantLocalReduce
    virtual HcclResult Prepare(
        void* inputMemPtr, DeviceMem& cclInMem, DeviceMem& outputMem, const Stream& stream,
        std::vector<Stream>& subStreams, std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, GroupSlicesInfo& grouSlicesInfo,
        const HcclReduceOp reductionOp, u32 all2allOffset, const HcclDataType dataType, bool isNeedSpaceBorrow,
        bool reverseMemUsage = false, bool isA3CrossNode = false);

    // ReduceScatterPlantLocalReduceCombine
    virtual HcclResult Prepare(
        DeviceMem& cclInMem, DeviceMem& outputMem, const Stream& stream, std::vector<Stream>& subStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        MemBlockInfo& memBlockInfo, const HcclReduceOp reductionOp, const HcclDataType dataType, bool isUseCclIn,
        bool isLevel0LastRank, bool isNeedSpaceBorrow);

    /* 11个参数 */
    // Prepare for AllGatherPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, u32 userRank, u64& count, DeviceMem& cclBufferPartOne, DeviceMem& cclBufferPartTwo,
        SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
        std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub);

    // Prepare for AllGatherUnifiedMarch
    virtual HcclResult Prepare(
        const Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput,
        DeviceMem& usrInMem, DeviceMem& usrOutMem, u64 blockDataByte, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain,
        const std::vector<std::vector<Slice>>& multRingsUserMemSlice, const u64 baseOffset = 0);

    // Prepare for AllGatherHccsSio
    virtual HcclResult Prepare(
        SubCommInfo& outerCommInfoHccs, SubCommInfo& outerCommInfoSio, DeviceMem& usrInMem, DeviceMem& usrOutMem,
        u64 totalCount, const HcclDataType dataType, const Stream& mainStream, std::vector<Stream>& meshStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        u32 userRank, HcomCollOpInfo* opInfo);

    // Prepare for AlltoAllvContinuousPipeline
    virtual HcclResult Prepare(
        const u32 userRank, const A2aPipelineMemory& a2aPipelineMemory, const SubCommInfo& level0CommInfo,
        const SubCommInfo& level1CommInfo, const Stream& mainStream, std::vector<Stream>& subStream,
        std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
        std::vector<SendRecvInfo>& sendRecvInfoList, const HcclDataType dataType, const HcclWorkflowMode workMode,
        u32 timeOut);

    /* 12个参数 */
    // AlltoAllVFor310P
    virtual HcclResult Prepare(
        DeviceMem& userInput, DeviceMem& userOutput, DeviceMem& cclInMem, DeviceMem& cclOutMem,
        const std::vector<std::shared_ptr<LocalNotify>>& signalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& signalSubToMain, Stream& mainStream,
        std::vector<Stream>& subStreams, const std::vector<LINK>& links, u32 userRank, u32 userRankSize,
        std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo);

    // AlltoAllVStagedMesh
    virtual HcclResult Prepare(
        DeviceMem& sendMem, DeviceMem& recvMem, DeviceMem& scratchInputMem, DeviceMem& scratchOutputMem,
        StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo, bool isAlltoAllZCopyMode,
        u32 userRank, Stream& mainStream, std::vector<Stream>& subStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain);

    // ReduceScatterPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 count, const u64 bufferSize, const u64 offset,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap);

    // ReduceScatterVPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& cclBuffer, const u64 bufferSize, const std::vector<Slice>& slices,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub, u64 reduceAttrBitMap);

    // BroadcastStar
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, std::vector<u32> nicRankList, u32 userRank);

    // Prepare for AllGatherVPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, u32 userRank, u64& count, DeviceMem& cclBufferPartOne, DeviceMem& cclBufferPartTwo,
        SubCommInfo& level0CommInfo, SubCommInfo& level1CommInfo, Stream& mainStream, std::vector<Stream>& subStream,
        std::vector<std::shared_ptr<LocalNotify>>& notifyMain, std::vector<std::shared_ptr<LocalNotify>>& notifySub,
        std::vector<Slice>& userOutSlice);

    /* 13个参数 */
    // BroadcastHD
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, std::vector<Stream>& meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, const HcomCollOpInfo* opInfo);

    /* 14个参数 */
    // ReduceScatterUnifiedMarch
    virtual HcclResult Prepare(
        Stream& mainStream, SubCommInfo& level0CommInfo, DeviceMem& userInput, DeviceMem& userOutput,
        DeviceMem& usrInMem, DeviceMem& scratchMem, u64 totalCount, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain, const HcclDataType dataType,
        const HcclReduceOp reductionOp, const std::vector<std::vector<Slice>>& multRingsUserMemSlice,
        u64 reduceAttrBitMap);

    // ReduceScatterHalvingDoubling
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u32 blockSize, const u64 reduceAttrBitMap, const UserMemType hdInputMemType,
        const UserMemType hdOutputMemType);

    /* 15个参数 */
    // AlltoAllVMeshReadOnly
    virtual HcclResult Prepare(
        DeviceMem& userInput, DeviceMem& userOutput, DeviceMem& scratchPingMem, DeviceMem& scratchPongMem,
        StageAlltoAllVAddrInfo& sendAddrInfo, StageAlltoAllVAddrInfo& recvAddrInfo, HcclWorkflowMode workMode,
        Stream& mainStream, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalMainToSub,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalSubToMain, u32 userRank, u32 intraRankSize,
        const std::vector<LINK>& links, std::vector<SendRecvInfo>& allMeshAggregationSendRecvInfo);

    /* 16个参数 */
    // ReduceScatterHDStage, ReduceScatterLocalReduce, ReduceScatterMeshAtomic, ReduceScatterMeshDirect
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        u32 userRank, const HcomCollOpInfo* opInfo = nullptr);

    // ReduceScatterHccsSio
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const u64 baseOffset,
        const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
        std::vector<std::shared_ptr<LocalNotify>>& meshSignal, std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux,
        u32 userRank, SubCommInfo subCommInfoHccs, SubCommInfo subCommInfoSio, HcomCollOpInfo* opInfo);

    /* 17个参数 */
    // ReduceScatterMeshMix
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const HcclReduceOp reductionOp, const u32 root, const std::vector<Slice>& slices,
        const u64 baseOffset, const u64 reduceAttrBitMap, std::vector<Stream>& meshStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignal,
        const std::vector<std::shared_ptr<LocalNotify>>& meshSignalAux, u32 interRank, u32 interRankSize,
        HcomCollOpInfo* opInfo);

    /* 19个参数 */
    // AlignedReduceScatterDoubleRing, AlignedReduceScatter, DoubleRingWithSerialLocalCopy
    virtual HcclResult Prepare(
        DeviceMem& inputMem, DeviceMem& outputMem, DeviceMem& scratchMem, const u64 count, const HcclDataType dataType,
        const Stream& stream, const std::vector<std::vector<Slice>>& multRingsSlices, const HcclReduceOp reductionOp,
        const u32 root, const u64 baseOffset, const bool disableDMAReduce, const u64 reduceAttrBitMap,
        const HcomCollOpInfo* opInfo, const u32 userRank, std::vector<Stream>& subStreams,
        const std::vector<std::shared_ptr<LocalNotify>>& mainSignals,
        const std::vector<std::shared_ptr<LocalNotify>>& subSignals, const std::vector<std::vector<u32>>& ringsOrders,
        const std::vector<std::vector<Slice>>& userMemInputSlicesOfDoubleRing);

    // ReduceScatterDeterPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& buffer, const u64 count, const u64 offset, const std::vector<Slice>& slices,
        const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo, Stream& mainStream,
        std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub);

    // AllReduceDeterPipeline
    virtual HcclResult Prepare(
        HcomCollOpInfo* opInfo, DeviceMem& inBuffer, DeviceMem& outBuffer, const u64 count,
        const std::vector<Slice>& slices, const SubCommInfo& level0CommInfo, const SubCommInfo& level1CommInfo,
        Stream& mainStream, std::vector<Stream>& subStream, std::vector<std::shared_ptr<LocalNotify>>& notifyMain,
        std::vector<std::shared_ptr<LocalNotify>>& notifySub);

    HcclResult Sum(const std::vector<Slice>& inputSlices, u32 start, u32 num, u64& sizeOut);
    HcclResult RegisterProfiler(s32 planeId, s32 stage, s32 step, const Stream& stream);
    static HcclResult
    ExecEmptyTask(DeviceMem& inputMem, DeviceMem& outputMem, Stream& stream, const HcclDispatcher dispatcher);
    HcclResult CheckConcurrentDirectParameters(const u32 rank, const u32 rankSize, const std::vector<LINK>& links);
    u32 DataUnitSize(HcclDataType dataType) const
    {
        if (dataType >= HCCL_DATA_TYPE_RESERVED) {
            HCCL_ERROR(
                "[AlgTemplateBase][DataUnitSize]data type[%s] out of range[%d, %d]",
                GetDataTypeEnumStr(dataType).c_str(), HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
            return 0;
        }

        return SIZE_TABLE[dataType];
    }

    static std::vector<bool> CalcLinksRelation(
        const u32 rank, const u32 rankSize, const u32 rootRank = 0,
        HalvingDoublingType algorithmType = HalvingDoublingType::RECURSIVE_HALVING_DOUBLING);

    static HcclResult
    PrepareSliceData(u64 dataCount, u32 unitSize, u32 sliceNum, u64 pipelineOffset, std::vector<Slice>& dataSlice);
    static HcclResult PrepareSliceMeshStreams(
        const std::vector<Slice>& rankSegsSlice, u32 streamCount, std::vector<std::vector<Slice>>& multiStreamsSlices);

    static inline u64 RoundUpWithDivisor(u64 value, u64 divisor)
    {
        if ((value == 0) || (divisor == 0)) {
            return divisor;
        }
        // divisor必须大于等于1, 返回value向上取divisor的整数倍的值
        return ((value + (divisor - 1)) / divisor) * divisor;
    }
    inline u64 ByteOffset(u64 countOffset) const { return countOffset * DataUnitSize(dataType_); }
    inline u64 SliceOffset(u32 sliceIndex, u64 countPerSlice) const
    {
        return sliceIndex * countPerSlice * DataUnitSize(dataType_);
    }
    inline void CloseBarrier() { barrierSwitchOn_ = false; }
    virtual HcclResult
    GetNslbAdjInfo(const u32 rank, const u32 rankSize, const std::vector<LINK>& links, AdjInfo& nslbAdjInfo);

    // 只用于alltoallv类算子的aicpu cache
    virtual HcclResult
    GetHcclOffsetDstRanksMap(std::unordered_map<uint64_t, std::vector<uint32_t>>& hcclOffsetDstRanksMap) const;

protected:
    HcclResult ExecuteBarrier(const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink);
    HcclResult ExecuteBarrier(
        const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink, Stream& stream);
    HcclResult
    ExecuteBarrier(const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink, u32 notifyIdx);
    HcclResult ExecuteBarrier(
        const std::shared_ptr<Transport>& preLink, const std::shared_ptr<Transport>& aftLink, u32 notifyIdx,
        Stream& stream);
    HcclResult ExecuteBarrier(std::shared_ptr<Transport> link, Stream& stream);
    HcclResult ExecuteRxSync(
        std::shared_ptr<Transport> link, UserMemType srcMemType, u64 srcOffset, void* dst, u64 len,
        Stream& stream) const;
    HcclResult ExecuteTxSync(
        std::shared_ptr<Transport> link, UserMemType dstMemType, u64 dstOffset, void* src, u64 len,
        Stream& stream) const;
    virtual HcclResult
    PrepareRunAsync(const u32 rank, const u32 rankSize, const std::vector<std::shared_ptr<Transport>>& links);
    const HcclDispatcher dispatcher_;
    std::vector<Slice> slicesDummy_;
    std::vector<Slice>& slices_;
    DeviceMem inputMem_;   /* * 输入memory */
    DeviceMem outputMem_;  /* * 输出memory */
    DeviceMem scratchMem_; /* * 草稿memory */

    u64 count_;     //  需处理的每块memory数据总个数
    u64 dataBytes_; //  数据所占的字节数
    HcclDataType dataType_;
    HcclReduceOp reductionOp_;
    u32 root_;
    bool disableDMAReduce_;

    // Added on Mar.24th, for profiling template
    StepData profilerInput_;
    u64 baseOffset_;

    Stream stream_;

    // 用于chunk算法
    std::vector<u32> nicRankList_;
    std::vector<std::vector<u32>> rankSliceLists_;
    bool barrierSwitchOn_;
    // 用于91093 aligned double ring算法
    std::vector<std::vector<Slice>> multRingsSlices_;
    AlgOpContext algOpContext_;

private:
    static void CalcBinaryBlockParams(
        u32 rank, u32 rankSize, u32& stepsInBlock, u32& lowerBlockSize, u32& myBlockSize, u32& rankInMyBlock,
        u32& myBlockOffset, u32& higherBlockSize);
    static HcclResult
    CalcBinaryBlockHalvingDoubleLinkReleation(u32 rank, u32 rankSize, std::vector<bool>& linkRelation);

    static void CalcLinkInBlock(u32 blockSize, u32 rankInBlock, std::list<u32>& linkRankIndexInBlock);
    static void CalcLinkBetweenParts(
        u32 part1Size, std::list<u32>& linkRankIndexInBlock, std::list<u32>& linkRankIndex, bool oddRank);
    static void
    CalcRecursiveHalvingDobuleLinkReleation(u32 rank, u32 rankSize, u32 rootRank, std::vector<bool>& linkRelation);
    static void
    CalcRecursiveHdLinkRelationForFirstScene(u32 rank, u32 part1Size, u32 blockSize, std::vector<bool>& linkRelation);
    static void
    CalcRecursiveHdLinkRelationForSecondScene(u32 rank, u32 part1Size, u32 blockSize, std::vector<bool>& linkRelation);
};
using AlgTemplateBase = ExecutorBase;
} // namespace hccl

#endif /* EXECUTOR_BASE_PUB_H */
