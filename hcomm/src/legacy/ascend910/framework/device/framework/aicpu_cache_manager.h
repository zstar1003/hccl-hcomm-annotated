/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_CACHE_MANAGER_H__
#define __AICPU_CACHE_MANAGER_H__

#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>
#include <string>
#include <stdint.h>

#include "aicpu_zero_copy_exchanger.h"
#include "coll_alg_param.h"
#include "coll_executor_base.h"
#include "hccl_types.h"
#include "stream_pub.h"
#include "topo_matcher.h"
#include "op_context.h"
#include "workflow.h"
#include "op_unfold_cache.h"
#include "mem_device_pub.h"

namespace hccl {

constexpr uint8_t FORCE_OP_BASE_DELTA = 10;

// A3消息语义算子展开的动态缓存
class AicpuCacheManager {
public:
    AicpuCacheManager();
    ~AicpuCacheManager();

    // 初始化op-unfold cache
    HcclResult InitOpUnfoldCache();

    // 查找op-unfold cache, hit则直接刷新并下发缓存的SQE
    HcclResult LookupOpUnfoldCache(
        const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource, bool& needExecute,
        bool& isCacheMiss, Stream& mainStream, std::vector<Stream>& slaveStreams, void* dispatcherPtr,
        const bool isDeviceMode, const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr,
        const AlgOpContext& algContext, std::shared_ptr<AicpuZeroCopyExchanger>& zeroCopyExchangerPtr,
        const HcclWorkflowMode workflowMode, const DeviceMem& tinySendRecvMem,
        std::function<HcclResult()> setProfStartCallback);

    // 针对cache miss的预处理, 包括使能alltoallv executor感知cache
    HcclResult PreProcessForCacheMiss(const OpParam& param, std::unique_ptr<CollExecutorBase>& executor);
    // 针对cache miss的后处理, 包括清理launch context、更新op-unfold cache中后续cache hit需要使用到的相关信息等
    HcclResult PostProcessForCacheMiss(
        const OpParam& param, std::unique_ptr<CollExecutorBase>& executor, Stream& mainStream,
        std::vector<Stream>& slaveStreams, void* dispatcherPtr, const HcclTopoInfo& topoinfo,
        const AlgOpContext& algContext, const HcclWorkflowMode workflowMode);

    // 故障快恢/重执行时清理中断的cache entry (if any), 避免命中不完整的缓存
    HcclResult ClearOpUnfoldCacheEntry(
        const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource,
        const bool isDeviceMode, const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr,
        const AlgOpContext& algContext, const HcclWorkflowMode workflowMode);

    // 设置对称内存模式
    void SetSymmetricMemoryEnable(bool enable);

private:
    // 故障快恢/重执行时清理alltoallv metadata, 避免复用不完整的metadata
    HcclResult ClearMetadataForFirstAlltoallv();

    // 判断是否需要使用aicpu op-unfold cache
    HcclResult NeedOpUnfoldCache(
        const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource,
        const bool isDeviceMode, const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr,
        const AlgOpContext& algContext, const HcclWorkflowMode workflowMode, bool& needCache);
    HcclResult IsInplace(
        const OpParam& param, bool& isInplace, const HcclTopoInfo& topoinfo); // 是否为inplace场景 (inplace则不做cache)
    bool IsAlltoallvType(const HcclCMDType opType);                           // 是否为alltoallv类型的算子
    HcclResult IsSmallDataAlltoallv(
        const OpParam& param, bool& isSmallData, const HcclTopoInfo& topoinfo); // 是否为小数据量的alltoallv类型的算子
    HcclResult HasRemoteDataAlltoallv(const OpParam& param, bool& hasRemoteData, const HcclTopoInfo& topoinfo);

    // 为第一个可能被cache的alltoallv算子计算metadata
    HcclResult CalcMetadataForFirstAlltoallv(
        const AlgResourceResponse& algResource, const bool isDeviceMode, const HcclTopoInfo& topoinfo,
        std::unique_ptr<TopoMatcher>& topoMatcherPtr, const AlgOpContext& algContext);

    // 获得op-unfold key for cache
    HcclResult GetOpUnfoldKey(
        const OpParam& param, OpUnfoldKey& key, const HcclTopoInfo& topoinfo, const AlgOpContext& algContext,
        const HcclWorkflowMode workflowMode);
    HcclResult IsBigCountForAlltoallv(const OpParam& param, const HcclTopoInfo& topoinfo, bool& isBigCount);

    // 准备user input/output memory ranges
    HcclResult PrepareUserMemRanges(
        const OpParam& param, const AlgResourceResponse& algResource, std::vector<OpUnfoldMemRange>& userInputMemRanges,
        std::vector<OpUnfoldMemRange>& userOutputMemRanges, const HcclTopoInfo& topoinfo,
        std::shared_ptr<AicpuZeroCopyExchanger>& zeroCopyExchangerPtr, const HcclWorkflowMode workflowMode,
        const DeviceMem& tinySendRecvMem);

    // 从param中解析相关字段
    HcclResult ParseOpParamForCache(
        const OpParam& param, HcclDataType& sendType, HcclDataType& recvType, uint64_t& inputSize, uint64_t& outputSize,
        const HcclTopoInfo& topoinfo);

    // 为alltoallv算子准备send/recv information用于cache刷新
    HcclResult PrepareAlltoallvSendRecvInfo(
        const OpParam& param, AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const HcclTopoInfo& topoinfo);

    // aicpu cache
    OpUnfoldCache* opUnfoldCachePtr_ = nullptr;

    // 是否开启对称内存模式
    bool isSymmetricMemory_ = false;

    // 对于第一次可能被cache的alltoallv算子 (不一定是第一个alltoallv, 因为MC2/RDMA/inplace场景下一定不会被cache,
    // 无需计算下列metadata) 记录 是否已计算过alltoallvMetadata_ (sdmaDataBlockSize, hcclInputMemRanges,
    // notifyIdRankRflagMap, signalAddrRankRflagMap) 记录 是否已初始化过alltoallvMetadata_ (hcclOffsetDstRanksIdxMap)
    bool isCalcAlltoallvMetadata_ = false;
    bool isInitAlltoallvMetadata_ = false;

    // alltoallv类型算子的metadata (alltoallv/alltoallvc均视为alltoallv类型算子, 只是对上提供的接口不同,
    // 但实际算法编排都相同) 注意: 给定通信域下可能会有多个alltoallv的cache entry (e.g., opType=alltoallv/alltoallvc,
    // workflowType=单算子/图模式), 但不影响AlltoallvMetadata 注意: 每个通信域下的HcclCommAicpu是互相独立的,
    // 因此只需要在这里维护metadata, 并在alltoallv第一次出现时计算, 后续可直接复用
    AlltoallvMetadata alltoallvMetadata_; // 当前通信域下所有alltoallv cache entry共享一个AlltoallvMetadata
                                          // (只由HCCL_BUFFSIZE和通信域拓扑决定, 与OpUnfoldCacheKey相关字段无关)
};

} // namespace hccl

#endif // __AICPU_CACHE_MANAGER_H__
