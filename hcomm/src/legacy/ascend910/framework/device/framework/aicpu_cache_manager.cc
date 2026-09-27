/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_cache_manager.h"

#include "comm_utils.h"
#include "transport_pub.h"
#include "dispatcher.h"
#include "dispatcher_aicpu_pub.h"
#include "log.h"
#include "task_logic_info_pub.h"
#include "profiling_manager_device.h"
#include "alltoall_utils_pub.h"

namespace hccl {
AicpuCacheManager::AicpuCacheManager() { HCCL_RUN_INFO("Construct AicpuCacheManager complete."); }

AicpuCacheManager::~AicpuCacheManager()
{
    // 释放算子展开的动态缓存 (if any)
    if (opUnfoldCachePtr_ != nullptr) {
        delete opUnfoldCachePtr_;
        opUnfoldCachePtr_ = nullptr;
    }

    HCCL_RUN_INFO("Destruct AicpuCacheManager success!");
}

void AicpuCacheManager::SetSymmetricMemoryEnable(bool enable)
{
    isSymmetricMemory_ = enable;
    HCCL_INFO("[AicpuCacheManager][SetSymmetricMemoryEnable] enable[%u]", enable);
}

HcclResult AicpuCacheManager::InitOpUnfoldCache()
{
    // 创建算子展开的动态缓存 (不区分单算子/图模式)
    HCCL_INFO("[AicpuCacheManager][InitOpUnfoldCache] create aicpu cache for operator unfolding");
    opUnfoldCachePtr_ = (new (std::nothrow) OpUnfoldCache());
    CHK_PTR_NULL(opUnfoldCachePtr_);

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::LookupOpUnfoldCache(
    const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource, bool& needExecute,
    bool& isCacheMiss, Stream& mainStream, std::vector<Stream>& slaveStreams, void* dispatcherPtr,
    const bool isDeviceMode, const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr,
    const AlgOpContext& algContext, std::shared_ptr<AicpuZeroCopyExchanger>& zeroCopyExchangerPtr,
    const HcclWorkflowMode workflowMode, const DeviceMem& tinySendRecvMem,
    std::function<HcclResult()> setProfStartCallback)
{
    needExecute = true;
    isCacheMiss = false;

    CHK_PTR_NULL(opUnfoldCachePtr_);

    // Dump main stream and slave streams addr and id for debug
    HCCL_INFO(
        "[AicpuCacheManager][LookupOpUnfoldCache] mainStream with streamId[%u] and id[%u]",
        mainStream.GetHcclStreamInfo().actualStreamId, mainStream.id());
    for (size_t i = 0; i < slaveStreams.size(); ++i) {
        Stream& slaveStream = slaveStreams[i];
        HCCL_INFO(
            "[AicpuCacheManager][LookupOpUnfoldCache] %uth slaveStream with streamId[%u] and id[%u]", i,
            slaveStream.GetHcclStreamInfo().actualStreamId, slaveStream.id());
    }

    // 判断是否需要cache
    bool needCache = false;
    CHK_RET(NeedOpUnfoldCache(
        algName, param, algResource, isDeviceMode, topoinfo, topoMatcherPtr, algContext, workflowMode, needCache));
    HCCL_INFO("[AicpuCacheManager][LookupOpUnfoldCache] needCache[%u]", needCache);

    // Cacheable算子
    if (needCache) {
        // 将streams中已有的task强制下发, 放置cache缓存跟算子编排无关的SQE
        // 注意: cache miss需要先强制下发, 避免缓存和算子展开无关的SQE; cache hit也需要强制下发,
        // 否则LaunchNewTask只会下发cache里的, 而不会下发stream里的
        CHK_PTR_NULL(dispatcherPtr);
        CHK_RET(LaunchTaskExtend(dispatcherPtr, mainStream, slaveStreams));

        // 准备key
        OpUnfoldKey opUnfoldKey;
        CHK_RET(GetOpUnfoldKey(param, opUnfoldKey, topoinfo, algContext, workflowMode));
        HCCL_INFO(
            "[AicpuCacheManager][LookupOpUnfoldCache] prepare key[%s] for op-unfold cache",
            opUnfoldKey.GetKeyString().c_str());

        // 准备 memory ranges
        std::vector<OpUnfoldMemRange> userInputMemRanges;
        std::vector<OpUnfoldMemRange> userOutputMemRanges;
        CHK_RET(PrepareUserMemRanges(
            param, algResource, userInputMemRanges, userOutputMemRanges, topoinfo, zeroCopyExchangerPtr, workflowMode,
            tinySendRecvMem));

        // 查找算子展开的动态缓存
        HCCL_INFO(
            "[AicpuCacheManager][LookupOpUnfoldCache] look up op-unfold cache for key %s",
            opUnfoldKey.GetKeyString().c_str());
        OpUnfoldCacheEntry* entryPtr = nullptr;
        CHK_RET(opUnfoldCachePtr_->FindEntry(opUnfoldKey, &entryPtr));
        if (entryPtr != nullptr) { // Cache hit
            HCCL_INFO(
                "[AicpuCacheManager][LookupOpUnfoldCache] cache hit for key %s", opUnfoldKey.GetKeyString().c_str());

            // 判断是否为alltoallv算子
            CHK_PTR_NULL(dispatcherPtr);
            CHK_PTR_NULL(setProfStartCallback);
            const bool profL1Enable = dfx::ProfilingManager::GetProfL1State(); // SQE-level profiling info
            if (IsAlltoallvType(param.opType)) { // alltoallv类算子, 需要额外的offset信息
                // 准备offset信息
                AlltoallvSendRecvInfo alltoallvSendRecvInfo;
                CHK_RET(PrepareAlltoallvSendRecvInfo(param, alltoallvSendRecvInfo, topoinfo));

                // 刷新缓存的SQE并直接下发到RTSQ
                // 注意: AicpuCacheManager下dispatcher一定是DispatcherAicpu
                (void)setProfStartCallback(); // Keep consistent with cache miss (调用kfcHandler for kSetProfTimeStart)
                CHK_RET((reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr))
                            ->LaunchNewTask(
                                entryPtr, userInputMemRanges, userOutputMemRanges, mainStream, slaveStreams,
                                profL1Enable, true, alltoallvMetadata_, alltoallvSendRecvInfo));
            } else { // 非V类算子, 无需offset信息
                // 刷新缓存的SQE并直接下发到RTSQ
                // 注意: AicpuCacheManager下dispatcher一定是DispatcherAicpu
                (void)setProfStartCallback(); // Keep consistent with cache miss (调用kfcHandler for kSetProfTimeStart)
                CHK_RET((reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr))
                            ->LaunchNewTask(
                                entryPtr, userInputMemRanges, userOutputMemRanges, mainStream, slaveStreams,
                                profL1Enable, false, AlltoallvMetadata(), AlltoallvSendRecvInfo()));
            }

            // 不需要执行算子展开的具体编排
            needExecute = false;
        } else { // Cache miss
            HCCL_INFO(
                "[AicpuCacheManager][LookupOpUnfoldCache] cache miss for key %s", opUnfoldKey.GetKeyString().c_str());

            // alltoallv类算子需要传入metadata, 用于HCCL input buffer的扫描, 判断SQE addr字段对应的rank
            // id用于后续地址刷新
            bool isAlltoallv = false;
            const AlltoallvMetadata* alltoallvMetadataPtr = nullptr;
            if (IsAlltoallvType(param.opType)) {
                isAlltoallv = true;
                alltoallvMetadataPtr = &alltoallvMetadata_;
                CHK_PTR_NULL(alltoallvMetadataPtr);
            }

            // 设置launch context, enable DispatcherAicpu在下发SQE时去执行cache admission
            // 注意: AicpuCacheManager下dispatcher一定是DispatcherAicpu
            CHK_RET((reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr))
                        ->SetLaunchContext(
                            opUnfoldKey, opUnfoldCachePtr_, userInputMemRanges, userOutputMemRanges, isAlltoallv,
                            alltoallvMetadataPtr));

            isCacheMiss = true;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::PreProcessForCacheMiss(const OpParam& param, std::unique_ptr<CollExecutorBase>& executor)
{
    // 第一个需要cache的alltoallv类算子
    // 注意: 只有当alltoallv的algName为"RunAlltoAllDirectFullmesh"时, 才会进入cache,
    // 所以使用的一定是CollRunAlltoAllDirectFullmesh executor
    if (IsAlltoallvType(param.opType)) {
        CHK_PTR_NULL(executor.get());
        HCCL_INFO(
            "[AicpuCacheManager][PreProcessForCacheMiss] mark NeedAlltoallvCache for CollRunAlltoAllDirectFullmesh");
        CHK_RET(executor->MarkNeedAlltoallvCache());
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::PostProcessForCacheMiss(
    const OpParam& param, std::unique_ptr<CollExecutorBase>& executor, Stream& mainStream,
    std::vector<Stream>& slaveStreams, void* dispatcherPtr, const HcclTopoInfo& topoinfo,
    const AlgOpContext& algContext, const HcclWorkflowMode workflowMode)
{
    // Cache miss会设置launch context to enable cache admission -> 需要清理launch context,
    // DispatcherAicpu不会再admit当前算子后续展开的SQE 注意: AicpuCacheManager下dispatcher一定是DispatcherAicpu
    CHK_RET((reinterpret_cast<DispatcherAiCpu*>(dispatcherPtr))->ClearLaunchContext());

    // 准备key
    OpUnfoldKey opUnfoldKey;
    CHK_RET(GetOpUnfoldKey(param, opUnfoldKey, topoinfo, algContext, workflowMode));

    // 校验cache entry (post cache miss前的orchestrate一定会add new cache entry)
    OpUnfoldCacheEntry* entryPtr = nullptr;
    CHK_PTR_NULL(opUnfoldCachePtr_);
    CHK_RET(opUnfoldCachePtr_->FindEntry(opUnfoldKey, &entryPtr));
    CHK_PTR_NULL(entryPtr); // Cache miss后刚刚admit的cache entry

    // 根据cache中的streamid计算是主流还是第几个从流
    HCCL_INFO(
        "[AicpuCacheManager][PostProcessForCacheMiss] calculate stream seq idxes for a newly-admitted entry of key %s",
        opUnfoldKey.GetKeyString().c_str());
    CHK_RET(entryPtr->CalcStreamSeqIdxes(mainStream, slaveStreams));

    // 针对alltoallv类算子, cache miss后处理
    if (IsAlltoallvType(param.opType)) {
        // 第一个需要cache的alltoallv类算子
        // 注意: 同一个通信域下, alltoallv类算子展开得到的hcclOffsetDstRanksMap是相同的, 所以只需要初始化一次
        if (!isInitAlltoallvMetadata_) {
            // 获得hcclOffset-dstRanks mapping
            // 注意: 只有当alltoallv的algName为"RunAlltoAllDirectFullmesh"时, 才会进入cache,
            // 所以使用的一定是CollRunAlltoAllDirectFullmesh executor
            CHK_PTR_NULL(executor.get());
            HCCL_INFO(
                "[AicpuCacheManager][PostProcessForCacheMiss] get hcclOffset-dstRank mapping of key[%s] for "
                "CollRunAlltoAllDirectFullmesh",
                opUnfoldKey.GetKeyString().c_str());
            std::unordered_map<uint64_t, std::vector<uint32_t>> hcclOffsetDstRanksMap;
            CHK_RET(executor->GetHcclOffsetDstRanksMap(hcclOffsetDstRanksMap));
            for (std::unordered_map<uint64_t, std::vector<uint32_t>>::const_iterator mapIter
                 = hcclOffsetDstRanksMap.cbegin();
                 mapIter != hcclOffsetDstRanksMap.cend(); ++mapIter) {
                alltoallvMetadata_.hcclOffsetDstRanksIdxMap.emplace(mapIter->first, std::make_pair(mapIter->second, 0));
            }

            // Dump hcclOffset-dstRanks mapping
            if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
                for (std::unordered_map<uint64_t, std::vector<uint32_t>>::const_iterator mapIter
                     = hcclOffsetDstRanksMap.cbegin();
                     mapIter != hcclOffsetDstRanksMap.cend(); ++mapIter) {
                    HCCL_INFO(
                        "[AicpuCacheManager][PostProcessForCacheMiss] hcclOffset[%llu]-dstRanks.size[%u]",
                        mapIter->first, mapIter->second.size());
                    for (uint32_t i = 0; i < mapIter->second.size(); ++i) {
                        HCCL_INFO(
                            "[AicpuCacheManager][PostProcessForCacheMiss] dstRanks[%u]: %u", i, mapIter->second[i]);
                    }
                }
            }

            isInitAlltoallvMetadata_ = true;
        }

        // 根据hcclOffset-dstRank mapping更新PrepareIntraData case下dstRefreshInfo中的rank id
        CHK_RET(entryPtr->UpdateRefreshAddrInfoForAlltoallv(topoinfo.userRank, alltoallvMetadata_));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::ClearOpUnfoldCacheEntry(
    const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource, const bool isDeviceMode,
    const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr, const AlgOpContext& algContext,
    const HcclWorkflowMode workflowMode)
{
    // 清理当前aicpu算子对应的cache entry, 避免异常状态下, cache命中
    CHK_PTR_NULL(opUnfoldCachePtr_);

    // 判断是否需要cache
    bool needCache = false;
    CHK_RET(NeedOpUnfoldCache(
        algName, param, algResource, isDeviceMode, topoinfo, topoMatcherPtr, algContext, workflowMode, needCache));
    HCCL_INFO("[AicpuCacheManager][ClearOpUnfoldCacheEntry] needCache[%u]", needCache);

    // Cacheable算子
    if (needCache) {
        // 准备key
        OpUnfoldKey opUnfoldKey;
        CHK_RET(GetOpUnfoldKey(param, opUnfoldKey, topoinfo, algContext, workflowMode));

        // 清理cache entry if any
        HCCL_RUN_INFO(
            "[AicpuCacheManager][ClearOpUnfoldCacheEntry] try to clear cache entry for key[%s]",
            opUnfoldKey.GetKeyString().c_str());
        CHK_RET(opUnfoldCachePtr_->ClearEntry(opUnfoldKey));
    }

    // 针对alltoallv算子的缓存进行metadata清理
    // 注意: 即使当前故障算子不是alltoallv类算子, 由于NS快恢可能会重新分配资源 (例如hccl input / notify/ signal),
    //     为了保证alltoallvMetadata_的正确性, 必须重新计算并初始化alltoallvMetadata_
    CHK_RET(ClearMetadataForFirstAlltoallv());

    // 清理alltoallv类算子的cache entry
    // 注意: 为了保证NS快恢/重执行后, 必定进入alltoallvMetadata_重新计算和初始化的流程,
    // 需要清理与alltoallv类算子相关的entry,
    //     但对aicpu cache影响有限, 因为alltoallv类算子的cache entry数量有限 (只区分opType/isBigCount), 所以性能影响有限
    CHK_RET(opUnfoldCachePtr_->ClearEntryForAlltoallv());

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::ClearMetadataForFirstAlltoallv()
{
    // 确保故障/重执行后第一次可能被cache的alltoallv算子仍然会重新计算/初始化metadata
    isCalcAlltoallvMetadata_ = false;
    isInitAlltoallvMetadata_ = false;
    alltoallvMetadata_.Clear();

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::NeedOpUnfoldCache(
    const std::string& algName, const OpParam& param, const AlgResourceResponse& algResource, const bool isDeviceMode,
    const HcclTopoInfo& topoinfo, std::unique_ptr<TopoMatcher>& topoMatcherPtr, const AlgOpContext& algContext,
    const HcclWorkflowMode workflowMode, bool& needCache)
{
    // 初始化为不需要op-unfold cache
    needCache = false;

    // 检查cache容量
    if (opUnfoldCachePtr_->IsCacheFull()) {
        HCCL_INFO("[AicpuCacheManager][NeedOpUnfoldCache] cache is full, disable cache for current operator");
        return HCCL_SUCCESS;
    }

    // 校验环境变量
    if (param.aicpuCacheEnable == 0) {
        HCCL_INFO(
            "[AicpuCacheManager][NeedOpUnfoldCache] disable aicpu cache for aicpuCacheEnable[%u]",
            param.aicpuCacheEnable);
        return HCCL_SUCCESS;
    }

    // 屏蔽MC2算子
    if (isDeviceMode) {
        HCCL_INFO("[AicpuCacheManager][NeedOpUnfoldCache] MC2 op is not supported for operator unfolding cache");
        return HCCL_SUCCESS;
    }
    HCCL_INFO("[AicpuCacheManager][NeedOpUnfoldCache] device mode is not MC2 op");

    // 判断当前通信域是否使用RDMA (例如跨超通信域), 使用则不cache (因为RoCE队列的WQE不可见)
    const std::unordered_map<u32, bool>& isUsedRdmaMap = topoinfo.isUsedRdmaMap;
    for (std::unordered_map<u32, bool>::const_iterator map_iter = isUsedRdmaMap.cbegin();
         map_iter != isUsedRdmaMap.end(); ++map_iter) {
        if (map_iter->second) {
            HCCL_INFO(
                "[AicpuCacheManager][NeedOpUnfoldCache] rank[%u] uses RDMA -> not supported for operator unfolding "
                "cache",
                map_iter->first);
            return HCCL_SUCCESS;
        }
    }
    HCCL_INFO("[AicpuCacheManager][NeedOpUnfoldCache] all ranks do not use RDMA");

    // 屏蔽inplace场景
    bool isInplace = false;
    CHK_RET(IsInplace(param, isInplace, topoinfo));
    if (isInplace) {
        HCCL_INFO("[AicpuCacheManager][NeedOpUnfoldCache] inplace case is not supported for operator unfolding cache");
        return HCCL_SUCCESS;
    }

    // 目前V类算子、batch类型算子、以及send/recv不考虑动态缓存 (使用白名单而非黑名单管理, 避免非预期算子进入cache机制)
    // 注意: 如果想要通过比较缓存刷新后的SQE与正常算子展开的SQE来debug, 可以将想要比较的算子从以下的cache白名单中移除,
    // 重新打包运行
    const HcclCMDType opType = param.opType;
    if (opType == HcclCMDType::HCCL_CMD_BROADCAST || opType == HcclCMDType::HCCL_CMD_REDUCE
        || opType == HcclCMDType::HCCL_CMD_ALLGATHER || opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLTOALL || opType == HcclCMDType::HCCL_CMD_SCATTER
        || opType == HcclCMDType::HCCL_CMD_ALLREDUCE) { // 非V类算子
        if (algName == "RunAlltoAllVStaged" || algName == "RunAlltoAllVFullMesh") {
            HCCL_INFO(
                "[AicpuCacheManager][%s] algName[%s] is not supported for unfolding cache", __func__, algName.c_str());
            return HCCL_SUCCESS;
        }
        HCCL_INFO(
            "[AicpuCacheManager][NeedOpUnfoldCache] opType[%d] is supported for operator unfolding cache", opType);
        needCache = true;
    } else if (IsAlltoallvType(opType)) { // alltoallv类算子
        // 注意: 暂不支持alltoallv 图模式 / 存在强制单算子模式转换 (即图模式建链+单算子模式展开)
        // 原因: 这两个场景下, alltoallv每次执行会重新建链, 导致remote ranks' hccl buffer在本rank映射的虚拟地址发生变化;
        //     当前如果src是(local) user input, dst是local user output, 会当做LocalCopy根据recv offset进行src addr刷新;
        //     而图模式建链下, remote hccl input会作为remote user input传入cache, remote copy由原来的remote hccl input
        //     -> local user output变成(remote) user input -> local user output, 需要根据hccl offset进行src addr刷新
        // 结论: 由于现网下基本不存在alltoallv图模式调用, 暂对该场景不使能aicpu cache;
        //     如果需要支持, 应当在fullmesh算法中拦截srcRank-hcclOffset的映射, 并通过AlltoallvMetadata传入cache;
        //     识别到remote user input -> local user output时, 判断为图模式下的RemoteCopy;
        //     根据remote user input baseaddr (即remote hccl input在本rank映射的VA) + hccl offset更新src addr
        if (workflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB || // 图模式
            param.aicpuCacheEnable > FORCE_OP_BASE_DELTA) { // 存在强制单算子模式转换 (即图模式建链+单算子模式展开)
            HCCL_INFO(
                "[AicpuCacheManager][NeedOpUnfoldCache] graph mode[%u, %u > %u] is not supported for alltoallv's cache",
                workflowMode, param.aicpuCacheEnable, FORCE_OP_BASE_DELTA);
            return HCCL_SUCCESS;
        }

        // 注意: 假设CollRunAlltoAllDirectFullmesh一定只使用AlltoAllVDirectFullMesh作为algTemp
        if (algName != "RunAlltoAllDirectFullmesh") {
            HCCL_INFO(
                "[AicpuCacheManager][NeedOpUnfoldCache] algName[%s] is not supported for alltoallv's cache",
                algName.c_str());
            return HCCL_SUCCESS;
        }

        if (!isCalcAlltoallvMetadata_) { // 当前通信域下第一次可能被cache的alltoallv, 需要计算相应metadata
            CHK_RET(CalcMetadataForFirstAlltoallv(algResource, isDeviceMode, topoinfo, topoMatcherPtr, algContext));

            // 后续不再重复计算alltoallv metadata
            // 注意: (i) 虽然alltoallvMetadata_中的相关mapping还未被初始化,
            // 但isCalcAlltoallvMetadata_只是为了避免重复计算部分metadata (ii) 参考CalcMetadataForFirstAlltoallv,
            // 例如sdmaDataBlockSize, hcclInputMemRanges, notifyIdRankRflagMap等 如果有故障发生: (i)
            // 发生在isCalcAlltoallvMetadata_ = true前,
            // 则ClearOpUnfoldCacheEntry会重新计算sdmaDataBlockSize来判断是否需要清理cache entry (ii) 发生在设置true后,
            // 即使在PostProcessForCacheMiss初始化相关mapping前, 也不影响清理时needCache的判断 (只依赖sdmaDataBlockSize)
            isCalcAlltoallvMetadata_ = true;
        }

        // alltoallv/alltoallvc 没有对端数据时不走cache
        bool hasRemoteData = true;
        CHK_RET(HasRemoteDataAlltoallv(param, hasRemoteData, topoinfo));
        if (topoinfo.userRankSize > 1 && !hasRemoteData) {
            HCCL_INFO(
                "[AicpuCacheManager][NeedOpUnfoldCache] no remote data alltoallv[%u] is not supported for operator "
                "unfolding cache",
                opType);
            return HCCL_SUCCESS;
        }

        // 判断是否为小数据量的alltoallv类算子
        bool isSmallData = false;
        CHK_RET(IsSmallDataAlltoallv(param, isSmallData, topoinfo));
        if (!isSmallData) {
            HCCL_INFO(
                "[AicpuCacheManager][NeedOpUnfoldCache] large-data alltoallv[%u] is not supported for operator "
                "unfolding cache",
                opType);
            return HCCL_SUCCESS;
        }

        HCCL_INFO(
            "[AicpuCacheManager][NeedOpUnfoldCache] small-data alltoallv[%u] is supported for operator unfolding cache",
            opType);
        needCache = true;
    } else {
        HCCL_INFO(
            "[AicpuCacheManager][NeedOpUnfoldCache] opType[%d] is not supported for operator unfolding cache", opType);
        return HCCL_SUCCESS;
    }

    // 到这里needCache应该为true (如果为false则已经提前返回了)
    CHK_PRT_RET(
        !needCache, HCCL_ERROR("[AicpuCacheManager][NeedOpUnfoldCache] needCache should be true"), HCCL_E_INTERNAL);

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::IsInplace(const OpParam& param, bool& isInplace, const HcclTopoInfo& topoinfo)
{
    // 准备input/output size
    HcclDataType sendType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclDataType recvType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CHK_RET(ParseOpParamForCache(param, sendType, recvType, inputSize, outputSize, topoinfo));
    UNUSED_PARAM(sendType);
    UNUSED_PARAM(recvType);

    // 注意: alltoall/alltoallv/alltoallvc可能存在inputSize/outputSize为0的情况, 导致不分配user input/output
    // 但会使用tinySendRecvMem_更新algResource.paramInput/OutputMem用于建链, 导致cache无法区分给定地址字段的地址类型
    // 参考aicpu_communicator.cc中的SetAlltoAllInputAndOutPutMem
    if (inputSize == 0 && outputSize == 0) {
        isInplace = true;
        HCCL_INFO(
            "[AicpuCacheManager][IsInplace] inputSize[%u] is overlapping with outputSize[%u]", inputSize, outputSize);
        return HCCL_SUCCESS;
    }

    if (inputSize == 0 || outputSize == 0) {
        isInplace = false;
        HCCL_INFO(
            "[AicpuCacheManager][IsInplace] inputSize[%u] is not overlapping with outputSize[%u]", inputSize,
            outputSize);
        return HCCL_SUCCESS;
    }

    const uint64_t inputStart = reinterpret_cast<uint64_t>(param.inputPtr);
    const uint64_t inputEnd = inputStart + inputSize - 1;
    const uint64_t outputStart = reinterpret_cast<uint64_t>(param.outputPtr);
    const uint64_t outputEnd = outputStart + outputSize - 1;

    if (inputStart <= outputEnd && outputStart <= inputEnd) {
        isInplace = true;
        HCCL_INFO(
            "[AicpuCacheManager][IsInplace] input[0x%016llx, 0x%016llx] is overlapping with output[0x%016llx, "
            "0x%016llx]",
            inputStart, inputEnd, outputStart, outputEnd);
    } else {
        isInplace = false;
        HCCL_INFO(
            "[AicpuCacheManager][IsInplace] input[0x%016llx, 0x%016llx] is not overlapping with output[0x%016llx, "
            "0x%016llx]",
            inputStart, inputEnd, outputStart, outputEnd);
    }

    return HCCL_SUCCESS;
}

bool AicpuCacheManager::IsAlltoallvType(const HcclCMDType opType)
{
    // alltoallv/alltoallvc只是对上接口不同, 实际算法编排相同, 均视为alltoallv类型的算子
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV || opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
        return true;
    }
    return false;
}

HcclResult
AicpuCacheManager::IsSmallDataAlltoallv(const OpParam& param, bool& isSmallData, const HcclTopoInfo& topoinfo)
{
    // 根据SDMA data block size判断是否需要cache
    // 参考coll_all_to_all_v_direct_fullmesh_executor.cc下的CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallV
    const uint32_t rankSize = topoinfo.userRankSize;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) { // alltoallv
        HCCL_INFO("[AicpuCacheManager][IsSmallDataAlltoallv] check %u send/recv counts", rankSize);
        for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            const uint64_t curSendCounts = *(static_cast<const uint64_t*>(param.All2AllDataDes.sendCounts) + tmpRank);
            const uint64_t curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
            // 如果curSendLength超过SDMA data block size (即alltoallv需要切step), 或者超过HCCL_SDMA_MAX_COUNT_4GB
            // (即MemcpyAsync需要切split), 不做cache
            if (curSendLength > alltoallvMetadata_.sdmaDataBlockSize || curSendLength > HCCL_SDMA_MAX_COUNT_4GB) {
                HCCL_INFO(
                    "[AicpuCacheManager][IsSmallDataAlltoallv] large-sdata alltoallv[%u]: userRank[%u] tmpRank[%u]"
                    "curSendLength[%u] sdmaDataBlockSize[%u] 4GB[%u]",
                    param.opType, topoinfo.userRank, tmpRank, curSendLength, alltoallvMetadata_.sdmaDataBlockSize,
                    HCCL_SDMA_MAX_COUNT_4GB);
                isSmallData = false;
                return HCCL_SUCCESS;
            }

            const uint64_t curRecvCounts = *(static_cast<const uint64_t*>(param.All2AllDataDes.recvCounts) + tmpRank);
            const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
            // 如果curRecvLength超过SDMA data block size (即alltoallv需要切step), 或者超过HCCL_SDMA_MAX_COUNT_4GB
            // (即MemcpyAsync需要切split), 不做cache
            if (curRecvLength > alltoallvMetadata_.sdmaDataBlockSize || curRecvLength > HCCL_SDMA_MAX_COUNT_4GB) {
                HCCL_INFO(
                    "[AicpuCacheManager][IsSmallDataAlltoallv] large-rdata alltoallv[%u]: userRank[%u] tmpRank[%u]"
                    "curRecvLength[%u] sdmaDataBlockSize[%u] 4GB[%u]",
                    param.opType, topoinfo.userRank, tmpRank, curRecvLength, alltoallvMetadata_.sdmaDataBlockSize,
                    HCCL_SDMA_MAX_COUNT_4GB);
                isSmallData = false;
                return HCCL_SUCCESS;
            }
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) { // alltoallvc
        const uint32_t curRank = topoinfo.userRank;
        HCCL_INFO("[AicpuCacheManager][IsSmallDataAlltoallv] check %u-size sendCountMatrix", rankSize);
        for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            const uint64_t curSendCounts
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + curRank * rankSize
                    + tmpRank); // sendCountMatrix[curRank][tmpRank]
            const uint64_t curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
            // 如果curSendLength超过SDMA data block size (即alltoallv需要切step), 或者超过HCCL_SDMA_MAX_COUNT_4GB
            // (即MemcpyAsync需要切split), 不做cache
            if (curSendLength > alltoallvMetadata_.sdmaDataBlockSize || curSendLength > HCCL_SDMA_MAX_COUNT_4GB) {
                HCCL_INFO(
                    "[AicpuCacheManager][IsSmallDataAlltoallv] large-sdata alltoallvc[%u]: userRank[%u] tmpRank[%u]"
                    "curSendLength[%u] sdmaDataBlockSize[%u] 4GB[%u]",
                    param.opType, curRank, tmpRank, curSendLength, alltoallvMetadata_.sdmaDataBlockSize,
                    HCCL_SDMA_MAX_COUNT_4GB);
                isSmallData = false;
                return HCCL_SUCCESS;
            }

            const uint64_t curRecvCounts
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + tmpRank * topoinfo.userRankSize
                    + curRank); // sendCountMatrix[tmpRank][curRank]
            const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
            // 如果curRecvLength超过SDMA data block size (即alltoallv需要切step), 或者超过HCCL_SDMA_MAX_COUNT_4GB
            // (即MemcpyAsync需要切split), 不做cache
            if (curRecvLength > alltoallvMetadata_.sdmaDataBlockSize || curRecvLength > HCCL_SDMA_MAX_COUNT_4GB) {
                HCCL_INFO(
                    "[AicpuCacheManager][IsSmallDataAlltoallv] large-rdata alltoallvc[%u]: userRank[%u] tmpRank[%u]"
                    "curRecvLength[%u] sdmaDataBlockSize[%u] 4GB[%u]",
                    param.opType, topoinfo.userRank, tmpRank, curRecvLength, alltoallvMetadata_.sdmaDataBlockSize,
                    HCCL_SDMA_MAX_COUNT_4GB);
                isSmallData = false;
                return HCCL_SUCCESS;
            }
        }
    } else {
        HCCL_ERROR("[AicpuCacheManager][IsSmallDataAlltoallv] invalid opType[%u] for alltoallv", param.opType);
        return HCCL_E_INTERNAL;
    }

    isSmallData = true;
    return HCCL_SUCCESS;
}

HcclResult
AicpuCacheManager::HasRemoteDataAlltoallv(const OpParam& param, bool& hasRemoteData, const HcclTopoInfo& topoinfo)
{
    const uint32_t rankSize = topoinfo.userRankSize;
    const uint32_t curRank = topoinfo.userRank;
    hasRemoteData = false;
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) { // alltoallv
        for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            if (tmpRank == curRank) {
                continue;
            }

            // curRank发送到tmpRank的数据量
            const uint64_t curSendCounts = *(static_cast<const uint64_t*>(param.All2AllDataDes.sendCounts) + tmpRank);
            const uint64_t curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
            if (curSendLength > 0) {
                hasRemoteData = true;
                return HCCL_SUCCESS;
            }

            // curRank从tmpRank接收的数据量
            const uint64_t curRecvCounts = *(static_cast<const uint64_t*>(param.All2AllDataDes.recvCounts) + tmpRank);
            const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
            if (curRecvLength > 0) {
                hasRemoteData = true;
                return HCCL_SUCCESS;
            }
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) { // alltoallvc
        for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            if (tmpRank == curRank) {
                continue;
            }

            // curRank发送到tmpRank的数据量
            const uint64_t curSendCounts
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + curRank * rankSize
                    + tmpRank); // sendCountMatrix[curRank][tmpRank]
            const uint64_t curSendLength = curSendCounts * SIZE_TABLE[param.All2AllDataDes.sendType];
            if (curSendLength > 0) {
                hasRemoteData = true;
                return HCCL_SUCCESS;
            }

            // curRank从tmpRank接收的数据量
            const uint64_t curRecvCounts
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + tmpRank * topoinfo.userRankSize
                    + curRank); // sendCountMatrix[tmpRank][curRank]
            const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[param.All2AllDataDes.recvType];
            if (curRecvLength > 0) {
                hasRemoteData = true;
                return HCCL_SUCCESS;
            }
        }
    } else {
        HCCL_ERROR("[AicpuCacheManager][HasRemoteDataAlltoallv] invalid opType[%u] for alltoallv", param.opType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::CalcMetadataForFirstAlltoallv(
    const AlgResourceResponse& algResource, const bool isDeviceMode, const HcclTopoInfo& topoinfo,
    std::unique_ptr<TopoMatcher>& topoMatcherPtr, const AlgOpContext& algContext)
{
    alltoallvMetadata_.Clear();
    HCCL_INFO("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] clear alltoallv metadata before calc.");

    // Part 1: 计算SDMA data block size, 用于根据数据量判断是否需要cache
    // 参考alltoallv_direct_fullmesh.cc下的AlltoAllVDirectFullMesh::Prepare (当前alltoallv算子只会使用direct full
    // mesh算法)

    // 获取local pod中的device数量
    // 参考coll_all_to_all_v_direct_fullmesh_executor.cc下的CollRunAlltoAllDirectFullmesh::GetLocalSDMAGroupInfo
    uint32_t devNumInlocalPod = 0;
    uint32_t rankIdxInPod = 0;
    const bool isA2MultiModule = topoinfo.deviceType == DevType::DEV_TYPE_910B && !topoinfo.isSingleMeshAggregation;
    if (topoMatcherPtr->GetExternalInputInterHccsDisable() || isA2MultiModule) {
        CHK_RET(topoMatcherPtr->GetLocalServerRankSize(topoinfo.userRank, devNumInlocalPod, rankIdxInPod));
    } else {
        CHK_RET(topoMatcherPtr->GetLocalSuperPodRankSize(topoinfo.userRank, devNumInlocalPod, rankIdxInPod));
    }
    CHK_PRT_RET(
        devNumInlocalPod == INVALID_VALUE_RANKSIZE,
        HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] get local superPod total ranksize failed."),
        HCCL_E_PARA);
    UNUSED_PARAM(rankIdxInPod);

    // 计算SDMA在alltoallv下的最大并发数量
    const uint32_t sdmaConcurrentNum = (devNumInlocalPod > ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) ?
                                           (ALLTOALLV_DIRECT_FULLMESH_SDMA_CONCURRENT_SIZE) :
                                           (devNumInlocalPod);

    // 注意: MC2算子不会进入cache, 不需要根据MC2 stepSize调整sdmaConcurrentNum
    CHK_PRT_RET(
        algContext.mc2Handler.stepSize > 0,
        HCCL_ERROR(
            "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] isDeviceMode[%u] mc2Handler.stepSize[%u]", isDeviceMode,
            algContext.mc2Handler.stepSize),
        HCCL_E_INTERNAL);

    // 计算SDMA data block大小
    constexpr uint32_t blockGroup = 2;
    alltoallvMetadata_.sdmaDataBlockSize
        = (algResource.cclInputMem.size() / std::max(1u, sdmaConcurrentNum * blockGroup));

    // 向下对齐到16k Byte
    if (alltoallvMetadata_.sdmaDataBlockSize > HCCL_MIN_SLICE_ALIGN_910B) {
        alltoallvMetadata_.sdmaDataBlockSize
            = (alltoallvMetadata_.sdmaDataBlockSize / HCCL_MIN_SLICE_ALIGN_910B) * HCCL_MIN_SLICE_ALIGN_910B;
    }

    // sdmaDataBlockSize应该大于0
    CHK_PRT_RET(
        alltoallvMetadata_.sdmaDataBlockSize == 0,
        HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] sdmaDataBlockSize is zero."), HCCL_E_INTERNAL);

    HCCL_INFO(
        "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] first alltoallv, devNumInlocalPod[%u],"
        "sdmaConcurrentNum[%u] cclInputSize[%u] sdmaDataBlockSize[%u]",
        devNumInlocalPod, sdmaConcurrentNum, algResource.cclInputMem.size(), alltoallvMetadata_.sdmaDataBlockSize);

    // Part 2: 计算每个rank的HCCL input buffer memory range
    // 参考alltoallv_direct_fullmesh.cc下的SDMAwithRemoteRankAndNotifyEnd,
    // coll_all_to_all_v_direct_fullmesh_executor.cc下的KernelRun,
    //     和coll_native_executor_base.cc下的GetSubCommInfo

    // 初始化hcclInputMemRanges
    const uint32_t rankSize = topoinfo.userRankSize;
    HCCL_INFO(
        "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] prepare %u hccl input memory ranges for op-unfold cache",
        rankSize);
    std::vector<OpUnfoldMemRange>& hcclInputMemRanges = alltoallvMetadata_.hcclInputMemRanges;
    hcclInputMemRanges.resize(rankSize);

    // 准备hccl input size
    // 来自于cclInputBuffer_ (size来自于HcclOpResParam commParam.winSize, 单位是bytes)
    const uint64_t hcclInputSize = algResource.cclInputMem.size();

    // 设置当前rank的hccl input memory range
    const uint32_t curRank = topoinfo.userRank; // NOTE: 不应该使用param.srcRank (某些算子始终为0)
    CHK_PRT_RET(
        curRank >= rankSize,
        HCCL_ERROR(
            "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] invalid curRank %u >= rankSize %u", curRank, rankSize),
        HCCL_E_INTERNAL);
    OpUnfoldMemRange& curHcclInputMemRange = hcclInputMemRanges[curRank];
    curHcclInputMemRange.isValid = true;
    curHcclInputMemRange.baseAddr = reinterpret_cast<uint64_t>(algResource.cclInputMem.ptr());
    curHcclInputMemRange.memSize
        = hcclInputSize; // NOTE: 不应该使用param.inputSize (user memory input size, 且alltoall类始终为0)

    // 设置其他rank的hccl input memory range
    const std::vector<LINK>& links = algResource.opTransportResponse[COMM_COMBINE_ORDER][COMM_INDEX_0].links;
    CHK_PRT_RET(
        links.size() != rankSize,
        HCCL_ERROR(
            "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] links.size[%u] != rankSize[%u]", links.size(),
            rankSize),
        HCCL_E_INTERNAL);
    for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
        if (tmpRank == curRank) {
            continue;
        }

        // 获取curRank与tmpRank之间的LINK
        const LINK& intraNeighboorTransport = links[tmpRank];
        CHK_PTR_NULL(intraNeighboorTransport);

        // 获取tmpRank的hccl input memory baseaddr
        void* tmpHcclInputBaseAddr = nullptr;
        CHK_RET(intraNeighboorTransport->GetRemoteMem(UserMemType::INPUT_MEM, &tmpHcclInputBaseAddr));
        CHK_PTR_NULL(tmpHcclInputBaseAddr);

        // 设置tmpRank对应的hccl input memory range
        OpUnfoldMemRange& tmpHcclInputMemRange = hcclInputMemRanges[tmpRank];
        tmpHcclInputMemRange.isValid = true;
        tmpHcclInputMemRange.baseAddr = reinterpret_cast<uint64_t>(tmpHcclInputBaseAddr);
        tmpHcclInputMemRange.memSize = hcclInputSize;
    }

    // 打印debug信息
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        for (size_t rankId = 0; rankId < hcclInputMemRanges.size(); ++rankId) {
            HCCL_INFO(
                "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] hcclInputMemRanges[%u] isValid: %d,"
                "baseAddr: 0x%016llx, memSize: %llu",
                rankId, hcclInputMemRanges[rankId].isValid, hcclInputMemRanges[rankId].baseAddr,
                hcclInputMemRanges[rankId].memSize);
        }
    }

    // Part 3: 计算notifyId与remoteRank间的映射

    HCCL_INFO(
        "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] prepare %u notify info for op-unfold cache", rankSize - 1);
    for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
        if (tmpRank == curRank) {
            continue;
        }

        // 获取curRank与tmpRank之间的LINK
        const LINK& intraNeighboorTransport = links[tmpRank];
        CHK_PTR_NULL(intraNeighboorTransport);

        // 获取RxAck相关的NotifyId (与recv count相关)
        HcclSignalInfo recvNotifyInfo;
        bool recvIsValid = false;
        CHK_RET(intraNeighboorTransport->GetSpecificNotify(recvNotifyInfo, recvIsValid, "localSendDone"));
        CHK_PRT_RET(
            !recvIsValid, HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] invalid localSendDoneNotify_"),
            HCCL_E_INTERNAL);
        const uint32_t recvNotifyId = static_cast<uint32_t>(recvNotifyInfo.resId);
        alltoallvMetadata_.notifyIdRankRflagMap.emplace(recvNotifyId, std::make_pair(tmpRank, true));

        // 获取RxDataSignal相关的NotifyId (与send count相关)
        HcclSignalInfo sendNotifyInfo;
        bool sendIsValid = false;
        CHK_RET(intraNeighboorTransport->GetSpecificNotify(sendNotifyInfo, sendIsValid, "localSendReady"));
        CHK_PRT_RET(
            !sendIsValid,
            HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] invalid localSendReadyNotify_"),
            HCCL_E_INTERNAL);
        const uint32_t sendNotifyId = static_cast<uint32_t>(sendNotifyInfo.resId);
        alltoallvMetadata_.notifyIdRankRflagMap.emplace(sendNotifyId, std::make_pair(tmpRank, false));
    }

    // Part 4: 计算signalAddr与remoteRank的映射

    HCCL_INFO(
        "[AicpuCacheManager][CalcMetadataForFirstAlltoallv] prepare %u signal info for op-unfold cache", rankSize - 1);
    for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
        if (tmpRank == curRank) {
            continue;
        }

        // 获取curRank与tmpRank之间的LINK
        const LINK& intraNeighboorTransport = links[tmpRank];
        CHK_PTR_NULL(intraNeighboorTransport);

        // 获取TxDataSignal相关的SignalAddr (与recv count相关)
        HcclSignalInfo recvNotifyInfo;
        bool recvIsValid = false;
        CHK_RET(intraNeighboorTransport->GetSpecificNotify(recvNotifyInfo, recvIsValid, "remoteSendReady"));
        CHK_PRT_RET(
            !recvIsValid,
            HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] invalid remoteSendReadyNotify_"),
            HCCL_E_INTERNAL);
        const uint64_t recvSignalAddr = recvNotifyInfo.addr;
        alltoallvMetadata_.signalAddrRankRflagMap.emplace(recvSignalAddr, std::make_pair(tmpRank, true));

        // 获取TxAck相关的SignalAddr (与send count相关)
        HcclSignalInfo sendNotifyInfo;
        bool sendIsValid = false;
        CHK_RET(intraNeighboorTransport->GetSpecificNotify(sendNotifyInfo, sendIsValid, "remoteSendDone"));
        CHK_PRT_RET(
            !sendIsValid,
            HCCL_ERROR("[AicpuCacheManager][CalcMetadataForFirstAlltoallv] invalid remoteSendDoneNotify_"),
            HCCL_E_INTERNAL);
        const uint64_t sendSignalAddr = sendNotifyInfo.addr;
        alltoallvMetadata_.signalAddrRankRflagMap.emplace(sendSignalAddr, std::make_pair(tmpRank, false));
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::GetOpUnfoldKey(
    const OpParam& param, OpUnfoldKey& key, const HcclTopoInfo& topoinfo, const AlgOpContext& algContext,
    const HcclWorkflowMode workflowMode)
{
    // 注意: 由于GetOpUnfoldKey前已经做过NeedOpUnfoldCache检查, 这里不再做重复检验

    // 准备sendType和inputSize
    // 注意: 如果是alltoallv类算子, sendType设置为RESERVED, inputSize设置为0, 保证即使dataType, sendCounts,
    // recvCounts发生变化, 仍然能够缓存命中
    HcclDataType sendType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t inputSize = 0;
    if (!IsAlltoallvType(param.opType)) { // 非alltoallv类算子, 需要根据sendType和inputSize生成不同的key
        HcclDataType recvType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
        uint64_t outputSize = 0;
        CHK_RET(ParseOpParamForCache(param, sendType, recvType, inputSize, outputSize, topoinfo));
        UNUSED_PARAM(recvType);
        UNUSED_PARAM(outputSize);
    } else { // alltoallv类算子, 需要根据isBigCount生成不同的key, 决定SQE编排是否需要并发
        bool isBigCountForAlltoallv = false;
        CHK_RET(IsBigCountForAlltoallv(param, topoinfo, isBigCountForAlltoallv));
        if (isBigCountForAlltoallv) {
            inputSize = 1;
        } else {
            inputSize = 0;
        }
    }

    // 准备root: 仅 scatter / broadcast / reduce 三类算子的SQE模板会随root变化, 必须纳入key
    // 注意: 其他算子 (allgather/allreduce/alltoall/...) 在不同root下SQE模板相同, 固定为0即可
    uint32_t root = 0;
    if (param.opType == HcclCMDType::HCCL_CMD_SCATTER || param.opType == HcclCMDType::HCCL_CMD_BROADCAST
        || param.opType == HcclCMDType::HCCL_CMD_REDUCE) {
        root = param.root;
    }

    // 设置key for op-unfold cache
    CHK_RET(key.Init(
        param.opType, sendType, param.reduceType, param.isZeroCopy, isSymmetricMemory_, inputSize,
        algContext.opRetryHandler.isInplacePreSync, workflowMode, param.isCapture, root));

    return HCCL_SUCCESS;
}

HcclResult
AicpuCacheManager::IsBigCountForAlltoallv(const OpParam& param, const HcclTopoInfo& topoinfo, bool& isBigCount)
{
    isBigCount = false;
    if (IsAlltoallvType(param.opType)) { // alltoallv类算子
        // 计算maxSendCount
        // 参考coll_all_to_all_v_direct_fullmesh_executor.cc中的GetLocalSendRecvInfoforAlltoallV
        const uint32_t curRank = topoinfo.userRank;
        const uint32_t rankSize = topoinfo.userRankSize;
        uint64_t maxSendCount = 0;
        for (size_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            // 获得curRank -> dstRank的sendCount
            uint64_t curSendCount = 0;
            if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) { // alltoallv
                curSendCount = *(static_cast<const uint64_t*>(param.All2AllDataDes.sendCounts) + dstRank);
            } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) { // alltoallvc
                curSendCount
                    = *(static_cast<const uint64_t*>(param.All2AllDataDes.sendCountMatrix) + curRank * rankSize
                        + dstRank); // sendCountMatrix[curRank][dstRank]
            } else {
                HCCL_ERROR(
                    "[AicpuCacheManager][IsBigCountForAlltoallv] invalid opType[%u] for alltoallv", param.opType);
                return HCCL_E_INTERNAL;
            }

            // 更新maxSendCount
            if (curSendCount > maxSendCount) {
                maxSendCount = curSendCount;
            }
        }

        // 参考alltoallv_direct_fullmesh.cc中的Prepare()
        uint64_t maxSendLen = maxSendCount * SIZE_TABLE[param.All2AllDataDes.sendType];
        isBigCount = (maxSendLen > ALLTOALLV_DIRECT_FULLMESH_BIG_SIZE) ? true : false;

        HCCL_INFO(
            "[AicpuCacheManager][IsBigCountForAlltoallv] maxSendCount[%llu] maxSendLen[%llu] isBigCount[%u]",
            maxSendCount, maxSendLen, isBigCount);
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::PrepareUserMemRanges(
    const OpParam& param, const AlgResourceResponse& algResource, std::vector<OpUnfoldMemRange>& userInputMemRanges,
    std::vector<OpUnfoldMemRange>& userOutputMemRanges, const HcclTopoInfo& topoinfo,
    std::shared_ptr<AicpuZeroCopyExchanger>& zeroCopyExchangerPtr, const HcclWorkflowMode workflowMode,
    const DeviceMem& tinySendRecvMem)
{
    // 注意: 由于PrepareUserMemRanges前已经做过NeedOpUnfoldCache检查, 这里不再做重复检验

    const uint32_t rankSize = topoinfo.userRankSize;
    HCCL_INFO(
        "[AicpuCacheManager][PrepareUserMemRanges] prepare %u user input/output memory ranges for op-unfold cache",
        rankSize);

    // 准备memory ranges
    userInputMemRanges.resize(rankSize);
    userOutputMemRanges.resize(rankSize);

    // 准备input/output size
    HcclDataType sendType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    HcclDataType recvType = HcclDataType::HCCL_DATA_TYPE_RESERVED;
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    CHK_RET(ParseOpParamForCache(param, sendType, recvType, inputSize, outputSize, topoinfo));

    // 校验input/output size (应该在NeedOpUnfoldCache中被IsInplace拦截)
    CHK_PRT_RET(
        inputSize == 0 && outputSize == 0,
        HCCL_ERROR("[AicpuCacheManager][PrepareUserMemRanges] inputSize[%llu] outputSize[%llu]", inputSize, outputSize),
        HCCL_E_INTERNAL);

    // 当前只有alltoall/alltoallv/alltoallvc才会存在inputSize/outputSize为0的情况
    // 注意: 对于alltoall算子, outputSize一定等于inputSize, 所以不会存在inputSize/outputSize之一为0的情况
    // -> 这里其实只考虑alltoallv/alltoallvc
    // 参考aicpu_communicator.cc中的SetAlltoAllInputAndOutPutMem
    const HcclCMDType opType = param.opType;
    CHK_PRT_RET(
        (inputSize == 0 || outputSize == 0) && opType != HCCL_CMD_ALLTOALL && opType != HCCL_CMD_ALLTOALLV
            && opType != HCCL_CMD_ALLTOALLVC,
        HCCL_ERROR(
            "[AicpuCacheManager][PrepareUserMemRanges] opType[%u] inputSize[%llu] outputSize[%llu]", opType, inputSize,
            outputSize),
        HCCL_E_INTERNAL);

    // 设置当前rank的input/output usermem addr
    const uint32_t curRank = topoinfo.userRank; // NOTE: 不应该使用param.srcRank (某些算子始终为0)
    CHK_PRT_RET(
        curRank >= rankSize,
        HCCL_ERROR("[AicpuCacheManager][PrepareUserMemRanges] invalid curRank %u >= rankSize %u", curRank, rankSize),
        HCCL_E_INTERNAL);
    HCCL_INFO("[AicpuCacheManager][PrepareUserMemRanges] prepare user memory range of current rank %u", curRank);
    OpUnfoldMemRange& curUserInputMemRange = userInputMemRanges[curRank];
    curUserInputMemRange.isValid = true;
    if (inputSize == 0) { // 处理alltoallv/alltoallvc的corner case
        HCCL_INFO(
            "[AicpuCacheManager][PrepareUserMemRanges] use tinySendRecvMem[0x%016llx, %llu]"
            "as local user input for opType[%u]",
            tinySendRecvMem.ptr(), tinySendRecvMem.size(), opType);
        curUserInputMemRange.baseAddr = reinterpret_cast<uint64_t>(tinySendRecvMem.ptr());
        curUserInputMemRange.memSize = tinySendRecvMem.size();
    } else {
        curUserInputMemRange.baseAddr = reinterpret_cast<uint64_t>(param.inputPtr);
        curUserInputMemRange.memSize = inputSize; // NOTE: 不应该使用param.inputSize (alltoall类始终为0)
    }
    OpUnfoldMemRange& curUserOutputMemRange = userOutputMemRanges[curRank];
    curUserOutputMemRange.isValid = true;
    if (outputSize == 0) { // 处理alltoallv/alltoallvc的corner case
        HCCL_INFO(
            "[AicpuCacheManager][PrepareUserMemRanges] use tinySendRecvMem[0x%016llx, %llu]"
            "as local user output for opType[%u]",
            tinySendRecvMem.ptr(), tinySendRecvMem.size(), opType);
        curUserOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(tinySendRecvMem.ptr());
        curUserOutputMemRange.memSize = tinySendRecvMem.size();
    } else {
        curUserOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(param.outputPtr);
        curUserOutputMemRange.memSize = outputSize; // NOTE: 不应该使用param.outputSize (alltoall类始终为0)
    }

    // 针对zero copy, 设置remote rank的input/output usermem addr
    if (param.isZeroCopy) {
        // 注意: 只有非V类算子可能使用zero copy (因此假设remote ranks' input/output size与local rank相同)
        // 注意: 而V类算子一定是buffer copy, 只会存在local user/hccl <-> remote hccl之间的搬运
        // (否则PrepareRemoteUserMemRanges需要额外的输入作为remote ranks' input/output size)
        CHK_PRT_RET(
            opType == HCCL_CMD_ALLTOALLV || opType == HCCL_CMD_ALLTOALLVC || opType == HCCL_CMD_ALLGATHER_V
                || opType == HCCL_CMD_REDUCE_SCATTER_V || opType == HCCL_CMD_HALF_ALLTOALLV,
            HCCL_ERROR("[AicpuCacheManager][PrepareUserMemRanges] opType[%u] should not use zero copy", opType),
            HCCL_E_INTERNAL);

        // 直接传入local rank's input/output size用于remote ranks' memory ranges
        HCCL_INFO("[AicpuCacheManager][PrepareUserMemRanges] prepare user memory ranges of other remote ranks");
        CHK_PTR_NULL(zeroCopyExchangerPtr.get());
        CHK_RET(zeroCopyExchangerPtr->PrepareRemoteUserMemRanges(
            inputSize, outputSize, userInputMemRanges, userOutputMemRanges));
    } else if (
        isSymmetricMemory_
        || // 对称内存场景：remote ranks 的 user mem 需从 opTransportResponse 的 links 中获取
           // PrepareSymmetricMemory 已在 ExecOp 中通过 HcclSymWinGetPeerPointer 刷新各 link 的 remote addr
        workflowMode == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB || // 图模式
        param.aicpuCacheEnable > FORCE_OP_BASE_DELTA) { // 存在强制单算子模式转换 (即图模式建链+单算子模式展开)
        HCCL_INFO("[AicpuCacheManager][PrepareUserMemRanges] check transport resource for potential user memory of "
                  "remote ranks");

        // 遍历所有transport信息, 更新remote ranks' user input/output memory ranges
        for (size_t planeIdx = 0; planeIdx < algResource.opTransportResponse.size(); ++planeIdx) {
            const LevelNSubCommTransport& subCommTransport = algResource.opTransportResponse[planeIdx];
            for (size_t commIdx = 0; commIdx < subCommTransport.size(); ++commIdx) {
                const SingleSubCommTransport& commTransport = subCommTransport[commIdx];

                // 注意: 假设SingleSubCommTransport中的transportRequests和links是一一对应的
                const std::vector<TransportRequest>& transportRequests = commTransport.transportRequests;
                const std::vector<LINK>& links = commTransport.links;
                HCCL_INFO(
                    "[AicpuCacheManager][PrepareUserMemRanges] planeIdx[%u] commIdx[%u] links.size[%u]", planeIdx,
                    commIdx, links.size());
                CHK_PRT_RET(
                    transportRequests.size() != links.size(),
                    HCCL_ERROR(
                        "[AicpuCacheManager][PrepareUserMemRanges] transportRequests.size[%u] != links.size[%u]",
                        transportRequests.size(), links.size()),
                    HCCL_E_INTERNAL);

                // 遍历每个remote rank对应的link信息
                for (size_t reqIdx = 0; reqIdx < transportRequests.size(); ++reqIdx) {
                    const TransportRequest& curReq = transportRequests[reqIdx];
                    if (curReq.isValid) {
                        if (curReq.remoteUserRank == curRank) { // 本rank无需从link获取user memory range
                            continue;
                        } else if (curReq.remoteUserRank == INVALID_VALUE_RANKID) { // 本rank无需从link获取user memory
                                                                                    // range
                            continue;
                        }

                        CHK_PRT_RET(
                            curReq.remoteUserRank >= rankSize,
                            HCCL_ERROR(
                                "[AicpuCacheManager][PrepareUserMemRanges] invalid remoteRank %u >= rankSize %u",
                                curReq.remoteUserRank, rankSize),
                            HCCL_E_INTERNAL);

                        // 获取curRank与remoteRank之间的LINK
                        const LINK& curLink = links[reqIdx];
                        CHK_PTR_NULL(curLink);

                        // 获取user input memory range if any
                        if (curReq.inputMemType == TransportMemType::PARAM_INPUT
                            || curReq.inputMemType == TransportMemType::CCL_INPUT) {
                            // 获取remoteRank的user input memory baseaddr
                            void* remoteUserInputBaseAddr = nullptr;
                            CHK_RET(curLink->GetRemoteMem(UserMemType::INPUT_MEM, &remoteUserInputBaseAddr));
                            CHK_PTR_NULL(remoteUserInputBaseAddr);

                            HCCL_INFO(
                                "[AicpuCacheManager][PrepareUserMemRanges] prepare user input of remoteRank[%u] for"
                                "graph mode; baseAddr[0x%016llx]",
                                curReq.remoteUserRank, remoteUserInputBaseAddr);

                            // 设置remoteRank对应的user input memory range
                            OpUnfoldMemRange& remoteUserInputMemRange = userInputMemRanges[curReq.remoteUserRank];
                            remoteUserInputMemRange.isValid = true;
                            remoteUserInputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserInputBaseAddr);
                            if (curReq.inputMemType == TransportMemType::PARAM_INPUT) { // user input
                                remoteUserInputMemRange.memSize = inputSize;
                            } else if (curReq.inputMemType == TransportMemType::CCL_INPUT) { // hccl input
                                remoteUserInputMemRange.memSize = algResource.cclInputMem.size();
                            } else {
                                HCCL_ERROR(
                                    "[AicpuCacheManager][PrepareUserMemRanges] invalid curReq.inputMemType[%u]",
                                    curReq.inputMemType);
                                return HCCL_E_INTERNAL;
                            }
                        }

                        // 获取user output memory range if any
                        if (curReq.outputMemType == TransportMemType::PARAM_OUTPUT
                            || curReq.outputMemType == TransportMemType::CCL_OUTPUT) {
                            // 获取remoteRank的user output memory baseaddr
                            void* remoteUserOutputBaseAddr = nullptr;
                            CHK_RET(curLink->GetRemoteMem(UserMemType::OUTPUT_MEM, &remoteUserOutputBaseAddr));
                            CHK_PTR_NULL(remoteUserOutputBaseAddr);

                            HCCL_INFO(
                                "[AicpuCacheManager][PrepareUserMemRanges] prepare user output of remoteRank[%u] for"
                                "graph mode; baseAddr[0x%016llx]",
                                curReq.remoteUserRank, remoteUserOutputBaseAddr);

                            // 设置remoteRank对应的user output memory range
                            OpUnfoldMemRange& remoteUserOutputMemRange = userOutputMemRanges[curReq.remoteUserRank];
                            remoteUserOutputMemRange.isValid = true;
                            remoteUserOutputMemRange.baseAddr = reinterpret_cast<uint64_t>(remoteUserOutputBaseAddr);
                            if (curReq.outputMemType == TransportMemType::PARAM_OUTPUT) { // user output
                                remoteUserOutputMemRange.memSize = outputSize;
                            } else if (curReq.outputMemType == TransportMemType::CCL_OUTPUT) { // hccl output
                                remoteUserOutputMemRange.memSize = algResource.cclOutputMem.size();
                            } else {
                                HCCL_ERROR(
                                    "[AicpuCacheManager][PrepareUserMemRanges] invalid curReq.outputMemType[%u]",
                                    curReq.outputMemType);
                                return HCCL_E_INTERNAL;
                            }
                        }
                    } // curReq.isValid
                } // Each TransportRequest
            } // Each SingleSubCommTransport
        } // Each LevelNSubCommTransport
    }

    // 打印debug信息
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        for (size_t rankId = 0; rankId < userInputMemRanges.size(); ++rankId) {
            const OpUnfoldMemRange& userInputMemRange = userInputMemRanges[rankId];
            HCCL_INFO(
                "[AicpuCacheManager][PrepareUserMemRanges] userInputMemRanges[%u] isValid: %d, baseAddr: 0x%016llx, "
                "memSize: %llu, endAddr: 0x%016llx",
                rankId, userInputMemRange.isValid, userInputMemRange.baseAddr, userInputMemRange.memSize,
                userInputMemRange.baseAddr + userInputMemRange.memSize);

            const OpUnfoldMemRange& userOutputMemRange = userOutputMemRanges[rankId];
            HCCL_INFO(
                "[AicpuCacheManager][PrepareUserMemRanges] userOutputMemRanges[%u] isValid: %d, baseAddr: 0x%016llx, "
                "memSize: %llu, endAddr: 0x%016llx",
                rankId, userOutputMemRange.isValid, userOutputMemRange.baseAddr, userOutputMemRange.memSize,
                userOutputMemRange.baseAddr + userOutputMemRange.memSize);
        }
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::ParseOpParamForCache(
    const OpParam& param, HcclDataType& sendType, HcclDataType& recvType, uint64_t& inputSize, uint64_t& outputSize,
    const HcclTopoInfo& topoinfo)
{
    // 注意: 由于ParseOpParamForCache前已经做过NeedOpUnfoldCache检查, 这里不再做重复检验

    const HcclCMDType opType = param.opType;
    const uint32_t rankSize = topoinfo.userRankSize;

    // 准备data type和count
    // NOTE: 非V类算子 (DataRes), V类算子 (VDataDes), All2All类算子 (All2AllDataDes), batch类算子
    // (BatchSendRecvDataDes/BatchWriteDataDes)
    if (opType == HcclCMDType::HCCL_CMD_ALLTOALL) { // alltoall算子
        // 注意: sendType和recvType一定相同
        sendType = param.All2AllDataDes.sendType;
        recvType = param.All2AllDataDes.recvType;

        // 注意: 对于alltoall算子, inputSize和outputSize一定相同 (但不能直接使用param.input/outputSize,
        // alltoall算子不会设置这两个字段)
        inputSize = param.All2AllDataDes.sendCount * rankSize * SIZE_TABLE[sendType];
        outputSize = inputSize; // 注意: 不能使用param.All2AllDataDes.recvCount * rankSize * SIZE_TABLE[recvType],
                                // 因为alltoall使用sendCount来表示send/recvCount, 而recvCount本身为0
    } else if (IsAlltoallvType(opType)) { // alltoallv类算子
        // 计算相应字段 (虽然GetOpUnfoldKey不需要, 但是PrepareUserMemRanges需要)

        // 注意: sendType和recvType一定相同
        sendType = param.All2AllDataDes.sendType;
        recvType = param.All2AllDataDes.recvType;

        // 注意: 对于alltoallv算子, inputSize和outputSize不一定相同 (但不能直接使用param.input/outputSize,
        // alltoallv算子不会设置这两个字段)
        // 参考coll_all_to_all_v_direct_fullmesh_executor.cc下的CollRunAlltoAllDirectFullmesh::GetLocalSendRecvInfoforAlltoallV
        inputSize = 0;
        outputSize = 0;
        if (opType == HcclCMDType::HCCL_CMD_ALLTOALLV) { // alltoallv算子
            HCCL_INFO(
                "[AicpuCacheManager][ParseOpParamForCache] sum %u send/recv counts for input/output size", rankSize);
            for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
                // curRank发送到tmpRank的数据量
                const uint64_t curSendCounts = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + tmpRank);
                const uint64_t curSendLength = curSendCounts * SIZE_TABLE[sendType];
                inputSize += curSendLength;

                // curRank从tmpRank接收的数据量
                const uint64_t curRecvCounts = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + tmpRank);
                const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
                outputSize += curRecvLength;
            }
        } else if (opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) { // alltoallvc算子
            const uint32_t curRank = topoinfo.userRank;
            HCCL_INFO(
                "[AicpuCacheManager][ParseOpParamForCache] sum %u-size sendCountMatrix for input/output size",
                rankSize);
            for (uint32_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
                // curRank发送到tmpRank的数据量
                const uint64_t curSendCounts
                    = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + curRank * rankSize
                        + tmpRank); // sendCountMatrix[curRank][tmpRank]
                const uint64_t curSendLength = curSendCounts * SIZE_TABLE[sendType];
                inputSize += curSendLength;

                // curRank从tmpRank接收到的数据量
                const uint64_t curRecvCounts
                    = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + tmpRank * topoinfo.userRankSize
                        + curRank); // sendCountMatrix[tmpRank][curRank]
                const uint64_t curRecvLength = curRecvCounts * SIZE_TABLE[recvType];
                outputSize += curRecvLength;
            }
        } else {
            HCCL_ERROR("[AicpuCacheManager][ParseOpParamForCache] invalid opType[%u] for alltoallv", opType);
            return HCCL_E_INTERNAL;
        }
    } else { // 非V类算子
        sendType = param.DataDes.dataType;
        recvType = param.DataDes.dataType;
        inputSize = param.inputSize;
        outputSize = param.outputSize;
    }

    HCCL_DEBUG(
        "[AicpuCacheManager][ParseOpParamForCache] opType[%u] rankSize[%u] sendType[%u] recvType[%u] inputSize[%u] "
        "outputSize[%u]",
        opType, rankSize, sendType, recvType, inputSize, outputSize);

    return HCCL_SUCCESS;
}

HcclResult AicpuCacheManager::PrepareAlltoallvSendRecvInfo(
    const OpParam& param, AlltoallvSendRecvInfo& alltoallvSendRecvInfo, const HcclTopoInfo& topoinfo)
{
    const uint32_t rankSize = topoinfo.userRankSize;
    HCCL_INFO("[AicpuCacheManager][PrepareAlltoallvSendRecvInfo] prepare %u send/recv info", rankSize);

    // 准备send/recv data type
    const HcclDataType sendType = param.All2AllDataDes.sendType;
    alltoallvSendRecvInfo.sendType = sendType;
    const HcclDataType recvType = param.All2AllDataDes.recvType;
    alltoallvSendRecvInfo.recvType = recvType;

    // 初始化send/recv counts
    alltoallvSendRecvInfo.sendCounts.resize(rankSize);
    alltoallvSendRecvInfo.recvCounts.resize(rankSize);

    // 初始化send/recv offsets
    alltoallvSendRecvInfo.sendOffsets.resize(rankSize);
    alltoallvSendRecvInfo.recvOffsets.resize(rankSize);

    // 参考coll_all_to_all_v_direct_fullmesh_executor.cc中的GetLocalSendRecvInfoforAlltoallV
    const uint32_t sendTypeSize = SIZE_TABLE[sendType];    // Size of sendType in units of bytes
    const uint32_t recvTypeSize = SIZE_TABLE[recvType];    // Size of recvType in units of bytes
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV) { // alltoallv
        // 准备send counts
        for (size_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            alltoallvSendRecvInfo.sendCounts[dstRank]
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCounts) + dstRank);
        }

        // 准备recv counts
        for (size_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            alltoallvSendRecvInfo.recvCounts[dstRank]
                = *(static_cast<const u64*>(param.All2AllDataDes.recvCounts) + dstRank);
        }

        // 准备send offsets
        for (size_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            const uint64_t curSendDispls = *(static_cast<const uint64_t*>(param.All2AllDataDes.sdispls) + dstRank);
            alltoallvSendRecvInfo.sendOffsets[dstRank] = curSendDispls * sendTypeSize;
        }

        // 准备recv offsets
        for (size_t dstRank = 0; dstRank < rankSize; ++dstRank) {
            const uint64_t curRecvDispls = *(static_cast<const uint64_t*>(param.All2AllDataDes.rdispls) + dstRank);
            alltoallvSendRecvInfo.recvOffsets[dstRank] = curRecvDispls * recvTypeSize;
        }
    } else if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) { // alltoallvc
        const uint32_t curRank = topoinfo.userRank;

        // 准备send counts
        for (size_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            alltoallvSendRecvInfo.sendCounts[tmpRank]
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + curRank * rankSize
                    + tmpRank); // sendCountMatrix[curRank][tmpRank]
        }

        // 准备recv counts
        for (size_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            alltoallvSendRecvInfo.recvCounts[tmpRank]
                = *(static_cast<const u64*>(param.All2AllDataDes.sendCountMatrix) + tmpRank * topoinfo.userRankSize
                    + curRank); // sendCountMatrix[tmpRank][curRank]
        }

        // 准备send offsets
        uint64_t curSendDispls = 0;
        for (size_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            alltoallvSendRecvInfo.sendOffsets[tmpRank] = curSendDispls * sendTypeSize;
            curSendDispls += alltoallvSendRecvInfo.sendCounts[tmpRank];
        }

        // 准备recv offsets
        uint64_t curRecvDispls = 0;
        for (size_t tmpRank = 0; tmpRank < rankSize; ++tmpRank) {
            alltoallvSendRecvInfo.recvOffsets[tmpRank] = curRecvDispls * recvTypeSize;
            curRecvDispls += alltoallvSendRecvInfo.recvCounts[tmpRank];
        }
    } else {
        HCCL_ERROR("[AicpuCacheManager][PrepareAlltoallvSendRecvInfo] invalid opType[%u] for alltoallv", param.opType);
        return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

} // namespace hccl
