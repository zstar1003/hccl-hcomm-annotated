/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <cstdint>
#include <cstdlib>
#include <cstring>

#include "aicpu_task_cache_entry.h"
#include "aicpu_task_utils.h"

#include "aicpu_indop_env.h"
#include "log.h"
#include "rma_buffer_lite.h"
#include "buffer.h"
#include "rmt_rma_buf_slice_lite.h"
#include "sqe_v82.h"

using Hccl::GetPlfDebugConfigValue;

namespace {
constexpr uint8_t SDMA_OPCODE_MEMCPY = 0;
}
using Hccl::PLF_TASK;
#ifdef HCCL_V2 // hccl_v2
using Hccl::HCCL_LOG_DEBUG;
using Hccl::HCCL_LOG_INFO;
using Hccl::HcclCheckLogLevel;
#endif

using Hccl::AC_SQE_SIZE;
using hccl::AicpuTsThread;
using Hccl::DbSqeProfInfo;
using Hccl::Rt91095StarsMemcpySqe;
using Hccl::Rt91095StarsSqeHeader;
using Hccl::Rt91095StarsSqeType;
using Hccl::Rt91095StarsUbdmaDBmodeSqe;
using Hccl::Rt91095StarsWriteValueSqe;
using Hccl::RtsqA5;
using Hccl::StreamLite;
using Hccl::TaskParamTypeVal;
using Hccl::UbConnLite;
using Hccl::UbTransportLiteImpl;
using Hccl::UdmaSqeCommon;
using Hccl::UdmaSqOpcode;
using Hccl::WqeTask;
using std::vector;

namespace hcomm {

AddrRefreshInfo::AddrRefreshInfo() : needRefresh(false), memIdx(0), offset(0) {}

AddrRefreshInfo::AddrRefreshInfo(const uint32_t curMemIdx) : needRefresh(true), memIdx(curMemIdx), offset(0) {}

AddrRefreshInfo::AddrRefreshInfo(const AddrRefreshInfo& other)
    : needRefresh(other.needRefresh),
      memIdx(other.memIdx),
      offset(other.offset)
{}

AddrRefreshInfo::~AddrRefreshInfo() {}

const AddrRefreshInfo& AddrRefreshInfo::operator=(const AddrRefreshInfo& other)
{
    if (this != &other) {
        needRefresh = other.needRefresh;
        offset = other.offset;
        memIdx = other.memIdx;
    }
    return *this;
}

AicpuTaskCacheEntry::AicpuTaskCacheEntry()
{
    if ((UNLIKELY((GetPlfDebugConfigValue() & PLF_TASK) != 0)) || UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        isTaskConfigDebug_ = true;
    }
}

AicpuTaskCacheEntry::~AicpuTaskCacheEntry()
{
    size_t sqeArrayCount = sqeArrayInfos_.size();
    size_t totalSqeCount = 0;
    for (size_t arrayIdx = 0; arrayIdx < sqeArrayCount; ++arrayIdx) {
        totalSqeCount += sqeArrayInfos_[arrayIdx].srcAddrRefreshInfoArray.size();

        uint8_t* curSqeArray = sqeArrayInfos_[arrayIdx].sqeArray;
        if (UNLIKELY(curSqeArray == nullptr)) {
            HCCL_ERROR("[AicpuTaskCacheEntry][~AicpuTaskCacheEntry] curSqeArray is nullptr");
        } else {
            free(curSqeArray);
            sqeArrayInfos_[arrayIdx].sqeArray = nullptr;
        }
    }

    size_t wqeArrayCount = wqeTaskArrayInfos_.size();
    size_t totalWqeCount = 0;
    for (size_t arrayIdx = 0; arrayIdx < wqeArrayCount; ++arrayIdx) {
        totalWqeCount += wqeTaskArrayInfos_[arrayIdx].wqeTaskArray.size();
    }

    HCCL_INFO(
        "[AicpuTaskCacheEntry][~AicpuTaskCacheEntry] release %u SQE arrays "
        "(%u SQEs in total) and %u WQE arrays (%u WQEs in total) from a cache entry",
        sqeArrayCount, totalSqeCount, wqeArrayCount, totalWqeCount);
}

HcclResult
AicpuTaskCacheEntry::InitCacheEntry(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 校验count (当前rank的userIn和userOut)
    constexpr uint64_t ADDRS_COUNT = 2;
    CHK_PRT_RET(
        count != ADDRS_COUNT,
        HCCL_ERROR("[AicpuTaskCacheEntry][%s] count[%llu] != ADDRS_COUNT[%llu]", __func__, count, ADDRS_COUNT),
        HCCL_E_PARA);

    // 空指针及溢出检查
    for (uint32_t i = 0; i < count; i++) {
        CHK_PRT_RET(
            baseAddrs[i] == 0,
            HCCL_ERROR("[AicpuTaskCacheEntry][InitCacheEntry] baseAddrs[%u] is 0, memSize[%llu]", i, memSizes[i]),
            HCCL_E_PARA);
        CHK_PRT_RET(
            baseAddrs[i] + memSizes[i] < baseAddrs[i],
            HCCL_ERROR("[AicpuTaskCacheEntry][InitCacheEntry] baseAddrs[%u] + memSizes[%u] overflows", i, i),
            HCCL_E_PARA);
    }

    // 缓存baseAddrs
    CHK_PRT_RET(
        cachedBaseAddrs_.size() != 0,
        HCCL_ERROR("[AicpuTaskCacheEntry][%s] cachedBaseAddrs_.size[%u] != 0", __func__, cachedBaseAddrs_.size()),
        HCCL_E_INTERNAL);
    cachedBaseAddrs_.resize(count);
    entryBytes_ += count * sizeof(uint64_t);
    CHK_SAFETY_FUNC_RET(
        memcpy_s(cachedBaseAddrs_.data(), count * sizeof(uint64_t), baseAddrs, count * sizeof(uint64_t)));

    // 缓存sizes
    CHK_PRT_RET(
        cachedMemSizes_.size() != 0,
        HCCL_ERROR("[AicpuTaskCacheEntry][%s] cachedMemSizes_.size[%u] != 0", __func__, cachedMemSizes_.size()),
        HCCL_E_INTERNAL);
    cachedMemSizes_.resize(count);
    entryBytes_ += count * sizeof(uint64_t);
    CHK_SAFETY_FUNC_RET(memcpy_s(cachedMemSizes_.data(), count * sizeof(uint64_t), memSizes, count * sizeof(uint64_t)));

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::AddSqeArray(
    RtsqA5* rtsqPtr, AicpuTsThread* aicpuTsThreadPtr, const uint64_t sqeCount, const uint8_t* sqeArray,
    const uint32_t streamId)
{
    CHK_PTR_NULL(rtsqPtr);
    CHK_PTR_NULL(aicpuTsThreadPtr);
    CHK_PTR_NULL(sqeArray);
    CHK_PRT_RET(sqeCount == 0, HCCL_ERROR("[AicpuTaskCacheEntry][AddSqeArray] sqeCount is 0"), HCCL_E_INTERNAL);

    const size_t sqeBytes = sqeCount * AC_SQE_SIZE;
    uint8_t* newSqeArray = ReinterpretAs<uint8_t*>(malloc(sqeBytes));
    CHK_PTR_NULL(newSqeArray);
    HcclResult ret = AddSqeArray_(newSqeArray, sqeBytes, sqeArray, streamId);
    if (ret != HCCL_SUCCESS) {
        free(newSqeArray);
        return ret;
    }

    // 更新SQE数组
    sqeArrayInfos_.emplace_back();
    SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_.back();
    sqeArrayInfo.sqeArray = newSqeArray;
    sqeArrayInfo.rtsqPtr = rtsqPtr;
    sqeArrayInfo.aicpuTsThreadPtr = aicpuTsThreadPtr;
    sqeArrayInfo.sqeCount = sqeCount;
    sqeArrayInfo.srcAddrRefreshInfoArray.resize(sqeCount);
    sqeArrayInfo.dstAddrRefreshInfoArray.resize(sqeCount);
    entryBytes_ += sqeArrayInfo.GetSize();

    // 更新下发顺序
    launchOrder_.emplace_back(TaskArrayType::kTaskArrayTypeSqe);
    entryBytes_ += sizeof(TaskArrayType);

    HCCL_INFO(
        "[AicpuTaskCacheEntry][AddSqeArray] add %uth sqe array with sqeCount[%llu] streamId[%u]",
        sqeArrayInfos_.size() - 1, sqeCount, streamId);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::AddWqeArray(
    UbConnLite* ubConnLitePtr, UbTransportLiteImpl* ubTransportLiteImplPtr, const vector<WqeTask>& wqeTasks,
    const uint32_t streamId, const uint32_t dbSqeIdx, const bool isReportTask, const DbSqeProfInfo& dbSqeProfInfo)
{
    CHK_PRT_RET(
        isReportTask != dbSqeProfInfo.isValid,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][AddWqeArray] isReportTask[%d] != dbSqeProfInfo.isValid[%d]", isReportTask,
            dbSqeProfInfo.isValid),
        HCCL_E_INTERNAL);
    CHK_PTR_NULL(ubConnLitePtr);
    CHK_PTR_NULL(ubTransportLiteImplPtr);

    const uint32_t wqeCount = wqeTasks.size();
    CHK_PRT_RET(wqeCount == 0, HCCL_ERROR("[AicpuTaskCacheEntry][AddWqeArray] wqeCount is 0"), HCCL_E_INTERNAL);

    // 更新streamIdToDbSqeTmpInfoMap_ (仅cache miss时临时维护, 展开完成后一定为空)
    DbSqeTmpInfo dbSqeTmpInfo;
    dbSqeTmpInfo.wqeArrayIdx = wqeTaskArrayInfos_.size();
    dbSqeTmpInfo.dbSqeIdx = dbSqeIdx;
    dbSqeTmpInfo.isReportTask = isReportTask;
    if (isReportTask) {
        dbSqeTmpInfo.dbSqeProfInfo = dbSqeProfInfo;
    }
    std::unordered_map<uint32_t, vector<DbSqeTmpInfo>>::iterator mapIter = streamIdToDbSqeTmpInfoMap_.find(streamId);
    if (mapIter == streamIdToDbSqeTmpInfoMap_.end()) {
        mapIter = streamIdToDbSqeTmpInfoMap_.emplace(streamId, vector<DbSqeTmpInfo>()).first;
    }
    mapIter->second.push_back(dbSqeTmpInfo);

    // 更新WQE数组
    wqeTaskArrayInfos_.emplace_back();
    WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_.back();
    wqeTaskArrayInfo.wqeTaskArray = wqeTasks;
    wqeTaskArrayInfo.ubConnLitePtr = ubConnLitePtr;
    wqeTaskArrayInfo.ubTransportLiteImplPtr = ubTransportLiteImplPtr;
    wqeTaskArrayInfo.dbSqeLocation.dbSqeIdx = dbSqeIdx;
    wqeTaskArrayInfo.locAddrRefreshInfoArray.resize(wqeCount);
    wqeTaskArrayInfo.rmtAddrRefreshInfoArray.resize(wqeCount);
    entryBytes_ += wqeTaskArrayInfo.GetSize();

    // 更新下发顺序
    launchOrder_.emplace_back(TaskArrayType::kTaskArrayTypeWqe);
    entryBytes_ += sizeof(TaskArrayType);

    // 初始化token info
    if (tokenInfosMap_.find(ubTransportLiteImplPtr) == tokenInfosMap_.end()) {
        const uint64_t addrCnt = cachedBaseAddrs_.size();
        tokenInfosMap_.emplace(ubTransportLiteImplPtr, vector<TokenInfo>(addrCnt));
        entryBytes_ += (sizeof(UbTransportLiteImplHandle) + addrCnt * sizeof(TokenInfo));
    }

    HCCL_INFO(
        "[AicpuTaskCacheEntry][AddWqeArray] add %uth wqe array with wqeCount[%u] dbSqeTmpInfo[%u, %d, %u] streamId[%u]",
        wqeTaskArrayInfos_.size() - 1, wqeCount, dbSqeTmpInfo.wqeArrayIdx, dbSqeTmpInfo.isReportTask,
        dbSqeTmpInfo.dbSqeIdx, streamId);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::SubmitCacheEntry()
{
    // 校验streamIdToDbSqeTmpInfoMap_
    // 注意: SubmitCacheEntry时, 算子展开一定完成且task全部launch, 因此所有DbSqe的临时信息一定均已消耗
    if (UNLIKELY(streamIdToDbSqeTmpInfoMap_.size() != 0)) {
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][SubmitCacheEntry] streamIdToDbSqeTmpInfoMap_.size[%u] != 0",
            streamIdToDbSqeTmpInfoMap_.size());
        for (const auto& mapIter : streamIdToDbSqeTmpInfoMap_) {
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] streamId[%u] wqeArrayIdx[%u] dbSqeIdx[%u] isReportTask[%d]",
                mapIter.first, mapIter.second[0].wqeArrayIdx, mapIter.second[0].dbSqeIdx,
                mapIter.second[0].isReportTask);
        }
        return HCCL_E_INTERNAL;
    }

    // 地址信息已经通过InitCacheEntry保存
    const uint64_t count = cachedBaseAddrs_.size();
    CHK_PRT_RET(
        count == 0, HCCL_ERROR("[AicpuTaskCacheEntry][SubmitCacheEntry] cachedBaseAddrs_.size is 0"), HCCL_E_INTERNAL);

    // 打印dynamic memory ranges
    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        for (size_t memIdx = 0; memIdx < count; memIdx++) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] memRanges[%u]: [0x%016llx, 0x%016llx)", memIdx,
                cachedBaseAddrs_[memIdx], cachedBaseAddrs_[memIdx] + cachedMemSizes_[memIdx]);
        }
    }

    CHK_RET(SubmitSqeAddrRefreshInfo_());
    CHK_RET(SubmitWqeAddrRefreshInfoAndTokenInfo_());
    CHK_RET(SubmitDbSqeProfRefreshInfo_());
    CHK_RET(ValidateLaunchOrder_());

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitSqeAddrRefreshInfo_()
{
    // 更新每段SQE数组的AddrRefreshInfo
    for (size_t arrayIdx = 0; arrayIdx < sqeArrayInfos_.size(); arrayIdx++) {
        SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[arrayIdx];
        const uint64_t sqeCount = sqeArrayInfo.sqeCount;
        const uint8_t* sqePtr = sqeArrayInfo.sqeArray;
        CHK_PTR_NULL(sqePtr);
        for (size_t sqeIdx = 0; sqeIdx < sqeCount; sqeIdx++) {
            CHK_RET(UpdateSqeAddrRefreshInfo_(
                sqePtr, sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx], sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx]));

            // 打印SQE AddrRefreshInfo
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] sqeArrayInfos_[%u][%u]: "
                "srcAddrRefreshInfo[needRefresh-%d memIdx-%u] dstAddrRefreshInfo[needRefresh-%d memIdx-%u]",
                arrayIdx, sqeIdx, sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx].needRefresh,
                sqeArrayInfo.srcAddrRefreshInfoArray[sqeIdx].memIdx,
                sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx].needRefresh,
                sqeArrayInfo.dstAddrRefreshInfoArray[sqeIdx].memIdx);
            sqePtr += AC_SQE_SIZE;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitWqeAddrRefreshInfoAndTokenInfo_()
{
    // 更新每段WQE数组的AddrRefreshInfo, 并更新token info
    for (size_t arrayIdx = 0; arrayIdx < wqeTaskArrayInfos_.size(); arrayIdx++) {
        // 获取token info
        WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_[arrayIdx];
        UbTransportLiteImpl* ubTransportLiteImplPtr
            = wqeTaskArrayInfo.ubTransportLiteImplPtr; // 注意: AddWqeArray时已校验非空
        std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter
            = tokenInfosMap_.find(ubTransportLiteImplPtr);

        CHK_PRT_RET(
            iter == tokenInfosMap_.end(),
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] ubTransportLiteImplPtr[%p] not found in tokenInfosMap_",
                ubTransportLiteImplPtr),
            HCCL_E_INTERNAL);
        vector<TokenInfo>& tokenInfos = iter->second;

        // 更新AddrRefreshInfo和token info
        const uint32_t wqeCount = wqeTaskArrayInfo.wqeTaskArray.size();
        vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
        for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
            CHK_RET(UpdateWqeAddrRefreshInfoAndTokenInfo_(
                wqeTasks[wqeIdx], wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx],
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx], tokenInfos));

            // 打印WQE AddrRefreshInfo
            HCCL_INFO(
                "[AicpuTaskCacheEntry][SubmitCacheEntry] wqeTaskArrayInfos_[%u][%u]: "
                "locAddrRefreshInfo[needRefresh-%d memIdx-%u] needLocTokenIdFlag[%d] "
                "rmtAddrRefreshInfo[needRefresh-%d memIdx-%u] needRmtTokenIdAndValueFlag[%d]",
                arrayIdx, wqeIdx, wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].needRefresh,
                wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].memIdx,
                tokenInfos[wqeTaskArrayInfo.locAddrRefreshInfoArray[wqeIdx].memIdx].needLocTokenIdFlag,
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].needRefresh,
                wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].memIdx,
                tokenInfos[wqeTaskArrayInfo.rmtAddrRefreshInfoArray[wqeIdx].memIdx].needRmtTokenIdAndValueFlag);
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::SubmitDbSqeProfRefreshInfo_()
{
    // 更新每个DbSqeProfAndRefreshInfo的AddrRefreshInfo
    for (std::unordered_map<DbSqeLocation, DbSqeProfAndRefreshInfo>::iterator iter = dbSqeLocInfoMap_.begin();
         iter != dbSqeLocInfoMap_.end(); iter++) {
        DbSqeProfAndRefreshInfo& dbSqeProfAndRefreshInfo = iter->second;
        const DbSqeProfInfo& dbSqeProfInfo = dbSqeProfAndRefreshInfo.dbSqeProfInfo;
        switch (static_cast<u8>(dbSqeProfInfo.taskParamType)) {
            case TaskParamTypeVal::TASK_UB:
            case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
            case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.locAddr, dbSqeProfAndRefreshInfo.locAddrRefreshInfo, AddrType::kAddrTypeLocal));
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.rmtAddr, dbSqeProfAndRefreshInfo.rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
                break;
            case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
                CHK_RET(UpdateAddrRefreshInfo_(
                    dbSqeProfInfo.rmtAddr, dbSqeProfAndRefreshInfo.rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
                break;
            default:
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][%s] invalid taskParamType[%u]", __func__, dbSqeProfInfo.taskParamType);
                return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::ValidateLaunchOrder_()
{
    // 按照缓存时的下发顺序, 校验launchOrder_对应的sqe/wqeArrayIdx, 避免后续命中时重复校验
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;
    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            // 校验arrayIdx
            CHK_PRT_RET(
                sqeArrayIdx >= sqeArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][SubmitCacheEntry] sqeArrayIdx[%u] >= sqeArrayInfos_.size[%u]", sqeArrayIdx,
                    sqeArrayInfos_.size()),
                HCCL_E_PARA);
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            // 校验arrayIdx
            CHK_PRT_RET(
                wqeArrayIdx >= wqeTaskArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][SubmitCacheEntry] wqeArrayIdx[%u] >= wqeTaskArrayInfos_.size[%u]",
                    wqeArrayIdx, wqeTaskArrayInfos_.size()),
                HCCL_E_PARA);
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][SubmitCacheEntry] invalid task array type");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
AicpuTaskCacheEntry::RefreshAndLaunch(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 校验count
    CHK_PRT_RET(
        cachedBaseAddrs_.size() != count || cachedMemSizes_.size() != count,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][RefreshAndLaunch] cachedBaseAddrs_.size[%u] / cachedMemSizes_.size[%u] != count[%u]",
            cachedBaseAddrs_.size(), cachedMemSizes_.size(), count),
        HCCL_E_INTERNAL);

    // 校验memSizes
    for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
        CHK_PRT_RET(
            cachedMemSizes_[memIdx] != memSizes[memIdx],
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] cachedMemSizes_[%u] != memSizes[%u]", memIdx, memIdx),
            HCCL_E_INTERNAL);
    }

    CHK_RET(RefreshTokenInfos_(baseAddrs, memSizes, count));

    const bool enableTaskException = hcomm::GetTaskExceptionEnable();
    const bool l1State = Hccl::DfxProfilingHandlerLite::GetInstance().GetProfL1State();
    const bool needTaskParam = (l1State || enableTaskException);
    HCCL_INFO(
        "[AicpuTaskCacheEntry][RefreshAndLaunch] l1State[%d] enableTaskException[%d] -> needTaskParam[%d]", l1State,
        enableTaskException, needTaskParam);

    CHK_RET(LaunchTasksByOrder_(baseAddrs, memSizes, count, needTaskParam));

    if (UNLIKELY(HcclCheckLogLevel(HCCL_LOG_INFO))) {
        CHK_RET(PrintRefreshResult_(baseAddrs, memSizes, count));
    }

    return HCCL_SUCCESS;
}

inline HcclResult
AicpuTaskCacheEntry::RefreshTokenInfos_(const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 按需统一获取新传入内存地址的token信息用于WQE刷新, 避免重复构造Buffer
    for (std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter = tokenInfosMap_.begin();
         iter != tokenInfosMap_.end(); iter++) {
        vector<TokenInfo>& tokenInfos = iter->second;
        CHK_PRT_RET(
            tokenInfos.size() != count,
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] tokenInfos.size[%u] != count[%u]", tokenInfos.size(), count),
            HCCL_E_INTERNAL);

        // 注意: AddWqeArray时已校验非空, 无需重复校验
        UbTransportLiteImpl* ubTransportLitePtr = ReinterpretAs<UbTransportLiteImpl*>(iter->first);
        for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
            const uint64_t baseAddr = baseAddrs[memIdx];
            const uint64_t memSize = memSizes[memIdx];
            TokenInfo& tokenInfo = tokenInfos[memIdx];

            if (tokenInfo.needLocTokenIdFlag) {
                // 参考hccl_api_data_aicpu_ts.cc (例如HcommWriteOnThread), 获取新loc token id
                // 注意: 无需通过ubTransportLitePtr->GetRmaBufSlicelite(locRmaBuf)构造Hccl::RmaBufSliceLite
                // locRmaBufSlicelite,
                //     再调用locRmaBufSlicelite.GetTokenId()获取token id (一定与Hccl::RmaBufferLite locRmaBuf的token
                //     id相同)
                Hccl::RmaBufferLite locRmaBuf;
                CHK_RET(
                    ubTransportLitePtr->BuildLocRmaBufferLite(ReinterpretAs<uintptr_t>(baseAddr), memSize, locRmaBuf));
                tokenInfo.locTokenId = locRmaBuf.GetTokenId();
            }

            if (tokenInfo.needRmtTokenIdAndValueFlag) {
                // 参考hccl_api_data_aicpu_ts.cc (例如HcommRead/Write/WriteReduce/WriteWithNotifyOnThread), 获取新rmt
                // token id/value 注意: Hccl::Buffer本身不含token id/value,
                // 必须通过调用ubTransportLitePtr->GetRmtRmaBufSliceLite,
                //     构造Hccl::RmtRmaBufSliceLite, 再调用GetTokenId/Value获取token id/value
                const Hccl::Buffer rmtBuf{ReinterpretAs<uintptr_t>(baseAddr), memSize};
                Hccl::RmtRmaBufSliceLite rmtRmaBufSlicelite = ubTransportLitePtr->GetRmtRmaBufSliceLite(rmtBuf);
                tokenInfo.rmtTokenId = rmtRmaBufSlicelite.GetTokenId();
                tokenInfo.rmtTokenValue = rmtRmaBufSlicelite.GetTokenValue();
            }
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchTasksByOrder_(
    const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count, bool needTaskParam)
{
    // 按照缓存时的下发顺序, 依次刷新并下发task
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;
    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            // 注意: sqeArrayIdx已在SubmitCacheEntry校验, 无需重复校验
            const SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[sqeArrayIdx];

            // 刷新SQE数组
            CHK_RET(RefreshSqeTasks_(sqeArrayInfo, baseAddrs));

            // 下发SQE
            CHK_RET(LaunchSqeTasks_(sqeArrayInfo));

            // 对刷新后的SQE填充DfxTaskInfo并经NextTaskSlot上报
            if (needTaskParam) {
                CHK_RET(ReportSqeArrayProfiling_(sqeArrayIdx, baseAddrs, memSizes, count));
            }
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            // 注意: wqeArrayIdx已在SubmitCacheEntry校验, 无需重复校验
            WqeTaskArrayInfo& wqeTaskArrayInfo = wqeTaskArrayInfos_[wqeArrayIdx];

            // 刷新WQE数组
            CHK_RET(RefreshWqeTasks_(wqeTaskArrayInfo, baseAddrs, memSizes, count));

            // 下发WQE数组
            CHK_RET(LaunchWqeTasks_(wqeTaskArrayInfo));

            // 刷新对应DbSqe
            CHK_RET(RefreshDbSqe_(wqeTaskArrayInfo));
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid task array type[%u]", taskType);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::PrintRefreshResult_(
    [[maybe_unused]] const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes, const uint32_t count)
{
    // 打印更新后的token信息
    for (std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::iterator iter = tokenInfosMap_.begin();
         iter != tokenInfosMap_.end(); iter++) {
        UbTransportLiteImpl* ubTransportLitePtr = ReinterpretAs<UbTransportLiteImpl*>(iter->first);
        vector<TokenInfo>& tokenInfos = iter->second;
        for (uint32_t memIdx = 0; memIdx < count; memIdx++) {
            TokenInfo& tokenInfo = tokenInfos[memIdx];
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] tokenInfosMap_[0x%016llx][%u]: "
                "needLocTokenIdFlag[%d]; rmtTokenIdAndValueFlag[%d]",
                ubTransportLitePtr, memIdx, tokenInfo.needLocTokenIdFlag, tokenInfo.needRmtTokenIdAndValueFlag);
        }
    }
    // 打印刷新的顺序
    size_t sqeArrayIdx = 0;
    size_t wqeArrayIdx = 0;
    for (size_t launchIdx = 0; launchIdx < launchOrder_.size(); launchIdx++) {
        const TaskArrayType taskType = launchOrder_[launchIdx];
        if (taskType == TaskArrayType::kTaskArrayTypeSqe) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] launchedArray[%u] is sqeArray[%u]", launchIdx, sqeArrayIdx);
            ++sqeArrayIdx;
        } else if (taskType == TaskArrayType::kTaskArrayTypeWqe) {
            HCCL_INFO(
                "[AicpuTaskCacheEntry][RefreshAndLaunch] launchedArray[%u] is wqeArray[%u]", launchIdx, wqeArrayIdx);
            ++wqeArrayIdx;
        } else {
            HCCL_ERROR("[AicpuTaskCacheEntry][RefreshAndLaunch] invalid task array type[%u]", taskType);
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::AddSqeArray_(
    uint8_t* newSqeArray, const size_t sqeBytes, const uint8_t* sqeArray, const uint32_t streamId)
{
    CHK_SAFETY_FUNC_RET(memcpy_s(newSqeArray, sqeBytes, sqeArray, sqeBytes));

    // 按需更新DbSqe相关信息
    std::unordered_map<uint32_t, vector<DbSqeTmpInfo>>::iterator mapIter = streamIdToDbSqeTmpInfoMap_.find(streamId);
    if (mapIter != streamIdToDbSqeTmpInfoMap_.end()) {
        // 当前SQE数组可能存在多个DbSqe, 需要逐一更新
        for (size_t i = 0; i < mapIter->second.size(); ++i) {
            // 更新DbSqeLocation (AddWqeArray时已分配并设置相关字段, 此处只需设置sqeArrayIdx)
            const DbSqeTmpInfo& dbSqeTmpInfo = mapIter->second[i];
            CHK_PRT_RET(
                dbSqeTmpInfo.wqeArrayIdx >= wqeTaskArrayInfos_.size(),
                HCCL_ERROR(
                    "[AicpuTaskCacheEntry][AddSqeArray_] streamIdToDbSqeTmpInfoMap_[%u][%u].wqeArrayIdx[%u] >= "
                    "wqeTaskArrayInfos_.size[%u]",
                    streamId, i, dbSqeTmpInfo.wqeArrayIdx, wqeTaskArrayInfos_.size()),
                HCCL_E_INTERNAL);
            DbSqeLocation& dbSqeLocation = wqeTaskArrayInfos_[dbSqeTmpInfo.wqeArrayIdx].dbSqeLocation;
            dbSqeLocation.sqeArrayIdx = static_cast<uint32_t>(sqeArrayInfos_.size());

            // 按需更新dbSqeLocInfoMap_
            if (dbSqeTmpInfo.isReportTask) {
                // dbSqeLocInfoMap_中一定不存在对应的DbSqeProfInfo
                CHK_PRT_RET(
                    dbSqeLocInfoMap_.find(dbSqeLocation) != dbSqeLocInfoMap_.end(),
                    HCCL_ERROR(
                        "[AicpuTaskCacheEntry][AddSqeArray_] DbSqeLocation[%u, %u] already exists in "
                        "dbSqeLocInfoMap_ for streamId[%u]",
                        dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx, streamId),
                    HCCL_E_INTERNAL);

                // 保存到dbSqeLocInfoMap_
                // 注意: dbSqeProfAndRefreshInfo中的src/dstAddrRefreshInfo在SubmitCacheEntry时更新
                DbSqeProfAndRefreshInfo dbSqeProfAndRefreshInfo;
                dbSqeProfAndRefreshInfo.dbSqeProfInfo = dbSqeTmpInfo.dbSqeProfInfo;
                dbSqeProfAndRefreshInfo.wqeArrayIdx
                    = dbSqeTmpInfo.wqeArrayIdx; // 注意: 本函数已校验dbSqeTmpInfo.wqeArrayIdx
                dbSqeLocInfoMap_.emplace(dbSqeLocation, dbSqeProfAndRefreshInfo);
            }

            HCCL_INFO(
                "[AicpuTaskCacheEntry][AddSqeArray_] update dbSqeLocation[%u, %u] isReportTask[%d]",
                dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx, dbSqeTmpInfo.isReportTask);
        }

        // 更新后清理当前SQE数组对应的DbSqe临时信息
        streamIdToDbSqeTmpInfoMap_.erase(mapIter);
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateSqeAddrRefreshInfo_(
    const uint8_t* sqePtr, AddrRefreshInfo& srcAddrRefreshInfo, AddrRefreshInfo& dstAddrRefreshInfo) const
{
    // 参考sqe_build_a5.h, 提取给定SQE的AddrRefreshInfo

    // 提取sqeType，频繁调用私有函数，上层保证指针不为空
    const Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<const Rt91095StarsSqeHeader*>(static_cast<const void*>(sqePtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);

    // 根据sqeType提取AddrRefreshInfo
    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // 无地址字段, 直接跳过
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            const Rt91095StarsMemcpySqe* memcpySqePtr
                = static_cast<const Rt91095StarsMemcpySqe*>(static_cast<const void*>(sqePtr));
            CHK_RET(UpdateAddrRefreshInfo_(
                memcpySqePtr->u.strideMode0.srcAddrLow, memcpySqePtr->u.strideMode0.srcAddrHigh, srcAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            CHK_RET(UpdateAddrRefreshInfo_(
                memcpySqePtr->u.strideMode0.dstAddrLow, memcpySqePtr->u.strideMode0.dstAddrHigh, dstAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE: {
            const Rt91095StarsWriteValueSqe* writeValueSqePtr
                = static_cast<const Rt91095StarsWriteValueSqe*>(static_cast<const void*>(sqePtr));
            CHK_RET(UpdateAddrRefreshInfo_(
                writeValueSqePtr->writeAddrLow, writeValueSqePtr->writeAddrHigh, dstAddrRefreshInfo,
                AddrType::kAddrTypeUnknown));
            break;
        }
        default:
            // sqe_build_a5.h中未使用的SQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][UpdateSqeAddrRefreshInfo_] unexpected sqeType[%u]", sqeType);
            return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateWqeAddrRefreshInfoAndTokenInfo_(
    const WqeTask& wqeTask, AddrRefreshInfo& locAddrRefreshInfo, AddrRefreshInfo& rmtAddrRefreshInfo,
    vector<TokenInfo>& tokenInfos)
{
    // 参考ub_conn_lite.cc, 提取给定WQE的AddrRefreshInfo

    // 提取wqeCode
    // 注意: BatchOneSidedRead/Write只是对multi-slice封装的接口, 最终还是规约到normal read/write
    const UdmaSqeCommon* wqeCommonPtr = static_cast<const UdmaSqeCommon*>(static_cast<const void*>(&wqeTask));
    const uint8_t wqeCode = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内
    switch (wqeCode) {
        case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
            // normal read
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo,
                AddrType::kAddrTypeLocal));
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                AddrType::kAddrTypeRemote));
            break;
        case UdmaSqOpcode::UDMA_OPC_WRITE: { // UdmaSqeWrite
            const uint32_t inlineEn = wqeTask.wqeWrite.comm.inlineEn;
            if (inlineEn != 0) { // inline write
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                    AddrType::kAddrTypeRemote));
            } else { // normal write or write reduce
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo,
                    AddrType::kAddrTypeLocal));
                CHK_RET(UpdateAddrRefreshInfo_(
                    wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo,
                    AddrType::kAddrTypeRemote));
            }
            break;
        }
        case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
            // write with notify (对于给定slice的最后一个UB chunk)
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWriteWithNotify.localU.sge.dataAddrLow, wqeTask.wqeWriteWithNotify.localU.sge.dataAddrHigh,
                locAddrRefreshInfo, AddrType::kAddrTypeLocal));
            CHK_RET(UpdateAddrRefreshInfo_(
                wqeTask.wqeWriteWithNotify.comm.rmtAddrLow, wqeTask.wqeWriteWithNotify.comm.rmtAddrHigh,
                rmtAddrRefreshInfo, AddrType::kAddrTypeRemote));
            break;
        default:
            // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][UpdateWqeAddrRefreshInfo_] unexpected wqeCode[%u]", wqeCode);
            return HCCL_E_INTERNAL;
    }

    CHK_RET(UpdateTokenFlagsByAddrRefreshInfo_(locAddrRefreshInfo, tokenInfos, true));
    CHK_RET(UpdateTokenFlagsByAddrRefreshInfo_(rmtAddrRefreshInfo, tokenInfos, false));

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::UpdateTokenFlagsByAddrRefreshInfo_(
    const AddrRefreshInfo& addrRefreshInfo, vector<TokenInfo>& tokenInfos, bool isLoc) const
{
    if (!addrRefreshInfo.needRefresh) {
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        addrRefreshInfo.memIdx >= tokenInfos.size(),
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][UpdateWqeAddrRefreshInfo_] %s memIdx[%u] >= tokenInfos.size[%u]",
            isLoc ? "loc" : "rmt", addrRefreshInfo.memIdx, tokenInfos.size()),
        HCCL_E_INTERNAL);
    if (isLoc) {
        tokenInfos[addrRefreshInfo.memIdx].needLocTokenIdFlag = true;
    } else {
        tokenInfos[addrRefreshInfo.memIdx].needRmtTokenIdAndValueFlag = true;
    }
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::UpdateAddrRefreshInfo_(
    const uint64_t addr, AddrRefreshInfo& addrRefreshInfo, const AddrType addrType) const
{
    // 默认不是dynamic memory (e.g., user input/output), 认为无需刷新
    addrRefreshInfo.needRefresh = false;

    // 注意: 假设地址vector中前两个为local地址, 后续为remote地址
    constexpr uint32_t kNumLocAddr = 2;
    uint32_t startMemIdx = 0;
    uint32_t endMemIdx = 0;
    if (addrType == AddrType::kAddrTypeUnknown) { // 遍历所有地址信息
        startMemIdx = 0;
        endMemIdx = cachedBaseAddrs_.size();
    } else if (addrType == AddrType::kAddrTypeLocal) { // 遍历local地址信息
        startMemIdx = 0;
        endMemIdx = kNumLocAddr;
    } else { // 遍历remote地址信息
        startMemIdx = kNumLocAddr;
        endMemIdx = cachedBaseAddrs_.size();
    }

    // 检查是否为任意一段dynamic memory
    for (uint32_t memIdx = startMemIdx; memIdx < endMemIdx; memIdx++) {
        // Memory range: [baseAddr, baseAddr + memSize)
        const uint64_t baseAddr = cachedBaseAddrs_[memIdx];
        const uint64_t memSize = cachedMemSizes_[memIdx];

        // 在InitCacheEntry已经做了baseAddr的空指针，及baseAddr + memsize的溢出检查。

        // 检查是否在当前dynamic memory range
        if (AicpuTaskCacheEntry::InRange(baseAddr, memSize, addr)) {
            addrRefreshInfo.needRefresh = true;
            addrRefreshInfo.memIdx = memIdx;
            addrRefreshInfo.offset = addr - baseAddr;
            break;
        }
    }

    return HCCL_SUCCESS;
}

inline bool AicpuTaskCacheEntry::InRange(const uint64_t baseAddr, const uint64_t memSize, const uint64_t addr)
{
    return (addr >= baseAddr && addr < baseAddr + memSize);
}

inline HcclResult AicpuTaskCacheEntry::RefreshSqeTasks_(const SqeArrayInfo& sqeArrayInfo, const uint64_t* baseAddrs)
{
    // sqe数组
    uint8_t* sqeArrayPtr = sqeArrayInfo.sqeArray; // 注意: sqeArrayPtr已在AddSqeArray校验, 无需再校验
    uint64_t sqeCount = sqeArrayInfo.sqeCount;
    RtsqA5* rtsqA5Ptr = sqeArrayInfo.rtsqPtr; // 注意: rtsqPtr已在AddSqeArray校验, 无需再校验
    const vector<AddrRefreshInfo>& sqeSrcAddrRefreshInfoArray = sqeArrayInfo.srcAddrRefreshInfoArray;
    const vector<AddrRefreshInfo>& sqeDstAddrRefreshInfoArray = sqeArrayInfo.dstAddrRefreshInfoArray;
    if (UNLIKELY(isTaskConfigDebug_)) {
        PLF_CONFIG_INFO(
            PLF_TASK, "[AicpuTaskCacheEntry][RefreshSqeTasks_] dump %llu cached SQEs in stream[%u]", sqeCount,
            rtsqA5Ptr->GetStreamId());
    }
    for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
        // 获取当前SQE的信息
        Rt91095StarsSqeHeader* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqeArrayPtr));
        CHK_RET(RefreshOneSqe_(
            sqeArrayPtr, sqeSrcAddrRefreshInfoArray[sqeIdx], sqeDstAddrRefreshInfoArray[sqeIdx], baseAddrs));

        // 刷新streamId, taskId
        rtsqA5Ptr->RefreshSqeHeaderTaskField(sqeHeaderPtr);

        // 切换至下一个SQE
        sqeArrayPtr += AC_SQE_SIZE;
    }

    // 按需打印刷新后的SQE内容
    if (UNLIKELY(isTaskConfigDebug_)) {
        // 循环打印刷新后的SQE内容
        sqeArrayPtr = sqeArrayInfo.sqeArray;
        for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
            // 打印当前SQE
            PLF_CONFIG_INFO(
                PLF_TASK, "[AicpuTaskCacheEntry][RefreshSqeTasks_] %uth cached SQE in stream[%u]", sqeIdx,
                rtsqA5Ptr->GetStreamId());
            CHK_RET(AicpuTaskUtils::DumpSqeContent(sqeArrayPtr));

            // 切换至下一个SQE
            sqeArrayPtr += AC_SQE_SIZE;
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshOneSqe_(
    uint8_t* sqeArrayPtr, const AddrRefreshInfo& srcAddrRefreshInfo, const AddrRefreshInfo& dstAddrRefreshInfo,
    const uint64_t* baseAddrs) const
{
    Rt91095StarsSqeHeader* sqeHeaderPtr = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqeArrayPtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    // 根据SQE type进行对应刷新 (task id始终要刷新; addr相关字段有条件刷新)
    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // 无地址字段, 直接跳过
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            Rt91095StarsMemcpySqe* memcpySqePtr = static_cast<Rt91095StarsMemcpySqe*>(static_cast<void*>(sqeArrayPtr));
            // 刷新地址
            if (srcAddrRefreshInfo.needRefresh) {
                RefreshTaskAddr_(
                    memcpySqePtr->u.strideMode0.srcAddrLow, memcpySqePtr->u.strideMode0.srcAddrHigh, srcAddrRefreshInfo,
                    baseAddrs);
            }
            if (dstAddrRefreshInfo.needRefresh) {
                RefreshTaskAddr_(
                    memcpySqePtr->u.strideMode0.dstAddrLow, memcpySqePtr->u.strideMode0.dstAddrHigh, dstAddrRefreshInfo,
                    baseAddrs);
            }
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE: {
            Rt91095StarsWriteValueSqe* writeValueSqePtr
                = static_cast<Rt91095StarsWriteValueSqe*>(static_cast<void*>(sqeArrayPtr));
            // 刷新地址
            if (dstAddrRefreshInfo.needRefresh) {
                // 注意: writeAddrHigh是位域, 无法直接作为u32&传入
                uint32_t tmpAddrHigh = writeValueSqePtr->writeAddrHigh;
                RefreshTaskAddr_(writeValueSqePtr->writeAddrLow, tmpAddrHigh, dstAddrRefreshInfo, baseAddrs);
                writeValueSqePtr->writeAddrHigh = tmpAddrHigh;
            }
            break;
        }
        default:
            // sqe_build_a5.h中未使用的SQE类型, 告警后报错
            HCCL_ERROR("[AicpuTaskCacheEntry][%s] unexpected sqeType[%u]", __func__, sqeType);
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchSqeTasks_(const SqeArrayInfo& sqeArrayInfo) const
{
    // 注意: rtsqPtr和sqeArray在AddSqeArray时已校验, 这里无需再校验
    RtsqA5* rtsqA5Ptr = sqeArrayInfo.rtsqPtr;
    rtsqA5Ptr->LaunchNewTask(sqeArrayInfo.sqeArray, static_cast<u32>(sqeArrayInfo.sqeCount));
    return HCCL_SUCCESS;
}

// 频繁调用函数，参数外层已经校验，不会失败，无需返回HcclResult让调用方判断。
inline void AicpuTaskCacheEntry::RefreshTaskAddr_(
    uint32_t& addrLow, uint32_t& addrHigh, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const
{
    // 注意: 调用RefreshTaskAddr_时已判断addrRefreshInfo.needRefresh, 无需再次校验

    // 注意: memIdx已经在SubmitCacheEntry的UpdateAddrRefreshInfo_中校验, 确保在cachedMemSizes_范围内

    // 计算newAddr
    const uint64_t newAddr = baseAddrs[addrRefreshInfo.memIdx] + addrRefreshInfo.offset;

    // 返回新地址
    AicpuTaskCacheEntry::SplitUint64ToUint32(newAddr, addrHigh, addrLow);
}

// 频繁调用函数，参数外层已经校验，不会失败，无需返回HcclResult让上层判断。
inline void AicpuTaskCacheEntry::RefreshTaskAddr_(
    uint64_t& addr, const AddrRefreshInfo& addrRefreshInfo, const uint64_t* baseAddrs) const
{
    // 注意: 调用RefreshTaskAddr_时已判断addrRefreshInfo.needRefresh, 无需再次校验

    // 注意: memIdx已经在SubmitCacheEntry的UpdateAddrRefreshInfo_中校验, 确保在cachedMemSizes_范围内

    // 计算newAddr, 返回新地址
    addr = baseAddrs[addrRefreshInfo.memIdx] + addrRefreshInfo.offset;
}

inline void AicpuTaskCacheEntry::DumpWqeTasksHeader_(uint64_t wqeCount, const UbConnLite* ubConnLitePtr) const
{
    PLF_CONFIG_INFO(
        PLF_TASK, "[AicpuTaskCacheEntry][RefreshWqeTasks_] dump %llu cached WQEs in jetty[%u, %u, %u]", wqeCount,
        ubConnLitePtr->GetUbJettyLiteId().GetDieId(), ubConnLitePtr->GetUbJettyLiteId().GetFuncId(),
        ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
}

inline HcclResult
AicpuTaskCacheEntry::DumpWqeTasksPerWqe_(size_t wqeIdx, const WqeTask& wqeTask, const UbConnLite* ubConnLitePtr) const
{
    PLF_CONFIG_INFO(
        PLF_TASK, "[AicpuTaskCacheEntry][%s] %uth cached WQE in jetty[%u, %u, %u]", __func__, wqeIdx,
        ubConnLitePtr->GetUbJettyLiteId().GetDieId(), ubConnLitePtr->GetUbJettyLiteId().GetFuncId(),
        ubConnLitePtr->GetUbJettyLiteId().GetJettyId());
    CHK_RET(AicpuTaskUtils::DumpWqeContent(ReinterpretAs<const uint8_t*>(&wqeTask)));
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeTasks_(
    WqeTaskArrayInfo& wqeTaskArrayInfo, const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes,
    [[maybe_unused]] const uint32_t count)
{
    // 逐个刷新地址
    vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
    const vector<AddrRefreshInfo>& wqeLocAddrRefreshInfoArray = wqeTaskArrayInfo.locAddrRefreshInfoArray;
    const vector<AddrRefreshInfo>& wqeRmtAddrRefreshInfoArray = wqeTaskArrayInfo.rmtAddrRefreshInfoArray;
    UbTransportLiteImpl* ubTransportLitePtr
        = wqeTaskArrayInfo.ubTransportLiteImplPtr; // 注意: 已在AddWqeArray时校验非空
    const uint64_t wqeCount = wqeTasks.size();
    if (UNLIKELY(isTaskConfigDebug_)) {
        DumpWqeTasksHeader_(wqeCount, wqeTaskArrayInfo.ubConnLitePtr);
    }
    // 当前UbTransportLiteImpl在对应内存上, 一定已经提前获取了rmtTokenId/Value
    std::unordered_map<UbTransportLiteImplHandle, vector<TokenInfo>>::const_iterator constIter
        = tokenInfosMap_.find(ubTransportLitePtr);
    CHK_PRT_RET(
        constIter == tokenInfosMap_.end(),
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][RefreshWqeTasks_] ubTransportLitePtr[%p] not found in tokenInfosMap_",
            ubTransportLitePtr),
        HCCL_E_INTERNAL);
    const vector<TokenInfo>& tokenInfos = constIter->second;
    for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
        WqeTask& wqeTask = wqeTasks[wqeIdx];
        const AddrRefreshInfo& locAddrRefreshInfo = wqeLocAddrRefreshInfoArray[wqeIdx];
        const AddrRefreshInfo& rmtAddrRefreshInfo = wqeRmtAddrRefreshInfoArray[wqeIdx];

        // 根据WQE类型刷新对应地址字段及token id/value
        UdmaSqeCommon* wqeCommonPtr = static_cast<UdmaSqeCommon*>(static_cast<void*>(&wqeTask));
        const uint8_t wqeCode
            = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内
        switch (wqeCode) {
            case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
                // normal read
                CHK_RET(RefreshWqeRead_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs, tokenInfos));
                break;
            case UdmaSqOpcode::UDMA_OPC_WRITE: { // UdmaSqeWrite
                CHK_RET(RefreshWqeWrite_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs, tokenInfos));
                break;
            }
            case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
                // write with notify (对于给定slice的最后一个UB chunk)
                CHK_RET(
                    RefreshWqeWriteWithNotify_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs, tokenInfos));
                break;
            default:
                // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
                HCCL_ERROR("[AicpuTaskCacheEntry][%s] unexpected wqeCode[%u]", __func__, wqeCode);
                return HCCL_E_INTERNAL;
        }
        if (UNLIKELY(isTaskConfigDebug_)) {
            CHK_RET(DumpWqeTasksPerWqe_(wqeIdx, wqeTask, wqeTaskArrayInfo.ubConnLitePtr));
        }
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeRead_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs, const vector<TokenInfo>& tokenInfos) const
{
    // 如果需要刷新loc地址信息
    if (locAddrRefreshInfo.needRefresh) {
        // 刷新loc addr
        RefreshTaskAddr_(
            wqeTask.wqeWrite.u.sge.dataAddrLow, wqeTask.wqeWrite.u.sge.dataAddrHigh, locAddrRefreshInfo, baseAddrs);
        // 根据刷新后的loc addr, 刷新loc token id (注意: loc不需要刷新token value)
        CHK_RET(RefreshWqeLocTokenId_(wqeTask.wqeWrite.u.sge.tokenId, locAddrRefreshInfo, tokenInfos));
    }
    // 如果需要刷新rmt地址信息
    if (rmtAddrRefreshInfo.needRefresh) {
        // 刷新rmt addr
        RefreshTaskAddr_(
            wqeTask.wqeWrite.comm.rmtAddrLow, wqeTask.wqeWrite.comm.rmtAddrHigh, rmtAddrRefreshInfo, baseAddrs);

        // 注意: rmtObjId是位域, 不能直接传引用
        uint32_t rmtObjId = wqeTask.wqeWrite.comm.rmtObjId;
        CHK_RET(RefreshWqeRmtTokenIdAndValue_(
            rmtObjId, wqeTask.wqeWrite.comm.rmtTokenValue, rmtAddrRefreshInfo, tokenInfos));
        wqeTask.wqeWrite.comm.rmtObjId = rmtObjId;
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeWrite_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs, const vector<TokenInfo>& tokenInfos)
{
    const uint32_t inlineEn = wqeTask.wqeWrite.comm.inlineEn;
    if (inlineEn != 0) { // inline write
        // 注意: inline write使用WriteWqe实现notify功能 (类似A3使用WriteValue实现notify功能)
        // 因为notify token id/value以及notify addr不会改变, 因此inline write无需刷新WQE
        if (UNLIKELY(rmtAddrRefreshInfo.needRefresh)) {
            // 打印告警信息
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshWqeTasks_] inline write should not refresh rmt addr: "
                "rmtAddrRefreshInfo.needRefresh[%d] rmtAddrRefreshInfo.memIdx[%u]",
                rmtAddrRefreshInfo.needRefresh, rmtAddrRefreshInfo.memIdx);

            // 打印地址信息
            // 注意: memIdx在SubmitCacheEntry中已校验, 这里直接使用
            uint64_t rmtAddr = 0;
            AicpuTaskCacheEntry::CombineUint32ToUint64(
                rmtAddr, wqeTask.wqeWrite.comm.rmtAddrHigh, wqeTask.wqeWrite.comm.rmtAddrLow);
            const uint64_t cachedBaseAddr = cachedBaseAddrs_[rmtAddrRefreshInfo.memIdx];
            const uint64_t cachedMemSize = cachedMemSizes_[rmtAddrRefreshInfo.memIdx];
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][RefreshWqeTasks_] rmtAddr[0x%016llx] "
                "cachedBaseAddr[0x%016llx] endAddr[0x%016llx] memSize[%llu]",
                rmtAddr, cachedBaseAddr, cachedBaseAddr + cachedMemSize, cachedMemSize);

            return HCCL_E_INTERNAL;
        }
        return HCCL_SUCCESS;
    }
    // normal write or write reduce
    return RefreshWqeRead_(wqeTask, locAddrRefreshInfo, rmtAddrRefreshInfo, baseAddrs, tokenInfos);
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeWriteWithNotify_(
    WqeTask& wqeTask, const AddrRefreshInfo& locAddrRefreshInfo, const AddrRefreshInfo& rmtAddrRefreshInfo,
    const uint64_t* baseAddrs, const vector<TokenInfo>& tokenInfos) const
{
    // 如果需要刷新loc地址信息
    if (locAddrRefreshInfo.needRefresh) {
        // 刷新loc addr
        RefreshTaskAddr_(
            wqeTask.wqeWriteWithNotify.localU.sge.dataAddrLow, wqeTask.wqeWriteWithNotify.localU.sge.dataAddrHigh,
            locAddrRefreshInfo, baseAddrs);
        // 根据刷新后的loc addr, 刷新loc token id (注意: loc不需要刷新token value)
        CHK_RET(RefreshWqeLocTokenId_(wqeTask.wqeWriteWithNotify.localU.sge.tokenId, locAddrRefreshInfo, tokenInfos));
    }
    // 如果需要刷新rmt地址信息
    if (rmtAddrRefreshInfo.needRefresh) {
        // 刷新rmt addr
        RefreshTaskAddr_(
            wqeTask.wqeWriteWithNotify.comm.rmtAddrLow, wqeTask.wqeWriteWithNotify.comm.rmtAddrHigh, rmtAddrRefreshInfo,
            baseAddrs);

        // 注意: rmtObjId是位域, 不能直接传引用
        uint32_t rmtObjId = wqeTask.wqeWriteWithNotify.comm.rmtObjId;
        CHK_RET(RefreshWqeRmtTokenIdAndValue_(
            rmtObjId, wqeTask.wqeWriteWithNotify.comm.rmtTokenValue, rmtAddrRefreshInfo, tokenInfos));
        wqeTask.wqeWriteWithNotify.comm.rmtObjId = rmtObjId;
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeLocTokenId_(
    uint32_t& tokenId, const AddrRefreshInfo& addrRefreshInfo, const vector<TokenInfo>& tokenInfos) const
{
    // 调用方保证addrRefreshInfo.needRefresh为true

    // 当前UbTransportLiteImpl在对应内存上, 一定已经提前获取了locTokenId
    // 注意: ubTransportLitePtr已在AddWqeArray时校验非空, 这里无需再校验
    const TokenInfo& tokenInfo = tokenInfos[addrRefreshInfo.memIdx];

    CHK_PRT_RET(
        !tokenInfo.needLocTokenIdFlag,
        HCCL_ERROR("[AicpuTaskCacheEntry][RefreshWqeLocTokenId_] needLocTokenIdFlag is false"), HCCL_E_INTERNAL);

    // 刷新loc token id
    tokenId = tokenInfo.locTokenId;

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshWqeRmtTokenIdAndValue_(
    uint32_t& tokenId, uint32_t& tokenValue, const AddrRefreshInfo& addrRefreshInfo,
    const vector<TokenInfo>& tokenInfos) const
{
    // 调用方保证addrRefreshInfo.needRefresh为true

    // 当前UbTransportLiteImpl在对应内存上, 一定已经提前获取了rmtTokenId/Value
    const TokenInfo& tokenInfo = tokenInfos[addrRefreshInfo.memIdx];

    CHK_PRT_RET(
        !tokenInfo.needRmtTokenIdAndValueFlag,
        HCCL_ERROR("[AicpuTaskCacheEntry][RefreshWqeRmtTokenIdAndValue_] needRmtTokenIdAndValueFlag is false"),
        HCCL_E_INTERNAL);

    // 刷新remote token id/value
    tokenId = tokenInfo.rmtTokenId;
    tokenValue = tokenInfo.rmtTokenValue;

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::LaunchWqeTasks_(WqeTaskArrayInfo& wqeTaskArrayInfo) const
{
    // 逐个下发WQE
    vector<WqeTask>& wqeTasks = wqeTaskArrayInfo.wqeTaskArray;
    UbConnLite* ubConnLitePtr = wqeTaskArrayInfo.ubConnLitePtr; // 注意: AddWqeArray时已校验非空
    const size_t wqeCount = wqeTasks.size();
    for (size_t wqeIdx = 0; wqeIdx < wqeCount; wqeIdx++) {
        WqeTask& wqeTask = wqeTasks[wqeIdx];

        // 根据WQE类型下发 (下发过程中会更新ubConnLitePtr中的pi)
        UdmaSqeCommon* wqeCommonPtr = static_cast<UdmaSqeCommon*>(static_cast<void*>(&wqeTask));
        const uint8_t wqeCode
            = static_cast<uint8_t>(wqeCommonPtr->opcode); // opcode来自于UdmaSqOpcode, 一定在uint8范围内
        switch (wqeCode) {
            case UdmaSqOpcode::UDMA_OPC_READ: // UdmaSqeWrite
                // normal read
                ubConnLitePtr->LaunchOneWqe(&wqeTask.wqeWrite, UdmaSqOpcode::UDMA_OPC_READ);
                break;
            case UdmaSqOpcode::UDMA_OPC_WRITE: // UdmaSqeWrite
                // inline write, normal write, or write reduce
                ubConnLitePtr->LaunchOneWqe(&wqeTask.wqeWrite, UdmaSqOpcode::UDMA_OPC_WRITE);
                break;
            case WRITE_WITH_NOTIFY_OPCODE: // UdmaSqeWriteWithNotify
                // write with notify (对于给定slice的最后一个UB chunk)
                ubConnLitePtr->LaunchOneWqeWithNotify(&wqeTask.wqeWriteWithNotify, WRITE_WITH_NOTIFY_OPCODE);
                break;
            default:
                // ub_conn_lite.cc中未使用的WQE类型, 告警后报错
                HCCL_ERROR("[AicpuTaskCacheEntry][LaunchWqeTasks_] unexpected wqeCode[%u]", wqeCode);
                return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshDbSqe_(WqeTaskArrayInfo& wqeTaskArrayInfo)
{
    // 获取pi
    UbConnLite* ubConnLitePtr = wqeTaskArrayInfo.ubConnLitePtr; // 注意: AddWqeArray时已校验非空
    const uint16_t pi = ubConnLitePtr->GetPi();

    // 校验dbSqeLocation, AddSqeArray_中已经校验
    const DbSqeLocation& dbSqeLocation = wqeTaskArrayInfo.dbSqeLocation;

    // 校验SQE type
    uint8_t* sqePtr = sqeArrayInfos_[dbSqeLocation.sqeArrayIdx].sqeArray + dbSqeLocation.dbSqeIdx * AC_SQE_SIZE;

    // 更新SQE pi value
    Rt91095StarsUbdmaDBmodeSqe* dbSqePtr = static_cast<Rt91095StarsUbdmaDBmodeSqe*>(static_cast<void*>(sqePtr));
    dbSqePtr->piValue1 = pi;

    // 注意: UbTransportLiteImpl只针对WQE按需填充DfxTaskInfo上报profiling, DB SQE无需上报profiling

    HCCL_INFO(
        "[AicpuTaskCacheEntry][RefreshDbSqe_] update pi[%u] at dbSqeLocation[%u, %u]", pi, dbSqeLocation.sqeArrayIdx,
        dbSqeLocation.dbSqeIdx);

    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportDbSqeProfiling_(
    uint8_t* dbSqePtr, size_t arrayIdx, uint32_t dbSqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
    const uint32_t count, StreamLite* streamLite, const u32 sqId, const u32 taskId)
{
    // 注意: 参考ub_transport_lite_impl.cc填充DfxTaskInfo并经NextTaskSlot上报
    DbSqeLocation dbSqeLocation;
    dbSqeLocation.sqeArrayIdx = arrayIdx;
    dbSqeLocation.dbSqeIdx = dbSqeIdx;
    auto it = dbSqeLocInfoMap_.find(dbSqeLocation);

    // 注意: 少数场景下DbSqe不需要上报, 例如Drain, BatchOneSidedRead, BatchOneSidedWrite
    if (UNLIKELY(it == dbSqeLocInfoMap_.end())) {
        HCCL_INFO(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] dbSqeLocInfoMap_[%u, %u] is not found",
            dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx);
        return HCCL_SUCCESS;
    }

    CHK_PRT_RET(
        it->second.dbSqeProfInfo.isValid == false,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] dbSqeLocInfoMap_[%u, %u] is invalid",
            dbSqeLocation.sqeArrayIdx, dbSqeLocation.dbSqeIdx),
        HCCL_E_INTERNAL);

    // 校验SQE type
    Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(dbSqePtr)); // 已在AddSqeArray校验, 无需再校验
    CHK_PRT_RET(
        static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type) != Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA,
        HCCL_ERROR(
            "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] sqeHeaderPtr->type[%u] is not RT_91095_SQE_TYPE_UBDMA",
            sqeHeaderPtr->type),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] report %lluth DbSqe profiling in %lluth SQE array: "
        "dbSqeType[%u], taskId[%u], sqId[%u]",
        dbSqeIdx, arrayIdx, sqeHeaderPtr->type, taskId, sqId);

    CHK_RET(RefreshDbSqeProfAddrs_(it->second, baseAddrs, memSizes, count));

    // 从DbSqeProfAndRefreshInfo中获取DbSqe对应的UbTransportLiteImpl
    // 注意: wqeArrayIdx已在AddSqeArray_校验, 无需再校验
    const uint32_t wqeArrayIdx = it->second.wqeArrayIdx;
    // 注意: ubTransportLiteImplPtr已在AddWqeArray校验, 无需再校验
    UbTransportLiteImpl* ubTransportLitePtr = wqeTaskArrayInfos_[wqeArrayIdx].ubTransportLiteImplPtr;

    Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
    switch (static_cast<u8>(it->second.dbSqeProfInfo.taskParamType)) {
        case TaskParamTypeVal::TASK_UB_INLINE_WRITE:
        case TaskParamTypeVal::TASK_UB:
        case TaskParamTypeVal::TASK_WRITE_WITH_NOTIFY:
            CHK_RET(FillSlotUbDma_(slot, dbSqePtr, it->second, ubTransportLitePtr, streamLite, taskId));
            break;
        case TaskParamTypeVal::TASK_UB_REDUCE_INLINE:
        case TaskParamTypeVal::TASK_WRITE_REDUCE_WITH_NOTIFY:
            CHK_RET(FillSlotReduce_(slot, dbSqePtr, it->second, ubTransportLitePtr, streamLite, taskId));
            break;
        default:
            HCCL_ERROR(
                "[AicpuTaskCacheEntry][ReportDbSqeProfiling_] taskType[%u] is unsupported",
                it->second.dbSqeProfInfo.taskParamType);
            return HCCL_E_INTERNAL;
    }

    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::RefreshDbSqeProfAddrs_(
    DbSqeProfAndRefreshInfo& profAndRefreshInfo, const uint64_t* baseAddrs, [[maybe_unused]] const uint64_t* memSizes,
    [[maybe_unused]] const uint32_t count) const
{
    // 注意: dbSqeProfInfo中的src/dstAddr, 需要根据DbSqeProfAndRefreshInfo中的src/dstAddrRefreshInfo进行刷新,
    // 才能填充DfxTaskInfo
    if (profAndRefreshInfo.locAddrRefreshInfo.needRefresh) {
        RefreshTaskAddr_(profAndRefreshInfo.dbSqeProfInfo.locAddr, profAndRefreshInfo.locAddrRefreshInfo, baseAddrs);
    }
    if (profAndRefreshInfo.rmtAddrRefreshInfo.needRefresh) {
        RefreshTaskAddr_(profAndRefreshInfo.dbSqeProfInfo.rmtAddr, profAndRefreshInfo.rmtAddrRefreshInfo, baseAddrs);
    }
    return HCCL_SUCCESS;
}

inline void AicpuTaskCacheEntry::FillSlotCommonFields_(
    Hccl::DfxTaskInfo* slot, const StreamLite* streamLite, u32 taskId, u8 linkType, u8 transportType,
    u64 channelHandle) const
{
    slot->sqId = streamLite->GetSqId();
    slot->taskId = taskId;
    const void* opInfo = streamLite->GetLatestDfxOpInfo();
    slot->dfxOpInfo = (opInfo != nullptr) ? ReinterpretAs<u64>(opInfo) : DFX_INVALID_U64;
    slot->linkType = linkType;
    slot->transportType = transportType;
    slot->channelHandle = channelHandle;
}

inline u8 AicpuTaskCacheEntry::GetUbLinkTypeVal_(const UbTransportLiteImpl* ubTransportLiteImplPtr) const
{
    return (ubTransportLiteImplPtr->linkType_ == Hccl::DfxLinkType::UB) ? Hccl::DfxLinkTypeVal::LINK_UB :
                                                                          Hccl::DfxLinkTypeVal::LINK_UBoE;
}

inline u8 AicpuTaskCacheEntry::ConvertSdmaOpCodeToReduceOp_(uint8_t opcode) const
{
    // opcode 低4位为 RtStarsMemcpyAsyncOperationKind: ADD=0x01, MAX=0x02, MIN=0x03
    // 映射到 HcclReduceOp: SUM=0, MAX=2, MIN=3
    const uint8_t opKind = opcode & 0x0F;
    switch (opKind) {
        case 0x01:
            return static_cast<u8>(HCCL_REDUCE_SUM);
        case 0x02:
            return static_cast<u8>(HCCL_REDUCE_MAX);
        case 0x03:
            return static_cast<u8>(HCCL_REDUCE_MIN);
        default:
            return static_cast<u8>(HCCL_REDUCE_RESERVED);
    }
}

inline HcclResult AicpuTaskCacheEntry::FillSlotUbDma_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
    UbTransportLiteImpl* ubTransportLiteImplPtr, StreamLite* streamLite, u32 taskId) const
{
    const DbSqeProfInfo& profInfo = profAndRefreshInfo.dbSqeProfInfo;
    slot->taskType = static_cast<u8>(profInfo.taskParamType);
    FillSlotCommonFields_(
        slot, streamLite, taskId, GetUbLinkTypeVal_(ubTransportLiteImplPtr),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_UB), ReinterpretAs<u64>(ubTransportLiteImplPtr));
    slot->taskPara.ubDma.sqeAddr = ReinterpretAs<u64>(sqePtr);
    slot->taskPara.ubDma.srcAddr = profInfo.locAddr;
    slot->taskPara.ubDma.dstAddr = profInfo.rmtAddr;
    slot->taskPara.ubDma.size = profInfo.size;
    slot->taskPara.ubDma.notifyId = profInfo.notifyId;
    slot->taskPara.ubDma.jettyHandle = profInfo.jettyHandle;
    slot->taskPara.ubDma.jettyId = profInfo.jettyId;
    slot->taskPara.ubDma.tpn = profInfo.tpn;
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotReduce_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, const DbSqeProfAndRefreshInfo& profAndRefreshInfo,
    UbTransportLiteImpl* ubTransportLiteImplPtr, StreamLite* streamLite, u32 taskId) const
{
    const DbSqeProfInfo& profInfo = profAndRefreshInfo.dbSqeProfInfo;
    slot->taskType = static_cast<u8>(profInfo.taskParamType);
    FillSlotCommonFields_(
        slot, streamLite, taskId, GetUbLinkTypeVal_(ubTransportLiteImplPtr),
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_UB), ReinterpretAs<u64>(ubTransportLiteImplPtr));
    slot->taskPara.Reduce.sqeAddr = ReinterpretAs<u64>(sqePtr);
    slot->taskPara.Reduce.srcAddr = profInfo.locAddr;
    slot->taskPara.Reduce.dstAddr = profInfo.rmtAddr;
    slot->taskPara.Reduce.size = profInfo.size;
    slot->taskPara.Reduce.reduceOp = static_cast<u8>(profInfo.reduceOp);
    slot->taskPara.Reduce.notifyId = profInfo.notifyId;
    slot->taskPara.Reduce.jettyHandle = profInfo.jettyHandle;
    slot->taskPara.Reduce.jettyId = profInfo.jettyId;
    slot->taskPara.Reduce.tpn = profInfo.tpn;
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportSqeProfiling_(
    uint8_t* sqePtr, size_t arrayIdx, uint32_t sqeIdx, const uint64_t* baseAddrs, const uint64_t* memSizes,
    const uint32_t count, StreamLite* streamLite, const u32 sqId)
{
    // 注意: 参考aicpu_ts_thread.cc填充DfxTaskInfo并经NextTaskSlot上报

    // 获取SQE对应的sqeType和taskId
    Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<Rt91095StarsSqeHeader*>(static_cast<void*>(sqePtr)); // 已在AddSqeArray校验, 无需再校验
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    const u32 taskId = (sqeHeaderPtr->taskId << 16) | (sqeHeaderPtr->rtStreamId);

    switch (sqeType) {
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA:
            // DbSqe由ReportDbSqeProfiling_内部按需获取NextTaskSlot并填充
            CHK_RET(
                ReportDbSqeProfiling_(sqePtr, arrayIdx, sqeIdx, baseAddrs, memSizes, count, streamLite, sqId, taskId));
            break;
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD:
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT: {
            Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
            CHK_RET(FillSlotNotify_(slot, sqePtr, streamLite, taskId));
            break;
        }
        case Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA: {
            Hccl::DfxTaskInfo* slot = streamLite->NextTaskSlot();
            CHK_RET(FillSlotSdma_(slot, sqePtr, streamLite, taskId));
            break;
        }
        default:
            HCCL_ERROR("[AicpuTaskCacheEntry][ReportSqeProfiling_] sqeType[%u] is unsupported", sqeType);
            return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotNotify_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, StreamLite* streamLite, u32 taskId) const
{
    const Rt91095StarsSqeHeader* sqeHeaderPtr
        = static_cast<const Rt91095StarsSqeHeader*>(static_cast<const void*>(sqePtr));
    const Rt91095StarsSqeType sqeType = static_cast<Rt91095StarsSqeType>(sqeHeaderPtr->type);
    slot->taskType = static_cast<u8>(
        (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD) ? Hccl::TaskParamTypeVal::TASK_NOTIFY_RECORD :
                                                                            Hccl::TaskParamTypeVal::TASK_NOTIFY_WAIT);
    FillSlotCommonFields_(
        slot, streamLite, taskId, Hccl::DfxLinkTypeVal::LINK_ONCHIP,
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL), DFX_INVALID_U64);
    slot->taskPara.Notify.sqeAddr = ReinterpretAs<u64>(sqePtr);
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

inline HcclResult AicpuTaskCacheEntry::FillSlotSdma_(
    Hccl::DfxTaskInfo* slot, const uint8_t* sqePtr, StreamLite* streamLite, u32 taskId) const
{
    const Hccl::Rt91095StarsMemcpySqe* sdmaSqe
        = static_cast<const Hccl::Rt91095StarsMemcpySqe*>(static_cast<const void*>(sqePtr));
    FillSlotCommonFields_(
        slot, streamLite, taskId, Hccl::DfxLinkTypeVal::LINK_ONCHIP,
        static_cast<u8>(Hccl::DfxTransportType::DFX_TRANSPORT_TYPE_LOCAL), DFX_INVALID_U64);
    if (sdmaSqe->opcode == SDMA_OPCODE_MEMCPY) {
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_SDMA);
        slot->taskPara.Dma.sqeAddr = ReinterpretAs<u64>(sqePtr);
    } else {
        constexpr uint32_t UINT32_BIT_WIDTH = 32;
        slot->taskType = static_cast<u8>(Hccl::TaskParamTypeVal::TASK_REDUCE_INLINE);
        slot->taskPara.Reduce.sqeAddr = ReinterpretAs<u64>(sqePtr);
        slot->taskPara.Reduce.srcAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.srcAddrHigh) << UINT32_BIT_WIDTH)
                                        | sdmaSqe->u.strideMode0.srcAddrLow;
        slot->taskPara.Reduce.dstAddr = (static_cast<uint64_t>(sdmaSqe->u.strideMode0.dstAddrHigh) << UINT32_BIT_WIDTH)
                                        | sdmaSqe->u.strideMode0.dstAddrLow;
        slot->taskPara.Reduce.size = sdmaSqe->u.strideMode0.lengthMove;
        slot->taskPara.Reduce.notifyId = INVALID_U32;
        slot->taskPara.Reduce.reduceOp = ConvertSdmaOpCodeToReduceOp_(sdmaSqe->opcode);
    }
    PLF_CONFIG_INFO(PLF_TASK, "[%s] %s", __func__, slot->Describe().c_str());
    return HCCL_SUCCESS;
}

HcclResult AicpuTaskCacheEntry::ReportSqeArrayProfiling_(
    size_t arrayIdx, const uint64_t* baseAddrs, const uint64_t* memSizes, const uint32_t count)
{
    // 注意: arrayIdx已在SubmitCacheEntry_校验, 这里无需重复校验
    const SqeArrayInfo& sqeArrayInfo = sqeArrayInfos_[arrayIdx];

    // sqe数组
    uint8_t* sqePtr = sqeArrayInfo.sqeArray; // 注意: sqePtr已在AddSqeArray校验, 无需再校验
    uint64_t sqeCount = sqeArrayInfo.sqeCount;

    // 获取SQE对应的sqId
    // 注意: aicpuTsThreadPtr已在AddSqeArray校验, 无需再校验
    StreamLite* streamLite = ReinterpretAs<StreamLite*>(sqeArrayInfo.aicpuTsThreadPtr->GetStreamLitePtr());
    CHK_PTR_NULL(streamLite);
    const u32 sqId = streamLite->GetSqId();
    for (size_t sqeIdx = 0; sqeIdx < sqeCount; ++sqeIdx) {
        CHK_PRT(ReportSqeProfiling_(sqePtr, arrayIdx, sqeIdx, baseAddrs, memSizes, count, streamLite, sqId));

        // 切换到下一个SQE
        sqePtr += AC_SQE_SIZE;
    }

    return HCCL_SUCCESS;
}

} // namespace hcomm
