/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <array>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <string>
#include <thread>
#include <atomic>
#include <mutex>
#include <cstring>
#include <sstream>
#include <utility>
#include <nlohmann_json/json.hpp>
#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/validation_issue_recorder.h"
#include "dump_v3/dump_v3_manager.h"
#include "setting_manager.h"
#include "storage_manager.h"
#include "task_utils.h"
#include "checker.h"
#include "framework/big_graph_check/big_graph_checker.h"
#include "hccl_verifier.h"
#include "sim_log.h"
#include "ccu_all_rank_param_recorder.h"
#include "framework/task_graph_generator_v3/ccu_graph_generator_v3/ccu_all_rank_param_recorder_v3.h"
#include "mem_conflict_check_utils.h"
#include "sim_loader.h"
#include "sim_common_defs.h"
#include "utils/check_utils.h"
#include "utils/error_codes.h"

using json = nlohmann::json;
loader::Loader g_loader;

// 使用原子变量控制程序生命周期
std::atomic<bool> g_keep_running{true};
std::mutex g_run_checker_mutex;
std::mutex g_worker_mutex;
std::thread g_worker_thread;

enum class CheckerStatus : uint8_t { SUCCESS, FAILED, DISABLE, NOT_EXECUTED };
using CheckerResult = std::array<CheckerStatus, 2>;
static constexpr size_t OLD_CHECKER_RESULT = 0;
static constexpr size_t NEW_CHECKER_RESULT = 1;
static constexpr const char* CHECKER_STATUS_TEXT[] = {"success", "failed", "disable", "not_executed"};

static std::vector<std::map<uint32_t, sim::CompositeOpDetail>>
TransposeCompositeOpMap(const std::map<uint32_t, std::vector<sim::CompositeOpDetail>>& compositeDataMap)
{
    size_t maxOps = 0;
    for (const auto& entry : compositeDataMap) {
        maxOps = std::max(maxOps, entry.second.size());
    }
    std::vector<std::map<uint32_t, sim::CompositeOpDetail>> opGroups(maxOps);
    for (const auto& entry : compositeDataMap) {
        uint32_t rankId = entry.first;
        const auto& ops = entry.second;
        for (size_t i = 0; i < ops.size(); i++) {
            opGroups[i][rankId] = ops[i];
        }
    }
    return opGroups;
}

json BuildOpParamSummaryJson(const HcclSim::CheckerParam& param);

static bool IsAivOpExpansionMode(uint32_t opExpansionMode)
{
    constexpr uint32_t SIM_OP_EXPANSION_MODE_AIV = 2U;
    return opExpansionMode == SIM_OP_EXPANSION_MODE_AIV;
}

static bool HasAivGraphTask(const std::vector<std::vector<sim::OpTaskTab>>& allTasks)
{
    for (const auto& rankTasks : allTasks) {
        for (const auto& task : rankTasks) {
            if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
                continue;
            }
            HcclTaskMetaData metaData;
            std::memcpy(&metaData, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
            if (metaData.taskType == HccLTaskMetaType::AIV_GRAPH) {
                return true;
            }
        }
    }
    return false;
}

static bool IsSingleRankWithNoTask(const std::map<uint32_t, sim::CompositeOpDetail>& opGroup)
{
    if (opGroup.size() != 1) {
        return false;
    }

    const sim::CompositeOpDetail& op = opGroup.begin()->second;
    return op.detail.rankSize == 1 && op.tasks.empty();
}

static bool IsSingleRankWithNoTask(const HcclSim::BigGraphCheckV3::BigGraphData& data)
{
    if (data.operators.empty()) {
        return false;
    }

    for (const HcclSim::BigGraphCheckV3::OpParam& opParam : data.operators) {
        if (opParam.ranks.size() != 1) {
            return false;
        }
        const HcclSim::BigGraphCheckV3::OperatorRankData& rankData = opParam.ranks.front();
        if (rankData.op.detail.rankSize != 1 || !rankData.taskMetas.empty()) {
            return false;
        }
    }
    return true;
}

static const char* HcclReduceOpToString(HcclReduceOp t)
{
    switch (t) {
        case HCCL_REDUCE_SUM:
            return "SUM";
        case HCCL_REDUCE_PROD:
            return "PROD";
        case HCCL_REDUCE_MAX:
            return "MAX";
        case HCCL_REDUCE_MIN:
            return "MIN";
        default:
            return "Unknown";
    }
}

static void AppendApplicableRoleFields(std::ostringstream& os, const HcclSim::CheckerParam& param)
{
    switch (param.cmdType) {
        case HCCL_CMD_SEND:
        case HCCL_CMD_RECEIVE:
            os << ", sourceRank=" << param.srcRank << ", targetRank=" << param.dstRank;
            break;
        case HCCL_CMD_BROADCAST:
        case HCCL_CMD_REDUCE:
        case HCCL_CMD_SCATTER:
            os << ", rootRank=" << param.root;
            break;
        default:
            break;
    }
}

static HcclResult LoadCheckerDataBase(
    std::vector<sim::CcuChannelTab>& channels, std::vector<sim::CcuInstrResTab>& instrRes,
    std::vector<sim::SyncRecordTab>& syncRecords, uint32_t& syncIterMaxNum)
{
    HcclResult ret = g_loader.GetCcuChannelInfo(channels);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU channel information",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    ret = g_loader.GetInstrResInfo(instrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load CCU instruction resource information",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    ret = g_loader.GetSyncInfo(syncRecords);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load sync records", HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }

    if (syncRecords.empty()) {
        HCCL_VM_WARN("No sync records were found");
        return HcclResult::HCCL_E_PARA;
    }

    std::sort(syncRecords.begin(), syncRecords.end(), [](const sim::SyncRecordTab& a, const sim::SyncRecordTab& b) {
        return a.syncIter < b.syncIter;
    });

    syncIterMaxNum = syncRecords.back().syncIter;
    HCCL_VM_INFO(
        "Checker database loaded, channelCount={}, instrResCount={}, syncRecordCount={}, "
        "syncIterMaxNum={}",
        channels.size(), instrRes.size(), syncRecords.size(), syncIterMaxNum);
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult DispatchCheckByCmdType(HcclSim::AllRankTaskQueues& taskQueues, HcclSim::CheckerParam& param)
{
    HcclCMDType cmdType = param.cmdType;
    switch (cmdType) {
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return CheckAllReduce(taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType);
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            return CheckAllGather(taskQueues, param.rankSize, param.dataType, param.dataCount);
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return CheckReduceScatter(taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType);
        case HcclCMDType::HCCL_CMD_SEND:
            return CheckSend(taskQueues, param.rankSize, param.dataType, param.dataCount, param.srcRank, param.dstRank);
        case HcclCMDType::HCCL_CMD_RECEIVE:
            return CheckRecv(taskQueues, param.rankSize, param.dataType, param.dataCount, param.srcRank, param.dstRank);
        case HcclCMDType::HCCL_CMD_BROADCAST:
            return CheckBroadcast(taskQueues, param.rankSize, param.dataType, param.dataCount, param.root);
        case HcclCMDType::HCCL_CMD_REDUCE:
            return CheckReduce(
                taskQueues, param.rankSize, param.dataType, param.dataCount, param.reduceType, param.root);
        case HcclCMDType::HCCL_CMD_SCATTER:
            return CheckScatter(taskQueues, param.rankSize, param.dataType, param.dataCount, param.root);
        case HcclCMDType::HCCL_CMD_BATCH_SEND_RECV:
            return CheckBatchSendRecv(taskQueues, param.rankSize, param.dataType, param.dataCount);
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
            return CheckAllGatherV(taskQueues, param.rankSize, param.vDataDes);
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
            return CheckReduceScatterV(taskQueues, param.rankSize, param.reduceType, param.vDataDes);
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            return CheckAll2All(
                taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType),
                param.all2AllDataDes.sendCount);
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            return CheckAll2AllVC(
                taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType),
                param.all2AllDataDes.sendCountMatrix);
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            HCCL_VM_WARN("Checker does not support AllToAllV and will use the AllToAllVC validation path");
            return CheckAll2AllVC(
                taskQueues, param.rankSize, static_cast<HcclDataType>(param.all2AllDataDes.sendType),
                param.all2AllDataDes.sendCountMatrix);
        default:
            HCCL_VM_ERROR(
                "{} Unsupported collective type, collectiveTypeCode={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), static_cast<u32>(cmdType));
            return HcclResult::HCCL_E_NOT_SUPPORT;
    }
}

static HcclResult LoadOpDataForOneRank(
    HcclSim::StorageManager& storage, std::vector<sim::CcuChannelTab>& channels,
    std::vector<sim::CcuInstrResTab>& instrRes, uint32_t rankId, sim::CompositeOpDetail& op)
{
    HcclResult ret = storage.LoadHcclVmSynthesisData(rankId, op.memInfo, channels);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load synthesized memory information for this rank, rankId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), rankId);
        return ret;
    }

    ret = storage.LoadHcclVmInstrData(instrRes);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load instruction data for this rank, rankId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), rankId);
        return ret;
    }

    if (op.detail.opDetail.size() < sizeof(::OpDetails)) {
        HCCL_VM_ERROR(
            "{} Op detail payload is too small to parse, rankId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), rankId);
        return HcclResult::HCCL_E_PARA;
    }

    ::OpDetails opDetails{};
    std::memcpy(&opDetails, op.detail.opDetail.data(), sizeof(::OpDetails));
    ret = storage.Trans2CheckerParam(op.detail, opDetails);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to convert this rank into checker input parameters, rankId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), rankId);
        return ret;
    }
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult ProcessOneOpGroup(
    HcclSim::StorageManager& storage, std::vector<sim::CcuChannelTab>& channels,
    std::vector<sim::CcuInstrResTab>& instrRes, uint32_t opIdx, std::map<uint32_t, sim::CompositeOpDetail>& opGroup,
    CheckerResult& checkerResult)
{
    HcclSim::ValidationIssueRecorder::GetInstance().Reset();
    HcclSim::AllRankParamRecorder::Global()->Reset();
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    HcclSim::g_ccuGraphTaskOri2New.clear();
    HcclSim::DumpManager& dumpManager = HcclSim::DumpManager::GetInstance();
    HcclSim::SettingManager& settingManager = HcclSim::SettingManager::GetInstance();
    bool enableNewChecker = settingManager.IsNewCheckerEnabled();
    bool enableOldChecker = settingManager.IsOldCheckerEnabled();
    bool usesAivExpansionMode = false;
    checkerResult[OLD_CHECKER_RESULT] = enableOldChecker ? CheckerStatus::NOT_EXECUTED : CheckerStatus::DISABLE;
    checkerResult[NEW_CHECKER_RESULT] = enableNewChecker ? CheckerStatus::NOT_EXECUTED : CheckerStatus::DISABLE;

    if (IsSingleRankWithNoTask(opGroup)) {
        checkerResult[OLD_CHECKER_RESULT] = enableOldChecker ? CheckerStatus::SUCCESS : CheckerStatus::DISABLE;
        checkerResult[NEW_CHECKER_RESULT] = enableNewChecker ? CheckerStatus::SUCCESS : CheckerStatus::DISABLE;
        HCCL_VM_WARN(
            "Single-op check is skipped and treated as success because this is a single-rank "
            "operation with no tasks, opIndex={}",
            opIdx);
        return HcclResult::HCCL_SUCCESS;
    }

    HCCL_VM_INFO("Start checking one op group, opGroupSize={}", opGroup.size());
    storage.BeginOpGroup();
    std::vector<std::vector<sim::OpTaskTab>> allTasks;
    for (auto& entry : opGroup) {
        uint32_t rankId = entry.first;
        HCCL_VM_INFO("Load one rank from this op group, rankId={}", rankId);
        sim::CompositeOpDetail& op = entry.second;
        usesAivExpansionMode = usesAivExpansionMode || IsAivOpExpansionMode(op.detail.opExpansionMode);
        allTasks.push_back(op.tasks);
        HcclResult ret = LoadOpDataForOneRank(storage, channels, instrRes, rankId, op);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to load one rank from this op group, opIndex={}, rankId={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), opIdx, rankId);
            return ret;
        }
    }

    if (!enableNewChecker && !enableOldChecker) {
        HCCL_VM_ERROR(
            "{} This op is skipped because both the new checker and the old checker are disabled, "
            "opIndex={}, newCheckerEnabled={}, oldCheckerEnabled={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::SETTING_WARNING), opIdx, enableNewChecker, enableOldChecker);
        return HcclResult::HCCL_SUCCESS;
    }

    HcclResult ret = storage.FinalizeOpGroup();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to finalize operator parameters for this op group, opIndex={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), opIdx);
        return ret;
    }

    // AllToAll family continues to use the existing matrix merge path.
    storage.MergeAll2AllVSendCountMatrix();

    if (dumpManager.IsEnabled()) {
        HcclSim::DumpRunManifest::GetInstance().SetOpParam(BuildOpParamSummaryJson(storage.GetCheckerParam()));
    }

    ret = storage.LoadHcclVmTaskMetaData(allTasks);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load V3 task metadata for this op group",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR));
        return ret;
    }
    const bool hasAivGraphTask = HasAivGraphTask(allTasks);
    const bool isAivOp = usesAivExpansionMode || hasAivGraphTask;
    if (isAivOp) {
        if (!enableNewChecker) {
            HCCL_VM_ERROR(
                "{} This AIV op requires the V3 checker, but V3 is disabled by configuration, "
                "opIndex={}, action=abort, newCheckerEnabled={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::SETTING_WARNING), opIdx, enableNewChecker);
            return HcclResult::HCCL_E_NOT_SUPPORT;
        }
        if (enableOldChecker) {
            HCCL_VM_WARN(
                "AIV op detected, the old checker is skipped and only CheckerV3 will run, "
                "opIndex={}, oldCheckerEnabled={}",
                opIdx, enableOldChecker);
            enableOldChecker = false;
            checkerResult[OLD_CHECKER_RESULT] = CheckerStatus::DISABLE;
        }
        enableNewChecker = true;
    }

    {
        auto checkerParamBrief = storage.GetCheckerParam();
        std::ostringstream summary;
        summary << "[Main] Op summary, opIndex=" << opIdx
                << ", collectiveType=" << HcclSim::HcclCmdTypeToString(checkerParamBrief.cmdType)
                << ", rankCount=" << checkerParamBrief.rankSize
                << ", dataType=" << HcclSim::HcclDataTypeToString(checkerParamBrief.dataType)
                << ", elementCount=" << checkerParamBrief.dataCount
                << ", reduceType=" << HcclReduceOpToString(checkerParamBrief.reduceType);
        AppendApplicableRoleFields(summary, checkerParamBrief);
        summary << ", opGroupSize=" << opGroup.size() << ", usesAivExpansionMode=" << usesAivExpansionMode
                << ", hasAivGraphTask=" << hasAivGraphTask;
        HCCL_VM_INFO("{}", summary.str());
    }

    HcclResult newCheckerRet = HcclResult::HCCL_SUCCESS;
    if (enableNewChecker) {
        HCCL_VM_INFO("----------[Start CheckerV3]----------");
        newCheckerRet = HcclSim::GenAndCheckGraphV3();
        HCCL_VM_INFO("----------[CheckerV3 Finished]----------");
        HCCL_VM_INFO("CheckerV3 finished for this op, opIndex={}", opIdx);
        checkerResult[NEW_CHECKER_RESULT]
            = newCheckerRet == HcclResult::HCCL_SUCCESS ? CheckerStatus::SUCCESS : CheckerStatus::FAILED;
    } else {
        HCCL_VM_INFO("CheckerV3 is disabled by configuration");
    }

    HcclResult oldCheckerRet = HcclResult::HCCL_SUCCESS;
    if (enableOldChecker) {
        HCCL_VM_INFO("----------[Start Old Checker]----------");
        HCCL_VM_INFO("Start running the old checker, opIndex={}, dataId={}", opIdx, storage.GetDataId());
        HcclSim::AllRankTaskQueues taskQueues;
        const HcclResult convertRet = HcclSim::ConvertTaskQueue(taskQueues);
        if (convertRet != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "{} Failed to convert tasks, opIndex={}",
                HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), opIdx);
            oldCheckerRet = convertRet;
        } else {
            auto checkerParam = storage.GetCheckerParam();
            oldCheckerRet = DispatchCheckByCmdType(taskQueues, checkerParam);
        }
        checkerResult[OLD_CHECKER_RESULT]
            = oldCheckerRet == HcclResult::HCCL_SUCCESS ? CheckerStatus::SUCCESS : CheckerStatus::FAILED;
        HCCL_VM_INFO("----------[Old Checker Finished]----------");
        HCCL_VM_INFO("Old checker finished for this op, opIndex={}", opIdx);
    } else {
        HCCL_VM_INFO("Old checker is disabled by configuration");
    }

    if (!isAivOp && enableNewChecker && enableOldChecker && newCheckerRet != oldCheckerRet) {
        HCCL_VM_WARN(
            "CheckerV3 result differs from old checker result, opIndex={}, checkerV3Ret={}, "
            "oldCheckerRet={}",
            opIdx, static_cast<u32>(newCheckerRet), static_cast<u32>(oldCheckerRet));
    }
    if (enableNewChecker && newCheckerRet != HcclResult::HCCL_SUCCESS) {
        return newCheckerRet;
    }
    return oldCheckerRet;
}

static HcclResult ProcessOneBigGraphSyncIter(
    loader::Loader& loader, uint32_t syncIter, HcclSim::BigGraphCheckV3::BigGraphCheckerV3& bigGraphChecker)
{
    HCCL_VM_INFO("----------[Start BigGraphCheckerV3]----------");
    HCCL_VM_INFO("Start building the big graph for one sync iteration, syncIter={}", syncIter);

    // Each sync window owns an independent CCU register state. Keep this reset at
    // the window boundary; the V3 CCU expansion must remain continuous within the window.
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    HcclResult ret = bigGraphChecker.LoadOpData(loader, syncIter);
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to load multi-operator data for big graph, syncIter={}, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), syncIter,
            static_cast<uint32_t>(ret));
        return ret;
    }

    if (IsSingleRankWithNoTask(bigGraphChecker.GetData())) {
        HCCL_VM_WARN(
            "Big-graph check is skipped and treated as success because this sync iteration "
            "contains only single-rank operations with no tasks, syncIter={}",
            syncIter);
        return HcclResult::HCCL_SUCCESS;
    }

    ret = bigGraphChecker.TranslateTask();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to translate multi-operator tasks for big graph, syncIter={}, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), syncIter,
            static_cast<uint32_t>(ret));
        return ret;
    }

    ret = bigGraphChecker.GenerateBigGraph();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to generate big graph, syncIter={}, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), syncIter,
            static_cast<uint32_t>(ret));
        return ret;
    }

    ret = bigGraphChecker.SyncCheck();
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Big graph sync-conflict check failed, syncIter={}, ret={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::CHECKER_RUNTIME_ERROR), syncIter,
            static_cast<uint32_t>(ret));
        return ret;
    }

    const auto* graph = bigGraphChecker.GetGraph();
    const size_t nodeCount = graph == nullptr ? 0 : graph->GetNodes().size();
    const size_t rankCount = graph == nullptr ? 0 : graph->GetTaskQueues().size();
    HCCL_VM_INFO(
        "BigGraphCheckerV3 generated graph successfully, syncIter={}, operatorCount={}, "
        "nodeCount={}, rankCount={}",
        syncIter, bigGraphChecker.GetOpParams().size(), nodeCount, rankCount);
    HCCL_VM_INFO("----------[BigGraphCheckerV3 Finished]----------");
    return HcclResult::HCCL_SUCCESS;
}

json BuildOpParamSummaryJson(const HcclSim::CheckerParam& param)
{
    json opParamJson = json::object();
    opParamJson["cmd_type"] = static_cast<u32>(param.cmdType);
    opParamJson["rank_size"] = param.rankSize;
    opParamJson["data_type"] = static_cast<u32>(param.dataType);
    opParamJson["data_count"] = param.dataCount;

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLREDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
        || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        opParamJson["reduce_type"] = static_cast<u32>(param.reduceType);
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_SEND || param.cmdType == HcclCMDType::HCCL_CMD_RECEIVE) {
        opParamJson["src_rank"] = param.srcRank;
        opParamJson["dst_rank"] = param.dstRank;
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_BROADCAST || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE
        || param.cmdType == HcclCMDType::HCCL_CMD_SCATTER) {
        opParamJson["root"] = param.root;
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLGATHER_V || param.cmdType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V) {
        json vDataDesJson = json::object();
        vDataDesJson["data_type"] = param.vDataDes.dataType;
        vDataDesJson["rank_count"] = param.vDataDes.count;
        vDataDesJson["counts_size"] = param.vDataDes.counts.size();
        vDataDesJson["displs_size"] = param.vDataDes.displs.size();
        vDataDesJson["counts"] = param.vDataDes.counts;
        vDataDesJson["displs"] = param.vDataDes.displs;
        opParamJson["v_data_des"] = std::move(vDataDesJson);
    }

    if (param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        || param.cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV) {
        json all2AllDataDesJson = json::object();
        all2AllDataDesJson["send_type"] = param.all2AllDataDes.sendType;
        all2AllDataDesJson["recv_type"] = param.all2AllDataDes.recvType;
        all2AllDataDesJson["send_count"] = param.all2AllDataDes.sendCount;
        all2AllDataDesJson["recv_count"] = param.all2AllDataDes.recvCount;
        all2AllDataDesJson["count"] = param.all2AllDataDes.count;
        all2AllDataDesJson["send_count_matrix_size"] = param.all2AllDataDes.sendCountMatrix.size();
        opParamJson["all2all_data_des"] = std::move(all2AllDataDesJson);
    }
    return opParamJson;
}

// --- 业务函数修正 ---
void RunChecker(const std::string& data_id)
{
    std::lock_guard<std::mutex> runLock(g_run_checker_mutex);
    HcclSim::StorageManager& storage = HcclSim::StorageManager::GetInstance();
    storage.Reset();
    storage.SetDataId(data_id);
    const HcclResult settingRefreshRet = HcclSim::SettingManager::GetInstance().Refresh();
    if (settingRefreshRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_WARN("Failed to refresh manifest settings, the previous checker settings will be kept");
    }
    HcclSim::DumpManager& dumpManager = HcclSim::DumpManager::GetInstance();
    dumpManager.Reset();
    HcclResult dumpInitRet = dumpManager.Initialize(data_id);
    if (dumpInitRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to initialize the old checker dump manager, old checker output files "
            "cannot be written, dataId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id);
        return;
    }
    HcclSim::DumpV3Manager& dumpV3Manager = HcclSim::DumpV3Manager::GetInstance();
    dumpV3Manager.Reset();
    dumpInitRet = dumpV3Manager.Initialize(data_id);
    if (dumpInitRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR(
            "{} Failed to initialize the V3 dump manager, checker output files cannot be written, "
            "dataId={}",
            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id);
        return;
    }
    HcclSim::DumpRunManifest::GetInstance().Reset(data_id);
    HcclSim::AllRankParamRecorder::Global()->Reset();
    HcclSim::TaskGraphGeneratorV3::AllRankParamRecorder::Global()->Reset();
    storage.InitCcuInfo(
        HcclSim::AllRankParamRecorder::Global()->devType_,
        HcclSim::AllRankParamRecorder::Global()->ccu_resource_base_addr_);
    HcclSim::g_ccuGraphTaskOri2New.clear();

    HcclResult loadRet = g_loader.LoadOpTaskFile();
    if (loadRet != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_ERROR("Failed to load the op task file");
        return;
    }

    std::vector<sim::CcuChannelTab> channels;
    std::vector<sim::CcuInstrResTab> instrRes;
    std::vector<sim::SyncRecordTab> syncRecords;
    uint32_t syncIterMaxNum = 0;
    HcclResult ret = LoadCheckerDataBase(channels, instrRes, syncRecords, syncIterMaxNum);
    if (ret != HcclResult::HCCL_SUCCESS || syncRecords.empty()) {
        return;
    }

    HCCL_VM_INFO("Start checker run, syncRecordCount={}", syncRecords.size());
    const HcclSim::CheckerSettings checkerSettings = HcclSim::SettingManager::GetInstance().GetSettings();
    const bool enableBigGraphChecker = checkerSettings.enableBigGraphChecker;
    const bool enableSingleOpChecker = checkerSettings.enableNewChecker || checkerSettings.enableOldChecker;
    std::vector<uint32_t> bigGraphSyncIters;
    bigGraphSyncIters.reserve(syncRecords.size());
    for (const auto& syncRecord : syncRecords) {
        if (bigGraphSyncIters.empty() || bigGraphSyncIters.back() != syncRecord.syncIter) {
            bigGraphSyncIters.push_back(syncRecord.syncIter);
        }
    }
    std::vector<CheckerStatus> multiOpCheckerResults(
        bigGraphSyncIters.size(), enableBigGraphChecker ? CheckerStatus::NOT_EXECUTED : CheckerStatus::DISABLE);
    std::vector<CheckerResult> checkerResults;
    HcclSim::BigGraphCheckV3::BigGraphCheckerV3 bigGraphChecker;

    if (enableBigGraphChecker) {
        for (size_t iterIndex = 0; iterIndex < bigGraphSyncIters.size(); ++iterIndex) {
            const uint32_t syncIter = bigGraphSyncIters[iterIndex];
            const HcclResult bigGraphRet = ProcessOneBigGraphSyncIter(g_loader, syncIter, bigGraphChecker);
            multiOpCheckerResults[iterIndex]
                = bigGraphRet == HcclResult::HCCL_SUCCESS ? CheckerStatus::SUCCESS : CheckerStatus::FAILED;
            if (bigGraphRet != HcclResult::HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "BigGraphCheckerV3 failed, syncIter={}, ret={}", syncIter, static_cast<uint32_t>(bigGraphRet));
            }
        }
    }

    if (!enableSingleOpChecker) {
        HCCL_VM_INFO("Single-op checkers are disabled by configuration, skip sync iteration checks");
    } else {
        uint32_t opIdx = 0;
        for (uint32_t syncIter = 0; syncIter <= syncIterMaxNum; syncIter++) {
            std::map<uint32_t, std::vector<sim::CompositeOpDetail>> compositeDataMap;
            g_loader.LoadCompositeOpDetailBySyncIter(syncIter, compositeDataMap);
            auto opGroups = TransposeCompositeOpMap(compositeDataMap);
            HCCL_VM_INFO("Start one sync iteration, syncIter={}, opGroupCount={}", syncIter, opGroups.size());
            for (auto& opGroup : opGroups) {
                HCCL_VM_INFO("Check one op group in this sync iteration, opGroupSize={}", opGroup.size());
                const uint32_t currentOpIdx = opIdx++;
                CheckerResult checkerResult = {CheckerStatus::DISABLE, CheckerStatus::DISABLE};
                ret = ProcessOneOpGroup(storage, channels, instrRes, currentOpIdx, opGroup, checkerResult);
                checkerResults.push_back(checkerResult);
                if (dumpManager.IsEnabled()) {
                    HcclSim::DumpRunManifest::GetInstance().SetCheckResult(ret);
                    const HcclResult flushRet = HcclSim::ValidationIssueRecorder::GetInstance().Flush();
                    if (flushRet != HcclResult::HCCL_SUCCESS) {
                        HCCL_VM_WARN(
                            "{} Failed to flush the validation issue dump, dataId={}, opIndex={}, "
                            "dumpType=validation_issues",
                            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id, currentOpIdx);
                    }
                    const HcclResult manifestRet = HcclSim::DumpRunManifest::GetInstance().Flush();
                    if (manifestRet != HcclResult::HCCL_SUCCESS) {
                        HCCL_VM_WARN(
                            "{} Failed to flush the dump manifest, dataId={}, opIndex={}",
                            HcclSim::MakeErrorCodeText(HcclSim::ErrorCode::DUMP_FAILED), data_id, currentOpIdx);
                    }
                }
                const bool checkerResultFailed = checkerResult[OLD_CHECKER_RESULT] == CheckerStatus::FAILED
                                                 || checkerResult[NEW_CHECKER_RESULT] == CheckerStatus::FAILED;
                if (ret != HcclResult::HCCL_SUCCESS || checkerResultFailed) {
                    HCCL_VM_ERROR("op[{}] Checker failed", currentOpIdx);
                    continue;
                }
                const bool checkerResultNotExecuted
                    = checkerResult[OLD_CHECKER_RESULT] == CheckerStatus::NOT_EXECUTED
                      || checkerResult[NEW_CHECKER_RESULT] == CheckerStatus::NOT_EXECUTED;
                if (checkerResultNotExecuted) {
                    HCCL_VM_INFO("op[{}] Checker not executed", currentOpIdx);
                    continue;
                }
                HCCL_VM_INFO("op[{}] Checker Success", currentOpIdx);
            }
        }
    }
    if (!checkerResults.empty()) {
        constexpr int OP_COLUMN_WIDTH = 8;
        constexpr int CHECKER_COLUMN_WIDTH = 13;
        HCCL_VM_INFO("Checker execution result (success/failed/disable/not_executed):");
        HCCL_VM_INFO("Single-op checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(OP_COLUMN_WIDTH) << "op[id]"
               << " | " << std::setw(CHECKER_COLUMN_WIDTH) << "old checker"
               << " | " << std::setw(CHECKER_COLUMN_WIDTH) << "new checker" << " |";
        HCCL_VM_INFO("{}", header.str());
        for (size_t opIdx = 0; opIdx < checkerResults.size(); ++opIdx) {
            const CheckerResult& checkerResult = checkerResults[opIdx];
            std::ostringstream row;
            row << "| " << std::left << std::setw(OP_COLUMN_WIDTH) << opIdx << " | " << std::setw(CHECKER_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(checkerResult[OLD_CHECKER_RESULT])] << " | "
                << std::setw(CHECKER_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(checkerResult[NEW_CHECKER_RESULT])] << " |";
            HCCL_VM_INFO("{}", row.str());
        }
    } else {
        HCCL_VM_WARN("Checker execution result is unavailable because no single-op checker was executed");
    }
    if (!multiOpCheckerResults.empty()) {
        constexpr int SYNC_ITER_COLUMN_WIDTH = 10;
        constexpr int MULTI_OP_COLUMN_WIDTH = 17;
        HCCL_VM_INFO("Multi-op checker result:");
        std::ostringstream header;
        header << "| " << std::left << std::setw(SYNC_ITER_COLUMN_WIDTH) << "syncIter"
               << " | " << std::setw(MULTI_OP_COLUMN_WIDTH) << "multi op checker" << " |";
        HCCL_VM_INFO("{}", header.str());
        for (size_t iterIndex = 0; iterIndex < multiOpCheckerResults.size(); ++iterIndex) {
            std::ostringstream row;
            row << "| " << std::left << std::setw(SYNC_ITER_COLUMN_WIDTH) << bigGraphSyncIters[iterIndex] << " | "
                << std::setw(MULTI_OP_COLUMN_WIDTH)
                << CHECKER_STATUS_TEXT[static_cast<size_t>(multiOpCheckerResults[iterIndex])] << " |";
            HCCL_VM_INFO("{}", row.str());
        }
    }
    bool hasCheckerFailure = false;
    bool hasCheckerExecution = false;
    bool hasCheckerNotExecuted = false;
    for (const auto& checkerResult : checkerResults) {
        for (const CheckerStatus status : checkerResult) {
            hasCheckerFailure = hasCheckerFailure || status == CheckerStatus::FAILED;
            hasCheckerExecution
                = hasCheckerExecution || status == CheckerStatus::SUCCESS || status == CheckerStatus::FAILED;
            hasCheckerNotExecuted = hasCheckerNotExecuted || status == CheckerStatus::NOT_EXECUTED;
        }
    }
    for (const CheckerStatus status : multiOpCheckerResults) {
        hasCheckerFailure = hasCheckerFailure || status == CheckerStatus::FAILED;
        hasCheckerExecution
            = hasCheckerExecution || status == CheckerStatus::SUCCESS || status == CheckerStatus::FAILED;
        hasCheckerNotExecuted = hasCheckerNotExecuted || status == CheckerStatus::NOT_EXECUTED;
    }
    if (hasCheckerExecution && !hasCheckerFailure && !hasCheckerNotExecuted) {
        HCCL_VM_INFO(
            "[CHECKER_RUN_SUMMARY] All Success (Total Op: {}, Total SyncIter: {})", checkerResults.size(),
            multiOpCheckerResults.size());
    } else {
        HCCL_VM_INFO(
            "[CHECKER_RUN_SUMMARY] Failed (Total Op: {}, Total SyncIter: {})", checkerResults.size(),
            multiOpCheckerResults.size());
    }
    std::cout << "(hvm)$> " << std::flush;
    FlushLog(); // 将本轮完整日志落盘
}

void StartCheckerWorker(const std::string& dataId)
{
    std::lock_guard<std::mutex> workerLock(g_worker_mutex);
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }
    g_worker_thread = std::thread([dataId]() {
        RunChecker(dataId);
    });
}

void JoinCheckerWorker()
{
    std::lock_guard<std::mutex> workerLock(g_worker_mutex);
    if (g_worker_thread.joinable()) {
        g_worker_thread.join();
    }
}

// --- 分发函数修正 ---
// 不再使用 exit(0)，而是通过标记位通知主线程
void ProcessCommand(const std::string& line)
{
    try {
        auto j = json::parse(line);
        std::string action = j.value("action", "");
        auto payload = j.value("payload", json::object());

        if (action == "status") {
            std::string status = payload.value("status", "");
            HCCL_VM_INFO("Received checker status signal, status={}", status);
            if (status != "finish") {
                return;
            }
            // 运行前刷新设置，确保最新的配置生效
            const HcclResult settingRefreshRet = HcclSim::SettingManager::GetInstance().Refresh();
            if (settingRefreshRet != HcclResult::HCCL_SUCCESS) {
                HCCL_VM_WARN("Failed to refresh manifest settings, use the previous settings");
            }

            std::string data_id = payload.value("data_id", "");
            StartCheckerWorker(data_id);
        } else if (action == "stop") {
            HCCL_VM_INFO("Received checker stop signal, shutdown...");
            g_keep_running.store(false); // 仅仅修改标志位
        }
        // 其他 action...
    } catch (const std::exception& e) {
        HCCL_VM_ERROR("Command processing failed: {}", e.what());
    }
}

int main()
{
    LogConfig config = LoadLogConfig("checker");
    InitLogger(config);

    HCCL_VM_INFO("Plugin process active. Listening for commands...");

    // 主循环检查标志位
    std::string line;
    // 注意：std::getline 是阻塞的。如果 stop 指令后没有后续输入，
    // 循环会卡在 getline。但在插件管理场景下，发送完 stop 后通常会关闭管道，
    // 导致 getline 返回 false。
    while (g_keep_running.load() && std::getline(std::cin, line)) {
        if (line.empty())
            continue;
        ProcessCommand(line);
    }

    JoinCheckerWorker();
    HCCL_VM_INFO("shutdown.");
    return 0; // 整个进程唯一的正常出口
}
