/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <algorithm>
#include <unordered_set>
#include "mc2_compont.h"
#include "ccu_assist.h"
#include "mc2_context.h"
#include "ccu_task_arg_mc2.h"
#include "coll_service_device_mode.h"
#include "mc2_global_mirror_tasks.h"
#include "op_params_checker.h"
#include "runtime_api_exception.h"

namespace Hccl {

Mc2Compont::~Mc2Compont()
{
    auto deviceLogicId = comm->GetDeviceLogicId();
    for (const auto& server : ccuServerMap) {
        auto ret = InsExeQue::DeregisterExtendInstruction(deviceLogicId, server.first);
        HCCL_INFO("[Mc2Compont:%s]Destroy ccuServer execId[%u]", __func__, server.first);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_ERROR("DeregisterExtendInstruction execId[%u] failed, ret[%d]", server.first, ret);
        }
    }
    ccuServerMap.clear();
}

void Mc2Compont::AllocCommResource(void* mc2Tiling, void** commContext)
{
    auto tilingVersion = *static_cast<uint32_t*>(mc2Tiling);
    HCCL_INFO("[Mc2Compont:%s] Tiling version [%u]", __func__, tilingVersion);
    if (tilingVersion != UNKNOWN_TILING_V1 && tilingVersion != UNKNOWN_TILING_V2) {
        THROW<NotSupportException>(StringFormat("Tiling version not support, version[%u]", tilingVersion));
    }

    if (comm->GetRankSize() == 1) {
        HCCL_WARNING("Comm[%s] rank size is 1, Mc2 not support", comm->GetId().c_str());
        return;
    }

    std::unordered_set<uint64_t> algoTemplateRequire;
    if (tilingVersion == UNKNOWN_TILING_V1) {
        // 申请deviceMem、通信域信息获取、commContext赋值
        Alloc();
        // 生成本次需要的算子模板
        GenerateAlgoTemplates(reinterpret_cast<Mc2Tiling*>(mc2Tiling), algoTemplateRequire);
    } else {
        // 申请deviceMem、通信域信息获取、commContext赋值
        AllocV2();
        // 生成本次需要的算子模板
        GenerateAlgoTemplatesV2(reinterpret_cast<Mc2InitTilingInner*>(mc2Tiling), algoTemplateRequire);
    }

    HCCL_RUN_INFO(
        "hcclCombinOpParam info: workSpace = [%llu], rankId = [%u], rankDim = [%u], xnAddr = [%llu], "
        "ckeAddr = [%llu], winSize = [%llu], windowsOut[0] = [%llu]",
        combinOpParam.workSpace, combinOpParam.rankId, combinOpParam.rankDim, combinOpParam.xnAddr,
        combinOpParam.ckeAddr, combinOpParam.winSize, combinOpParam.windowsOut[0]);
    HCCL_RUN_INFO(
        "opType[0] = [%u], opType[1] = [%u], opType[2] = [%u], opType[3] = [%u], opType[4] = [%u], "
        "opType[5] = [%u], opType[6] = [%u], opType[7] = [%u], ",
        combinOpParam.opType[0], combinOpParam.opType[1], combinOpParam.opType[2], combinOpParam.opType[3],
        combinOpParam.opType[4], combinOpParam.opType[5], combinOpParam.opType[6], combinOpParam.opType[7]);
    HCCL_RUN_INFO(
        "algorithmType[0] = [%u], algorithmType[1] = [%u], algorithmType[2] = [%u], algorithmType[3] = [%u], "
        "algorithmType[4] = [%u], algorithmType[5] = [%u], algorithmType[6] = [%u], algorithmType[7] = [%u]",
        combinOpParam.algorithmType[0], combinOpParam.algorithmType[1], combinOpParam.algorithmType[2],
        combinOpParam.algorithmType[3], combinOpParam.algorithmType[4], combinOpParam.algorithmType[5],
        combinOpParam.algorithmType[6], combinOpParam.algorithmType[7]);
    auto paramSize = sizeof(HcclCombinOpParam);
    if (combinOpParamBuffer == nullptr) {
        combinOpParamBuffer = std::make_shared<DevBuffer>(paramSize);
    }
    HrtMemcpy(
        reinterpret_cast<void*>(combinOpParamBuffer->GetAddr()), paramSize, static_cast<void*>(&combinOpParam),
        paramSize, RT_MEMCPY_HOST_TO_DEVICE);
    *commContext = reinterpret_cast<void*>(combinOpParamBuffer->GetAddr());
    // 生成ccuServer指令，将注册得到的execId保存在curExecId，GetCcuTaskInfo时通过curExecId获取TaskParam
    GenerateCcuServer(algoTemplateRequire);
}

static bool GetArgSizeFlag(std::vector<std::vector<CcuTaskParam>>& taskParams)
{
    /*
     * 在MC2场景下，暂定只支持三种场景：1）单die且一个mission，2）双die且每die一个mission，3）单die且多个mission
     * 一次调用HCCL接口只能支持一种场景，其中2）包括所有双die算法，3）包括带尾块处理的算法与HalfAlltoAllV算子，1）包括其余算法
     * 只有场景3）时需要使argSize=1，1）和2）argSize均不变
     */
    std::unordered_set<uint8_t> dieIdSet;
    std::unordered_set<uint8_t> missionIdSet;
    for (auto& task : taskParams) {
        dieIdSet.emplace(task[0].dieId);
        missionIdSet.emplace(task[0].missionId);
        HCCL_INFO("TaskParam: dieId = [%u], missionId = [%u]", task[0].dieId, task[0].missionId);
    }
    bool oneDieOneMission = (dieIdSet.size() == 1) && (missionIdSet.size() == 1);
    bool twoDieOneMission = (dieIdSet.size() == 2) && (missionIdSet.size() == 1);
    bool oneDieMultiMission = (dieIdSet.size() == 1) && (missionIdSet.size() > 1);
    if (!oneDieOneMission && !twoDieOneMission && !oneDieMultiMission) {
        THROW<NotSupportException>(
            StringFormat("MC2 Scene cannot support: not OneDieOneMission, TwoDieOneMission or OneDieMultiMission !"));
    }
    return oneDieMultiMission;
}

std::vector<CcuTaskParam> Mc2Compont::GetCcuTaskInfo(void* tilingData)
{
    HCCL_INFO("%s start.", __func__);
    HCCL_INFO("tilingData=%llu", tilingData);
    std::vector<std::vector<CcuTaskParam>> taskParams;
    std::vector<CcuTaskParam> ccuTaskParam;
    if (tilingData == nullptr) {
        return ccuTaskParam;
    }
    auto mc2Tiling = reinterpret_cast<Mc2Tiling*>(tilingData);
    HCCL_INFO("mc2Tiling=%s", mc2Tiling->ToString().c_str());
    auto version = mc2Tiling->version;

    if (version != UNKNOWN_TILING_V1 && version != UNKNOWN_TILING_V2) {
        THROW<NotSupportException>(StringFormat("Tiling version not support, version[%u]", version));
    }

    // 校验curExecId是否有效
    if (ccuServerMap.find(curExecId) == ccuServerMap.end()) {
        THROW<Hccl::InternalException>(
            StringFormat("CcuServer not find, curExecId[%llu], ccuServerSize[%d]", curExecId, ccuServerMap.size()));
    }

    CcuTaskArgMc2 ccuTaskArg(tokenInfo);
    HcclResult ret = CcuCtxMgr::GetTaskParam(comm->GetDeviceLogicId(), ccuTaskArg, curExecId, taskParams);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>(StringFormat("GetTaskParam failed, ret[%u]", ret));
    }
    bool argSizeFlag = GetArgSizeFlag(taskParams);
    for (auto& task : taskParams) {
        if (task.size() != 1) {
            THROW<Hccl::InternalException>(
                StringFormat("Task Num In TaskParams Should Be 1, While It Is %d", task.size()));
        }
        if (argSizeFlag) {
            task[0].argSize = 1;
        }
        u32 notifyTimeout = comm->GetNotifyTimeoutCfg().GetNotifyTimeout();
        task[0].timeout
            = (notifyTimeout > UINT16_MAX) ? static_cast<uint16_t>(UINT16_MAX) : static_cast<uint16_t>(notifyTimeout);
        ccuTaskParam.push_back(task[0]);
        SaveMc2DfxTaskInfo(task[0], curExecId);
    }
    std::sort(ccuTaskParam.begin(), ccuTaskParam.end(), [](const CcuTaskParam& a, const CcuTaskParam& b) {
        return a.missionId < b.missionId;
    });
    HCCL_INFO("GetCcuTaskInfo success");
    return ccuTaskParam;
}

void Mc2Compont::Alloc()
{
    // inputMem给算法编排使用，只需要申请一次，按照最大数据类型申请
    inputMem = std::make_shared<DevBuffer>(dataCount * DataTypeSizeGet(DataType::INT64) * comm->GetRankSize());
    HCCL_INFO("[Mc2Compont][Alloc]inputMem addr[%p] size = [%llu]", inputMem->GetAddr(), inputMem->GetSize());
    for (uint32_t i = 0; i < MAX_OP_NUM; i++) {
        combinOpParam.opType[i] = 0;
        combinOpParam.algorithmType[i] = 0;
    }
    if (ccuResourceAlloced) {
        return;
    }

    constexpr uint32_t comSyncNum = 2; // 每轮同步使用2个同步信号
    uint32_t comParamBufSize = CCU_TASK_NUM_MAX * CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE;
    uint32_t comSyncBufSize = CCU_TASK_NUM_MAX * comSyncNum * CCU_ONE_PARAM_SIZE;
    workspaceBuffer = std::make_shared<DevBuffer>(MC2_WORKSPACE_SIZE);
    comParamBuffer = std::make_shared<DevBuffer>(comParamBufSize);
    comSyncBuffer = std::make_shared<DevBuffer>(comSyncBufSize);

    combinOpParam.workSpace = static_cast<uint64_t>(workspaceBuffer->GetAddr());
    combinOpParam.workSpaceSize = MC2_WORKSPACE_SIZE;
    combinOpParam.rankId = comm->GetMyRank();
    combinOpParam.rankDim = comm->GetRankSize();
    combinOpParam.xnAddr = static_cast<uint64_t>(comParamBuffer->GetAddr());
    combinOpParam.ckeAddr = static_cast<uint64_t>(comSyncBuffer->GetAddr());
    // add cclbuffer info
    if (comm->GetCclBuffer() == nullptr) {
        THROW<Hccl::InternalException>(StringFormat("Cannot get CCL Buffer to fill window!"));
    }
    combinOpParam.winSize = static_cast<uint64_t>(comm->GetCclBuffer()->GetSize());
    combinOpParam.windowsOut[0] = static_cast<uint64_t>(comm->GetCclBuffer()->GetAddr());
    ccuResourceAlloced = true;

    tokenInfo = CcuRep::GetTokenInfo(
        static_cast<uint64_t>(workspaceBuffer->GetAddr()), static_cast<uint64_t>(workspaceBuffer->GetSize()));
}

void Mc2Compont::AllocV2()
{
    inputMem = std::make_shared<DevBuffer>(dataCount * DataTypeSizeGet(DataType::INT64) * comm->GetRankSize());
    HCCL_INFO("[Mc2Compont][AllocV2]inputMem addr[%p] size = [%llu]", inputMem->GetAddr(), inputMem->GetSize());
    for (uint32_t i = 0; i < MAX_OP_NUM; i++) {
        combinOpParam.opType[i] = 0;
        combinOpParam.algorithmType[i] = 0;
    }
    if (ccuResourceAlloced) {
        return;
    }

    constexpr uint32_t comSyncNum = 2; // 每轮同步使用2个同步信号
    uint32_t comParamBufSize = CCU_TASK_NUM_MAX * CCU_PARAM_NUM_MAX * CCU_ONE_PARAM_SIZE;
    uint32_t comSyncBufSize = CCU_TASK_NUM_MAX * comSyncNum * CCU_ONE_PARAM_SIZE;
    workspaceBuffer = std::make_shared<DevBuffer>(MC2_WORKSPACE_SIZE);
    comParamBuffer = std::make_shared<DevBuffer>(comParamBufSize);
    comSyncBuffer = std::make_shared<DevBuffer>(comSyncBufSize);
    HrtMemset(reinterpret_cast<void*>(workspaceBuffer->GetAddr()), MC2_WORKSPACE_SIZE, MC2_WORKSPACE_SIZE);
    HrtMemset(reinterpret_cast<void*>(comParamBuffer->GetAddr()), comParamBufSize, comParamBufSize);
    HrtMemset(reinterpret_cast<void*>(comSyncBuffer->GetAddr()), comSyncBufSize, comSyncBufSize);
    combinOpParam.workSpace = static_cast<uint64_t>(workspaceBuffer->GetAddr());
    combinOpParam.workSpaceSize = MC2_WORKSPACE_SIZE;
    combinOpParam.rankId = comm->GetMyRank();
    combinOpParam.rankDim = comm->GetRankSize();
    combinOpParam.xnAddr = static_cast<uint64_t>(comParamBuffer->GetAddr());
    combinOpParam.ckeAddr = static_cast<uint64_t>(comSyncBuffer->GetAddr());
    // add cclbuffer info
    if (comm->GetCclBuffer() == nullptr) {
        THROW<Hccl::InternalException>(StringFormat("Cannot get CCL Buffer to fill window!"));
    }
    combinOpParam.winSize = static_cast<uint64_t>(comm->GetCclBuffer()->GetSize());
    combinOpParam.windowsOut[0] = static_cast<uint64_t>(comm->GetCclBuffer()->GetAddr());
    ccuResourceAlloced = true;

    tokenInfo = CcuRep::GetTokenInfo(
        static_cast<uint64_t>(workspaceBuffer->GetAddr()), static_cast<uint64_t>(workspaceBuffer->GetSize()));
}

void Mc2Compont::MC2Orchestrate(
    const CollAlgParams& params, std::shared_ptr<InsQueue>& insQueue, uint8_t commEngine) const
{
    auto op = comm->GetCurrentCollOperator();

    CollOpParams opParams;
    opParams.commEngine = static_cast<HcclAccelerator::Value>(commEngine);
    opParams.opType = op->opType;
    opParams.dataType = op->dataType;
    opParams.count = op->dataCount;
    opParams.reduceOp = op->reduceOp;
    opParams.isMc2 = params.isMc2;
    comm->ExecAlgSelect(opParams, op->opMode);
    if (!comm->GetOpCcuFeatureFlag()) { // 算子粒度
        auto msg = StringFormat(
            "[Mc2Compont:%s]AlgSelect not ccu, accState[%s]", __func__,
            comm->GetOpExecuteConfig().accState.Describe().c_str());
        THROW<InternalException>(msg);
    }

    std::string algName = comm->GetCurAlgName();
    // 算子编排获取InsQueue
    auto ret = comm->GetCollAlgComponent()->Orchestrate(*op, params, algName, insQueue);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<InternalException>(
            StringFormat("Error occurs when call collAlgComponent.orchestrate(), error code: %d", ret));
    }
}

void Mc2Compont::MC2AllocCommRes(
    const CollAlgParams& params, std::shared_ptr<InsQueue>& insQueue, uint8_t commEngine) const
{
    MC2Orchestrate(params, insQueue, commEngine);
    // 获取LinkData
    auto collService = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService());
    auto ccuLinks = collService->GetUniqueLinks(insQueue);
    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(ccuLinks);
    // 对insQueue中ccuIns进行预处理(创建transport、ccuCtx、分配资源、注册等)
    collService->GetCcuInsPreprocessor()->Preprocess(insQueue, true);
    if (collService->GetCcuInsPreprocessor()->IsRollback()) { // mc2暂不能回退到aicpu
        THROW<InternalException>("[Mc2Compont][%s]ResAlloc unsuccessful.", __func__);
    }
}

void Mc2Compont::SaveAlgoInfo(uint32_t index, uint64_t templateSign, uint32_t opType, uint8_t algorithmType)
{
    combinOpParam.opType[index] = opType;
    combinOpParam.algorithmType[index] = algorithmType;
    HcclAlgoInfo hcclAlgoInfo{};
    hcclAlgoInfo.opType = opType;
    hcclAlgoInfo.algorithmType = algorithmType;
    algoInfoMap_[templateSign] = hcclAlgoInfo;
    return;
}

void Mc2Compont::GenerateAlgoTemplates(Mc2Tiling* mc2TilingPtr, std::unordered_set<uint64_t>& algoTemplateRequire)
{
    HCCL_INFO("GenerateAlgoTemplates start v1");

    auto tmpMemSize = comm->GetBufferSize();
    CollAlgParams params;
    params.opMode = OpMode::OPBASE;
    params.maxTmpMemSize = tmpMemSize;
    params.isMc2 = true;
    // 从mc2Tiling中获取需要的算法信息
    Mc2CommConfig* commConfigPtr = reinterpret_cast<Mc2CommConfig*>(
        reinterpret_cast<uint8_t*>(mc2TilingPtr) + sizeof(uint32_t) + sizeof(uint32_t) + sizeof(Mc2ServerCfg));
    for (uint32_t index = 0; index < mc2TilingPtr->commConfigNum; index++) {
        const auto& commConfig = *(commConfigPtr + index);
        OpParamsChecker::CheckOpDataTypeMC2(commConfig);
        uint64_t templateSign = GetTemplateSignature(commConfig);
        algoTemplateRequire.insert(templateSign);
        // 已经生成过的算法模板不再生成
        if (algoTemplateMap.find(templateSign) != algoTemplateMap.end()) {
            HCCL_INFO(
                "A algoTemplate that meets the requirement already exists, index = [%u], templateSign = [%llu]", index,
                templateSign);
            if (algoInfoMap_.find(templateSign) != algoInfoMap_.end()) {
                combinOpParam.opType[index] = algoInfoMap_[templateSign].opType;
                combinOpParam.algorithmType[index] = algoInfoMap_[templateSign].algorithmType;
                continue;
            } else {
                THROW<Hccl::InternalException>(
                    StringFormat("algoInfoMap_ do not has templateSign = [%llu]", templateSign));
            }
        }

        FillCollOperator(commConfig);

        auto insQueue = make_shared<InsQueue>();
        MC2AllocCommRes(params, insQueue, commConfig.communicationEngine);

        std::string algName = comm->GetCurAlgName();
        HCCL_INFO(
            "Orchestrate: index = [%u], algName = [%s], templateSign = [%llu]", index, algName.c_str(), templateSign);
        if (insQueue->Iter()->GetType() != InstructionType::CCU_INS) {
            THROW<Hccl::InternalException>(
                StringFormat("InstructionType is not ccu ins, algName = [%s]", algName.c_str()));
        }

        // 获取taskParam
        const CcuInstruction& ccuInstruction = static_cast<const CcuInstruction&>(*insQueue->Iter());
        std::vector<std::vector<CcuTaskParam>> taskParams;
        ccuInstruction.Translate(taskParams);
        if (taskParams.empty()) {
            THROW<Hccl::InternalException>(
                StringFormat("CcuInstruction translate failed, index = [%u], algName = [%s]", index, algName.c_str()));
        }
        algoTemplateMap[templateSign] = taskParams;
        SaveAlgoInfo(index, templateSign, commConfig.opType, comm->GetAlgorithmType());
        for (const auto& task : taskParams) {
            HCCL_INFO("taskParam: dieId = [%u], instStartId = [%u]", task[0].dieId, task[0].instStartId);
            SaveMc2DfxTaskInfo(task[0], ccuInstruction.GetExecId());
        }
    }
    HCCL_INFO("GenerateAlgoTemplates success");
}

void Mc2Compont::GenerateAlgoTemplatesV2(
    const Mc2InitTilingInner* mc2TilingPtr, std::unordered_set<uint64_t>& algoTemplateRequire)
{
    HCCL_INFO("GenerateAlgoTemplates start v2");

    auto tmpMemSize = comm->GetBufferSize();
    CollAlgParams params;
    params.opMode = OpMode::OPBASE;
    params.maxTmpMemSize = tmpMemSize;
    params.isMc2 = true;
    if (mc2TilingPtr->mc2HcommCnt > MAX_OP_NUM) {
        THROW<Hccl::InternalException>(
            StringFormat("mc2HcommCnt is lager than MAX_OP_NUM, mc2HcommCnt = [%u]", mc2TilingPtr->mc2HcommCnt));
    }

    for (uint32_t index = 0; index < mc2TilingPtr->mc2HcommCnt; index++) {
        const auto offset = mc2TilingPtr->offset[index];
        const auto& commConfig
            = *(reinterpret_cast<const Mc2CcTilingInner*>(reinterpret_cast<const uint8_t*>(mc2TilingPtr) + offset));
        OpParamsChecker::CheckOpDataTypeMC2V2(commConfig);
        uint64_t templateSign = GetTemplateSignatureV2(commConfig);
        algoTemplateRequire.insert(templateSign);
        // 已经生成过的算法模板不再生成
        if (algoTemplateMap.find(templateSign) != algoTemplateMap.end()) {
            HCCL_INFO(
                "A algoTemplate that meets the requirement already exists, index = [%u], templateSign = [%llu]", index,
                templateSign);
            if (algoInfoMap_.find(templateSign) != algoInfoMap_.end()) {
                combinOpParam.opType[index] = algoInfoMap_[templateSign].opType;
                combinOpParam.algorithmType[index] = algoInfoMap_[templateSign].algorithmType;
                continue;
            } else {
                THROW<Hccl::InternalException>(
                    StringFormat("algoInfoMap_ do not has templateSign = [%llu]", templateSign));
            }
        }

        FillCollOperatorV2(commConfig);

        auto insQueue = make_shared<InsQueue>();
        MC2AllocCommRes(params, insQueue, commConfig.communicationEngine);

        std::string algName = comm->GetCurAlgName();
        HCCL_INFO(
            "Orchestrate: index = [%u], algName = [%s], templateSign = [%llu]", index, algName.c_str(), templateSign);
        if (insQueue->Iter()->GetType() != InstructionType::CCU_INS) {
            THROW<Hccl::InternalException>(
                StringFormat("InstructionType is not ccu ins, algName = [%s]", algName.c_str()));
        }

        // 获取taskParam
        const CcuInstruction& ccuInstruction = static_cast<const CcuInstruction&>(*insQueue->Iter());
        std::vector<std::vector<CcuTaskParam>> taskParams;
        ccuInstruction.Translate(taskParams);
        if (taskParams.empty()) {
            THROW<Hccl::InternalException>(
                StringFormat("CcuInstruction translate failed, index = [%u], algName = [%s]", index, algName.c_str()));
        }
        algoTemplateMap[templateSign] = taskParams;
        SaveAlgoInfo(index, templateSign, commConfig.opType, comm->GetAlgorithmType());
        for (const auto& task : taskParams) {
            HCCL_INFO("taskParam: dieId = [%u], instStartId = [%u]", task[0].dieId, task[0].instStartId);
            SaveMc2DfxTaskInfo(task[0], ccuInstruction.GetExecId());
        }
    }
    HCCL_INFO("GenerateAlgoTemplates success");
}

static std::map<uint8_t, std::map<uint32_t, uint32_t>> TransToMap(const std::vector<std::vector<CcuTaskParam>>& params)
{
    std::map<uint8_t, std::map<uint32_t, uint32_t>> dieIdToInstrIdMap;

    for (const auto& param : params) {
        uint8_t dieId = param[0].dieId;
        uint32_t instrId = param[0].instStartId;

        // 检查 dieId 是否已经存在于 map 中
        if (dieIdToInstrIdMap.find(dieId) == dieIdToInstrIdMap.end()) {
            // 如果不存在，创建一个新的 map
            std::map<uint32_t, uint32_t> indexIdToInstrIdMap;
            indexIdToInstrIdMap[0] = instrId;
            dieIdToInstrIdMap[dieId] = indexIdToInstrIdMap;
        } else {
            // 如果存在，获取对应的 map
            std::map<uint32_t, uint32_t>& indexIdToInstrIdMap = dieIdToInstrIdMap[dieId];
            // 插入新的 IndexId 和 InstrId
            int indexId = indexIdToInstrIdMap.size();
            indexIdToInstrIdMap[indexId] = instrId;
        }
    }
    return dieIdToInstrIdMap;
}

static std::map<uint8_t, std::vector<uint32_t>>
TransToDieIdMissionIdMap(const std::vector<std::vector<CcuTaskParam>>& params)
{
    std::map<uint8_t, std::vector<uint32_t>> dieIdMissionIdMap;

    for (const auto& param : params) {
        uint8_t dieId = param[0].dieId;
        uint32_t missionId = param[0].missionId;

        if (dieIdMissionIdMap.find(dieId) == dieIdMissionIdMap.end()) {
            dieIdMissionIdMap[dieId] = std::vector<uint32_t>();
        }
        dieIdMissionIdMap[dieId].push_back(missionId);
    }
    return dieIdMissionIdMap;
}

bool Mc2Compont::CompareMissionMap(
    const std::map<uint8_t, std::map<uint32_t, uint32_t>>& mapA,
    const std::map<uint8_t, std::map<uint32_t, uint32_t>>& mapB) const
{
    if (mapA.size() != mapB.size()) {
        return false;
    }
    for (auto& entry : mapA) {
        uint8_t curDieId = entry.first;
        if (mapB.find(curDieId) == mapB.end()) {
            return false;
        }
        const std::map<uint32_t, uint32_t>& curSubMapA = entry.second;
        const std::map<uint32_t, uint32_t>& curSubMapB = mapB.at(curDieId);
        if (curSubMapA.size() != curSubMapB.size()) {
            return false;
        }
        for (auto& elem : curSubMapA) {
            if (curSubMapB.find(elem.first) == curSubMapB.end()) {
                return false;
            }
        }
    }
    return true;
}

u32 Mc2Compont::GetCcuMc2ServerNum() { return ccuServerMap.size(); }

void Mc2Compont::GenerateCcuServer(const std::unordered_set<uint64_t>& algoTemplateRequire)
{
    HCCL_INFO("GenerateCcuServer start");
    if (algoTemplateRequire.empty()) {
        THROW<InvalidParamsException>(StringFormat("AlgoTemplate require num is zero!"));
    }

    InsExeQue::ExtInsExeEntityId execId;
    // 查找当前是否存在符合条件的ccuServer
    if (FindCcuServer(algoTemplateRequire, execId)) {
        curExecId = execId;
        HCCL_INFO("A CcuServer that meets the requirement already exists, execId = [%llu]", execId);
        return;
    }

    // 没有符合条件的ccuServer, 生成一个新的ccuServer
    std::map<uint64_t, std::map<uint8_t, std::map<uint32_t, uint32_t>>> signatureMap;
    std::map<uint8_t, std::map<uint32_t, uint32_t>> compareMap;
    std::map<uint8_t, std::vector<uint32_t>> dieIdMissionIdMap;
    bool initFlag = false;
    for (uint64_t templateSignature : algoTemplateRequire) {
        auto tmpMap = TransToMap(algoTemplateMap[templateSignature]);
        if (!initFlag) {
            compareMap = tmpMap;
            dieIdMissionIdMap = TransToDieIdMissionIdMap(algoTemplateMap[templateSignature]);
            initFlag = true;
        } else {
            if (!CompareMissionMap(compareMap, tmpMap)) {
                THROW<InvalidParamsException>(StringFormat("AlgoTemplate require is not the same!"));
            }
        }
        // algoTemplateRequire为unordered_set，可以保证键值唯一
        signatureMap[templateSignature] = tmpMap;
    }

    std::map<uint8_t, std::map<uint32_t, std::map<uint64_t, uint32_t>>> algoTemplate;
    for (const auto& signature : signatureMap) {
        for (const auto& dieId : signature.second) {
            for (const auto& indexId : dieId.second) {
                algoTemplate[dieId.first][indexId.first][signature.first] = indexId.second;
            }
        }
    }

    // 实例化Mc2Context
    CcuCtxGroup ctxGroup;
    uint32_t dieNum = algoTemplate.size();
    for (const auto& item : algoTemplate) {
        uint8_t dieId = item.first;
        for (const auto& mission : item.second) {
            std::unique_ptr<Mc2ContextBase> mc2Context;
            if (mission.first == 0) {
                mc2Context = std::make_unique<Mc2Context>();
                mc2Context->SetDieId(dieId);
                static_cast<Mc2Context*>(mc2Context.get())->SetDieNum(dieNum);
                static_cast<Mc2Context*>(mc2Context.get())
                    ->SetCommAddr(
                        static_cast<uint64_t>(comSyncBuffer->GetAddr()),
                        static_cast<uint64_t>(comParamBuffer->GetAddr()));
            } else {
                mc2Context = std::make_unique<Mc2SlaveContext>();
                mc2Context->SetDieId(dieId);
            }
            mc2Context->SetMissionNumAndId(item.second.size(), mission.first);
            mc2Context->SetAlgoTemplateInfo(mission.second);
            ctxGroup.ctxs.push_back(std::move(mc2Context));
        }
    }

    // 申请资源
    HcclResult ret = CcuCtxMgr::AllocRes(comm->GetDeviceLogicId(), ctxGroup, ccuResPack);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GenerateCcuServer AllocRes failed, ret[%d]", ret);
    }

    // needtodo 检查各个templateSignature对应的missionId相同
    std::map<uint8_t, uint32_t> dieIndex;
    for (auto& ctx : ctxGroup.ctxs) {
        if (dieIndex.find(ctx->GetDieId()) == dieIndex.end()) {
            dieIndex[ctx->GetDieId()] = 0;
        }
        ctx->SetMissionId(dieIdMissionIdMap[ctx->GetDieId()][dieIndex[ctx->GetDieId()]++]);
    }

    // 指令注册
    InsExeQue::ExtInsExeEntity entity;
    entity.ctxGroup = std::move(ctxGroup);
    ret = InsExeQue::RegisterExtendInstruction(comm->GetDeviceLogicId(), entity, execId);
    if (ret != HcclResult::HCCL_SUCCESS) {
        THROW<CcuApiException>("GenerateCcuServer RegisterExtendInstruction failed, ret[%d]", ret);
    }
    ccuServerMap[execId] = algoTemplateRequire;
    curExecId = execId;
    HCCL_INFO("GenerateCcuServer success, execId[%llu]", execId);
}

void Mc2Compont::DisplayRPCMsg() const
{
    void* xnAddr = nullptr;
    try {
        xnAddr = HrtMallocHost(comParamBuffer->GetSize() + comSyncBuffer->GetSize());
        void* ckeAddr = reinterpret_cast<uint8_t*>(xnAddr) + comParamBuffer->GetSize();
        HrtMemcpy(
            xnAddr, comParamBuffer->GetSize(), reinterpret_cast<uint8_t*>(comParamBuffer->GetAddr()),
            comParamBuffer->GetSize(), RT_MEMCPY_DEVICE_TO_HOST);
        HrtMemcpy(
            ckeAddr, comSyncBuffer->GetSize(), reinterpret_cast<uint8_t*>(comSyncBuffer->GetAddr()),
            comSyncBuffer->GetSize(), RT_MEMCPY_DEVICE_TO_HOST);
    } catch (const RuntimeApiException& e) {
        HCCL_ERROR("[Mc2Compont:%s] host memory operation fail: %s", __func__, e.what());
        if (xnAddr != nullptr) {
            HrtFreeHost(xnAddr);
        }
        return;
    }
    void* ckeAddr = static_cast<uint8_t*>(xnAddr) + comParamBuffer->GetSize();
    uint64_t* xnBaseAddr = static_cast<uint64_t*>(xnAddr);
    uint64_t* waitCkeAddr = static_cast<uint64_t*>(ckeAddr);
    uint64_t* setCkeAddr = static_cast<uint64_t*>(ckeAddr) + CCU_TASK_NUM_MAX;
    uint32_t xnNumPerMsg = CCU_PARAM_NUM_MAX;
    for (uint32_t msgId = 0; msgId < CCU_TASK_NUM_MAX; msgId++) {
        uint64_t* xnMsg = xnBaseAddr + msgId * xnNumPerMsg;
        uint64_t* waitCkeMsg = waitCkeAddr + msgId;
        uint64_t* setCkeMsg = setCkeAddr + msgId;
        // print cke value and xn value
        HCCL_ERROR(
            "msgId[%u]: waitCkeAddr=[%p], setCkeAddr=[%p], waitCke=[%llu], setCke=[%llu], xnAddr=[%p]", msgId,
            waitCkeMsg, setCkeMsg, *waitCkeMsg, *setCkeMsg, xnMsg);
        std::string tmpXnInfo = "xnMsg=[";
        for (uint32_t xnId = 0; xnId < xnNumPerMsg; xnId++) {
            tmpXnInfo += StringFormat("%llx", *(xnMsg + xnId));
            if (xnId != xnNumPerMsg - 1) {
                tmpXnInfo += ", ";
            }
        }
        tmpXnInfo += "]";
        HCCL_ERROR("[Mc2Compont:%s] RPC MsgId[%u], %s", __func__, msgId, tmpXnInfo.c_str());
    }
    HrtFreeHost(xnAddr);
}

bool Mc2Compont::FindCcuServer(
    const std::unordered_set<uint64_t>& algoTemplateRequire, InsExeQue::ExtInsExeEntityId& execId) const
{
    // 查找是否有符合的ccuServer存在，本次算子需求是已有sever中包含算子的子集既符合要求
    for (const auto& server : ccuServerMap) {
        bool isMatch = true;
        for (auto& templateSign : algoTemplateRequire) {
            if (server.second.find(templateSign) == server.second.end()) {
                isMatch = false;
                break;
            }
        }
        if (isMatch) {
            execId = server.first;
            return true;
        }
    }
    return false;
}

uint64_t Mc2Compont::GetTemplateSignature(const Mc2CommConfig& config) const
{
    // 根据Mc2CommConfig生成算子模板签名
    if (config.opType > UINT8_MAX || config.reduceType > UINT8_MAX || config.dataType > UINT8_MAX
        || config.outputDataType > UINT8_MAX) {
        THROW<InvalidParamsException>(
            StringFormat("MC2 High Level API GetTemplateSignature Failed, Mc2CommConfig value is bigger than 256!"));
    }
    constexpr uint16_t algoTypeShift = 32;
    constexpr uint16_t outputDataTypeShift = 24;
    constexpr uint16_t reduceTypeShift = 16;
    constexpr uint16_t dataTypeShift = 8;
    constexpr uint16_t opTypeShift = 0;
    uint64_t opType = config.opType;
    uint64_t dataType = config.dataType;
    uint64_t reduceType = config.reduceType;
    uint64_t outputDataType = config.outputDataType;
    uint64_t algoType = 0; // 用于算法选择，当前暂不支持，固定为0

    uint64_t templateSignature = ((opType & 0xff) << opTypeShift) | ((dataType & 0xff) << dataTypeShift)
                                 | ((reduceType & 0xff) << reduceTypeShift)
                                 | ((outputDataType & 0xff) << outputDataTypeShift)
                                 | ((algoType & 0xff) << algoTypeShift);
    HCCL_INFO(
        "[GetTemplateSignature]: opType[%s] dataType[%s] reduceType[%s] outputDataType[%s] algoType[%u] "
        "templateSignature[%llu]",
        MC2OpType(static_cast<AicpuComType>(config.opType)).Describe().c_str(),
        MC2DataType(static_cast<HcclDataType>(config.dataType)).Describe().c_str(),
        MC2ReduceType(static_cast<HcclReduceOp>(config.reduceType)).Describe().c_str(),
        MC2DataType(static_cast<HcclDataType>(config.outputDataType)).Describe().c_str(), algoType, templateSignature);
    return templateSignature;
}

uint64_t Mc2Compont::GetTemplateSignatureV2(const Mc2CcTilingInner& config) const
{
    // 根据Mc2CommConfig生成算子模板签名
    if (config.opType > UINT8_MAX || config.reduceType > UINT8_MAX) {
        THROW<InvalidParamsException>(
            StringFormat("MC2 High Level API GetTemplateSignature Failed, Mc2CommConfig value is bigger than 256!"));
    }
    constexpr uint16_t algoTypeShift = 32;
    constexpr uint16_t outputDataTypeShift = 24;
    constexpr uint16_t reduceTypeShift = 16;
    constexpr uint16_t dataTypeShift = 8;
    constexpr uint16_t opTypeShift = 0;
    uint64_t opType = config.opType;
    uint64_t dataType = config.srcDataType;
    uint64_t reduceType = config.reduceType;
    uint64_t outputDataType = config.dstDataType;
    uint64_t algoType = 0; // 用于算法选择，当前暂不支持，固定为0

    uint64_t templateSignature = ((opType & 0xff) << opTypeShift) | ((dataType & 0xff) << dataTypeShift)
                                 | ((reduceType & 0xff) << reduceTypeShift)
                                 | ((outputDataType & 0xff) << outputDataTypeShift)
                                 | ((algoType & 0xff) << algoTypeShift);
    HCCL_INFO(
        "[GetTemplateSignature]: opType[%s] dataType[%s] reduceType[%s] outputDataType[%s] algoType[%u] "
        "templateSignature[%llu]",
        MC2OpType(static_cast<AicpuComType>(config.opType)).Describe().c_str(),
        MC2DataType(static_cast<HcclDataType>(config.srcDataType)).Describe().c_str(),
        MC2ReduceType(static_cast<HcclReduceOp>(config.reduceType)).Describe().c_str(),
        MC2DataType(static_cast<HcclDataType>(config.dstDataType)).Describe().c_str(), algoType, templateSignature);
    return templateSignature;
}

void Mc2Compont::FillCollOperatorV2(const Mc2CcTilingInner& config)
{
    CollOpParams opParams;
    opParams.opType = MC2OpType(static_cast<AicpuComType>(config.opType));
    opParams.reduceOp = MC2ReduceType(static_cast<HcclReduceOp>(config.reduceType));
    opParams.dataType = MC2DataType(static_cast<HcclDataType>(config.srcDataType));
    opParams.outputDataType = MC2DataType(static_cast<HcclDataType>(config.dstDataType));
    opParams.count = dataCount;
    opParams.sendBuf = reinterpret_cast<void*>(inputMem->GetAddr());
    opParams.recvBuf = reinterpret_cast<void*>(inputMem->GetAddr());
    if (opParams.opType == OpType::ALLTOALL) {
        opParams.all2AllDataDes.sendType = opParams.dataType;
        opParams.all2AllDataDes.recvType = opParams.outputDataType;
        opParams.all2AllDataDes.sendCount = dataCount;
        opParams.all2AllDataDes.recvCount = dataCount;
    }
    std::string opTag = comm->GetId();

    if (opParams.opType == OpType::ALLTOALLV) {
        opParams.all2AllVDataDes.sendType = opParams.dataType;
        opParams.all2AllVDataDes.recvType = opParams.outputDataType;
        dataCounts.resize(comm->GetRankSize());
        displs.resize(comm->GetRankSize());
        u64 countSum = 0;
        for (u32 i = 0; i < comm->GetRankSize(); i++) {
            dataCounts.at(i) = 1;
            displs.at(i) = countSum++;
        }
        opParams.all2AllVDataDes.sendCounts = reinterpret_cast<void*>(&dataCounts[0]);
        opParams.all2AllVDataDes.recvCounts = reinterpret_cast<void*>(&dataCounts[0]);
        opParams.all2AllVDataDes.sdispls = reinterpret_cast<void*>(&displs[0]);
        opParams.all2AllVDataDes.rdispls = reinterpret_cast<void*>(&displs[0]);
    }

    comm->CovertToCurrentCollOperator(opTag, opParams, OpMode::OPBASE);
}

void Mc2Compont::FillCollOperator(const Mc2CommConfig& config)
{
    CollOpParams opParams;
    opParams.opType = MC2OpType(static_cast<AicpuComType>(config.opType));
    opParams.reduceOp = MC2ReduceType(static_cast<HcclReduceOp>(config.reduceType));
    opParams.dataType = MC2DataType(static_cast<HcclDataType>(config.dataType));
    opParams.outputDataType = MC2DataType(static_cast<HcclDataType>(config.outputDataType));
    opParams.count = dataCount;
    opParams.sendBuf = reinterpret_cast<void*>(inputMem->GetAddr());
    opParams.recvBuf = reinterpret_cast<void*>(inputMem->GetAddr());
    if (opParams.opType == OpType::ALLTOALL) {
        opParams.all2AllDataDes.sendType = opParams.dataType;
        opParams.all2AllDataDes.recvType = opParams.outputDataType;
        opParams.all2AllDataDes.sendCount = dataCount;
        opParams.all2AllDataDes.recvCount = dataCount;
    }
    std::string opTag = comm->GetId();

    if (opParams.opType == OpType::ALLTOALLV) {
        opParams.all2AllVDataDes.sendType = opParams.dataType;
        opParams.all2AllVDataDes.recvType = opParams.outputDataType;
        dataCounts.resize(comm->GetRankSize());
        displs.resize(comm->GetRankSize());
        u64 countSum = 0;
        for (u32 i = 0; i < comm->GetRankSize(); i++) {
            dataCounts.at(i) = 1;
            displs.at(i) = countSum++;
        }
        opParams.all2AllVDataDes.sendCounts = reinterpret_cast<void*>(&dataCounts[0]);
        opParams.all2AllVDataDes.recvCounts = reinterpret_cast<void*>(&dataCounts[0]);
        opParams.all2AllVDataDes.sdispls = reinterpret_cast<void*>(&displs[0]);
        opParams.all2AllVDataDes.rdispls = reinterpret_cast<void*>(&displs[0]);
    }

    comm->CovertToCurrentCollOperator(opTag, opParams, OpMode::OPBASE);
}

void Mc2Compont::SaveMc2DfxTaskInfo(const CcuTaskParam& ccuTaskParam, uint64_t execId) const
{
    shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
    dfxOpInfo->comm_ = comm;

    TaskParam taskParam{};
    taskParam.taskType = TaskParamType::TASK_CCU;
    taskParam.taskPara.Ccu.dieId = ccuTaskParam.dieId;
    taskParam.taskPara.Ccu.missionId = ccuTaskParam.missionId;
    taskParam.taskPara.Ccu.instrId = ccuTaskParam.instStartId;
    taskParam.taskPara.Ccu.executeId = execId;

    shared_ptr<TaskInfo> taskInfo = std::make_shared<TaskInfo>(0, 0, 0, taskParam, dfxOpInfo);

    MC2GlobalMirrorTasks::GetInstance().AddTaskInfo(comm->GetDeviceLogicId(), taskInfo);
}

std::vector<CcuTaskParam> Mc2Compont::GetAlgoCcuTaskInfo(InsExeQue::ExtInsExeEntityId execId) const
{
    std::vector<CcuTaskParam> ccuTaskParam{};
    auto serverItor = ccuServerMap.find(execId);
    if (serverItor == ccuServerMap.end()) {
        HCCL_WARNING("[Mc2Compont]Failed to find ccuServer by executeId[%llu]", execId);
        return ccuTaskParam;
    }
    for (uint64_t algoSign : serverItor->second) {
        auto algoTemplateItor = algoTemplateMap.find(algoSign);
        if (algoTemplateItor == algoTemplateMap.end()) {
            HCCL_WARNING("[Mc2Compont]Failed to find ccuTaskParam by algoSign[%llu]", algoSign);
            continue;
        }
        for (const auto& taskParam : algoTemplateItor->second) {
            ccuTaskParam.push_back(taskParam[0]);
        }
    }
    return ccuTaskParam;
}
} // namespace Hccl
