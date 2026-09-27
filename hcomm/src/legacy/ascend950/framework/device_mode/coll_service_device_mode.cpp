/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_service_device_mode.h"
#include "exception_util.h"
#include "communicator_impl.h"
#include "ccu_dev_mgr.h"
#include "types.h"
#include "aiv_ins.h"
#include "stream_utils.h"
#include "orion_adapter_rts.h"

namespace Hccl {

constexpr u32 SIZE_TABLE_ORION[HCCL_DATA_TYPE_RESERVED]
    = {sizeof(s8),
       sizeof(s16),
       sizeof(s32),
       2,
       sizeof(float),
       sizeof(s64),
       sizeof(u64),
       sizeof(u8),
       sizeof(u16),
       sizeof(u32),
       8,
       2,
       16,
       2,
       1,
       1,
       1,
       1};

void CollServiceDeviceMode::Init()
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    // 算子执行次数计数器buffer申请
    AddOpCounterMems();
    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

static void AddCcuInsAndAicpuInsLinks(std::vector<LinkData>& linkDatas, const Instruction& ins)
{
    InstructionType insType = ins.GetType();
    std::vector<LinkData> tmpLinkDatas;
    if (insType == InstructionType::CCU_INS) {
        const CcuInstruction& ccuIns = dynamic_cast<const CcuInstruction&>(ins);
        tmpLinkDatas = ccuIns.GetLinks();
    } else if (insType == InstructionType::AICPU_INS) {
        const AicpuInstruction& aicpuIns = dynamic_cast<const AicpuInstruction&>(ins);
        tmpLinkDatas = aicpuIns.GetLinks();
    } else if (insType == InstructionType::AIV_INS) {
        const AivInstruction& aivIns = dynamic_cast<const AivInstruction&>(ins);
        tmpLinkDatas = aivIns.GetLinks();
    }
    linkDatas.insert(linkDatas.end(), tmpLinkDatas.begin(), tmpLinkDatas.end());
}

std::vector<LinkData> CollServiceDeviceMode::GetUniqueLinks(std::shared_ptr<InsQueue>& insQueue) const
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    // 返回队列中所有ins的links
    std::vector<LinkData> links;
    for (auto slaveIter = insQueue->IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        for (auto ins = slaveIter->Iter(); ins.HasNext(); ++ins) {
            AddCcuInsAndAicpuInsLinks(links, *ins);
        }
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] slaves end.", __func__);

    for (auto ins = insQueue->Iter(); ins.HasNext(); ++ins) {
        AddCcuInsAndAicpuInsLinks(links, *ins);
    }

    std::unordered_set<LinkData> linkDataSet(links.begin(), links.end());
    links.assign(linkDataSet.begin(), linkDataSet.end());

    HCCL_INFO("[CollServiceDeviceMode::%s] end, links size[%zu]", __func__, links.size());
    return links;
}

void CollServiceDeviceMode::LoadWithOpBasedMode(CollOperator& op, std::unique_ptr<Stream> stream)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    // AIV aclgrah 流程
    if (comm->GetOpExecuteConfig().accState == AcceleratorState::AIV
        || comm->GetOpExecuteConfig().accState == AcceleratorState::AIV_ONLY) {
        HandleAclGraphFirstOpAivBuff(stream->GetPtr());
    }

    // 入参buffer和stream注册
    RegisterOpBufToBufMgr(op);

    RegisterOpbasedStream(std::move(stream));

    if (comm->GetOpExecuteConfig().accState == AcceleratorState::AIV
        || comm->GetOpExecuteConfig().accState == AcceleratorState::AIV_ONLY) {
        auto insQueue = make_shared<InsQueue>();

        AivOpCacheArgs opCacheParam{
            comm->GetCurAlgName(), op.dataCount,      op.dataType, op.opType, op.reduceOp, op.root,
            op.numBlocksLimit,     op.outputDataType, {},          {}};
        if (op.opType == OpType::ALLTOALL) {
            opCacheParam.all2allDataDes
                = {op.all2AllDataDes.sendType, op.all2AllDataDes.recvType, op.all2AllDataDes.sendCount,
                   op.all2AllDataDes.recvCount};
        }
        if (op.opType == OpType::ALLTOALLV) {
            opCacheParam.all2allVDataDes
                = {op.all2AllVDataDes.sendType,   op.all2AllVDataDes.recvType, op.all2AllVDataDes.sendCounts,
                   op.all2AllVDataDes.recvCounts, op.all2AllVDataDes.sdispls,  op.all2AllVDataDes.rdispls};
        }
        auto it = comm->hcclCacheMap_.find(opCacheParam);
        bool isCache = false;
        bool isSendRecv
            = ((op.opType == OpType::SEND) || (op.opType == OpType::RECV) || (op.opType == OpType::BATCHSENDRECV));
        bool isAlltoAllV = (op.opType == OpType::ALLTOALLV);
        if ((it != comm->hcclCacheMap_.end()) && (!isSendRecv) && (!isAlltoAllV)) {
            isCache = true;
            insQueue = it->second;
        } else {
            // 算法编排返回insQueue, 包含ccu扩展指令和aicpu扩展指令
            insQueue = Orchestrate(op);
        }
        AllocQueueNotify(*insQueue);
        // 日志打印
        if (comm->GetAivTag() == 1) {
            std::vector<LinkData> uniqueLinks = comm->GetFullMeshLinks();
            comm->SetCommStatus(CommStatus::COMM_BUILDING);
            // Socket建链
            comm->GetSocketManager().BatchCreateSockets(uniqueLinks);
            aivInsPreprocessor.Preprocess(insQueue);
        }
        // translate
        SaveMirrorDfxOpInfo();
        Interpreter interpreter(*comm);
        interpreter.Submit(*insQueue);
        if (!isCache) {
            comm->GetCacheMap(opCacheParam, insQueue);
        }
    } else {
        // 用于aicpu专用流
        comm->GetAicpuStreamManager().AllocFreeStream();
        // 算法编排返回insQueue, 包含ccu扩展指令和aicpu扩展指令
        shared_ptr<InsQueue> insQueue = Orchestrate(op);
        AllocQueueNotify(*insQueue);
        // 获取insQueue中所有Ins的linkDats
        std::vector<LinkData> uniqueLinks = GetUniqueLinks(insQueue);
        // 将通讯域设置为transport建链中状态
        comm->SetCommStatus(CommStatus::COMM_BUILDING);

        // Socket建链
        comm->GetSocketManager().BatchCreateSockets(uniqueLinks);

        // 对insQueue中ccuIns进行预处理(transport建链和交换, 资源申请、注册等)
        ccuInsPreprocessor.Preprocess(insQueue);

        if (ccuInsPreprocessor.IsRollback()) { // 如果是回退，流程退出
            return;
        }
        SaveMirrorDfxOpInfo();
        // translate
        Interpreter interpreter(*comm);
        interpreter.Submit(*insQueue);
    }
    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

void CollServiceDeviceMode::LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    RegisterOpBufToBufMgr(op);

    RegisterOffloadMasterStream(op.opTag, std::move(stream));

    // 算法编排返回insQueue, 包含ccu扩展指令和aicpu扩展指令
    shared_ptr<InsQueue> insQueue = Orchestrate(op);

    AllocQueueNotify(*insQueue);

    // 获取insQueue中所有Ins的linkDats
    std::vector<LinkData> uniqueLinks = GetUniqueLinks(insQueue);

    // 将通讯域设置为transport建链中状态
    comm->SetCommStatus(CommStatus::COMM_BUILDING);

    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(uniqueLinks);

    // 对insQueue中ccuIns进行预处理(transport建链和交换, 资源申请、注册等)
    aivInsPreprocessor.Preprocess(insQueue);
    ccuInsPreprocessor.Preprocess(insQueue);

    if (ccuInsPreprocessor.IsRollback()) { // 如果是回退，流程退出
        return;
    }

    SaveMirrorDfxOpInfo();

    // 下发head算子执行计数器task
    AddCountTask(true);

    // translate
    Interpreter interpreter(*comm);
    interpreter.Submit(*insQueue);

    // 下发tail算子执行计数器task
    AddCountTask(false);

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

shared_ptr<InsQueue> CollServiceDeviceMode::Orchestrate(const CollAlgOperator& op) const
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    u64 tmpMemSize = 0;
    if (op.opMode == OpMode::OPBASE || comm->GetOpExecuteConfig().accState == AcceleratorState::AIV
        || comm->GetOpExecuteConfig().accState == AcceleratorState::AIV_ONLY) {
        tmpMemSize = comm->GetBufferSize();
    } else if (op.scratchMem != nullptr) {
        tmpMemSize = op.scratchMem->GetSize();
    } else {
        HCCL_WARNING("[CollServiceDeviceMode::%s] no need scratchMem.", __func__);
    }
    CollAlgParams params;
    auto insQueue = make_shared<InsQueue>();

    params.opMode = op.opMode;
    params.maxTmpMemSize = tmpMemSize;
    HCCL_INFO("[CollServiceDeviceMode::%s] opMode[%d], tmpMemSize[%llu]", __func__, op.opMode, tmpMemSize);
    HcclResult errCode = comm->GetCollAlgComponent()->Orchestrate(op, params, comm->GetCurAlgName(), insQueue);
    HCCL_INFO("[CollServiceDeviceMode::%s] orchestrate with Ins end", __func__);

    if (errCode != HcclResult::HCCL_SUCCESS) {
        auto msg = StringFormat("Error occurs when call collAlgComponent.orchestrate(), error code: %d", errCode);
        THROW<InternalException>(msg);
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
    return insQueue;
}

void CollServiceDeviceMode::RecoverTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    // ccu支持快照保存和恢复
    RecoverCcuTransport(links, linkGroupPair);

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

void CollServiceDeviceMode::RecoverCcuTransport(
    vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    // 创建TransPort所需的Socket
    comm->GetSocketManager().BatchCreateSockets(links);

    auto ret = GetCcuInsPreprocessor()->RecoverCcuTransportCtx(links, linkGroupPair);
    if (ret != HcclResult::HCCL_SUCCESS) {
        auto msg = StringFormat("Error occurs when call CollServiceDeviceMode::%s, error code: %d", __func__, ret);
        THROW<InternalException>(msg);
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

// 功能说明：等待transport建链完成
// 输入说明：string &opTag：通信域ID，唯一标记一个通信域
bool CollServiceDeviceMode::IsAllTransportRecoveredReady(const std::string& opTag)
{
    auto ret = GetCcuInsPreprocessor()->RecoverCcuTransportConfirm();
    if (ret == HcclResult::HCCL_SUCCESS) {
        HCCL_INFO(
            "[CollServiceDeviceMode][IsAllTransportRecoveredReady] opTag[%s] recover transport success", opTag.c_str());
        return true;
    }

    HCCL_ERROR("[CollServiceDeviceMode][IsAllTransportRecoveredReady] fail, ret[%d]", ret);
    return false;
}

constexpr u32 TEMP_UES_CNTCKE_NUM = 16;

HcclResult CollServiceDeviceMode::GetSnapShotDynamicBuf(CollOperator& op, BinaryStream& buf)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    auto req = comm->GetCollAlgComponent()->GetCollAlgOpReq(op, comm->GetCurAlgName());
    HCCL_INFO(
        "CollServiceAiCpuImpl::GetSnapShotDynamicBuf GetCollAlgOpReq OrchestMode::INSTRUCTION, algName %s",
        req.algName.c_str());
    buf << req.resReq.levelRankPairs.size();
    for (auto levelRankPair : req.resReq.levelRankPairs) {
        buf << levelRankPair.first << levelRankPair.second;
    }

    auto transportLinkGroup = ccuInsPreprocessor.GetCcuComm()->GetCcuTransportGrpMgr()->GetAllTransportGroups();
    vector<std::pair<LinkGroup, u32>> linkGroupPairs;

    //  临时规避多轮不同算子导致CNTCKE资源不足，cntCkeNum采用硬编码形式，待后续正式方案修改
    for (LinkGroup& group : transportLinkGroup) {
        linkGroupPairs.push_back({group, TEMP_UES_CNTCKE_NUM});
    }
    buf << linkGroupPairs.size();
    HCCL_INFO("[CollServiceDeviceMode::%s] linkGroupPairs size[%zu].", __func__, linkGroupPairs.size());
    for (auto linkGroupPair : linkGroupPairs) {
        LinkGroup& linkGroup = linkGroupPair.first;
        u32 cntCkeNum = linkGroupPair.second;
        buf << linkGroup.GetLinks().size();
        HCCL_INFO("[CollServiceDeviceMode::%s] linkGroup size[%zu].", __func__, linkGroup.GetLinks().size());
        for (auto& linkInfo : linkGroup.GetLinks()) {
            buf << linkInfo.rankId << linkInfo.dieId;
            linkInfo.localAddr.GetBinStream(buf);
            linkInfo.remoteAddr.GetBinStream(buf);
            HCCL_INFO(
                "[CollServiceDeviceMode::%s] rankId[%d], dieId[%u], localAddr[%s], remoteAddr[%s].", __func__,
                linkInfo.rankId, linkInfo.dieId, linkInfo.localAddr.Describe().c_str(),
                linkInfo.remoteAddr.Describe().c_str());
        }
        buf << cntCkeNum;
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
    return HcclResult::HCCL_SUCCESS;
}

void CollServiceDeviceMode::AllocCommResource(
    void* mc2Tiling, void** commContext, const AcceleratorState& tilingAccelerator)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    if (tilingAccelerator == AcceleratorState::AIV || tilingAccelerator == AcceleratorState::AIV_ONLY) {
        aivMc2Compont.AllocCommResource(mc2Tiling, commContext);
    } else {
        mc2Compont.AllocCommResource(mc2Tiling, commContext);
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

void CollServiceDeviceMode::GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);

    auto taskParams = mc2Compont.GetCcuTaskInfo(tilingData);
    if (taskParams.size() > FUSION_SUB_TASK_MAX_CCU_NUM) {
        THROW<InternalException>(StringFormat(
            "Get %d task params, which is bigger than the maximum size %d.", taskParams.size(),
            FUSION_SUB_TASK_MAX_CCU_NUM));
    }

    auto group = static_cast<rtCcuTaskGroup_t*>(ccuTaskGroup);
    group->taskNum = taskParams.size();

    for (size_t index = 0; index < taskParams.size(); ++index) {
        group->ccuTaskInfo[index].dieId = taskParams[index].dieId;
        group->ccuTaskInfo[index].missionId = taskParams[index].missionId;
        group->ccuTaskInfo[index].timeout = taskParams[index].timeout;
        group->ccuTaskInfo[index].instStartId = taskParams[index].instStartId;
        group->ccuTaskInfo[index].instCnt = taskParams[index].instCnt;
        group->ccuTaskInfo[index].key = taskParams[index].key;
        group->ccuTaskInfo[index].argSize = taskParams[index].argSize;
        std::copy(
            std::begin(taskParams[index].args), std::end(taskParams[index].args),
            std::begin(group->ccuTaskInfo[index].args));
        HCCL_INFO(
            "ccu task info, dieId[%u] missionId[%u] instStartId[%u] instCnt[%u]", taskParams[index].dieId,
            taskParams[index].missionId, taskParams[index].instStartId, taskParams[index].instCnt);
    }

    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

u32 CollServiceDeviceMode::GetCcuMc2ServerNum() { return mc2Compont.GetCcuMc2ServerNum(); }

CcuInsPreprocessor* CollServiceDeviceMode::GetCcuInsPreprocessor() { return &ccuInsPreprocessor; }

AivInsPreprocessor* CollServiceDeviceMode::GetAivInsPreprocessor() { return &aivInsPreprocessor; }

AicpuInsPreprocessor* CollServiceDeviceMode::GetAicpuInsPreprocessor() { return &aicpuInsPreprocessor; }

bool CollServiceDeviceMode::IsAicpuResExisted(std::string algName)
{
    return aicpuInsPreprocessor.IsAicpuResExisted(algName);
}

DevBuffer* CollServiceDeviceMode::GetAicpuResBuffer(std::string algName)
{
    return aicpuInsPreprocessor.GetAicpuResBuffer(algName);
}

constexpr u32 TEMP_MAX_CNTCKE_NUM = 16; // 临时规避多轮不同算子导致CNTCKE资源不足，待后续正式方案修改

void CollServiceDeviceMode::Resume()
{
    CcuCommunicator* ccuComm = ccuInsPreprocessor.GetCcuComm();
    CHECK_NULLPTR(ccuComm, "[CollServiceDeviceMode::Resume] ccuComm is nullptr!");

    CcuTransportMgr* ccuTransportMgr = ccuComm->GetCcuTransportMgr();
    CHECK_NULLPTR(ccuTransportMgr, "[CollServiceDeviceMode::Resume] ccuTransportMgr is nullptr!");
    ccuTransportMgr->Resume();
    ccuTransportMgr->Confirm();
    HCCL_INFO("[CollServiceDeviceMode][%s] resource confirm end.", __func__);

    int32_t devLogicId = HrtGetDevice();
    for (uint8_t dieId = 0; dieId < MAX_CCU_IODIE_NUM; ++dieId) {
        CHK_RET_THROW(
            InternalException,
            StringFormat(
                "[CollServiceDeviceMode][%s]Error occurs when call CcuCleanDieCkes, "
                "die[%u], devLogicId[%d].",
                __func__, dieId, devLogicId),
            CcuCleanDieCkes(devLogicId, dieId));
    }
}

HcclResult CollServiceDeviceMode::HandleAclGraphFirstOpAivBuff(rtStream_t mainStream)
{
    rtModel_t rtModel = nullptr;
    bool isCapture = false;
    u32 modelId = 0;
    CHK_RET(GetStreamCaptureInfo(mainStream, rtModel, isCapture));
    if (isCapture) {
        CHK_PTR_NULL(rtModel);
        // 获取不到modelId会报错
        CHK_RET(GetModelId(rtModel, modelId));
        if (captureModelIds.find(modelId) == captureModelIds.end()) {
            // aclgraph场景，首算子清理AIV buff
            comm->SetAivClearEnable(true);
            comm->SetAivTag(1);
            captureModelIds.insert(modelId);
            HCCL_INFO("[CollServiceDeviceMode][%s] modelId[%u] is inserted to captureModelIds_", __func__, modelId);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult CollServiceDeviceMode::GenerateAivOpArgs(const AivInstruction& aivInstruction, AivOpArgs& aivOpArgs) const
{
    aivInstruction.GetAivInsArgs(aivOpArgs);
    aivOpArgs.aivTag = aivOpArgs.isOpBase ? (static_cast<uint32_t>(comm->GetAivTag()) << AIV_TAG_MOVE_LEFT_BITS)
                                                | static_cast<uint32_t>(aivOpArgs.aivTag) :
                                            (static_cast<uint32_t>(comm->GetAivOffloadTag()) << AIV_TAG_MOVE_LEFT_BITS)
                                                | static_cast<uint32_t>(aivOpArgs.aivTag);
    void* buffersInAddr;
    void* buffersInAddrSrc;
    u64 buffersIn[MAX_RANK_SIZE_] = {};
    if (static_cast<unsigned int>(comm->GetMyRank()) >= MAX_RANK_SIZE_) {
        HCCL_ERROR("[CollServiceDeviceMode][%s] myRank is greater than or equal MAX_RANK_SIZE", __func__);
        return HCCL_E_PARA;
    }
    buffersIn[comm->GetMyRank()] = comm->GetCclBuffer()->GetAddr();
    auto ubMemLink2TransportMap = comm->GetUbMemoryTransportMgr()->GetRmtRankId2RmtIpcRmaBufList();
    for (auto ubMemLink2TransportIter : ubMemLink2TransportMap) {
        auto rmtRank = ubMemLink2TransportIter.first;
        auto rmtMemBuffer = ubMemLink2TransportIter.second->GetAddr();
        if (static_cast<unsigned int>(rmtRank) >= MAX_RANK_SIZE_) {
            HCCL_ERROR("[CollServiceDeviceMode][%s] rmtRank is greater than or equal MAX_RANK_SIZE", __func__);
            return HCCL_E_PARA;
        }
        buffersIn[rmtRank] = rmtMemBuffer;
    }

    buffersInAddr = aivOpArgs.isOpBase ? reinterpret_cast<void*>(comm->GetAivTagBuffer()->GetAddr()) :
                                         reinterpret_cast<void*>(comm->GetAivOffloadTagBuffer()->GetAddr());
    HCCL_INFO("%s AivTag[%u]", __func__, aivOpArgs.aivTag);
    aivOpArgs.buffersIn = buffersInAddr;
    HrtMemcpy(
        buffersInAddr, MAX_RANK_SIZE_ * sizeof(uint64_t), buffersIn, MAX_RANK_SIZE_ * sizeof(uint64_t),
        RT_MEMCPY_HOST_TO_DEVICE);
    u64 buffersOut[MAX_RANK_SIZE_] = {};
    auto ubMemLink2TransportMap_ = aivOpArgs.isOpBase ?
                                       comm->GetUbMemoryTransportMgr()->GetAllRankId2AivTagBufAddrList() :
                                       comm->GetUbMemoryTransportMgr()->GetAllRankId2AivOffloadTagBufAddrList();
    for (auto ubMemLink2TransportIter : ubMemLink2TransportMap_) {
        auto rmtRank = ubMemLink2TransportIter.first;
        auto rmtMemBuffer = ubMemLink2TransportIter.second;
        if (static_cast<unsigned int>(rmtRank) >= MAX_RANK_SIZE_) {
            HCCL_ERROR("[CollServiceDeviceMode][%s] rmtRank is greater than or equal MAX_RANK_SIZE", __func__);
            return HCCL_E_PARA;
        }
        buffersOut[rmtRank] = rmtMemBuffer;
    }
    buffersInAddr = aivOpArgs.isOpBase ?
                        reinterpret_cast<void*>(comm->GetAivTagBuffer()->GetAddr() + AIV_TAG_ADDR_OFFSET) :
                        reinterpret_cast<void*>(comm->GetAivOffloadTagBuffer()->GetAddr() + AIV_TAG_ADDR_OFFSET);
    HrtMemcpy(
        buffersInAddr, MAX_RANK_SIZE_ * sizeof(uint64_t), buffersOut, MAX_RANK_SIZE_ * sizeof(uint64_t),
        RT_MEMCPY_HOST_TO_DEVICE);

    buffersInAddr = aivOpArgs.isOpBase ?
                        reinterpret_cast<void*>(comm->GetAivTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET) :
                        reinterpret_cast<void*>(comm->GetAivOffloadTagBuffer()->GetAddr() + AIV_FLAG_ADDR_OFFSET);
    buffersInAddrSrc = aivOpArgs.isOpBase ?
                           reinterpret_cast<void*>(comm->GetAivTagBuffer()->GetAddr() + AIV_FLAG_CLEAR_OFFSET) :
                           reinterpret_cast<void*>(comm->GetAivOffloadTagBuffer()->GetAddr() + AIV_FLAG_CLEAR_OFFSET);
    bool isAivClearEnable = comm->GetAivClearEnable();
    if (isAivClearEnable && (aivOpArgs.aivTag & AIV_LOW_16_BITS) == 1
        && (aivOpArgs.aivTag >> AIV_TAG_MOVE_LEFT_BITS) == 1) {
        HrtMemcpy(buffersInAddr, AIV_FLAG_AREA_SIZE, buffersInAddrSrc, AIV_FLAG_AREA_SIZE, RT_MEMCPY_DEVICE_TO_DEVICE);
    }
    if (comm->GetCurrentCollOperator()->inputMem == nullptr) {
        HCCL_INFO("%s comm->GetCurrentCollOperator()->inputMem is nullptr", __func__);
    } else {
        u64 localInputAddr = static_cast<uint64_t>(comm->GetCurrentCollOperator()->inputMem->GetAddr());
        aivOpArgs.input += localInputAddr;
    }

    if (comm->GetCurrentCollOperator()->outputMem == nullptr) {
        HCCL_INFO("%s comm->GetCurrentCollOperator()->outputMem is nullptr", __func__);
    } else {
        u64 localOutputAddr = static_cast<uint64_t>(comm->GetCurrentCollOperator()->outputMem->GetAddr());
        aivOpArgs.output += localOutputAddr;
    }
    return HCCL_SUCCESS;
}

void CollServiceDeviceMode::GeneratorAivSuperKernelArgs(
    const AivOpArgs& aivOpArgs, bool clearEnable, u32 numBlocks, AivSuperKernelArgs& superArgs) const
{
    auto op = comm->GetCurrentCollOperator();
    superArgs.buffersIn = aivOpArgs.buffersIn;
    superArgs.rank = comm->GetMyRank();
    superArgs.rankSize = comm->GetRankSize();
    u64 dataCount = 0;
    DataType dataType = Hccl::DataType::INVALID;
    if (op->opType == OpType::ALLTOALL) {
        dataCount = op->all2AllDataDes.sendCount;
        dataType = op->all2AllDataDes.sendType;
    } else {
        dataCount = op->dataCount;
        dataType = op->dataType;
    }
    superArgs.len = dataCount;
    superArgs.dataType = dataType;
    superArgs.unitSize = SIZE_TABLE_ORION[dataType];
    superArgs.reduceOp = op->reduceOp;
    superArgs.numBlocks = numBlocks;
    superArgs.tag = comm->GetAivTag();
    superArgs.clearEnable = (clearEnable ? 1 : 0);
    superArgs.inputSliceStride = 0;
    superArgs.outputSliceStride = 0;
    superArgs.repeatNum = 1;
    superArgs.inputRepeatStride = 0;
    superArgs.outputRepeatStride = 0;
    superArgs.input = aivOpArgs.input;
    superArgs.output = aivOpArgs.output;
    superArgs.cclBufferSize = comm->GetBufferSize();

    HCCL_INFO(
        "[CollServiceDeviceMode::%s] Tag %lld, clearEnable %lld, numBlocks %llu, dataCount %llu, cclBufferSize %llu.",
        __func__, superArgs.tag, superArgs.clearEnable, superArgs.numBlocks, dataCount, superArgs.cclBufferSize);
}

HcclResult CollServiceDeviceMode::GetAlgExecParam(bool clearEnable, u32 numBlocks, void*& commContext, u64& len)
{
    auto op = comm->GetCurrentCollOperator();
    HCCL_INFO(
        "[CollServiceDeviceMode][%s] op[%p] sendCount[%u], recvCount[%u]", __func__, op, op->all2AllDataDes.sendCount,
        op->all2AllDataDes.recvCount);
    // 建链
    shared_ptr<InsQueue> insQueue = Orchestrate(*op);
    AllocQueueNotify(*insQueue);
    std::vector<LinkData> uniqueLinks = GetUniqueLinks(insQueue);
    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(uniqueLinks);
    GetAivInsPreprocessor()->Preprocess(insQueue);
    // 组装AivOpArgs
    AivOpArgs aivOpArgs{};
    for (auto ins = insQueue->Iter(); ins.HasNext(); ++ins) {
        if (ins->GetType() != InstructionType::AIV_INS) {
            continue;
        }
        const AivInstruction& aivIns = dynamic_cast<const AivInstruction&>(*ins);
        CHK_RET(GenerateAivOpArgs(aivIns, aivOpArgs));
        break;
    }

    // aivOpArgs转为aivSuperKernelArgs的参数
    AivSuperKernelArgs aivSuperKernelArgs{};
    GeneratorAivSuperKernelArgs(aivOpArgs, clearEnable, numBlocks, aivSuperKernelArgs);

    void* sendAlgParamMemPtr = nullptr;
    // alloc device 地址
    sendAlgParamMemPtr = HrtMalloc(sizeof(AivSuperKernelArgs), static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    CHK_PTR_NULL(sendAlgParamMemPtr);
    HCCL_INFO("SPK sendalgparam %p.", sendAlgParamMemPtr);

    // 拷贝到Device
    HrtMemcpy(
        sendAlgParamMemPtr, sizeof(AivSuperKernelArgs), &aivSuperKernelArgs, sizeof(AivSuperKernelArgs),
        RT_MEMCPY_HOST_TO_DEVICE);
    commContext = sendAlgParamMemPtr;
    len = sizeof(AivSuperKernelArgs);
    return HCCL_SUCCESS;
}

} // namespace Hccl
