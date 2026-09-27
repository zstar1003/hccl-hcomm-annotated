/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_service_default_impl.h"
#include <unordered_set>
#include <sstream>
#include <string>
#include <chrono>
#include "connections_builder.h"
#include "communicator_impl.h"
#include "dev_ub_connection.h"
#include "stl_util.h"
#include "exception_util.h"

using HcclUs = std::chrono::steady_clock::time_point;

namespace Hccl {

void CollServiceDefaultImpl::LoadWithOpBasedModeNoRegister(CollOperator& op)
{
    shared_ptr<InsQueue> insQueue;
    insQueue = OrchestrateWithIns(op);

    AllocQueueNotify(*insQueue);

    vector<LinkData> links = insQueue->GetUniqueLinks();

    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(links);
    // 建立RmaConnection并建链
    auto connBuilderPair = connectionsBuilders.emplace(comm->GetId(), make_unique<ConnectionsBuilder>(*comm));
    connBuilderPair.first->second->BatchBuild(comm->GetId(), links);

    AllocNotifies(links);

    AllocLocCntNotifies(*insQueue);

    comm->GetMemTransportManager()->BatchBuildOpbasedTransports(links);
    WaitOpbasedTransportReady();

    SaveMirrorDfxOpInfo();

    Interpreter interpreter(*comm);
    interpreter.Submit(*insQueue);

    UpdateUbCiIfNeed(op.opTag);
}

void CollServiceDefaultImpl::LoadWithOpBasedMode(CollOperator& op, unique_ptr<Stream> stream)
{
    HCCL_INFO("LoadWithOpBasedMode START");
    HCCL_INFO("RegisterOpbasedBuf start");
    RegisterOpBufToBufMgr(op);
    RegisterOpbasedStream(std::move(stream));

    LoadWithOpBasedModeNoRegister(op);
    HCCL_INFO("LoadWithOpBasedMode END");
}

void CollServiceDefaultImpl::UpdateUbCiIfNeed(const std::string& opTag)
{
    HCCL_INFO("CollServiceDefaultImpl::UpdateUbCiIfNeed start, opTag[%s]", opTag.c_str());
    if (updatingUbCiEvent == nullptr) {
        HCCL_INFO("updatingUbCiEvent is null");
        std::vector<DevUbConnection*> devUbConns = GetStarsPollUbConns(comm->GetRmaConnManager().GetOpTagConns(opTag));
        HCCL_INFO("starsPoll devUbConns size: %lu", devUbConns.size());
        if (IfNeedUpdatingUbCi(devUbConns)) {
            HCCL_INFO("need update ub ci");
            ubCiUpdaterMgr->SaveConnsCi(opTag);
            HCCL_INFO("ubCiUpdaterMgr saveConnsCi finished");
            updatingUbCiEvent = make_unique<MaskEvent>();
            updatingUbCiEvent->Record(*(comm->GetStreamManager().opbase->GetMaster()));
            HCCL_INFO("submit event record finished");
        }
        HCCL_INFO("need not update ub ci");
    } else {
        HCCL_INFO("updatingUbCiEvent is not nullptr");
        auto status = updatingUbCiEvent->QueryStatus();
        if (status == HrtEventStatus::EVENT_RECORDED) {
            HCCL_INFO("updatingUbCiEvent status is EVENT_RECORDED");
            ubCiUpdaterMgr->UpdateConnsCi(opTag);
            HCCL_INFO("ubCiUpdaterMgr updateConnsCi finished");
            updatingUbCiEvent = nullptr;
            HCCL_INFO("updatingUbCiEvent reset as nullptr");
        }
        HCCL_INFO("updatingUbCiEvent status is %u", static_cast<u32>(status));
    }
}

void CollServiceDefaultImpl::LoadWithOffloadModeNoRegister(CollOperator& op)
{
    RegisterOffloadLocalRmaBuf(op.opTag);

    shared_ptr<InsQueue> insQueue;
    insQueue = OrchestrateWithIns(op);

    vector<LinkData> links = insQueue->GetUniqueLinks();

    // Socket建链
    comm->GetSocketManager().BatchCreateSockets(links);
    // 建立RmaConnection并建链
    auto connBuilderPair = connectionsBuilders.emplace(op.opTag, make_unique<ConnectionsBuilder>(*comm));
    connBuilderPair.first->second->BatchBuild(op.opTag, links);

    AllocNotifies(links);

    AllocLocCntNotifies(*insQueue);

    comm->GetMemTransportManager()->BatchBuildOffloadTransports(op.opTag, links);
    WaitOffloadTransportReady(op.opTag);
    HCCL_INFO("Offload Interpret start");

    SaveMirrorDfxOpInfo();

    // 下发head算子执行计数器task
    AddCountTask(true);

    Interpreter interpreter(*comm);
    interpreter.Submit(*insQueue);
    HCCL_INFO("Offload Interpret end");

    // 下发tail算子执行计数器task
    AddCountTask(false);

    // 基于opTag+link找到connection; connection提供方法AddNop; 调用HCCP提供的AddNop(qpHandle)接口
    AddNop(op.opTag, links);
}

void CollServiceDefaultImpl::LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream)
{
    HCCL_INFO("LoadWithOffloadMode START");
    HCCL_INFO("RegisterOffloadBuf start");
    RegisterOpBufToBufMgr(op);

    RegisterOffloadMasterStream(op.opTag, std::move(stream));

    LoadWithOffloadModeNoRegister(op);

    HCCL_INFO("LoadWithOffloadMode END");
}

shared_ptr<PrimQueue> CollServiceDefaultImpl::OrchestrateWithPrim(const CollAlgOperator& op) const
{
    u64 tmpMemSize = comm->GetBufferSize();
    CollAlgParams params{};
    auto primQueue = make_shared<PrimQueue>();

    params.opMode = op.opMode;
    params.maxTmpMemSize = tmpMemSize;

    HCCL_INFO("orchestrate with Prim start");
    HcclResult errCode = comm->GetCollAlgComponent()->Orchestrate(op, params, comm->GetCurAlgName(), primQueue);
    HCCL_INFO("orchestrate with Prim end");

    if (errCode != HcclResult::HCCL_SUCCESS) {
        auto msg = StringFormat("Error occurs when call collAlgComponent.orchestrate(), error code: %d", errCode);
        THROW<InternalException>(msg);
    }

    return primQueue;
}

shared_ptr<InsQueue> CollServiceDefaultImpl::OrchestrateWithIns(const CollAlgOperator& op) const
{
    u64 tmpMemSize = 0;
    // 图模式部分算子不需要scratchMem
    if (op.scratchMem != nullptr) {
        tmpMemSize = op.scratchMem->GetSize();
    }
    CollAlgParams params{};
    auto insQueue = make_shared<InsQueue>();

    params.opMode = op.opMode;
    params.maxTmpMemSize = tmpMemSize;

    HCCL_INFO("orchestrate with Ins start");
    HcclResult errCode = comm->GetCollAlgComponent()->Orchestrate(op, params, comm->GetCurAlgName(), insQueue);
    HCCL_INFO("orchestrate with Ins end");

    if (errCode != HcclResult::HCCL_SUCCESS) {
        auto msg = StringFormat("Error occurs when call collAlgComponent.orchestrate(), error code: %d", errCode);
        THROW<InternalException>(msg);
    }
    return insQueue;
}

void CollServiceDefaultImpl::AllocNotifies(const vector<LinkData>& links)
{
    vector<LinkData> pendingLinks;
    for (auto& link : links) {
        if (Contain(availableLinks, link)) {
            continue;
        }
        pendingLinks.emplace_back(link);
    }

    if (pendingLinks.empty()) {
        return;
    }

    for (auto& link : pendingLinks) {
        // 待修改: 申请数量
        comm->GetConnLocalNotifyManager().ApplyFor(link.GetRemoteRankId(), link);
    }

    availableLinks.insert(pendingLinks.begin(), pendingLinks.end());
}

void CollServiceDefaultImpl::AllocOneLocCntNotify(const Instruction& ins) const
{
    HCCL_INFO("AllocOneLocCntNotify %s begin", ins.Describe().c_str());
    vector<LinkData> links;
    const InsWaitGroupFin& insWaitGroupFin = reinterpret_cast<const InsWaitGroupFin&>(ins);
    for (auto iter = insWaitGroupFin.Iter(); iter.HasNext(); ++iter) {
        links.push_back(*iter);
    }
    comm->GetConnLocalCntNotifyManager().ApplyFor(insWaitGroupFin.GetTopicId(), links);
    HCCL_INFO("AllocOneLocCntNotify %s end", ins.Describe().c_str());
}

void CollServiceDefaultImpl::AllocLocCntNotifies(const InsQueue& insQueue) const
{
    for (auto ins = insQueue.Iter(); ins.HasNext(); ++ins) {
        if (ins->GetType() == InstructionType::WAIT_GROUP_FIN) {
            AllocOneLocCntNotify(*ins);
        }
    }

    for (auto slaveIter = insQueue.IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        for (auto iterSlave = slaveIter->Iter(); iterSlave.HasNext(); ++iterSlave) {
            if (iterSlave->GetType() == InstructionType::WAIT_GROUP_FIN) {
                AllocOneLocCntNotify(*iterSlave);
            }
        }
    }
}

void CollServiceDefaultImpl::Init()
{
    ubCiUpdaterMgr = make_unique<UbCiUpdaterManager>(&comm->GetRmaConnManager());
    primTranslator = make_unique<PrimTranslator>();
    RegisterCclLocRmaBuffer();
}

void CollServiceDefaultImpl::AddNop(const std::string& opTag, const vector<LinkData>& linkDataVec) const
{
    for (auto& linkData : linkDataVec) {
        auto conn = comm->GetRmaConnManager().Get(opTag, linkData);
        Stream* mainStream = comm->GetStreamManager().offload->GetMaster(opTag);
        if (conn == nullptr) {
            THROW<NullPtrException>(StringFormat("CollServiceDefaultImpl::AddNop ptr is null"));
        }
        conn->AddNop(*mainStream);
    }
}

void CollServiceDefaultImpl::RecoverTransport(
    [[maybe_unused]] vector<LinkData>& links, [[maybe_unused]] vector<std::pair<LinkGroup, u32>> linkGroupPair)
{
    THROW<NotSupportException>(StringFormat("CollServiceDefaultImpl::RecoverTransport not support yet."));
}

void CollServiceDefaultImpl::ReLoadWithOpBasedMode(CollOperator& op)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    LoadWithOpBasedModeNoRegister(op);
    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

void CollServiceDefaultImpl::ReLoadWithOffloadMode(CollOperator& op)
{
    HCCL_INFO("[CollServiceDeviceMode::%s] start.", __func__);
    LoadWithOffloadModeNoRegister(op);
    HCCL_INFO("[CollServiceDeviceMode::%s] end.", __func__);
}

} // namespace Hccl
