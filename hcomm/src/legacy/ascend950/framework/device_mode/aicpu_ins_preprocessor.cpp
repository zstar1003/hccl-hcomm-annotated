/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_ins_preprocessor.h"
#include "null_ptr_exception.h"
#include "orion_adapter_rts.h"
#include "stl_util.h"
#include "aicpu_res_package_helper.h"
#include "alg_topo_package_helper.h"

namespace Hccl {

constexpr u8 QUEUE_NOTIFY_POST_QID_POS = 0;
constexpr u8 QUEUE_NOTIFY_WAIT_QID_POS = 1;
constexpr u8 QUEUE_NOTIFY_TOPIC_ID_POS = 2;

void AicpuInsPreprocessor::Preprocess(std::shared_ptr<InsQueue>& insQueue)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] insQueue Preprocess start.", __func__);

    // 对每个queue中每个aicpuIns进行预处理
    for (auto slaveIter = insQueue->IterSlaves(); slaveIter.HasNext(); ++slaveIter) {
        for (auto ins = slaveIter->Iter(); ins.HasNext(); ++ins) {
            if (ins->GetType() != InstructionType::AICPU_INS) { // todo:InstructionType
                HCCL_INFO(
                    "[AicpuInsPreprocessor::%s] slave insQueue ins type[%s] not aicpu type.", __func__,
                    ins->GetType().Describe().c_str());
                continue;
            }
            InsPreprocess(ins);
        }
    }

    // 对每主queue中每个aicpuIns进行预处理
    for (auto ins = insQueue->Iter(); ins.HasNext(); ++ins) {
        if (ins->GetType() != InstructionType::AICPU_INS) {
            HCCL_INFO(
                "[AicpuInsPreprocessor::%s] master insQueue ins type[%s] not aicpu type.", __func__,
                ins->GetType().Describe().c_str());
            continue;
        }
        InsPreprocess(ins);
    }

    HCCL_INFO("[AicpuInsPreprocessor::%s] insQueue Preprocess end.", __func__);
}

bool AicpuInsPreprocessor::IsAicpuResExisted(const std::string& algName)
{
    if (aicpuResExistedMap.find(algName) == aicpuResExistedMap.end()
        || aicpuResMap.find(algName) == aicpuResMap.end()) {
        THROW<NullPtrException>(StringFormat(
            "[AicpuInsPreprocessor::%s] aicpuRes for algName[%s] is not exited on aicpuResExistedMap.", __func__,
            algName.c_str()));
    }
    HCCL_INFO("[AicpuInsPreprocessor::%s] end, aicpuResExisted [%d].", __func__, aicpuResExistedMap[algName]);
    return aicpuResExistedMap[algName];
}

DevBuffer* AicpuInsPreprocessor::GetAicpuResBuffer(const std::string& algName)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    if (aicpuResMap.find(algName) == aicpuResMap.end()) {
        THROW<NullPtrException>(StringFormat(
            "[AicpuInsPreprocessor::%s] aicpuRes for algName[%s] is not exited on device buffer.", __func__,
            algName.c_str()));
    }

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
    return aicpuResMap[algName].get();
}

void AicpuInsPreprocessor::InsPreprocess(InsIterator& insIter)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    const AicpuInstruction& aicpuIns = dynamic_cast<const AicpuInstruction&>(*insIter);

    CollAlgResReq collAlgResReq = aicpuIns.GetCollAlgResReq();
    AllocWorkStream(collAlgResReq.primQueueNum);
    AllocQueueNotify(collAlgResReq.queueNotifys);
    AllocBcastPostCntNotify(collAlgResReq.localBcastPostCntNotify);
    AllocWaitGroupCntNotify(collAlgResReq.localWaitGroupCntNotify);
    AllocInterRankNotifies(collAlgResReq.links);

    // 创建MemTransport并建链、交换
    BatchBuildTransports(collAlgResReq.links);

    std::string algName = aicpuIns.GetAlgName();
    if (aicpuResMap.find(algName) != aicpuResMap.end()) { // 已经向Device Mem写过资源
        HCCL_INFO("[AicpuInsPreprocessor::%s] aicpuRes for algName[%s] has existed.", __func__, algName.c_str());
        return;
    }

    PackResAndCopyToDev(algName, collAlgResReq);

    AllocAlltoallVOpMem();

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::AllocWorkStream(u32 workStreamNum) const
{
    comm->GetAicpuStreamManager().AllocStreams(workStreamNum);
}

void AicpuInsPreprocessor::AllocQueueNotify(std::vector<std::tuple<QId, QId, u32>>& queueNotifyReq) const
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    QueueNotifyManager& queueNotifyMgr = comm->GetAicpuQueueNotifyManager();

    std::for_each(queueNotifyReq.begin(), queueNotifyReq.end(), [&queueNotifyMgr](auto item) {
        queueNotifyMgr.ApplyFor(
            std::get<QUEUE_NOTIFY_POST_QID_POS>(item), std::get<QUEUE_NOTIFY_WAIT_QID_POS>(item),
            std::get<QUEUE_NOTIFY_TOPIC_ID_POS>(item));
    });

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::AllocBcastPostCntNotify(std::vector<std::pair<QId, u32>>& bcastPostCntNotifyReq) const
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    QueueBcastPostCntNotifyManager& bcastPostCntNotifyMgr = comm->GetBcastPostCntNotifyManager();

    std::for_each(bcastPostCntNotifyReq.begin(), bcastPostCntNotifyReq.end(), [&bcastPostCntNotifyMgr](auto item) {
        bcastPostCntNotifyMgr.ApplyFor(item.first, item.second);
        HCCL_INFO("[AicpuInsPreprocessor::%s] qid[%u] topicId[%u]", __func__, item.first, item.second);
    });

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::AllocWaitGroupCntNotify(std::vector<std::pair<QId, u32>>& waitGroupCntNotifyReq) const
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    QueueWaitGroupCntNotifyManager& waitGroupCntNotifyMgr = comm->GetQueueWaitGroupCntNotifyManager();

    std::for_each(waitGroupCntNotifyReq.begin(), waitGroupCntNotifyReq.end(), [&waitGroupCntNotifyMgr](auto item) {
        waitGroupCntNotifyMgr.ApplyFor(item.first, item.second);
        HCCL_INFO("[AicpuInsPreprocessor::%s] qid[%u] topicId[%u]", __func__, item.first, item.second);
    });

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::AllocInterRankNotifies(const vector<LinkData>& links)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

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

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::BatchBuildTransports(const vector<LinkData>& links)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    std::string opTag = comm->GetCurrentCollOperator()->opTag;

    // 创建RmaConnectiuon
    auto connBuilderPair = connectionsBuilders.emplace(opTag, make_unique<ConnectionsBuilder>(*comm));
    connBuilderPair.first->second->BatchBuild(opTag, links);

    // 创建MemTransport并进行异步建链、交换
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) {
        comm->GetMemTransportManager()->BatchBuildOpbasedTransports(links);
    } else if (op->opMode == OpMode::OFFLOAD) {
        comm->GetMemTransportManager()->BatchBuildOffloadTransports(opTag, links);
    }

    // 等待异步建链完成
    comm->GetCollService()->WaitTransportReady(opTag);

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

static void SetModuleName(ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("strcpy_s name %s failed", name.c_str()));
    }
}

std::vector<char>
AicpuInsPreprocessor::PackOpData(const std::string& opTag, const std::string& algName, const CollAlgResReq& resReq)
{
    std::vector<ModuleData> dataVec;
    dataVec.resize(AicpuResMgrType::__COUNT__);

    AicpuResMgrType resType = AicpuResMgrType::STREAM;
    SetModuleName(dataVec[resType], "StreamManager");
    dataVec[resType].data = comm->GetAicpuStreamManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_NOTIFY;
    SetModuleName(dataVec[resType], "QueueNotifyManager");
    dataVec[resType].data = comm->GetAicpuQueueNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_WAIT_GROUP_CNT_NOTIFY;
    SetModuleName(dataVec[resType], "QueueWaitGroupCntNotifyManager");
    dataVec[resType].data = comm->GetQueueWaitGroupCntNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_BCAST_POST_CNT_NOTIFY;
    SetModuleName(dataVec[resType], "GetBcastPostCntNotifyManager");
    dataVec[resType].data = comm->GetBcastPostCntNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::HOST_DEV_SYNC_NOTIFY;
    SetModuleName(dataVec[resType], "HostDeviceSyncNotifyManager");
    dataVec[resType].data = comm->GetHostDeviceSyncNotifyManager().GetPackedData();
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::TRANSPORT;
    SetModuleName(dataVec[resType], "MemTransportManager");
    auto op = comm->GetCurrentCollOperator();
    if (op->opMode == OpMode::OPBASE) { // 单算子模式
        dataVec[resType].data = comm->GetMemTransportManager()->GetOpbasedPackedData();
    } else if (op->opMode == OpMode::OFFLOAD) { // 图下沉模式
        dataVec[resType].data = comm->GetMemTransportManager()->GetOffloadPackedData(opTag);
    } else {
        THROW<InternalException>(StringFormat("opMode=%s failed", op->opMode.Describe().c_str()));
    }
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::ALG_TOPO;
    SetModuleName(dataVec[resType], algName);
    AlgTopoPackageHelper algTopoHelper;
    dataVec[resType].data = algTopoHelper.GetPackedData(resReq.topoInfo);
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::CONNECTD_MGR;
    SetModuleName(dataVec[resType], "ConnectedManager");
    dataVec[resType].data = comm->GetRankGraph()->GetPackedData(resReq.levelRankPairs);
    HCCL_INFO("CollServiceAiCpuImpl::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    AicpuResPackageHelper helper;
    return helper.GetPackedData(dataVec);
}

void AicpuInsPreprocessor::PackResAndCopyToDev(const std::string& algName, const CollAlgResReq& resReq)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    std::string opTag = comm->GetCurrentCollOperator()->opTag;
    auto buffer = PackOpData(opTag, algName, resReq);
    shared_ptr<DevBuffer> devMem = make_shared<DevBuffer>(buffer.size()); // 申请device内存
    HrtMemcpy(
        reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), buffer.data(), buffer.size(),
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到device内存
    HCCL_INFO("[AicpuInsPreprocessor::%s] PackedData %s", __func__, Bytes2hex(buffer.data(), buffer.size()).c_str());

    aicpuResMap.insert(std::make_pair(algName, devMem));
    aicpuResExistedMap.insert(std::make_pair(algName, false));

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::AllocAlltoallVOpMem()
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    auto op = comm->GetCurrentCollOperator();
    if (op->opType != OpType::ALLTOALLV) {
        HCCL_INFO("[AllocAlltoallVOpMem] op->opType[%d]", op->opType);
        return;
    }

    size_t size = static_cast<size_t>(comm->GetRankSize() * sizeof(u64)); // counts内存大小
    if (!isCountMemInited) {
        for (u32 i = 0; i < MAX_ALLTOALLV_MEM_NUM; i++) {                 // 64: 初始化countMem
            shared_ptr<DevBuffer> sendMem = make_shared<DevBuffer>(size); // 申请send device内存
            sendCountsMem.push_back(sendMem);

            shared_ptr<DevBuffer> recvMem = make_shared<DevBuffer>(size); // 申请recv device内存
            recvCountsMem.push_back(recvMem);

            shared_ptr<DevBuffer> sdisplMem = make_shared<DevBuffer>(size); // 申请sdispls device内存
            sdisplsMem.push_back(sdisplMem);

            shared_ptr<DevBuffer> rdisplMem = make_shared<DevBuffer>(size); // 申请rdispls device内存
            rdisplsMem.push_back(rdisplMem);
        }
        isCountMemInited = true;
    }

    HrtMemcpy(
        reinterpret_cast<void*>(sendCountsMem[resIndex].get()->GetAddr()), sendCountsMem[resIndex].get()->GetSize(),
        op->all2AllVDataDes.sendCounts, size,
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到SEND内存
    HrtMemcpy(
        reinterpret_cast<void*>(recvCountsMem[resIndex].get()->GetAddr()), recvCountsMem[resIndex].get()->GetSize(),
        op->all2AllVDataDes.recvCounts, size,
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到RECV内存
    HrtMemcpy(
        reinterpret_cast<void*>(sdisplsMem[resIndex].get()->GetAddr()), sdisplsMem[resIndex].get()->GetSize(),
        op->all2AllVDataDes.sdispls, size,
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到SDISPLS内存
    HrtMemcpy(
        reinterpret_cast<void*>(rdisplsMem[resIndex].get()->GetAddr()), rdisplsMem[resIndex].get()->GetSize(),
        op->all2AllVDataDes.rdispls, size,
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到RDISPLS内存

    resIndex++;
    if (resIndex >= MAX_ALLTOALLV_MEM_NUM) { // MAX_ALLTOALLV_MEM_NUM: 初始化countMem
        resIndex = 0;
    }

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::SetAicpuKernelLaunchParam(HcclKernelLaunchParam& param)
{
    HCCL_INFO("[AicpuInsPreprocessor::%s] start.", __func__);

    auto op = comm->GetCurrentCollOperator();
    if (op->opType != OpType::ALLTOALLV) {
        HCCL_INFO("[SetAicpuKernelLaunchParam] op->opType[%d]", op->opType);
        return;
    }

    param.kernel.op.algOperator.all2AllVDataDes.sendCounts
        = reinterpret_cast<void*>(sendCountsMem[launchResIndex].get()->GetAddr());
    param.kernel.op.algOperator.all2AllVDataDes.recvCounts
        = reinterpret_cast<void*>(recvCountsMem[launchResIndex].get()->GetAddr());
    param.kernel.op.algOperator.all2AllVDataDes.sdispls
        = reinterpret_cast<void*>(sdisplsMem[launchResIndex].get()->GetAddr());
    param.kernel.op.algOperator.all2AllVDataDes.rdispls
        = reinterpret_cast<void*>(rdisplsMem[launchResIndex].get()->GetAddr());
    param.kernel.op.algOperator.all2AllVDataDes.sendType = op->all2AllVDataDes.sendType;
    param.kernel.op.algOperator.all2AllVDataDes.recvType = op->all2AllVDataDes.recvType;

    HCCL_INFO(
        "AicpuKernelLauncher::SetHcclKernelLaunchParam param.kernel.op.algOperator.sendCounts[%p] "
        "param.kernel.op.algOperator.recvCounts[%p] param.kernel.op.algOperator.sdispls[%p] "
        "param.kernel.op.algOperator.rdispls[%p], launchResIndex[%u]",
        param.kernel.op.algOperator.all2AllVDataDes.sendCounts, param.kernel.op.algOperator.all2AllVDataDes.recvCounts,
        param.kernel.op.algOperator.all2AllVDataDes.sdispls, param.kernel.op.algOperator.all2AllVDataDes.rdispls,
        launchResIndex);

    launchResIndex++;
    if (launchResIndex >= MAX_ALLTOALLV_MEM_NUM) { // MAX_ALLTOALLV_MEM_NUM: 初始化countMem
        launchResIndex = 0;
    }

    HCCL_INFO("[AicpuInsPreprocessor::%s] end.", __func__);
}

void AicpuInsPreprocessor::SetAicpuResExisted(const std::string& algName)
{
    if (aicpuResExistedMap.find(algName) == aicpuResExistedMap.end()) {
        THROW<NullPtrException>(StringFormat(
            "[AicpuInsPreprocessor::%s] aicpuRes for algName[%s] is not exited on aicpuResExistedMap.", __func__,
            algName.c_str()));
    }
    aicpuResExistedMap[algName] = true;
}

} // namespace Hccl
