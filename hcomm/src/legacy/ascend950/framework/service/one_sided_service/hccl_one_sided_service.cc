/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <climits>
#include "hccl_one_sided_service_v2.h"
#include "communicator_impl.h"
#include "virtual_topo.h"
#include "alg_topo_package_helper.h"
#include "aicpu_res_package_helper.h"
#include "hccl_mem.h"
#include "exception_util.h"
#include "env_config_v2.h"

namespace Hccl {
using namespace std;

// 设置最大注册内存数量为256
constexpr u32 maxregisteredMem = 256;

static void OneSidedSetModuleDataName(ModuleData& module, const std::string& name)
{
    int ret = strcpy_s(module.name, sizeof(module.name), name.c_str());
    if (ret != 0) {
        THROW<InternalException>(StringFormat("strcpy_s name %s failed. ret[%d]", name.c_str(), ret));
    }
}

template <class T, class U>
u16 CalcFieldOffset(T* target, U* base)
{
    return static_cast<u16>(
        static_cast<const char*>(static_cast<void*>(target)) - static_cast<const char*>(static_cast<void*>(base)));
}

HcclOneSidedService::HcclOneSidedService(CommunicatorImpl& comm) : comm_(&comm) { AddOpCounterMems(); }

HcclOneSidedService::~HcclOneSidedService()
{
    for (const auto& pair : desc2netDevMap_) {
        const HcclNetDev& hcclNetDev = pair.second;
        HcclResult ret = HcclNetDevClose(hcclNetDev);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclOneSidedService][~HcclOneSidedService]HcclNetDevClose failed, descStr[%s], ret[%d].",
                pair.first.c_str(), ret);
        }
    }
}

LinkData HcclOneSidedService::GetLinkData(RankId remoteRankId)
{
    if (linkDataMap_.find(remoteRankId) == linkDataMap_.end()) {
        // 组建linkData
        LinkData linkData(comm_->GetRankGraph()->GetPaths(0, comm_->GetMyRank(), remoteRankId)[0]);
        linkDataMap_.emplace(remoteRankId, linkData);
    }
    HCCL_INFO("[HcclOneSidedService][GetLinkData] linkData[%s]", linkDataMap_.at(remoteRankId).Describe().c_str());
    return linkDataMap_.at(remoteRankId);
}

HcclResult HcclOneSidedService::CheckLink(LinkData linkData) const
{
    HCCL_INFO("[HcclOneSidedService][CheckLink] linkData[%s]", linkData.Describe().c_str());
    CHK_PRT_RET(
        (linkData.GetLinkProtocol() != LinkProtocol::UB_CTP && linkData.GetLinkProtocol() != LinkProtocol::UB_TP),
        HCCL_ERROR("[HcclOneSidedService][CheckLink] Proto is not UB, not support"), HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        linkData.GetHop() > 1, HCCL_ERROR("[HcclOneSidedService][CheckLink]Hop is greater than 1, not support"),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

HcclResult
HcclOneSidedService::RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc& localMemDesc)
{
    CHK_PTR_NULL(addr);
    CHK_PRT_RET(
        type == HcclMemType::HCCL_MEM_TYPE_HOST,
        HCCL_ERROR("[HcclOneSidedService][RegMem]HCCL_MEM_TYPE_HOST is not supported"), HCCL_E_NOT_SUPPORT);
    HCCL_INFO(
        "[HcclOneSidedService][RegMem]addr[%p], size[%llu], type[%d], remoteRankId[%u]", addr, size, type,
        remoteRankId);
    LinkData linkData = GetLinkData(remoteRankId);
    RmaMemDesc* localRmaMemDesc = static_cast<RmaMemDesc*>(static_cast<void*>(localMemDesc.desc));
    CHK_PTR_NULL(localRmaMemDesc);
    CHK_PRT_RET(
        registeredMemCnt_ >= maxregisteredMem,
        HCCL_ERROR(
            "[HcclOneSidedService][RegMem]registered memory counts=[%u] exceeds limit[%u]", registeredMemCnt_,
            maxregisteredMem),
        HCCL_E_UNAVAIL);
    HcclNetDevInfos info;
    info.addr.protoType = HcclNetDevice::ConvertHcclProtoToLinkProto(linkData.GetLocalPort().GetProto());
    info.addr.type = HCCL_ADDR_TYPE_IP_V4;
    info.netdevDeployment = HcclNetDevice::ConvertDeploymentType(linkData.GetLocalPort().GetType());
    info.devicePhyId = comm_->GetDevicePhyId();
    info.addr.addr = linkData.GetLocalPort().GetAddr().GetBinaryAddress().addr;
    HcclNetDev netDev;
    HcclResult ret = HcclNetDevOpen(&info, &netDev);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclOneSidedService][RegMem]HcclNetDevOpen failed, ret[%d].", ret);
        return ret;
    }
    HcclMem localMem{type, addr, size};
    HcclBuf buf;
    ret = HcclMemReg(netDev, &localMem, &buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) {
        HCCL_ERROR("[HcclOneSidedService][RegMem]HcclMemReg failed, ret[%d].", ret);
        CHK_RET(HcclNetDevClose(netDev));
        return ret;
    }
    string logInfo = ret == HCCL_SUCCESS ? "Register memory success!" :
                                           "Memory is already registered, just increase the reference count.";
    HCCL_INFO("[HcclOneSidedService][RegMem]:%s Add key {%p, %llu}", logInfo.c_str(), addr, size);
    localRmaMemDesc->localRankId = comm_->GetMyRank();
    localRmaMemDesc->remoteRankId = remoteRankId;
    char* desc = localRmaMemDesc->memDesc;
    uint64_t descLen = 0;
    ret = HcclMemExport(&buf, &desc, &descLen);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclOneSidedService][RegMem]HcclMemExport failed, ret[%d]", ret);
        CHK_RET(HcclNetDevClose(netDev));
        return ret;
    }
    registeredMemCnt_++;
    std::string descStr(localRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE);
    desc2HcclBufMapLocalUb_.emplace(descStr, buf);
    desc2netDevMap_.emplace(descStr, netDev);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::DeregMem(const HcclMemDesc& localMemDesc)
{
    // 若当前内存注册数量为0，则返回找不到内存
    if (registeredMemCnt_ == 0) {
        HCCL_ERROR("[HcclOneSidedService][DeregMem]Registered memory is 0, please register first.");
        return HCCL_E_NOT_FOUND;
    }
    const RmaMemDesc* localRmaMemDesc = static_cast<const RmaMemDesc*>(static_cast<const void*>(localMemDesc.desc));
    std::string descStr(localRmaMemDesc->memDesc, TRANSPORT_EMD_ESC_SIZE);
    if (desc2HcclBufMapLocalUb_.find(descStr) == desc2HcclBufMapLocalUb_.end()) {
        HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]memory is not registered, please register first.");
        return HCCL_E_NOT_FOUND;
    }
    HcclBuf buf = desc2HcclBufMapLocalUb_.at(descStr);
    HcclResult ret = HcclMemDereg(&buf);
    if (ret == HCCL_SUCCESS) {
        registeredMemCnt_--;
        desc2HcclBufMapLocalUb_.erase(descStr);
    }
    if (desc2netDevMap_.find(descStr) == desc2netDevMap_.end()) {
        HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]NetDev is not open, please register first.");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::CreateConnection(std::shared_ptr<HcclOneSidedConn>& tempConn, LinkData linkData)
{
    if (isOpModeReady_ == false) {
        CHK_RET(comm_->RecoverOpMode(1));
        isOpModeReady_ = true;
    }
    HCCL_INFO("[HcclOneSidedService][CreateConnection] start");
    HCCL_INFO("[HcclOneSidedService][CreateConnection] linkData[%s]", linkData.Describe().c_str());
    tempConn = make_shared<HcclOneSidedConn>(comm_, linkData);

    CHK_PTR_NULL(tempConn);
    HCCL_INFO("[HcclOneSidedService][CreateConnection] end");
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ExchangeMemDesc(
    RankId remoteRankId, const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote)
{
    HcclUs startut = TIME_NOW();
    if (comm_->GetCommExecuteConfig().accState != AcceleratorState::AICPU_TS) {
        HCCL_ERROR(
            "[HcclOneSidedService][%s] only support aicpu, current accelerator[%s]", __func__,
            comm_->GetCommExecuteConfig().accState.Describe().c_str());
        return HCCL_E_NOT_SUPPORT;
    }
    comm_->SetOpExecuteConfig(comm_->GetCommExecuteConfig());
    // 组装linkData
    LinkData linkData = GetLinkData(remoteRankId);

    HCCL_INFO("[HcclOneSidedService][ExchangeMemDesc] Find HcclOneSidedConn");
    shared_ptr<HcclOneSidedConn> tempConn;
    // 查找是否已存在对端连接，不存在则创建
    std::unique_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        // 检测对端是否符合建链要求
        CHK_RET(CheckLink(linkData));
        // 创建Conn对象
        CHK_RET(CreateConnection(tempConn, linkData));
        oneSidedConns_.emplace(remoteRankId, tempConn);
    } else {
        tempConn = it->second;
    }

    CHK_PTR_NULL(tempConn);
    HCCL_INFO("[HcclOneSidedService][ExchangeMemDesc] tempConn linkData[%s]", linkData.Describe().c_str());
    HCCL_INFO("[HcclOneSidedService][ExchangeMemDesc] ExchangeMemDesc");
    HcclResult ret = tempConn->ExchangeMemDesc(localMemDescs, remoteMemDescs, actualNumOfRemote);
    HCCL_INFO(
        "[HcclOneSidedService][ExchangeMemDesc] finished. ret[%d], take time [%lld]us.", ret,
        DURATION_US(TIME_NOW() - startut).count());
    return ret;
}

HcclResult HcclOneSidedService::EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem)
{
    CHK_PTR_NULL(remoteMemDesc.desc);
    // 将HcclMemDesc转化为RmaMemDesc
    const RmaMemDesc* remoteRmaMemDesc = static_cast<const RmaMemDesc*>(static_cast<const void*>(remoteMemDesc.desc));
    RankId remoteRankId = remoteRmaMemDesc->localRankId;

    HCCL_INFO("[HcclOneSidedService][EnableMemAccess] Get remoteRankId[%u]", remoteRankId);
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    if (oneSidedConns_.find(remoteRankId) == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclOneSidedService][EnableMemAccess]connection not found, remoteRank[%u].", remoteRankId);
        return HCCL_E_NOT_FOUND;
    }

    HCCL_INFO("[HcclOneSidedService][EnableMemAccess] EnableMemAccess.");
    oneSidedConns_.at(remoteRankId)->EnableMemAccess(remoteMemDesc, remoteMem);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::DisableMemAccess(const HcclMemDesc& remoteMemDesc)
{
    CHK_PTR_NULL(remoteMemDesc.desc);
    // 将HcclMemDesc转化为RmaMemDesc
    const RmaMemDesc* remoteRmaMemDesc = static_cast<const RmaMemDesc*>(static_cast<const void*>(remoteMemDesc.desc));

    // 获取Conn对象
    RankId remoteRankId = remoteRmaMemDesc->localRankId;
    HCCL_INFO("[HcclOneSidedService][DisableMemAccess] Get remoteRankId[%u]", remoteRankId);
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    if (oneSidedConns_.find(remoteRankId) == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclOneSidedService][DisableMemAccess]connection not found by remoteRankId[%u].", remoteRankId);
        return HCCL_E_NOT_FOUND;
    }

    HCCL_INFO("[HcclOneSidedService][DisableMemAccess] DisableMemAccess");
    oneSidedConns_.at(remoteRankId)->DisableMemAccess(remoteMemDesc);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::BatchPutGetDevBufs(
    const HcclOneSideOpDesc* desc, u32 descNum, std::shared_ptr<HcclOneSidedConn> oneSidedConn)
{
    vector<HcclAicpuLocBufLite> hostBatchPutGetLocalBufferSliceBufs(descNum);
    vector<HcclAicpuLocBufLite> hostBatchPutGetRemoteBufferSliceBufs(descNum);
    CHK_RET(oneSidedConn->BatchBufferSlice(
        desc, descNum, hostBatchPutGetLocalBufferSliceBufs, hostBatchPutGetRemoteBufferSliceBufs));

    devBatchPutGetLocalBufs = make_shared<DevBuffer>(sizeof(HcclAicpuLocBufLite) * descNum);
    devBatchPutGetRemoteBufs = make_shared<DevBuffer>(sizeof(HcclAicpuLocBufLite) * descNum);

    HrtMemcpy(
        reinterpret_cast<void*>(devBatchPutGetLocalBufs->GetAddr()), devBatchPutGetLocalBufs->GetSize(),
        static_cast<void*>(hostBatchPutGetLocalBufferSliceBufs.data()), sizeof(HcclAicpuLocBufLite) * descNum,
        RT_MEMCPY_HOST_TO_DEVICE);

    HrtMemcpy(
        reinterpret_cast<void*>(devBatchPutGetRemoteBufs->GetAddr()), devBatchPutGetRemoteBufs->GetSize(),
        static_cast<void*>(hostBatchPutGetRemoteBufferSliceBufs.data()), sizeof(HcclAicpuLocBufLite) * descNum,
        RT_MEMCPY_HOST_TO_DEVICE);

    return HCCL_SUCCESS;
}

std::vector<char> HcclOneSidedService::PackOpData(const CollAlgOpReq& req) const
{
    std::vector<ModuleData> dataVec;
    dataVec.resize(AicpuResMgrType::__COUNT__);

    AicpuResMgrType resType = AicpuResMgrType::STREAM;
    OneSidedSetModuleDataName(dataVec[resType], "StreamManager");
    dataVec[resType].data = comm_->GetAicpuStreamManager().GetPackedData();
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_NOTIFY;
    OneSidedSetModuleDataName(dataVec[resType], "QueueNotifyManager");
    dataVec[resType].data = comm_->GetAicpuQueueNotifyManager().GetPackedData();
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_WAIT_GROUP_CNT_NOTIFY;
    OneSidedSetModuleDataName(dataVec[resType], "QueueWaitGroupCntNotifyManager");
    dataVec[resType].data = comm_->GetQueueWaitGroupCntNotifyManager().GetPackedData();
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::QUEUE_BCAST_POST_CNT_NOTIFY;
    OneSidedSetModuleDataName(dataVec[resType], "GetBcastPostCntNotifyManager");
    dataVec[resType].data = comm_->GetBcastPostCntNotifyManager().GetPackedData();
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::HOST_DEV_SYNC_NOTIFY;
    OneSidedSetModuleDataName(dataVec[resType], "HostDeviceSyncNotifyManager");
    dataVec[resType].data = comm_->GetHostDeviceSyncNotifyManager().GetPackedData();
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::TRANSPORT;
    OneSidedSetModuleDataName(dataVec[resType], "MemTransportManager");
    auto op = comm_->GetCurrentCollOperator();
    // GetOpbasedPackedData由于单边通信隔离，会找不到Transport
    if (op->opMode == OpMode::OPBASE) { // 单算子模式
        dataVec[resType].data = comm_->GetMemTransportManager()->GetOneSidedPackedData();
    } else {
        THROW<InternalException>(StringFormat("opMode=%s failed", op->opMode.Describe().c_str()));
    }
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::ALG_TOPO;
    OneSidedSetModuleDataName(dataVec[resType], req.algName);
    AlgTopoPackageHelper algTopoHelper;
    dataVec[resType].data = algTopoHelper.GetPackedData(req.resReq.topoInfo);
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    resType = AicpuResMgrType::CONNECTD_MGR;
    OneSidedSetModuleDataName(dataVec[resType], "ConnectedManager");
    dataVec[resType].data = comm_->GetRankGraph()->GetPackedData(req.resReq.levelRankPairs);
    HCCL_INFO("HcclOneSidedService::PackOpData: GetResMgr %s Data", resType.Describe().c_str());

    AicpuResPackageHelper helper;
    return helper.GetPackedData(dataVec);
}

void HcclOneSidedService::FillOneSidedOperator(OpType type, RankId remoteRankId, const HcclOneSideOpDesc* desc) const
{
    CollOpParams opParams;

    opParams.dataType = HcclDataTypeToDataType(desc->dataType);
    opParams.count = desc->count;

    // sendBuf/recvBuf当前不使用，等待后续扩展
    opParams.sendBuf = desc->localAddr;
    opParams.recvBuf = desc->remoteAddr;

    opParams.opType = type;
    opParams.dstRank = remoteRankId;
    std::string opTag = comm_->GetId();

    HCCL_INFO(
        "[HcclOneSidedService][FillOneSidedOperator] CovertToCurrentCollOperator opType[%s], dstRank[%u], opTag[%s]",
        opParams.opType.Describe().c_str(), opParams.dstRank, opTag.c_str());
    comm_->CovertToCurrentCollOperator(opTag, opParams, OpMode::OPBASE);
}

void HcclOneSidedService::AddPostToUserStream(const Stream& stream) const
{
    auto postNotify = comm_->GetHostDeviceSyncNotifyManager().GetDeviceWaitNotify();

    postNotify->Post(stream);
}

void HcclOneSidedService::AddWaitToUserStream(const Stream& stream) const
{
    auto waitNotify = comm_->GetHostDeviceSyncNotifyManager().GetHostWaitNotify();

    waitNotify->Wait(stream, 1000); // host 和 device sync流程，等待1000ms
}

void HcclOneSidedService::SetOneSidedKernelLaunchParam(HcclKernelLaunchParam& param, const DevBuffer* mem) const
{
    CollOperator op = *comm_->GetCurrentCollOperator();

    HCCL_INFO("[HcclOneSidedService][SetOneSidedKernelLaunchParam] op.opType[%s]", op.opType.Describe().c_str());
    param.kernel.comm.idIndex = comm_->GetIdIndex();
    param.kernel.comm.myRank = comm_->GetMyRank();
    param.kernel.comm.rankSize = comm_->GetRankSize();
    param.kernel.comm.devType = comm_->GetDevType();
    param.kernel.comm.devPhyId = comm_->GetDevicePhyId();
    param.kernel.comm.opCounterAddr = static_cast<u64>(counterBuf->GetAddr());
    auto ret = strcpy_s(param.kernel.comm.commId, sizeof(param.kernel.comm.commId), comm_->GetId().data());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("HcclOneSidedService::SetOneSidedKernelLaunchParam, strcpy_s commId failed! ret[%d]", ret));
    }

    param.kernel.oneSidedComm = true;

    param.kernel.op.algOperator.opMode = op.opMode;
    param.kernel.op.algOperator.opType = op.opType;

    param.kernel.binaryResAddr = mem->GetAddr();
    param.kernel.binaryResSize = mem->GetSize();

    param.kernel.op.sendRecvRemoteRank = op.sendRecvRemoteRank;

    param.kernel.kfcControlTransferH2DParams = comm_->GetKfcControlTransferH2D().GetCommunicateParams();
    param.kernel.kfcControlTransferD2HParams = comm_->GetKfcStatusTransferD2H().GetCommunicateParams();
}

void HcclOneSidedService::OneSidedAicpuKernelLaunch(HcclKernelLaunchParam& param, Stream& stream) const
{
    const aclrtFuncHandle funcHandle = comm_->GetAicpuKernelFuncHandle(param.kernelName);
    constexpr u32 numBlocks = 1;
    aclrtLaunchKernelCfg cfg;
    aclrtLaunchKernelAttr attr;
    attr.id = ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT;
    auto timeoutCheck = EnvConfig::GetInstance().GetRtsConfig().GetExecTimeOut();
    // aicpu kernel超时时间: X+30s
    attr.value.timeout = static_cast<u16>((timeoutCheck == 0) ? timeoutCheck : (timeoutCheck + 30));
    cfg.numAttrs = 1;
    cfg.attrs = &attr;
    AddPostToUserStream(stream);
    HCCL_INFO(
        "[HcclOneSidedService::AicpuKernelLaunch] param.soName: %s, param.kernelName: %s", param.soName,
        param.kernelName);
    HrtAicpuLaunchKernelWithHostArgs(
        funcHandle, numBlocks, comm_->GetAicpuStreamManager().GetFreeStream()->GetPtr(), &cfg, &param.kernel,
        sizeof(HcclKernelParamLite));
    HCCL_INFO(
        "[HcclOneSidedService][AicpuKernelLaunch] param.kernel.algName: %s HrtAicpuLaunchKernelWithHostArgs end!",
        param.kernel.algName);
    AddWaitToUserStream(stream);
}

DevBuffer* HcclOneSidedService::PackResToKernelLanuch(CollAlgOpReq& opReq)
{
    auto it = OneSidedLoadMap.find(opReq.algName);
    if (it != OneSidedLoadMap.end()) { // 已经向Device Mem写过资源
        HCCL_INFO("[OpBasedCollProcess] tag[%s] devMem has been allocated, reuse it", opReq.algName.c_str());
        return it->second.get();
    }

    HCCL_INFO("[HcclOneSidedService][PackResToKernelLanuch], PackOpData start");
    // 打包单边通信资源信息到device
    auto buffer = PackOpData(opReq);
    shared_ptr<DevBuffer> devMem = make_shared<DevBuffer>(buffer.size()); // 申请device内存

    HCCL_INFO("[HcclOneSidedService][PackResToKernelLanuch], HrtMemSyncCopy start");
    HrtMemcpy(
        reinterpret_cast<void*>(devMem->GetAddr()), devMem->GetSize(), buffer.data(), buffer.size(),
        RT_MEMCPY_HOST_TO_DEVICE); // H2D拷贝，将资源拷贝到device内存
    HCCL_INFO(
        "HcclOneSidedService::BatchGet PackOpData: PackedData %s", Bytes2hex(buffer.data(), buffer.size()).c_str());
    OneSidedLoadMap.insert(make_pair(opReq.algName, devMem));

    return devMem.get();
}

HcclResult HcclOneSidedService::BatchOpKernelLaunch(
    OpType opType, RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, shared_ptr<Stream> stream)
{
    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] start");
    comm_->GetAicpuStreamManager().AllocFreeStream();
    HCCL_INFO("[HcclOneSidedService][AllocStreams] start");
    comm_->GetAicpuStreamManager().AllocStreams(1);

    CHK_PTR_NULL(desc);
    HCCL_INFO(
        "[HcclOneSidedService][BatchOpKernelLaunch] desc: localAddr:[%p],remoteAddr:[%p],count:[%llu],dataType:[%d]",
        desc->localAddr, desc->remoteAddr, desc->count, desc->dataType);

    CollAlgOpReq opReq;
    opReq.algName = OpTypeToString(opType);
    opReq.resReq.levelRankPairs.push_back(make_pair(0, remoteRankId));
    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] FillOneSidedOperator start");
    // 填充通信算子信息
    FillOneSidedOperator(opType, remoteRankId, desc);
    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] PackResToKernelLanuch start");
    // 打包device展开资源信息
    DevBuffer* devMem = PackResToKernelLanuch(opReq);
    // 组kernelLaunch参数
    HcclKernelLaunchParam param;

    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] SetOneSidedKernelLaunchParam start");
    // 构造单边通信公共参数
    SetOneSidedKernelLaunchParam(param, devMem);

    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchGet] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] BatchPutGetDevBufs start");
    CHK_RET(BatchPutGetDevBufs(desc, descNum, it->second));
    oneSidedConnslock.unlock();

    param.kernel.op.batchPutGetDescNum = descNum;
    param.kernel.op.batchPutGetLocalAddr = reinterpret_cast<void*>(devBatchPutGetLocalBufs.get()->GetAddr());
    param.kernel.op.batchPutGetRemoteAddr = reinterpret_cast<void*>(devBatchPutGetRemoteBufs.get()->GetAddr());
    auto ret = strcpy_s(param.kernel.tagKey, sizeof(param.kernel.tagKey), opReq.algName.c_str());
    if (ret != EOK) {
        THROW<InternalException>(
            StringFormat("[HcclOneSidedService][BatchOpKernelLaunch], strcpy_s opReq.algName failed! ret[%d]", ret));
    }

    HCCL_INFO("[HcclOneSidedService][BatchOpKernelLaunch] OneSidedAicpuKernelLaunch start");
    // 启动kernel
    OneSidedAicpuKernelLaunch(param, *stream);
    return HCCL_SUCCESS;
}

HcclResult
HcclOneSidedService::BatchPut(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream)
{
    HCCL_INFO("[HcclOneSidedService][BatchPut] start");
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchPut] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    oneSidedConnslock.unlock();

    HCCL_INFO("[HcclOneSidedService][BatchPut] BatchOpKernelLaunch start");
    CHK_RET(BatchOpKernelLaunch(OpType::BATCHPUT, remoteRankId, desc, descNum, std::make_shared<Stream>(stream)));

    HCCL_INFO("[HcclOneSidedService][BatchPut] end");
    return HCCL_SUCCESS;
}

HcclResult
HcclOneSidedService::BatchGet(RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t stream)
{
    HCCL_INFO("[HcclOneSidedService][BatchGet] start");
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchGet] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    oneSidedConnslock.unlock();

    HCCL_INFO("[HcclOneSidedService][BatchGet] BatchOpKernelLaunch start");
    CHK_RET(BatchOpKernelLaunch(OpType::BATCHGET, remoteRankId, desc, descNum, std::make_shared<Stream>(stream)));

    HCCL_INFO("[HcclOneSidedService][BatchGet] end");
    return HCCL_SUCCESS;
}

void HcclOneSidedService::AddOpCounterMems()
{
    HCCL_INFO("[HcclOneSidedService::%s] start.", __func__);

    constexpr u64 FOUR_BYTES = 4;
    u64 size = FOUR_BYTES * 3; // 第一个四字节用于计数加1, 后面两个四字节分别保存headCounter和tailCounter
    counterBuf = std::make_shared<DevBuffer>(size);

    // 初始化第一个四字节置1, 用于计数加1, reduce task add 1
    u64 srcSize = FOUR_BYTES;
    float srcValue = 1;
    void* srcAddr = reinterpret_cast<void*>(counterBuf->GetAddr());
    HrtMemcpy(srcAddr, srcSize, &srcValue, srcSize, RT_MEMCPY_HOST_TO_DEVICE);

    // 初始化后面两个四字节置0
    u64 countMemSize = srcSize;
    float startValue = 0; // value为0表示从0开始计数
    void* headCountAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + srcSize);
    void* tailCountAddr = reinterpret_cast<void*>(counterBuf->GetAddr() + srcSize * 2);
    HrtMemcpy(headCountAddr, countMemSize, &startValue, countMemSize, RT_MEMCPY_HOST_TO_DEVICE);
    HrtMemcpy(tailCountAddr, countMemSize, &startValue, countMemSize, RT_MEMCPY_HOST_TO_DEVICE);

    HCCL_INFO(
        "[HcclOneSidedService::%s] end, counterBuf[%llu] srcAddr[%p] headCountAddr[%p] tailCountAddr[%p].", __func__,
        counterBuf->GetAddr(), srcAddr, headCountAddr, tailCountAddr);
}

DevBuffer* HcclOneSidedService::GetOpCounterBuf() { return counterBuf.get(); }
} // namespace Hccl
