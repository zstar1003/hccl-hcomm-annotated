/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <arpa/inet.h>
#include <securec.h>
#include <string>
#include "network/hccp_common.h"
#include "network_manager_pub.h"
#include "externalinput_pub.h"
#include "hccl_network.h"
#include "transport_direct_npu.h"
#include "launch_aicpu.h"
#include "acl/acl_rt.h"

using namespace std;

namespace hccl {
UniversalConcurrentMap<u64, TransportDirectNpu*> TransportDirectNpu::g_qpn2IbversLinkMap_;
bool TransportDirectNpu::g_flag = false;
bool TransportDirectNpu::g_isSupCqeErrInfoListConfig = false;
u32 TransportDirectNpu::cqeErrQpn_ = 0;

constexpr u32 DEV_PHY_ID_BIT = 32;
constexpr u32 CQE_ARRAY_SIZE = 128;

TransportDirectNpu::TransportDirectNpu(
    DispatcherPub* dispatcher, const std::unique_ptr<NotifyPool>& notifyPool, MachinePara& machinePara,
    std::chrono::milliseconds timeout)
    : TransportNet(dispatcher, notifyPool, machinePara, timeout),
      qpsPerConnection_(1),
      access_(RA_ACCESS_LOCAL_WRITE | RA_ACCESS_REMOTE_WRITE | RA_ACCESS_REMOTE_READ),
      workFlowMode_(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB),
      currentQP_(0),
      qpMode_(machinePara.qpMode)
{}

TransportDirectNpu::~TransportDirectNpu()
{
    HCCL_DEBUG("~TransportDirectNpu Enter!");

    (void)DeInit();
    UnloadAICPUKernel();

    HCCL_DEBUG("~TransportDirectNpu Success!");
}

HcclResult TransportDirectNpu::DeInit()
{
    (void)DeRegMR();

    (void)DestroyQP();

    (void)DestroyAicpuMem();

    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetRemoteMem(UserMemType memType, void** remotePtr)
{
    HCCL_INFO("[TransportDirectNpu][GetRemoteMem] direct npu getRemoteMem");
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            *remotePtr = remoteMemMsg_[static_cast<u32>(memType)].addr;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetLocalNotifyValueAddrKey(std::vector<AddrKey>& notifyValue)
{
    // 目前这里时随意填充了一些数据上去，不然会调用到基类接口，导致报错
    AddrKey notifyDetails;
    notifyDetails.addr = reinterpret_cast<u64>(memMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(memMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey);
    notifyValue.push_back(notifyDetails);
    notifyValue.push_back(notifyDetails);
    notifyValue.push_back(notifyDetails);
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetLocalRdmaNotify(std::vector<HcclSignalInfo>& rdmaNotify)
{
    HcclSignalInfo signalInfo = {};
    rdmaNotify.push_back(signalInfo);
    rdmaNotify.push_back(signalInfo);
    rdmaNotify.push_back(signalInfo);
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetRemoteRdmaNotifyAddrKey(std::vector<AddrKey>& rdmaNotifyAddr)
{
    AddrKey notifyDetails;
    notifyDetails.addr = reinterpret_cast<u64>(memMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr);
    notifyDetails.key = reinterpret_cast<u32>(memMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey);
    rdmaNotifyAddr.push_back(notifyDetails);
    rdmaNotifyAddr.push_back(notifyDetails);
    rdmaNotifyAddr.push_back(notifyDetails);
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetRemoteMemSize(UserMemType memType, u64& size)
{
    HCCL_INFO("[TransportDirectNpu][GetRemoteMem] direct npu GetRemoteMemSize");
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            size = remoteMemMsg_[static_cast<u32>(memType)].len;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::LoadBinaryFromFile(
    [[maybe_unused]] const char* binPath, [[maybe_unused]] aclrtBinaryLoadOptionType optionType,
    [[maybe_unused]] uint32_t cpuKernelMode, [[maybe_unused]] aclrtBinHandle& binHandle)
{
#ifndef CCL_KERNEL
    CHK_PRT_RET(binPath == nullptr, HCCL_ERROR("[LoadBinaryFromFile] binary path is nullptr"), HCCL_E_PTR);

    char realPath[PATH_MAX] = {0};
    CHK_PRT_RET(
        realpath(binPath, realPath) == nullptr,
        HCCL_ERROR("LoadBinaryFromFile: %s is not a valid real path, err[%d]", binPath, errno), HCCL_E_INTERNAL);
    HCCL_INFO("[LoadBinaryFromFile]realPath: %s", realPath);

    aclrtBinaryLoadOptions loadOptions = {};
    aclrtBinaryLoadOption option;
    loadOptions.numOpt = 1;
    loadOptions.options = &option;
    option.type = optionType;
    option.value.cpuKernelMode = cpuKernelMode;
    aclError aclRet
        = aclrtBinaryLoadFromFile(realPath, &loadOptions, &binHandle); // ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[LoadBinaryFromFile]errNo[0x%016llx] load binary from file error.", aclRet),
        HCCL_E_OPEN_FILE_FAILURE);
#else
    HCCL_ERROR("[AicpuAclKernelLaunch]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::LoadAICPUKernel(void)
{
#ifndef CCL_KERNEL
    std::string jsonPath;
    CHK_RET(GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";
    HcclResult ret = LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHandle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[LoadAICPUKernel]errNo[0x%016llx]load aicpu file fail, path[%s] optionType[%u] "
            "cpuKernelMode[%u].",
            ret, jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0),
        ret);
#endif
    return HCCL_SUCCESS;
}

void TransportDirectNpu::UnloadAICPUKernel(void)
{
#ifndef CCL_KERNEL
    if (binHandle_ != nullptr) {
        aclError aclRet = aclrtBinaryUnLoad(binHandle_);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[UnloadAICPUKernel]errNo[0x%016llx] unload binary from binHandle[%p] error.", aclRet, binHandle_);
        }
        binHandle_ = nullptr;
    }
#endif
    return;
}

HcclResult TransportDirectNpu::DeRegOneMR(QpHandle& qpHandle, MemMsg& memMsg)
{
    struct MrInfoT mrInfo = {};
    mrInfo.addr = memMsg.addr;
    HcclResult ret = HrtRaMrDereg(qpHandle, &mrInfo);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("errNo[0x%016llx] in link lbv, In lbv exp deconstruct, mr dereg failed.", HCCL_ERROR_CODE(ret)),
        ret);
    return HCCL_SUCCESS;
}

void TransportDirectNpu::DeRegMRForQPhandles(MemMsg& memMsg)
{
    for (u32 j = 0; j < qpHandles_.size(); j++) {
        if (qpHandles_[j] == nullptr) {
            continue;
        }
        (void)DeRegOneMR(qpHandles_[j], memMsg);
    }
}

HcclResult TransportDirectNpu::DeRegMR()
{
    /* 销毁mr */
    std::map<uintptr_t, s32> addrMap;
    for (s32 i = 0; i < static_cast<s32>(MemType::MEM_TYPE_RESERVED); i++) {
        if (memMsg_[i].mrRegFlag == REG_VALID) {
            std::pair<std::map<uintptr_t, s32>::iterator, bool> res
                = addrMap.insert(std::pair<uintptr_t, s32>(reinterpret_cast<uintptr_t>(memMsg_[i].addr), 0));
            if (res.second) {
                DeRegMRForQPhandles(memMsg_[i]);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::DestroyQpVct(std::vector<QpHandle>& qpHandles)
{
    HcclResult ret;
    for (u32 i = 0; i < qpHandles.size(); i++) {
        if (qpHandles[i] != nullptr) {
            struct QpAttr attr {};
            CHK_RET(hrtRaGetQpAttr(qpHandles[i], &attr));

            g_qpn2IbversLinkMap_.Erase(((static_cast<u64>(machinePara_.localDeviceId) << DEV_PHY_ID_BIT) | attr.qpn));

            ret = HrtRaQpDestroy(qpHandles[i]);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "errNo[0x%016llx] in link lbv, lbv exp deconstruct, qp destroy failed.", HCCL_ERROR_CODE(ret));
            }
            qpHandles[i] = nullptr;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::DestroyQP()
{
    CHK_RET(DestroyQpVct(qpHandles_));
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::DestroyAicpuMem()
{
    if (aicpuMem_.ptr() != nullptr) {
        aicpuMem_.free();
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::CreateAicpuMem()
{
    CHK_RET(DeviceMem::alloc(aicpuMem_, AICPU_FLAG_AREA));

    CHK_RET(hrtMemSet(aicpuMem_.ptr(), aicpuMem_.size(), aicpuMem_.size()));

    HCCL_INFO("[TransportDirectNpu][CreateAicpuMem] buffer ptr[%p]", aicpuMem_.ptr());
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::Init()
{
    HCCL_INFO(
        "machineType=[%d], serverId=[%s], localDeviceId=[%d], remoteDeviceId=[%d], "
        "localRank=[%u], localUserRank=[%u], remoteRank=[%u], remoteUserrank=[%u], "
        "deviceType=[%d], inputMem=[%p], outputMem=[%p], isAicpuModeEn[%d], notifyNum[%u], "
        "custom exchange data size [%llu].",
        machinePara_.machineType, machinePara_.serverId.c_str(), machinePara_.localDeviceId,
        machinePara_.remoteDeviceId, machinePara_.localUserrank, machinePara_.localWorldRank,
        machinePara_.remoteUserrank, machinePara_.remoteWorldRank, machinePara_.deviceType, machinePara_.inputMem.ptr(),
        machinePara_.outputMem.ptr(), machinePara_.isAicpuModeEn, machinePara_.notifyNum,
        machinePara_.exchangeInfo.size());
    HcclUs startut = TIME_NOW();

    CHK_SMART_PTR_NULL(machinePara_.inputMem);
    CHK_SMART_PTR_NULL(machinePara_.outputMem);
    CHK_RET(CheckDeviceId());
    CHK_RET(CheckExchangeData());
    CHK_RET(CreateAicpuMem());
    CHK_RET(LoadAICPUKernel());

    // 上层初始化时保证 machinePara_.sockets 非空
    if (machinePara_.sockets.size() == 0) {
        HCCL_ERROR("machinePara sockets is empty.");
        return HCCL_E_INTERNAL;
    }
    defaultSocket_ = machinePara_.sockets[0];
    CHK_PTR_NULL(defaultSocket_);

    CHK_RET(hrtGetDeviceType(localDeviceType));
    HCCL_INFO("localDeviceType=[%d], remoteDeviceType=[%d]", localDeviceType, machinePara_.deviceType);
    CHK_RET(GetNicHandle());

    // 设置linkType
    transportAttr_.linkType = hccl::LinkType::LINK_ROCE;

    /* 获取当前的连接模式，offline模式或者op base模式 */
    workFlowMode_ = GetWorkflowMode();
    HCCL_INFO("current work mode is [%d]", workFlowMode_);

    /* 创建QP连接 */
    CHK_RET(InitQpConnect());

    HCCL_INFO("linkexp initialization success,Time:%lld us", DURATION_US(TIME_NOW() - startut));

    CHK_RET(GetQpAttr());
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetQpAttr()
{
    char stackLogBuffer[LOG_TMPBUF_SIZE];
    s32 ret = snprintf_s(
        stackLogBuffer, LOG_TMPBUF_SIZE, LOG_TMPBUF_SIZE - 1U,
        "communicator[%s], local rank[%u], ip[%s], remote rank[%u], ip[%s], transporttype[%s]",
        machinePara_.tag.c_str(), machinePara_.localUserrank, machinePara_.localIpAddr.GetReadableAddress(),
        machinePara_.remoteUserrank, machinePara_.remoteIpAddr.GetReadableAddress(),
        GetLinkTypeEnumStr(GetLinkType()).c_str());
    CHK_PRT_RET(
        ret == -1, HCCL_ERROR("[GetQpAttr]errNo[0x%016llx] sal snprintf_s error", HCCL_ERROR_CODE(HCCL_E_INTERNAL)),
        HCCL_E_INTERNAL);
    std::string logInfo = "create hccl transport:" + std::string(stackLogBuffer);
    for (u32 i = 0; i < qpHandles_.size(); i++) {
        struct QpAttr attr {};
        hrtRaGetQpAttr(qpHandles_[i], &attr);
        HCCL_USER_CRITICAL_LOG(
            "%s, rdma qpn[%u], rdma qp sport[%u], rdma TC[%u], rdma SL[%u]", logInfo.c_str(), attr.qpn, attr.udpSport,
            machinePara_.tc, machinePara_.sl);
    }

    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::IsUseQpCreateWithAttrs(bool& isUseQpCreateWithAttrs, s32 qpMode)
{
    isUseQpCreateWithAttrs = false;
    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        bool is910Bor91093
            = machinePara_.deviceType == DevType::DEV_TYPE_910B || machinePara_.deviceType == DevType::DEV_TYPE_910_93;
        if (is910Bor91093 && (qpMode == OFFLINE_QP_MODE_EXT || qpMode == OPBASE_QP_MODE_EXT)) {
            isUseQpCreateWithAttrs = true;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::FillExchangeDataTotalSize()
{
    const uint8_t memMsgCount = 3;
    exchangeDataTotalSize_ = 0;
    exchangeDataTotalSize_ += sizeof(u32);                  // 首个内容放qp数量
    exchangeDataTotalSize_ += sizeof(MemMsg) * memMsgCount; // output and input mem and aicpu
    exchangeDataTotalSize_ += machinePara_.exchangeInfo.size();

    HCCL_DEBUG("[TransportDirectNpu][FillExchangeDataTotalSize] exchangeDataTotalSize[%llu]", exchangeDataTotalSize_);
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::ConstructExchangeForSend()
{
    exchangeDataForSend_.resize(exchangeDataTotalSize_);
    u8* exchangeDataPtr = exchangeDataForSend_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;
    // 把qp对数量放在最前头，第一个做检验
    u32 qpNum = 1;
    s32 sRet = memcpy_s(exchangeDataPtr, sizeof(u32), reinterpret_cast<void*>(&qpNum), sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Set][LocalMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    CHK_RET(RegUserMem(MemType::AICPU_SYNC_MEM, exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(RegUserMem(MemType::USER_OUTPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(RegUserMem(MemType::USER_INPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));
    CHK_RET(ConstructExchangeDataForSend(exchangeDataPtr, exchangeDataBlankSize));

    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportDirectNpu][ConstructExchangeForSend] failed to construct exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[TransportDirectNpu] ConstructExchangeForSend finished.");
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::ParseReceivedExchangeData()
{
    u8* exchangeDataPtr = exchangeDataForRecv_.data();
    u64 exchangeDataBlankSize = exchangeDataTotalSize_;

    // 首先解析qp对数量，并作一致性校验
    u32 localQpNum = 1;
    u32 remoteQpNum = 0;
    s32 sRet = memcpy_s(reinterpret_cast<void*>(&remoteQpNum), sizeof(u32), exchangeDataPtr, sizeof(u32));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteMem]errNo[0x%016llx] memory copy failed. errorno[%d], params:dstMaxSize[%zu],cnt[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(u32), sizeof(u32)),
        HCCL_E_MEMORY);
    CHK_PRT_RET(
        localQpNum != remoteQpNum,
        HCCL_ERROR(
            "[TransportDirectNpu][ParseReceivedExchangeData]"
            "local qps[%u] not equal to remote qps[%u], rank:local[%u],remote[%u]",
            localQpNum, remoteQpNum, machinePara_.localUserrank, machinePara_.remoteUserrank),
        HCCL_E_INTERNAL);
    exchangeDataPtr += sizeof(u32);
    exchangeDataBlankSize -= sizeof(u32);

    CHK_RET(GetRemoteAddr(MemType::AICPU_SYNC_MEM, exchangeDataPtr, exchangeDataBlankSize));

    CHK_RET(GetRemoteAddr(MemType::USER_OUTPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));

    CHK_RET(GetRemoteAddr(MemType::USER_INPUT_MEM, exchangeDataPtr, exchangeDataBlankSize));

    CHK_RET(ParseExchangeData(exchangeDataPtr, exchangeDataBlankSize));

    if (exchangeDataBlankSize != 0) {
        HCCL_ERROR(
            "[TransportDirectNpu][ParseReceivedExchangeData] failed to Parse exchange Data "
            "exchangeDataBlankSize[%llu]",
            exchangeDataBlankSize);
        return HCCL_E_INTERNAL;
    }
    HCCL_DEBUG("Parse Received ExchangeData success!");
    return HCCL_SUCCESS;
}

u32 TransportDirectNpu::GetQpsPerConnection()
{
    u32 externalQps = std::max(static_cast<u32>(machinePara_.srcPorts.size()), 1U);
    s32 qpMode = GetQpMode();
    if (GetWorkflowMode() != HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE
        && externalQps != HCCL_QPS_PER_CONNECTION_DEFAULT) {
        HCCL_RUN_INFO(
            "HCCL_RDMA_QPS_PER_CONNECTION is set to [%u] but it is not effective in offline mode.", externalQps);
    } else if (qpMode != OPBASE_QP_MODE_EXT && externalQps > 1) {
        HCCL_RUN_INFO(
            "HCCL_RDMA_QPS_PER_CONNECTION is set to [%u] but current devType[%d] does not support multi-QP.",
            externalQps, machinePara_.deviceType);
        return 1; // 非单算子模式仅支持单QP， QPS = 1
    }
    if (GetWorkflowMode() == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
        return externalQps; // only work for opbase mode
    }
    return 1; // 非单算子模式仅支持单QP， QPS = 1
}

HcclResult TransportDirectNpu::GetNicHandle()
{
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(machinePara_.deviceLogicId).GetRaResourceInfo(raResourceInfo));
    std::map<HcclIpAddress, IpSocket>& tmpSocketMap = machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE ?
                                                          raResourceInfo.nicSocketMap :
                                                          raResourceInfo.hostNetSocketMap;

    HcclIpAddress localIpAddr = machinePara_.localIpAddr;

    // 获取 nicRdmaHandle
    auto itSocket = tmpSocketMap.find(localIpAddr);
    if (itSocket == tmpSocketMap.end()) {
        HCCL_ERROR(
            "[Get][NicHandle]In get nic handle, can not find socket handle, handle size[%u], "
            "local ip[%s]",
            tmpSocketMap.size(), localIpAddr.GetReadableAddress());
        return HCCL_E_PARA;
    }

    nicRdmaHandle_ = itSocket->second.nicRdmaHandle;
    CHK_PTR_NULL(nicRdmaHandle_);

    return HCCL_SUCCESS;
}

// 创建一个QP
HcclResult TransportDirectNpu::CreateOneQp(
    s32 qpMode, u32 qpsPerConnection, QpHandle& qpHandle, AiQpInfo& aiQpInfo, bool useAicpu, u32 udpSport)
{
    bool isUseQpCreateWithAttrs = false;
    CHK_RET(IsUseQpCreateWithAttrs(isUseQpCreateWithAttrs, qpMode));
    HcclResult ret;
    std::string useAicpuTitle = useAicpu ? std::string("aicpu ") : std::string("");
    std::string qpInfo = useAicpuTitle + std::string("rank:") + std::to_string(machinePara_.localWorldRank)
                         + std::string(",localUserrank:") + std::to_string(machinePara_.localUserrank)
                         + std::string(",localIpAddr: ") + std::string(machinePara_.localIpAddr.GetReadableAddress())
                         + std::string(",deviceLogicId:") + std::to_string(machinePara_.deviceLogicId);
    struct QpExtAttrs attrs {};
    // 判断是否为NORMALQP需要使用qpMode_; hostnic场景的qpmode也是NORMALQP
    if (useAicpu || qpMode_ == QPMode::NORMAL) {
        bool isWorkFlowLib = (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        CHK_RET(ConstructQpAttrs(qpMode, attrs, machinePara_.queueDepthAttr, isWorkFlowLib));

        // A3 aicpu图模式使用单个qp, qp深度为socket数量*128
        bool isAicpuLib = machinePara_.isAicpuModeEn && (machinePara_.deviceType == DevType::DEV_TYPE_910_93)
                          && (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
        if (isAicpuLib) {
            attrs.qpAttr.cap.max_send_wr = machinePara_.sockets.size() * DEFAULT_OFFLINE_MAX_SEND_WR;
        }
        HCCL_DEBUG(
            "qp set max_send_wr %u, socket size %u, isWorkFlowLib %d", attrs.qpAttr.cap.max_send_wr,
            machinePara_.sockets.size(), isWorkFlowLib);

        attrs.udpSport = udpSport;
        ret = hrtRaAiQpCreate(machinePara_.localDeviceId, nicRdmaHandle_, &attrs, &aiQpInfo, qpHandle);
        HCCL_DEBUG("aiQpAddr:%llu db_index:%u, sq_index=%u", aiQpInfo.aiQpAddr, aiQpInfo.dbIndex, aiQpInfo.sqIndex);
        qpInfo = qpInfo + std::string(",sendCqDepth:") + std::to_string(attrs.cqAttr.sendCqDepth);
    } else if (!isUseQpCreateWithAttrs && qpsPerConnection == HCCL_QPS_PER_CONNECTION_DEFAULT) {
        ret = HrtRaQpCreate(nicRdmaHandle_, QP_FLAG_RC, qpMode, qpHandle);
    } else if (!isUseQpCreateWithAttrs && qpsPerConnection != HCCL_QPS_PER_CONNECTION_DEFAULT) {
        HCCL_ERROR("qpsPerConnection[%u] is set but qpMode[%d] is not supported", qpsPerConnection, qpMode);
        return HCCL_E_PARA;
    } else {
        CHK_RET(ConstructQpAttrs(qpMode, attrs, machinePara_.queueDepthAttr));
        attrs.udpSport = udpSport;
        ret = hrtRaQpCreateWithAttrs(nicRdmaHandle_, &attrs, qpHandle);
        qpInfo = qpInfo + std::string(",sendCqDepth:") + std::to_string(attrs.cqAttr.sendCqDepth);
    }

    RPT_ENV_ERR(
        ret != 0 || (qpHandle == nullptr), "EI0007", vector<string>({"resource_type", "resource_info"}),
        vector<string>({"qp", "CreateOneQp"}));

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]create qp failed, localDeviceId[%d], qpMode[%d]", LOG_KEYWORDS_INIT_GROUP.c_str(),
            LOG_KEYWORDS_RESOURCE.c_str(), machinePara_.localDeviceId, qpMode),
        HCCL_E_ROCE_CONNECT);

    // 表示没有通过config配置，则使用环境变量配置
    CHK_RET(SetQpAttrQos(qpHandle, machinePara_.tc, machinePara_.sl));
    // 配置RDMA Timeout时间
    CHK_RET(SetQpAttrTimeOut(qpHandle));
    // 配置RDMA Retry Cnt重传次数
    CHK_RET(SetQpAttrRetryCnt(qpHandle));
    // qpn map 插入
    struct QpAttr attr {};
    CHK_RET(hrtRaGetQpAttr(qpHandle, &attr));

    g_qpn2IbversLinkMap_.Emplace(((static_cast<u64>(machinePara_.localDeviceId) << DEV_PHY_ID_BIT) | attr.qpn), this);

    HCCL_DEBUG("ra qp create success.");
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::CreateSingleQp(s32 qpMode) // 根据socket个数创建QP（下沉模板不够用多QP）
{
    u32 socketNum = 1;
    // A3 aicpu图模式只使用1个qp，qp深度为socketNum*128，最大不超过32K
    bool isAicpuLib = machinePara_.isAicpuModeEn && (machinePara_.deviceType == DevType::DEV_TYPE_910_93)
                      && (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OPS_KERNEL_INFO_LIB);
    if (!isAicpuLib) {
        socketNum = std::max(static_cast<u32>(machinePara_.sockets.size()), socketNum);
    }
    // 原来是 machinePara_.socketFdHandles 换成 machinePara_.sockets
    for (u32 i = 0; i < socketNum; i++) {
        QpHandle qpHandle = nullptr;
        u32 udpSport = machinePara_.srcPorts.empty() ? 0 : machinePara_.srcPorts[0];
        CHK_RET(CreateOneQp(
            qpMode, HCCL_QPS_PER_CONNECTION_DEFAULT, qpHandle, aiQpInfo_, machinePara_.isAicpuModeEn, udpSport));
        qpHandles_.push_back(qpHandle);
    }
    return HCCL_SUCCESS;
}

s32 TransportDirectNpu::GetQpMode()
{
    s32 qpMode = NORMAL_QP_MODE;

    if (qpMode_ == QPMode::NORMAL) {
        return qpMode;
    }

    if (machinePara_.nicDeploy == NICDeployment::NIC_DEPLOYMENT_DEVICE) {
        if (workFlowMode_ == HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE) {
            qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                      || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                         OPBASE_QP_MODE_EXT :
                         OPBASE_QP_MODE;
            // isCapture需要创建下沉QP
            qpMode = (qpMode == OPBASE_QP_MODE_EXT && qpMode_ == QPMode::OFFLOAD) ? OFFLINE_QP_MODE_EXT : qpMode;
        } else {
            qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                      || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                         OFFLINE_QP_MODE_EXT :
                         OFFLINE_QP_MODE;
        }
    }
    if (machinePara_.isAicpuModeEn) {
        qpMode = (machinePara_.deviceType == DevType::DEV_TYPE_910B
                  || machinePara_.deviceType == DevType::DEV_TYPE_910_93) ?
                     OPBASE_QP_MODE_EXT :
                     OPBASE_QP_MODE;
    }
    return qpMode;
}

HcclResult TransportDirectNpu::CreateQp()
{
    s32 qpMode = GetQpMode();
    HCCL_DEBUG("[TransportDirectNpu][CreateQp] QpMode[%u]", qpMode);
    CHK_RET(CreateSingleQp(qpMode));
    HCCL_DEBUG("ra qp create %u qp success.", qpHandles_.size());
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::InitQpConnect()
{
    /* 创建QP操作句柄 */
    qpsPerConnection_ = GetQpsPerConnection();

    CHK_RET(CreateQp());

    CHK_RET(FillExchangeDataTotalSize());

    CHK_RET(ConstructExchangeForSend());

    HCCL_DEBUG("[TransportDirectNpu] resource create done exchangeDataTotalSize_[%llu]", exchangeDataTotalSize_);

    HcclResult ret = defaultSocket_->Send(exchangeDataForSend_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportDirectNpu][InitQpConnect] failed to send exchangeData exchangeDataTotalSize[%llu], "
            "custom exchange data size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);
    HCCL_DEBUG("[TransportDirectNpu]Socket Send finished, exchangeDataTotalSize[%llu]", exchangeDataTotalSize_);

    exchangeDataForRecv_.resize(exchangeDataTotalSize_);
    ret = defaultSocket_->Recv(exchangeDataForRecv_.data(), exchangeDataTotalSize_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[TransportDirectNpu][InitQpConnect] failed to recv exchangeData exchangeDataTotalSize[%llu], "
            "custom exchange data size [%llu].",
            exchangeDataTotalSize_, machinePara_.exchangeInfo.size()),
        ret);

    HCCL_DEBUG("[TransportDirectNpu][Init] Socket Data Recved");

    CHK_RET(ParseReceivedExchangeData());

    // 连接Qp
    CHK_RET(ConnectQp());
    HCCL_INFO("In link ibv, qp status has ready");
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::ConnectSingleQp(std::function<bool()> needStop)
{
    // QP建链
    for (u32 i = 0; i < qpHandles_.size(); i++) {
        CHK_RET(HrtRaQpConnectAsync(qpHandles_[i], machinePara_.sockets[i]->GetFdHandle(), needStop));
    }
    // 查询QP建链是否成功
    s32 qpStatus = 0;
    s32 raRet = 0;
    auto startTime = std::chrono::steady_clock::now();
    HCCL_INFO("In link ibv, waiting for qp status ready...");
    for (u32 i = 0; i < qpHandles_.size(); i++) {
        while (true) {
            CHK_PRT_RET(needStop(), HCCL_ERROR("Terminating operation due to external request"), HCCL_E_INTERNAL);

            if ((std::chrono::steady_clock::now() - startTime) >= timeout_) {
                HCCL_ERROR("[Connect][Qp]get qp status timeout_=[%lld ms], qp_status=[%d]", timeout_, qpStatus);
                return HCCL_E_TIMEOUT;
            }
            raRet = hrtGetRaQpStatus(qpHandles_[i], &qpStatus);
            if ((raRet == 0) && (qpStatus == 1)) { // 为1时，qp 建链成功
                HCCL_INFO("In link ibv, %u of %u QP get status success.", (i + 1), qpHandles_.size());
                break;
            } else {
                // qp建链需要时间，获取qp状态直至超时
                SaluSleep(WAIT_US_COUNT);
            }
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::ConnectQp()
{
    CHK_RET(ConnectSingleQp([this]() -> bool {
        return this->GetStopFlag();
    }));
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::TxAsync(UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, Stream& stream)
{
    CHK_RET(TxData(dstMemType, dstOffset, src, len, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::TxAsync(std::vector<TxMemoryInfo>& txMems, Stream& stream)
{
    for (auto& mem : txMems) {
        CHK_RET(TxAsync(mem.dstMemType, mem.dstOffset, mem.src, mem.len, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::RxAsync(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, Stream& stream)
{
    CHK_PRT(RxData(srcMemType, srcOffset, dst, len, stream));
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::RxAsync(std::vector<RxMemoryInfo>& rxMems, Stream& stream)
{
    for (auto& mem : rxMems) {
        CHK_RET(RxAsync(mem.srcMemType, mem.srcOffset, mem.dst, mem.len, stream));
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::DataReceivedAck(Stream& stream)
{
    CHK_RET(PostFinAck(stream));
    CHK_RET(WaitFinAck(stream));

    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::TxWaitDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

/* 发送ack消息(同步模式) */
HcclResult TransportDirectNpu::TxAck([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

/* 接收ack消息(同步模式) */
HcclResult TransportDirectNpu::RxAck([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::TxDataSignal([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::RxDataSignal([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::RegUserMem(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    void* memPtr = nullptr;
    u64 memSize;
    switch (memType) {
        case MemType::USER_INPUT_MEM: {
            memPtr = machinePara_.inputMem.ptr();
            memSize = machinePara_.inputMem.size();
            break;
        }

        case MemType::USER_OUTPUT_MEM: {
            memPtr = machinePara_.outputMem.ptr();
            memSize = machinePara_.outputMem.size();
            break;
        }

        case MemType::AICPU_SYNC_MEM: {
            memPtr = aicpuMem_.ptr();
            memSize = aicpuMem_.size();
            break;
        }

        default: {
            HCCL_ERROR("[Reg][UserMem]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    struct MrInfoT mrInfo = {};
    mrInfo.addr = memPtr;
    mrInfo.size = memSize;
    mrInfo.access = access_;
    for (u32 i = 0; i < qpHandles_.size(); i++) {
        CHK_RET(HrtRaMrReg(qpHandles_[i], &mrInfo));
    }

    memMsg_[static_cast<u32>(memType)].mrRegFlag = REG_VALID;
    memMsg_[static_cast<u32>(memType)].addr = memPtr;
    memMsg_[static_cast<u32>(memType)].len = memSize;
    memMsg_[static_cast<u32>(memType)].memType = memType;
    memMsg_[static_cast<u32>(memType)].lkey = mrInfo.lkey;

    CHK_SAFETY_FUNC_RET(memcpy_s(
        exchangeDataPtr, exchangeDataBlankSize, reinterpret_cast<void*>(&memMsg_[static_cast<u32>(memType)]),
        sizeof(MemMsg)));

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);

    HCCL_DEBUG("memType=%d mem_ptr=%p mem_size=%llu Byte, key = %u", memType, memPtr, memSize, mrInfo.lkey);

    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetMemInfo(UserMemType memType, void** dstMemPtr, u64* dstMemSize)
{
    switch (memType) {
        case UserMemType::INPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr;
            *dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].len;
            break;
        }

        case UserMemType::OUTPUT_MEM: {
            *dstMemPtr = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr;
            *dstMemSize = remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].len;
            break;
        }

        default: {
            HCCL_ERROR("[Get][MemInfo]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetRemoteAddr(MemType memType, u8*& exchangeDataPtr, u64& exchangeDataBlankSize)
{
    s32 sRet = memcpy_s(&remoteMemMsg_[static_cast<u32>(memType)], sizeof(MemMsg), exchangeDataPtr, sizeof(MemMsg));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Get][RemoteAddr]errNo[0x%016llx] In lbv exp get remote addr, "
            "memcpy failed. errorno[%d], params:destMaxSize[%zu],count[%zu]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), sRet, sizeof(MemMsg), sizeof(MemMsg)),
        HCCL_E_MEMORY);
    CHK_PTR_NULL(remoteMemMsg_[static_cast<u32>(memType)].addr);

    exchangeDataPtr += sizeof(MemMsg);
    exchangeDataBlankSize -= sizeof(MemMsg);
    HCCL_INFO(
        "GetRemoteAddr success: memType=%d, addr=%p len=%llu, notifyId=%u", static_cast<int32_t>(memType),
        remoteMemMsg_[static_cast<u32>(memType)].addr, remoteMemMsg_[static_cast<u32>(memType)].len,
        remoteMemMsg_[static_cast<u32>(memType)].notifyId);
    return HCCL_SUCCESS;
}

/* 发送ack消息(同步模式) */
HcclResult TransportDirectNpu::TxPrepare([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

/* 接收ack消息(同步模式) */
HcclResult TransportDirectNpu::RxPrepare([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::TxData(
    UserMemType dstMemType, u64 dstOffset, const void* src, u64 len, [[maybe_unused]] Stream& stream)
{
    CHK_PTR_NULL(src);
    struct ApiParamDef {
        u32 lKey;
        u32 rKey;
        HcclQpInfoV2 qpInfo;
        u64 remoteAddr;
        u64 localAddr;
        u64 dataSize;
        u64 timeout;
        u64 localFlagAddr;
        u64 remoteFlagAddr;
        u32 lfKey;
        u32 rfKey;
    };
    const std::string kernelName = "RunTransportRoceTx";
    struct ApiParamDef apiParam = {};
    MemDetails inputMemDetails;
    MemDetails outputMemDetails;
    CHK_PRT(GetLocalMemDetails(UserMemType::INPUT_MEM, inputMemDetails));
    CHK_PRT(GetLocalMemDetails(UserMemType::OUTPUT_MEM, outputMemDetails));

    if (reinterpret_cast<u64>(src) >= inputMemDetails.addr
        && reinterpret_cast<u64>(src) < inputMemDetails.addr + inputMemDetails.size) {
        apiParam.lKey = inputMemDetails.key;
    } else if (
        reinterpret_cast<u64>(src) >= outputMemDetails.addr
        && reinterpret_cast<u64>(src) <= outputMemDetails.addr + outputMemDetails.size) {
        apiParam.lKey = outputMemDetails.key;
    } else {
        HCCL_ERROR(
            "[TransportDirectNpu][TxData]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            src, inputMemDetails.addr, inputMemDetails.size, outputMemDetails.addr, outputMemDetails.size);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetRemoteMemKey(dstMemType, &apiParam.rKey));
    void* remoteAddr = nullptr;
    u64 memSize = 0;
    CHK_RET(GetMemInfo(dstMemType, &remoteAddr, &memSize));
    apiParam.remoteFlagAddr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].addr);
    apiParam.lfKey = memMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].lkey;
    apiParam.rfKey = remoteMemMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].lkey;
    apiParam.remoteAddr = reinterpret_cast<u64>(remoteAddr) + dstOffset;
    apiParam.dataSize = len;
    apiParam.localAddr = reinterpret_cast<u64>(src);
    apiParam.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
        || dispatcher_->GetExecTimeOutSet()) {
        apiParam.timeout = dispatcher_->GetExecTimeOut();
    }
    apiParam.localFlagAddr = reinterpret_cast<u64>(aicpuMem_.ptr());
    std::vector<HcclQpInfoV2> aiQpInfos;
    CHK_RET(GetAiQpInfo(aiQpInfos));
    apiParam.qpInfo = aiQpInfos[0];
    HCCL_INFO(
        "[TransportDirectNpu][TxData]localRank %u remoteRank %u lkey %u rkey %u remoteAddr %p localAddr %p dataSize "
        "%llu "
        "timeout %llu localFlagAddr %p remoteFlagAddr %p lfkey %u rfkey %u qpinfo %llu",
        machinePara_.localUserrank, machinePara_.remoteUserrank, apiParam.lKey, apiParam.rKey, apiParam.remoteAddr,
        apiParam.localAddr, apiParam.dataSize, apiParam.timeout, apiParam.localFlagAddr, apiParam.remoteFlagAddr,
        apiParam.lfKey, apiParam.rfKey, apiParam.qpInfo.qpPtr);

#ifndef CCL_KERNEL
    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      NOTIFY_DEFAULT_WAIT_TIME;
    CHK_PRT(AicpuAclKernelLaunch(
        stream.ptr(), reinterpret_cast<void*>(&apiParam), sizeof(apiParam), binHandle_, kernelName, true, timeOut));
#else
    HCCL_ERROR("[AicpuAclKernelLaunch]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
    HCCL_INFO("[TransportDirectNpu][TxData] exec succ.");
    return HCCL_SUCCESS;
}

HcclResult
TransportDirectNpu::RxData(UserMemType srcMemType, u64 srcOffset, void* dst, u64 len, [[maybe_unused]] Stream& stream)
{
    CHK_PTR_NULL(dst);
    struct ApiParamDef {
        u32 lKey;
        u32 rKey;
        HcclQpInfoV2 qpInfo;
        u64 remoteAddr;
        u64 localAddr;
        u64 dataSize;
        u64 timeout;
        u64 localFlagAddr;
        u64 remoteFlagAddr;
        u32 lfKey;
        u32 rfKey;
    };
    const std::string kernelName = "RunTransportRoceRx";
    struct ApiParamDef apiParam = {};
    MemDetails inputMemDetails;
    MemDetails outputMemDetails;
    CHK_PRT(GetLocalMemDetails(UserMemType::INPUT_MEM, inputMemDetails));
    CHK_PRT(GetLocalMemDetails(UserMemType::OUTPUT_MEM, outputMemDetails));

    if (reinterpret_cast<u64>(dst) >= inputMemDetails.addr
        && reinterpret_cast<u64>(dst) < inputMemDetails.addr + inputMemDetails.size) {
        apiParam.lKey = inputMemDetails.key;
    } else if (
        reinterpret_cast<u64>(dst) >= outputMemDetails.addr
        && reinterpret_cast<u64>(dst) <= outputMemDetails.addr + outputMemDetails.size) {
        apiParam.lKey = outputMemDetails.key;
    } else {
        HCCL_ERROR(
            "[TransportDirectNpu][RxData]src_ptr=%p is out of range, inputmem src[%p], size[%llu];"
            " outputmem src[%p] size[%llu]",
            dst, inputMemDetails.addr, inputMemDetails.size, outputMemDetails.addr, outputMemDetails.size);
        return HCCL_E_INTERNAL;
    }
    CHK_RET(GetRemoteMemKey(srcMemType, &apiParam.rKey));
    void* remoteAddr = nullptr;
    u64 memSize = 0;
    CHK_RET(GetMemInfo(srcMemType, &remoteAddr, &memSize));
    apiParam.remoteFlagAddr = reinterpret_cast<u64>(remoteMemMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].addr);
    apiParam.lfKey = memMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].lkey;
    apiParam.rfKey = remoteMemMsg_[static_cast<u32>(MemType::AICPU_SYNC_MEM)].lkey;
    apiParam.remoteAddr = reinterpret_cast<u64>(remoteAddr) + srcOffset;
    apiParam.dataSize = len;
    apiParam.localAddr = reinterpret_cast<u64>(dst);
    apiParam.timeout = NOTIFY_DEFAULT_WAIT_TIME;
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
        || dispatcher_->GetExecTimeOutSet()) {
        apiParam.timeout = dispatcher_->GetExecTimeOut();
    }
    apiParam.localFlagAddr = reinterpret_cast<u64>(aicpuMem_.ptr());
    std::vector<HcclQpInfoV2> aiQpInfos;
    CHK_RET(GetAiQpInfo(aiQpInfos));
    apiParam.qpInfo = aiQpInfos[0];
    HCCL_INFO(
        "[TransportDirectNpu][RxData]localRank %u remoteRank %u lkey %u rkey %u remoteAddr %p localAddr %p dataSize "
        "%llu "
        "timeout %llu localFlagAddr %p remoteFlagAddr %p lfkey %u rfkey %u qpinfo %llu",
        machinePara_.localUserrank, machinePara_.remoteUserrank, apiParam.lKey, apiParam.rKey, apiParam.remoteAddr,
        apiParam.localAddr, apiParam.dataSize, apiParam.timeout, apiParam.localFlagAddr, apiParam.remoteFlagAddr,
        apiParam.lfKey, apiParam.rfKey, apiParam.qpInfo.qpPtr);

#ifndef CCL_KERNEL
    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      NOTIFY_DEFAULT_WAIT_TIME;
    CHK_PRT(AicpuAclKernelLaunch(
        stream.ptr(), reinterpret_cast<void*>(&apiParam), sizeof(apiParam), binHandle_, kernelName, true, timeOut));
#else
    HCCL_ERROR("[AicpuAclKernelLaunch]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::TxDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::RxDone([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::PostFin([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::WaitFin([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::PostFinAck([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::WaitFinAck([[maybe_unused]] Stream& stream) { return HCCL_SUCCESS; }

HcclResult TransportDirectNpu::GetRemoteMemKey(UserMemType memType, uint32_t* remoteMemKey)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            *remoteMemKey = remoteMemMsg_[static_cast<u32>(memType)].lkey;
            break;

        default:
            HCCL_ERROR("[Get][RemoteMemKey]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetLocalMemDetails(UserMemType memType, MemDetails& memDetails)
{
    switch (memType) {
        case UserMemType::INPUT_MEM:
        case UserMemType::OUTPUT_MEM:
            memDetails.addr = reinterpret_cast<u64>(memMsg_[static_cast<u32>(memType)].addr);
            memDetails.size = memMsg_[static_cast<u32>(memType)].len;
            memDetails.key = memMsg_[static_cast<u32>(memType)].lkey;
            break;

        default:
            HCCL_ERROR("[Get][LocalMemDetails]not support dst_mem_type=%d", memType);
            return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetAiQpInfo(std::vector<HcclQpInfoV2>& aiQpInfo)
{
    aiQpInfo.resize(aiQpInfos_.size() + 1);

    aiQpInfo[0].qpPtr = aiQpInfo_.aiQpAddr;
    aiQpInfo[0].sqIndex = aiQpInfo_.sqIndex;
    aiQpInfo[0].dbIndex = aiQpInfo_.dbIndex;
    HCCL_DEBUG(
        "[TransportDirectNpu][GetAiQpInfo] i[0] qpPtr[%llu] sqIndex[%u] dbIndex[%u]", aiQpInfo[0].qpPtr,
        aiQpInfo[0].sqIndex, aiQpInfo[0].dbIndex);
    for (u32 i = 1, j = 0; i < aiQpInfo.size(); i++, j++) {
        aiQpInfo[i].qpPtr = aiQpInfos_[j].aiQpAddr;
        aiQpInfo[i].sqIndex = aiQpInfos_[j].sqIndex;
        aiQpInfo[i].dbIndex = aiQpInfos_[j].dbIndex;
        HCCL_DEBUG(
            "[TransportDirectNpu][GetAiQpInfo] i[%u] qpPtr[%llu] sqIndex[%u] dbIndex[%u]", i, aiQpInfo[i].qpPtr,
            aiQpInfo[i].sqIndex, aiQpInfo[i].dbIndex);
    }
    return HCCL_SUCCESS;
}

HcclResult TransportDirectNpu::GetTransportErrorCqe(
    const HcclNetDevCtx netDevCtx, std::vector<std::pair<TransportBase*, CqeInfo>>& infos, u32& num)
{
    if (g_qpn2IbversLinkMap_.Size() == 0) {
        num = 0;
        return HCCL_SUCCESS;
    }

    if (UNLIKELY(!g_flag)) {
        CHK_RET(IsSuppCqeErrInfoListConfig(g_isSupCqeErrInfoListConfig));
        g_flag = true;
    }

    CHK_PTR_NULL(netDevCtx);
    s32 deviceLogicId = (static_cast<NetDevContext*>(netDevCtx))->GetLogicId();
    s32 devicePhyId = (static_cast<NetDevContext*>(netDevCtx))->GetPhyId();
    HcclIpAddress localIp = (static_cast<NetDevContext*>(netDevCtx))->GetLocalIp();
    NicType nicType = (static_cast<NetDevContext*>(netDevCtx))->GetNicType();
    CHK_PRT_RET(
        nicType == NicType::HOST_NIC_TYPE,
        HCCL_WARNING("[TransportDirectNpu][GetTransportErrorCqe] nicType[%d] not support", nicType), HCCL_SUCCESS);
    RaResourceInfo raResourceInfo;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId).GetRaResourceInfo(raResourceInfo));
    RdmaHandle rdmaHandle = raResourceInfo.nicSocketMap[localIp].nicRdmaHandle;
    CHK_PTR_NULL(rdmaHandle);

    if (g_isSupCqeErrInfoListConfig) {
        u32 loop = 0;
        if (num > CQE_ARRAY_SIZE) {
            loop = (num % CQE_ARRAY_SIZE != 0) ? (num / CQE_ARRAY_SIZE) : ((num / CQE_ARRAY_SIZE) - 1);
        }

        struct CqeErrInfo infolist[CQE_ARRAY_SIZE] = {};
        u32 cqeNum = CQE_ARRAY_SIZE;
        for (u32 index = 0; index <= loop; index++) {
            cqeNum = (index == loop) ? (num - index * CQE_ARRAY_SIZE) : CQE_ARRAY_SIZE;
            u32 temNum = cqeNum;
            CHK_RET(hrtRaGetCqeErrInfoList(rdmaHandle, infolist, &temNum));
            ProcessCqeInfo(devicePhyId, infolist, temNum, infos);
            if (temNum < cqeNum) {
                break;
            }
        }
    } else {
        struct CqeErrInfo infolist[1] = {};
        CHK_RET(hrtRaGetCqeErrInfo(devicePhyId, &infolist[0]));
        if (infolist[0].status == 0) {
            num = 0;
            return HCCL_SUCCESS;
        }
        u32 cqeNum = 1;
        ProcessCqeInfo(devicePhyId, infolist, cqeNum, infos);
    }

    num = infos.size();

    return HCCL_SUCCESS;
}

void TransportDirectNpu::ProcessCqeInfo(
    const s32 deviceId, const struct CqeErrInfo* infolist, const u32 cqeNum,
    std::vector<std::pair<TransportBase*, CqeInfo>>& infos)
{
    for (u32 i = 0; i < cqeNum; i++) {
        // localPhyId + qpn
        auto it = g_qpn2IbversLinkMap_.Find(((static_cast<u64>(deviceId) << DEV_PHY_ID_BIT) | infolist[i].qpn));
        if (it.second) {
            TransportBase* ptr = reinterpret_cast<TransportBase*>(it.first->second);
            infos.push_back(
                std::make_pair(ptr, CqeInfo(infolist[i].time, infolist[i].status, it.first->second->GetRemoteIp())));
            cqeErrQpn_ = infolist[i].qpn;
        } else {
            HCCL_RUN_WARNING("[GetTransportErrorCqe]get err failed, transport was not found.");
        }
    }
    return;
}

HcclIpAddress& TransportDirectNpu::GetRemoteIp() { return machinePara_.remoteIpAddr; }

HcclResult TransportDirectNpu::GetTransportId(u32& id)
{
    id = cqeErrQpn_;
    return HCCL_SUCCESS;
}
} // namespace hccl
