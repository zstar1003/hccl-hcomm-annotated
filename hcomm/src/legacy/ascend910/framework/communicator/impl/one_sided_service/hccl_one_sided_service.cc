/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_one_sided_service.h"
#include <future>
#include "device_capacity.h"
#include "sal_pub.h"
#include "threads_guard.h"
#include "adapter_rts_common.h"
#include "adapter_prof.h"
#include "profiling_manager_pub.h"
#include "prof_common.h"
#include "launch_aicpu.h"
#include "launch_device.h"
#include "comm_configer.h"

namespace hccl {
using namespace std;
constexpr u32 INVALID_REMOTE_RANK_ID = 0xFFFFFFFF;
constexpr u64 TILINGDATA_BUF_SIZE = 32 * 1024;
constexpr u16 MAX_VALUE_U16 = 0xFFFF;

std::mutex HcclOneSidedService::regMutex_;

std::unique_ptr<Stream> g_launchStream = nullptr;
std::mutex g_launchMutex;

HcclOneSidedService::HcclOneSidedService(
    unique_ptr<HcclSocketManager>& socketManager, unique_ptr<NotifyPool>& notifyPool, const CommConfig& commConfig)
    : IHcclOneSidedService(socketManager, notifyPool)
{
    commConfig_ = commConfig;
}

HcclOneSidedService::~HcclOneSidedService()
{
    HCCL_RUN_INFO(
        "[~HcclOneSidedService] localRankId[%u] has registeredMemCnt[%u] mem didn't dereg", localRankInfo_.userRank,
        registedMemCnt_);
    HcclResult ret = HCCL_SUCCESS;
    for (auto it = desc2HcclBufMapIpc_.begin(); it != desc2HcclBufMapIpc_.end(); ++it) {
        HcclBuf& buf = it->second;
        do {
            ret = HcclMemDereg(&buf); // 需循环调用DeregMem来去注册内存(因为存在一块内存多次Reg的情况)
            // 失败场景记录log即可，接着处理后面的mem
            CHK_PRT_CONT(
                ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)),
                HCCL_ERROR(
                    "[~HcclOneSidedService] DeregMem IPC localRankId[%u] addr[%p] size[%lu] failed",
                    localRankInfo_.userRank, buf.addr, buf.len));
        } while (ret == HCCL_E_AGAIN);
    }

    for (auto it = desc2HcclBufMapRoce_.begin(); it != desc2HcclBufMapRoce_.end(); ++it) {
        HcclBuf& buf = it->second;
        do {
            ret = HcclMemDereg(&buf); // 需循环调用DeregMem来去注册内存(因为存在一块内存多次Reg的情况)
            // 失败场景记录log即可，接着处理后面的mem
            CHK_PRT_CONT(
                ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)),
                HCCL_ERROR(
                    "[~HcclOneSidedService] DeregMem ROCE localRankId[%u] addr[%p] size[%lu] failed",
                    localRankInfo_.userRank, buf.addr, buf.len));
        } while (ret == HCCL_E_AGAIN);
    }

    for (u32 i = 0; i < localAicpuNotify_.size(); ++i) {
        if (localAicpuNotify_[i] != nullptr) {
            ret = localAicpuNotify_[i]->Destroy();
            localAicpuNotify_[i] = nullptr;
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[Destroy][AiCpuNotify] errNo[0x%016llx] notify destroy fail, aicpuNotify[%u], ret[%d].",
                    HCCL_ERROR_CODE(HCCL_E_RUNTIME), i, ret);
            }
        }
    }
    UnloadAICPUKernel();
}

HcclResult HcclOneSidedService::IsUsedRdma(RankId remoteRankId, bool& useRdma)
{
    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));

    RankInfo_t localRankInfo = (rankTable_->rankList).at(localRankInfo_.userRank);
    RankInfo_t remoteRankInfo = (rankTable_->rankList).at(remoteRankId);
    if (deviceType == DevType::DEV_TYPE_910B) {
        // 外部使能RDMA，或者节点间通信
        if (GetExternalInputIntraRoceSwitch() != 0 || localRankInfo.serverId != remoteRankInfo.serverId) {
            useRdma = true;
            return HCCL_SUCCESS;
        }

        // 同一节点的 PCIe 连接判断
        s32 localDeviceId = localRankInfo_.devicePhyId;
        s32 remoteDeviceId = remoteRankInfo.deviceInfo.devicePhyId;
        LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
        CHK_RET(hrtGetPairDeviceLinkType(static_cast<u32>(localDeviceId), static_cast<u32>(remoteDeviceId), linkType));
        if (linkType != LinkTypeInServer::HCCS_TYPE) {
            HCCL_ERROR(
                "[HcclOneSidedService][IsUsedRdma]localDeviceId: %d, remoteDeviceId: %d, linkType %u is not supported",
                localDeviceId, remoteDeviceId, linkType);
            return HCCL_E_NOT_SUPPORT;
        }

        // 节点内通信，默认不使用 RDMA
        useRdma = false;
        return HCCL_SUCCESS;
    } else if (deviceType == DevType::DEV_TYPE_910_93) {
        if (GetExternalInputIntraRoceSwitch() != 0 || localRankInfo.superPodId != remoteRankInfo.superPodId) {
            useRdma = true;
            return HCCL_SUCCESS;
        }

        useRdma = false;
        return HCCL_SUCCESS;
    }

    // 其他情况默认使用 RDMA
    useRdma = true;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::GetIsUsedRdma(RankId remoteRankId, bool& useRdma)
{
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    } else {
        useRdma = isUsedRdmaMap_[remoteRankId];
    }

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ReMapMem(HcclMem* memInfoArray, u64 arraySize)
{
    HcclResult ret = HCCL_SUCCESS;
    if (netDevRdmaCtx_) { // 非roce场景不进行remap，返回success
        ret = HcclMemRemap(netDevRdmaCtx_, memInfoArray, arraySize);
    } else {
        HCCL_RUN_INFO("[HcclOneSidedService][ReMapMem] doesn't support remap ipc mem, just return success");
    }
    return ret;
}

HcclResult
HcclOneSidedService::RegMem(void* addr, u64 size, HcclMemType type, RankId remoteRankId, HcclMemDesc& localMemDesc)
{
    bool useRdma = true;
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    }
    useRdma = isUsedRdmaMap_[remoteRankId];

    HcclMem localMem{type, addr, size};
    HcclBuf buf;
    HcclResult ret = HcclMemReg(useRdma ? netDevRdmaCtx_ : netDevIpcCtx_, &localMem, &buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) { // HCCL_E_AGAIN:调用HcclMemReg前，内存已注册过
        return ret;
    }
    bool firstReg = (ret == HCCL_SUCCESS);

    char* desc = nullptr;
    uint64_t descLen = 0;
    ret = HcclMemExport(&buf, &desc, &descLen);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclOneSidedService][RegMem] get mem desc failed, ret[%d]", ret);
        throw logic_error("[HcclOneSidedService][RegMem] get mem desc failed");
    }

    HcclMemDescData* ptr = static_cast<HcclMemDescData*>(static_cast<void*>(localMemDesc.desc));
    ptr->localRankId = localRankInfo_.userRank;
    ptr->remoteRankId = remoteRankId;
    CHK_SAFETY_FUNC_RET(memset_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, 0, HCCL_MEM_DESC_STR_LEN));
    if (memcpy_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, desc, descLen + 1) != EOK) {
        HCCL_ERROR("[HcclOneSidedService][RegMem] memcpy_s memDesc failed");
        return HCCL_E_INTERNAL;
    }

    if (firstReg) {
        registedMemCnt_++;
        std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
        if (useRdma) {
            desc2HcclBufMapRoce_.emplace(descStr, buf);
        } else {
            desc2HcclBufMapIpc_.emplace(descStr, buf);
        }
    }
    HCCL_DEBUG(
        "[HcclOneSidedService][RegMem] localRankId[%u] remoteRankId[%u] size[%lu] useRdma[%d] "
        "desc2HcclBufMap[%u] registeredMemCnt[%u]",
        ptr->localRankId, ptr->remoteRankId, size, useRdma,
        useRdma ? desc2HcclBufMapRoce_.size() : desc2HcclBufMapIpc_.size(), registedMemCnt_);
    return HCCL_SUCCESS;
}

HcclBuf* HcclOneSidedService::GetHcclBufByDesc(std::string& descStr, bool useRdma)
{
    HcclBuf* buf = nullptr;
    if (useRdma) {
        auto iter = desc2HcclBufMapRoce_.find(descStr);
        if (iter == desc2HcclBufMapRoce_.end()) {
            HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]Roce memory is not registered, please register first.");
            return nullptr;
        }
        buf = &(iter->second);
    } else {
        auto iter = desc2HcclBufMapIpc_.find(descStr);
        if (iter == desc2HcclBufMapIpc_.end()) {
            HCCL_ERROR("[HcclOneSidedService][GetHcclBufByDesc]Ipc memory is not registered, please register first.");
            return nullptr;
        }
        buf = &(iter->second);
    }
    return buf;
}

HcclResult HcclOneSidedService::DeregMem(const HcclMemDesc& localMemDesc)
{
    const HcclMemDescData* ptr = static_cast<const HcclMemDescData*>(static_cast<const void*>(localMemDesc.desc));
    u32 remoteRankId = ptr->remoteRankId;
    if (registedMemCnt_ == 0) {
        HCCL_ERROR("[HcclOneSidedService][DeregMem]The number of registered memory is 0, please register first.");
        return HCCL_E_NOT_FOUND;
    }

    bool useRdma = true;
    if (isUsedRdmaMap_.find(remoteRankId) == isUsedRdmaMap_.end()) {
        CHK_RET(IsUsedRdma(remoteRankId, useRdma));
        isUsedRdmaMap_[remoteRankId] = useRdma;
    }
    useRdma = isUsedRdmaMap_[remoteRankId];

    std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
    HcclBuf* buf = GetHcclBufByDesc(descStr, useRdma);
    CHK_PRT_RET(
        buf == nullptr, HCCL_ERROR("[HcclOneSidedService][DeregMem] GetHcclBufByDesc failed."), HCCL_E_INTERNAL);
    HcclResult ret = HcclMemDereg(buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) { // 调用DeregMem后，去注册的内存还需继续使用（即有多次注册
        return ret;
    }

    HCCL_DEBUG(
        "[HcclOneSidedService][DeregMem] localRankId[%u] remoteRankId[%u] size[%lu] useRdma[%d] "
        "desc2HcclBufMap[%u] registeredMemCnt[%u]",
        ptr->localRankId, ptr->remoteRankId, buf->len, useRdma,
        useRdma ? desc2HcclBufMapRoce_.size() : desc2HcclBufMapIpc_.size(), registedMemCnt_);

    if (ret == HCCL_SUCCESS) {
        registedMemCnt_--;
        if (useRdma) {
            desc2HcclBufMapRoce_.erase(descStr);
        } else {
            desc2HcclBufMapIpc_.erase(descStr);
        }
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::SetupRemoteRankInfo(RankId remoteRankId, HcclRankLinkInfo& remoteRankInfo)
{
    // 检查 rankId 是否有效
    CHK_PRT_RET(
        rankTable_->rankList.size() <= remoteRankId,
        HCCL_ERROR(
            "[HcclOneSidedService][SetupRemoteRankInfo] the size of rankList is less than remoteRankId[%u].",
            remoteRankId),
        HCCL_E_NOT_FOUND);

    RankInfo_t tempRankInfo = rankTable_->rankList.at(remoteRankId);
    remoteRankInfo.userRank = tempRankInfo.rankId;
    remoteRankInfo.devicePhyId = tempRankInfo.deviceInfo.devicePhyId;

    // 检查 deviceIp 是否为空
    CHK_PRT_RET(
        tempRankInfo.deviceInfo.deviceIp.empty(),
        HCCL_ERROR("[HcclOneSidedService][SetupRemoteRankInfo] deviceIp is empty. RemoteRankId is [%u]", remoteRankId),
        HCCL_E_NOT_FOUND);
    remoteRankInfo.ip = tempRankInfo.deviceInfo.deviceIp[0];

    if (isUsedRdmaMap_.find(remoteRankId) != isUsedRdmaMap_.end() && !isUsedRdmaMap_[remoteRankId]) {
        bool useSuperPodMode = false;
        CHK_RET(IsSuperPodMode(useSuperPodMode));

        HcclIpAddress localVnicIp = HcclIpAddress(localRankInfo_.devicePhyId);
        HcclIpAddress remoteVnicIp = HcclIpAddress(remoteRankInfo.devicePhyId);
        RankInfo_t tRankInfo = rankTable_->rankList.at(localRankInfo_.userRank);

        if (useSuperPodMode) {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_SDID, tRankInfo.superDeviceId, localVnicIp));
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_SDID, tempRankInfo.superDeviceId,
                remoteVnicIp));
        } else {
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, localRankInfo_.devicePhyId,
                localVnicIp));
            CHK_RET(hrtRaGetSingleSocketVnicIpInfo(
                localRankInfo_.devicePhyId, DeviceIdType::DEVICE_ID_TYPE_PHY_ID, remoteRankInfo.devicePhyId,
                remoteVnicIp));
        }

        localRankVnicInfo_.ip = localVnicIp;
        remoteRankInfo.ip = remoteVnicIp;
    }
    remoteRankInfo.port = tempRankInfo.deviceInfo.port == 0 || tempRankInfo.deviceInfo.port == HCCL_INVALID_PORT ?
                              HETEROG_CCL_PORT :
                              tempRankInfo.deviceInfo.port;
    remoteRankInfo.socketsPerLink = 1;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::CreateLaunchStream()
{
    g_launchStream = nullptr;
    constexpr u32 streamMode = 1; // 使能遇错即停
    EXCEPTION_CATCH(g_launchStream = std::make_unique<Stream>(StreamType::STREAM_TYPE_ONLINE), return HCCL_E_PTR);
    CHK_PTR_NULL(g_launchStream);
    CHK_PTR_NULL(g_launchStream->ptr());
    HCCL_INFO("[HcclOneSidedService][CreateLaunchStream] launchStream[%u]", g_launchStream->id());
    CHK_RET(hrtStreamSetMode(g_launchStream->ptr(), streamMode));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::InitAicpuUnfoldMode()
{
    if (isAicpuModeInited_) {
        return HCCL_SUCCESS;
    }

    DevType deviceType;
    CHK_RET(hrtGetDeviceType(deviceType));
    aicpuUnfoldMode_ = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B)
                       && commConfig_.GetConfigAicpuUnfold(); // keep env flag for perf test
    HCCL_INFO(
        "[InitAicpuUnfoldMode] deviceType[%u] rdma[%u] aicpu[%u]", deviceType, (netDevRdmaCtx_ != nullptr),
        aicpuUnfoldMode_);
    if (aicpuUnfoldMode_) {
        CHK_PRT(LoadAICPUKernel());
        CHK_RET(AicpuResourceInit()); // 初始化service粒度资源
        CHK_RET(AicpuInitKernelLaunch());
    }

    isAicpuModeInited_ = true;

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::LoadAICPUKernel(void)
{
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
    return HCCL_SUCCESS;
}

void HcclOneSidedService::UnloadAICPUKernel(void)
{
    if (binHandle_ != nullptr) {
        aclError aclRet = aclrtBinaryUnLoad(binHandle_);
        if (aclRet != ACL_SUCCESS) {
            HCCL_ERROR(
                "[UnloadAICPUKernel]errNo[0x%016llx] unload binary from binHandle[%p] error.", aclRet, binHandle_);
        }
        binHandle_ = nullptr;
    }
    return;
}

HcclResult HcclOneSidedService::CreateConnection(
    RankId remoteRankId, const HcclRankLinkInfo& remoteRankInfo, std::shared_ptr<HcclOneSidedConn>& tempConn)
{
    CHK_RET(InitAicpuUnfoldMode());
    HcclNetDevCtx* ctx = isUsedRdmaMap_.at(remoteRankId) ? &netDevRdmaCtx_ : &netDevIpcCtx_;
    HcclRankLinkInfo* rankInfo = isUsedRdmaMap_.at(remoteRankId) ? &localRankInfo_ : &localRankVnicInfo_;
    u32 sdid = isUsedRdmaMap_.at(remoteRankId) ? 0 : rankTable_->rankList.at(localRankInfo_.userRank).superDeviceId;
    u32 serverId = isUsedRdmaMap_.at(remoteRankId) ? 0 : rankTable_->rankList.at(localRankInfo_.userRank).serverIdx;
    // 新增isNeedEnableP2P，用于判断remoteRank和本Rank是否在同一server上，在则需要enableP2P，反之则不需要
    bool isNeedEnableP2P = enableP2PRankIds_.find(remoteRankId) != enableP2PRankIds_.end();
    EXCEPTION_CATCH(
        tempConn = std::make_shared<HcclOneSidedConn>(
            *ctx, *rankInfo, remoteRankInfo, socketManager_, notifyPool_, dispatcher_, isUsedRdmaMap_[remoteRankId],
            sdid, serverId, trafficClass_, serviceLevel_, aicpuUnfoldMode_, isStandardCard_, isNeedEnableP2P),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(tempConn);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::Grant(const HcclMemDesc& localMemDesc, const ProcessInfo& remoteProcess)
{
    const HcclMemDescData* ptr = static_cast<const HcclMemDescData*>(static_cast<const void*>(localMemDesc.desc));
    std::string descStr(ptr->memDesc, HCCL_MEM_DESC_STR_LEN);
    HCCL_DEBUG("[HcclOneSidedService][Grant] desc[%s] length[%u]", descStr.c_str(), descStr.length());
    HcclBuf* buf = GetHcclBufByDesc(descStr, false);
    if (buf == nullptr) {
        return HCCL_E_INTERNAL;
    }

    HcclMemGrantInfo grantInfo = {remoteProcess.sdid, static_cast<int32_t>(remoteProcess.pid)};
    HcclResult ret = HcclMemGrant(buf, &grantInfo);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclOneSidedService][Grant] Grant error"), ret);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ExchangeMemDesc(
    RankId remoteRankId, const HcclMemDescs& localMemDescs, HcclMemDescs& remoteMemDescs, u32& actualNumOfRemote,
    const std::string& commIdentifier, s32 timeoutSec)
{
    std::shared_ptr<HcclOneSidedConn> tempConn;
    std::unique_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HcclRankLinkInfo remoteRankInfo;
        CHK_RET(SetupRemoteRankInfo(remoteRankId, remoteRankInfo));
        CHK_RET(CreateConnection(remoteRankId, remoteRankInfo, tempConn));
        timeoutSec = timeoutSec == 0 ? GetExternalInputHcclLinkTimeOut() : timeoutSec;
        CHK_RET(tempConn->Connect(commIdentifier, timeoutSec));
        oneSidedConns_.emplace(remoteRankId, tempConn);
    } else {
        tempConn = it->second;
    }
    std::unique_lock<std::mutex> lock(descMtx_);
    for (u32 i = 0; i < localMemDescs.arrayLength; ++i) {
        localMemDescs_[remoteRankId].push_back(localMemDescs.array[i]);
    }
    lock.unlock();

    return tempConn->ExchangeMemDesc(localMemDescs, remoteMemDescs, actualNumOfRemote);
}

void HcclOneSidedService::EnableMemAccess(const HcclMemDesc& remoteMemDesc, HcclMem& remoteMem)
{
    HcclResult ret = HCCL_SUCCESS;
    const TransportMem::RmaMemDesc* ptr = reinterpret_cast<const TransportMem::RmaMemDesc*>(remoteMemDesc.desc);
    u32 remoteRank = ptr->localRankId;
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRank);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR(
            "[HcclOneSidedService][EnableMemAccess]connection not found, remoteRank[%u], "
            "please exchange mem desc to create connection first.",
            remoteRank);
        throw logic_error("[HcclOneSidedService][EnableMemAccess]connection not found.");
    }
    std::unique_lock<std::mutex> lock(descMtx_);
    auto descIt = localMemDescs_.find(remoteRank);
    // HCCS下进行权限授予
    if (!isUsedRdmaMap_[remoteRank] && descIt != localMemDescs_.end()) {
        s32 pid;
        SalGetBareTgid(&pid);
        RankId localRankId = localRankInfo_.userRank;
        u32 sid = rankTable_->rankList.at(localRankId).superDeviceId;
        u32 serverId = rankTable_->rankList.at(localRankId).serverIdx;

        // 收发进程信息
        ProcessInfo localProcess = {pid, sid, serverId};
        ProcessInfo remoteProcess = {};

        ret = it->second->ExchangeIpcProcessInfo(localProcess, remoteProcess);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HcclOneSidedService][EnableMemAccess] Exchange ipc processInfo failed, ret[%d], "
                "remoteRank[%u].",
                ret, remoteRank);
            throw logic_error("[HcclOneSidedService][EnableMemAccess] Exchange ipc processInfo failed.");
        }
        remoteProcess.sdid = localProcess.serverId == remoteProcess.serverId ? INVALID_INT : remoteProcess.sdid;

        for (u32 i = 0; i < descIt->second.size(); ++i) {
            ret = Grant(descIt->second.at(i), remoteProcess);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR(
                    "[HcclOneSidedService][EnableMemAccess] Grant remote process failed, ret[%d], "
                    "remoteRank[%u].",
                    ret, remoteRank);
                throw logic_error("[HcclOneSidedService][EnableMemAccess] Grant remote process failed.");
            }
        }
        localMemDescs_.erase(descIt);
    }
    lock.unlock();
    it->second->EnableMemAccess(remoteMemDesc, remoteMem);
}

void HcclOneSidedService::DisableMemAccess(const HcclMemDesc& remoteMemDesc)
{
    const TransportMem::RmaMemDesc* ptr = reinterpret_cast<const TransportMem::RmaMemDesc*>(remoteMemDesc.desc);
    u32 remoteRank = ptr->localRankId;
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    if (oneSidedConns_.find(remoteRank) == oneSidedConns_.end()) {
        HCCL_ERROR(
            "[HcclOneSidedService][DisableMemAccess]connection not found by remoteRankId[%u], "
            "please exchange mem desc to create connection first.",
            remoteRank);
        throw logic_error("[HcclOneSidedService][DisableMemAccess]connection not found.");
    }
    oneSidedConns_.at(remoteRank)->DisableMemAccess(remoteMemDesc);
}

void HcclOneSidedService::BatchPut(
    RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t& stream)
{
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchPut] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    if (aicpuUnfoldMode_) {
        EXCEPTION_THROW_IF_ERR(
            OrchestrateAicpu(remoteRankId, HcclCMDType::HCCL_CMD_BATCH_PUT, it->second, desc, descNum, stream),
            "[BatchPut] AICPU launch failed");
    } else {
        it->second->BatchWrite(desc, descNum, stream);
    }
}

void HcclOneSidedService::BatchGet(
    RankId remoteRankId, const HcclOneSideOpDesc* desc, u32 descNum, const rtStream_t& stream)
{
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    auto it = oneSidedConns_.find(remoteRankId);
    if (it == oneSidedConns_.end()) {
        HCCL_ERROR("[HcclMemCommunication][BatchGet] Can't find oneSidedConn by remoteRank %u", remoteRankId);
        throw out_of_range("Can't find oneSidedConn by remoteRank.");
    }
    if (aicpuUnfoldMode_) {
        EXCEPTION_THROW_IF_ERR(
            OrchestrateAicpu(remoteRankId, HcclCMDType::HCCL_CMD_BATCH_GET, it->second, desc, descNum, stream),
            "[BatchGet] AICPU launch failed");
    } else {
        it->second->BatchRead(desc, descNum, stream);
    }
}

// 绑定一块全局内存
HcclResult HcclOneSidedService::BindMem(void* memRecordHandle, const std::string& commIdentifier)
{
    auto memRecordPtr = static_cast<GlobalMemRecord*>(memRecordHandle);
    CHK_RET(memRecordPtr->BindToComm(commIdentifier));

    // 是否重复绑定在前面BindToComm已经检查过了
    std::unique_lock lock(boundMemPtrSetMtx_);
    auto emplaceResult = boundMemPtrSet_.emplace(memRecordPtr);
    CHK_PRT_RET(
        emplaceResult.second == false,
        HCCL_ERROR(
            "[HcclOneSidedService][BindMem] Emplace mem record ptr failed, memRecordPtr[%p], comm[%s].", memRecordPtr,
            commIdentifier.c_str()),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[HcclOneSidedService][BindMem] Bind mem successfully, memHandle[%p], comm[%s].", memRecordHandle,
        commIdentifier.c_str());
    return HCCL_SUCCESS;
}

// 解绑一块全局内存
HcclResult HcclOneSidedService::UnbindMem(void* memRecordHandle, const std::string& commIdentifier)
{
    auto memRecordPtr = static_cast<GlobalMemRecord*>(memRecordHandle);
    CHK_RET(memRecordPtr->UnbindFromComm(commIdentifier));

    std::unique_lock lock(boundMemPtrSetMtx_);
    const auto eraseCount = boundMemPtrSet_.erase(memRecordPtr);
    CHK_PRT_RET(
        eraseCount == 0,
        HCCL_ERROR(
            "[HcclOneSidedService][UnbindMem] Erase mem record ptr failed, memRecordPtr[%p], comm[%s].",
            memRecordHandle, commIdentifier.c_str()),
        HCCL_E_INTERNAL);

    HCCL_INFO(
        "[HcclOneSidedService][UnbindMem] Unbind mem successfully, memHandle[%p], comm[%s].", memRecordHandle,
        commIdentifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::DeInit()
{
    if (aicpuUnfoldMode_) {
        std::unique_lock<std::mutex> guard{g_launchMutex};
        CHK_RET(CreateLaunchStream());
        CHK_RET(OrchestrateAicpu(0, HcclCMDType::HCCL_CMD_BATCH_GET, nullptr, nullptr, 0, g_launchStream->ptr()));
        CHK_RET(hcclStreamSynchronize(
            g_launchStream->ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(identifier_)));
        HCCL_INFO("[HcclOneSidedService][DeInit] destroy launchStream[%u]", g_launchStream->id());
        g_launchStream = nullptr;
    }

    // 检查是否还绑定着全局内存
    {
        std::shared_lock lock(boundMemPtrSetMtx_);
        if (!boundMemPtrSet_.empty()) {
            HCCL_ERROR("[HcclOneSidedService][DeInit] There are memories still bound to this comm; please unbind them "
                       "before destroying the comm.");
            HCCL_ERROR("[HcclOneSidedService][DeInit] List of bound memories:");
            for (auto handle : boundMemPtrSet_) {
                auto memRecordPtr = static_cast<GlobalMemRecord*>(handle);
                const auto info = memRecordPtr->PrintInfo();
                HCCL_ERROR("[HcclOneSidedService][DeInit][Bound mem] ptr:%p, %s", handle, info.c_str());
            }
            return HCCL_E_PARA;
        }
    }

    if (prepared_) {
        // 去使能内存
        CHK_RET(DisableMemAccess());
        prepared_ = false;
    }
    UnloadAICPUKernel();
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::RunFuncWithTimeout(
    std::function<HcclResult()> func, const std::string& commIdentifier, s32 timeoutSec, std::string functionName)
{
    std::future<HcclResult> futureResult;
    futureResult = std::async(std::launch::async, func);

    CHK_PRT_RET(
        !futureResult.valid(),
        HCCL_ERROR("[HcclOneSidedService][%s] futureResult is not assigned.", functionName.c_str()), HCCL_E_INTERNAL);

    // 超时检查，若timeout设置为-1则不检查，上层已经保证timeout不会为0
    if (timeoutSec != -1 && futureResult.wait_for(std::chrono::seconds(timeoutSec)) == std::future_status::timeout) {
        // 发生超时，设置stop flag让socket线程停止，避免进程长时间无法退出
        CHK_RET(socketManager_->SetStopFlag(true));
        HCCL_ERROR(
            "[HcclOneSidedService][%s]timeout. commIdentifier[%s], timeout[%ds]", functionName.c_str(),
            commIdentifier.c_str(), timeoutSec);
        futureResult.wait();
        CHK_RET(socketManager_->SetStopFlag(false));
        return HCCL_E_TIMEOUT;
    }

    HcclResult ret = futureResult.get();
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[HcclOneSidedService][%s] Prepare failed. commIdentifier[%s]", functionName.c_str(),
            commIdentifier.c_str()),
        ret);

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::PrepareFullMesh(const std::string& commIdentifier, s32 timeoutSec)
{
    HcclUs startut = TIME_NOW();
    // 创建连接
    CHK_RET(CreateLinkFullmesh(commIdentifier, timeoutSec));
    // 注册内存
    CHK_RET(RegisterBoundMems());
    // 交换内存描述符
    CHK_RET(RunFuncWithTimeout(
        [this]() -> HcclResult {
            return this->ExchangeMemDescFullMesh();
        },
        commIdentifier, timeoutSec, "ExchangeMemDescFullMesh"));
    // 使能访问
    CHK_RET(RunFuncWithTimeout(
        [this]() -> HcclResult {
            return this->EnableMemAccessByThread();
        },
        commIdentifier, timeoutSec, "EnableMemAccessByThread"));

    HCCL_INFO(
        "[HcclOneSidedService][PrepareFullMesh] Prepare finished. comm[%s], take time [%lld us].",
        commIdentifier.c_str(), DURATION_US(TIME_NOW() - startut));
    return HCCL_SUCCESS;
}

HcclResult
HcclOneSidedService::Prepare(const std::string& commIdentifier, const HcclPrepareConfig* prepareConfig, s32 timeoutSec)
{
    // 如果已经prepare过，直接返回Success
    CHK_PRT_RET(
        prepared_, HCCL_WARNING("[HcclOneSidedService][Prepare] This comm[%s] has prepared.", commIdentifier.c_str()),
        HCCL_SUCCESS);

    CHK_RET(hrtGetDevice(&deviceLogicId_));

    if (needRegIpcMem_) {
        SalGetBareTgid(&localProcess_.pid);
        RankId localRankId = localRankInfo_.userRank;
        localProcess_.sdid = rankTable_->rankList.at(localRankId).superDeviceId;
        localProcess_.serverId = rankTable_->rankList.at(localRankId).serverIdx;
    }

    HcclTopoType configTopoType = prepareConfig->topoType;
    std::future<HcclResult> futureResult;
    timeoutSec = timeoutSec == 0 ? GetExternalInputHcclLinkTimeOut() : timeoutSec;
    if (configTopoType == HcclTopoType::HCCL_TOPO_FULLMESH) {
        HCCL_INFO("[HcclOneSidedService][Prepare] topoType is fullmesh.");

        auto ret = PrepareFullMesh(commIdentifier, timeoutSec);
        if (ret != HCCL_SUCCESS) {
            u32 rankSize = (rankTable_->rankList).size();
            std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
            for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
                auto it = oneSidedConns_.find(remoteRankId);
                if (it == oneSidedConns_.end()) {
                    // remoteRankId超出oneSidedConns_的范围，直接退出
                    HCCL_ERROR(
                        "[HcclOneSidedService][Prepare] remoteRankId[%u] "
                        "is not found in map oneSidedConns.",
                        remoteRankId);
                    break;
                }
                if (remoteRankId == localRankInfo_.userRank || oneSidedConns_.at(remoteRankId) == nullptr) {
                    HCCL_INFO("[HcclOneSidedService][CleanSocketResource] remoteRank[%u] skip.", remoteRankId);
                    continue;
                }
                HCCL_INFO("[HcclOneSidedService][CleanSocketResource] remote[%u]", remoteRankId);
                oneSidedConns_.at(remoteRankId)->CleanSocketResource(commIdentifier);
            }
            HCCL_ERROR("[HcclOneSidedService][Prepare] Prepare failed. commIdentifier[%s]", commIdentifier.c_str());
            return ret;
        }
    }

    prepared_ = true;
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::InitIsUsedRdmaMap(bool& needInitNic, bool& needInitVnic)
{
    u32 rankSize = (rankTable_->rankList).size();
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        bool isUseRdma;
        CHK_RET(IsUsedRdma(remoteRankId, isUseRdma));
        isUsedRdmaMap_[remoteRankId] = isUseRdma;

        if (isUseRdma) {
            needRegRoceMem_ = true;
        } else {
            needRegIpcMem_ = true;
        }
    }
    needInitNic = needRegRoceMem_;
    needInitVnic = needRegIpcMem_;

    HCCL_INFO(
        "[HcclOneSidedService][InitIsUsedRdmaMap] needInitNic is [%d], needInitVnic is [%d]", needInitNic,
        needInitVnic);
    return HCCL_SUCCESS;
}

void HcclOneSidedService::ConnectByThread(
    std::shared_ptr<HcclOneSidedConn>& conn, const std::string& commIdentifier, s32 timeoutSec, HcclResult& retOut)
{
    if (deviceLogicId_ != HOST_DEVICE_ID) {
        hrtSetDevice(deviceLogicId_);
    }
    HcclResult ret = conn->ConnectWithRemote(commIdentifier, localProcess_, timeoutSec);
    retOut = ret;
    if (ret != HCCL_SUCCESS) {
        hasErrorFlag_ = true;
        if (ret == HCCL_E_TIMEOUT) {
            hasTimeoutErrorFlag_ = true;
        }
        HCCL_ERROR("[ConnectByThread] Connect failed. userRank[%u], ret[%d].", localRankInfo_.userRank, ret);
    }
    hrtResetDevice(deviceLogicId_);
}

HcclResult HcclOneSidedService::CreateLinkFullmesh(const std::string& commIdentifier, s32 timeoutSec)
{
    u32 rankSize = (rankTable_->rankList).size();

    std::unique_lock oneSidedConnslock(oneSidedConnsMutex_);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        HcclRankLinkInfo remoteRankInfo;
        CHK_RET(SetupRemoteRankInfo(remoteRankId, remoteRankInfo));
        CHK_RET(CreateConnection(remoteRankId, remoteRankInfo, oneSidedConns_[remoteRankId]));
    }

    std::vector<std::unique_ptr<std::thread>> linkThreads;
    std::vector<HcclResult> linkResult;
    linkThreads.resize(rankSize);
    linkResult.resize(rankSize, HCCL_SUCCESS);
    hasErrorFlag_ = false;
    ThreadsGuard threadsGuard(linkThreads);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        linkThreads[remoteRankId].reset(new (std::nothrow) std::thread(
            &HcclOneSidedService::ConnectByThread, this, std::ref(oneSidedConns_.at(remoteRankId)), commIdentifier,
            timeoutSec, std::ref(linkResult[remoteRankId])));
        CHK_SMART_PTR_NULL(linkThreads[remoteRankId]);
    }
    oneSidedConnslock.unlock();

    for (u32 remoteRankId = 0; remoteRankId < linkThreads.size(); remoteRankId++) {
        if (linkThreads[remoteRankId] == nullptr || !linkThreads[remoteRankId]->joinable()) {
            continue;
        }
        linkThreads[remoteRankId]->join(); // 等待线程执行完毕
    }
    linkThreads.clear();

    for (u32 remoteRankId = 0; remoteRankId < linkResult.size(); remoteRankId++) {
        CHK_PRT_RET(
            linkResult[remoteRankId] != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HcclOneSidedService][CreateLinkFullmesh] Create links failed. commIdentifier[%s].",
                commIdentifier.c_str()),
            linkResult[remoteRankId]);
    }

    CHK_PRT_RET(
        hasErrorFlag_ == true,
        HCCL_ERROR(
            "[HcclOneSidedService][CreateLinkFullmesh] Create links failed. commIdentifier[%s].",
            commIdentifier.c_str()),
        hasTimeoutErrorFlag_ ? HCCL_E_TIMEOUT : HCCL_E_INTERNAL);

    HCCL_INFO(
        "[HcclOneSidedService][CreateLinkFullmesh] Create links success. commIdentifier[%s].", commIdentifier.c_str());
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::RegBoundMem(
    HcclNetDevCtx netDevCtx, const HcclMem& localMem, HcclMemDesc& localMemDesc, HcclBuf& buf)
{
    std::unique_lock<std::mutex> lock(regMutex_);
    HcclResult ret = HcclMemReg(netDevCtx, &localMem, &buf);
    if ((ret != HCCL_SUCCESS) && (ret != HCCL_E_AGAIN)) { // HCCL_E_AGAIN:调用HcclMemReg前，内存已注册过
        return ret;
    }

    char* desc = nullptr;
    uint64_t descLen = 0;
    ret = HcclMemExport(&buf, &desc, &descLen);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HcclOneSidedService][RegBoundMem] get mem desc failed, ret[%d]", ret);
        throw logic_error("[HcclOneSidedService][RegBoundMem] get mem desc failed");
    }
    lock.unlock();

    HcclMemDescData* ptr = static_cast<HcclMemDescData*>(static_cast<void*>(localMemDesc.desc));
    ptr->localRankId = localRankInfo_.userRank;
    ptr->remoteRankId = INVALID_REMOTE_RANK_ID; // 进程粒度注册，不区分对端rank, 填为全F
    CHK_SAFETY_FUNC_RET(memset_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, 0, HCCL_MEM_DESC_STR_LEN));
    if (memcpy_s(ptr->memDesc, HCCL_MEM_DESC_STR_LEN, desc, descLen + 1) != EOK) {
        HCCL_ERROR("[HcclOneSidedService][RegBoundMem] memcpy_s memDesc failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[HcclOneSidedService][RegBoundMem] RegBoundMem success. addr[%p], size[%llu].", buf.addr, buf.len);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::RegisterBoundMems()
{
    std::shared_lock lock(boundMemPtrSetMtx_);
    localMemIpcDescs_.reserve(boundMemPtrSet_.size());
    localMemRoceDescs_.reserve(boundMemPtrSet_.size());
    localMemIpcDescs_.clear();
    localMemRoceDescs_.clear();
    for (auto& recordPtr : boundMemPtrSet_) {
        HcclMem mem{recordPtr->GetMemType(), const_cast<void*>(recordPtr->GetAddr()), recordPtr->GetSize()};
        if (needRegRoceMem_) {
            HcclBuf buf;
            HcclMemDesc localMemDesc;
            CHK_RET(RegBoundMem(netDevRdmaCtx_, mem, localMemDesc, buf));
            localMemRoceDescs_.push_back(localMemDesc);
            recordPtr->SaveRegBufInfo(netDevRdmaCtx_, buf);
        }
        if (needRegIpcMem_) {
            HcclBuf buf;
            HcclMemDesc localMemDesc;
            CHK_RET(RegBoundMem(netDevIpcCtx_, mem, localMemDesc, buf));
            if (recordPtr->GetMemType() == HCCL_MEM_TYPE_DEVICE) {
                localMemIpcDescs_.push_back(localMemDesc);
            }
            recordPtr->SaveRegBufInfo(netDevIpcCtx_, buf);
            CHK_RET(Grant(buf));
        }
    }
    HCCL_INFO("[HcclOneSidedService][RegisterBoundMems] Register bound mems success.");
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ExchangeMemDescFullMesh()
{
    u32 rankSize = (rankTable_->rankList).size();
    std::vector<std::unique_ptr<std::thread>> exchangeThreads;
    exchangeThreads.resize(rankSize);

    hasErrorFlag_ = false;
    ThreadsGuard threadsGuard(exchangeThreads);
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        exchangeThreads[remoteRankId].reset(new (std::nothrow) std::thread(
            &HcclOneSidedService::ExchangeMemDescByThread, this, std::ref(oneSidedConns_.at(remoteRankId)),
            isUsedRdmaMap_[remoteRankId]));
        CHK_SMART_PTR_NULL(exchangeThreads[remoteRankId]);
    }
    oneSidedConnslock.unlock();

    for (u32 remoteRankId = 0; remoteRankId < exchangeThreads.size(); remoteRankId++) {
        if (exchangeThreads[remoteRankId] == nullptr || !exchangeThreads[remoteRankId]->joinable()) {
            continue;
        }
        exchangeThreads[remoteRankId]->join(); // 等待线程执行完毕
    }
    CHK_PRT_RET(
        hasErrorFlag_ == true, HCCL_ERROR("[HcclOneSidedService][ExchangeMemDescFullMesh] Exchange mem desc failed."),
        HCCL_E_INTERNAL);

    HCCL_INFO("[HcclOneSidedService][ExchangeMemDescFullMesh] Exchange mem desc success.");
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ExchangeMemDescByThread(std::shared_ptr<HcclOneSidedConn>& conn, bool isUseRdma)
{
    if (deviceLogicId_ != HOST_DEVICE_ID) {
        hrtSetDevice(deviceLogicId_);
    }

    HcclMemDescs localMemDescs;
    if (isUseRdma) {
        localMemDescs.array = localMemRoceDescs_.data();
        localMemDescs.arrayLength = localMemRoceDescs_.size();
    } else {
        localMemDescs.array = localMemIpcDescs_.data();
        localMemDescs.arrayLength = localMemIpcDescs_.size();
    }

    HcclResult ret = conn->ExchangeMemDesc(localMemDescs);
    if (ret != HCCL_SUCCESS) {
        hasErrorFlag_ = true;
        HCCL_ERROR(
            "[ExchangeMemDescByThread] ExchangeMemDescByThread failed. userRank[%u], ret[%d].", localRankInfo_.userRank,
            ret);
    }
    CHK_RET(hrtResetDevice(deviceLogicId_));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::EnableMemAccessByThread()
{
    if (deviceLogicId_ != HOST_DEVICE_ID) {
        hrtSetDevice(deviceLogicId_);
    }
    CHK_RET(EnableMemAccess());
    CHK_RET(hrtResetDevice(deviceLogicId_));
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::EnableMemAccess()
{
    u32 rankSize = (rankTable_->rankList).size();
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        CHK_RET(oneSidedConns_.at(remoteRankId)->EnableMemAccess());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::DisableMemAccess()
{
    u32 rankSize = (rankTable_->rankList).size();
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank) {
            continue;
        }
        CHK_RET(oneSidedConns_.at(remoteRankId)->DisableMemAccess());
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::Grant(HcclBuf& buf)
{
    u32 rankSize = (rankTable_->rankList).size();
    std::shared_lock oneSidedConnslock(oneSidedConnsMutex_);
    for (u32 remoteRankId = 0; remoteRankId < rankSize; remoteRankId++) {
        if (remoteRankId == localRankInfo_.userRank || isUsedRdmaMap_[remoteRankId] == true) {
            continue;
        }
        ProcessInfo remoteProcess;
        CHK_RET(oneSidedConns_.at(remoteRankId)->GetRemoteProcessInfo(remoteProcess));
        HcclMemGrantInfo grantInfo = {remoteProcess.sdid, static_cast<int32_t>(remoteProcess.pid)};

        HcclResult ret = HcclMemGrant(&buf, &grantInfo);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[HcclOneSidedService][Grant] Grant error"), ret);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::OrchestrateAicpu(
    RankId remoteRankId, HcclCMDType cmdType, const std::shared_ptr<HcclOneSidedConn>& conn,
    const HcclOneSideOpDesc* desc, u32 descNum, rtStream_t stream)
{
    bool useRdma;
    CHK_RET(GetIsUsedRdma(remoteRankId, useRdma));

    HCCL_DEBUG(
        "[OrchestrateAicpu] aicpu unfold launch kernel: desc[%p] descNum[%u] cmdType[%u] tag[%s] localRank[%u] "
        "remoteRank[%u] useRdma[%d]",
        desc, descNum, cmdType, identifier_.c_str(), localRankInfo_.userRank, remoteRankId, useRdma);

    AicpuOneSideCommTiling tilingInfo;
    tilingInfo.cmdType = cmdType;
    tilingInfo.tag = identifier_;
    tilingInfo.stream = stream;
    tilingInfo.dumpDebug = GetExternalInputHcclDumpDebug();
    tilingInfo.useRdma = useRdma;
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    CHK_RET(hrtGetDeviceSatMode(&floatOverflowMode));
    tilingInfo.floatOverflowMode = floatOverflowMode;
    const u64 dynamicDataSize = CalcTilingDynamicDataSize(cmdType, descNum);
    CHK_RET(InitAicpuTilingDataBuf(tilingInfo, remoteRankId, conn, desc, descNum, dynamicDataSize));
    // 根据算子类型，获取 Aicpu Kernel 名称
    auto iter = HCOM_CMD_TYPE_STR_MAP.find(cmdType);
    CHK_PRT_RET(
        (iter == HCOM_CMD_TYPE_STR_MAP.end()),
        HCCL_ERROR("[%s] RunAicpuRpcSrvLaunchV2 kernel not found, cmdType=[%d]", __func__, static_cast<int>(cmdType)),
        HCCL_E_INTERNAL);
    std::string kernelName = std::string("RunAicpuRpcSrvLaunchV2") + "_" + iter->second;
    HcclResult ret = AicpuKernelLaunch(conn, kernelName, tilingInfo, sizeof(struct OpTilingData) + dynamicDataSize);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[OrchestrateAicpu] aicpu unfold launch kernel[%s] failed. ret[%u], "
            "desc[%p] descNum[%u] cmdType[%u] tag[%s]",
            kernelName.c_str(), ret, desc, descNum, cmdType, identifier_.c_str()),
        ret);
    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::AicpuResourceInit()
{
    const u64 beginTime = hrtMsprofSysCycleTime();

    execStream_ = Stream(StreamType::STREAM_TYPE_DEVICE);
    const u32 streamContextSize = sizeof(SqCqeContext);
    commResPara_.execStreamParam.streamInfo.streamIds = execStream_.id();
    commResPara_.execStreamParam.streamInfo.sqIds = execStream_.sqId();
    commResPara_.execStreamParam.streamInfo.cqIds = execStream_.cqId();
    commResPara_.execStreamParam.streamInfo.logicCqids = execStream_.logicCqId();
    CHK_RET(DeviceMem::alloc(execStreamContext_, streamContextSize));
    CHK_RET(hrtMemSet(execStreamContext_.ptr(), streamContextSize, streamContextSize));
    commResPara_.execStreamParam.sqCqContextAddr = reinterpret_cast<u64>(execStreamContext_.ptr());
    commResPara_.execStreamParam.sqCqContextSize = streamContextSize;

    const u32 postNotifyIdx = static_cast<u32>(AicpuLocalNotify::HOST_TO_AICPU_POST);
    HcclResult ret = CreateAicpuNotify(localAicpuNotify_[postNotifyIdx], commResPara_.aicpuOpNotify[postNotifyIdx]);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuResourceInit] create aicpu post notify failed, errNo[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);
    const u32 waitNotifyIdx = static_cast<u32>(AicpuLocalNotify::HOST_TO_AICPU_WAIT);
    ret = CreateAicpuNotify(localAicpuNotify_[waitNotifyIdx], commResPara_.aicpuOpNotify[waitNotifyIdx]);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[AicpuResourceInit] create aicpu wait notify failed, errNo[0x%016llx]", HCCL_ERROR_CODE(ret)), ret);

    CHK_RET(DeviceMem::alloc(commResParaDevice_, sizeof(HcclOneSideCommResParam)));

    const u64 endTime = hrtMsprofSysCycleTime();
    HCCL_DEBUG("[AicpuResourceInit] done, time cost[%llu us]", (endTime - beginTime));

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::ReportProfilingCommInfo(const Stream& kfcStream, const Stream& aicpuStream)
{
    ProfilingDeviceCommResInfo profCommInfo;
    profCommInfo.groupName = hrtMsprofGetHashId(identifier_.c_str(), identifier_.length());
    profCommInfo.rankSize = rankTable_->rankNum;
    profCommInfo.rankId = localRankInfo_.userRank;
    profCommInfo.usrRankId = localRankInfo_.userRank;
    profCommInfo.aicpuKfcStreamId = static_cast<uint32_t>(kfcStream.id());
    profCommInfo.reserve = 0;
    HCCL_INFO(
        "[ReportProfilingCommInfo] group[%s], groupHashId[%llu], streamId[%u]", identifier_.c_str(),
        profCommInfo.groupName, aicpuStream.id());
    profCommInfo.commStreamIds[0] = aicpuStream.id();
    profCommInfo.commStreamSize = 1; // 只有1条执行流
    return ProfilingManagerPub::CallMsprofReportMc2CommInfo(
        hrtMsprofSysCycleTime(), &profCommInfo, sizeof(profCommInfo));
}

HcclResult HcclOneSidedService::AicpuInitKernelLaunch()
{
    const u64 beginTime = hrtMsprofSysCycleTime();

    {
        std::unique_lock<std::mutex> guard{g_launchMutex};
        struct InitTask {
            u64 context; // A矩阵地址，通信在前时为sendbuffer
            bool isCustom;
        };
        InitTask initTask = {};
        initTask.context = 0ULL;
        initTask.isCustom = false;
        u16 timeOut = 0;
        char kernelName[64] = "RunAicpuKfcResInitV2";
        CHK_RET(CreateLaunchStream());
        CHK_RET(AicpuAclKernelLaunch(
            g_launchStream->ptr(), static_cast<void*>(&initTask), sizeof(initTask), binHandle_, kernelName, true,
            timeOut));
        CHK_RET(hcclStreamSynchronize(
            g_launchStream->ptr(), CommConfiger::GetInstance().GetCommConfigExecTimeOut(identifier_)));
        HCCL_RUN_INFO(
            "[AicpuInitKernelLaunch] launch in launchStream[%u], execStream[%u]", g_launchStream->id(),
            execStream_.id());
        g_launchStream = nullptr;
    }

    const u64 endTime = hrtMsprofSysCycleTime();
    s32 threadId = SalGetTid();
    std::string profName = "OneSideCommAicpuInit";
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::CreateAicpuNotify(std::shared_ptr<LocalNotify>& localNotify, HcclSignalInfo& notifyInfo)
{
    EXCEPTION_CATCH((localNotify = std::make_shared<LocalNotify>()), return HCCL_E_PTR);
    CHK_RET(localNotify->Init(NotifyLoadType::DEVICE_NOTIFY));
    CHK_RET(localNotify->SetIpc());
    CHK_RET(localNotify->GetNotifyData(notifyInfo));
    HCCL_INFO(
        "[HcclOneSidedService][CreateAicpuNotify]resId[%llu], addr[%llu], devId[%u], tsId[%u].", notifyInfo.resId,
        notifyInfo.addr, notifyInfo.devId, notifyInfo.tsId);
    return HCCL_SUCCESS;
}

u64 HcclOneSidedService::CalcTilingDynamicDataSize(HcclCMDType cmdType, u32 descNum)
{
    u64 dynamicDataSize = 0ULL;
    if (cmdType == HcclCMDType::HCCL_CMD_BATCH_GET || cmdType == HcclCMDType::HCCL_CMD_BATCH_PUT) {
        dynamicDataSize = sizeof(OpTilingOneSideCommDataDes) + sizeof(HcclOneSideOpDescParam) * (descNum + 1); // signal
    }
    return dynamicDataSize;
}

HcclResult HcclOneSidedService::InitAicpuTilingDataBuf(
    const AicpuOneSideCommTiling& tilingInfo, u32 remoteRankId, const std::shared_ptr<HcclOneSidedConn>& conn,
    const HcclOneSideOpDesc* desc, u32 descNum, u64 dynamicDataSize)
{
    const u64 tilingDataSize = sizeof(struct OpTilingData) + dynamicDataSize;
    if (tilingDataMem_.ptr() == nullptr) {
        tilingDataMem_ = HostMem::alloc(std::max(tilingDataSize, TILINGDATA_BUF_SIZE));
        CHK_PRT_RET(
            tilingDataMem_.ptr() == nullptr, HCCL_ERROR("[InitAicpuTilingDataBuf] Alloc tilingDataMem failed!"),
            HCCL_E_MEMORY);
    }

    if (tilingDataSize > tilingDataMem_.size()) {
        HCCL_INFO(
            "[InitAicpuTilingDataBuf] Increase tilingDataMem from size[%llu] to tilingDataSize[%llu]",
            tilingDataMem_.size(), tilingDataSize);
        tilingDataMem_.free();
        tilingDataMem_ = HostMem::alloc(tilingDataSize);
        CHK_PRT_RET(
            tilingDataMem_.ptr() == nullptr,
            HCCL_ERROR(
                "[InitAicpuTilingDataBuf] Increase tilingDataMem to "
                "tilingDataSize[%llu] failed!",
                tilingDataSize),
            HCCL_E_MEMORY);
    }

    const HcclCMDType cmdType = tilingInfo.cmdType;
    HCCL_DEBUG(
        "[InitAicpuTilingDataBuf] [%s] tilingDataSize[%llu] dynamicDataSize[%llu] desc[%p] descNum[%u] "
        "cmdType[%u] tilingDataMem[%p] tilingDataMem.size[%llu]",
        tilingInfo.tag.c_str(), tilingDataSize, dynamicDataSize, desc, descNum, cmdType, tilingDataMem_.ptr(),
        tilingDataMem_.size());

    // 填充固定内容
    HostMem tilingDataMem = tilingDataMem_.range(0, tilingDataSize);
    CHK_PTR_NULL(tilingDataMem.ptr());
    struct OpTilingData* tilingData = static_cast<struct OpTilingData*>(tilingDataMem.ptr());
    CHK_SAFETY_FUNC_RET(
        memcpy_s(tilingData->tag, sizeof(tilingData->tag), tilingInfo.tag.c_str(), tilingInfo.tag.length() + 1));
    tilingData->floatOverflowMode = tilingInfo.floatOverflowMode;
    tilingData->dumpDebug = tilingInfo.dumpDebug;
    tilingData->debugMode = 0;
    tilingData->srcRank = localRankInfo_.userRank;
    tilingData->dstRank = remoteRankId;
    tilingData->opType = static_cast<u8>(tilingInfo.cmdType);
    tilingData->length = dynamicDataSize;
    tilingData->customDataLength = 0;

    // 填充动态内容
    HostMem dynamicDataMem = tilingDataMem_.range(sizeof(struct OpTilingData), dynamicDataSize);
    CHK_PTR_NULL(dynamicDataMem.ptr());
    auto* vDataPtr = reinterpret_cast<struct OpTilingOneSideCommDataDes*>(dynamicDataMem.ptr());
    vDataPtr->commResParaAddr = reinterpret_cast<u64>(commResParaDevice_.ptr());
    vDataPtr->commResParaSize = commResParaDevice_.size();
    vDataPtr->rankSize = rankTable_->rankNum;
    vDataPtr->linkTimeout = 0;       // deprecated; 改成在AICPU侧使用qpInfo里的配置计算
    vDataPtr->descNum = descNum + 1; // signal
    vDataPtr->descDataLen = sizeof(HcclOneSideOpDescParam) * vDataPtr->descNum;
    vDataPtr->linkType
        = tilingInfo.useRdma ? static_cast<u8>(LinkType::LINK_ROCE) : static_cast<u8>(LinkType::LINK_HCCS);
    if (conn != nullptr && desc != nullptr) {
        vDataPtr->finalize = false;
        auto* descParam = reinterpret_cast<HcclOneSideOpDescParam*>(
            reinterpret_cast<u8*>(dynamicDataMem.ptr()) + sizeof(OpTilingOneSideCommDataDes));
        CHK_RET(conn->GetTransInfo(
            descParam, desc, vDataPtr->descNum, vDataPtr->transportDataAddr, vDataPtr->transportDataSize));
        CHK_RET(hrtMemSyncCopy(
            commResParaDevice_.ptr(), commResParaDevice_.size(), static_cast<void*>(&commResPara_),
            sizeof(commResPara_), HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE));
    } else {
        vDataPtr->finalize = true;
    }

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::AicpuKernelLaunch(
    const std::shared_ptr<HcclOneSidedConn>& conn, const std::string& kernelName,
    const AicpuOneSideCommTiling& tilingInfo, u64 tilingDataSize)
{
    const u64 beginTime = hrtMsprofSysCycleTime();
    std::string profName = GetCMDTypeEnumStr(tilingInfo.cmdType);
    if (profName == "Invalid HcclCMDType" || profName == "invalid") {
        profName = "HcclOneSideOpAicpuKernel";
    } else {
        profName += "AicpuKernel";
    }

    s32 streamId = 0;
    Stream mainStream = Stream(tilingInfo.stream);
    CHK_RET(hrtGetStreamId(mainStream.ptr(), streamId));
    HCCL_DEBUG("[%s] profName[%s] streamId[%d]", __func__, profName.c_str(), streamId);

    Stream launchStream = Stream(tilingInfo.stream); // 在用户流展开
    if (!isContextLaunched_) {
        CHK_RET(ReportProfilingCommInfo(launchStream, execStream_));
        isContextLaunched_ = true;
    }

    HostMem tilingDataMem = tilingDataMem_.range(0, tilingDataSize);
    CHK_RET(AicpuUnfoldKernelLaunchV2(kernelName, tilingDataMem.ptr(), tilingDataSize, launchStream.ptr()));

    // 省略下发流，在用户流展开，已经可以和用户流任务保序，不再需要前置Post/Wait，否则会导致Kernel任务不能边展开边执行
    HCCL_DEBUG(
        "[AicpuKernelLaunch] launch in user[%u] stream[%u], launchStream[%u], execStream[%u]",
        (mainStream.id() == launchStream.id()), mainStream.id(), launchStream.id(), execStream_.id());

    const u64 endTime = hrtMsprofSysCycleTime();
    const s32 threadId = SalGetTid();
    CHK_RET(ProfilingManagerPub::CallMsprofReportNodeInfo(beginTime, endTime, profName, threadId));

    if (conn != nullptr) {
        if (tilingInfo.useRdma) {
            CHK_RET(conn->WaitOpFence(tilingInfo.stream));
        } else {
            CHK_RET(LocalNotify::Wait(mainStream, dispatcher_, localAicpuNotify_[1], INVALID_VALUE_STAGE));
        }
    }

    HCCL_INFO(
        "[HcclOneSidedService][AicpuKernelLaunch] exec succ, conn[%p], streamId[%u]. time[%u us]", conn.get(),
        mainStream.id(), (endTime - beginTime));

    return HCCL_SUCCESS;
}

HcclResult HcclOneSidedService::AicpuUnfoldKernelLaunchV2(
    const std::string& kernelName, void* tilingDataPtr, u64 tilingDataSize, const rtStream_t stream)
{
    u64 commContext = 0ULL;
    u16 timeOut = NOTIFY_DEFAULT_WAIT_TIME > std::numeric_limits<uint16_t>::max() ?
                      std::numeric_limits<uint16_t>::max() :
                      NOTIFY_DEFAULT_WAIT_TIME;
    if (GetExternalInputHcclExecTimeoutSet() != HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_NOT_SET
        || CommConfiger::GetInstance().GetCommConfigExecTimeOutSet(identifier_)) {
        s32 execTimeOut = CommConfiger::GetInstance().GetCommConfigExecTimeOut(identifier_);
        if (execTimeOut >= MAX_VALUE_U16) {
            timeOut = MAX_VALUE_U16;
        } else {
            timeOut = execTimeOut;
        }
    }

    if (tilingDataSize > std::numeric_limits<uint32_t>::max()) {
        HCCL_ERROR(
            "[AicpuUnfoldKernelLaunchV2] tilingDataSize[%llu] exceeds the "
            "maximum allowed value for u32 [%u].",
            tilingDataSize, std::numeric_limits<uint32_t>::max());
        return HCCL_E_RUNTIME;
    }

    CHK_RET(AicpuAclKernelLaunchV2(
        stream, static_cast<void*>(&commContext), sizeof(commContext), binHandle_, kernelName, false, timeOut,
        tilingDataPtr, tilingDataSize, identifier_));
    HCCL_DEBUG("[HcclOneSidedService][AicpuUnfoldKernelLaunchV2] exec succ.");
    return HCCL_SUCCESS;
}
} // namespace hccl
