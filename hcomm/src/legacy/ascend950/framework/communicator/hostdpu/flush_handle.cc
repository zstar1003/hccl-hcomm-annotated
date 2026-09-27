/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "flush_handle.h"
#include <stdlib.h>
#include "hccp.h"
#include "orion_adapter_rts.h"
#include "orion_adapter_hccp.h"
#include "config_plf_log_v2.h"

namespace Hccl {

FlushHandle::FlushHandle() : flushIsInitialized(false) {}

FlushHandle::~FlushHandle() { Destroy(); }

HcclResult FlushHandle::Init(IpAddress ip, u32 devPhyId)
{
    int lbMax = 0;
    // 获取 RDMA handle
    CHK_RET(GetRdmaHandle(ip, devPhyId, &rdmaHandle));

    // 获取 LbMax
    CHK_RET(GetLbMax(&lbMax));

    if (lbMax > 0) {
        SetFlushOpcodeSupport();
    }

    // 分配 Local Memory
    CHK_RET(AllocateLocalMemory());

    // 分配 Device Memory
    CHK_RET(AllocateDeviceMemory());

    // 创建环回 QP
    CHK_RET(CreateLoopbackQp());

    // 注册 Local MR
    CHK_RET(RegisterLocalMr());

    // 注册 Remote MR
    CHK_RET(RegisterRemoteMr());

    flushIsInitialized = true;
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::GetLbMax(int* lbMax) const
{
    int ret = RaGetLbMax(rdmaHandle, lbMax);
    if (ret != 0) {
        HCCL_ERROR("[GetLbMax]Failed to get load balance max value. error_code=%d.", ret);
        return HCCL_E_ROCE_CONNECT;
    }
    HCCL_INFO("[GetLbMax]Get load balance max value successfully, lbMax = %d", *lbMax);
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::Destroy()
{
    HcclResult finalResult = HCCL_SUCCESS;
    if (remoteMrHandle != nullptr) {
        finalResult = std::max(finalResult, DeregisterMr(remoteMrHandle, "Remote"));
    }
    if (localMrHandle != nullptr) {
        finalResult = std::max(finalResult, DeregisterMr(localMrHandle, "Local"));
    }
    finalResult = std::max(finalResult, DestroyLoopbackQp());
    finalResult = std::max(finalResult, FreeLocalMemory());
    finalResult = std::max(finalResult, FreeDeviceMemory());
    return finalResult;
}

HcclResult FlushHandle::GetRdmaHandle(IpAddress ip, u32 devPhyId, void** rdmaHandle) const
{
    *rdmaHandle
        = RdmaHandleManager::GetInstance().GetByAddr(devPhyId, LinkProtoType::RDMA, ip, PortDeploymentType::HOST_NET);
    CHK_PTR_NULL(*rdmaHandle);

    HCCL_DEBUG("[GetRdmaHandle]RDMA handle initialized. ");

    return HCCL_SUCCESS;
}

HcclResult FlushHandle::AllocateLocalMemory()
{
    u64 bufferSize = FLUSH_BUFFER_SIZE;
    if (flushOpcodeSupport_) {
        // 1825 主动排空要求使用device 内存
        localMem = HrtMalloc(bufferSize, static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    } else {
        localMem = malloc(bufferSize);
    }

    if (localMem == nullptr) {
        HcclResult eRet = Destroy();
        HCCL_ERROR("[%s]Failed to Allocate Local Memory. Destroy Flush code=%d", __func__, eRet);
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG("[%s]Local memory allocated at %p, size=%u", __func__, localMem, bufferSize);
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::AllocateDeviceMemory()
{
    u64 bufferSize = FLUSH_BUFFER_SIZE;
    deviceMem = HrtMalloc(bufferSize, static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH));
    if (deviceMem == nullptr) {
        HcclResult eRet = Destroy();
        HCCL_ERROR("[AllocateDeviceMemory]Failed to Allocate Device Memory. Destroy Flush code=%d", eRet);
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG("[AllocateDeviceMemory]Device memory allocated at %p, size=%u", deviceMem, bufferSize);
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::CreateLoopbackQp()
{
    int ret = RaLoopbackQpCreate(rdmaHandle, &loopBackQpParam, &qpHandle);
    if (ret != 0) {
        HcclResult eRet = Destroy();
        HCCL_ERROR("[CreateLoopbackQp]Failed to create loopback QP. error_code=%d. Destroy Flush code=%d", ret, eRet);
        return HCCL_E_ROCE_CONNECT;
    }
    HCCL_DEBUG("[CreateLoopbackQp]Loopback QP created successfully. QP Handle=%p", qpHandle);
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::RegisterLocalMr()
{
    u64 bufferSize = FLUSH_BUFFER_SIZE;
    loopBackQpMrLocalInfo.addr = localMem;
    loopBackQpMrLocalInfo.size = bufferSize;
    loopBackQpMrLocalInfo.access = RA_ACCESS_LOCAL_WRITE;

    int localRet = RaRegisterMr(rdmaHandle, &loopBackQpMrLocalInfo, &localMrHandle);
    if (localRet != 0 || localMrHandle == nullptr) {
        HCCL_ERROR(
            "[RegisterLocalMr]Failed to register local MR. localMrHandle=0x%p, error_code=%d", localMrHandle, localRet);
        HcclResult eRet = Destroy();
        HCCL_ERROR(
            "[RegisterLocalMr]Failed to register local MR. error_code=%d. Destroy Flush code=%d", localRet, eRet);
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG("[RegisterLocalMr]Local MR registered successfully. MR Handle=0x%p", localMrHandle);
    return HCCL_SUCCESS;
}

HcclResult FlushHandle::RegisterRemoteMr()
{
    u64 bufferSize = FLUSH_BUFFER_SIZE;
    loopBackQpMrRemoteInfo.addr = deviceMem;
    loopBackQpMrRemoteInfo.size = bufferSize;
    loopBackQpMrRemoteInfo.access = static_cast<int>(
        static_cast<unsigned int>(RA_ACCESS_REMOTE_WRITE) | static_cast<unsigned int>(RA_ACCESS_LOCAL_WRITE)
        | static_cast<unsigned int>(RA_ACCESS_REMOTE_READ) | static_cast<unsigned int>(RA_ACCESS_REMOTE_ATOMIC));

    int remoteRet = RaRegisterMr(rdmaHandle, &loopBackQpMrRemoteInfo, &remoteMrHandle);
    if (remoteRet != 0 || remoteMrHandle == nullptr) {
        HCCL_ERROR(
            "[RegisterRemoteMr]Failed to register remote MR. remoteMrHandle=0x%p, error_code=%d", remoteMrHandle,
            remoteRet);
        HcclResult eRet = Destroy();
        HCCL_ERROR(
            "[RegisterLocalMr]Failed to register remote MR. error_code=%d. Destroy Flush code=%d", remoteRet, eRet);
        return HCCL_E_MEMORY;
    }
    HCCL_DEBUG("[RegisterRemoteMr]Remote MR registered successfully. MR Handle=0x%p", remoteMrHandle);
    return HCCL_SUCCESS;
}

// 销毁 MR
HcclResult FlushHandle::DeregisterMr(MrHandle& mrHandle, std::string logTag) const
{
    HCCL_DEBUG("[DeregisterMr] Starting to destroy %s MR...", logTag.c_str());

    if (mrHandle == nullptr || rdmaHandle == nullptr) {
        HCCL_DEBUG("[DeregisterMr] %s MR is already null, skipping.", logTag.c_str());
        return HCCL_SUCCESS;
    }

    int ret = RaDeregisterMr(rdmaHandle, mrHandle);
    if (ret != 0) {
        HCCL_ERROR(
            "[DeregisterMr] Failed to deregister %s MR, mrHandle=0x%p, error_code=%d.", logTag.c_str(), mrHandle, ret);
        mrHandle = nullptr; // 防止重复调用
        return HCCL_E_INTERNAL;
    }

    mrHandle = nullptr;
    HCCL_DEBUG("[DeregisterMr] %s MR successfully deregistered.", logTag.c_str());
    return HCCL_SUCCESS;
}

// 销毁环回 QP
HcclResult FlushHandle::DestroyLoopbackQp()
{
    HCCL_DEBUG("[DestroyLoopbackQp] Starting to destroy loopback QP...");

    if (qpHandle == nullptr) {
        HCCL_DEBUG("[DestroyLoopbackQp] QP already null, skipping.");
        return HCCL_SUCCESS;
    }

    if ((GetPlfDebugConfigValue() & PLF_RES) != 0) {
        u32 qpn = HrtGetQpNum(qpHandle);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
    }
    int ret = RaQpDestroy(qpHandle);
    if (ret != 0) {
        HCCL_ERROR("[DestroyLoopbackQp] Failed to destroy QP. qpHandle=%p, error=%d", qpHandle, ret);
        qpHandle = nullptr;
        return HCCL_E_INTERNAL;
    }

    qpHandle = nullptr;
    HCCL_DEBUG("[DestroyLoopbackQp] Loopback QP successfully destroyed.");
    return HCCL_SUCCESS;
}

// 释放 Local MR 内存
HcclResult FlushHandle::FreeLocalMemory()
{
    HCCL_DEBUG("[%s] Starting to free local memory...", __func__);

    if (localMem == nullptr) {
        HCCL_DEBUG("[%s] Local memory already null, skipping.", __func__);
        return HCCL_SUCCESS;
    }

    try {
        if (flushOpcodeSupport_) {
            HrtFree(localMem);
        } else {
            free(localMem);
        }
    } catch (HcclException& e) {
        HCCL_ERROR("[%s] Exception occurred: %s", __func__, e.what());
        return e.GetErrorCode();
    } catch (...) {
        HCCL_ERROR("[%s] Exception caught while freeing local memory.", __func__);
        return HcclResult::HCCL_E_INTERNAL;
    }
    localMem = nullptr;
    HCCL_DEBUG("[%s] Local memory successfully freed.", __func__);
    return HCCL_SUCCESS;
}

// 释放 Device 内存
HcclResult FlushHandle::FreeDeviceMemory()
{
    HCCL_DEBUG("[FreeDeviceMemory] Starting to free device memory...");

    if (deviceMem == nullptr) {
        HCCL_DEBUG("[FreeDeviceMemory] Device memory already null, skipping.");
        return HCCL_SUCCESS;
    }

    try {
        HrtFree(deviceMem);
        deviceMem = nullptr;
        HCCL_DEBUG("[FreeDeviceMemory] Device memory successfully freed.");
        return HCCL_SUCCESS;
    } catch (...) {
        HCCL_ERROR("[FreeDeviceMemory] Exception caught while freeing device memory.");
        deviceMem = nullptr;
        return HCCL_E_RUNTIME;
    }
}

} // namespace Hccl
