/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hcomm_res_mgr.h"

#include <mutex>

#include "hccl_common.h"
#include "comm_engine_utils.h"
#include "launch_device.h"
#include "launch_aicpu.h"

// orion 通用平台层单例
#include "hccp_hdc_manager.h"
#include "hccp_peer_manager.h"
#include "hccp_tlv_hdc_manager.h"
#include "rdma_handle_manager.h"
#include "inner_net_dev_manager.h"
#include "socket_handle_manager.h"
#include "host_socket_handle_manager.h"
#include "tp_manager.h"
#include "endpoint_monitor.h"
// legacy ccu单例
#include "ccu_component.h"
#include "ccu_res_batch_allocator_legacy.h"
#include "../../../legacy/ascend950/unified_platform/ccu/ccu_context/ccu_context_mgr_imp.h"
// 开源开放 ccu单例
#include "hccp_tlv_hdc_mgr.h"
#include "tp_mgr.h"
#include "ccu_comp.h"
#include "resources/ccu/ccu_device/ccu_res_batch_allocator.h"
#include "ccu_kernel_mgr.h"
#include "ccu_instance_mgr.h"
#include "../endpoint_pairs/sockets/socket_process.h"
#include "dpu_notify/dpu_notify_manager.h"
#include "server_socket_mgr.h"
#include "server_socket_manager.h"
#include "adapter_rts_common.h"

namespace hcomm {

static std::mutex g_deviceResetRegMutex;
static std::mutex g_deviceRefreshRegMutex;
static std::mutex g_deviceRefreshMutex;
static bool g_deviceRefreshCallbackRegistered = false;
static bool g_deviceResetCallbackRegistered = false;

aclrtBinHandle HcommResMgr::binHandle_ = nullptr;
std::mutex HcommResMgr::binHandleMtx_;

HcclResult HcommResMgr::EnsureKernelBinLoaded(CommEngine engine)
{
    if (engine != COMM_ENGINE_AICPU && engine != COMM_ENGINE_AICPU_TS) {
        HCCL_INFO(
            "[%s] engine[%s] kernel loading not required", __func__,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_SUCCESS;
    }
    std::lock_guard<std::mutex> lock(binHandleMtx_);
    if (binHandle_ != nullptr) {
        return HCCL_SUCCESS;
    }
    std::string jsonPath;
    CHK_RET(hccl::GetKernelFilePath(jsonPath));
    jsonPath += "ccl_kernel.json";

    HcclResult ret = hccl::LoadBinaryFromFile(jsonPath.c_str(), ACL_RT_BINARY_LOAD_OPT_CPU_KERNEL_MODE, 0, binHandle_);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[%s] load aicpu file fail, path[%s]", __func__, jsonPath.c_str()), ret);
    return HCCL_SUCCESS;
}

aclrtBinHandle HcommResMgr::GetBinHandle() { return binHandle_; }

// HcommBaseResMgr

void HcommBaseResMgr::Init() const
{
    // 临时方案：只声明单例对象做生命周期控制，不执行业务动作
    // 未来需要将各种单例转为该数据结构的成员变量
    // devicePhyId 目前不影响流程，只是触发静态对象声明
    DpuNotifyManager::GetInstance();
    Hccl::HccpHdcManager::GetInstance();
    Hccl::HccpPeerManager::GetInstance();
    Hccl::HccpTlvHdcManager::GetInstance();
    Hccl::RdmaHandleManager::GetInstance();
    Hccl::InnerNetDevManager::GetInstance();
    Hccl::SocketHandleManager::GetInstance();
    Hccl::HostSocketHandleManager::GetInstance();
    SocketMgr::GetInstance(devPhyId_);
    Hccl::TpManager::GetInstance(devPhyId_);
    (void)EndpointMonitor::GetHolder(devPhyId_);

    Hccl::CcuComponent::GetInstance(devPhyId_);
    Hccl::CcuResBatchAllocator::GetInstance(devPhyId_);
    Hccl::CtxMgrImp::GetInstance(devPhyId_);

    // 开源开放架构下CCU模式新增类型单例，当前混跑时不使用
    HccpTlvHdcMgr::GetInstance(devPhyId_);
    TpMgr::GetInstance(devPhyId_);
    CcuComponent::GetInstance(devPhyId_);
    CcuResBatchAllocator::GetInstance(devPhyId_);
    CcuKernelMgr::GetInstance(devPhyId_);
    CcuInstanceMgr::GetInstance(devPhyId_);
    SocketProcess::GetInstance(devPhyId_);
}

// HcommResMgr

HcommResMgr::HcommResMgr() = default;

// 进程销毁兜底：先清全局 endpoint 句柄表残留，再遍历各设备 EndpointCtxMgr 强制释放残留 ctx
HcommResMgr::~HcommResMgr()
{
    endpointMgr_.DeInit();
    for (uint32_t devPhyId = 0; devPhyId <= MAX_MODULE_DEVICE_NUM; ++devPhyId) {
        deviceResMgrs_[devPhyId].GetEndpointCtxMgr().DeInit();
    }
    g_deviceRefreshCallbackRegistered = false;
    g_deviceResetCallbackRegistered = false;
    UnregisterDeviceRefreshCallback();
}

HcommResMgr& HcommResMgr::GetInstance()
{
    static HcommResMgr instance;
    return instance;
}

void HcommResMgr::InitDevice(uint32_t devPhyId)
{
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        HCCL_WARNING(
            "[HcommResMgr][%s] use the backup device, devPhyId[%u] should be "
            "less than %u.",
            __func__, devPhyId, MAX_MODULE_DEVICE_NUM);
        devPhyId = MAX_MODULE_DEVICE_NUM; // 使用备份设备
    }
    if (!isInitialized_[devPhyId]) {
        deviceResMgrs_[devPhyId].SetDevPhyId(devPhyId);
        deviceResMgrs_[devPhyId].Init();
        isInitialized_[devPhyId] = true;
    }
}

HcommBaseResMgr& HcommResMgr::GetDeviceResMgr(uint32_t devicePhyId)
{
    InitDevice(devicePhyId);
    uint32_t devPhyId = devicePhyId;
    if (devPhyId >= MAX_MODULE_DEVICE_NUM) {
        devPhyId = MAX_MODULE_DEVICE_NUM;
    }
    return deviceResMgrs_[devPhyId];
}

ConfigMgr& HcommResMgr::GetConfigMgr() { return configMgr_; }

static void OnDeviceResetPre(int32_t deviceId, aclrtDeviceState state, [[maybe_unused]] void* args)
{
    try {
        if (state != ACL_RT_DEVICE_STATE_RESET_PRE) {
            return;
        }
        HCCL_INFO("[%s] deviceId[%d] state[%d] ", __func__, deviceId, static_cast<int>(state));

        u32 devPhyId = 0;
        HcclResult ret = hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceId), devPhyId);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDevicePhyIdByIndex failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }
        SocketMgr::DeInit(devPhyId);
        ServerSocketMgr::DeInit(devPhyId);
        ServerSocketManager::GetInstance().DeInit(devPhyId);
        Hccl::RdmaHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::SocketHandleManager::GetInstance().DeInit(devPhyId);
        Hccl::HccpHdcManager::GetInstance().DeInit(deviceId);
    } catch (const std::exception& e) {
        HCCL_WARNING("[%s] exception caught:%s", __func__, e.what());
    } catch (...) {
        HCCL_WARNING("[%s] unknown exception caught", __func__);
    }
}

void HcommResMgr::RegisterDeviceResetCallback()
{
    std::lock_guard<std::mutex> lock(g_deviceResetRegMutex);
    if (g_deviceResetCallbackRegistered) {
        return;
    }
    aclError ret = aclrtRegDeviceStateCallback("hcomm_res_mgr", OnDeviceResetPre, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[RegisterDeviceResetCallback] aclrtRegDeviceStateCallback failed, ret[%d]", ret);
        return;
    }
    g_deviceResetCallbackRegistered = true;
    HCCL_INFO("[%s] aclrtRegDeviceStateCallback success", __func__);
}

static void OnDeviceStateRefresh(int32_t deviceId, aclrtDeviceState state, [[maybe_unused]] void* args)
{
    std::lock_guard<std::mutex> lock(g_deviceRefreshMutex);
    try {
        if (state != ACL_RT_DEVICE_STATE_SET_POST) {
            return;
        }
        HCCL_INFO("[%s] deviceId[%d] state[%d]", __func__, deviceId, static_cast<int>(state));
        s32 deviceLogicId = 0;
        HcclResult ret = hrtGetDeviceRefresh(&deviceLogicId);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDeviceRefresh failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }

        u32 devicePhyId = 0;
        ret = hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId), devicePhyId, true);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDevicePhyIdByIndex failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }

        DevType deviceType = DevType::DEV_TYPE_COUNT;
        ret = hrtGetDeviceType(deviceType);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] hrtGetDeviceType failed, deviceId[%d] ret[%d]", __func__, deviceId, ret);
            return;
        }
        HCCL_INFO(
            "[%s] refresh success, deviceLogicId[%d] devicePhyId[%d], deviceType[%d]", __func__, deviceLogicId,
            devicePhyId, static_cast<int>(deviceType));
    } catch (const std::exception& e) {
        HCCL_WARNING("[%s] exception caught:%s", __func__, e.what());
    } catch (...) {
        HCCL_WARNING("[%s] unknown exception caught", __func__);
    }
}

void HcommResMgr::RegisterDeviceRefreshCallback()
{
    std::lock_guard<std::mutex> lock(g_deviceRefreshRegMutex);
    if (g_deviceRefreshCallbackRegistered) {
        return;
    }
    aclError ret = aclrtRegDeviceStateCallback("hcomm_refresh_device", OnDeviceStateRefresh, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING("[%s] aclrtRegDeviceStateCallback failed, ret[%d]", __func__, ret);
        return;
    }
    g_deviceRefreshCallbackRegistered = true;
    HCCL_INFO("[%s] aclrtRegDeviceStateCallback success, regName[%s]", __func__, "hcomm_refresh_device");
}

void HcommResMgr::UnregisterDeviceRefreshCallback()
{
    aclError ret = aclrtRegDeviceStateCallback("hcomm_refresh_device", nullptr, nullptr);
    if (ret != ACL_SUCCESS) {
        HCCL_WARNING(
            "[%s] aclrtRegDeviceStateCallback unregister failed, "
            "regName[%s] ret[%d]",
            __func__, "hcomm_refresh_device", ret);
    }
    HCCL_INFO("[%s] unregister success", __func__);
}

} // namespace hcomm
