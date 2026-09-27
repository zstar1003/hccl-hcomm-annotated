/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_RES_MGR_H
#define HCOMM_RES_MGR_H

#include <array>
#include <cstdint>
#include <mutex>
#include "acl/acl_rt.h"
#include "hcomm_res_defs.h"
#include "config_mgr/config_mgr.h"
#include "hccl_common.h"
#include "resources/endpoints/mgr/endpoint_mgr.h"
#include "resources/endpoints/mgr/endpoint_ctx_mgr.h"

namespace hcomm {

/**
 * @brief 单设备资源管理基类。
 *
 * 每个设备对应一个实例，持有 devPhyId，负责触发该设备下各子模块单例创建，
 * 持有该设备的 EndpointCtxMgr（EndpointCtx 去重缓存，per-device）。
 */
class HcommBaseResMgr {
public:
    HcommBaseResMgr() = default;
    explicit HcommBaseResMgr(uint32_t devPhyId) : devPhyId_(devPhyId) {}
    ~HcommBaseResMgr() = default;

    void Init() const;
    void SetDevPhyId(uint32_t devPhyId) { devPhyId_ = devPhyId; }
    uint32_t GetDevPhyId() const { return devPhyId_; }
    // EndpointCtx 去重缓存按 devPhyId 隔离，经 GetDeviceResMgr(devPhyId).GetEndpointCtxMgr() 访问
    EndpointCtxMgr& GetEndpointCtxMgr() { return endpointCtxMgr_; }

private:
    uint32_t devPhyId_{0};
    EndpointCtxMgr endpointCtxMgr_;
};

/**
 * @brief 全局资源管理类，聚合各设备实例和环境变量配置。
 *
 * 持有 HcommBaseResMgr[MAX_MODULE_DEVICE_NUM + 1] 和 ConfigMgr，
 * 通过 GetInstance() 获取单例，GetDeviceResMgr() 访问指定设备，GetConfigMgr() 访问环境变量配置。
 */
class HcommResMgr {
public:
    static HcommResMgr& GetInstance();
    static void RegisterDeviceResetCallback();
    static void RegisterDeviceRefreshCallback();
    static void UnregisterDeviceRefreshCallback();

    HcommBaseResMgr& GetDeviceResMgr(uint32_t devicePhyId);
    ConfigMgr& GetConfigMgr();
    // EndpointMgr 作为 HcommResMgr 成员：endpoint 句柄表（handle→Endpoint）全局管理层
    EndpointMgr& GetEndpointMgr() { return endpointMgr_; }

    // 进程级 kernel bin 句柄管理（不分 device，与 per-device 实例隔离）
    static HcclResult EnsureKernelBinLoaded(CommEngine engine);
    static aclrtBinHandle GetBinHandle();

private:
    HcommResMgr();
    // 进程销毁兜底：endpointMgr_.DeInit() 清残留 endpoint（析构链逐设备回调 EndpointCtxMgr 释放 ctx），
    // 再遍历 deviceResMgrs_ 调各 EndpointCtxMgr::DeInit() 清残留 EndpointCtx
    ~HcommResMgr();
    HcommResMgr(const HcommResMgr& that) = delete;
    HcommResMgr& operator=(const HcommResMgr& that) = delete;

    void InitDevice(uint32_t devPhyId);

    std::array<bool, MAX_MODULE_DEVICE_NUM + 1> isInitialized_{false};
    HcommBaseResMgr deviceResMgrs_[MAX_MODULE_DEVICE_NUM + 1];
    ConfigMgr configMgr_;
    EndpointMgr endpointMgr_; // 声明在最后：先于其它成员析构，析构期 DeInit 依赖的 deviceResMgrs_ 成员仍存活

    static aclrtBinHandle binHandle_;
    static std::mutex binHandleMtx_;
};

} // namespace hcomm

#endif // HCOMM_RES_MGR_H
