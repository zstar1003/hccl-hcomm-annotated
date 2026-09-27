/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_MONITOR_H
#define ENDPOINT_MONITOR_H
#include <memory>
#include <thread>
#include <unordered_set>
#include <mutex>
#include <atomic>
#include "log.h"
#include "hccl/hccl_types.h"
#include "hcomm_res_defs.h"
#include "hccp_ctx.h"

namespace hcomm {

/**
 * @note 每张卡一格 Endpoint 监控。Device UB 才 Register 进 set 并起线程。
 *
 * 仍是单例（每格一份）。构造禁止拷贝，GetHolder(id) 里 static 数组只构造一次，
 * 之后每次调用返回同一格的 shared_ptr 拷贝。变的是生命周期，不是实例个数。
 * 旧写法是 Meyers 数组（static array + GetInstance 返回 T&），寿命绑在静态析构上；
 * 静态对象先构造的晚析构。GetHolder 往往比其它静态对象更晚才第一次调用，
 * 退出时自己的 static 会先拆。已 Register 的 Endpoint 多持有一份，就能活过那些先构造的静态对象的析构。
 *
 * 用法：
 * 1. 入口只留 GetHolder()。Register：Endpoint AttachMonitor 拷贝后，用持有的指针 Register。ResMgr warmup 调
 * GetHolder()。
 * 2. Destroy / 析构：走持有的指针 Remove，再 reset。未 Register 的空指针直接返回。
 * 3. 析构路径不要再调 GetHolder()。函数内那份 static 数组拆掉后，入口已悬空。
 */
class EndpointMonitor {
public:
    EndpointMonitor() = default;
    EndpointMonitor(const EndpointMonitor&) = delete;
    EndpointMonitor& operator=(const EndpointMonitor&) = delete;
    EndpointMonitor(EndpointMonitor&&) = delete;
    EndpointMonitor& operator=(EndpointMonitor&&) = delete;
    ~EndpointMonitor();

    static std::shared_ptr<EndpointMonitor> GetHolder(s32 deviceId);
    HcclResult RegisterToEndpointMonitor(s32 deviceLogicId, EndpointHandle epHandle);
    HcclResult UnRegisterToEndpointMonitor();
    void RemoveEpHandleFromEndpointMonitor(EndpointHandle epHandle);

private:
    HcclResult RunMonitorThread();
    void MonitorThread();
    HcclResult DeInit(s32 deviceLogicId);

    void ProcessUbAsyncEvents();
    void PrintUbAsyncEventsContext(const void* epHandle, u32 seq, const struct AsyncEvent& event) const;

    static constexpr u32 MONITOR_INTERVAL = 1000;
    std::unique_ptr<std::thread> endpointMonitorThread_;

    std::atomic<bool> initialized_{false};
    std::atomic<bool> endpointMonitorThreadFlag_{false};
    std::mutex threadLock_;
    u32 devPhyId_{0};
    s32 deviceLogicId_{0};
    std::unordered_set<u64> epHandleSet_;

    struct AsyncEvent events_[ASYNC_EVENT_MAX_NUM];
};
} // namespace hcomm
#endif // ENDPOINT_MONITOR_H
