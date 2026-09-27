/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ORDER_LAUNCH_THREAD_MGR_H
#define HCCLV2_ORDER_LAUNCH_THREAD_MGR_H

#include <mutex>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include "hccl/base.h"
#include "hccl/hccl_types.h"
#include "hccl/hccl_res.h"
#include "hcomm_res.h"
#include "log.h"

namespace hccl {

class CollComm;

/** 保序 thread 模式：单算子 / aclgraph（内部使用） */
enum class OrderThreadMode : u8 {
    OPBASE,   /**< 单算子模式保序 thread */
    ACLGRAPH, /**< aclgraph 模式保序 thread */
};

/**
 * @brief 单个 context 下的保序资源描述。
 *
 * thread 通过 HcommThreadAlloc 创建（进程粒度，不与通信域绑定），
 * 注册到全局映射表。thread 的生命周期由本结构通过 HcommThreadFree 管理。
 */
struct OrderLaunchContextRes {
    u64 context{UINT64_MAX};
    ThreadHandle opbaseThread{0};
    ThreadHandle aclgraphThread{0};
    bool resValid{false};

    void DestroyResources();
};

/**
 * @brief 保序资源管理器（资源管理面，进程粒度）。
 *
 * thread 通过 HcommThreadAlloc/HcommThreadFree 管理生命周期，不与通信域绑定。
 * 可通过 HcommThreadExportToCommEngine 导出到目标引擎。
 */
class OrderLaunchThreadMgr {
public:
    OrderLaunchThreadMgr();
    ~OrderLaunchThreadMgr();

    HcclResult RegisterOrderLaunch(const std::string& group);
    HcclResult UnRegisterOrderLaunch(const std::string& group);

    HcclResult SetAttachedStream(const std::string& group, u32 graphId, void* stream);
    HcclResult EnsureOrderThread(
        OrderThreadMode mode, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread);
    HcclResult EnsureDeviceOrderThread(
        CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread, ThreadHandle& thread);
    ThreadHandle GetHcomAttachedThreadByGroup(const std::string& group);
    HcclResult OrderLaunchThreadAcquire(
        HcclDedicatedThreadType useType, CollComm* collComm, const std::string& group, uint32_t notifyNumPerThread,
        ThreadHandle& thread);

private:
    void Destroy();
    HcclResult RegisterThreadToComm(CollComm* collComm, ThreadHandle thread) const;
    HcclResult RegisterDfx(
        CollComm* collComm, HcclDedicatedThreadType useType, ThreadHandle thread, u64 beginTime,
        const std::string& commId) const;
    HcclResult GetCurrentContext(u64& currentContext) const;
    HcclResult EnsureContextRes(u64 context);
    void UpdateGroupContextMapping(const std::string& group, u64 currentContext);
    bool IsOrderLaunchDisabled(u64 currentContext);

    std::mutex mutex_;
    std::unordered_map<std::string, u64> groupCtxMap_;
    std::unordered_map<u64, std::unordered_set<std::string>> contextGroupsMap_;
    std::unordered_map<u64, OrderLaunchContextRes> contextResMap_;
    std::unordered_map<u32, ThreadHandle> hcomAttachedThreadMap_; // graphId -> 附属从 thread（图模式使用）
    std::unordered_map<std::string, u32> groupGraphMap_;          // group -> graphId 映射
    u32 blockNum_{0};                                             // AICPU block 数（懒加载缓存，0=未查询）
};

} // namespace hccl

#endif // HCCLV2_ORDER_LAUNCH_THREAD_MGR_H
