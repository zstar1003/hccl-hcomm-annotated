/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_ORDER_LAUNCH_H
#define HCCL_ORDER_LAUNCH_H

#include <unordered_set>
#include <map>
#include <memory>
#include "hccl/hccl_types.h"
#include "stream_pub.h"
#include "hccl_common.h"
#include "local_notify.h"
#include "adapter_rts.h"
#include "log.h"

namespace hccl {
struct OrderLaunchResMgr {
    u64 context;
    Stream opbaseStream;   // 单算子模式使用
    Stream aclgraphStream; // aclgraph模式使用
    bool resValid;
    bool contextInitialized; // 标记context是否已初始化，一旦初始化就不会变回INVALID_U64）

    OrderLaunchResMgr()
        : context(INVALID_U64),
          opbaseStream(),
          aclgraphStream(),
          resValid(false),
          contextInitialized(false)
    {}

    // 检查context是否匹配，不匹配时需要更新context但保留stream
    bool IsCtxMatch(u64 currentContext) const { return context == currentContext && resValid; }

    // 更新context
    void UpdateContext(u64 newContext)
    {
        if (context != newContext) {
            HCCL_INFO("[OrderLaunchResMgr] context updated: [0x%llx] -> [0x%llx]", context, newContext);
            context = newContext;
            resValid = false; // context变化认为资源暂时无效
        }
    }

    // 设置context为已初始化状态（一旦调用，context就不会再变回INVALID_U64）
    void MarkContextInitialized(u64 newContext)
    {
        if (context == INVALID_U64 || context != newContext) {
            HCCL_INFO("[OrderLaunchResMgr] Mark context initialized: [0x%llx] -> [0x%llx]", context, newContext);
            context = newContext;
            contextInitialized = true; // 标记为已初始化
            resValid = false;
        }
    }

    // 析构资源
    void DestroyResources() const
    {
        // 析构资源位置预留，后续根据需要实现
    }
};

class OrderLaunch {
public:
    static OrderLaunch& GetInstance(s32 deviceLogicID);
    HcclResult RegisterOrderLaunch(const std::string& group);
    HcclResult UnRegisterOrderLaunch(const std::string& group);

    HcclResult AclgraphLaunchInOrderToOrderStream(
        std::string& group, const Stream& kernelStream, const Stream& mainStream, std::shared_ptr<LocalNotify> notify0,
        std::shared_ptr<LocalNotify> notify1, u32 timeOut, HcclRtEvent event);
    HcclResult AclgraphLaunchInOrderToKernelStream(std::string& group, const Stream& kernelStream, HcclRtEvent event);

    HcclResult OpbaseLaunchInOrder(
        std::string& group, const Stream& kernelStream, std::shared_ptr<LocalNotify> notify0,
        std::shared_ptr<LocalNotify> notify1, u32 timeOut);

    HcclResult HcomLaunchInOrder(
        std::string& group, const Stream& kernelStream, u32 graphId, std::shared_ptr<LocalNotify> notify0,
        std::shared_ptr<LocalNotify> notify1, u32 timeOut);

    HcclResult SetHcomStream(u32 graphId, const Stream& hcomAttachedStream);

private:
    OrderLaunch();
    ~OrderLaunch();

    void DestroyRes();
    HcclResult LaunchInOrder(
        const std::string& group, const Stream& kernelStream, const Stream& hostOrderStream,
        std::shared_ptr<LocalNotify> notify0, std::shared_ptr<LocalNotify> notify1, u32 timeOut);
    HcclResult InitGroupCtx(const std::string& group);
    HcclResult EnsureOrderStreamForGroup(std::string& group, u64 context, Stream& orderStream);
    HcclResult GetCurrentContext(u64& currentContext) const;

    std::mutex streamMutex_;
    std::unordered_map<std::string, u64> groupCtxMap_;                          // group -> context 映射
    std::unordered_map<u64, std::unordered_set<std::string>> contextGroupsMap_; // context -> groups 映射
    std::unordered_map<u64, OrderLaunchResMgr> contextResMgrMap_; // context -> OrderLaunchResMgr (stream由context管理)
    std::unordered_map<u32, Stream> hcomStreamMap_;               // 图模式使用
    bool initialized_ = false;
};
} // namespace hccl
#endif
