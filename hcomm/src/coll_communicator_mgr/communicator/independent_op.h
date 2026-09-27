/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef INDEPENDENT_OP_RESOURCE_MANAGER_H
#define INDEPENDENT_OP_RESOURCE_MANAGER_H

#include <memory>
#include <string>
#include <unordered_set>
#include <atomic>
#include "hccl/hccl_res.h"
#include "hccl_independent_common.h"
#include "comm_mem_manager.h"
#include "comm_engine_res_manager.h"
#include "rank_graph.h"
#include "comm_config_pub.h"
#include "independent_op_context_manager.h"
#include "channel_manager.h"
#include "aicpu_init_param.h"

namespace hccl {
constexpr int32_t HCCL_COMM_ENGINE_CONFIG_NOT_SET = -1;
constexpr uint32_t HCCL_COMM_THREADNUM_CONFIG_NOT_SET = 0xffffffff;
constexpr uint32_t HCCL_COMM_NOTIFY_NUM_PER_THREAD_CONFIG_NOT_SET = 0xffffffff;

class IndependentOp {
public:
    IndependentOp();
    ~IndependentOp() = default;

    // 初始化资源管理器
    HcclResult SetIndependentOpConfig(
        const CommConfig& commConfig, const RankTable_t& rankTable, const HcclTopoAttr& topoAttr,
        const aclrtBinHandle binHandle, HDCommunicateParams& kfcControlTransferH2DParams,
        HDCommunicateParams& kfcStatusTransferD2HParams, CCLBufferManager& bufferManager);
    HcclResult SetChannelCallbacks(const ChannelManagerCallbacks& channelCallbacks);

    // 获取配置信息
    u32 GetThreadNum() const { return threadNum_; }
    u32 GetNotifyNumPerThread() const { return notifyNumPerThread_; }

    bool GetAicpuCommState(); // 是否线程安全
    void SetAicpuCommState(bool aicpuCommState);

    inline CommMemMgr& GetCommMemMgr() { return commMemMgr_; }

    inline CommEngineResMgr& GetCommEngineResMgr() { return engineResMgr_; }

    inline ContextManager& GetContextManager() { return contextMgr_; }

    inline ChannelManager& GetChannelManager() { return channelMgr_; }

    // 自定义算子Aicpu通信域公共初始化
    HcclResult KernelLaunchAicpuCommInit();

private:
    // config内容
    int32_t commEngine_ = -1;
    u32 threadNum_ = 0;
    u32 notifyNumPerThread_ = 0;
    u64 cclBufferSize_ = 0;
    std::string commId_;
    aclrtBinHandle binHandle_ = nullptr;

    // 管理器
    CommMemMgr commMemMgr_;
    CommEngineResMgr engineResMgr_;
    ContextManager contextMgr_;
    ChannelManager channelMgr_;

    std::atomic<bool> isAicpuCommInit_{false};
    CommAicpuParam commAicpuParam_{};
};

} // namespace hccl
#endif
