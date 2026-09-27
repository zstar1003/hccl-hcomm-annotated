/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_MICROCODE_OPT_MICROCODE_OPTIMIZER_H
#define CCU_MICROCODE_OPT_MICROCODE_OPTIMIZER_H

#include "ccu_instr_info_v1.h"
#include "instruction_scheduler.h"

namespace hcomm {
namespace CcuOpt {

    struct OptimizerStats {
        SchedulerStats sched{};
        // 后端优化耗时 (单位: 微秒 us), 由 Optimize() 填充.
        uint64_t pass1DurationUs = 0;
        uint64_t totalDurationUs = 0;
    };

    // 极简后端优化只有 CkeOnly 一档, 无寄存器重排; 保留结构体仅为对外形态统一.
    struct OptimizerOptions {
        SchedLevel schedLevel = SchedLevel::CkeOnly;
    };

    class MicrocodeOptimizer {
    public:
        MicrocodeOptimizer();

        // 档位设置 (schedLevel 会同步到 InstructionSchedulerOptions::level).
        void SetOptions(const OptimizerOptions& opts)
        {
            opts_ = opts;
            schedOpts_.level = opts.schedLevel;
        }
        const OptimizerOptions& Options() const { return opts_; }

        // 设置 XN 写后读归约用的 pinned 组 (LoadX/StoreX/ClearX). 指针语义, 生命周期由调用方持有;
        // 仅在随后的 Optimize() 调用栈内使用. 传 nullptr 表示不做 array 归约.
        void SetPinnedGroups(const std::vector<PinnedGroup>* pinnedGroups) { schedOpts_.pinnedGroups = pinnedGroups; }

        CcuRep::CcuInstrInfo Optimize(const CcuRep::CcuInstrInfo& input);

        const OptimizerStats& Stats() const { return stats_; }

        // 编译期默认档位: 恒为 CkeOnly (极简后端优化无其他档位).
        static OptimizerOptions DefaultOptions();

        // 后端优化统一入口: V2 场景无条件启用 (V1 由调用方保证不进来).
        // reserveXnId / reserveCkeId 为 translator 保留寄存器, CkeOnly 不重命名寄存器,
        // 故这两个参数当前仅用于日志观测.
        // pinnedGroups: XN 写后读 (LoadX/StoreX/ClearX) array 归约用的 pinned 组; 默认空 (不归约).
        static CcuRep::CcuInstrInfo
        Run(const CcuRep::CcuInstrInfo& input, uint16_t reserveXnId, uint16_t reserveCkeId,
            const std::vector<PinnedGroup>& pinnedGroups = {});

    private:
        // dfx / opt_log: 把"优化前指令序列 + 优化后指令序列 + 后→前指令下标映射"一次性写入
        // HCCL_INFO, 供用户结合硬件 runlog 定位错误来源.
        void DumpOptLog(const CcuRep::CcuInstrInfo& input, const CcuRep::CcuInstrInfo& output) const;

        InstructionSchedulerOptions schedOpts_;
        OptimizerOptions opts_;
        OptimizerStats stats_{};
    };

} // namespace CcuOpt
} // namespace hcomm

#endif // CCU_MICROCODE_OPT_MICROCODE_OPTIMIZER_H
