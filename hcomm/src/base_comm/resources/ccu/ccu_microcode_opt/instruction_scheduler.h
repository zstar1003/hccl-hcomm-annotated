/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H
#define CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H

#include <cstdint>
#include <vector>

#include "ccu_instr_info_v1.h"
#include "extract_operands.h"

namespace hcomm {
namespace CcuOpt {

    struct SchedulerStats {
        uint32_t nopRemoved = 0;     // 输入序列里被剥离的 NOP 数 (CkeOnly 不剥离, 恒 0)
        uint32_t nopInserted = 0;    // 调度过程因 latency 阻塞而新插入的 NOP 数
        uint32_t instrReordered = 0; // 与输入顺序不同的真实指令数 (CkeOnly 不重排, 恒 0)
        uint32_t basicBlocks = 0;    // 切分得到的 BB 个数 (CkeOnly 不切分, 占位为 1)

        // originIndex[i] = 调度后第 i 条指令对应的输入本地下标; -1 表示新插入的填充 NOP.
        std::vector<int32_t> originIndex{};
        std::vector<uint16_t> strippedNopIndices{}; // CkeOnly 不剥离, 恒为空.
    };

    // 指令调度算法档位. 极简后端优化只提供 CkeOnly 一档:
    //  * CkeOnly: 默认档 (唯一档). 完全不重排 / 不剥离, 按固定 latency 补 NOP, 覆盖:
    //             (1) CKE 写后读 (setcke/clearcke -> waitcke/clearcke);
    //             (2) LoadX/StoreX/ClearX 写 xn/array -> 后续任意指令读 (half-rtt 写者方向);
    //             (3) 任意前序指令写 xn/array -> LoadX/StoreX/ClearX 读 (硬件对 lsx/clearx 读操作数
    //                 interlock 失效, half-rtt 读者方向).
    //             普通 XN 写后读 (非 lsx/clearx 读者) 及 MS 写后读仍交由硬件 interlock, 不补 NOP.
    //             (2)(3) 与指令空间预留同源, 保证不越界.
    enum class SchedLevel : uint8_t {
        CkeOnly = 0,
    };

    // 优化器侧轻量 pinned 组描述: [baseId, baseId + count) 表示一批连续 XN 寄存器构成的 array.
    // 各组互不相交, 用 baseId 作为整组唯一 key. 与 ccu_kernel 的 PinnedGroupEntry 解耦, 避免后端
    // 优化 (base_comm 内更底层) 反向依赖 kernel 头, 满足分层约束; 由 translator 在接入点转换填充.
    struct PinnedGroup {
        uint16_t baseId = 0;
        uint16_t count = 0;
    };

    struct InstructionSchedulerOptions {
        SchedLevel level = SchedLevel::CkeOnly;
        // XN 写后读 (LoadX/StoreX/ClearX) 归约用的 pinned 组列表; 空 (nullptr) 表示不做 array 归约.
        // 生命周期契约: 指针为只读弱引用, 不拥有所指对象; 被指 vector 由调用方 (translator 的
        // RunV2BackendOptimizer 局部变量) 持有. Run() -> Optimize() -> Schedule() 全程同步调用,
        // 该 vector 在整条调用栈内始终存活, scheduler 仅在此调用栈内读, 不跨调用缓存该指针,
        // 故无悬垂风险. 调用方须保证在 Schedule 返回前不销毁/搬移该 vector.
        const std::vector<PinnedGroup>* pinnedGroups = nullptr;
    };

    class InstructionScheduler {
    public:
        explicit InstructionScheduler(InstructionSchedulerOptions opts = {}) : opts_(opts) {}

        CcuRep::CcuInstrInfo Schedule(const CcuRep::CcuInstrInfo& input);

        const SchedulerStats& Stats() const { return stats_; }

    private:
        InstructionSchedulerOptions opts_;
        SchedulerStats stats_{};

        // CkeOnly (默认档) 实现: 仅顺序扫描 + 只对 CKE 写后读按固定 cke latency 补 NOP,
        // XN / MS 写后读不补 (硬件 interlock).
        CcuRep::CcuInstrInfo ScheduleCkeOnly(const CcuRep::CcuInstrInfo& input);
    };

} // namespace CcuOpt
} // namespace hcomm

#endif // CCU_MICROCODE_OPT_INSTRUCTION_SCHEDULER_H
