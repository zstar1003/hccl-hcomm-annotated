/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// ============================================================================
// A6 后端优化 (ccu_microcode_opt) CKE 补 NOP 行为验证 —— 算子 Demo
// ----------------------------------------------------------------------------
// 用 ccu:: 高层 API 搭真实 kernel, 交给框架翻译(RegisterEnd 会跑 A6 后端优化),
// 由配套 UT 检查翻译后指令序列里 NOP 的插入情况。
//
// 关键语义:
//   - EventRecord(evt): 翻译成 setcke, setCKEId=evt (置位 evt 对应 bit)。置位不算 CKE 写者。
//   - EventWait(evt): 循环体外 (isProfiling=true) 翻译成 setcke, waitCKEId=evt, clearType=1
//     (读 evt + clearType=1 自动清零 evt)。
//   - CKE 写者身份: 只有 clearType=1 自动清零的 waitCKEId 才是触发写后读的写者。setCKEId (record 置位)
//     与主动清位的 clearCKEId 都不算写者。据此, 写后读只在"前一个自清 wait" 与 "后一个读同 bit 的 wait/read"
//     之间成立, 后端优化按 CCU_CKE_RAW_LATENCY 在两者之间补 NOP。
//
// 硬件约束:
//   1) set 与 wait 必须成对, 单独 wait 一个从未被 set 的 event 会在硬件上死锁。故场景统一用
//      **成对且近邻的 set/wait**, 每个 wait 都有前置 record 置位, 既不死锁又能构造写后读。
//   2) EventRecord 与 EventWait 都只能在 **循环体外** 使用 (二者在 LOOP_BLOCK 内均非法), 故 set/wait
//      对统一放在循环体外, wait 走 profiling 分支翻译成 setcke。
//
// 提供两个场景:
//   场景1 (单对 set/wait): record(非写者)+wait(自清=写者)。wait 是写者但其后无读同一 bit 的读者,
//                         不构成写后读 -> 不补 NOP。
//   场景2 (两对近邻 set/wait): record+wait#1+record+wait#2。wait#1(自清=写者) 与 wait#2(读) 同 bit
//                         构成写后读 -> 在 wait#1 与 wait#2 之间补 NOP。
// ============================================================================

#include "ccu_primitives.hpp"
#include "ccu_types.h"

namespace ccu = ::AscendC::ccu;

// ---------------------------------------------------------------------------
// 场景2 kernel: 循环体外两对近邻的 EventRecord + EventWait (同一 event)。
// 翻译: 循环体外 wait 走 profiling 分支 -> 四条 setcke: record#1(置位,非写者) -> wait#1(waitId=evt,
//       clearType=1 自清=写者) -> record#2(置位,非写者) -> wait#2(读同 bit)。wait#1(写者) 与 wait#2(读)
//       构成写后读 -> 在 wait#1 与 wait#2 之间补 NOP。set 与 wait 成对, 不死锁。
// ---------------------------------------------------------------------------
inline CcuResult CcuCkeSetClearPairsHazardDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Event evtGuard; // 占位, 避免真正使用的 event 落到 CKE id 0 (id 0 会被依赖图兜底丢弃)
    (void)evtGuard;
    Event evt;

    // 两对近邻 set/wait: record(置位,非写者) -> wait#1(自清=写者) -> record(置位) -> wait#2(读)。
    // wait#1 与 wait#2 同 bit 构成写后读, 且成对不死锁。
    (void)EventRecord(evt, 1);
    (void)EventWait(evt, 1);
    (void)EventRecord(evt, 1);
    (void)EventWait(evt, 1);

    return CcuResult::CCU_SUCCESS;
}

// ---------------------------------------------------------------------------
// 场景1 kernel: 循环体外单对 EventRecord + EventWait (同一 event)。
// 翻译: record(setcke, 置位, 非写者) + wait(setcke, waitId=evt, clearType=1 自清=写者)。wait 虽是写者,
//       但其后无读同一 bit 的读者 -> 不构成写后读 -> 不补 NOP。set 与 wait 成对, 不死锁。
// ---------------------------------------------------------------------------
inline CcuResult CcuCkeSetClearPairHazardDemoKernel(CcuKernelArg arg)
{
    using namespace ccu;
    (void)arg;

    Event evtGuard; // 占位, 避免真正使用的 event 落到 CKE id 0 (id 0 会被依赖图兜底丢弃)
    (void)evtGuard;
    Event evt;

    // 单对 set/wait: record 置位(非写者), wait 读并自清(写者), 无后续读者 -> 不构成写后读。
    (void)EventRecord(evt, 1);
    (void)EventWait(evt, 1);

    return CcuResult::CCU_SUCCESS;
}
