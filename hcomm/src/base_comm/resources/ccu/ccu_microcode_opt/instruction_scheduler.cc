/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "instruction_scheduler.h"

#include <limits>
#include <unordered_map>

#include "ccu_log.h"

#include "config/barrier_config.h" // 提供 InstrCodeV2 opcode 常量

namespace hcomm {
namespace CcuOpt {

    namespace {

        inline uint32_t RegKey(const RegOperand& operand)
        {
            constexpr uint32_t REG_TYPE_SHIFT = 16;
            return (static_cast<uint32_t>(operand.type) << REG_TYPE_SHIFT) | static_cast<uint32_t>(operand.regId);
        }

        // ---------------- XN 写后读 (half-rtt LoadX/StoreX/ClearX) 归约与判定 ----------------

        // XN 写者/读者归约到统一 key 的编码域. XN regId 为 15 位 (< 0x8000), 用高位 flag 与
        // ClearX 区间 key 隔离, 保证三类 key (单 xn / array 组 / ClearX 区间) 互不碰撞:
        //   * 单 xn / array 组: 直接用归约后的 XN id (< 0x8000), 无 flag;
        //   * ClearX 区间:      kClearXRangeFlag | (lo << 15) | hi.
        constexpr uint32_t kClearXRangeFlag = 1u << 30;
        constexpr uint32_t kXnIdShift = 15;

        // 把任意 XN regId 归约成整组 key: 命中某 pinnedGroup [baseId, baseId+count) 则返回 baseId,
        // 否则原样返回 regId. 各组互不相交, 命中唯一.
        inline uint16_t ReduceXnToGroupKey(uint16_t regId, const std::vector<PinnedGroup>* pinnedGroups)
        {
            if (pinnedGroups == nullptr) {
                return regId;
            }
            for (const auto& g : *pinnedGroups) {
                if (g.count == 0) {
                    continue;
                }
                if (regId >= g.baseId && regId < static_cast<uint32_t>(g.baseId) + g.count) {
                    return g.baseId;
                }
            }
            return regId;
        }

        // 归约后的单 xn / array 组 key (无 flag).
        inline uint32_t XnGroupKey(uint16_t regId, const std::vector<PinnedGroup>* pinnedGroups)
        {
            return static_cast<uint32_t>(ReduceXnToGroupKey(regId, pinnedGroups));
        }

        // ClearX 区间规范化 key: 以较小端点为 lo, 较大端点为 hi, 与普通单 xn key 天然隔离,
        // 只有相同区间的两条 ClearX 才会命中同一 key.
        inline uint32_t ClearXRangeKey(uint16_t a, uint16_t b)
        {
            uint16_t lo = a <= b ? a : b;
            uint16_t hi = a <= b ? b : a;
            return kClearXRangeFlag | (static_cast<uint32_t>(lo) << kXnIdShift) | static_cast<uint32_t>(hi);
        }

        inline bool IsLoadX(const CcuRep::CcuInstr& instr)
        {
            return instr.header.type == InstrCodeV2::LOAD_TYPE && instr.header.code == InstrCodeV2::LOADX_CODE;
        }
        inline bool IsStoreX(const CcuRep::CcuInstr& instr)
        {
            return instr.header.type == InstrCodeV2::LOAD_TYPE && instr.header.code == InstrCodeV2::STOREX_CODE;
        }
        inline bool IsClearX(const CcuRep::CcuInstr& instr)
        {
            return instr.header.type == InstrCodeV2::LOAD_TYPE && instr.header.code == InstrCodeV2::CLEARX_CODE;
        }

        // 收集当前指令作为 XN 写者登记进写者表的 key. 新机制单向: 只有 LoadX 写的 i / StoreX 写的
        // array / ClearX 区间进表; 其余指令的 XN def (Add 等) 交硬件 interlock, 不进表.
        // 字段落位以 ccu_microcode_v2.cc 为准 (LoadX: xdId=i; StoreX: xdo=array; ClearX: [xnId,xmId]).
        void CollectXnWriterKeys(
            const CcuRep::CcuInstr& instr, const std::vector<PinnedGroup>* pinnedGroups, std::vector<uint32_t>& keys)
        {
            if (IsLoadX(instr)) {
                keys.push_back(XnGroupKey(instr.v2.loadStoreX.xdId, pinnedGroups));
            } else if (IsStoreX(instr)) {
                keys.push_back(XnGroupKey(instr.v2.loadStoreX.xdo, pinnedGroups));
            } else if (IsClearX(instr)) {
                keys.push_back(ClearXRangeKey(instr.v2.clearX.xnId, instr.v2.clearX.xmId));
            }
        }

        inline CcuRep::CcuInstr MakeNop()
        {
            // CcuInstr 为 POD (header + union, 无非平凡成员), {} 值初始化已将全部字节零化,
            // 无需再 memset_s; 仅设置 NOP 的 header 即可.
            CcuRep::CcuInstr nopInstr{};
            nopInstr.header = CcuRep::InstrHeader(InstrCodeV2::LOAD_TYPE, InstrCodeV2::NOP_CODE);
            return nopInstr;
        }

        // RelJmp 模板内部使用的固定常量 (与 ccu_microcode_v2.cc 同源, 见 RelJmp 生成器):
        //   两条内部 Jump 的固定跳距 (IF 分支跳 5 / NOP 分支跳 3), 及两种 conditionType.
        namespace RelJmpConst {
            constexpr uint16_t JMP_TARGET_PC_IF_BRANCH = 5;  // P+1 LoadImdToXn(xn1, 5)
            constexpr uint16_t JMP_TARGET_PC_NOP_BRANCH = 3; // P+5 LoadImdToXn(xn1, 3)
            constexpr uint16_t JMP_COND_TYPE_IF = 2;         // P+2 换算 Jump 的 conditionType
            constexpr uint16_t JMP_COND_TYPE_UNCOND = 6;     // P+6 无条件 Jump 的 conditionType
            constexpr int TEMPLATE_LEN = 9;                  // RelJmp 展开固定 9 条
        } // namespace RelJmpConst

        // RelJmp 模板解析结果. matched=true 时各下标字段有效.
        struct RelJmpMatch {
            bool matched = false;
            size_t p0 = 0;         // 块起始 (LoadImdToXn(xn0, jmpInstrId))
            size_t innerJmp = 0;   // P+2 换算 Jump
            size_t p3 = 0;         // P+3 LoadImdToXn(xn0, 0x10000 - jmpInstrId)
            uint16_t xn0 = 0;      // jmpInstrId 基准寄存器
            uint16_t xn1 = 0;      // 模板内固定跳距寄存器
            uint16_t targetXn = 0; // 目标绝对 id 寄存器 (被 Add/Sub 换算)
        };

        inline bool IsLoadImd(const CcuRep::CcuInstr& instr)
        {
            return instr.header.type == InstrCodeV2::LOAD_TYPE && instr.header.code == InstrCodeV2::LOADIMDTOX_CODE;
        }
        inline bool IsCtrlJmp(const CcuRep::CcuInstr& instr)
        {
            return instr.header.type == InstrCodeV2::CTRL_TYPE && instr.header.code == InstrCodeV2::JMP_CODE;
        }
        inline bool IsArith(const CcuRep::CcuInstr& instr, uint16_t code)
        {
            return instr.header.type == InstrCodeV2::LOAD_TYPE && instr.header.code == code;
        }

        // RelJmp 模板寄存器角色: 由 P+2 换算 Jump 解出的三个寄存器 id, 供逐条校验共享.
        struct RelJmpRegs {
            uint16_t xn0 = 0;      // jmpInstrId 基准寄存器
            uint16_t xn1 = 0;      // 模板内固定跳距寄存器
            uint16_t targetXn = 0; // 目标绝对 id 寄存器
        };

        // P+4 ADD / P+7 SUB 共享校验: 算术码 + xd/xn==targetXn + xm==xn0.
        inline bool IsRelJmpArith(const CcuRep::CcuInstr& instr, uint16_t code, const RelJmpRegs& regs)
        {
            return IsArith(instr, code) && instr.v2.operate.xdId == regs.targetXn
                   && instr.v2.operate.xnId == regs.targetXn && instr.v2.operate.xmId == regs.xn0;
        }

        // 校验 P+0..P+1 (P+2 前两条): P+1 LoadImdToXn(xn1, 5), P+0 LoadImdToXn(xn0, *).
        inline bool
        CheckRelJmpHead(const std::vector<CcuRep::CcuInstr>& vec, size_t innerJmpIdx, const RelJmpRegs& regs)
        {
            using namespace RelJmpConst;
            const auto& p1 = vec[innerJmpIdx - 1];
            const auto& p0 = vec[innerJmpIdx - 2];
            if (!IsLoadImd(p1) || p1.v2.loadImdToX.xnId != regs.xn1
                || p1.v2.loadImdToX.immediate != JMP_TARGET_PC_IF_BRANCH) {
                return false;
            }
            return IsLoadImd(p0) && p0.v2.loadImdToX.xnId == regs.xn0;
        }

        // 校验 P+3..P+7 (P+2 后五条): P+3 LoadImd(xn0), P+4 Add, P+5 LoadImd(xn1,3), P+6 UNCOND Jump, P+7 Sub.
        inline bool
        CheckRelJmpTail(const std::vector<CcuRep::CcuInstr>& vec, size_t innerJmpIdx, const RelJmpRegs& regs)
        {
            using namespace InstrCodeV2;
            using namespace RelJmpConst;
            const auto& p3 = vec[innerJmpIdx + 1];
            const auto& p4 = vec[innerJmpIdx + 2];
            const auto& p5 = vec[innerJmpIdx + 3];
            const auto& p6 = vec[innerJmpIdx + 4];
            const auto& p7 = vec[innerJmpIdx + 5];
            if (!IsLoadImd(p3) || p3.v2.loadImdToX.xnId != regs.xn0) {
                return false;
            }
            if (!IsRelJmpArith(p4, ADD_CODE, regs)) {
                return false;
            }
            if (!IsLoadImd(p5) || p5.v2.loadImdToX.xnId != regs.xn1
                || p5.v2.loadImdToX.immediate != JMP_TARGET_PC_NOP_BRANCH) {
                return false;
            }
            if (!IsCtrlJmp(p6) || p6.v2.jmp.conditionType != JMP_COND_TYPE_UNCOND
                || p6.v2.jmp.relTarInstrXnId != regs.xn1) {
                return false;
            }
            return IsRelJmpArith(p7, SUB_CODE, regs);
        }

        // 强指纹校验: 以 innerJmpIdx (候选 P+2 换算 Jump) 为锚, 校验其所在的 9 条是否构成完整 RelJmp
        // 模板. 校验点 (任一不满足即判否), 互锁性极强, 普通条件跳转不可能全中:
        //   P+2 vec[i]      : CTRL/JMP, conditionType==IF(2); 取 xn1=relTar, targetXn=condition, xn0=expected
        //   P+1 vec[i-1]    : LOADIMDTOX, xnId==xn1, immediate==5 (IF 分支固定跳距)
        //   P+0 vec[i-2]    : LOADIMDTOX, xnId==xn0
        //   P+3 vec[i+1]    : LOADIMDTOX, xnId==xn0 (immediate 应为 0x10000 - P+0.immediate)
        //   P+4 vec[i+2]    : ADD, xdId==targetXn, xnId==targetXn, xmId==xn0
        //   P+5 vec[i+3]    : LOADIMDTOX, xnId==xn1, immediate==3 (NOP 分支固定跳距)
        //   P+6 vec[i+4]    : CTRL/JMP, conditionType==UNCOND(6), relTar==xn1
        //   P+7 vec[i+5]    : SUB, xdId==targetXn, xnId==targetXn, xmId==xn0
        // 普通条件跳转 (EQ/NE/GT/... + 用户 expected/condition) 因固定跳距 5/3、成对 Add/Sub、配套第二跳
        // 缺一即被排除, 从根本上杜绝把 "expected 比较值 load" 误判为 "jmpInstrId 基准 load".
        RelJmpMatch MatchRelJmpTemplate(const std::vector<CcuRep::CcuInstr>& vec, size_t innerJmpIdx)
        {
            using namespace RelJmpConst;
            RelJmpMatch match;

            // 边界: innerJmp 至少是 P+2, 其后还需 P+3..P+7 (5 条).
            if (innerJmpIdx < 2 || innerJmpIdx + 5 >= vec.size()) {
                return match;
            }
            const auto& jmp = vec[innerJmpIdx];
            if (!IsCtrlJmp(jmp) || jmp.v2.jmp.conditionType != JMP_COND_TYPE_IF) {
                return match;
            }
            const RelJmpRegs regs{jmp.v2.jmp.expectedXnId, jmp.v2.jmp.relTarInstrXnId, jmp.v2.jmp.conditionXnId};

            if (!CheckRelJmpHead(vec, innerJmpIdx, regs) || !CheckRelJmpTail(vec, innerJmpIdx, regs)) {
                return match;
            }

            match.matched = true;
            constexpr int REL_JMP_P0_OFFSET = 2; // P0 = innerJmp - 2 (P+2 为换算 Jump)
            match.p0 = innerJmpIdx - REL_JMP_P0_OFFSET;
            match.innerJmp = innerJmpIdx;
            match.p3 = innerJmpIdx + 1;
            match.xn0 = regs.xn0;
            match.xn1 = regs.xn1;
            match.targetXn = regs.targetXn;
            return match;
        }

        // 识别 RelJmp 原子块 (func-call / func-ret 运行期地址跳转), 返回逐指令保护掩码.
        // 用 MatchRelJmpTemplate 强指纹匹配: 命中的块为 [P+0, P+8] 共 9 条, 全部标记为块内禁止插 NOP
        // (块内固定跳距 5/3 与 P+2 相对 P+0 的换算关系依赖块内不被撕裂). 目标区间的距离修正由
        // FixRelJmpFunc 在 FixReferences 阶段完成, 不在此保护.
        std::vector<bool> MarkRelJmpProtectedRanges(const std::vector<CcuRep::CcuInstr>& vec)
        {
            using namespace RelJmpConst;
            const size_t count = vec.size();
            std::vector<bool> mask(count, false);
            for (size_t i = 0; i < count; ++i) {
                if (!IsCtrlJmp(vec[i])) {
                    continue;
                }
                RelJmpMatch match = MatchRelJmpTemplate(vec, i);
                if (!match.matched) {
                    continue;
                }
                // 块 = [P+0, P+8] = [match.p0, match.p0 + 8].
                const size_t blockEnd = match.p0 + static_cast<size_t>(TEMPLATE_LEN) - 1;
                for (size_t k = match.p0; k <= blockEnd && k < count; ++k) {
                    mask[k] = true;
                }
            }
            return mask;
        }

        // CkeOnly 顺序调度的可变状态: 输出序列、原->输出映射、统计信息, 以及只跟踪 CKE 写者发射
        // cycle 的表. 不做启发式放大, 保证补 NOP 有界.
        struct CkeOnlyState {
            std::vector<CcuRep::CcuInstr>& outVec;
            std::vector<int32_t>& origToOut;
            SchedulerStats& stats;
            const std::vector<bool>& relJmpProtected; // 逐指令保护掩码: true 表示属于 RelJmp 原子块, 块内禁止插 NOP.
            const std::vector<PinnedGroup>* pinnedGroups = nullptr; // XN array 归约用 pinned 组; 只读引用.
            std::unordered_map<uint32_t, int64_t> lastCkeWriterCycle{};
            std::unordered_map<uint32_t, int64_t> lastXnWriterCycle{}; // XN 写后读写者表, 与 CKE 表并行.
            // 通用 XN 写者表 (任意指令的 XN def 归约后进表): 只服务于 "任意指令写 xn/array -> 后续
            // LoadX/StoreX/ClearX 读该 xn/array" 这一方向. 硬件对 LoadX/StoreX/ClearX 读操作数的 interlock
            // 会失效, 故其读若命中前序写者且距离 < latency, 后端须补 NOP. 与 lastXnWriterCycle (只登记
            // lsx 写者、服务任意读者) 并存互补, 两方向各自查各自的表.
            std::unordered_map<uint32_t, int64_t> lastAnyXnWriterCycle{};
            int64_t cycle = 0;
        };

        inline void EmitNop(CkeOnlyState& state)
        {
            state.outVec.push_back(MakeNop());
            state.stats.originIndex.push_back(-1); // 无对应源.
            state.stats.nopInserted++;
            state.cycle++;
        }

        // 计算当前指令为满足 CKE 写后读 latency 所需的最早发射 cycle.
        inline int64_t EarliestCkeIssueCycle(const CkeOnlyState& state, const std::vector<RegOperand>& operands)
        {
            const int64_t ckeLatency = static_cast<int64_t>(CcuRep::CCU_CKE_RAW_LATENCY);
            int64_t earliest = state.cycle;
            for (const auto& operand : operands) {
                if (operand.isDef || operand.type != RegType::CKE) {
                    continue;
                }
                auto it = state.lastCkeWriterCycle.find(RegKey(operand));
                if (it == state.lastCkeWriterCycle.end()) {
                    continue;
                }
                int64_t needed = it->second + ckeLatency;
                if (needed > earliest) {
                    earliest = needed;
                }
            }
            return earliest;
        }

        // 计算当前指令为满足 XN 写后读 (half-rtt LoadX/StoreX/ClearX) latency 所需的最早发射 cycle.
        // 读者是任意指令的 XN use, 经 pinnedGroup 归约成 array/单 xn key 后查 XN 写者表; ClearX 额外
        // 以其规范化区间 key 作为读者查表 (同片自依赖). 命中写者则 needed = writerCycle + L.
        // 返回值语义: cycle 是发射节拍计数, 恒 >= 0. 初值取 state.cycle (>=0), 后续只与
        // writerCycle + latency (两者均非负) 取 max, 故结果永不为负. 之所以用有符号 int64_t 而非
        // uint: (1) 与 state.cycle / writerCycle 存储类型一致, 全程 64 位避免与 latency 运算时的
        // 隐式截断/无符号回绕; (2) 有符号便于表达 "无需求" 等潜在哨兵语义并让越界更易在调试期暴露.
        inline int64_t EarliestXnIssueCycle(
            const CkeOnlyState& state, const CcuRep::CcuInstr& instr, const std::vector<RegOperand>& operands)
        {
            const int64_t xnLatency = static_cast<int64_t>(CcuRep::CCU_XN_RAW_LATENCY);
            int64_t earliest = state.cycle;

            auto probe = [&](uint32_t key) {
                auto it = state.lastXnWriterCycle.find(key);
                if (it == state.lastXnWriterCycle.end()) {
                    return;
                }
                int64_t needed = it->second + xnLatency;
                if (needed > earliest) {
                    earliest = needed;
                }
            };

            // 任意指令的 XN use 作为读者归约后查表.
            for (const auto& operand : operands) {
                if (operand.isDef || operand.type != RegType::XN) {
                    continue;
                }
                probe(XnGroupKey(operand.regId, state.pinnedGroups));
            }
            // ClearX 同片自依赖: 用区间 key 再查一次.
            if (IsClearX(instr)) {
                probe(ClearXRangeKey(instr.v2.clearX.xnId, instr.v2.clearX.xmId));
            }
            return earliest;
        }

        // 计算当前指令为满足 "前序任意指令写 xn/array -> LoadX/StoreX/ClearX 读该 xn/array" 写后读
        // latency 所需的最早发射 cycle. 背景: 硬件 interlock 对 LoadX/StoreX/ClearX 的读操作数失效,
        // 故只有当前指令本身是 lsx/clearx 时, 才需要把它读的 xn/array 与前序任意写者 (通用写者表
        // lastAnyXnWriterCycle) 做写后读检查; 其余普通指令的读仍由硬件 interlock 兜底, 不查此表.
        // 读者操作数: lsx/clearx 提取出的所有 XN use (经 pinnedGroup 归约成 array/单 xn key), 以及
        // ClearX 的区间 key. 命中写者则 needed = writerCycle + L.
        inline int64_t EarliestXnReadByLsxCycle(
            const CkeOnlyState& state, const CcuRep::CcuInstr& instr, const std::vector<RegOperand>& operands)
        {
            // 只有读者本身是 LoadX/StoreX/ClearX 时其读才失去硬件 interlock, 需要后端补 NOP.
            if (!IsLoadX(instr) && !IsStoreX(instr) && !IsClearX(instr)) {
                return state.cycle;
            }
            const int64_t xnLatency = static_cast<int64_t>(CcuRep::CCU_XN_RAW_LATENCY);
            int64_t earliest = state.cycle;

            auto probe = [&](uint32_t key) {
                auto it = state.lastAnyXnWriterCycle.find(key);
                if (it == state.lastAnyXnWriterCycle.end()) {
                    return;
                }
                int64_t needed = it->second + xnLatency;
                if (needed > earliest) {
                    earliest = needed;
                }
            };

            // lsx/clearx 的所有 XN use 作为读者归约后查通用写者表.
            for (const auto& operand : operands) {
                if (operand.isDef || operand.type != RegType::XN) {
                    continue;
                }
                probe(XnGroupKey(operand.regId, state.pinnedGroups));
            }
            // ClearX 区间既读又写: 用规范化区间 key 再查一次前序区间写者.
            if (IsClearX(instr)) {
                probe(ClearXRangeKey(instr.v2.clearX.xnId, instr.v2.clearX.xmId));
            }
            return earliest;
        }

        // 处理单条指令: 先补齐 latency NOP, 再原序发射, 最后记录 CKE 写者的发射 cycle.
        void ScheduleOneCkeInstr(CkeOnlyState& state, const CcuRep::CcuInstr& instr, size_t originIdx)
        {
            auto operands = ExtractOperandsV2(instr);

            // RelJmp 原子块内禁止插 NOP: 块内指令(Load/Jump/Add/Sub/Nop)不产生 CKE 写者, 也不含 CKE 读者,
            // earliestIssueCycle 恒等于当前 cycle, 正常路径本就不会补 NOP; 这里显式跳过补 NOP 是防御, 确保
            // 即使块紧邻的 CKE 写者仍有残余 latency 需求, 也不会把 NOP 插进/插到块中间破坏运行期地址链.
            const bool isProtected = originIdx < state.relJmpProtected.size() && state.relJmpProtected[originIdx];
            if (!isProtected) {
                // CKE 与 XN 多路写后读需求并行计算, 取最大者决定最早发射 cycle; 各路结构对称:
                //   * EarliestCkeIssueCycle    : CKE 写后读 (setcke/clearcke -> waitcke/clearcke);
                //   * EarliestXnIssueCycle     : lsx 写 xn/array -> 任意后续指令读 (原方向);
                //   * EarliestXnReadByLsxCycle : 任意前序指令写 xn/array -> lsx/clearx 读 (新方向,
                //                                硬件对 lsx/clearx 读操作数 interlock 失效).
                int64_t earliestIssueCycle = EarliestCkeIssueCycle(state, operands);
                int64_t earliestXnCycle = EarliestXnIssueCycle(state, instr, operands);
                if (earliestXnCycle > earliestIssueCycle) {
                    earliestIssueCycle = earliestXnCycle;
                }
                int64_t earliestXnReadCycle = EarliestXnReadByLsxCycle(state, instr, operands);
                if (earliestXnReadCycle > earliestIssueCycle) {
                    earliestIssueCycle = earliestXnReadCycle;
                }
                while (state.cycle < earliestIssueCycle) {
                    EmitNop(state);
                }
            }

            state.origToOut[originIdx] = static_cast<int32_t>(state.outVec.size());
            state.outVec.push_back(instr);
            state.stats.originIndex.push_back(static_cast<int32_t>(originIdx)); // CkeOnly 顺序保持.
            int64_t issueCycle = state.cycle;
            state.cycle++;

            // 登记 CKE 写者 (现有): clearcke 自动清零的 waitCKEId.
            for (const auto& operand : operands) {
                if (!operand.isDef || operand.type != RegType::CKE) {
                    continue;
                }
                state.lastCkeWriterCycle[RegKey(operand)] = issueCycle;
            }
            // 登记 lsx 写者 (原方向, 单向): 仅 LoadX 写 i / StoreX 写 array / ClearX 区间进表,
            // 服务 "lsx 写 -> 任意后续指令读" 的 EarliestXnIssueCycle.
            std::vector<uint32_t> xnWriterKeys;
            CollectXnWriterKeys(instr, state.pinnedGroups, xnWriterKeys);
            for (uint32_t key : xnWriterKeys) {
                state.lastXnWriterCycle[key] = issueCycle;
            }
            // 登记通用 XN 写者 (新方向): 任意指令的每个 XN def 归约后进通用表, 服务 "任意前序写 ->
            // 后续 lsx/clearx 读" 的 EarliestXnReadByLsxCycle. 含 lsx/clearx 自身写的目标 (LoadX 写 i /
            // StoreX 写 array / ClearX 区间两端点), 使 lsx 之间的写读也被覆盖. ClearX 额外登记区间 key,
            // 供后续同区间 ClearX 的读命中 (与自依赖判定共用区间 key 域).
            for (const auto& operand : operands) {
                if (!operand.isDef || operand.type != RegType::XN) {
                    continue;
                }
                state.lastAnyXnWriterCycle[XnGroupKey(operand.regId, state.pinnedGroups)] = issueCycle;
            }
            if (IsClearX(instr)) {
                state.lastAnyXnWriterCycle[ClearXRangeKey(instr.v2.clearX.xnId, instr.v2.clearX.xmId)] = issueCycle;
            }
        }

        // 依据已确定的 out.missionStartInstrId 重新推导 missionInstrCount:
        // mission 起点落在输出序列内则取到序列尾部的长度, 否则计 0.
        inline void RecomputeMissionCount(CcuRep::CcuInstrInfo& out, uint16_t startId)
        {
            if (static_cast<uint32_t>(out.missionStartInstrId)
                < static_cast<uint32_t>(startId) + static_cast<uint32_t>(out.instrCount)) {
                out.missionInstrCount = static_cast<uint16_t>(startId + out.instrCount - out.missionStartInstrId);
            } else {
                out.missionInstrCount = 0;
            }
        }

        // 与 GetRelativeInstrId / jump_executor 的回绕空间一致. 用 uint64_t 承载, 使所有涉及 immediate
        // (字段本身为 uint64_t, 可存完整 64 位业务立即数) 的读取/运算全程 64 位, 杜绝中间隐式截断.
        constexpr uint64_t kInstrIdSpace = 0x10000ULL;

        // 普通相对跳转 offset 修正.
        // 背景: v2 的 jmp 目标不是直接写在 jmp 指令里的绝对 instrId, 而是"相对距离":
        // 生成端在紧邻 jmp 之前用一条 LoadImdToXn 把 offset = target - jmpPC 加载进 relTarInstrXnId,
        // 硬件按 nextInsIdx = jmpPC + offset (mod 0x10000) 跳转 (见 jump_executor.cc 相对跳转分支).
        // 因此只要在 jmp 与目标之间净插入了 k 条 NOP, 真实相对距离就变了, 而 offset 立即数是编译期
        // 写死的, 不修正会跳错. Loop/LoopGroup 用绝对 id 靠 remapGlobal 平移, jmp 则必须改写 offset.
        // 向前找最近一条写 tgtXn 的指令: 命中 LoadImdToXn 返回其原始下标; 命中算术等其它写者则告警并
        // 返回 -1 (放弃修正); 找不到任何写者也返回 -1 (保守不动).
        int32_t FindOffsetLoaderOrigIdx(const std::vector<CcuRep::CcuInstr>& origVec, size_t jmpOrigIdx, uint16_t tgtXn)
        {
            using namespace InstrCodeV2;
            for (int32_t j = static_cast<int32_t>(jmpOrigIdx) - 1; j >= 0; --j) {
                const auto& cur = origVec[j];
                if (cur.header.type == LOAD_TYPE && cur.header.code == LOADIMDTOX_CODE
                    && cur.v2.loadImdToX.xnId == tgtXn) {
                    return j;
                }
                for (const auto& op : ExtractOperandsV2(cur)) {
                    if (op.isDef && op.type == RegType::XN && op.regId == tgtXn) {
                        HCCL_WARNING(
                            "[InstructionScheduler] jmp@%zu target reg X%u overwritten by non-imm instr@%d; "
                            "skip relative-offset fix.",
                            jmpOrigIdx, static_cast<unsigned>(tgtXn), j);
                        return -1;
                    }
                }
            }
            return -1; // 没找到立即数来源, 保守不动.
        }

        void FixPlainJumpOffset(
            const std::vector<CcuRep::CcuInstr>& origVec, const std::vector<int32_t>& origToOut, size_t jmpOrigIdx,
            std::vector<CcuRep::CcuInstr>& outVec)
        {
            const uint16_t tgtXn = origVec[jmpOrigIdx].v2.jmp.relTarInstrXnId;

            const int32_t loadOrigIdx = FindOffsetLoaderOrigIdx(origVec, jmpOrigIdx, tgtXn);
            if (loadOrigIdx < 0) {
                return;
            }

            const int32_t loadOutPos = origToOut[loadOrigIdx];
            const int32_t jmpOutPos = origToOut[jmpOrigIdx];
            if (loadOutPos < 0 || jmpOutPos < 0) {
                return; // 理论上二者都保留 (CkeOnly 不删指令), 兜底防御.
            }

            const uint64_t oldOffset = outVec[loadOutPos].v2.loadImdToX.immediate;
            // 定位护栏 (非落点校验): offset 语义上是相对距离, 合法域即 [0, 0x10000). 向前扫描找 offset
            // loader 是启发式的, 若读到 >= 0x10000, 说明大概率误命中了往同一寄存器装 64 位业务数据的
            // load (而非真正的 offset loader). 此时改写会截断高位破坏业务数据 —— 放弃修正并告警.
            // 注: 这里判的是"偏移是否超出该字段合法域", 落点是否合法由下方 origTargetIdx 越界检查负责.
            if (oldOffset >= kInstrIdSpace) {
                HCCL_WARNING(
                    "[InstructionScheduler] jmp@%zu offset-loader immediate %llu out of relative-offset range "
                    "[0,0x10000); likely mismatched a 64-bit value load, skip fix to avoid truncation.",
                    jmpOrigIdx, static_cast<unsigned long long>(oldOffset));
                return;
            }
            // 旧 offset 基准是 jmp 原始位置, 反推原始目标下标 (回绕). 全程 uint64_t.
            const uint64_t origTargetIdx = (static_cast<uint64_t>(jmpOrigIdx) + oldOffset) % kInstrIdSpace;
            if (origTargetIdx >= origToOut.size() || origToOut[origTargetIdx] < 0) {
                HCCL_WARNING(
                    "[InstructionScheduler] jmp@%zu old offset %llu points outside sequence (target idx %llu); "
                    "skip relative-offset fix.",
                    jmpOrigIdx, static_cast<unsigned long long>(oldOffset),
                    static_cast<unsigned long long>(origTargetIdx));
                return;
            }

            const uint64_t newTargetPos = static_cast<uint64_t>(origToOut[origTargetIdx]);
            const uint64_t newJmpPos = static_cast<uint64_t>(jmpOutPos);
            const uint64_t newOffset = (newTargetPos + kInstrIdSpace - newJmpPos) % kInstrIdSpace;
            outVec[loadOutPos].v2.loadImdToX.immediate = newOffset;
        }

        // RelJmp (func-call / func-ret 运行期地址跳转) 修正.
        //
        // RelJmp 用绝对量表达跳转: 运行时寄存器换算值 = targetAbsId - jmpInstrId, 主 Jump (紧随 9 条模板
        // 之后) 按 nextPC = 主JumpPC + 换算值 跳转; 生成端令 jmpInstrId == 主JumpPC, 故最终 nextPC ==
        // targetAbsId. 插 NOP 后主 Jump 与目标各自位移, 二者相对距离改变, 必须把两个绝对量分别重映射:
        //   * jmpInstrId (RelJmp 模板 P+0 LoadImdToXn(xn0, jmpInstrId) 与 P+3 LoadImdToXn(xn0,
        //     0x10000 - jmpInstrId)): jmpInstrId == 主 Jump 旧 PC, 平移到主 Jump 新 PC (= remap(jmpInstrId)).
        //   * 目标绝对 id (由加载 targetXn 的 LoadImdToXn 提供): remap 到新位置; 若目标由 Add 加载
        //     (funcAddrVar 运行期变量, 外部绝对地址), 不受本段插 NOP 影响, 不动.
        // 块内 9 条由 MarkRelJmpProtectedRanges 保证不被 NOP 撕裂, 故模板内固定跳距 (5 / 3) 无需修改.
        //
        // 定位方式: P+0 / P+3 由已通过强指纹校验的 RelJmpMatch 按模板固定偏移直接给出 (match.p0 / match.p3),
        // 不再靠"解释 jmp 的 expectedXnId 字段 + 向前扫描"猜测 —— 后者在普通条件跳转上会误命中 expected
        // 比较值 load. 只有整段构成 RelJmp 模板才会走到这里, 故 match.p0 / match.p3 必为 jmpInstrId 基准 load.
        //
        // RelJmp 目标绝对 id 的 remap 改写: 命中"编译期常量绝对 id"的 LoadImdToXn 后, 越界告警 / 否则 remap 平移.
        // 从 FixRelJmpFunc 拆出, 消除 for -> if -> if -> if/else -> 赋值 的过深嵌套 (超大深度函数告警).
        //
        // 硬件约束: jmp 落点绝对 PC 必须落在指令空间 [0, 0x10000). 这里 rawTarget 是"目标绝对 instrId"
        // (由生成端 LoadImdToXn(targetXn, funcBlock->StartInstrId()/funcRet.Id()) 装入编译期常量绝对 id;
        // 运行期偏移由模板 P+4 Add(0x10000 - jmpInstrId) 现算, 不落在本立即数里). 故越界校验分两处:
        //   * 旧绝对 id (rawTarget): 强指纹已保证 < 0x10000, 越界说明数据异常, 放弃改写;
        //   * remap 后新绝对 id (newTarget): 插 NOP 后 PC 整体后移, 显式校验新落点仍在指令空间内.
        template <typename RemapFn>
        void RemapRelJmpTargetImmediate(
            const RelJmpMatch& match, int32_t outPos, std::vector<CcuRep::CcuInstr>& outVec, RemapFn remapId)
        {
            const uint64_t rawTarget = outVec[outPos].v2.loadImdToX.immediate;
            if (rawTarget >= kInstrIdSpace) {
                HCCL_WARNING(
                    "[InstructionScheduler] RelJmp@%zu: target absolute id %llu exceeds instrId space; "
                    "skip target remap.",
                    match.innerJmp, static_cast<unsigned long long>(rawTarget));
                return;
            }
            const uint64_t newTarget = static_cast<uint64_t>(remapId(static_cast<uint16_t>(rawTarget)));
            if (newTarget >= kInstrIdSpace) {
                HCCL_WARNING(
                    "[InstructionScheduler] RelJmp@%zu: remapped target absolute id %llu exceeds instrId space "
                    "after NOP insertion; skip target remap.",
                    match.innerJmp, static_cast<unsigned long long>(newTarget));
                return;
            }
            outVec[outPos].v2.loadImdToX.immediate = newTarget;
        }

        // RelJmp (func-call / func-ret 运行期地址跳转) 专用修正. 与普通相对跳转 (FixPlainJumpOffset)
        // 区分命名: 本函数处理的 P+0/target 立即数是"编译期常量绝对 instrId", 运行期相对偏移由模板
        // Add/Sub 现算; 普通相对跳转处理的是直接写死的相对偏移立即数.
        //
        // remapId: 把"旧全局 instrId"映射到"新全局 instrId"(与 FixReferences 的 remapGlobal 同语义).
        template <typename RemapFn>
        void FixRelJmpFunc(
            const std::vector<CcuRep::CcuInstr>& origVec, const std::vector<int32_t>& origToOut,
            const RelJmpMatch& match, std::vector<CcuRep::CcuInstr>& outVec, RemapFn remapId)
        {
            using namespace InstrCodeV2;

            // 1) 修 jmpInstrId 基准: P+0 (match.p0, 立即数 jmpInstrId) 与 P+3 (match.p3, 立即数 0x10000 - jmpInstrId).
            const int32_t p0Out = origToOut[match.p0];
            const int32_t p3Out = origToOut[match.p3];
            if (p0Out < 0 || p3Out < 0) {
                return; // 块内不删指令, 兜底防御.
            }
            const uint64_t rawJmpInstrId = outVec[p0Out].v2.loadImdToX.immediate;
            if (rawJmpInstrId >= kInstrIdSpace) {
                // jmpInstrId 是主 Jump 的绝对 instrId (< 0x10000). 强指纹已确保这是 RelJmp, 正常不会越界;
                // 越界则说明数据异常, 放弃并告警, 不做可能截断的改写.
                HCCL_WARNING(
                    "[InstructionScheduler] RelJmp@%zu: base absolute id %llu exceeds instrId space; skip fix.",
                    match.innerJmp, static_cast<unsigned long long>(rawJmpInstrId));
                return;
            }
            const uint64_t newJmpInstrId = static_cast<uint64_t>(remapId(static_cast<uint16_t>(rawJmpInstrId)));
            // 硬件约束: 主 Jump 落点绝对 PC 必须落在指令空间内. 插 NOP 后 PC 整体后移, 显式校验新绝对
            // id 仍 < 0x10000, 越界则放弃改写 (避免写出会被硬件回绕到错误 PC 的基准值).
            if (newJmpInstrId >= kInstrIdSpace) {
                HCCL_WARNING(
                    "[InstructionScheduler] RelJmp@%zu: remapped base absolute id %llu exceeds instrId space "
                    "after NOP insertion; skip fix.",
                    match.innerJmp, static_cast<unsigned long long>(newJmpInstrId));
                return;
            }
            outVec[p0Out].v2.loadImdToX.immediate = newJmpInstrId;
            outVec[p3Out].v2.loadImdToX.immediate = kInstrIdSpace - newJmpInstrId;

            // 2) 修目标绝对 id: 向前找加载 targetXn 的指令 (在块之前, 位置不固定, 但 targetXn 由强指纹给出).
            //    LoadImdToXn(xnId==targetXn) -> 目标是编译期常量绝对 id, remap 平移;
            //    Add/Sub(xdId==targetXn)     -> 目标是运行期变量 (funcAddrVar, 外部绝对地址), 不动.
            for (int32_t j = static_cast<int32_t>(match.p0) - 1; j >= 0; --j) {
                const auto& cur = origVec[j];
                if (cur.header.type == LOAD_TYPE && cur.header.code == LOADIMDTOX_CODE
                    && cur.v2.loadImdToX.xnId == match.targetXn) {
                    if (origToOut[j] >= 0) {
                        RemapRelJmpTargetImmediate(match, origToOut[j], outVec, remapId);
                    }
                    break;
                }
                if (cur.header.type == LOAD_TYPE && (cur.header.code == ADD_CODE || cur.header.code == SUB_CODE)
                    && cur.v2.operate.xdId == match.targetXn) {
                    // 目标为运行期变量 (外部绝对地址), 不随本段插 NOP 变化, 无需修正.
                    break;
                }
            }
        }

        // 处理单条 JMP 指令的引用修正 (从 FixReferences 主循环拆出, 降低单函数体量与圈复杂度).
        template <typename RemapFn>
        void FixOneJumpReference(
            const CcuRep::CcuInstrInfo& input, const std::vector<int32_t>& origToOut,
            const std::vector<bool>& relJmpProtected, size_t originIdx, CcuRep::CcuInstrInfo& out, RemapFn remapGlobal)
        {
            const auto& origVec = input.instrVec;
            const auto& origInstr = origVec[originIdx];
            RelJmpMatch relJmp = MatchRelJmpTemplate(origVec, originIdx);
            if (origInstr.v2.jmp.jumpMode != 0) {
                // 绝对跳转 (jumpMode == 1): 目标是绝对 instrId, 需按绝对 id 重映射, 与相对跳转不同.
                // 当前生成端从不产生绝对跳转 (CcuV2::Jump 恒留 jumpMode=0), 不做猜测性改写, 显式告警.
                HCCL_ERROR(
                    "[InstructionScheduler] jmp@%zu is absolute (jumpMode=1); unsupported, jump target may be "
                    "wrong after NOP insertion.",
                    originIdx);
            } else if (relJmp.matched) {
                // RelJmp 换算 Jump (P+2, 强指纹命中): 目标用绝对量表达, 按模板固定偏移修 jmpInstrId
                // 基准 (P+0/P+3) + 目标绝对 id, 不触碰任何 expected/condition 业务 load.
                FixRelJmpFunc(origVec, origToOut, relJmp, out.instrVec, remapGlobal);
            } else if (originIdx < relJmpProtected.size() && relJmpProtected[originIdx]) {
                // RelJmp 模板内的其它 jmp (P+6 无条件跳): 跳距是模板内固定常量, 块内不插 NOP, 不改.
            } else {
                // 普通相对跳转: 改写其前置 LoadImdToXn 的 offset 立即数, 而非 jmp 指令本身.
                FixPlainJumpOffset(origVec, origToOut, originIdx, out.instrVec);
            }
        }

        // 顺序扫描后处理: 修正 missionStartInstrId / missionInstrCount 与 Loop / LoopGroup 引用.
        // CkeOnly 只在原序上插入 NOP, 不重排, 故按 origToOut 平移引用即可.
        void FixReferences(
            const CcuRep::CcuInstrInfo& input, const std::vector<int32_t>& origToOut,
            const std::vector<bool>& relJmpProtected, CcuRep::CcuInstrInfo& out)
        {
            using namespace InstrCodeV2;
            const auto& origVec = input.instrVec;
            const size_t instrCount = origVec.size();
            const uint16_t startId = input.startInstrId;

            auto remapGlobal = [&startId, &origToOut](uint16_t globalId) -> uint16_t {
                // globalId 是"startId + localId" 编码的全局 id, 越界或未映射则原样返回.
                if (globalId < startId)
                    return globalId;
                uint32_t localId = static_cast<uint32_t>(globalId) - static_cast<uint32_t>(startId);
                if (localId >= origToOut.size())
                    return globalId;
                int32_t newPos = origToOut[localId];
                if (newPos < 0)
                    return globalId;
                return static_cast<uint16_t>(startId + newPos);
            };

            // missionStartInstrId 修正: 若 mission 起点原本在本序列范围内, 映射到新的位置;
            // missionInstrCount 用序列尾部长度重新推导.
            if (input.missionStartInstrId >= startId
                && static_cast<uint32_t>(input.missionStartInstrId) < static_cast<uint32_t>(startId) + instrCount) {
                out.missionStartInstrId = remapGlobal(input.missionStartInstrId);
                RecomputeMissionCount(out, startId);
            } else {
                out.missionStartInstrId = input.missionStartInstrId;
                out.missionInstrCount = input.missionInstrCount;
            }

            for (size_t originIdx = 0; originIdx < instrCount; ++originIdx) {
                const auto& origInstr = origVec[originIdx];
                int32_t outPos = origToOut[originIdx];
                if (outPos < 0)
                    continue;
                auto& outInstr = out.instrVec[outPos];
                if (origInstr.header.type == CTRL_TYPE && origInstr.header.code == LOOP_CODE) {
                    outInstr.v2.loop.startInstrId = remapGlobal(origInstr.v2.loop.startInstrId);
                    outInstr.v2.loop.endInstrId = remapGlobal(origInstr.v2.loop.endInstrId);
                } else if (origInstr.header.type == CTRL_TYPE && origInstr.header.code == LOOPGROUP_CODE) {
                    outInstr.v2.loopGroup.startLoopInstrId = remapGlobal(origInstr.v2.loopGroup.startLoopInstrId);
                } else if (origInstr.header.type == CTRL_TYPE && origInstr.header.code == JMP_CODE) {
                    FixOneJumpReference(input, origToOut, relJmpProtected, originIdx, out, remapGlobal);
                }
            }
        }

    } // namespace

    CcuRep::CcuInstrInfo InstructionScheduler::Schedule(const CcuRep::CcuInstrInfo& input)
    {
        return ScheduleCkeOnly(input);
    }

    // CkeOnly 默认档: 保持原序, 只对 CKE 寄存器的写后读 (某条 setcke 写 CKE, 之后 waitcke/
    // clearcke 读同一 CKE) 按固定 cke latency 补 NOP; XN / MS 写后读交由硬件 interlock, 不补
    // 任何 NOP. 每个 CKE 读者最多补 (L-1) 条 NOP, 与"每个 wait 类 rep 预留 L 条"精确对齐.
    CcuRep::CcuInstrInfo InstructionScheduler::ScheduleCkeOnly(const CcuRep::CcuInstrInfo& input)
    {
        stats_ = {};
        const auto& origVec = input.instrVec;
        const size_t instrCount = origVec.size();

        std::vector<CcuRep::CcuInstr> outVec;
        std::vector<int32_t> origToOut;
        outVec.reserve(instrCount);
        origToOut.assign(instrCount, -1);

        // 预扫描识别 RelJmp 原子块 (func-call / func-ret 运行期地址跳转), 块内禁止插 NOP.
        const std::vector<bool> relJmpProtected = MarkRelJmpProtectedRanges(origVec);

        // 只跟踪 CKE 写者的发射 cycle; 不做启发式放大, 保证补 NOP 有界.
        // 索引用 size_t 与 vector::size() 对齐, 避免 instrCount 逼近 65535 时 uint16_t 回绕死循环;
        // 输出条数是否越界预留区由上游 TransRepSequenceToMicrocode 按 instrVec.size() 快速失败兜底.
        CkeOnlyState state{outVec, origToOut, stats_, relJmpProtected, opts_.pinnedGroups};
        for (size_t i = 0; i < instrCount; ++i) {
            ScheduleOneCkeInstr(state, origVec[i], i);
        }

        // CkeOnly 不做 BB 切分, 用 1 作为占位 (仅统计意义).
        stats_.basicBlocks = instrCount > 0 ? 1 : 0;

        CcuRep::CcuInstrInfo out;
        out.instrVec = std::move(outVec);
        out.startInstrId = input.startInstrId;

        // instrCount 字段为 uint16_t. 正常路径下上游按 CKE 预留区申请, 优化后条数远小于 65535;
        // 但一旦补 NOP 后输出条数超过 uint16_t 上限, 直接截断会让 instrCount 与真实 instrVec 大小
        // 不一致, 进而使 FixReferences 的引用重映射错位. 此处显式记录错误再截断, 把静默数据损坏
        // 变成可观测告警; instrVec 保留完整大小, 由上游 TransRepSequenceToMicrocode 按
        // instrVec.size() > regionSize 快速失败兜底 (见 ccu_kernel_mgr.cc).
        constexpr size_t kMaxInstrCount = std::numeric_limits<uint16_t>::max();
        if (out.instrVec.size() > kMaxInstrCount) {
            HCCL_ERROR(
                "[InstructionScheduler] optimized instr count[%zu] exceeds uint16_t range[%zu]; "
                "instrCount field will be truncated, upstream region-size check will reject it.",
                out.instrVec.size(), kMaxInstrCount);
        }
        out.instrCount = static_cast<uint16_t>(out.instrVec.size());

        FixReferences(input, origToOut, relJmpProtected, out);

        return out;
    }

} // namespace CcuOpt
} // namespace hcomm
