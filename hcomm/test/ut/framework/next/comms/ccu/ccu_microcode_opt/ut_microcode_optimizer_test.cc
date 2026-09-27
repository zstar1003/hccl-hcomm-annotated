/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_microcode_v1.h"
#include "ccu_log.h"
#include "extract_operands.h"
#include "instruction_scheduler.h"
#include "microcode_optimizer.h"
#include "config/barrier_config.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <unordered_map>
#include <vector>

namespace hcomm {
namespace CcuOpt {
    namespace {

        using namespace hcomm::CcuRep;

        // 按 ExtractOperandsV2 的 Def/Use 顺序重放依赖图 (lastWriter / lastReaders),
        // 收集 WAW / WAR / RAW 三类边, 供 TEST_F 直接查询 HasEdge。
        class DepGraphReplay {
        public:
            struct Edge {
                int from;
                int to;
            };

            explicit DepGraphReplay(const std::vector<CcuInstr>& seq)
            {
                for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
                    auto ops = ExtractOperandsV2(seq[i]);
                    RecordUses(ops, i);
                    RecordDefs(ops, i);
                }
            }

            bool HasEdge(char kind, int from, int to) const
            {
                const std::vector<Edge>& es = (kind == 'w') ? waw_ : (kind == 'r' ? raw_ : war_);
                for (const auto& e : es) {
                    if (e.from == from && e.to == to)
                        return true;
                }
                return false;
            }

        private:
            static uint32_t KeyOf(RegType t, uint16_t id)
            {
                return (static_cast<uint32_t>(t) << 16) | static_cast<uint32_t>(id);
            }

            void RecordUses(const std::vector<RegOperand>& ops, int i)
            {
                for (const auto& op : ops) {
                    if (op.isDef)
                        continue;
                    uint32_t k = KeyOf(op.type, op.regId);
                    auto it = lastWriter_.find(k);
                    if (it != lastWriter_.end())
                        raw_.push_back({it->second, i});
                    lastReaders_[k].push_back(i);
                }
            }

            void RecordDefs(const std::vector<RegOperand>& ops, int i)
            {
                for (const auto& op : ops) {
                    if (!op.isDef)
                        continue;
                    uint32_t k = KeyOf(op.type, op.regId);
                    AddWarEdges(k, i);
                    auto wIt = lastWriter_.find(k);
                    if (wIt != lastWriter_.end() && wIt->second != i)
                        waw_.push_back({wIt->second, i});
                    lastWriter_[k] = i;
                }
            }

            void AddWarEdges(uint32_t k, int i)
            {
                auto rIt = lastReaders_.find(k);
                if (rIt == lastReaders_.end())
                    return;
                for (int r : rIt->second) {
                    if (r != i)
                        war_.push_back({r, i});
                }
                rIt->second.clear();
            }

            std::unordered_map<uint32_t, int> lastWriter_;
            std::unordered_map<uint32_t, std::vector<int>> lastReaders_;
            std::vector<Edge> waw_;
            std::vector<Edge> war_;
            std::vector<Edge> raw_;
        };

        class MicrocodeOptTest : public ::testing::Test {
        protected:
            static CcuInstr MakeNop()
            {
                CcuInstr ins{};
                CcuV2::Nop(&ins);
                return ins;
            }
            static CcuInstr MakeLoadImdToX(uint16_t xnId, uint64_t imm)
            {
                CcuInstr ins{};
                CcuV2::LoadImdToXn(&ins, xnId, imm);
                return ins;
            }
            // Xd = Xn + Xm (parMode = 1)
            static CcuInstr MakeAdd(uint16_t xd, uint16_t xn, uint16_t xm)
            {
                CcuInstr ins{};
                CcuV2::Add(&ins, xd, xn, xm);
                return ins;
            }
            static CcuInstr MakeSetCKE(uint16_t setId, uint16_t waitId)
            {
                CcuInstr ins{};
                CcuV2::SetCKE(&ins, setId, 0xffff, waitId, 0, /*clearType*/ 1);
                return ins;
            }
            // Loop(startInstrId, endInstrIdInclusive, iterNum, offset, contextId).
            // 注意: endInstrId 是闭区间, 指向 body 的最后一条指令的全局 id (与发射器
            // ccu_ins_generater_v2.cc: `start + count - 1` 一致).
            static CcuInstr MakeLoop(uint16_t start, uint16_t endInclusive, uint16_t iterXn)
            {
                CcuInstr ins{};
                CcuV2::Loop(&ins, start, endInclusive, iterXn, 0, 0);
                return ins;
            }
            // 相对跳转的 offset 加载指令: LoadImdToXn(targetXn, offset). 与生成端一致, offset 相对
            // Jump 指令自身 PC.
            static CcuInstr MakeJumpOffsetLoad(uint16_t targetXn, uint16_t offset)
            {
                CcuInstr ins{};
                CcuV2::LoadImdToXn(&ins, targetXn, offset);
                return ins;
            }
            // 无条件相对跳转 (jumpMode == 0). Jump 只引用 targetXn, 真正 offset 由前置 LoadImdToXn 提供.
            // conditionType 的具体取值不影响 offset 修正逻辑, 这里取 0 (等于) 即可.
            static CcuInstr MakeRelJump(uint16_t targetXn)
            {
                CcuInstr ins{};
                CcuV2::Jump(&ins, targetXn, /*condition*/ 0, /*expected*/ 0, /*conditionType*/ 0);
                return ins;
            }
            // 生成一段真实的 RelJmp 9 条模板 (func-call / func-ret 用的运行期地址跳转), 追加到 seq 尾部.
            // 内部为 LoadImdToXn / Jump / Add / Sub / Nop, 其中 Jump 的 targetXn 被 Add/Sub 换算/还原.
            static void
            AppendRelJmp(std::vector<CcuInstr>& seq, uint16_t targetXn, uint32_t jmpInstrId, uint16_t xn0, uint16_t xn1)
            {
                std::vector<CcuInstr> blk(9);
                for (auto& in : blk)
                    in = CcuInstr{};
                CcuV2::RelJmp(blk.data(), targetXn, jmpInstrId, xn0, xn1);
                for (const auto& in : blk)
                    seq.push_back(in);
            }
            // 通用条件跳转的 Jump 指令: Jump(relTar=offsetXn, cond=condXn, expected=expectedXn, condType).
            // 对应普通条件跳转 (jump if condXn <op> expectedXn). condType 非 IF(2) 组合 + 无 RelJmp 模板
            // 结构, 用于验证不会被误判为 RelJmp.
            static CcuInstr MakeCondJump(uint16_t offsetXn, uint16_t condXn, uint16_t expectedXn, uint16_t condType)
            {
                CcuInstr ins{};
                CcuV2::Jump(&ins, offsetXn, condXn, expectedXn, condType);
                return ins;
            }
            // Xd = Xn - Xm (parMode = 1)
            static CcuInstr MakeSub(uint16_t xd, uint16_t xn, uint16_t xm)
            {
                CcuInstr ins{};
                CcuV2::Sub(&ins, xd, xn, xm);
                return ins;
            }
            static CcuInstr MakeClearCKE(uint16_t clearId, uint16_t waitId)
            {
                CcuInstr ins{};
                CcuV2::ClearCKE(
                    &ins, clearId, /*clearMask*/ 0xffff, waitId,
                    /*waitMask*/ waitId == 0 ? uint16_t{0} : uint16_t{0xffff},
                    /*clearType*/ 1);
                return ins;
            }

            // half-rtt LoadX: xn[dst] = array[base + xn[srcOffset]] + imm. 字段落位以 CcuV2::LoadX 为准,
            // 后端优化里 dst(=xdId) 是唯一 def(写者), base/srcOffset 是 use(读者).
            static CcuInstr MakeLoadX(uint16_t dst, uint16_t base, uint16_t srcOffset, uint16_t imm)
            {
                CcuInstr ins{};
                CcuV2::LoadX(&ins, dst, base, srcOffset, imm, /*oMode*/ 0, /*setCKEId*/ 0, /*setCKEMask*/ 0);
                return ins;
            }
            // half-rtt StoreX: array[base + xn[dstOffset]] = xn[src] + imm. 字段落位以 CcuV2::StoreX 为准,
            // 后端优化里 base(=xdo) 是唯一 def(写者), src/dstOffset 是 use(读者).
            static CcuInstr MakeStoreX(uint16_t base, uint16_t src, uint16_t imm, uint16_t dstOffset)
            {
                CcuInstr ins{};
                CcuV2::StoreX(&ins, base, src, imm, dstOffset, /*oMode*/ 0, /*setCKEId*/ 0, /*setCKEMask*/ 0);
                return ins;
            }
            // half-rtt ClearX: 清空连续 xn 区间 [xnId, xmId], 整片既读又写.
            static CcuInstr MakeClearX(uint16_t xnId, uint16_t xmId)
            {
                CcuInstr ins{};
                CcuV2::ClearX(&ins, xnId, xmId, /*xnIdMode*/ 0, /*xmIdMode*/ 0, /*setCKEId*/ 0, /*setCKEMask*/ 0);
                return ins;
            }
            // 统计输出序列里被插入的 NOP 条数 (originIndex == -1). 便于与 nopInserted 交叉验证.
            static uint32_t CountInsertedNops(const SchedulerStats& stats)
            {
                uint32_t n = 0;
                for (int32_t idx : stats.originIndex) {
                    if (idx < 0) {
                        n++;
                    }
                }
                return n;
            }

            // 在 ExtractOperandsV2 结果里查找某个 CKE 寄存器是否以给定 def/use 身份出现。
            static bool HasCkeOperand(const std::vector<RegOperand>& ops, uint16_t regId, bool isDef)
            {
                for (const auto& o : ops) {
                    if (o.type == RegType::CKE && o.regId == regId && o.isDef == isDef)
                        return true;
                }
                return false;
            }

            // 定位输出序列里最后一条 Jump 的前置 LoadImdToXn (相对跳转的 offset 加载指令).
            // 断言 Jump 存在且其前一条为 LOADIMDTOX, 返回该 LoadImdToXn 指令引用.
            static const CcuInstr& FindLoadImdBeforeLastJump(const CcuInstrInfo& out)
            {
                int idxJump = -1;
                for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                    const auto& ins = out.instrVec[i];
                    if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::JMP_CODE) {
                        idxJump = i;
                    }
                }
                EXPECT_GE(idxJump, 1);
                const auto& loadIns = out.instrVec[idxJump - 1];
                EXPECT_EQ(loadIns.header.code, InstrCodeV2::LOADIMDTOX_CODE);
                return loadIns;
            }
        };

        TEST_F(MicrocodeOptTest, ExtractOperands_LoadImdToX)
        {
            auto ins = MakeLoadImdToX(/*xn*/ 5, /*imm*/ 0x1234);
            auto ops = ExtractOperandsV2(ins);
            ASSERT_EQ(ops.size(), 1u);
            EXPECT_EQ(ops[0].type, RegType::XN);
            EXPECT_EQ(ops[0].regId, 5);
            EXPECT_TRUE(ops[0].isDef);
        }

        TEST_F(MicrocodeOptTest, ExtractOperands_Add)
        {
            // xd = xn + xm (parMode = 1) => def xd; use xn, xm
            auto ins = MakeAdd(/*xd*/ 10, /*xn*/ 11, /*xm*/ 12);
            auto ops = ExtractOperandsV2(ins);
            bool foundDef = false, foundXn = false, foundXm = false;
            for (const auto& o : ops) {
                if (o.type != RegType::XN)
                    continue;
                if (o.isDef && o.regId == 10)
                    foundDef = true;
                if (!o.isDef && o.regId == 11)
                    foundXn = true;
                if (!o.isDef && o.regId == 12)
                    foundXm = true;
            }
            EXPECT_TRUE(foundDef);
            EXPECT_TRUE(foundXn);
            EXPECT_TRUE(foundXm);
        }

        // setcke 的 setCKEId 只是置位供别人等待, 不产生供后续指令读取的数据, 故 **不作为 CKE def**;
        // waitCKEId 仍是阻塞读 (use). 只有 clearType=1 自动清零的 waitCKEId 才是触发写后读的 def.
        TEST_F(MicrocodeOptTest, ExtractOperands_SetCKE_SetIsNotDef_WaitIsUse)
        {
            auto ins = MakeSetCKE(/*setId*/ 7, /*waitId*/ 3);
            auto ops = ExtractOperandsV2(ins);
            bool foundSetDef = false, foundWaitUse = false;
            for (const auto& o : ops) {
                if (o.type != RegType::CKE)
                    continue;
                if (o.isDef && o.regId == 7)
                    foundSetDef = true;
                if (!o.isDef && o.regId == 3)
                    foundWaitUse = true;
            }
            EXPECT_FALSE(foundSetDef) << "setcke 的 setCKEId 只是置位, 不应作为 CKE def";
            EXPECT_TRUE(foundWaitUse) << "setcke 的 waitCKEId 仍是阻塞读 (use)";
        }

        // clearType==1: 硬件 wait 到 waitCKEId 后会自动清零, 因此 waitCKEId 既是 use 也是 def.
        TEST_F(MicrocodeOptTest, ExtractOperands_ClearCKE_ClearType1_AutoClearsWaitId)
        {
            // MakeClearCKE 内部固定 clearType=1 (与 EventWait / NotifyWait 生产路径一致).
            // 生产路径下 clearCKEId==0, waitCKEId!=0: 仅 waitCKEId 参与依赖.
            auto ins = MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5);
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false)) << "waitCKEId 应作为 use (阻塞读)";
            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true))
                << "clearType=1 下 waitCKEId 应同时作为 def (自动清零)";
            // clearCKEId==0 应被兜底丢弃, 不进入依赖图 (def/use 都不该出现)。
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ true));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ false));
        }

        // setcke 与 clearcke 对称: clearType==1 时硬件 wait 命中后自动清零 waitCKEId, 因此 waitCKEId
        // 既是 use 也是 def (三种 wait 类 rep 的 profiling 分支即 setId=0/waitId!=0/clearType=1).
        TEST_F(MicrocodeOptTest, ExtractOperands_SetCKE_ClearType1_AutoClearsWaitId)
        {
            auto ins = MakeSetCKE(/*setId*/ 0, /*waitId*/ 5); // MakeSetCKE 默认 clearType=1
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false)) << "waitCKEId 应作为 use (阻塞读)";
            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true))
                << "clearType=1 下 waitCKEId 应同时作为 def (自动清零)";
            // setCKEId==0 应被兜底丢弃, 不进入依赖图 (def/use 都不该出现)。
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ true));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 0, /*isDef*/ false));
        }

        // 反面用例: clearType==0 时 waitCKEId 只 wait 不清零, 仅作为 use; clearCKEId 只是主动清位,
        // 不作为触发写后读的 def. 因此该指令不产生任何 CKE def.
        TEST_F(MicrocodeOptTest, ExtractOperands_ClearCKE_ClearType0_NoAutoClear)
        {
            CcuInstr ins{};
            CcuV2::ClearCKE(
                &ins, /*clearId*/ 3, /*clearMask*/ 0xffff,
                /*waitId*/ 5, /*waitMask*/ 0xffff, /*clearType*/ 0);
            auto ops = ExtractOperandsV2(ins);

            EXPECT_TRUE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ false));
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 5, /*isDef*/ true)) << "clearType=0 时不应把 waitCKEId 视为 def";
            EXPECT_FALSE(HasCkeOperand(ops, /*regId*/ 3, /*isDef*/ true))
                << "clearCKEId 只是主动清位, 不作为触发写后读的 def";
        }

        // 建模层验证: 直接按 lastWriter / lastReaders 重放 ExtractOperandsV2 的 Def/Use.
        // 只有 clearType=1 自动清零的 waitCKEId 才是 CKE 写者: setcke(setId=1,waitId=0) 不产生 CKE def,
        // 故 CKE[1] 的写者只有两条 clearcke(自清), 写后读只在 clearcke 之间.
        TEST_F(MicrocodeOptTest, ExtractOperands_OnlySelfClearDefsCke)
        {
            // 按输入顺序构造 4 条: SetCKE(setId=1) / ClearCKE(waitId=1) / SetCKE(setId=1) / ClearCKE(waitId=1)
            std::vector<CcuInstr> seq;
            seq.push_back(MakeSetCKE(/*setId*/ 1, /*waitId*/ 0));
            seq.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 1));
            seq.push_back(MakeSetCKE(/*setId*/ 1, /*waitId*/ 0));
            seq.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 1));

            // 重放依赖图, 只关心 CKE[1] 上的 def-def 链. 'w'=WAW, 'r'=RAW.
            DepGraphReplay g(seq);

            // setcke (ins0/ins2, setId=1/waitId=0) 不产生 CKE def, 不作为任何依赖端点.
            EXPECT_FALSE(g.HasEdge('w', 0, 1)) << "setcke 非 def, 不该有 setcke->clearcke 的 WAW";
            EXPECT_FALSE(g.HasEdge('w', 1, 2)) << "ins2 是 setcke, 非 def, 不该作为 WAW 后继";
            EXPECT_FALSE(g.HasEdge('w', 2, 3)) << "ins2 是 setcke, 非 def, 不该作为 WAW 前驱";
            // CKE[1] 的真实写者只有两条 clearcke 的自清: ins1 def, ins3 def => 唯一 WAW 1->3.
            EXPECT_TRUE(g.HasEdge('w', 1, 3)) << "两条 clearcke 自清之间构成唯一 WAW: 1->3";
            // 写后读只在 clearcke 之间: ins3 读 CKE[1], 前一个写者是 ins1 => RAW 1->3.
            EXPECT_TRUE(g.HasEdge('r', 1, 3)) << "clearcke 之间的写后读: 1->3";
            // ins1 读 CKE[1] 时没有更早的写者 (setcke 非 def) => 不该有指向 ins1 的 RAW.
            EXPECT_FALSE(g.HasEdge('r', 0, 1)) << "setcke 非 def, 不该产生 setcke->clearcke 的 RAW";
        }

        TEST_F(MicrocodeOptTest, ExtractOperands_NopHasNoOperands)
        {
            auto ins = MakeNop();
            auto ops = ExtractOperandsV2(ins);
            EXPECT_TRUE(ops.empty());
        }

        // setcke(仅 setCKEId 置位) 之后紧跟读同 id 的 clearcke: setCKEId 不是 CKE def (只是置位供别人
        // 等待), 不构成写后读, 不补 NOP. 对应 "record -> 紧邻 wait 同一 event" 不该多插 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_SetCkeThenClearCke_SameBit_NoNop)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeSetCKE(/*setId*/ 5, /*waitId*/ 0));     // 置位 cke5 (非 def)
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 读 cke5, 前无写者
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u) << "setCKEId 非 def, setcke->clearcke 不构成写后读";
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u); // 顺序保持, 无插入
        }

        // clearcke(clearType=1 自动清零) 之后紧跟同 id 的 clearcke: 构成 CKE 写后读, 按
        // CCU_CKE_RAW_LATENCY 补 (L-1) 个 NOP (前一条自身占 1 cycle), 且严格 < L.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_InsertsNopsForCkeRawHazard)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 读并清 cke5 (def)
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // 再读 cke5 (RAW)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            int idxFirst = -1, idxSecond = -1;
            for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::CLEARCKBIT_CODE) {
                    if (idxFirst < 0) {
                        idxFirst = i;
                    } else {
                        idxSecond = i;
                    }
                }
            }
            ASSERT_EQ(idxFirst, 0);
            EXPECT_EQ(idxSecond, static_cast<int>(CcuRep::CCU_CKE_RAW_LATENCY)); // clear(0) + (L-1) NOP + clear
            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);
            EXPECT_LT(sched.Stats().nopInserted, CcuRep::CCU_CKE_RAW_LATENCY); // 每读者补 NOP < L, 与预留对齐
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);                           // CkeOnly 不剥离
        }

        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_IgnoresXnRawHazard)
        {
            // [LoadImdToX(X1), Add(Xd=X2, X1, X5)]: XN 写后读交由硬件 interlock, CkeOnly 不补 NOP.
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadImdToX(/*xn*/ 1, 1));
            input.instrVec.push_back(MakeAdd(/*xd*/ 2, /*xn*/ 1, /*xm*/ 5));
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u); // XN RAW 不补
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u); // 顺序保持, 无插入
            EXPECT_EQ(out.instrVec[0].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[1].header.code, InstrCodeV2::ADD_CODE);
        }

        // ================= half-rtt XN 写后读 (LoadX / StoreX / ClearX) =================

        // ExtractOperandsV2: LoadX 只把 xdId(=i) 当 def, base/srcOffset 当 use, 立即数字段不提取.
        TEST_F(MicrocodeOptTest, ExtractOperands_LoadX_OnlyDstIsDef)
        {
            auto ins = MakeLoadX(/*dst*/ 10, /*base*/ 20, /*srcOffset*/ 30, /*imm*/ 0x55);
            auto ops = ExtractOperandsV2(ins);
            // 期望: def X10; use X20 (base); use X30 (offset). 立即数不入.
            bool defI = false, useBase = false, useOff = false;
            for (const auto& o : ops) {
                if (o.type != RegType::XN)
                    continue;
                if (o.isDef && o.regId == 10)
                    defI = true;
                if (!o.isDef && o.regId == 20)
                    useBase = true;
                if (!o.isDef && o.regId == 30)
                    useOff = true;
            }
            EXPECT_TRUE(defI);
            EXPECT_TRUE(useBase);
            EXPECT_TRUE(useOff);
            // xdId 不应同时以 use 身份出现.
            for (const auto& o : ops) {
                if (o.type == RegType::XN && o.regId == 10) {
                    EXPECT_TRUE(o.isDef);
                }
            }
        }

        // ExtractOperandsV2: StoreX 只把 xdo(=array 基址) 当 def, src(=i)/dstOffset 当 use.
        TEST_F(MicrocodeOptTest, ExtractOperands_StoreX_OnlyArrayBaseIsDef)
        {
            auto ins = MakeStoreX(/*base*/ 40, /*src*/ 50, /*imm*/ 0x66, /*dstOffset*/ 60);
            auto ops = ExtractOperandsV2(ins);
            bool defBase = false, useSrc = false, useOff = false;
            for (const auto& o : ops) {
                if (o.type != RegType::XN)
                    continue;
                if (o.isDef && o.regId == 40)
                    defBase = true;
                if (!o.isDef && o.regId == 50)
                    useSrc = true;
                if (!o.isDef && o.regId == 60)
                    useOff = true;
            }
            EXPECT_TRUE(defBase);
            EXPECT_TRUE(useSrc);
            EXPECT_TRUE(useOff);
        }

        // LoadX 写 i -> 后续任意指令读 i: 构成 XN 写后读, 补 (L-1) 条 NOP, 且严格 < L.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_LoadXWriterFollowedByReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadX(/*dst*/ 7, /*base*/ 20, /*srcOffset*/ 30, /*imm*/ 0)); // 写 X7
            input.instrVec.push_back(MakeAdd(/*xd*/ 8, /*xn*/ 7, /*xm*/ 9));                          // 读 X7 (RAW)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
            EXPECT_LT(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY); // 每写者补 NOP < L, 与预留对齐
            EXPECT_EQ(sched.Stats().nopRemoved, 0u);
            ASSERT_EQ(out.instrVec.size(), CcuRep::CCU_XN_RAW_LATENCY + 1);
            EXPECT_EQ(CountInsertedNops(sched.Stats()), CcuRep::CCU_XN_RAW_LATENCY - 1);
        }

        // 新方向: 别人写 X20 -> LoadX 读 X20 (作为 base). 硬件对 LoadX 读操作数 interlock 失效,
        // 后端须补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_WriterThenLoadXReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ 20, /*xn*/ 1, /*xm*/ 2));                 // 写 X20
            input.instrVec.push_back(MakeLoadX(/*dst*/ 7, /*base*/ 20, /*srcOffset*/ 30, 0)); // LoadX 读 X20 (RAW)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
            EXPECT_LT(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY); // 每读者补 NOP < L, 与预留对齐
            ASSERT_EQ(out.instrVec.size(), CcuRep::CCU_XN_RAW_LATENCY + 1);
            EXPECT_EQ(out.instrVec[0].header.code, InstrCodeV2::ADD_CODE);
            EXPECT_EQ(out.instrVec.back().header.code, InstrCodeV2::LOADX_CODE);
        }

        // 新方向: 别人写 offset 寄存器 X30 -> LoadX 读 X30 (作为 srcOffset). 同样补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_WriterThenLoadXOffsetReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ 30, /*xn*/ 1, /*xm*/ 2));                 // 写 X30 (offset)
            input.instrVec.push_back(MakeLoadX(/*dst*/ 7, /*base*/ 20, /*srcOffset*/ 30, 0)); // LoadX 读 X30
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
        }

        // 新方向: 别人写 src 寄存器 X50 -> StoreX 读 X50 (被存的数据 i). 补 (L-1) 条 NOP.
        // 注意: StoreX 的 base(=array 基址, xdo) 是 def 不是 read, 故写者须落在 StoreX 真正读的
        // src(xsId) 或 dstOffset(xdId) 上, 而非 base.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_WriterThenStoreXReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ 50, /*xn*/ 1, /*xm*/ 2)); // 写 X50 (src)
            // StoreX: base=40(def), src=50(读), dstOffset=60(读). 前序写 X50 -> StoreX 读 X50.
            input.instrVec.push_back(MakeStoreX(/*base*/ 40, /*src*/ 50, /*imm*/ 0, /*dstOffset*/ 60));
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
        }

        // 新方向 + pinnedGroup 归约: 别人写组内寄存器 (base+2) -> LoadX 读 array 基址 (base),
        // 经 pinnedGroup 归约到同一 array key -> 命中, 补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_PinnedGroupWriterThenLoadXReaderInsertsXnNops)
        {
            const uint16_t base = 100;
            const uint16_t count = 4; // array = [100, 104)

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ base + 2, /*xn*/ 1, /*xm*/ 2));             // 写 X102 (组内)
            input.instrVec.push_back(MakeLoadX(/*dst*/ 7, /*base*/ base, /*srcOffset*/ 30, 0)); // LoadX 读 array@100
            input.instrCount = 2;

            std::vector<PinnedGroup> groups{PinnedGroup{base, count}};
            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            o.pinnedGroups = &groups;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1)
                << "组内寄存器写经 pinnedGroup 归约应命中 LoadX 读的 array key";
        }

        // 新方向: 普通指令作为读者不查通用写者表 -> 别人写 X20 -> 普通 Add 读 X20 仍不补 (硬件 interlock).
        // 与"任意写 -> lsx 读"区分: 只有读者是 lsx/clearx 时才补.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_WriterThenNormalReaderStillNoNop)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ 20, /*xn*/ 1, /*xm*/ 2));  // 写 X20
            input.instrVec.push_back(MakeAdd(/*xd*/ 21, /*xn*/ 20, /*xm*/ 3)); // 普通读 X20 (非 lsx)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u); // 普通读者交硬件 interlock
            ASSERT_EQ(out.instrVec.size(), 2u);
        }

        // StoreX 写 array 基址 -> 后续读同一 array 基址寄存器: 补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_StoreXWriterFollowedByArrayReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeStoreX(/*base*/ 40, /*src*/ 50, /*imm*/ 0, /*dstOffset*/ 60)); // 写 X40(array)
            input.instrVec.push_back(MakeAdd(/*xd*/ 41, /*xn*/ 40, /*xm*/ 42));                         // 读 X40 (RAW)
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
            ASSERT_EQ(out.instrVec.size(), CcuRep::CCU_XN_RAW_LATENCY + 1);
        }

        // pinnedGroup 归约: StoreX 写 array 基址 baseId, 后续指令读组内另一个寄存器 (baseId+2),
        // 经 pinnedGroup 归约到同一 array key -> 命中写后读, 补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_PinnedGroupReducesArrayReaderToStoreXWriter)
        {
            const uint16_t base = 100;
            const uint16_t count = 4; // array = [100, 104)

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(
                MakeStoreX(/*base*/ base, /*src*/ 50, /*imm*/ 0, /*dstOffset*/ 60));    // 写 array@100
            input.instrVec.push_back(MakeAdd(/*xd*/ 200, /*xn*/ base + 2, /*xm*/ 201)); // 读 X102 (组内)
            input.instrCount = 2;

            std::vector<PinnedGroup> groups{PinnedGroup{base, count}};
            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            o.pinnedGroups = &groups;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1)
                << "组内寄存器经 pinnedGroup 归约应命中 StoreX 写的 array key";
        }

        // 无 pinnedGroup 时, 读组内另一寄存器 (base+2) 与写者 key(base) 不同 -> 不命中, 不补.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_NoPinnedGroupArrayReaderMisses)
        {
            const uint16_t base = 100;

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeStoreX(/*base*/ base, /*src*/ 50, /*imm*/ 0, /*dstOffset*/ 60));
            input.instrVec.push_back(MakeAdd(/*xd*/ 200, /*xn*/ base + 2, /*xm*/ 201)); // 读 X102, 无归约
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly; // pinnedGroups == nullptr
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u);
        }

        // ClearX 同片自依赖: 两条相同区间的 ClearX 构成写后读, 补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_ClearXSameRangeInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearX(/*xnId*/ 300, /*xmId*/ 310));
            input.instrVec.push_back(MakeClearX(/*xnId*/ 300, /*xmId*/ 310)); // 同片 RAW
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
            ASSERT_EQ(out.instrVec.size(), CcuRep::CCU_XN_RAW_LATENCY + 1);
        }

        // ClearX 不同片: 两条区间不相交的 ClearX 不构成写后读, 不补.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_ClearXDifferentRangeNoNop)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearX(/*xnId*/ 300, /*xmId*/ 310));
            input.instrVec.push_back(MakeClearX(/*xnId*/ 320, /*xmId*/ 330)); // 不同片
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u);
            ASSERT_EQ(out.instrVec.size(), 2u);
        }

        // 新方向: 别人写区间端点 X300 -> ClearX 读区间 [300,310] (端点 300 既读又写). ClearX 作为读者,
        // 硬件 interlock 对其读操作数失效, 命中前序写者补 (L-1) 条 NOP.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_WriterThenClearXReaderInsertsXnNops)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeAdd(/*xd*/ 300, /*xn*/ 1, /*xm*/ 2)); // 写 X300 (区间端点)
            input.instrVec.push_back(MakeClearX(/*xnId*/ 300, /*xmId*/ 310));  // ClearX 读端点 X300
            input.instrCount = 2;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, CcuRep::CCU_XN_RAW_LATENCY - 1);
            ASSERT_EQ(out.instrVec.size(), CcuRep::CCU_XN_RAW_LATENCY + 1);
        }

        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_LoopRefsRemappedAfterCkeNopInsertion)
        {
            // body 内含一次 CKE 写后读 (clearcke -> 同 id clearcke), CkeOnly 在 body 内插 NOP,
            // Loop.start/end 需同步平移.
            //   [0] ClearCKE(wait cke5)  <- body[0] (def cke5)
            //   [1] ClearCKE(wait cke5)  <- body[1] (RAW)
            //   [2] LoopGroup -> loop id 3
            //   [3] Loop(start=0, end=1, iter=X20)
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));
            {
                CcuInstr lg{};
                lg.header = CcuRep::InstrHeader(InstrCodeV2::CTRL_TYPE, InstrCodeV2::LOOPGROUP_CODE);
                lg.v2.loopGroup.startLoopInstrId = 3;
                input.instrVec.push_back(lg);
            }
            input.instrVec.push_back(MakeLoop(/*start*/ 0, /*endInclusive*/ 1, /*iter*/ 20));
            input.instrCount = 4;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            int idxLoop = -1, idxLg = -1, idxBody0 = -1, idxBody1 = -1;
            for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::LOOP_CODE) {
                    idxLoop = i;
                }
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::LOOPGROUP_CODE) {
                    idxLg = i;
                }
                // 两条 clearcke: 先出现的是 body0, 后出现的是 body1.
                if (ins.header.type == InstrCodeV2::CTRL_TYPE && ins.header.code == InstrCodeV2::CLEARCKBIT_CODE) {
                    if (idxBody0 < 0) {
                        idxBody0 = i;
                    } else {
                        idxBody1 = i;
                    }
                }
            }
            ASSERT_GE(idxLoop, 0);
            ASSERT_GE(idxLg, 0);
            ASSERT_EQ(idxBody0, 0);
            ASSERT_EQ(idxBody1, static_cast<int>(CcuRep::CCU_CKE_RAW_LATENCY));

            const auto& loopIns = out.instrVec[idxLoop];
            EXPECT_EQ(loopIns.v2.loop.startInstrId, static_cast<uint16_t>(idxBody0));
            EXPECT_EQ(loopIns.v2.loop.endInstrId, static_cast<uint16_t>(idxBody1));

            const auto& lgIns = out.instrVec[idxLg];
            EXPECT_EQ(lgIns.v2.loopGroup.startLoopInstrId, static_cast<uint16_t>(idxLoop));
        }

        // jmp 与其目标 label 之间插入了 CKE 补偿 NOP: 相对跳转 offset (前置 LoadImdToXn 的立即数)
        // 必须随之增大, 否则会跳到错误位置. 验证 FixReferences 的 jmp 分支正确改写 offset.
        //   [0] LoadImdToX(xn7, offset=3)   <- jmp 的目标偏移, 目标为下方 index 4
        //   [1] Jump(relTar=xn7)            <- jmp 自身, 基准 PC = 1
        //   [2] ClearCKE(wait cke5)         <- 区间内, CKE def
        //   [3] ClearCKE(wait cke5)         <- 区间内, CKE RAW => 在 [2] 与 [3] 间插 (L-1) 个 NOP
        //   [4] Nop                         <- 跳转目标 label
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_RelJumpOffsetRemappedWhenNopInsertedInside)
        {
            const uint16_t targetXn = 7;
            const uint16_t oldOffset = 3; // 4 - 1
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeJumpOffsetLoad(targetXn, oldOffset));   // [0]
            input.instrVec.push_back(MakeRelJump(targetXn));                     // [1]
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [2]
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [3]
            input.instrVec.push_back(MakeNop());                                 // [4] target
            input.instrCount = 5;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            const uint32_t inserted = sched.Stats().nopInserted;
            ASSERT_EQ(inserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);

            // 定位输出里的 Jump 与其前置 LoadImdToXn.
            const auto& loadIns = FindLoadImdBeforeLastJump(out);

            // 目标 (原 index 4) 被下移 inserted 条, jmp 位置不变 => 新 offset = 旧 offset + inserted.
            EXPECT_EQ(loadIns.v2.loadImdToX.immediate, static_cast<uint64_t>(oldOffset + inserted));
        }

        // 反面场景: NOP 插在跳转区间之外 (jmp 与目标都在插入点之后), 相对距离不变, offset 不应被改动.
        //   [0] ClearCKE(wait cke5)         <- CKE def (区间外, 在 jmp 之前)
        //   [1] ClearCKE(wait cke5)         <- CKE RAW => 在 [0] 与 [1] 间插 (L-1) 个 NOP
        //   [2] LoadImdToX(xn7, offset=2)   <- jmp 目标偏移, 目标为 index 4 (基准 jmp PC=3)
        //   [3] Jump(relTar=xn7)
        //   [4] Nop                         <- 跳转目标 label
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_RelJumpOffsetUnchangedWhenNopOutsideRange)
        {
            const uint16_t targetXn = 7;
            const uint16_t offset = 1; // jmp 在 index 3, 目标 index 4 => offset = 4 - 3 = 1.
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [0]
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [1]
            input.instrVec.push_back(MakeJumpOffsetLoad(targetXn, offset));      // [2]
            input.instrVec.push_back(MakeRelJump(targetXn));                     // [3]
            input.instrVec.push_back(MakeNop());                                 // [4] target
            input.instrCount = 5;

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            ASSERT_EQ(sched.Stats().nopInserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);

            const auto& loadIns = FindLoadImdBeforeLastJump(out);
            // jmp 与目标都在 NOP 之后, 相对距离不变.
            EXPECT_EQ(loadIns.v2.loadImdToX.immediate, static_cast<uint64_t>(offset));
        }

        // RelJmp 原子块 (func-call / func-ret 运行期地址跳转) 内部禁止插 NOP: 其立即数含绝对 id 且被
        // Add/Sub 换算, FixReferences 无法安全重映射, 必须保证 CkeOnly 不在块内补 NOP.
        // 单独一段 RelJmp 作为输入, 输出应原样 9 条、零 NOP 插入.
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_RelJmpBlockNotSplitByNop)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            AppendRelJmp(input.instrVec, /*targetXn*/ 8, /*jmpInstrId*/ 100, /*xn0*/ 1, /*xn1*/ 2);
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());
            ASSERT_EQ(input.instrVec.size(), 9u);

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            EXPECT_EQ(sched.Stats().nopInserted, 0u) << "RelJmp 块内不应插入任何 NOP";
            ASSERT_EQ(out.instrVec.size(), 9u) << "RelJmp 9 条应原样保留, 无插入";
            // 块内 originIndex 应全部为真实源 (无 -1 的新插 NOP).
            for (int32_t src : sched.Stats().originIndex) {
                EXPECT_GE(src, 0) << "RelJmp 块内不应出现新插入的 NOP (src=-1)";
            }
        }

        // 即使把 CKE 写后读的两端跨在 RelJmp 块两侧, 块内也不能被插 NOP: latency 补偿只能落在块外.
        //   [0]      ClearCKE(wait cke5)     <- CKE def (cycle 0)
        //   [1..9]   RelJmp 9 条             <- 受保护原子块 (紧跟在写者之后, 连续 9 条)
        //   [10]     ClearCKE(wait cke5)     <- CKE RAW; 因 L=14 > 9, 块之后仍需补 (14-10) 条 NOP
        // 关键: 块内 [1..9] 必须紧贴第一条 clearcke 连续排列, 补偿 NOP 只能落在块之后 (块外).
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_NopNotInsertedInsideRelJmpBlockAcrossCkeRaw)
        {
            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));                    // [0] def cke5
            AppendRelJmp(input.instrVec, /*targetXn*/ 8, /*jmpInstrId*/ 100, /*xn0*/ 1, /*xn1*/ 2); // [1..9]
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [10] RAW read cke5
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            const auto& originIndex = sched.Stats().originIndex;
            ASSERT_EQ(originIndex.size(), out.instrVec.size());

            // 第一条 clearcke 在输出下标 0 (原序保持), 其后紧跟的 9 条应是 RelJmp body (原始源下标 1..9),
            // 连续且无调度器新插 NOP (src=-1). 这正是"块内不插 NOP、块紧贴写者"的直接证据.
            ASSERT_GE(static_cast<int>(out.instrVec.size()), 10);
            ASSERT_EQ(out.instrVec[0].header.code, InstrCodeV2::CLEARCKBIT_CODE);
            for (int i = 1; i <= 9; ++i) {
                EXPECT_EQ(originIndex[i], i)
                    << "RelJmp 块内 (输出下标 " << i << ") 应为原始源 " << i << ", 无 NOP 插入或位移";
            }

            // 第二条 clearcke 应出现在 9 条 body 之后; 因 L=14 > 9, 其前会有 (14-10) 条块外补偿 NOP.
            int idxSecondClear = -1;
            for (int i = 10; i < static_cast<int>(out.instrVec.size()); ++i) {
                if (out.instrVec[i].header.type == InstrCodeV2::CTRL_TYPE
                    && out.instrVec[i].header.code == InstrCodeV2::CLEARCKBIT_CODE) {
                    idxSecondClear = i;
                    break;
                }
            }
            ASSERT_GT(idxSecondClear, 9) << "第二条 clearcke 应在 RelJmp 块之后";
            // 第二条 clearcke 发射 cycle 应满足 CKE RAW latency: 恰好落在 cycle L.
            EXPECT_EQ(idxSecondClear, static_cast<int>(CcuRep::CCU_CKE_RAW_LATENCY))
                << "块外补偿后, 第二条 clearcke 落在 cycle L (latency 满足)";
            // 补偿 NOP 全部在块之后 (下标 10 .. idxSecondClear-1), 均为 src=-1.
            for (int i = 10; i < idxSecondClear; ++i) {
                EXPECT_EQ(originIndex[i], -1) << "块外补偿位置 (下标 " << i << ") 应为新插 NOP";
            }
        }

        // RelJmp 与其目标之间插入 NOP: 跳转长度 (= 目标绝对id - jmpInstrId基准) 必须随之延长, 表现为
        // 目标绝对 id 的加载立即数被 remap 到新位置, jmpInstrId 基准跟随主 Jump 新 PC.
        //   [0]      LoadImdToX(targetXn=8, E)          <- 目标绝对 id 加载 (funcCall 风格)
        //   [1..9]   RelJmp(targetXn=8, jmpInstrId=10)  <- 主 Jump 在 [10]
        //   [10]     Jump(targetXn=8)                   <- 主 Jump (相对基准)
        //   [11]     ClearCKE(def cke5)                 <- 目标前的 CKE 写者
        //   [12]     ClearCKE(read cke5)                <- RAW: 在 [11][12] 间插 (L-1) NOP
        //   [13]     Nop                                <- 跳转目标 (E=13)
        // 插 NOP 点在主 Jump 之后、目标之前: 主 Jump PC 不变(=10), 目标后移 (L-1). 故 jmpInstrId 保持 10,
        // 目标立即数 remap 13 -> 13+(L-1).
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_RelJmpTargetRemappedWhenNopInsertedBeforeTarget)
        {
            const uint16_t targetXn = 8, xn0 = 1, xn1 = 2;
            const uint32_t jmpInstrId = 10; // 主 Jump 位置
            const uint16_t targetE = 13;    // 目标 (Nop) 位置

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadImdToX(targetXn, targetE));         // [0] 目标绝对 id
            AppendRelJmp(input.instrVec, targetXn, jmpInstrId, xn0, xn1);        // [1..9]
            input.instrVec.push_back(MakeRelJump(targetXn));                     // [10] 主 Jump
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [11] def cke5
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [12] RAW
            input.instrVec.push_back(MakeNop());                                 // [13] 目标
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());
            ASSERT_EQ(input.instrVec.size(), 14u);

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            const uint32_t inserted = sched.Stats().nopInserted;
            ASSERT_EQ(inserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);

            // 目标绝对 id 加载在输出下标 0 (原序保持, 其前无插入).
            ASSERT_EQ(out.instrVec[0].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[0].v2.loadImdToX.immediate, static_cast<uint64_t>(targetE + inserted))
                << "目标位移后, 目标绝对 id 立即数应 remap 到新位置";

            // RelJmp 块内 P+0 (jmpInstrId 基准) 与 P+3 (0x10000 - jmpInstrId) 仍指向主 Jump PC=10 (未位移).
            // 块在输出下标 [1..9] (原序保持, 块内不插 NOP). P+0 是块首条 LoadImdToX(xn0).
            int idxP0 = -1, idxP3 = -1;
            for (int i = 1; i <= 9; ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type == InstrCodeV2::LOAD_TYPE && ins.header.code == InstrCodeV2::LOADIMDTOX_CODE
                    && ins.v2.loadImdToX.xnId == xn0) {
                    if (idxP0 < 0)
                        idxP0 = i;
                    else
                        idxP3 = i;
                }
            }
            ASSERT_GE(idxP0, 1);
            ASSERT_GT(idxP3, idxP0);
            EXPECT_EQ(out.instrVec[idxP0].v2.loadImdToX.immediate, static_cast<uint64_t>(jmpInstrId))
                << "主 Jump 未位移, jmpInstrId 基准应保持不变";
            EXPECT_EQ(out.instrVec[idxP3].v2.loadImdToX.immediate, static_cast<uint64_t>(0x10000u - jmpInstrId))
                << "P+3 的 0x10000 - jmpInstrId 应与 P+0 同步";
        }

        // RelJmp 之前插入 NOP (jmpInstrId 基准与目标一起后移): 两个绝对量同步平移, 跳转仍落到目标.
        //   [0]      ClearCKE(def cke5)                 <- 在 RelJmp 之前的 CKE 写者
        //   [1]      ClearCKE(read cke5)                <- RAW: 在 [0][1] 间插 (L-1) NOP
        //   [2]      LoadImdToX(targetXn=8, E)          <- 目标绝对 id
        //   [3..11]  RelJmp(targetXn=8, jmpInstrId=12)  <- 主 Jump 在 [12]
        //   [12]     Jump(targetXn=8)                   <- 主 Jump
        //   [13]     Nop                                <- 目标 (E=13)
        // 插 NOP 在整段之前: 主 Jump 与目标同步后移 (L-1). jmpInstrId: 12 -> 12+(L-1); 目标: 13 -> 13+(L-1).
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_RelJmpBaseAndTargetShiftTogether)
        {
            const uint16_t targetXn = 8, xn0 = 1, xn1 = 2;
            const uint32_t jmpInstrId = 12;
            const uint16_t targetE = 13;

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [0] def
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5)); // [1] RAW
            input.instrVec.push_back(MakeLoadImdToX(targetXn, targetE));         // [2] 目标
            AppendRelJmp(input.instrVec, targetXn, jmpInstrId, xn0, xn1);        // [3..11]
            input.instrVec.push_back(MakeRelJump(targetXn));                     // [12] 主 Jump
            input.instrVec.push_back(MakeNop());                                 // [13] 目标
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            const uint32_t inserted = sched.Stats().nopInserted;
            ASSERT_EQ(inserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);

            // 找目标绝对 id 加载 (LoadImdToX 写 targetXn) 与 RelJmp P+0/P+3 (LoadImdToX 写 xn0).
            int idxTarget = -1, idxP0 = -1, idxP3 = -1;
            for (int i = 0; i < static_cast<int>(out.instrVec.size()); ++i) {
                const auto& ins = out.instrVec[i];
                if (ins.header.type != InstrCodeV2::LOAD_TYPE || ins.header.code != InstrCodeV2::LOADIMDTOX_CODE)
                    continue;
                if (ins.v2.loadImdToX.xnId == targetXn && idxTarget < 0)
                    idxTarget = i;
                if (ins.v2.loadImdToX.xnId == xn0) {
                    if (idxP0 < 0)
                        idxP0 = i;
                    else if (idxP3 < 0)
                        idxP3 = i;
                }
            }
            ASSERT_GE(idxTarget, 0);
            ASSERT_GE(idxP0, 0);
            ASSERT_GE(idxP3, 0);

            EXPECT_EQ(out.instrVec[idxTarget].v2.loadImdToX.immediate, static_cast<uint64_t>(targetE + inserted))
                << "目标随整段后移, 立即数 remap";
            EXPECT_EQ(out.instrVec[idxP0].v2.loadImdToX.immediate, static_cast<uint64_t>(jmpInstrId + inserted))
                << "jmpInstrId 基准随主 Jump 后移";
            EXPECT_EQ(
                out.instrVec[idxP3].v2.loadImdToX.immediate, static_cast<uint64_t>(0x10000u - (jmpInstrId + inserted)))
                << "P+3 与 P+0 同步";

            // 跳转长度 = 目标 - jmpInstrId 保持不变 (整段同步位移): (13-12) == (新目标 - 新基准).
            const uint64_t newTarget = out.instrVec[idxTarget].v2.loadImdToX.immediate;
            const uint64_t newBase = out.instrVec[idxP0].v2.loadImdToX.immediate;
            EXPECT_EQ(newTarget - newBase, static_cast<uint64_t>(targetE - jmpInstrId))
                << "整段同步位移, 相对跳转长度不变";
        }

        // 回归: 普通条件跳转 (jump if X3 == X5), 且跳转后紧跟一条 Add 写 condition 寄存器 X3 —— 这正是
        // 旧弱识别 (只看 conditionXnId 后有无 Add/Sub) 会误判成 RelJmp 的场景. 误判会让修复逻辑把承载
        // 64 位 expected 比较值的那条 LoadImdToXn 当成 "jmpInstrId 基准" 去 remap/截断. 强指纹修复后:
        //   - 该 jmp 不被识别为 RelJmp (走普通 jmp 分支);
        //   - expected 比较值 load 绝不被改动 (哪怕值 > 0x10000).
        //   [0] LoadImdToX(X5, expected=0x100000007)  <- 64 位比较值
        //   [1] LoadImdToX(X8, offset=3)              <- offset, 目标 [4]
        //   [2] Jump(relTar=X8, cond=X3, expected=X5, EQ)
        //   [3] Add(X3, X3, X1)                       <- 旧代码误判触发点
        //   [4] Nop                                   <- 跳转目标
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_CondJumpNotMisdetectedAsRelJmp_ExpectedLoadUntouched)
        {
            const uint16_t offXn = 8, condXn = 3, expXn = 5, otherXn = 1;
            const uint64_t expectedVal = 0x100000007ULL; // > 0x10000, 若被误 remap/截断会立刻暴露
            // 普通 jmp offset 基准是 jmp 自身 PC(=2), 目标 [4] => offset = 4 - 2 = 2.
            const uint16_t realOffset = 2;

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadImdToX(expXn, expectedVal));           // [0]
            input.instrVec.push_back(MakeJumpOffsetLoad(offXn, realOffset));        // [1]
            input.instrVec.push_back(MakeCondJump(offXn, condXn, expXn, /*EQ*/ 0)); // [2]
            input.instrVec.push_back(MakeAdd(condXn, condXn, otherXn));             // [3] 干扰: Add 写 cond
            input.instrVec.push_back(MakeNop());                                    // [4] 目标
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            // 无 CKE 依赖, 不插 NOP, 原序保持.
            EXPECT_EQ(sched.Stats().nopInserted, 0u);
            ASSERT_EQ(out.instrVec.size(), 5u);

            // 关键断言: expected 比较值 load ([0]) 立即数完全不变 (未被截断、未被 remap).
            ASSERT_EQ(out.instrVec[0].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[0].v2.loadImdToX.immediate, expectedVal)
                << "普通条件跳转的 expected 比较值 load 绝不该被修复逻辑改动";

            // offset load ([1]) 无位移, 仍为原值 (普通 jmp 分支处理, remap 后不变).
            ASSERT_EQ(out.instrVec[1].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[1].v2.loadImdToX.immediate, static_cast<uint64_t>(realOffset))
                << "无 NOP 插入时 offset 不变";
        }

        // 同上误判场景, 但这次跳转区间内插入 NOP: expected 仍不该动, 而 offset 应正确 +NOP 数 (普通 jmp).
        //   [0] LoadImdToX(X5, expected=0x100000007)
        //   [1] LoadImdToX(X8, offset=2)              <- jmp 在 [2], 目标 [5]; 基准 jmp PC=2
        //   [2] Jump(relTar=X8, cond=X3, expected=X5, EQ)   <- 目标原本 [4]? 见构造
        //   [3] ClearCKE(def cke5)
        //   [4] ClearCKE(read cke5)                   <- RAW: [3][4] 间插 (L-1) NOP
        //   [5] Nop                                   <- 跳转目标
        TEST_F(MicrocodeOptTest, SchedulerCkeOnly_CondJumpExpectedUntouched_OffsetRemappedWithNop)
        {
            const uint16_t offXn = 8, condXn = 3, expXn = 5, otherXn = 1;
            const uint64_t expectedVal = 0x100000007ULL;
            // jmp 在 [2], 目标 Nop 在 [5]. offset 基准是 jmp PC=2 => offset = 5 - 2 = 3.
            const uint16_t offset = 3;

            CcuInstrInfo input{};
            input.startInstrId = 0;
            input.instrVec.push_back(MakeLoadImdToX(expXn, expectedVal));           // [0]
            input.instrVec.push_back(MakeJumpOffsetLoad(offXn, offset));            // [1]
            input.instrVec.push_back(MakeCondJump(offXn, condXn, expXn, /*EQ*/ 0)); // [2]
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));    // [3] def cke5
            input.instrVec.push_back(MakeClearCKE(/*clearId*/ 0, /*waitId*/ 5));    // [4] RAW
            input.instrVec.push_back(MakeNop());                                    // [5] 目标
            input.instrVec.push_back(MakeAdd(condXn, condXn, otherXn));             // [6] 干扰: Add 写 cond
            input.instrCount = static_cast<uint16_t>(input.instrVec.size());

            InstructionSchedulerOptions o{};
            o.level = SchedLevel::CkeOnly;
            InstructionScheduler sched(o);
            auto out = sched.Schedule(input);

            const uint32_t inserted = sched.Stats().nopInserted;
            ASSERT_EQ(inserted, CcuRep::CCU_CKE_RAW_LATENCY - 1);

            // expected load 立即数完全不变.
            EXPECT_EQ(out.instrVec[0].v2.loadImdToX.immediate, expectedVal)
                << "跨区间插 NOP 时, expected 比较值仍不该被改动";

            // offset load ([1], 原序保持在下标 1): 目标 [5] 后移 inserted 条, jmp [2] 不动 => offset += inserted.
            ASSERT_EQ(out.instrVec[1].header.code, InstrCodeV2::LOADIMDTOX_CODE);
            EXPECT_EQ(out.instrVec[1].v2.loadImdToX.immediate, static_cast<uint64_t>(offset + inserted))
                << "普通条件跳转的 offset 应随 NOP 插入正确延长";
        }

        TEST_F(MicrocodeOptTest, DefaultOptions_SchedLevelIsCkeOnly)
        {
            // 极简后端优化只有 CkeOnly 一档, 编译期恒返回 CkeOnly.
            OptimizerOptions o = MicrocodeOptimizer::DefaultOptions();
            EXPECT_EQ(o.schedLevel, SchedLevel::CkeOnly);
        }

    } // namespace
} // namespace CcuOpt
} // namespace hcomm
