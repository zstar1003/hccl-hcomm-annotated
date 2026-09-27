/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <climits>

#include "ccu_ins_generator_v2.h"
#include "ccu_rep_v1.h"
#include "ccu_datatype_v1.h"
#include "ccu_rep_context_v1.h"
#include "ccu_api_exception.h"
#include "ccu_res_repo.h"

namespace hcomm {
namespace CcuRep {
    namespace {

        class HalfRttRepTest : public ::testing::Test {
        protected:
            CcuInsGeneratorV2 insGen{};
            CcuRepContext context{};
            // translate 需从 CcuKernel 按 handle 查询 CascCntBlock, 故持有真实 kernel 并注入块数据
            CcuKernel kernel{};
            std::unordered_map<HcommCcuCascCntHandle, CntXnBlock> injectedBlocks{};
            HcommCcuCascCntHandle nextHandle{0};

            void SetUp() override {}

            void TearDown() override {}

            // 向 CcuKernel 注入测试数据块, 返回对应的 handle
            // 布局遵循控制面分配惯例: 1022 个 wishCntXn + totalCntXn + expectedCntXn
            // 块按值存入 injectedBlocks, 再通过 SetCascCntBlock 灌入 kernel(与 CcuResPack 持块行为一致)
            HcommCcuCascCntHandle
            InjectCascCntBlock(uint32_t wishBegin, uint32_t wishEnd, uint32_t totalXn, uint32_t expectedXn)
            {
                CntXnBlock block{};
                block.wishCntXns = {wishBegin, wishEnd};
                block.totalCntXn = totalXn;
                block.expectedCntXn = expectedXn;
                nextHandle += 1;
                injectedBlocks.emplace(nextHandle, block);
                kernel.SetCascCntBlock(injectedBlocks);
                return nextHandle;
            }
        };

        /* ---------------- repTypeInstrCount 表条目 ---------------- */

        TEST_F(HalfRttRepTest, GetInstrCount_WriteVarAtomic)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::WRITE_VAR_ATOMIC), 1u);
        }

        TEST_F(HalfRttRepTest, GetInstrCount_WriteWithCntInc)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::WRITE_WITH_CNT_INC), 1u);
        }

        TEST_F(HalfRttRepTest, GetInstrCount_CascCntWait)
        {
            // CascCntWait = LoadImdToXn + Wait 两条指令
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::CASC_CNT_WAIT), 2u);
        }

        TEST_F(HalfRttRepTest, GetInstrCount_CascCntClear)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::CASC_CNT_CLEAR), 1u);
        }

        TEST_F(HalfRttRepTest, GetInstrCount_LoadAddImm)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::LOAD_ADD_IMM), 1u);
        }

        TEST_F(HalfRttRepTest, GetInstrCount_StoreAddImm)
        {
            EXPECT_EQ(insGen.GetInstrCount(CcuRepType::STORE_ADD_IMM), 1u);
        }

        /* ---------------- LoadAddImm: 构造/Describe ---------------- */

        TEST_F(HalfRttRepTest, LoadAddImm_ConstructorAndGetters)
        {
            Variable src{&context};
            Variable srcOffset{&context};
            Variable dst{&context};

            CcuRepLoadAddImm rep(&insGen, src, 2, srcOffset, 0x55, dst);
            EXPECT_EQ(rep.Type(), CcuRepType::LOAD_ADD_IMM);
            EXPECT_EQ(rep.GetSrcId(), src.Id());
            EXPECT_EQ(rep.GetSrcOffsetId(), srcOffset.Id());
            EXPECT_EQ(rep.GetDstId(), dst.Id());
            EXPECT_EQ(rep.GetImmAddValue(), 0x55);
        }

        TEST_F(HalfRttRepTest, LoadAddImm_Describe)
        {
            Variable src{&context};
            Variable srcOffset{&context};
            Variable dst{&context};

            CcuRepLoadAddImm rep(&insGen, src, 2, srcOffset, 0x55, dst);
            std::string desc = rep.Describe();
            EXPECT_TRUE(desc.find("Load Add Imm") != std::string::npos);
            EXPECT_TRUE(desc.find("immAddValue[85]") != std::string::npos); // 0x55
        }

        /* ---------------- LoadAddImm translate -> LoadX 指令 ---------------- */

        TEST_F(HalfRttRepTest, LoadAddImmTranslate_GeneratesLoadXInstr)
        {
            Variable src{&context};
            Variable srcOffset{&context};
            Variable dst{&context};

            CcuRepLoadAddImm rep(&insGen, src, 2, srcOffset, 0x55, dst);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepLoadAddImmTranslate(nullptr, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // 位段映射: dst->Xd, srcOffset->Xs, src->Xso, immAddValue->Xdo(立即数), O_Mode=0
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdId, dst.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.xsId, srcOffset.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.xso, src.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdo, 0x55);
            EXPECT_EQ(instrs[0].v2.loadStoreX.oMode, 0u);
        }

        /* ---------------- StoreAddImm: 构造/Describe/translate ---------------- */

        TEST_F(HalfRttRepTest, StoreAddImm_ConstructorAndGetters)
        {
            Variable dst{&context};
            Variable dstOffset{&context};
            Variable src{&context};

            CcuRepStoreAddImm rep(&insGen, dst, 3, dstOffset, 0x66, src);
            EXPECT_EQ(rep.Type(), CcuRepType::STORE_ADD_IMM);
            EXPECT_EQ(rep.GetDstId(), dst.Id());
            EXPECT_EQ(rep.GetDstOffsetId(), dstOffset.Id());
            EXPECT_EQ(rep.GetSrcId(), src.Id());
            EXPECT_EQ(rep.GetImmAddValue(), 0x66);
        }

        TEST_F(HalfRttRepTest, StoreAddImm_Describe)
        {
            Variable dst{&context};
            Variable dstOffset{&context};
            Variable src{&context};

            CcuRepStoreAddImm rep(&insGen, dst, 3, dstOffset, 0x66, src);
            std::string desc = rep.Describe();
            EXPECT_TRUE(desc.find("Store Add Imm") != std::string::npos);
            EXPECT_TRUE(desc.find("immAddValue[102]") != std::string::npos); // 0x66
        }

        TEST_F(HalfRttRepTest, StoreAddImmTranslate_GeneratesStoreXInstr)
        {
            Variable dst{&context};
            Variable dstOffset{&context};
            Variable src{&context};

            CcuRepStoreAddImm rep(&insGen, dst, 3, dstOffset, 0x66, src);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepStoreAddImmTranslate(nullptr, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // 位段映射: dstOffset->Xd, src->Xs, immAddValue->Xso(立即数), dst->Xdo, O_Mode=0
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdId, dstOffset.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.xsId, src.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.xso, 0x66);
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdo, dst.Id());
            EXPECT_EQ(instrs[0].v2.loadStoreX.oMode, 0u);
        }

        /* ---------------- WriteVarAtomic: 构造/translate ---------------- */

        TEST_F(HalfRttRepTest, WriteVarAtomic_ConstructorAndGetters)
        {
            Variable channel{&context};
            Address addr;
            Variable token;
            RemoteAddr varAddr{addr, token};
            Variable targetValue{&context};
            CompletedEvent sem{&context};

            CcuRepWriteVarAtomic rep(&insGen, channel, varAddr, targetValue, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::WRITE_VAR_ATOMIC);
            EXPECT_EQ(rep.GetVarAddrId(), varAddr.addr.Id());
            EXPECT_EQ(rep.GetVarTokenId(), varAddr.token.Id());
            EXPECT_EQ(rep.GetTargetId(), targetValue.Id());
            EXPECT_EQ(rep.GetSemId(), sem.Id());
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(HalfRttRepTest, WriteVarAtomicTranslate_GeneratesSyncAtXInstr)
        {
            Variable channel{&context};
            Address addr;
            Variable token;
            RemoteAddr varAddr{addr, token};
            Variable targetValue{&context};
            CompletedEvent sem{&context};

            CcuRepWriteVarAtomic rep(&insGen, channel, varAddr, targetValue, sem, 0xF);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepWriteVarAtomicTranslate(nullptr, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // SyncAtX: xdId=varAddr, xdtId=varToken, xsId=targetValue, xcId=channel, setCKE=sem/mask
            EXPECT_EQ(instrs[0].v2.syncAtX.xdId, varAddr.addr.Id());
            EXPECT_EQ(instrs[0].v2.syncAtX.xdtId, varAddr.token.Id());
            EXPECT_EQ(instrs[0].v2.syncAtX.xsId, targetValue.Id());
            EXPECT_EQ(instrs[0].v2.syncAtX.xcId, channel.Id());
            EXPECT_EQ(instrs[0].v2.syncAtX.setCKEId, sem.Id());
            EXPECT_EQ(instrs[0].v2.syncAtX.setCKEMask, 0xF);
        }

        /* ---------------- WriteWithCntInc: 构造/translate ---------------- */

        TEST_F(HalfRttRepTest, WriteWithCntInc_ConstructorAndGetters)
        {
            Variable channel{&context};
            Address addr;
            Variable token;
            RemoteAddr rem{addr, token};
            LocalAddr loc{addr, token};
            Variable len{&context};
            RemoteAddr incCnt{addr, token};

            CcuRepWriteWithCntInc rep(&insGen, channel, rem, loc, len, incCnt);
            EXPECT_EQ(rep.Type(), CcuRepType::WRITE_WITH_CNT_INC);
            EXPECT_EQ(rep.GetRemAddrId(), rem.addr.Id());
            EXPECT_EQ(rep.GetRemTokenId(), rem.token.Id());
            EXPECT_EQ(rep.GetLocAddrId(), loc.addr.Id());
            EXPECT_EQ(rep.GetLocTokenId(), loc.token.Id());
            EXPECT_EQ(rep.GetLenId(), len.Id());
            EXPECT_EQ(rep.GetIncCntAddrId(), incCnt.addr.Id());
            EXPECT_EQ(rep.GetIncCntTokenId(), incCnt.token.Id());
            EXPECT_EQ(rep.GetChannelVarId(), channel.Id());
        }

        TEST_F(HalfRttRepTest, WriteWithCntIncTranslate_GeneratesTransMemInstr)
        {
            Address addr;
            Variable token;
            Variable channel{&context};
            Variable len{&context};
            RemoteAddr incCnt{addr, token};
            RemoteAddr rem{addr, token};
            LocalAddr loc{addr, token};

            CcuRepWriteWithCntInc rep(&insGen, channel, rem, loc, len, incCnt);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepWriteWithCntIncTranslate(nullptr, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // TransMem 搬运字段: dst=rem, src=loc, len, channel
            EXPECT_EQ(instrs[0].v2.transMem.xdId, rem.addr.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xdtId, rem.token.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xsId, loc.addr.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xstId, loc.token.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xlId, len.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xcId, channel.Id());
            // notify 字段: incCnt 地址/token + 每次递增 1
            EXPECT_EQ(instrs[0].v2.transMem.xnId, incCnt.addr.Id());
            EXPECT_EQ(instrs[0].v2.transMem.xntId, incCnt.token.Id());
            EXPECT_EQ(instrs[0].v2.transMem.value, 1u);
            // config 字段: 写带计数递增 opcode 与按包切分模式(用字面量固化指令语义)
            EXPECT_EQ(instrs[0].v2.transMem.dmaOpCode, 0x1A);
            EXPECT_EQ(instrs[0].v2.transMem.splitMode, 1u);
        }

        /* ---------------- CascCntWait/CascCntClear: 构造 + 占位 translate ---------------- */

        TEST_F(HalfRttRepTest, CascCntWait_ConstructorAndGetters)
        {
            HcommCcuCascCntHandle cntHandle = 0x1234;
            CcuRepCascCntWait rep(&insGen, cntHandle, 100);
            EXPECT_EQ(rep.Type(), CcuRepType::CASC_CNT_WAIT);
            EXPECT_EQ(rep.GetCntHandle(), cntHandle);
            EXPECT_EQ(rep.GetOutCntTarget(), 100u);
        }

        TEST_F(HalfRttRepTest, CascCntWait_Describe)
        {
            CcuRepCascCntWait rep(&insGen, 0x1234, 100);
            std::string desc = rep.Describe();
            // 与 ccu_rep_casc_cnt_wait.cc 实现保持一致
            EXPECT_TRUE(desc.find("Wait Cascade OutCounter") != std::string::npos);
            EXPECT_TRUE(desc.find("counterHandle[4660]") != std::string::npos); // 0x1234
            EXPECT_TRUE(desc.find("targetValue[100]") != std::string::npos);
        }

        TEST_F(HalfRttRepTest, CascCntWaitTranslate_GeneratesLoadImdAndWaitInstr)
        {
            // 注入测试数据: wishCntXn[0~1021], totalCntXn=1022, expectedCntXn=1023(控制面分配惯例)
            HcommCcuCascCntHandle handle = InjectCascCntBlock(0, 1021, 1022, 1023);
            CcuRepCascCntWait rep(&insGen, handle, 100);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepCascCntWaitTranslate(&kernel, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // 第 1 条 LoadImdToXn: 将 outCntTarget(100) 写入 expectedCntXn(1023) 寄存器
            EXPECT_EQ(instrs[0].v2.loadImdToX.xnId, 1023u);
            EXPECT_EQ(instrs[0].v2.loadImdToX.immediate, static_cast<uint64_t>(100));
            // 第 2 条 Wait: 等待 totalCntXn(1022) == expectedCntXn(1023)
            EXPECT_EQ(instrs[1].v2.wait.conditionXnId, 1022u);
            EXPECT_EQ(instrs[1].v2.wait.expectedXnId, 1023u);
        }

        TEST_F(HalfRttRepTest, CascCntClear_ConstructorAndGetters)
        {
            HcommCcuCascCntHandle cntHandle = 0x5678;
            CompletedEvent sem{&context};

            CcuRepCascCntClear rep(&insGen, cntHandle, sem, 0xF);
            EXPECT_EQ(rep.Type(), CcuRepType::CASC_CNT_CLEAR);
            EXPECT_EQ(rep.GetCntHandle(), cntHandle);
            EXPECT_EQ(rep.GetSemId(), sem.Id());
            EXPECT_EQ(rep.GetMask(), 0xF);
        }

        TEST_F(HalfRttRepTest, CascCntClear_Describe)
        {
            CompletedEvent sem{&context};
            CcuRepCascCntClear rep(&insGen, 0x5678, sem, 0xF);
            std::string desc = rep.Describe();
            // 与 ccu_rep_casc_cnt_clear.cc 实现保持一致
            EXPECT_TRUE(desc.find("Clear Cascade Counters") != std::string::npos);
            EXPECT_TRUE(desc.find("counterHandle[22136]") != std::string::npos); // 0x5678
            EXPECT_TRUE(desc.find("mask[000f]") != std::string::npos);
        }

        TEST_F(HalfRttRepTest, CascCntClearTranslate_GeneratesClearXInstr)
        {
            // 注入测试数据: wishCntXn[0~1021], totalCntXn=1022, expectedCntXn=1023(控制面分配惯例, 与 CascCntWait
            // 用例一致)
            HcommCcuCascCntHandle handle = InjectCascCntBlock(0, 1021, 1022, 1023);
            CompletedEvent sem{&context};
            CcuRepCascCntClear rep(&insGen, handle, sem, 0xF);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            HcclResult ret = insGen.CcuRepCascCntClearTranslate(&kernel, instr, instrId, &rep, dep);
            EXPECT_EQ(ret, HcclResult::HCCL_SUCCESS);
            // 一条 ClearX 清除整个计数块区间 [wishCntXns.first(0), expectedCntXn(1023)]
            // (覆盖 1022 个 wish + totalCntXn + expectedCntXn, 见 b99367cf0 fix clearX xn range),
            // setCKE 取 rep 用户传入的 sem/mask
            EXPECT_EQ(instrs[0].v2.clearX.xnId, 0u);
            EXPECT_EQ(instrs[0].v2.clearX.xmId, 1023u);
            EXPECT_EQ(instrs[0].v2.clearX.setCKEId, sem.Id());
            EXPECT_EQ(instrs[0].v2.clearX.setCKEMask, 0xF);
        }

        /* ---------------- rep 层 Translate 端到端 ---------------- */

        TEST_F(HalfRttRepTest, LoadAddImm_RepTranslateEndToEnd)
        {
            Variable src{&context};
            Variable srcOffset{&context};
            Variable dst{&context};

            CcuRepLoadAddImm rep(&insGen, src, 2, srcOffset, 0x55, dst);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            EXPECT_TRUE(rep.Translate(nullptr, instr, instrId, dep));
            EXPECT_TRUE(rep.Translated());
            // instrCount = 1, 翻译后 instrId 推进 1
            EXPECT_EQ(instrId, 1u);
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdId, dst.Id());
        }

        TEST_F(HalfRttRepTest, StoreAddImm_RepTranslateEndToEnd)
        {
            Variable dst{&context};
            Variable dstOffset{&context};
            Variable src{&context};

            CcuRepStoreAddImm rep(&insGen, dst, 3, dstOffset, 0x66, src);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            EXPECT_TRUE(rep.Translate(nullptr, instr, instrId, dep));
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 1u);
            EXPECT_EQ(instrs[0].v2.loadStoreX.xdId, dst.Id());
        }

        TEST_F(HalfRttRepTest, WriteWithCntInc_RepTranslateEndToEnd)
        {
            Variable channel{&context};
            Address addr;
            Variable token;
            RemoteAddr rem{addr, token};
            LocalAddr loc{addr, token};
            Variable len{&context};
            RemoteAddr incCnt{addr, token};

            CcuRepWriteWithCntInc rep(&insGen, channel, rem, loc, len, incCnt);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            EXPECT_TRUE(rep.Translate(nullptr, instr, instrId, dep));
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 1u);
            EXPECT_EQ(instrs[0].v2.transMem.xnId, incCnt.addr.Id());
        }

        TEST_F(HalfRttRepTest, CascCntWait_RepTranslateEndToEnd)
        {
            // rep 层 Translate 端到端: 注入数据后走真实 translate, instrId 按 instrCount(=2) 推进
            HcommCcuCascCntHandle handle = InjectCascCntBlock(0, 1021, 1022, 1023);
            CcuRepCascCntWait rep(&insGen, handle, 100);
            CcuInstr instrs[4] = {};
            CcuInstr* instr = instrs;
            uint16_t instrId = 0;
            TransDep dep = {};

            EXPECT_TRUE(rep.Translate(&kernel, instr, instrId, dep));
            EXPECT_TRUE(rep.Translated());
            EXPECT_EQ(instrId, 2u);
            // 两条指令内容: LoadImdToXn 写 expectedCntXn + Wait 比较
            EXPECT_EQ(instrs[0].v2.loadImdToX.xnId, 1023u);
            EXPECT_EQ(instrs[1].v2.wait.conditionXnId, 1022u);
        }

    }; // namespace
}; // namespace CcuRep
}; // namespace hcomm
