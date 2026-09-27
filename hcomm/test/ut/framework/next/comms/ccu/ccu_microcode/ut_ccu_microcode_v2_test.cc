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
#include <gtest/gtest.h>

namespace hcomm {
namespace CcuRep {
    namespace CcuV2 {
        namespace {

            class CcuMicrocodeV2Test : public ::testing::Test {
            protected:
                CcuInstr instr{};
                void SetUp() override { memset(&instr, 0, sizeof(instr)); }
            };

            TEST_F(CcuMicrocodeV2Test, LoadSqeArgsToX)
            {
                uint16_t xnId = 3;
                uint16_t sqeArgsId = 7;
                uint16_t setCKEId = 1;
                uint16_t setCKEMask = 0xF;
                LoadSqeArgsToX(&instr, xnId, sqeArgsId, setCKEId, setCKEMask);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0x1);
                EXPECT_EQ(instr.v2.loadSqeArgsToX.xnId, xnId);
                EXPECT_EQ(instr.v2.loadSqeArgsToX.sqeArgsId, sqeArgsId);
                EXPECT_EQ(instr.v2.loadSqeArgsToX.setCKEId, setCKEId);
                EXPECT_EQ(instr.v2.loadSqeArgsToX.setCKEMask, setCKEMask);
            }

            TEST_F(CcuMicrocodeV2Test, LoadImdToXn)
            {
                uint16_t xnId = 4;
                uint64_t immediate = 0xDEADBEEF;
                uint16_t setCKEId = 2;
                uint16_t setCKEMask = 0xA;
                LoadImdToXn(&instr, xnId, immediate, setCKEId, setCKEMask);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0x2);
                EXPECT_EQ(instr.v2.loadImdToX.xnId, xnId);
                EXPECT_EQ(instr.v2.loadImdToX.immediate, immediate);
                EXPECT_EQ(instr.v2.loadImdToX.setCKEId, setCKEId);
                EXPECT_EQ(instr.v2.loadImdToX.setCKEMask, setCKEMask);
            }

            TEST_F(CcuMicrocodeV2Test, Nop)
            {
                Nop(&instr);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0x9);
            }

            TEST_F(CcuMicrocodeV2Test, Assign)
            {
                uint16_t result = 5;
                uint16_t operand = 3;
                Assign(&instr, result, operand, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xD);
                EXPECT_EQ(instr.v2.operate.parMode, 0);
                EXPECT_EQ(instr.v2.operate.xdId, result);
                EXPECT_EQ(instr.v2.operate.xnId, operand);
                EXPECT_EQ(instr.v2.operate.xmId, 0);
            }

            TEST_F(CcuMicrocodeV2Test, Add)
            {
                uint16_t result = 1;
                uint16_t operand1 = 2;
                uint16_t operand2 = 3;
                Add(&instr, result, operand1, operand2, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xD);
                EXPECT_EQ(instr.v2.operate.parMode, 1);
                EXPECT_EQ(instr.v2.operate.xdId, result);
                EXPECT_EQ(instr.v2.operate.xnId, operand1);
                EXPECT_EQ(instr.v2.operate.xmId, operand2);
            }

            TEST_F(CcuMicrocodeV2Test, AddI)
            {
                uint16_t result = 1;
                uint16_t operand = 2;
                uint16_t imm = 10;
                AddI(&instr, result, operand, imm, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xD);
                EXPECT_EQ(instr.v2.operate.parMode, 0);
                EXPECT_EQ(instr.v2.operate.xdId, result);
                EXPECT_EQ(instr.v2.operate.xnId, operand);
                EXPECT_EQ(instr.v2.operate.xmId, imm);
            }

            TEST_F(CcuMicrocodeV2Test, Mul)
            {
                uint16_t result = 2;
                uint16_t operand1 = 3;
                uint16_t operand2 = 4;
                Mul(&instr, result, operand1, operand2, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xF);
                EXPECT_EQ(instr.v2.operate.parMode, 1);
                EXPECT_EQ(instr.v2.operate.xdId, result);
            }

            TEST_F(CcuMicrocodeV2Test, MulI)
            {
                uint16_t result = 2;
                uint16_t operand = 3;
                uint16_t imm = 5;
                MulI(&instr, result, operand, imm, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xF);
                EXPECT_EQ(instr.v2.operate.parMode, 0);
                EXPECT_EQ(instr.v2.operate.xmId, imm);
            }

            TEST_F(CcuMicrocodeV2Test, LoadXFromMem)
            {
                CacheConfig config;
                config.allocHint = 1;
                config.victimHint = 2;
                LoadXFromMem(&instr, 1, 2, 3, 4, config, 5, 0xF);

                EXPECT_EQ(instr.v2.load.dstType, 0x0);
            }

            TEST_F(CcuMicrocodeV2Test, StoreXToMem)
            {
                CacheConfig config;
                config.allocHint = 2;
                config.victimHint = 1;
                StoreXToMem(&instr, 1, 2, 3, 4, config, 5, 0xF);

                EXPECT_EQ(instr.header.type, 0x0);
                EXPECT_EQ(instr.header.code, 0xB);
                EXPECT_EQ(instr.v2.store.xdId, 1);
                EXPECT_EQ(instr.v2.store.xdtId, 2);
                EXPECT_EQ(instr.v2.store.xsId, 3);
                EXPECT_EQ(instr.v2.store.xlId, 4);
                EXPECT_EQ(instr.v2.store.srcType, 0x0);
                EXPECT_EQ(instr.v2.store.allocHint, config.allocHint & 0x3);
                EXPECT_EQ(instr.v2.store.victimHint, config.victimHint & 0x3);
            }

            TEST_F(CcuMicrocodeV2Test, Loop)
            {
                uint16_t startInstrId = 10;
                uint16_t endInstrId = 20;
                uint16_t iterNum = 5;
                uint16_t offset = 100;
                uint16_t contextId = 3;
                Loop(&instr, startInstrId, endInstrId, iterNum, offset, contextId);

                EXPECT_EQ(instr.header.type, 0x1);
                EXPECT_EQ(instr.header.code, 0x0);
                EXPECT_EQ(instr.v2.loop.startInstrId, startInstrId);
                EXPECT_EQ(instr.v2.loop.endInstrId, endInstrId);
                EXPECT_EQ(instr.v2.loop.xmId, iterNum);
                EXPECT_EQ(instr.v2.loop.xnId, offset);
                EXPECT_EQ(instr.v2.loop.xpId, contextId);
                EXPECT_EQ(instr.v2.loop.mode, 0);
                EXPECT_EQ(instr.v2.loop.wishCKEBit, 0);
            }

            TEST_F(CcuMicrocodeV2Test, LoopGroup)
            {
                uint16_t startLoopInstrId = 5;
                uint16_t loopGroupConfig = 0x123;
                uint16_t resOffset = 0x456;
                uint16_t xnOffset = 0x789;
                LoopGroup(&instr, startLoopInstrId, loopGroupConfig, resOffset, xnOffset);

                EXPECT_EQ(instr.header.type, 0x1);
                EXPECT_EQ(instr.header.code, 0x1);
                EXPECT_EQ(instr.v2.loopGroup.startLoopInstrId, startLoopInstrId);
                EXPECT_EQ(instr.v2.loopGroup.xnId, loopGroupConfig);
                EXPECT_EQ(instr.v2.loopGroup.xmId, resOffset);
                EXPECT_EQ(instr.v2.loopGroup.xpId, xnOffset);
            }

            TEST_F(CcuMicrocodeV2Test, SetCKE)
            {
                uint16_t setCKEId = 1;
                uint16_t setCKEMask = 0xF;
                uint16_t waitCKEId = 2;
                uint16_t waitCKEMask = 0xA;
                uint16_t clearType = 1;
                SetCKE(&instr, setCKEId, setCKEMask, waitCKEId, waitCKEMask, clearType);

                EXPECT_EQ(instr.header.type, 0x1);
                EXPECT_EQ(instr.header.code, 0x2);
                EXPECT_EQ(instr.v2.setCKE.setCKEId, setCKEId);
                EXPECT_EQ(instr.v2.setCKE.setCKEMask, setCKEMask);
                EXPECT_EQ(instr.v2.setCKE.waitCKEId, waitCKEId);
                EXPECT_EQ(instr.v2.setCKE.waitCKEMask, waitCKEMask);
                EXPECT_EQ(instr.v2.setCKE.clearType, clearType & 0x1);
            }

            TEST_F(CcuMicrocodeV2Test, ClearCKE)
            {
                uint16_t clearCKEId = 3;
                uint16_t clearMask = 0x5;
                uint16_t waitCKEId = 1;
                uint16_t waitCKEMask = 0x3;
                uint16_t clearType = 0;
                ClearCKE(&instr, clearCKEId, clearMask, waitCKEId, waitCKEMask, clearType);

                EXPECT_EQ(instr.header.type, 0x1);
                EXPECT_EQ(instr.header.code, 0x4);
                EXPECT_EQ(instr.v2.clearCKE.clearCKEId, clearCKEId);
                EXPECT_EQ(instr.v2.clearCKE.clearMask, clearMask);
                EXPECT_EQ(instr.v2.clearCKE.clearType, clearType & 0x1);
            }

            TEST_F(CcuMicrocodeV2Test, Jump)
            {
                uint16_t relTarInstrXnId = 10;
                uint16_t conditionXnId = 5;
                uint16_t expectedXnId = 3;
                uint16_t conditionType = 0x5;
                Jump(&instr, relTarInstrXnId, conditionXnId, expectedXnId, conditionType);

                EXPECT_EQ(instr.header.type, 0x1);
                EXPECT_EQ(instr.header.code, 0x5);
                EXPECT_EQ(instr.v2.jmp.expectedXnId, expectedXnId);
                EXPECT_EQ(instr.v2.jmp.conditionXnId, conditionXnId);
                EXPECT_EQ(instr.v2.jmp.relTarInstrXnId, relTarInstrXnId);
                EXPECT_EQ(instr.v2.jmp.conditionType, conditionType & 0xF);
            }

            TEST_F(CcuMicrocodeV2Test, TransLocMemToLocMS)
            {
                CacheConfig config;
                config.allocHint = 1;
                config.victimHint = 2;
                TransLocMemToLocMS(&instr, 1, 2, 3, 4, 5, 6, 0xF, config);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0x0);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.msId, 1);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.xsId, 2);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.xstId, 3);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.xlId, 4);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.xoId, 5);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.allocHint, config.allocHint & 0x3);
                EXPECT_EQ(instr.v2.transLocMemToLocMS.victimHint, config.victimHint & 0x3);
            }

            TEST_F(CcuMicrocodeV2Test, TransLocMSToLocMem)
            {
                CacheConfig config;
                config.allocHint = 2;
                config.victimHint = 1;
                TransLocMSToLocMem(&instr, 1, 2, 3, 4, 5, 6, 0xF, config);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0x2);
                EXPECT_EQ(instr.v2.transLocMSToLocMem.xdId, 1);
                EXPECT_EQ(instr.v2.transLocMSToLocMem.xdtId, 2);
                EXPECT_EQ(instr.v2.transLocMSToLocMem.msId, 3);
                EXPECT_EQ(instr.v2.transLocMSToLocMem.xlId, 4);
                EXPECT_EQ(instr.v2.transLocMSToLocMem.xoId, 5);
            }

            TEST_F(CcuMicrocodeV2Test, TransLocMemToLocMem)
            {
                CacheConfig srcConfig;
                srcConfig.allocHint = 1;
                srcConfig.victimHint = 1;
                CacheConfig dstConfig;
                dstConfig.allocHint = 2;
                dstConfig.victimHint = 2;
                TransLocMemToLocMem(&instr, 1, 2, 3, 4, 5, 6, 7, 0xF, srcConfig, dstConfig);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0x6);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.xdId, 1);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.xdtId, 2);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.xsId, 3);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.xstId, 4);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.xlId, 5);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.usedMSId, 6);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.msNum, CCU_MS_INTERLEAVE);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.srcAllocHint, srcConfig.allocHint & 0x3);
                EXPECT_EQ(instr.v2.transLocMemToLocMem.dstAllocHint, dstConfig.allocHint & 0x3);
            }

            TEST_F(CcuMicrocodeV2Test, TransMem)
            {
                TransMemNotifyInfo notify;
                notify.xnId = 1;
                notify.xntId = 2;
                notify.value = 0x123;

                TransMemReduceInfo reduce;
                reduce.udfType = 0x10;
                reduce.reduceDataType = 0x5;
                reduce.reduceOpCode = 0x3;

                TransMemConfig config;
                config.order = 0x2;
                config.fence = 1;
                config.cqe = 0;
                config.nf = 1;
                config.udfEnable = 0;
                config.splitMode = 1;
                config.se = 0;
                config.rmtJettyType = 0x2;

                TransMem(&instr, 1, 2, 3, 4, 5, 6, notify, reduce, config, 7, 0xF);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0x10);
                EXPECT_EQ(instr.v2.transMem.xdId, 1);
                EXPECT_EQ(instr.v2.transMem.xdtId, 2);
                EXPECT_EQ(instr.v2.transMem.xsId, 3);
                EXPECT_EQ(instr.v2.transMem.xstId, 4);
                EXPECT_EQ(instr.v2.transMem.xlId, 5);
                EXPECT_EQ(instr.v2.transMem.xcId, 6);
                EXPECT_EQ(instr.v2.transMem.xnId, notify.xnId);
                EXPECT_EQ(instr.v2.transMem.udfType, reduce.udfType & 0xFF);
                EXPECT_EQ(instr.v2.transMem.reduceDataType, reduce.reduceDataType & 0xF);
                EXPECT_EQ(instr.v2.transMem.reduceOpCode, reduce.reduceOpCode & 0xF);
                EXPECT_EQ(instr.v2.transMem.order, config.order & 0x7);
                EXPECT_EQ(instr.v2.transMem.fence, config.fence & 0x1);
                EXPECT_EQ(instr.v2.transMem.cqe, config.cqe & 0x1);
                EXPECT_EQ(instr.v2.transMem.nf, config.nf & 0x1);
                EXPECT_EQ(instr.v2.transMem.udfEnable, config.udfEnable & 0x1);
                EXPECT_EQ(instr.v2.transMem.splitMode, config.splitMode & 0x1);
                EXPECT_EQ(instr.v2.transMem.se, config.se & 0x1);
                EXPECT_EQ(instr.v2.transMem.rmtJettyType, config.rmtJettyType & 0x3);
            }

            TEST_F(CcuMicrocodeV2Test, SyncWtX_NotifyOnly)
            {
                TransMemNotifyInfo notify;
                notify.xnId = 1;
                notify.xntId = 2;
                notify.value = 0x123;
                SyncWtX(&instr, notify, 5, 6, 0xF);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0xD);
                EXPECT_EQ(instr.v2.syncWtX.notifyValid, 0);
                EXPECT_EQ(instr.v2.syncWtX.parMode, 0);
            }

            TEST_F(CcuMicrocodeV2Test, SyncWtX_WithXn)
            {
                TransMemNotifyInfo notify;
                notify.xnId = 10;
                notify.xntId = 11;
                notify.value = 0x123;
                SyncWtX(&instr, 1, 2, 3, 4, notify, 5, 0xF);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0xD);
                EXPECT_EQ(instr.v2.syncWtX.xdId, 1);
                EXPECT_EQ(instr.v2.syncWtX.xdtId, 2);
                EXPECT_EQ(instr.v2.syncWtX.xsId, 3);
                EXPECT_EQ(instr.v2.syncWtX.xcId, 4);
                EXPECT_EQ(instr.v2.syncWtX.notifyValid, 1);
                EXPECT_EQ(instr.v2.syncWtX.parMode, 1);
            }

            TEST_F(CcuMicrocodeV2Test, SyncWtX_FullNotify)
            {
                TransMemNotifyInfo notify;
                notify.xnId = 1;
                notify.xntId = 2;
                notify.value = 0x123;
                SyncWtX(&instr, 3, 4, 5, 6, notify, 7, 0xF);

                EXPECT_EQ(instr.v2.syncWtX.xdId, 3);
                EXPECT_EQ(instr.v2.syncWtX.xdtId, 4);
                EXPECT_EQ(instr.v2.syncWtX.xsId, 5);
                EXPECT_EQ(instr.v2.syncWtX.xcId, 6);
                EXPECT_EQ(instr.v2.syncWtX.xnId, notify.xnId);
                EXPECT_EQ(instr.v2.syncWtX.xntId, notify.xntId);
                EXPECT_EQ(instr.v2.syncWtX.value, notify.value);
                EXPECT_EQ(instr.v2.syncWtX.notifyValid, 1);
                EXPECT_EQ(instr.v2.syncWtX.parMode, 1);
            }

            TEST_F(CcuMicrocodeV2Test, SyncAtX)
            {
                SyncAtX(&instr, 1, 2, 3, 4, 5, 0xF);

                EXPECT_EQ(instr.header.type, 0x2);
                EXPECT_EQ(instr.header.code, 0xE);
                EXPECT_EQ(instr.v2.syncAtX.xdId, 1);
                EXPECT_EQ(instr.v2.syncAtX.xdtId, 2);
                EXPECT_EQ(instr.v2.syncAtX.xsId, 3);
                EXPECT_EQ(instr.v2.syncAtX.xcId, 4);
                EXPECT_EQ(instr.v2.syncAtX.parMode, 1);
            }

            TEST_F(CcuMicrocodeV2Test, ReduceAdd)
            {
                uint16_t ms[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
                uint16_t count = 4;
                uint16_t castEn = 2;
                uint16_t dataType = 3;
                ReduceAdd(&instr, ms, count, castEn, dataType, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x3);
                EXPECT_EQ(instr.header.code, 0x0);
                EXPECT_EQ(instr.v2.reduce.count, (count - 2) & 0x7);
                EXPECT_EQ(instr.v2.reduce.castEn, castEn & 0x3);
                EXPECT_EQ(instr.v2.reduce.dataType, dataType & 0x1f);
            }

            TEST_F(CcuMicrocodeV2Test, ReduceMax)
            {
                uint16_t ms[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
                uint16_t count = 5;
                uint16_t dataType = 4;
                ReduceMax(&instr, ms, count, dataType, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x3);
                EXPECT_EQ(instr.header.code, 0x1);
                EXPECT_EQ(instr.v2.reduce.count, (count - 2) & 0x7);
                EXPECT_EQ(instr.v2.reduce.dataType, dataType & 0x1f);
                EXPECT_EQ(instr.v2.reduce.castEn, 0);
            }

            TEST_F(CcuMicrocodeV2Test, ReduceMin)
            {
                uint16_t ms[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
                uint16_t count = 6;
                uint16_t dataType = 5;
                ReduceMin(&instr, ms, count, dataType, 1, 0xF);

                EXPECT_EQ(instr.header.type, 0x3);
                EXPECT_EQ(instr.header.code, 0x2);
                EXPECT_EQ(instr.v2.reduce.count, (count - 2) & 0x7);
                EXPECT_EQ(instr.v2.reduce.dataType, dataType & 0x1f);
            }

        } // namespace
    } // namespace CcuV2
} // namespace CcuRep
} // namespace hcomm
