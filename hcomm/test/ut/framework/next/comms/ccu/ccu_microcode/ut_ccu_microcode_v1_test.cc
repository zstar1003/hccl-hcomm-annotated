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
#include <string>

namespace hcomm {
namespace CcuRep {
    namespace {

        class CcuMicrocodeV1Test : public ::testing::Test {
        protected:
            CcuInstr instr{};
            void SetUp() override { memset(&instr, 0, sizeof(instr)); }
        };

        TEST_F(CcuMicrocodeV1Test, LoadSqeArgsToGSAInstr)
        {
            uint16_t gsaId = 5;
            uint16_t sqeArgsId = 10;
            LoadSqeArgsToGSAInstr(&instr, gsaId, sqeArgsId);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x0);
            EXPECT_EQ(instr.v1.loadSqeArgsToGSA.gsaId, gsaId);
            EXPECT_EQ(instr.v1.loadSqeArgsToGSA.sqeArgsId, sqeArgsId);
        }

        TEST_F(CcuMicrocodeV1Test, LoadSqeArgsToXnInstr)
        {
            uint16_t xnId = 3;
            uint16_t sqeArgsId = 7;
            LoadSqeArgsToXnInstr(&instr, xnId, sqeArgsId);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x1);
            EXPECT_EQ(instr.v1.loadSqeArgsToXn.xnId, xnId);
            EXPECT_EQ(instr.v1.loadSqeArgsToXn.sqeArgsId, sqeArgsId);
        }

        TEST_F(CcuMicrocodeV1Test, LoadImdToGSAInstr)
        {
            uint16_t gsaId = 8;
            uint64_t immediate = 0x12345678ABCD;
            LoadImdToGSAInstr(&instr, gsaId, immediate);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x2);
            EXPECT_EQ(instr.v1.loadImdToGSA.gsaId, gsaId);
            EXPECT_EQ(instr.v1.loadImdToGSA.immediate, immediate);
        }

        TEST_F(CcuMicrocodeV1Test, LoadImdToXnInstr_Normal)
        {
            uint16_t xnId = 4;
            uint64_t immediate = 0xDEADBEEF;
            uint16_t secFlag = 0;
            LoadImdToXnInstr(&instr, xnId, immediate, secFlag);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x3);
            EXPECT_EQ(instr.v1.loadImdToXn.xnId, xnId);
            EXPECT_EQ(instr.v1.loadImdToXn.immediate, immediate);
            EXPECT_EQ(instr.v1.loadImdToXn.secFlag, secFlag);
        }

        TEST_F(CcuMicrocodeV1Test, LoadImdToXnInstr_SecFlag)
        {
            uint16_t xnId = 6;
            uint64_t immediate = 0x12345;
            uint16_t secFlag = CCU_LOAD_TO_XN_SEC_INFO;
            LoadImdToXnInstr(&instr, xnId, immediate, secFlag);

            EXPECT_EQ(instr.v1.loadImdToXn.secFlag, CCU_LOAD_TO_XN_SEC_INFO);
        }

        TEST_F(CcuMicrocodeV1Test, LoadGSAXnInstr)
        {
            uint16_t gsAdId = 1;
            uint16_t gsAmId = 2;
            uint16_t xnId = 3;
            LoadGSAXnInstr(&instr, gsAdId, gsAmId, xnId);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x4);
            EXPECT_EQ(instr.v1.loadGSAXn.gsAdId, gsAdId);
            EXPECT_EQ(instr.v1.loadGSAXn.gsAmId, gsAmId);
            EXPECT_EQ(instr.v1.loadGSAXn.xnId, xnId);
        }

        TEST_F(CcuMicrocodeV1Test, LoadGSAGSAInstr)
        {
            uint16_t gsAdId = 1;
            uint16_t gsAmId = 2;
            uint16_t gsAnId = 3;
            LoadGSAGSAInstr(&instr, gsAdId, gsAmId, gsAnId);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x5);
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAdId, gsAdId);
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAmId, gsAmId);
            EXPECT_EQ(instr.v1.loadGSAGSA.gsAnId, gsAnId);
        }

        TEST_F(CcuMicrocodeV1Test, LoadXXInstr)
        {
            uint16_t xdId = 1;
            uint16_t xmId = 2;
            uint16_t xnId = 3;
            LoadXXInstr(&instr, xdId, xmId, xnId);

            EXPECT_EQ(instr.header.type, 0x0);
            EXPECT_EQ(instr.header.code, 0x6);
            EXPECT_EQ(instr.v1.loadXX.xdId, xdId);
            EXPECT_EQ(instr.v1.loadXX.xmId, xmId);
            EXPECT_EQ(instr.v1.loadXX.xnId, xnId);
        }

        TEST_F(CcuMicrocodeV1Test, LoopInstr)
        {
            uint16_t startInstrId = 10;
            uint16_t endInstrId = 20;
            uint16_t xnId = 5;
            LoopInstr(&instr, startInstrId, endInstrId, xnId);

            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x0);
            EXPECT_EQ(instr.v1.loop.startInstrId, startInstrId);
            EXPECT_EQ(instr.v1.loop.endInstrId, endInstrId);
            EXPECT_EQ(instr.v1.loop.xnId, xnId);
        }

        TEST_F(CcuMicrocodeV1Test, LoopGroupInstr)
        {
            uint16_t startLoopInstrId = 5;
            uint16_t xnId = 2;
            uint16_t xmId = 3;
            uint16_t highPerfModeEn = 1;
            LoopGroupInstr(&instr, startLoopInstrId, xnId, xmId, highPerfModeEn);

            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x1);
            EXPECT_EQ(instr.v1.loopGroup.startLoopInstrId, startLoopInstrId);
            EXPECT_EQ(instr.v1.loopGroup.xnId, xnId);
            EXPECT_EQ(instr.v1.loopGroup.xmId, xmId);
            EXPECT_EQ(instr.v1.loopGroup.highPerfModeEn, highPerfModeEn & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, LoopGroupInstr_HighPerfModeZero)
        {
            uint16_t highPerfModeEn = 0xFE;
            LoopGroupInstr(&instr, 0, 0, 0, highPerfModeEn);
            EXPECT_EQ(instr.v1.loopGroup.highPerfModeEn, 0);
        }

        TEST_F(CcuMicrocodeV1Test, SetCKEInstr)
        {
            uint16_t setCKEId = 1;
            uint16_t setCKEMask = 0xF;
            uint16_t waitCKEId = 2;
            uint16_t waitCKEMask = 0xA;
            uint16_t clearType = 1;
            SetCKEInstr(&instr, setCKEId, setCKEMask, waitCKEId, waitCKEMask, clearType);

            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x2);
            EXPECT_EQ(instr.v1.setCKE.setCKEId, setCKEId);
            EXPECT_EQ(instr.v1.setCKE.setCKEMask, setCKEMask);
            EXPECT_EQ(instr.v1.setCKE.waitCKEId, waitCKEId);
            EXPECT_EQ(instr.v1.setCKE.waitCKEMask, waitCKEMask);
            EXPECT_EQ(instr.v1.setCKE.clearType, clearType & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, ClearCKEInstr)
        {
            uint16_t clearCKEId = 3;
            uint16_t clearMask = 0x5;
            uint16_t waitCKEId = 1;
            uint16_t waitCKEMask = 0x3;
            uint16_t clearType = 0;
            ClearCKEInstr(&instr, clearCKEId, clearMask, waitCKEId, waitCKEMask, clearType);

            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x4);
            EXPECT_EQ(instr.v1.clearCKE.clearCKEId, clearCKEId);
            EXPECT_EQ(instr.v1.clearCKE.clearMask, clearMask);
            EXPECT_EQ(instr.v1.clearCKE.clearType, clearType & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, JumpInstr)
        {
            uint16_t dstInstrXnId = 10;
            uint16_t conditionXnId = 5;
            uint64_t expectData = 0x123;
            JumpInstr(&instr, dstInstrXnId, conditionXnId, expectData);

            EXPECT_EQ(instr.header.type, 0x1);
            EXPECT_EQ(instr.header.code, 0x5);
            EXPECT_EQ(instr.v1.jmp.dstInstrXnId, dstInstrXnId);
            EXPECT_EQ(instr.v1.jmp.conditionXnId, conditionXnId);
            EXPECT_EQ(instr.v1.jmp.expectData, expectData);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMemToLocMSInstr)
        {
            TransLocMemToLocMSInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 1, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x0);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.locMSId, 1);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.locGSAId, 2);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.locXnId, 3);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.lengthXnId, 4);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.channelId, 5);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.clearType, 1 & 0x1);
            EXPECT_EQ(instr.v1.transLocMemToLocMS.lengthEn, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, TransRmtMemToLocMSInstr)
        {
            TransRmtMemToLocMSInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 0, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x1);
            EXPECT_EQ(instr.v1.transRmtMemToLocMS.clearType, 0 & 0x1);
            EXPECT_EQ(instr.v1.transRmtMemToLocMS.lengthEn, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMSToLocMemInstr)
        {
            TransLocMSToLocMemInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 1, 0);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x2);
            EXPECT_EQ(instr.v1.transLocMSToLocMem.locGSAId, 1);
            EXPECT_EQ(instr.v1.transLocMSToLocMem.locXnId, 2);
            EXPECT_EQ(instr.v1.transLocMSToLocMem.locMSId, 3);
            EXPECT_EQ(instr.v1.transLocMSToLocMem.clearType, 1 & 0x1);
            EXPECT_EQ(instr.v1.transLocMSToLocMem.lengthEn, 0 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMSToRmtMemInstr)
        {
            TransLocMSToRmtMemInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 1, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x3);
            EXPECT_EQ(instr.v1.transLocMSToRmtMem.rmtGSAId, 1);
            EXPECT_EQ(instr.v1.transLocMSToRmtMem.rmtXnId, 2);
            EXPECT_EQ(instr.v1.transLocMSToRmtMem.locMSId, 3);
        }

        TEST_F(CcuMicrocodeV1Test, TransRmtMSToLocMemInstr)
        {
            TransRmtMSToLocMemInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 0, 0);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x4);
            EXPECT_EQ(instr.v1.transRmtMSToLocMem.locGSAId, 1);
            EXPECT_EQ(instr.v1.transRmtMSToLocMem.locXnId, 2);
            EXPECT_EQ(instr.v1.transRmtMSToLocMem.rmtMSId, 3);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMSToLocMSInstr)
        {
            TransLocMSToLocMSInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x5);
            EXPECT_EQ(instr.v1.transLocMSToLocMS.dstMSId, 1);
            EXPECT_EQ(instr.v1.transLocMSToLocMS.srcMSId, 2);
            EXPECT_EQ(instr.v1.transLocMSToLocMS.lengthXnId, 3);
            EXPECT_EQ(instr.v1.transLocMSToLocMS.channelId, 4);
            EXPECT_EQ(instr.v1.transLocMSToLocMS.setCKEId, 5);
        }

        TEST_F(CcuMicrocodeV1Test, TransRmtMSToLocMSInstr)
        {
            TransRmtMSToLocMSInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x6);
            EXPECT_EQ(instr.v1.transRmtMSToLocMS.locMSId, 1);
            EXPECT_EQ(instr.v1.transRmtMSToLocMS.rmtMSId, 2);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMSToRmtMSInstr)
        {
            TransLocMSToRmtMSInstr(&instr, 1, 2, 3, 4, 5, 6, 0xF, 7, 0xA, 8, 0xB, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x7);
            EXPECT_EQ(instr.v1.transLocMSToRmtMS.rmtMSId, 1);
            EXPECT_EQ(instr.v1.transLocMSToRmtMS.locMSId, 2);
            EXPECT_EQ(instr.v1.transLocMSToRmtMS.setRmtCKEId, 5);
            EXPECT_EQ(instr.v1.transLocMSToRmtMS.setRmtCKEMask, 6);
        }

        TEST_F(CcuMicrocodeV1Test, TransRmtMemToLocMemInstr)
        {
            TransRmtMemToLocMemInstr(&instr, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xF, 10, 0xA, 1, 1, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x8);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.locGSAId, 1);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.locXnId, 2);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.rmtGSAId, 3);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.rmtXnId, 4);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.reduceDataType, 7 & 0xf);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.reduceOpCode, 8 & 0xf);
            EXPECT_EQ(instr.v1.transRmtMemToLocMem.reduceEn, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMemToRmtMemInstr)
        {
            TransLocMemToRmtMemInstr(&instr, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0xF, 10, 0xA, 0, 0, 0);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0x9);
            EXPECT_EQ(instr.v1.transLocMemToRmtMem.rmtGSAId, 1);
            EXPECT_EQ(instr.v1.transLocMemToRmtMem.rmtXnId, 2);
            EXPECT_EQ(instr.v1.transLocMemToRmtMem.locGSAId, 3);
            EXPECT_EQ(instr.v1.transLocMemToRmtMem.locXnId, 4);
            EXPECT_EQ(instr.v1.transLocMemToRmtMem.reduceEn, 0 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, TransLocMemToLocMemInstr)
        {
            TransLocMemToLocMemInstr(&instr, 1, 2, 3, 4, 5, 6, 7, 0xF, 8, 0xA, 1, 0);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xa);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.dstGSAId, 1);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.dstXnId, 2);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.srcGSAId, 3);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.srcXnId, 4);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.clearType, 1 & 0x1);
            EXPECT_EQ(instr.v1.transLocMemToLocMem.lengthEn, 0 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, SyncCKEInstr)
        {
            SyncCKEInstr(&instr, 1, 2, 0xF, 3, 4, 0xA, 5, 0xB, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xb);
            EXPECT_EQ(instr.v1.syncCKE.rmtCKEId, 1);
            EXPECT_EQ(instr.v1.syncCKE.locCKEId, 2);
            EXPECT_EQ(instr.v1.syncCKE.locCKEMask, 0xF);
            EXPECT_EQ(instr.v1.syncCKE.channelId, 3);
            EXPECT_EQ(instr.v1.syncCKE.clearType, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, SyncGSAInstr)
        {
            SyncGSAInstr(&instr, 1, 2, 3, 4, 0xF, 5, 0xA, 6, 0xB, 0);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xc);
            EXPECT_EQ(instr.v1.syncGSA.rmtGSAId, 1);
            EXPECT_EQ(instr.v1.syncGSA.locGSAId, 2);
            EXPECT_EQ(instr.v1.syncGSA.channelId, 3);
            EXPECT_EQ(instr.v1.syncGSA.setRmtCKEId, 4);
            EXPECT_EQ(instr.v1.syncGSA.setRmtCKEMask, 0xF);
            EXPECT_EQ(instr.v1.syncGSA.clearType, 0 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, SyncXnInstr)
        {
            SyncXnInstr(&instr, 1, 2, 3, 4, 0xF, 5, 0xA, 6, 0xB, 1);

            EXPECT_EQ(instr.header.type, 0x2);
            EXPECT_EQ(instr.header.code, 0xd);
            EXPECT_EQ(instr.v1.syncXn.rmtXnId, 1);
            EXPECT_EQ(instr.v1.syncXn.locXnId, 2);
            EXPECT_EQ(instr.v1.syncXn.channelId, 3);
            EXPECT_EQ(instr.v1.syncXn.clearType, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, AddInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
            uint16_t count = 4;
            uint16_t castEn = 2;
            uint16_t dataType = 3;
            AddInstr(&instr, msId, count, castEn, dataType, 1, 0xF, 2, 0xA, 1, 5);

            EXPECT_EQ(instr.header.type, 0x3);
            EXPECT_EQ(instr.header.code, 0x0);
            EXPECT_EQ(instr.v1.add.count, (count - 2) & 0x7);
            EXPECT_EQ(instr.v1.add.castEn, castEn & 0x3);
            EXPECT_EQ(instr.v1.add.dataType, dataType & 0x1f);
            EXPECT_EQ(instr.v1.add.clearType, 1 & 0x1);
            EXPECT_EQ(instr.v1.add.XnIdLength, 5);
        }

        TEST_F(CcuMicrocodeV1Test, AddInstr_CountMinValue)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
            uint16_t count = 2;
            AddInstr(&instr, msId, count, 0, 0, 0, 0, 0, 0, 0, 0);
            EXPECT_EQ(instr.v1.add.count, 0);
        }

        TEST_F(CcuMicrocodeV1Test, MaxInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
            uint16_t count = 5;
            uint16_t dataType = 4;
            MaxInstr(&instr, msId, count, dataType, 1, 0xF, 2, 0xA, 1, 3);

            EXPECT_EQ(instr.header.type, 0x3);
            EXPECT_EQ(instr.header.code, 0x1);
            EXPECT_EQ(instr.v1.max.count, (count - 2) & 0x7);
            EXPECT_EQ(instr.v1.max.dataType, dataType & 0x1f);
            EXPECT_EQ(instr.v1.max.clearType, 1 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, MinInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0, 1, 2, 3, 4, 5, 6, 7};
            uint16_t count = 6;
            uint16_t dataType = 5;
            MinInstr(&instr, msId, count, dataType, 1, 0xF, 2, 0xA, 0, 7);

            EXPECT_EQ(instr.header.type, 0x3);
            EXPECT_EQ(instr.header.code, 0x2);
            EXPECT_EQ(instr.v1.min.count, (count - 2) & 0x7);
            EXPECT_EQ(instr.v1.min.dataType, dataType & 0x1f);
            EXPECT_EQ(instr.v1.min.clearType, 0 & 0x1);
        }

        TEST_F(CcuMicrocodeV1Test, ParseLoadSqeArgsToGSAInstr)
        {
            LoadSqeArgsToGSAInstr(&instr, 5, 10);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("GSA[5]"), std::string::npos);
            EXPECT_NE(result.find("SqeArg[10]"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseLoadImdToXnInstr_SecFlag)
        {
            LoadImdToXnInstr(&instr, 3, 0xDEAD, CCU_LOAD_TO_XN_SEC_INFO);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("tokenInfo"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseLoadImdToXnInstr_Normal)
        {
            LoadImdToXnInstr(&instr, 3, 0xDEAD, 0);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Xn[3]"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseLoopInstr)
        {
            LoopInstr(&instr, 10, 20, 5);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("startInstrId[10]"), std::string::npos);
            EXPECT_NE(result.find("endInstrId[20]"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseAddInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0x8000, 0x8001, 0x8002, 0x8003};
            AddInstr(&instr, msId, 4, 1, 2, 0, 0, 0, 0, 0, 0);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Add"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseMaxInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0x8000, 0x8001, 0x8002, 0x8003};
            MaxInstr(&instr, msId, 4, 2, 0, 0, 0, 0, 0, 0);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Max"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseMinInstr)
        {
            uint16_t msId[CCU_REDUCE_MAX_MS] = {0x8000, 0x8001, 0x8002, 0x8003};
            MinInstr(&instr, msId, 4, 2, 0, 0, 0, 0, 0, 0);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Min"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseSetCKEInstr)
        {
            SetCKEInstr(&instr, 1, 0xF, 2, 0xA, 1);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Set CKE"), std::string::npos);
            EXPECT_NE(result.find("clearType[1]"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseClearCKEInstr)
        {
            ClearCKEInstr(&instr, 3, 0x5, 1, 0x3, 0);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Clear CKE"), std::string::npos);
            EXPECT_NE(result.find("clearType[0]"), std::string::npos);
        }

        TEST_F(CcuMicrocodeV1Test, ParseJumpInstr)
        {
            JumpInstr(&instr, 10, 5, 0x123);
            std::string result = ParseInstr(&instr);
            EXPECT_NE(result.find("Jump"), std::string::npos);
        }

    } // namespace
} // namespace CcuRep
} // namespace hcomm
