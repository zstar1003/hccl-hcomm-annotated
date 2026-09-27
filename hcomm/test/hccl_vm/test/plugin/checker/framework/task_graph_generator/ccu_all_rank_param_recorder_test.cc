/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"

#include <cstdint>
#include <gtest/gtest.h>

#include "ccu_all_rank_param_recorder.h"
#include "type_conversion.h"

using namespace HcclSim;

class AllRankParamRecorderTest : public testing::Test {
protected:
    void SetUp() override
    {
        AllRankParamRecorder::Global()->Reset();
        AllRankParamRecorder::Global()->curHBM.clear();
    }
    void TearDown() override
    {
        AllRankParamRecorder::Global()->Reset();
        AllRankParamRecorder::Global()->curHBM.clear();
    }
};

TEST_F(AllRankParamRecorderTest, Global_ReturnsSameInstance)
{
    auto* inst1 = AllRankParamRecorder::Global();
    auto* inst2 = AllRankParamRecorder::Global();
    EXPECT_EQ(inst1, inst2);
}

TEST_F(AllRankParamRecorderTest, InitParam_DoesNotCrash)
{
    EXPECT_NO_THROW(AllRankParamRecorder::Global()->InitParam());
}

TEST_F(AllRankParamRecorderTest, Reset_ClearsAllMaps)
{
    AllRankParamRecorder::Global()->SetXn(0, 0, 1, 0x100);
    AllRankParamRecorder::Global()->SetGSA(0, 0, 1, 0x200);
    AllRankParamRecorder::Global()->SetCKE(0, 0, 1, 3);
    AllRankParamRecorder::Global()->Reset();
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetXn(0, 0, 1, val), HCCL_E_PARA);
    EXPECT_EQ(AllRankParamRecorder::Global()->GetGSA(0, 0, 1, val), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, SetXn_GetXn_Success)
{
    AllRankParamRecorder::Global()->SetXn(1, 2, 3, 0xABCD);
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetXn(1, 2, 3, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 0xABCDu);
}

TEST_F(AllRankParamRecorderTest, GetXn_NotSet_ReturnsError)
{
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetXn(99, 99, 99, val), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, SetXn_OverwriteValue)
{
    AllRankParamRecorder::Global()->SetXn(0, 0, 1, 100);
    AllRankParamRecorder::Global()->SetXn(0, 0, 1, 200);
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetXn(0, 0, 1, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 200u);
}

TEST_F(AllRankParamRecorderTest, SetGSA_GetGSA_Success)
{
    AllRankParamRecorder::Global()->SetGSA(0, 1, 5, 0x1234);
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetGSA(0, 1, 5, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 0x1234u);
}

TEST_F(AllRankParamRecorderTest, GetGSA_NotSet_ReturnsError)
{
    uint64_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetGSA(99, 99, 99, val), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, SetCKE_GetCKE_Success)
{
    AllRankParamRecorder::Global()->SetCKE(0, 0, 10, 5);
    uint16_t val = 0;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetCKE(0, 0, 10, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 5u);
}

TEST_F(AllRankParamRecorderTest, GetCKE_NotSet_ReturnsZero)
{
    uint16_t val = 99;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetCKE(99, 99, 99, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 0u);
}

TEST_F(AllRankParamRecorderTest, SetHBM_GetHBM_Success)
{
    std::vector<uint64_t> data(8, 0);
    data[0] = 0x10;
    data[3] = 0x40;
    EXPECT_EQ(AllRankParamRecorder::Global()->SetHBM(0, 0, 0x1000, data), HCCL_SUCCESS);
    std::vector<uint64_t> out;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetHBM(0, 0, 0x1000, out), HCCL_SUCCESS);
    EXPECT_EQ(out.size(), 8u);
    EXPECT_EQ(out[0], 0x10u);
    EXPECT_EQ(out[3], 0x40u);
}

TEST_F(AllRankParamRecorderTest, SetHBM_UnalignedSize_ReturnsError)
{
    std::vector<uint64_t> data = {0x10, 0x20, 0x30};
    EXPECT_EQ(AllRankParamRecorder::Global()->SetHBM(0, 0, 0x1000, data), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, GetHBM_NotSet_ReturnsError)
{
    std::vector<uint64_t> out;
    EXPECT_EQ(AllRankParamRecorder::Global()->GetHBM(99, 99, 0xFFFF, out), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, SetHBM_EightByteAligned_Success)
{
    std::vector<uint64_t> data(8, 0);
    data[0] = 1;
    data[7] = 8;
    EXPECT_EQ(AllRankParamRecorder::Global()->SetHBM(1, 0, 0x2000, data), HCCL_SUCCESS);
}

TEST_F(AllRankParamRecorderTest, GetXnIdByAddr_ValidDieId_Success)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0, 0x200000};
    uint16_t xnId = 0;
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::XN_A6, 0x100000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(xnId, 0u);
}

TEST_F(AllRankParamRecorderTest, GetXnIdByAddr_OutOfRangeDieId_ReturnsError)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0};
    uint16_t xnId = 0;
    EXPECT_EQ(rec->GetXnIdByAddr(5, CcuComponerntType::XN_A6, 0x100000, xnId), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, GetAddrByXnId_ValidDieId_Success)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0};
    uint64_t addr = 1;
    EXPECT_EQ(rec->GetAddrByXnId(0, CcuComponerntType::XN_A6, 0, addr), HCCL_SUCCESS);
}

TEST_F(AllRankParamRecorderTest, GetAddrByXnId_OutOfRangeDieId_ReturnsError)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0};
    uint64_t addr = 0;
    EXPECT_EQ(rec->GetAddrByXnId(5, CcuComponerntType::XN_A6, 0, addr), HCCL_E_PARA);
}

TEST_F(AllRankParamRecorderTest, GetDevType_Default_ReturnsInvalid)
{
    EXPECT_EQ(AllRankParamRecorder::Global()->GetDevType(), DevType::DEV_TYPE_COUNT);
}

TEST_F(AllRankParamRecorderTest, CheckAllPostMatch_Empty_ReturnsSuccess)
{
    EXPECT_EQ(AllRankParamRecorder::Global()->CheckAllPostMatch(), HCCL_SUCCESS);
}

TEST_F(AllRankParamRecorderTest, MultipleRanksAndDies_SetGetSuccess)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->SetXn(0, 0, 1, 100);
    rec->SetXn(0, 1, 2, 200);
    rec->SetXn(1, 0, 1, 300);
    uint64_t val = 0;
    EXPECT_EQ(rec->GetXn(0, 0, 1, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 100u);
    EXPECT_EQ(rec->GetXn(0, 1, 2, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 200u);
    EXPECT_EQ(rec->GetXn(1, 0, 1, val), HCCL_SUCCESS);
    EXPECT_EQ(val, 300u);
}

TEST_F(AllRankParamRecorderTest, GetXnIdByAddr_AllComponentTypes_Success)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0};
    uint16_t xnId = 0xFFFF;
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::XN_A6, 0x100000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::CKE_A6, 0x140000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::PFE_A6, 0x148000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::CHANNEL_A6, 0x150000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::JETTY_A6, 0x170000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::MISSION_A6, 0x178000, xnId), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetXnIdByAddr(0, CcuComponerntType::LOOP_A6, 0x180000, xnId), HCCL_SUCCESS);
}

TEST_F(AllRankParamRecorderTest, GetAddrByXnId_AllComponentTypes_Success)
{
    auto* rec = AllRankParamRecorder::Global();
    rec->ccu_resource_base_addr_ = {0x0};
    uint64_t addr = 0;
    EXPECT_EQ(rec->GetAddrByXnId(0, CcuComponerntType::XN_A6, 0, addr), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetAddrByXnId(0, CcuComponerntType::CKE_A6, 0, addr), HCCL_SUCCESS);
    EXPECT_EQ(rec->GetAddrByXnId(0, CcuComponerntType::CHANNEL_A6, 0, addr), HCCL_SUCCESS);
}

class TypeConversionTest : public testing::Test {};

TEST_F(TypeConversionTest, DataType2CheckerDataType_ContainsAllEntries)
{
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu.size(), 16u);
}

TEST_F(TypeConversionTest, DataType2CheckerDataType_MapsCorrectly)
{
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::INT8], HcclDataType::HCCL_DATA_TYPE_INT8);
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::FP16], HcclDataType::HCCL_DATA_TYPE_FP16);
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::FP32], HcclDataType::HCCL_DATA_TYPE_FP32);
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::INT64], HcclDataType::HCCL_DATA_TYPE_INT64);
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::FP8E4M3], HcclDataType::HCCL_DATA_TYPE_FP8E4M3);
    EXPECT_EQ(g_DataType2CheckerDataType_aicpu[DataType::FP8E5M2], HcclDataType::HCCL_DATA_TYPE_FP8E5M2);
}

TEST_F(TypeConversionTest, ReduceOp2CheckerReduceOp_Ccu_ContainsEntries)
{
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu.size(), 3u);
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu[10], HcclReduceOp::HCCL_REDUCE_SUM);
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu[9], HcclReduceOp::HCCL_REDUCE_MIN);
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu[8], HcclReduceOp::HCCL_REDUCE_MAX);
}

TEST_F(TypeConversionTest, ReduceOp2CheckerReduceOp_Ccu_InvalidKey_ReturnsZero)
{
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu.count(0), 0u);
    EXPECT_EQ(g_ReduceOp2CheckerReduceOp_ccu.count(99), 0u);
}
