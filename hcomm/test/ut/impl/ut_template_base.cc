/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>
#include <stdio.h>

#define protected public
#define private public
#include "alg_template_register.h"
#include "alg_template_base_pub.h"
#include "alltoallv_staged_base_pub.h"
#include "alltoallv_for_310p_pub.h"
#undef private
#undef protected

using namespace std;
using namespace hccl;

class AlgTemplateBaseTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AlgTemplateBaseTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AlgTemplateBaseTest Testcase TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "AlgTemplateBaseTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "AlgTemplateBaseTest TearDown" << std::endl;
    }
};

TEST_F(AlgTemplateBaseTest, alg_template_base_prepare)
{
    HcclDispatcher disp;
    struct PrepareData params;
    DeviceMem devMem;
    Stream mainStream;
    std::vector<Stream> subStreams;
    std::vector<LINK> links;
    std::vector<SendRecvInfo> sendrecvInfo;
    std::vector<std::shared_ptr<LocalNotify>> signals;
    std::map<u32, std::vector<u64>> displsMap;
    StageAlltoAllVAddrInfo alltoallAddr;
    AlltoAllVBufferInfo alltoallBuf;
    u64 attr;
    u32 idx;
    HcomCollOpInfo opInfo;
    std::vector<u32> order;
    std::vector<std::vector<u32>> multiOrders;
    std::vector<Slice> slices;
    std::vector<std::vector<Slice>> multiSlices;
    SubCommInfo comInfo;

    AlgTemplateBase tempBase(disp);

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(params));

    // AlltoAllVFor310P
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, devMem, signals, signals, mainStream, subStreams, links, 0, 0, sendrecvInfo));

    // AlltoAllVPairWise
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     alltoallBuf, alltoallBuf, true, mainStream,
                                     HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, displsMap, displsMap));

    // AlltoAllVPairWise
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     alltoallBuf, alltoallBuf, devMem, devMem, true, mainStream,
                                     HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, displsMap, displsMap));

    // AlltoAllVStagedPairwise
    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(devMem, devMem, alltoallAddr, alltoallAddr, true, mainStream));

    // AlltoAllVStagedPairwise
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, true, mainStream));

    // AlltoAllVStagedMesh
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, alltoallAddr, alltoallAddr, true, 0, mainStream, subStreams, signals, signals));

    // AlltoAllVStagedMesh
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, true, 0, mainStream,
                                     subStreams, signals, signals));

    // ReduceScatterNB, ReduceScatterNHRV1, ReduceScatterRing, ReduceScatterRecursiveHalvingDoubling
    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, nullptr));

    // ReduceScatterNHR
    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, false));

    // ReduceScatterMeshMixSingleStream, ReduceScatterMesh
    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, idx));

    // ReduceScatterRingConcurrentDirect
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(attr, &opInfo, idx, subStreams, signals, signals, order, slices, false));

    // ReduceScatterPipeline
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            &opInfo, devMem, attr, attr, attr, comInfo, comInfo, mainStream, subStreams, signals, signals, attr));

    // BroadcastStar
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     HcclReduceOp::HCCL_REDUCE_SUM, idx, slices, attr, order, idx));

    // BroadcastHD
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     HcclReduceOp::HCCL_REDUCE_SUM, idx, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterUnifiedMarch
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            mainStream, comInfo, devMem, devMem, devMem, devMem, attr, subStreams, signals, signals,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, multiSlices, attr));

    // ReduceScatterHalvingDoubling
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, idx, attr, UserMemType::INPUT_MEM, UserMemType::INPUT_MEM));

    // ReduceScatterHDStage, ReduceScatterLocalReduce, ReduceScatterMeshAtomic, ReduceScatterMeshDirect
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterMeshMix
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, idx, &opInfo));

    // AlignedReduceScatterDoubleRing, AlignedReduceScatter, DoubleRingWithSerialLocalCopy
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     multiSlices, HcclReduceOp::HCCL_REDUCE_SUM, idx, attr, false, attr, &opInfo, idx,
                                     subStreams, signals, signals, multiOrders, multiSlices));
}

TEST_F(AlgTemplateBaseTest, alltoallv_staged_base_prepare)
{
    HcclDispatcher disp;
    DeviceMem devMem;
    StageAlltoAllVAddrInfo alltoallAddr;
    Stream mainStream;

    AlltoAllVStagedBase tempBase(disp);

    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempBase.Prepare(devMem, devMem, alltoallAddr, alltoallAddr, true, mainStream));
}

TEST_F(AlgTemplateBaseTest, alltoallv_template_instance_init)
{
    HcclDispatcher disp;
    DeviceMem devMem = DeviceMem::create(nullptr, 1024);
    Stream mainStream;
    std::vector<Stream> subStreams(2);
    std::vector<LINK> links;
    std::vector<SendRecvInfo> sendrecvInfo;
    std::shared_ptr<LocalNotify> s0 = std::make_shared<LocalNotify>();
    std::shared_ptr<LocalNotify> s1 = std::make_shared<LocalNotify>();
    std::vector<std::shared_ptr<LocalNotify>> signals = {s0, s1};
    StageAlltoAllVAddrInfo alltoallAddr;

    std::unique_ptr<AlgTemplateBase> tempAlg;

    // AlltoAllVFor310P
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_2_ALL_V_FOR310P, disp);
    HcclResult ret = tempAlg->Prepare(
        devMem, devMem, devMem, devMem, signals, signals, mainStream, subStreams, links, 0, 1, sendrecvInfo);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, ret);

    // AlltoAllVStagedMesh
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_ALL_2_ALL_V_STAGED_MESH, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, alltoallAddr, alltoallAddr, false, 0, mainStream, subStreams, signals, signals));
    EXPECT_NE(
        HcclResult::HCCL_SUCCESS, tempAlg->Prepare(
                                      devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, false, 0, mainStream,
                                      subStreams, signals, signals));
}

TEST_F(AlgTemplateBaseTest, reducescatter_template_instance_init)
{
    HcclDispatcher disp;
    DeviceMem devMem;
    Stream mainStream;
    std::vector<Stream> subStreams(2);
    std::vector<std::shared_ptr<LocalNotify>> signals(2);
    u64 attr = 0ULL;
    u32 idx = 0UL;
    HcomCollOpInfo opInfo;
    std::vector<Slice> slices;
    std::vector<std::vector<Slice>> multiSlices(2);
    SubCommInfo comInfo;

    opInfo.dataType = HcclDataType::HCCL_DATA_TYPE_INT8;
    opInfo.reduceOp = HcclReduceOp::HCCL_REDUCE_SUM;
    comInfo.localRankSize = 0;

    std::unique_ptr<AlgTemplateBase> tempAlg;

    // ReduceScatterHDStage
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_HDSTAGE, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterLocalReduce
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_LOCAL_REDUCE, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterNB
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_NB, disp);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempAlg->Prepare(attr));
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS, tempAlg->Prepare(
                                      devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                      HcclReduceOp::HCCL_REDUCE_SUM));

    // ReduceScatterPipeline
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_PIPELINE, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            &opInfo, devMem, attr, attr, attr, comInfo, comInfo, mainStream, subStreams, signals, signals, attr));

    // ReduceScatterUnifiedMarch
    multiSlices[0].resize(2);
    comInfo.localRankSize = 2;
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_UNIFIED_MARCH, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            mainStream, comInfo, devMem, devMem, devMem, devMem, attr, subStreams, signals, signals,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, multiSlices, attr));

    // ReduceScatterMeshDirect
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_DIRECT, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterMeshAtomic
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_ATOMIC, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    // ReduceScatterMeshMixSingleStream
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX_SS, disp);
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempAlg->Prepare(attr, idx));
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS, tempAlg->Prepare(
                                      devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                      HcclReduceOp::HCCL_REDUCE_SUM, idx, slices, attr));

    // ReduceScatterMeshMix
    tempAlg = AlgTemplateRegistry::Instance().GetAlgTemplate(TemplateType::TEMPLATE_REDUCESCATTER_MESH_MIX, disp);
    EXPECT_EQ(
        HcclResult::HCCL_SUCCESS,
        tempAlg->Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, idx, &opInfo));
}
