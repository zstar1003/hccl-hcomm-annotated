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

    virtual void SetUp() { std::cout << "AlgTemplateBaseTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        // GlobalMockObject::verify();
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
    AHCExtendPreparePara extendPara;

    AlgTemplateBase tempBase(disp);

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(params));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(extendPara));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, devMem, signals, signals, mainStream, subStreams, links, 0, 0, sendrecvInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE,
            mainStream, subStreams, signals, signals, 0, 0, links, sendrecvInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     alltoallBuf, alltoallBuf, true, mainStream,
                                     HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, displsMap, displsMap));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     alltoallBuf, alltoallBuf, devMem, devMem, true, mainStream,
                                     HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE, displsMap, displsMap));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(devMem, devMem, alltoallAddr, alltoallAddr, true, mainStream));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, true, mainStream));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, alltoallAddr, alltoallAddr, true, 0, mainStream, subStreams, signals, signals));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, devMem, alltoallAddr, alltoallAddr, true, 0, mainStream,
                                     subStreams, signals, signals));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, nullptr));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, false));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, idx));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(attr, &opInfo, idx, subStreams, signals, signals, order, slices, false));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            &opInfo, devMem, attr, attr, attr, comInfo, comInfo, mainStream, subStreams, signals, signals, attr));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     HcclReduceOp::HCCL_REDUCE_SUM, idx, slices, attr, order, idx));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     HcclReduceOp::HCCL_REDUCE_SUM, idx, subStreams, signals, signals, idx, &opInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            mainStream, comInfo, devMem, devMem, devMem, devMem, attr, subStreams, signals, signals,
            HcclDataType::HCCL_DATA_TYPE_INT8, HcclReduceOp::HCCL_REDUCE_SUM, multiSlices, attr));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, idx, attr, UserMemType::INPUT_MEM, UserMemType::INPUT_MEM));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, &opInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream, HcclReduceOp::HCCL_REDUCE_SUM,
            idx, slices, attr, attr, subStreams, signals, signals, idx, idx, &opInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     devMem, devMem, devMem, attr, HcclDataType::HCCL_DATA_TYPE_INT8, mainStream,
                                     multiSlices, HcclReduceOp::HCCL_REDUCE_SUM, idx, attr, false, attr, &opInfo, idx,
                                     subStreams, signals, signals, multiOrders, multiSlices));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(false));

    u32 userRank = 1;
    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(idx));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(idx, idx));

    UserMemType hdInput;
    UserMemType hdOutput;
    std::vector<std::vector<std::vector<u32>>> subGroups;
    std::map<AHCConcOpType, TemplateType> ahcAlgOption;
    bool extendFlag = false;

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(idx, hdInput, hdOutput));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(&opInfo, idx, order, slices));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(&opInfo, idx, slices, false));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, subGroups, ahcAlgOption, extendFlag, extendPara));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, subStreams, signals, signals, idx, &opInfo, false));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(&opInfo, idx, subStreams, signals, signals, multiOrders, multiSlices));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(subStreams, signals, signals, idx, &opInfo, idx, idx));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(&opInfo, idx, subStreams, signals, signals, order, slices, false));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(&opInfo, idx, idx, subStreams, signals, signals, multiOrders, multiSlices, multiSlices));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            &opInfo, idx, attr, devMem, devMem, comInfo, comInfo, mainStream, subStreams, signals, signals));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(
            mainStream, comInfo, devMem, devMem, devMem, devMem, idx, subStreams, signals, signals, multiSlices));

    EXPECT_EQ(HcclResult::HCCL_E_PARA, tempBase.Prepare(attr, subStreams, signals, signals, idx, idx, idx, &opInfo));

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA,
        tempBase.Prepare(&opInfo, devMem, devMem, attr, comInfo, comInfo, mainStream, subStreams, signals, signals));

    DeviceMem cclInMem;
    DeviceMem outputMem;
    Stream stream;
    std::vector<std::shared_ptr<LocalNotify>> meshSignal;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalAux;
    MemBlockInfo memBlockInfo;
    bool isUseCclIn;
    bool isLevel0LastRank;
    bool isNeedSpaceBorrow;
    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.Prepare(
                                     cclInMem, outputMem, stream, subStreams, meshSignal, meshSignalAux, memBlockInfo,
                                     HcclReduceOp::HCCL_REDUCE_SUM, HcclDataType::HCCL_DATA_TYPE_INT8, isUseCclIn,
                                     isLevel0LastRank, isNeedSpaceBorrow));

    std::vector<std::shared_ptr<Transport>> sharedTransport;
    RunStage stage;
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempBase.RunAsync());
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempBase.RunAsync(idx, idx, sharedTransport));
    EXPECT_EQ(HcclResult::HCCL_SUCCESS, tempBase.RunAsyncStaged(idx, idx, sharedTransport, stage));

    u32 rank, rankSize, stepsInBlock, lowerBlockSize, myBlockSize, rankInMyBlock, myBlockOffset, higherBolockSize;
    rank = stepsInBlock = lowerBlockSize = myBlockSize = rankInMyBlock = myBlockOffset = higherBolockSize = 1;
    rankSize = 3;
    tempBase.CalcBinaryBlockParams(
        rank, rankSize, stepsInBlock, lowerBlockSize, myBlockSize, rankInMyBlock, myBlockOffset, higherBolockSize);

    std::vector<bool> linkRelation(10, false);
    rank = rankSize = myBlockSize = 0;

    EXPECT_EQ(
        HcclResult::HCCL_E_PARA, tempBase.CalcBinaryBlockHalvingDoubleLinkReleation(rank, rankSize, linkRelation));
    tempBase.CalcRecursiveHdLinkRelationForSecondScene(rank, rankSize, myBlockSize, linkRelation);
    rankSize = 3;
    tempBase.CalcRecursiveHdLinkRelationForSecondScene(rank, rankSize, myBlockSize, linkRelation);
    rank = 1;
    tempBase.CalcRecursiveHdLinkRelationForSecondScene(rank, rankSize, myBlockSize, linkRelation);
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
