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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#define protected public
#include "mc2_compont.h"
#include "orion_adapter_rts.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "ccu_ins.h"
#include "ccu_assist.h"
#include "coll_service_device_mode.h"
#include "communicator_impl.h"
#include "mc2_global_mirror_tasks.h"
#include "ccu_ins_preprocessor.h"
#include "op_params_checker.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class Mc2CompontTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "Mc2CompontTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "Mc2CompontTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in Mc2CompontTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        MC2GlobalMirrorTasks::GetInstance().Clear();
        std::cout << "A Test case in Mc2CompontTest TearDown" << std::endl;
    }
};

class FakeCollAlgComponent : public CollAlgComponent {
public:
    FakeCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue)
    {
        queue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
        return HCCL_SUCCESS;
    }

    HcclResult
    Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, const string& algName, PrimQuePtr queue)
    {
        return HCCL_SUCCESS;
    }
};

TEST(Mc2CompontTest, should_return_fail_when_calling_AllocCommResource_comm_rank_size_1)
{
    Mc2Tiling mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V1;
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    comm->rankSize = 1;
    EXPECT_NO_THROW(mc2Compont.AllocCommResource((void*)&mc2Tiling, nullptr));
}

TEST(Mc2CompontTest, should_return_success_when_calling_Alloc)
{
    // when
    MOCKER(CcuRep::GetTokenInfo).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1000));
    HcclCombinOpParam opParam;
    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 1;
    Mc2Compont mc2Compont(comm.get());
    comm->cclBuffer = DevBuffer::Create(0x100, 0x100);

    void* commContext;
    // check
    EXPECT_NO_THROW(mc2Compont.Alloc(&commContext));
    EXPECT_NE(nullptr, mc2Compont.combinOpParamBuffer);
}

TEST(Mc2CompontTest, should_return_success_when_calling_generateCcuServer)
{
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::RegisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::DeregisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    CcuTaskParam param;
    param.dieId = 0;
    std::vector<std::vector<CcuTaskParam>> taskParams(1);
    taskParams[0].push_back(param);
    uint64_t templateSignature = 0xf000f06;
    mc2Compont.algoTemplateMap[templateSignature] = taskParams;
    mc2Compont.comParamBuffer = std::make_shared<DevBuffer>(128 * 8);
    mc2Compont.comSyncBuffer = std::make_shared<DevBuffer>(16 * 8);
    std::unordered_set<uint64_t> algoTemplateRequire = {templateSignature};
    // check
    EXPECT_NO_THROW(mc2Compont.GenerateCcuServer(algoTemplateRequire));
    EXPECT_EQ(1, mc2Compont.ccuServerMap.size());
}

TEST(Mc2CompontTest, should_return_success_when_calling_generateCcuServer_and_server_exist)
{
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::RegisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::DeregisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    CcuTaskParam param;
    param.dieId = 0;
    std::vector<std::vector<CcuTaskParam>> taskParams(1);
    taskParams[0].push_back(param);
    uint64_t templateSignature = 0xf000f06;
    mc2Compont.algoTemplateMap[templateSignature] = taskParams;
    std::unordered_set<uint64_t> algoTemplateRequire = {templateSignature};
    mc2Compont.ccuServerMap[1] = algoTemplateRequire;
    mc2Compont.comParamBuffer = std::make_shared<DevBuffer>(128 * 8);
    mc2Compont.comSyncBuffer = std::make_shared<DevBuffer>(16 * 8);
    // check
    EXPECT_NO_THROW(mc2Compont.GenerateCcuServer(algoTemplateRequire));
    EXPECT_EQ(1, mc2Compont.curExecId);
    EXPECT_EQ(1, mc2Compont.ccuServerMap.size());
}

TEST(Mc2CompontTest, should_return_false_when_calling_CompareMissionMap)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    std::map<uint8_t, std::map<uint32_t, uint32_t>> mapA;
    std::map<uint8_t, std::map<uint32_t, uint32_t>> mapB;

    mapA[0] = std::map<uint32_t, uint32_t>();
    EXPECT_EQ(false, mc2Compont.CompareMissionMap(mapA, mapB));

    mapB[0] = std::map<uint32_t, uint32_t>();
    mapA[0][0] = 1;
    EXPECT_EQ(false, mc2Compont.CompareMissionMap(mapA, mapB));

    mapB[0][0] = 2;
    mapA[0][1] = 2;
    EXPECT_EQ(false, mc2Compont.CompareMissionMap(mapA, mapB));
}

TEST(Mc2CompontTest, should_return_success_when_calling_generateCcuServer_and_server_exist_multiOp)
{
    MOCKER(CcuCtxMgr::AllocRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER(InsExeQue::RegisterExtendInstruction)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());

    CcuTaskParam param;
    std::vector<std::vector<CcuTaskParam>> taskParams(4);
    param.dieId = 0;
    param.missionId = 0;
    taskParams[0].push_back(param);
    param.missionId = 1;
    taskParams[1].push_back(param);

    param.dieId = 1;
    param.missionId = 0;
    taskParams[2].push_back(param);
    param.missionId = 1;
    taskParams[3].push_back(param);

    uint64_t templateSignature0 = 0xf000f06;
    mc2Compont.algoTemplateMap[templateSignature0] = taskParams;
    uint64_t templateSignature1 = 0x3000302;
    mc2Compont.algoTemplateMap[templateSignature1] = taskParams;

    std::unordered_set<uint64_t> algoTemplateRequire = {templateSignature0, templateSignature1};
    mc2Compont.comParamBuffer = std::make_shared<DevBuffer>(128 * 8);
    mc2Compont.comSyncBuffer = std::make_shared<DevBuffer>(16 * 8);
    // check
    EXPECT_NO_THROW(mc2Compont.GenerateCcuServer(algoTemplateRequire));
}

TEST(Mc2CompontTest, should_return_success_when_calling_getCcuTaskInfo)
{
    CcuTaskParam param;
    param.dieId = 0;
    std::vector<std::vector<CcuTaskParam>> taskParams(1);
    taskParams[0].push_back(param);
    MOCKER(CcuCtxMgr::GetTaskParam)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBound(taskParams))
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    mc2Compont.curExecId = 0;
    uint64_t templateSignature = 0xf000f06;
    mc2Compont.algoTemplateMap[templateSignature] = taskParams;
    std::unordered_set<uint64_t> algoTemplateRequire = {templateSignature};
    mc2Compont.ccuServerMap[0] = algoTemplateRequire;

    uint32_t kfcArgsFmtOffset = 397;
    auto args = new uint8_t[kfcArgsFmtOffset * sizeof(void*) + sizeof(HcclCommParamDesc)];
    *reinterpret_cast<uint64_t*>(args + sizeof(void*)) = 0xff;
    auto desc = reinterpret_cast<HcclCommParamDesc*>(args + kfcArgsFmtOffset * sizeof(void*));
    desc->version = 1;
    desc->groupNum = 1;
    desc->hasFfts = 1;
    desc->tilingDataPtrOff = 7;
    desc->isDyn = 0;

    auto tilingData = reinterpret_cast<KFCTilingData*>(args + (desc->tilingDataPtrOff + 2) * sizeof(void*));
    tilingData->preparePosition = 3;
    *reinterpret_cast<uint64_t*>(args + (desc->tilingDataPtrOff) * sizeof(void*))
        = reinterpret_cast<uint64_t>(tilingData);
    std::vector<CcuTaskParam> taskParam = mc2Compont.GetCcuTaskInfo(tilingData);
    delete[] args;
    // check
    EXPECT_EQ(1, taskParam.size());
}

TEST(Mc2CompontTest, func_SaveMc2DfxTaskInfo_test)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->devLogicId = 1;
    Mc2Compont mc2Compont(comm.get());

    CcuTaskParam ccuTaskParam{};
    ccuTaskParam.dieId = 0;
    ccuTaskParam.missionId = 3;
    ccuTaskParam.instStartId = 10;
    mc2Compont.SaveMc2DfxTaskInfo(ccuTaskParam, 2);

    shared_ptr<TaskInfo> taskInfo = MC2GlobalMirrorTasks::GetInstance().GetTaskInfo(1, 0, 3, 10);
    EXPECT_NE(taskInfo, nullptr);
    EXPECT_EQ(taskInfo->taskParam_.taskType, TaskParamType::TASK_CCU);
    EXPECT_EQ(taskInfo->taskParam_.taskPara.Ccu.dieId, 0);
    EXPECT_EQ(taskInfo->taskParam_.taskPara.Ccu.missionId, 3);
    EXPECT_EQ(taskInfo->taskParam_.taskPara.Ccu.instrId, 10);
    EXPECT_EQ(taskInfo->taskParam_.taskPara.Ccu.executeId, 2);
}

TEST(Mc2CompontTest, func_GetAlgoCcuTaskInfo_test)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());

    vector<CcuTaskParam> empRet = mc2Compont.GetAlgoCcuTaskInfo(100);
    EXPECT_EQ(empRet.size(), 0);

    mc2Compont.ccuServerMap[0] = {1, 2};
    CcuTaskParam ccuTaskParam{};
    std::vector<std::vector<CcuTaskParam>> ccuTaskParams1{};
    ccuTaskParams1.push_back({ccuTaskParam});
    ccuTaskParams1.push_back({ccuTaskParam});
    mc2Compont.algoTemplateMap[1] = ccuTaskParams1;
    vector<CcuTaskParam> ret = mc2Compont.GetAlgoCcuTaskInfo(0);
    EXPECT_EQ(ret.size(), 2);
}

TEST(Mc2CompontTest, test_MC2AllocCommRes)
{
    CommunicatorImpl comm;
    comm.opExecuteConfig.accState = AcceleratorState::CCU_MS;
    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect).stubs().will(ignoreReturnValue());
    CollServiceDeviceMode collService{&comm};
    collService.GetCcuInsPreprocessor()->isRollback = true;
    comm.collService = &collService;
    comm.socketManager = std::make_unique<SocketManager>(comm, 0, 0, 0);
    MOCKER_CPP(&CcuInsPreprocessor::Preprocess).stubs();

    Mc2Compont mc2Compont{&comm};

    MOCKER_CPP(&CommunicatorImpl::SetCommExecuteConfig).stubs().will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::SelectCollService).stubs().will(ignoreReturnValue());

    std::shared_ptr<FakeCollAlgComponent> collAlgComponent = std::make_shared<FakeCollAlgComponent>();
    comm.collAlgComponent = collAlgComponent;
    MOCKER_CPP_VIRTUAL(
        *collAlgComponent, &CollAlgComponent::Orchestrate,
        HcclResult(CollAlgComponent::*)(
            const CollAlgOperator& op, const CollAlgParams& params, const string& algName, InsQuePtr queue))
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    comm.currentCollOperator = std::make_unique<CollOperator>();
    comm.currentCollOperator->opType = OpType::ALLREDUCE;
    comm.currentCollOperator->opMode = OpMode::OPBASE;

    CollAlgParams collAlgParams;
    std::shared_ptr<InsQueue> insQueue = std::make_shared<InsQueue>();
    EXPECT_THROW(mc2Compont.MC2AllocCommRes(collAlgParams, insQueue), InternalException);
}

TEST(Mc2CompontTest, should_success_when_calling_AllocCommResource_V2)
{
    Mc2InitTilingInner mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V2;
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());
    comm->rankSize = 2;
    MOCKER_CPP(&Mc2Compont::AllocV2).stubs();
    MOCKER_CPP(&Mc2Compont::GenerateAlgoTemplatesV2).stubs();
    MOCKER_CPP(&Mc2Compont::GenerateCcuServer).stubs();
    EXPECT_NO_THROW(mc2Compont.AllocCommResource((void*)&mc2Tiling, nullptr));
}

TEST(Mc2CompontTest, should_return_success_when_calling_AllocV2)
{
    // when
    MOCKER(CcuRep::GetTokenInfo).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1000));
    HcclCombinOpParam opParam;
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(&opParam)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x10000));

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 1;
    Mc2Compont mc2Compont(comm.get());
    comm->cclBuffer = DevBuffer::Create(0x100, 0x100);

    void* commContext;
    // check
    void* mem = malloc(sizeof(Mc2InitTilingInner) + sizeof(Mc2CcTilingInner));
    Mc2InitTilingInner* mc2TilingPtr = reinterpret_cast<Mc2InitTilingInner*>(mem);
    mc2TilingPtr->version = 100;
    mc2TilingPtr->mc2HcommCnt = 1;
    mc2TilingPtr->offset[0] = sizeof(Mc2InitTilingInner);
    Mc2CcTilingInner* commConfigPtr
        = reinterpret_cast<Mc2CcTilingInner*>(reinterpret_cast<uint8_t*>(mc2TilingPtr) + mc2TilingPtr->offset[0]);
    commConfigPtr->opType = AicpuComType::HCCL_CMD_ALLTOALLV;
    commConfigPtr->reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    commConfigPtr->srcDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    commConfigPtr->dstDataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    EXPECT_NO_THROW(mc2Compont.AllocV2(&commContext));

    free(mc2TilingPtr);
}

TEST(Mc2CompontTest, should_skip_GenerateAlgoTemplatesV2_when_has_cache)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());

    std::unordered_set<uint64_t> algoTemplateRequire{};
    void* mem = malloc(sizeof(Mc2InitTilingInner) + sizeof(Mc2CcTilingInner));
    Mc2InitTilingInner* mc2TilingPtr = reinterpret_cast<Mc2InitTilingInner*>(mem);
    mc2TilingPtr->version = 100;
    mc2TilingPtr->mc2HcommCnt = 1;
    mc2TilingPtr->offset[0] = sizeof(Mc2InitTilingInner);
    Mc2CcTilingInner* commConfigPtr
        = reinterpret_cast<Mc2CcTilingInner*>(reinterpret_cast<uint8_t*>(mc2TilingPtr) + mc2TilingPtr->offset[0]);
    commConfigPtr->opType = AicpuComType::HCCL_CMD_ALLTOALLV;
    commConfigPtr->reduceType = HcclReduceOp::HCCL_REDUCE_PROD;
    commConfigPtr->srcDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    commConfigPtr->dstDataType = HcclDataType::HCCL_DATA_TYPE_FP32;

    mc2Compont.algoTemplateMap[0x0000000004010408] = {}; // mock algo template sign cache

    EXPECT_NO_THROW(mc2Compont.GenerateAlgoTemplatesV2(mc2TilingPtr, algoTemplateRequire));

    free(mc2TilingPtr);
}

TEST(Mc2CompontTest, func_FillCollOperatorV2_test)
{
    Mc2InitTilingInner mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V2;
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 1;
    Mc2Compont mc2Compont(comm.get());
    mc2Compont.inputMem = DevBuffer::Create(0x100, 0x100);

    Mc2CcTilingInner config;
    config.opType = AicpuComType::HCCL_CMD_ALLTOALLV;
    config.reduceType = HcclReduceOp::HCCL_REDUCE_SUM;
    config.srcDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    config.dstDataType = HcclDataType::HCCL_DATA_TYPE_FP32;
    MOCKER_CPP(&CommunicatorImpl::CovertToCurrentCollOperator)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any());
    mc2Compont.FillCollOperatorV2(config);
}

TEST(Mc2CompontTest, func_GetTemplateSignatureV2_test)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    Mc2Compont mc2Compont(comm.get());

    Mc2CcTilingInner config;
    config.opType = AicpuComType::HCCL_CMD_ALLTOALLV;       // 8
    config.reduceType = HcclReduceOp::HCCL_REDUCE_PROD;     // 1
    config.srcDataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 4
    config.dstDataType = HcclDataType::HCCL_DATA_TYPE_FP32; // 4

    EXPECT_EQ(mc2Compont.GetTemplateSignatureV2(config), 0x0000000004010408);
}

TEST(Mc2CompontTest, should_suc_when_check_datatype_mc2_highP_V2)
{
    Mc2CcTilingInner config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> optypeWithoutReduce = {
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLGATHER), static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALL),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLTOALLV),
        static_cast<uint32_t>(AicpuComType::HCCL_CMD_HALF_ALLTOALLV)};
    std::vector<uint32_t> dataTypeHighP
        = {static_cast<uint32_t>(DataType::INT16), static_cast<uint32_t>(DataType::INT32),
           static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.srcDataType = dtype;
            config.dstDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
        }
    }
    for (auto optype : optypeWithoutReduce) {
        config.opType = optype;
        for (auto dtype : dataTypeHighP) {
            config.srcDataType = dtype;
            config.dstDataType = dtype;
            EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
        }
    }
}

TEST(Mc2CompontTest, should_suc_when_check_datatype_mc2_lowP_V2)
{
    Mc2CcTilingInner config;

    std::vector<uint32_t> optypeWithReduce
        = {static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER),
           static_cast<uint32_t>(AicpuComType::HCCL_CMD_ALLREDUCE)};
    std::vector<uint32_t> inputDataType
        = {static_cast<uint32_t>(DataType::INT8), static_cast<uint32_t>(DataType::FP8E5M2),
           static_cast<uint32_t>(DataType::FP8E4M3), static_cast<uint32_t>(DataType::HIF8)};
    std::vector<uint32_t> outputDataType
        = {static_cast<uint32_t>(DataType::FP16), static_cast<uint32_t>(DataType::FP32),
           static_cast<uint32_t>(DataType::BFP16)};

    for (auto optype : optypeWithReduce) {
        config.opType = optype;
        for (auto dtype : inputDataType) {
            config.srcDataType = dtype;
            for (auto outDtype : outputDataType) {
                config.dstDataType = outDtype;
                EXPECT_EQ(OpParamsChecker::CheckOpDataTypeMC2V2(config), HcclResult::HCCL_SUCCESS);
            }
        }
    }
}

TEST(Mc2CompontTest, should_fail_when_check_unsupported_datatype_mc2_lowP_V2)
{
    Mc2CcTilingInner config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.srcDataType = static_cast<uint32_t>(DataType::INT16);
    config.dstDataType = static_cast<uint32_t>(DataType::FP16);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);

    config.srcDataType = static_cast<uint32_t>(DataType::INT8);
    config.dstDataType = static_cast<uint32_t>(DataType::INT32);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);
}

TEST(Mc2CompontTest, should_fail_when_check_unsupported_datatype_mc2_highP_V2)
{
    Mc2CcTilingInner config;

    config.opType = static_cast<uint32_t>(AicpuComType::HCCL_CMD_REDUCE_SCATTER);
    config.srcDataType = static_cast<uint32_t>(DataType::INT64);
    config.dstDataType = static_cast<uint32_t>(DataType::INT64);
    EXPECT_THROW(OpParamsChecker::CheckOpDataTypeMC2V2(config), InvalidParamsException);
}
