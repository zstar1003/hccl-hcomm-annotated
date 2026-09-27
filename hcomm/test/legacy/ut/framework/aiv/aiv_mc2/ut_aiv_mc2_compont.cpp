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
#include "orion_adapter_rts.h"
#include "ccu_instruction_all_gather_mesh1d.h"
#include "coll_service_device_mode.h"
#include "ccu_ins.h"
#include "ccu_assist.h"
#include "mc2_global_mirror_tasks.h"
#include "mc2_compont.h"
#include "communicator_impl.h"
#include "ccu_ins_preprocessor.h"
#include "dev_buffer.h"
#include "rma_buffer.h"
#include "internal_exception.h"
#include "invalid_params_exception.h"
#undef private
#undef protected

using namespace Hccl;

using namespace std;

class AivMc2CompontTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AivMc2CompontTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AivMc2CompontTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AivMc2CompontTest SetUP" << std::endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        MC2GlobalMirrorTasks::GetInstance().Clear();
        std::cout << "A Test case in AivMc2CompontTest TearDown" << std::endl;
    }
};

class FakeCollAlgComponent : public CollAlgComponent {
public:
    FakeCollAlgComponent() : CollAlgComponent(nullptr, DevType::DEV_TYPE_950, 0, 1) {};
    HcclResult Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, InsQuePtr queue, string& algName)
    {
        queue->Append(std::move(std::make_unique<CcuInstructionAllGatherMesh1D>()));
        return HCCL_SUCCESS;
    }

    HcclResult Orchestrate(const CollAlgOperator& op, const CollAlgParams& params, PrimQuePtr queue, string& algName)
    {
        return HCCL_SUCCESS;
    }
};

TEST_F(AivMc2CompontTest, should_return_fail_when_calling_AllocCommResource_comm_rank_size_1)
{
    Mc2Tiling mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V1;
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    AivMc2Compont aivMc2Compont(comm.get());
    comm->rankSize = 1;
    EXPECT_THROW(aivMc2Compont.AllocCommResource((void*)&mc2Tiling, nullptr), NotSupportException);
}

TEST_F(AivMc2CompontTest, should_return_fail_when_calling_AllocCommResource_tilingVersion_not_UNKNOWN_TILING)
{
    Mc2Tiling mc2Tiling;
    mc2Tiling.version = 0;
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    AivMc2Compont aivMc2Compont(comm.get());
    comm->rankSize = 1;
    EXPECT_THROW(aivMc2Compont.AllocCommResource((void*)&mc2Tiling, nullptr), NotSupportException);
}

TEST_F(AivMc2CompontTest, should_return_success_when_calling_GenerateCommContext)
{
    // when
    MOCKER(CcuRep::GetTokenInfo).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1000));
    HcclCombinOpParam opParam;
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(&opParam)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x10000));

    // then
    CommunicatorImpl comm{};
    comm.rankSize = 1;
    comm.myRank = 0;
    AivMc2Compont aivMc2Compont(&comm);
    comm.cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm.ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(comm);

    auto tmp = std::make_shared<CollServiceDeviceMode>(&comm);
    comm.collService = tmp.get();
    auto collService = dynamic_cast<CollServiceDeviceMode*>(comm.GetCollService());
    collService->GetAivInsPreprocessor()->SetProtocol(0);

    void* commContext;
    // check
    Mc2Tiling mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V1;
    mc2Tiling.commConfig.dataType = 5; // int64
    EXPECT_NO_THROW(aivMc2Compont.GenerateCommContext(&commContext));
    EXPECT_NE(nullptr, aivMc2Compont.combinOpParamBuffer);
    EXPECT_NO_THROW(aivMc2Compont.GenerateCommContext(&commContext)); // combinOpParamBuffer != nullptr退出
}

TEST_F(AivMc2CompontTest, should_throw_InternalException_when_calling_GenerateCommContext_CclBuffer_Null)
{
    // when
    MOCKER(CcuRep::GetTokenInfo).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1000));
    HcclCombinOpParam opParam;
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(&opParam)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x10000));

    // then
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 1;
    AivMc2Compont aivMc2Compont(comm.get());
    comm->cclBuffer = nullptr;

    auto tmp = std::make_shared<CollServiceDeviceMode>(comm.get());
    comm->collService = tmp.get();
    auto collService = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService());
    collService->GetAivInsPreprocessor()->SetProtocol(0);

    void* commContext;
    // check
    Mc2Tiling mc2Tiling;
    mc2Tiling.version = UNKNOWN_TILING_V1;
    mc2Tiling.commConfig.dataType = 5; // int64
    EXPECT_THROW(aivMc2Compont.GenerateCommContext(&commContext), InternalException);
}

TEST_F(AivMc2CompontTest, should_return_early_when_mc2Tiling_is_nullptr)
{
    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 2;
    AivMc2Compont aivMc2Compont(comm.get());

    void* commContext = nullptr;
    aivMc2Compont.AllocCommResource(nullptr, &commContext);
    EXPECT_EQ(nullptr, commContext);
}

// 覆盖 GenerateAivUrmaCommContext 的 links.empty() 异常路径及 opTag 改动（GetAivUrmaBufferTag）
TEST_F(AivMc2CompontTest, should_throw_when_calling_GenerateCommContext_urma_links_empty)
{
    MOCKER(CcuRep::GetTokenInfo).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(1000));
    HcclCombinOpParam opParam;
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(&opParam)));
    MOCKER(HrtMalloc).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue((void*)0x10000));

    std::unique_ptr<CommunicatorImpl> comm = std::make_unique<CommunicatorImpl>();
    comm->rankSize = 1;
    comm->myRank = 0;
    AivMc2Compont aivMc2Compont(comm.get());
    comm->cclBuffer = DevBuffer::Create(0x100, 0x100);
    comm->ubMemoryTransportMgr = std::make_unique<UbMemoryTransportMgr>(*comm);

    auto tmp = std::make_shared<CollServiceDeviceMode>(comm.get());
    comm->collService = tmp.get();
    auto collService = dynamic_cast<CollServiceDeviceMode*>(comm->GetCollService());
    collService->GetAivInsPreprocessor()->SetProtocol(1); // urma 走 GenerateAivUrmaCommContext

    // fullMeshLinks 默认为空，触发 links.empty() 异常
    void* commContext;
    EXPECT_THROW(aivMc2Compont.GenerateCommContext(&commContext), InvalidParamsException);
}
