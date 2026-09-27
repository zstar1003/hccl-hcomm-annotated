/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "independent_op_context_manager.h"
#include "log.h"
#include "hccl_comm_pub.h"
#include "independent_op.h"
#include "llt_hccl_stub_rank_graph.h"
#include <string>
#include "mockcpp/mockcpp.hpp"

#define private public

using namespace hccl;

class HcclEngineCtxDestroyV2Test : public BaseInit {
public:
    void SetUp() override
    {
        BaseInit::SetUp();
        const char* fakeA5SocName = "Ascend950PR_958b";
        MOCKER(aclrtGetSocName).stubs().will(returnValue(fakeA5SocName));
    }
    void TearDown() override
    {
        BaseInit::TearDown();
        GlobalMockObject::verify();
    }

protected:
    void SetUpCommAndGraph(
        std::shared_ptr<hccl::hcclComm>& hcclCommPtr, std::shared_ptr<Hccl::RankGraph>& rankGraphV2, void*& comm,
        HcclResult& ret)
    {
        MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));

        bool isDeviceSide{false};
        MOCKER(GetRunSideIsDevice).stubs().with(outBound(isDeviceSide)).will(returnValue(HCCL_SUCCESS));
        MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
        setenv("HCCL_INDEPENDENT_OP", "1", 1);
        RankGraphStub rankGraphStub;
        rankGraphV2 = rankGraphStub.Create2PGraph();
        void* commV2 = (void*)0x2000;
        uint32_t rank = 1;
        HcclMem cclBuffer;
        cclBuffer.size = 1;
        cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
        cclBuffer.addr = (void*)0x1000;
        char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
        hcclCommPtr = std::make_shared<hccl::hcclComm>(1, 1, commName);
        HcclCommConfig config;
        UtInitHcclCommConfig(config);
        config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
        config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
        config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
        unsetenv("HCCL_DFS_CONFIG");
        ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
        CollComm* collComm = hcclCommPtr->GetCollComm();
        comm = static_cast<HcclComm>(hcclCommPtr.get());
    }
};

// 测试空comm指针传入
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_InputParamNull_Expect_Return_ERROR)
{
    const char* ctxTag = "1";
    CommEngine engine = COMM_ENGINE_CPU;

    HcclResult result = HcclEngineCtxDestroy(nullptr, ctxTag, engine);
    EXPECT_EQ(result, HCCL_E_PTR);
}

// 测试空ctxTag指针传入，tag替换为空字符串，预期不报错
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_CtxTagNull_Expect_Return_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    CommEngine engine = COMM_ENGINE_CPU;
    void* ctx;
    uint64_t size = 256;

    // 创建COMM_ENGINE_CPU类型的Context
    HcclResult createResult = HcclEngineCtxCreate(comm, nullptr, engine, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    HcclResult result = HcclEngineCtxDestroy(comm, nullptr, engine);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

// 测试销毁不存在的Context
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_TagNotExist_Expect_Return_EPARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const char* ctxTag = "non_existent_tag";
    CommEngine engine = COMM_ENGINE_CPU;

    HcclResult result = HcclEngineCtxDestroy(comm, ctxTag, engine);
    EXPECT_EQ(result, HCCL_E_PARA);
}

// 测试Tag存在但engine不存在
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_EngineNotExistInTag_Expect_Return_EPARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const char* ctxTag = "tag1";
    void* ctx;
    uint64_t size = 256;

    // 先创建tag1的CPU类型Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 尝试销毁未创建的AICPU类型engine
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyResult, HCCL_E_PARA);

    // 清理已创建的CPU Context
    HcclResult cleanResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(cleanResult, HCCL_SUCCESS);
}

// 测试销毁Host类型内存
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_SuccessDestroyHostMem_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const char* ctxTag = "host_tag";
    void* ctx;
    uint64_t size = 256;

    // 创建COMM_ENGINE_CPU类型的Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 销毁Host类型内存
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(destroyResult, HCCL_SUCCESS);
}

// 测试销毁Device类型内存
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_SuccessDestroyDeviceMem_Expect_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    const char* ctxTag = "device_tag";
    void* ctx;
    uint64_t size = 256;

    // 创建COMM_ENGINE_AICPU类型的Context
    HcclResult createResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_AICPU, size, &ctx);
    EXPECT_EQ(createResult, HCCL_SUCCESS);

    // 销毁Device类型内存
    HcclResult destroyResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyResult, HCCL_SUCCESS);
}

// 测试同tag下多engine销毁
TEST_F(HcclEngineCtxDestroyV2Test, Ut_HcclEngineCtxDestroy_When_MultipleEnginesUnderSameTag_Expect_OnlySpecifiedDeleted)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    const char* ctxTag = "multi_engine_tag";
    void* cpuCtx;
    void* aicpuCtx;
    uint64_t size = 256;

    // 创建tag1的CPU和AICPU两种Context
    HcclResult createCpuResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_CPU, size, &cpuCtx);
    EXPECT_EQ(createCpuResult, HCCL_SUCCESS);

    HcclResult createAicpuResult = HcclEngineCtxCreate(comm, ctxTag, COMM_ENGINE_AICPU, size, &aicpuCtx);
    EXPECT_EQ(createAicpuResult, HCCL_SUCCESS);

    // 只销毁CPU类型的engine
    HcclResult destroyCpuResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_CPU);
    EXPECT_EQ(destroyCpuResult, HCCL_SUCCESS);

    // 验证AICPU内存仍存在（尝试销毁AICPU应该成功）
    HcclResult destroyAicpuResult = HcclEngineCtxDestroy(comm, ctxTag, COMM_ENGINE_AICPU);
    EXPECT_EQ(destroyAicpuResult, HCCL_SUCCESS);
}
