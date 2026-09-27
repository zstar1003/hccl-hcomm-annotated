/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "../../../hccl_api_base_test.h"
#include "hccl/hccl_res.h"
#include "llt_hccl_stub_rank_graph.h"

class TestHcclCommMemReg : public BaseInit {
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
};

// 公共：构造一个已经 InitCollComm 成功的 hcclComm，供成功/失败用例复用
static void BuildV2HcclComm(std::shared_ptr<hccl::hcclComm>& hcclCommPtr)
{
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(IsSupportHCCLV2).stubs().will(returnValue(true));
    setenv("HCCL_INDEPENDENT_OP", "1", 1);

    void* commV2 = (void*)0x2000;
    RankGraphStub rankGraphStub;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2 = rankGraphStub.Create2PGraph();
    u32 rank = 1;
    HcclMem cclBuffer;
    cclBuffer.size = 1024;
    cclBuffer.type = HcclMemType::HCCL_MEM_TYPE_HOST;
    cclBuffer.addr = (void*)0x1000;
    char commName[ROOTINFO_INDENTIFIER_MAX_LENGTH] = {};
    hcclCommPtr = make_shared<hccl::hcclComm>(1, 1, commName);
    HcclCommConfig config;
    UtInitHcclCommConfig(config);
    config.hcclOpExpansionMode = 1;           // 非CCU模式，避免拉起CCU平台层
    config.hcclRdmaTrafficClass = 0xFFFFFFFF; // 不配置RDMA Traffic Class
    config.hcclRdmaServiceLevel = 0xFFFFFFFF; // 不配置RDMA Service Level
    unsetenv("HCCL_DFS_CONFIG");
    HcclResult ret = hcclCommPtr->InitCollComm(commV2, rankGraphV2.get(), rank, cclBuffer, commName, &config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_Normal_Return_HCCL_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_Comm_Nullptr_Return_HCCL_E_PTR)
{
    void* comm = nullptr;
    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemTag_Nullptr_Return_HCCL_E_PTR)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = nullptr;
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemTag_Empty_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemTag_TooLong_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    // HCCL_RES_TAG_MAX_LEN = 255，超过该长度应返回 HCCL_E_PARA
    std::string longTag(HCCL_RES_TAG_MAX_LEN + 1, 'a');
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, longTag.c_str(), &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_Mem_Nullptr_Return_HCCL_E_PTR)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem* mem = nullptr;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemHandle_Nullptr_Return_HCCL_E_PTR)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle* memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemType_Invalid_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_INVALID;
    mem.addr = (void*)0x3000;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_Addr_Nullptr_Return_HCCL_E_PTR)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = nullptr;
    mem.size = 1024;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_Size_Zero_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagA";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_HOST;
    mem.addr = (void*)0x3000;
    mem.size = 0;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(TestHcclCommMemReg, Ut_HcclCommMemReg_When_MemType_Device_Return_HCCL_Success)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    BuildV2HcclComm(hcclCommPtr);
    void* comm = static_cast<HcclComm>(hcclCommPtr.get());

    const char* memTag = "userTagDevice";
    CommMem mem;
    mem.type = COMM_MEM_TYPE_DEVICE;
    mem.addr = (void*)0x4000;
    mem.size = 2048;
    HcclMemHandle memHandle = nullptr;

    HcclResult ret = HcclCommMemReg(comm, memTag, &mem, &memHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
