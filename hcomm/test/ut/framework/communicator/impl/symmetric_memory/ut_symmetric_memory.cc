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
#include <iostream>
#include <vector>
#include <algorithm>

// 假设SimpleVaAllocator在这些头文件中
#define private public
#define protected public
#include "symmetric_memory.cc" // 替换为实际的头文件
#undef private
#undef protected

using namespace std;
using namespace hccl;

constexpr size_t TWO_M = 2097152;

class SymmetricMemoryTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SymmetricMemoryTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "SymmetricMemoryTest Testcase TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A SymmetricMemoryTest SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A SymmetricMemoryTest TearDown" << std::endl;
    }
};

TEST_F(SymmetricMemoryTest, ut_SymmetricMemoryCtor_When_UrmaMode_Expect_NoVaAllocator)
{
    SymmetricMemory urmaMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    EXPECT_EQ(urmaMemory.vaAllocator_, nullptr);

    SymmetricMemory hccsMemory(0, 2, 4096, SymmetricMemoryMode::HCCS);
    EXPECT_NE(hccsMemory.vaAllocator_, nullptr);
}

class SimpleVaAllocatorTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "SimpleVaAllocatorTest Testcase SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "SimpleVaAllocatorTest Testcase TearDown" << std::endl; }
    virtual void SetUp()
    {
        std::cout << "A SimpleVaAllocatorTest SetUP" << std::endl;
        symmetricMemory = new SymmetricMemory(0, 2, 4096, nullptr);
    }
    virtual void TearDown()
    {
        delete symmetricMemory;
        GlobalMockObject::verify();
        std::cout << "A SimpleVaAllocatorTest TearDown" << std::endl;
    }
    SymmetricMemory* symmetricMemory;
};

// ==================== Init 方法测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_Init_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    HcclResult ret = symmetricMemory->vaAllocator_->Init(1024 * 1024); // 1MB
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SimpleVaAllocatorTest, ut_Init_When_SizeIsZero_Expect_ReturnHCCL_E_PARA)
{
    HcclResult ret = symmetricMemory->vaAllocator_->Init(0);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SimpleVaAllocatorTest, ut_Init_When_AlreadyInitialized_Expect_ReinitSuccess)
{
    HcclResult ret = symmetricMemory->vaAllocator_->Init(1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 再次初始化
    ret = symmetricMemory->vaAllocator_->Init(2048);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ==================== Reserve 方法测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_NormalAlloc_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset = 0;
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(1024, 1, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset, 0u); // 从0开始分配
}

TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_Align4_Expect_AlignedOffset)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset = 0;

    // 分配一个不对齐的块，然后释放
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(3, 1, offset); // 占用[0,3)
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 现在分配4字节对齐的块
    ret = symmetricMemory->vaAllocator_->Reserve(8, 4, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset % 4, 0u); // 确保对齐
    EXPECT_GE(offset, 3u);     // 应该在第一个块之后
}

TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_AlignPowerOfTwo_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(8192);
    size_t offset = 0;

    // 测试各种2的幂对齐
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(256, 8, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset % 8, 0u);

    ret = symmetricMemory->vaAllocator_->Reserve(512, 64, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset % 64, 0u);

    ret = symmetricMemory->vaAllocator_->Reserve(1024, 128, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset % 128, 0u);
}

TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_NoEnoughMemory_Expect_ReturnHCCL_E_NOMEM)
{
    symmetricMemory->vaAllocator_->Init(1024);
    size_t offset = 0;

    // 尝试分配超过总大小的内存
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(2048, 1, offset);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_FragmentedMemory_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset1, offset2, offset3;

    // 分配三个块，然后释放中间一个，造成碎片
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset1); // [0, 1024)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset2); // [1024, 2048)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset3); // [2048, 3072)

    // 释放中间块
    symmetricMemory->vaAllocator_->Release(1024, 1024);

    // 现在应该能在碎片中分配
    size_t offset = 0;
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(512, 1, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(offset, 1024u); // 应该在释放的块中分配
}

TEST_F(SimpleVaAllocatorTest, ut_Reserve_When_AlignCausesNoSpace_Expect_Fail)
{
    symmetricMemory->vaAllocator_->Init(100);
    size_t offset = 0;

    // 分配占用[0, 90)
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(90, 1, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 现在尝试分配16字节对齐的块，但剩余空间不够对齐
    ret = symmetricMemory->vaAllocator_->Reserve(10, 16, offset);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

// ==================== Release 方法测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_Release_When_Normal_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset = 0;

    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset);
    HcclResult ret = symmetricMemory->vaAllocator_->Release(offset, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_MergeWithPrevBlock_Expect_OneBlock)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset1, offset2;

    // 分配两个连续块
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset1); // [0, 1024)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset2); // [1024, 2048)

    // 释放第一个块
    symmetricMemory->vaAllocator_->Release(offset1, 1024);

    // 释放第二个块，应该与第一个合并
    HcclResult ret = symmetricMemory->vaAllocator_->Release(offset2, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 现在应该只有一个空闲块 [0, 2048)
    // 可以通过内部状态验证，这里省略
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_MergeWithNextBlock_Expect_OneBlock)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset1, offset2;

    // 分配两个连续块
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset1); // [0, 1024)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset2); // [1024, 2048)

    // 先释放第二个块
    symmetricMemory->vaAllocator_->Release(offset2, 1024);

    // 再释放第一个块，应该与第二个合并
    HcclResult ret = symmetricMemory->vaAllocator_->Release(offset1, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_MergeWithBothBlocks_Expect_OneBlock)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset1, offset2, offset3;

    // 分配三个连续块
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset1); // [0, 1024)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset2); // [1024, 2048)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset3); // [2048, 3072)

    // 释放第一个和第三个
    symmetricMemory->vaAllocator_->Release(offset1, 1024);
    symmetricMemory->vaAllocator_->Release(offset3, 1024);

    // 现在释放中间块，应该与前后都合并
    HcclResult ret = symmetricMemory->vaAllocator_->Release(offset2, 1024);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 应该只有一个空闲块 [0, 3072)
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_OutOfBounds_Expect_ReturnHCCL_E_PARAM)
{
    symmetricMemory->vaAllocator_->Init(1024);

    // 尝试释放超出边界的块
    HcclResult ret = symmetricMemory->vaAllocator_->Release(512, 1024); // 512 + 1024 > 1024
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_OverlapWithExisting_Expect_ReturnHCCL_E_PARAM)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset1, offset2, offset3;

    // 分配两个连续块
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset1);            // [0, 1024)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset2);            // [1024, 2048)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset3);            // [1024, 2048)
    HcclResult ret = symmetricMemory->vaAllocator_->Release(1024, 1024); // [512, 1536) 与 [0, 1024) 重叠

    // 尝试释放与第一个块重叠的区域
    ret = symmetricMemory->vaAllocator_->Release(512, 1024); // [512, 1536) 与 [0, 1024) 重叠
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SimpleVaAllocatorTest, ut_Release_When_OverlapWithNextBlock_Expect_ReturnHCCL_E_PARAM)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset;

    // 分配一个块
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset); // [0, 1024)

    // 尝试释放与未来可能分配的区域重叠
    // 先释放一个未分配的区域（这是允许的）
    HcclResult ret = symmetricMemory->vaAllocator_->Release(1024, 1024); // 未分配区域
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 现在分配[1024, 2048)
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset);

    // 尝试释放与已释放区域重叠
    ret = symmetricMemory->vaAllocator_->Release(2000, 100); // [2000, 2100) 与 [2048, 3072) 重叠
    EXPECT_EQ(ret, HCCL_E_PARA);
}

// ==================== 综合场景测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_ComplexScenario_When_MultipleAllocFree_Expect_Correct)
{
    symmetricMemory->vaAllocator_->Init(10000);
    std::vector<std::pair<u32, size_t>> allocations;
    size_t offset;

    // 分配多个不同大小的块
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(1000, 1, offset), HCCL_SUCCESS);
    allocations.emplace_back(offset, 1000);

    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(2000, 64, offset), HCCL_SUCCESS);
    allocations.emplace_back(offset, 2000);

    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(1500, 128, offset), HCCL_SUCCESS);
    allocations.emplace_back(offset, 1500);

    // 释放一些块
    for (int i = 0; i < allocations.size(); i += 2) {
        EXPECT_EQ(symmetricMemory->vaAllocator_->Release(allocations[i].first, allocations[i].second), HCCL_SUCCESS);
    }

    // 再次分配，应该能利用释放的空间
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(800, 1, offset), HCCL_SUCCESS);
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(1200, 1, offset), HCCL_SUCCESS);
}

TEST_F(SimpleVaAllocatorTest, ut_FragmentationScenario_When_SmallHoles_Expect_Utilization)
{
    symmetricMemory->vaAllocator_->Init(1000);
    size_t offset;

    // 创建多个小碎片
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(100, 1, offset), HCCL_SUCCESS);
        // 不释放，创建连续分配
    }

    // 释放奇数位置的块，创建碎片
    for (int i = 1; i < 10; i += 2) {
        EXPECT_EQ(symmetricMemory->vaAllocator_->Release(i * 100, 100), HCCL_SUCCESS);
    }

    // 现在应该能在碎片中分配
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(50, 1, offset), HCCL_SUCCESS);
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(50, 1, offset), HCCL_SUCCESS);

    // 但分配大块可能失败
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(200, 1, offset), HCCL_E_MEMORY);
}

TEST_F(SimpleVaAllocatorTest, ut_AlignmentStressTest_When_VariousAlignments_Expect_AllSuccess)
{
    symmetricMemory->vaAllocator_->Init(1000000); // 1MB
    size_t offset;

    // 测试各种对齐要求
    std::vector<u32> alignments = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};

    for (u32 align : alignments) {
        HcclResult ret = symmetricMemory->vaAllocator_->Reserve(align * 2, align, offset);
        EXPECT_EQ(ret, HCCL_SUCCESS) << "Failed with alignment " << align;
        if (ret == HCCL_SUCCESS) {
            EXPECT_EQ(offset % align, 0u) << "Misaligned with alignment " << align;
        }
    }
}

// ==================== Destroy 方法测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_Destroy_When_HasAllocations_Expect_Cleared)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset;

    // 分配一些内存
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset);
    symmetricMemory->vaAllocator_->Reserve(1024, 1, offset);

    // 销毁分配器
    symmetricMemory->vaAllocator_->Destroy();

    // 重新初始化应该成功
    HcclResult ret = symmetricMemory->vaAllocator_->Init(2048);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 应该可以分配全部空间
    ret = symmetricMemory->vaAllocator_->Reserve(2048, 1, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ==================== 线程安全测试（概念性） ====================
// 注意：实际线程安全测试需要多线程环境
TEST_F(SimpleVaAllocatorTest, ut_ThreadSafety_When_ConcurrentAccess_Expect_NoCrash)
{
    // 这个测试在实际中需要多线程实现
    // 这里只是概念性测试
    symmetricMemory->vaAllocator_->Init(1000000);

    // 可以在这里添加多线程分配和释放的测试
    // 但由于是单元测试框架，通常不包含多线程测试
    // 可以标记为需要进一步测试
    SUCCEED();
}

// ==================== 边界条件测试 ====================
TEST_F(SimpleVaAllocatorTest, ut_BoundaryCondition_When_FullAllocation_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset;

    // 分配全部空间
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(4096, 1, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 再次分配应该失败
    ret = symmetricMemory->vaAllocator_->Reserve(1, 1, offset);
    EXPECT_EQ(ret, HCCL_E_MEMORY);
}

TEST_F(SimpleVaAllocatorTest, ut_BoundaryCondition_When_ExactFitWithAlignment_Expect_Success)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset;

    // 分配几乎全部空间，留出对齐空间
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(4092, 1, offset), HCCL_SUCCESS);

    // 释放
    EXPECT_EQ(symmetricMemory->vaAllocator_->Release(offset, 4092), HCCL_SUCCESS);

    // 现在分配4字节对齐的4字节块
    EXPECT_EQ(symmetricMemory->vaAllocator_->Reserve(4, 4, offset), HCCL_SUCCESS);
    EXPECT_EQ(offset % 4, 0u);
}

TEST_F(SimpleVaAllocatorTest, ut_BoundaryCondition_When_ZeroSizeAllocation_ShouldWork)
{
    symmetricMemory->vaAllocator_->Init(4096);
    size_t offset;

    // 分配0字节（如果允许的话）
    // 注意：这可能取决于实现，当前代码可能允许也可能不允许
    HcclResult ret = symmetricMemory->vaAllocator_->Reserve(0, 1, offset);
    // 根据实际实现设置期望
    // EXPECT_EQ(ret, HCCL_SUCCESS); 或 EXPECT_EQ(ret, HCCL_E_PARAM);
}

// ==================== Symmetric Memory Init 方法测试 ====================
TEST_F(SymmetricMemoryTest, ut_Init_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().with().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_VaAllocator_Is_Nullptr_Expect_ReturnHCCL_E_PTR)
{
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    symmetricMemory.vaAllocator_ = nullptr;
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_SymmetricMemoryAgent_Is_Nullptr_Expect_ReturnHCCL_E_PTR)
{
    SymmetricMemory symmetricMemory(0, 2, TWO_M, nullptr);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_SingleRankCommunicator_Expect_ReturnHCCL_SUCCESS)
{
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 1, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_GetAllocationGranularityFails_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER_CPP(aclrtMemGetAllocationGranularity).stubs().will(returnValue(500000));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_GranularityZero_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER(aclrtMemGetAllocationGranularity).stubs().will(returnValue(ACL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_StrideNotMultipleOfGranularity_Expect_ReturnHCCL_E_PARA)
{
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 4096, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_ReserveMemAddressFails_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER(aclrtReserveMemAddress).stubs().will(returnValue(500000));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_VaAllocatorInitFails_Expect_ReturnHCCL_E_PARA)
{
    MOCKER_CPP(&SymmetricMemory::SimpleVaAllocator::Init).stubs().with().will(returnValue(HCCL_E_PARA));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_SymmetricMemoryAgentInitFails_Expect_ReturnHCCL_E_PARA)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().with().will(returnValue(HCCL_E_PARA));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_GetPidFails_Expect_ReturnHCCL_E_DRV)
{
    MOCKER_CPP(aclrtDeviceGetBareTgid).stubs().with().will(returnValue(500000));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(SymmetricMemoryTest, ut_Init_When_ExchangeInfoFails_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo).stubs().with().will(returnValue(HCCL_E_INTERNAL));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    HcclResult ret = symmetricMemory.Init(); // 1MB
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryTest, ut_AllocSymmetricMem_When_Normal_Expect_ReturnNonNullptr)
{
    u32 tmp = 1;
    void* mockDevWin = static_cast<void*>(&tmp);
    MOCKER_CPP(HcommMemAlloc).stubs().with().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER_CPP(&SymmetricMemory::RegisterSymmetricMem)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockDevWin, sizeof(mockDevWin)))
        .will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    void* devWin = symmetricMemory.AllocSymmetricMem(1048576); // 1MB
    EXPECT_NE(devWin, nullptr);
}

TEST_F(SymmetricMemoryTest, ut_AllocSymmetricMem_When_HcommMemAllocFails_Expect_ReturnNull)
{
    MOCKER_CPP(HcommMemAlloc).stubs().with().will(returnValue(static_cast<HcommResult>(HCCL_E_INTERNAL)));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    void* devWin = symmetricMemory.AllocSymmetricMem(1048576); // 1MB
    EXPECT_EQ(devWin, nullptr);
}

TEST_F(SymmetricMemoryTest, ut_AllocSymmetricMem_When_RegisterSymmetricMemFails_Expect_ReturnNull)
{
    MOCKER_CPP(HcommMemAlloc).stubs().with().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER_CPP(&SymmetricMemory::RegisterSymmetricMem).stubs().with().will(returnValue(HCCL_E_INTERNAL));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    void* devWin = symmetricMemory.AllocSymmetricMem(1048576); // 1MB
    EXPECT_EQ(devWin, nullptr);
}

TEST_F(SymmetricMemoryTest, ut_AddSymmetricWindow_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemory::DeregisterSymmetricMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    HcclResult ret = symmetricMemory.AddSymmetricWindow(pWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_DeleteSymmetricWindow_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemory::DeregisterSymmetricMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    HcclResult ret = symmetricMemory.AddSymmetricWindow(pWin);
    ret = symmetricMemory.DeleteSymmetricWindow(pWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 tmp = 1;
    void* mockDevWin = static_cast<void*>(&tmp);
    MOCKER_CPP(HcommMemAlloc).stubs().with().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER_CPP(&SymmetricMemory::RegisterSymmetricMem)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockDevWin, sizeof(mockDevWin)))
        .will(returnValue(HCCL_SUCCESS));
    void* devWin = symmetricMemory.AllocSymmetricMem(1048576); // 1MB
    ret = symmetricMemory.DeleteSymmetricWindow(devWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_FreeSymmetricMem_When_Normal_Expect_ReturnNonNullptr)
{
    void* mockDevWin = new u32(1);
    MOCKER_CPP(HcommMemAlloc).stubs().with().will(returnValue(static_cast<HcommResult>(HCCL_SUCCESS)));
    MOCKER_CPP(&SymmetricMemory::RegisterSymmetricMem)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&mockDevWin, sizeof(mockDevWin)))
        .will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    void* devWin = symmetricMemory.AllocSymmetricMem(1048576); // 1MB
    HcclResult ret = symmetricMemory.FreeSymmetricMem(devWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_FindSymmetricWindow_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemory::DeregisterSymmetricMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    pWin->userVa = reinterpret_cast<void*>(0x1000);
    pWin->userSize = 4096;
    HcclResult ret = symmetricMemory.AddSymmetricWindow(pWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* ptr = reinterpret_cast<void*>(0x1000);
    void* win;
    u64 offset;
    ret = symmetricMemory.FindSymmetricWindow(ptr, 1024, &win, &offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_FindSymmetricWindow_When_PtrNotInAnyWindow_Expect_ReturnHCCL_E_NOT_FOUND)
{
    MOCKER_CPP(&SymmetricMemory::DeregisterSymmetricMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    void* ptr = reinterpret_cast<void*>(0x1000);
    void* win;
    u64 offset;
    HcclResult ret = symmetricMemory.FindSymmetricWindow(ptr, 1024, &win, &offset);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(SymmetricMemoryTest, ut_FindSymmetricWindow_When_PtrBeforeFirstWindow_Expect_ReturnHCCL_E_NOT_FOUND)
{
    MOCKER_CPP(&SymmetricMemory::DeregisterSymmetricMem).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    std::shared_ptr<SymmetricWindow> pWin(new (std::nothrow) SymmetricWindow());
    pWin->userVa = reinterpret_cast<void*>(0x2000);
    pWin->userSize = 4096;
    HcclResult ret = symmetricMemory.AddSymmetricWindow(pWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* ptr = reinterpret_cast<void*>(0x1000);
    void* win;
    u64 offset;
    ret = symmetricMemory.FindSymmetricWindow(ptr, 1024, &win, &offset);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_ParaError_Expect_ReturnHCCL_E_PARA)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    ret = symmetricMemory.RegisterSymmetricMem(nullptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = symmetricMemory.RegisterSymmetricMem(ptr, 0, &win);
    EXPECT_EQ(ret, HCCL_E_PARA);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(eq(ptr + 1), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    ret = symmetricMemory.RegisterSymmetricMem(ptr + 1, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_PTR);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(eq(ptr + 2), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_ERROR_INTERNAL_ERROR));
    ret = symmetricMemory.RegisterSymmetricMem(ptr + 2, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_PARA);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(eq(ptr + 3), outBoundP(&ptr, sizeof(ptr)), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    ret = symmetricMemory.RegisterSymmetricMem(ptr + 3, TWO_M, &win);
    size_t asize = TWO_M - 1;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(eq(ptr + 4), outBoundP(&ptr, sizeof(ptr)), outBoundP(&asize, sizeof(asize)))
        .will(returnValue(ACL_SUCCESS));
    ret = symmetricMemory.RegisterSymmetricMem(ptr + 4, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_PARA);
    size_t csize = TWO_M;
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(eq(ptr + 5), outBoundP(&ptr, sizeof(ptr)), outBoundP(&csize, sizeof(csize)))
        .will(returnValue(ACL_SUCCESS));
    ret = symmetricMemory.RegisterSymmetricMem(ptr + 5, 2 * TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_TwoRegister_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_Offset_Is_Failed_Expect_ReturnHCCL_E_INTERNAL)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {1, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_MapFailed_Expect_ReturnHCCL_E_DRV)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(aclrtMapMem).stubs().will(returnValue(1));
    HcclResult ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(SymmetricMemoryTest, ut_RegisterSymmetricMem_When_AddSymmetricWindowFailed_Expect_ReturnHCCL_E_OOM)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(returnValue(HCCL_E_OOM));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_E_OOM);
}

HcclResult stub_hrtMalloc(void** devPtr, u64 size, bool Level2Address)
{
    *devPtr = (void*)0x4000000;
    return HCCL_SUCCESS;
}

HcclResult stub_hrtMallocUrma(void** devPtr, u64 size, bool Level2Address)
{
    static uintptr_t devAddr = 0x5000000;
    *devPtr = reinterpret_cast<void*>(devAddr);
    devAddr += 0x1000;
    return HCCL_SUCCESS;
}

TEST_F(SymmetricMemoryTest, ut_DeRegisterSymmetricMem_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMalloc));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(win, (void*)0x4000000);
    ret = symmetricMemory.DeregisterSymmetricMem(win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_DeRegisterSymmetricMem_When_FreePhysical_Is_Failed_Expect_ReturnHCCL_E_DRV)
{
    MOCKER_CPP(&SymmetricMemoryAgent::Init).stubs().will(returnValue(HCCL_SUCCESS));
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, 2 * TWO_M, symmetricMemoryAgent_);
    size_t bsize = TWO_M;
    void* heapBase_ = reinterpret_cast<void*>(0x2000000);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* win;
    void* handle = reinterpret_cast<void*>(0x3000000);
    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(mockcpp::any(), outBoundP(&ptr, sizeof(ptr)), outBoundP(&bsize, sizeof(bsize)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtReserveMemAddressNoUCMemory)
        .stubs()
        .with(outBoundP(&heapBase_, sizeof(heapBase_)), mockcpp::any(), mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handle, sizeof(handle)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMalloc));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    aclrtMemFabricHandle shareableHandle;
    ShareableInfo remoteShareableInfos[2] = {{0, TWO_M, shareableHandle}, {0, TWO_M, shareableHandle}};
    int32_t remoteShareablePids[2] = {0, 1};
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareablePids, sizeof(remoteShareablePids)),
            eq((u64)sizeof(int32_t)))
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&SymmetricMemoryAgent::ExchangeInfo)
        .expects(once())
        .with(
            mockcpp::any(), outBoundP((void*)remoteShareableInfos, sizeof(remoteShareableInfos)),
            eq((u64)sizeof(ShareableInfo)))
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HCCL_SUCCESS;
    ret = symmetricMemory.RegisterSymmetricMem(ptr, TWO_M, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(win, (void*)0x4000000);
    MOCKER_CPP(aclrtFreePhysical).stubs().will(returnValue(1));
    ret = symmetricMemory.DeregisterSymmetricMem(win);
    EXPECT_EQ(ret, HCCL_E_DRV);
}

TEST_F(SymmetricMemoryTest, ut_RegisterUrmaSymmetricMem_When_RepeatRegister_Expect_ReuseSameWin)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    void* ptr = reinterpret_cast<void*>(0x1000000);
    void* firstWin = nullptr;
    void* secondWin = nullptr;

    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(ptr, 0x2000, &firstWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(firstWin, nullptr);
    EXPECT_EQ(symmetricMemory.sortedWindows_.size(), 1U);

    ret = symmetricMemory.RegisterUrmaSymmetricMem(ptr, 0x2000, &secondWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(secondWin, firstWin);
    EXPECT_EQ(symmetricMemory.sortedWindows_.size(), 1U);
    EXPECT_EQ(symmetricMemory.windowMap_.size(), 1U);
    EXPECT_EQ(symmetricMemory.remoteMemMap_.size(), 1U);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(firstWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = symmetricMemory.DeregisterUrmaSymmetricMem(firstWin);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(SymmetricMemoryTest, ut_DeregisterUrmaSymmetricMem_When_SingleRankRepeatDeregister_Expect_NotFound)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 1, 0, SymmetricMemoryMode::URMA);
    void* win = nullptr;
    void* ptr = reinterpret_cast<void*>(0x18000000);

    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(ptr, 0x2000, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(win, nullptr);
    EXPECT_EQ(symmetricMemory.singleRankUrmaWindows_.size(), 1U);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(symmetricMemory.singleRankUrmaWindows_.empty());

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(win);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(SymmetricMemoryTest, ut_RegisterUrmaSymmetricMem_When_SubsetRegister_Expect_ReuseParentWin)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    void* parentPtr = reinterpret_cast<void*>(0x2000000);
    void* subsetPtr = reinterpret_cast<void*>(0x2000800);
    void* parentWin = nullptr;
    void* subsetWin = nullptr;

    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(parentPtr, 0x4000, &parentWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = symmetricMemory.RegisterUrmaSymmetricMem(subsetPtr, 0x1000, &subsetWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(subsetWin, parentWin);
    EXPECT_EQ(symmetricMemory.sortedWindows_.size(), 1U);
    EXPECT_EQ(symmetricMemory.windowMap_.size(), 1U);
    EXPECT_EQ(symmetricMemory.remoteMemMap_.size(), 1U);

    void* foundWin = nullptr;
    size_t offset = 0;
    ret = symmetricMemory.FindUrmaSymmetricWindow(subsetPtr, 0x1000, &foundWin, &offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(foundWin, parentWin);
    EXPECT_EQ(offset, 0x800U);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(parentWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_RegisterUrmaSymmetricMem_When_PartialOverlap_Expect_RegisterIndependently)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    void* firstPtr = reinterpret_cast<void*>(0x3000000);
    void* overlapPtr = reinterpret_cast<void*>(0x3001000);
    void* firstWin = nullptr;
    void* overlapWin = nullptr;

    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(firstPtr, 0x3000, &firstWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = symmetricMemory.RegisterUrmaSymmetricMem(overlapPtr, 0x3000, &overlapWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(overlapWin, nullptr);
    EXPECT_NE(overlapWin, firstWin);
    EXPECT_EQ(symmetricMemory.sortedWindows_.size(), 2U);
    EXPECT_EQ(symmetricMemory.windowMap_.size(), 2U);
    EXPECT_EQ(symmetricMemory.remoteMemMap_.size(), 2U);
    EXPECT_EQ(symmetricMemory.sortedWindows_[0]->userVa, firstPtr);
    EXPECT_EQ(symmetricMemory.sortedWindows_[0]->userSize, 0x3000U);
    EXPECT_EQ(symmetricMemory.sortedWindows_[1]->userVa, overlapPtr);
    EXPECT_EQ(symmetricMemory.sortedWindows_[1]->userSize, 0x3000U);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(firstWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = symmetricMemory.DeregisterUrmaSymmetricMem(overlapWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_TryReuseRegisteredUrmaWindow_When_ExactOrSubset_Expect_Reused)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    void* parentPtr = reinterpret_cast<void*>(0x4000000);
    void* parentWin = nullptr;
    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(parentPtr, 0x4000, &parentWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    bool reused = false;
    void* reusedWin = nullptr;
    ret = symmetricMemory.TryReuseRegisteredUrmaWindow(parentPtr, 0x4000, &reusedWin, reused);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(reused);
    EXPECT_EQ(reusedWin, parentWin);

    reused = false;
    reusedWin = nullptr;
    void* subsetPtr = reinterpret_cast<void*>(0x4001000);
    ret = symmetricMemory.TryReuseRegisteredUrmaWindow(subsetPtr, 0x1000, &reusedWin, reused);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(reused);
    EXPECT_EQ(reusedWin, parentWin);

    reused = false;
    reusedWin = nullptr;
    void* overlapPtr = reinterpret_cast<void*>(0x4003000);
    ret = symmetricMemory.TryReuseRegisteredUrmaWindow(overlapPtr, 0x3000, &reusedWin, reused);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(reused);
    EXPECT_EQ(reusedWin, nullptr);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(parentWin);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_UpdateRemoteMemByTag_When_TagMatched_Expect_UpdateAndMatched)
{
    MOCKER_CPP(hrtMalloc).stubs().will(invoke(stub_hrtMallocUrma));
    MOCKER_CPP(hrtMemSyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(hrtFree).stubs().will(returnValue(HCCL_SUCCESS));

    SymmetricMemory symmetricMemory(0, 2, 0, SymmetricMemoryMode::URMA);
    void* ptr = reinterpret_cast<void*>(0x5000000);
    void* win = nullptr;
    constexpr size_t winSize = 0x2000;
    HcclResult ret = symmetricMemory.RegisterUrmaSymmetricMem(ptr, winSize, &win);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    SymmetricMemoryResource resource;
    resource.memHandle = reinterpret_cast<void*>(0x5100000);
    resource.memTag = std::string(HCCL_SYMMETRIC_MEMORY_TAG_PREFIX) + "ut_addr_"
                      + std::to_string(reinterpret_cast<uintptr_t>(ptr)) + "_size_" + std::to_string(winSize);
    ret = symmetricMemory.SetRegisteredMemoryResource(win, resource);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unordered_map<void*, bool> matchedResources;
    matchedResources.emplace(win, false);
    std::unordered_map<std::string, std::vector<void*>> tagIndex;
    symmetricMemory.BuildRemoteMemTagIndex(tagIndex);
    std::vector<void*> dirtyResources;
    CommMem remoteMem{};
    remoteMem.type = COMM_MEM_TYPE_DEVICE;
    remoteMem.addr = reinterpret_cast<void*>(0x5200000);
    remoteMem.size = winSize;
    ret = symmetricMemory.UpdateRemoteMemByTag(
        1, remoteMem, resource.memTag.c_str(), tagIndex, matchedResources, dirtyResources);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(matchedResources[win]);
    ASSERT_EQ(dirtyResources.size(), 1U);
    EXPECT_EQ(dirtyResources[0], win);

    ret = symmetricMemory.UpdateRemoteMemByTag(
        1, remoteMem, resource.memTag.c_str(), tagIndex, matchedResources, dirtyResources);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ASSERT_EQ(dirtyResources.size(), 1U);

    ret = symmetricMemory.SyncDirtyRemoteMems(dirtyResources);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = symmetricMemory.CheckAllRemoteMemMatched(1, matchedResources);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto remoteMemIt = symmetricMemory.remoteMemMap_.find(win);
    ASSERT_NE(remoteMemIt, symmetricMemory.remoteMemMap_.end());
    ASSERT_EQ(remoteMemIt->second.size(), 2U);
    EXPECT_EQ(remoteMemIt->second[1].addr, remoteMem.addr);
    EXPECT_EQ(remoteMemIt->second[1].size, remoteMem.size);

    ret = symmetricMemory.DeregisterUrmaSymmetricMem(win);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(SymmetricMemoryTest, ut_GetMemoryInfo_When_PtrExceedsBlock_Expect_ReturnHCCL_E_PARA)
{
    std::vector<RankInfo> rankInfoList(2);
    HcclIpAddress localVnicIp;
    std::shared_ptr<SymmetricMemoryAgent> symmetricMemoryAgent_
        = std::make_shared<SymmetricMemoryAgent>(nullptr, 0, 0, localVnicIp, rankInfoList, 0, true, "1");
    SymmetricMemory symmetricMemory(0, 2, TWO_M, symmetricMemoryAgent_);
    symmetricMemory.granularity_ = 4096;

    void* ptr = reinterpret_cast<void*>(0x1000000);
    size_t size = TWO_M;
    void* baseUserVaVal = reinterpret_cast<void*>(0x1000000);
    size_t baseVaSizeVal = 4096;
    void* handleVal = reinterpret_cast<void*>(0x3000000);

    MOCKER_CPP(aclrtMemGetAddressRange)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&baseUserVaVal, sizeof(baseUserVaVal)),
            outBoundP(&baseVaSizeVal, sizeof(baseVaSizeVal)))
        .will(returnValue(ACL_SUCCESS));
    MOCKER_CPP(aclrtMemRetainAllocationHandle)
        .stubs()
        .with(mockcpp::any(), outBoundP(&handleVal, sizeof(handleVal)))
        .will(returnValue(ACL_SUCCESS));

    void* baseUserVa = nullptr;
    size_t baseVaSize = 0;
    aclrtDrvMemHandle paHandle = nullptr;
    HcclResult ret = symmetricMemory.GetMemoryInfo(ptr, size, &baseUserVa, &baseVaSize, &paHandle);
    EXPECT_EQ(ret, HCCL_E_PARA);
}
