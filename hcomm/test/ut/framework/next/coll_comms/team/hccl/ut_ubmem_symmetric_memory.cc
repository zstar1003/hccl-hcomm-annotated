/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "mockcpp/mockcpp.hpp"
#include "coll_comm.h"

#define private public
#include "ubmem_symmetric_memory.h"
#include "ubmem_symmetric_memory_agent.h"
#undef private
#include "hcomm_team.h"
#include "hcomm_team_mgr.h"

namespace hccl {
namespace {

    constexpr size_t TEST_GRANULARITY = 0x1000U;
    constexpr uint64_t TEST_STRIDE = 0x4000U;
    constexpr uintptr_t TEST_WINDOW_BASE = 0x100000U;
    constexpr uintptr_t TEST_DEVICE_WINDOW = 0x200000U;

    std::vector<std::vector<uint8_t>> g_recvFrames;
    size_t g_recvFrameIndex = 0U;
    uint32_t g_mapMemCallCount = 0U;
    std::vector<void*> g_boundBaseVas;
    std::vector<size_t> g_boundUserSizes;
    std::vector<CommMem> g_boundFirstMemberMems;

    HcommResult StubBindUbSymmetricWindow(
        HcclCommSymWindow handle, HcommTeamHandle lsaTeam, uint32_t netLayer, const CommMem* memberMems,
        uint32_t memberNum, void* baseVa, size_t stride, size_t userSize)
    {
        (void)handle;
        (void)lsaTeam;
        (void)netLayer;
        (void)stride;
        if (memberMems == nullptr || memberNum == 0U) {
            return HCOMM_E_PTR;
        }
        g_boundBaseVas.emplace_back(baseVa);
        g_boundUserSizes.emplace_back(userSize);
        g_boundFirstMemberMems.emplace_back(memberMems[0]);
        return HCOMM_SUCCESS;
    }

    aclError StubMapMemFailOnSecondCall(
        void* virtualAddress, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
    {
        (void)virtualAddress;
        (void)size;
        (void)offset;
        (void)handle;
        (void)flags;
        ++g_mapMemCallCount;
        return g_mapMemCallCount == 1U ? ACL_SUCCESS : ACL_ERROR_RT_PARAM_INVALID;
    }

    HcclResult StubSocketSend(SocketHandler socketHandle, void* sendBuffer, uint64_t sendSize, uint64_t* sentSize)
    {
        (void)socketHandle;
        (void)sendBuffer;
        *sentSize = sendSize;
        return HCCL_SUCCESS;
    }

    HcclResult StubSocketRecv(SocketHandler socketHandle, void* recvBuffer, uint64_t recvSize, uint64_t* receivedSize)
    {
        (void)socketHandle;
        if (g_recvFrameIndex >= g_recvFrames.size() || g_recvFrames[g_recvFrameIndex].size() != recvSize) {
            return HCCL_E_INTERNAL;
        }
        const auto& frame = g_recvFrames[g_recvFrameIndex++];
        if (memcpy_s(recvBuffer, recvSize, frame.data(), frame.size()) != EOK) {
            return HCCL_E_MEMORY;
        }
        *receivedSize = recvSize;
        return HCCL_SUCCESS;
    }

    void
    SetupLifecycleMocks(void*& allocationBase, size_t& allocationSize, aclrtDrvMemHandle& localHandle, int32_t& pid)
    {
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(once())
            .with(
                allocationBase, outBoundP(&allocationBase, sizeof(allocationBase)),
                outBoundP(&allocationSize, sizeof(allocationSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(once())
            .with(allocationBase, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid).stubs().with(outBoundP(&pid, sizeof(pid))).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER(HcommTeamBindUbSymmetricWindow)
            .expects(once())
            .will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
        MOCKER_CPP(aclrtUnmapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));
    }

    std::unique_ptr<UbMemSymmetricMemory> MakeSymmetricMemory(const std::vector<uint32_t>& worldRankIds = {0U, 2U})
    {
        return std::make_unique<UbMemSymmetricMemory>(nullptr, nullptr, 0U, worldRankIds);
    }

    class TestUbMemSymmetricMemoryWithMock : public testing::Test {
    protected:
        void TearDown() override
        {
            GlobalMockObject::verify();
            g_recvFrames.clear();
            g_recvFrameIndex = 0U;
            g_mapMemCallCount = 0U;
            g_boundBaseVas.clear();
            g_boundUserSizes.clear();
            g_boundFirstMemberMems.clear();
        }
    };

    TEST(TestUbMemSymmetricMemory, Ut_ValidateRegisterRange_When_RangeVaries_Expect_CorrectResult)
    {
        auto memory = MakeSymmetricMemory();

        EXPECT_EQ(memory->ValidateRegisterRange(nullptr, TEST_GRANULARITY), HCCL_E_PTR);
        EXPECT_EQ(memory->ValidateRegisterRange(reinterpret_cast<void*>(TEST_WINDOW_BASE), 0U), HCCL_E_PARA);
        EXPECT_EQ(
            memory->ValidateRegisterRange(reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max() - 1U), 4U),
            HCCL_E_PARA);
        EXPECT_EQ(
            memory->ValidateRegisterRange(reinterpret_cast<void*>(TEST_WINDOW_BASE), TEST_GRANULARITY), HCCL_SUCCESS);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_InitSymmetricVa_When_StrideExceedsHbm_Expect_ParaError)
    {
        CollComm collComm(nullptr, 0U, "ut_stride_hbm", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U};
        UbMemSymmetricMemory memory(&collComm, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        size_t freeHbmSize = 0U;
        size_t totalHbmSize = 1U;
        MOCKER_CPP(aclrtGetMemInfo)
            .expects(once())
            .with(
                ACL_HBM_MEM_HUGE, outBoundP(&freeHbmSize, sizeof(freeHbmSize)),
                outBoundP(&totalHbmSize, sizeof(totalHbmSize)))
            .will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory.InitSymmetricVa(), HCCL_E_PARA);
        EXPECT_EQ(memory.arenaBase_, nullptr);
    }

    TEST(TestUbMemSymmetricMemory, Ut_AllocateWindowOffset_When_AllocateAndRelease_Expect_ReuseGap)
    {
        auto memory = MakeSymmetricMemory();
        memory->granularity_ = TEST_GRANULARITY;
        memory->stride_ = TEST_STRIDE;
        ASSERT_EQ(memory->InitWindowOffsetAllocator(), HCCL_SUCCESS);

        uint64_t firstOffset = TEST_STRIDE;
        uint64_t secondOffset = TEST_STRIDE;
        ASSERT_EQ(memory->AllocateWindowOffset(TEST_GRANULARITY, firstOffset), HCCL_SUCCESS);
        ASSERT_EQ(memory->AllocateWindowOffset(2U * TEST_GRANULARITY, secondOffset), HCCL_SUCCESS);
        EXPECT_EQ(firstOffset, 0U);
        EXPECT_EQ(secondOffset, TEST_GRANULARITY);

        ASSERT_EQ(memory->ReleaseWindowOffset(firstOffset, TEST_GRANULARITY), HCCL_SUCCESS);
        uint64_t reusedOffset = TEST_STRIDE;
        ASSERT_EQ(memory->AllocateWindowOffset(TEST_GRANULARITY, reusedOffset), HCCL_SUCCESS);
        EXPECT_EQ(reusedOffset, 0U);

        ASSERT_EQ(memory->ReleaseWindowOffset(reusedOffset, TEST_GRANULARITY), HCCL_SUCCESS);
        ASSERT_EQ(memory->ReleaseWindowOffset(secondOffset, 2U * TEST_GRANULARITY), HCCL_SUCCESS);
        uint64_t mergedOffset = TEST_STRIDE;
        EXPECT_EQ(memory->AllocateWindowOffset(TEST_STRIDE, mergedOffset), HCCL_SUCCESS);
        EXPECT_EQ(mergedOffset, 0U);
    }

    TEST(TestUbMemSymmetricMemory, Ut_AllocateWindowOffset_When_InvalidOrExhausted_Expect_Error)
    {
        auto memory = MakeSymmetricMemory();
        memory->granularity_ = TEST_GRANULARITY;
        memory->stride_ = TEST_STRIDE;
        ASSERT_EQ(memory->InitWindowOffsetAllocator(), HCCL_SUCCESS);

        uint64_t offset = 0U;
        EXPECT_EQ(memory->AllocateWindowOffset(TEST_GRANULARITY - 1U, offset), HCCL_E_PARA);
        EXPECT_EQ(memory->AllocateWindowOffset(TEST_STRIDE + TEST_GRANULARITY, offset), HCCL_E_MEMORY);
        ASSERT_EQ(memory->AllocateWindowOffset(TEST_STRIDE, offset), HCCL_SUCCESS);
        EXPECT_EQ(memory->AllocateWindowOffset(TEST_GRANULARITY, offset), HCCL_E_MEMORY);
    }

    TEST(TestUbMemSymmetricMemory, Ut_ValidateMappingRange_When_RangeVaries_Expect_CorrectResult)
    {
        auto memory = MakeSymmetricMemory();
        memory->arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        memory->stride_ = TEST_STRIDE;
        memory->granularity_ = TEST_GRANULARITY;

        EXPECT_EQ(memory->ValidateMappingRange(TEST_GRANULARITY, 2U * TEST_GRANULARITY), HCCL_SUCCESS);
        EXPECT_EQ(memory->ValidateMappingRange(1U, TEST_GRANULARITY), HCCL_E_PARA);
        EXPECT_EQ(memory->ValidateMappingRange(TEST_GRANULARITY, TEST_GRANULARITY - 1U), HCCL_E_PARA);
        EXPECT_EQ(memory->ValidateMappingRange(TEST_STRIDE, TEST_GRANULARITY), HCCL_E_PARA);

        memory->arenaBase_ = nullptr;
    }

    TEST(TestUbMemSymmetricMemory, Ut_ImportLocalHandle_When_SelfMember_Expect_BorrowedHandle)
    {
        auto memory = MakeSymmetricMemory();
        memory->selfMember_ = 1U;
        UbMemSymmetricMemory::VaMappingInfo mapping;
        mapping.localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        aclrtDrvMemHandle importedHandle = nullptr;
        bool ownsHandle = true;

        EXPECT_EQ(memory->ImportMemberHandle(1U, {}, mapping, importedHandle, ownsHandle), HCCL_SUCCESS);
        EXPECT_EQ(importedHandle, mapping.localHandle);
        EXPECT_FALSE(ownsHandle);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_ImportMemberHandle_When_RemoteMember_Expect_OwnedImportedHandle)
    {
        auto memory = MakeSymmetricMemory();
        memory->selfMember_ = 0U;
        UbMemSymmetricMemory::VaMappingInfo mapping;
        std::vector<uint8_t> shareableDesc(sizeof(aclrtMemFabricHandle), 1U);
        aclrtDrvMemHandle expectedHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        aclrtDrvMemHandle importedHandle = nullptr;
        bool ownsHandle = false;

        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(
                mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U,
                outBoundP(&expectedHandle, sizeof(expectedHandle)))
            .will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory->ImportMemberHandle(1U, shareableDesc, mapping, importedHandle, ownsHandle), HCCL_SUCCESS);
        EXPECT_EQ(importedHandle, expectedHandle);
        EXPECT_TRUE(ownsHandle);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_MapAllMembers_When_PartialMapFails_Expect_ImmediateCleanup)
    {
        auto memory = MakeSymmetricMemory();
        memory->arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + TEST_STRIDE);
        memory->stride_ = TEST_STRIDE;
        memory->granularity_ = TEST_GRANULARITY;
        memory->selfMember_ = 0U;
        memory->lsaTeamSize_ = 2U;
        UbMemSymmetricMemory::VaMappingInfo mapping;
        mapping.allocationSize = TEST_GRANULARITY;
        mapping.localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        std::vector<std::vector<uint8_t>> shareableDescs(2U, std::vector<uint8_t>(sizeof(aclrtMemFabricHandle), 1U));
        aclrtDrvMemHandle remoteHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        std::vector<CommMem> memberMems;

        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U, outBoundP(&remoteHandle, sizeof(remoteHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).stubs().will(invoke(StubMapMemFailOnSecondCall));
        MOCKER_CPP(aclrtUnmapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(once()).with(remoteHandle).will(returnValue(ACL_SUCCESS));

        EXPECT_EQ(memory->MapAllMembers(0U, shareableDescs, mapping, TEST_GRANULARITY, memberMems), HCCL_E_RUNTIME);
        EXPECT_EQ(g_mapMemCallCount, 2U);
        EXPECT_EQ(memory->activeMappingCount_, 0U);
        EXPECT_TRUE(mapping.peers.empty());
        EXPECT_EQ(mapping.localHandle, nullptr);

        memory->arenaBase_ = nullptr;
    }

    TEST(TestUbMemSymmetricMemoryAgent, Ut_ExchangeInfo_When_SingleMember_Expect_LocalData)
    {
        const std::vector<uint32_t> ranks{3U};
        UbMemSymmetricMemoryAgent agent(nullptr, 3U, ranks, 0U, "ut_single");
        agent.neighborLinksChecked_ = true;
        agent.initialized_ = true;
        uint32_t localValue = 7U;
        uint32_t output = 0U;

        EXPECT_EQ(agent.ExchangeInfo(&localValue, &output, sizeof(localValue)), HCCL_SUCCESS);
        EXPECT_EQ(output, localValue);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_ExchangeInfo_When_TwoMembers_Expect_MemberOrderedData)
    {
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemoryAgent agent(nullptr, 0U, ranks, 0U, "ut_multi");
        agent.neighborLinksChecked_ = true;
        agent.initialized_ = true;
        agent.leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        agent.rightSocket_ = agent.leftSocket_;
        uint32_t localValue = 3U;
        uint32_t remoteValue = 5U;
        UbmemPacket remotePacket{};
        remotePacket.type = UbmemPacketType::DATA;
        remotePacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePacket.data, sizeof(remotePacket.data), &remoteValue, sizeof(remoteValue)), EOK);
        const auto* packetBegin = reinterpret_cast<const uint8_t*>(&remotePacket);
        g_recvFrames = {std::vector<uint8_t>(packetBegin, packetBegin + sizeof(remotePacket))};
        MOCKER(SocketSendNb).stubs().will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).stubs().will(invoke(StubSocketRecv));

        std::vector<uint32_t> output(ranks.size(), 0U);
        EXPECT_EQ(agent.ExchangeInfo(&localValue, output.data(), sizeof(localValue)), HCCL_SUCCESS);
        EXPECT_EQ(output[0], localValue);
        EXPECT_EQ(output[1], remoteValue);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());

        agent.leftSocket_ = nullptr;
        agent.rightSocket_ = nullptr;
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_Finalize_When_WindowActive_Expect_AllResourcesReleased)
    {
        auto memory = MakeSymmetricMemory({0U});
        auto paMapping = std::make_shared<UbMemSymmetricMemory::PaMappingInfo>();
        paMapping->paHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE);
        paMapping->heapBaseOffset = 0U;
        paMapping->refCount = 1U;
        paMapping->vaMapping.allocationSize = TEST_GRANULARITY;
        paMapping->vaMapping.localHandle = paMapping->paHandle;
        paMapping->vaMapping.peers.resize(1U);
        paMapping->vaMapping.peers[0].handle = paMapping->paHandle;
        paMapping->vaMapping.peers[0].address = reinterpret_cast<void*>(TEST_WINDOW_BASE + TEST_STRIDE);
        paMapping->vaMapping.peers[0].mapped = true;
        auto record = std::make_unique<UbMemSymmetricMemory::WindowRecord>();
        record->userBase = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        record->userSize = TEST_GRANULARITY;
        record->deviceWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        record->paMapping = paMapping;
        record->mappingRefHeld = true;
        record->state = UbMemSymmetricMemory::WindowState::ACTIVE;
        memory->arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE);
        memory->granularity_ = TEST_GRANULARITY;
        memory->stride_ = TEST_STRIDE;
        ASSERT_EQ(memory->InitWindowOffsetAllocator(), HCCL_SUCCESS);
        uint64_t windowOffset = TEST_STRIDE;
        ASSERT_EQ(memory->AllocateWindowOffset(TEST_GRANULARITY, windowOffset), HCCL_SUCCESS);
        memory->activeMappingCount_ = 1U;
        memory->paMappingMap_.emplace(paMapping->paHandle, paMapping);
        memory->windowsByHandle_.emplace(record->deviceWindow, record.get());
        memory->windowsByAddress_.emplace(TEST_WINDOW_BASE, std::move(record));
        HcclComm ownerComm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        ASSERT_EQ(RecordHcommWindowOwner(reinterpret_cast<void*>(TEST_DEVICE_WINDOW), ownerComm), HCCL_SUCCESS);

        MOCKER_CPP(aclrtUnmapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(never());
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        memory->Finalize();

        EXPECT_TRUE(memory->finalized_);
        EXPECT_TRUE(memory->windowsByAddress_.empty());
        EXPECT_TRUE(memory->windowsByHandle_.empty());
        EXPECT_TRUE(memory->paMappingMap_.empty());
        EXPECT_EQ(memory->activeMappingCount_, 0U);
        EXPECT_EQ(memory->arenaBase_, nullptr);
        ownerComm = nullptr;
        EXPECT_EQ(GetHcommWindowComm(reinterpret_cast<void*>(TEST_DEVICE_WINDOW), ownerComm), HCCL_E_NOT_FOUND);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterAndDeregister_When_SingleMember_Expect_LifecycleComplete)
    {
        CollComm owner(nullptr, 0U, "ut_lifecycle", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U};
        UbMemSymmetricMemory memory(&owner, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0U, ranks, 0U, "ut_lifecycle");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->initialized_ = true;
        memory.arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE);
        memory.granularity_ = TEST_GRANULARITY;
        memory.stride_ = TEST_STRIDE;
        ASSERT_EQ(memory.InitWindowOffsetAllocator(), HCCL_SUCCESS);
        void* allocationBase = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t allocationSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        void* deviceWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        int32_t pid = 1;
        SetupLifecycleMocks(allocationBase, allocationSize, localHandle, pid);
        MOCKER_CPP(aclrtFreePhysical).expects(never());

        HcclCommSymWindow window = deviceWindow;
        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        ASSERT_EQ(memory.RegisterWindow(allocationBase, allocationSize, window, comm), HCCL_SUCCESS);
        EXPECT_EQ(window, deviceWindow);
        HcclComm ownerComm = nullptr;
        EXPECT_EQ(GetHcommWindowComm(window, ownerComm), HCCL_SUCCESS);
        EXPECT_EQ(ownerComm, comm);
        EXPECT_EQ(memory.windowsByHandle_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.activeMappingCount_, 1U);

        EXPECT_EQ(memory.DeregisterWindow(window), HCCL_SUCCESS);
        EXPECT_TRUE(memory.windowsByAddress_.empty());
        EXPECT_TRUE(memory.windowsByHandle_.empty());
        EXPECT_TRUE(memory.paMappingMap_.empty());
        EXPECT_EQ(memory.activeMappingCount_, 0U);
        EraseHcommWindowOwner(window);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterWindow_When_PostExchangeMapFails_Expect_BrokenAndUnavailable)
    {
        CollComm owner(nullptr, 0U, "ut_broken", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U};
        UbMemSymmetricMemory memory(&owner, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0U, ranks, 0U, "ut_broken");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->initialized_ = true;
        memory.arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE);
        memory.granularity_ = TEST_GRANULARITY;
        memory.stride_ = TEST_STRIDE;
        ASSERT_EQ(memory.InitWindowOffsetAllocator(), HCCL_SUCCESS);

        void* allocationBase = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t allocationSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        int32_t pid = 1;

        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(once())
            .with(
                allocationBase, outBoundP(&allocationBase, sizeof(allocationBase)),
                outBoundP(&allocationSize, sizeof(allocationSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(once())
            .with(allocationBase, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid).stubs().with(outBoundP(&pid, sizeof(pid))).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(once()).will(returnValue(ACL_ERROR_RT_PARAM_INVALID));
        MOCKER_CPP(aclrtFreePhysical).expects(never());
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclCommSymWindow window = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        EXPECT_EQ(memory.RegisterWindow(allocationBase, allocationSize, window, comm), HCCL_E_RUNTIME);
        EXPECT_TRUE(memory.broken_);
        EXPECT_TRUE(memory.windowsByHandle_.empty());
        EXPECT_TRUE(memory.paMappingMap_.empty());
        EXPECT_EQ(memory.RegisterWindow(allocationBase, allocationSize, window, comm), HCCL_E_UNAVAIL);
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterAndDeregister_When_TwoMembers_Expect_LifecycleComplete)
    {
        CollComm collComm(nullptr, 0U, "ut_multi_lifecycle", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U, 2U};
        UbMemSymmetricMemory memory(&collComm, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0U, ranks, 0U, "ut_multi_lifecycle");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->initialized_ = true;
        memory.agent_->leftSocket_ = reinterpret_cast<SocketHandler>(TEST_WINDOW_BASE);
        memory.agent_->rightSocket_ = memory.agent_->leftSocket_;
        memory.arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE);
        memory.granularity_ = TEST_GRANULARITY;
        memory.stride_ = TEST_STRIDE;
        ASSERT_EQ(memory.InitWindowOffsetAllocator(), HCCL_SUCCESS);

        void* allocationBase = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t allocationSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        aclrtDrvMemHandle remoteHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + 2U * TEST_GRANULARITY);
        int32_t localPid = 1;
        int32_t remotePid = 2;

        UbmemPacket remotePidPacket{};
        remotePidPacket.type = UbmemPacketType::DATA;
        remotePidPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remotePidPacket.data, sizeof(remotePidPacket.data), &remotePid, sizeof(remotePid)), EOK);
        UbmemShareableInfo remoteInfo{};
        remoteInfo.offset = 0U;
        remoteInfo.size = allocationSize;
        int32_t remotePrepareResult = static_cast<int32_t>(HCCL_SUCCESS);
        UbmemPacket remotePreparePacket{};
        remotePreparePacket.type = UbmemPacketType::DATA;
        remotePreparePacket.memberId = 1U;
        ASSERT_EQ(
            memcpy_s(
                remotePreparePacket.data, sizeof(remotePreparePacket.data), &remotePrepareResult,
                sizeof(remotePrepareResult)),
            EOK);
        UbmemPacket remoteGrantPacket = remotePreparePacket;
        UbmemPacket remoteInfoPacket{};
        remoteInfoPacket.type = UbmemPacketType::DATA;
        remoteInfoPacket.memberId = 1U;
        ASSERT_EQ(memcpy_s(remoteInfoPacket.data, sizeof(remoteInfoPacket.data), &remoteInfo, sizeof(remoteInfo)), EOK);
        const auto* pidPacketBegin = reinterpret_cast<const uint8_t*>(&remotePidPacket);
        const auto* preparePacketBegin = reinterpret_cast<const uint8_t*>(&remotePreparePacket);
        const auto* infoPacketBegin = reinterpret_cast<const uint8_t*>(&remoteInfoPacket);
        const auto* grantPacketBegin = reinterpret_cast<const uint8_t*>(&remoteGrantPacket);
        g_recvFrames
            = {std::vector<uint8_t>(pidPacketBegin, pidPacketBegin + sizeof(remotePidPacket)),
               std::vector<uint8_t>(preparePacketBegin, preparePacketBegin + sizeof(remotePreparePacket)),
               std::vector<uint8_t>(infoPacketBegin, infoPacketBegin + sizeof(remoteInfoPacket)),
               std::vector<uint8_t>(grantPacketBegin, grantPacketBegin + sizeof(remoteGrantPacket))};

        MOCKER(SocketSendNb).expects(exactly(4)).will(invoke(StubSocketSend));
        MOCKER(SocketRecvNb).expects(exactly(4)).will(invoke(StubSocketRecv));
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(once())
            .with(
                allocationBase, outBoundP(&allocationBase, sizeof(allocationBase)),
                outBoundP(&allocationSize, sizeof(allocationSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(once())
            .with(allocationBase, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid)
            .expects(once())
            .with(outBoundP(&localPid, sizeof(localPid)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemImportFromShareableHandleV2)
            .expects(once())
            .with(mockcpp::any(), ACL_MEM_SHARE_HANDLE_TYPE_FABRIC, 0U, outBoundP(&remoteHandle, sizeof(remoteHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(exactly(2)).will(returnValue(ACL_SUCCESS));
        MOCKER(HcommTeamBindUbSymmetricWindow)
            .expects(once())
            .will(returnValue(static_cast<HcommResult>(HCOMM_SUCCESS)));
        MOCKER_CPP(aclrtUnmapMem).expects(exactly(2)).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(once()).with(remoteHandle).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclCommSymWindow window = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        ASSERT_EQ(memory.RegisterWindow(allocationBase, allocationSize, window, comm), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        const auto& paMapping = memory.paMappingMap_.begin()->second;
        ASSERT_NE(paMapping, nullptr);
        ASSERT_EQ(paMapping->memberMems.size(), ranks.size());
        ASSERT_EQ(paMapping->vaMapping.peers.size(), ranks.size());
        EXPECT_FALSE(paMapping->vaMapping.peers[0].ownsHandle);
        EXPECT_TRUE(paMapping->vaMapping.peers[1].ownsHandle);
        EXPECT_EQ(paMapping->memberMems[0].addr, memory.arenaBase_);
        EXPECT_EQ(
            paMapping->memberMems[1].addr, static_cast<void*>(static_cast<uint8_t*>(memory.arenaBase_) + TEST_STRIDE));
        EXPECT_EQ(paMapping->memberMems[0].size, allocationSize);
        EXPECT_EQ(paMapping->memberMems[1].size, allocationSize);
        EXPECT_EQ(memory.activeMappingCount_, ranks.size());

        EXPECT_EQ(memory.DeregisterWindow(window), HCCL_SUCCESS);
        EXPECT_TRUE(memory.windowsByAddress_.empty());
        EXPECT_TRUE(memory.windowsByHandle_.empty());
        EXPECT_TRUE(memory.paMappingMap_.empty());
        EXPECT_EQ(memory.activeMappingCount_, 0U);
        EXPECT_EQ(g_recvFrameIndex, g_recvFrames.size());
        EraseHcommWindowOwner(window);
        memory.agent_->leftSocket_ = nullptr;
        memory.agent_->rightSocket_ = nullptr;
    }

    TEST_F(TestUbMemSymmetricMemoryWithMock, Ut_RegisterOverlap_When_SameAllocation_Expect_ReusePaMapping)
    {
        CollComm owner(nullptr, 0U, "ut_overlap", ManagerCallbacks{}, CollCommInitMode::simpleMode);
        const std::vector<uint32_t> ranks{0U};
        UbMemSymmetricMemory memory(&owner, reinterpret_cast<HcommTeamHandle>(TEST_WINDOW_BASE), 0U, ranks);
        memory.agent_ = std::make_unique<UbMemSymmetricMemoryAgent>(nullptr, 0U, ranks, 0U, "ut_overlap");
        memory.agent_->neighborLinksChecked_ = true;
        memory.agent_->initialized_ = true;
        memory.arenaBase_ = reinterpret_cast<void*>(TEST_WINDOW_BASE + 2U * TEST_STRIDE);
        memory.granularity_ = TEST_GRANULARITY;
        memory.stride_ = TEST_STRIDE;
        ASSERT_EQ(memory.InitWindowOffsetAllocator(), HCCL_SUCCESS);

        void* allocationBase = reinterpret_cast<void*>(TEST_WINDOW_BASE);
        size_t allocationSize = TEST_GRANULARITY;
        aclrtDrvMemHandle localHandle = reinterpret_cast<aclrtDrvMemHandle>(TEST_WINDOW_BASE + TEST_GRANULARITY);
        int32_t pid = 1;
        MOCKER_CPP(aclrtMemGetAddressRange)
            .expects(exactly(3))
            .with(
                mockcpp::any(), outBoundP(&allocationBase, sizeof(allocationBase)),
                outBoundP(&allocationSize, sizeof(allocationSize)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemRetainAllocationHandle)
            .expects(exactly(3))
            .with(allocationBase, outBoundP(&localHandle, sizeof(localHandle)))
            .will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemExportToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtDeviceGetBareTgid).stubs().with(outBoundP(&pid, sizeof(pid))).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMemSetPidToShareableHandleV2).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtMapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER(HcommTeamBindUbSymmetricWindow).expects(exactly(3)).will(invoke(StubBindUbSymmetricWindow));
        MOCKER_CPP(aclrtUnmapMem).expects(once()).will(returnValue(ACL_SUCCESS));
        MOCKER_CPP(aclrtFreePhysical).expects(never());
        MOCKER_CPP(aclrtReleaseMemAddress).expects(once()).will(returnValue(ACL_SUCCESS));

        HcclComm comm = reinterpret_cast<HcclComm>(TEST_WINDOW_BASE + 3U * TEST_STRIDE);
        HcclCommSymWindow firstWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW);
        HcclCommSymWindow overlapWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW + TEST_GRANULARITY);
        HcclCommSymWindow duplicateWindow = reinterpret_cast<void*>(TEST_DEVICE_WINDOW + 2U * TEST_GRANULARITY);
        ASSERT_EQ(memory.RegisterWindow(allocationBase, allocationSize, firstWindow, comm), HCCL_SUCCESS);
        ASSERT_EQ(
            memory.RegisterWindow(
                reinterpret_cast<void*>(TEST_WINDOW_BASE + 0x80U), allocationSize - 0x80U, overlapWindow, comm),
            HCCL_SUCCESS);
        ASSERT_EQ(memory.RegisterWindow(allocationBase, allocationSize, duplicateWindow, comm), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 3U);
        auto overlapRecordIter = memory.windowsByHandle_.find(overlapWindow);
        ASSERT_NE(overlapRecordIter, memory.windowsByHandle_.end());
        ASSERT_NE(overlapRecordIter->second, nullptr);
        EXPECT_EQ(overlapRecordIter->second->userBase, reinterpret_cast<void*>(TEST_WINDOW_BASE + 0x80U));
        EXPECT_EQ(overlapRecordIter->second->userSize, allocationSize - 0x80U);
        EXPECT_EQ(overlapRecordIter->second->userOffset, 0x80U);
        ASSERT_EQ(g_boundBaseVas.size(), 3U);
        ASSERT_EQ(g_boundUserSizes.size(), 3U);
        ASSERT_EQ(g_boundFirstMemberMems.size(), 3U);
        void* mappedBase = memory.arenaBase_;
        EXPECT_EQ(g_boundBaseVas[0], mappedBase);
        EXPECT_EQ(g_boundBaseVas[1], static_cast<void*>(static_cast<uint8_t*>(mappedBase) + 0x80U));
        EXPECT_EQ(g_boundBaseVas[2], mappedBase);
        EXPECT_EQ(g_boundUserSizes[0], allocationSize);
        EXPECT_EQ(g_boundUserSizes[1], allocationSize - 0x80U);
        EXPECT_EQ(g_boundUserSizes[2], allocationSize);
        EXPECT_EQ(g_boundFirstMemberMems[1].addr, g_boundBaseVas[1]);
        EXPECT_EQ(g_boundFirstMemberMems[1].size, allocationSize - 0x80U);

        EXPECT_EQ(memory.DeregisterWindow(duplicateWindow), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 2U);
        EXPECT_EQ(memory.activeMappingCount_, 1U);
        EXPECT_EQ(memory.DeregisterWindow(overlapWindow), HCCL_SUCCESS);
        ASSERT_EQ(memory.paMappingMap_.size(), 1U);
        EXPECT_EQ(memory.paMappingMap_.begin()->second->refCount, 1U);
        EXPECT_EQ(memory.activeMappingCount_, 1U);
        EXPECT_EQ(memory.DeregisterWindow(firstWindow), HCCL_SUCCESS);
        EXPECT_TRUE(memory.paMappingMap_.empty());
        EXPECT_EQ(memory.activeMappingCount_, 0U);
        EraseHcommWindowOwner(firstWindow);
        EraseHcommWindowOwner(overlapWindow);
        EraseHcommWindowOwner(duplicateWindow);
    }

} // namespace
} // namespace hccl
