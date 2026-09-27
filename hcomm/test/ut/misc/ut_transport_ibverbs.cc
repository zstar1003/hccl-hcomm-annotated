/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#define private public
#define protected public
#include "dispatcher_pub.h"
#include "transport_device_ibverbs_pub.h"
#include "dlhns_function.h"
#include "network/hccp_common.h"
#undef protected
#undef private

using namespace hccl;

class TransportDeviceIbverbsTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

static void FillMinimalMemDetailsQpData(TransportDeviceIbverbsData& d)
{
    d.useMemDetailsMgr = true;
    d.qpsPerConnection = 1U;
    d.qpInfo.resize(1U);
    d.qpInfo[0].qpPtr = 0x5000ULL;
    d.qpInfo[0].sqIndex = 1U;
    d.qpInfo[0].dbIndex = 2U;
    d.multiQpThreshold = HCCL_MULTI_QP_THRESHOLD_DEFAULT;
    d.useAtomicWrite = false;

    RoceMemDetails loc{};
    loc.addr = 0x20000ULL;
    loc.devAddr = 0xC20000ULL;
    loc.size = 4096ULL;
    loc.key = 11U;
    d.localRoceMemDetailsList = {loc};

    RoceMemDetails rem{};
    rem.addr = 0x10000ULL;
    rem.devAddr = 0xC10000ULL;
    rem.size = 4096ULL;
    rem.key = 22U;
    d.remoteRoceMemDetailsList = {rem};
}

TEST_F(TransportDeviceIbverbsTest, Ut_InitMemDetails_InvalidQpSize_Returns_INTERNAL)
{
    HcclDispatcher dispatcherPtr = nullptr;
    ASSERT_EQ(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr), HCCL_SUCCESS);
    ASSERT_NE(dispatcherPtr, nullptr);
    auto* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);

    MachinePara machinePara{};
    machinePara.deviceLogicId = 0;
    TransportDeviceIbverbsData d{};
    d.useMemDetailsMgr = true;
    d.qpsPerConnection = 1U;
    d.qpInfo.resize(3U);
    std::chrono::milliseconds timeout{1};

    TransportDeviceIbverbs link(dispatcher, nullptr, machinePara, timeout, d);
    EXPECT_EQ(link.Init(), HCCL_E_INTERNAL);

    ASSERT_EQ(HcclDispatcherDestroy(dispatcherPtr), HCCL_SUCCESS);
}

TEST_F(TransportDeviceIbverbsTest, Ut_InitMemDetails_Success_Then_ResolveRdmaAddrsAndKeys)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcherPtr = nullptr;
    ASSERT_EQ(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr), HCCL_SUCCESS);
    ASSERT_NE(dispatcherPtr, nullptr);
    auto* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);

    MachinePara machinePara{};
    machinePara.deviceLogicId = 0;
    TransportDeviceIbverbsData d{};
    FillMinimalMemDetailsQpData(d);
    std::chrono::milliseconds timeout{1};

    TransportDeviceIbverbs link(dispatcher, nullptr, machinePara, timeout, d);
    ASSERT_EQ(link.Init(), HCCL_SUCCESS);
    EXPECT_TRUE(link.useMemDetailsLookup_);

    TransportDeviceIbverbs::RdmaAddrKeyResolveParam param{};
    param.remoteAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x10000ULL));
    param.localAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x20000ULL));
    param.length = 4096ULL;
    ASSERT_EQ(link.ResolveRdmaAddrsAndKeys(param), HCCL_SUCCESS);
    EXPECT_EQ(param.dstKey, 22U);
    EXPECT_EQ(param.srcKey, 11U);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(param.transRemoteAddr), static_cast<uintptr_t>(0xC10000ULL));
    EXPECT_EQ(reinterpret_cast<uintptr_t>(param.transLocalAddr), static_cast<uintptr_t>(0xC20000ULL));

    ASSERT_EQ(HcclDispatcherDestroy(dispatcherPtr), HCCL_SUCCESS);
}

TEST_F(TransportDeviceIbverbsTest, Ut_ResolveRdmaAddrsAndKeys_MemDetails_Miss_Returns_INTERNAL)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcherPtr = nullptr;
    ASSERT_EQ(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr), HCCL_SUCCESS);
    auto* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);

    MachinePara machinePara{};
    machinePara.deviceLogicId = 0;
    TransportDeviceIbverbsData d{};
    FillMinimalMemDetailsQpData(d);
    std::chrono::milliseconds timeout{1};

    TransportDeviceIbverbs link(dispatcher, nullptr, machinePara, timeout, d);
    ASSERT_EQ(link.Init(), HCCL_SUCCESS);

    TransportDeviceIbverbs::RdmaAddrKeyResolveParam param{};
    param.remoteAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(0xDEADBEEFULL));
    param.localAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(0x20000ULL));
    param.length = 4096ULL;
    EXPECT_EQ(link.ResolveRdmaAddrsAndKeys(param), HCCL_E_INTERNAL);

    ASSERT_EQ(HcclDispatcherDestroy(dispatcherPtr), HCCL_SUCCESS);
}

TEST_F(TransportDeviceIbverbsTest, Ut_ResolveRdmaKeysFromIoMemRanges_InputAndOutputRanges_Returns_SUCCESS)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcherPtr = nullptr;
    ASSERT_EQ(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr), HCCL_SUCCESS);
    ASSERT_NE(dispatcherPtr, nullptr);
    auto* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);

    MachinePara machinePara{};
    machinePara.deviceLogicId = 0;
    TransportDeviceIbverbsData d{};
    FillMinimalMemDetailsQpData(d);
    std::chrono::milliseconds timeout{1};

    TransportDeviceIbverbs link(dispatcher, nullptr, machinePara, timeout, d);
    ASSERT_EQ(link.Init(), HCCL_SUCCESS);

    link.useMemDetailsLookup_ = false;
    constexpr u64 kRemoteIn = 0x700000ULL;
    constexpr u64 kRemoteOut = 0x800000ULL;
    constexpr u32 kRemoteInKey = 101U;
    constexpr u32 kRemoteOutKey = 202U;
    link.remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].addr
        = reinterpret_cast<void*>(static_cast<uintptr_t>(kRemoteIn));
    link.remoteMemMsg_[static_cast<u32>(MemType::USER_INPUT_MEM)].lkey = kRemoteInKey;
    link.remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].addr
        = reinterpret_cast<void*>(static_cast<uintptr_t>(kRemoteOut));
    link.remoteMemMsg_[static_cast<u32>(MemType::USER_OUTPUT_MEM)].lkey = kRemoteOutKey;

    constexpr u64 kLocalInBase = 0x900000ULL;
    constexpr u64 kLocalInSize = 0x1000ULL;
    constexpr u64 kLocalOutBase = 0xA00000ULL;
    constexpr u64 kLocalOutSize = 0x800ULL;
    constexpr u32 kLocalInKey = 303U;
    constexpr u32 kLocalOutKey = 404U;
    link.localInputMem_.addr = kLocalInBase;
    link.localInputMem_.size = kLocalInSize;
    link.localInputMem_.key = kLocalInKey;
    link.localOutputMem_.addr = kLocalOutBase;
    link.localOutputMem_.size = kLocalOutSize;
    link.localOutputMem_.key = kLocalOutKey;

    TransportDeviceIbverbs::RdmaAddrKeyResolveParam paramIn{};
    paramIn.remoteAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(kRemoteIn + 0x100ULL));
    paramIn.localAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(kLocalInBase + 0x10ULL));
    paramIn.length = 64ULL;
    ASSERT_EQ(link.ResolveRdmaKeysFromIoMemRanges(paramIn), HCCL_SUCCESS);
    EXPECT_EQ(paramIn.dstKey, kRemoteInKey);
    EXPECT_EQ(paramIn.srcKey, kLocalInKey);

    TransportDeviceIbverbs::RdmaAddrKeyResolveParam paramOut{};
    paramOut.remoteAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(kRemoteOut + kLocalOutSize));
    paramOut.localAddr = reinterpret_cast<const void*>(static_cast<uintptr_t>(kLocalOutBase + kLocalOutSize));
    paramOut.length = 1ULL;
    ASSERT_EQ(link.ResolveRdmaKeysFromIoMemRanges(paramOut), HCCL_SUCCESS);
    EXPECT_EQ(paramOut.dstKey, kRemoteOutKey);
    EXPECT_EQ(paramOut.srcKey, kLocalOutKey);

    TransportDeviceIbverbs::RdmaAddrKeyResolveParam viaAddrs{};
    viaAddrs.remoteAddr = paramIn.remoteAddr;
    viaAddrs.localAddr = paramIn.localAddr;
    viaAddrs.length = paramIn.length;
    ASSERT_EQ(link.ResolveRdmaAddrsAndKeys(viaAddrs), HCCL_SUCCESS);
    EXPECT_EQ(viaAddrs.dstKey, kRemoteInKey);
    EXPECT_EQ(viaAddrs.srcKey, kLocalInKey);

    ASSERT_EQ(HcclDispatcherDestroy(dispatcherPtr), HCCL_SUCCESS);
}

class TransportIbverbsTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(TransportIbverbsTest, Ut_Init_UserMemEnableTrue_WithoutInputMem_Returns_PTR)
{
    MachinePara mp{};
    mp.userMemEnable = true;
    mp.notifyNum = 0;
    std::chrono::milliseconds timeout{1};
    TransportIbverbs ib(nullptr, nullptr, mp, timeout);
    EXPECT_EQ(ib.Init(), HCCL_E_PTR);
}

TEST_F(TransportIbverbsTest, Ut_FillExchangeDataTotalSize_UserMemEnable_AddsFiveMemMsg)
{
    MachinePara mpOff{};
    mpOff.userMemEnable = false;
    mpOff.notifyNum = 0;
    mpOff.isIndOp = false;
    std::chrono::milliseconds timeout{1};
    TransportIbverbs ibOff(nullptr, nullptr, mpOff, timeout);
    ibOff.qpsPerConnection_ = 1U;
    ibOff.notifyNum_ = 0U;
    ASSERT_EQ(ibOff.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    const u64 baseSize = ibOff.exchangeDataTotalSize_;

    // userMemEnable=true, notifyNum=0: 只添加 2 个 MemMsg (output and input)
    MachinePara mpOn{};
    mpOn.userMemEnable = true;
    mpOn.notifyNum = 0;
    mpOn.isIndOp = false;
    TransportIbverbs ibOn(nullptr, nullptr, mpOn, timeout);
    ibOn.qpsPerConnection_ = 1U;
    ibOn.notifyNum_ = 0U;
    ASSERT_EQ(ibOn.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    // userMemEnable=true, notifyNum=0: 添加 2 + 3 = 5 个 MemMsg
    EXPECT_EQ(ibOn.exchangeDataTotalSize_, baseSize + 5ULL * static_cast<u64>(sizeof(MemMsg)));
}

TEST_F(TransportIbverbsTest, Ut_FillExchangeDataTotalSize_UserMemEnableTrue_AddsNotifyMemMsg)
{
    std::chrono::milliseconds timeout{1};
    // userMemEnable=false: 不添加 notify 相关内容
    MachinePara base{};
    base.userMemEnable = false;
    base.notifyNum = 0;
    base.isIndOp = false;
    TransportIbverbs ibBase(nullptr, nullptr, base, timeout);
    ibBase.qpsPerConnection_ = 1U;
    ibBase.notifyNum_ = 2U;
    ASSERT_EQ(ibBase.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    const u64 baseSize = ibBase.exchangeDataTotalSize_;

    // userMemEnable=true, notifyNum=3: 添加 2 + 3 = 5 个 MemMsg
    MachinePara mp{};
    mp.userMemEnable = true;
    mp.notifyNum = 3U;
    mp.isIndOp = false;
    TransportIbverbs ib(nullptr, nullptr, mp, timeout);
    ib.qpsPerConnection_ = 1U;
    ib.notifyNum_ = 2U;
    ASSERT_EQ(ib.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    EXPECT_EQ(ib.exchangeDataTotalSize_, baseSize + 5ULL * static_cast<u64>(sizeof(MemMsg)));
}
// 当drainenbale设置为true，此时应创建dataNotify
TEST_F(TransportIbverbsTest, Ut_FillExchangeDataTotalSize_DrainEnableTrue_AddsNotifyMemMsg)
{
    std::chrono::milliseconds timeout{1};
    MachinePara base{};
    base.userMemEnable = false;
    // 一个dataNotify
    base.drainEnable = true;
    base.notifyNum = 0;
    base.isIndOp = false;
    TransportIbverbs ibBase(nullptr, nullptr, base, timeout);
    ibBase.qpsPerConnection_ = 1U;
    ibBase.notifyNum_ = 2U;
    ASSERT_EQ(ibBase.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    const u64 baseSize = ibBase.exchangeDataTotalSize_;

    // userMemEnable=true, notifyNum=3: 添加 2 + 3 = 5 个 MemMsg
    MachinePara mp{};
    mp.userMemEnable = true;
    mp.notifyNum = 3U;
    mp.isIndOp = false;
    TransportIbverbs ib(nullptr, nullptr, mp, timeout);
    ib.qpsPerConnection_ = 1U;
    ib.notifyNum_ = 2U;
    ASSERT_EQ(ib.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    EXPECT_EQ(ib.exchangeDataTotalSize_, baseSize + 3ULL * static_cast<u64>(sizeof(MemMsg)));
}

TEST_F(TransportIbverbsTest, Ut_FillExchangeDataTotalSize_UserMemEnableMultiQp_AddsAllMemMsg)
{
    std::chrono::milliseconds timeout{1};
    MachinePara mp{};
    mp.userMemEnable = false;
    mp.notifyNum = 0U;
    mp.isIndOp = false;
    mp.qpMode = QPMode::OFFLOAD;
    mp.isAicpuModeEn = true;
    mp.deviceType = DevType::DEV_TYPE_910B;
    TransportIbverbs ib0(nullptr, nullptr, mp, timeout);
    ib0.qpsPerConnection_ = 3U;
    ib0.notifyNum_ = 1U;
    ASSERT_EQ(ib0.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    const u64 s0 = ib0.exchangeDataTotalSize_;

    // userMemEnable=true, notifyNum=2: 添加 2 + 3 + 3(multiqp) = 8 个 MemMsg
    MachinePara mp2 = mp;
    mp2.userMemEnable = true;
    mp2.notifyNum = 2U;
    TransportIbverbs ib1(nullptr, nullptr, mp2, timeout);
    ib1.qpsPerConnection_ = 3U;
    ib1.notifyNum_ = 1U;
    ASSERT_EQ(ib1.FillExchangeDataTotalSize(), HCCL_SUCCESS);
    const u64 extra = 8ULL * static_cast<u64>(sizeof(MemMsg));
    EXPECT_EQ(ib1.exchangeDataTotalSize_, s0 + extra);
}

TEST_F(TransportDeviceIbverbsTest, Ut_BatchTransferAsync_NullPtr_Returns_PTR)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcherPtr = nullptr;
    ASSERT_EQ(HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr), HCCL_SUCCESS);
    auto* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);

    MachinePara machinePara{};
    machinePara.deviceLogicId = 0;
    TransportDeviceIbverbsData d{};
    FillMinimalMemDetailsQpData(d);
    std::chrono::milliseconds timeout{1};

    TransportDeviceIbverbs link(dispatcher, nullptr, machinePara, timeout, d);
    ASSERT_EQ(link.Init(), HCCL_SUCCESS);

    Stream stream;
    HcommBatchTransferDesc transferDescs[1];
    transferDescs[0].transType = HCOMM_TRANSFER_TYPE_WRITE;
    transferDescs[0].transferInfo.write.dst = nullptr;
    transferDescs[0].transferInfo.write.src = reinterpret_cast<void*>(static_cast<uintptr_t>(0x20000ULL));
    transferDescs[0].transferInfo.write.len = 1024;

    HcclResult ret = link.BatchTransferAsync(transferDescs, 1, stream);
    EXPECT_EQ(ret, HCCL_E_PTR);

    ASSERT_EQ(HcclDispatcherDestroy(dispatcherPtr), HCCL_SUCCESS);
}
