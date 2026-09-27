/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <vector>

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "adapter_hal.h"
#include "aicpu/device/aicpu_channel_process.h"
#include "aicpu/device/dev_aicpu_ts_channel_mgr.h"
#include "channel_param.h"
#include "dlhns_function.h"
#include "externalinput_pub.h"
#include "hccl_dispatcher_ctx.h"
#include "workflow_pub.h"
#include "binary_stream.h"
#include "roce_transport_lite_impl.h"
#include "ub_transport_lite_impl.h"

using namespace hccl;

namespace {
constexpr uint32_t kKindTsRoce = 2U;

HcclResult StubHrtDrvGetLocalDevIDByHostDevID(u32 /*hostUdevid*/, u32* localDevid)
{
    if (localDevid != nullptr) {
        *localDevid = 0U;
    }
    return HCCL_SUCCESS;
}

HcclResult StubHrtDrvGetPlatformInfo(uint32_t* info)
{
    if (info != nullptr) {
        *info = 1U;
    }
    return HCCL_SUCCESS;
}
} // namespace

namespace {
// 单个 RmaBuffer 的 uniqueId（addr/size/lkey）。ROCE 的 ParseNotifyValueBuffer 无 ==0 guard，
// 最小分发 blob 的 notifyValue 段必须塞一个合法 buffer，故保留此 helper。
static std::vector<char> BuildSingleRmaBufferUniqueId(u64 addr, u64 size, u32 key)
{
    Hccl::BinaryStream bs;
    bs << addr;
    bs << size;
    bs << key;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

// ParsePackData 入参 data 是「首元素为 vector<char> transpUniqueId」的 BinaryStream 序列化，
// 等价于 InitUrmaChannel 里 AicpuResPackageHelper::ParsePackedData(...)[STREAM].data。
static std::vector<char> BuildOuterPackData(const std::vector<char>& transpUniqueId)
{
    Hccl::BinaryStream bs;
    bs << transpUniqueId;
    std::vector<char> result;
    bs.Dump(result);
    return result;
}

// 构造完整的最小 ROCE 分发 blob（外层 pack 后）：type=ROCE + notifyNum/bufferNum/connNum=0 + 各段空 vector，
// notifyValue 段必须塞一个合法 RmaBuffer uniqueId（ROCE 的 ParseNotifyValueBuffer 无 ==0 guard）。
// 供 Ut_ParsePackData_WhenRoceBlob 与 Ut_ParsePackData_WhenTwoRoceBlobs 复用。
static std::vector<char> BuildRocePackData()
{
    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::ROCE);
    inner << static_cast<u32>(0); // notifyNum
    inner << static_cast<u32>(0); // bufferNum
    inner << static_cast<u32>(0); // connNum
    inner << std::vector<char>(); // locNotify（notifyNum==0，ParseLocNotifyVec 提前 return）
    inner << std::vector<char>(); // rmtNotify
    inner << BuildSingleRmaBufferUniqueId(0x3000, 0x200, 0x300); // notifyValue（无 guard，必须非空）
    inner << std::vector<char>();                                // locBuffer（bufferNum==0，提前 return）
    inner << std::vector<char>();                                // rmtBuffer
    inner << std::vector<char>();                                // conn（connNum==0，提前 return）
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);
    return BuildOuterPackData(transpUniqueId);
}

// 构造完整的最小 P2P 分发 blob（外层 pack 后）。P2P 的 Init 无 connNum/notifyValue 段，
// 仅 type=P2P + notifyNum/bufferNum=0 + 4 个空 vector。
static std::vector<char> BuildP2pPackData()
{
    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::P2P);
    inner << static_cast<u32>(0); // notifyNum
    inner << static_cast<u32>(0); // bufferNum
    inner << std::vector<char>(); // locNotify（notifyNum==0，ParseLocNotifyVec 提前 return）
    inner << std::vector<char>(); // rmtNotify
    inner << std::vector<char>(); // locBuffer（bufferNum==0，ParseRmtBufferVec 提前 return）
    inner << std::vector<char>(); // rmtBuffer
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);
    return BuildOuterPackData(transpUniqueId);
}

// 分发 + 断言 handle 非空 + 销毁单个 channel：单 channel 成功用例的公共尾部。
static void ParsePackDataAndDestroy(std::vector<char>& data)
{
    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    EXPECT_NE(handle, 0ULL);
    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}
} // namespace

class AicpuChannelProcessTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenCommParamNull_Returns_PTR)
{
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(nullptr), HCCL_E_PTR);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenChannelListNull_Returns_PTR)
{
    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = nullptr;
    res.channelDataListAddr = reinterpret_cast<void*>(0x1U);
    res.channelDataSizeListAddr = reinterpret_cast<void*>(0x1U);
    res.channelTypeListAddr = reinterpret_cast<void*>(0x1U);
    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_PTR);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenUnsupportedKind_Returns_NOT_SUPPORT)
{
    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.QpInfo[0].qpPtr = 0x7000ULL;

    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = 999U;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;

    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    EXPECT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_E_NOT_SUPPORT);
}

TEST_F(AicpuChannelProcessTest, Ut_InitHcommChannelRes_WhenValidTsRoceBlob_Returns_SUCCESS)
{
    MOCKER_CPP(&DlHnsFunction::DlHnsFunctionInit).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetPlatformInfo).stubs().will(invoke(StubHrtDrvGetPlatformInfo));
    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtDrvGetLocalDevIDByHostDevID).stubs().will(invoke(StubHrtDrvGetLocalDevIDByHostDevID));
    MOCKER(hrtSetWorkModeAicpu).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtSetlocalDeviceType).stubs().will(returnValue(HCCL_SUCCESS));

    HcommRoceChannelRes blob{};
    blob.qpsPerConnection = 1U;
    blob.localMemCount = 0U;
    blob.remoteMemCount = 0U;
    blob.localMem = nullptr;
    blob.remoteMem = nullptr;
    blob.QpInfo[0].qpPtr = 0x7000ULL;
    blob.QpInfo[0].sqIndex = 1U;
    blob.QpInfo[0].dbIndex = 2U;
    blob.localDataNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x8000ULL));
    blob.remoteNotifyAddr = reinterpret_cast<void*>(static_cast<uintptr_t>(0x9000ULL));
    void* dataPtr = static_cast<void*>(&blob);
    u64 sizeVal = sizeof(blob);
    u32 typeVal = kKindTsRoce;
    ChannelHandle outHandle = 0ULL;

    HcommChannelRes res{};
    res.listNum = 1U;
    res.channelList = static_cast<void*>(&outHandle);
    res.channelDataListAddr = static_cast<void*>(&dataPtr);
    res.channelDataSizeListAddr = static_cast<void*>(&sizeVal);
    res.channelTypeListAddr = static_cast<void*>(&typeVal);
    res.deviceInfo.deviceLogicId = 0;
    res.deviceInfo.devicePhyId = 0U;
    res.deviceInfo.deviceType = static_cast<u32>(DevType::DEV_TYPE_910_93);

    ASSERT_EQ(AicpuChannelProcess::InitHcommChannelRes(&res), HCCL_SUCCESS);
    ASSERT_NE(outHandle, 0ULL);
    EXPECT_TRUE(DevAicpuTsChannelMgr::Instance().DestroyChannel(outHandle));
}

// ROCE_V2 建链收编走 V2 后，device 侧构造 RoceTransportLiteImpl 的路径由 ParsePackData ROCE 分支承担。
// 等价于被删 ut_dev_aicpu_ts_roce_channel_v2.cc 的 Ut_When_Construct_Expect_Success。
// 此处只测「分发」：最小 uniqueId（type=ROCE + 各段计数为 0 + notifyValue 段），完整字段解析由
// legacy ut_roce_transport_lite_impl.cc 覆盖。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenRoceBlob_Returns_SUCCESS_AndHandle)
{
    std::vector<char> data = BuildRocePackData();
    ParsePackDataAndDestroy(data);
}

TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenInvalidTransportType_Returns_PARA)
{
    Hccl::BinaryStream inner;
    u32 invalidType = 0xFFFFFFFFU;
    inner << invalidType;
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);

    std::vector<char> data = BuildOuterPackData(transpUniqueId);

    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_E_PARA);
}

// P2P 收编走 V2 后，device 侧构造 P2PTransportLiteImpl 的路径由 ParsePackData P2P 分支承担。
// P2P 的 Init 无 connNum/notifyValue 段，最小分发 blob 仅 type=P2P + notifyNum=0 + bufferNum=0 + 4 个空 vector。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenP2pBlob_Returns_SUCCESS_AndHandle)
{
    std::vector<char> data = BuildP2pPackData();
    ParsePackDataAndDestroy(data);
}

// task cache 收编：device ParsePackData UB 分支直接注册 needCacheTask / addWqeArray 两个回调，
// 复刻 comm 侧 ChannelAicpuMgr::RegisterChannelCacheCallback 的 dynamic_cast→两个 setter。
// 两个 setter 是 inline 且无 getter，利用 -fno-access-control 直接断言私有成员由空置为非空。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenUbBlob_RegistersCacheCallbacks)
{
    Hccl::BinaryStream inner;
    inner << static_cast<u32>(Hccl::TransportType::UB);
    inner << static_cast<u32>(0); // notifyNum
    inner << static_cast<u32>(0); // bufferNum
    inner << static_cast<u32>(0); // rmtbufferNum
    inner << static_cast<u32>(0); // connNum
    inner << std::vector<char>(); // locNotify
    inner << std::vector<char>(); // rmtNotify
    inner << std::vector<char>(); // locBuffer
    inner << std::vector<char>(); // rmtBuffer
    inner << std::vector<char>(); // drain
    inner << std::vector<char>(); // conn
    std::vector<char> transpUniqueId;
    inner.Dump(transpUniqueId);

    std::vector<char> data = BuildOuterPackData(transpUniqueId);

    ChannelHandle handle = 0ULL;
    EXPECT_EQ(AicpuChannelProcess::ParsePackData(data, handle), HCCL_SUCCESS);
    ASSERT_NE(handle, 0ULL);

    auto* ub = reinterpret_cast<Hccl::UbTransportLiteImpl*>(handle);
    EXPECT_NE(ub->needCacheTaskCallback_, nullptr);
    EXPECT_NE(ub->addWqeArrayCallback_, nullptr);

    HcclChannelUrmaRes destroyRes{};
    destroyRes.channelList = static_cast<void*>(&handle);
    destroyRes.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes), HCCL_SUCCESS);
}

// 行 26 等价迁移：V3 单 unique_ptr transport_ 覆写 bug 已随 DevAicpuTsRoceChannelV2 删除而消除，
// V2 用 type-erased transportMap_（handle→unique_ptr）存储，多 channel 各占一条、handle 互异、独立销毁互不影响。
TEST_F(AicpuChannelProcessTest, Ut_ParsePackData_WhenTwoRoceBlobs_IndependentHandles_NoDangling)
{
    std::vector<char> data = BuildRocePackData();

    ChannelHandle handle1 = 0ULL;
    ChannelHandle handle2 = 0ULL;
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle1), HCCL_SUCCESS);
    ASSERT_EQ(AicpuChannelProcess::ParsePackData(data, handle2), HCCL_SUCCESS);
    EXPECT_NE(handle1, 0ULL);
    EXPECT_NE(handle2, 0ULL);
    EXPECT_NE(handle1, handle2);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(2));

    // 独立销毁：删 handle1 后剩 1 条，handle2 仍存活
    HcclChannelUrmaRes destroyRes1{};
    destroyRes1.channelList = static_cast<void*>(&handle1);
    destroyRes1.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes1), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(1));

    HcclChannelUrmaRes destroyRes2{};
    destroyRes2.channelList = static_cast<void*>(&handle2);
    destroyRes2.listNum = 1U;
    EXPECT_EQ(AicpuChannelProcess::AicpuChannelDestroy(&destroyRes2), HCCL_SUCCESS);
    EXPECT_EQ(AicpuChannelProcess::transportMap_.size(), static_cast<size_t>(0));
}
