/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstring>
#include <vector>

#include "gtest/gtest.h"

#ifndef private
#define private public
#define protected public
#endif

#include "aicpu_task_cache_entry.h"
#include "ub_conn_lite.h"
#include "ub_transport_lite_impl.h"
#include "rtsq_a5.h"
#include "aicpu_ts_thread.h"
#include "sqe_v82.h"
#include "udma_data_struct.h"
#include "ub_jetty_lite.h"
#include "ip_address.h"
#include "config_plf_log_v2.h"
#include "stream_lite.h"

#undef private
#undef protected

using namespace hcomm;
using namespace Hccl;

TEST(AicpuTaskCacheEntryDebugConfigTest, TaskDebugFlagFollowsPlfTaskConfig)
{
    Hccl::SetPlfDebugConfigValue(0ULL);
    AicpuTaskCacheEntry disabledEntry;
    EXPECT_EQ(disabledEntry.isTaskConfigDebug_, HcclCheckLogLevel(HCCL_LOG_DEBUG));

    Hccl::SetPlfDebugConfigValue(Hccl::PLF_TASK);
    AicpuTaskCacheEntry enabledEntry;
    EXPECT_TRUE(enabledEntry.isTaskConfigDebug_);
    Hccl::SetPlfDebugConfigValue(0ULL);
}

// Provide strong definition for weak symbol aicpu::GetSqeId (called by RtsqBase constructor)
namespace aicpu {
void GetSqeId(const uint32_t num, uint32_t& start, uint32_t& end)
{
    start = 1;
    end = start + num;
}
} // namespace aicpu

namespace {
constexpr uint64_t TEST_BASE_ADDR_0 = 0x10000ULL;
constexpr uint64_t TEST_MEM_SIZE_0 = 0x1000ULL;
constexpr uint64_t TEST_BASE_ADDR_1 = 0x20000ULL;
constexpr uint64_t TEST_MEM_SIZE_1 = 0x2000ULL;
constexpr uint32_t TEST_SQ_DEPTH = 128;

std::vector<uint8_t> MakeSqeArray(Rt91095StarsSqeType sqeType, uint64_t srcAddr = 0, uint64_t dstAddr = 0)
{
    std::vector<uint8_t> sqeArray(Hccl::AC_SQE_SIZE, 0);
    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(sqeArray.data());
    header->type = static_cast<uint8_t>(sqeType);

    if (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA) {
        Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(sqeArray.data());
        sdmaSqe->u.strideMode0.srcAddrLow = static_cast<uint32_t>(srcAddr);
        sdmaSqe->u.strideMode0.srcAddrHigh = static_cast<uint32_t>(srcAddr >> 32);
        sdmaSqe->u.strideMode0.dstAddrLow = static_cast<uint32_t>(dstAddr);
        sdmaSqe->u.strideMode0.dstAddrHigh = static_cast<uint32_t>(dstAddr >> 32);
    } else if (sqeType == Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE) {
        Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(sqeArray.data());
        wvSqe->writeAddrLow = static_cast<uint32_t>(dstAddr);
        wvSqe->writeAddrHigh = static_cast<uint32_t>(dstAddr >> 32);
    }
    return sqeArray;
}

Hccl::WqeTask MakeWqeTaskRead(uint64_t locAddr, uint64_t rmtAddr)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(Hccl::UdmaSqOpcode::UDMA_OPC_READ);
    wqe.wqeWrite.u.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWrite.u.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWrite.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWrite.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskWrite(uint64_t locAddr, uint64_t rmtAddr, bool inlineEn)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(Hccl::UdmaSqOpcode::UDMA_OPC_WRITE);
    common->inlineEn = inlineEn ? 1 : 0;
    wqe.wqeWrite.u.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWrite.u.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWrite.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWrite.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskWriteWithNotify(uint64_t locAddr, uint64_t rmtAddr)
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = static_cast<uint8_t>(WRITE_WITH_NOTIFY_OPCODE);
    wqe.wqeWriteWithNotify.localU.sge.dataAddrLow = static_cast<uint32_t>(locAddr);
    wqe.wqeWriteWithNotify.localU.sge.dataAddrHigh = static_cast<uint32_t>(locAddr >> 32);
    wqe.wqeWriteWithNotify.comm.rmtAddrLow = static_cast<uint32_t>(rmtAddr);
    wqe.wqeWriteWithNotify.comm.rmtAddrHigh = static_cast<uint32_t>(rmtAddr >> 32);
    return wqe;
}

Hccl::WqeTask MakeWqeTaskInvalid()
{
    Hccl::WqeTask wqe;
    memset(&wqe, 0, sizeof(wqe));
    UdmaSqeCommon* common = reinterpret_cast<UdmaSqeCommon*>(&wqe);
    common->opcode = 0xFF;
    return wqe;
}

Hccl::DbSqeProfInfo MakeDbSqeProfInfo(bool isValid)
{
    Hccl::DbSqeProfInfo info{};
    memset(&info, 0, sizeof(info));
    info.isValid = isValid;
    return info;
}
} // namespace

// ===================== AddrRefreshInfo Tests =====================

TEST(AddrRefreshInfoTest, DefaultConstructor)
{
    AddrRefreshInfo info;
    EXPECT_FALSE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 0U);
}

TEST(AddrRefreshInfoTest, ParamConstructor)
{
    AddrRefreshInfo info(42);
    EXPECT_TRUE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 42U);
}

TEST(AddrRefreshInfoTest, CopyConstructor)
{
    AddrRefreshInfo orig(7);
    AddrRefreshInfo copy(orig);
    EXPECT_TRUE(copy.needRefresh);
    EXPECT_EQ(copy.memIdx, 7U);
}

TEST(AddrRefreshInfoTest, AssignmentOperator)
{
    AddrRefreshInfo orig(5);
    AddrRefreshInfo assigned;
    assigned = orig;
    EXPECT_TRUE(assigned.needRefresh);
    EXPECT_EQ(assigned.memIdx, 5U);
}

TEST(AddrRefreshInfoTest, SelfAssignment)
{
    AddrRefreshInfo info(3);
    info = info;
    EXPECT_TRUE(info.needRefresh);
    EXPECT_EQ(info.memIdx, 3U);
}

// ===================== AicpuTaskCacheEntry Fixture =====================

class AicpuTaskCacheEntryTest : public testing::Test {
protected:
    std::unique_ptr<Hccl::RtsqA5> rtsq_;
    Hccl::RtsqA5* rtsqPtr_ = nullptr;
    std::unique_ptr<hccl::AicpuTsThread> aicpuTsThread_;
    std::unique_ptr<Hccl::UbConnLite> ubConnLite_;
    std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransport_;
    std::vector<char> emptyUniqueId_;

    void SetUp() override
    {
        rtsq_ = std::make_unique<Hccl::RtsqA5>(0, 0, 0);
        rtsqPtr_ = rtsq_.get();
        aicpuTsThread_ = std::make_unique<hccl::AicpuTsThread>(std::string("ut_thread"));
        ubConnLite_ = std::make_unique<Hccl::UbConnLite>(
            Hccl::UbJettyLiteId(0, 0, 0), Hccl::UbJettyLiteAttr(0, 0, TEST_SQ_DEPTH, 0, true), Hccl::Eid{});
        ubTransport_ = std::make_unique<Hccl::UbTransportLiteImpl>(emptyUniqueId_);
    }

    void TearDown() override {}

    HcclResult InitEntryWithTwoAddrs(hcomm::AicpuTaskCacheEntry& entry)
    {
        uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
        uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
        return entry.InitCacheEntry(baseAddrs, memSizes, 2);
    }

    // AddWqeArray always creates a DbSqeTmpInfo entry that must be cleared by AddSqeArray
    // with the same streamId before SubmitCacheEntry can succeed.
    void AddUbdmaSqeToClearTmpMap(hcomm::AicpuTaskCacheEntry& entry, uint32_t streamId = 0)
    {
        auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
        ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), streamId), HCCL_SUCCESS);
    }
};

// ===================== GetUbLinkTypeVal Tests =====================

TEST_F(AicpuTaskCacheEntryTest, GetUbLinkTypeVal_Returns_LINK_UB_When_linkType_Is_UB)
{
    hcomm::AicpuTaskCacheEntry entry;
    ubTransport_->linkType_ = Hccl::DfxLinkType::UB;
    EXPECT_EQ(entry.GetUbLinkTypeVal_(ubTransport_.get()), static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_UB));
}

TEST_F(AicpuTaskCacheEntryTest, GetUbLinkTypeVal_Returns_LINK_UBoE_When_linkType_Is_UBoE)
{
    hcomm::AicpuTaskCacheEntry entry;
    ubTransport_->linkType_ = Hccl::DfxLinkType::UBoE;
    EXPECT_EQ(entry.GetUbLinkTypeVal_(ubTransport_.get()), static_cast<u8>(Hccl::DfxLinkTypeVal::LINK_UBoE));
}

// ===================== Constructor / Destructor Tests =====================

TEST_F(AicpuTaskCacheEntryTest, Constructor_Default)
{
    hcomm::AicpuTaskCacheEntry entry;
    EXPECT_EQ(entry.entryBytes_, 0U);
    EXPECT_EQ(entry.sqeArrayInfos_.size(), 0U);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 0U);
}

TEST_F(AicpuTaskCacheEntryTest, Destructor_EmptyEntry_NoCrash)
{
    {
        hcomm::AicpuTaskCacheEntry entry;
    }
    SUCCEED();
}

TEST_F(AicpuTaskCacheEntryTest, Destructor_WithSqeArray_FreesMemory)
{
    {
        hcomm::AicpuTaskCacheEntry entry;
        ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
        auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
        ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    }
    SUCCEED();
}

// ===================== InitCacheEntry Tests =====================

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};

    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_SUCCESS);
    EXPECT_EQ(entry.cachedBaseAddrs_.size(), 2U);
    EXPECT_EQ(entry.cachedMemSizes_.size(), 2U);
    EXPECT_EQ(entry.cachedBaseAddrs_[0], TEST_BASE_ADDR_0);
    EXPECT_EQ(entry.cachedBaseAddrs_[1], TEST_BASE_ADDR_1);
    EXPECT_EQ(entry.cachedMemSizes_[0], TEST_MEM_SIZE_0);
    EXPECT_EQ(entry.cachedMemSizes_[1], TEST_MEM_SIZE_1);
    EXPECT_GT(entry.entryBytes_, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_InvalidCount)
{
    hcomm::AicpuTaskCacheEntry entry;
    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0};

    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 1), HCCL_E_PARA);
    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 3), HCCL_E_PARA);
}

TEST_F(AicpuTaskCacheEntryTest, InitCacheEntry_DoubleInit)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0, TEST_BASE_ADDR_1};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.InitCacheEntry(baseAddrs, memSizes, 2), HCCL_E_INTERNAL);
}

// ===================== AddSqeArray Tests =====================

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullRtsqPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(entry.AddSqeArray(nullptr, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullAicpuTsThreadPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(entry.AddSqeArray(rtsqPtr_, nullptr, 1, sqeArray.data(), 0), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_NullSqeArray)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    EXPECT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, nullptr, 0), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_ZeroSqeCount)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 0, sqeArray.data(), 0), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddSqeArray_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    EXPECT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    EXPECT_EQ(entry.sqeArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.sqeArrayInfos_[0].sqeCount, 1U);
    EXPECT_NE(entry.sqeArrayInfos_[0].sqeArray, nullptr);
    EXPECT_EQ(entry.launchOrder_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeSqe);
    EXPECT_GT(entry.entryBytes_, 0U);
}

// ===================== AddWqeArray Tests =====================

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_ReportTaskMismatch)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(true)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_EmptyWqeTasks)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> emptyTasks;

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), emptyTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_NullUbConnLitePtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(nullptr, ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_NullUbTransportLiteImplPtr)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), nullptr, wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)), HCCL_E_PTR);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_.size(), 1U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeWqe);
    EXPECT_EQ(entry.tokenInfosMap_.size(), 1U);
}

TEST_F(AicpuTaskCacheEntryTest, AddWqeArray_Success_WithReportTask)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};

    EXPECT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(true)),
        HCCL_SUCCESS);
    EXPECT_EQ(entry.wqeTaskArrayInfos_.size(), 1U);
    EXPECT_EQ(entry.streamIdToDbSqeTmpInfoMap_.size(), 1U);
}

// ===================== SubmitCacheEntry Tests =====================

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_EmptyCache)
{
    hcomm::AicpuTaskCacheEntry entry;
    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_TmpMapNotEmpty)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(0, 0)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, true, MakeDbSqeProfInfo(true)),
        HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_NotifyWait)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray.size(), 1U);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_NotifyRecord)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Ubdma)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Sdma_AddrInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].memIdx, 0U);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].memIdx, 1U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_Sdma_AddrNotInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t outOfRangeAddr = 0x999000;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, outOfRangeAddr, outOfRangeAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlySqe_WriteValue)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t dstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, dstAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].memIdx, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_InvalidSqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_AIC);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_Read)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].memIdx, 0U);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false, memIdx预期为默认值0U
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].memIdx, 0U);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteNonInline)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWrite(locAddr, rmtAddr, false)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteInline)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWrite(0, rmtAddr, true)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_OnlyWqe_WriteWithNotify)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskWriteWithNotify(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_InvalidWqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskInvalid()};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_BothSqeWqe)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(TEST_BASE_ADDR_0 + 0x10, TEST_BASE_ADDR_1 + 0x20)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    EXPECT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);
    EXPECT_EQ(entry.launchOrder_.size(), 2U);
    EXPECT_EQ(entry.launchOrder_[0], TaskArrayType::kTaskArrayTypeWqe);
    EXPECT_EQ(entry.launchOrder_[1], TaskArrayType::kTaskArrayTypeSqe);
}

TEST_F(AicpuTaskCacheEntryTest, SubmitCacheEntry_Success_TokenInfoFlagSet)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);
    AddUbdmaSqeToClearTmpMap(entry);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    auto& tokenInfos = entry.tokenInfosMap_.begin()->second;
    EXPECT_TRUE(tokenInfos[0].needLocTokenIdFlag);

    // 注意: WQE remote addr不再检查本地地址, 因此needRmtTokenIdAndValueFlag预期为false
    EXPECT_FALSE(tokenInfos[1].needRmtTokenIdAndValueFlag);
}

// ===================== RefreshAndLaunch Tests =====================

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_CountMismatch)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t baseAddrs[] = {TEST_BASE_ADDR_0};
    uint64_t memSizes[] = {TEST_MEM_SIZE_0};
    EXPECT_EQ(entry.RefreshAndLaunch(baseAddrs, memSizes, 1), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshAndLaunch_Success_SqeAndWqe)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    // 添加WQE数组 (Read类型, loc/rmt地址均在cached range内)
    uint64_t locAddr = TEST_BASE_ADDR_0 + 0x10;
    uint64_t rmtAddr = TEST_BASE_ADDR_1 + 0x20;
    std::vector<Hccl::WqeTask> wqeTasks = {MakeWqeTaskRead(locAddr, rmtAddr)};
    ASSERT_EQ(
        entry.AddWqeArray(ubConnLite_.get(), ubTransport_.get(), wqeTasks, 0, 0, false, MakeDbSqeProfInfo(false)),
        HCCL_SUCCESS);

    // 添加SQE数组 (SDMA类型, src/dst地址均在cached range内)
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    // 验证SubmitCacheEntry后AddrRefreshInfo已正确设置
    EXPECT_TRUE(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].locAddrRefreshInfoArray[0].memIdx, 0U);

    // 注意: WQE remote addr不再检查本地地址, 因此needRefresh预期为false, memIdx为默认值0U
    EXPECT_FALSE(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].needRefresh);
    EXPECT_EQ(entry.wqeTaskArrayInfos_[0].rmtAddrRefreshInfoArray[0].memIdx, 0U);

    EXPECT_TRUE(entry.sqeArrayInfos_[0].srcAddrRefreshInfoArray[0].needRefresh);
    EXPECT_TRUE(entry.sqeArrayInfos_[0].dstAddrRefreshInfoArray[0].needRefresh);

    // 手动刷新SQE, 验证地址刷新结果
    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    ASSERT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);

    // 手动刷新WQE, 验证地址刷新结果
    ASSERT_EQ(entry.RefreshWqeTasks_(entry.wqeTaskArrayInfos_[0], newBaseAddrs, newMemSizes, 2), HCCL_SUCCESS);
    uint64_t newLocAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newLocAddr, entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.u.sge.dataAddrHigh,
        entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.u.sge.dataAddrLow);
    EXPECT_EQ(newLocAddr, 0x50000ULL + 0x10);
    // 注意: WQE remote addr不再检查本地地址, 因此地址不需要刷新, 仍然为初始值
    uint64_t newRmtAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newRmtAddr, entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.comm.rmtAddrHigh,
        entry.wqeTaskArrayInfos_[0].wqeTaskArray[0].wqeWrite.comm.rmtAddrLow);
    EXPECT_EQ(newRmtAddr, TEST_BASE_ADDR_1 + 0x20);

    // 清空launchOrder_避免调用硬件接口, 验证RefreshAndLaunch的校验和RefreshTokenInfos_逻辑
    entry.launchOrder_.clear();
    EXPECT_EQ(entry.RefreshAndLaunch(newBaseAddrs, newMemSizes, 2), HCCL_SUCCESS);

    // 验证tokenInfos已刷新 (needLocTokenIdFlag/needRmtTokenIdAndValueFlag在SubmitCacheEntry时已设置)
    auto& tokenInfos = entry.tokenInfosMap_.begin()->second;
    EXPECT_TRUE(tokenInfos[0].needLocTokenIdFlag);
    // 注意: WQE remote addr不再检查本地地址, 因此needRmtTokenIdAndValueFlag预期为false
    EXPECT_FALSE(tokenInfos[1].needRmtTokenIdAndValueFlag);
}

// ===================== RefreshSqeTasks_ Tests =====================

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_NotifyWait_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsSqeHeader* header = reinterpret_cast<Rt91095StarsSqeHeader*>(entry.sqeArrayInfos_[0].sqeArray);
    EXPECT_EQ(static_cast<Rt91095StarsSqeType>(header->type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_NotifyRecord_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_RECORD);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Ubdma_Success)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_UBDMA);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Sdma_BothAddrsRefreshed)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);

    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_Sdma_AddrNotInRange)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t outOfRangeAddr = 0x999000;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, outOfRangeAddr, outOfRangeAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, outOfRangeAddr);

    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, outOfRangeAddr);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_WriteValue_DstAddrRefreshed)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t dstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, dstAddr);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(entry.sqeArrayInfos_[0].sqeArray);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(newDstAddr, wvSqe->writeAddrHigh, wvSqe->writeAddrLow);
    EXPECT_EQ(newDstAddr, 0x50000ULL + 0x50);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_MultipleSqes)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);

    std::vector<uint8_t> sqeArray(3 * Hccl::AC_SQE_SIZE, 0);
    auto sqe0 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    memcpy(sqeArray.data(), sqe0.data(), Hccl::AC_SQE_SIZE);
    uint64_t srcAddr = TEST_BASE_ADDR_0 + 0x100;
    uint64_t dstAddr = TEST_BASE_ADDR_1 + 0x200;
    auto sqe1 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_SDMA, srcAddr, dstAddr);
    memcpy(sqeArray.data() + Hccl::AC_SQE_SIZE, sqe1.data(), Hccl::AC_SQE_SIZE);
    uint64_t wvDstAddr = TEST_BASE_ADDR_0 + 0x50;
    auto sqe2 = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_WRITE_VALUE, 0, wvDstAddr);
    memcpy(sqeArray.data() + 2 * Hccl::AC_SQE_SIZE, sqe2.data(), Hccl::AC_SQE_SIZE);

    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 3, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);

    uint8_t* base = entry.sqeArrayInfos_[0].sqeArray;

    Rt91095StarsSqeHeader* header0 = reinterpret_cast<Rt91095StarsSqeHeader*>(base);
    EXPECT_EQ(static_cast<Rt91095StarsSqeType>(header0->type), Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);

    Rt91095StarsMemcpySqe* sdmaSqe = reinterpret_cast<Rt91095StarsMemcpySqe*>(base + Hccl::AC_SQE_SIZE);
    uint64_t newSrcAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newSrcAddr, sdmaSqe->u.strideMode0.srcAddrHigh, sdmaSqe->u.strideMode0.srcAddrLow);
    EXPECT_EQ(newSrcAddr, 0x50000ULL + 0x100);
    uint64_t newDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(
        newDstAddr, sdmaSqe->u.strideMode0.dstAddrHigh, sdmaSqe->u.strideMode0.dstAddrLow);
    EXPECT_EQ(newDstAddr, 0x60000ULL + 0x200);

    Rt91095StarsWriteValueSqe* wvSqe = reinterpret_cast<Rt91095StarsWriteValueSqe*>(base + 2 * Hccl::AC_SQE_SIZE);
    uint64_t newWvDstAddr = 0;
    hcomm::AicpuTaskCacheEntry::CombineUint32ToUint64(newWvDstAddr, wvSqe->writeAddrHigh, wvSqe->writeAddrLow);
    EXPECT_EQ(newWvDstAddr, 0x50000ULL + 0x50);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_InvalidSqeType)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_AIC);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);

    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_E_INTERNAL);
}

TEST_F(AicpuTaskCacheEntryTest, RefreshSqeTasks_TaskConfigDebug)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    ASSERT_EQ(entry.SubmitCacheEntry(), HCCL_SUCCESS);

    entry.isTaskConfigDebug_ = true;
    uint64_t newBaseAddrs[] = {0x50000, 0x60000};
    uint64_t newMemSizes[] = {TEST_MEM_SIZE_0, TEST_MEM_SIZE_1};
    EXPECT_EQ(entry.RefreshSqeTasks_(entry.sqeArrayInfos_[0], newBaseAddrs), HCCL_SUCCESS);
    entry.isTaskConfigDebug_ = false;
}

// ===================== ReportSqeArrayProfiling_ (profiling disabled path) =====================

TEST_F(AicpuTaskCacheEntryTest, GetEntryBytes_AfterInit)
{
    hcomm::AicpuTaskCacheEntry entry;
    ASSERT_EQ(InitEntryWithTwoAddrs(entry), HCCL_SUCCESS);
    uint64_t bytesAfterInit = entry.GetEntryBytes();
    EXPECT_GT(bytesAfterInit, 0U);

    auto sqeArray = MakeSqeArray(Rt91095StarsSqeType::RT_91095_SQE_TYPE_NOTIFY_WAIT);
    ASSERT_EQ(entry.AddSqeArray(rtsqPtr_, aicpuTsThread_.get(), 1, sqeArray.data(), 0), HCCL_SUCCESS);
    EXPECT_GT(entry.GetEntryBytes(), bytesAfterInit);
}

TEST_F(AicpuTaskCacheEntryTest, FillSlot_UbDma_JettyPassthrough)
{
    hcomm::AicpuTaskCacheEntry entry;
    const u64 jettyHandle = 0xABCDEF1234ULL;
    const u32 jettyId = 42;

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    DbSqeProfAndRefreshInfo profAndRefreshInfo;
    profAndRefreshInfo.dbSqeProfInfo.jettyHandle = jettyHandle;
    profAndRefreshInfo.dbSqeProfInfo.jettyId = jettyId;

    HcclResult ret = entry.FillSlotUbDma_(&slot, nullptr, profAndRefreshInfo, ubTransport_.get(), &streamLite, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.taskPara.ubDma.jettyHandle, jettyHandle);
    EXPECT_EQ(slot.taskPara.ubDma.jettyId, jettyId);
}

TEST_F(AicpuTaskCacheEntryTest, FillSlot_Reduce_JettyPassthrough)
{
    hcomm::AicpuTaskCacheEntry entry;
    const u64 jettyHandle = 0x1122334455ULL;
    const u32 jettyId = 99;

    Hccl::StreamLite streamLite(0, 0, 0, 0);
    Hccl::DfxTaskInfo slot{};
    DbSqeProfAndRefreshInfo profAndRefreshInfo;
    profAndRefreshInfo.dbSqeProfInfo.jettyHandle = jettyHandle;
    profAndRefreshInfo.dbSqeProfInfo.jettyId = jettyId;

    HcclResult ret = entry.FillSlotReduce_(&slot, nullptr, profAndRefreshInfo, ubTransport_.get(), &streamLite, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(slot.taskPara.Reduce.jettyHandle, jettyHandle);
    EXPECT_EQ(slot.taskPara.Reduce.jettyId, jettyId);
}
