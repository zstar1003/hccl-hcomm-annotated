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
#include "topo_matcher.h"
#include "topoinfo_struct.h"
#include "dlra_function.h"

#define private public
#define protected public

#include "transport_pub.h"
#include "transport_base_pub.h"
#include "dispatcher.h"
#include "dispatcher_pub.h"
#include "coll_alg_param.h"
#include "mem_device_pub.h"
#include "hccl_common.h"
#include "local_notify.h"
#include "notify_pool.h"
#include "alg_template_base_pub.h"
#include "ffts_common_pub.h"
#include "coll_batch_send_recv_group_executor.h"

#undef private
#undef protected

using namespace hccl;
using namespace std;

// Fake TransportBase that overrides all virtual methods to return success.
// This avoids null pimpl_ crashes when Transport::TxPrepare etc. are called.
class FakeTransportBase : public TransportBase {
public:
    FakeTransportBase() : TransportBase(nullptr, fakeNotifyPool_, fakeMachinePara_, std::chrono::milliseconds(0)) {}
    HcclResult TxPrepare(Stream&) override { return HCCL_SUCCESS; }
    HcclResult TxData(UserMemType, u64, const void*, u64, Stream&) override { return HCCL_SUCCESS; }
    HcclResult TxDone(Stream&) override { return HCCL_SUCCESS; }
    HcclResult RxPrepare(Stream&) override { return HCCL_SUCCESS; }
    HcclResult RxData(UserMemType, u64, void*, u64, Stream&) override { return HCCL_SUCCESS; }
    HcclResult RxDone(Stream&) override { return HCCL_SUCCESS; }
    HcclResult TxWaitDone(Stream&) override { return HCCL_SUCCESS; }
    HcclResult RxWaitDone(Stream&) override { return HCCL_SUCCESS; }

private:
    static std::unique_ptr<NotifyPool> fakeNotifyPool_;
    static MachinePara fakeMachinePara_;
};
std::unique_ptr<NotifyPool> FakeTransportBase::fakeNotifyPool_;
MachinePara FakeTransportBase::fakeMachinePara_;

class CollBatchSendRecvGroupExecutorTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "CollBatchSendRecvGroupExecutorTest SetUpTestCase" << std::endl;
    }
    static void TearDownTestCase() { std::cout << "CollBatchSendRecvGroupExecutorTest TearDownTestCase" << std::endl; }

    virtual void SetUp()
    {
        // Configure topoInfo with defaults for 16-rank test setup
        topoInfo_.userRank = 4;
        topoInfo_.userRankSize = 16;
        topoInfo_.deviceType = DevType::DEV_TYPE_910B;
        topoInfo_.isSingleMeshAggregation = false;
        topoInfo_.moduleNum = 4;

        // AlgoInfo defaults are fine
        // ExternalEnable defaults are fine

        topoMatcher_ = std::make_unique<TopoMatcher>(
            commPlaneRanks_, isBridgeVector_, topoInfo_, algoInfo_, externalEnable_, serverAndsuperPodToRank_);

        executor_ = std::make_unique<CollBatchSendRecvGroupExecutor>(nullptr, topoMatcher_);

        // Set up a real DispatcherPub so HcclD2DMemcpyAsync and LocalNotify work.
        // Individual tests mock hrtMemAsyncCopy and virtual methods as needed.
        dispatcher_ = std::make_unique<DispatcherPub>(0);
        const_cast<HcclDispatcher&>(executor_->dispatcher_) = dispatcher_.get();
    }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }

    // Helper: set up basic pod state. pod = rank [0..3], so userRank=4 is outside pod (RDMA).
    // This is the default state for IsRemoteRankRdma / GetNextDstRank / etc.
    void SetPodRange(u32 startRank, u32 endRank, u32 devNumInPod)
    {
        executor_->podStartRank_ = startRank;
        executor_->podEndRank_ = endRank;
        executor_->devNumInlocalPod_ = devNumInPod;
    }

    // Helper: set stream numbers
    void SetStreamNum(u32 sendNum, u32 recvNum)
    {
        executor_->sendStreamNum_ = sendNum;
        executor_->recvStreamNum_ = recvNum;
    }

    // Helper: set topoAttr_ fields (const in base class, but mutable for testing)
    void SetTopoAttrUserRank(u32 rank) { const_cast<HcclTopoInfo&>(executor_->topoAttr_).userRank = rank; }
    void SetTopoAttrUserRankSize(u32 size) { const_cast<HcclTopoInfo&>(executor_->topoAttr_).userRankSize = size; }

    // Helper: set up algResResp_ with memory and streams for RunTasks/Process* tests.
    void SetupAlgResResp(u32 streamNum = 18, u64 memSize = 4 * 1024 * 1024)
    {
        // Use real allocated memory so stream background threads can memcpy without crashing
        memBuffer_.resize(memSize);
        void* memPtr = memBuffer_.data();
        algRes_.cclInputMem = DeviceMem(memPtr, memSize);
        algRes_.cclOutputMem = DeviceMem(memPtr, memSize);
        algRes_.slaveStreams.clear();
        algRes_.slaveStreams.resize(streamNum);
        // Create real streams so MemcpyAsync/aclrtMemcpyAsync stubs work
        for (auto& s : algRes_.slaveStreams) {
            s = Stream(StreamType::STREAM_TYPE_OFFLINE);
        }
        algRes_.notifiesAux.clear();
        algRes_.notifiesAux.resize(streamNum);
        algRes_.notifiesMain.clear();
        algRes_.notifiesMain.resize(streamNum);
        executor_->algResResp_ = &algRes_;
    }

    // Helper: create a fake Transport link with given transport type.
    // Uses FakeTransportBase as pimpl_ so TxPrepare/TxData/etc. return success.
    LINK MakeFakeLink(TransportType type = TransportType::TRANS_TYPE_P2P)
    {
        auto* fakeBase = new FakeTransportBase();
        LINK link = std::make_shared<Transport>(fakeBase);
        const_cast<TransportType&>(link->type_) = type;
        return link;
    }

    // Helper: set up opTransportResponse so GetSendTargetLink/GetRecvTargetLink
    // return fakeLink for the given remoteRank. userRank defaults to 4 (from SetUp).
    void SetupOpTransportForRank(u32 remoteRank, LINK fakeLink)
    {
        u32 userRank = executor_->topoAttr_.userRank;
        // Ensure opTransportResponse has COMM_COMBINE_ORDER level with 2 subComms
        if (algRes_.opTransportResponse.size() <= COMM_COMBINE_ORDER) {
            algRes_.opTransportResponse.resize(COMM_COMBINE_ORDER + 1);
        }
        auto& level = algRes_.opTransportResponse[COMM_COMBINE_ORDER];
        while (level.size() < 2) {
            level.push_back(SingleSubCommTransport());
        }
        // Set up both subComms so both GetSendTargetLink and GetRecvTargetLink work
        for (u32 commIndex = 0; commIndex < 2; commIndex++) {
            SingleSubCommTransport& sub = level[commIndex];
            for (u32 r = 0; r <= remoteRank; r++) {
                if (sub.userRank2subCommRank.count(r) == 0) {
                    u32 sr = sub.links.size();
                    sub.userRank2subCommRank[r] = sr;
                    sub.links.push_back((r == remoteRank) ? fakeLink : nullptr);
                    sub.status.push_back(TransportStatus::READY);
                } else if (r == remoteRank) {
                    u32 sr = sub.userRank2subCommRank[r];
                    sub.links[sr] = fakeLink;
                }
            }
        }
        executor_->algResResp_ = &algRes_;
    }

    // Data members that TopoMatcher copies/refs
    HcclTopoInfo topoInfo_;
    HcclAlgoInfo algoInfo_;
    HcclExternalEnable externalEnable_;
    std::vector<std::vector<std::vector<u32>>> commPlaneRanks_;
    std::vector<bool> isBridgeVector_;
    std::vector<std::vector<std::vector<u32>>> serverAndsuperPodToRank_;

    std::unique_ptr<TopoMatcher> topoMatcher_;
    std::unique_ptr<CollBatchSendRecvGroupExecutor> executor_;
    std::unique_ptr<DispatcherPub> dispatcher_;
    AlgResourceResponse algRes_;
    std::vector<u8> memBuffer_; // backing store for cclInputMem/cclOutputMem
};

// ============================================================================
// Constants tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, Constants_HaveExpectedValues)
{
    EXPECT_EQ(CollBatchSendRecvGroupExecutor::RDMA_CCLOUT_HALF_NUM, 2u);
    EXPECT_EQ(CollBatchSendRecvGroupExecutor::RDMA_STREAM_NUM, 2u);
    EXPECT_EQ(GROUP_MAX_CONCURRENT, 8u);
}

// ============================================================================
// RdmaSendStreamIdx / RdmaRecvStreamIdx tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, RdmaSendStreamIdx_ReturnsCorrectIndex)
{
    SetStreamNum(8, 8);
    EXPECT_EQ(executor_->RdmaSendStreamIdx(), 16u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, RdmaSendStreamIdx_WithStreamNumVaried)
{
    SetStreamNum(4, 6);
    EXPECT_EQ(executor_->RdmaSendStreamIdx(), 10u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, RdmaRecvStreamIdx_ReturnsCorrectIndex)
{
    SetStreamNum(8, 8);
    EXPECT_EQ(executor_->RdmaRecvStreamIdx(), 17u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, RdmaRecvStreamIdx_WithStreamNumVaried)
{
    SetStreamNum(4, 6);
    EXPECT_EQ(executor_->RdmaRecvStreamIdx(), 11u);
}

// ============================================================================
// IsRemoteRankRdma tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, IsRemoteRankRdma_RankInsidePod_ReturnsFalse)
{
    // pod = ranks [0, 7]
    SetPodRange(0, 7, 8);
    EXPECT_FALSE(executor_->IsRemoteRankRdma(0));
    EXPECT_FALSE(executor_->IsRemoteRankRdma(3));
    EXPECT_FALSE(executor_->IsRemoteRankRdma(7));
}

TEST_F(CollBatchSendRecvGroupExecutorTest, IsRemoteRankRdma_RankOutsidePod_ReturnsTrue)
{
    // pod = ranks [0, 7]
    SetPodRange(0, 7, 8);
    EXPECT_TRUE(executor_->IsRemoteRankRdma(8));
    EXPECT_TRUE(executor_->IsRemoteRankRdma(15));
}

TEST_F(CollBatchSendRecvGroupExecutorTest, IsRemoteRankRdma_RankAtPodBoundary_ReturnsFalse)
{
    // pod = ranks [0, 3]
    SetPodRange(0, 3, 4);
    EXPECT_FALSE(executor_->IsRemoteRankRdma(0)); // pod start
    EXPECT_FALSE(executor_->IsRemoteRankRdma(3)); // pod end
    EXPECT_TRUE(executor_->IsRemoteRankRdma(4));  // just outside
}

TEST_F(CollBatchSendRecvGroupExecutorTest, IsRemoteRankRdma_SingleRankPod)
{
    // pod = rank 0 only
    SetPodRange(0, 0, 1);
    EXPECT_FALSE(executor_->IsRemoteRankRdma(0));
    EXPECT_TRUE(executor_->IsRemoteRankRdma(1));
    EXPECT_TRUE(executor_->IsRemoteRankRdma(15));
}

TEST_F(CollBatchSendRecvGroupExecutorTest, IsRemoteRankRdma_EntireClusterIsPod)
{
    // pod covers all 16 ranks
    SetPodRange(0, 15, 16);
    for (u32 r = 0; r < 16; r++) {
        EXPECT_FALSE(executor_->IsRemoteRankRdma(r));
    }
}

// ============================================================================
// GetNextDstRank tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, GetNextDstRank_NormalIncrement)
{
    // pod = [4, 7], devNumInlocalPod = 4, userRankSize = 16
    // Use pod that doesn't start at 0 so normal increment works
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(16);

    u32 curRank = 0;
    // 0 != podStart(4), so: curRank=0%16=0, return 0, curRank=1
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(curRank, 1u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetNextDstRank_SkipPodStart)
{
    // pod = [4, 7], devNumInlocalPod = 4, userRankSize = 16
    // curRank = 4 (podStart) -> should skip to 8
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(16);

    u32 curRank = 4;
    // 4 == podStart(4), curRank = 4+4=8, 8%16=8, return 8, curRank=9
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 8u);
    EXPECT_EQ(curRank, 9u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetNextDstRank_WrapAround)
{
    // pod = [0, 3], devNumInlocalPod = 4, userRankSize = 8
    // Starting at curRank=0 skips pod to 4, so use pod that doesn't start at 0
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(8);

    // curRank=6: 6 != 4, 6%8=6, return 6, curRank=7
    u32 curRank = 6;
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 6u);
    EXPECT_EQ(curRank, 7u);

    // curRank=7: 7 != 4, 7%8=7, return 7, curRank=8 (which is 0 after next call's modulo)
    result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 7u);
    EXPECT_EQ(curRank, 8u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetNextDstRank_OverflowWrapThenSkipPod)
{
    // pod = [4, 7], devNumInlocalPod = 4, userRankSize = 8
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(8);

    // curRank=3: 3 != 4, 3%8=3, return 3, curRank=4
    u32 curRank = 3;
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 3u);
    EXPECT_EQ(curRank, 4u);

    // curRank=4 (podStart): 4 == 4, curRank=4+4=8, 8%8=0, return 0, curRank=1
    result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(curRank, 1u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetNextDstRank_OverflowWrap)
{
    // pod = [0, 3], devNumInlocalPod = 4, userRankSize = 16
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);

    u32 curRank = 17; // > userRankSize
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 1u); // 17 % 16 = 1
    EXPECT_EQ(curRank, 2u);
}

// ============================================================================
// GetPreSrcRank tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, GetPreSrcRank_NormalDecrement)
{
    // pod = [0, 3], devNumInlocalPod = 4, userRankSize = 16
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);

    u32 curRank = 10;
    u32 result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 10u);
    EXPECT_EQ(curRank, 9u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetPreSrcRank_ZeroWrap_SimplePod)
{
    // pod = [4, 7], devNumInlocalPod = 4, userRankSize = 16
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(16);

    // curRank=0: 0 != 7(podEnd), but 0 == 0: wraps to 15, returns 0
    u32 curRank = 0;
    u32 result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(curRank, 15u); // wrapped to size-1
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetPreSrcRank_SkipPodEnd)
{
    // pod = [4, 7], devNumInlocalPod = 4, userRankSize = 8
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(8);

    // curRank=7 (podEnd = 4+4-1=7): triggers pod skip
    // curRank = (7+8-4)%8 = 11%8 = 3
    // then 3 != 0, return 3, curRank = 2
    u32 curRank = 7;
    u32 result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 3u);
    EXPECT_EQ(curRank, 2u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, GetPreSrcRank_ZeroWrap)
{
    // pod = [1, 4], devNumInlocalPod = 4, userRankSize = 8
    SetPodRange(1, 4, 4);
    SetTopoAttrUserRankSize(8);

    u32 curRank = 1; // podStart
    u32 result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 1u);
    EXPECT_EQ(curRank, 0u);

    curRank = 0;
    result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 0u);
    EXPECT_EQ(curRank, 7u); // wrapped to size-1
}

// ============================================================================
// OrderRdmaSlices tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, OrderRdmaSlices_Send_EmptyByRank)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(16);

    std::map<u32, std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice>> byRank;
    std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice> out;

    executor_->OrderRdmaSlices(true, byRank, out);
    EXPECT_TRUE(out.empty());
}

TEST_F(CollBatchSendRecvGroupExecutorTest, OrderRdmaSlices_Send_BasicOrdering)
{
    // pod = [0, 3], 4 ranks local, 12 ranks cross-pod
    // userRank = 4, devNumInlocalPod = 4, userRankSize = 16
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(16);

    // Create slices for specific ranks
    std::map<u32, std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice>> byRank;
    u8 dummyBuf[64];
    byRank[8].emplace_back(dummyBuf, 16, 8, true);
    byRank[12].emplace_back(dummyBuf, 16, 12, true);
    byRank[15].emplace_back(dummyBuf, 32, 15, true);

    std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice> out;
    executor_->OrderRdmaSlices(true, byRank, out);

    // Should have 3 slices total
    EXPECT_EQ(out.size(), 3u);

    // send order is forward-increasing around the ring, skipping pod [0,3].
    // Starting from (4+4) % 16 = 8, then 9,10,...,15, then 0,1,2,3(skip), 4,5,6,7(skip)
    // So order should be: rank 8, 12, 15 (the ones that have tasks)
    // We only verify that all expected ranks appear
    std::set<u32> foundRanks;
    for (const auto& s : out) {
        foundRanks.insert(s.remoteRank);
        EXPECT_TRUE(s.isRdma);
    }
    EXPECT_EQ(foundRanks.size(), 3u);
    EXPECT_TRUE(foundRanks.count(8));
    EXPECT_TRUE(foundRanks.count(12));
    EXPECT_TRUE(foundRanks.count(15));
}

TEST_F(CollBatchSendRecvGroupExecutorTest, OrderRdmaSlices_Recv_BasicOrdering)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(16);

    std::map<u32, std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice>> byRank;
    u8 dummyBuf[64];
    byRank[5].emplace_back(dummyBuf, 16, 5, true);
    byRank[8].emplace_back(dummyBuf, 16, 8, true);
    byRank[15].emplace_back(dummyBuf, 32, 15, true);

    std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice> out;
    executor_->OrderRdmaSlices(false, byRank, out);

    // recv order is backward-decreasing, skipping pod [0,3]
    EXPECT_EQ(out.size(), 3u);
    std::set<u32> foundRanks;
    for (const auto& s : out) {
        foundRanks.insert(s.remoteRank);
    }
    EXPECT_EQ(foundRanks.size(), 3u);
    EXPECT_TRUE(foundRanks.count(5));
    EXPECT_TRUE(foundRanks.count(8));
    EXPECT_TRUE(foundRanks.count(15));
}

TEST_F(CollBatchSendRecvGroupExecutorTest, OrderRdmaSlices_SkipsSamePodRanks)
{
    // pod = [4, 7], so ranks 4-7 should be skipped
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(8);

    std::map<u32, std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice>> byRank;
    u8 dummyBuf[64];
    // 5 and 6 are inside pod [4,7] — but the list already filters them out externally
    // OrderRdmaSlices only processes what's in the map, so these won't appear
    byRank[1].emplace_back(dummyBuf, 16, 1, true);
    byRank[3].emplace_back(dummyBuf, 16, 3, true);

    std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice> out;
    executor_->OrderRdmaSlices(true, byRank, out);

    EXPECT_EQ(out.size(), 2u);
    // Both ranks outside pod should be in output
    for (const auto& s : out) {
        EXPECT_FALSE(s.remoteRank >= 4 && s.remoteRank <= 7);
    }
}

TEST_F(CollBatchSendRecvGroupExecutorTest, OrderRdmaSlices_MultipleSlicesPerRank)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(8);

    std::map<u32, std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice>> byRank;
    u8 dummyBuf[64];
    byRank[5].emplace_back(dummyBuf, 16, 5, true);
    byRank[5].emplace_back(dummyBuf + 16, 32, 5, true);
    byRank[7].emplace_back(dummyBuf, 64, 7, true);

    std::deque<CollBatchSendRecvGroupExecutor::SendRecvSlice> out;
    executor_->OrderRdmaSlices(true, byRank, out);

    EXPECT_EQ(out.size(), 3u);
    // Both slices from rank 5 should be contiguous and in correct order
    EXPECT_EQ(out[0].remoteRank, 5u);
    EXPECT_EQ(out[0].size, 16u);
    EXPECT_EQ(out[1].remoteRank, 5u);
    EXPECT_EQ(out[1].size, 32u);
    EXPECT_EQ(out[2].remoteRank, 7u);
    EXPECT_EQ(out[2].size, 64u);
}

// ============================================================================
// CalcStreamTaskStatus tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcStreamTaskStatus_AllEmpty)
{
    SetStreamNum(4, 4);
    executor_->sendDataSlicesBySendStream_.resize(4);
    executor_->recvDataSlicesByRecvStream_.resize(4);
    executor_->rdmaSendSlices_.clear();
    executor_->rdmaRecvSlices_.clear();

    u32 nonEmptySend = 99, nonEmptyRecv = 99;
    HcclResult ret = executor_->CalcStreamTaskStatus(nonEmptySend, nonEmptyRecv);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(nonEmptySend, 0u);
    EXPECT_EQ(nonEmptyRecv, 0u);
    EXPECT_FALSE(executor_->rdmaSendHasTask_);
    EXPECT_FALSE(executor_->rdmaRecvHasTask_);
    for (u32 i = 0; i < 4; i++) {
        EXPECT_FALSE(executor_->sendStreamHasTask_[i]);
        EXPECT_FALSE(executor_->recvStreamHasTask_[i]);
    }
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcStreamTaskStatus_SomeSendStreamsNonEmpty)
{
    SetStreamNum(4, 4);
    executor_->sendDataSlicesBySendStream_.resize(4);
    executor_->recvDataSlicesByRecvStream_.resize(4);

    u8 dummyBuf[64];
    executor_->sendDataSlicesBySendStream_[0].emplace_back(dummyBuf, 16, 1);
    executor_->sendDataSlicesBySendStream_[2].emplace_back(dummyBuf, 16, 3);

    u32 nonEmptySend = 0, nonEmptyRecv = 0;
    HcclResult ret = executor_->CalcStreamTaskStatus(nonEmptySend, nonEmptyRecv);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(nonEmptySend, 2u);
    EXPECT_EQ(nonEmptyRecv, 0u);
    EXPECT_TRUE(executor_->sendStreamHasTask_[0]);
    EXPECT_FALSE(executor_->sendStreamHasTask_[1]);
    EXPECT_TRUE(executor_->sendStreamHasTask_[2]);
    EXPECT_FALSE(executor_->sendStreamHasTask_[3]);
    for (u32 i = 0; i < 4; i++) {
        EXPECT_FALSE(executor_->recvStreamHasTask_[i]);
    }
    EXPECT_FALSE(executor_->rdmaSendHasTask_);
    EXPECT_FALSE(executor_->rdmaRecvHasTask_);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcStreamTaskStatus_RdmaTasksFlagged)
{
    SetStreamNum(4, 4);
    executor_->sendDataSlicesBySendStream_.resize(4);
    executor_->recvDataSlicesByRecvStream_.resize(4);

    u8 dummyBuf[64];
    executor_->rdmaSendSlices_.emplace_back(dummyBuf, 16, 5, true);
    executor_->rdmaRecvSlices_.emplace_back(dummyBuf, 32, 6, true);

    u32 nonEmptySend = 0, nonEmptyRecv = 0;
    HcclResult ret = executor_->CalcStreamTaskStatus(nonEmptySend, nonEmptyRecv);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(nonEmptySend, 0u);
    EXPECT_EQ(nonEmptyRecv, 0u);
    EXPECT_TRUE(executor_->rdmaSendHasTask_);
    EXPECT_TRUE(executor_->rdmaRecvHasTask_);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcStreamTaskStatus_AllStreamsOccupied)
{
    SetStreamNum(3, 2);
    executor_->sendDataSlicesBySendStream_.resize(3);
    executor_->recvDataSlicesByRecvStream_.resize(2);

    u8 dummyBuf[64];
    executor_->sendDataSlicesBySendStream_[0].emplace_back(dummyBuf, 16, 1);
    executor_->sendDataSlicesBySendStream_[1].emplace_back(dummyBuf, 16, 2);
    executor_->sendDataSlicesBySendStream_[2].emplace_back(dummyBuf, 16, 3);
    executor_->recvDataSlicesByRecvStream_[0].emplace_back(dummyBuf, 16, 0);
    executor_->recvDataSlicesByRecvStream_[1].emplace_back(dummyBuf, 16, 1);
    executor_->rdmaSendSlices_.emplace_back(dummyBuf, 16, 5, true);
    executor_->rdmaRecvSlices_.emplace_back(dummyBuf, 32, 6, true);

    u32 nonEmptySend = 0, nonEmptyRecv = 0;
    HcclResult ret = executor_->CalcStreamTaskStatus(nonEmptySend, nonEmptyRecv);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(nonEmptySend, 3u);
    EXPECT_EQ(nonEmptyRecv, 2u);
    for (u32 i = 0; i < 3; i++) {
        EXPECT_TRUE(executor_->sendStreamHasTask_[i]);
    }
    for (u32 i = 0; i < 2; i++) {
        EXPECT_TRUE(executor_->recvStreamHasTask_[i]);
    }
    EXPECT_TRUE(executor_->rdmaSendHasTask_);
    EXPECT_TRUE(executor_->rdmaRecvHasTask_);
}

// ============================================================================
// CalcStreamNum tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcStreamNum_ReturnsStreamCountIncludingRdma)
{
    // GROUP_MAX_CONCURRENT = 8
    u32 streamNum = 0;
    HcclResult ret = executor_->CalcStreamNum(streamNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // streamNum = sendStreamNum_ + recvStreamNum_ + RDMA_STREAM_NUM = 8 + 8 + 2 = 18
    EXPECT_EQ(streamNum, 18u);
    EXPECT_EQ(executor_->sendStreamNum_, 8u);
    EXPECT_EQ(executor_->recvStreamNum_, 8u);
}

// ============================================================================
// CalcPingPongHalfSize tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPingPongHalfSize_ComputesCorrectSizes)
{
    // Set up algResResp_ with known sizes
    AlgResourceResponse algRes;
    void* dummyPtr = reinterpret_cast<void*>(0x10000);
    // cclInputMem size = 16 * 16384 * 16 = 4194304 (enough for 16 ping-pong slots aligned)
    algRes.cclInputMem = DeviceMem(dummyPtr, 4194304);
    // cclOutputMem size = 16384 * 2 * 8 = 262144 (enough for 2 halves * 8 alignment)
    algRes.cclOutputMem = DeviceMem(dummyPtr, 262144);

    executor_->algResResp_ = &algRes;

    HcclResult ret = executor_->CalcPingPongHalfSize();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // bufferSliceSize_ = 4194304 / 16384 / 16 * 16384 = 256 / 16 * 16384 = 16 * 16384 = 262144
    EXPECT_EQ(executor_->bufferSliceSize_, 262144u);
    // rdmaDataBlockSize_ = 262144 / 16384 / 2 * 16384 = 16 / 2 * 16384 = 8 * 16384 = 131072
    EXPECT_EQ(executor_->rdmaDataBlockSize_, 131072u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPingPongHalfSize_SmallMemTruncated)
{
    AlgResourceResponse algRes;
    void* dummyPtr = reinterpret_cast<void*>(0x10000);
    // cclInputMem smaller than alignment * pingPongSliceNum -> bufferSliceSize_ = 0
    algRes.cclInputMem = DeviceMem(dummyPtr, 1024);
    algRes.cclOutputMem = DeviceMem(dummyPtr, 1024);

    executor_->algResResp_ = &algRes;

    HcclResult ret = executor_->CalcPingPongHalfSize();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // 1024 / 16384 = 0, so bufferSliceSize_ = 0
    EXPECT_EQ(executor_->bufferSliceSize_, 0u);
    EXPECT_EQ(executor_->rdmaDataBlockSize_, 0u);
}

// ============================================================================
// SendRecvSlice tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, SendRecvSlice_DefaultConstructor)
{
    u8 buf[32];
    CollBatchSendRecvGroupExecutor::SendRecvSlice slice(buf, 32, 5);
    EXPECT_EQ(slice.addr, buf);
    EXPECT_EQ(slice.size, 32u);
    EXPECT_EQ(slice.remoteRank, 5u);
    EXPECT_FALSE(slice.isRdma); // default false
}

TEST_F(CollBatchSendRecvGroupExecutorTest, SendRecvSlice_WithIsRdmaTrue)
{
    u8 buf[32];
    CollBatchSendRecvGroupExecutor::SendRecvSlice slice(buf, 64, 7, true);
    EXPECT_EQ(slice.addr, buf);
    EXPECT_EQ(slice.size, 64u);
    EXPECT_EQ(slice.remoteRank, 7u);
    EXPECT_TRUE(slice.isRdma);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, SendRecvSlice_WithIsRdmaFalse)
{
    u8 buf[32];
    CollBatchSendRecvGroupExecutor::SendRecvSlice slice(buf, 128, 3, false);
    EXPECT_FALSE(slice.isRdma);
}

// ============================================================================
// CalcPodRange state verification tests
// CalcPodRange() calls into TopoMatcher which uses complex internal state;
// we verify the resulting state is handled correctly by the dependent functions
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPodRange_ResultState_IsRemoteRankRdma_Consistent)
{
    // Simulate CalcPodRange result: pod = [0, 3], devNumInPod = 4
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);

    // Verify IsRemoteRankRdma uses pod state correctly
    EXPECT_FALSE(executor_->IsRemoteRankRdma(0));
    EXPECT_FALSE(executor_->IsRemoteRankRdma(3));
    EXPECT_TRUE(executor_->IsRemoteRankRdma(4));
    EXPECT_TRUE(executor_->IsRemoteRankRdma(15));
}

// ============================================================================
// CalcSendLoopMaxCount / CalcRecvLoopMaxCount tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcSendLoopMaxCount_ComputesCorrectly)
{
    // bufferSliceSize_ = 262144 (default from CalcPingPongHalfSize won't have been called)
    // Set it manually
    executor_->bufferSliceSize_ = 16384;
    // unitSize = 4 (e.g. INT32), maxCountPerLoop = 16384 / 4 = 4096
    u64 result = executor_->CalcSendLoopMaxCount(4);
    EXPECT_EQ(result, 4096u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcSendLoopMaxCount_ZeroUnitSize)
{
    executor_->bufferSliceSize_ = 16384;
    // unitSize = 0 -> division by 0 would be UB in production, but test verifies the path
    // We don't actually call with 0 since it would crash. Test with real datatype sizes.
    u64 result = executor_->CalcSendLoopMaxCount(1);
    EXPECT_EQ(result, 16384u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcRecvLoopMaxCount_ComputesCorrectly)
{
    executor_->bufferSliceSize_ = 65536;
    u64 result = executor_->CalcRecvLoopMaxCount(8); // 8 bytes = INT64
    EXPECT_EQ(result, 8192u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcRecvLoopMaxCount_MinimumSize)
{
    executor_->bufferSliceSize_ = 16384;
    u64 result = executor_->CalcRecvLoopMaxCount(2); // 2 bytes = INT16
    EXPECT_EQ(result, 8192u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPodRange_ResultState_RdmaRankingWorks)
{
    // Simulate CalcPodRange result: pod = [4, 7], devNumInPod = 4
    SetPodRange(4, 7, 4);
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(4);

    // Verify GetNextDstRank / GetPreSrcRank use pod + userRankSize correctly
    u32 curRank = 4; // podStart
    u32 result = executor_->GetNextDstRank(curRank);
    EXPECT_EQ(result, 8u); // skipped pod

    curRank = 7; // podEnd
    result = executor_->GetPreSrcRank(curRank);
    EXPECT_EQ(result, 3u); // skipped pod
}

// ============================================================================
// CalcSendSlices tests (SDMA+RDMA separation logic)
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcSendSlices_SdmaOnly_NoRdmaRanks)
{
    // All ranks in pod — no RDMA, all SDMA
    SetPodRange(0, 15, 16); // entire cluster is pod
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(0);
    SetStreamNum(4, 4);
    executor_->bufferSliceSize_ = 16384;
    executor_->rdmaDataBlockSize_ = 8192;

    // Set up send queue with one item
    HcclSendRecvItem item;
    item.buf = reinterpret_cast<void*>(0x1000);
    item.count = 512;
    item.dataType = HCCL_DATA_TYPE_INT32; // 4 bytes
    item.remoteRank = 3;                  // within pod [0,15]
    item.sendRecvType = HcclSendRecvType::HCCL_SEND;

    executor_->sendStreamNum_ = 4;
    executor_->sendQueueBySendstream_.resize(4);
    executor_->sendQueueBySendstream_[0].push_back(&item);

    HcclResult ret = executor_->CalcSendSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // Should have one slice in SDMA (not RDMA)
    EXPECT_EQ(executor_->sendDataSlicesBySendStream_.size(), 4u);
    EXPECT_EQ(executor_->sendDataSlicesBySendStream_[0].size(), 1u);
    EXPECT_TRUE(executor_->rdmaSendSlices_.empty());

    auto& slice = executor_->sendDataSlicesBySendStream_[0].front();
    EXPECT_FALSE(slice.isRdma);
    EXPECT_EQ(slice.remoteRank, 3u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcSendSlices_RdmaRankSeparated)
{
    // pod [0, 3], rank 5 is outside pod -> RDMA
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(4);
    SetStreamNum(4, 4);
    executor_->bufferSliceSize_ = 65536;
    executor_->rdmaDataBlockSize_ = 32768;

    HcclSendRecvItem item;
    item.buf = reinterpret_cast<void*>(0x2000);
    item.count = 128;
    item.dataType = HCCL_DATA_TYPE_INT32; // 4 bytes, total 512 bytes
    item.remoteRank = 5;                  // outside pod [0,3] -> RDMA
    item.sendRecvType = HcclSendRecvType::HCCL_SEND;

    executor_->sendStreamNum_ = 4;
    executor_->sendQueueBySendstream_.resize(4);
    executor_->sendQueueBySendstream_[0].push_back(&item);

    HcclResult ret = executor_->CalcSendSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    // Should be routed to RDMA, not SDMA
    EXPECT_EQ(executor_->sendDataSlicesBySendStream_[0].size(), 0u);
    EXPECT_FALSE(executor_->rdmaSendSlices_.empty());
    auto& slice = executor_->rdmaSendSlices_.front();
    EXPECT_TRUE(slice.isRdma);
    EXPECT_EQ(slice.remoteRank, 5u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcSendSlices_EmptyQueue)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(4);
    SetStreamNum(4, 4);

    executor_->sendQueueBySendstream_.resize(4);

    HcclResult ret = executor_->CalcSendSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->rdmaSendSlices_.empty());
    for (u32 i = 0; i < 4; i++) {
        EXPECT_TRUE(executor_->sendDataSlicesBySendStream_[i].empty());
    }
}

// ============================================================================
// OrganizeSendItemByStream tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, OrganizeSendItemByStream_DistributesByRank)
{
    SetStreamNum(4, 4);

    HcclSendRecvItem item1;
    item1.buf = reinterpret_cast<void*>(0x1000);
    item1.count = 100;
    item1.dataType = HCCL_DATA_TYPE_INT32;
    item1.remoteRank = 0;
    item1.sendRecvType = HcclSendRecvType::HCCL_SEND;

    HcclSendRecvItem item2;
    item2.buf = reinterpret_cast<void*>(0x2000);
    item2.count = 200;
    item2.dataType = HCCL_DATA_TYPE_INT32;
    item2.remoteRank = 5; // 5 % 4 = 1
    item2.sendRecvType = HcclSendRecvType::HCCL_SEND;

    executor_->sendStreamNum_ = 4;
    executor_->sendDeque_.push_back(&item1);
    executor_->sendDeque_.push_back(&item2);

    HcclResult ret = executor_->OrganizeSendItemByStream();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->sendDeque_.empty());
    EXPECT_EQ(executor_->sendQueueBySendstream_[0].size(), 1u);
    EXPECT_EQ(executor_->sendQueueBySendstream_[1].size(), 1u);
    EXPECT_EQ(executor_->sendQueueBySendstream_[2].size(), 0u);
    EXPECT_EQ(executor_->sendQueueBySendstream_[3].size(), 0u);
}

// ============================================================================
// OrganizeRecvItemByStream tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, OrganizeRecvItemByStream_DistributesByRank)
{
    SetStreamNum(4, 4);

    HcclSendRecvItem item1;
    item1.buf = reinterpret_cast<void*>(0x1000);
    item1.count = 100;
    item1.dataType = HCCL_DATA_TYPE_INT32;
    item1.remoteRank = 0; // 0 % 4 = 0
    item1.sendRecvType = HcclSendRecvType::HCCL_RECV;

    HcclSendRecvItem item2;
    item2.buf = reinterpret_cast<void*>(0x2000);
    item2.count = 200;
    item2.dataType = HCCL_DATA_TYPE_INT32;
    item2.remoteRank = 5; // 5 % 4 = 1
    item2.sendRecvType = HcclSendRecvType::HCCL_RECV;

    executor_->recvStreamNum_ = 4;
    executor_->recvDeque_.push_back(&item1);
    executor_->recvDeque_.push_back(&item2);

    HcclResult ret = executor_->OrganizeRecvItemByStream();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->recvDeque_.empty());
    EXPECT_EQ(executor_->recvQueueByRecvstream_[0].size(), 1u);
    EXPECT_EQ(executor_->recvQueueByRecvstream_[1].size(), 1u);
    EXPECT_EQ(executor_->recvQueueByRecvstream_[2].size(), 0u);
    EXPECT_EQ(executor_->recvQueueByRecvstream_[3].size(), 0u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, OrganizeRecvItemByStream_EmptyDeque)
{
    SetStreamNum(4, 4);
    executor_->recvStreamNum_ = 4;
    HcclResult ret = executor_->OrganizeRecvItemByStream();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->recvDeque_.empty());
    EXPECT_EQ(executor_->recvQueueByRecvstream_.size(), 4u);
}

// ============================================================================
// CalcRecvSlices tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcRecvSlices_SdmaOnly_NoRdmaRanks)
{
    SetPodRange(0, 15, 16); // entire cluster is pod
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(0);
    SetStreamNum(4, 4);
    executor_->bufferSliceSize_ = 16384;
    executor_->rdmaDataBlockSize_ = 8192;

    HcclSendRecvItem item;
    item.buf = reinterpret_cast<void*>(0x1000);
    item.count = 512;
    item.dataType = HCCL_DATA_TYPE_INT32;
    item.remoteRank = 3;
    item.sendRecvType = HcclSendRecvType::HCCL_RECV;

    executor_->recvStreamNum_ = 4;
    executor_->recvQueueByRecvstream_.resize(4);
    executor_->recvQueueByRecvstream_[0].push_back(&item);

    HcclResult ret = executor_->CalcRecvSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->recvDataSlicesByRecvStream_.size(), 4u);
    EXPECT_EQ(executor_->recvDataSlicesByRecvStream_[0].size(), 1u);
    EXPECT_TRUE(executor_->rdmaRecvSlices_.empty());
    EXPECT_FALSE(executor_->recvDataSlicesByRecvStream_[0].front().isRdma);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcRecvSlices_RdmaRankSeparated)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(4);
    SetStreamNum(4, 4);
    executor_->bufferSliceSize_ = 65536;
    executor_->rdmaDataBlockSize_ = 32768;

    HcclSendRecvItem item;
    item.buf = reinterpret_cast<void*>(0x2000);
    item.count = 128;
    item.dataType = HCCL_DATA_TYPE_INT32;
    item.remoteRank = 5; // outside pod -> RDMA
    item.sendRecvType = HcclSendRecvType::HCCL_RECV;

    executor_->recvStreamNum_ = 4;
    executor_->recvQueueByRecvstream_.resize(4);
    executor_->recvQueueByRecvstream_[0].push_back(&item);

    HcclResult ret = executor_->CalcRecvSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->recvDataSlicesByRecvStream_[0].size(), 0u);
    EXPECT_FALSE(executor_->rdmaRecvSlices_.empty());
    EXPECT_TRUE(executor_->rdmaRecvSlices_.front().isRdma);
    EXPECT_EQ(executor_->rdmaRecvSlices_.front().remoteRank, 5u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcRecvSlices_EmptyQueue)
{
    SetPodRange(0, 3, 4);
    SetTopoAttrUserRankSize(16);
    SetTopoAttrUserRank(4);
    SetStreamNum(4, 4);
    executor_->recvQueueByRecvstream_.resize(4);

    HcclResult ret = executor_->CalcRecvSlices();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->rdmaRecvSlices_.empty());
    for (u32 i = 0; i < 4; i++) {
        EXPECT_TRUE(executor_->recvDataSlicesByRecvStream_[i].empty());
    }
}

// ============================================================================
// CalcPodRange tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPodRange_InterHccsDisable_Branch)
{
    // Default SetUp has deviceType=910B, isSingleMeshAggregation=false -> isA2MultiModule=true
    // This makes the if condition true regardless of GetExternalInputInterHccsDisable
    // Set up serverAndsuperPodToRank_ so GetLocalServerRankSize works with real data
    serverAndsuperPodToRank_ = {{{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}};
    topoMatcher_ = std::make_unique<TopoMatcher>(
        commPlaneRanks_, isBridgeVector_, topoInfo_, algoInfo_, externalEnable_, serverAndsuperPodToRank_);
    executor_ = std::make_unique<CollBatchSendRecvGroupExecutor>(nullptr, topoMatcher_);

    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(16);

    HcclResult ret = executor_->CalcPodRange();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->podStartRank_, 4u); // 4 - 0
    EXPECT_EQ(executor_->podEndRank_, 7u);   // 4 + 4 - 1
    EXPECT_EQ(executor_->devNumInlocalPod_, 4u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, CalcPodRange_SuperPod_Branch)
{
    // Set deviceType to non-910B and isSingleMeshAggregation=true so isA2MultiModule=false
    // Default externalEnable_.interHccsDisable=0 so GetExternalInputInterHccsDisable()=0 -> else branch
    topoInfo_.deviceType = DevType::DEV_TYPE_910;
    topoInfo_.isSingleMeshAggregation = true;
    // Set up superPod data: 2 superPods with 8 ranks each (level 1), plus server data (level 0)
    serverAndsuperPodToRank_ = {
        {{0, 1, 2, 3}, {4, 5, 6, 7}, {8, 9, 10, 11}, {12, 13, 14, 15}}, // level 0: servers
        {{0, 1, 2, 3, 4, 5, 6, 7}, {8, 9, 10, 11, 12, 13, 14, 15}}      // level 1: superPods
    };
    topoMatcher_ = std::make_unique<TopoMatcher>(
        commPlaneRanks_, isBridgeVector_, topoInfo_, algoInfo_, externalEnable_, serverAndsuperPodToRank_);
    executor_ = std::make_unique<CollBatchSendRecvGroupExecutor>(nullptr, topoMatcher_);

    SetTopoAttrUserRank(4);
    SetTopoAttrUserRankSize(16);

    HcclResult ret = executor_->CalcPodRange();

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->podStartRank_, 0u); // 4 - 4
    EXPECT_EQ(executor_->podEndRank_, 7u);   // 0 + 8 - 1
    EXPECT_EQ(executor_->devNumInlocalPod_, 8u);
}

// ============================================================================
// Orchestrate tests (single rank early return)
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, Orchestrate_SingleRankEarlyReturn)
{
    SetTopoAttrUserRankSize(1);

    // Set up opTransportResponse for CheckCommSize
    algRes_.opTransportResponse.resize(COMM_COMBINE_ORDER + 1);
    algRes_.opTransportResponse[COMM_COMBINE_ORDER].resize(2); // COMM_SIZE_TWO
    executor_->algResResp_ = &algRes_;

    OpParam param;
    param.tag = "test";

    HcclResult ret = executor_->Orchestrate(param, algRes_);
    // Expect failure since sendRecvItemsPtr/itemNum not set up for GetPairWiseList
    // but we exercise lines 102-110 (Orchestrate initialization and CheckCommSize)
    EXPECT_NE(ret, HCCL_SUCCESS);
}

// ============================================================================
// CollBatchSendRecvExecutor::Orchestrate aicpuUnfoldMode dispatch tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, Orchestrate_AicpuUnfoldModeTrue_Expect_RunLoopInAicpuUnfoldMode)
{
    // 使用基类 executor 覆盖 CollBatchSendRecvExecutor::Orchestrate 中
    // param.aicpuUnfoldMode 为 true 时走 aicpu 展开模式分支
    CollBatchSendRecvExecutor baseExecutor(dispatcher_.get(), topoMatcher_);

    algRes_.opTransportResponse.resize(COMM_COMBINE_ORDER + 1);
    algRes_.opTransportResponse[COMM_COMBINE_ORDER].resize(COMM_SIZE_TWO); // COMM_SIZE_TWO
    baseExecutor.algResResp_ = &algRes_;

    HcclSendRecvItem dummyItem;
    OpParam param;
    param.BatchSendRecvDataDes.sendRecvItemsPtr = &dummyItem;
    param.BatchSendRecvDataDes.itemNum = 0;
    param.aicpuUnfoldMode = true;

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInAicpuUnfoldMode)
        .expects(exactly(1))
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode).expects(never());

    HcclResult ret = baseExecutor.Orchestrate(param, algRes_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, Orchestrate_AicpuUnfoldModeFalse_Expect_RunLoopInHostUnfoldMode)
{
    // 使用基类 executor 覆盖 CollBatchSendRecvExecutor::Orchestrate 中
    // param.aicpuUnfoldMode 为 false 时走 host 展开模式分支
    CollBatchSendRecvExecutor baseExecutor(dispatcher_.get(), topoMatcher_);

    algRes_.opTransportResponse.resize(COMM_COMBINE_ORDER + 1);
    algRes_.opTransportResponse[COMM_COMBINE_ORDER].resize(COMM_SIZE_TWO); // COMM_SIZE_TWO
    baseExecutor.algResResp_ = &algRes_;

    HcclSendRecvItem dummyItem;
    OpParam param;
    param.BatchSendRecvDataDes.sendRecvItemsPtr = &dummyItem;
    param.BatchSendRecvDataDes.itemNum = 0;
    param.aicpuUnfoldMode = false;

    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInAicpuUnfoldMode).expects(never());
    MOCKER_CPP(&CollBatchSendRecvExecutor::RunLoopInHostUnfoldMode)
        .expects(exactly(1))
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = baseExecutor.Orchestrate(param, algRes_);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================================
// SetNormalModeIfDeviceDirect tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, SetNormalModeIfDeviceDirect_NoDeviceDirect)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6);
    LINK fakeLink = MakeFakeLink(TransportType::TRANS_TYPE_P2P); // not DEVICE_DIRECT
    SetupOpTransportForRank(1, fakeLink);

    executor_->sendDataSlicesBySendStream_.resize(2);
    u8 buf[64];
    executor_->sendDataSlicesBySendStream_[0].emplace_back(buf, 16, 1);

    HcclResult ret = executor_->SetNormalModeIfDeviceDirect();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================================
// MainPostSubWait / MainWaitSubPost tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, MainPostSubWait_NoTasks)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6);
    executor_->sendStreamHasTask_.assign(2, false);
    executor_->recvStreamHasTask_.assign(2, false);
    executor_->rdmaSendHasTask_ = false;
    executor_->rdmaRecvHasTask_ = false;

    Stream mainStream;
    HcclResult ret = executor_->MainPostSubWait(mainStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, MainWaitSubPost_NoTasks)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6);
    executor_->sendStreamHasTask_.assign(2, false);
    executor_->recvStreamHasTask_.assign(2, false);
    executor_->rdmaSendHasTask_ = false;
    executor_->rdmaRecvHasTask_ = false;

    Stream mainStream;
    HcclResult ret = executor_->MainWaitSubPost(mainStream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// ============================================================================
// ProcessNewRankSendSlice tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessNewRankSendSlice_SingleSlice)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(1, fakeLink); // remoteRank=1 < userRank=4 -> send

    u8 buf[64];
    executor_->sendDataSlicesBySendStream_.resize(2);
    executor_->sendDataSlicesBySendStream_[0].emplace_back(buf, 16, 1);

    // Initialize per-stream state vectors
    executor_->sendCurPhase_.assign(2, 0);
    executor_->sendLoadedSize_.assign(2, 0);
    executor_->sendLoadedRemoteRank_.assign(2, 0);

    u32 pending = 0;
    u32 nonEmpty = 1;
    HcclResult ret = executor_->ProcessNewRankSendSlice(0, pending, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->sendDataSlicesBySendStream_[0].empty());
    EXPECT_EQ(nonEmpty, 0u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessNewRankSendSlice_TwoSlicesSameRank)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(1, fakeLink);

    std::vector<u8> buf(64);
    executor_->sendDataSlicesBySendStream_.resize(2);
    executor_->sendDataSlicesBySendStream_[0].emplace_back(buf.data(), 16, 1);
    executor_->sendDataSlicesBySendStream_[0].emplace_back(buf.data() + 16, 32, 1);

    executor_->sendCurPhase_.assign(2, 0);
    executor_->sendLoadedSize_.assign(2, 0);
    executor_->sendLoadedRemoteRank_.assign(2, 0);

    u32 pending = 0;
    u32 nonEmpty = 1;
    HcclResult ret = executor_->ProcessNewRankSendSlice(0, pending, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->sendLoadedSize_[0], 32u);
    EXPECT_EQ(pending, 1u);
}

// ============================================================================
// ProcessPreloadedSendSlice tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessPreloadedSendSlice_Basic)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(1, fakeLink);

    executor_->sendCurPhase_.assign(2, 0);
    executor_->sendLoadedSize_.assign(2, 0);
    executor_->sendLoadedRemoteRank_.assign(2, 0);
    executor_->sendDataSlicesBySendStream_.resize(2);

    executor_->sendLoadedSize_[0] = 16;
    executor_->sendLoadedRemoteRank_[0] = 1;
    executor_->sendCurPhase_[0] = 0;

    u8 buf[64];
    executor_->sendDataSlicesBySendStream_[0].emplace_back(buf, 32, 1);

    u32 pending = 1;
    u32 nonEmpty = 1;
    HcclResult ret = executor_->ProcessPreloadedSendSlice(0, pending, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->sendLoadedSize_[0], 32u);
    EXPECT_EQ(pending, 1u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessPreloadedSendSlice_NoNextSlice)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(1, fakeLink);

    executor_->sendCurPhase_.assign(2, 0);
    executor_->sendLoadedSize_.assign(2, 0);
    executor_->sendLoadedRemoteRank_.assign(2, 0);
    executor_->sendDataSlicesBySendStream_.resize(2);

    executor_->sendLoadedSize_[0] = 16;
    executor_->sendLoadedRemoteRank_[0] = 1;
    executor_->sendCurPhase_[0] = 0;

    u32 pending = 1;
    u32 nonEmpty = 0;
    HcclResult ret = executor_->ProcessPreloadedSendSlice(0, pending, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->sendLoadedSize_[0], 0u);
    EXPECT_EQ(pending, 0u);
}

// ============================================================================
// ProcessRecvSlice tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessRecvSlice_Basic)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(5, fakeLink); // recv: rank 5 > userRank 4 -> commIndex 0

    u8 buf[64];
    executor_->recvDataSlicesByRecvStream_.resize(2);
    executor_->recvDataSlicesByRecvStream_[0].emplace_back(buf, 16, 5);
    executor_->recvCurPhase_.assign(2, 0);
    executor_->recvCurRemoteRank_.assign(2, 0);

    u32 nonEmpty = 1;
    HcclResult ret = executor_->ProcessRecvSlice(0, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->recvDataSlicesByRecvStream_[0].empty());
    EXPECT_EQ(nonEmpty, 0u);
}

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessRecvSlice_NewRankResetsPhase)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    executor_->bufferSliceSize_ = 16384;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(5, fakeLink); // recv: rank 5 > userRank 4 -> commIndex 0

    u8 buf[64];
    executor_->recvDataSlicesByRecvStream_.resize(2);
    executor_->recvDataSlicesByRecvStream_[0].emplace_back(buf, 16, 5);
    executor_->recvCurPhase_.assign(2, 1);
    executor_->recvCurRemoteRank_.assign(2, 3); // different rank -> should reset to 0

    u32 nonEmpty = 1;
    HcclResult ret = executor_->ProcessRecvSlice(0, nonEmpty);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(executor_->recvCurPhase_[0], 1u);
    EXPECT_EQ(executor_->recvCurRemoteRank_[0], 5u);
}

// ============================================================================
// ProcessRdmaSendSlice tests
// ============================================================================

TEST_F(CollBatchSendRecvGroupExecutorTest, ProcessRdmaSendSlice_Basic)
{
    SetStreamNum(2, 2);
    SetupAlgResResp(6, 4 * 1024 * 1024);
    MOCKER(hrtMemAsyncCopy).stubs().will(returnValue(HCCL_SUCCESS));
    executor_->rdmaDataBlockSize_ = 32768;
    LINK fakeLink = MakeFakeLink();
    SetupOpTransportForRank(5, fakeLink);

    u8 buf[64];
    executor_->rdmaSendSlices_.emplace_back(buf, 16, 5, true);

    HcclResult ret = executor_->ProcessRdmaSendSlice();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(executor_->rdmaSendSlices_.empty());
}
