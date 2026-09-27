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
#include "aicpu/aicpu_ts_hccs_channel.h"
#include "endpoint.h"
#include "channel_param.h"
#include "hccs_reged_mem_mgr.h"
#undef private
#undef protected

using namespace hcomm;

namespace {
class StubAicpuTsHccsEndpoint : public AicpuTsHccsEndpoint {
public:
    explicit StubAicpuTsHccsEndpoint(const EndpointDesc& desc) : AicpuTsHccsEndpoint(desc) {}
    HcclResult Init() override { return HCCL_SUCCESS; }
};
} // namespace

class AicpuTsHccsChannelTest : public testing::Test {
protected:
    void TearDown() override { GlobalMockObject::verify(); }
};

static HcclResult StubGetTransportAttr(hccl::Transport* /*self*/, hccl::TransportAttr& attr)
{
    attr = hccl::TransportAttr{};
    return HCCL_SUCCESS;
}

TEST_F(AicpuTsHccsChannelTest, UT_BuildHcclChannelHccsRes_WhenQosConfigured_ExpectQosPropagated)
{
    EndpointDesc epDesc{};
    epDesc.protocol = COMM_PROTOCOL_HCCS;
    epDesc.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    epDesc.loc.device.devPhyId = 3;
    StubAicpuTsHccsEndpoint stubEp(epDesc);
    stubEp.regedMemMgr_ = std::make_shared<HccsRegedMemMgr>(nullptr);

    HcommChannelDesc desc{};
    desc.qos = 42;
    desc.role = HCOMM_SOCKET_ROLE_CLIENT;
    desc.remoteEndpoint.protocol = COMM_PROTOCOL_HCCS;
    desc.remoteEndpoint.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    desc.remoteEndpoint.loc.device.devPhyId = 4;

    AicpuTsHccsChannel ch(reinterpret_cast<EndpointHandle>(&stubEp), desc);
    ch.socketTag_ = "ut_test";
    ch.isSocketServer_ = false;
    ch.localEp_.protocol = COMM_PROTOCOL_HCCS;
    ch.localEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.localEp_.loc.device.devPhyId = 3;
    ch.remoteEp_.protocol = COMM_PROTOCOL_HCCS;
    ch.remoteEp_.loc.locType = ENDPOINT_LOC_TYPE_DEVICE;
    ch.remoteEp_.loc.device.devPhyId = 4;
    ch.localEpPtr_ = &stubEp;
    ch.transport_ = std::make_shared<hccl::Transport>();

    MOCKER_CPP(&hccl::Transport::GetNotifyNum).stubs().will(returnValue(8U));
    MOCKER_CPP(&hccl::Transport::GetTransportAttr).stubs().will(invoke(StubGetTransportAttr));
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_950)).will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDeviceIndexByPhyId).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HccsRegedMemMgr::GetRemoteIpcRmaBufferEx).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HccsRegedMemMgr::GetLocalIpcRmaBufferEx).stubs().will(returnValue(HCCL_SUCCESS));

    HcclChannelHccsRes res;
    HcclResult ret = ch.BuildHcclChannelHccsRes(res);

    ASSERT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(res.channelP2p.qos, static_cast<u32>(desc.qos));
}
