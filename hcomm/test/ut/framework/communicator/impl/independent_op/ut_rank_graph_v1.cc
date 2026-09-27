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
#include "rank_graph.h"
#include "rank_graph_base.h"
#include "topoinfo_struct.h"
#include "hccl_common.h"

using namespace hccl;

class RankGraphV1DirectTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RankGraphV1DirectTest SetUp" << std::endl; }
    static void TearDownTestCase() { std::cout << "RankGraphV1DirectTest TearDown" << std::endl; }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }
};

TEST_F(RankGraphV1DirectTest, Ut_GetTopoInstsByLayer_When_ValidLayer_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    std::vector<u32> instRanks = {0, 1, 2, 3};
    rankGraph.rankSizeList_[0] = instRanks;

    uint32_t netLayer = 0;
    uint32_t* topoInsts = nullptr;
    uint32_t topoInstNum = 0;

    HcclResult ret = rankGraph.GetTopoInstsByLayer(netLayer, &topoInsts, &topoInstNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoInstNum, 4u);
    EXPECT_NE(topoInsts, nullptr);
}

TEST_F(RankGraphV1DirectTest, Ut_GetTopoType_When_ValidParams_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.topoAttr_.deviceType = DevType::DEV_TYPE_910_93;

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;

    HcclResult ret = rankGraph.GetTopoType(netLayer, &topoType);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, CommTopo::COMM_TOPO_910_93);
}

TEST_F(RankGraphV1DirectTest, Ut_GetRanksByTopoInst_When_ValidParams_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.devType_ = DevType::DEV_TYPE_910B;
    std::vector<u32> rankList = {0, 1, 2, 3};
    rankGraph.rankList_[0] = rankList;
    std::vector<u32> rankSizeList = {2, 2};
    rankGraph.rankSizeList_[0] = rankSizeList;

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t* ranks = nullptr;
    uint32_t rankNum = 0;

    HcclResult ret = rankGraph.GetRanksByTopoInst(netLayer, topoInstId, &ranks, &rankNum);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(ranks, nullptr);
    EXPECT_EQ(rankNum, 2u);
}

TEST_F(RankGraphV1DirectTest, Ut_GetRanksInTopoInst_When_ValidParams_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.netLayer_.push_back(1);
    std::vector<u32> rankList = {0, 1};
    rankGraph.rankList_[1] = rankList;
    std::vector<u32> rankSizeList = {2};
    rankGraph.rankSizeList_[1] = rankSizeList;

    RankInfo_t graphInfo;
    graphInfo.rankId = 0;
    rankGraph.rankGraph_.push_back(graphInfo);
    graphInfo.rankId = 1;
    rankGraph.rankGraph_.push_back(graphInfo);

    uint32_t netLayer = 1;
    uint32_t topoInstId = 0;

    auto ranks = rankGraph.GetRanksInTopoInst(netLayer, topoInstId);

    EXPECT_EQ(ranks.size(), 2u);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointInfo_When_NormalParams_Expect_Success)
{
    RankGraphV1 rankGraph;

    RankInfo_t rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.devicePhyId = 0;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&endpoint.commAddr.addr, 0, sizeof(endpoint.commAddr.addr));

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo = rankInfo;
    graphInfo.endPoints.push_back(endpoint);

    rankGraph.rankIndex_[0] = graphInfo;

    uint32_t rankId = 0;
    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&queryEndpoint.commAddr.addr, 0, sizeof(queryEndpoint.commAddr.addr));

    EndpointAttrBwCoeff bwCoeff = 0;
    HcclResult ret = rankGraph.GetEndpointInfo(
        rankId, &queryEndpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), &bwCoeff);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(bwCoeff, 1);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointDesc_When_Success_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.rankData_.userRank = 0;

    RankInfo_t rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.devicePhyId = 0;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&endpoint.commAddr.addr, 0, sizeof(endpoint.commAddr.addr));

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo = rankInfo;
    graphInfo.endPoints.push_back(endpoint);

    rankGraph.rankIndex_[0] = graphInfo;

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t descNum = 1;
    EndpointDesc endpointDesc[1];

    HcclResult ret = rankGraph.GetEndpointDesc(netLayer, topoInstId, &descNum, endpointDesc);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointNum_When_Success_Expect_Success)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.rankData_.userRank = 0;

    RankInfo_t rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.devicePhyId = 0;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&endpoint.commAddr.addr, 0, sizeof(endpoint.commAddr.addr));

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo = rankInfo;
    graphInfo.endPoints.push_back(endpoint);

    rankGraph.rankIndex_[0] = graphInfo;

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t num = 0;

    HcclResult ret = rankGraph.GetEndpointNum(netLayer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RankGraphV1DirectTest, Ut_GetProtocolsByConnections_When_Success_Expect_NonEmpty)
{
    RankGraphV1 rankGraph;

    rankGraph.netLayer_.push_back(0);
    rankGraph.rankData_.userRank = 0;
    rankGraph.devType_ = DevType::DEV_TYPE_910_93;

    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    rankInfo0.serverIdx = 0;
    rankInfo0.superPodId = "";
    rankInfo0.deviceInfo.deviceType = DevType::DEV_TYPE_910_93;

    RankInfo_t rankInfo1;
    rankInfo1.rankId = 1;
    rankInfo1.serverIdx = 0;
    rankInfo1.superPodId = "";
    rankInfo1.deviceInfo.deviceType = DevType::DEV_TYPE_910_93;

    rankGraph.rankGraph_.push_back(rankInfo0);
    rankGraph.rankGraph_.push_back(rankInfo1);

    std::vector<const RankInfo_t*> topoInstRanks;
    topoInstRanks.push_back(&rankGraph.rankGraph_[1]);

    std::set<CommProtocol> protocols = rankGraph.GetProtocolsByConnections(0, topoInstRanks);
    EXPECT_EQ(protocols.empty(), false);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_EndPointAttrDieId_Expect_Success)
{
    RankGraphV1 rankGraph;

    EndpointDesc foundEndpoint;
    foundEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    foundEndpoint.loc.device.superDevId = 123;

    EndpointAttrDieId dieId = 0;
    HcclResult ret = rankGraph.FillAttr(ENDPOINT_ATTR_DIE_ID, &foundEndpoint, sizeof(EndpointAttrDieId), &dieId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(dieId, 123u);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_EndPointAttrLocation_Expect_Success)
{
    RankGraphV1 rankGraph;

    EndpointDesc foundEndpoint;
    foundEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    foundEndpoint.loc.device.superDevId = 0;

    EndpointAttrLocation locType = 0;
    HcclResult ret = rankGraph.FillAttr(ENDPOINT_ATTR_LOCATION, &foundEndpoint, sizeof(EndpointAttrLocation), &locType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(locType, EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE);
}

TEST_F(RankGraphV1DirectTest, Ut_GetRankSize_When_Normal_Expect_Success)
{
    RankGraphV1 rankGraph;

    RankInfo_t rankInfo0;
    rankInfo0.rankId = 0;
    rankGraph.rankGraph_.push_back(rankInfo0);
    RankInfo_t rankInfo1;
    rankInfo1.rankId = 1;
    rankGraph.rankGraph_.push_back(rankInfo1);

    uint32_t rankSize = 0;
    HcclResult ret = rankGraph.GetRankSize(&rankSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, 2u);
}

TEST_F(RankGraphV1DirectTest, Ut_GetRankSize_When_RankSizeNullptr_Expect_HCCL_E_PTR)
{
    RankGraphV1 rankGraph;
    HcclResult ret = rankGraph.GetRankSize(nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphV1DirectTest, Ut_GetDevicePort_When_RankExists_Expect_Success)
{
    RankGraphV1 rankGraph;

    RankInfo_t rankInfo;
    rankInfo.rankId = 0;
    rankInfo.deviceInfo.port = 123;

    // GetDevicePort uses FindRank which looks up in rankIndex_, not rankGraph_
    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo = rankInfo;
    rankGraph.rankIndex_[0] = graphInfo;

    uint32_t devPort = 0;
    HcclResult ret = rankGraph.GetDevicePort(0, &devPort);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(devPort, 123u);
}

TEST_F(RankGraphV1DirectTest, Ut_GetDevicePort_When_RankNotFound_Expect_HCCL_E_PARA)
{
    RankGraphV1 rankGraph;
    uint32_t devPort = 0;
    HcclResult ret = rankGraph.GetDevicePort(999, &devPort);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetDevicePort_When_DevPortNullptr_Expect_HCCL_E_PTR)
{
    RankGraphV1 rankGraph;
    HcclResult ret = rankGraph.GetDevicePort(0, nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphV1DirectTest, Ut_GetLinks_When_NormalCase_Expect_Success)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.devType_ = DevType::DEV_TYPE_910_93;

    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(LinkTypeInServer::HCCS_TYPE))
        .will(returnValue(HCCL_SUCCESS));

    RankGraphV1::RankGraphInfo srcGraphInfo;
    srcGraphInfo.rankInfo.rankId = 0;
    srcGraphInfo.rankInfo.serverIdx = 0;
    srcGraphInfo.rankInfo.superPodId = "";
    srcGraphInfo.rankInfo.deviceInfo.deviceType = DevType::DEV_TYPE_910_93;
    srcGraphInfo.rankInfo.deviceInfo.devicePhyId = 0;

    EndpointDesc srcEndpoint;
    srcEndpoint.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    srcGraphInfo.endPoints.push_back(srcEndpoint);
    rankGraph.rankIndex_[0] = srcGraphInfo;

    RankGraphV1::RankGraphInfo dstGraphInfo;
    dstGraphInfo.rankInfo.rankId = 1;
    dstGraphInfo.rankInfo.serverIdx = 0;
    dstGraphInfo.rankInfo.superPodId = "";
    dstGraphInfo.rankInfo.deviceInfo.deviceType = DevType::DEV_TYPE_910_93;
    dstGraphInfo.rankInfo.deviceInfo.devicePhyId = 1;

    EndpointDesc dstEndpoint;
    dstEndpoint.protocol = CommProtocol::COMM_PROTOCOL_HCCS;
    dstGraphInfo.endPoints.push_back(dstEndpoint);
    rankGraph.rankIndex_[1] = dstGraphInfo;

    uint32_t netLayer = 0;
    uint32_t srcRank = 0;
    uint32_t dstRank = 1;
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;

    HcclResult ret = rankGraph.GetLinks(netLayer, srcRank, dstRank, &linkList, &listSize);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 1u);
    EXPECT_NE(linkList, nullptr);
}

TEST_F(RankGraphV1DirectTest, Ut_MatchEndpointByAddr_When_IpV6Match_Expect_Pointer)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.devType_ = DevType::DEV_TYPE_910_93;

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;
    graphInfo.rankInfo.serverIdx = 0;
    graphInfo.rankInfo.deviceInfo.deviceType = DevType::DEV_TYPE_910_93;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V6;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&endpoint.commAddr.addr6, 0xAB, sizeof(endpoint.commAddr.addr6));
    graphInfo.endPoints.push_back(endpoint);

    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V6;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&queryEndpoint.commAddr.addr6, 0xAB, sizeof(queryEndpoint.commAddr.addr6));

    const EndpointDesc* result = rankGraph.MatchEndpointByAddr(graphInfo, &queryEndpoint);
    EXPECT_NE(result, nullptr);
}

TEST_F(RankGraphV1DirectTest, Ut_MatchEndpointByAddr_When_IdType_Expect_Pointer)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_ID;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_PCIE;
    graphInfo.endPoints.push_back(endpoint);

    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_ID;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_PCIE;

    const EndpointDesc* result = rankGraph.MatchEndpointByAddr(graphInfo, &queryEndpoint);
    EXPECT_NE(result, nullptr);
}

TEST_F(RankGraphV1DirectTest, Ut_MatchEndpointByAddr_When_UnknownType_Expect_Nullptr)
{
    RankGraphV1 rankGraph;

    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;

    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_RESERVED;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    graphInfo.endPoints.push_back(endpoint);

    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_RESERVED;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;

    const EndpointDesc* result = rankGraph.MatchEndpointByAddr(graphInfo, &queryEndpoint);
    EXPECT_EQ(result, nullptr);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointNum_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    uint32_t num = 0;
    HcclResult ret = rankGraph.GetEndpointNum(5, 0, &num);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointNum_When_RankIndexEmpty_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t num = 0;
    HcclResult ret = rankGraph.GetEndpointNum(0, 0, &num);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointDesc_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    uint32_t descNum = 0;
    EndpointDesc endpointDesc;
    HcclResult ret = rankGraph.GetEndpointDesc(5, 0, &descNum, &endpointDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointDesc_When_NullPtr_Expect_EPara)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t descNum = 1;
    EndpointDesc endpointDesc;
    HcclResult ret = rankGraph.GetEndpointDesc(0, 0, nullptr, &endpointDesc);
    EXPECT_EQ(ret, HCCL_E_PARA);
    ret = rankGraph.GetEndpointDesc(0, 0, &descNum, nullptr);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointDesc_When_RankIndexEmpty_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t descNum = 1;
    EndpointDesc endpointDesc;
    HcclResult ret = rankGraph.GetEndpointDesc(0, 0, &descNum, &endpointDesc);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstSizeByNetLayer_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    uint32_t rankNum = 0;
    HcclResult ret = rankGraph.GetInstSizeByNetLayer(5, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstSizeByNetLayer_When_RankListNotFound_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t rankNum = 0;
    HcclResult ret = rankGraph.GetInstSizeByNetLayer(0, &rankNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstRanksByNetLayer_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    uint32_t* rankList = nullptr;
    uint32_t rankNum = 0;
    HcclResult ret = rankGraph.GetInstRanksByNetLayer(5, &rankList, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstRanksByNetLayer_When_RankListNotFound_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t* rankList = nullptr;
    uint32_t rankNum = 0;
    HcclResult ret = rankGraph.GetInstRanksByNetLayer(0, &rankList, &rankNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstSizeListByNetLayer_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    uint32_t* instSizeList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = rankGraph.GetInstSizeListByNetLayer(5, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstSizeListByNetLayer_When_RankSizeListNotFound_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    uint32_t* instSizeList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = rankGraph.GetInstSizeListByNetLayer(0, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetLinks_When_SrcRankNotFound_Expect_EPara)
{
    RankGraphV1 rankGraph;
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = rankGraph.GetLinks(0, 999, 0, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetLinks_When_NetLayerTooLarge_Expect_EPara)
{
    RankGraphV1 rankGraph;
    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;
    rankGraph.rankIndex_[0] = graphInfo;
    graphInfo.rankInfo.rankId = 1;
    rankGraph.rankIndex_[1] = graphInfo;
    CommLink* linkList = nullptr;
    uint32_t listSize = 0;
    HcclResult ret = rankGraph.GetLinks(3, 0, 1, &linkList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointInfo_When_NullPtr_Expect_EPtr)
{
    RankGraphV1 rankGraph;
    EndpointDesc queryEndpoint;
    HcclResult ret
        = rankGraph.GetEndpointInfo(0, nullptr, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), &queryEndpoint);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = rankGraph.GetEndpointInfo(0, &queryEndpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), nullptr);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointInfo_When_RankIndexEmpty_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    EndpointDesc queryEndpoint;
    EndpointAttrBwCoeff bwCoeff = 0;
    HcclResult ret
        = rankGraph.GetEndpointInfo(0, &queryEndpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), &bwCoeff);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetEndpointInfo_When_RankNotFound_Expect_ENotFound)
{
    RankGraphV1 rankGraph;
    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;
    rankGraph.rankIndex_[0] = graphInfo;
    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    EndpointAttrBwCoeff bwCoeff = 0;
    HcclResult ret
        = rankGraph.GetEndpointInfo(999, &queryEndpoint, ENDPOINT_ATTR_BW_COEFF, sizeof(EndpointAttrBwCoeff), &bwCoeff);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_InvalidAttr_Expect_EPara)
{
    RankGraphV1 rankGraph;
    EndpointDesc foundEndpoint;
    foundEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    foundEndpoint.loc.device.superDevId = 0;
    u32 invalidInfo = 0;
    HcclResult ret = rankGraph.FillAttr(static_cast<EndpointAttr>(999), &foundEndpoint, sizeof(u32), &invalidInfo);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_SizeMismatchBwCoeff_Expect_EPara)
{
    RankGraphV1 rankGraph;
    EndpointDesc foundEndpoint;
    u32 wrongSize = 999;
    EndpointAttrBwCoeff bwCoeff = 0;
    HcclResult ret = rankGraph.FillAttr(ENDPOINT_ATTR_BW_COEFF, &foundEndpoint, wrongSize, &bwCoeff);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_SizeMismatchDieId_Expect_EPara)
{
    RankGraphV1 rankGraph;
    EndpointDesc foundEndpoint;
    foundEndpoint.loc.device.superDevId = 0;
    u32 wrongSize = 999;
    EndpointAttrDieId dieId = 0;
    HcclResult ret = rankGraph.FillAttr(ENDPOINT_ATTR_DIE_ID, &foundEndpoint, wrongSize, &dieId);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_FillAttr_When_SizeMismatchLocation_Expect_EPara)
{
    RankGraphV1 rankGraph;
    EndpointDesc foundEndpoint;
    foundEndpoint.loc.locType = EndpointLocType::ENDPOINT_LOC_TYPE_DEVICE;
    u32 wrongSize = 999;
    EndpointAttrLocation locType = 0;
    HcclResult ret = rankGraph.FillAttr(ENDPOINT_ATTR_LOCATION, &foundEndpoint, wrongSize, &locType);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetHeterogMode_Expect_Success)
{
    RankGraphV1 rankGraph;
    rankGraph.heterogMode_ = HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS;
    HcclHeterogMode mode = HcclHeterogMode::HCCL_HETEROG_MODE_INVALID;
    HcclResult ret = rankGraph.GetHeterogMode(&mode);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(mode, HcclHeterogMode::HCCL_HETEROG_MODE_HOMOGENEOUS);
}

TEST_F(RankGraphV1DirectTest, Ut_GetNetLayers_When_Empty_Expect_EInternal)
{
    RankGraphV1 rankGraph;
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = rankGraph.GetNetLayers(&netLayers, &netLayerNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(RankGraphV1DirectTest, Ut_GetNetLayers_When_Valid_Expect_Success)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.netLayer_.push_back(1);
    uint32_t* netLayers = nullptr;
    uint32_t netLayerNum = 0;
    HcclResult ret = rankGraph.GetNetLayers(&netLayers, &netLayerNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netLayerNum, 2u);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstTopoTypeByNetLayer_When_InvalidNetLayer_Expect_EPara)
{
    RankGraphV1 rankGraph;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;
    HcclResult ret = rankGraph.GetInstTopoTypeByNetLayer(5, &topoType);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstTopoTypeByNetLayer_When_910B_L0_Expect_1DMesh)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;
    HcclResult ret = rankGraph.GetInstTopoTypeByNetLayer(0, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, CommTopo::COMM_TOPO_1DMESH);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstTopoTypeByNetLayer_When_910B_L1_Expect_CLOS)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.netLayer_.push_back(1);
    rankGraph.topoAttr_.deviceType = DevType::DEV_TYPE_910B;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;
    HcclResult ret = rankGraph.GetInstTopoTypeByNetLayer(1, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, CommTopo::COMM_TOPO_CLOS);
}

TEST_F(RankGraphV1DirectTest, Ut_GetInstTopoTypeByNetLayer_When_310P3_L0_Expect_310P)
{
    RankGraphV1 rankGraph;
    rankGraph.netLayer_.push_back(0);
    rankGraph.topoAttr_.deviceType = DevType::DEV_TYPE_310P3;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;
    HcclResult ret = rankGraph.GetInstTopoTypeByNetLayer(0, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, CommTopo::COMM_TOPO_310P);
}

TEST_F(RankGraphV1DirectTest, Ut_GetRankGraphInfo_When_InvalidType_Expect_ENotSupport)
{
    RankGraphV1 rankGraph;
    void* graph = nullptr;
    uint32_t len = 0;
    HcclResult ret = rankGraph.GetRankGraphInfo(static_cast<GraphType>(999), &graph, &len);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(RankGraphV1DirectTest, Ut_NeedIgnoreEndPoints_When_DiffProtocol_Expect_True)
{
    RankGraphV1 rankGraph;
    EXPECT_TRUE(rankGraph.NeedIgnoreEndPoints(
        CommProtocol::COMM_PROTOCOL_ROCE, CommProtocol::COMM_PROTOCOL_HCCS, CommProtocol::COMM_PROTOCOL_ROCE));
}

TEST_F(RankGraphV1DirectTest, Ut_NeedIgnoreEndPoints_When_SameHccsAndSioLink_Expect_False)
{
    RankGraphV1 rankGraph;
    EXPECT_FALSE(rankGraph.NeedIgnoreEndPoints(
        CommProtocol::COMM_PROTOCOL_HCCS, CommProtocol::COMM_PROTOCOL_HCCS, CommProtocol::COMM_PROTOCOL_SIO));
}

TEST_F(RankGraphV1DirectTest, Ut_NeedIgnoreEndPoints_When_SameProtocolMatchLink_Expect_False)
{
    RankGraphV1 rankGraph;
    EXPECT_FALSE(rankGraph.NeedIgnoreEndPoints(
        CommProtocol::COMM_PROTOCOL_ROCE, CommProtocol::COMM_PROTOCOL_ROCE, CommProtocol::COMM_PROTOCOL_ROCE));
}

TEST_F(RankGraphV1DirectTest, Ut_MatchEndpointByAddr_When_NoMatch_Expect_Nullptr)
{
    RankGraphV1 rankGraph;
    RankGraphV1::RankGraphInfo graphInfo;
    graphInfo.rankInfo.rankId = 0;
    EndpointDesc endpoint;
    endpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    endpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&endpoint.commAddr.addr, 0xAA, sizeof(endpoint.commAddr.addr));
    graphInfo.endPoints.push_back(endpoint);
    EndpointDesc queryEndpoint;
    queryEndpoint.commAddr.type = COMM_ADDR_TYPE_IP_V4;
    queryEndpoint.protocol = CommProtocol::COMM_PROTOCOL_ROCE;
    memset(&queryEndpoint.commAddr.addr, 0xBB, sizeof(queryEndpoint.commAddr.addr));
    const EndpointDesc* result = rankGraph.MatchEndpointByAddr(graphInfo, &queryEndpoint);
    EXPECT_EQ(result, nullptr);
}
