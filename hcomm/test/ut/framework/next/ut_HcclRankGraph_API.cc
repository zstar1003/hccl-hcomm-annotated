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
#include "hccl_comm_pub.h"
#include "llt_hccl_stub_rank_graph.h"

class HcclRankGraphTest : public BaseInit {
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

TEST_F(HcclRankGraphTest, Ut_HcclGetRankSize_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankSize = 0;
    ret = HcclGetRankSize(comm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankSize, 2);
}

TEST_F(HcclRankGraphTest, Ut_HcclGetRankSize_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankSize = 0;
    ret = HcclGetRankSize(nullptr, &rankSize);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLayers_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t netLayerNum = 0;
    uint32_t* netLayers;
    ret = HcclRankGraphGetLayers(comm, &netLayers, &netLayerNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(netLayerNum, 1);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLayers_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t netLayerNum = 0;
    uint32_t* netLayers;
    ret = HcclRankGraphGetLayers(nullptr, &netLayers, &netLayerNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetLayers(comm, nullptr, &netLayerNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRankSizeByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankNum = 0;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetRankSizeByLayer(comm, netLayer, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankNum, 2);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRankSizeByLayer_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankNum = 0;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetRankSizeByLayer(nullptr, netLayer, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetRankSizeByLayer(comm, 10, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankNum = 0;
    uint32_t* ranks;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetRanksByLayer(comm, netLayer, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankNum, 2);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByLayer_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t rankNum = 0;
    uint32_t netLayer = 0;
    uint32_t* ranks;
    ret = HcclRankGraphGetRanksByLayer(nullptr, netLayer, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetRanksByLayer(comm, netLayer, nullptr, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetRanksByLayer(comm, 10, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoTypeByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    CommTopo type;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetTopoTypeByLayer(comm, netLayer, &type);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(CommTopo::COMM_TOPO_CUSTOM, type);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoTypeByLayer_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    CommTopo type;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetTopoTypeByLayer(nullptr, netLayer, &type);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetTopoTypeByLayer(comm, 10, &type);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetInstSizeListByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t netLayer = 0;
    uint32_t listSize = 0;
    uint32_t* instSizeList;
    ret = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(listSize, 1);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetInstSizeListByLayer_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t netLayer = 0;
    uint32_t listSize = 0;
    uint32_t* instSizeList;
    ret = HcclRankGraphGetInstSizeListByLayer(nullptr, netLayer, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetInstSizeListByLayer(comm, netLayer, nullptr, &listSize);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetInstSizeListByLayer(comm, 10, &instSizeList, &listSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLinks_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t linkNum = 0;
    uint32_t netLayer = 0;
    uint32_t srcRank = 0;
    uint32_t dstRank = 1;
    CommLink* links;
    ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, &links, &linkNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(linkNum, 1);
    EXPECT_EQ(links[0].linkAttr.hop, 1);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetLinks_When_Param_Is_InVaild_Expect_Return_Error)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t linkNum = 0;
    uint32_t netLayer = 0;
    uint32_t srcRank = 0;
    uint32_t dstRank = 1;
    CommLink* links;
    ret = HcclRankGraphGetLinks(nullptr, netLayer, srcRank, dstRank, &links, &linkNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetLinks(comm, netLayer, srcRank, dstRank, nullptr, &linkNum);
    EXPECT_EQ(ret, HCCL_E_PTR);
    ret = HcclRankGraphGetLinks(comm, 10, srcRank, dstRank, &links, &linkNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoInstsByLayer_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);

    uint32_t netLayer = 0;
    uint32_t* topoInsts;
    uint32_t topoInstNum;
    ret = HcclRankGraphGetTopoInstsByLayer(comm, netLayer, &topoInsts, &topoInstNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoInstNum, 1);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoType_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    CommTopo topoType;
    ret = HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(topoType, CommTopo::COMM_TOPO_1DMESH);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetTopoType_When_TopoInstIdNotExist_Expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 1;
    CommTopo topoType = CommTopo::COMM_TOPO_CLOS;
    ret = HcclRankGraphGetTopoType(comm, netLayer, topoInstId, &topoType);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByTopoInst_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 0;
    uint32_t rankNum;
    uint32_t* ranks;
    ret = HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rankNum, 2);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetRanksByTopoInst_When_TopoInstIdNotExist_Expect_Return_HCCL_E_PARA)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);

    uint32_t netLayer = 0;
    uint32_t topoInstId = 1;
    uint32_t rankNum = 0;
    uint32_t* ranks = nullptr;
    ret = HcclRankGraphGetRanksByTopoInst(comm, netLayer, topoInstId, &ranks, &rankNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcclRankGraphTest, Ut_HcclRankGraphGetEndpointInfo_When_ValidParam_Expect_Return_HCCL_SUCCESS)
{
    std::shared_ptr<hccl::hcclComm> hcclCommPtr;
    std::shared_ptr<Hccl::RankGraph> rankGraphV2;
    void* comm;
    HcclResult ret;
    SetUpCommAndGraph(hcclCommPtr, rankGraphV2, comm, ret);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    uint32_t num = 0;
    uint32_t topoInstId = 0;
    uint32_t netLayer = 0;
    ret = HcclRankGraphGetEndpointNum(comm, netLayer, topoInstId, &num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(num, 1);

    uint32_t descNum = num;
    std::unique_ptr<EndpointDesc[]> endpointDesc(new EndpointDesc[descNum]);
    ret = HcclRankGraphGetEndpointDesc(comm, netLayer, topoInstId, &num, endpointDesc.get());
    EXPECT_EQ(ret, HCCL_SUCCESS);
    for (uint32_t i = 0; i < num; i++) {
        EXPECT_EQ(endpointDesc[i].protocol, COMM_PROTOCOL_UB_CTP);
        uint32_t infoLen = sizeof(EndpointAttrBwCoeff);
        EndpointAttrBwCoeff bwCoeff{};
        ret = HcclRankGraphGetEndpointInfo(comm, 0, &endpointDesc[i], ENDPOINT_ATTR_BW_COEFF, infoLen, &bwCoeff);
        EXPECT_EQ(ret, HCCL_SUCCESS);
    }
}
