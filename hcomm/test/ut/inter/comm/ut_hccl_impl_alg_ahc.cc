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

#include <string>

#define private public
#define protected public
#include "coll_all_reduce_comm_executor.h"
#include "coll_all_reduce_mesh_executor.h"
#include "hccl_alg.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#include "hccl_comm_pub.h"
#include "comm_impl.h"
#include "broadcast_operator.h"
#include "all_reduce_operator.h"
#include "reduce_operator.h"
#include "reduce_scatter_operator.h"
#include "all_gather_operator.h"
#include "coll_comm_executor.h"
#include "coll_all_gather_comm_executor.h"
#include "coll_reduce_scatter_comm_executor.h"
#include "coll_alg_param.h"
#include "coll_reduce_scatter_executor.h"
#include "coll_reduce_scatter_ring_for_910_93_executor.h"
#include "coll_reduce_scatter_fast_double_ring_for_910_93_executor.h"
#include "dispatcher_pub.h"
#include "dispatcher.h"
#include "externalinput.h"
#include "ffts_common_pub.h"
#include "reduce_scatter_ring_pub.h"
#include "all_reduce_ring_pub.h"
#include "all_gather_ring_pub.h"
#include "transport_base_pub.h"
#include "heartbeat.h"
#undef private
#undef protected
#include "dlra_function.h"
#include "llt_hccl_stub_sal_pub.h"
#include "dispatcher_pub.h"
#include "reduce_scatter_nb_pub.h"
#include "all_gather_nb_pub.h"
#include "all_reduce_nb_pub.h"
#include "adapter_prof.h"

using namespace hccl;
using namespace std;

enum AHCCommType {
    COMM_SYM,
    COMM_COPRIME,
    COMM_SPILT,
};

enum AHCEnvType {
    AHC,
    AHC_BROKE,
    DEFAULT,
};

class HcclImplAlgTestAHCAllreduce : public testing::Test {
public:
    static HcclDispatcher dispatcherPtr;
    static DispatcherPub* dispatcher;

protected:
    static void SetUpTestCase()
    {
        s32 ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
        if (ret != HCCL_SUCCESS)
            return;
        if (dispatcherPtr == nullptr)
            return;
        dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
        DlRaFunction::GetInstance().DlRaFunctionInit();
        std::cout << "HcclImplAlgTestAHCAllreduce SetUP" << std::endl;
    }
    static void TearDownTestCase()
    {
        if (dispatcherPtr != nullptr) {
            s32 ret = HcclDispatcherDestroy(dispatcherPtr);
            EXPECT_EQ(ret, HCCL_SUCCESS);
            dispatcherPtr = nullptr;
            dispatcher = nullptr;
        }
        std::cout << "HcclImplAlgTestAHCAllreduce TearDown" << std::endl;
    }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));
        MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};
HcclDispatcher HcclImplAlgTestAHCAllreduce::dispatcherPtr = nullptr;
DispatcherPub* HcclImplAlgTestAHCAllreduce::dispatcher = nullptr;

HcclResult ParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType);

HcclResult FakeParserHcclAlgoLevel(const std::string& algoLevel, u32& level, HcclAlgoType& algoType)
{
    std::size_t found = algoLevel.find(":");
    if ((found == 0) || (found == (algoLevel.length() - 1))) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid.");
        return HCCL_E_PARA;
    }

    std::string orginalLevel = algoLevel.substr(0, found);
    std::string orginalAlgo = algoLevel.substr(found + 1);

    const std::map<std::string, u32> hcclAlgoLevelMap
        = {{"level0", HCCL_ALGO_LEVEL_0},
           {"level1", HCCL_ALGO_LEVEL_1},
           {"level2", HCCL_ALGO_LEVEL_2},
           {"level3", HCCL_ALGO_LEVEL_3}};

    const std::map<std::string, HcclAlgoType> hcclAlgoTypeMap = {
        {"null", HcclAlgoType::HCCL_ALGO_TYPE_NULL},
        {"ring", HcclAlgoType::HCCL_ALGO_TYPE_RING},
        {"pipeline", HcclAlgoType::HCCL_ALGO_TYPE_PIPELINE},
        {"fullmesh", HcclAlgoType::HCCL_ALGO_TYPE_FULLMESH},
        {"H-D_R", HcclAlgoType::HCCL_ALGO_TYPE_HDR},
        {"pairwise", HcclAlgoType::HCCL_ALGO_TYPE_PAIRWISE},
        {"NHR", HcclAlgoType::HCCL_ALGO_TYPE_NHR},
        {"NHR_V1", HcclAlgoType::HCCL_ALGO_TYPE_NHR_V1},
        {"AHC", HcclAlgoType::HCCL_ALGO_TYPE_AHC},
        {"AHC_BROKE", HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE},
        {"NB", HcclAlgoType::HCCL_ALGO_TYPE_NB},
        {"NA", HcclAlgoType::HCCL_ALGO_TYPE_NA},
    };

    auto iterAlgoLevel = hcclAlgoLevelMap.find(orginalLevel);
    if (iterAlgoLevel == hcclAlgoLevelMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, level %s is not supported.", orginalLevel.c_str());
        return HCCL_E_PARA;
    }

    auto iterAlgoType = hcclAlgoTypeMap.find(orginalAlgo);
    if (iterAlgoType == hcclAlgoTypeMap.end()) {
        HCCL_ERROR("[Parser][HcclAlgoLevel] algo config is invalid, algo %s is not supported.", orginalAlgo.c_str());
        return HCCL_E_PARA;
    }

    level = iterAlgoLevel->second;
    algoType = iterAlgoType->second;

    return HCCL_SUCCESS;
}

HcclResult FakeParseAlgoString(std::string opName, std::string& algoString, std::vector<HcclAlgoType>& algType)
{
    algType = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
    std::vector<std::string> algoLevels;
    HcclResult ret = SplitHcclAlgoLevel(algoString, algoLevels);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
            "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
            algoString.c_str()),
        ret);
    for (auto algoLevel : algoLevels) {
        u32 level = 0;
        HcclAlgoType algo = HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT;
        ret = FakeParserHcclAlgoLevel(algoLevel, level, algo);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str()),
            ret);
        // 检查是否存在重复配置level
        if (algType[level] != HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) {
            HCCL_ERROR(
                "[Set][HcclAlgoConfig]hccl algo config[%s] is invalid. "
                "expect: level0:NA;level1:<algo> or <op0>=level0:NA;level1:<algo0>/<op1>=level0:NA;level1:<algo1>",
                algoString.c_str());
            return HCCL_E_PARA;
        }
        algType[level] = algo;
    }
    auto level0Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_0]);
    auto level1Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_1]);
    auto level2Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_2]);
    auto level3Iter = HcclAlgoTypeMap.find(algType[HCCL_ALGO_LEVEL_3]);
    HCCL_RUN_INFO(
        "hccl algo op %s config: config level0:%s, level1:%s, level2:%s, level3:%s", opName.c_str(),
        level0Iter->second.c_str(), level1Iter->second.c_str(), level2Iter->second.c_str(), level3Iter->second.c_str());
    return HCCL_SUCCESS;
}

HcclResult FakeRunTemplateCoprimeCase(const std::unique_ptr<AlgTemplateBase>& tempAlg, const SubCommInfo& commInfo)
{
    u32 localRank = 0;
    u32 localRankSize = 5;
    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    std::vector<std::shared_ptr<Transport>> links;
    links.resize(localRankSize);

    for (int i = 0; i < localRankSize; i++) {
        links[i].reset(new (std::nothrow) Transport(
            new (std::nothrow) TransportBase(HcclImplAlgTestAHCAllreduce::dispatcher, nullptr, machinePara, timeout)));
        links[i]->Init();
    }

    HcclResult ret = tempAlg->RunAsync(localRank, localRankSize, links);
    return ret;
}

HcclResult FakeRunTemplateSpiltCase(const std::unique_ptr<AlgTemplateBase>& tempAlg, const SubCommInfo& commInfo)
{
    u32 localRank = 0;
    u32 localRankSize = 8;
    MachinePara machinePara;
    std::chrono::milliseconds timeout;
    std::vector<std::shared_ptr<Transport>> links;
    links.resize(localRankSize);

    for (int i = 0; i < localRankSize; i++) {
        links[i].reset(new (std::nothrow) Transport(
            new (std::nothrow) TransportBase(HcclImplAlgTestAHCAllreduce::dispatcher, nullptr, machinePara, timeout)));
        links[i]->Init();
    }

    HcclResult ret = tempAlg->RunAsync(localRank, localRankSize, links);
    return ret;
}

// 对称 2 通信域，每通信域2卡构造
static void TestConstructParamSymComm(HcclCommParams& params, RankTable_t& rankTable, DevType deviceType)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 4;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = deviceType;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(4);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.0.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";

    if (deviceType == DevType::DEV_TYPE_910B) {
        rankVec[2].rankId = 2;
        rankVec[2].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr2(1694542016);
        rankVec[2].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
        rankVec[2].serverIdx = 0;
        rankVec[2].serverId = "192.168.0.101";

        rankVec[3].rankId = 3;
        rankVec[3].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr3(1711319232);
        rankVec[3].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
        rankVec[3].serverIdx = 1;
        rankVec[3].serverId = "192.168.0.102";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 4;
        rankTable.serverNum = 2;
    } else {
        rankVec[2].rankId = 2;
        rankVec[2].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr2(1728096448);
        rankVec[2].deviceInfo.deviceIp.push_back(ipAddr2); // 101.0.168.192
        rankVec[2].serverIdx = 2;
        rankVec[2].serverId = "192.168.0.103";

        rankVec[3].rankId = 3;
        rankVec[3].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr3(1744873664);
        rankVec[3].deviceInfo.deviceIp.push_back(ipAddr3); // 101.0.168.192
        rankVec[3].serverIdx = 3;
        rankVec[3].serverId = "192.168.0.104";

        rankVec[0].superPodId = "192.168.0.105";
        rankVec[1].superPodId = "192.168.0.105";
        rankVec[2].superPodId = "192.168.0.106";
        rankVec[3].superPodId = "192.168.0.106";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 4;
        rankTable.serverNum = 4;
    }
}

// 非对称 2 通信域，2、3卡构造
static void TestConstructParamCoprimeComm(HcclCommParams& params, RankTable_t& rankTable, DevType deviceType)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 5;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = deviceType;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(5);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";

    if (deviceType == DevType::DEV_TYPE_910B) {
        rankVec[2].rankId = 2;
        rankVec[2].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr3(1694542016);
        rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
        rankVec[2].serverIdx = 0;
        rankVec[2].serverId = "192.168.0.101";

        rankVec[3].rankId = 3;
        rankVec[3].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr4(1711319232);
        rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
        rankVec[3].serverIdx = 1;
        rankVec[3].serverId = "192.168.0.102";

        rankVec[4].rankId = 4;
        rankVec[4].deviceInfo.devicePhyId = 2;
        HcclIpAddress ipAddr5(1711319232);
        rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
        rankVec[4].serverIdx = 1;
        rankVec[4].serverId = "192.168.0.102";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 5;
        rankTable.serverNum = 2;
    } else {
        rankVec[2].rankId = 2;
        rankVec[2].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr3(1728096448);
        rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
        rankVec[2].serverIdx = 2;
        rankVec[2].serverId = "192.168.0.103";

        rankVec[3].rankId = 3;
        rankVec[3].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr4(1744873664);
        rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
        rankVec[3].serverIdx = 3;
        rankVec[3].serverId = "192.168.0.104";

        rankVec[4].rankId = 4;
        rankVec[4].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr5(1761650880);
        rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
        rankVec[4].serverIdx = 4;
        rankVec[4].serverId = "192.168.0.105";

        rankVec[0].superPodId = "192.168.0.106";
        rankVec[1].superPodId = "192.168.0.106";
        rankVec[2].superPodId = "192.168.0.107";
        rankVec[3].superPodId = "192.168.0.107";
        rankVec[4].superPodId = "192.168.0.107";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 5;
        rankTable.serverNum = 5;
    }

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1728096448);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
    rankVec[2].serverIdx = 2;
    rankVec[2].serverId = "192.168.0.103";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr4(1744873664);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
    rankVec[3].serverIdx = 3;
    rankVec[3].serverId = "192.168.0.104";

    rankVec[4].rankId = 4;
    rankVec[4].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr5(1761650880);
    rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
    rankVec[4].serverIdx = 4;
    rankVec[4].serverId = "192.168.0.105";

    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.rankNum = 5;
    rankTable.deviceNum = 5;
    rankTable.serverNum = 5;
}

// 非对称 可拆分 4 通信域，1、1、2、4卡构造
static void TestConstructParamSpiltComm(HcclCommParams& params, RankTable_t& rankTable, DevType deviceType)
{
    string commId = "comm ";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 8;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.commWorkMode = WorkMode::HCCL_MODE_NORMAL;
    params.deviceType = deviceType;

    rankTable.collectiveId = "192.168.0.101-8000-8001";
    vector<RankInfo_t> rankVec(8);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1694542016);
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1);
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";

    rankVec[1].rankId = 1;
    rankVec[1].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr2(1711319232);
    rankVec[1].deviceInfo.deviceIp.push_back(ipAddr2);
    rankVec[1].serverIdx = 1;
    rankVec[1].serverId = "192.168.0.102";

    rankVec[2].rankId = 2;
    rankVec[2].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr3(1728096448);
    rankVec[2].deviceInfo.deviceIp.push_back(ipAddr3);
    rankVec[2].serverIdx = 2;
    rankVec[2].serverId = "192.168.0.103";

    rankVec[3].rankId = 3;
    rankVec[3].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr4(1744873664);
    rankVec[3].deviceInfo.deviceIp.push_back(ipAddr4);
    rankVec[3].serverIdx = 3;
    rankVec[3].serverId = "192.168.0.104";

    if (deviceType == DevType::DEV_TYPE_910B) {
        rankVec[4].rankId = 4;
        rankVec[4].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr5(1728096448);
        rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
        rankVec[4].serverIdx = 2;
        rankVec[4].serverId = "192.168.0.103";

        rankVec[5].rankId = 5;
        rankVec[5].deviceInfo.devicePhyId = 1;
        HcclIpAddress ipAddr6(1744873664);
        rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6);
        rankVec[5].serverIdx = 3;
        rankVec[5].serverId = "192.168.0.104";

        rankVec[6].rankId = 6;
        rankVec[6].deviceInfo.devicePhyId = 2;
        HcclIpAddress ipAddr7(1744873664);
        rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7);
        rankVec[6].serverIdx = 3;
        rankVec[6].serverId = "192.168.0.104";

        rankVec[7].rankId = 7;
        rankVec[7].deviceInfo.devicePhyId = 3;
        HcclIpAddress ipAddr8(1744873664);
        rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8);
        rankVec[7].serverIdx = 3;
        rankVec[7].serverId = "192.168.0.104";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 8;
        rankTable.serverNum = 4;
    } else {
        rankVec[4].rankId = 4;
        rankVec[4].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr5(1761650880);
        rankVec[4].deviceInfo.deviceIp.push_back(ipAddr5);
        rankVec[4].serverIdx = 4;
        rankVec[4].serverId = "192.168.0.105";

        rankVec[5].rankId = 5;
        rankVec[5].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr6(1778428096);
        rankVec[5].deviceInfo.deviceIp.push_back(ipAddr6);
        rankVec[5].serverIdx = 5;
        rankVec[5].serverId = "192.168.0.106";

        rankVec[6].rankId = 6;
        rankVec[6].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr7(1795205312);
        rankVec[6].deviceInfo.deviceIp.push_back(ipAddr7);
        rankVec[6].serverIdx = 6;
        rankVec[6].serverId = "192.168.0.107";

        rankVec[7].rankId = 7;
        rankVec[7].deviceInfo.devicePhyId = 0;
        HcclIpAddress ipAddr8(1811982528);
        rankVec[7].deviceInfo.deviceIp.push_back(ipAddr8);
        rankVec[7].serverIdx = 7;
        rankVec[7].serverId = "192.168.0.108";

        rankVec[0].superPodId = "192.168.0.109";
        rankVec[1].superPodId = "192.168.0.110";
        rankVec[2].superPodId = "192.168.0.111";
        rankVec[3].superPodId = "192.168.0.111";
        rankVec[4].superPodId = "192.168.0.112";
        rankVec[5].superPodId = "192.168.0.112";
        rankVec[6].superPodId = "192.168.0.112";
        rankVec[7].superPodId = "192.168.0.112";

        rankTable.rankList.assign(rankVec.begin(), rankVec.end());
        rankTable.deviceNum = 8;
        rankTable.serverNum = 8;
    }
}

static inline void ConstructCommTestCase910B(AHCEnvType ahcEnvType, AHCCommType ahcCommType)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    if (ahcEnvType == AHCEnvType::AHC) {
        setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);
    } else if (ahcEnvType == AHCEnvType::AHC_BROKE) {
        setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
        MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));
    } else {
        unsetenv("HCCL_ALGO");
    }

    ResetInitState();
    InitExternalInput();

    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    if (ahcCommType == AHCCommType::COMM_SYM) {
        TestConstructParamSymComm(params, rankTable, DevType::DEV_TYPE_910B);
    } else if (ahcCommType == AHCCommType::COMM_COPRIME) {
        TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    } else if (ahcCommType == AHCCommType::COMM_SPILT) {
        TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    }
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllReduceCommExecutor* executor = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    if (ahcEnvType == AHCEnvType::AHC) {
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
        executor->SetAlgType(algType);
    } else if (ahcEnvType == AHCEnvType::AHC_BROKE) {
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
        executor->SetAlgType(algType);
    } else {
        algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_WHOLE_RING;
        algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_WHOLE_RING;
        executor->SetAlgType(algType);
    }

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 4096;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    if (ahcEnvType != AHCEnvType::DEFAULT) {
        unsetenv("HCCL_ALGO");
    }
    ResetInitState();
    InitExternalInput();
}

static inline void ConstructCommTestCase91093(AHCEnvType ahcEnvType, AHCCommType ahcCommType)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    if (ahcEnvType == AHCEnvType::AHC) {
        setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);
    } else if (ahcEnvType == AHCEnvType::AHC_BROKE) {
        setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);
        MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));
    }
    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    if (ahcCommType == AHCCommType::COMM_SYM) {
        TestConstructParamSymComm(params, rankTable, DevType::DEV_TYPE_910_93);
    } else if (ahcCommType == AHCCommType::COMM_COPRIME) {
        TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910_93);
    } else if (ahcCommType == AHCCommType::COMM_SPILT) {
        TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910_93);
    }
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();

    if (ahcEnvType != AHCEnvType::DEFAULT) {
        unsetenv("HCCL_ALGO");
    }
    ResetInitState();
    InitExternalInput();
}

// 910B AHC 对称通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSymComm_AHC_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC, AHCCommType::COMM_SYM);
}

// 910B AHC-Broke 对称通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSymComm_AHCBroke_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC_BROKE, AHCCommType::COMM_SYM);
}

// 910B AHC 非对称互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceCoprimeComm_AHC_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC, AHCCommType::COMM_COPRIME);
}

// 910B AHC-Broke 非对称互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceCoprimeComm_AHCBroke_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC_BROKE, AHCCommType::COMM_COPRIME);
}

// 910B AHC 非对称可切分通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSpiltComm_AHC_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC, AHCCommType::COMM_SPILT);
}

// 910B AHC-Broke 非对称可切分互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSpiltComm_AHCBroke_910B)
{
    ConstructCommTestCase910B(AHCEnvType::AHC_BROKE, AHCCommType::COMM_SPILT);
}

// 910B 无配置算法 非对称互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceCoprimeComm_Default_910B)
{
    ConstructCommTestCase910B(AHCEnvType::DEFAULT, AHCCommType::COMM_COPRIME);
}

// 910B 无配置算法 非对称可切分通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSpiltComm_Default_910B)
{
    ConstructCommTestCase910B(AHCEnvType::DEFAULT, AHCCommType::COMM_SPILT);
}

// 91093 AHC 对称通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSymComm_AHC_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC, AHCCommType::COMM_SYM);
}

// 91093 AHC-Broke 对称通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSymComm_AHCBroke_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC_BROKE, AHCCommType::COMM_SYM);
}

// 91093 AHC 非对称互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceCoprimeComm_AHC_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC, AHCCommType::COMM_COPRIME);
}

// 91093 AHC-Broke 非对称互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceCoprimeComm_AHCBroke_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC_BROKE, AHCCommType::COMM_COPRIME);
}

// 91093 AHC 非对称可切分通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSpiltComm_AHC_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC, AHCCommType::COMM_SPILT);
}

// 91093 AHC-Broke 非对称可切分互素通信域初始化
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceSpiltComm_AHCBroke_91093)
{
    ConstructCommTestCase91093(AHCEnvType::AHC_BROKE, AHCCommType::COMM_SPILT);
}

// 910B AHC 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceAHCExecute910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);

    ResetInitState();
    InitExternalInput();

    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllReduceCommExecutor* executor = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 4096;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&AlgTemplateBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllReduceRing allreduceRing(dispatcher);
    allreduceRing.Prepare(reduceAttrBitMap);
    ReduceScatterRing reducescatterRing(dispatcher);
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceRing, &AllReduceRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllReduceNB allreduceNB(dispatcher);
    allreduceNB.Prepare(reduceAttrBitMap);
    ReduceScatterNB reducescatterNB(dispatcher);
    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceNB, &AllReduceNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC-Broke 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceAHCBrokeExecute910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    InitExternalInput();

    ret = InitEnvVarParam();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllReduceCommExecutor* executor = new CollAllReduceCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 1024;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 1024;
    opParam.DataDes.count = 4096;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&AlgTemplateBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllReduceRing allreduceRing(dispatcher);
    allreduceRing.Prepare(reduceAttrBitMap);
    ReduceScatterRing reducescatterRing(dispatcher);
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceRing, &AllReduceRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllReduceNB allreduceNB(dispatcher);
    allreduceNB.Prepare(reduceAttrBitMap);
    ReduceScatterNB reducescatterNB(dispatcher);
    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceNB, &AllReduceNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLREDUCE, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 91093 AHC 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceAHCExecute91093)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);
    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910_93);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&AlgTemplateBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllReduceRing allreduceRing(dispatcher);
    allreduceRing.Prepare(reduceAttrBitMap);
    ReduceScatterRing reducescatterRing(dispatcher);
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceRing, &AllReduceRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllReduceNB allreduceNB(dispatcher);
    allreduceNB.Prepare(reduceAttrBitMap);
    ReduceScatterNB reducescatterNB(dispatcher);
    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceNB, &AllReduceNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 91093 AHC-Broke 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceAHCBrokeExecute91093)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910_93);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_DOUBLE_RING;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_DOUBLE_RING;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&AlgTemplateBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllReduceRing allreduceRing(dispatcher);
    allreduceRing.Prepare(reduceAttrBitMap);
    ReduceScatterRing reducescatterRing(dispatcher);
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceRing, &AllReduceRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllReduceNB allreduceNB(dispatcher);
    allreduceNB.Prepare(reduceAttrBitMap);
    ReduceScatterNB reducescatterNB(dispatcher);
    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceNB, &AllReduceNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 91093 null-AHC 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllReduceNULLAHCExecute91093)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);
    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910_93);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&AlgTemplateBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllReduceRing allreduceRing(dispatcher);
    allreduceRing.Prepare(reduceAttrBitMap);
    ReduceScatterRing reducescatterRing(dispatcher);
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceRing, &AllReduceRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllReduceNB allreduceNB(dispatcher);
    allreduceNB.Prepare(reduceAttrBitMap);
    ReduceScatterNB reducescatterNB(dispatcher);
    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allreduceNB, &AllReduceNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910_93 AI_CPU AHC 切换默认算法
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AICPU_AHC_Default_91093)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);
    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910_93);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(GetExternalInputHcclAicpuUnfold).stubs().will(returnValue(true));

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    u32 moduleNum = 8;
    AlgTypeLevel1 algType1 = AlgTypeLevel1::ALG_LEVEL1_RESERVED;
    algConfigurator->SetAlgoLevel1(
        HcclAlgoType::HCCL_ALGO_TYPE_AHC, moduleNum, algType1, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(algType1, AlgTypeLevel1::ALG_LEVEL1_AHC); // server 数为 8 以上：默认流程使用 HD 算法

    algConfigurator->SetAlgoLevel1(
        HcclAlgoType::HCCL_ALGO_TYPE_AHC_BROKE, moduleNum, algType1, HcclCMDType::HCCL_CMD_ALLREDUCE);
    EXPECT_EQ(algType1, AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE); // server 数为 8 以上：默认流程使用 HD 算法

    implBase = nullptr;

    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B 不同拓扑 AHC 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AHC_Mesh_910_93)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:NA;level1:AHC", 1);
    ResetInitState();
    InitExternalInput();

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSymComm(params, rankTable, DevType::DEV_TYPE_910_93);
    params.deviceType = DevType::DEV_TYPE_910_93;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&CollNativeExecutorBase::CheckCommSize).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AtomicInitSet();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;

    impl->topoAttr_.deviceLogicId = 0;
    impl->topoAttr_.devicePhyId = 0;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_NP_MESH;
    algConfigurator->algType_[HcclCMDType::HCCL_CMD_ALLREDUCE].algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    impl->topoType_ = TopoType::TOPO_TYPE_NP_MESH;

    const std::vector<std::vector<u32>> tmpRingNics
        = {{0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3, 4, 5, 6, 7}};

    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalRecord,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u64, s32, bool, u64, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP_VIRTUAL(
        *HcclImplAlgTestAHCAllreduce::dispatcher, &DispatcherPub::SignalWait,
        HcclResult(DispatcherPub::*)(HcclRtNotify, hccl::Stream&, u32, u32, s32, bool, u32, u32))
        .stubs()
        .will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(returnValue(HCCL_SUCCESS));

    ret = implBase->AllReduce(tag, inputMem.ptr(), outputMem.ptr(), count, dataType, op, stream.ptr());
    implBase = nullptr;

    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC All-Gather 单 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllGatherAHCExecuteSingleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllGatherCommExecutor* executor = new CollAllGatherCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC-Broke All-Gather 单 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllGatherAHCBrokeExecuteSingleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllGatherCommExecutor* executor = new CollAllGatherCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC All-Gather 双 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllGatherAHCExecuteDoubleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllGatherCommExecutor* executor = new CollAllGatherCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096 * 5);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096 * 5;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC-Broke All-Gather 双 Buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_AllGatherAHCBrokeExecuteDoubleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollAllGatherCommExecutor* executor = new CollAllGatherCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096 * 5);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096 * 5;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    // MOCKER(InitTask)
    // .stubs()
    // .will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    u64 reduceAttrBitMap;
    AllGatherRing allgatherRing(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherRing, &AllGatherRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AllGatherNB allgatherNB(dispatcher);
    MOCKER_CPP_VIRTUAL(allgatherNB, &AllGatherNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_ALLGATHER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC Reduce-Scatter 单 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_ReduceScatterAHCExecuteSingleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollReduceScatterCommExecutor* executor = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    CollReduceScatterExecutor collReduceScatterExecutor(dispatcher, topoMatcher);
    MOCKER_CPP_VIRTUAL(collReduceScatterExecutor, &CollReduceScatterExecutor::RunLoop)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    u64 reduceAttrBitMap;
    ReduceScatterRing reducescatterRing(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ReduceScatterNB reducescatterNB(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC-Broke Reduce-Scatter 单 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_ReduceScatterAHCBrokeExecuteSingleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollReduceScatterCommExecutor* executor = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));
    HcclDispatcher dispatcher;
    CollReduceScatterExecutor collReduceScatterExecutor(dispatcher, topoMatcher);
    MOCKER_CPP_VIRTUAL(collReduceScatterExecutor, &CollReduceScatterExecutor::RunLoop)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    u64 reduceAttrBitMap;
    ReduceScatterRing reducescatterRing(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ReduceScatterNB reducescatterNB(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}
// 910B AHC Reduce-Scatter 双 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_ReduceScatterAHCExecuteDoubleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC", 1);

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamCoprimeComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollReduceScatterCommExecutor* executor = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096 * 5);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096 * 5);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096 * 5;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateCoprimeCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    CollReduceScatterExecutor collReduceScatterExecutor(dispatcher, topoMatcher);
    MOCKER_CPP_VIRTUAL(collReduceScatterExecutor, &CollReduceScatterExecutor::RunLoop)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    u64 reduceAttrBitMap;
    ReduceScatterRing reducescatterRing(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ReduceScatterNB reducescatterNB(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}

// 910B AHC-Broke Reduce-Scatter 双 buffer 执行流程
TEST_F(HcclImplAlgTestAHCAllreduce, ut_ReduceScatterAHCBrokeExecuteDoubleBuffer910B)
{
    HcclResult ret = HCCL_SUCCESS;
    std::string tag = "test";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_FP32;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    Stream stream(StreamType::STREAM_TYPE_ONLINE);

    setenv("HCCL_ALGO", "level0:null;level1:AHC_BROKE", 1);
    MOCKER(ParseAlgoString).stubs().will(invoke(FakeParseAlgoString));

    ResetInitState();
    ret = InitExternalInput();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclCommParams params;
    RankTable_t rankTable;
    TestConstructParamSpiltComm(params, rankTable, DevType::DEV_TYPE_910B);
    params.deviceType = DevType::DEV_TYPE_910B;
    std::unique_ptr<HcclCommunicator> implBase(new (std::nothrow) HcclCommunicator());

    MOCKER_CPP(&HcclCommunicator::InitRaResource).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    ret = implBase->Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::unique_ptr<hcclImpl>& impl = implBase->implAlg_->pimpl_;
    std::shared_ptr<AlgConfigurator> algConfigurator = implBase->implAlg_->algConfigurator_;
    implBase->InitCCLbuffer(200 * 1024 * 1024, 200 * 1024 * 1024);

    impl->deviceLogicId_ = 0;
    impl->deviceType_ = DevType::DEV_TYPE_910B;
    impl->topoType_ = TopoType::TOPO_TYPE_COMMON;

    std::unique_ptr<TopoMatcher>& topoMatcher = implBase->implAlg_->topoMatcher_;
    topoMatcher->topoInfo_.deviceLogicId = 0;
    topoMatcher->topoInfo_.deviceType = DevType::DEV_TYPE_910B;
    topoMatcher->topoInfo_.topoType = TopoType::TOPO_TYPE_COMMON;
    CollReduceScatterCommExecutor* executor = new CollReduceScatterCommExecutor(impl->dispatcher_, topoMatcher);
    AlgType algType;
    algType.algoLevel0 = AlgTypeLevel0::ALG_LEVEL0_RESERVED;
    algType.algoLevel1 = AlgTypeLevel1::ALG_LEVEL1_AHC_BROKE;
    executor->SetAlgType(algType);

    DeviceMem inputMem = DeviceMem::alloc(4096 * 5);
    DeviceMem outputMem = DeviceMem::alloc(4096);
    DeviceMem scratchMem = DeviceMem::alloc(4096 * 5);
    OpParam opParam;
    opParam.tag = "test";
    opParam.inputPtr = inputMem.ptr();
    opParam.inputSize = 4096 * 5;
    opParam.outputPtr = outputMem.ptr();
    opParam.outputSize = 4096;
    opParam.DataDes.count = 1024;
    opParam.DataDes.dataType = HCCL_DATA_TYPE_FP32;
    opParam.reduceType = HCCL_REDUCE_SUM;
    opParam.stream = Stream(StreamType::STREAM_TYPE_ONLINE);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(LaunchTask).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(CollExecutorBase::RunTemplate).stubs().will(invoke(FakeRunTemplateSpiltCase));
    MOCKER_CPP(&ExecutorBase::RegisterProfiler).stubs().will(returnValue(HCCL_SUCCESS));

    HcclDispatcher dispatcher;
    CollReduceScatterExecutor collReduceScatterExecutor(dispatcher, topoMatcher);
    MOCKER_CPP_VIRTUAL(collReduceScatterExecutor, &CollReduceScatterExecutor::RunLoop)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    u64 reduceAttrBitMap;
    ReduceScatterRing reducescatterRing(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterRing, &ReduceScatterRing::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    ReduceScatterNB reducescatterNB(dispatcher);
    MOCKER_CPP_VIRTUAL(reducescatterNB, &ReduceScatterNB::RunAsync).stubs().will(returnValue(HCCL_SUCCESS));

    AlgResourceRequest resourceRequest;
    AlgResourceResponse resourceResponse;
    ret = executor->CalcResRequest(opParam, resourceRequest);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    implBase->AllocAlgResource(
        opParam.tag, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, opParam, resourceRequest, resourceResponse);
    resourceResponse.cclInputMem = inputMem;
    resourceResponse.cclOutputMem = outputMem;
    ret = executor->Orchestrate(opParam, resourceResponse);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    delete executor;
    GlobalMockObject::verify();

    unsetenv("HCCL_ALGO");
    ResetInitState();
    InitExternalInput();
}
