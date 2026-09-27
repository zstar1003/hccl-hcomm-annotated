/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_IMPL_PUB_H
#define HCCL_IMPL_PUB_H

#include "comm_base_pub.h"
#include "threadManage.h"

namespace hccl {
constexpr s32 PROF_RANKSIZE_OFFSET_OF_PLANEID = 16;
constexpr s32 PROF_RINGINDEX_OFFSET_OF_PLANEID = 28;

constexpr u32 PROF_STAGE_0 = 0;
constexpr u32 PROF_STAGE_1 = 1;
constexpr u32 PROF_STAGE_2 = 2;

using level1StreamInfo_t = struct Level1StreamInfo {
    u32 ringNum; /* 至少有1个ring */
    std::vector<std::shared_ptr<LocalNotify>> ringSignal;
    std::vector<std::shared_ptr<LocalNotify>> ringSignalAux;
    std::vector<Stream> ringStreams;
    std::vector<std::shared_ptr<ThreadManage>> ringThreadsManage;
    std::vector<uint32_t> tidInfo;
    std::vector<Stream> ringDeviceStreams;
    std::vector<std::shared_ptr<LocalNotify>> ringDeviceSignal;
    std::vector<std::shared_ptr<LocalNotify>> ringDeviceSignalAux;

    Level1StreamInfo() : ringNum(1) {}
};

using tagStreamInfo_t = std::map<std::string, Level1StreamInfo>;

using CommInfo = struct TagCommInfo {
    std::vector<std::unique_ptr<CommBase>> commLevel1;
    std::vector<std::unique_ptr<CommBase>> commLevel1Rdma;
    std::vector<std::unique_ptr<CommBase>> commLevel0;
    std::vector<std::unique_ptr<CommBase>> commLevel0Rdma;
    std::vector<std::unique_ptr<CommBase>> commLevel2;
    std::vector<std::unique_ptr<CommBase>> commP2P;
    std::unique_ptr<CommBase> commIntraServer;

    TagCommInfo()
        : commLevel1(0),
          commLevel1Rdma(0),
          commLevel0(0),
          commLevel0Rdma(0),
          commP2P(0),
          commIntraServer(nullptr)
    {}
};
using tagCommInfo_t = std::map<std::string, CommInfo>;

using HcclAlgoAttr = struct HcclAlgoAttrDef {
    bool isHaveCpuRank;        // 是否有cpu参与通信
    bool inlineReduceSwitchOn; // 收到数量时同时完成Reduce计算
    bool isUsedRdmaLevel0;     // Level0 通信域是否使用RDMA
    bool isUsedInterHccsMode;  // 超节点内节点间是否使用HCCS模式
    std::string identifier;
    std::string collectiveId;
    NICDeployment nicDeployment;
    WorkMode commWorkMode;
    std::map<HcclCMDType, std::vector<HcclAlgoType>> commAlgoConfig;

    HcclAlgoAttrDef()
        : isHaveCpuRank(false),
          inlineReduceSwitchOn(true),
          isUsedRdmaLevel0(false),
          isUsedInterHccsMode(false),
          identifier(""),
          collectiveId(""),
          nicDeployment(NICDeployment::NIC_DEPLOYMENT_DEVICE),
          commWorkMode(WorkMode::HCCL_MODE_NORMAL)
    {
        SetDefaultAlgo();
    }
    void SetDefaultAlgo()
    {
        for (u32 opType = 0; opType < static_cast<u32>(HcclCMDType::HCCL_CMD_MAX); opType++) {
            commAlgoConfig[static_cast<HcclCMDType>(opType)]
                = std::vector<HcclAlgoType>(HCCL_ALGO_LEVEL_NUM, HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT);
        }
    }
};

struct HcclTopoAttr {
    u32 serverNum;               // 集群中总的服务器数
    u32 superPodNum;             // 集群中总的超节点数
    u32 moduleNum;               // 集群中的总的module数
    u32 deviceNumPerServer;      // 服务器上的Device数量
    u32 deviceNumPerAggregation; // 每个module中的Device数量
    bool multiModuleDiffDeviceNumMode; // 每个module内的设备数是否相等，如果不相同即为多module不同卡模式 （走大RING环）
    bool multiSuperPodDiffServerNumMode; // 每个超节点内的server数是否相等
    bool multiSuperPodDiffDeviceNumMode; // 每个超节点内的总rank数是否相等

    bool isDiffDeviceType;
    u32 gcdDeviceNumPerAggregation;

    u32 meshAggregationRankSize;
    bool isDiffDeviceModule;
    bool isSingleMeshAggregation;
    bool isAllRankSamePlane; // 通信域所有Rank是否在同一平面

    u32 userRank; // 通信域 RankID
    u32 realUserRank;
    u32 userRankSize;                   // 通信域的 Rank数量
    std::vector<RankInfo> rankInfoList; // world group内rank的信息, 按照rank id递增依次排列

    u32 devicePhyId;
    s32 deviceLogicId;
    bool useSuperPodMode; // 使用SDID作为DeviceId做相关查询操作

    DevType deviceType;
    bool isStandardCard;
    bool is310PDuoCard;
    bool isCommon310P3DUO; // 310p duo 2卡4芯场景
    s32 hccsPortNum;

    std::vector<u32> nicList;
    std::unordered_map<u32, u32> pairLinkCounter; // server内所有device间的链路类型计数
    std::unordered_map<u32, std::unordered_map<int, std::vector<int>>> pairLinkInfo; // server内所有device间的链路类型
    bool isSupportRdmaLite;                                                          // 是否支持rdma lite
    bool isSupportHccsAndSio;                                                        // 是否支持hccs sio并发
    u32 localNicPort;
    bool isNeedInitNic; // 是否需要初始化Nic，心跳使用
    bool isARSDoubleRing;

    HcclTopoAttr()
        : serverNum(0),
          superPodNum(0),
          moduleNum(0),
          deviceNumPerServer(0),
          deviceNumPerAggregation(0),
          multiModuleDiffDeviceNumMode(false),
          multiSuperPodDiffServerNumMode(false),
          multiSuperPodDiffDeviceNumMode(false),
          isDiffDeviceType(false),
          gcdDeviceNumPerAggregation(0),
          meshAggregationRankSize(0),
          isDiffDeviceModule(false),
          isSingleMeshAggregation(false),
          isAllRankSamePlane(false),
          userRank(0),
          realUserRank(0),
          userRankSize(0),
          rankInfoList(0),
          devicePhyId(0),
          deviceLogicId(0),
          useSuperPodMode(false),
          deviceType(DevType::DEV_TYPE_COUNT),
          isStandardCard(false),
          is310PDuoCard(false),
          isCommon310P3DUO(false),
          hccsPortNum(-1),
          nicList(0),
          pairLinkCounter(0),
          pairLinkInfo(0),
          isSupportRdmaLite(false),
          isSupportHccsAndSio(false),
          localNicPort(0),
          isNeedInitNic(false),
          isARSDoubleRing(true)
    {}
};

using RegisterToHeartBeatCallBack = HcclResult (*)(
    s32 deviceLogicID, u32 userRank, DevType devType, std::vector<RankInfo>& rankInfoList, const u32 port,
    const bool isNeedNic, u32 peerRankId, const std::string& commIdentifier, const std::string& tag,
    bool useSuperPodMode, bool isUsedRdmaLevel0);
using UnRegisterToHeartBeatCallBack
    = void (*)(s32 deviceLogicID, DevType devType, const std::string& commIdentifier, const std::string& tag);
using SetRankPortInfoCallBack = HcclResult (*)(s32 deviceLogicID, bool isUseRankPort, std::vector<u32>& ranksPort);
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus
void RegisterHeartBeatCallBack(
    RegisterToHeartBeatCallBack p1, UnRegisterToHeartBeatCallBack p2, SetRankPortInfoCallBack p3);
#ifdef __cplusplus
}
#endif // __cplusplus
} // namespace hccl
#endif /** HCCL_IMPL_PUB_H */
