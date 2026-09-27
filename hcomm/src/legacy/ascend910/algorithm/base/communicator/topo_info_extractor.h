/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TOPO_INFO_EXTRACTOR_H
#define TOPO_INFO_EXTRACTOR_H

#include <array>
#include <vector>
#include <map>

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "log.h"
#include "hccl_common.h"
#include "common.h"
#include "hccl_impl_pub.h"
#include "comm_ahc_base_pub.h"
#include "alg_template_base_pub.h"
#include "env_config.h"
namespace hccl {

class TopoInfoExtractor {
public:
    explicit TopoInfoExtractor(HcclAlgoAttr& algoAttr, HcclTopoAttr& topoAttr, const TopoType topoType);
#ifdef CCL_LLT
    TopoInfoExtractor(
        std::string identifier, u32 userRank, u32 userRankSize, TopoType topoType, DevType deviceType,
        std::vector<RankInfo>& rankVector, u32 meshAggregationRankSize = 0, bool isUsedRdmaLevel0 = false,
        bool isUsedInterHccsMode = false, bool multiModuleDiffDeviceNumMode = false,
        bool multiSuperPodDiffServerNumMode = false, bool multiSuperPodDiffDeviceNumMode = false,
        bool isDiffDeviceType = false, u32 gcdDeviceNumPerAggregation = 0);
#endif
    ~TopoInfoExtractor();
    HcclResult Init(std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfig);
    HcclResult CheckInitInfo();
    HcclResult SetRankInfo();
    HcclResult SetTopologyInfo();
    HcclResult SetTopoDefaultInfo();
    HcclResult SetTopoInfoForLevel0();
    HcclResult SetTopoInfoForLevel1(bool prepareAHC = false);
    HcclResult SetTopoInfoForLevel2();
    HcclResult SetTopoInfoForMeshL0();
    HcclResult SetTopoInfoForMeshL1();
    HcclResult SetTopoInfoForARS();
    HcclResult SetTopoInfoForCombineL1();
    HcclResult CheckPlaneInfo();
    HcclResult CheckSuperPodInfo();
    HcclResult CheckServerInfo();
    HcclResult SetSingleLevel0For8P();
    HcclResult SetSingleLevel0();
    HcclResult GetServerIdx(const RankInfo& rankInfo, u32& serverIdx) const;
    bool IsDiffDeviceModuleInServer() const;
    HcclResult SetMultiLevel0(u32 ringNum);
    HcclResult SetMultiLevel0AnyPath(std::vector<std::vector<u32>> multiOrder);
    HcclResult GetModuleIdx(const RankInfo& rankInfo, u32& moduleIdx);
    HcclResult SetBridgeLinkInfo(RankInfo& bridgePara, u32 bridgeUserRank);
    HcclResult SetTopoDefaultInfoFor8P();
    HcclResult GetIsUsedRdmaMap(std::unordered_map<u32, bool>& isUsedRdmaMap);
    HcclResult GetCommPlaneRanks(std::vector<std::vector<std::vector<u32>>>& CommPlaneRanks);
    void GetCommPlaneVector(std::vector<std::vector<std::vector<RankInfo>>>& commPlaneVector);
    void GetIsBridgeVector(std::vector<bool>& isBridgeVector);
    void InitAHCConfig(std::map<HcclCMDType, std::vector<HcclAlgoType>>& algoConfig);
    void AHCCommSubgroupInit();
    HcclResult SetAHCSubGroupsAndAlgOption();
    void GetCommPlaneSubGroupVector(std::vector<std::vector<std::vector<std::vector<u32>>>>& CommPlaneSubGroupVector);
    void SetAHCLevelAlgOption(AHCConcOpType ahcConcOpType, TemplateType templateType);
    void GetAHCAlgOption(std::map<AHCConcOpType, TemplateType>& ahcAlgOption);
    HcclResult GetRankVecInfo(std::vector<std::vector<std::vector<u32>>>& serverAndsuperPodToRank);
    HcclResult SetRankMap();
    void GetRankData(RankInfo& rankData);
    void GetServerToRank(std::map<u32, std::vector<RankInfo>>& serverToRank);
    void GetSuperPodToRank(std::map<u32, std::vector<RankInfo>>& superPodToRank);
    void GetDeviceLinkTypeMap(std::map<s32, LinkTypeInServer>& deviceLinkTypeMap);

private:
    HcclResult AHCSubGroupInit(
        CommPlane algLevel, std::vector<std::vector<std::vector<std::vector<u32>>>>& CommPlaneSubGroupVector) const;

    const std::string identifier_; // 本节点所在的通信域ID
    const u32 userRank_;           //  本节点的用户原始rank号
    const u32 userRankSize_;       // 本节点所在的用户通信域rank size
    RankInfo rankData_;            // 当前rank的相关信息
    const TopoType topoType_;      // 当前通信域内服务器间拓扑组合类型
    const DevType deviceType_;     // 当前rank所归属设备的类型

    // 子通信域内当前userrank是否为bridge rank的属性(多环)
    std::vector<bool> isBridgeVector_;

    // 通信域在当前superPod内, 按照serverIdx划分的所有rank信息
    std::map<u32, std::vector<RankInfo>> serverToRank_;
    // 通信域所有rank的信息, 按照superPodIdx -> RankInfo 的结构划分
    std::map<u32, std::vector<RankInfo>> superPodToRank_;
    // 通信域内，按照 serverIdx 划分的 server 层和 superPod 层合并的 rank 信息
    std::map<u32, std::vector<RankInfo>> serverToRankMerge_;
    // 通信域内，按照 gcdDeviceNumPerAggregation_ 划分的 server 层拆分后的 rank 信息
    std::map<u32, std::vector<RankInfo>> gcdToRankSplit_;
    // 记录server内, 本rank和其他rank的连接关系
    std::map<s32, LinkTypeInServer> deviceLinkTypeMap_;

    // 8pring 多环的commlevel0 顺序
    std::vector<std::vector<u32>> multiLevel0Order_;
    // 整个通信域内rank的信息(直接调用exchanger生成，下标为userrank)
    std::vector<RankInfo> rankVector_;
    u32 meshAggregationRankSize_;
    // 不同拓扑场景每个server上的设备数
    std::array<u32, static_cast<u32>(TopoType::TOPO_TYPE_RESERVED)> ranksOneNode_;

    std::vector<std::vector<std::map<u32, u32>>> subCommRank2UserRank_;
    std::vector<std::vector<std::map<u32, u32>>> userRank2subCommRank_;

    const bool isUsedRdmaLevel0_;
    bool isUsedInterHccsMode_;
    bool isDiffAggregation_;

    // 是否静态配置 AHC 模式
    bool isConfigAHC_;
    bool isConfigNULL_;

    // 当前层次是否为非对称
    bool multiModuleDiffDeviceNumMode_;   // 每个module内的设备数是否相等
    bool multiSuperPodDiffServerNumMode_; // 每个超节点内的server数是否相等
    bool multiSuperPodDiffDeviceNumMode_; // 每个超节点内的设备数是否相等(超节点内的总rank数)

    bool isDiffDeviceType_;
    u32 gcdDeviceNumPerAggregation_;

    // 保存所有 level 的通信分组关系， CommPlaneSubGroupVector_[CommPlane] : 第 CommPlane 级通信域内分组信息
    std::vector<std::vector<std::vector<std::vector<u32>>>> CommPlaneSubGroupVector_;

    // 保存 AHC 算法中不同 level 以及组内、组间需要使用的算子配置
    std::map<AHCConcOpType, TemplateType> ahcAlgOption_;

    // 保存所有级别的通信rank关系, CommPlaneVector_[CommPlane][ringIndex]: 第CommPlane级 第ringIndex个环
    std::vector<std::vector<std::vector<RankInfo>>> CommPlaneVector_;
};

bool Ascending(const RankInfo& first, const RankInfo& second);
bool CompareWithUserRankAscend(const RankInfo& left, const RankInfo& right);
bool CheckSdmaWithRohTopo(const std::vector<u32>& nicList, std::vector<u32>& topoList);
std::vector<std::vector<u32>> GetRingsOrderByTopoType(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList);
std::vector<std::vector<u32>> GetRingsOrderForAnyPath(u32 ranksSize, TopoType topoType, std::vector<u32>& nicList);
} // namespace hccl

#endif
