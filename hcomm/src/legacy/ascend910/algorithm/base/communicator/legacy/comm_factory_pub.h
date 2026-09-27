/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_FACTORY_PUB_H
#define COMM_FACTORY_PUB_H

#include <vector>
#include <map>
#include <memory>

#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "comm_base_pub.h"
#include "hccl_socket_manager.h"
#include "dispatcher.h"
#include "coll_alg_param.h"
#include "topo_info_extractor.h"

namespace hccl {
constexpr u32 COMM_P2P_QUERRY_WAIT_TIME = 100;

class ExchangerNetwork;
class CommFactory {
public:
    explicit CommFactory(
        const std::string& identifier, const u32 userRank, const u32 userRankSize, const HcclDispatcher dispatcher,
        const std::unique_ptr<NotifyPool>& notifyPool, std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap,
        std::shared_ptr<TopoInfoExtractor> topoInfoEx, const bool isUsedRdmaLevel0 = false,
        const TopoType topoFlag = TopoType::TOPO_TYPE_COMMON, const DevType deviceType = DevType::DEV_TYPE_910,
        const std::vector<RankInfo> rankVector = std::vector<RankInfo>(0),
        const NICDeployment nicDeploymentInner = NICDeployment::NIC_DEPLOYMENT_DEVICE, bool isHeterogComm = false,
        u32 meshAggregationRankSize = 0, bool isHaveCpuRank = false, bool isUsedInterHccsMode = false,
        bool useSuperPodMode = false);

    virtual ~CommFactory();

    HcclResult Init();     // 初始化必要信息
    HcclResult InitComm(); // 310初始化必要信息

    std::vector<std::unique_ptr<CommBase>> CreateCommP2PAsync(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const u32 dstUserRank,
        u32& status);
    HcclResult CreateCommP2PQuarry(std::vector<std::unique_ptr<CommBase>>& comm, u32& status);
    // 创建单层通信域
    HcclResult CreateCommPlane(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        std::vector<std::unique_ptr<CommBase>>& commVec, DeviceMem expMem = DeviceMem());

    // 提供bcast多环使用,根据指定root节点,获取与当前userrank所在同一平面的子root节点
    u32 GetSubRootUserRank(const u32 userRank, const u32 rootUserRank);
    u32 GetSubRootUserRankWithSuperPod(const u32 userRank, const u32 rootUserRank);
    // 提供scatter使用,根据指定root节点和当前节点的userRank,获取与当前userRank所在同一平面的子root节点
    u32 GetSubRootForScatter(const u32 root);
    // 提供网口裁剪使用，在无节点间通信域场景下，获取本rank在节点间子通信域(多平面)内当前平面的rank号
    u32 GetLevel1CommRank(const u32 ringIdx);
    HcclResult SetHDCModeInfo(
        std::unordered_map<std::string, std::map<u32, HcclIpAddress>>& rankDevicePhyIdNicInfoMap,
        std::vector<u32>& ranksPort, std::vector<u32>& vnicRanksPort, bool isSetHDCModeInfo, bool isUseRankPort);

protected:
    /* 禁止用户对工厂类的实体做拷贝构造或拷贝赋值的操作，内部有指针成员变量 */
    CommFactory(const CommFactory&) = delete;
    CommFactory& operator=(const CommFactory&) = delete;

private:
    HcclResult CheckCommPara(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem,
        const CommParaInfo& commParaInfo) const;

    HcclResult GetIsUsedRdma(const CommParaInfo& commParaInfo, bool& isUsedRdma);

    HcclResult CreateCommRing(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec);

    HcclResult CreateCommHD(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec);

    HcclResult CreateCommStar(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec);

    HcclResult CreateCommMesh(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec, DeviceMem& expMem);

    HcclResult CreateCommP2P(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec);

    HcclResult CreateCommP2PSync(
        const std::string& tag, const DeviceMem& inputMem, const DeviceMem& outputMem, const CommParaInfo& commParaInfo,
        const std::vector<std::vector<RankInfo>>& commPlaneVec, bool isUsedRdma,
        std::vector<std::unique_ptr<CommBase>>& commVec);

    HcclResult SetIsUsedRdma(
        const CommParaInfo& commParaInfo, std::vector<SingleSubCommTransport>& commTransport, bool isUsedRdma) const;

    // 获取本rank在子通信域(多平面)内当前平面的rank号
    u32 GetSubCollectiveRank(const std::vector<RankInfo>& vecPara) const;
    bool JudgmentSetHeterogP2p(u32 rank) const;
    void CreateStarLinkPara(std::vector<RankInfo>& linkParas);
    bool IsUseSdidForVnicIp(); // 是否使用sdid作为vnicip

    const std::string identifier_; // 本节点所在的通信域ID
    const u32 userRank_;           //  本节点的用户原始rank号
    const u32 userRankSize_;       // 本节点所在的用户通信域rank size
    RankInfo rankData_;            // 当前rank的相关信息
    const TopoType topoFlag_;      // 当前通信域内服务器间拓扑组合类型
    const DevType deviceType_;     // 当前rank所归属设备的类型
    // comm特性位图:0x1=支持inline-reduce;0x2=支持RDMA,1=RDMA,
    // 0=TCP;0x4=支持RDMA异步,前提是支持RDMA;Others=保留;

    const HcclDispatcher dispatcher_; // signal调度句柄(event/notify机制)
    const std::unique_ptr<NotifyPool>& notifyPool_;
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap_;
    std::shared_ptr<TopoInfoExtractor> topoInfoEx_;
    const bool isUsedRdmaLevel0_;

    // 保存所有级别的通信rank关系, CommPlaneVector_[CommPlane][ringIndex]: 第CommPlane级 第ringIndex个环
    std::vector<std::vector<std::vector<RankInfo>>> CommPlaneVector_;
    // 子通信域内当前userrank是否为bridge rank的属性(多环)
    std::vector<bool> isBridgeVector_;

    // 通信域在当前superPod内, 按照serverIdx划分的所有rank信息
    std::map<u32, std::vector<RankInfo>> serverToRank_;
    // 通信域所有rank的信息, 按照superPodId -> RankInfo 的结构划分
    std::map<u32, std::vector<RankInfo>> superPodToRank_;
    // 记录server内, 本rank和其他rank的连接关系
    std::map<s32, LinkTypeInServer> deviceLinkTypeMap_;

    // 整个通信域内rank的信息(直接调用exchanger生成，下标为userrank)
    std::vector<RankInfo> rankVector_;

    NICDeployment nicDeployInner_;
    bool isHeterogComm_;
    bool isHaveCpuRank_;
    // 复用Socket时, 复用SocketManager
    std::shared_ptr<HcclSocketManager> reusedSocketManager_;
    s32 deviceLogicId_;

    bool isUsedInterHccsMode_;
    bool useSuperPodMode_;
    std::unordered_map<std::string, std::map<u32, HcclIpAddress>> rankDevicePhyIdNicInfoMap_;
    std::vector<u32> ranksPort_;
    bool isSetHDCModeInfo_{false};
    bool isUseRankPort_{false};
    std::vector<u32> vnicRanksPort_;
};
} // namespace hccl

#endif /* COMM_FACTORY_PUB_H */
