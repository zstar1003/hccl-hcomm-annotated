/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MY_RANK_H
#define MY_RANK_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "hccl/hccl_types.h"
#include "hccl/base.h"
#include "hccl/hccl_res.h"
#include "hccl_mem_defs.h"
#include "acl/acl_rt.h"
#include "socket_manager.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "mem_host_pub.h"
#include "rank_pair_mgr.h"
#include "endpoints/endpoint_mgr.h"
#include "comm_config_pub.h"
#include "manager_common.h"
#include "common.h"
#include "comm_mems/comm_mems.h"
#include "engine_ctxs.h"
#include "../../../dfx/ns_recovery/ns_recovery.h"
#include "hdc_pub.h"
#include "rank_graph.h"
#include "orion_adapter_hccp.h"
#include "coll_comm_config_consistency.h"
#include "exchange_info_mgr.h"

#include "ccu_types.h"
#include "ccu_drv_handle.h"
#include "ccu_device_res.h"
#include "dev_type.h"

namespace hccl {

constexpr uint32_t DEFAULT_MODE = 0;
constexpr uint32_t AICPU_TS_MODE = 2;
constexpr uint32_t CCU_MS_MODE = 5;
constexpr uint32_t CCU_SCHED_MODE = 6;
// opExpansionMode 到 CcuInstanceType 的映射，仅作数据类型转换，不做逻辑处理。
// 供 MyRank 内部及外部适配层（如 coll_comm_ccu_c_adpt.cc）共用。
inline CcuInstanceType OpExpansionModeToCcuInstanceType(uint32_t opExpansionMode)
{
    if (opExpansionMode == CCU_SCHED_MODE) {
        return CcuInstanceType::CCU_SCHED;
    }
    if (opExpansionMode == CCU_MS_MODE) {
        return CcuInstanceType::CCU_MS;
    }
    return CcuInstanceType::CCU_UNUSED;
}

/**
 * @note 职责：管理当前通信域下本Rank的信息和通信资源
 */
class MyRank {
public:
    MyRank(
        aclrtBinHandle binHandle, uint32_t rankId, const CommConfig& config, const ManagerCallbacks& callbacks,
        RankGraph* rankGraph, const Hccl::RankIpPortMapPtr& rankIpPortMap);
    ~MyRank();

    HcclResult Init(HcclMem cclBuffer, const uint32_t opExpansionMode, uint32_t rankNum);

    CommMems* GetCommMems() const { return commMems_.get(); }

    EngineCtxs* GetEngineCtxs() const { return engineCtxs_.get(); }

    HcclResult UnregMemByTag(const std::string& tag);
    uint32_t GetOpExpansionMode() { return opExpansionMode_; }
    CcuInsHandle GetCcuInstance() const { return ccuInsHandle_; }
    void SetCcuInstance(CcuInsHandle ccuInsHandle) { ccuInsHandle_ = ccuInsHandle; }
    CcuInsHandle GetAssignedCcuInstance() const { return assignedCcuInsHandle_; }
    void SetAssignedCcuInstance(CcuInsHandle ccuInsHandle) { assignedCcuInsHandle_ = ccuInsHandle; }

    CollCommConfigConsistency& GetCollCommConfigConsistency();

    hccl::EndpointMgr* GetEndpointMgr() const { return endpointMgr_.get(); }

    HcclResult CreateChannels(
        CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
        ChannelHandle* channels);

    HcclResult
    QueryChannels(CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels);
    HcclResult DestroyChannels(const ChannelHandle* channels, uint32_t channelNum);

    HcclResult ChannelGetHcclBuffer(ChannelHandle channel, void** buffer, uint64_t* size);
    HcclResult
    ChannelGetRemoteMems(ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, char*** memTags) const;
    HcclResult ChannelGetRemoteMems(
        ChannelHandle channel, uint32_t* memNum, CommMem** remoteMem, std::vector<std::string>& memTags) const;

    // Ns recovery
    void SetKfcControlTransfer(
        std::shared_ptr<HDCommunicate> kfcControlTransferH2D, std::shared_ptr<HDCommunicate> kfcStatusTransferD2H);
    std::vector<ChannelHandle> GetAllChannelList();
    HcclResult StopLaunch();
    HcclResult Clean();
    HcclResult Resume();

    /**
     * @brief 批量预建 socket（server 监听 + client 连接），与非共享路径 MyRank::CreateChannels 一致。
     *        共享 jetty 路径在 createFunc 中调用，确保两端 socket 体系一致、可对接。
     * @param[in] channelDescs HCCL 层 channel desc 数组
     * @param[in] channelNum 数量
     * @param[in] socketTag socket 标签（commTag + "_engine_" + engine）
     * @param[out] hcommDescs 输出的 HcommChannelDesc 数组（调用前需用 ChannelDescHccl2Hcomm 填充基础字段，
     *             本方法补上 socket/role/port 字段）
     */
    HcclResult BatchCreateSockets(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        std::vector<HcommChannelDesc>& hcommDescs);

    /**
     * @brief 在已建好的 socket 上执行通信域一致性校验交换（CheckFrameV2 + 用户信息）。
     *        与非共享路径 MyRank::CreateChannels 内部调用 exchangeInfoMgr_ 的逻辑一致，
     *        供共享 jetty 路径在 createFunc 中调用，确保两端无论 shared/非shared 配置是否对称，
     *        都会在 socket 上完成定长 CheckFrameV2（120B）的对称收发，避免一端死等。
     * @param[in] channelDescs HCCL 层 channel desc 数组
     * @param[in] hcommDescs 已通过 BatchCreateSockets 填充 socket 字段的 HcommChannelDesc 数组
     * @param[in] channelNum 数量
     * @param[in] newChannels 新建 channel 的 (idx, reuseIdx) 列表；空表示全部按新建处理
     * @param[in] engine 通信引擎
     * @note 仅 DEV_TYPE_950 实际执行交换，其它设备类型直接返回 SUCCESS，与非共享路径保持一致。
     */
    HcclResult BatchExchangeAndCheckConsistency(
        const HcclChannelDesc* channelDescs, const std::vector<HcommChannelDesc>& hcommDescs, uint32_t channelNum,
        const std::vector<std::pair<u32, u32>>& newChannels, CommEngine engine);
    void GetAbnormalChannelTlsStatus(
        const HcclChannelDesc* channelDescs, const int32_t* statusList, uint32_t channelNum,
        std::vector<Hccl::TlsStatus>& tlsStatusList) const;
    HcclResult PrepareMemHandles(
        EndpointHandle epHandle, void** memHandles, uint32_t memHandleNum, std::vector<MemHandle>& memHandleVec);

private:
    using ReuseSocketIdxMap = std::unordered_map<RankPair*, std::unordered_map<hcomm::EndpointPair*, u32>>;
    HcclResult GetEndpointPairFromChannel(
        const HcclChannelDesc& channelDesc, uint32_t channelIndex, uint32_t channelNum, uint32_t& remoteRank,
        hcomm::EndpointPair*& endpointPair, RankPair*& rankPair);
    HcclResult BatchServerInitForChannels(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        ReuseSocketIdxMap& reuseSocketIdxMap);
    HcclResult BatchGetSocketsForChannels(
        const HcclChannelDesc* channelDescs, uint32_t channelNum, const std::string& socketTag,
        std::vector<HcommChannelDesc>& hcommDescs, ReuseSocketIdxMap& reuseSocketIdxMap);
    HcclResult BatchCreateChannels(
        CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
        std::vector<HcommChannelDesc>& hcommDescs, ChannelHandle* channelHandles,
        std::vector<std::vector<MemHandle>>& allHandles);
    HcclResult
    BatchConnectChannels(const HcclChannelDesc* channelDescs, ChannelHandle* channelHandles, uint32_t channelNum);
    void LogChannelCreationInfo(
        CommEngine engine, const std::string& commTag, const HcclChannelDesc* channelDescs, uint32_t channelNum,
        const ChannelHandle* hostChannelHandleList) const;
    HcclResult FinalizeChannelsByEngine(
        CommEngine engine, const std::string& commTag, uint32_t channelNum, std::vector<HcommChannelDesc>& hcommDescs,
        ChannelHandle* hostChannelHandleList, ChannelHandle* channelHandles);
    HcclResult CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t channelNum) const;
    HcclResult QueryListenPort(
        uint32_t localRank, uint32_t remoteRank, const EndpointDesc& localEndpointDesc,
        const EndpointDesc& remoteEndpointDesc, uint32_t& listenPort, HcommChannelDesc& hcommDesc);
    HcclResult GetLocalTlsStatus(EndpointLocType localType, Hccl::TlsStatus& tlsStatus) const;
    HcclResult RegisterCommMemsToEndpoint(EndpointHandle epHandle);
    HcclResult TryInitCcuInstance();
    HcclResult ReserveCcuMsCommOrFallback();
    HcclResult TryInitCcuInstanceOnDemand();
    void ReconcileCcuMsCommReservation(HcclResult initRet);
    void ReleaseCcuMsCommReservation();
    HcclResult ConfigSqDepthByExpansionMode(CommEngine engine, HcommChannelDesc& hcommDesc) const;
    HcclResult DestroyNewChannels(
        CommEngine engine, const HcclChannelDesc* channelDescs, const std::vector<std::pair<u32, u32>>& newChannels);
    HcclResult
    QueryOneChannel(CommEngine engine, const HcclChannelDesc& channelDesc, u32 reuseIdx, ChannelHandle& handle);
    HcclResult
    DestroyOneChannel(ChannelHandle userHandle, u32 index, HcclResult& firstErr, u32& invalidHandleCnt, u32& failedCnt);
    aclrtBinHandle binHandle_{nullptr};
    uint32_t rankId_{};
    int32_t devLogicId_{};
    CommConfig config_{};

    // 当前通信域初始化没有处理CommConfig，暂时只使用展开模式
    uint32_t opExpansionMode_{0};

    std::unique_ptr<RankPairMgr> rankPairMgr_{nullptr};
    std::unique_ptr<hccl::EndpointMgr> endpointMgr_{nullptr};
    std::unique_ptr<CommMems> commMems_{nullptr};
    std::unique_ptr<EngineCtxs> engineCtxs_{nullptr};

    CcuInsHandle ccuInsHandle_{0}; // 按固定量CCU资源方式创建的 ccuInsHandle（HcclCommQueryCcuIns）（兼容旧版本）
    CcuInsHandle assignedCcuInsHandle_{0}; // 按需申请CCU资源方式绑定的 ccuInsHandle（HcommCcuInsCreate + Assign）

    ManagerCallbacks callbacks_;

    // RankGraph (临时放在myRank里面，后面会随着createchannel整体迁移到RankPairMgr上)
    RankGraph* rankGraph_{nullptr};

    // 记录每次调用BatchCreateChannels时新增的channelIndex, reuseIdx
    std::vector<std::pair<u32, u32>> newChannels_{};

    // channelHandle(host) -> EndpointPair 裸指针反查索引：EndpointPair 由 EndpointPairMgr 以 unique_ptr 持有，
    // 其生命周期与 RankPairMgr 一致（MyRank 析构时先 clear 反查表再释放 rankPairMgr_）
    std::unordered_map<ChannelHandle, hcomm::EndpointPair*> handleToEpPair_{};

    // 保护 newChannels_ / handleToEpPair_ 的并发访问
    std::mutex channelIndexMtx_{};

    // Ns recovery
    std::unique_ptr<NsRecoveryProcessor> nsRecoveryProcessor_{nullptr};
    // 内部获取 port 的方法，根据 mode_ 区分 v1/v2
    HcclResult GetDevicePortInternal(uint32_t rank, uint32_t* devPort, EndpointLocType locType);

    Hccl::RankIpPortMapPtr rankIpPortMap_;

    CollCommConfigConsistency collCommConfigConsistency_;
    ExchangeInfoMgr exchangeInfoMgr_;
    std::shared_ptr<hcomm::CcuDrvHandle> ccuDrvHandle_{};
    bool ccuMsCommReserved_{false};
    Hccl::HcclMainboardId mainBoardType_{Hccl::HcclMainboardId::MAINBOARD_OTHERS};
};

} // namespace hccl

namespace MyRankUtils {

// 将 HCCL 通道描述转换为 HCOMM 通道描述。
HcommChannelDesc ChannelDescHccl2Hcomm(const HcclChannelDesc& hcclDesc, const hccl::CommConfig& commConfig);

} // namespace MyRankUtils

#endif // MY_RANK_H
