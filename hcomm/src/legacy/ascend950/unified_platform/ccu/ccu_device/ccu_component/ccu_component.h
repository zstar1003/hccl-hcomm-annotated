/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_COMPONENT_H
#define HCCL_CCU_COMPONENT_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <shared_mutex>

#include "hccl/hccl_types.h"
#include "types.h"

#include "ccu_channel_mgr.h"
#include "ccu_res_allocator_legacy.h"
#include "ccu_device_manager.h"
#include "tp_manager.h"
#include "ccu_error_handler.h"

namespace Hccl {

class CcuComponent {
public:
    CcuComponent(const CcuComponent& that) = delete;
    CcuComponent& operator=(const CcuComponent& that) = delete;

    static CcuComponent& GetInstance(const int32_t deviceLogicId);
    void Init();
    void Deinit();

    HcclResult GetCcuResourceSpaceBufInfo(const uint8_t dieId, uint64_t& addr, uint64_t& size) const;
    HcclResult GetCcuResourceSpaceTokenInfo(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const;
    HcclResult GetCcuResourceSpaceTokenInfoForLocal(const uint8_t dieId, uint64_t& tokenId, uint64_t& tokenValue) const;

    HcclResult
    AllocChannels(const uint8_t dieId, const ChannelPara& channelPara, std::vector<ChannelInfo>& channelInfos);
    HcclResult ConfigChannel(const uint8_t dieId, const ChannelCfg& cfg);
    HcclResult ReleaseChannel(const uint8_t dieId, const uint32_t channelId);

    HcclResult GetLoopChannelId(const uint8_t srcDieId, const uint8_t dstDieId, uint32_t& channelId) const;

    HcclResult AllocRes(
        const uint8_t dieId, const ResType resType, const uint32_t num, const bool consecutive,
        vector<ResInfo>& resInfos);
    HcclResult ReleaseRes(const uint8_t dieId, const ResType resType, const uint32_t startId, const uint32_t num);

    HcclResult AllocIns(const uint8_t dieId, const uint32_t num, ResInfo& insInfo);
    HcclResult ReleaseIns(const uint8_t dieId, const ResInfo& insInfo);
    uint32_t GetInsConsecutiveRemainSize(const uint8_t dieId) const;
    HcclResult AllocCke(const uint8_t dieId, const uint32_t num, vector<ResInfo>& ckeInfos);
    HcclResult ReleaseCke(const uint8_t dieId, const vector<ResInfo>& ckeInfos);
    HcclResult AllocXn(const uint8_t dieId, const uint32_t num, vector<ResInfo>& xnInfos);
    HcclResult ReleaseXn(const uint8_t dieId, const vector<ResInfo>& xnInfos);

    HcclResult CleanDieCkes(const uint8_t dieId) const;
    HcclResult SetTaskKill();
    HcclResult SetTaskKillDone();
    HcclResult CleanTaskKillState() const;

    const std::array<bool, MAX_CCU_IODIE_NUM>& GetDieEnableFlags() const;
    bool IsInited() const { return ifInit; }

private:
    static constexpr uint32_t INVALID_DEV_ID = 0xFFFFFFFF;
    bool ifInit{false};
    int32_t devLogicId{static_cast<int32_t>(INVALID_DEV_ID)};
    uint32_t devPhyId{INVALID_DEV_ID};
    CcuVersion ccuVersion{CcuVersion::CCU_INVALID};
    std::array<bool, MAX_CCU_IODIE_NUM> dieEnableFlags{}; // 根据资源规格的记录可用的die

    // 记录环回设备信息，dieId, (feId, ipAddr)
    std::unordered_map<uint8_t, std::pair<uint32_t, IpAddress>> loopFeIpAddrMap{};
    // 记录CCU资源空间Buffer，避免重复内存注册
    std::unordered_map<uint8_t, std::unique_ptr<LocalUbRmaBuffer>> ccuRmaBufferMap{};
    std::unordered_map<uint8_t, std::unique_ptr<LocalUbRmaBuffer>> localCcuRmaBufferMap{};
    std::vector<std::unique_ptr<LocalUbRmaBuffer>> additionalCcuRmaBufferMap{};
    // 资源管理器
    std::array<std::unique_ptr<CcuChannelMgr>, MAX_CCU_IODIE_NUM> channelMgrs{};
    std::array<std::unique_ptr<CcuResAllocator>, MAX_CCU_IODIE_NUM> resAllocators{};
    // 环回channel编号
    std::array<uint32_t, MAX_CCU_IODIE_NUM> loopChannelIds{};
    // 环回jetty资源信息
    std::unordered_map<uint8_t, std::vector<HrtRaUbJettyCreatedOutParam>> createdOutParamMap{};
    using ImportOutParamPair = std::pair<RdmaHandle, HrtRaUbJettyImportedOutParam>;
    std::unordered_map<uint8_t, std::vector<ImportOutParamPair>> importedOutParamMap{};
    std::unordered_map<IpAddress, TpInfo> tpInfoMap{};
    std::unordered_map<IpAddress, TpAttrInfo> tpAttrInfoMap{};
    std::unordered_map<IpAddress, uint32_t> psnMap{};

    // CCU Task Kill相关状态
    enum class CcuTaskKillStatus : uint8_t { INIT = 0, TASK_KILL = 1, KILL_DONE = 2, CLEAN_TIF = 3, INVALID = 4 };
    CcuTaskKillStatus status{CcuTaskKillStatus::INVALID};
    mutable std::shared_mutex innerMutex;
    std::mutex taskKillMutex_;

    explicit CcuComponent() = default;
    ~CcuComponent();

    void CheckDiesEnable();
    void ChooseLoopEid(bool& dieDrvEnableFlag, uint8_t dieId);
    HcclResult GetLoopFeIpByDieId(const uint8_t dieId, uint32_t& feId, IpAddress& ipAddr);
    void CreateCcuRmaBuffer();
    void CreateResourceManagers();
    void CreateLoopChannels();
    HcclResult CreateLoopChannel(const uint8_t dieId, uint32_t& channelId);
    HcclResult
    CreateAndImportLoopJettys(const uint8_t dieId, const IpAddress& ipAddr, const vector<JettyInfo>& jettyInfos);
    TpInfo RequestNewTpInfo(const IpAddress& srcIpAddr, const IpAddress& dstIpAddr) const;
    TpInfo GetTpInfo(const IpAddress& ipAddr);
    TpAttrInfo GetLoopTpAttr(const IpAddress& ipAddr, const TpHandle tpHandle);
    HcclResult GetLoopJettyTimeout(const IpAddress& ipAddr, const TpHandle tpHandle, uint8_t& errTimeout);
    uint32_t GetPsn(const IpAddress& ipAddr);
    HcclResult ConfigLoopChannel(const uint8_t dieId, const IpAddress& ipAddr, const ChannelInfo& channelInfo);
    void ConfigMsIdToken();

    void ReleaseJettyRes();
    void UnimportAllJetty();
    void DestroyAllJetty();

    void SetProcess(CcuOpcodeType opCode) const;
    void PrintCcuMissionStatus(int32_t devLogicId) const;
};

}; // namespace Hccl

#endif
