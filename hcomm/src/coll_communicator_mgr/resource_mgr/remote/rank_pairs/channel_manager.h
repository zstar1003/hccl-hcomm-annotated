/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_MANAGER_H
#define CHANNEL_MANAGER_H

#include "hccl/hccl_res.h"
#include "hccl_types.h"
#include "transport_pub.h"
#include "hccl_common.h"
#include "hccl_mem_defs.h"
#include "transport_pub.h"
#include "aicpu_operator_pub.h"
#include "channel_param.h"
#include "manager_common.h"
#include "hccl_independent_common.h"

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>

namespace std {
template <>
struct hash<HcclChannelDesc> {
    size_t operator()(const HcclChannelDesc& desc) const
    {
        size_t hash = 0;
        // 仅区分remoteRank和protocol
        hash ^= std::hash<uint32_t>()(desc.remoteRank);
        hash ^= std::hash<int32_t>()(static_cast<int32_t>(desc.channelProtocol));
        return hash;
    }
};
} // namespace std

namespace hccl {

struct HcclChannelDescEqual {
    bool operator()(const HcclChannelDesc& lcd, const HcclChannelDesc& rcd) const
    {
        // 需要扩展增加EndpointDesc的有关内容
        return lcd.remoteRank == rcd.remoteRank && lcd.channelProtocol == rcd.channelProtocol;
    }
};

class ChannelManager {
public:
    ChannelManager() = default;
    ~ChannelManager() = default;
    HcclResult Init(aclrtBinHandle binHandle, u32 userRank, const ManagerCallbacks& callbacks);
    HcclResult SetChannelCallbacks(const ChannelManagerCallbacks& channelCallbacks);
    HcclResult ChannelCommCreate(
        const std::string& commId, CommEngine engine, const HcclChannelDesc* channelDescList, uint32_t listNum,
        ChannelHandle* channelList);
    HcclResult ChannelCommGetNotifyNum(ChannelHandle channel, uint32_t* notifyNum);
    HcclResult ChannelCommDestroy(ChannelHandle* channelList, uint32_t channelNum);
    HcclResult ChannelCommGetHcclBuffer(ChannelHandle channel, CommBuffer* buffer);
    HcclResult ChannelCommGetRemoteMem(ChannelHandle channel, HcclMem** remoteMem, uint32_t* memNum);
    HcclResult ReleaseChannel();
    HcclResult SetHcclQos(u32 hcclQos);

private:
    template <typename T>
    HcclResult CopyVectorToDeviceMem(const u64 len, DeviceMem& dstDeviceMem, const std::vector<T>& srcVec);
    HcclResult AllocAndClearHostMem(u64 size, std::shared_ptr<HostMem>& bufferPtr) const;
    HcclResult CreateWorkSpace(u64 size, DeviceMem& buffer) const;
    HcclResult CheckNotifyOrQPMaxNum(u64& existNum, const u64& MaxNum, const bool& isNotifyRes);
    HcclResult DeepCopyH2DChannelP2p(const HcclChannelP2p& hostChannelP2p, HcclChannelP2p& deviceChannelP2p);
    HcclResult DeepCopyH2DChannelRoce(const HcclChannelRoce& hostChannelRoce, HcclChannelRoce& deviceChannelRoce);
    HcclResult DeepCopyH2DChannelRemoteResV2(
        const HcclIndOpChannelRemoteResV2& hostRemoteResV2, HcclIndOpChannelRemoteResV2& deviceRemoteResV2);
    HcclResult DeepCopyH2DchannelParam(
        const HcclIndOpChannelRemoteResV3& hostChannelParam, HcclIndOpChannelRemoteResV3& deviceChannelParam);
    HcclResult ReleaseChannelParam(HcclIndOpChannelRemoteResV3& channelParam);
    HcclResult BuildOpRemoteChannelP2pResParam(const LINK& link, HcclIndOpChannelRemoteResV2& remoteRes);
    HcclResult BuildOpRemoteChannelRoceResParam(const LINK& link, HcclIndOpChannelRemoteResV2& remoteRes);
    HcclResult
    ParseChannelRemoteDataToMem(const OpCommTransport& opTransportResponse, HcclIndOpChannelRemoteResV3& channelParam);
    HcclResult AicpuChannelInit(
        const std::string& commId, const std::string& tag, CommEngine engine,
        const OpCommTransport& opTransportResponse, ChannelHandle* channelList, uint32_t listNum);
    void ClearOpTransportResponseLinks(OpCommTransport& opTransportResponse);
    OpCommTransport BuildChannelRequests(const std::vector<HcclChannelDesc>& descs);

    HcclResult CheckChannelParam(CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t descNum);
    HcclResult RegisterHandle(
        const std::string& tag, CommEngine engine, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle);
    HcclResult RegisterHandleHDPair(ChannelHandle deviceChannelHandle, ChannelHandle hostChannelHandle);
    HcclResult UnregisterHandle(ChannelHandle channel);
    HcclResult PrepareHandleArray(
        const std::string& tag, CommEngine engine, const HcclChannelDesc* channelDesc, uint32_t descNum,
        ChannelHandle* channelHandleArray, std::vector<HcclChannelDesc>& needCreateDescs,
        std::vector<uint32_t>& needCreateIndices);
    HcclResult IsChannelExist(ChannelHandle channel);
    HcclResult GetHostChannel(ChannelHandle channel, ChannelHandle& hostChannel);

    std::unordered_map<std::string, ChannelHandle> channelHandleMap_;
    std::unordered_map<ChannelHandle, std::string> keyMap_;
    std::unordered_map<ChannelHandle, CommEngine> engineMap_;
    std::unordered_map<ChannelHandle, ChannelHandle> channelD2HMap_;
    std::unordered_set<ChannelHandle*> channelHandleArraySet_;
    std::unordered_map<ChannelHandle, LINK> linkMap_;
    std::vector<LINK> channelLinks_{};
    u32 userRank_;
    std::vector<RankInfo> rankInfoList_;
    std::vector<std::shared_ptr<DeviceMem>> channelParamMemVector_;
    std::list<DeviceMem> channelParamMemList_;
    aclrtBinHandle binHandle_;
    ManagerCallbacks callbacks_;               // 存储回调函数
    ChannelManagerCallbacks channelCallbacks_; // channelMgr的回调函数
    u32 hcclQos_;
    std::vector<HcclMemHandle> CollectMemHandles(const std::vector<HcclChannelDesc>& descs) const;
};

} // namespace hccl

#endif // CHANNEL_MANAGER_H
