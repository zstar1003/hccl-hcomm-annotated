/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include <cstring>
#include <chrono>
#include <vector>

#include "ccu_types.h"
#include "hcomm_c_adpt.h"
#include "hcomm_c_adpt_common.h"
#include "hcomm_res_mgr.h"
#include "hcomm_res.h"
#include "hcomm_result_defs.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"
#include "log.h"
#include "param_check_pub.h"
#include "comm_engine_utils.h"
#include "channel_process.h"
#include "aicpu_ts_channel_helper.h"
#include "channel_config.h"
#include "shared_jetty_mgr.h"
#include "endpoint.h"
#include "nic_plugin_holder.h"
#include "nic_plugin_manager.h"
#include "acl/acl_rt.h"
#include "adapter_rts_common.h"
#include "tp_qos.h"
#include "hccl/hccl_types.h"
#include "ccu_log.h"
#include "env_config.h"
#include "hccp_common.h"

using namespace hcomm;

constexpr uint32_t kDscpToRoceTcShift = 2U; // RoCE TC = DSCP << 2（DiffServ 高 6 位为 DSCP）
constexpr uint8_t UB_PATH_MODE_DEFAULT = 0xFF;
constexpr uint8_t UB_PATH_MODE_MAX = 2;

static HcommResult GetNetworkModeByLocType(EndpointLocType localLocType, NetworkMode& networkModeOut)
{
    // host RoCE：PEER_ONLINE 读 /etc/hcomm.cfg；device RoCE：OFFLINE 经 HDC 读 /etc/hccl.cfg
    if (localLocType == ENDPOINT_LOC_TYPE_HOST) {
        networkModeOut = NETWORK_PEER_ONLINE;
        return HCCL_SUCCESS;
    }
    if (localLocType == ENDPOINT_LOC_TYPE_DEVICE) {
        networkModeOut = NETWORK_OFFLINE;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[GetNetworkModeByLocType] unsupported localLocType[%d].", static_cast<int>(localLocType));
    return HCCL_E_PARA;
}

static HcommResult GetEndPointLocType(EndpointHandle endpointHandle, EndpointLocType& localLocTypeOut)
{
    void* endpointRaw = nullptr;
    CHK_RET(static_cast<HcclResult>(HcommEndpointGet(endpointHandle, &endpointRaw)));
    auto* endpoint = static_cast<Endpoint*>(endpointRaw);
    localLocTypeOut = endpoint->GetEndpointDesc().loc.locType;
    return HCCL_SUCCESS;
}

static HcommResult ApplyRoceQosCompatToSlTc(HcommChannelDesc& channelDesc, EndpointLocType localLocType)
{
    if (channelDesc.qos == HCCL_COMM_QOS_CONFIG_NOT_SET) {
        return HCCL_SUCCESS;
    }

    // qos_dscp→TC 映射仅 950/960 生效（host/device RoCE 均支持；按 DevType 枚举匹配，避免代际数值比较误伤）
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType >= DevType::DEV_TYPE_COUNT) {
        HCCL_ERROR("[ApplyRoceQosCompatToSlTc] invalid deviceType[%d].", static_cast<int>(deviceType));
        return HCCL_E_PARA;
    }
    if (deviceType != DevType::DEV_TYPE_950 && deviceType != DevType::DEV_TYPE_960) {
        return HCCL_SUCCESS;
    }

    const uint8_t sl = static_cast<uint8_t>(channelDesc.qos & 0xFFU);
    uint8_t dscp = Hccl::kUboeDefaultDscp;
    NetworkMode networkMode = NETWORK_OFFLINE;
    CHK_RET(static_cast<HcclResult>(GetNetworkModeByLocType(localLocType, networkMode)));
    s32 userDevId = 0;
    s32 phyDevId = 0;
    if (hrtGetDevice(&userDevId) != HCCL_SUCCESS || aclrtGetPhyDevIdByUserDevId(userDevId, &phyDevId) != ACL_SUCCESS) {
        HCCL_RUN_WARNING(
            "[ApplyRoceQosCompatToSlTc] get phyDevId failed, userDevId[%d], fallback to default dscp[%u].", userDevId,
            static_cast<unsigned>(dscp));
    } else {
        CHK_RET(Hccl::GetDscpByQos(static_cast<uint32_t>(phyDevId), sl, dscp, networkMode));
    }

    channelDesc.roceAttr.sl = sl;
    channelDesc.roceAttr.tc = static_cast<uint8_t>((static_cast<uint32_t>(dscp) << kDscpToRoceTcShift) & 0xFFU);
    HCCL_INFO(
        "[ApplyRoceQosCompatToSlTc] qos compat: qos[%u] localLocType[%d] networkMode[%d] userDevId[%d] phyDevId[%d] "
        "dscp[%u] sl[%u] tc[%u].",
        channelDesc.qos, static_cast<int>(localLocType), static_cast<int>(networkMode), userDevId, phyDevId,
        static_cast<unsigned>(dscp), static_cast<unsigned>(channelDesc.roceAttr.sl),
        static_cast<unsigned>(channelDesc.roceAttr.tc));
    return HCCL_SUCCESS;
}

namespace {
void DestroyPluginCtx(const HcommNicChannelOps* ops, void* pluginCtx)
{
    if (ops != nullptr && ops->destroy != nullptr) {
        int32_t ret = ops->destroy(pluginCtx);
        if (ret != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] plugin channel destroy failed, ret[%d].", __func__, ret);
        }
    }
}

void RollbackPluginChannels(ChannelHandle* channels, uint32_t count)
{
    for (uint32_t i = 0; i < count; ++i) {
        if (channels[i] == 0)
            continue;
        auto* ch = CHANNEL_FROM_HANDLE(channels[i]);
        if (ch != nullptr) {
            HcclResult ret = ChannelProcess::RemovePluginChannelFromMap(ReinterpretAs<ChannelHandle>(ch));
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[%s] plugin channel not found in map during rollback, handle[0x%llx], ret[%d].", __func__,
                    channels[i], ret);
            }
        }
        channels[i] = 0;
    }
}

HcommResult CreateOnePluginChannel(
    const NicPluginEntry* entry, void* epCtx, HcommChannelDesc* channelDesc, ChannelHandle* outChannel)
{
    *outChannel = 0;

    void* pluginCtx = nullptr;
    HcommNicChannelOps* pluginOps = nullptr;
    HcommResult ret = static_cast<HcommResult>(entry->createChannel(epCtx, channelDesc, &pluginCtx, &pluginOps));
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[NicPlugin][%s] createChannel failed, ret[%d].", __func__, ret), ret);

    if (!ValidateChannelOps(pluginOps)) {
        HCCL_ERROR("[NicPlugin][%s] invalid channel ops.", __func__);
        DestroyPluginCtx(pluginOps, pluginCtx);
        return HCCL_E_INTERNAL;
    }

    HcommNicChannelOps* filledOps = nullptr;
    ret = FillDefaultChannelOps(pluginOps, &filledOps);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[NicPlugin][%s] FillDefaultChannelOps failed, ret[%d].", __func__, ret);
        DestroyPluginCtx(pluginOps, pluginCtx);
        return ret;
    }

    ret = static_cast<HcommResult>(filledOps->init(pluginCtx));
    if (ret != HCCL_SUCCESS) {
        int32_t destroyRet = filledOps->destroy(pluginCtx);
        if (destroyRet != HCCL_SUCCESS) {
            HCCL_WARNING("[%s] plugin channel destroy failed after init failure, ret[%d].", __func__, destroyRet);
        }
        delete filledOps;
        HCCL_ERROR("[NicPlugin][%s] plugin channel init failed, ret[%d].", __func__, ret);
        return ret;
    }

    auto holder = std::make_shared<hcomm::PluginChannelHolder>(entry);
    holder->SetNicChannelCtx(filledOps, pluginCtx);
    ChannelHandle handle = ReinterpretAs<ChannelHandle>(holder.get());

    ret = static_cast<HcommResult>(ChannelProcess::InsertPluginChannelToMap(handle, std::move(holder)));
    CHK_PRT_RET(
        (ret != HCCL_SUCCESS), HCCL_ERROR("[NicPlugin][%s] InsertChannelToMap failed, ret[%d].", __func__, ret), ret);

    *outChannel = MAKE_PLUGIN_CH_HANDLE(handle);
    HCCL_INFO("[%s] plugin channel created, handle[0x%llx].", __func__, handle);
    return HCCL_SUCCESS;
}

} // namespace

HcommResult CheckUbAttr(HcommChannelDesc& channelDesc, [[maybe_unused]] CommEngine engine)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBC_TP
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UBOE
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UB_RTP
        && channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UB_CTP) {
        return HCCL_SUCCESS;
    }

    // 暂不支持UBOE场景下配置SqDepth
    if (channelDesc.remoteEndpoint.protocol == COMM_PROTOCOL_UBOE) {
        return HCCL_SUCCESS;
    }

    // check sqDepth
    if (channelDesc.ubAttr.sqDepth == UB_SQ_DEPTH_NOT_SET) {
        HCCL_INFO("[%s] use default ubAttr.sqDepth.", __func__);
        return HCCL_SUCCESS;
    }

    // channelDesc.ubAttr.sqDepth调整到2的整数次幂
    auto GetNextPowerOfTwo = [](uint32_t n) -> uint32_t {
        n--;
        n |= n >> 1;
        n |= n >> 2;
        n |= n >> 4;
        n |= n >> 8;
        n |= n >> 16;
        return n + 1;
    };

    channelDesc.ubAttr.sqDepth = GetNextPowerOfTwo(channelDesc.ubAttr.sqDepth);

    return HCCL_SUCCESS;
}

HcommResult CheckUbMemAttr(HcommChannelDesc& channelDesc)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_UB_MEM) {
        return HCOMM_SUCCESS;
    }

    if (channelDesc.ubMemAttr.pathMode == UB_PATH_MODE_DEFAULT) {
        HCCL_INFO("[%s] use default ubMemAttr.pathMode, set to 0.", __func__);
        channelDesc.ubMemAttr.pathMode = 0;
        return HCOMM_SUCCESS;
    }

    if (channelDesc.ubMemAttr.pathMode > UB_PATH_MODE_MAX) {
        HCCL_ERROR("[%s] invalid ubMemAttr.pathMode[%u], should be 0 ~ 2.", __func__, channelDesc.ubMemAttr.pathMode);
        return HCCL_E_PARA;
    }
    return HCOMM_SUCCESS;
}

HcommResult CheckChannelDescQos(const HcommChannelDesc& channelDesc)
{
    if (channelDesc.qos == HCCL_COMM_QOS_CONFIG_NOT_SET
        || (channelDesc.qos >= EnvConfig::HCCL_QOS_MIN && channelDesc.qos <= EnvConfig::HCCL_QOS_MAX)) {
        return HCOMM_SUCCESS;
    }
    HCCL_ERROR(
        "[%s] invalid channelDesc.qos[%u], must be 0xFFFFFFFF or in [%u,%u]", __func__, channelDesc.qos,
        EnvConfig::HCCL_QOS_MIN, EnvConfig::HCCL_QOS_MAX);
    return HCOMM_E_PARA;
}

HcommResult CheckRoceAttr(HcommChannelDesc& channelDesc, EndpointLocType localLocType)
{
    if (channelDesc.remoteEndpoint.protocol != COMM_PROTOCOL_ROCE) {
        return HCCL_SUCCESS;
    }

    if (channelDesc.roceAttr.queueNum == INVALID_UINT) {
        channelDesc.roceAttr.queueNum = 1;
        HCCL_INFO("[%s] set roceAttr.queueNum to 1.", __func__);
    }

    if (channelDesc.roceAttr.cqAttrFlags == INVALID_UINT) {
        channelDesc.roceAttr.cqAttrFlags = 0;
        HCCL_INFO("[%s] set roceAttr.cqAttrFlags to 0.", __func__);
    }

    // 校验RoCE队列深度：0/INVALID_UINT表示使用默认值；否则需为2^n且在[64, 32K]范围内，不满足则报错退出
    auto validateQueueDepth = [](uint32_t depth, const char* fieldName) -> HcclResult {
        constexpr uint32_t kRoceQueueDepthMin = 64U;
        constexpr uint32_t kRoceQueueDepthMax = 32768U;
        if (depth == 0U || depth == INVALID_UINT) {
            return HCCL_SUCCESS; // 使用默认值
        }
        if (depth < kRoceQueueDepthMin || depth > kRoceQueueDepthMax || (depth & (depth - 1U)) != 0U) {
            HCCL_ERROR(
                "[CheckRoceAttr] invalid roceAttr.%s[%u], should be power of 2 and in [%u, %u], or 0 for default.",
                fieldName, depth, kRoceQueueDepthMin, kRoceQueueDepthMax);
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    };
    CHK_RET(validateQueueDepth(channelDesc.roceAttr.sqDepth, "sqDepth"));
    CHK_RET(validateQueueDepth(channelDesc.roceAttr.scqDepth, "scqDepth"));

    return ApplyRoceQosCompatToSlTc(channelDesc, localLocType);
}

namespace {
void ApplyHcommChannelDescV1Fields(const HcommChannelDesc& channelDesc, HcommChannelDesc& channelDescFinal)
{
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION_ONE) {
        return;
    }

    channelDescFinal.remoteEndpoint = channelDesc.remoteEndpoint;
    channelDescFinal.notifyNum = channelDesc.notifyNum;
    channelDescFinal.exchangeAllMems = channelDesc.exchangeAllMems;
    channelDescFinal.memHandles = channelDesc.memHandles;
    channelDescFinal.memHandleNum = channelDesc.memHandleNum;
    channelDescFinal.socket = channelDesc.socket;
    channelDescFinal.role = channelDesc.role;
    channelDescFinal.port = channelDesc.port;
}

HcommResult ProcessHcommChannelDescs(const HcommChannelDesc& channelDesc, HcommChannelDesc& channelDescFinal)
{
    if (channelDesc.header.size < sizeof(CommAbiHeader)) {
        HCCL_ERROR("[%s] invalid channelDesc.header.size[%u].", __func__, channelDesc.header.size);
        return HCCL_E_PARA;
    }

    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR(
            "[%s] channelDesc.header.magicWord[0x%08x] is invalid, expected[0x%08x].", __func__,
            channelDesc.header.magicWord, channelDescFinal.header.magicWord);
        return HCCL_E_PARA;
    }

    const uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ? channelDescFinal.header.size :
                                                                                        channelDesc.header.size)
                              - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(
        ReinterpretAs<uint8_t*>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        ReinterpretAs<const uint8_t*>(&channelDesc) + sizeof(CommAbiHeader), copySize));
    ApplyHcommChannelDescV1Fields(channelDesc, channelDescFinal);
    if (channelDesc.header.version > HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING(
            "The version of provided [%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        HCCL_RUN_WARNING(
            "The version of provided [%u] is lower than the current version[%u], "
            "configurations supported by later versions will be ignored.",
            channelDesc.header.version, HCOMM_CHANNEL_VERSION);
    }

    // qos：低版本时置默认值
    if (channelDesc.header.version <= HCOMM_CHANNEL_VERSION_ONE) {
        channelDescFinal.qos = 0xFFFFFFFFU;
    } else {
        channelDescFinal.qos = channelDesc.qos;
    }

    // v3：channelName，低版本时置 NULL
    constexpr uint32_t HCOMM_CHANNEL_VERSION_THREE = 3U;
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION_THREE) {
        channelDescFinal.channelName = nullptr;
    } else {
        channelDescFinal.channelName = channelDesc.channelName;
        if (channelDescFinal.channelName != nullptr
            && ReinterpretAs<uintptr_t>(channelDescFinal.channelName) == static_cast<uintptr_t>(-1)) {
            channelDescFinal.channelName = nullptr;
        }
    }

    if (channelDescFinal.channelName != nullptr) {
        size_t nameLen = strnlen(channelDescFinal.channelName, HCOMM_CHANNEL_NAME_MAX_LEN + 1);
        if (nameLen > HCOMM_CHANNEL_NAME_MAX_LEN) {
            HCCL_ERROR("[%s] channelName too long, max len[%u].", __func__, HCOMM_CHANNEL_NAME_MAX_LEN);
            return HCCL_E_PARA;
        }
    }

    // v4：roceAttr.srcPortList，低版本时 union 内该位置为脏数据，置 NULL
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        channelDescFinal.roceAttr.srcPortList = nullptr;
    } else {
        channelDescFinal.roceAttr.srcPortList = channelDesc.roceAttr.srcPortList;
    }

    // v4：roceAttr 队列深度（sqDepth/scqDepth），低版本时 union 内该位置为脏数据，置默认哨兵
    if (channelDesc.header.version < HCOMM_CHANNEL_VERSION) {
        channelDescFinal.roceAttr.sqDepth = INVALID_UINT;
        channelDescFinal.roceAttr.scqDepth = INVALID_UINT;
    }

    return HCOMM_SUCCESS;
}

HcommResult NormalizeHcommChannelDescs(
    EndpointHandle endpointHandle, HcommChannelDesc* channelDescs, uint32_t channelNum,
    std::vector<HcommChannelDesc>& channelDescFinals, CommEngine engine)
{
    channelDescFinals.clear();
    channelDescFinals.reserve(channelNum);
    EndpointLocType localLocType = ENDPOINT_LOC_TYPE_DEVICE;
    CHK_RET(static_cast<HcclResult>(GetEndPointLocType(endpointHandle, localLocType)));
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HcommChannelDesc channelDescFinal{};
        HcommResult ret = HcommChannelDescInit(&channelDescFinal, 1);
        if (ret != HCOMM_SUCCESS) {
            return ret;
        }
        ret = ProcessHcommChannelDescs(channelDescs[idx], channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] failed to normalize channelDesc[%u], ret[%d].", __func__, idx, ret);
            return ret;
        }
        ret = CheckChannelDescQos(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckChannelDescQos failed, ret[%d].", __func__, ret);
            return ret;
        }
        ret = CheckUbAttr(channelDescFinal, engine);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckUbAttr failed, ret[%d].", __func__, ret);
            return ret;
        }
        ret = CheckUbMemAttr(channelDescFinal);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckUbMemAttr failed, ret[%d].", __func__, ret);
            return ret;
        }
        ret = CheckRoceAttr(channelDescFinal, localLocType);
        if (ret != HCOMM_SUCCESS) {
            HCCL_ERROR("[%s] CheckRoceAttr failed, ret[%d].", __func__, ret);
            return ret;
        }

        channelDescFinals.push_back(channelDescFinal);
    }
    return HCOMM_SUCCESS;
}
} // namespace

// 集合通信使用，待归一到HcommChannelCreate
HcommResult HcommCollectiveChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels)
{
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(
        NormalizeHcommChannelDescs(endpointHandle, channelDescs, channelNum, channelDescFinals, engine)));
    auto startut = std::chrono::steady_clock::now();
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);
    HcommResult ret
        = ChannelProcess::CreateChannelsLoop(endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
    HCCL_INFO(
        "[%s] END. channelNum[%u], take time [%lld]us.", __func__, channelNum,
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startut).count());
    return ret;
}

HcommResult HcommChannelUpdateMemInfo(HcommMemHandle* memHandles, uint32_t memHandleNum, ChannelHandle channelHandle)
{
    CHK_PTR_NULL(memHandles);
    CHK_PRT_RET(
        (memHandleNum == 0), HCCL_ERROR("[%s] Invalid memHandleNum, memHandleNum is 0.", __func__), HCCL_E_PARA);
    return ChannelProcess::ChannelUpdateMemInfo(memHandles, memHandleNum, channelHandle);
}

HcommResult CreatePluginChannels(
    hcomm::Endpoint* endpoint, HcommChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    auto* epHolder = dynamic_cast<hcomm::PluginEndpointHolder*>(endpoint);
    CHK_PTR_NULL(epHolder);
    const NicPluginEntry* entry = epHolder->GetPluginEntry();
    CHK_PTR_NULL(entry);
    void* epCtx = epHolder->GetNicCtx();

    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        HcommResult ret = CreateOnePluginChannel(entry, epCtx, &channelDescs[idx], &channels[idx]);
        if (ret != HCCL_SUCCESS) {
            (void)RollbackPluginChannels(channels, idx);
            return ret;
        }
    }

    return HCCL_SUCCESS;
}

HcommResult HcommChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels)
{
    CHK_PTR_NULL(endpointHandle);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(
        NormalizeHcommChannelDescs(endpointHandle, channelDescs, channelNum, channelDescFinals, engine)));
    auto endpoint = HcommResMgr::GetInstance().GetEndpointMgr().Get(endpointHandle);
    auto startut = std::chrono::steady_clock::now();
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum);
    if (endpoint != nullptr && dynamic_cast<hcomm::PluginEndpointHolder*>(endpoint) != nullptr) {
        CHK_RET(
            static_cast<HcclResult>(CreatePluginChannels(endpoint, channelDescFinals.data(), channelNum, channels)));
        HCCL_INFO(
            "[%s] END. channelNum[%u], take time [%lld]us.", __func__, channelNum,
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startut).count());
        return HCCL_SUCCESS;
    }
    (void)HcommResMgrInit();
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle* targetChannels = hostChannelHandles.data();
    CHK_RET(ChannelProcess::CreateChannelsLoop(
        endpointHandle, engine, channelDescFinals.data(), channelNum, targetChannels));
    CHK_RET(
        ChannelProcess::PrepareUserChannels(targetChannels, channels, channelDescFinals.data(), channelNum, engine));
    HCCL_INFO(
        "[%s] END. channelNum[%u], take time [%lld]us.", __func__, channelNum,
        std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startut).count());
    HCCL_RUN_INFO(
        "[%s] channels created, channelNum[%u], engine[%s]", __func__, channelNum,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
    return HCCL_SUCCESS;
}

HcommResult HcommChannelGet(ChannelHandle channelHandle, void** channel)
{
    CHK_PTR_NULL(channel);
    return ChannelProcess::ChannelGet(channelHandle, channel);
}

HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList)
{
    CHK_PTR_NULL(channelList);
    CHK_PTR_NULL(statusList);
    CHK_PRT_RET((listNum == 0), HCCL_ERROR("[%s] Invalid listNum, listNum[%u]", __func__, listNum), HCCL_E_PARA);

    if (IS_PLUGIN_HANDLE(channelList[0])) {
        for (uint32_t i = 0; i < listNum; i++) {
            auto* ch = CHANNEL_FROM_HANDLE(channelList[i]);
            CHK_PTR_NULL(ch);
            int32_t status = 0;
            HcommResult ret = static_cast<HcommResult>(ch->GetNicOps()->getStatus(ch->GetNicCtx(), &status));
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s] plugin getStatus failed, idx[%u], ret[%d].", __func__, i, ret);
                return ret;
            }
            statusList[i] = status;
        }
        return HCCL_SUCCESS;
    } else {
        (void)HcommResMgrInit();
        std::vector<CommEngine> engines;
        std::vector<HcommChannelDesc> channelDescFinals;
        std::vector<ChannelStatus> internalStatus(listNum);
        auto startut = std::chrono::steady_clock::now();
        HcclResult ret
            = ChannelProcess::GetChannelsInfo(channelList, listNum, engines, channelDescFinals, internalStatus);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] GetChannelsInfo failed, ret[%d]", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        ret = ChannelProcess::HandleStatusByEngine(
            channelList, listNum, engines, channelDescFinals, internalStatus, statusList);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[%s] HandleStatusByEngine failed, ret[%d]", __func__, ret);
            return HCCL_E_INTERNAL;
        }
        HCCL_INFO(
            "[%s] END. listNum[%u], take time [%lld]us.", __func__, listNum,
            std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::steady_clock::now() - startut).count());
        return HCCL_SUCCESS;
    }
}

HcommResult HcommChannelGetNotifyNum(ChannelHandle channelHandle, uint32_t* notifyNum)
{
    CHK_PTR_NULL(notifyNum);
    return ChannelProcess::ChannelGetNotifyNum(channelHandle, notifyNum);
}

static HcclResult DestroyBuiltinChannels(std::vector<ChannelHandle>& builtinChannels)
{
    // 即使 plugin channel 销毁失败，也需继续销毁 builtin channel，避免 RDMA/jetty 资源泄漏
    // 及 SharedJettyMgr 残留记录永久阻塞 Endpoint 销毁。最终返回首个错误（优先 plugin 错误）。
    HcclResult builtinRet = HCCL_SUCCESS;
    if (builtinChannels.empty()) {
        return builtinRet;
    }
    builtinRet = ChannelProcess::ChannelDestroy(
        builtinChannels.data(), builtinChannels.size(), AicpuTsChannelHelper::GetBinHandle());
    // 无论 ChannelDestroy 成功与否都注销 SharedJettyMgr 记录：
    // 成功时正常清理；失败时 channel 已不可用，若不注销会永久阻塞 Endpoint 销毁。
    if (builtinRet != HCCL_SUCCESS) {
        HCCL_WARNING(
            "[%s] ChannelDestroy failed, ret[%d], force unregister shared jetty channels.", __func__, builtinRet);
    }
    (void)hcomm::SharedJettyMgr::GetInstance().UnregisterChannels(builtinChannels.data(), builtinChannels.size());
    return builtinRet;
}

HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum)
{
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    if (IS_PLUGIN_HANDLE(channels[0])) {
        for (uint32_t idx = 0; idx < channelNum; ++idx) {
            auto* ch = CHANNEL_FROM_HANDLE(channels[idx]);
            HcclResult ret = ChannelProcess::RemovePluginChannelFromMap(ReinterpretAs<ChannelHandle>(ch));
            if (ret != HCCL_SUCCESS) {
                HCCL_WARNING(
                    "[%s] plugin channel not found in map during destroy, handle[0x%llx], ret[%d].", __func__,
                    channels[idx], ret);
            }
        }
        return HCCL_SUCCESS;
    }
    (void)HcommResMgrInit();
    std::vector<ChannelHandle> builtinChannels;
    builtinChannels.reserve(channelNum);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        builtinChannels.push_back(channels[idx]);
    }
    return static_cast<HcommResult>(DestroyBuiltinChannels(builtinChannels));
}

HcommResult HcommChannelConfigCreate(HcommChannelConfig* config)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigCreate(config));
}

HcommResult HcommChannelConfigDestroy(HcommChannelConfig config)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigDestroy(config));
}

HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value)
{
    return static_cast<HcommResult>(hcomm::ChannelConfigSetInt(config, type, value));
}

static bool IsUbProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_RTP;
}

static HcclResult ValidateSharedQueueConfig(const std::vector<HcommChannelDesc>& channelDescs)
{
    for (uint32_t i = 0; i < channelDescs.size(); ++i) {
        CommProtocol protocol = channelDescs[i].remoteEndpoint.protocol;
        if (!IsUbProtocol(protocol)) {
            HCCL_ERROR(
                "[%s] IS_SHARED_QUEUE only supports UB protocols (UB_CTP/UBC_TP/UB_RTP)"
                "channelDesc[%u] protocol[%d].",
                __func__, i, protocol);
            return HCCL_E_NOT_SUPPORT;
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult CreateAndRegisterSharedQueueBuiltinChannels(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescFinals, uint32_t channelNum,
    ChannelHandle* channels)
{
    // 共享模式建链流程与 HcommChannelCreate 一致：CreateChannelsLoop 传 isSharedQueue=true，
    // channel 的 BuildConnection 据此走共享 jetty 复用路径；PrepareUserChannels 完成 AICPU/AIV 预分配。
    std::vector<ChannelHandle> hostChannelHandles(channelNum);
    ChannelHandle* targetChannels = hostChannelHandles.data();

    CHK_RET(ChannelProcess::CreateChannelsLoop(
        endpointHandle, engine, channelDescFinals, channelNum, targetChannels, true));
    HcclResult prepRet
        = ChannelProcess::PrepareUserChannels(targetChannels, channels, channelDescFinals, channelNum, engine);
    if (prepRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] PrepareUserChannels failed, ret[%d], destroying created channels.", __func__, prepRet);
        (void)ChannelProcess::ChannelDestroy(targetChannels, channelNum, AicpuTsChannelHelper::GetBinHandle());
        return prepRet;
    }

    HcclResult regRet = hcomm::SharedJettyMgr::GetInstance().RegisterChannels(endpointHandle, channels, channelNum);
    if (regRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] failed to register shared jetty channels, ret[%d].", __func__, regRet);
        (void)ChannelProcess::ChannelDestroy(channels, channelNum, AicpuTsChannelHelper::GetBinHandle());
        return regRet;
    }
    return HCCL_SUCCESS;
}

HcommResult HcommChannelCreateWithConfig(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    HcommChannelConfig config, ChannelHandle* channels)
{
    CHK_PTR_NULL(endpointHandle);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0), HCCL_ERROR("[%s] Invalid channelNum, channelNum[%u]", __func__, channelNum), HCCL_E_PARA);
    HCCL_INFO(
        "[%s] START. endpointHandle[0x%llx], engine[%s], channelNum[%u], config[%p].", __func__, endpointHandle,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum, config);

    bool isSharedQueue = false;
    if (config != nullptr) {
        auto* cfg = static_cast<hcomm::HcommChannelConfigData*>(config);
        isSharedQueue = cfg->isSharedQueue;
    }

    // 非共享模式直接复用 HcommChannelCreate 流程，避免重复维护两套建链逻辑
    if (!isSharedQueue) {
        return HcommChannelCreate(endpointHandle, engine, channelDescs, channelNum, channels);
    }

    // 共享 jetty 仅支持 AIV 引擎：AICPU 等 channel 的 BuildConnection 不处理共享 jetty 路径，
    // 强行创建会导致 channel 注册到 SharedJettyMgr 但无实际 jetty 共享，多 channel 共用同一 SQ
    // 但 PI/CI 未协调，引发 WQE 覆盖、doorbell 不前进、notify 超时。
    if (engine != COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[%s] IS_SHARED_QUEUE currently only supports AIV engine, engine[%d].", __func__, static_cast<int>(engine));
        return HCCL_E_NOT_SUPPORT;
    }

    auto endpoint = HcommResMgr::GetInstance().GetEndpointMgr().Get(endpointHandle);
    if (endpoint != nullptr) {
        CHK_RET(RefreshEndpointContext(endpoint->GetEndpointDesc()));
    }
    (void)HcommResMgrInit();

    std::vector<HcommChannelDesc> channelDescFinals;
    CHK_RET(static_cast<HcclResult>(
        NormalizeHcommChannelDescs(endpointHandle, channelDescs, channelNum, channelDescFinals, engine)));
    // NormalizeHcommChannelDescs 内部已调 CheckUbAttr，此处仅补共享模式专有校验
    CHK_RET(ValidateSharedQueueConfig(channelDescFinals));

    HcclResult ret = CreateAndRegisterSharedQueueBuiltinChannels(
        endpointHandle, engine, channelDescFinals.data(), channelNum, channels);
    if (ret != HCCL_SUCCESS) {
        return static_cast<HcommResult>(ret);
    }

    HCCL_INFO("[%s] SUCCESS. isSharedQueue[%d], channelNum[%u].", __func__, isSharedQueue, channelNum);
    return HCCL_SUCCESS;
}

HcommResult
HcommChannelGetRemoteMems(ChannelHandle channelHandle, uint32_t* memNum, CommMem** remoteMem, char*** memInfos)
{
    CHK_PTR_NULL(remoteMem);
    CHK_PTR_NULL(memNum);
    CHK_PTR_NULL(memInfos);

    return ChannelProcess::ChannelGetRemoteMems(channelHandle, memNum, remoteMem, memInfos);
}

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

CcuResult HcommCcuGetRmtMemToken(ChannelHandle channelHandle, uint64_t srcVa, uint64_t* tokenInfo)
{
    if (channelHandle == 0) {
        HCCL_ERROR("[%s] failed, invalid channelHandle[%llu].", __func__, channelHandle);
        return CcuResult::CCU_E_PARA;
    }
    CCU_CHK_PTR_NULL(tokenInfo);
    // srcVa 为 0 不是已注册内存的合法地址。若不在此拦截，底层遍历 rmtBufferVec_ 匹配不到，
    // 只会返回 HCCL_E_NOT_FOUND 并被收敛为 CCU_E_INTERNAL，与接口约定（非法 srcVa 返回 CCU_E_PARA）不符。
    if (srcVa == 0) {
        HCCL_ERROR("[%s] failed, invalid srcVa[%llu].", __func__, srcVa);
        return CcuResult::CCU_E_PARA;
    }
    // CcuChannelGetRmtMemToken 返回内部 HcclResult，与本接口对外的 CcuResult 分属两套错误码体系，
    // 故不做数值透传（不使用 HCCL_TO_CCU_RET），非 HCCL_SUCCESS 一律收敛为 CCU_E_INTERNAL。
    // 具体失败原因（channelHandle 查不到、channel 非 CcuUrmaChannel、srcVa 未注册）
    // 已在 ChannelProcess::CcuChannelGetRmtMemToken 与 CcuTransport::CcuGetRmtMemToken 内部打印。
    return ChannelProcess::CcuChannelGetRmtMemToken(channelHandle, srcVa, *tokenInfo) == HCCL_SUCCESS ?
               CcuResult::CCU_SUCCESS :
               CcuResult::CCU_E_INTERNAL;
}

#ifdef __cplusplus
}
#endif // __cplusplus
