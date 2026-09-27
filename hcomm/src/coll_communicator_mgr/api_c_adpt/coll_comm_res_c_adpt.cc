/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "my_rank.h"
#include <algorithm>
#include <array>
#include <iterator>
#include <limits>
#include <functional>
#include "hccl_comm_pub.h"
#include "exception_handler.h"
#include "config_log.h"
#include "config/env_config.h"
#include "env_config/env_config_v2.h"

#include "coll_comm_mgr.h"
#include "hcclCommOp.h"
#include "channel_process.h"
#include "aicpu_ts_roce_channel_v2.h"
#include "dfx_dlprof_function.h"
#include "aiv_urma_channel.h"
#include "hccl_group.h"
#include "../resource_mgr/local/my_rank/comm_engine/kernel_launch/hccl_kernel_launch_aicpu.h"
#include "param_check_basic_v2.h"
#include "comm_engine_utils.h"
#include "rank_consistency_checker_v2.h"
#include "rank_table_crc_bridge.h"
#include "hccl_channel_config.h"
#include "shared_jetty_channel_pool.h"
#include "endpoints/endpoint_mgr.h"
#include "hcomm_res.h"
#include "channel_config.h"
#include "hcclCommDfx.h"
#include "coll_comm_res_c_adpt.h"
#include "roce_channel_desc_configurator.h"
#include "common/loggers/channel_logger.h"

using namespace hccl;
/**
 * @note 职责：集合通信的通信域资源管理的C接口的C到C++适配
 */

/**
 * @note C接口适配参考示例
 * @code {.c}
 * HcclResult HcclThreadAcquire(HcclComm comm, CommEngine engine, uint32_t threadNum,
 *     uint32_t notifyNumPerThread, ThreadHandle *threads) {
 *     return HCCL_SUCCESS;
 * }
 * @endcode
 */

constexpr uint32_t HCCL_CHANNEL_VERSION_ONE = 1;
constexpr uint32_t MULTIPLE = 4;                // 用于A5判断TC是否为4的倍数
constexpr uint32_t TC_MAX = 255;                // TC的最大值（不区分芯片类型）
constexpr uint32_t RETRY_INTERVAL_MIN = 5u;     // retryInterval范围的最小值（不区分芯片类型）
constexpr uint32_t A5_RETRY_INTERVAL_MAX = 24u; // A5的retryInterval范围的最大值
constexpr uint32_t RETRY_CNT_MIN = 1u;          // retryCnt范围的最小值（不区分芯片类型）
constexpr uint32_t RETRY_CNT_MAX = 7u;          // retryCnt范围的最大值（不区分芯片类型）
constexpr uint32_t SL_MAX = 7u;                 // sl范围的最大值，sl即serviceLevel（不区分芯片类型）
constexpr uint32_t TC_DEFAULT = 0xFFFFFFFFu;    // TC的默认值（不区分芯片类型）
constexpr uint32_t SL_DEFAULT = 0xFFFFFFFFu;    // SL的默认值（不区分芯片类型）

static u32 ResolveQueueNum(const Hccl::EnvRdmaConfig& rdmaConfig, const HcclChannelDesc& channelDesc)
{
    if (channelDesc.roceAttr.queueNum != INVALID_UINT) { // 用户有配置qp数量，使用用户配置的
        return channelDesc.roceAttr.queueNum;
    }

    if (channelDesc.channelProtocol == COMM_PROTOCOL_ROCE
        && channelDesc.localEndpoint.loc.locType == ENDPOINT_LOC_TYPE_HOST) {
        u32 hostQpCount = 0;
        RoceChannelDescConfigurator::ReadHostNicMultiQpCount(hostQpCount);
        if (hostQpCount > 0) {
            return hostQpCount;
        }
    }

    const auto& qpSrcPortConfig = rdmaConfig.GetMultiQpSrcPortConfig();
    const CommAddr& localCommAddr = channelDesc.localEndpoint.commAddr;
    const CommAddr& remoteCommAddr = channelDesc.remoteEndpoint.commAddr;
    char localIpStr[INET6_ADDRSTRLEN] = {0};
    char remoteIpStr[INET6_ADDRSTRLEN] = {0};
    s32 localFamily = (localCommAddr.type == COMM_ADDR_TYPE_IP_V6) ? AF_INET6 : AF_INET;
    s32 remoteFamily = (remoteCommAddr.type == COMM_ADDR_TYPE_IP_V6) ? AF_INET6 : AF_INET;
    const void* localSrc = (localFamily == AF_INET6) ? static_cast<const void*>(&localCommAddr.addr6) :
                                                       static_cast<const void*>(&localCommAddr.addr);
    const void* remoteSrc = (remoteFamily == AF_INET6) ? static_cast<const void*>(&remoteCommAddr.addr6) :
                                                         static_cast<const void*>(&remoteCommAddr.addr);
    (void)inet_ntop(localFamily, localSrc, localIpStr, sizeof(localIpStr));
    (void)inet_ntop(remoteFamily, remoteSrc, remoteIpStr, sizeof(remoteIpStr));
    Hccl::IpAddress localIp(localIpStr, localFamily);
    Hccl::IpAddress remoteIp(remoteIpStr, remoteFamily);
    // 根据ip对，查HCCL_RDMA_QP_PORT_CONFIG_PATH环境变量对应的源端口号
    u32 srcPortNum = Hccl::GetMultiQpPortsNumByIpPair(qpSrcPortConfig, localIp, remoteIp);
    if (srcPortNum > 0) { // 查看源端口号是否有配置，有则使用
        return srcPortNum;
    }
    return rdmaConfig.GetRdmaQueueNum();
}

static void FillChannelDescFinal(
    hccl::CommConfig commConfig, const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal,
    bool isCommunicatorV2)
{
    if (isCommunicatorV2) { // A5
        auto& rdmaConfig = Hccl::EnvConfig::GetInstance().GetRdmaConfig();
        channelDescFinal.roceAttr.retryCnt = (channelDesc.roceAttr.retryCnt == INVALID_UINT) ?
                                                 rdmaConfig.GetRdmaRetryCnt() :
                                                 channelDesc.roceAttr.retryCnt;
        channelDescFinal.roceAttr.retryInterval = (channelDesc.roceAttr.retryInterval == INVALID_UINT) ?
                                                      rdmaConfig.GetRdmaTimeOut() :
                                                      channelDesc.roceAttr.retryInterval;
        channelDescFinal.roceAttr.tc = static_cast<uint8_t>(
            (commConfig.GetConfigTrafficClass() == INVALID_UINT) ? rdmaConfig.GetRdmaTrafficClass() :
                                                                   commConfig.GetConfigTrafficClass());
        channelDescFinal.roceAttr.sl = static_cast<uint8_t>(
            (commConfig.GetConfigServiceLevel() == INVALID_UINT) ? rdmaConfig.GetRdmaServerLevel() :
                                                                   commConfig.GetConfigServiceLevel());
        channelDescFinal.roceAttr.queueNum = ResolveQueueNum(rdmaConfig, channelDesc);
        if (channelDesc.roceAttr.tc != 0xFF || channelDesc.roceAttr.sl != 0xFF) {
            HCCL_RUN_WARNING(
                "[FillChannelDescFinal] ignore HcclChannelDesc tc/sl, actually used tc[%u], sl[%u]",
                channelDescFinal.roceAttr.tc, channelDescFinal.roceAttr.sl);
        }
    } else {
        channelDescFinal.roceAttr.retryCnt = (channelDesc.roceAttr.retryCnt == INVALID_UINT) ?
                                                 EnvConfig::GetExternalInputRdmaRetryCnt() :
                                                 channelDesc.roceAttr.retryCnt;
        channelDescFinal.roceAttr.retryInterval = (channelDesc.roceAttr.retryInterval == INVALID_UINT) ?
                                                      EnvConfig::GetExternalInputRdmaTimeOut() :
                                                      channelDesc.roceAttr.retryInterval;
        channelDescFinal.roceAttr.tc = (channelDesc.roceAttr.tc == 0xFF) ?
                                           EnvConfig::GetExternalInputRdmaTrafficClass() :
                                           channelDesc.roceAttr.tc;
        channelDescFinal.roceAttr.sl = (channelDesc.roceAttr.sl == 0xFF) ?
                                           EnvConfig::GetExternalInputRdmaServerLevel() :
                                           channelDesc.roceAttr.sl;
        channelDescFinal.roceAttr.queueNum = (channelDesc.roceAttr.queueNum == INVALID_UINT) ?
                                                 GetExternalInputQpsPerConnection() :
                                                 channelDesc.roceAttr.queueNum;
    }
}

static HcclResult CheckA5Config(hccl::CommConfig commConfig, const HcclChannelDesc& channelDesc)
{
    u32 tc = commConfig.GetConfigTrafficClass();
    CHK_PRT_RET(
        (tc != TC_DEFAULT) && (tc > TC_MAX || (tc % MULTIPLE != 0)),
        HCCL_ERROR(
            "[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaTrafficClass[%u], must be 0xFFFFFFFF or in "
            "[0,255] and a multiple of 4",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HCCL_E_PARA)), tc),
        HCCL_E_PARA);

    u32 sl = commConfig.GetConfigServiceLevel();
    CHK_PRT_RET(
        (sl != SL_DEFAULT) && (sl > SL_MAX),
        HCCL_ERROR(
            "[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaServiceLevel[%u], must be 0xFFFFFFFF or in [0,7]",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HCCL_E_PARA)), sl),
        HCCL_E_PARA);

    u32 retryInterval = channelDesc.roceAttr.retryInterval;
    CHK_PRT_RET(
        (retryInterval != INVALID_UINT)
            && (retryInterval < RETRY_INTERVAL_MIN || retryInterval > A5_RETRY_INTERVAL_MAX),
        HCCL_ERROR(
            "[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaRetryInterval[%u], must be 0xFFFFFFFF or in "
            "[5,24]",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HCCL_E_PARA)), retryInterval),
        HCCL_E_PARA);

    u32 retryCnt = channelDesc.roceAttr.retryCnt;
    CHK_PRT_RET(
        (retryCnt != INVALID_UINT) && (retryCnt < RETRY_CNT_MIN || retryCnt > RETRY_CNT_MAX),
        HCCL_ERROR(
            "[ProcessRoceChannelDesc]errNo[0x%016llx] invalid hcclRdmaRetryCnt[%u], must be 0xFFFFFFFF or in [1,7]",
            static_cast<unsigned long long>(HCCL_ERROR_CODE(HCCL_E_PARA)), retryCnt),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult
ProcessRoceChannelDesc(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hccl::hcclComm* hcclComm)
{
    bool isCommunicatorV2 = hcclComm->IsCommunicatorV2();
    hccl::CommConfig commConfig{}; // A5使用
    if (isCommunicatorV2) {        // A5
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        commConfig = collComm->GetCommConfig();
        CHK_RET(CheckA5Config(commConfig, channelDesc));
    }
    FillChannelDescFinal(commConfig, channelDesc, channelDescFinal, isCommunicatorV2);
    HCCL_INFO(
        "[%s]queueNum[%u], retryCnt[%u], retryInterval[%u], tc[%u], sl[%u]", __func__,
        channelDescFinal.roceAttr.queueNum, channelDescFinal.roceAttr.retryCnt, channelDescFinal.roceAttr.retryInterval,
        channelDescFinal.roceAttr.tc, channelDescFinal.roceAttr.sl);
    return HCCL_SUCCESS;
}

HcclResult ProcessUbChannelDesc(
    const HcclChannelDesc& channelDesc, const HcclChannelDesc& channelDescFinal, const hccl::hcclComm* hcclComm)
{
    (void)channelDescFinal;
    (void)hcclComm;

    if (channelDesc.channelProtocol != COMM_PROTOCOL_UB_CTP && channelDesc.channelProtocol != COMM_PROTOCOL_UBC_TP
        && channelDesc.channelProtocol != COMM_PROTOCOL_UBOE && channelDesc.channelProtocol != COMM_PROTOCOL_UB_RTP) {
        HCCL_ERROR(
            "[%s] unexpected channelProtocol[%d], expect UB_CTP/UBC_TP/UBOE/UB_RTP", __func__,
            static_cast<int>(channelDesc.channelProtocol));
        return HCCL_E_PARA;
    }
    HCCL_INFO(
        "[%s] channelProtocol[%d] ub comm-domain qos applied in HcommChannelDesc::qos when converting (HcclChannelDesc "
        "has no qos field)",
        __func__, static_cast<int>(channelDesc.channelProtocol));
    return HCCL_SUCCESS;
}

HcclResult
ProcessHcclChannelDesc(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hccl::hcclComm* hcclComm)
{
    channelDescFinal.remoteRank = channelDesc.remoteRank;
    channelDescFinal.channelProtocol = channelDesc.channelProtocol;
    channelDescFinal.localEndpoint = channelDesc.localEndpoint;
    channelDescFinal.remoteEndpoint = channelDesc.remoteEndpoint;
    channelDescFinal.notifyNum = channelDesc.notifyNum;
    channelDescFinal.memHandles = channelDesc.memHandles;
    channelDescFinal.memHandleNum = channelDesc.memHandleNum;

    // 根据协议类型拷贝union中的相应成员
    switch (channelDesc.channelProtocol) {
        case COMM_PROTOCOL_HCCS:
        case COMM_PROTOCOL_HCCS_ONLY:
        case COMM_PROTOCOL_PCIE:
        case COMM_PROTOCOL_SIO:
            break;
        case COMM_PROTOCOL_UB_MEM:
            channelDescFinal.ubMemAttr.pathMode = channelDesc.ubMemAttr.pathMode;
            HCCL_INFO("[%s] ubMemAttr.pathMode[%u]", __func__, channelDescFinal.ubMemAttr.pathMode);
            break;
        case COMM_PROTOCOL_UB_CTP:
        case COMM_PROTOCOL_UBC_TP:
        case COMM_PROTOCOL_UBOE:
        case COMM_PROTOCOL_UB_RTP:
            return ProcessUbChannelDesc(channelDesc, channelDescFinal, hcclComm);
        case COMM_PROTOCOL_ROCE:
            return ProcessRoceChannelDesc(channelDesc, channelDescFinal, hcclComm);
        default: {
            auto ProtocolToString = [](const CommProtocol proto) -> const char* {
                switch (proto) {
                    case COMM_PROTOCOL_HCCS:
                        return "COMM_PROTOCOL_HCCS";
                    case COMM_PROTOCOL_PCIE:
                        return "COMM_PROTOCOL_PCIE";
                    case COMM_PROTOCOL_SIO:
                        return "COMM_PROTOCOL_SIO";
                    case COMM_PROTOCOL_UB_CTP:
                        return "COMM_PROTOCOL_UB_CTP";
                    case COMM_PROTOCOL_UB_MEM:
                        return "COMM_PROTOCOL_UB_MEM";
                    case COMM_PROTOCOL_ROCE:
                        return "COMM_PROTOCOL_ROCE";
                    case COMM_PROTOCOL_UBC_TP:
                        return "COMM_PROTOCOL_UBC_TP";
                    case COMM_PROTOCOL_UBOE:
                        return "COMM_PROTOCOL_UBOE";
                    case COMM_PROTOCOL_UB_RTP:
                        return "COMM_PROTOCOL_UB_RTP";
                    case COMM_PROTOCOL_HCCS_ONLY:
                        return "COMM_PROTOCOL_HCCS_ONLY";
                    default:
                        return "UNKNOWN_PROTOCOL";
                }
            };
            HCCL_ERROR(
                "[%s] Unsupported protocol[%s] found in HcclChannelDesc.", __func__,
                ProtocolToString(channelDesc.channelProtocol));
            return HCCL_E_PARA;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult
ProcessHcclResPackReq(const HcclChannelDesc& channelDesc, HcclChannelDesc& channelDescFinal, hccl::hcclComm* hcclComm)
{
    if (channelDesc.header.size < channelDescFinal.header.size) {
        // 需要前向兼容HcclChannelDesc，末尾部分字段不支持处理
    } else if (channelDesc.header.size > channelDescFinal.header.size) {
        // 需要后向向兼容HcclChannelDesc，末尾部分字段会被忽略
    }

    if (channelDesc.header.magicWord != channelDescFinal.header.magicWord) {
        HCCL_ERROR(
            "[%s]channelDescFinal.header.magicWord[%u] not equal to channelDesc.header.magicWord[%u]", __func__,
            channelDescFinal.header.magicWord, channelDesc.header.magicWord);
        return HCCL_E_PARA;
    }

    uint32_t copySize = (channelDescFinal.header.size < channelDesc.header.size ? channelDescFinal.header.size :
                                                                                  channelDesc.header.size)
                        - sizeof(CommAbiHeader);
    CHK_SAFETY_FUNC_RET(memcpy_s(
        reinterpret_cast<uint8_t*>(&channelDescFinal) + sizeof(CommAbiHeader), copySize,
        reinterpret_cast<const uint8_t*>(&channelDesc) + sizeof(CommAbiHeader), copySize));

    if (channelDesc.header.version >= HCCL_CHANNEL_VERSION_ONE) {
        CHK_RET(ProcessHcclChannelDesc(channelDesc, channelDescFinal, hcclComm));
    }

    if (channelDesc.header.version > HCCL_CHANNEL_VERSION) {
        // 传入的版本高于当前版本，警告不支持的配置项将被忽略
        HCCL_WARNING(
            "The version of provided [%u] is higher than the current version[%u], "
            "unsupported configuration will be ignored.",
            channelDesc.header.version, HCCL_CHANNEL_VERSION);
    } else if (channelDesc.header.version < HCCL_CHANNEL_VERSION) {
        // 传入的版本低于当前版本，警告高版本支持的配置项将被忽略
        HCCL_WARNING(
            "The version of provided [%u] is lower than the current version[%u], "
            "configurations supported by later versions will be ignored.",
            channelDesc.header.version, HCCL_CHANNEL_VERSION);
    }

    // 如果扩展到version=2后
    // 1) 在底层为新的结构体和版本（version为2）上，会正常执行下面的判断处理逻辑；
    // 2) 在底层为旧的结构体和版本（version为1）上，下面的逻辑没有，version的2 > 1的部分会被忽略掉；
    if (channelDesc.header.version >= 2) {
    }

    return HCCL_SUCCESS;
}

static HcclResult
BuildAivDeviceChannelEntity(const HcclChannelDesc& channelDesc, ChannelHandle hostChannel, ChannelHandle& deviceChannel)
{
    void* channel = nullptr;
    CHK_RET(hcomm::ChannelProcess::ChannelGet(hostChannel, &channel));
    hcomm::Channel* baseChannel = static_cast<hcomm::Channel*>(channel);
    CHK_PTR_NULL(baseChannel);

    if (channelDesc.channelProtocol == COMM_PROTOCOL_ROCE) {
        auto* aicpuTsRoceChannelV2 = dynamic_cast<hcomm::AicpuTsRoceChannelV2*>(baseChannel);
        CHK_PTR_NULL(aicpuTsRoceChannelV2);
        HCCL_INFO(
            "[%s] build AIV direct device channel by AICPU+Host RoCE flow, protocol[%d], "
            "hostHandle[0x%llx]",
            __func__, channelDesc.channelProtocol, static_cast<unsigned long long>(hostChannel));
        CHK_RET(aicpuTsRoceChannelV2->BuildAndGetDevChannelEntity(&deviceChannel));
        return HCCL_SUCCESS;
    }

    if (channelDesc.channelProtocol == COMM_PROTOCOL_UB_CTP || channelDesc.channelProtocol == COMM_PROTOCOL_UBC_TP
        || channelDesc.channelProtocol == COMM_PROTOCOL_UB_RTP) {
        auto* aivUrmaChannel = dynamic_cast<hcomm::AivUrmaChannel*>(baseChannel);
        CHK_PTR_NULL(aivUrmaChannel);
        HCCL_INFO(
            "[%s] build AIV direct device channel by AIV+URMA flow, protocol[%d], "
            "hostHandle[0x%llx]",
            __func__, channelDesc.channelProtocol, static_cast<unsigned long long>(hostChannel));
        void* devChannelEntity = nullptr;
        CHK_RET(aivUrmaChannel->BuildChannelEntityToDevice(&devChannelEntity));
        CHK_PTR_NULL(devChannelEntity);
        deviceChannel = static_cast<ChannelHandle>(reinterpret_cast<uintptr_t>(devChannelEntity));
        return HCCL_SUCCESS;
    }

    HCCL_ERROR("[%s] protocol[%d] is not AIV direct channel protocol", __func__, channelDesc.channelProtocol);
    return HCCL_E_PARA;
}

static HcclResult ConvertAivChannelHandlesToDevicePtrs(
    CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    if (engine != COMM_ENGINE_AIV) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> hostChannels(channels, channels + channelNum);
    std::vector<ChannelHandle> deviceChannels(hostChannels);
    std::vector<ChannelHandle> mappedDeviceChannels;
    std::vector<ChannelHandle> mappedHostChannels;
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        if (channelDescs[idx].channelProtocol != COMM_PROTOCOL_ROCE
            && channelDescs[idx].channelProtocol != COMM_PROTOCOL_UB_CTP
            && channelDescs[idx].channelProtocol != COMM_PROTOCOL_UBC_TP
            && channelDescs[idx].channelProtocol != COMM_PROTOCOL_UB_RTP) {
            continue;
        }
        CHK_RET(BuildAivDeviceChannelEntity(channelDescs[idx], hostChannels[idx], deviceChannels[idx]));
        mappedDeviceChannels.emplace_back(deviceChannels[idx]);
        mappedHostChannels.emplace_back(hostChannels[idx]);
        HCCL_INFO(
            "[%s] convert AIV channel success, idx[%u], protocol[%d], hostHandle[0x%llx], devEntity[0x%llx]", __func__,
            idx, channelDescs[idx].channelProtocol, static_cast<unsigned long long>(hostChannels[idx]),
            static_cast<unsigned long long>(deviceChannels[idx]));
    }

    if (!mappedDeviceChannels.empty()) {
        CHK_RET(hcomm::ChannelProcess::RegisterChannelD2HMap(
            mappedDeviceChannels.data(), mappedHostChannels.data(),
            static_cast<uint32_t>(mappedDeviceChannels.size())));
    }

    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        channels[idx] = deviceChannels[idx];
    }
    return HCCL_SUCCESS;
}
static bool IsUbUrmaChannelProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UBOE
           || protocol == COMM_PROTOCOL_UB_RTP;
}

static bool HasUbUrmaChannel(const std::vector<HcclChannelDesc>& channelDescFinals)
{
    for (const HcclChannelDesc& channelDesc : channelDescFinals) {
        if (IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
            return true;
        }
    }
    return false;
}

static void AppendUniqueMemHandle(std::vector<HcclMemHandle>& mergedHandles, HcclMemHandle memHandle)
{
    if (memHandle == nullptr) {
        return;
    }
    if (std::find(mergedHandles.begin(), mergedHandles.end(), memHandle) == mergedHandles.end()) {
        mergedHandles.emplace_back(memHandle);
    }
}

static HcclResult MergeSymmetricMemHandles(
    HcclChannelDesc& channelDesc, const std::vector<HcclMemHandle>& symmetricMemHandles,
    std::vector<HcclMemHandle>& mergedHandles)
{
    if (!IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
        return HCCL_SUCCESS;
    }
    // 保留通道原有的业务内存句柄，再与对称内存句柄去重合并，避免直接覆盖调用方入参。
    if (channelDesc.memHandleNum != 0) {
        CHK_PTR_NULL(channelDesc.memHandles);
        for (uint32_t handleIdx = 0; handleIdx < channelDesc.memHandleNum; ++handleIdx) {
            AppendUniqueMemHandle(mergedHandles, channelDesc.memHandles[handleIdx]);
        }
    }
    for (HcclMemHandle memHandle : symmetricMemHandles) {
        AppendUniqueMemHandle(mergedHandles, memHandle);
    }
    CHK_PRT_RET(
        mergedHandles.size() > static_cast<size_t>(std::numeric_limits<uint32_t>::max()),
        HCCL_ERROR("[MergeSymmetricMemHandles] merged memHandleNum[%zu] exceeds uint32 max.", mergedHandles.size()),
        HCCL_E_PARA);
    channelDesc.memHandles = mergedHandles.data();
    channelDesc.memHandleNum = static_cast<uint32_t>(mergedHandles.size());
    return HCCL_SUCCESS;
}

static HcclResult BuildChannelSymMemHandles(
    hccl::CollComm* collComm, hccl::MyRank* myRank, const std::vector<HcclChannelDesc>& channelDescFinals,
    const std::vector<ChannelHandle>& existingChannelHandles, const std::vector<HcclMemHandle>& registeredSymMemHandles,
    std::vector<std::vector<HcclMemHandle>>& channelSymMemHandles, std::vector<bool>& channelSymMemAppended,
    size_t& appendedChannelCount)
{
    for (size_t idx = 0; idx < channelDescFinals.size(); ++idx) {
        if (!IsUbUrmaChannelProtocol(channelDescFinals[idx].channelProtocol)) {
            continue;
        }

        std::vector<std::string> remoteMemTags;
        if (existingChannelHandles[idx] == 0) {
            // 新建通道尚未交换过对称内存，需要携带本地全部已注册句柄。
            channelSymMemHandles[idx] = registeredSymMemHandles;
        } else {
            // 复用通道仅需携带其远端内存记录中缺失的句柄。
            CommMem* remoteMems = nullptr;
            uint32_t memNum = 0;
            // 读取已有通道上次双向交换后缓存的远端memTag，不会发起新的远端查询。
            CHK_RET(myRank->ChannelGetRemoteMems(existingChannelHandles[idx], &memNum, &remoteMems, remoteMemTags));
            // 各Rank的对称窗口使用相同memTag，据此筛出该通道本轮尚未交换的本地句柄。
            CHK_RET(collComm->GetRemoteMissingSymMemHandles(remoteMemTags, channelSymMemHandles[idx]));
        }
        HCCL_INFO(
            "[BuildChannelSymMemHandles] channelIdx[%zu], remoteRank[%u], existing[%d], "
            "remoteMemTagNum[%zu], missingSymmetricMemHandleNum[%zu].",
            idx, channelDescFinals[idx].remoteRank, existingChannelHandles[idx] != 0, remoteMemTags.size(),
            channelSymMemHandles[idx].size());
        if (!channelSymMemHandles[idx].empty()) {
            channelSymMemAppended[idx] = true;
            ++appendedChannelCount;
        }
    }
    return HCCL_SUCCESS;
}

static HcclResult ApplyChannelSymMemHandles(
    std::vector<HcclChannelDesc>& channelDescFinals,
    const std::vector<std::vector<HcclMemHandle>>& channelSymMemHandles, const std::vector<bool>& channelSymMemAppended,
    std::vector<std::vector<HcclMemHandle>>& mergedMemHandles)
{
    mergedMemHandles.clear();
    mergedMemHandles.resize(channelDescFinals.size());
    for (size_t idx = 0; idx < channelDescFinals.size(); ++idx) {
        if (channelSymMemAppended[idx]) {
            // mergedMemHandles持有channelDescFinals[idx].memHandles指向的内存，保证建链期间有效。
            CHK_RET(MergeSymmetricMemHandles(channelDescFinals[idx], channelSymMemHandles[idx], mergedMemHandles[idx]));
        }
    }
    return HCCL_SUCCESS;
}

// 建链前完成pending窗口注册，并按每条通道的新建或复用状态准备需要交换的对称内存句柄。
static HcclResult PrepareChannelSymMemHandles(
    hccl::CollComm* collComm, hccl::MyRank* myRank, CommEngine engine, std::vector<HcclChannelDesc>& channelDescFinals,
    std::vector<std::vector<HcclMemHandle>>& mergedMemHandles, std::vector<bool>& channelSymMemAppended)
{
    CHK_PTR_NULL(collComm);
    CHK_PTR_NULL(myRank);
    channelSymMemAppended.assign(channelDescFinals.size(), false);
    if (!HasUbUrmaChannel(channelDescFinals)) {
        return HCCL_SUCCESS;
    }

    // 对称内存窗口在建链时延迟注册，并加入本地持久化句柄索引。
    CHK_RET(collComm->RegisterPendingSymmetricMemHandles());

    std::vector<HcclMemHandle> registeredSymMemHandles;
    // 获取本地全部已注册句柄，包括历史注册的句柄，供新建通道使用。
    CHK_RET(collComm->GetAllRegisteredSymMemHandles(registeredSymMemHandles));
    if (registeredSymMemHandles.empty()) {
        return HCCL_SUCCESS;
    }

    std::vector<ChannelHandle> existingChannelHandles(channelDescFinals.size(), 0);
    // QueryChannels返回0表示通道未创建，返回有效句柄表示可以复用已有通道。
    CHK_RET(myRank->QueryChannels(
        engine, channelDescFinals.data(), static_cast<uint32_t>(channelDescFinals.size()),
        existingChannelHandles.data()));

    std::vector<std::vector<HcclMemHandle>> channelSymMemHandles(channelDescFinals.size());
    size_t appendedChannelCount = 0;
    CHK_RET(BuildChannelSymMemHandles(
        collComm, myRank, channelDescFinals, existingChannelHandles, registeredSymMemHandles, channelSymMemHandles,
        channelSymMemAppended, appendedChannelCount));
    if (appendedChannelCount == 0) {
        return HCCL_SUCCESS;
    }

    CHK_RET(
        ApplyChannelSymMemHandles(channelDescFinals, channelSymMemHandles, channelSymMemAppended, mergedMemHandles));
    HCCL_INFO(
        "[PrepareChannelSymMemHandles] prepare symmetric memHandles success, channelNum[%zu], "
        "appendedChannelCount[%zu], protocols[UB_CTP/UBC_TP/UBOE/UB_RTP].",
        channelDescFinals.size(), appendedChannelCount);
    return HCCL_SUCCESS;
}

static HcclResult UpdateSymmetricRemoteMems(
    hccl::CollComm* collComm, const hccl::MyRank* myRank, const std::vector<HcclChannelDesc>& channelDescFinals,
    const ChannelHandle* channels, uint32_t channelNum, const std::vector<bool>& channelSymMemAppended)
{
    CHK_PTR_NULL(collComm);
    CHK_PTR_NULL(myRank);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        channelSymMemAppended.size() != channelNum,
        HCCL_ERROR(
            "[UpdateSymmetricRemoteMems] appended state size[%zu] does not match channelNum[%u].",
            channelSymMemAppended.size(), channelNum),
        HCCL_E_PARA);
    for (uint32_t idx = 0; idx < channelNum; ++idx) {
        // 仅对本次交换了新句柄的通道更新远端对称内存信息。
        if (!channelSymMemAppended[idx]) {
            continue;
        }
        const HcclChannelDesc& channelDesc = channelDescFinals[idx];
        if (!IsUbUrmaChannelProtocol(channelDesc.channelProtocol)) {
            continue;
        }
        CommMem* remoteMems = nullptr;
        uint32_t memNum = 0;
        std::vector<std::string> memTags;
        // CreateChannels完成后，从channel取回交换到的remoteMem/memTag并回填window。
        CHK_RET(myRank->ChannelGetRemoteMems(channels[idx], &memNum, &remoteMems, memTags));
        if (memNum == 0) {
            continue;
        }
        CHK_RET(collComm->UpdateSymmetricRemoteMem(channelDesc.remoteRank, remoteMems, memTags));
    }
    return HCCL_SUCCESS;
}

bool CheckCommEngine(const CommEngine engine, const uint32_t opExpansionMode)
{
    constexpr uint32_t DEFAULT_MODE = 0;
    constexpr uint32_t CCU_MS_MODE = 5;
    constexpr uint32_t CCU_SCHE_MODE = 6;
    if (engine == CommEngine::COMM_ENGINE_CCU) {
        return opExpansionMode == DEFAULT_MODE || opExpansionMode == CCU_MS_MODE || opExpansionMode == CCU_SCHE_MODE;
    }
    if (engine == COMM_ENGINE_RESERVED) {
        return false;
    }
    return true;
}

static bool IsAicpuEngine(CommEngine engine) { return engine == COMM_ENGINE_AICPU || engine == COMM_ENGINE_AICPU_TS; }

constexpr uint32_t CHANNEL_NUM_MAX = 1024 * 1024; // channel的默认限制最大为1024 * 1024

HcclResult RegisterToClusterMonitor(HcclComm comm)
{
    HCCL_INFO("[%s] START, comm[%p].", __func__, comm);
    CHK_PRT_RET(comm == nullptr, HCCL_ERROR("[%s] comm is null", __func__), HCCL_E_PTR);
    auto* hcclComm = static_cast<hccl::hcclComm*>(comm);
    CHK_PTR_NULL(hcclComm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] comm is not support", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    CHK_RET(CollCommMgr::GetInstance().GetClusterMonitor(collComm->GetDeviceLogicId()).RegisterToClusterMonitor(comm));
    HCCL_INFO("%s Success", __func__);
    return HCCL_SUCCESS;
}

// V2 通信域 channel acquire 公共前置准备：一致性记录、引擎校验、debug 初始化、集群监控注册。
// 非共享路径 HcclChannelAcquire 与共享路径 HcclChannelAcquireWithConfig 共用。
static HcclResult PrepareV2ChannelAcquire(hccl::hcclComm* hcclComm, HcclComm comm, CommEngine engine)
{
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    s32 deviceLogicId = 0;
    (void)hrtGetDeviceRefresh(&deviceLogicId);
    u32 rankTableCrc = RankTableCrcBridge::GetInstance().ConsumeRankTableJsonCrc(deviceLogicId);
    if (rankTableCrc != 0) {
        CHK_RET(RankConsistencyCheckerV2::GetInstance(deviceLogicId).RecordRankTableCrcV2(rankTableCrc));
    }
    // 用 sizeof 自动推导包名长度，避免魔法数 6 与字面量 "hcomm" 长度耦合后忘记同步
    static constexpr char HCOMM_PKG_NAME[] = "hcomm";
    std::array<char, sizeof(HCOMM_PKG_NAME)> hcommPkgName = {};
    std::copy(std::begin(HCOMM_PKG_NAME), std::end(HCOMM_PKG_NAME), hcommPkgName.begin());
    std::array<char, CANN_VERSION_MAX_LEN + 1> hcommVersionStr = {0};
    aclError aclRet = aclsysGetVersionStr(hcommPkgName.data(), hcommVersionStr.data());
    CHK_PRT_RET(
        aclRet != ACL_SUCCESS, HCCL_ERROR("[%s] aclsysGetVersionStr failed, aclRet[%d].", __func__, aclRet),
        HCCL_E_INTERNAL);
    std::string curVersion(hcommVersionStr.data());
    CHK_RET(RankConsistencyCheckerV2::GetInstance(deviceLogicId).RecordCannVersionV2(curVersion));

    const uint32_t opExpansionMode = myRank->GetOpExpansionMode();
    if (!CheckCommEngine(engine, opExpansionMode)) {
        HCCL_ERROR(
            "[%s] opExpansionMode[%d] not supported by engine[%s].", __func__, opExpansionMode,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }

    if (!GetDebugConfigInited()) {
        InitDebugConfigByEnv();
    }

    if (engine != CommEngine::COMM_ENGINE_CPU) {
        HcclResult monRet = RegisterToClusterMonitor(comm);
        CHK_PRT_RET(
            monRet != HCCL_SUCCESS,
            HCCL_ERROR(
                "[%s] RegisterToClusterMonitor failed, group[%s], ret[%d].", __func__,
                hcclComm->GetIdentifier().c_str(), monRet),
            monRet);
    }

    return HCCL_SUCCESS;
}

// V2 通信域 channel acquire 公共后置处理：symmetric remoteMem 回填、CPU DFX callback、AICPU ReportKernel。
// 非共享路径 HcclChannelAcquire 与共享路径 HcclChannelAcquireWithConfig 共用。
static HcclResult FinalizeV2ChannelAcquire(
    hccl::hcclComm* hcclComm, CommEngine engine, const std::vector<HcclChannelDesc>& channelDescFinals,
    ChannelHandle* channels, uint32_t channelNum, const std::vector<bool>& channelSymMemAppended, u64 beginTime)
{
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);

    if (std::any_of(channelSymMemAppended.begin(), channelSymMemAppended.end(), [](bool appended) {
            return appended;
        })) {
        hccl::MyRank* myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CHK_RET(UpdateSymmetricRemoteMems(
            collComm, myRank, channelDescFinals, channels, channelNum, channelSymMemAppended));
    }

    if (engine == COMM_ENGINE_CPU) {
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        CHK_PTR_NULL(hcclCommDfx);
        auto callback = hcclCommDfx->GetDpuCallback();
        for (uint32_t idx = 0; idx < channelNum; idx++) {
            int32_t dpuRet = HcommDpuChannelRegisterDfx(channels[idx], callback);
            CHK_PRT_RET(
                dpuRet != HCCL_SUCCESS,
                HCCL_ERROR("[%s] Failed to register DFX callback for channel[%u], ret[%d].", __func__, idx, dpuRet),
                static_cast<HcclResult>(dpuRet));
        }
    }

    if (IsAicpuEngine(engine)) {
        HcclCommDfx* hcclCommDfx = collComm->GetHcclCommDfx();
        CHK_PTR_NULL(hcclCommDfx);
        std::string kernelName = "RunAicpuIndOpChannelInitV2";
        HcclResult reportRet
            = hcclCommDfx->ReportKernel(beginTime, hcclComm->GetIdentifier(), kernelName, SalGetTid(), false);
        CHK_PRT_RET(
            reportRet != HCCL_SUCCESS,
            HCCL_ERROR("[%s] ReportKernel failed, kernelName[%s], ret[%d].", __func__, kernelName.c_str(), reportRet),
            reportRet);
    }

    return HCCL_SUCCESS;
}

// 入参校验：HcclChannelAcquire / HcclChannelQuery / HcclChannelAcquireWithConfig 共用，消除重复参数检查
static HcclResult CheckChannelResParams(
    const HcclComm comm, const HcclChannelDesc* channelDescs, const ChannelHandle* channels, uint32_t channelNum)
{
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(channelDescs);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0 || channelNum > CHANNEL_NUM_MAX),
        HCCL_ERROR(
            "[%s]Invalid channelNum, channelNum[%u], max channel num[%u]", __func__, channelNum, CHANNEL_NUM_MAX),
        HCCL_E_PARA);
    return HCCL_SUCCESS;
}

HcclResult HcclChannelAcquire(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    HcclUs startut = TIME_NOW();
    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    EXCEPTION_HANDLE_BEGIN

    CHK_RET(CheckChannelResParams(comm, channelDescs, channels, channelNum));

    HcclResult ret = HCCL_SUCCESS;
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HCCL_RUN_INFO(
        "Entry-%s channelNum[%u], engine[%s] group[%s]", __func__, channelNum,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), hcclComm->GetIdentifier().c_str());
    std::vector<HcclChannelDesc> channelDescFinals;
    std::vector<std::vector<HcclMemHandle>> mergedMemHandles;
    for (uint32_t idx = 0; idx < channelNum; idx++) {
        HcclChannelDesc channelDescFinal;
        HcclChannelDescInit(&channelDescFinal, 1);
        ret = ProcessHcclResPackReq(channelDescs[idx], channelDescFinal, hcclComm);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "ProcessHcclResPackReq failed. channelDesc idx[%u], group[%s], engine[%s] channelNum[%u], ret[%d]", idx,
                hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                channelNum, ret),
            ret);
        channelDescFinals.push_back(channelDescFinal);
    }

    if (hcclComm->IsCommunicatorV2()) { // A5
        const std::string& commTag = hcclComm->GetIdentifier();
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);

        CHK_RET(PrepareV2ChannelAcquire(hcclComm, comm, engine));

        hccl::MyRank* myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        std::vector<bool> channelSymMemAppended;
        if (IsAicpuEngine(engine) || engine == COMM_ENGINE_AIV) {
            CHK_RET(PrepareChannelSymMemHandles(
                collComm, myRank, engine, channelDescFinals, mergedMemHandles, channelSymMemAppended));
        }
        const bool hasSymmetricMemHandles
            = std::any_of(channelSymMemAppended.begin(), channelSymMemAppended.end(), [](bool appended) {
                  return appended;
              });
        HCCL_INFO(
            "[HcclChannelAcquire] PrepareChannelSymMemHandles done, group[%s], engine[%d], channelNum[%u], "
            "hasSymmetricMemHandles[%d], mergedMemHandleGroups[%zu].",
            commTag.c_str(), engine, channelNum, hasSymmetricMemHandles, mergedMemHandles.size());

        ret = myRank->CreateChannels(engine, commTag, channelDescFinals.data(), channelNum, channels);
        CHK_PRT_RET(
            (ret == HCCL_E_AGAIN || ret == HCCL_E_UNAVAIL),
            HCCL_WARNING(
                "CreateChannels group[%s], engine[%s] ret[%d]", commTag.c_str(),
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "CreateChannels failed. group[%s], engine[%s] ret[%d]", commTag.c_str(),
                GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), ret),
            ret);

        CHK_RET(FinalizeV2ChannelAcquire(
            hcclComm, engine, channelDescFinals, channels, channelNum, channelSymMemAppended, beginTime));
    } else {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        if (collComm != nullptr) {
            hccl::MyRank* myRank = collComm->GetMyRank();
            if (hcclComm->GetConnectMode() != 0 && engine == COMM_ENGINE_CPU && myRank != nullptr) {
                const std::string& commTag = hcclComm->GetIdentifier();
                ret = myRank->CreateChannels(engine, commTag, channelDescFinals.data(), channelNum, channels);
            } else {
                auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
                ret = channelMgr.ChannelCommCreate(
                    hcclComm->GetIdentifier(), engine, channelDescFinals.data(), channelNum, channels);
            }
        } else {
            auto& channelMgr = hcclComm->GetIndependentOp().GetChannelManager();
            ret = channelMgr.ChannelCommCreate(
                hcclComm->GetIdentifier(), engine, channelDescFinals.data(), channelNum, channels);
        }
    }

    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to acquire channel, group[%s], engine[%s], channelNum[%u], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum,
            ret),
        ret);

    CHK_RET(ConvertAivChannelHandlesToDevicePtrs(engine, channelDescFinals.data(), channelNum, channels));

    HCCL_RUN_INFO(
        "[%s] acquire channel success, group[%s], engine[%s], channelNum[%u], take time [%lld]us.", __func__,
        hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum,
        DURATION_US(TIME_NOW() - startut).count());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

static HcclResult PackChannelDescs(
    const HcclChannelDesc* channelDescs, uint32_t channelNum, hccl::hcclComm* hcclComm, CommEngine engine,
    std::vector<HcclChannelDesc>& channelDescFinals)
{
    for (uint32_t idx = 0; idx < channelNum; idx++) {
        HcclChannelDesc channelDescFinal;
        HcclChannelDescInit(&channelDescFinal, 1);
        HcclResult ret = ProcessHcclResPackReq(channelDescs[idx], channelDescFinal, hcclComm);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "ProcessHcclResPackReq failed. channelDesc idx[%u], group[%s], engine[%s] channelNum[%u], ret[%d]", idx,
                hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
                channelNum, ret),
            ret);
        channelDescFinals.push_back(channelDescFinal);
    }
    return HCCL_SUCCESS;
}

HcclResult HcclChannelQuery(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum, ChannelHandle* channels)
{
    HcclUs startut = TIME_NOW();
    EXCEPTION_HANDLE_BEGIN

    CHK_RET(CheckChannelResParams(comm, channelDescs, channels, channelNum));

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HCCL_RUN_INFO(
        "Entry-%s channelNum[%u], engine[%s] group[%s]", __func__, channelNum,
        GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), hcclComm->GetIdentifier().c_str());

    // 仅 V2（A5）路径支持；legacy 通信域不支持查询，返回 NOT_SUPPORT（符合 legacy 不承接新特性）
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_WARNING("[%s] legacy communicator not supported, return NOT_SUPPORT.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    const uint32_t opExpansionMode = myRank->GetOpExpansionMode();
    if (!CheckCommEngine(engine, opExpansionMode)) {
        HCCL_ERROR(
            "[%s] opExpansionMode[%d] not supported by engine[%s].", __func__, opExpansionMode,
            GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str());
        return HCCL_E_PARA;
    }

    // 打包 channelDesc（与 HcclChannelAcquire 一致的兼容处理流程）
    std::vector<HcclChannelDesc> channelDescFinals;
    CHK_RET(PackChannelDescs(channelDescs, channelNum, hcclComm, engine, channelDescFinals));

    HcclResult ret = myRank->QueryChannels(engine, channelDescFinals.data(), channelNum, channels);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to query channel, group[%s], engine[%s], channelNum[%u], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum,
            ret),
        ret);

    HCCL_RUN_INFO(
        "[%s] query channel success, group[%s], engine[%s], channelNum[%u], take time [%lld]us.", __func__,
        hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(), channelNum,
        DURATION_US(TIME_NOW() - startut).count());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclChannelDestroy(HcclComm comm, const ChannelHandle* channels, uint32_t channelNum)
{
    HcclUs startut = TIME_NOW();
    EXCEPTION_HANDLE_BEGIN

    // 入参校验
    CHK_PTR_NULL(comm);
    CHK_PTR_NULL(channels);
    CHK_PRT_RET(
        (channelNum == 0 || channelNum > CHANNEL_NUM_MAX),
        HCCL_ERROR("[%s]Invalid channelNum[%u], max channel num[%u]", __func__, channelNum, CHANNEL_NUM_MAX),
        HCCL_E_PARA);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    HCCL_RUN_INFO("Entry-%s channelNum[%u] group[%s]", __func__, channelNum, hcclComm->GetIdentifier().c_str());

    // 仅 V2（A5）路径支持；legacy 通信域不支持销毁，返回 NOT_SUPPORT
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_WARNING("[%s] legacy communicator not supported, return NOT_SUPPORT.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }

    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    HcclResult ret = myRank->DestroyChannels(channels, channelNum);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] Failed to destroy channel, group[%s], channelNum[%u], ret[%d]", __func__,
            hcclComm->GetIdentifier().c_str(), channelNum, ret),
        ret);

    HCCL_RUN_INFO(
        "[%s] destroy channel success, group[%s], channelNum[%u], take time [%lld]us.", __func__,
        hcclComm->GetIdentifier().c_str(), channelNum, DURATION_US(TIME_NOW() - startut).count());
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}

HcclResult HcclGroupStart() { return HcclLegacyGroupStart(); }

HcclResult HcclGroupEndV2()
{
    CHK_RET(groupLaunchA5());
    HCCL_INFO("[GroupEnd] to the end");
    return HCCL_SUCCESS;
}

HcclResult HcclGroupEnd()
{
    if (hcclGroupDepth == 0) {
        HCCL_ERROR("HcclGroupEnd: not in a group call. Didn't call HcclGroupStart before.");
        return HCCL_E_NOT_SUPPORT;
    }
    if (--hcclGroupDepth > 0) {
        return HCCL_SUCCESS;
    }

    HCCL_INFO("[HcclGroupEnd] hcclGroupDepth=[%d]", hcclGroupDepth);
    /*遇到最后一个HcclGroupEnd才处理group内的所有任务*/
    HCCLV2_FUNC_RUN([&]() -> HcclResult {
        CHK_RET(HcclLegacyAsyncJobLaunch());
        return HcclGroupEndV2();
    }());
    return HcclLegacyGroupEnd();
}

HcclResult HcclGroupStatusGet(bool* isGroupEnabled)
{
    CHK_PTR_NULL(isGroupEnabled);
    *isGroupEnabled = (hcclGroupDepth > 0);
    return HCCL_SUCCESS;
}

static bool IsSharedQueueUbProtocol(CommProtocol protocol)
{
    return protocol == COMM_PROTOCOL_UB_CTP || protocol == COMM_PROTOCOL_UBC_TP || protocol == COMM_PROTOCOL_UB_RTP;
}

static bool IsSameLocalEndpoint(const EndpointDesc& a, const EndpointDesc& b)
{
    return a.protocol == b.protocol && a.commAddr.type == b.commAddr.type
           && std::memcmp(a.commAddr.raws, b.commAddr.raws, sizeof(a.commAddr.raws)) == 0
           && a.loc.locType == b.loc.locType && std::memcmp(a.loc.raws, b.loc.raws, sizeof(a.loc.raws)) == 0;
}

static HcclResult ValidateSharedQueueDescs(const std::vector<HcclChannelDesc>& channelDescs)
{
    for (uint32_t i = 0; i < channelDescs.size(); ++i) {
        if (!IsSharedQueueUbProtocol(channelDescs[i].channelProtocol)) {
            HCCL_ERROR(
                "[%s] IS_SHARED_QUEUE only supports UB protocols (UB_CTP/UBC_TP/UB_RTP), "
                "channelDesc[%u] protocol[%d].",
                __func__, i, channelDescs[i].channelProtocol);
            return HCCL_E_NOT_SUPPORT;
        }
    }

    if (channelDescs.size() > 1) {
        const EndpointDesc& firstLocal = channelDescs[0].localEndpoint;
        for (uint32_t i = 1; i < channelDescs.size(); ++i) {
            if (!IsSameLocalEndpoint(firstLocal, channelDescs[i].localEndpoint)) {
                HCCL_ERROR(
                    "[%s] all channelDescs must have the same localEndpoint for shared jetty, "
                    "channelDesc[0] != channelDesc[%u].",
                    __func__, i);
                return HCCL_E_PARA;
            }
        }
    }
    return HCCL_SUCCESS;
}

struct SharedJettyRemoteGroup {
    EndpointDesc remoteEp;
    std::vector<uint32_t> descIndices;
};

static HcclResult RegisterMemForSharedJettyChannels(
    hccl::MyRank* myRank, EndpointHandle epHandle, std::vector<HcclChannelDesc>& channelDescs,
    std::vector<std::vector<MemHandle>>& memHandleStorage)
{
    uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());
    for (uint32_t i = 0; i < channelNum; ++i) {
        CHK_RET(myRank->PrepareMemHandles(
            epHandle, channelDescs[i].memHandles, channelDescs[i].memHandleNum, memHandleStorage[i]));
        channelDescs[i].memHandles = memHandleStorage[i].data();
        channelDescs[i].memHandleNum = static_cast<uint32_t>(memHandleStorage[i].size());
    }
    return HCCL_SUCCESS;
}

static void GroupChannelDescsByRemoteEp(
    const std::vector<HcclChannelDesc>& channelDescs, std::vector<SharedJettyRemoteGroup>& groups)
{
    auto FindGroup = [&groups](const EndpointDesc& remoteEp) -> SharedJettyRemoteGroup* {
        for (auto& g : groups) {
            if (g.remoteEp.protocol == remoteEp.protocol && g.remoteEp.commAddr.type == remoteEp.commAddr.type
                && std::memcmp(g.remoteEp.commAddr.raws, remoteEp.commAddr.raws, sizeof(remoteEp.commAddr.raws)) == 0
                && g.remoteEp.loc.locType == remoteEp.loc.locType
                && std::memcmp(g.remoteEp.loc.raws, remoteEp.loc.raws, sizeof(remoteEp.loc.raws)) == 0) {
                return &g;
            }
        }
        return nullptr;
    };
    for (uint32_t i = 0; i < channelDescs.size(); ++i) {
        const EndpointDesc& remoteEp = channelDescs[i].remoteEndpoint;
        SharedJettyRemoteGroup* g = FindGroup(remoteEp);
        if (g == nullptr) {
            groups.push_back({remoteEp, {i}});
        } else {
            g->descIndices.push_back(i);
        }
    }
}

static HcclResult CreateSharedJettyChannelsForGroup(
    CommEngine engine, EndpointHandle epHandle, const std::vector<HcclChannelDesc>& channelDescs, uint32_t repIdx,
    const std::string& commTag, hccl::MyRank* myRank, uint32_t needCreate, ChannelHandle* outCh)
{
    std::vector<HcclChannelDesc> hcclDescs(needCreate, channelDescs[repIdx]);
    std::vector<HcommChannelDesc> hcommDescs(needCreate);
    const std::string channelNameStr = commTag;
    for (uint32_t j = 0; j < needCreate; ++j) {
        hcommDescs[j] = MyRankUtils::ChannelDescHccl2Hcomm(hcclDescs[j], hccl::CommConfig{});
        hcommDescs[j].channelName = channelNameStr.c_str();
    }
    std::string socketTag = commTag + "_engine_" + std::to_string(static_cast<uint32_t>(engine));
    HcclResult sockRet = myRank->BatchCreateSockets(hcclDescs.data(), needCreate, socketTag, hcommDescs);
    CHK_PRT_RET(
        sockRet != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] BatchCreateSockets failed, repIdx[%u], remoteRank[%u], ret[%d].", __func__, repIdx,
            channelDescs[repIdx].remoteRank, sockRet),
        sockRet);
    HCCL_INFO("[%s] shared jetty sockets created, repIdx[%u], needCreate[%u].", __func__, repIdx, needCreate);

    HcommChannelConfig hcommConfig = nullptr;
    HcclResult cfgRet = static_cast<HcclResult>(hcomm::ChannelConfigCreate(&hcommConfig));
    CHK_PRT_RET(
        cfgRet != HCCL_SUCCESS, HCCL_ERROR("[%s] ChannelConfigCreate failed, ret[%d].", __func__, cfgRet), cfgRet);
    auto* hcommCfg = static_cast<hcomm::HcommChannelConfigData*>(hcommConfig);
    hcommCfg->isSharedQueue = true;

    uint32_t created = 0;
    for (uint32_t j = 0; j < needCreate; ++j) {
        HcclResult ret = static_cast<HcclResult>(
            HcommChannelCreateWithConfig(epHandle, engine, &hcommDescs[j], 1, hcommConfig, &outCh[j]));
        if (ret != HCCL_SUCCESS) {
            if (created > 0) {
                (void)HcommChannelDestroy(outCh, created);
            }
            HCCL_ERROR("[%s] HcommChannelCreateWithConfig failed, j[%u], ret[%d].", __func__, j, ret);
            (void)hcomm::ChannelConfigDestroy(hcommConfig);
            return ret;
        }
        created++;
    }
    (void)hcomm::ChannelConfigDestroy(hcommConfig);
    return HCCL_SUCCESS;
}

static HcclResult AcquireSharedJettyGroupChannels(
    const HcclComm comm, CommEngine engine, const std::vector<HcclChannelDesc>& channelDescs,
    const SharedJettyRemoteGroup& group, const EndpointHandle epHandle, const std::string& commTag,
    const std::string& sharedTag, hccl::MyRank* myRank, const EndpointDesc& localEp, ChannelHandle* channels,
    std::vector<bool>* outIsNewChannel)
{
    (void)comm;
    uint32_t requestedNum = static_cast<uint32_t>(group.descIndices.size());
    hccl::EndpointDescPair epPair = std::make_pair(localEp, group.remoteEp);
    uint32_t repIdx = group.descIndices[0];

    auto createFunc = [engine, &channelDescs, repIdx, epHandle, &commTag,
                       myRank](uint32_t needCreate, ChannelHandle* outCh) -> HcclResult {
        return CreateSharedJettyChannelsForGroup(
            engine, epHandle, channelDescs, repIdx, commTag, myRank, needCreate, outCh);
    };

    std::vector<ChannelHandle> groupOut(requestedNum, 0);
    uint32_t reusedCount = 0;
    HcclResult acqRet = hccl::SharedJettyChannelPool::GetInstance().AcquireChannels(
        myRank, sharedTag, epPair, requestedNum, createFunc, groupOut.data(), &reusedCount);
    if (acqRet != HCCL_SUCCESS) {
        HCCL_ERROR("[%s] AcquireChannels failed for group, ret[%d].", __func__, acqRet);
        return acqRet;
    }

    // 池返回的 handle 按组内 descIndices 回填到 channels 的原位置
    for (uint32_t k = 0; k < requestedNum; ++k) {
        uint32_t descIdx = group.descIndices[k];
        channels[descIdx] = groupOut[k];
        // k >= reusedCount 的为新建 channel，回滚时需销毁并从池移除；
        // 复用的 channel 仍由池和其他调用方持有，不可销毁
        if (outIsNewChannel != nullptr && k >= reusedCount) {
            (*outIsNewChannel)[descIdx] = true;
        }
        u32 remoteRank = channelDescs[descIdx].remoteRank;
        HcclCommDfx::AddChannelRemoteRankId(commTag, static_cast<u64>(groupOut[k]), remoteRank);
    }
    return HCCL_SUCCESS;
}

static void RollbackAcquiredSharedJettyChannels(
    uint32_t channelNum, ChannelHandle* channels, const std::vector<bool>* isNewChannel, const EndpointDesc& localEp,
    const std::vector<HcclChannelDesc>& channelDescs, hccl::MyRank* myRank, const std::string& sharedTag)
{
    // 多组部分失败时回滚已成功的新建 channel
    // 复用的 channel 仍由池和其他调用方持有，不可销毁，否则导致 use-after-free
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (channels[i] != 0 && isNewChannel != nullptr && (*isNewChannel)[i]) {
            (void)HcommChannelDestroy(&channels[i], 1);
            hccl::EndpointDescPair epPair = std::make_pair(localEp, channelDescs[i].remoteEndpoint);
            hccl::SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, sharedTag, epPair, &channels[i], 1);
            channels[i] = 0;
        }
    }
}

static HcclResult AcquireSharedJettyChannels(
    HcclComm comm, CommEngine engine, std::vector<HcclChannelDesc>& channelDescs,
    const hccl::HcclChannelConfigData* cfg, ChannelHandle* channels, std::vector<bool>* outIsNewChannel)
{
    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(comm);
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    const std::string& commTag = hcclComm->GetIdentifier();
    const std::string& sharedTag = cfg->sharedQueueTag;
    uint32_t channelNum = static_cast<uint32_t>(channelDescs.size());

    if (outIsNewChannel != nullptr) {
        outIsNewChannel->assign(channelNum, false);
    }

    const EndpointDesc& localEp = channelDescs[0].localEndpoint;
    EndpointHandle epHandle = nullptr;
    hccl::EndpointMgr* endpointMgr = myRank->GetEndpointMgr();
    CHK_PTR_NULL(endpointMgr);
    // 共享 jetty 按 sharedQueueTag 区分 Endpoint：不同 tag 创建独立 Endpoint → 独立底层 jetty 资源。
    // 同一 tag 复用同一 Endpoint（JettyContext 引用计数复用）。
    CHK_RET(endpointMgr->GetWithTag(localEp, sharedTag, epHandle));

    // memHandleStorage 持有 memHandleVec 的生命周期，确保 channelDescs[].memHandles 在本函数内有效。
    // 无论 memVec 是否为空都执行 RegisterMemory 并覆盖 memHandles：
    // 空时 memHandleStorage[i] 为空 → memHandles=nullptr/memHandleNum=0，避免残留用户传入的无效句柄。
    std::vector<std::vector<MemHandle>> memHandleStorage(channelNum);
    CHK_RET(RegisterMemForSharedJettyChannels(myRank, epHandle, channelDescs, memHandleStorage));

    std::vector<SharedJettyRemoteGroup> groups;
    GroupChannelDescsByRemoteEp(channelDescs, groups);

    HcclResult groupRet = HCCL_SUCCESS;
    for (const auto& group : groups) {
        groupRet = AcquireSharedJettyGroupChannels(
            comm, engine, channelDescs, group, epHandle, commTag, sharedTag, myRank, localEp, channels,
            outIsNewChannel);
        if (groupRet != HCCL_SUCCESS) {
            break;
        }
    }

    if (groupRet != HCCL_SUCCESS) {
        RollbackAcquiredSharedJettyChannels(
            channelNum, channels, outIsNewChannel, localEp, channelDescs, myRank, sharedTag);
        return groupRet;
    }

    HCCL_INFO(
        "[%s] shared jetty channels acquired, comm[%p], tag[%s], channelNum[%u], remoteGroups[%zu].", __func__, comm,
        sharedTag.c_str(), channelNum, groups.size());

    // memHandleStorage 即将析构，清空 channelDescs 中的悬空指针，防止调用方误用
    for (uint32_t i = 0; i < channelNum; ++i) {
        channelDescs[i].memHandles = nullptr;
        channelDescs[i].memHandleNum = 0;
    }
    return HCCL_SUCCESS;
}

static HcclResult ParseSharedQueueConfig(
    HcclChannelConfig config, CommEngine engine, HcclComm comm, bool& isSharedQueue, std::string& sharedQueueTag,
    hccl::hcclComm*& hcclComm)
{
    isSharedQueue = false;
    if (config != nullptr) {
        auto* cfg = static_cast<hccl::HcclChannelConfigData*>(config);
        isSharedQueue = cfg->isSharedQueue;
        sharedQueueTag = cfg->sharedQueueTag;
    }

    if (!isSharedQueue) {
        return HCCL_SUCCESS;
    }

    if (sharedQueueTag.empty()) {
        HCCL_ERROR("[%s] SHARED_QUEUE_TAG must be set when IS_SHARED_QUEUE is true.", __func__);
        return HCCL_E_PARA;
    }

    if (engine != COMM_ENGINE_AIV) {
        HCCL_ERROR(
            "[%s] IS_SHARED_QUEUE currently only supports AIV engine, engine[%d].", __func__, static_cast<int>(engine));
        return HCCL_E_NOT_SUPPORT;
    }

    hcclComm = static_cast<hccl::hcclComm*>(comm);
    if (!hcclComm->IsCommunicatorV2()) {
        HCCL_ERROR("[%s] IS_SHARED_QUEUE only supports V2 communicator.", __func__);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}

static void DestroyAndClearSharedJettyChannels(
    hccl::hcclComm* hcclComm, const std::string& sharedQueueTag, uint32_t channelNum, ChannelHandle* channels,
    const std::vector<bool>& isNewChannel, const std::vector<ChannelHandle>& channelsCopy,
    const std::vector<HcclChannelDesc>& channelDescFinals)
{
    // 仅销毁本轮新建的 channel，复用的 channel 保留在池中供其他调用方使用
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (channels[i] != 0 && isNewChannel[i]) {
            (void)HcommChannelDestroy(&channels[i], 1);
            channels[i] = 0;
        }
    }
    // 从池中移除已销毁的新建句柄，避免重试时返回已销毁的 channel
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    if (collComm == nullptr) {
        return;
    }
    hccl::MyRank* myRank = collComm->GetMyRank();
    if (myRank == nullptr) {
        return;
    }
    const EndpointDesc& localEp = channelDescFinals[0].localEndpoint;
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (channelsCopy[i] == 0 || !isNewChannel[i]) {
            continue;
        }
        const EndpointDesc& remoteEp = channelDescFinals[i].remoteEndpoint;
        hccl::EndpointDescPair epPair = std::make_pair(localEp, remoteEp);
        hccl::SharedJettyChannelPool::GetInstance().RemoveChannels(myRank, sharedQueueTag, epPair, &channelsCopy[i], 1);
    }
}

constexpr uint32_t SHARED_JETTY_POLL_INTERVAL_MS = 2; // 共享jetty建链状态轮询间隔（ms）

static HcclResult WaitForSharedJettyChannelsReady(
    uint32_t channelNum, ChannelHandle* channels, hccl::hcclComm* hcclComm,
    const std::vector<HcclChannelDesc>& channelDescFinals)
{
    std::vector<int32_t> statusList(channelNum, 0);
    auto linkTimeout = std::chrono::seconds(Hccl::EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut());
    auto startTime = std::chrono::steady_clock::now();
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    uint32_t localRank = (collComm != nullptr) ? collComm->GetMyRankId() : 0;
    hccl::MyRank* myRank = (collComm != nullptr) ? collComm->GetMyRank() : nullptr;

    auto printChannelErrors = [&](int64_t elapsed) {
        std::vector<int32_t> internalStatus(channelNum, 0);
        (void)hcomm::ChannelProcess::ChannelGetStatus(channels, channelNum, internalStatus.data());
        std::vector<Hccl::TlsStatus> tlsStatusList(channelNum, Hccl::TlsStatus::UNKNOWN);
        if (myRank != nullptr) {
            myRank->GetAbnormalChannelTlsStatus(
                channelDescFinals.data(), internalStatus.data(), channelNum, tlsStatusList);
        }
        hcomm::logger::ChannelLogger::PrintChannelErrorDetails(
            localRank, channelNum, channelDescFinals.data(), channels, internalStatus.data(),
            static_cast<uint64_t>(elapsed), tlsStatusList.data());
    };

    while (true) {
        HcclResult statusRet = static_cast<HcclResult>(HcommChannelGetStatus(channels, channelNum, statusList.data()));
        if (statusRet != HCCL_SUCCESS && statusRet != HCCL_E_AGAIN) {
            HCCL_ERROR("[%s] HcommChannelGetStatus failed during shared jetty connect, ret[%d].", __func__, statusRet);
            return statusRet;
        }
        bool allReady = true;
        for (uint32_t i = 0; i < channelNum; ++i) {
            if (statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_FAILED
                || statusList[i] == hcomm::HCOMM_CHANNEL_STATUS_TIMEOUT) {
                auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
                                   std::chrono::steady_clock::now() - startTime)
                                   .count();
                HCCL_ERROR("[%s] shared jetty channel[%u] connect failed, status[%d].", __func__, i, statusList[i]);
                printChannelErrors(elapsed);
                return HCCL_E_NETWORK;
            }
            if (statusList[i] != hcomm::HCOMM_CHANNEL_STATUS_READY) {
                allReady = false;
            }
        }
        if (allReady) {
            return HCCL_SUCCESS;
        }
        if ((std::chrono::steady_clock::now() - startTime) >= linkTimeout) {
            auto elapsed
                = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - startTime)
                      .count();
            HCCL_ERROR(
                "[%s] shared jetty channel connect timeout, group[%s], elapsed[%lld]ms.", __func__,
                hcclComm->GetIdentifier().c_str(), elapsed);
            printChannelErrors(elapsed);
            return HCCL_E_TIMEOUT;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(SHARED_JETTY_POLL_INTERVAL_MS));
    }
}

static HcclResult ExchangeConsistencyForSharedJetty(
    hccl::hcclComm* hcclComm, CommEngine engine, uint32_t channelNum,
    const std::vector<HcclChannelDesc>& channelDescFinals, const std::vector<bool>& isNewChannel)
{
    hccl::CollComm* collComm = hcclComm->GetCollComm();
    CHK_PTR_NULL(collComm);
    hccl::MyRank* myRank = collComm->GetMyRank();
    CHK_PTR_NULL(myRank);

    const std::string identifier = hcclComm->GetIdentifier();
    std::vector<HcommChannelDesc> consistencyDescs(channelNum);
    for (uint32_t i = 0; i < channelNum; ++i) {
        consistencyDescs[i] = MyRankUtils::ChannelDescHccl2Hcomm(channelDescFinals[i], hccl::CommConfig{});
        consistencyDescs[i].channelName = identifier.c_str();
    }

    std::string consistencySocketTag = identifier + "_engine_" + std::to_string(static_cast<uint32_t>(engine));
    HcclResult sockRet
        = myRank->BatchCreateSockets(channelDescFinals.data(), channelNum, consistencySocketTag, consistencyDescs);
    CHK_PRT_RET(
        sockRet != HCCL_SUCCESS,
        HCCL_ERROR("[%s] BatchCreateSockets for consistency failed, ret[%d].", __func__, sockRet), sockRet);

    std::vector<std::pair<u32, u32>> newChannelIdxs;
    for (uint32_t i = 0; i < channelNum; ++i) {
        if (isNewChannel[i]) {
            newChannelIdxs.emplace_back(i, 0U);
        }
    }
    HcclResult exchRet = myRank->BatchExchangeAndCheckConsistency(
        channelDescFinals.data(), consistencyDescs, channelNum, newChannelIdxs, engine);
    CHK_PRT_RET(
        exchRet != HCCL_SUCCESS,
        HCCL_ERROR(
            "[%s] BatchExchangeAndCheckConsistency failed, group[%s], ret[%d].", __func__,
            hcclComm->GetIdentifier().c_str(), exchRet),
        exchRet);
    return HCCL_SUCCESS;
}

// 推进建链状态机至 READY + 一致性交换，失败时销毁已获取的新建 channel 并从池中移除
static HcclResult FinalizeSharedJettyAcquisition(
    hccl::hcclComm* hcclComm, CommEngine engine, uint32_t channelNum, ChannelHandle* channels,
    const std::vector<bool>& isNewChannel, const std::vector<HcclChannelDesc>& channelDescFinals,
    const std::string& sharedQueueTag)
{
    std::vector<ChannelHandle> channelsCopy(channels, channels + channelNum);

    HcclResult waitRet = WaitForSharedJettyChannelsReady(channelNum, channels, hcclComm, channelDescFinals);
    if (waitRet != HCCL_SUCCESS) {
        DestroyAndClearSharedJettyChannels(
            hcclComm, sharedQueueTag, channelNum, channels, isNewChannel, channelsCopy, channelDescFinals);
        return waitRet;
    }

    HcclResult exchRet
        = ExchangeConsistencyForSharedJetty(hcclComm, engine, channelNum, channelDescFinals, isNewChannel);
    if (exchRet != HCCL_SUCCESS) {
        DestroyAndClearSharedJettyChannels(
            hcclComm, sharedQueueTag, channelNum, channels, isNewChannel, channelsCopy, channelDescFinals);
        return exchRet;
    }
    return HCCL_SUCCESS;
}

HcclResult HcclChannelAcquireWithConfig(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    HcclChannelConfig config, ChannelHandle* channels)
{
    HcclUs startut = TIME_NOW();
    EXCEPTION_HANDLE_BEGIN

    CHK_RET(CheckChannelResParams(comm, channelDescs, channels, channelNum));

    bool isSharedQueue = false;
    std::string sharedQueueTag;
    hccl::hcclComm* hcclComm = nullptr;
    CHK_RET(ParseSharedQueueConfig(config, engine, comm, isSharedQueue, sharedQueueTag, hcclComm));
    if (!isSharedQueue) {
        return HcclChannelAcquire(comm, engine, channelDescs, channelNum, channels);
    }

    u64 beginTime = Hccl::DfxDlProfFunction::GetInstance().dlMsprofSysCycleTime();
    CHK_RET(PrepareV2ChannelAcquire(hcclComm, comm, engine));

    // 复用 HcclChannelAcquire 的前置校验（ProcessHcclResPackReq），保证共享/非共享路径校验一致
    std::vector<HcclChannelDesc> channelDescFinals;
    CHK_RET(PackChannelDescs(channelDescs, channelNum, hcclComm, engine, channelDescFinals));
    CHK_RET(ValidateSharedQueueDescs(channelDescFinals));

    std::vector<std::vector<HcclMemHandle>> mergedMemHandles;
    std::vector<bool> channelSymMemAppended;
    if (IsAicpuEngine(engine) || engine == COMM_ENGINE_AIV) {
        hccl::CollComm* collComm = hcclComm->GetCollComm();
        CHK_PTR_NULL(collComm);
        hccl::MyRank* myRank = collComm->GetMyRank();
        CHK_PTR_NULL(myRank);
        CHK_RET(PrepareChannelSymMemHandles(
            collComm, myRank, engine, channelDescFinals, mergedMemHandles, channelSymMemAppended));
    }

    auto* cfg = static_cast<hccl::HcclChannelConfigData*>(config);
    std::vector<bool> isNewChannel;
    HcclResult ret = AcquireSharedJettyChannels(comm, engine, channelDescFinals, cfg, channels, &isNewChannel);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR(
                                 "[%s] AcquireSharedJettyChannels failed, group[%s], ret[%d].", __func__,
                                 hcclComm->GetIdentifier().c_str(), ret);
        for (uint32_t i = 0; i < channelNum; ++i) { channels[i] = 0; }, ret);

    // 推进建链状态机至 READY + 一致性交换，失败时自动清理
    CHK_RET(FinalizeSharedJettyAcquisition(
        hcclComm, engine, channelNum, channels, isNewChannel, channelDescFinals, sharedQueueTag));

    CHK_RET(FinalizeV2ChannelAcquire(
        hcclComm, engine, channelDescFinals, channels, channelNum, channelSymMemAppended, beginTime));

    HCCL_RUN_INFO(
        "[%s] acquire shared jetty channels success, group[%s], engine[%s], channelNum[%u], take time [%lld]us.",
        __func__, hcclComm->GetIdentifier().c_str(), GetEnumToString(GetCommEngineStatusStrMap(), engine).c_str(),
        channelNum, DURATION_US(TIME_NOW() - startut));
    EXCEPTION_HANDLE_END
    return HCCL_SUCCESS;
}
