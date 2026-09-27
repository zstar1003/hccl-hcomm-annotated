/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "orion_adapter_hccp.h"
#include <chrono>
#include <unistd.h>
#include <memory>
#include <unordered_map>
#include "sal.h"
#include "network_api_exception.h"
#include "internal_exception.h"
#include "hccp.h"
#include "hccp_tlv.h"
#include "hccp_ctx.h"
#include "hccp_async.h"
#include "env_config_v2.h"
#include "hccp_common.h"
#include "exception_util.h"
#include "adapter_error_manager_pub.h"
#include "hccl_log_keywords.h"
#include "config_plf_log_v2.h"

using namespace std;

namespace Hccl {
constexpr u32 ONE_HUNDRED_MICROSECOND_OF_USLEEP = 100;
constexpr u32 ONE_MILLISECOND_OF_USLEEP = 1000;
constexpr unsigned int SOCKET_NUM_ONE = 1;
constexpr u32 MAX_NUM_OF_WHITE_LIST_NUM = 16;
constexpr u32 AUTO_LISTEN_PORT = 0;
constexpr u64 SOCKET_SEND_MAX_SIZE = 0x7FFFFFFFFFFFFFFF;
constexpr u32 MAX_WR_NUM = 1024;
constexpr u32 MAX_SEND_SGE_NUM = 1;
constexpr u32 MAX_RECV_SGE_NUM = 1;
constexpr u32 MAX_CQ_DEPTH = 65535;
constexpr u32 NDA_CQ_DEPTH_FOR_UBNIC = 31 * 1024;
constexpr u32 NDA_CQ_DEPTH_FOR_XSCDV = 32 * 1024;
constexpr u32 MAX_INLINE_DATA = 64;
constexpr u32 RA_TLV_REQUEST_UNAVAIL = 128308;
constexpr u32 ROCE_ENOMEM_RET = 328100;
constexpr u32 GET_TP_ATTR_OPCODE = 106;
constexpr u32 GET_TLS_ENABLE_OPCODE = 95;
constexpr u32 GET_TLS_ENABLE_VERSION = 1;
constexpr u32 GET_TP_ATTR_VERSION = 2;

const std::unordered_map<HrtNetworkMode, NetworkMode, EnumClassHash> HRT_NETWORK_MODE_MAP
    = {{HrtNetworkMode::PEER, NetworkMode::NETWORK_PEER_ONLINE}, {HrtNetworkMode::HDC, NetworkMode::NETWORK_OFFLINE}};

s32 g_linkTimeout = 0;
inline s32 EnvLinkTimeoutGet()
{
    g_linkTimeout = g_linkTimeout != 0 ? g_linkTimeout : EnvConfig::GetInstance().GetSocketConfig().GetLinkTimeOut();
    return g_linkTimeout;
}

HcclResult HrtRaGetTlsStatus(struct RaInfo* info, TlsStatus& tlsStatus)
{
    tlsStatus = TlsStatus::UNKNOWN;
    CHK_PTR_NULL(info);

    u32 tlsVersion = 0;
    s32 versionRet = RaGetInterfaceVersion(info->phyId, GET_TLS_ENABLE_OPCODE, &tlsVersion);
    if (versionRet != 0 || tlsVersion < GET_TLS_ENABLE_VERSION) {
        HCCL_WARNING(
            "[HrtRaGetTlsStatus] this package does not support RaGetTlsEnable for device, "
            "please change new package. ret[%d], tlsVersion[%u].",
            versionRet, tlsVersion);
        return HCCL_E_NOT_SUPPORT;
    }

    bool tlsEnable = false;
    s32 ret = RaGetTlsEnable(info, &tlsEnable);
    if (ret != 0) {
        tlsStatus = TlsStatus::DISABLE;
        HCCL_ERROR(
            "[HrtRaGetTlsStatus] errNo[0x%016llx] failed ret[%d], phyId[%u]", HCCL_ERROR_CODE(HCCL_E_NETWORK), ret,
            info->phyId);
        return HCCL_E_NETWORK;
    }

    tlsStatus = tlsEnable ? TlsStatus::ENABLE : TlsStatus::DISABLE;
    HCCL_INFO(
        "[HrtRaGetTlsStatus] phyId[%u], tlsEnable[%d], tlsStatus[%d]", info->phyId, tlsEnable,
        static_cast<s32>(tlsStatus));
    return HCCL_SUCCESS;
}

inline union HccpIpAddr IpAddressToHccpIpAddr(IpAddress& addr)
{
    union HccpIpAddr hccpIpAddr;
    if (addr.GetFamily() == AF_INET) {
        hccpIpAddr.addr = addr.GetBinaryAddress().addr;
    } else {
        hccpIpAddr.addr6 = addr.GetBinaryAddress().addr6;
    }
    return hccpIpAddr;
}

inline IpAddress IfAddrInfoToIpAddress(struct InterfaceInfo info)
{
    BinaryAddr addr;
    if (info.family == AF_INET) {
        addr.addr = info.ifaddr.ip.addr;
    } else {
        addr.addr6 = info.ifaddr.ip.addr6;
    }
    return IpAddress(addr, info.family, info.scopeId);
}

void* HrtRaTlvInit(HRaTlvInitConfig& cfg)
{
    HCCL_INFO("[Init][RaTlv] Input params: version=[%d], phyId=[%u], mode=[%u]", cfg.version, cfg.phyId, cfg.mode);
    struct TlvInitInfo init_info {};
    init_info.version = cfg.version;
    init_info.phyId = cfg.phyId;
    init_info.nicPosition = HRT_NETWORK_MODE_MAP.at(cfg.mode);

    s32 ret = 0;
    unsigned int buffer_size;
    void* tlv_handle;

    ret = RaTlvInit(&init_info, &buffer_size, &tlv_handle);
    if (ret != 0 || tlv_handle == nullptr) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[Init][RaTlv]errNo[0x%016llx] ra tlv init fail. params: mode=%u, device id=%u, "
                                     "version=%d, tlv_handle=%p, return: ret[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), cfg.mode, init_info.phyId,
                                     init_info.version, tlv_handle, ret));
    }

    HCCL_INFO("tlv init success, device id[%u]", init_info.phyId);

    return tlv_handle;
}

HcclResult HrtRaTlvRequest(void* tlv_handle, u32 tlv_module_type, u32 tlv_ccu_msg_type)
{
    CHK_PTR_NULL(tlv_handle);

    HCCL_INFO(
        "[Request][RaTlv] Input params: tlv_handle=[%p], tlv_module_type=[%u], tlv_ccu_msg_type=[%u]", tlv_handle,
        tlv_module_type, tlv_ccu_msg_type);
    s32 ret = 0;

    struct TlvMsg send_msg {};
    struct TlvMsg recv_msg {};
    send_msg.type = tlv_ccu_msg_type;

    ret = RaTlvRequest(tlv_handle, tlv_module_type, &send_msg, &recv_msg);
    if (ret != 0) {
        if (ret == RA_TLV_REQUEST_UNAVAIL || ret == OTHERS_ENOTSUPP) {
            HCCL_WARNING("[HrtRaTlvRequest]ra tlv request UNAVAIL. return: ret[%d]", ret);
            return HCCL_E_UNAVAIL;
        }
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[Request][RaTlv]errNo[0x%016llx] ra tlv request fail. params: tlv_handle=%p, tlv_module_type=%u, "
                "tlv_ccu_msg_type=%u, return: ret=%d",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), tlv_handle, tlv_module_type, tlv_ccu_msg_type, ret));
    }

    HCCL_INFO("tlv request success, tlv module type[%u], message type[%u]", tlv_module_type, tlv_ccu_msg_type);
    return HCCL_SUCCESS;
}

void HrtRaTlvRequestForCustomChannel(void* tlvHandle, u32 msgType, void* customIn, void* customOut)
{
    CHECK_NULLPTR(tlvHandle, "[HrtRaTlvRequestForCustomChannel] tlvHandle is nullptr!");
    CHECK_NULLPTR(customIn, "[HrtRaTlvRequestForCustomChannel] customIn is nullptr!");
    CHECK_NULLPTR(customOut, "[HrtRaTlvRequestForCustomChannel] customOut is nullptr!");

    struct TlvMsg sendMsg {};
    sendMsg.type = msgType;
    sendMsg.length = sizeof(CustomChanInfoIn);
    sendMsg.data = static_cast<char*>(customIn);

    struct TlvMsg recvMsg {};
    recvMsg.type = msgType;
    recvMsg.length = sizeof(CustomChanInfoOut);
    recvMsg.data = static_cast<char*>(customOut);

    s32 ret = RaTlvRequest(tlvHandle, TLV_MODULE_TYPE_CCU, &sendMsg, &recvMsg);
    if (ret != 0) {
        MACRO_THROW(NetworkApiException, StringFormat("[%s] RaTlvRequest fail, ret[%d]", __func__, ret));
    }
}

void HrtRaTlvDeInit(void* tlv_handle)
{
    CHECK_NULLPTR(tlv_handle, "[HrtRaTlvDeInit] tlv_handle is nullptr!");

    s32 ret = 0;

    ret = RaTlvDeinit(tlv_handle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[DeInit][RaTlv]errNo[0x%016llx] ra tlv deinit fail. params: tlv_handle=%p, return: ret=%d",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), tlv_handle, ret));
    }
}

// hccp 模块日志掩码，bit0~bit3 依次对应 INIT/RDMA_OP/SOCKET_OP/OTHERS（见 ra_rs_opcode.h 的 ModuleType）
constexpr u64 HCCP_DEBUG_ALL_MODULES = (1ULL << 0) | (1ULL << 1) | (1ULL << 2) | (1ULL << 3);

static void SetHccpDebugConfig(HrtNetworkMode mode, uint32_t phyId)
{
    auto modeIter = HRT_NETWORK_MODE_MAP.find(mode);
    if (modeIter == HRT_NETWORK_MODE_MAP.end()) {
        HCCL_ERROR("[SetHccpDebugConfig] invalid network mode[%u], skip debug config.", mode);
        return;
    }
    struct RaInfo debugInfo {};
    debugInfo.mode = modeIter->second;
    debugInfo.phyId = phyId;
    u64 hccpDebugConfig = 0;
    if (GetPlfDebugConfigValue() & PLF_CHANNEL) {
        hccpDebugConfig = HCCP_DEBUG_ALL_MODULES;
    }
    s32 ret = RaSetDebugConfig(&debugInfo, hccpDebugConfig);
    if (ret != 0) {
        HCCL_WARNING("[SetHccpDebugConfig] RaSetDebugConfig failed, ret[%d], mode[%u], phyId[%u].", ret, mode, phyId);
    }
}

void HrtRaInit(HRaInitConfig& cfg)
{
    HCCL_INFO("[Init][Ra] Input params: phyId=[%u], mode=[%u]", cfg.phyId, cfg.mode);

    struct RaInitConfig config {};
    config.phyId = cfg.phyId;
    config.nicPosition = HRT_NETWORK_MODE_MAP.at(cfg.mode);
    config.hdcType = PID_HDC_TYPE;
    config.enableHdcAsync = true;

    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());

    while (true) {
        ret = RaInit(&config);
        if (!ret) {
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat(
                        "[Init][Ra]errNo[0x%016llx], ra init timeout[%lld s], phy_id=%u, nic_position=%u, ret=%d",
                        HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), timeout, config.phyId, config.nicPosition, ret));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            // 非ra限速场景错误，不轮询。直接退出
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Init][Ra]errNo[0x%016llx] ra init fail, phy_id=%u, nic_position=%u, ret=%d",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), config.phyId, config.nicPosition, ret));
        }
    }
    HCCL_INFO("init ra success,return: ret[%d]", ret);
    SetHccpDebugConfig(cfg.mode, cfg.phyId);
}

void HrtRaDeInit(HRaInitConfig& cfg)
{
    HCCL_INFO("[DeInit][Ra] Input params: phyId=[%u], mode=[%u]", cfg.phyId, cfg.mode);
    struct RaInitConfig config {};
    config.phyId = cfg.phyId;
    config.nicPosition = HRT_NETWORK_MODE_MAP.at(cfg.mode);
    config.hdcType = PID_HDC_TYPE;

    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        ret = RaDeinit(&config);
        if (!ret) {
            HCCL_INFO("deinit ra success,return: ret[%d]", ret);
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat(
                        "[DeInit][Ra]errNo[0x%016llx] ra deinit timeout[%lld s], phy_id=%u, nic_position=%u, ret=%d",
                        HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), timeout, config.phyId, config.nicPosition, ret));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            // 非ra限速场景错误，不轮询。直接退出
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[DeInit][Ra]errNo[0x%016llx] ra deinit fail, phy_id=%u, nic_position=%u, ret=%d",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), config.phyId, config.nicPosition, ret));
        }
    }
}

static void SocketBatchConnect(SocketConnectInfoT conn[], u32 num)
{
    CHECK_NULLPTR(conn, "[SocketBatchConnect] conn is nullptr!");
    HCCL_INFO("[BatchConnect][RaSocket] Input params: num=%u", num);
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        ret = RaSocketBatchConnect(conn, num);
        if (!ret) {
            HCCL_INFO("socket batch connect success, ret=%d", ret);
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException, StringFormat(
                                             "[BatchConnect][RaSocket]errNo[0x%016llx] ra socket batch connect, "
                                             "timeout[%lld s]. return[%d], params: num[%u]",
                                             HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), timeout, ret, num));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            MACRO_THROW(
                NetworkApiException, StringFormat(
                                         "[BatchConnect][RaSocket]errNo[0x%016llx] ra socket batch connect fail, "
                                         "return[%d], params: num[%u]",
                                         HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret, num));
        }
    }
}

void HrtRaSocketConnectOne(RaSocketConnectParam& in)
{
    HCCL_INFO(
        "[ConnectOne][RaSocket] Input params: socketHandle=%p, remoteIp=%s, port=%u, tag=%s", in.socketHandle,
        in.remoteIp.Describe().c_str(), in.port, in.tag.c_str());

    struct SocketConnectInfoT connInfo {};
    connInfo.socketHandle = in.socketHandle;
    connInfo.remoteIp = IpAddressToHccpIpAddr(in.remoteIp);
    connInfo.port = in.port;

    int sret = strcpy_s(connInfo.tag, sizeof(connInfo.tag), in.tag.c_str());
    if (sret != 0) {
        string msg = StringFormat(
            "[HrtRaSocketConnectOne] copy tag[%s] to hccp tag failed, in.tag size[%d], connInfo.tag size[%d], ret[%d]",
            in.tag.c_str(), sizeof(in.tag.c_str()), sizeof(connInfo.tag), sret);
        MACRO_THROW(NetworkApiException, msg);
    }

    HCCL_INFO("Socket Connect tag=[%s], remoteIp[%s]", connInfo.tag, in.remoteIp.Describe().c_str());
    SocketBatchConnect(&connInfo, 1);
}

static void HRaSocketBatchClose(struct SocketCloseInfoT conn[], u32 num)
{
    CHECK_NULLPTR(conn, "[HRaSocketBatchClose] conn is nullptr!");
    HCCL_INFO("[BatchClose][RaSocket] Input params: num=%u", num);
    HCCL_INFO("ra socket batch close");
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        ret = RaSocketBatchClose(conn, num);
        if (!ret) {
            HCCL_INFO("socket batch close success, ret=%d", ret);
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException, StringFormat(
                                             "[BatchClose][RaSocket]errNo[0x%016llx] ra socket batch close, timeout[%d "
                                             "s], return[%d], params: num[%u]",
                                             HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), timeout, ret, num));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            // 非ra限速场景错误，不轮询，直接退出
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[BatchClose][RaSocket]errNo[0x%016llx] ra socket batch close fail, return[%d], params: num[%u]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret, num));
        }
    }
}

void HrtRaSocketCloseOne(RaSocketCloseParam& in)
{
    HCCL_INFO("[CloseOne][RaSocket] Input params: socketHandle=%p, fdHandle=%p", in.socketHandle, in.fdHandle);
    struct SocketCloseInfoT closeInfo = {};
    closeInfo.fdHandle = in.fdHandle;
    closeInfo.socketHandle = in.socketHandle;

    HRaSocketBatchClose(&closeInfo, 1);
}

static void ReportAddrInUseError(const IpAddress& localIp, u32 port, HrtNetworkMode netMode)
{
    std::string errMsg = "The IP address " + std::string(localIp.Describe().c_str()) + " and port "
                         + std::to_string(port) + " have already been bound.";
    if (netMode == HrtNetworkMode::PEER) {
        RPT_INPUT_ERR(true, "EI0019", std::vector<std::string>({"reason"}), std::vector<std::string>({errMsg}));
        HCCL_ERROR(
            "[%s][%s] socket type[0], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), errMsg.c_str());
    } else {
        RPT_INPUT_ERR(true, "EI0020", std::vector<std::string>({"reason"}), std::vector<std::string>({errMsg}));
        HCCL_ERROR(
            "[%s][%s] socket type[1], %s Please check the port status and whether the port is being used by other "
            "process.",
            LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), errMsg.c_str());
    }
}

static void
HRaSocketListenStart(struct SocketListenInfoT conn[], u32 num, const IpAddress& localIp, HrtNetworkMode netMode)
{
    CHECK_NULLPTR(conn, "[HRaSocketListenStart] conn is nullptr!");
    HCCL_INFO("[ListenStart][RaSocket] Input params: num=%u", num);
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());

    while (true) {
        ret = RaSocketListenStart(conn, num);
        if (ret == 0) {
            HCCL_INFO("socket listen start success, ret=%d", ret);
            break;
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat(
                        "[ListenStart][RaSocket]errNo[0x%016llx] ra socket listen start, timeout[%d s], return[%d], "
                        "params: num[%u]",
                        HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), EnvLinkTimeoutGet(), ret, num));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (ret == SOCK_EADDRINUSE) {
            u32 port = (num > 0) ? conn[0].port : HCCL_INVALID_PORT;
            ReportAddrInUseError(localIp, port, netMode);
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[%s]ra socket listen could not start, due to the port[%u] has already been bound. please try"
                    " another port or check the port status",
                    __func__, port));
        } else if (ret == SOCK_EADDRNOTAVAIL) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[%s] Socket listen start fail: "
                    "IP address is not available, please check the IP address configuration, return[%d]",
                    __func__, ret));
        } else {
            // 非ra限速场景错误，不轮询，直接退出
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[ListenStart][RaSocket]errNo[0x%016llx] ra socket listen start fail, return[%d], params: num[%u]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), EnvLinkTimeoutGet(), ret, num));
        }
    }
}

static bool RaSocketTryListenStart(
    struct SocketListenInfoT conn[], u32 num, [[maybe_unused]] const IpAddress& localIp,
    [[maybe_unused]] HrtNetworkMode netMode)
{
    CHECK_NULLPTR(conn, "[RaSocketTryListenStart] conn is nullptr!");
    HCCL_INFO("[TryListenStart][RaSocket] Input params: num=%u", num);
    s32 ret = RaSocketListenStart(conn, num);
    if (ret == 0) {
        return true;
    } else if (ret == SOCK_EAGAIN) {
        HCCL_INFO("[%s] listen eagain", __func__);
        return true;
    } else if (ret == SOCK_EADDRINUSE) {
        u32 port = (num > 0) ? conn[0].port : HCCL_INVALID_PORT;
        HCCL_INFO(
            "[%s]ra socket listen could not start, due to the port[%u] has already been bound. please try"
            " another port or check the port status",
            __func__, port);
        return false;
    } else if (ret == SOCK_EADDRNOTAVAIL) {
        RPT_INPUT_ERR(
            true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}),
            std::vector<std::string>({std::to_string(ret), "socket listen start", "0"}));
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] Socket listen start fail: "
                "IP address is not available, please check the IP address configuration, return[%d]",
                __func__, ret));
    } else {
        // 非ra限速场景错误，不轮询，直接退出
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[TryListenStart][RaSocket]errNo[0x%016llx] ra socket listen start fail, return[%d], params: num[%u]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret, num));
    }
}

static void HRaSocketListenStop(struct SocketListenInfoT conn[], u32 num)
{
    CHECK_NULLPTR(conn, "[HRaSocketListenStop] conn is nullptr!");
    HCCL_INFO("[ListenStop][RaSocket] Input params: num=%u", num);
    s32 ret = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        ret = RaSocketListenStop(conn, num);
        if (!ret || ret == 228202) { // 待修改: 同步版本后 228202 修改为 SOCK_ENODEV
            HCCL_INFO("socket listen stop success, ret=%d", ret);
            break; // 成功跳出
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                MACRO_THROW(
                    NetworkApiException, StringFormat(
                                             "[ListenStop][RaSocket]errNo[0x%016llx] ra socket listen stop fail, "
                                             "timeout[%d s], return[%d], params: num[%u]",
                                             HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), timeout, ret, num));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            // 非ra限速场景错误，不轮询，直接退出
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[ListenStop][RaSocket]errNo[0x%016llx] ra socket listen stop fail, return[%d], params: num[%u]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret, num));
        }
    }
}

void HrtRaSocketListenOneStart(RaSocketListenParam& in, HrtNetworkMode netMode)
{
    HCCL_INFO("[ListenStart][RaSocket] Input params: socketHandle: %p, port: %u", in.socketHandle, in.port);
    struct SocketListenInfoT listenInfo {};
    listenInfo.socketHandle = in.socketHandle;
    listenInfo.port = in.port;
    HRaSocketListenStart(&listenInfo, 1, in.localIp, netMode);
}

bool HrtRaSocketTryListenOneStart(RaSocketListenParam& in, HrtNetworkMode netMode)
{
    HCCL_INFO("[TryListenOneStart][RaSocket] Input params: socketHandle: %p, port: %u", in.socketHandle, in.port);
    struct SocketListenInfoT listenInfo {};
    listenInfo.socketHandle = in.socketHandle;
    listenInfo.port = in.port;
    bool ret = RaSocketTryListenStart(&listenInfo, 1, in.localIp, netMode);
    if (ret && in.port == AUTO_LISTEN_PORT) {
        in.port = listenInfo.port;
    }
    return ret;
}

void HrtRaSocketListenOneStop(RaSocketListenParam& in)
{
    HCCL_INFO("[ListenOneStop][RaSocket] Input params: socketHandle: %p, port: %u", in.socketHandle, in.port);
    struct SocketListenInfoT listenInfo {};
    listenInfo.socketHandle = in.socketHandle;
    listenInfo.port = in.port;
    HRaSocketListenStop(&listenInfo, 1);
}

void RaBlockGetSockets(u32 role, SocketInfoT conn[], u32 num, u32 timeoutSec) // 修改为内部函数，不对外
{
    CHECK_NULLPTR(conn, "[RaBlockGetSockets] conn is nullptr!");
    HCCL_INFO("[GetSockets][RaBlock] Input params: role=[%u], num=[%u], timeoutSec=[%u s]", role, num, timeoutSec);
    s32 sockRet;
    u32 gotSocketsCnt = 0;
    auto startTime = std::chrono::steady_clock::now();
    auto linkTimeout = std::chrono::seconds(EnvLinkTimeoutGet());
    auto timeout
        = (timeoutSec > 0 && timeoutSec < linkTimeout.count()) ? std::chrono::seconds(timeoutSec) : linkTimeout;
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[HrtRaBlockGetSockets] get rasocket timeout role[%u], num[%u], gotSocketsCnt[%u], timeout[%lld]s",
                    role, num, gotSocketsCnt, timeout));
        }
        u32 connectedNum = 0;
        sockRet = RaGetSockets(role, conn, num, &connectedNum);
        if ((connectedNum == 0 && sockRet == 0) || (sockRet == SOCK_EAGAIN)) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else if (sockRet != 0) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Get][RaSocket]get rasocket error. role[%u], num[%u], sockRet[%d], connectednum[%u]", role, num,
                    sockRet, connectedNum));
        } else {
            gotSocketsCnt += connectedNum;
            if (gotSocketsCnt == num) {
                HCCL_INFO("block get sockets success, socket num[%u]", gotSocketsCnt);
                break;
            } else if (gotSocketsCnt > num) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat("[Get][RaSocket]total Sockets[%u], more than needed num[%u]!", gotSocketsCnt, num));
            } else {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            }
        }
    }
}

RaSocketFdHandleParam HrtRaBlockGetOneSocket(u32 role, RaSocketGetParam& param, u32 timeout)
{
    HCCL_INFO(
        "[GetOneSocket][RaSocket] Input params: role=[%u],socketHandle=[%p], fdHandle=[%p], remoteIp=[%s], timeout=[%u "
        "s]",
        role, param.socketHandle, param.fdHandle, param.remoteIp.Describe().c_str(), timeout);
    struct SocketInfoT socketInfo {};

    socketInfo.socketHandle = param.socketHandle;
    socketInfo.fdHandle = param.fdHandle;
    socketInfo.remoteIp = IpAddressToHccpIpAddr(param.remoteIp);
    socketInfo.status = SOCKET_NOT_CONNECTED;

    int sret = strcpy_s(socketInfo.tag, sizeof(socketInfo.tag), param.tag.c_str());
    if (sret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[HrtRaBlockGetOneSocket] copy tag[%s] to hccp failed, ret=%d, role=%u,socketHandle=%p, fdHandle=%p, "
                "remoteIp=%s, socketInfo.tag size=%d, param.tag size=%d",
                param.tag.c_str(), sret, role, param.socketHandle, param.fdHandle, param.remoteIp.Describe().c_str(),
                sizeof(socketInfo.tag), sizeof(param.tag.c_str())));
    }

    HCCL_INFO("Socket Get tag=[%s], remoteIp[%s], ret[%d]", socketInfo.tag, param.remoteIp.Describe().c_str(), sret);
    RaBlockGetSockets(role, &socketInfo, 1, timeout);

    return RaSocketFdHandleParam(socketInfo.fdHandle, socketInfo.status);
}

void HrtRaSocketBlockSend(const FdHandle fdHandle, const void* data, u32 sendSize)
{
    CHECK_NULLPTR(fdHandle, "[HrtRaSocketBlockSend] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketBlockSend] data is nullptr!");
    s32 ret = 0;
    void* sendData = const_cast<void*>(data);
    const std::chrono::seconds timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    const auto start = std::chrono::steady_clock::now();
    u32 totalSentSize = 0;
    unsigned long long sentSize = 0;

    HCCL_INFO("before ra socket send, para: fdHandle[%p], data[%p], size[%u]", fdHandle, sendData, sendSize);

    while (true) {
        // 底层ra_socket_send host网卡无限制，device网卡由于HDC通道限制的限制有大小限制(目前大小为64KB)
        ret = RaSocketSend(
            fdHandle, ReinterpretAs<void*>(ReinterpretAs<uintptr_t>(sendData) + totalSentSize),
            sendSize - totalSentSize, &sentSize);
        HCCL_INFO("ra socket send, data[%p], size[%u] send size[%u]", sendData, sendSize, totalSentSize);
        if (ret == 0) {
            totalSentSize += sentSize;
            if (totalSentSize == sendSize) { // 只有完全发送完才返回成功
                break;
            }

            if (totalSentSize > sendSize) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat(
                        "[Send][RaSocket]errNo[0x%016llx] ra socket send failed, fdHandle=%p, data=%p, size=%u, "
                        "retSize=%u",
                        HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), fdHandle, data, sendSize, sentSize));
            }
            SaluSleep(ONE_HUNDRED_MICROSECOND_OF_USLEEP);
        } else if (ret == SOCK_EAGAIN) {
            /* ra速率限制 retry */
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Send][RaSocket]errNo[0x%016llx] ra socket send failed, fdHandle=%p, data=%p, size=%u, "
                    "retSize=%u, ret=%d",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), fdHandle, data, sendSize, sentSize, ret));
        }
        /* 获取当前时间，如果耗时超过timeout，则返回错误 */
        const auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(std::chrono::steady_clock::now() - start);
        if (elapsed > timeout) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Send][RaSocket]errNo[0x%016llx] Wait timeout for sockets send, fdHandle[%p], data[%p], size[%u], "
                    "retsize[%u], ret[%d]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), fdHandle, data, sendSize, sentSize, ret));
        }
    }
    HCCL_INFO("ra socket send finished,ret[%d]", ret);
}

s32 HrtRaSocketNonBlockSendNormal(const FdHandle fdHandle, void* data, u64 size, u64* sentSize)
{
    CHECK_NULLPTR(fdHandle, "[HrtRaSocketNonBlockSend] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketNonBlockSend] data is nullptr!");
    CHECK_NULLPTR(sentSize, "[HrtRaSocketNonBlockSend] sentSize is nullptr!");
    HCCL_INFO(
        "[HrtRaSocketNonBlockSend] Input params: fdHandle=%p,data=%p, size=%llu, sentSize=%llu", fdHandle, data, size,
        *sentSize);
    if (size > SOCKET_SEND_MAX_SIZE) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[hrtRaSocketNonBlockSend]errNo[0x%016llx] ra socket send size is too large, "
                                     "data[%p], size[%llu], fdHandle[%p], send size[%llu]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), data, size, fdHandle, *sentSize));
    }

    return RaSocketSend(fdHandle, data, size, sentSize);
}

bool HrtRaSocketNonBlockSend(const FdHandle fdHandle, void* data, u64 size, u64* sentSize)
{
    s32 ret = HrtRaSocketNonBlockSendNormal(fdHandle, data, size, sentSize);
    if (ret == 0 || ret == SOCK_EAGAIN) {
        HCCL_INFO(
            "[HrtRaSocketNonBlockSend] ra socket send, data[%p], size[%llu], send size[%llu], ret[%d]", data, size,
            *sentSize, ret);
        return true;
    } else {
        HCCL_ERROR(
            "call RaSocketSend failed, fdHandle=%p, data=%p, size=%llu, sentSize=%llu, ret[%d]", fdHandle, data, size,
            *sentSize, ret);
        return false;
    }
}

HcclResult HrtRaSocketNonBlockSendHeart(const FdHandle fdHandle, void* data, u64 size, u64* sentSize)
{
    s32 ret = HrtRaSocketNonBlockSendNormal(fdHandle, data, size, sentSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else if (ret == SOCK_CLOSE) {
        return HCCL_E_INTERNAL; // 暂时用这个错误码表示hccp进程异常退出
    } else {
        HCCL_WARNING(
            "[HrtRaSocketNonBlockSend]ra socket send failed, data[%p], size[%llu], send size[%llu], ret[%d]", data,
            size, *sentSize, ret);
        return HCCL_E_NETWORK;
    }
}

HcclResult HrtRaSocketNonBlockRecvHeart(const FdHandle fdHandle, void* data, u64 size, u64* recvSize)
{
    CHECK_NULLPTR(fdHandle, "[HrtRaSocketNonBlockRecv] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketNonBlockRecv] data is nullptr!");
    CHECK_NULLPTR(recvSize, "[HrtRaSocketNonBlockRecv] recvSize is nullptr!");
    HCCL_DEBUG(
        "[HrtRaSocketNonBlockRecv] Input params: fdHandle=%p,data=%p, size=%llu, recvSize=%llu", fdHandle, data, size,
        *recvSize);

    s32 ret = RaSocketRecv(fdHandle, data, size, recvSize);
    if (ret == 0) {
        return HCCL_SUCCESS;
    } else if (ret == SOCK_EAGAIN) {
        return HCCL_E_AGAIN;
    } else if (ret == SOCK_CLOSE) {
        return HCCL_E_INTERNAL; // 暂时用这个错误码表示hccp进程异常退出
    } else {
        HCCL_WARNING(
            "[HrtRaSocketNonBlockRecv]ra socket recv failed, data[%p], size[%llu], "
            "recv[%llu], ret[%d], errno[%d][%s]",
            data, size, recvSize, ret, errno, strerror(errno));
        return HCCL_E_TCP_TRANSFER;
    }
    return HCCL_SUCCESS;
}

void HrtRaSocketBlockRecv(const FdHandle fdHandle, void* data, u32 size)
{
    auto startTime = std::chrono::steady_clock::now();
    unsigned long long recvSize = 0;
    s32 rtRet = 0;
    u32 getedLen = 0;
    const std::chrono::seconds timeout = std::chrono::seconds(EnvLinkTimeoutGet());

    CHECK_NULLPTR(fdHandle, "[HrtRaSocketBlockRecv] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketBlockRecv] data is nullptr!");
    HCCL_INFO("before ra socket recv, para: fdHandle[%p], data[%p], size[%u]", fdHandle, data, size);
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= timeout) {
            std::string errMsg = StringFormat(
                "[Recv][RaSocket]errNo[0x%016llx] Wait timeout for sockets recv, data[%p], "
                "size[%u], recvSize[%u], fdHandle[%p], ret[%d]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), data, size, recvSize, fdHandle, rtRet);
            HCCL_ERROR("%s", errMsg.c_str());
            HCCL_ERROR("Please check the following reasons:");
            HCCL_ERROR("1. check the firewall configuration or try to disable the firewall.");
            HCCL_ERROR("2. check error log on the other process or thread.");
            MACRO_THROW(NetworkApiException, errMsg);
        }
        rtRet = RaSocketRecv(
            fdHandle, ReinterpretAs<void*>(ReinterpretAs<uintptr_t>(data) + getedLen), size - getedLen, &recvSize);
        if ((rtRet == 0) && (recvSize > 0)) { // 接收完成，也有可能要多次接收
            getedLen += recvSize;
            if (getedLen > size) {
                MACRO_THROW(
                    NetworkApiException,
                    StringFormat(
                        "[Recv][RaSocket]errNo[0x%016llx] socket receive call RaSocketRecv failed,"
                        "rtSize[%u], bigger size[%zu], fdHandle[%p], data[%p], retSize[%u], ret[%d]",
                        HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_TRANSFER), getedLen, size, fdHandle, data, recvSize,
                        rtRet));
            }
            if (getedLen == size) {
                break;
            }
        } else if ((rtRet == 0) && (recvSize == 0)) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Recv][RaSocket]recv fail, fdHandle=%p, data=%p, bufLen=%u, recLen=%lld, ret=%d", fdHandle, data,
                    size, recvSize, rtRet));
        } else if (rtRet == SOCK_ESOCKCLOSED || rtRet == SOCK_CLOSE) { // 连接关闭，出错
            RPT_INPUT_ERR(
                true, "EI0015", std::vector<std::string>({"error_reason"}),
                std::vector<std::string>(
                    {StringFormat("socket closed during recv, fdHandle[%p], ret[%d]", fdHandle, rtRet)}));
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Recv][RaSocket]errNo[0x%016llx] recv fail, call RaSocketRecv failed, sock_esockclosed, "
                    "fdhandle=%p, data=%p, bufLen=%u, recLen=%lld, ret=%d",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_TRANSFER), fdHandle, data, size, recvSize, rtRet));
        } else if (rtRet != 0) {
            SaluSleep(ONE_MILLISECOND_OF_USLEEP); // 尚未接收到数据,延时1ms
            continue;
        }
    }
    HCCL_INFO("ra socket receive finished. ret[%d]", rtRet);
}

SocketHandle HrtRaSocketInit(HrtNetworkMode netMode, RaInterface& in)
{
    int mode = HRT_NETWORK_MODE_MAP.at(netMode);
    struct rdev rdevInfo {};
    rdevInfo.phyId = in.phyId;
    rdevInfo.family = in.address.GetFamily();
    rdevInfo.localIp = IpAddressToHccpIpAddr(in.address);

    HCCL_INFO(
        "[HrtRaSocketInit] Input params: mode=%u, ip=%u, device id=%u, family=%u", mode, rdevInfo.localIp.addr.s_addr,
        rdevInfo.phyId, rdevInfo.family);

    SocketHandle socketHandle = nullptr;
    s32 ret = RaSocketInit(mode, rdevInfo, &socketHandle);
    if (ret != 0 || (socketHandle == nullptr)) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[Init][RaSock]errNo[0x%016llx] ra socket init fail, call RaSocketInit failed, "
                                     "params: mode=%u, ip=%u, device id=%u, family=%u. return: ret=%d",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), mode, rdevInfo.localIp.addr.s_addr,
                                     rdevInfo.phyId, rdevInfo.family, ret));
    }

    HCCL_INFO("socket init success, ip[%u], device id[%u], ret[%d]", rdevInfo.localIp.addr.s_addr, rdevInfo.phyId, ret);
    return socketHandle;
}

void HrtRaSocketDeInit(SocketHandle socketHandle)
{
    CHECK_NULLPTR(socketHandle, "[HrtRaSocketDeInit] socketHandle is nullptr!");
    HCCL_INFO("[HrtRaSocketDeInit] Input params: socketHandle=%p", socketHandle);

    s32 ret = RaSocketDeinit(socketHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[DeInit][RaSocket]errNo[0x%016llx] rt socket deinit fail. call RaSocketDeinit "
                                     "failed, params: socketHandle[%p], return: ret[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), socketHandle, ret));
    }
}

void HrtRaSocketSetWhiteListStatus(u32 enable)
{
    HCCL_INFO("[HrtRaSocketSetWhiteListStatus] Input params: enable=%u", enable);
    s32 ret = RaSocketSetWhiteListStatus(enable);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[Set][WhiteListStatus]errNo[0x%016llx] ra socket set white list fail, call "
                                     "RaSocketSetWhiteListStatus failed, params: enable[%u], return: ret[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), enable, ret));
    }

    HCCL_INFO("set host socket whitelist status[%u] success.", enable);
}

u32 HrtRaSocketGetWhiteListStatus()
{
    u32 enable;
    s32 ret = RaSocketGetWhiteListStatus(&enable);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[Get][WhiteListStatus]errNo[0x%016llx] ra socket get white list fail, call "
                                     "RaSocketGetWhiteListStatus failed, return: ret[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret));
    }

    HCCL_INFO("get host socket whitelist status[%u] success.", enable);
    return enable;
}

void HrtRaSocketWhiteListAdd(SocketHandle socketHandle, vector<RaSocketWhitelist>& wlists)
{
    CHECK_NULLPTR(socketHandle, "[HrtRaSocketWhiteListAdd] socketHandle is nullptr!");
    PLF_CONFIG_INFO(PLF_CHANNEL, "[HrtRaSocketWhiteListAdd] Input params: socketHandle=%p", socketHandle);

    vector<struct SocketWlistInfoT> wlistInfoVec;
    wlistInfoVec.reserve(MAX_NUM_OF_WHITE_LIST_NUM);
    size_t wlistNum = wlists.size();
    size_t startIdx = 0;
    while (wlistNum > 0) {
        size_t addListNum = wlistNum > MAX_NUM_OF_WHITE_LIST_NUM ? MAX_NUM_OF_WHITE_LIST_NUM : wlistNum;
        for (size_t idx = startIdx; idx < addListNum + startIdx; idx++) {
            struct SocketWlistInfoT wlistInfo {};
            wlistInfo.connLimit = wlists[idx].connLimit;
            wlistInfo.remoteIp = IpAddressToHccpIpAddr(wlists[idx].remoteIp);

            int sret = strcpy_s(wlistInfo.tag, sizeof(wlistInfo.tag), wlists[idx].tag.c_str());
            if (sret != EOK) {
                MACRO_THROW(
                    InternalException,
                    StringFormat(
                        "[Add][RaSocketWhiteList]errNo[0x%016llx]errName[HCCL_E_MEMORY] memory copy failed. params: "
                        "socketHandle[%p], return: ret[%d], wlistInfo.tag size=%zu, wlists[%zu].tag size=%zu",
                        HCOM_ERROR_CODE(HcclResult::HCCL_E_MEMORY), socketHandle, sret, sizeof(wlistInfo.tag), idx,
                        sizeof(wlists[idx].tag.c_str())));
            }
            HCCL_INFO(
                "add whitelistInfo tag=[%s], remoteIp[%s]", wlistInfo.tag, wlists[idx].remoteIp.Describe().c_str());
            wlistInfoVec.push_back(wlistInfo);
        }

        s32 ret = RaSocketWhiteListAdd(socketHandle, wlistInfoVec.data(), wlistInfoVec.size());
        if (ret != 0) {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Add][RaSocketWhiteList]errNo[0x%016llx]errName[HCCL_E_TCP_CONNECT] ra white list add fail, call "
                    "RaSocketWhiteListAdd failed, socketHandle[%p], num=%zu, return[%d].",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), socketHandle, wlistInfoVec.size() + startIdx,
                    ret));
        }
        HCCL_INFO("add white list: num[%zu], remain [%zu].", addListNum, (wlistNum - addListNum));

        wlistInfoVec.clear();
        wlistNum -= addListNum;
        startIdx += addListNum;
    }
    HCCL_INFO("[HrtRaSocketWhiteListAdd] Success. Total add num [%zu]", wlists.size());
}

void HrtRaSocketWhiteListDel(SocketHandle socketHandle, vector<RaSocketWhitelist>& wlists)
{
    CHECK_NULLPTR(socketHandle, "[HrtRaSocketWhiteListDel] socketHandle is nullptr!");
    PLF_CONFIG_INFO(PLF_CHANNEL, "[HrtRaSocketWhiteListDel] Input params: socketHandle=%p", socketHandle);

    vector<struct SocketWlistInfoT> wlistInfoVec;
    wlistInfoVec.reserve(MAX_NUM_OF_WHITE_LIST_NUM);
    size_t wlistNum = wlists.size();
    size_t startIdx = 0;
    while (wlistNum > 0) {
        size_t delListNum = wlistNum > MAX_NUM_OF_WHITE_LIST_NUM ? MAX_NUM_OF_WHITE_LIST_NUM : wlistNum;
        for (size_t idx = startIdx; idx < delListNum + startIdx; idx++) {
            struct SocketWlistInfoT wlistInfo {};
            wlistInfo.connLimit = wlists[idx].connLimit;
            wlistInfo.remoteIp = IpAddressToHccpIpAddr(wlists[idx].remoteIp);

            int sret = strcpy_s(wlistInfo.tag, sizeof(wlistInfo.tag), wlists[idx].tag.c_str());
            if (sret != EOK) {
                auto msg = StringFormat(
                    "[Del][RaSocketWhiteList]errNo[0x%016llx] memory copy failed. ret[%d], wlistInfo.tag size[%zu], "
                    "wlists[%zu].tag size[%zu]",
                    HCOM_ERROR_CODE(HcclResult::HCCL_E_MEMORY), sret, sizeof(wlistInfo.tag), idx,
                    sizeof(wlists[idx].tag.c_str()));
                MACRO_THROW(InternalException, msg);
            }
            wlistInfoVec.push_back(wlistInfo);
        }

        s32 ret = RaSocketWhiteListDel(socketHandle, wlistInfoVec.data(), wlistInfoVec.size());
        if (ret != 0) {
            MACRO_THROW(
                NetworkApiException, StringFormat(
                                         "[Del][RaSocketWhiteList]errNo[0x%016llx] ra white list del fail, call "
                                         "RaSocketWhiteListDel failed, num=%zu, return[%d].",
                                         HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), wlists.size(), ret));
        }
        HCCL_INFO("del white list: num[%zu], remain [%zu].", delListNum, (wlistNum - delListNum));

        wlistInfoVec.clear();
        wlistNum -= delListNum;
        startIdx += delListNum;
    }
    HCCL_INFO("[HrtRaSocketWhiteListDel] Success. Total delete num[%zu]", wlists.size());
}

std::mutex g_deviceVnicIpMutex;
std::map<u32, IpAddress> g_deviceIdVnicInfoMap; // 记录deviceid和vnic ip的关系，用于server内查询，避免重复查询

void HrtRaSocketGetVnicIpInfos(u32 phyId, DeviceIdType deviceIdType, u32 deviceId, IpAddress& vnicIP)
{
    std::lock_guard<std::mutex> lock(g_deviceVnicIpMutex);
    auto iter = g_deviceIdVnicInfoMap.find(deviceId);
    if (iter != g_deviceIdVnicInfoMap.end()) {
        // 缓存查找到，直接从缓存获取
        vnicIP = iter->second;
        HCCL_INFO(
            "[HrtRaSocketGetVnicIpInfos] vnicInfoMap deviceId[%u] found, Ip[%s]", deviceId, vnicIP.Describe().c_str());
        return;
    }
    struct IpInfo vnicIpInfo = {};
    (void)memset_s(&vnicIpInfo, sizeof(IpInfo), 0, sizeof(IpInfo));
    IdType idType = static_cast<IdType>(deviceIdType);
    auto ret = RaSocketGetVnicIpInfos(phyId, idType, &deviceId, 1, &vnicIpInfo);
    if (ret != 0) {
        HCCL_ERROR("[hrtRaGetSocketVnicIpInfo]ra get VnicIpfail. ret[%d]", ret);
        throw NetworkApiException(StringFormat("call hrtRaGetSocketVnicIpInfo failed, ret=%llu", ret));
    }
    BinaryAddr temp;
    temp.addr = vnicIpInfo.ip.addr;
    temp.addr6 = vnicIpInfo.ip.addr6;
    IpAddress ipInfo(temp, vnicIpInfo.family);
    if (ipInfo.IsInvalid()) {
        HCCL_ERROR("vnicIp is invalid.");
        throw NetworkApiException("vnicIp is invalid.");
    }
    g_deviceIdVnicInfoMap.insert({deviceId, ipInfo});
    vnicIP = ipInfo;
    HCCL_INFO(
        "[hrtRaGetSocketVnicIpInfos] add vnicInfoMap, deviceIds[%u], Ip[%s]", deviceId, vnicIP.Describe().c_str());
}

static u32 HrtGetIfNum(struct RaGetIfattr& config)
{
    HCCL_INFO("[HrtGetIfNum] Input params: phyId=%u, nicPosition=%u", config.phyId, config.nicPosition);

    u32 num = 0;
    s32 ret = RaGetIfnum(&config, &num);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[Get][IfNum]errNo[0x%016llx] ra get if num fail. call RaGetIfnum failed, Input params: phyId=%u, "
                "nicPosistion=%u, return: ret[%d], num[%u]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), config.phyId, config.nicPosition, ret, num));
    }
    return num;
}

static void HrtGetIfAddress(struct RaGetIfattr& config, InterfaceInfo ifaddrInfos[], u32& num)
{
    CHECK_NULLPTR(ifaddrInfos, "[HrtGetIfAddress] ifaddrInfos is nullptr!");
    HCCL_INFO(
        "[HrtGetIfAddress] Input params: phyId=%u, nicPosition=%u, num=%u", config.phyId, config.nicPosition, num);

    s32 ret = RaGetIfaddrs(&config, ifaddrInfos, &num);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[Get][IfAddress]errNo[0x%016llx] ra get if address fail. call RaGetIfaddrs failed, Input params: "
                "phyId=%u, nicPosistion=%u, return: ret[%d], num[%u]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), config.phyId, config.nicPosition, ret, num));
    }
}

std::vector<std::pair<std::string, IpAddress>> HrtGetHostIf(u32 devPhyId)
{
    HCCL_INFO("[HrtGetHostIf] Input params: devPhyId=%u", devPhyId);
    std::vector<std::pair<std::string, IpAddress>> hostIfs;
    struct RaGetIfattr config = {};
    config.phyId = devPhyId;
    config.nicPosition = static_cast<u32>(NetworkMode::NETWORK_PEER_ONLINE);

    u32 ifAddrNum = HrtGetIfNum(config);
    HCCL_RUN_INFO("[Get][HostIf]hrtGetIfNum success. ifAddrNum[%u].", ifAddrNum);
    if (ifAddrNum == 0) {
        HCCL_WARNING("[Get][HostIf]there is no valid host interface, ifAddrNum[%u].", ifAddrNum);
        return hostIfs;
    }

    std::shared_ptr<struct InterfaceInfo> ifAddrInfoPtrs(
        new InterfaceInfo[ifAddrNum](), std::default_delete<InterfaceInfo[]>());
    struct InterfaceInfo* ifAddrInfos = ifAddrInfoPtrs.get();

    (void)memset_s(ifAddrInfos, ifAddrNum * sizeof(InterfaceInfo), 0, ifAddrNum * sizeof(InterfaceInfo));

    HrtGetIfAddress(config, ifAddrInfos, ifAddrNum);

    for (u32 i = 0; i < ifAddrNum; i++) {
        IpAddress ip = IfAddrInfoToIpAddress(ifAddrInfos[i]);
        hostIfs.emplace_back(ifAddrInfos[i].ifname, ip);
        HCCL_INFO("HrtGetIfAddress: idx[%u], ifName[%s], ip[%s]", i, ifAddrInfos[i].ifname, ip.GetIpStr().c_str());
    }

    return hostIfs;
}

vector<IpAddress> HrtGetDeviceIp(u32 devicePhyId, NetworkMode netWorkMode)
{
    HCCL_INFO("[HrtGetDeviceIp] Input params: devicePhyId=%u", devicePhyId);
    vector<IpAddress> ipAddr;
    struct RaGetIfattr config = {};
    config.phyId = devicePhyId;
    config.nicPosition = static_cast<u32>(netWorkMode);

    u32 ifAddrNum = HrtGetIfNum(config);
    HCCL_RUN_INFO("[Get][DeviceIP]hrtGetIfNum success. ifAddrNum[%u].", ifAddrNum);

    if (ifAddrNum == 0) {
        HCCL_WARNING("[Get][DeviceIP]device has no ip information, phy_id[%u]", devicePhyId);
        return ipAddr;
    }

    std::shared_ptr<struct InterfaceInfo> ifAddrInfoPtrs(
        new InterfaceInfo[ifAddrNum](), std::default_delete<InterfaceInfo[]>());
    struct InterfaceInfo* ifAddrInfos = ifAddrInfoPtrs.get();

    (void)memset_s(ifAddrInfos, ifAddrNum * sizeof(InterfaceInfo), 0, ifAddrNum * sizeof(InterfaceInfo));

    HrtGetIfAddress(config, ifAddrInfos, ifAddrNum);

    for (u32 i = 0; i < ifAddrNum; i++) {
        IpAddress ip = IfAddrInfoToIpAddress(ifAddrInfos[i]);
        ipAddr.emplace_back(ip);
        HCCL_INFO("HrtGetIfAddress: idx[%u], ifName[%s], ip[%s]", i, ifAddrInfos[i].ifname, ip.GetIpStr().c_str());
    }

    return ipAddr;
}

RdmaHandle HrtRaRdmaInit(HrtNetworkMode netMode, RaInterface& in)
{
    RdmaHandle rdmaHandle = nullptr;
    int mode = HRT_NETWORK_MODE_MAP.at(netMode);
    unsigned int notifyType = netMode == HrtNetworkMode::PEER ? NO_USE : NOTIFY;
    HCCL_INFO("[HrtRaRdmaInit] Input params: mode=%d, phyId=%u", mode, in.phyId);
    struct rdev rdevInfo {};
    rdevInfo.phyId = in.phyId;
    rdevInfo.family = in.address.GetFamily();
    rdevInfo.localIp = IpAddressToHccpIpAddr(in.address);
    s32 ret = RaRdevInit(mode, notifyType, rdevInfo, &rdmaHandle);
    RPT_INPUT_ERR(
        ret == HCCP_ELINKDOWN, "EI0009", vector<string>({"device_id", "reason"}),
        vector<string>({std::to_string(rdevInfo.phyId), "The network port is down"}));
    if (ret == HCCP_ELINKDOWN) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] rdma init fail, device[%u] network port is down.",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RUN_FAILED.c_str(),
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), rdevInfo.phyId);
    }
    if (ret != 0 || (rdmaHandle == nullptr)) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[Init][RaRdma]errNo[0x%016llx] rdma init fail. call RaRdevInit failed, Input "
                                     "params: phyId=%u, mode=%u, return: ret[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), in.phyId, mode, ret));
    }
    return rdmaHandle;
}

s32 HrtRaRdmaInit(HrtNetworkMode netMode, RaInterface& in, RdmaHandle& rdmaHandle)
{
    rdmaHandle = nullptr;
    int mode = HRT_NETWORK_MODE_MAP.at(netMode);
    unsigned int notifyType = netMode == HrtNetworkMode::PEER ? NO_USE : NOTIFY;
    HCCL_INFO("[HrtRaRdmaInit] Input params: mode=%d, phyId=%u", mode, in.phyId);
    struct rdev rdevInfo {};
    rdevInfo.phyId = in.phyId;
    rdevInfo.family = in.address.GetFamily();
    rdevInfo.localIp = IpAddressToHccpIpAddr(in.address);
    return RaRdevInit(mode, notifyType, rdevInfo, &rdmaHandle);
}

void HrtRaRdmaDeInit(RdmaHandle rdmaHandle, HrtNetworkMode netMode)
{
    CHECK_NULLPTR(rdmaHandle, "[HrtRaRdmaDeInit] rdmaHandle is nullptr!");
    HCCL_INFO("[HrtRaRdmaDeInit] Input params: rdmaHandle=%p, netMode=%d", rdmaHandle, netMode);
    unsigned int notifyType = netMode == HrtNetworkMode::PEER ? NO_USE : NOTIFY;
    s32 ret = RaRdevDeinit(rdmaHandle, notifyType);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[DeInit][RaRdma]errNo[0x%016llx] rt rdev deinit fail. call RaRdevDeinit failed, "
                                     "rdmaHandle=%p, return[%d].",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), rdmaHandle, ret));
    }
}

void HrtRaGetNotifyBaseAddr(RdmaHandle rdmaHandle, u64* va, u64* size)
{
    CHECK_NULLPTR(rdmaHandle, "[HrtRaGetNotifyBaseAddr] rdmaHandle is nullptr!");
    CHECK_NULLPTR(va, "[HrtRaGetNotifyBaseAddr] va is nullptr!");
    CHECK_NULLPTR(size, "[HrtRaGetNotifyBaseAddr] size is nullptr!");

    HCCL_INFO("[HrtRaGetNotifyBaseAddr] Input params: rdmaHandle=%p, va=%llu, size=%llu", rdmaHandle, *va, *size);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        unsigned long long notifyVa;
        unsigned long long notifySize;
        s32 ret = RaGetNotifyBaseAddr(rdmaHandle, &notifyVa, &notifySize);
        if (ret == 0) {
            *va = notifyVa;
            *size = notifySize;
            break;
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout != 0) {
                HCCL_ERROR(
                    "[Get][RaNotifyBaseAddr]errNo[0x%016llx] ra get notify base addr "
                    "timeout[%lld s]. return[%d], params: rdmaHandle[%p], va[0x%llx], size[%llu]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), timeout, ret, rdmaHandle, notifyVa, notifySize);
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            MACRO_THROW(
                NetworkApiException,
                StringFormat(
                    "[Get][RaNotifyBaseAddr]errNo[0x%016llx] ra get notify base addr fail, call RaGetNotifyBaseAddr "
                    "failed,"
                    "return[%d], params: va[0x%llx], size[%llu], rdmaHandle=%p",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), ret, notifyVa, notifySize, rdmaHandle));
        }
    }
}

QpHandle HrtRaQpCreate(RdmaHandle rdmaHandle, int flag, int qpMode)
{
    CHECK_NULLPTR(rdmaHandle, "[HrtRaQpCreate] rdmaHandle is nullptr!");
    HCCL_INFO("[HrtRaQpCreate] Input params: rdmaHandle=%p, flag=%d, qpMode=%d", rdmaHandle, flag, qpMode);
    QpHandle connHandle = nullptr;

    s32 ret = RaQpCreate(rdmaHandle, flag, qpMode, &connHandle);
    if (ret != 0 || connHandle == nullptr) {
        RPT_INPUT_ERR(
            ret == ROCE_ENOMEM_RET, "EI0011",
            std::vector<std::string>({"memory_size"}), // A3是当ROCE_ENOMEM_RET才上报EI0011,内存大小取决于qp深度配置
            std::vector<std::string>({"262144~3145728"}));
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] ra qp create fail, return[%d], rdmaHandle[%p], flag[%d], qpMode[%d], "
            "connHandle[%p]",
            LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_RESOURCE.c_str(),
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), ret, rdmaHandle, flag, qpMode, connHandle);
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[Create][RaQp]errNo[0x%016llx] ra qp create fail. call RaGetNotifyBaseAddr, params: rdmaHandle[%p], "
                "flag[%d], qpMode[%d], connHandle[%p]. return: ret[%d]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), rdmaHandle, flag, qpMode, connHandle, ret));
    }
    return connHandle;
}

u32 HrtGetQpNum(QpHandle qpHandle)
{
    struct QpAttr attr {};
    if (qpHandle == nullptr || RaGetQpAttr(qpHandle, &attr) != 0) {
        HCCL_ERROR("[HrtGetQpNum] RaGetQpAttr failed, qpHandle[%p].", qpHandle);
        return HRT_INVALID_QPN;
    }
    return attr.qpn;
}

static u32 HrtGetJettyQpNum(void* ctxQpHandle)
{
    if (ctxQpHandle == nullptr) {
        return HRT_INVALID_QPN;
    }
    return *reinterpret_cast<const u32*>(ctxQpHandle);
}

void HrtRaQpDestroy(QpHandle qpHandle)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaQpDestroy] qpHandle is nullptr!");
    HCCL_INFO("[HrtRaQpDestroy] Input params: qpHandle=%p", qpHandle);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    if (GetPlfDebugConfigValue() & PLF_RES) {
        u32 qpn = HrtGetQpNum(qpHandle);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
    }
    while (true) {
        s32 ret = RaQpDestroy(qpHandle);
        if (ret == 0) {
            break;
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout != 0) {
                MACRO_THROW(
                    NetworkApiException, StringFormat(
                                             "[Destroy][RaQp]errNo[0x%016llx] ra qp destroy timeout[%d s]. "
                                             "qpHandle[%p], return[%d].",
                                             HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), timeout, qpHandle, ret));
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            MACRO_THROW(
                NetworkApiException, StringFormat(
                                         "[Destroy][RaQp]errNo[0x%016llx] ra qp destroy fail. call RaQpDestroy failed, "
                                         "qpHandle[%p], return[%d].",
                                         HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), qpHandle, ret));
        }
    }
}

void HrtRaQpConnectAsync(QpHandle qpHandle, FdHandle fdHandle)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaQpConnectAsync] qpHandle is nullptr!");
    CHECK_NULLPTR(fdHandle, "[HrtRaQpConnectAsync] fdHandle is nullptr!");

    HCCL_INFO("[HrtRaQpConnectAsync] Input params: qpHandle=%p, fdHandle=%p", qpHandle, fdHandle);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        s32 ret = RaQpConnectAsync(qpHandle, fdHandle);
        if (ret == 0) {
            break;
        } else if (ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout != 0) {
                HCCL_ERROR(
                    "[ConnectAsync][RaQp]errNo[0x%016llx] ra qp connect async "
                    "timeout[%lld s]. qpHandle=[%p], fdHandle=[%p], return[%d].",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), timeout, qpHandle, fdHandle, ret);
            }
            SaluSleep(ONE_MILLISECOND_OF_USLEEP);
        } else {
            MACRO_THROW(
                NetworkApiException, StringFormat(
                                         "[ConnectAsync][RaQp]errNo[0x%016llx] ra qp connect async fail. call "
                                         "RaQpConnectAsync failed, qpHandle=%p, fdHandle=%p, return[%d]",
                                         HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), qpHandle, fdHandle, ret));
        }
    }
}

int HrtGetRaQpStatus(QpHandle qpHandle)
{
    CHECK_NULLPTR(qpHandle, "[HrtGetRaQpStatus] qpHandle is nullptr!");
    HCCL_INFO("[HrtGetRaQpStatus] Input params: qpHandle=%p", qpHandle);
    int status = 0;
    s32 ret = RaGetQpStatus(qpHandle, &status);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[GetStatus][RaQp]errNo[0x%016llx] ra qp get status failed. call ra_get_status "
                                     "failed, qpHandle[%p], return[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), qpHandle, ret));
    }
    return status;
}

void HrtRaMrReg(QpHandle qpHandle, RaMrInfo& info)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaMrReg] qpHandle is nullptr!");
    struct MrInfoT mrInfo = {};
    mrInfo.addr = info.addr;
    mrInfo.size = info.size;
    mrInfo.access = info.access;
    mrInfo.lkey = info.lkey;
    HCCL_INFO(
        "ra mr reg: qpHandle[%p], addr[%p], size[%llu], access[%d]", qpHandle, mrInfo.addr, mrInfo.size, mrInfo.access);
    s32 ret = RaMrReg(qpHandle, &mrInfo);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[Reg][RaMr]errNo[0x%016llx] ra mr reg fail. call RaMrReg failed, return[%d], params: qpHandle[%p], "
                "addr[%p], size[%llu], access[%d]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), ret, qpHandle, mrInfo.addr, mrInfo.size, mrInfo.access));
    }
}

void HrtRaMrDereg(QpHandle qpHandle, RaMrInfo& info)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaMrDereg] qpHandle is nullptr!");
    struct MrInfoT mrInfo = {};
    mrInfo.addr = info.addr;
    mrInfo.size = info.size;
    mrInfo.access = info.access;
    mrInfo.lkey = info.lkey;
    HCCL_INFO(
        "ra mr dereg: qpHandle[%p], addr[%p], size[%llu], access[%d]", qpHandle, mrInfo.addr, mrInfo.size,
        mrInfo.access);
    s32 ret = RaMrDereg(qpHandle, &mrInfo);
    if (ret != 0) {
        string msg = StringFormat(
            "call RaMrDereg failed, qpHandle=%p, addr=%p, size=%llu, access=%d", qpHandle, mrInfo.addr, mrInfo.size,
            mrInfo.access);
        MACRO_THROW(NetworkApiException, msg);
    }
}

static void HrtRaSendWr(QpHandle qpHandle, struct SendWr* wr, struct SendWrRsp* opRsp)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaSendWr] qpHandle is nullptr!");
    CHECK_NULLPTR(wr, "[HrtRaSendWr] wr is nullptr!");
    CHECK_NULLPTR(opRsp, "[HrtRaSendWr] opRsp is nullptr!");
    HCCL_INFO("[HrtRaSendWr] Input params: qpHandle=%p, send_wrAddr=%p, opRspAddr=%p", qpHandle, wr, opRsp);
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(EnvLinkTimeoutGet());
    while (true) {
        s32 ret = RaSendWr(qpHandle, wr, opRsp);
        if (ret == 0) {
            break;
        } else if (ret == SOCK_ENOENT || ret == SOCK_EAGAIN) {
            bool bTimeout = ((std::chrono::steady_clock::now() - startTime) >= timeout);
            if (bTimeout) {
                HCCL_ERROR(
                    "[Send][RaWr]errNo[0x%016llx] ra get send async timeout[%d s]. "
                    "return[%d], params: qpHandle[%p], send_wrAddr[%p], opRspAddr[%p]",
                    HCCL_ERROR_CODE(HcclResult::HCCL_E_ROCE_TRANSFER), timeout, ret, qpHandle, wr, opRsp);
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
            }
        } else {
            string msg
                = StringFormat("call RaSendWr failed, qpHandle=%p, send_wrAddr=%p opRspAddr=%p", qpHandle, wr, opRsp);
            MACRO_THROW(NetworkApiException, msg);
        }
    }
}

RaSendWrResp HrtRaSendOneWr(QpHandle qpHandle, HRaSendWr& in)
{
    CHECK_NULLPTR(qpHandle, "[HrtRaSendOneWr] qpHandle is nullptr!");
    HCCL_INFO(
        "[HrtRaSendOneWr] Input params: qpHandle=%p, locAddr=0x%llx, len=%u, rmtAddr=0x%llx, op=%u, sendFlag=%d",
        qpHandle, in.locAddr, in.len, in.rmtAddr, in.op, in.sendFlag);
    struct SgList bufList {};
    bufList.addr = in.locAddr;
    bufList.len = in.len;

    struct SendWr wr = {};
    wr.op = in.op;
    wr.dstAddr = in.rmtAddr;
    wr.sendFlag = in.sendFlag;
    wr.bufNum = 1; // 此处list只有一个，设置为1
    wr.bufList = &bufList;
    struct SendWrRsp opRsp = {};
    HrtRaSendWr(qpHandle, &wr, &opRsp);

    return RaSendWrResp(opRsp.wqeTmp.sqIndex, opRsp.wqeTmp.wqeIndex, opRsp.db.dbIndex, opRsp.db.dbInfo);
}

string HrtRaGetKeyDescribe(const u8* key, u32 len)
{
    CHECK_NULLPTR(key, "[HrtRaGetKeyDescribe] key is nullptr!");
    HCCL_INFO("[HrtRaGetKeyDescribe] Input params: key=%d, len=%u", *key, len);
    string desc = "0x";
    for (u32 idx = 0; idx < len; idx++) {
        desc += StringFormat("%02x", key[idx]);
    }
    return desc;
}

RdmaHandle HrtRaUbCtxInit(const HrtRaUbCtxInitParam& in)
{
    HCCL_INFO(
        "[HrtRaUbCtxInit] Input params: mode=%d, phyId=%u, addr=%s", in.mode, in.phyId, in.addr.GetIpStr().c_str());
    struct CtxInitCfg initCfg {};
    initCfg.mode = HRT_NETWORK_MODE_MAP.at(in.mode);

    struct CtxInitAttr ctxInfo {};
    ctxInfo.phyId = in.phyId;
    // urma_create_context(eidIndex) 决定 ctx 的 local EID，须与 GetTpList/Import 使用的链路 EID 一致
    ctxInfo.ub.eidIndex = 0U;
    const Eid linkEid = in.addr.GetEid();
    try {
        const vector<HrtDevEidInfo> eidInfoList = HrtRaGetDevEidInfoList(HRaInfo(in.mode, in.phyId));
        bool matched = false;
        for (const auto& eidInfo : eidInfoList) {
            if (eidInfo.ipAddress.GetEid() == linkEid) {
                ctxInfo.ub.eidIndex = eidInfo.eidIndex;
                matched = true;
                HCCL_INFO(
                    "[HrtRaUbCtxInit] linkEid[%s] matched eidIndex[%u].", in.addr.Describe().c_str(), eidInfo.eidIndex);
                break;
            }
        }
        if (!matched) {
            HCCL_WARNING(
                "[HrtRaUbCtxInit] linkEid[%s] not found in dev eid list(size[%zu]), "
                "fallback eidIndex[0].",
                in.addr.Describe().c_str(), eidInfoList.size());
        }
    } catch (const NetworkApiException&) {
        HCCL_WARNING(
            "[HrtRaUbCtxInit] HrtRaGetDevEidInfoList failed, fallback eidIndex[0], addr[%s].",
            in.addr.Describe().c_str());
    }
    HCCL_INFO("[HrtRaUbCtxInit] use eid[%s] eidIndex[%u]", in.addr.Describe().c_str(), ctxInfo.ub.eidIndex);
    s32 sRet
        = memcpy_s(ctxInfo.ub.eid.raw, sizeof(ctxInfo.ub.eid.raw), in.addr.GetEid().raw, sizeof(in.addr.GetEid().raw));
    if (sRet != EOK) {
        MACRO_THROW(InternalException, StringFormat("[HrtRaUbCtxInit]memcpy_s failed. sRet[%d]", sRet));
    }

    RdmaHandle handle;
    s32 ret = RaCtxInit(&initCfg, &ctxInfo, &handle);
    if (ret != 0) {
        string msg = StringFormat(
            "[Init][RaUbCtx]errNo[0x%016llx] ub ctx init fail, mode[%d], phyId[%u], addr[%s], ret[%d]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), in.mode, in.phyId, in.addr.GetIpStr().c_str(), ret);
        MACRO_THROW(NetworkApiException, msg);
    }
    return handle;
}

void HrtRaUbCtxDestroy(RdmaHandle handle)
{
    CHECK_NULLPTR(handle, "[HrtRaUbCtxDestroy] handle is nullptr!");
    HCCL_INFO("[HrtRaUbCtxDestroy] rdmaHandle[%llu].", handle);
    s32 ret = RaCtxDeinit(handle);
    if (ret != 0) {
        string msg = StringFormat(
            "[DeInit][RaRdma]errNo[0x%016llx] rt ctx deinit fail. handle[%p], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), handle, ret);
        MACRO_THROW(NetworkApiException, msg);
    }
}

std::pair<TokenIdHandle, uint32_t> RaUbAllocTokenIdHandle(RdmaHandle handle)
{
    CHECK_NULLPTR(handle, "[RaUbAllocTokenIdHandle] handle is nullptr!");
    HCCL_INFO("[RaUbAllocTokenIdHandle] rdmaHandle[%p].", handle);
    struct HccpTokenId out {};
    void* tokenIdHandle = nullptr;
    s32 ret = RaCtxTokenIdAlloc(handle, &out, &tokenIdHandle);
    if (ret != 0) {
        string msg = StringFormat("%s failed, set=%d, rdmaHandle=%p", __func__, ret, handle);
        MACRO_THROW(NetworkApiException, msg);
    }
    HCCL_INFO("[RaUbAllocTokenIdHandle] tokenIdHandle[%p], rdmaHandle[%p]", tokenIdHandle, handle);
    return {ReinterpretAs<TokenIdHandle>(tokenIdHandle), out.tokenId >> URMA_TOKEN_ID_RIGHT_SHIFT};
}

void RaUbFreeTokenIdHandle(RdmaHandle handle, TokenIdHandle tokenIdHandle)
{
    CHECK_NULLPTR(handle, "[RaUbFreeTokenIdHandle] handle is nullptr!");
    HCCL_INFO("[RaUbFreeTokenIdHandle] rdmaHandle[%p], tokenIdHandle[0x%llx].", handle, tokenIdHandle);
    s32 ret = RaCtxTokenIdFree(handle, ReinterpretAs<void*>(tokenIdHandle));
    if (ret != 0) {
        string msg = StringFormat(
            "%s failed, set=%d, rdmaHandle=%p, tokenIdHandle=0x%llx.", __func__, ret, handle, tokenIdHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

constexpr u64 UB_MEM_PAGE_SIZE = 4096;

std::pair<u64, u64> BufAlign(u64 addr, u64 size)
{
    HCCL_INFO("[BufAlign] Input params: addr=0x%llx, size=%llu", addr, size);
    // 待解决: 正式方案待讨论
    u64 pageSize = UB_MEM_PAGE_SIZE;
    u64 newAddr = addr & (~(static_cast<u64>(pageSize - 1))); // UB内存注册要求起始地址4k对齐
    u64 offset = addr - newAddr;
    u64 newSize = size + offset;
    HCCL_INFO("UB mem info: newAddr[%llx], newSize[%llu]", newAddr, newSize);

    return std::make_pair(newAddr, newSize);
}

HrtRaUbLocalMemRegOutParam HrtRaUbLocalMemReg(RdmaHandle handle, const HrtRaUbLocMemRegParam& in)
{
    CHECK_NULLPTR(handle, "[HrtRaUbLocalMemReg] handle is nullptr!");
    HCCL_INFO("[HrtRaUbLocalMemReg] Input params: handle=%p, addr=0x%llx, size=%llu", handle, in.addr, in.size);
    struct MrRegInfoT info {};
    info.in.mem.addr = in.addr;
    info.in.mem.size = in.size;

    info.in.ub.flags.value = 0;
    info.in.ub.flags.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;
    info.in.ub.flags.bs.tokenIdValid = 1;
    info.in.ub.flags.bs.access = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC;
    info.in.ub.flags.bs.nonPin = in.nonPin;
    info.in.ub.tokenValue = in.tokenValue;
    info.in.ub.tokenIdHandle = ReinterpretAs<void*>(in.tokenIdHandle);

    void* lmemHandle = nullptr;
    s32 ret = RaCtxLmemRegister(handle, &info, &lmemHandle);
    if (ret != 0) {
        string msg = StringFormat("localMemReg failed, addr=0x%llx, size=0x%llx", in.addr, in.size);
        MACRO_THROW(NetworkApiException, msg);
    }

    HrtRaUbLocalMemRegOutParam out;
    s32 sRet = memcpy_s(out.key, sizeof(out.key), info.out.key.value, info.out.key.size);
    if (sRet != EOK) {
        MACRO_THROW(InternalException, StringFormat("[HrtRaUbLocalMemReg]memcpy_s failed. sRet[%d]", sRet));
    }

    HCCL_INFO("[HrtRaUbLocalMemReg]UbLocalMemReg key.size=%u", info.out.key.size);
    out.keySize = info.out.key.size;
    out.handle = ReinterpretAs<LocMemHandle>(lmemHandle);
    out.targetSegVa = info.out.ub.targetSegHandle;
    info.in.ub.tokenValue = 0;
    HCCL_INFO(
        "[HrtRaUbLocalMemReg]UB mem reg info: in.addr[%llx], in.size[%llu], out.targetSegVa[%llu]", in.addr, in.size,
        out.targetSegVa);
    return out;
}

void HrtRaUbLocalMemUnreg(RdmaHandle rdmaHandle, LocMemHandle lmemHandle)
{
    CHECK_NULLPTR(rdmaHandle, "[HrtRaUbLocalMemUnreg] rdmaHandle is nullptr!");
    HCCL_INFO("[HrtRaUbLocalMemUnreg] Input params: rdmaHandle=%p, lmemHandle=0x%llx", rdmaHandle, lmemHandle);
    s32 ret = RaCtxLmemUnregister(rdmaHandle, ReinterpretAs<void*>(lmemHandle));
    if (ret != 0) {
        string msg = StringFormat("localMemUnreg failed, rdmaHandle=%p, lmemHandle=0x%llx", rdmaHandle, lmemHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

HrtRaUbRemMemImportedOutParam HrtRaUbRemoteMemImport(RdmaHandle handle, u8* key, u32 keyLen, u32 tokenValue)
{
    CHECK_NULLPTR(handle, "[HrtRaUbRemoteMemImport] handle is nullptr!");
    CHECK_NULLPTR(key, "[HrtRaUbRemoteMemImport] key is nullptr!");
    HCCL_INFO("[HrtRaUbRemoteMemImport] Input params: handle=%p, key=%d, keyLen=%u", handle, *key, keyLen);
    struct MrImportInfoT info {};
    int res = memcpy_s(info.in.key.value, sizeof(info.in.key.value), key, keyLen);
    if (res != 0) {
        MACRO_THROW(
            InternalException, StringFormat(
                                   "[%s] memcpy_s failed, ret = %d, params: handle=%p, key=%d, keyLen=%u", __func__,
                                   res, handle, *key, keyLen));
    }
    info.in.key.size = keyLen;

    info.in.ub.tokenValue = tokenValue;
    info.in.ub.mappingAddr = 0;
    info.in.ub.flags.value = 0;
    info.in.ub.flags.bs.access = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC;

    void* rmemHandle = nullptr;
    s32 ret = RaCtxRmemImport(handle, &info, &rmemHandle);
    if (ret != 0) {
        string msg = StringFormat("ubRemoteMemImport failed!");
        MACRO_THROW(NetworkApiException, msg);
    }

    HrtRaUbRemMemImportedOutParam out;
    out.handle = ReinterpretAs<LocMemHandle>(rmemHandle);
    out.targetSegVa = info.out.ub.targetSegHandle;
    info.in.ub.tokenValue = 0;
    return out;
}
void HrtRaUbRemoteMemUnimport(RdmaHandle rdmaHandle, RemMemHandle rmemHandle)
{
    CHECK_NULLPTR(rdmaHandle, "[HrtRaUbRemoteMemUnimport] rdmaHandle is nullptr!");
    HCCL_INFO("[HrtRaUbRemoteMemUnimport] Input params: rdmaHandle=%p, rmemHandle=0x%llx", rdmaHandle, rmemHandle);
    s32 ret = RaCtxRmemUnimport(rdmaHandle, ReinterpretAs<void*>(rmemHandle));
    if (ret != 0) {
        string msg
            = StringFormat("ubRemoteMemUnimport failed, rdmaHandle=%p, rmemHandle=0x%llx", rdmaHandle, rmemHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

const std::map<HrtUbJfcMode, JfcMode> HRT_UB_JFC_MODE_MAP
    = {{HrtUbJfcMode::NORMAL, JfcMode::JFC_MODE_NORMAL},
       {HrtUbJfcMode::STARS_POLL, JfcMode::JFC_MODE_STARS_POLL},
       {HrtUbJfcMode::CCU_POLL, JfcMode::JFC_MODE_CCU_POLL},
       {HrtUbJfcMode::USER_CTL, JfcMode::JFC_MODE_USER_CTL_NORMAL}};

constexpr u32 CQ_DEPTH = 2 * 1024 * 1024 / 64;
constexpr u32 CCU_CQ_DEPTH = 64;

JfcHandle HrtRaUbCreateJfc(RdmaHandle handle, CqCreateInfo& cqInfo, HrtUbJfcMode mode)
{
    CHECK_NULLPTR(handle, "[HrtRaUbCreateJfc] handle is nullptr!");
    HCCL_INFO("[HrtRaUbCreateJfc] Input params: handle=%p, mode=%d", handle, mode);
    struct CqInfoT info {};

    info.in.chanHandle = nullptr;
    if (mode == HrtUbJfcMode::CCU_POLL) {
        info.in.depth = CCU_CQ_DEPTH;
    } else {
        info.in.depth = CQ_DEPTH;
    }
    info.in.ub.userCtx = 0;
    info.in.ub.mode = HRT_UB_JFC_MODE_MAP.at(mode);
    info.in.ub.ceqn = 0;
    info.in.ub.flag.value = 0;

    void* jfcHandle = nullptr;

    s32 ret = RaCtxCqCreate(handle, &info, &jfcHandle);
    if (ret != 0) {
        string msg = StringFormat("ubCreateCq failed, rdmaHandle=%p,", handle);
        MACRO_THROW(NetworkApiException, msg);
    }

    cqInfo.va = info.out.va;
    HCCL_INFO("HrtRaUbCreateJfc va[%llu] mode[%u] jfcHandle[%p]", cqInfo.va, info.in.ub.mode, jfcHandle);
    return ReinterpretAs<JfcHandle>(jfcHandle);
}

void HrtRaUbDestroyJfc(RdmaHandle handle, JfcHandle jfcHandle)
{
    CHECK_NULLPTR(handle, "[HrtRaUbDestroyJfc] handle is nullptr!");
    HCCL_INFO("[HrtRaUbDestroyJfc] Input params: handle=%p, jfcHandle=0x%llx", handle, jfcHandle);
    s32 ret = RaCtxCqDestroy(handle, ReinterpretAs<void*>(jfcHandle));
    if (ret != 0) {
        string msg = StringFormat("ubCqDestroy failed, rdmaHandle=%p, jfcHandle=0x%llx", handle, jfcHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

JfcHandle HrtRaUbCreateJfcUserCtl(RdmaHandle handle, CqCreateInfo& cqInfo)
{
    CHECK_NULLPTR(handle, "[HrtRaUbCreateJfcUserCtl] handle is nullptr!");
    HCCL_INFO("[HrtRaUbCreateJfcUserCtl] Input params: handle=%p", handle);
    struct CqInfoT info {};

    info.in.chanHandle = nullptr;
    info.in.depth = CQ_DEPTH;
    info.in.ub.userCtx = 0;
    info.in.ub.mode = JfcMode::JFC_MODE_USER_CTL_NORMAL;
    info.in.ub.ceqn = 0;
    info.in.ub.flag.value = 0;

    void* jfcHandle = nullptr;

    s32 ret = RaCtxCqCreate(handle, &info, &jfcHandle);
    if (ret != 0) {
        string msg = StringFormat("ubCreateCq failed, rdmaHandle=%p,", handle);
        THROW<NetworkApiException>(msg);
    }

    HCCL_INFO(
        "[HrtRaUbCreateJfcUserCtl] jfcId[%u], cqVA[%llx], cqeSize[%u], cqDepth[%u], dbAddr[%llx]", info.out.id,
        info.out.bufAddr, info.out.cqeSize, CQ_DEPTH, info.out.swdbAddr);

    cqInfo.va = info.out.bufAddr;
    cqInfo.id = info.out.id;
    cqInfo.cqeSize = info.out.cqeSize;
    cqInfo.cqDepth = CQ_DEPTH;
    cqInfo.swdbAddr = info.out.swdbAddr;

    return ReinterpretAs<JfcHandle>(jfcHandle);
}

const std::map<HrtTransportMode, TransportModeT> HRT_TRANSPORT_MODE_MAP
    = {{HrtTransportMode::RM, TransportModeT::CONN_RM}};

const std::map<HrtJettyMode, JettyMode> HRT_JETTY_MODE_MAP
    = {{HrtJettyMode::STANDARD, JettyMode::JETTY_MODE_URMA_NORMAL},
       {HrtJettyMode::HOST_OFFLOAD, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::HOST_OPBASE, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::DEV_USED, JettyMode::JETTY_MODE_USER_CTL_NORMAL},
       {HrtJettyMode::CACHE_LOCK_DWQE, JettyMode::JETTY_MODE_CACHE_LOCK_DWQE},
       {HrtJettyMode::CCU_CCUM_CACHE, JettyMode::JETTY_MODE_CCU}};

constexpr u8 RNR_RETRY = 7;
constexpr u32 RQ_DEPTH = 256;

static struct QpCreateAttr GetQpCreateAttr(const HrtRaUbCreateJettyParam& in)
{
    struct QpCreateAttr attr {};
    attr.scqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rcqHandle = ReinterpretAs<void*>(in.rjfcHandle);
    attr.srqHandle = ReinterpretAs<void*>(in.sjfcHandle);
    attr.rqDepth = RQ_DEPTH;
    attr.sqDepth = in.sqDepth;
    attr.transportMode = HRT_TRANSPORT_MODE_MAP.at(in.transMode);
    attr.ub.mode = HRT_JETTY_MODE_MAP.at(in.jettyMode);

    attr.ub.tokenValue = in.tokenValue;
    attr.ub.tokenIdHandle = ReinterpretAs<void*>(in.tokenIdHandle);
    attr.ub.flag.value = 0;
    /* errTime配置值：0-31
        0-7代表芯片配置值b00:512ms
        8-15代表芯片配置值b01:1s
        16-23代表芯片配置值b10:8s
        24-31代表芯片配置值b11:32s
    */
    attr.ub.errTimeout = in.errTimeout;
    attr.ub.priority = static_cast<uint8_t>(in.qos & 0xFU);
    attr.ub.rnrRetry = RNR_RETRY;
    attr.ub.flag.bs.shareJfr = 1;
    attr.ub.jettyId = in.jettyId;
    // 在continue模式下+配置了wqe的fence标记，并且远端有一些权限校验错误/内存异常错误，硬件会直接挂死
    // jfs_flag 的 error_suspend 设置为 1，
    attr.ub.jfsFlag.bs.errorSuspend = 1;

    attr.ub.extMode.sqebbNum = in.sqDepth;
    if (in.jettyMode == HrtJettyMode::HOST_OFFLOAD) {
        attr.ub.extMode.piType = 1;
        attr.ub.extMode.cstmFlag.bs.sqCstm = 0; // 表示不指定Va，由HCCP返回Va
    } else if (in.jettyMode == HrtJettyMode::CCU_CCUM_CACHE) {
        attr.ub.tokenValue = in.tokenValue;
        attr.ub.extMode.cstmFlag.bs.sqCstm = 1;
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    } else if (in.jettyMode == HrtJettyMode::DEV_USED || in.jettyMode == HrtJettyMode::CACHE_LOCK_DWQE) {
        attr.ub.extMode.cstmFlag.bs.sqCstm = 0; // 表示不指定Va，由HCCP返回Va
        attr.ub.extMode.sq.buffSize = in.sqBufSize;
        attr.ub.extMode.sq.buffVa = in.sqBufVa;
    } // 预埋HrtJettyMode::CACHE_LOCK_DWQE类型，当前流程暂未使用

    // 其他Mode暂时不需要额外更新特定字段
    HCCL_INFO(
        "Create jetty, input params: attr.ub.jettyId[%u], attr.rqDepth[%u], "
        "attr.sqDepth[%u], attr.transportMode[%d], attr.ub.mode[%d], "
        "attr.ub.extMode.sqebbNum[%u], attr.ub.extMode.sq.buffVa[%llx], "
        "attr.ub.extMode.sq.buffSize[%u], attr.ub.extMode.piType[%u], attr.ub.priority[%u], timeout[%u].",
        attr.ub.jettyId, attr.rqDepth, attr.sqDepth, attr.transportMode, attr.ub.mode, attr.ub.extMode.sqebbNum,
        attr.ub.extMode.sq.buffVa, attr.ub.extMode.sq.buffSize, attr.ub.extMode.piType, attr.ub.priority,
        attr.ub.errTimeout);
    return attr;
}

HrtRaUbJettyCreatedOutParam HrtRaUbCreateJetty(RdmaHandle handle, const HrtRaUbCreateJettyParam& in)
{
    CHECK_NULLPTR(handle, "[HrtRaUbCreateJetty] handle is nullptr!");
    HCCL_INFO("[HrtRaUbCreateJetty] Input params: handle=%p", handle);
    struct QpCreateAttr attr = GetQpCreateAttr(in);

    struct QpCreateInfo info {};
    void* qpHandle = nullptr;
    s32 ret = RaCtxQpCreate(handle, &attr, &info, &qpHandle);
    if (ret != 0) {
        string msg = StringFormat("ubCreateJetty failed, rdmaHandle=%p,", handle);
        MACRO_THROW(NetworkApiException, msg);
    }
    PLF_CONFIG_INFO(PLF_RES, "Create Qp para: qpn[%u]", info.ub.id);

    HrtRaUbJettyCreatedOutParam out;
    out.handle = ReinterpretAs<JettyHandle>(qpHandle);
    out.id = info.ub.id;
    out.uasid = info.ub.uasid;
    out.jettyVa = info.va;
    out.dbVa = info.ub.dbAddr;
    out.dbTokenId = info.ub.dbTokenId >> URMA_TOKEN_ID_RIGHT_SHIFT;
    out.sqBuffVa = info.ub.sqBuffVa; // 适配HCCP修改，jettybufva由HCCP提供，不再由HCCL分配

    s32 sRet = memcpy_s(out.key, sizeof(out.key), info.key.value, info.key.size);
    if (sRet != EOK) {
        MACRO_THROW(
            InternalException,
            StringFormat("HrtRaUbCreateJetty memcpy_s failed. sRet[%d], params: handle=%p", sRet, handle));
    }
    out.keySize = info.key.size;
    attr.ub.tokenValue = 0;
    HCCL_INFO("Create jetty success, handle[%llu] jettyVa[%llu]", out.handle, out.jettyVa);
    return out;
}

void HrtRaUbDestroyJetty(JettyHandle jettyHandle)
{
    HCCL_INFO("[HrtRaUbDestroyJetty] Input params: jettyHandle=0x%llx", jettyHandle);
    if (GetPlfDebugConfigValue() & PLF_RES) {
        u32 qpn = HrtGetJettyQpNum(ReinterpretAs<void*>(jettyHandle));
        PLF_CONFIG_INFO(PLF_RES, "Destroy Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
    }
    s32 ret = RaCtxQpDestroy(ReinterpretAs<void*>(jettyHandle));
    if (ret != 0) {
        string msg = StringFormat("ubDestroyJetty failed, jettyHandle=0x%llx", jettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

static HrtRaUbJettyImportedOutParam ImportJetty(
    RdmaHandle handle, u8* key, u32 keyLen, u32 tokenValue, JettyImportExpCfg cfg, JettyImportMode mode,
    TpProtocol protocol = TpProtocol::INVALID)
{
    CHECK_NULLPTR(handle, "[ImportJetty] handle is nullptr!");
    CHECK_NULLPTR(key, "[ImportJetty] key is nullptr!");
    HCCL_INFO("[ImportJetty] Input params: handle=%p, key=%d, keyLen=%u, mode=%d", handle, *key, keyLen, mode);
    if (mode == JettyImportMode::JETTY_IMPORT_MODE_NORMAL) {
        MACRO_THROW(
            NotSupportException, StringFormat("[%s] currently not support JETTY_IMPORT_MODE_NORMAL.", __func__));
    }

    struct QpImportInfoT info {};

    int res = memcpy_s(info.in.key.value, sizeof(info.in.key.value), key, keyLen);
    if (res != 0) {
        MACRO_THROW(InternalException, StringFormat("[%s] memcpy_s failed, ret = %d", __func__, res));
    }
    info.in.key.size = keyLen;

    info.in.ub.mode = mode;
    info.in.ub.tokenValue = tokenValue;
    info.in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
    info.in.ub.type = TargetType::TARGET_TYPE_JETTY;

    info.in.ub.flag.value = 0;
    info.in.ub.flag.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;

    info.in.ub.expImportCfg = cfg;

    if (protocol != TpProtocol::TP && protocol != TpProtocol::CTP && protocol != TpProtocol::UBOE
        && protocol != TpProtocol::UB_RTP) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat("[%s] failed, tp protocol[%s] is not expected.", __func__, protocol.Describe().c_str()));
    }
    // tpType: 0->RTP, 1->CTP
    info.in.ub.tpType = protocol == TpProtocol::TP ? 0 : 1;

    void* remQpHandle = nullptr;
    s32 ret = RaCtxQpImport(handle, &info, &remQpHandle);
    if (ret != 0) {
        string msg = StringFormat("UbImportJetty failed, rdmaHandle=%p,", handle);
        MACRO_THROW(NetworkApiException, msg);
    }

    HrtRaUbJettyImportedOutParam out;
    out.handle = ReinterpretAs<TargetJettyHandle>(remQpHandle);
    out.targetJettyVa = info.out.ub.tjettyHandle;
    out.tpn = info.out.ub.tpn;

    HCCL_INFO("ImportJetty handle[%llu] targetJettyVa[%llu] tpn[%u]", out.handle, out.targetJettyVa, out.tpn);
    info.in.ub.tokenValue = 0;
    return out;
}

static struct JettyImportExpCfg GetTpImportCfg(const JettyImportCfg& jettyImportCfg)
{
    struct JettyImportExpCfg cfg = {};

    cfg.tpHandle = jettyImportCfg.localTpHandle;
    cfg.peerTpHandle = jettyImportCfg.remoteTpHandle;
    cfg.tag = jettyImportCfg.localTag;
    cfg.txPsn = jettyImportCfg.localPsn;
    cfg.rxPsn = jettyImportCfg.remotePsn;

    HCCL_INFO(
        "GetTpImportCfg tpHandle[%llu] peerTpHandle[%llu] tag[%llu] txPsn[%llu] rxPsn[%llu]", cfg.tpHandle,
        cfg.peerTpHandle, cfg.tag, cfg.txPsn, cfg.rxPsn);

    return cfg;
}

HrtRaUbJettyImportedOutParam RaUbImportJetty(RdmaHandle handle, u8* key, u32 keyLen, u32 tokenValue)
{
    CHECK_NULLPTR(handle, "[RaUbImportJetty] handle is nullptr!");
    CHECK_NULLPTR(key, "[RaUbImportJetty] key is nullptr!");
    HCCL_INFO("[RaUbImportJetty] Input params: handle=%p, key=%d, keyLen=%u", handle, *key, keyLen);
    // 该接口仅适配非管控面模式，当前不期望使用
    struct JettyImportExpCfg cfg = {};
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_NORMAL;
    return ImportJetty(handle, key, keyLen, tokenValue, cfg, mode);
}

HrtRaUbJettyImportedOutParam
RaUbTpImportJetty(RdmaHandle handle, u8* key, u32 keyLen, u32 tokenValue, const JettyImportCfg& jettyImportCfg)
{
    CHECK_NULLPTR(handle, "[RaUbTpImportJetty] handle is nullptr!");
    CHECK_NULLPTR(key, "[RaUbTpImportJetty] key is nullptr!");
    HCCL_INFO("[RaUbTpImportJetty] Input params: handle=%p", handle);
    struct JettyImportExpCfg cfg = GetTpImportCfg(jettyImportCfg);
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_EXP;
    return ImportJetty(handle, key, keyLen, tokenValue, cfg, mode, jettyImportCfg.protocol);
}

void HrtRaUbUnimportJetty(RdmaHandle handle, TargetJettyHandle targetJettyHandle)
{
    CHECK_NULLPTR(handle, "[HrtRaUbUnimportJetty] handle is nullptr!");
    HCCL_INFO("[HrtRaUbUnimportJetty] Input params: handle=%p, targetJettyHandle=0x%llx", handle, targetJettyHandle);
    s32 ret = RaCtxQpUnimport(ReinterpretAs<void*>(handle), ReinterpretAs<void*>(targetJettyHandle));
    if (ret != 0) {
        string msg
            = StringFormat("ubCqDestroy failed, rdmaHandle=%p, targetJettyHandle=0x%llx", handle, targetJettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

void HrtRaUbJettyBind(JettyHandle jettyHandle, TargetJettyHandle targetJettyHandle)
{
    HCCL_INFO(
        "[HrtRaUbJettyBind] Input params: jettyHandle=0x%llx, targetJettyHandle=0x%llx", jettyHandle,
        targetJettyHandle);
    s32 ret = RaCtxQpBind(ReinterpretAs<void*>(jettyHandle), ReinterpretAs<void*>(targetJettyHandle));
    if (ret != 0) {
        string msg = StringFormat(
            "ubJettyBind failed, jettyHandle=0x%llx, targetJettyHandle=0x%llx", jettyHandle, targetJettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

void HrtRaUbJettyUnbind(JettyHandle jettyHandle)
{
    HCCL_INFO("[HrtRaUbJettyUnbind] Input params: jettyHandle=0x%llx", jettyHandle);
    s32 ret = RaCtxQpUnbind(ReinterpretAs<void*>(jettyHandle));
    if (ret != 0) {
        string msg = StringFormat("ubJettyUnbind failed, jettyHandle=0x%llx", jettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

const std::map<HrtUbSendWrOpCode, RaUbOpcode> HRT_UB_SEND_WR_OP_CODE_MAP
    = {{HrtUbSendWrOpCode::WRITE, RaUbOpcode::RA_UB_OPC_WRITE},
       {HrtUbSendWrOpCode::WRITE_WITH_NOTIFY, RaUbOpcode::RA_UB_OPC_WRITE_NOTIFY},
       {HrtUbSendWrOpCode::READ, RaUbOpcode::RA_UB_OPC_READ},
       {HrtUbSendWrOpCode::NOP, RaUbOpcode::RA_UB_OPC_NOP}};

const std::map<ReduceOp, u8> HRT_UB_REDUCE_OP_CODE_MAP
    = {{ReduceOp::SUM, 0xA}, {ReduceOp::MAX, 0x8}, {ReduceOp::MIN, 0x9}};

const std::map<DataType, u8> HRT_UB_REDUCE_DATA_TYPE_MAP
    = {{DataType::INT8, 0x0},   {DataType::INT16, 0x1},   {DataType::INT32, 0x2}, {DataType::UINT8, 0x3},
       {DataType::UINT16, 0x4}, {DataType::UINT32, 0x5},  {DataType::FP16, 0x6},  {DataType::FP32, 0x7},
       {DataType::BFP16, 0x8},  {DataType::BF16_SAT, 0x9}};

static void ConstructWrSge(HrtRaUbSendWrReqParam& in, struct WrSgeList& sge)
{
    sge.addr = in.localAddr;
    sge.len = in.size;
    sge.lmemHandle = ReinterpretAs<void*>(in.lmemHandle);
}

static void ConstructSendWrReq(HrtRaUbSendWrReqParam& in, struct WrSgeList& sge, struct SendWrData& sendWr)
{
    // 看一下hccp测试用例的入参
    sendWr.numSge = 1;
    sendWr.sges = &sge;
    sendWr.remoteAddr = in.remoteAddr;
    sendWr.rmemHandle = ReinterpretAs<void*>(in.rmemHandle);
    sendWr.ub.userCtx = 0;
    sendWr.ub.opcode = HRT_UB_SEND_WR_OP_CODE_MAP.at(in.opcode);
    sendWr.ub.flags.value = 0;
    sendWr.ub.flags.bs.compOrder = 1;
    sendWr.ub.flags.bs.completeEnable = in.cqeEn;
    sendWr.ub.flags.bs.fence = 1;
    sendWr.ub.flags.bs.solicitedEnable = 1;
    sendWr.ub.remQpHandle = ReinterpretAs<void*>(in.handle);
    sendWr.ub.flags.bs.inlineFlag = in.inlineFlag;
    if (sendWr.ub.flags.bs.inlineFlag) {
        sendWr.inlineData = in.inlineData;
        sendWr.inlineSize = in.size;
    }
    sendWr.ub.reduceInfo.reduceEn = in.inlineReduceFlag;
    if (sendWr.ub.reduceInfo.reduceEn) {
        sendWr.ub.reduceInfo.reduceOpcode = HRT_UB_REDUCE_OP_CODE_MAP.at(in.reduceOp);
        sendWr.ub.reduceInfo.reduceDataType = HRT_UB_REDUCE_DATA_TYPE_MAP.at(in.dataType);
    }
    if (sendWr.ub.opcode == RaUbOpcode::RA_UB_OPC_WRITE_NOTIFY) {
        sendWr.ub.notifyInfo.notifyData = in.notifyData;
        sendWr.ub.notifyInfo.notifyAddr = in.notifyAddr;
        sendWr.ub.notifyInfo.notifyHandle = ReinterpretAs<void*>(in.notifyHandle);
    }
}

HrtRaUbSendWrRespParam HrtRaUbPostSend(JettyHandle jettyHandle, HrtRaUbSendWrReqParam& in)
{
    struct WrSgeList sge = {};
    struct SendWrData sendWr {};

    ConstructWrSge(in, sge);
    ConstructSendWrReq(in, sge, sendWr);

    HCCL_INFO("Sge addr = 0x%llx", in.localAddr);
    HCCL_INFO("SendWR lmemHandle = 0x%llx", in.lmemHandle); // 和notifyFixedValue能否对齐
    HCCL_INFO("SendWR rmemHandle = 0x%llx", in.rmemHandle); // remote
    HCCL_INFO("SendWR remote addr = 0x%llx", in.remoteAddr);
    HCCL_INFO("SendWR remote qp handle = 0x%llx", in.handle);
    HCCL_INFO("SendWR jetty handle = 0x%llx", jettyHandle);

    SendWrResp sendWrResp{};

    u32 compNum = 0;
    s32 ret = RaBatchSendWr(ReinterpretAs<void*>(jettyHandle), &sendWr, &sendWrResp, 1, &compNum);
    if (ret != 0) {
        string msg = StringFormat("UbJettySendWr failed, jettyHandle=0x%llx,", jettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
    HrtRaUbSendWrRespParam out;
    out.dieId = sendWrResp.doorbellInfo.dieId;
    out.funcId = sendWrResp.doorbellInfo.funcId;
    out.jettyId = sendWrResp.doorbellInfo.jettyId;
    out.piVal = sendWrResp.doorbellInfo.piVal;
    out.dwqeSize = sendWrResp.doorbellInfo.dwqeSize;
    ret = memcpy_s(out.dwqe, sizeof(out.dwqe), sendWrResp.doorbellInfo.dwqe, out.dwqeSize);
    if (ret != 0) {
        string msg = StringFormat("HrtRaUbPostSend copy dwqe failed, ret=%d", ret);
        MACRO_THROW(InternalException, msg);
    }

    return out;
}

std::pair<uint32_t, uint32_t> HraGetDieAndFuncId(RdmaHandle handle)
{
    CHECK_NULLPTR(handle, "[HraGetDieAndFuncId] handle is nullptr!");
    HCCL_INFO("[HraGetDieAndFuncId] Input params: handle=%p", handle);
    struct DevBaseAttr out {};
    auto ret = RaGetDevBaseAttr(handle, &out);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat("[%s] call ra_get_dev_base_attr failed, error code =%d.", __func__, ret));
    }
    return std::make_pair(out.ub.dieId, out.ub.funcId);
}

bool HraGetRtpEnable(RdmaHandle handle)
{
    struct DevBaseAttr out {};
    auto ret = RaGetDevBaseAttr(handle, &out);
    if (ret != 0) {
        THROW<NetworkApiException>(StringFormat("[%s] call RaGetDevBaseAttr failed, error code =%d.", __func__, ret));
    }

    HCCL_RUN_INFO(
        "[%s] rmTpCap[%u] rcTpCap[%u] umTpCap[%u] tpFeat[%u]", __func__, out.ub.rmTpCap.value, out.ub.rcTpCap.value,
        out.ub.umTpCap.value, out.ub.tpFeat.value);

    for (int i = 0; i < MAX_PRIORITY_CNT; i++) {
        const CtxSlInfo& priorityInfo = out.ub.priorityInfo[i];
        HCCL_RUN_INFO(
            "[%s] priorityInfo[%d]: SL[%u] tpType[%u] rtp[%u]", __func__, i, priorityInfo.SL, priorityInfo.tpType.value,
            priorityInfo.tpType.bs.rtp);
        if (priorityInfo.tpType.bs.rtp == 1) {
            return true;
        }
    }
    return false;
}

void HrtRaUbPostNops(JettyHandle jettyHandle, JettyHandle remoteJettyHandle, const u32 numNop)
{
    HCCL_INFO(
        "HrtRaUbPostNops: jettyHandle[0x%llx], remoteJettyHandle[0x%llx], numNop[%u]", jettyHandle, remoteJettyHandle,
        numNop);
    struct SendWrData sendWrList[numNop] = {};
    for (auto& sendWr : sendWrList) {
        sendWr.ub.opcode = HRT_UB_SEND_WR_OP_CODE_MAP.at(HrtUbSendWrOpCode::NOP);
        HCCL_INFO("SendWR opcode = %u", static_cast<u32>(sendWr.ub.opcode));
    }
    sendWrList[numNop - 1].ub.flags.bs.completeEnable = 1;

    SendWrResp sendWrRespList[numNop] = {};
    u32 compNum = 0;
    s32 ret = RaBatchSendWr(ReinterpretAs<void*>(jettyHandle), sendWrList, sendWrRespList, numNop, &compNum);
    if (ret != 0) {
        string msg = StringFormat("UbJettySendWr failed, jettyHandle=0x%llx,", jettyHandle);
        MACRO_THROW(NetworkApiException, msg);
    }
}

void RaUbUpdateCi(JettyHandle jettyHandle, u32 ci)
{
    HCCL_INFO("RaUbUpdateCi: jettyHandle=0x%llx, ci=%u", jettyHandle, ci);
    s32 ret = RaCtxUpdateCi(ReinterpretAs<void*>(jettyHandle), ci);
    if (ret != 0) {
        string msg = StringFormat("UbUpdateCi failed, ret=%d, jettyHandle=0x%llx, ci=%u", ret, jettyHandle, ci);
        MACRO_THROW(NetworkApiException, msg);
    }
}

inline string HccpEidDesc(union HccpEid& hccpEid)
{
    return StringFormat(
        "HccpEid[%016llx:%016llx]", static_cast<unsigned long long>(be64toh(hccpEid.in6.subnetPrefix)),
        static_cast<unsigned long long>(be64toh(hccpEid.in6.interfaceId)));
}

inline IpAddress HccpEidToIpAddress(union HccpEid& hccpEid)
{
    Eid eid{};
    HCCL_INFO("[HccpEidToIpAddress] %s", HccpEidDesc(hccpEid).c_str());
    s32 sRet = memcpy_s(eid.raw, sizeof(eid.raw), hccpEid.raw, sizeof(hccpEid.raw));
    if (sRet != EOK) {
        MACRO_THROW(InternalException, StringFormat("[HccpEidToIpAddress]memcpy_s failed. sRet[%d]", sRet));
    }
    return IpAddress(eid);
}

std::vector<HrtDevEidInfo> HrtRaGetDevEidInfoList(const HRaInfo& raInfo)
{
    std::vector<HrtDevEidInfo> hrtDevEidInfo;
    struct RaInfo info {};
    u32 num = 0;

    info.mode = HRT_NETWORK_MODE_MAP.at(raInfo.mode);
    info.phyId = raInfo.phyId;

    HCCL_INFO("[HrtRaGetDevEidInfoList] Input params: mode=%d, phyId=%u", info.mode, info.phyId);
    s32 ret = RaGetDevEidInfoNum(info, &num);
    if (ret != 0) {
        string msg = StringFormat("call RaGetDevEidInfoNum failed, error code =%d.", ret);
        MACRO_THROW(NetworkApiException, msg);
    }

    struct HccpDevEidInfo infoList[num] = {};
    ret = RaGetDevEidInfoList(info, infoList, &num);
    if (ret != 0) {
        string msg = StringFormat("call RaGetDevEidInfoList failed, error code =%d.", ret);
        MACRO_THROW(NetworkApiException, msg);
    }

    hrtDevEidInfo.resize(num);
    for (u32 i = 0; i < num; i++) {
        hrtDevEidInfo[i].name = (infoList[i].name);
        hrtDevEidInfo[i].ipAddress = HccpEidToIpAddress(infoList[i].eid);
        hrtDevEidInfo[i].type = infoList[i].type;
        hrtDevEidInfo[i].eidIndex = infoList[i].eidIndex;
        hrtDevEidInfo[i].dieId = infoList[i].dieId;
        hrtDevEidInfo[i].chipId = infoList[i].chipId;
        hrtDevEidInfo[i].funcId = infoList[i].funcId;
        hrtDevEidInfo[i].devFeature = infoList[i].devFeature;
        HCCL_INFO(
            "[%s] HrtDevEidInfo[%d]: name[%s], ipAddress[%s], type[%u], "
            "eidIndex[%u], dieId[%u], chipId[%u], funcId[%u], devFeature[%u]",
            __func__, i, hrtDevEidInfo[i].name.c_str(), hrtDevEidInfo[i].ipAddress.Describe().c_str(),
            hrtDevEidInfo[i].type, hrtDevEidInfo[i].eidIndex, hrtDevEidInfo[i].dieId, hrtDevEidInfo[i].chipId,
            hrtDevEidInfo[i].funcId, hrtDevEidInfo[i].devFeature);
    }

    return hrtDevEidInfo;
}

ReqHandleResult HrtRaGetAsyncReqResult(RequestHandle& reqHandle)
{
    if (reqHandle == 0) {
        HCCL_ERROR("[%s] failed, reqHandle is 0.params: reqHandle=0x%llx", __func__, reqHandle);
        return ReqHandleResult::INVALID_PARA;
    }

    int reqResult = 0;
    s32 ret = RaGetAsyncReqResult(ReinterpretAs<void*>(reqHandle), &reqResult);
    // 返回 OTHERS_EAGAIN 代表查询到异步任务未完成，需要重新查询，此时保留handle
    if (ret == OTHERS_EAGAIN) {
        return ReqHandleResult::NOT_COMPLETED;
    }

    // 返回码非0代表调用查询接口失败，当前仅入参错误时触发
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d], "
                                     "reqhandle[%llu].",
                                     __func__, ret, reqHandle));
    }

    RequestHandle tmpReqHandle = reqHandle;
    reqHandle = 0;
    // 返回码为 0 时，reqResult为异步任务完成结果，0代表成功，其他值代表失败
    // SOCK_EAGAIN 为 socket 类执行结果，代表 socket 接口失败需要重试
    if (reqResult == SOCK_EAGAIN) {
        return ReqHandleResult::SOCK_E_AGAIN;
    }

    if (reqResult != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, the asynchronous request "
                                     "error[%d], reqhandle[%llu].",
                                     __func__, reqResult, tmpReqHandle));
    }

    return ReqHandleResult::COMPLETED;
}

RequestHandle RaSocketConnectOneAsync(RaSocketConnectParam& in)
{
    HCCL_INFO(
        "[RaSocketConnectOneAsync] Input params: socketHandle=%p, remoteIp=%s, port=%u, tag=%s", in.socketHandle,
        in.remoteIp.Describe().c_str(), in.port, in.tag.c_str());
    struct SocketConnectInfoT connInfo {};
    connInfo.socketHandle = in.socketHandle;
    connInfo.remoteIp = IpAddressToHccpIpAddr(in.remoteIp);
    connInfo.port = in.port;

    int sret = strcpy_s(connInfo.tag, sizeof(connInfo.tag), in.tag.c_str());
    if (sret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] copy tag[%s] to hccp tag failed, ret=%d, connInfo.tag size=%zu, in.tag size=%zu", __func__,
                in.tag.c_str(), sret, sizeof(connInfo.tag), sizeof(in.tag.c_str())));
    }

    HCCL_INFO("Socket Connect tag=[%s], remoteIp[%s]", connInfo.tag, in.remoteIp.Describe().c_str());
    void* raReqHandle = nullptr;
    int ret = RaSocketBatchConnectAsync(&connInfo, SOCKET_NUM_ONE, &raReqHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[BatchConnect][RaSocket]errNo[0x%016llx] ra socket batch connect fail. return[%d]",
                HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaSocketCloseOneAsync(RaSocketCloseParam& in)
{
    HCCL_INFO("[RaSocketCloseOneAsync] Input params: socketHandle=%p, fdHandle=%p", in.socketHandle, in.fdHandle);
    struct SocketCloseInfoT closeInfo = {};
    closeInfo.fdHandle = in.fdHandle;
    closeInfo.socketHandle = in.socketHandle;

    void* raReqHandle = nullptr;
    int ret = RaSocketBatchCloseAsync(&closeInfo, SOCKET_NUM_ONE, &raReqHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[BatchClose][RaSocket]errNo[0x%016llx] ra socket batch close fail. return[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaSocketListenOneStartAsync(SocketListenInfoT* listenInfo)
{
    if (listenInfo == nullptr) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat("errNo[0x%016llx] listenInfo is nullptr.", HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT)));
    }
    HCCL_INFO("[RaSocketListenOneStartAsync] Input params: listenInfo=%p, port=%u", listenInfo, listenInfo->port);

    void* raReqHandle = nullptr;
    int ret = RaSocketListenStartAsync(listenInfo, SOCKET_NUM_ONE, &raReqHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "errNo[0x%016llx] ra socket listen start fail. return[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaSocketListenOneStopAsync(RaSocketListenParam& in)
{
    HCCL_INFO("[RaSocketListenOneStopAsync] Input params: socketHandle=%p, port=%u", in.socketHandle, in.port);
    struct SocketListenInfoT listenInfo {};
    listenInfo.socketHandle = in.socketHandle;
    listenInfo.port = in.port;

    void* raReqHandle = nullptr;
    int ret = RaSocketListenStopAsync(&listenInfo, SOCKET_NUM_ONE, &raReqHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[ListenStop][RaSocket]errNo[0x%016llx] ra socket listen stop fail. return[%d]",
                                     HCCL_ERROR_CODE(HcclResult::HCCL_E_TCP_CONNECT), ret));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RaSocketFdHandleParam RaGetOneSocket(u32 role, RaSocketGetParam& param)
{
    struct SocketInfoT socketInfo {};

    socketInfo.socketHandle = param.socketHandle;
    socketInfo.fdHandle = param.fdHandle;
    socketInfo.remoteIp = IpAddressToHccpIpAddr(param.remoteIp);
    socketInfo.status = SOCKET_NOT_CONNECTED;

    int sret = strcpy_s(socketInfo.tag, sizeof(socketInfo.tag), param.tag.c_str());
    if (sret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] failed, copy tag[%s] to hccp failed, ret=%d, socketInfo.tag size=%zu, param.tag size=%zu",
                __func__, param.tag.c_str(), sret, sizeof(socketInfo.tag), sizeof(param.tag.c_str())));
    }

    u32 connectedNum = 0;
    s32 sockRet = RaGetSockets(role, &socketInfo, SOCKET_NUM_ONE, &connectedNum);
    if ((connectedNum == 0 && sockRet == 0) || sockRet == SOCK_EAGAIN) {
        // 更新为 connecting 状态，表示连接未完成
        socketInfo.status = SOCKET_CONNECTING;
        return RaSocketFdHandleParam(socketInfo.fdHandle, socketInfo.status);
    }

    if (sockRet != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d], "
                                     "role[%u], num[%u], connectednum[%u]",
                                     __func__, sockRet, role, SOCKET_NUM_ONE, connectedNum));
    }

    if (connectedNum > SOCKET_NUM_ONE) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] failed, connetedNum[%u] is more "
                "than expected[%u], role[%u], num[%u], connectednum[%u]",
                __func__, connectedNum, SOCKET_NUM_ONE, sockRet, role, SOCKET_NUM_ONE, connectedNum));
    }
    return RaSocketFdHandleParam(socketInfo.fdHandle, socketInfo.status);
}

RequestHandle HrtRaSocketSendAsync(const FdHandle fdHandle, const void* data, u32 size, unsigned long long& sentSize)
{
    CHECK_NULLPTR(fdHandle, "[HrtRaSocketSendAsync] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketSendAsync] data is nullptr!");
    HCCL_INFO(
        "[HrtRaSocketSendAsync] Input params: fdHandle=%p, data=%p, size=%u, sentSize=%llu", fdHandle, data, size,
        sentSize);
    void* raReqHandle = nullptr;
    s32 ret = RaSocketSendAsync(fdHandle, data, size, &sentSize, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d] "
                                     "raReqHandle[%p], fdHandle[%p], data[%p], size[%u], sentSize[%u].",
                                     __func__, ret, raReqHandle, fdHandle, data, size, sentSize));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle HrtRaSocketRecvAsync(const FdHandle fdHandle, void* data, u32 size, unsigned long long& recvSize)
{
    CHECK_NULLPTR(fdHandle, "[HrtRaSocketRecvAsync] fdHandle is nullptr!");
    CHECK_NULLPTR(data, "[HrtRaSocketRecvAsync] data is nullptr!");
    PLF_CONFIG_INFO(
        PLF_CHANNEL, "[HrtRaSocketRecvAsync] Input params: fdHandle=%p, data=%p, size=%u, recvSize=%llu", fdHandle,
        data, size, recvSize);

    void* raReqHandle = nullptr;
    s32 ret = RaSocketRecvAsync(fdHandle, data, size, &recvSize, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d], "
                                     "raReqHandle[%p], fdHandle[%p], data[%p], size[%u], recvSize[%u].",
                                     __func__, ret, raReqHandle, fdHandle, data, size, recvSize));
    }

    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle
RaUbLocalMemRegAsync(RdmaHandle handle, const HrtRaUbLocMemRegParam& in, vector<char_t>& out, void*& lmemHandle)
{
    CHECK_NULLPTR(handle, "[RaUbLocalMemRegAsync] handle is nullptr!");
    CHECK_NULLPTR(lmemHandle, "[RaUbLocalMemRegAsync] lmemHandle is nullptr!");
    HCCL_INFO(
        "[RaUbLocalMemRegAsync] Input params: handle=%p, addr=0x%llx, size=0x%llx, lmemHandle=%p", handle, in.addr,
        in.size, lmemHandle);
    u64 pageSize = UB_MEM_PAGE_SIZE;
    u64 newAddr = in.addr & (~(static_cast<u64>(pageSize - 1))); // UB内存注册要求起始地址4k对齐
    u64 offset = in.addr - newAddr;
    u64 newSize = in.size + offset + 4;

    out.resize(sizeof(struct MrRegInfoT));
    struct MrRegInfoT* info = ReinterpretAs<struct MrRegInfoT*>(out.data());
    info->in.mem.addr = newAddr;
    info->in.mem.size = newSize;

    info->in.ub.flags.value = 0;
    info->in.ub.flags.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;
    info->in.ub.flags.bs.tokenIdValid = 1;
    info->in.ub.flags.bs.access = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE | MEM_SEG_ACCESS_ATOMIC;
    info->in.ub.flags.bs.nonPin = in.nonPin;
    info->in.ub.tokenValue = in.tokenValue;
    info->in.ub.tokenIdHandle = ReinterpretAs<void*>(in.tokenIdHandle);

    void* raReqHandle = nullptr;
    s32 ret = RaCtxLmemRegisterAsync(handle, info, &lmemHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface "
                                     "error[%d], raReqHandle[%p], addr=0x%llx, size=0x%llx",
                                     __func__, ret, raReqHandle, in.addr, in.size));
    }
    info->in.ub.tokenValue = 0;
    HCCL_INFO(
        "[%s] RaCtxLmemRegisterAsync success, reqHandle[%llu] addr[0x%llx] size[0x%llx].", __func__,
        ReinterpretAs<RequestHandle>(raReqHandle), in.addr, in.size);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaUbLocalMemUnregAsync(RdmaHandle rdmaHandle, LocMemHandle lmemHandle)
{
    CHECK_NULLPTR(rdmaHandle, "[RaUbLocalMemUnregAsync] rdmaHandle is nullptr!");
    HCCL_INFO("[RaUbLocalMemUnregAsync] Input params: rdmaHandle=%p, lmemHandle=0x%llx", rdmaHandle, lmemHandle);
    void* raReqHandle = nullptr;
    s32 ret = RaCtxLmemUnregisterAsync(rdmaHandle, ReinterpretAs<void*>(lmemHandle), &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d] "
                                     "raReqResult[%p], rdmaHandle=%p, lmemHandle=0x%llx.",
                                     __func__, ret, raReqHandle, rdmaHandle, lmemHandle));
    }

    HCCL_INFO(
        "[%s] RaCtxLmemUnregisterAsync success, reqHandle[%llu] lmemHandle[0x%llx].", __func__,
        ReinterpretAs<RequestHandle>(raReqHandle), lmemHandle);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaUbCreateJettyAsync(
    const RdmaHandle handle, const HrtRaUbCreateJettyParam& in, vector<char_t>& out, void*& jettyHandle)
{
    struct QpCreateAttr attr = GetQpCreateAttr(in);

    void* raReqHandle = nullptr;
    out.resize(sizeof(QpCreateInfo));
    s32 ret = RaCtxQpCreateAsync(handle, &attr, ReinterpretAs<QpCreateInfo*>(out.data()), &jettyHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d], raReqHandle[%p], "
                                     "rdmaHanlde[%p].",
                                     __func__, ret, raReqHandle, handle));
    }
    attr.ub.tokenValue = 0;
    PLF_CONFIG_INFO(
        PLF_RES, "Create RaCtxQpCreateAsync success, reqHandle[%llu] jettyHandle[%p].",
        ReinterpretAs<RequestHandle>(raReqHandle), jettyHandle);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaUbDestroyJettyAsync(void* jettyHandle)
{
    CHECK_NULLPTR(jettyHandle, "[RaUbDestroyJettyAsync] jettyHandle is nullptr!");
    HCCL_INFO("[RaUbDestroyJettyAsync] Input params: jettyHandle=%p", jettyHandle);
    void* raReqHandle = nullptr;
    s32 ret = RaCtxQpDestroyAsync(jettyHandle, &raReqHandle);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d] raReqHandle[%p], "
                                     "jettyHandle[%p].",
                                     __func__, ret, raReqHandle, jettyHandle));
    }

    PLF_CONFIG_INFO(
        PLF_RES, "Destroy RaCtxQpDestroyAsync success, reqHandle[%llu] jettyHandle[%p].",
        ReinterpretAs<RequestHandle>(raReqHandle), jettyHandle);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

inline HccpEid IpAddressToHccpEid(const IpAddress& ipAddr)
{
    HccpEid eid = {};
    HCCL_INFO("EID ipAddr[%s]", ipAddr.Describe().c_str());
    s32 sRet = memcpy_s(eid.raw, sizeof(eid.raw), ipAddr.GetEid().raw, sizeof(ipAddr.GetEid().raw));
    if (sRet != EOK) {
        MACRO_THROW(
            InternalException,
            StringFormat(
                "[IpAddressToHccpEid]memcpy_s failed. sRet[%d], dest[%p], destSize[%zu], src[%p], srcSize[%zu]", sRet,
                eid.raw, sizeof(eid.raw), ipAddr.GetEid().raw, sizeof(ipAddr.GetEid().raw)));
    }
    HCCL_INFO("[IpAddressToHccpEid] %s", HccpEidDesc(eid).c_str());
    return eid;
}

RequestHandle
RaUbGetTpInfoAsync(const RdmaHandle rdmaHandle, const RaUbGetTpInfoParam& param, vector<char_t>& out, uint32_t& num)
{
    CHECK_NULLPTR(rdmaHandle, "[RaUbGetTpInfoAsync] rdmaHandle is nullptr!");
    HCCL_INFO("[RaUbGetTpInfoAsync] Input params: rdmaHandle=%p, num=%u", rdmaHandle, num);
    const auto& locAddr = param.locAddr;
    const auto& rmtAddr = param.rmtAddr;
    const auto& tpProtocol = param.tpProtocol;

    struct GetTpCfg cfg {};
    // UB_RTP与TP同属RTP传输，需使能rtp位；UBOE走独立uboe位
    cfg.flag.bs.rtp = (tpProtocol == TpProtocol::TP || tpProtocol == TpProtocol::UB_RTP) ? 1 : 0;
    cfg.flag.bs.ctp = tpProtocol == TpProtocol::CTP ? 1 : 0;
    cfg.flag.bs.uboe = (tpProtocol == TpProtocol::UBOE) ? 1 : 0;
    cfg.transMode = TransportModeT::CONN_RM; // 当前只使用RM Jetty
    cfg.localEid = IpAddressToHccpEid(locAddr);
    HCCL_INFO("RaUbGetTpInfoAsync cfg.localEid=%s", HccpEidDesc(cfg.localEid).c_str());
    cfg.peerEid = IpAddressToHccpEid(rmtAddr);
    HCCL_INFO("RaUbGetTpInfoAsync cfg.peerEid=%s", HccpEidDesc(cfg.peerEid).c_str());

    // 须至少容纳 TP_HANDLE_REQUEST_NUM 条 HccpTpInfo，避免 RS 按 num 写多条时越界破坏堆
    out.resize(static_cast<size_t>(TP_HANDLE_REQUEST_NUM) * sizeof(struct HccpTpInfo));
    struct HccpTpInfo* info = ReinterpretAs<struct HccpTpInfo*>(out.data());

    void* raReqHandle = nullptr;
    num = TP_HANDLE_REQUEST_NUM; // 指定需要从管控面申请tp handle的数量, hccp 会返回实际个数
    s32 ret = RaGetTpInfoListAsync(rdmaHandle, &cfg, info, &num, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] failed, call interface error[%d] raReqHandle[%p], "
                "rdmaHandle[%p], locAddr[%s], rmtAddr[%s].",
                __func__, ret, raReqHandle, rdmaHandle, locAddr.Describe().c_str(), rmtAddr.Describe().c_str()));
    }

    HCCL_INFO(
        "[%s] RaGetTpInfoListAsync success, reqHandle[%llu] locAddr[%s] rmtAddr[%s] tpNum[%u].", __func__,
        ReinterpretAs<RequestHandle>(raReqHandle), locAddr.Describe().c_str(), rmtAddr.Describe().c_str(), num);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

void RaUbGetTpInfo(const RdmaHandle rdmaHandle, const RaUbGetTpInfoParam& param, vector<char_t>& out, uint32_t& num)
{
    CHECK_NULLPTR(rdmaHandle, "[RaUbGetTpInfo] rdmaHandle is nullptr!");
    HCCL_INFO("[RaUbGetTpInfo] Input params: rdmaHandle=%p, num=%u", rdmaHandle, num);
    const auto& locAddr = param.locAddr;
    const auto& rmtAddr = param.rmtAddr;
    const auto& tpProtocol = param.tpProtocol;

    struct GetTpCfg cfg {};
    // UB_RTP与TP同属RTP传输，需使能rtp位；UBOE走独立uboe位
    cfg.flag.bs.rtp = (tpProtocol == TpProtocol::TP || tpProtocol == TpProtocol::UB_RTP) ? 1 : 0;
    cfg.flag.bs.ctp = tpProtocol == TpProtocol::CTP ? 1 : 0;
    cfg.transMode = TransportModeT::CONN_RM; // 当前只使用RM Jetty
    cfg.localEid = IpAddressToHccpEid(locAddr);
    HCCL_INFO("RaUbGetTpInfo cfg.localEid=%s", HccpEidDesc(cfg.localEid).c_str());
    cfg.peerEid = IpAddressToHccpEid(rmtAddr);
    HCCL_INFO("RaUbGetTpInfo cfg.peerEid=%s", HccpEidDesc(cfg.peerEid).c_str());

    out.resize(static_cast<size_t>(TP_HANDLE_REQUEST_NUM) * sizeof(struct HccpTpInfo));
    struct HccpTpInfo* info = ReinterpretAs<struct HccpTpInfo*>(out.data());

    num = TP_HANDLE_REQUEST_NUM; // 指定需要从管控面申请tp handle的数量, hccp 会返回实际个数
    s32 ret = RaCtxGetTpInfoList(rdmaHandle, &cfg, info, &num);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat(
                "[%s] failed, call interface error[%d], "
                "rdmaHandle[%p], locAddr[%s], rmtAddr[%s].",
                __func__, ret, rdmaHandle, locAddr.Describe().c_str(), rmtAddr.Describe().c_str()));
    }

    HCCL_INFO(
        "[%s] RaCtxGetTpInfoList success, locAddr[%s] rmtAddr[%s] tpNum[%u].", __func__, locAddr.Describe().c_str(),
        rmtAddr.Describe().c_str(), num);
}

static RequestHandle ImportJettyAsync(
    RdmaHandle rdmaHandle, const HrtRaUbJettyImportedInParam& in, vector<char_t>& out, void*& remQpHandle,
    const JettyImportExpCfg& cfg, JettyImportMode mode, TpProtocol protocol = TpProtocol::INVALID)
{
    CHECK_NULLPTR(rdmaHandle, "[ImportJettyAsync] rdmaHandle is nullptr!");
    HCCL_INFO("[ImportJettyAsync] Input params: rdmaHandle=%p, remQpHandle=%p", rdmaHandle, remQpHandle);
    if (mode == JettyImportMode::JETTY_IMPORT_MODE_NORMAL) {
        MACRO_THROW(
            NotSupportException, StringFormat("[%s] currently not support JETTY_IMPORT_MODE_NORMAL.", __func__));
    }

    out.resize(sizeof(QpImportInfoT));
    struct QpImportInfoT* info = ReinterpretAs<QpImportInfoT*>(out.data());

    s32 ret = memcpy_s(info->in.key.value, sizeof(info->in.key.value), in.key, in.keyLen);
    if (ret != 0) {
        MACRO_THROW(InternalException, StringFormat("[%s] memcpy_s failed, ret=%d.", __func__, ret));
    }

    info->in.key.size = in.keyLen;
    info->in.ub.mode = mode;
    info->in.ub.tokenValue = in.tokenValue;
    info->in.ub.policy = JettyGrpPolicy::JETTY_GRP_POLICY_RR;
    info->in.ub.type = TargetType::TARGET_TYPE_JETTY;

    info->in.ub.flag.value = 0;
    info->in.ub.flag.bs.tokenPolicy = TOKEN_POLICY_PLAIN_TEXT;

    info->in.ub.expImportCfg = cfg;

    if (protocol != TpProtocol::TP && protocol != TpProtocol::CTP && protocol != TpProtocol::UBOE
        && protocol != TpProtocol::UB_RTP) {
        MACRO_THROW(
            NetworkApiException,
            StringFormat("[%s] failed, tp protocol[%s] is not expected, %s.", __func__, protocol.Describe().c_str()));
    }
    // tpType: 0->RTP, 1->CTP
    info->in.ub.tpType = protocol == TpProtocol::TP ? 0 : 1;

    void* raReqHandle = nullptr;
    ret = RaCtxQpImportAsync(rdmaHandle, info, &remQpHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d] raReqHandle[%p], "
                                     "rdmaHandle[%p].",
                                     __func__, ret, raReqHandle, rdmaHandle));
    }
    info->in.ub.tokenValue = 0;
    HCCL_INFO(
        "[%s] RaCtxQpImportAsync success, reqHandle[%llu] remQpHandle[%p].", __func__,
        ReinterpretAs<RequestHandle>(raReqHandle), remQpHandle);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

RequestHandle RaUbImportJettyAsync(
    const RdmaHandle rdmaHandle, const HrtRaUbJettyImportedInParam& in, vector<char_t>& out, void*& remQpHandle)
{
    CHECK_NULLPTR(rdmaHandle, "[RaUbImportJettyAsync] rdmaHandle is nullptr!");
    HCCL_INFO("[RaUbImportJettyAsync] Input params: rdmaHandle=%p, remQpHandle=%p", rdmaHandle, remQpHandle);
    // 该接口仅适配非管控面模式，当前不期望使用
    struct JettyImportExpCfg cfg = {};
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_NORMAL;
    return ImportJettyAsync(rdmaHandle, in, out, remQpHandle, cfg, mode);
}

RequestHandle RaUbTpImportJettyAsync(
    const RdmaHandle rdmaHandle, const HrtRaUbJettyImportedInParam& in, vector<char_t>& out, void*& remQpHandle)
{
    CHECK_NULLPTR(rdmaHandle, "[RaUbTpImportJettyAsync] rdmaHandle is nullptr!");
    HCCL_INFO("[RaUbTpImportJettyAsync] Input params: rdmaHandle=%p, remQpHandle=%p", rdmaHandle, remQpHandle);
    struct JettyImportExpCfg cfg = GetTpImportCfg(in.jettyImportCfg);
    const auto mode = JettyImportMode::JETTY_IMPORT_MODE_EXP;
    return ImportJettyAsync(rdmaHandle, in, out, remQpHandle, cfg, mode, in.jettyImportCfg.protocol);
}

RequestHandle RaUbUnimportJettyAsync(void* targetJettyHandle)
{
    CHECK_NULLPTR(targetJettyHandle, "[RaUbUnimportJettyAsync] targetJettyHandle is nullptr!");
    HCCL_INFO("[RaUbUnimportJettyAsync] Input params: targetJettyHandle=%p", targetJettyHandle);
    void* raReqHandle = nullptr;
    s32 ret = RaCtxQpUnimportAsync(targetJettyHandle, &raReqHandle);
    if (ret != 0 || !raReqHandle) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d] raReqHandle[%p], "
                                     "targetJettyHandle[%p].",
                                     __func__, ret, raReqHandle, targetJettyHandle));
    }

    HCCL_INFO(
        "[%s] RaCtxQpUnimportAsync success, reqHandle[%llu] targetJettyHandle[%p].", __func__,
        ReinterpretAs<RequestHandle>(raReqHandle), targetJettyHandle);
    return ReinterpretAs<RequestHandle>(raReqHandle);
}

HcclResult HrtRaWaitEventHandle(
    int event_handle, std::vector<SocketEventInfo>& event_infos, int timeout, unsigned int maxevents, u32& events_num)
{
    HCCL_INFO(
        "[HrtRaWaitEventHandle] Input params: event_handle=[%d], timeout=[%d ms], maxevents=[%u], events_num=[%u]",
        event_handle, timeout, maxevents, events_num);
    std::vector<struct SocketEventInfoT> raEventInfos(maxevents);
    s32 ret = RaWaitEventHandle(event_handle, raEventInfos.data(), timeout, maxevents, &events_num);
    CHK_PRT_RET(
        ret != 0, HCCL_ERROR("[%s] failed, call RaWaitEventHandle error ret[%d].", __func__, ret), HCCL_E_NETWORK);
    for (u32 i = 0; i < events_num; i++) {
        event_infos[i].fdHandle = raEventInfos[i].fdHandle;
    }
    return HCCL_SUCCESS;
}

void HrtRaGetSecRandom(u32* value, u32& devPhyId)
{
    CHECK_NULLPTR(value, "[HrtRaGetSecRandom] value is nullptr!");
    HCCL_INFO("[HrtRaGetSecRandom] Input params: value=%u, devPhyId=%u", *value, devPhyId);
    struct RaInfo raInfo = {};
    raInfo.mode = HrtNetworkMode::HDC;
    raInfo.phyId = devPhyId;

    s32 ret = RaGetSecRandom(&raInfo, value);
    if (ret != 0) {
        MACRO_THROW(
            NetworkApiException, StringFormat(
                                     "[%s] failed, call interface error[%d]. params: value=%u, devPhyId=%u", __func__,
                                     ret, *value, devPhyId));
    }
    HCCL_INFO("[HrtRaGetSecRandom] Input params: value=%u, devPhyId=%u", *value, devPhyId);
}
HcclResult HrtRaCreateQpWithCq(
    RdmaHandle rdmaHandle, s32 sqEvent, s32 rqEvent, void* sendChannel, void* recvChannel, QpInfo& info, bool isHdcMode)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(sendChannel);
    CHK_PTR_NULL(recvChannel);
    HCCL_INFO(
        "[HrtRaCreateQpWithCq] Input params: rdmaHandle=%p, sqEvent=%d, rqEvent=%d, sendChannel=%p, recvChannel=%p",
        rdmaHandle, sqEvent, rqEvent, sendChannel, recvChannel);
    struct ibv_comp_channel* sChannel = ReinterpretAs<struct ibv_comp_channel*>(sendChannel);
    struct ibv_comp_channel* rChannel = ReinterpretAs<struct ibv_comp_channel*>(recvChannel);

    QpConfig config(MAX_WR_NUM, MAX_SEND_SGE_NUM, MAX_RECV_SGE_NUM, sqEvent, rqEvent);
    CqInfo cq(
        nullptr, nullptr, nullptr, MAX_CQ_DEPTH, config.sqEvent, config.rqEvent, info.srqContext, sChannel, rChannel);
    // hdc模式下hccp没有对外提供创建CQ的接口
    if (!isHdcMode) {
        CHK_RET(HrtRaCreateCq(rdmaHandle, cq));
    }
    info.attr = config;
    info.rdmaHandle = rdmaHandle;
    info.context = cq.context;
    info.sendCq = cq.sq;
    info.recvCq = cq.rq;
    info.recvChannel = rChannel;
    info.sendChannel = sChannel;

    if (isHdcMode) {
        TRY_CATCH_RETURN(info.qpHandle = HrtRaQpCreate(rdmaHandle, info.flag, info.qpMode));
    } else {
        CHK_RET(HrtRaNormalQpCreate(rdmaHandle, info));
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaDestroyQpWithCq(const QpInfo& info, bool isHdcMode)
{
    if (info.qpHandle == nullptr) {
        return HCCL_SUCCESS;
    }

    if (isHdcMode) {
        TRY_CATCH_RETURN(HrtRaQpDestroy(info.qpHandle));
    } else {
        CHK_RET(HrtRaNormalQpDestroy(info.qpHandle));
        CqInfo cq;
        cq.context = info.context;
        cq.rq = info.recvCq;
        cq.sq = info.sendCq;
        CHK_RET(HrtRaDestroyCq(info.rdmaHandle, cq));
    }

    return HCCL_SUCCESS;
}

// ra_cq_create
HcclResult HrtRaCreateCq(RdmaHandle rdmaHandle, CqInfo& cq)
{
    CHK_PTR_NULL(rdmaHandle);
    HCCL_INFO(
        "[HrtRaCreateCq] Input params: rdmaHandle=%p, sq=%p, rq=%p, context=%p", rdmaHandle, cq.sq, cq.rq, cq.context);

    struct CqAttr attr {};
    attr.qpContext = &(cq.context);
    attr.ibSendCq = &(cq.sq);
    attr.ibRecvCq = &(cq.rq);
    attr.sendCqDepth = cq.depth;
    attr.recvCqDepth = cq.depth;
    attr.sendCqEventId = cq.sqEvent;
    attr.recvCqEventId = cq.rqEvent;
    attr.sendChannel = cq.sendChannel;
    attr.recvChannel = cq.recvChannel;
    attr.srqContext = cq.srqContext;

    HCCL_DEBUG(
        "ra create cq: send_cq_depth[%d], recv_cq_depth[%d], send_cq_event_id[%d], recv_cq_event_id[%d]",
        attr.sendCqDepth, attr.recvCqDepth, attr.sendCqEventId, attr.recvCqEventId);
    s32 ret = RaCqCreate(rdmaHandle, &attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaCreateCq] errNo[0x%016llx] RaCqCreate fail. "
            "return[%d], params: rdmaHandle[%p], sq[%p], rq[%p], context[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, cq.sq, cq.rq, cq.context),
        HCCL_E_NETWORK);
    if (cq.sq == nullptr || cq.rq == nullptr || cq.context == nullptr) {
        HCCL_ERROR(
            "[HrtRaCreateCq] cq member[sq:%p, rq:%p, context:%p] is nullptr, ret[%d]", cq.sq, cq.rq, cq.context, ret);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}
// ra_cq_destroy
HcclResult HrtRaDestroyCq(RdmaHandle rdmaHandle, CqInfo& cq)
{
    CHK_PTR_NULL(rdmaHandle);
    HCCL_INFO(
        "[HrtRaDestroyCq] Input params: rdmaHandle=%p, sq=%p, rq=%p, context=%p", rdmaHandle, cq.sq, cq.rq, cq.context);
    struct CqAttr attr = {};
    attr.qpContext = &cq.context;
    attr.ibSendCq = &cq.sq;
    attr.ibRecvCq = &cq.rq;
    s32 ret = RaCqDestroy(rdmaHandle, &attr);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[HrtRaDestroyCq] errNo[0x%016llx] RaCqDestroy failed, call interface error. "
            "return[%d], params: rdmaHandle[%p], sq[%p], rq[%p], context[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, cq.sq, cq.rq, cq.context),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

// ra_normal_qp_create
HcclResult HrtRaNormalQpCreate(RdmaHandle rdmaHandle, QpInfo& qp)
{
    CHK_PTR_NULL(rdmaHandle);
    HCCL_INFO("[HrtRaNormalQpCreate] Input params: rdmaHandle=%p, context=%p", rdmaHandle, qp.context);
    struct ibv_qp_init_attr ibQpAttr = {};
    CHK_SAFETY_FUNC_RET(memset_s(&ibQpAttr, sizeof(ibv_qp_init_attr), 0, sizeof(ibv_qp_init_attr)));
    ibQpAttr.qp_context = qp.context;
    ibQpAttr.send_cq = qp.sendCq;
    ibQpAttr.recv_cq = qp.recvCq;
    ibQpAttr.srq = qp.srq;
    ibQpAttr.qp_type = IBV_QPT_RC;
    ibQpAttr.cap.max_inline_data = MAX_INLINE_DATA;
    ibQpAttr.cap.max_send_wr = qp.attr.maxWr;
    ibQpAttr.cap.max_send_sge = qp.attr.maxSendSge;
    ibQpAttr.cap.max_recv_wr = (qp.srq == nullptr ? qp.attr.maxWr : 0);
    ibQpAttr.cap.max_recv_sge = (qp.srq == nullptr ? qp.attr.maxRecvSge : 0);
    s32 ret = RaNormalQpCreate(rdmaHandle, &ibQpAttr, &(qp.qpHandle), ReinterpretAs<void**>(&(qp.qp)));
    RPT_INPUT_ERR(
        ret == ROCE_ENOMEM_RET, "EI0011",
        std::vector<std::string>({"memory_size"}), // A3是当ROCE_ENOMEM_RET才上报EI0011,内存大小取决于qp深度配置
        std::vector<std::string>({"262144~3145728"}));
    if (ret == ROCE_ENOMEM_RET) {
        HCCL_ERROR(
            "[%s][%s] errNo[0x%016llx] RaNormalQpCreate fail due to insufficient memory, "
            "return[%d], rdmaHandle[%p], context[%p]",
            LOG_KEYWORDS_INIT_CHANNEL.c_str(), LOG_KEYWORDS_RESOURCE.c_str(),
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), ret, rdmaHandle, qp.context);
    }
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Create][NormalQp]errNo[0x%016llx] RaNormalQpCreate fail. return[%d], params: rdmaHandle[%p], context[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, qp.context),
        HCCL_E_NETWORK);
    if (GetPlfDebugConfigValue() & PLF_RES) {
        u32 qpn = HrtGetQpNum(qp.qpHandle);
        PLF_CONFIG_INFO(PLF_RES, "Create Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaNormalQpDestroy(QpHandle qpHandle)
{
    CHK_PTR_NULL(qpHandle);
    HCCL_INFO("[HrtRaNormalQpDestroy] Input params: qpHandle=%p", qpHandle);
    if (GetPlfDebugConfigValue() & PLF_RES) {
        u32 qpn = HrtGetQpNum(qpHandle);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
    }
    s32 ret = RaNormalQpDestroy(qpHandle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[Destroy][NormalQp]errNo[0x%016llx] ra destroy normal qp fail. return[%d], params: rdmaHandle[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, qpHandle),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaNdaQpCreate(
    RdmaHandle rdmaHandle, NdaOps* ndaOps, uint32_t dmaMode, NdaCqInfo* cqInfo, NdaQpInfo* qpInfo, QpHandle* qpHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(ndaOps);
    HCCL_INFO("[HrtRaNdaQpCreate] Input params: rdmaHandle=%p dmaMode=%u", rdmaHandle, dmaMode);

    struct ibv_qp_init_attr ibQpAttr;
    CHK_SAFETY_FUNC_RET(memset_s(&ibQpAttr, sizeof(ibv_qp_init_attr), 0, sizeof(ibv_qp_init_attr)));
    ibQpAttr.qp_context = nullptr;
    ibQpAttr.send_cq = cqInfo->cq;
    ibQpAttr.recv_cq = cqInfo->cq;
    ibQpAttr.srq = nullptr;
    ibQpAttr.qp_type = IBV_QPT_RC;
    ibQpAttr.cap.max_inline_data = MAX_INLINE_DATA;
    ibQpAttr.cap.max_send_wr = MAX_WR_NUM;
    ibQpAttr.cap.max_send_sge = MAX_SEND_SGE_NUM;
    ibQpAttr.cap.max_recv_wr = MAX_WR_NUM;
    ibQpAttr.cap.max_recv_sge = MAX_RECV_SGE_NUM;

    struct NdaQpInitAttr qpAttr;
    qpAttr.attr = ibQpAttr;
    qpAttr.qpCapFlag = 0;
    qpAttr.dmaMode = dmaMode;
    qpAttr.ops = ndaOps;

    s32 ret = RaNdaQpCreate(rdmaHandle, &qpAttr, qpInfo, qpHandle);
    CHK_PRT_RET(
        ret != 0 || qpInfo == nullptr || qpHandle == nullptr,
        HCCL_ERROR(
            "[Create][NdaQp]errNo[0x%016llx] RaNdaQpCreate fail. return[%d], "
            "params: rdmaHandle[%p] dmaMode[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, dmaMode),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaNdaCqCreate(
    RdmaHandle rdmaHandle, NdaOps* ndaOps, uint32_t dmaMode, uint32_t cqAttrFlags, NdaCqInfo* cqInfo,
    CqHandle* cqHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(ndaOps);
    HCCL_INFO(
        "[HrtRaNdaCqCreate] Input params: rdmaHandle=%p dmaMode=%u cqAttrFlags=%u", rdmaHandle, dmaMode, cqAttrFlags);

    struct ibv_cq_init_attr_ex ibCqAttr;
    CHK_SAFETY_FUNC_RET(memset_s(&ibCqAttr, sizeof(ibv_cq_init_attr_ex), 0, sizeof(ibv_cq_init_attr_ex)));
    if (dmaMode == QBUF_DMA_MODE_INDEP_UB) {
        ibCqAttr.cqe = NDA_CQ_DEPTH_FOR_UBNIC;
    } else {
        ibCqAttr.cqe = NDA_CQ_DEPTH_FOR_XSCDV;
    }
    ibCqAttr.cq_context = nullptr;
    ibCqAttr.channel = nullptr;
    ibCqAttr.comp_vector = 0;
    ibCqAttr.wc_flags = 0;
    ibCqAttr.comp_mask = 0;
    ibCqAttr.flags = cqAttrFlags;

    struct NdaCqInitAttr cqAttr;
    cqAttr.attr = ibCqAttr;
    cqAttr.cqCapFlag = 0;
    cqAttr.dmaMode = dmaMode;
    cqAttr.ops = ndaOps;

    s32 ret = RaNdaCqCreate(rdmaHandle, &cqAttr, cqInfo, cqHandle);
    CHK_PRT_RET(
        ret != 0 || cqInfo == nullptr || cqHandle == nullptr,
        HCCL_ERROR(
            "[Create][NdaCq]errNo[0x%016llx] RaNdaCqCreate fail. return[%d], "
            "params: rdmaHandle[%p] dmaMode[%u]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, dmaMode),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult HrtRaNdaCqDestroy(RdmaHandle rdmaHandle, CqHandle cqHandle)
{
    CHK_PTR_NULL(rdmaHandle);
    CHK_PTR_NULL(cqHandle);
    HCCL_INFO("[HrtRaNdaCqDestroy] Input params: rdmaHandle=%p cqHandle=%p", rdmaHandle, cqHandle);

    s32 ret = RaNdaCqDestroy(rdmaHandle, cqHandle);
    CHK_PRT_RET(
        ret != 0,
        HCCL_ERROR(
            "[RaNdaCqDestroy] errNo[0x%016llx] RaNdaCqDestroy failed, call interface error. "
            "return[%d], params: rdmaHandle[%p] cqHandle[%p]",
            HCCL_ERROR_CODE(HCCL_E_NETWORK), ret, rdmaHandle, cqHandle),
        HCCL_E_NETWORK);
    return HCCL_SUCCESS;
}

HcclResult
RaBatchQueryJettyStatus(const std::vector<JettyHandle>& jettyHandles, std::vector<JettyStatus>& jettyAttrs, u32& num)
{
    if (jettyHandles.size() != num) {
        HCCL_ERROR("jettyHandles size[%zu] not equal to num[%u]", jettyHandles.size(), num);
        return HCCL_E_PARA;
    }
    std::vector<struct JettyAttr> raJettyAttrs(MAX_JETTY_QUERY_NUM);
    void* qp_handle[jettyHandles.size()];
    for (size_t i = 0; i < jettyHandles.size(); ++i) {
        qp_handle[i] = ReinterpretAs<void*>(jettyHandles[i]);
    }
    auto ret = RaCtxQpQueryBatch(qp_handle, raJettyAttrs.data(), &num);
    if (ret != 0) {
        HCCL_ERROR("RaBatchQueryJettyAttr failed.");
        return HCCL_E_NETWORK;
    }
    if (num != jettyHandles.size()) {
        HCCL_ERROR("jettyAttrs num[%zu] not equal to input jettyHandles size[%zu]", num, jettyHandles.size());
        return HCCL_E_PARA;
    }

    for (u32 i = 0; i < num; i++) {
        JettyStatus jettyStatus = static_cast<JettyStatus::Value>(static_cast<int>(raJettyAttrs[i].state));
        jettyAttrs.push_back(jettyStatus);
    }
    return HCCL_SUCCESS;
}

HcclResult RaGetAuxInfo(const RdmaHandle rdmaHandle, AuxInfoIn auxInfoIn, AuxInfoOut& auxInfoOut)
{
    HccpAuxInfoIn in;
    in.type = static_cast<HccpAuxInfoInType>(static_cast<int>(auxInfoIn.auxInfoInType));
    if (auxInfoIn.auxInfoInType == AuxInfoInType::AUX_INFO_IN_TYPE_CQE) {
        in.cqe.status = auxInfoIn.cqe.status;
        in.cqe.sR = auxInfoIn.cqe.sR;
    } else if (auxInfoIn.auxInfoInType == AuxInfoInType::AUX_INFO_IN_TYPE_AE) {
        in.ae.eventType = auxInfoIn.ae.eventType;
    }

    HccpAuxInfoOut out;
    auto ret = RaCtxGetAuxInfo(rdmaHandle, &in, &out);
    if (ret != 0) {
        HCCL_ERROR("RaGetAuxInfo failed.");
        return HCCL_E_NETWORK;
    }

    auxInfoOut.auxInfoNum = out.auxInfoNum;
    for (uint32_t i = 0; i < out.auxInfoNum; i++) {
        auxInfoOut.auxInfoTypes[i] = out.auxInfoType[i];
        auxInfoOut.auxInfoValues[i] = out.auxInfoValue[i];
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaCtxQpDestoryBatch(
    const RdmaHandle handle, const std::unordered_set<JettyHandle>& jettyHandles,
    std::vector<JettyHandle>& failJettyHandles)
{
    std::vector<void*> qp_handle;
    failJettyHandles.clear();
    for (auto jettyHandle : jettyHandles) {
        qp_handle.push_back(ReinterpretAs<void*>(jettyHandle));
    }
    unsigned int delNum = min(qp_handle.size(), static_cast<size_t>(MAX_DELETE_JETTY_NUMS));
    std::vector<void*> del_qp_handle;
    while (true) {
        void* raReqHandle = nullptr;
        delNum = min(qp_handle.size(), static_cast<size_t>(MAX_DELETE_JETTY_NUMS));
        del_qp_handle.assign(qp_handle.begin(), qp_handle.begin() + delNum);
        if (GetPlfDebugConfigValue() & PLF_RES) {
            for (void* qp : del_qp_handle) {
                u32 qpn = HrtGetJettyQpNum(qp);
                PLF_CONFIG_INFO(
                    PLF_RES, "Destroy Qp para: qpn[%u]%s", qpn, (qpn == HRT_INVALID_QPN) ? " (invalid)" : "");
            }
        }
        auto ret = RaCtxQpDestroyBatchAsync(handle, del_qp_handle.data(), &delNum, &raReqHandle);
        if (ret != 0) {
            HCCL_ERROR("[%s] failed, ret is [%d].", __func__, ret);
            return HCCL_E_INTERNAL;
        }

        RequestHandle reqHandle = ReinterpretAs<RequestHandle>(raReqHandle);
        auto startTime = std::chrono::steady_clock::now();
        constexpr uint32_t pollTimeoutMs = 10000; // 轮询超时时间10s
        auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);
        while (true) {
            if ((std::chrono::steady_clock::now() - startTime) >= waitPollTimeOutMs) {
                HCCL_ERROR(
                    "[%s]poll timeout, originalJettyCount[%zu], undeleteJettyCount[%zu].", __func__,
                    jettyHandles.size(), failJettyHandles.size());
                return HCCL_E_TIMEOUT;
            }
            ReqHandleResult result = ReqHandleResult::INVALID_PARA;
            TRY_CATCH_RETURN(result = HrtRaGetAsyncReqResult(reqHandle));
            if (result == ReqHandleResult::NOT_COMPLETED) {
                continue;
            } else if (result == ReqHandleResult::COMPLETED) {
                break;
            } else {
                HCCL_ERROR("[%s] failed, result[%s] is unexpected.", __func__, result.Describe().c_str());
                return HCCL_E_INTERNAL;
            }
        }

        // 检查是否删除完成
        if (delNum > del_qp_handle.size()) {
            HCCL_ERROR(
                "[%s] run RaCtxQpDestroyBatchAsync error, del jetty num[%u] greater than all jetty num[%zu].", __func__,
                delNum, del_qp_handle.size());
            return HCCL_E_INTERNAL;
        } else if (del_qp_handle.size() == delNum) {
            qp_handle.erase(qp_handle.begin(), qp_handle.begin() + delNum);
        } else {
            failJettyHandles.push_back(ReinterpretAs<JettyHandle>(del_qp_handle[delNum]));
            qp_handle.erase(qp_handle.begin(), qp_handle.begin() + delNum + 1);
        }
        if (qp_handle.size() == 0) {
            break;
        }
    }
    HCCL_INFO(
        "[%s] run success, originalJettyCount[%zu], undeleteJettyCount[%zu].", __func__, jettyHandles.size(),
        failJettyHandles.size());
    return HCCL_SUCCESS;
}

struct ccu_mem_info {
    unsigned int long long mem_va;
    unsigned int mem_size;
    unsigned int resv[1];
};

struct ccu_mem_rsp {
    unsigned int die_id;
    unsigned int num;
    struct ccu_mem_info list[64U];
};

void HrtSetMemInfoList(struct CcuMemInfo* memInfoList, uint32_t count, struct ccu_mem_info* recvMemList)
{
    for (size_t i = 0; i < count; ++i) {
        memInfoList[i].memVa = recvMemList[i].mem_va;
        memInfoList[i].memSize = recvMemList[i].mem_size;
    }
}

HcclResult HrtGetCcuMemInfo(
    void* tlv_handle, uint32_t udieIdx, uint64_t memTypeBitmap, struct CcuMemInfo* memInfoList, uint32_t count)
{
    s32 ret = 0;
    u32 tlv_module_type = TLV_MODULE_TYPE_CCU;

    struct TlvMsg send_msg = {};
    struct TlvMsg recv_msg = {};
    // 使用unique_ptr管理动态分配的内存，实现RAII
    auto send_data = std::make_unique<char[]>(sizeof(CcuMemReq));
    auto recv_data = std::make_unique<char[]>(sizeof(ccu_mem_rsp));

    // 初始化请求消息
    send_msg.type = MSG_TYPE_CCU_GET_MEM_INFO;
    send_msg.length = sizeof(CcuMemReq);
    send_msg.data = send_data.get();

    auto req = ReinterpretAs<CcuMemReq*>(send_msg.data);
    req->udieIdx = udieIdx;
    req->memTypeBitmap = memTypeBitmap;

    // 初始化响应消息
    recv_msg.type = 0;
    recv_msg.length = sizeof(ccu_mem_rsp);
    recv_msg.data = recv_data.get();

    auto rsp = ReinterpretAs<ccu_mem_rsp*>(recv_msg.data);
    rsp->die_id = 0;
    rsp->num = 0;
    std::fill(std::begin(rsp->list), std::end(rsp->list), ccu_mem_info{});

    ret = RaTlvRequest(tlv_handle, tlv_module_type, &send_msg, &recv_msg);
    if (ret != 0) {
        if (ret == RA_TLV_REQUEST_UNAVAIL) {
            HCCL_WARNING("[HrtGetCcuMemInfo]ra tlv request UNAVAIL. return: ret[%d]", ret);
            return HCCL_E_UNAVAIL;
        }
        HCCL_ERROR(
            "[Request][RaTlv]errNo[0x%016llx] ra tlv request fail. return: ret[%d], module type[%u], message type[%u]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_NETWORK), ret, tlv_module_type, send_msg.type);
        throw NetworkApiException(StringFormat("call ra_tlv_request failed"));
    }
    HrtSetMemInfoList(memInfoList, count, rsp->list);
    HCCL_INFO("tlv request success, tlv module type[%u], message type[%u]", tlv_module_type, send_msg.type);
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetEidByIp(RdmaHandle handle, const vector<IpAddress>& ipV4AddrList, vector<IpAddress>& eidAddrList)
{
    HCCL_INFO("[HrtRaGetEidByIp] begin, ipV4AddrList size=%zu", ipV4AddrList.size());
    size_t ipV4AddrListSize = ipV4AddrList.size();
    unsigned int num = ipV4AddrListSize;
    IpInfo ipInfoList[num] = {};
    for (size_t i = 0; i < num; i++) {
        auto ipAddress = ipV4AddrList.at(i);
        HCCL_INFO("[HrtRaGetEidByIp] ipV4AddrList[%d][%s]", i, ipAddress.Describe().c_str());
        ipInfoList[i].family = ipAddress.GetFamily();
        ipInfoList[i].ip = IpAddressToHccpIpAddr(ipAddress);
    }

    union HccpEid eidList[num] = {};
    s32 ret = RaGetEidByIp(handle, ipInfoList, eidList, &num);
    if (ret != 0) {
        HCCL_WARNING("call RaGetEidByIp failed, error code =%d.", ret);
        return HCCL_E_INTERNAL;
    }

    if (num != ipV4AddrList.size()) {
        HCCL_ERROR(
            "call RaGetEidByIp failed, The number of ipInfoList and eidList is inconsistent, "
            "ipV4AddrList size =%zu, eidList size =%u",
            ipV4AddrList.size(), num);
        return HCCL_E_INTERNAL;
    }

    for (unsigned int i = 0; i < num; i++) {
        IpAddress eidAddr = HccpEidToIpAddress(eidList[i]);
        eidAddrList.push_back(eidAddr);
    }
    HCCL_INFO("[HrtRaGetEidByIp] success, eidAddrList size=%zu", eidAddrList.size());
    return HCCL_SUCCESS;
}

HcclResult WaitRequestResult(void* raReqHandle, RequestHandle& reqHandle)
{
    reqHandle = ReinterpretAs<RequestHandle>(raReqHandle);
    auto startTime = std::chrono::steady_clock::now();
    constexpr uint32_t pollTimeoutMs = 10000; // 轮询超时时间
    auto waitPollTimeOutMs = std::chrono::milliseconds(pollTimeoutMs);
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) >= waitPollTimeOutMs) {
            HCCL_ERROR("[WaitRequestResult] poll timeout.");
            return HCCL_E_TIMEOUT; // 超时报错
        }

        ReqHandleResult result = ReqHandleResult::INVALID_PARA;
        TRY_CATCH_RETURN(result = HrtRaGetAsyncReqResult(reqHandle));

        // 结果判断
        if (result == ReqHandleResult::NOT_COMPLETED) {
            continue;
        } else if (result == ReqHandleResult::COMPLETED) {
            break;
        } else {
            HCCL_ERROR("[WaitRequestResult] failed, result[%s] is unexpected.", result.Describe().c_str());
            return HCCL_E_INTERNAL;
        }
    }

    return HCCL_SUCCESS;
}

HcclResult HrtRaGetTpAttr(RdmaHandle handle, uint64_t tpHandle, uint32_t& attrBitmap, TpAttr& attr)
{
    const s32 ret = RaCtxGetTpAttr(handle, tpHandle, &attrBitmap, &attr);
    if (ret != 0) {
        HCCL_ERROR(
            "[HrtRaGetTpAttr] RaCtxGetTpAttr failed ret[%d] tpHandle[%llu] attrBitmap[0x%x].", ret, tpHandle,
            attrBitmap);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtRaSetTpAttr(RdmaHandle handle, uint64_t tpHandle, uint32_t attrBitmap, TpAttr& attr)
{
    const s32 ret = RaCtxSetTpAttr(handle, tpHandle, attrBitmap, &attr);
    if (ret != 0) {
        HCCL_ERROR(
            "[HrtRaSetTpAttr] RaCtxSetTpAttr failed ret[%d] tpHandle[%llu] attrBitmap[0x%x].", ret, tpHandle,
            attrBitmap);
        return HCCL_E_NETWORK;
    }
    return HCCL_SUCCESS;
}

bool HrtRaSupportsGetTpAttr(u32 phyId)
{
    u32 tpAttrVersion = 0;
    const s32 ret = RaGetInterfaceVersion(phyId, GET_TP_ATTR_OPCODE, &tpAttrVersion);
    return (ret == 0 && tpAttrVersion >= GET_TP_ATTR_VERSION);
}

HcclResult
HrtRaSetTpAttrAsync(RdmaHandle handle, uint64_t tpHandle, uint32_t attrBitmap, TpAttr& attr, RequestHandle& reqHandle)
{
    HCCL_INFO("[HrtRaSetTpAttrAsync] begin, reqHandle[%llu]", reqHandle);
    void* raReqHandle = nullptr;
    s32 ret = RaSetTpAttrAsync(handle, tpHandle, attrBitmap, &attr, &raReqHandle);
    if (ret != 0) {
        string msg = StringFormat("call RaSetTpAttrAsync failed, error code =%d.", ret);
        THROW<NetworkApiException>(msg);
    }

    CHK_RET(WaitRequestResult(raReqHandle, reqHandle));
    HCCL_INFO("[HrtRaSetTpAttrAsync] success, reqHandle[%llu]", reqHandle);
    return HCCL_SUCCESS;
}

HcclResult HrtRaGetTpAttrAsync(
    u32 phyId, RdmaHandle handle, uint64_t tpHandle, uint32_t& attrBitmap, TpAttr& attr, RequestHandle& reqHandle)
{
    HCCL_INFO("[HrtRaGetTpAttrAsync] begin, reqHandle[%llu]", reqHandle);
    if (!HrtRaSupportsGetTpAttr(phyId)) {
        HCCL_ERROR(
            "this package does not support RaGetTpAttrAsync for device, please change new package, phyId=%u.", phyId);
        return HCCL_E_NOT_SUPPORT;
    }
    void* raReqHandle = nullptr;
    s32 ret = RaGetTpAttrAsync(handle, tpHandle, &attrBitmap, &attr, &raReqHandle);
    if (ret != 0) {
        string msg = StringFormat("call RaGetTpAttrAsync failed, error code =%d.", ret);
        THROW<NetworkApiException>(msg);
    }

    CHK_RET(WaitRequestResult(raReqHandle, reqHandle));
    HCCL_INFO("[HrtRaGetTpAttrAsync] success, reqHandle[%llu]", reqHandle);
    return HCCL_SUCCESS;
}

HcclResult HrtRaStartGetTpAttrAsync(
    RdmaHandle handle, uint64_t tpHandle, uint32_t& attrBitmap, TpAttr& attr, RequestHandle& reqHandle)
{
    void* raReqHandle = nullptr;
    const s32 ret = RaGetTpAttrAsync(handle, tpHandle, &attrBitmap, &attr, &raReqHandle);
    if (ret != 0 || raReqHandle == nullptr) {
        HCCL_ERROR(
            "[HrtRaStartGetTpAttrAsync] RaGetTpAttrAsync failed ret[%d] raReqHandle[%p] tpHandle[%llu].", ret,
            raReqHandle, tpHandle);
        return HCCL_E_NETWORK;
    }
    reqHandle = ReinterpretAs<RequestHandle>(raReqHandle);
    return HCCL_SUCCESS;
}

HcclResult HrtGetUboeFlagEnable(const u32 devPhyId)
{
    u32 uboeVersion = 0;
    s32 versionRet = RaGetInterfaceVersion(devPhyId, GET_UBOE_FLAG_ENABLE_OPCODE, &uboeVersion);
    CHK_PRT_RET(
        versionRet != 0,
        HCCL_ERROR("[%s] RaGetInterfaceVersion failed, devPhyId=%u, versionRet=%d", __func__, devPhyId, versionRet),
        HCCL_E_INTERNAL);
    CHK_PRT_RET(
        uboeVersion < GET_UBOE_FLAG_ENABLE_VERSION,
        HCCL_ERROR(
            "[%s] this package does not support to get uboe flag, "
            "please change new package. uboeVersion[%u].",
            __func__, uboeVersion),
        HCCL_E_NOT_SUPPORT);
    return HCCL_SUCCESS;
}

} // namespace Hccl
