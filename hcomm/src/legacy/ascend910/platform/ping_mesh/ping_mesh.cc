/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <chrono>
#include "securec.h"
#include "adapter_hccp.h"
#include "adapter_tdt.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include "adapter_rts_common.h"
#include "dispatcher_task_types.h"
#include "network_manager_pub.h"
#include "externalinput.h"
#include "dlra_function.h"
#include "sal_pub.h"
#include "ping_mesh.h"
#include "hccp_ping.h"
#include "hccp_ctx.h"
#include "local_ub_rma_buffer.h"
#include "orion_adapter_hccp.h"

namespace hccl {
constexpr int HOP_MAX_TIMES = 64;              // 最大跳数
constexpr int LOG_CHK_TIMES = 50;              // 需要打印日志的轮询次数
constexpr u32 WR_DEPTH_MULTIPLE = 4;           // wr深度扩展倍数
constexpr u32 BYTE_PER_TARGET_DEFAULT = 2048;  // 记录每个target的result时默认需要的buffersize大小
constexpr u32 BYTE_IPV4_SHIFT_IN_GID = 12;     // ipv4地址相比于gid首地址的偏移值
constexpr u32 RPING_PAYLOAD_REFILL_LEN = 136;  // A2/3payload头需要重填的长度
constexpr u32 RPING_PAYLOAD_RSVD_LEN = 44;     // A2/3payload头需要清零的长度
constexpr u32 RPING_PAYLOAD_UB_HEAD_LEN = 256; // A5payload头需要填充的内存大小
constexpr u32 RPING_PAYLOAD_UB_TIME_LEN = 64;  // A5payload头需要填充的times
constexpr u32 QPINFO_UB_KEY_LEN = 28;          // jetty存储

enum class RpingInitState {
    HCCL_INIT_SUCCESS,
    HCCL_TSD_NEED_CLOSE,
    HCCL_RA_NEED_DEINIT,
    HCCL_RAPING_NEED_DEINIT,
    HCCL_NET_NEED_CLOSE,
    RESERVED
};

PingMesh::PingMesh() {}

PingMesh::~PingMesh()
{
    if (!isDeinited_) { // 如果没有主动释放过资源，析构时需要释放一下
        HccnRpingDeinit(deviceLogicId_);
    }
}

static bool isInitialized = false; // 标记是否已经初始化
static std::mutex ubTokenMutex;

inline HcclResult
GetUbToken(u32 devicePhyId, u32* client_qp_token, u32* client_seg_token, u32* server_qp_token, u32* server_seg_token)
{
    std::lock_guard<std::mutex> lock(ubTokenMutex);
    if (!isInitialized) {
        u32 devPhyId = devicePhyId;
        struct RaInfo raInfo = {};
        raInfo.mode = HrtNetworkMode::HDC;
        raInfo.phyId = devPhyId;
        HcclResult ret = hrtRaGetSecRandom(&raInfo, client_qp_token);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("get hrtRaGetSecRandom client_qp_token failed, ret:%d", ret);
            return ret;
        }
        ret = hrtRaGetSecRandom(&raInfo, client_seg_token);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("get hrtRaGetSecRandom client_seg_token failed, ret:%d", ret);
            return ret;
        }
        ret = hrtRaGetSecRandom(&raInfo, server_qp_token);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("get hrtRaGetSecRandom server_qp_token failed, ret:%d", ret);
            return ret;
        }
        ret = hrtRaGetSecRandom(&raInfo, server_seg_token);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("get hrtRaGetSecRandom server_seg_token failed, ret:%d", ret);
            return ret;
        }
        isInitialized = true;
    }

    return HCCL_SUCCESS;
}

bool IsSupportHCCLV2(const char* socNamePtr)
{
    std::string targetChipVerStr = socNamePtr;
    HCCL_DEBUG("[%s]SocVersion = %s.", __func__, targetChipVerStr.c_str());
    if (targetChipVerStr.find("Ascend950") != std::string::npos) {
        return true;
    }

    if (targetChipVerStr.find("Ascend960") != std::string::npos
        || targetChipVerStr.find("ascend960") != std::string::npos) {
        return true;
    }

    return false;
}

HcclResult GetAddrType(u32* addrType)
{
    CHK_PTR_NULL(addrType);
    const char* socNamePtr = aclrtGetSocName();
    CHK_PTR_NULL(socNamePtr);
    if (IsSupportHCCLV2(socNamePtr)) {
        *addrType = HCCN_RPING_ADDR_TYPE_EID;
    } else {
        *addrType = HCCN_RPING_ADDR_TYPE_IP;
    }
    return HCCL_SUCCESS;
}

inline HcclResult UninitStateCheck(RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (nextState) {
        case RpingState::UNINIT:
            HCCL_INFO("[HCCN][UninitStateCheck]Device is uninited, does not need uninit.");
            break;
        case RpingState::INITED:
            break;
        case RpingState::READY:
            HCCL_ERROR("[HCCN][UninitStateCheck]Device is not inited yet.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        case RpingState::RUN:
            HCCL_ERROR("[HCCN][UninitStateCheck]Device is not inited yet.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        case RpingState::STOP:
            HCCL_ERROR("[HCCN][UninitStateCheck]Device is not inited yet.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        default:
            HCCL_ERROR("[HCCN][UninitStateCheck]Undefined behavior.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

inline HcclResult InitedStateCheck(RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (nextState) {
        case RpingState::UNINIT:
            break;
        case RpingState::INITED:
            HCCL_INFO("[HCCN][InitedStateCheck]Device is inited already.");
            break;
        case RpingState::READY:
            break;
        case RpingState::RUN:
            HCCL_ERROR("[HCCN][InitedStateCheck]Device is not ready.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        case RpingState::STOP:
            HCCL_ERROR("[HCCN][InitedStateCheck]Device is not ready.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        default:
            HCCL_ERROR("[HCCN][InitedStateCheck]Undefined behavior.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

inline HcclResult ReadyStateCheck(RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (nextState) {
        case RpingState::UNINIT:
            break;
        case RpingState::INITED:
            break;
        case RpingState::READY:
            break;
        case RpingState::RUN:
            break;
        case RpingState::STOP:
            HCCL_ERROR("[HCCN][ReadyStateCheck]Device has not run tasks yet.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
        default:
            HCCL_ERROR("[HCCN][ReadyStateCheck]Undefined behavior.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

inline HcclResult RunStateCheck(RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (nextState) {
        case RpingState::UNINIT:
            HCCL_WARNING("[HCCN][RunStateCheck]Make sure the task is finished and result has already gotten.");
            break;
        case RpingState::INITED:
            break;
        case RpingState::READY:
            break;
        case RpingState::RUN:
            break;
        case RpingState::STOP:
            break;
        default:
            HCCL_ERROR("[HCCN][RunStateCheck]Undefined behavior.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

inline HcclResult StopStateCheck(RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (nextState) {
        case RpingState::UNINIT:
            break;
        case RpingState::INITED:
            break;
        case RpingState::READY:
            break;
        case RpingState::RUN:
            break;
        case RpingState::STOP:
            HCCL_WARNING("[HCCN][StopStateCheck]Task is stopped.");
            break;
        default:
            HCCL_ERROR("[HCCN][StopStateCheck]Undefined behavior.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

inline HcclResult RpingstateCheck(RpingState currState, RpingState nextState)
{
    HcclResult ret = HCCL_SUCCESS;
    switch (currState) {
        case RpingState::UNINIT:
            ret = UninitStateCheck(nextState);
            break;
        case RpingState::INITED:
            ret = InitedStateCheck(nextState);
            break;
        case RpingState::READY:
            ret = ReadyStateCheck(nextState);
            break;
        case RpingState::RUN:
            ret = RunStateCheck(nextState);
            break;
        case RpingState::STOP:
            ret = StopStateCheck(nextState);
            break;
        default:
            HCCL_ERROR("[HCCN][RpingstateCheck]Current state doesn't exist.");
            ret = HCCL_E_NOT_SUPPORT;
            break;
    }

    return ret;
}

const std::string extPam[TSD_EXT_PARA_NUM]
    = {std::string("--hdcType=" + std::to_string(HDC_SERVICE_TYPE_RDMA_V2)),
       std::string("--whiteListStatus=" + std::to_string(WHITE_LIST_CLOSE))};
inline void TsdProcessOpenInit(rtNetServiceOpenArgs& openArgs, rtProcExtParam* extParam)
{
    for (u32 i = 0; i < TSD_EXT_PARA_NUM; i++) {
        extParam[i].paramInfo = extPam[i].c_str();
        extParam[i].paramLen = extPam[i].size();
    }
    openArgs.extParamList = extParam;
    openArgs.extParamCnt = TSD_EXT_PARA_NUM;
    HCCL_INFO(
        "[HCCN]TsdProcessOpenInit extPar0[%s] size[%llu], extPar1[%s] size[%llu]", extParam[0].paramInfo,
        extParam[0].paramLen, extParam[1].paramInfo, extParam[1].paramLen);
}

inline void RpingRoceAttrInit(
    u32 deviceId, HcclIpAddress ipAddr, [[maybe_unused]] u32 port, u32 nodeNum, u32 bufferSize, u32 sl, u32 tc,
    PingInitAttr& initAttr)
{
    u32 maxWrDepth = nodeNum * WR_DEPTH_MULTIPLE;
    maxWrDepth = (maxWrDepth > DEFAULT_OPBASE_MAX_SEND_WR) ? DEFAULT_OPBASE_MAX_SEND_WR : maxWrDepth;

    initAttr.version = 0;            // 暂时无用，默认给0
    initAttr.mode = NETWORK_OFFLINE; // net work mode 枚举值
    initAttr.dev.rdma.phyId = deviceId;
    initAttr.dev.rdma.family = ipAddr.GetFamily(); // AF_INET(ipv4) or AF_INET6(ipv6)
    initAttr.dev.rdma.localIp.addr = ipAddr.GetBinaryAddress().addr;
    initAttr.dev.rdma.localIp.addr6 = ipAddr.GetBinaryAddress().addr6;
    initAttr.bufferSize = bufferSize == 0 ? (maxWrDepth * BYTE_PER_TARGET_DEFAULT) : bufferSize; // 发送接收缓存区大小
    initAttr.protocol = PROTOCOL_RDMA; // pingmesh支持兼容UB驱动，新增protocol字段

    // client的初始化信息
    initAttr.client.rdma.cqAttr.sendCqDepth = maxWrDepth;
    initAttr.client.rdma.cqAttr.recvCqDepth = maxWrDepth;
    initAttr.client.rdma.cqAttr.sendCqCompVector = 0; // 一组cqe组成的集合，这里给0
    initAttr.client.rdma.cqAttr.recvCqCompVector = 1; // 一组cqe组成的集合，这里给1
    initAttr.client.rdma.qpAttr.cap.maxSendWr = maxWrDepth;
    initAttr.client.rdma.qpAttr.cap.maxRecvWr = maxWrDepth;
    initAttr.client.rdma.qpAttr.cap.maxSendSge = DEFAULT_MAX_SEND_SGE;
    initAttr.client.rdma.qpAttr.cap.maxRecvSge = DEFAULT_MAX_RECV_SGE;
    initAttr.client.rdma.qpAttr.cap.maxInlineData = DEFAULT_MAX_INLINE_DATA;
    initAttr.client.rdma.qpAttr.udpSport = 0;

    // server的初始化信息
    initAttr.server.rdma.cqAttr.sendCqDepth = maxWrDepth;
    initAttr.server.rdma.cqAttr.recvCqDepth = maxWrDepth;
    initAttr.server.rdma.cqAttr.sendCqCompVector = 0; // 一组cqe组成的集合，这里给0
    initAttr.server.rdma.cqAttr.recvCqCompVector = 1; // 一组cqe组成的集合，这里给1
    initAttr.server.rdma.qpAttr.cap.maxSendWr = maxWrDepth;
    initAttr.server.rdma.qpAttr.cap.maxRecvWr = maxWrDepth;
    initAttr.server.rdma.qpAttr.cap.maxSendSge = DEFAULT_MAX_SEND_SGE;
    initAttr.server.rdma.qpAttr.cap.maxRecvSge = DEFAULT_MAX_RECV_SGE;
    initAttr.server.rdma.qpAttr.cap.maxInlineData = DEFAULT_MAX_INLINE_DATA;
    initAttr.server.rdma.qpAttr.udpSport = 0;

    // ip协议信息
    initAttr.commInfo.version = 0;
    initAttr.commInfo.rdma.flowLabel = 0;
    initAttr.commInfo.rdma.hopLimit = HOP_MAX_TIMES;
    initAttr.commInfo.rdma.qosAttr.sl = sl;
    initAttr.commInfo.rdma.qosAttr.tc = tc;
}

inline HcclResult RpingUbAttrInit(
    u32 deviceId, HcclIpAddress ipAddr, [[maybe_unused]] u32 port, u32 nodeNum, u32 bufferSize, u32 sl, u32 tc,
    PingInitAttr& initAttr, std::map<Eid, uint32_t> eidmap)
{
    u32 maxWrDepth = nodeNum * WR_DEPTH_MULTIPLE;
    maxWrDepth = (maxWrDepth > DEFAULT_OPBASE_MAX_SEND_WR) ? DEFAULT_OPBASE_MAX_SEND_WR : maxWrDepth;

    initAttr.version = 0;            // 暂时无用，默认给0
    initAttr.mode = NETWORK_OFFLINE; // net work mode 枚举值
    initAttr.ub.phyId = deviceId;
    if (eidmap.find(ipAddr.GetEid()) == eidmap.end()) {
        HCCL_ERROR("eidmap does not have the input Eid, input Eid: %s", ipAddr.GetEid().Describe().c_str());
        return HCCL_E_NOT_FOUND;
    }
    initAttr.dev.ub.eidIndex = eidmap.at(ipAddr.GetEid()); // 从eid_list获取eidIndex
    u32 ret = memcpy_s(
        initAttr.dev.ub.eid.raw, sizeof(initAttr.dev.ub.eid.raw), ipAddr.GetEid().raw, sizeof(ipAddr.GetEid().raw));
    if (ret != 0) {
        HCCL_ERROR("memcpy_s Eid failed");
        return HCCL_E_MEMORY;
    }
    initAttr.bufferSize = bufferSize == 0 ? (maxWrDepth * BYTE_PER_TARGET_DEFAULT) : bufferSize; // 发送接收缓存区大小
    initAttr.protocol = PROTOCOL_UDMA; // pingmesh支持兼容UB驱动，新增protocol字段

    // 获取安全随机数
    u32 client_qp_token = 0;
    u32 client_seg_token = 0;
    u32 server_qp_token = 0;
    u32 server_seg_token = 0;
    HcclResult token_ret
        = GetUbToken(deviceId, &client_qp_token, &client_seg_token, &server_qp_token, &server_seg_token);
    CHK_RET(token_ret);
    // client的初始化信息
    initAttr.client.ub.cqAttr.sendCqDepth = maxWrDepth;
    initAttr.client.ub.cqAttr.recvCqDepth = maxWrDepth;
    initAttr.client.ub.cqAttr.sendCqCompVector = 0; // 一组cqe组成的集合，这里给0
    initAttr.client.ub.cqAttr.recvCqCompVector = 1; // 一组cqe组成的集合，这里给1
    initAttr.client.ub.qpAttr.cap.maxSendWr = maxWrDepth;
    initAttr.client.ub.qpAttr.cap.maxRecvWr = maxWrDepth;
    initAttr.client.ub.qpAttr.cap.maxSendSge = DEFAULT_MAX_SEND_SGE;
    initAttr.client.ub.qpAttr.cap.maxRecvSge = DEFAULT_MAX_RECV_SGE;
    initAttr.client.ub.qpAttr.cap.maxInlineData = DEFAULT_MAX_INLINE_DATA;
    initAttr.client.ub.qpAttr.tokenValue = client_qp_token;
    initAttr.client.ub.segAttr.tokenValue = client_seg_token;

    // server的初始化信息
    initAttr.server.ub.cqAttr.sendCqDepth = maxWrDepth;
    initAttr.server.ub.cqAttr.recvCqDepth = maxWrDepth;
    initAttr.server.ub.cqAttr.sendCqCompVector = 0; // 一组cqe组成的集合，这里给0
    initAttr.server.ub.cqAttr.recvCqCompVector = 1; // 一组cqe组成的集合，这里给1
    initAttr.server.ub.qpAttr.cap.maxSendWr = maxWrDepth;
    initAttr.server.ub.qpAttr.cap.maxRecvWr = maxWrDepth;
    initAttr.server.ub.qpAttr.cap.maxSendSge = DEFAULT_MAX_SEND_SGE;
    initAttr.server.ub.qpAttr.cap.maxRecvSge = DEFAULT_MAX_RECV_SGE;
    initAttr.server.ub.qpAttr.cap.maxInlineData = DEFAULT_MAX_INLINE_DATA;
    initAttr.server.ub.qpAttr.tokenValue = server_qp_token;
    initAttr.server.ub.segAttr.tokenValue = server_seg_token;

    // ip协议信息
    initAttr.commInfo.version = 0;
    initAttr.commInfo.ub.qosAttr.sl = sl;
    initAttr.commInfo.ub.qosAttr.tc = tc;
    return HCCL_SUCCESS;
}
const std::unordered_map<HrtNetworkMode, NetworkMode, std::EnumClassHash> HRT_NETWORK_MODE_MAP
    = {{HrtNetworkMode::PEER, NetworkMode::NETWORK_PEER_ONLINE}, {HrtNetworkMode::HDC, NetworkMode::NETWORK_OFFLINE}};

// add查询eidIndex
inline HcclResult RaGetEidMap(std::map<Eid, uint32_t>& eidmap, const HRaInfo& raInfo)
{
    struct RaInfo info {};
    u32 num = 0;
    s32 ret = 0;

    auto iter = HRT_NETWORK_MODE_MAP.find(raInfo.mode);
    if (iter == HRT_NETWORK_MODE_MAP.end()) {
        HCCL_ERROR("[RaGetEidMap]HRT_NETWORK_MODE_MAP not found mode[%d].", raInfo.mode);
        return HCCL_E_NOT_FOUND;
    }
    info.mode = iter->second;
    info.phyId = raInfo.phyId;

    ret = hrtRaGetDevEidInfoNum(info, &num);
    if (ret != 0) {
        HCCL_ERROR("call RaGetDevEidInfoNum failed, error code = %d.", ret);
        return HCCL_E_NETWORK; // ra接口是网络相关调用
    }
    if (num == 0) {
        HCCL_WARNING("call hrtRaGetDevEidInfoNum return num = 0.");
        return HCCL_SUCCESS;
    }

    HccpDevEidInfo* infoList = new (std::nothrow) HccpDevEidInfo[num];
    CHK_PTR_NULL(infoList);
    ret = hrtRaGetDevEidInfoList(info, infoList, &num);
    if (ret != 0 || num == 0) {
        HCCL_ERROR("call RaGetDevEidInfoList failed num = %u, error code = %d.", num, ret);
        delete[] infoList;
        return HCCL_E_NETWORK;
    }

    // 填充map
    for (u32 i = 0; i < num; i++) {
        Eid eid;
        ret = memcpy_s(eid.raw, sizeof(eid.raw), infoList[i].eid.raw, sizeof(infoList[i].eid.raw));
        if (ret != 0) {
            HCCL_ERROR("[RaGetEidMap]memcpy_s failed, error code = %d.", ret);
            delete[] infoList;
            return HCCL_E_INTERNAL;
        }
        HCCL_RUN_INFO("[RaGetEidMap] eid[%s], eidIndex[%u] get.", eid.Describe().c_str(), infoList[i].eidIndex);
        eidmap.insert(std::make_pair(eid, infoList[i].eidIndex));
    }
    delete[] infoList;

    return HCCL_SUCCESS;
}

inline HcclResult
RpingTargetAttrInitWithUb(PingTargetInfo& ubtarget, RpingInput ubinput, PingQpInfo* ubinfo, bool isAddTargetUb)
{
    ubtarget.remoteInfo.qpInfo.version = ubinfo->version;
    ubtarget.remoteInfo.qpInfo.ub.size = ubinfo->ub.size;
    u32 ret = 0;
    ret = memcpy_s(
        ubtarget.remoteInfo.qpInfo.ub.key, sizeof(ubtarget.remoteInfo.qpInfo.ub.key), ubinfo->ub.key,
        QPINFO_UB_KEY_LEN);
    if (ret != 0) {
        HCCL_ERROR("[RpingTargetAttrInitWithUb]memcpy_s key failed, error code = %d.", ret);
        return HCCL_E_INTERNAL;
    }
    ubtarget.remoteInfo.qpInfo.ub.tokenValue = ubinfo->ub.tokenValue;
    ret = memcpy_s(
        ubtarget.remoteInfo.eid.raw, sizeof(ubtarget.remoteInfo.eid.raw), ubinput.dip.GetEid().raw, URMA_EID_LEN);
    if (ret != 0) {
        HCCL_ERROR("[RpingTargetAttrInitWithUb]memcpy_s eid failed, error code = %d.", ret);
        return HCCL_E_INTERNAL;
    }
    ubtarget.localInfo.ub.qosAttr.tc = ubinput.tc;
    ubtarget.localInfo.ub.qosAttr.sl = ubinput.sl;
    if (!isAddTargetUb) { // 并非添加target的时候调用，不需要拷贝payload信息
        return HCCL_SUCCESS;
    }
    if (ubinput.len > PING_USER_PAYLOAD_MAX_SIZE) {
        HCCL_WARNING(
            "[HCCN][RpingTargetAttrInit]Payload length is %u, should be less than %u byte.", ubinput.len,
            PING_USER_PAYLOAD_MAX_SIZE);
        ubtarget.payload.size = 0;
        return HCCL_SUCCESS;
    }
    ubtarget.payload.size = ubinput.len;
    errno_t memRet = memcpy_s(ubtarget.payload.buffer, ubtarget.payload.size, ubinput.payload, ubinput.len);
    if (memRet != EOK) {
        HCCL_ERROR(
            "[HCCN][RpingTargetAttrInit]Memcpy ret %d, dst:%p, dstMax:%u, src:%p, length:%u", memRet,
            ubtarget.payload.buffer, ubtarget.payload.size, ubinput.payload, ubinput.len);
        return HCCL_E_MEMORY;
    }

    return HCCL_SUCCESS;
}

inline HcclResult
RpingTargetAttrInit(PingTargetInfo& target, RpingInput input, PingQpInfo* rdmainfo, bool isAddTargetUb)
{
    target.remoteInfo.qpInfo.version = rdmainfo->version;
    target.remoteInfo.qpInfo.rdma.gid = rdmainfo->rdma.gid;
    target.remoteInfo.qpInfo.rdma.qpn = rdmainfo->rdma.qpn;
    target.remoteInfo.qpInfo.rdma.qkey = rdmainfo->rdma.qkey;
    target.remoteInfo.ip.addr = input.dip.GetBinaryAddress().addr;
    target.remoteInfo.ip.addr6 = input.dip.GetBinaryAddress().addr6;
    target.localInfo.rdma.qosAttr.tc = input.tc;
    target.localInfo.rdma.qosAttr.sl = input.sl;
    target.localInfo.rdma.flowLabel = 0;
    target.localInfo.rdma.hopLimit = HOP_MAX_TIMES;
    target.localInfo.rdma.udpSport = input.srcPort;
    if (!isAddTargetUb) { // 并非添加target的时候调用，不需要拷贝payload信息
        return HCCL_SUCCESS;
    }
    if (input.len > PING_USER_PAYLOAD_MAX_SIZE) {
        HCCL_WARNING(
            "[HCCN][RpingTargetAttrInit]Payload length is %u, should be less than %u byte.", input.len,
            PING_USER_PAYLOAD_MAX_SIZE);
        target.payload.size = 0;
        return HCCL_SUCCESS;
    }
    target.payload.size = input.len;
    errno_t memRet = memcpy_s(target.payload.buffer, target.payload.size, input.payload, input.len);
    if (memRet != EOK) {
        HCCL_ERROR(
            "[HCCN][RpingTargetAttrInit]Memcpy ret %d, dst:%p, dstMax:%u, src:%p, length:%u", memRet,
            target.payload.buffer, target.payload.size, input.payload, input.len);
        return HCCL_E_MEMORY;
    }

    return HCCL_SUCCESS;
}

HcclResult PingMesh::RpingResultInfoInit(
    PingTargetResult* resultInfo, UniversalConcurrentMap<std::string, PingQpInfo>& rdmaInfoMaps, RpingInput* input,
    u32 targetNum)
{
    u32 addressType = 0;
    HcclResult addrTypeRet = GetAddrType(&addressType);
    if (addrTypeRet != HCCL_SUCCESS) {
        HCCL_ERROR("[RpingResultInfoInit]GetAddrType Fail ret %d", addrTypeRet);
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < targetNum; i++) {
        if (!rdmaInfoMaps.Find(std::string(input[i].dip.GetReadableIP())).second) {
            HCCL_WARNING("[HCCN][RpingResultInfoInit]Target[%s] info doesn't exist.", input[i].dip.GetReadableIP());
            continue;
        }
        PingQpInfo* rdmainfo = &rdmaInfoMaps[std::string(input[i].dip.GetReadableIP())];
        if (addressType == HCCN_RPING_ADDR_TYPE_IP) {
            resultInfo[i].remoteInfo.ip.addr = input[i].dip.GetBinaryAddress().addr;
            resultInfo[i].remoteInfo.ip.addr6 = input[i].dip.GetBinaryAddress().addr6;
            resultInfo[i].remoteInfo.qpInfo.version = 0;
            resultInfo[i].remoteInfo.qpInfo.rdma.gid = rdmainfo->rdma.gid;
            resultInfo[i].remoteInfo.qpInfo.rdma.qpn = rdmainfo->rdma.qpn;
            resultInfo[i].remoteInfo.qpInfo.rdma.qkey = rdmainfo->rdma.qkey;
        }
        const char* socNamePtr = aclrtGetSocName();
        CHK_PTR_NULL(socNamePtr);
        if (addressType == HCCN_RPING_ADDR_TYPE_EID && IsSupportHCCLV2(socNamePtr)) {
            u32 ret = 0;
            ret = memcpy_s(
                resultInfo[i].remoteInfo.eid.raw, sizeof(resultInfo[i].remoteInfo.eid.raw), input[i].dip.GetEid().raw,
                URMA_EID_LEN);
            if (ret != 0) {
                HCCL_ERROR("[RpingResultInfoInit]memcpy_s eid failed, error code = %d.", ret);
                return HCCL_E_INTERNAL;
            }
            resultInfo[i].remoteInfo.qpInfo.version = 0;
            resultInfo[i].remoteInfo.qpInfo.ub.size = rdmainfo->ub.size;
            ret = memcpy_s(
                resultInfo[i].remoteInfo.qpInfo.ub.key, sizeof(resultInfo[i].remoteInfo.qpInfo.ub.key),
                rdmainfo->ub.key, QPINFO_UB_KEY_LEN);
            if (ret != 0) {
                HCCL_ERROR("[RpingResultInfoInit]memcpy_s key failed, error code = %d.", ret);
                return HCCL_E_INTERNAL;
            }
            resultInfo[i].remoteInfo.qpInfo.ub.tokenValue = rdmainfo->ub.tokenValue;
        }
        HCCL_INFO("[HCCN][RpingResultInfoInit]Target[%s] info init success.", input[i].dip.GetReadableIP());
    }
    return HCCL_SUCCESS;
}

inline void GetResultFromReturnValue(PingTargetResult* resultInfo, RpingOutput* output, u32 targetNum)
{
    for (u32 i = 0; i < targetNum; i++) {
        output[i].state = resultInfo[i].result.state;
        output[i].txPkt = resultInfo[i].result.summary.sendCnt;
        output[i].rxPkt = resultInfo[i].result.summary.recvCnt;
        output[i].minRTT = resultInfo[i].result.summary.rttMin;
        output[i].maxRTT = resultInfo[i].result.summary.rttMax;
        output[i].avgRTT = resultInfo[i].result.summary.rttAvg;
    }
}

inline void LogRecordbyTimes(int& count)
{
    // 日志过滤, 50次才打印一次
    if (count % LOG_CHK_TIMES == 0) {
        HCCL_DEBUG("[HCCN][LogRecordbyTimes]socket is connecting...");
    }
    count++;
}

inline void RemoveMapInfo(
    RpingInput* input, u32 targetNum, std::map<std::string, std::shared_ptr<HcclSocket>>& socketMaps,
    UniversalConcurrentMap<std::string, PingQpInfo>& rdmaInfoMaps,
    UniversalConcurrentMap<std::string, u32>& payloadLenMap)
{
    for (u32 i = 0; i < targetNum; i++) {
        socketMaps.erase(std::string(input[i].dip.GetReadableIP()));
        rdmaInfoMaps.Erase(std::string(input[i].dip.GetReadableIP()));
        payloadLenMap.Erase(std::string(input[i].dip.GetReadableIP()));
    }
}

HcclResult PingMesh::RpingSendInitInfo(
    u32 deviceId, [[maybe_unused]] u32 port, HcclIpAddress ipAddr, PingInitInfo initInfo,
    [[maybe_unused]] std::shared_ptr<HcclSocket> socket)
{
    // 给当前线程添加名字
    SetThreadName("Hccl_PingMesh");
    // 等待client端发送的建链请求
    HcclIpAddress remoteIp = HcclIpAddress();
    std::string tag = "PingMesh" + std::string(ipAddr.GetReadableIP());
    HCCL_INFO("[HCCN][RpingSendInitInfo]socket tag[%s].", tag.c_str());
    // 持续在后台等待建链，保证可以处理多个client端的建链请求
    while (true) {
        HcclSocket realSocket(tag, netCtx_, remoteIp, 0, HcclSocketRole::SOCKET_ROLE_SERVER);
        CHK_RET(realSocket.Init());
        int count = 0; // 轮询计数
        while (true) {
            bool isStop = connThreadStop_.load();
            if (isStop == true) {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
                HCCL_INFO("[HCCN][RpingSendInitInfo]Device[%u] stop waiting connect.", deviceId);
                break;
            }
            HcclSocketStatus status = realSocket.GetStatus();
            if (status == HcclSocketStatus::SOCKET_OK) {
                HCCL_DEBUG(
                    "[HCCN][RpingSendInitInfo]socket is established. localIp[%s], remoteIp[%s]",
                    realSocket.GetLocalIp().GetReadableIP(), realSocket.GetRemoteIp().GetReadableIP());
                break;
            } else if (status == HcclSocketStatus::SOCKET_CONNECTING) {
                SaluSleep(ONE_MILLISECOND_OF_USLEEP);
                LogRecordbyTimes(count);
                continue;
            } else if (status == HcclSocketStatus::SOCKET_TIMEOUT) {
                HCCL_WARNING("[HCCN][RpingSendInitInfo]socket connect timeout.");
                break;
            } else {
                HCCL_WARNING("[HCCN][RpingSendInitInfo]socket connect failed.");
                break;
            }
        }

        // 判断否需要中止线程
        bool isStop = connThreadStop_.load();
        if (isStop == true) {
            HCCL_INFO("[HCCN][RpingSendInitInfo]Device[%u] background thread stopped.", deviceId);
            break;
        }

        // 建链成功，发送rping初始化信息
        u64 sendSize = sizeof(initInfo);
        CHK_RET(realSocket.Send(&initInfo, sendSize));
        HCCL_INFO("[HCCN][RpingSendInitInfo]Device[%u] rdma info send success.", deviceId);
    }

    return HCCL_SUCCESS;
}

HcclResult
PingMesh::RpingRecvTargetInfo(void* clientNetCtx, u32 port, HcclIpAddress ipAddr, PingInitInfo& recvInfo, u32 timeout)
{
    // 确认是否添加过该IP
    std::unique_lock<std::mutex> lock(socketMapsMtx_);
    if (socketMaps_.find(std::string(ipAddr.GetReadableIP())) != socketMaps_.end()) {
        HCCL_WARNING("[HCCN][RpingRecvTargetInfo]IP address[%s] has already exist.", ipAddr.GetReadableIP());
        return HCCL_SUCCESS;
    }
    lock.unlock();
    // socket建链 这里的建链流程与init里的侦听动作应当使用同一套接口
    std::string tag = "PingMesh" + std::string(ipAddr.GetReadableIP());
    HCCL_INFO("[HCCN][RpingRecvTargetInfo]socket tag[%s].", tag.c_str());
    std::shared_ptr<HcclSocket> socket = nullptr;
    EXCEPTION_CATCH(
        (socket = std::make_shared<HcclSocket>(tag, clientNetCtx, ipAddr, port, HcclSocketRole::SOCKET_ROLE_CLIENT)),
        return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(socket);
    CHK_RET(socket->Init());
    CHK_RET(socket->Connect());
    auto startTime = std::chrono::steady_clock::now();
    while (true) {
        auto endTime = std::chrono::steady_clock::now();
        // 计算毫秒差值并转为u32
        auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(endTime - startTime);
        u32 ms = static_cast<u32>(duration_ms.count());
        if (ms >= timeout) {
            HCCL_ERROR(
                "[HCCN][RpingRecvTargetInfo]Get socket timeout! cost time [%u ms], timeout [%u ms]", ms, timeout);
            socket->SetStatus(HcclSocketStatus::SOCKET_TIMEOUT);
            return HCCL_E_TIMEOUT;
        }

        auto status = socket->GetStatus();
        if (status == HcclSocketStatus::SOCKET_CONNECTING) {
            SaluSleep(ONE_MILLISEC);
            HCCL_INFO("[HCCN][RpingRecvTargetInfo]connecting to server [%s] port [%u]", ipAddr.GetReadableIP(), port);
            continue;
        } else if (status != HcclSocketStatus::SOCKET_OK) {
            HCCL_ERROR("[HCCN][RpingRecvTargetInfo]Get socket failed, ret [%d]", status);
            return HCCL_E_TCP_CONNECT;
        } else {
            HCCL_INFO(
                "[HCCN][RpingRecvTargetInfo]Get socket success with server [%s] port [%u]", ipAddr.GetReadableIP(),
                port);
            break;
        }
    }

    // 接收发送的信息
    u32 recvBufLen = sizeof(recvInfo);
    CHK_RET(socket->Recv(&recvInfo, recvBufLen));
    HCCL_INFO("[HCCN][RpingRecvTargetInfo]Server[%s] info received success.", ipAddr.GetReadableIP());

    // 记录socket
    lock.lock();
    socketMaps_.insert({std::string(ipAddr.GetReadableIP()), socket});

    return HCCL_SUCCESS;
}

inline RpingLinkState ConvertHcclSocketStatus(HcclSocketStatus socketStatus)
{
    RpingLinkState status = RpingLinkState::DISCONNECTED;
    switch (socketStatus) {
        case HcclSocketStatus::SOCKET_INIT:
            status = RpingLinkState::DISCONNECTED;
            break;
        case HcclSocketStatus::SOCKET_OK:
            status = RpingLinkState::CONNECTED;
            break;
        case HcclSocketStatus::SOCKET_TIMEOUT:
            status = RpingLinkState::TIMEOUT;
            break;
        case HcclSocketStatus::SOCKET_CONNECTING:
            status = RpingLinkState::CONNECTING;
            break;
        default:
            status = RpingLinkState::ERROR;
            break;
    }
    return status;
}

HcclResult PingMesh::HccnRaInit(u32 deviceId)
{
    RaInitConfig config
        = {devicePhyId_, static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_DEVICE), HDC_SERVICE_TYPE_RDMA_V2, false};
    u32 rpingInterfaceVersion = 0;
    CHK_RET(NetworkManager::GetInstance(deviceLogicId_).PingMeshRaPingInit(deviceLogicId_, devicePhyId_, &config));
    CHK_RET(hrtRaGetInterfaceVersion(devicePhyId_, RPING_INTERFACE_OPCODE, &rpingInterfaceVersion));
    if (rpingInterfaceVersion < RPING_INTERFACE_VERSION) {
        HCCL_ERROR(
            "[HCCN][HccnRpingInit]this package[%u] does not support rpingInterface for device.", rpingInterfaceVersion);
        return HCCL_E_NOT_SUPPORT;
    }
    HCCL_INFO("[HCCN][HccnRpingInit]Device[%u] init hccp success.", deviceId);
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnCloseSubProc(u32 deviceId)
{
    hrtCloseNetService();
    HCCL_INFO("[HCCN][HccnCloseSubProc]Device[%u] close hccp process success.", deviceId);
    return HCCL_SUCCESS;
}

HcclResult PingMesh::StartSocketThread(u32 deviceId, HcclIpAddress ipAddr, u32 port)
{
    socket_ = std::make_shared<HcclSocket>(netCtx_, port);
    // 初始化socket并启动侦听
    CHK_RET(socket_->Init());
    CHK_RET(SetTcpMode(true));
    CHK_RET(socket_->Listen());
    HCCL_INFO("[HCCN][StartSocketThread]Device[%u] starts listen port[%u].", deviceId, port);
    // 等待客户端建链
    connThread_.reset(new (std::nothrow)
                          std::thread(&PingMesh::RpingSendInitInfo, this, deviceId, port, ipAddr, initInfo_, socket_));
    CHK_SMART_PTR_NULL(connThread_);
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnSupportedAndGetphyid(u32 deviceId, LinkType netMode)
{
    if (netMode != LinkType::LINK_ROCE && netMode != LinkType::LINK_UB) {
        HCCL_ERROR("[HCCN][HccnSupportedAndGetphyid]only support ROCE or UB mode.");
        return HCCL_E_NOT_SUPPORT;
    }
    // 获取并验证设备物理id
    deviceLogicId_ = deviceId;
    CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<u32>(deviceLogicId_), devicePhyId_));
    if (deviceId != static_cast<u32>(deviceLogicId_)) {
        HCCL_ERROR(
            "[HCCN][HccnSupportedAndGetphyid]Input device logicId[%u] don't match real logicId[%d].", deviceId,
            deviceLogicId_);
        return HCCL_E_PARA;
    }
    HCCL_INFO("[HCCN][HccnSupportedAndGetphyid]Device logic id is [%d], phy id is [%u].", deviceLogicId_, devicePhyId_);
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingOpenTsd(u32 deviceId, u32 mode, u32 port, u32 nodeNum, u32 bufferSize, u32 sl, u32 tc)
{
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::INITED));
    HCCL_DEBUG(
        "[HccnRpingOpenTsd]deviceid %u, mode %u, port %u, nodeNum %u, bufferSize %u, sl %u, tc %u", deviceId, mode,
        port, nodeNum, bufferSize, sl, tc);
    // 当前只支持RoCE和UB
    LinkType netMode = static_cast<LinkType>(mode);
    HcclResult ret = HCCL_SUCCESS;
    ret = HccnSupportedAndGetphyid(deviceId, netMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HCCN][HccnRpingOpenTsd]HccnSupportedAndGetphyid Failed, deviceId[%u] ret[%d].", deviceId, ret);
        return HCCL_E_NOT_SUPPORT;
    }

    // 拉起hccp进程
    rtProcExtParam extParam[TSD_EXT_PARA_NUM]{};

    rtNetServiceOpenArgs openArgs;
    TsdProcessOpenInit(openArgs, extParam);
    CHK_RET(DlTdtFunction::GetInstance().DlTdtFunctionHeterogInit());

    CHK_RET(hrtOpenNetService(&openArgs));
    HCCL_INFO("[HCCN][HccnRpingOpenTsd]Device[%u] open process success", deviceId);
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingInit(
    u32 deviceId, u32 mode, HcclIpAddress ipAddr, u32 port, u32 nodeNum, u32 bufferSize, u32 sl, u32 tc)
{
    CHK_RET(HccnRpingOpenTsd(deviceId, mode, port, nodeNum, bufferSize, sl, tc));

    LinkType netMode = static_cast<LinkType>(mode);
    RpingInitState status = RpingInitState::HCCL_INIT_SUCCESS;
    HcclResult ret = HCCL_SUCCESS;
    void* pingHandle = nullptr;
    const char* socNamePtr = aclrtGetSocName();
    do {
        // hccp侧初始化ping mesh资源
        ret = HccnRaInit(deviceId);
        if (ret != HCCL_SUCCESS) {
            status = RpingInitState::HCCL_TSD_NEED_CLOSE;
            HCCL_ERROR("[HCCN][HccnRpingInit]HccnRaInit fail, deviceId[%u] ret[%d].", deviceId, ret);
            break;
        }
        PingInitAttr initAttr{};
        if (netMode == LinkType::LINK_ROCE) {
            RpingRoceAttrInit(devicePhyId_, ipAddr, port, nodeNum, bufferSize, sl, tc, initAttr);
        }
        if (netMode == LinkType::LINK_UB && IsSupportHCCLV2(socNamePtr)) {
            HRaInfo info(HrtNetworkMode::HDC, devicePhyId_);
            std::map<Eid, uint32_t> eidmap;
            ret = RaGetEidMap(eidmap, info);
            if (ret != HCCL_SUCCESS) {
                status = RpingInitState::HCCL_TSD_NEED_CLOSE;
                HCCL_ERROR(
                    "[HccnRpingInit]call ra_get_dev_eid_map failed, deviceId[%u], error code =%d.", deviceId, ret);
                break;
            }
            ret = RpingUbAttrInit(devicePhyId_, ipAddr, port, nodeNum, bufferSize, sl, tc, initAttr, eidmap);
            if (ret != HCCL_SUCCESS) {
                status = RpingInitState::HCCL_TSD_NEED_CLOSE;
                HCCL_ERROR("[HccnRpingInit]RpingUbAttrInit failed, deviceId[%u], error code =%d.", deviceId, ret);
                break;
            }
        }
        ret = hrtRaPingInit(&initAttr, &initInfo_, &pingHandle);
        if (ret != HCCL_SUCCESS || pingHandle == nullptr) {
            status = RpingInitState::HCCL_RA_NEED_DEINIT;
            HCCL_ERROR(
                "[HCCN][HccnRpingInit]hrtRaPingInit fail, deviceId[%u] ret[%d] pingHandle[%p].", deviceId, ret,
                pingHandle);
            break;
        }
        HCCL_INFO("[HCCN][HccnRpingInit]Device[%u] init success.", deviceId);

        // 建链并发送初始化信息
        ret = HcclNetOpenDev(&netCtx_, NicType::DEVICE_NIC_TYPE, devicePhyId_, deviceLogicId_, ipAddr);
        if (ret != HCCL_SUCCESS || netCtx_ == nullptr) {
            status = RpingInitState::HCCL_RAPING_NEED_DEINIT;
            HCCL_ERROR(
                "[HCCN][HccnRpingInit]HcclNetOpenDev fail, deviceId[%u] ret[%d] netCtx_[%p].", deviceId, ret, netCtx_);
            break;
        }

        ret = StartSocketThread(deviceId, ipAddr, port);
        if (ret != HCCL_SUCCESS) {
            status = RpingInitState::HCCL_NET_NEED_CLOSE;
            HCCL_ERROR("[HCCN][HccnRpingInit]StartSocketThread fail, deviceId[%u] port[%d].", deviceId, port);
            break;
        }
    } while (0);

    switch (status) {
        case RpingInitState::HCCL_INIT_SUCCESS:
            break;
        case RpingInitState::HCCL_NET_NEED_CLOSE:
            if (netCtx_ != nullptr) {
                HcclNetCloseDev(netCtx_);
                netCtx_ = nullptr;
            }
            [[fallthrough]];
        case RpingInitState::HCCL_RAPING_NEED_DEINIT:
            if (pingHandle != nullptr) {
                (void)hrtRaPingDeinit(pingHandle);
            }
            [[fallthrough]];
        case RpingInitState::HCCL_RA_NEED_DEINIT:
            (void)NetworkManager::GetInstance(static_cast<s32>(deviceId)).PingMeshRaPingDeinit();
            [[fallthrough]];
        case RpingInitState::HCCL_TSD_NEED_CLOSE:
            (void)HccnCloseSubProc(deviceId);
            [[fallthrough]];
        default:
            HCCL_ERROR("[HCCN][HccnRpingInit]HccnRpingInit ret[%d], status[%d].", ret, status);
            return ret;
    }
    // 绑定信息
    pingHandle_ = pingHandle;
    rpingState_ = RpingState::INITED;
    ipAddr_ = ipAddr;
    isUsePayload_ = bufferSize == 0 ? false : true;

    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingDeinit(u32 deviceId)
{
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::UNINIT));
    CHK_PRT_RET(
        rpingState_ == RpingState::UNINIT, HCCL_WARNING("[HCCN][HccnRpingDeinit]Device[%u] has not inited.", deviceId),
        HCCL_SUCCESS);
    HCCL_DEBUG("[HccnRpingDeinit]deviceid %u", deviceId);
    // 释放payload内存
    if (payload_ != nullptr) {
        delete[] payload_;
        payload_ = nullptr;
    }

    // 手动结束背景线程
    connThreadStop_.store(true);
    if (connThread_ != nullptr && connThread_->joinable()) {
        connThread_->join();
        HCCL_INFO("[HCCN][HccnRpingDeinit]Device[%u] end background thread success.", deviceId);
    }

    // 清空map
    std::unique_lock<std::mutex> lock(socketMapsMtx_);
    for (auto& socket : socketMaps_) {
        if (socket.second->DeInit() != HCCL_SUCCESS) {
            HCCL_WARNING("[HCCN][HccnRpingDeinit]socket deinit failed");
        }
    }
    socketMaps_.clear();
    HCCL_INFO("[HCCN][HccnRpingDeinit]Socket map clear.");
    lock.unlock();
    rdmaInfoMaps_.Clear();
    HCCL_INFO("[HCCN][HccnRpingDeinit]Rdma info map clear.");
    payloadLenMap_.Clear();
    HCCL_INFO("[HCCN][HccnRpingDeinit]payloadLen map clear.");

    // 关闭socket链路
    if ((socket_ != nullptr) && (!isSocketClosed_)) {
        CHK_RET(socket_->DeInit());
        isSocketClosed_ = true;
        HCCL_INFO("[HCCN][HccnRpingDeinit]Device[%u] deinit socket success.", deviceId);
    }

    // 释放资源
    if (netCtx_ != nullptr) {
        HcclNetCloseDev(netCtx_);
        netCtx_ = nullptr;
    }

    if (pingHandle_ == nullptr) {
        HCCL_WARNING("[HCCN][HccnRpingDeinit]Device[%u] don't need to deinit because it is not inited.", deviceId);
        return HCCL_SUCCESS;
    }
    CHK_RET(hrtRaPingDeinit(pingHandle_));
    HCCL_INFO("[HCCN][HccnRpingDeinit]Device[%u] deinit hccp success.", deviceId);

    // 关闭hccp进程
    CHK_RET(NetworkManager::GetInstance(static_cast<s32>(deviceId)).PingMeshRaPingDeinit());
    CHK_RET(HccnCloseSubProc(deviceId));
    isDeinited_ = true;
    rpingState_ = RpingState::UNINIT;
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnTargetAttrInter(
    u32 targetNumInter, RpingInput* inputInter, HccnRpingAddTargetConfig* configInter, PingTargetInfo* targetInter)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 addressType = 0;
    ret = GetAddrType(&addressType);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnTargetAttrInter]GetAddrType Fail ret %d", ret);
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < targetNumInter; i++) {
        PingInitInfo recvInfo;
        ret = RpingRecvTargetInfo(
            netCtx_, inputInter[i].port, inputInter[i].dip, recvInfo, configInter->connectTimeout);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HCCN][HccnRpingAddTarget]Target[%s] added failed because of error[%d].",
                inputInter[i].dip.GetReadableIP(), ret);
            break;
        }
        PingQpInfo* rdmaInfo = &(recvInfo.client);
        if (rdmaInfoMaps_.Find(std::string(inputInter[i].dip.GetReadableIP())).second) {
            HCCL_RUN_INFO("[HCCN][HccnRpingAddTarget]Target[%s] has already added.", inputInter[i].dip.GetReadableIP());
            continue;
        }
        rdmaInfoMaps_.Emplace(
            std::pair<std::string, PingQpInfo>(std::string(inputInter[i].dip.GetReadableIP()), recvInfo.client));
        if (payloadLenMap_.Find(std::string(inputInter[i].dip.GetReadableIP())).second) {
            HCCL_RUN_INFO("[HCCN][HccnRpingAddTarget]Target[%s] has already added.", inputInter[i].dip.GetReadableIP());
            continue;
        }
        payloadLenMap_.Emplace(
            std::pair<std::string, u32>(std::string(inputInter[i].dip.GetReadableIP()), inputInter[i].len));
        if (addressType == HCCN_RPING_ADDR_TYPE_IP) {
            ret = RpingTargetAttrInit(targetInter[0], inputInter[i], rdmaInfo, true);
        }
        const char* socNamePtr = aclrtGetSocName();
        CHK_PTR_NULL(socNamePtr);
        if (addressType == HCCN_RPING_ADDR_TYPE_EID && IsSupportHCCLV2(socNamePtr)) {
            ret = RpingTargetAttrInitWithUb(targetInter[0], inputInter[i], rdmaInfo, true);
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("[HCCN][HccnRpingAddTarget]Target[%s] payload added failed.", inputInter[i].dip.GetReadableIP());
            break;
        }
        ret = hrtRaPingTargetAdd(pingHandle_, targetInter, 1); // hccp侧只能一个一个处理，因此数组大小固定为1
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HCCN][HccnRpingAddTarget]Target[%s] added failed because of error[%d]",
                inputInter[i].dip.GetReadableIP(), ret);
            break;
        }
        HCCL_INFO("[HCCN][HccnRpingAddTarget]Target[%s] added success.", inputInter[i].dip.GetReadableIP());
        rpingTargetNum_++;
    }
    return ret;
}

HcclResult
PingMesh::HccnRpingAddTarget(u32 deviceId, u32 targetNum, RpingInput* input, HccnRpingAddTargetConfig* config)
{
    // 校验入参
    CHK_PRT_RET(config == nullptr, HCCL_ERROR("[PingMesh::HccnRpingAddTarget]config is null."), HCCL_E_PARA);
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::READY));
    // 调用hccp接口添加目标
    if (pingHandle_ == nullptr) {
        HCCL_ERROR("[HCCN][HccnRpingAddTarget]Device[%u] cannot add targets because it is not inited.", deviceId);
        return HCCL_E_NOT_FOUND;
    }
    HCCL_INFO("[HccnRpingAddTarget]deviceId %u, targetNum %u", deviceId, targetNum);
    HcclResult ret = HCCL_SUCCESS;
    PingTargetInfo target[1] = {{}}; // hccp侧只能一个一个处理，因此数组大小固定为1

    ret = HccnTargetAttrInter(targetNum, input, config, target);
    if ((ret == HCCL_SUCCESS) && (rpingState_ == RpingState::INITED)) { // 从初始化完成的状态切换到ready to start的状态
        rpingState_ = RpingState::READY;
    }

    return ret;
}

HcclResult PingMesh::HccnTarRemoveAttrInter(
    u32 targetNumInter, RpingInput* inputInter, PingTargetCommInfo* targetInter,
    std::shared_ptr<HcclSocket>& socketInter)
{
    HcclResult retInter = HCCL_SUCCESS;
    u32 addressType = 0;
    retInter = GetAddrType(&addressType);
    if (retInter != HCCL_SUCCESS) {
        HCCL_ERROR("[HccnTarRemoveAttrInter]GetAddrType Fail retInter %d", retInter);
        return HCCL_E_PARA;
    }
    for (u32 i = 0; i < targetNumInter; i++) {
        // 删除链路
        std::unique_lock<std::mutex> lock(socketMapsMtx_);
        if (socketMaps_.find(std::string(inputInter[i].dip.GetReadableIP())) == socketMaps_.end()) {
            HCCL_ERROR("[HCCN][HccnRpingRemoveTarget]Socket[%s] doesn't exist.", inputInter[i].dip.GetReadableIP());
            retInter = HCCL_E_NOT_FOUND;
            break;
        }
        socketInter = socketMaps_[std::string(inputInter[i].dip.GetReadableIP())];
        retInter = socketInter->DeInit();
        if (retInter != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[HCCN][HccnRpingRemoveTarget]Socket[%u][%s] deinit failed, ret[%d].", i,
                inputInter[i].dip.GetReadableIP(), retInter);
            break;
        }
        lock.unlock();
        if (!rdmaInfoMaps_.Find(std::string(inputInter[i].dip.GetReadableIP())).second) {
            HCCL_ERROR("[HCCN][HccnRpingRemoveTarget]Target[%s] doesn't exist.", inputInter[i].dip.GetReadableIP());
            retInter = HCCL_E_NOT_FOUND;
            break;
        }
        if (!payloadLenMap_.Find(std::string(inputInter[i].dip.GetReadableIP())).second) {
            HCCL_ERROR("[HCCN][HccnRpingRemoveTarget]Target[%s] doesn't exist.", inputInter[i].dip.GetReadableIP());
            retInter = HCCL_E_NOT_FOUND;
            break;
        }
        PingQpInfo* rdmainfo = &rdmaInfoMaps_[std::string(inputInter[i].dip.GetReadableIP())];
        PingTargetInfo targetInfo{};
        if (addressType == HCCN_RPING_ADDR_TYPE_IP) {
            retInter = RpingTargetAttrInit(targetInfo, inputInter[i], rdmainfo, false);
        }
        const char* socNamePtr = aclrtGetSocName();
        CHK_PTR_NULL(socNamePtr);
        if (addressType == HCCN_RPING_ADDR_TYPE_EID && IsSupportHCCLV2(socNamePtr)) {
            retInter = RpingTargetAttrInitWithUb(targetInfo, inputInter[i], rdmainfo, false);
        }

        targetInter[i] = targetInfo.remoteInfo;
    }
    return retInter;
}
HcclResult PingMesh::HccnRpingRemoveTarget(u32 deviceId, u32 targetNum, RpingInput* input)
{
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::READY));
    CHK_RET(RpingstateCheck(rpingState_, RpingState::INITED)); // 所有目标都被移除时回到READY前的状态
    if (pingHandle_ == nullptr) {
        HCCL_ERROR("[HCCN][HccnRpingRemoveTarget]Device[%u] cannot add targets because it is not inited.", deviceId);
        return HCCL_E_NOT_FOUND;
    }
    HCCL_INFO("[HccnRpingRemoveTarget]deviceId %u, targetNum %u", deviceId, targetNum);
    // 调用hccp接口删除目标
    HcclResult ret = HCCL_SUCCESS;
    PingTargetCommInfo* target = new (std::nothrow) PingTargetCommInfo[targetNum];
    CHK_PTR_NULL(target);
    std::shared_ptr<HcclSocket> socket = nullptr;
    ret = HccnTarRemoveAttrInter(targetNum, input, target, socket);
    if (ret != HCCL_SUCCESS) {
        delete[] target;
        HCCL_ERROR("[HCCN][HccnRpingRemoveTarget]Target info is not correct, ret[%d].", ret);
        return ret;
    }
    ret = hrtRaPingTargetDel(pingHandle_, target, targetNum);
    delete[] target;
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HCCN][HccnRpingRemoveTarget]Device[%u] remove targetNum %u failed, ret[%d].", deviceId, targetNum, ret);
        return ret;
    }
    rpingTargetNum_ = rpingTargetNum_ - targetNum;
    HCCL_INFO("[HCCN][HccnRpingRemoveTarget]Device[%u] remove targetNum %u success.", deviceId, targetNum);

    // 清除需要删掉的socket和rdma信息
    std::unique_lock<std::mutex> lock(socketMapsMtx_);
    RemoveMapInfo(input, targetNum, socketMaps_, rdmaInfoMaps_, payloadLenMap_);
    lock.unlock();
    if (rpingTargetNum_ <= 0) { // 目标数量小于等于0时, 记录的目标数量设为0,  切回初始化完成状态
        rpingTargetNum_ = 0;
        rpingState_ = RpingState::INITED;
    }

    return HCCL_SUCCESS;
}

HcclResult
PingMesh::HccnRpingGetTarget([[maybe_unused]] u32 deviceId, u32 targetNum, RpingInput* input, int* targetStat)
{
    CHK_PTR_NULL(input);
    CHK_PTR_NULL(targetStat);
    std::unique_lock<std::mutex> lock(socketMapsMtx_);
    for (u32 i = 0; i < targetNum; i++) {
        // 查询链路状态
        if (socketMaps_.find(std::string(input[i].dip.GetReadableIP())) == socketMaps_.end()) {
            HCCL_WARNING("[HCCN][HccnRpingGetTarget]Cannot get socket[%s]'s status.", input[i].dip.GetReadableIP());
            targetStat[i] = static_cast<int>(RpingLinkState::DISCONNECTED);
            continue;
        }
        HcclSocketStatus socketStatus = socketMaps_[std::string(input[i].dip.GetReadableIP())]->GetStatus();
        // 转换状态信息
        RpingLinkState linkStatus = ConvertHcclSocketStatus(socketStatus);
        // 记录查询结果
        targetStat[i] = static_cast<int>(linkStatus);
    }
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingBatchPingStart(u32 deviceId, u32 pktNum, u32 interval, u32 timeout)
{
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::RUN));
    HCCL_INFO(
        "[HCCN][HccnRpingBatchPingStart] deviceId[%u], pktNum[%u], interval[%u s], timeout[%u s].", deviceId, pktNum,
        interval, timeout);
    // 调用hccp接口发起ping请求
    if (pingHandle_ == nullptr) {
        HCCL_ERROR("[HCCN][HccnRpingBatchPingStart]Device[%u] cannot start ping because it is not inited.", deviceId);
        return HCCL_E_NOT_FOUND;
    }
    // 计算内存空间能否保存全部的payload信息，内存不足的话不可以发起ping请求
    PingBufferInfo* bufferInfo = &(initInfo_.result);
    u32 targetNum = rpingTargetNum_;
    u32 payloadLen = pktNum * PING_TOTAL_PAYLOAD_MAX_SIZE * targetNum;
    if ((bufferInfo->bufferSize != 0) && (payloadLen >= bufferInfo->bufferSize)) {
        HCCL_ERROR(
            "[HCCN][HccnRpingBatchPingStart]Buffer[%u] overflow threshold[%u], pktNum[%u], targetNum[%u].", payloadLen,
            bufferInfo->bufferSize, pktNum, targetNum);
        return HCCL_E_MEMORY;
    }
    PingTaskAttr attr = {};
    attr.packetCnt = pktNum;
    attr.packetInterval = interval;
    attr.timeoutInterval = timeout;
    CHK_RET(hrtRaPingTaskStart(pingHandle_, &attr));
    pktNum_ = pktNum;
    HCCL_INFO("[HCCN][HccnRpingBatchPingStart]pingmesh task is started on device[%u].", deviceId);
    rpingState_ = RpingState::RUN;
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingBatchPingStop(u32 deviceId)
{
    // 判断当前状态
    CHK_RET(RpingstateCheck(rpingState_, RpingState::STOP));
    HCCL_INFO("[HCCN][HccnRpingBatchPingStop]deviceId %u", deviceId);
    // 调用hccp接口中止ping请求
    CHK_RET(hrtRaPingTaskStop(pingHandle_));
    HCCL_INFO("[HCCN][HccnRpingBatchPingStop]Device[%u] pingmesh task is manually stopped.", deviceId);

    rpingState_ = RpingState::STOP;
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingGetResult(u32 deviceId, u32 targetNum, RpingInput* input, RpingOutput* output)
{
    CHK_PTR_NULL(input);
    CHK_PTR_NULL(output);
    PingTargetResult* resultInfo = new (std::nothrow) PingTargetResult[targetNum];
    CHK_PRT_RET(
        resultInfo == nullptr, HCCL_ERROR("[HCCN][HccnRpingGetResult]Alloc result memory failed."), HCCL_E_MEMORY);
    HCCL_INFO("[HCCN][HccnRpingGetResult]deviceId %u targetNum %u", deviceId, targetNum);
    HcclResult ret = RpingResultInfoInit(resultInfo, rdmaInfoMaps_, input, targetNum);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR(
            "[HCCN][HccnRpingGetResult]RpingResultInfoInit failed,Device[%u] ret[%d] num[%u].", deviceId, ret,
            targetNum);
        delete[] resultInfo;
        return ret;
    }
    // 调用hccp接口获取探测结果
    u32 num = targetNum; // resultinfo是一个带有返回值信息的数组, 对应的数组大小也需要返回，因此这里数组大小也传递指针
    ret = hrtRaPingGetResults(pingHandle_, resultInfo, &num);
    if (ret == HCCL_E_AGAIN) {
        HCCL_WARNING("[HCCN][HccnRpingGetResult]Try again.");
        delete[] resultInfo;
        return HCCL_E_AGAIN;
    }
    if (ret != HCCL_SUCCESS || num > targetNum) {
        HCCL_ERROR(
            "[HCCN][HccnRpingGetResult]Device[%u] get result failed, ret[%d] num[%u], targetNum[%u].", deviceId, ret,
            num, targetNum);
        delete[] resultInfo;
        return ret;
    }
    HCCL_INFO("[HCCN][HccnRpingGetResult]Device[%u] successfully gets [%u] results.", deviceId, num);

    GetResultFromReturnValue(resultInfo, output, num);

    delete[] resultInfo;
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingRefillPayloadHead(u8* originalHead, u32 payloadNum)
{
    for (u32 i = 0; i < payloadNum; i++) {
        RpingIpHead* ipHead = reinterpret_cast<RpingIpHead*>(originalHead);
        RpingIpHead ipHeadTmp;
        // 清零之前记录头信息
        errno_t memRet = memcpy_s(&ipHeadTmp, sizeof(RpingIpHead), ipHead, sizeof(RpingIpHead));
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillPayloadHead]copy head fail, ret %d, dst:%p, dstMax:%u, src:%p, length:%u",
                memRet, &ipHeadTmp, sizeof(RpingIpHead), ipHead, sizeof(RpingIpHead)),
            HCCL_E_MEMORY);

        // 重填payload头
        RpingPayloadHead* head = reinterpret_cast<RpingPayloadHead*>(originalHead);
        // 清零要重填的内存
        memRet = memset_s(originalHead, RPING_PAYLOAD_REFILL_LEN, 0, RPING_PAYLOAD_REFILL_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillPayloadHead]clear first 136B fail, ret %d, destMaxSize %u, count %u", memRet,
                RPING_PAYLOAD_REFILL_LEN, RPING_PAYLOAD_REFILL_LEN),
            HCCL_E_MEMORY);
        // 清零payload头rsvd字段的内存
        memRet = memset_s(head->reserved, RPING_PAYLOAD_RSVD_LEN, 0, RPING_PAYLOAD_RSVD_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillPayloadHead]clear last 44B fail, ret %d, destMaxSize %u, count %u", memRet,
                RPING_PAYLOAD_RSVD_LEN, RPING_PAYLOAD_RSVD_LEN),
            HCCL_E_MEMORY);
        // 填充ip
        HcclInAddr srcIpBinary;
        HcclInAddr dstIpBinary;
        // 报文来自对端，因此srcIp和dstIp需要调换过来
        if (ipAddr_.GetFamily() == AF_INET) {
            srcIpBinary.addr.s_addr = ipHeadTmp.ipv4.dstIp;
            dstIpBinary.addr.s_addr = ipHeadTmp.ipv4.srcIp;
        } else {
            HcclInAddr* srcIpBinary6 = reinterpret_cast<HcclInAddr*>(ipHeadTmp.ipv6.srcIp);
            HcclInAddr* dstIpBinary6 = reinterpret_cast<HcclInAddr*>(ipHeadTmp.ipv6.dstIp);
            srcIpBinary = *dstIpBinary6;
            dstIpBinary = *srcIpBinary6;
        }
        HcclIpAddress srcIp = HcclIpAddress(ipAddr_.GetFamily(), srcIpBinary);
        u32 ipAddrStrLen = std::string(srcIp.GetReadableIP()).size();
        memRet = memcpy_s(head->srcIp, IP_ADDRESS_BUFFER_LEN, srcIp.GetReadableIP(), ipAddrStrLen);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillPayloadHead]Memcpy ret %d, dst:%p, dstMax:%u, src:%p, length:%u", memRet,
                head->srcIp, IP_ADDRESS_BUFFER_LEN, srcIp.GetReadableIP(), ipAddrStrLen),
            HCCL_E_MEMORY);
        HcclIpAddress dstIp = HcclIpAddress(ipAddr_.GetFamily(), dstIpBinary);
        ipAddrStrLen = std::string(dstIp.GetReadableIP()).size();
        memRet = memcpy_s(head->dstIp, IP_ADDRESS_BUFFER_LEN, dstIp.GetReadableIP(), ipAddrStrLen);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillPayloadHead]Memcpy ret %d, dst:%p, dstMax:%u, src:%p, length:%u", memRet,
                head->dstIp, IP_ADDRESS_BUFFER_LEN, dstIp.GetReadableIP(), ipAddrStrLen),
            HCCL_E_MEMORY);
        // 填充payloadLen
        if (payloadLenMap_.Find(dstIp.GetReadableIP()).second) {
            head->payloadLen = payloadLenMap_[dstIp.GetReadableIP()];
        }
        // 填充addrtype
        head->addrType = HCCN_RPING_ADDR_TYPE_IP;
        originalHead += BYTE_PER_TARGET_DEFAULT;
    }

    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingRefillUbPayloadHead(u8* originalHead, u32 payloadNum)
{
    for (u32 i = 0; i < payloadNum; i++) {
        RpingEidHead* EidHead = reinterpret_cast<RpingEidHead*>(originalHead);
        RpingEidHead EidHeadTmp;
        // 清零之前记录头信息
        errno_t memRet = memcpy_s(&EidHeadTmp, sizeof(RpingEidHead), EidHead, sizeof(RpingEidHead));
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillUbPayloadHead]copy head fail, ret %d, dst:%p, dstMax:%u, src:%p, length:%u",
                memRet, &EidHeadTmp, sizeof(RpingEidHead), EidHead, sizeof(RpingEidHead)),
            HCCL_E_MEMORY);

        // 重填payload头
        RpingPayloadHead* head = reinterpret_cast<RpingPayloadHead*>(originalHead);
        // 清零要重填的内存
        memRet = memset_s(originalHead, RPING_PAYLOAD_UB_HEAD_LEN, 0, RPING_PAYLOAD_UB_HEAD_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillUbPayloadHead]clear first 256B fail, ret %d, destMaxSize %u, count %u", memRet,
                RPING_PAYLOAD_UB_HEAD_LEN, RPING_PAYLOAD_UB_HEAD_LEN),
            HCCL_E_MEMORY);

        // 填充Eid
        Eid srcEid;
        Eid dstEid;
        // 报文来自对端，因此srcEid和dstEid需要调换过来
        Eid* srcEid6 = reinterpret_cast<Eid*>(EidHeadTmp.srcEid);
        Eid* dstEid6 = reinterpret_cast<Eid*>(EidHeadTmp.dstEid);
        srcEid = *dstEid6;
        dstEid = *srcEid6;

        memRet = memcpy_s(head->srcEid, URMA_EID_LEN, srcEid.raw, URMA_EID_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillUbPayloadHead]Exchange eid fail. ret %d, dst:%p, dstMax:%u, src:%p, length:%u",
                memRet, head->srcEid, URMA_EID_LEN, srcEid.raw, URMA_EID_LEN),
            HCCL_E_MEMORY);

        memRet = memcpy_s(head->dstEid, URMA_EID_LEN, dstEid.raw, URMA_EID_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillUbPayloadHead]Exchange eid fail. ret %d, dst:%p, dstMax:%u, src:%p, length:%u",
                memRet, head->dstEid, URMA_EID_LEN, dstEid.raw, URMA_EID_LEN),
            HCCL_E_MEMORY);
        // 填充payloadLen
        HcclIpAddress dstEidAddress = HcclIpAddress(dstEid);
        if (payloadLenMap_.Find(dstEidAddress.Describe()).second) {
            head->payloadLen = payloadLenMap_[dstEidAddress.Describe()];
        }
        // 填充times
        memRet = memcpy_s(head->timestamp, RPING_PAYLOAD_UB_TIME_LEN, EidHeadTmp.times, RPING_PAYLOAD_UB_TIME_LEN);
        CHK_PRT_RET(
            memRet != EOK,
            HCCL_ERROR(
                "[HCCN][HccnRpingRefillUbPayloadHead]copy times fail. ret %d, dst:%p, dstMax:%u, src:%p, length:%u",
                memRet, head->timestamp, RPING_PAYLOAD_UB_TIME_LEN, EidHeadTmp.times, RPING_PAYLOAD_UB_TIME_LEN),
            HCCL_E_MEMORY);
        // 填充taskID
        head->rpingBatchId = EidHeadTmp.taskId;
        head->addrType = HCCN_RPING_ADDR_TYPE_EID;
        originalHead += BYTE_PER_TARGET_DEFAULT;
    }
    return HCCL_SUCCESS;
}

HcclResult PingMesh::HccnRpingGetPayload(u32 deviceId, void** payload, u32* payloadLen, HccnRpingMode mode)
{
    CHK_PTR_NULL(payload);
    CHK_PTR_NULL(payloadLen);
    // 判断是否为payload配置过内存
    if (!isUsePayload_) {
        HCCL_DEBUG("[HCCN][HccnRpingGetPayload]not alloc memory on device[%u] for payload.", deviceId);
        *payload = nullptr;
        *payloadLen = 0;
        return HCCL_SUCCESS;
    }
    // 将payload信息从device拷贝到host
    PingBufferInfo* bufferInfo = &(initInfo_.result);
    CHK_PRT_RET(
        bufferInfo->bufferSize == 0,
        HCCL_ERROR("[HCCN][HccnRpingGetPayload]no memory on device[%u] for payload.", deviceId), HCCL_E_MEMORY);
    // payload_为空时，需要为其申请内存资源
    if (payload_ == nullptr) {
        payload_ = new (std::nothrow) u8[bufferInfo->bufferSize];
        CHK_PRT_RET(
            payload_ == nullptr, HCCL_ERROR("[HCCN][HccnRpingGetPayload]Get payload from device[%u] failed.", deviceId),
            HCCL_E_MEMORY);
    }
    // 从device拷贝内存
    HcclResult ret = HCCL_SUCCESS;
    bool errorFlag = false;
    do {
        ret = hrtMemcpyEx(
            payload_, bufferInfo->bufferSize, reinterpret_cast<void*>(bufferInfo->bufferVa), bufferInfo->bufferSize,
            rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_HOST);
        CHK_PRT_BREAK(
            ret != HCCL_SUCCESS,
            HCCL_ERROR(
                "[HCCN][HccnRpingGetPayload]Get payload from device[%u] failed, bufferSize[%u], bufferVa[%llu].",
                deviceId, bufferInfo->bufferSize, bufferInfo->bufferVa),
            (errorFlag = true));
        // 重填payload头
        u32 payloadNum = bufferInfo->bufferSize / BYTE_PER_TARGET_DEFAULT;
        // 实际有效记录数为发包数×目标数，不应遍历整个buffer中未填充的槽位
        u32 actualNum = pktNum_ * static_cast<u32>(rpingTargetNum_);
        HCCL_INFO(
            "[HCCN][HccnRpingGetPayload]payloadNum[%u], actualNum[%u], pktNum[%u], targetNum[%d], bufferSize[%u].",
            payloadNum, actualNum, pktNum_, rpingTargetNum_, bufferInfo->bufferSize);
        if (actualNum > 0 && actualNum < payloadNum) {
            HCCL_INFO(
                "[HCCN][HccnRpingGetPayload]Adjust payloadNum from [%u] to [%u] to skip unfilled slots.", payloadNum,
                actualNum);
            payloadNum = actualNum;
        }
        u8* payloadTmp = payload_;
        if (mode == HCCN_RPING_MODE_ROCE) {
            ret = HccnRpingRefillPayloadHead(payloadTmp, payloadNum);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[HCCN][HccnRpingGetPayload]HccnRpingRefillPayloadHead failed."),
                errorFlag = true);
        }
        const char* socNamePtr = aclrtGetSocName();
        CHK_PRT_BREAK(
            socNamePtr == nullptr, HCCL_ERROR("[HCCN][HccnRpingGetPayload]aclrtGetSocName failed."), errorFlag = true);
        if (mode == HCCN_RPING_MODE_UB && IsSupportHCCLV2(socNamePtr)) {
            ret = HccnRpingRefillUbPayloadHead(payloadTmp, payloadNum);
            CHK_PRT_BREAK(
                ret != HCCL_SUCCESS, HCCL_ERROR("[HCCN][HccnRpingGetPayload]HccnRpingRefillUbPayloadHead failed."),
                errorFlag = true);
        }
        *payload = payload_;
        *payloadLen = bufferInfo->bufferSize;
    } while (0);

    if (errorFlag) {
        HCCL_ERROR(
            "[HCCN][HccnRpingGetPayload]Get payload from device[%u] failed, bufferSize[%u], bufferVa[%llu].", deviceId,
            bufferInfo->bufferSize, bufferInfo->bufferVa);
        delete[] payload_;
        payload_ = nullptr;
        return ret;
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
