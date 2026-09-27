/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_INC_ADAPTER_HCCP_COMMON_H
#define HCCL_INC_ADAPTER_HCCP_COMMON_H

#include "hccl_common.h"
#include "hccl_ip_address.h"

#if T_DESC("ip管理", true)

enum class DeviceIdType { DEVICE_ID_TYPE_PHY_ID = 0, DEVICE_ID_TYPE_SDID };

#define SOCK_CONN_TAG_SIZE 192
struct SocketWlistInfo {
    union hccl::HcclInAddr remoteIp; /**< IP address of remote */
    unsigned int connLimit;          /**< limit of white list */
    char tag[SOCK_CONN_TAG_SIZE];    /**< tag used for whitelist must ended by '\0' */
};

struct SocketEventInfo {
    u32 event;
    FdHandle fdHandle;
};

enum class HcclEpollEvent {
    HCCL_EPOLLIN = 0,
    HCCL_EPOLLOUT,
    HCCL_EPOLLPRI,
    HCCL_EPOLLERR,
    HCCL_EPOLLHUP,
    HCCL_EPOLLET,
    HCCL_EPOLLONESHOT,
    HCCL_EPOLLOUT_LET_ONESHOT,
    HCCL_EPOLLINVALD
};

enum class HcclSaveSnapShotAction {
    HCCL_SAVE_SNAPSHOT_ACTION_PRE_PROCESSING = 0,
    HCCL_SAVE_SNAPSHOT_ACTION_POST_PROCESSING = 1,
};

enum class HccnCfgKeyT {
    HCCN_UDP_PORT_MODE = 0,
    HCCN_MULTI_QP_COUNT = 1,
    HCCN_MULTI_QP_UDP_PORTS = 2,
    HCCN_RESV_MEM_INFO = 3
};

using QueueDepthAttr = struct QueueDepthAttrDef { // 有效配置 128 - 32K
    u32 sendCqDepth{INVALID_UINT};
    u32 recvCqDepth{INVALID_UINT};
    u32 sqDepth{INVALID_UINT};
    u32 rqDepth{INVALID_UINT};
};

HcclResult
hrtRaGetSingleSocketVnicIpInfo(u32 phyId, DeviceIdType deviceIdType, u32 deviceId, hccl::HcclIpAddress& vnicIP);
HcclResult hrtGetHostIf(
    std::vector<std::pair<std::string, hccl::HcclIpAddress>>& hostIfs, u32 devPhyId = 0); // key: if name, value ip addr
HcclResult hrtRaGetDeviceIP(u32 devicePhyId, std::vector<hccl::HcclIpAddress>& ipAddr);
HcclResult hrtRaGetDeviceAllNicIP(std::vector<std::vector<hccl::HcclIpAddress>>& ipAddr);
HcclResult GetIsSupSockBatchCloseImmed(u32 phyId, bool& isSupportBatchClose);

HcclResult hrtRaCreateEventHandle(s32& eventHandle);
HcclResult hrtRaCtlEventHandle(s32 eventHandle, const FdHandle fdHandle, int opCode, HcclEpollEvent event);

HcclResult hrtRaWaitEventHandle(
    s32 eventHandle, std::vector<SocketEventInfo>& eventInfos, s32 timeOut, u32 maxEvents, u32& eventsNum);
HcclResult H2DTlvInit(struct TlvInitInfo* init_info, uint32_t* buffer_size, void** tlv_handle);
HcclResult H2DTlvRequest(void* tlv_handle, unsigned int module_type, struct TlvMsg* send_msg, struct TlvMsg* recv_msg);
HcclResult H2DTlvDeinit(void* tlv_handle);
HcclResult hrtRaDestroyEventHandle(s32& eventHandle);

HcclResult SnapShotSaveAction(s32 networkMode, u32 devicePhyId, HcclSaveSnapShotAction action);
HcclResult SnapShotRestoreAction(s32 networkMode, u32 devicePhyId);
HcclResult HrtRaGetHccnCfg(s32 networkMode, u32 devicePhyId, enum HccnCfgKeyT key, std::string& value);
#endif

#endif
