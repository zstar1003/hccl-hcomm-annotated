/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include <sys/epoll.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>
#include "securec.h"
#include "hccp.h"
#include "ra_client_host.h"
#include "ra.h"
#include "ra_init.h"
#include "ra_rdma.h"
#include "ra_hdc.h"
#include "ra_hdc_rdma_notify.h"
#include "ra_hdc_rdma.h"
#include "ra_hdc_rdma_verbs.h"
#include "ra_hdc_socket.h"
#include "ra_peer.h"
#include "ra_peer_nda.h"
#include "ra_peer_socket.h"
#include "ra_rs_comm.h"
#include "ra_rs_err.h"

static unsigned int gWhiteListSwitch = 0;

/* socket: nic on device, need use hdc channel, support: 910, 310 */
struct RaSocketOps gRaHdcSocketOps = {
    .raSocketInit = RaHdcSocketInit,
    .raSocketDeinit = RaHdcSocketDeinit,
    .raSocketBatchConnect = RaHdcSocketBatchConnect,
    .raSocketBatchClose = RaHdcSocketBatchClose,
    .raSocketBatchAbort = RaHdcSocketBatchAbort,
    .raSocketListenStart = RaHdcSocketListenStart,
    .raSocketListenStop = RaHdcSocketListenStop,
    .raGetSockets = RaHdcGetSockets,
    .raSocketSend = RaHdcSocketSend,
    .raSocketRecv = RaHdcSocketRecv,
    .raGetClientSocketErrInfo = NULL,
    .raGetServerSocketErrInfo = NULL,
    .raSocketSetWhiteListStatus = NULL,
    .raSocketGetWhiteListStatus = NULL,
    .raSocketWhiteListAdd = RaHdcSocketWhiteListAdd,
    .raSocketWhiteListDel = RaHdcSocketWhiteListDel,
    .raSocketAcceptCreditAdd = RaHdcSocketAcceptCreditAdd,
};

/* socket: nic on host/device, support: cx6, 1822 */
struct RaSocketOps gRaPeerSocketOps = {
    .raSocketInit = NULL,
    .raSocketDeinit = RaPeerSocketDeinit,
    .raSocketBatchConnect = RaPeerSocketBatchConnect,
    .raSocketBatchClose = RaPeerSocketBatchClose,
    .raSocketBatchAbort = RaPeerSocketBatchAbort,
    .raSocketListenStart = RaPeerSocketListenStart,
    .raSocketListenStop = RaPeerSocketListenStop,
    .raGetSockets = RaPeerGetSockets,
    .raSocketSend = RaPeerSocketSend,
    .raSocketRecv = RaPeerSocketRecv,
    .raGetClientSocketErrInfo = RaPeerGetClientSocketErrInfo,
    .raGetServerSocketErrInfo = RaPeerGetServerSocketErrInfo,
    .raSocketSetWhiteListStatus = NULL,
    .raSocketGetWhiteListStatus = NULL,
    .raSocketWhiteListAdd = RaPeerSocketWhiteListAdd,
    .raSocketWhiteListDel = RaPeerSocketWhiteListDel,
    .raSocketAcceptCreditAdd = RaPeerSocketAcceptCreditAdd,
};

/* rdma: nic on device, need use hdc channel, support:910 */
struct RaRdmaOps gRaHdcRdmaOps = {
    .raRdevInit = RaHdcRdevInit,
    .raRdevGetPortStatus = RaHdcRdevGetPortStatus,
    .raGetLbMax = NULL,
    .raRdevDeinit = RaHdcRdevDeinit,
    .raSetTsqpDepth = RaHdcSetTsqpDepth,
    .raGetTsqpDepth = RaHdcGetTsqpDepth,
    .raQpCreate = RaHdcQpCreate,
    .raQpCreateWithAttrs = RaHdcQpCreateWithAttrs,
    .raAiQpCreate = RaHdcAiQpCreate,
    .raAiQpCreateWithAttrs = RaHdcAiQpCreateWithAttrs,
    .raTypicalQpCreate = RaHdcTypicalQpCreate,
    .raTypicalCqCreate = RaHdcTypicalCqCreate,
    .raTypicalCqDestroy = RaHdcTypicalCqDestroy,
    .raQpCreateWithCQWithAttrs = RaHdcQpCreateWithCQWithAttrs,
    .raLoopbackQpCreate = NULL,
    .raQpDestroy = RaHdcQpDestroy,
    .raQpDestroyWithoutCQ = RaHdcQpDestroyWithoutCQ,
    .raTypicalQpModify = RaHdcTypicalQpModify,
    .raQpBatchModify = RaHdcQpBatchModify,
    .raSetQpLbValue = NULL,
    .raGetQpLbValue = NULL,
    .raQpConnectAsync = RaHdcQpConnectAsync,
    .raGetQpStatus = RaHdcGetQpStatus,
    .raMrReg = RaHdcMrReg,
    .raMrDereg = RaHdcMrDereg,
    .raRegisterMr = RaHdcTypicalMrReg,
    .raRemapMr = RaHdcRemapMr,
    .raDeregisterMr = RaHdcTypicalMrDereg,
    .raSendWr = RaHdcSendWr,
    .raSendWrV2 = RaHdcSendWrV2,
    .raSendWrVerbs = RaHdcSendWrVerbs,
    .raRecvWrVerbs = RaHdcRecvWrVerbs,
    .raTypicalSendWr = RaHdcTypicalSendWr,
    .raSendWrlist = RaHdcSendWrlist,
    .raSendWrlistExt = RaHdcSendWrlistExt,
    .raSendNormalWrlist = RaHdcSendNormalWrlist,
    .raGetNotifyBaseAddr = RaHdcGetNotifyBaseAddr,
    .raGetNotifyMrInfo = RaHdcGetNotifyMrInfo,
    .raRecvWrlist = RaHdcRecvWrlist,
    .raPollCq = RaHdcPollCq,
    .raPollTypicalCq = RaHdcPollTypicalCq,
    .raGetQpContext = NULL,
    .raNormalQpCreate = NULL,
    .raNormalQpDestroy = NULL,
    .raCqCreate = NULL,
    .raCqDestroy = NULL,
    .raSetQpAttrQos = RaHdcSetQpAttrQos,
    .raSetQpAttrTimeout = RaHdcSetQpAttrTimeout,
    .raSetQpAttrRetryCnt = RaHdcSetQpAttrRetryCnt,
    .raGetQpHyperFeature = NULL,
    .raCreateCompChannel = NULL,
    .raDestroyCompChannel = NULL,
    .raCreateSrq = NULL,
    .raDestroySrq = NULL,
};

/* rdma: nic on host/device, support:cx6 1822 */
struct RaRdmaOps gRaPeerRdmaOps = {
    .raRdevInit = RaPeerRdevInit,
    .raRdevGetPortStatus = RaPeerRdevGetPortStatus,
    .raGetLbMax = RaPeerGetLbMax,
    .raRdevDeinit = RaPeerRdevDeinit,
    .raSetTsqpDepth = RaPeerSetTsqpDepth,
    .raGetTsqpDepth = RaPeerGetTsqpDepth,
    .raQpCreate = RaPeerQpCreate,
    .raQpCreateWithAttrs = RaPeerQpCreateWithAttrs,
    .raAiQpCreate = NULL,
    .raAiQpCreateWithAttrs = NULL,
    .raTypicalQpCreate = NULL,
    .raTypicalCqCreate = NULL,
    .raTypicalCqDestroy = NULL,
    .raQpCreateWithCQWithAttrs = NULL,
    .raLoopbackQpCreate = RaPeerLoopbackQpCreate,
    .raQpDestroy = RaPeerQpDestroy,
    .raQpDestroyWithoutCQ = NULL,
    .raTypicalQpModify = RaPeerTypicalQpModify,
    .raQpBatchModify = NULL,
    .raSetQpLbValue = RaPeerSetQpLbValue,
    .raGetQpLbValue = RaPeerGetQpLbValue,
    .raQpConnectAsync = RaPeerQpConnectAsync,
    .raGetQpStatus = RaPeerGetQpStatus,
    .raMrReg = RaPeerMrReg,
    .raMrDereg = RaPeerMrDereg,
    .raRegisterMr = RaPeerRegisterMr,
    .raRemapMr = NULL,
    .raDeregisterMr = RaPeerDeregisterMr,
    .raSendWr = RaPeerSendWr,
    .raSendWrV2 = NULL,
    .raSendWrVerbs = NULL,
    .raRecvWrVerbs = NULL,
    .raTypicalSendWr = NULL,
    .raSendWrlist = RaPeerSendWrlist,
    .raSendWrlistExt = NULL,
    .raSendNormalWrlist = NULL,
    .raGetNotifyBaseAddr = RaPeerGetNotifyBaseAddr,
    .raGetNotifyMrInfo = NULL,
    .raRecvWrlist = RaPeerRecvWrlist,
    .raPollCq = NULL,
    .raPollTypicalCq = NULL,
    .raGetQpContext = RaPeerGetQpContext,
    .raNormalQpCreate = RaPeerNormalQpCreate,
    .raNormalQpDestroy = RaPeerNormalQpDestroy,
    .raCqCreate = RaPeerCqCreate,
    .raCqDestroy = RaPeerCqDestroy,
    .raSetQpAttrQos = RaPeerSetQpAttrQos,
    .raSetQpAttrTimeout = RaPeerSetQpAttrTimeout,
    .raSetQpAttrRetryCnt = RaPeerSetQpAttrRetryCnt,
    .raGetQpHyperFeature = RaPeerGetQpHyperFeature,
    .raCreateCompChannel = RaPeerCreateCompChannel,
    .raDestroyCompChannel = RaPeerDestroyCompChannel,
    .raCreateSrq = RaPeerCreateSrq,
    .raDestroySrq = RaPeerDestroySrq,
};

struct ErrcodeInfo gErrcodeInfoList[] = {
    {-EPERM, 1, 0},
    {-EAGAIN, 1, 1},
    {-EACCES, 1, 2},
    {-EINVAL, 1, 3},
    {-ESYSFUNC, 1, 4},
    {-EADDRINUSE, 1, 5},
    {-EADDRNOTAVAIL, 1, 6},
    {-ESOCKCLOSED, 1, 7},
    {-EUSERS, TYPE_CODE_OR_ENV_ERR, 8},
    {-ENOENT, 2, 0},
    {-ESRCH, 2, 1},
    {-ENODEV, 2, 2},
    {-ENOSPC, 2, 3},
    {-EPROTONOSUPPORT, 2, 4},
    {-EFILEOPER, 2, 5},
    {-ENOMEM, 3, 0},
    {-EFAULT, 3, 1},
    {-EEXIST, 3, 2},
    {-EPIPE, 3, 3},
    {-ENOLINK, 3, 4},
    {-ENETUNREACH, 3, 5},
    {-ESAFEFUNC, 3, 6},
    {-EDEFAULT, 3, 7},
    {-EINVALIDIP, 3, 8},
    {-EOPENSRC, 5, 1},
    {-ENOTSUPP, 5, 2},
};

int RaInetPton(int family, union HccpIpAddr ip, char netAddr[], unsigned int len)
{
    const char *str = inet_ntop(family, &(ip.addr), netAddr, len);
    CHK_PRT_RETURN(str == NULL, hccp_err("[ntop_convert][ra_inet]the ip failed err(%d)", errno), -EINVAL);
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSocketInit(int mode, struct rdev rdevInfo, void **socketHandle)
{
    struct RaSocketHandle *socketHandleTmp = NULL;
    int ret;
    char localIp[MAX_IP_LEN] = {0};

    CHK_PRT_RETURN(rdevInfo.phyId >= RA_MAX_PHY_ID_NUM || socketHandle == NULL,
        hccp_err("[init][ra_socket]phyId(%u) is invalid! it must be [0,%d) or socket is null!", rdevInfo.phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    ret = RaInetPton(rdevInfo.family, rdevInfo.localIp, localIp, MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, ret));

    hccp_run_info("socket init:mode=%d phyId=%u family=%d ip=%s", mode, rdevInfo.phyId, rdevInfo.family, localIp);

    socketHandleTmp = calloc(1, sizeof(struct RaSocketHandle));
    CHK_PRT_RETURN(socketHandleTmp == NULL, hccp_err("[init][ra_socket]calloc socketHandleTmp failed, ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, -ENOMEM));

    if (mode == NETWORK_OFFLINE) {
        socketHandleTmp->socketOps = &gRaHdcSocketOps;
    } else if (mode == NETWORK_PEER_ONLINE) {
        socketHandleTmp->socketOps = &gRaPeerSocketOps;
    } else {
        hccp_err("[init][ra_socket]Wrong mode(%d), do not support", mode);
        ret = -EINVAL;
        goto err;
    }

    ret = memcpy_s(&(socketHandleTmp->rdevInfo), sizeof(struct rdev), &rdevInfo, sizeof(struct rdev));
    if (ret) {
        hccp_err("[init][ra_socket]memcpy_s for rdev_info failed, ret(%d)", ret);
        ret = -ESAFEFUNC;
        goto err;
    }

    if (rdevInfo.family == AF_INET && rdevInfo.localIp.addr.s_addr < RA_VNIC_MAX &&
        socketHandleTmp->socketOps->raSocketInit != NULL) {
        // HDC模式只支持IPv4
        if (rdevInfo.localIp.addr.s_addr < RA_VNIC_MAX) {
            ret = socketHandleTmp->socketOps->raSocketInit(rdevInfo);
            if (ret) {
                hccp_err("[init][ra_socket]ra socket init failed, ret(%d)", ret);
                goto err;
            }
        }
    }
    *socketHandle = (void *)socketHandleTmp;
    return ret;
err:
    free(socketHandleTmp);
    socketHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketInitV1(int mode, struct SocketInitInfoT socketInit, void **socketHandle)
{
    // 支持IPv4/IPv6 socket初始化
    // IPv6需要输入scope id
    struct RaSocketHandle *socketHandleTmp = NULL;
    char localIp[MAX_IP_LEN] = {0};
    int ret;

    CHK_PRT_RETURN(socketInit.rdevInfo.phyId >= RA_MAX_PHY_ID_NUM || socketHandle == NULL,
        hccp_err("[init][ra_socket]phyId(%u) is invalid! it must be [0,%d) or socket is null!",
            socketInit.rdevInfo.phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    ret = RaInetPton(socketInit.rdevInfo.family, socketInit.rdevInfo.localIp, localIp, MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, ret));

    hccp_run_info("socket init:mode=%d phyId=%u scope_id=%d family=%d ip=%s", mode, socketInit.rdevInfo.phyId,
        socketInit.scopeId, socketInit.rdevInfo.family, localIp);

    socketHandleTmp = calloc(1, sizeof(struct RaSocketHandle));
    CHK_PRT_RETURN(socketHandleTmp == NULL, hccp_err("[init][ra_socket]calloc for socket_handle failed"),
        ConverReturnCode(HCCP_INIT, -ENOMEM));

    if (mode == NETWORK_OFFLINE) {
        socketHandleTmp->socketOps = &gRaHdcSocketOps;
        socketHandleTmp->scopeId = socketInit.scopeId;
    } else if (mode == NETWORK_PEER_ONLINE) {
        socketHandleTmp->socketOps = &gRaPeerSocketOps;
        socketHandleTmp->scopeId = socketInit.scopeId;
    } else {
        hccp_err("[init][ra_socket]Wrong mode(%d), do not support", mode);
        ret = -EINVAL;
        goto err;
    }

    ret = memcpy_s(&(socketHandleTmp->rdevInfo), sizeof(struct rdev), &socketInit.rdevInfo, sizeof(struct rdev));
    if (ret) {
        hccp_err("[init][ra_socket]memcpy_s for rdev_info failed, ret(%d)", ret);
        ret = -ESAFEFUNC;
        goto err;
    }

    if (socketInit.rdevInfo.family == AF_INET && socketInit.rdevInfo.localIp.addr.s_addr < RA_VNIC_MAX &&
        socketHandleTmp->socketOps->raSocketInit != NULL) {
        // HDC模式只支持IPv4
        if (socketInit.rdevInfo.localIp.addr.s_addr < RA_VNIC_MAX) {
            ret = socketHandleTmp->socketOps->raSocketInit(socketInit.rdevInfo);
            if (ret) {
                hccp_err("[init][ra_socket]ra socket init v1 failed, ret(%d)", ret);
                goto err;
            }
        }
    }
    *socketHandle = (void *)socketHandleTmp;
    return ret;
err:
    free(socketHandleTmp);
    socketHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketDeinit(void *socketHandle)
{
    struct RaSocketHandle *socketHandleTmp = NULL;
    struct rdev rdevInfo;
    char localIp[MAX_IP_LEN] = {0};
    int ret;

    CHK_PRT_RETURN(socketHandle == NULL, hccp_err("[deinit][ra_socket]socket_handle is NULL"),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    socketHandleTmp = (struct RaSocketHandle *)socketHandle;
    rdevInfo = socketHandleTmp->rdevInfo;
    CHK_PRT_RETURN(rdevInfo.phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[deinit][ra_socket]phyId(%u) >= %u. invalid", rdevInfo.phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    ret = RaInetPton(rdevInfo.family, rdevInfo.localIp, localIp, MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, ret));

    hccp_run_info("Input parameters: phyId[%u] family[%d] local_ip[%s]", rdevInfo.phyId, rdevInfo.family, localIp);

    ret = socketHandleTmp->socketOps->raSocketDeinit(rdevInfo);
    if (ret) {
        hccp_err("[deinit][ra_socket]ra socket deinit failed, ret(%d)", ret);
    }

    socketHandleTmp->socketOps = NULL;
    free(socketHandleTmp);
    socketHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

STATIC int RaRdevInitCheckIp(int mode, struct rdev rdevInfo, char localIp[])
{
    struct InterfaceInfo *interfaceInfos = NULL;
    char interfaceIp[MAX_IP_LEN] = {0};
    struct RaGetIfattr config = {0};
    unsigned int i, interfaceVersion;
    unsigned int num = 0;
    int ret;

    config.phyId = rdevInfo.phyId;
    config.nicPosition = (unsigned int)mode;
    if (config.nicPosition == NETWORK_OFFLINE) {
        ret = RaGetInterfaceVersion(config.phyId, RA_RS_GET_IFNUM, &interfaceVersion);
        if (ret != 0 || interfaceVersion != RA_RS_GET_IFNUM_VERSION) {
            num = MAX_INTERFACE_NUM;
            goto get_addrs;
        }
    }
    /* get the number of interfaces */
    ret = RaGetIfnum(&config, &num);
    CHK_PRT_RETURN(ret != 0 || num == 0, hccp_err("[check][ip]get_ifnum failed, ret(%d) or num is 0", ret), -EINVAL);

get_addrs:
    /* calloc for interface_infos according to the real num */
    interfaceInfos = calloc(num, sizeof(struct InterfaceInfo));
    CHK_PRT_RETURN(interfaceInfos == NULL, hccp_err("[check][ip]calloc for interface_infos failed"), -EINVAL);

    ret = RaGetIfaddrs(&config, interfaceInfos, &num);
    if (ret != 0 || (num == 0)) {
        hccp_err("[check][ip]ra_get_ifaddrs for interface_infos failed, ret(%d), num(%u)", ret, num);
        free(interfaceInfos);
        return -EINVALIDIP;
    }

    for (i = 0; i < num; i++) {
        if (interfaceInfos[i].family != rdevInfo.family) {
            continue;
        }

        ret = RaInetPton(interfaceInfos[i].family, interfaceInfos[i].ifaddr.ip, interfaceIp, MAX_IP_LEN);
        if (ret != 0) {
            hccp_err("[check][ip]ra_inet_pton for interface_infos[%u] failed, ret(%d)", i, ret);
            free(interfaceInfos);
            return ret;
        }

        ret = strncmp(interfaceIp, localIp, MAX_IP_LEN - 1);
        if (ret == 0) {
            free(interfaceInfos);
            return 0;
        }
    }
    hccp_err("[check][ip]failed, ret(%d) the IP address(%s) in the ranktable is inconsistent with the IP(%s)"
             "address of the network adapter, please make sure they're consistent. "
             "num(%u)",
        ret, localIp, interfaceIp, num);

    free(interfaceInfos);
    return -EINVALIDIP;
}

int RaRdevInitCheck(int mode, struct rdev rdevInfo, char localIp[], unsigned int num, void *rdmaHandle)
{
    int ret;

    CHK_PRT_RETURN(rdevInfo.phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[check][ra_rdev_init]phyId(%u) is invalid! "
                 "it must be greater than or equal to 0 and less than %d!",
            rdevInfo.phyId, RA_MAX_PHY_ID_NUM),
        -EINVAL);

    CHK_PRT_RETURN(rdmaHandle == NULL, hccp_err("[check][ra_rdev_init]phyId(%u) rdma_handle is null!", rdevInfo.phyId),
        -EINVAL);

    ret = RaInetPton(rdevInfo.family, rdevInfo.localIp, localIp, num);
    CHK_PRT_RETURN(ret, hccp_err("[check][ra_rdev_init]ra_inet_pton for local_ip failed, ret(%d)", ret), -EINVAL);

    ret = RaRdevInitCheckIp(mode, rdevInfo, localIp);
    CHK_PRT_RETURN(ret, hccp_err("[check][ra_rdev_init]ra_rdev_init_check_ip failed, ret(%d)", ret), ret);

    return 0;
}

STATIC int RaGetInitRdmaHandle(int mode, struct RaRdmaHandle *rdmaHandle)
{
    if (mode == NETWORK_OFFLINE) {
        (void)RaHdcRdmaSetOps(rdmaHandle, &gRaHdcRdmaOps);
    } else if (mode == NETWORK_PEER_ONLINE) {
        (void)RaHdcRdmaSetOps(rdmaHandle, &gRaPeerRdmaOps);
    } else {
        hccp_err("[init][ra_rdev]Wrong mode(%d), do not support", mode);
        return -EINVAL;
    }

    CHK_PRT_RETURN(rdmaHandle->rdmaOps->raRdevInit == NULL, hccp_err("[init][ra_rdev] ra_rdev_init is NULL!"), -EINVAL);
    return 0;
}

STATIC void RaGenerateGidByRdevInfo(struct RaRdmaHandle *rdmaHandle)
{
#define RA_GID_SEQ_NUM 4
#define RA_GID_SEQ_ZERO 0
#define RA_GID_SEQ_ONE 1
#define RA_GID_SEQ_TWO 2
#define RA_GID_SEQ_THREE 3
    union HccpIpAddr localIp = rdmaHandle->rdevInfo.localIp;
    int family = rdmaHandle->rdevInfo.family;
    unsigned int gidV4[RA_GID_SEQ_NUM];

    if (family == AF_INET6) {
        (void)memcpy_s(rdmaHandle->gid, HCCP_GID_RAW_LEN, &(localIp.addr6), HCCP_GID_RAW_LEN);
    } else {
        gidV4[RA_GID_SEQ_ZERO] = 0;
        gidV4[RA_GID_SEQ_ONE] = 0;
        /* The gid format generated by ipv4 is filled with 0xFFFF in [33, 48] */
        gidV4[RA_GID_SEQ_TWO] = htonl(0x0000FFFF);
        gidV4[RA_GID_SEQ_THREE] = localIp.addr.s_addr;
        (void)memcpy_s(rdmaHandle->gid, HCCP_GID_RAW_LEN, gidV4, HCCP_GID_RAW_LEN);
    }

    return;
}

STATIC int RaRdevInitWithBackupInfo(struct RdevInitInfo initInfo, struct rdev rdevInfo, struct RaBackupInfo backupInfo,
    void **rdmaHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int rdevIndex;
    int ret;

    ret = RaRdevInitCheck(initInfo.mode, rdevInfo, localIp, MAX_IP_LEN, rdmaHandle);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_rdev]ra_rdev_init_check failed ,ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, ret));

    hccp_run_info("rdev_init:mode=%d phyId=%u family=%d ip=%s notify_type=%u", initInfo.mode, rdevInfo.phyId,
        rdevInfo.family, localIp, initInfo.notifyType);

    rdmaHandleTmp = calloc(1, sizeof(struct RaRdmaHandle));
    CHK_PRT_RETURN(rdmaHandleTmp == NULL, hccp_err("[init][ra_rdev]calloc for rdma_handle failed"),
        ConverReturnCode(HCCP_INIT, -ENOMEM));

    // disabled_lite_thread will be invalid if enabled_910a_lite is false
    rdmaHandleTmp->disabledLiteThread = initInfo.disabledLiteThread;
    rdmaHandleTmp->enabled910aLite = initInfo.enabled910aLite;
    rdmaHandleTmp->enabled2mbLite = initInfo.enabled2mbLite;
    (void)memcpy_s(&rdmaHandleTmp->backupInfo, sizeof(struct RaBackupInfo), &backupInfo, sizeof(struct RaBackupInfo));

    ret = RaGetInitRdmaHandle(initInfo.mode, rdmaHandleTmp);
    if (ret) {
        hccp_err("[init][ra_rdev] get rdma handle failed, ret(%d)", ret);
        goto err;
    }

    ret = memcpy_s(&(rdmaHandleTmp->rdevInfo), sizeof(struct rdev), &rdevInfo, sizeof(struct rdev));
    if (ret) {
        hccp_err("[init][ra_rdev]memcpy_s for rdev_info failed, ret(%d)", ret);
        ret = -ESAFEFUNC;
        goto err;
    }

    ret = rdmaHandleTmp->rdmaOps->raRdevInit(rdmaHandleTmp, initInfo.notifyType, rdevInfo, &rdevIndex);
    if (ret) {
        hccp_err("[init][ra_rdev]ra rdev init failed, ret(%d)", ret);
        goto err;
    }

    rdmaHandleTmp->rdevIndex = rdevIndex;
    RaGenerateGidByRdevInfo(rdmaHandleTmp);
    *rdmaHandle = (void *)rdmaHandleTmp;

    // save rdev handle for helper
    RaRdevSetHandle(rdevInfo.phyId, *rdmaHandle);

    return ret;
err:
    free(rdmaHandleTmp);
    rdmaHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaRdevInitWithBackup(struct RdevInitInfo *initInfo, struct rdev *rdevInfo,
    struct rdev *backupRdevInfo, void **rdmaHandle)
{
    struct RaBackupInfo backupInfo = {0};

    if (initInfo == NULL || rdevInfo == NULL || backupRdevInfo == NULL || rdmaHandle == NULL) {
        hccp_err("[init][ra_rdev]init_info or rdev_info or backup_rdev_info or rdma_handle is NULL");
        return -EINVAL;
    }

    backupInfo.backupFlag = true;
    (void)memcpy_s(&backupInfo.rdevInfo, sizeof(struct rdev), backupRdevInfo, sizeof(struct rdev));

    return RaRdevInitWithBackupInfo(*initInfo, *rdevInfo, backupInfo, rdmaHandle);
}

HCCP_ATTRI_VISI_DEF int RaRdevInitV2(struct RdevInitInfo initInfo, struct rdev rdevInfo, void **rdmaHandle)
{
    struct RaBackupInfo backupInfo = {0};

    return RaRdevInitWithBackupInfo(initInfo, rdevInfo, backupInfo, rdmaHandle);
}

HCCP_ATTRI_VISI_DEF int RaRdevInit(int mode, unsigned int notifyType, struct rdev rdevInfo, void **rdmaHandle)
{
    struct RdevInitInfo initInfo = {0};

    initInfo.mode = mode;
    initInfo.notifyType = notifyType;
    initInfo.disabledLiteThread = false; // will start lite thread by default
    initInfo.enabled910aLite = false;    // will disabled lite on 910A by default
    initInfo.enabled2mbLite = false;     // will disabled lite on 2MB page align scenario by default

    return RaRdevInitV2(initInfo, rdevInfo, rdmaHandle);
}

STATIC int RaRdevDeinitParaCheck(void *rdmaHandle, struct RaRdmaHandle **rdmaHandleTmp)
{
    int ret;
    char localIp[MAX_IP_LEN] = {0};
    struct rdev rdevInfo;

    *rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    rdevInfo = (*rdmaHandleTmp)->rdevInfo;
    CHK_PRT_RETURN(rdevInfo.phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[deinit][ra_rdev]phyId(%u) "
                 "must be smaller than %u",
            rdevInfo.phyId, RA_MAX_PHY_ID_NUM),
        -EINVAL);

    ret = RaInetPton(rdevInfo.family, rdevInfo.localIp, localIp, MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_rdev]ra_inet_pton for local_ip failed, ret(%d)", ret), ret);

    CHK_PRT_RETURN((*rdmaHandleTmp)->rdmaOps == NULL || (*rdmaHandleTmp)->rdmaOps->raRdevDeinit == NULL,
        hccp_err("[deinit][ra_rdev]rdma_ops is NULL or ra_rdev_deinit is NULL"), -EINVAL);

    hccp_run_info("Input parameters: phyId[%u] rdev_index[%u] family[%d] local_ip[%s]", rdevInfo.phyId,
        (*rdmaHandleTmp)->rdevIndex, rdevInfo.family, localIp);

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaRdevGetPortStatus(void *rdmaHandle, enum PortStatus *status)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdmaHandle == NULL || status == NULL,
        hccp_err("[get][ra_port_status]rdma_handle or status is NULL, invalid"), ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_port_status]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raRdevGetPortStatus == NULL,
        hccp_err("[get][ra_port_status]rdma_ops is NULL or ra_rdev_get_port_status is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdmaHandleTmp->rdmaOps->raRdevGetPortStatus(rdmaHandleTmp, status);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaRdevDeinit(void *rdmaHandle, unsigned int notifyType)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdmaHandle == NULL, hccp_err("[deinit][ra_rdev] rdma_handle is NULL"),
        ConverReturnCode(HCCP_INIT, -EINVAL));

    ret = RaRdevDeinitParaCheck(rdmaHandle, &rdmaHandleTmp);
    CHK_PRT_RETURN(ret, hccp_err("[deinit][ra_rdev] para check failed, ret(%d)", ret),
        ConverReturnCode(HCCP_INIT, ret));

    phyId = rdmaHandleTmp->rdevInfo.phyId;

    ret = rdmaHandleTmp->rdmaOps->raRdevDeinit(rdmaHandleTmp, notifyType);
    if (ret) {
        hccp_err("[deinit][ra_rdev]ra rdev deinit failed, ret(%d)", ret);
        goto free_rdma_handle;
    }

free_rdma_handle:
    rdmaHandleTmp->rdmaOps = NULL;
    free(rdmaHandleTmp);
    rdmaHandleTmp = NULL;
    RaRdevSetHandle(phyId, NULL);
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaRdevGetSupportLite(void *rdmaHandle, int *supportLite)
{
    struct RaRdmaHandle *rdmaHandleTmp = NULL;

    CHK_PRT_RETURN(rdmaHandle == NULL || supportLite == NULL,
        hccp_err("[get][ra_rdev]rdma_handle is NULL or support_lite is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    rdmaHandleTmp = (struct RaRdmaHandle *)rdmaHandle;
    *supportLite = rdmaHandleTmp->supportLite;

    hccp_dbg("[get][ra_rdev]support_lite:%d", *supportLite);

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSocketBatchConnect(struct SocketConnectInfoT conn[], unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char remoteIp[MAX_IP_LEN] = {0};
    char localIp[MAX_IP_LEN] = {0};
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[batch_connect][ra_socket]conn is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        if (socketHandle == NULL || socketHandle->socketOps == NULL ||
            socketHandle->socketOps->raSocketBatchConnect == NULL) {
            hccp_err("[batch_connect][ra_socket]socket_handle or func is NULL");
            return ConverReturnCode(SOCKET_OP, -EINVAL);
        }

        CHK_PRT_RETURN(socketHandle->rdevInfo.phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[batch_connect][ra_socket]phyId(%u) must be smaller than %u", socketHandle->rdevInfo.phyId,
                RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        CHK_PRT_RETURN(strlen(conn[i].tag) >= SOCK_CONN_TAG_SIZE,
            hccp_err("[batch_connect][ra_socket]conn tag len(%d) more than max len(%d)", strlen(conn[i].tag),
                SOCK_CONN_TAG_SIZE),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[batch_connect][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        ret = RaInetPton(socketHandle->rdevInfo.family, conn[i].remoteIp, remoteIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[batch_connect][ra_socket]ra_inet_pton for remote_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], remoteIp[%s], port[%u], tag[%s], cnt[%u]", i,
            socketHandle->rdevInfo.phyId, localIp, remoteIp, conn[i].port, conn[i].tag, socketHandle->connectCnt);
    }

    socketHandle->connectCnt++;
    ret = socketHandle->socketOps->raSocketBatchConnect(socketHandle->rdevInfo.phyId, conn, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketBatchClose(struct SocketCloseInfoT conn[], unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[batch_close][ra_socket]conn is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        if (socketHandle == NULL || socketHandle->socketOps == NULL ||
            socketHandle->socketOps->raSocketBatchClose == NULL) {
            hccp_warn_socket("[batch_close][ra_socket]socket_handle or func is NULL, no need to close");
            return 0;
        }
        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[batch_close][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[batch_connect][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], cnt[%u]", i, phyId, localIp,
            socketHandle->closeCnt);
    }

    socketHandle->closeCnt++;
    ret = socketHandle->socketOps->raSocketBatchClose(phyId, conn, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketBatchAbort(struct SocketConnectInfoT conn[], unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char remoteIp[MAX_IP_LEN] = {0};
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[batch_abort][ra_socket]conn is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        CHK_PRT_RETURN(socketHandle == NULL || socketHandle->socketOps == NULL ||
                           socketHandle->socketOps->raSocketBatchAbort == NULL,
            hccp_err("[batch_abort][ra_socket]socket_handle or func is NULL"), ConverReturnCode(SOCKET_OP, -EINVAL));

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[batch_abort][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[batch_abort][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        ret = RaInetPton(socketHandle->rdevInfo.family, conn[i].remoteIp, remoteIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[batch_abort][ra_socket]ra_inet_pton for remote_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], remoteIp[%s], tag[%s], cnt[%u]", i, phyId,
            localIp, remoteIp, conn[i].tag, socketHandle->abortCnt);
    }

    socketHandle->abortCnt++;
    ret = socketHandle->socketOps->raSocketBatchAbort(phyId, conn, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketListenStart(struct SocketListenInfoT conn[], unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[listen_start][ra_socket]conn is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        if (socketHandle == NULL || socketHandle->socketOps == NULL ||
            socketHandle->socketOps->raSocketListenStart == NULL) {
            hccp_err("[listen_start][ra_socket]socket_handle or func is NULL");
            return ConverReturnCode(SOCKET_OP, -EINVAL);
        }
        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[listen_start][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[listen_start][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s], port[%u]", i, phyId, localIp, conn[i].port);
    }

    ret = socketHandle->socketOps->raSocketListenStart(phyId, conn, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketListenStop(struct SocketListenInfoT conn[], unsigned int num)
{
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId = 0;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(conn == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[listen_stop][ra_socket]conn is NULL or num[%u] is zero or num is greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        if (socketHandle == NULL || socketHandle->socketOps == NULL ||
            socketHandle->socketOps->raSocketListenStop == NULL) {
            hccp_err("[listen_stop][ra_socket]socket_handle or func is NULL");
            return ConverReturnCode(SOCKET_OP, -EINVAL);
        }

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[listen_stop][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[listen_stop][ra_socket]ra_inet_pton for server_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        hccp_run_info("Input parameters: [%u]th, phyId[%u], localIp[%s]", i, phyId, localIp);
    }

    ret = socketHandle->socketOps->raSocketListenStop(phyId, conn, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetSockets(unsigned int role, struct SocketInfoT conn[], unsigned int num,
    unsigned int *connectedNum)
{
    unsigned int i;
    struct RaSocketHandle *socketHandle = NULL;
    char localIp[MAX_IP_LEN] = {0};
    char remoteIp[MAX_IP_LEN] = {0};
    int ret;
    unsigned int phyId = 0;

    CHK_PRT_RETURN(conn == NULL || connectedNum == NULL || num == 0 || num > MAX_SOCKET_NUM,
        hccp_err("[get][ra_socket]conn or connected_num is NULL or num[%u] is zero or num greater than %d", num,
            MAX_SOCKET_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    for (i = 0; i < num; i++) {
        socketHandle = (struct RaSocketHandle *)conn[i].socketHandle;
        if (socketHandle == NULL || socketHandle->socketOps == NULL || socketHandle->socketOps->raGetSockets == NULL) {
            hccp_err("[get][ra_socket]socket_handle or func is NULL");
            return ConverReturnCode(SOCKET_OP, -EINVAL);
        }

        phyId = socketHandle->rdevInfo.phyId;
        CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
            hccp_err("[get][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
            ConverReturnCode(SOCKET_OP, -EINVAL));

        ret = RaInetPton(socketHandle->rdevInfo.family, socketHandle->rdevInfo.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket]ra_inet_pton for local_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));

        ret = RaInetPton(socketHandle->rdevInfo.family, conn[i].remoteIp, remoteIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_socket]ra_inet_pton for remote_ip failed, ret(%d)", ret),
            ConverReturnCode(SOCKET_OP, ret));
    }

    ret = socketHandle->socketOps->raGetSockets(phyId, role, conn, num);
    if (ret >= 0) {
        *connectedNum = (unsigned int)ret;
        return 0;
    }

    *connectedNum = 0;
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketRecv(const void *fdHandle, void *data, unsigned long long size,
    unsigned long long *receivedSize)
{
    int ret;
    unsigned int phyId;
    const struct RaSocketHandle *socketHandleTmp = NULL;
    const struct SocketHdcInfo *fdHandleTmp = (const struct SocketHdcInfo *)fdHandle;

    CHK_PRT_RETURN(fdHandle == NULL || data == NULL || size == 0 || receivedSize == NULL,
        hccp_err("[recv][ra_socket]fd_handle or data or received_size is NULL or size[%llu] is 0", size),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    socketHandleTmp = (const struct RaSocketHandle *)(fdHandleTmp->socketHandle);
    if (socketHandleTmp == NULL || socketHandleTmp->socketOps == NULL ||
        socketHandleTmp->socketOps->raSocketRecv == NULL) {
        hccp_err("[recv][ra_socket]socket_handle_tmp or func is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    phyId = socketHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[recv][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    ret = socketHandleTmp->socketOps->raSocketRecv(phyId, fdHandle, data, size);
    if (ret > 0) {
        *receivedSize = (unsigned long long)(unsigned int)ret;
        return 0;
    } else if (ret == 0) {
        hccp_warn_socket("[recv][ra_socket]socket has been closed. received_size is 0");
        ret = -ESOCKCLOSED;
    }

    *receivedSize = 0;
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketSend(const void *fdHandle, const void *data, unsigned long long size,
    unsigned long long *sentSize)
{
    int ret;
    unsigned int phyId;
    const struct RaSocketHandle *socketHandleTmp = NULL;
    const struct SocketHdcInfo *fdHandleTmp = (const struct SocketHdcInfo *)fdHandle;

    CHK_PRT_RETURN(fdHandle == NULL || data == NULL || sentSize == NULL || size == 0,
        hccp_err("[send][ra_socket]fd_handle or data or sent_size is NULL or size[%llu] is 0", size),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    socketHandleTmp = (const struct RaSocketHandle *)(fdHandleTmp->socketHandle);
    if (socketHandleTmp == NULL || socketHandleTmp->socketOps == NULL ||
        socketHandleTmp->socketOps->raSocketSend == NULL) {
        hccp_err("[send][ra_socket]socket_handle_tmp or func is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    phyId = socketHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[send][ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    ret = socketHandleTmp->socketOps->raSocketSend(phyId, fdHandle, data, size);
    if (ret > 0) {
        *sentSize = (unsigned long long)(unsigned int)ret;
        return 0;
    } else if (ret == 0) {
        hccp_warn_socket("[send][ra_socket]socket has been closed. sent_size is 0");
        ret = -ESOCKCLOSED;
    }

    *sentSize = 0;
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaEpollCtlAdd(const void *fdHandle, enum RaEpollEvent event)
{
    CHK_PRT_RETURN(fdHandle == NULL, hccp_err("[ra_epoll_ctl_add]fd_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(event != RA_EPOLLIN && event != RA_EPOLLONESHOT,
        hccp_err("[ra_epoll_ctl_add]wrong event, only RA_EPOLLIN and RA_EPOLLONESHOT are supported."),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    int ret = RaPeerEpollCtlAdd(fdHandle, event);
    CHK_PRT_RETURN(ret, hccp_err("[ra_epoll_ctl_add]ra_peer_epoll_ctl_add failed ret(%d)", ret),
        ConverReturnCode(SOCKET_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaEpollCtlMod(const void *fdHandle, enum RaEpollEvent event)
{
    CHK_PRT_RETURN(fdHandle == NULL, hccp_err("[ra_epoll_ctl_mod]fd_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    CHK_PRT_RETURN(event != RA_EPOLLIN && event != RA_EPOLLONESHOT,
        hccp_err("[ra_epoll_ctl_mod]wrong event, only RA_EPOLLIN and RA_EPOLLONESHOT are supported."),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    int ret = RaPeerEpollCtlMod(fdHandle, event);
    CHK_PRT_RETURN(ret, hccp_err("[ra_epoll_ctl_mod]ra_peer_epoll_ctl_mod failed ret(%d)", ret),
        ConverReturnCode(SOCKET_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaEpollCtlDel(const void *fdHandle)
{
    CHK_PRT_RETURN(fdHandle == NULL, hccp_err("[ra_epoll_ctl_del]fd_handle is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    int ret = RaPeerEpollCtlDel(fdHandle);
    CHK_PRT_RETURN(ret, hccp_err("[ra_epoll_ctl_del]ra_peer_epoll_ctl_del failed ret(%d)", ret),
        ConverReturnCode(SOCKET_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSetTcpRecvCallback(const void *socketHandle, const void *callback)
{
    const struct RaSocketHandle *socketHandleTmp = (const struct RaSocketHandle *)socketHandle;
    unsigned int phyId;

    CHK_PRT_RETURN(socketHandleTmp == NULL || callback == NULL,
        hccp_err("[ra_socket]socket_handle is NULL or callback is NULL"), ConverReturnCode(SOCKET_OP, -EINVAL));

    phyId = socketHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[ra_socket]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    RaPeerSetTcpRecvCallback(phyId, callback);

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaGetTsqpDepth(void *rdevHandle, unsigned int *tempDepth, unsigned int *qpNum)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raGetTsqpDepth == NULL,
        hccp_err("[get][ra_tsqp_depth]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(tempDepth == NULL || qpNum == NULL, hccp_err("[get][ra_tsqp_depth]temp_depth or qp_num is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_tsqp_depth]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!",
            phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_info_rma("Input parameters: phyId[%u], rdevIndex[%u]", phyId, rdmaHandleTmp->rdevIndex);

    ret = rdmaHandleTmp->rdmaOps->raGetTsqpDepth(rdmaHandleTmp, tempDepth, qpNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetTsqpDepth(void *rdevHandle, unsigned int tempDepth, unsigned int *qpNum)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raSetTsqpDepth == NULL,
        hccp_err("[set][ra_tsqp_depth]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpNum == NULL, hccp_err("[set][ra_tsqp_depth]qp_num is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(tempDepth < RA_MIN_TEMPTH_DEPTH || tempDepth > RA_MAX_TEMPTH_DEPTH,
        hccp_err("[set][ra_tsqp_depth]param error! temp_depth(%u) cannot be smaller than %d or bigger than %d",
            tempDepth, RA_MIN_TEMPTH_DEPTH, RA_MAX_TEMPTH_DEPTH),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[set][ra_tsqp_depth]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!",
            phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], rdevIndex[%u], tempDepth[%u]", phyId, rdmaHandleTmp->rdevIndex,
        tempDepth);

    ret = rdmaHandleTmp->rdmaOps->raSetTsqpDepth(rdmaHandleTmp, tempDepth, qpNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaQpCreate(void *rdevHandle, int flag, int qpMode, void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raQpCreate == NULL,
        hccp_err("[create][ra_qp]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[create][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    /* 0 means RC mode */
    CHK_PRT_RETURN(flag != 0, hccp_err("[create][ra_qp]The flag(%d) is invalid, expect 0", flag),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_qp]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpMode < 0 || qpMode >= RA_RS_ERR_QP_MODE,
        hccp_err("[create][ra_qp]QP mode(%d) must be greater than or equal to 0 and less than %d", qpMode,
            RA_RS_ERR_QP_MODE),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], flag[%d] qpMode [%d]", phyId, flag, qpMode);

    ret = rdmaHandleTmp->rdmaOps->raQpCreate(rdmaHandleTmp, flag, qpMode, qpHandle);
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_qp]create qp failed, ret(%d) phyId(%u)", ret, phyId), ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaQpCreateWithAttrs(void *rdevHandle, struct QpExtAttrs *extAttrs, void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raQpCreateWithAttrs == NULL,
        hccp_err("[create][ra_qp_with_attrs]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[create][ra_qp_with_attrs]qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs == NULL, hccp_err("[create][ra_qp_with_attrs]ext_attrs is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs->version != QP_CREATE_WITH_ATTR_VERSION,
        hccp_err("[create][ra_qp_with_attrs]attr version[%d] mismatch, expect [%d]", extAttrs->version,
            QP_CREATE_WITH_ATTR_VERSION),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(extAttrs->qpMode < 0 || extAttrs->qpMode >= RA_RS_ERR_QP_MODE,
        hccp_err("[create][ra_qp_with_attrs]QP mode[%d] must be greater than or equal to 0 and less than %d",
            extAttrs->qpMode, RA_RS_ERR_QP_MODE),
        ConverReturnCode(RDMA_OP, -EINVAL));
    // no need and disallow to set data_plane_flag, set it to default value 0
    extAttrs->dataPlaneFlag.value = 0;

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_qp_with_attrs]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u] qp_mode[%d] cq_attr{%d,%d,%d,%d} qpAttr.cap{%u,%u,%u,%u,%u}"
                  " qp_type[%u] sqSigAll[%d], cnt[%u], useResvMem[%u], resvMemPoolId[%u]",
        phyId, extAttrs->qpMode, extAttrs->cqAttr.sendCqDepth, extAttrs->cqAttr.sendCqCompVector,
        extAttrs->cqAttr.recvCqDepth, extAttrs->cqAttr.recvCqCompVector, extAttrs->qpAttr.cap.max_send_wr,
        extAttrs->qpAttr.cap.max_recv_wr, extAttrs->qpAttr.cap.max_send_sge, extAttrs->qpAttr.cap.max_recv_sge,
        extAttrs->qpAttr.cap.max_inline_data, extAttrs->qpAttr.qp_type, extAttrs->qpAttr.sq_sig_all,
        rdmaHandleTmp->qpCnt, extAttrs->cstmFlag.bs.useResvMem, extAttrs->resvMemPoolId);

    rdmaHandleTmp->qpCnt++;
    ret = rdmaHandleTmp->rdmaOps->raQpCreateWithAttrs(rdmaHandleTmp, extAttrs, qpHandle);
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_qp_with_attrs]create qp failed, ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaAiQpCreate(void *rdevHandle, struct QpExtAttrs *attrs, struct AiQpInfo *info, void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int interfaceVersion = 0;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raAiQpCreate == NULL ||
                       rdmaHandleTmp->rdmaOps->raAiQpCreateWithAttrs == NULL,
        hccp_err("[create][ra_ai_qp]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(info == NULL || qpHandle == NULL, hccp_err("[create][ra_ai_qp]info is NULL or qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(attrs == NULL, hccp_err("[create][ra_ai_qp]attrs is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(attrs->version != QP_CREATE_WITH_ATTR_VERSION,
        hccp_err("[create][ra_ai_qp]attr version[%d] mismatch, expect [%d]", attrs->version,
            QP_CREATE_WITH_ATTR_VERSION),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(attrs->qpMode < 0 || attrs->qpMode >= RA_RS_ERR_QP_MODE,
        hccp_err("[create][ra_ai_qp]QP mode[%d] must be greater than or equal to 0 and less than %d", attrs->qpMode,
            RA_RS_ERR_QP_MODE),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_ai_qp]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u] qp_mode[%d] cq_attr{%d,%d,%d,%d} cqCstm[%d] "
                  "qpAttr.cap{%u,%u,%u,%u,%u} qp_type[%u] sqSigAll[%d]",
        phyId, attrs->qpMode, attrs->cqAttr.sendCqDepth, attrs->cqAttr.sendCqCompVector, attrs->cqAttr.recvCqDepth,
        attrs->cqAttr.recvCqCompVector, attrs->dataPlaneFlag.bs.cqCstm, attrs->qpAttr.cap.max_send_wr,
        attrs->qpAttr.cap.max_recv_wr, attrs->qpAttr.cap.max_send_sge, attrs->qpAttr.cap.max_recv_sge,
        attrs->qpAttr.cap.max_inline_data, attrs->qpAttr.qp_type, attrs->qpAttr.sq_sig_all);

    ret = RaGetInterfaceVersion(phyId, RA_RS_AI_QP_CREATE_WITH_ATTRS, &interfaceVersion);
    if (ret == 0 && interfaceVersion >= RA_RS_OPCODE_BASE_VERSION) {
        ret = rdmaHandleTmp->rdmaOps->raAiQpCreateWithAttrs(rdmaHandleTmp, attrs, info, qpHandle);
    } else {
        // origin procedure: not support to process data_plane_flag.bs.cq_cstm
        ret = rdmaHandleTmp->rdmaOps->raAiQpCreate(rdmaHandleTmp, attrs, info, qpHandle);
    }
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_ai_qp]create qp failed, ret(%d) phyId(%u)", ret, phyId), ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaTypicalQpCreate(void *rdevHandle, int flag, int qpMode, struct TypicalQp *qpInfo,
    void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raTypicalQpCreate == NULL,
        hccp_err("[create][ra_typical_qp]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[create][ra_typical_qp]qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpInfo == NULL, hccp_err("[create][ra_typical_qp]qp_info is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(flag != 0, hccp_err("[create][ra_typical_qp]The flag(%d) is invalid, expect 0", flag),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_typical_qp]phyId(%u) must greater or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpMode < 0 || qpMode >= RA_RS_ERR_QP_MODE,
        hccp_err("[create][ra_typical_qp]QP mode(%d) must be greater than or equal to 0 and less than %d", qpMode,
            RA_RS_ERR_QP_MODE),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], flag[%d] qpMode[%d]", phyId, flag, qpMode);

    ret = rdmaHandleTmp->rdmaOps->raTypicalQpCreate(rdmaHandleTmp, flag, qpMode, qpInfo, qpHandle);
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_typical_qp]create qp failed, ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaLoopbackQpCreate(void *rdevHandle, struct LoopbackQpPair *qpPair, void **qpHandle)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdmaHandleTmp == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raLoopbackQpCreate == NULL,
        hccp_err("[create][ra_loopback_qp]rdev_handle is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpPair == NULL, hccp_err("[create][ra_loopback_qp]qp_pair is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[create][ra_loopback_qp]qp_handle is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_loopback_qp]phyId(%u) must greater or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u]", phyId);
    ret = rdmaHandleTmp->rdmaOps->raLoopbackQpCreate(rdmaHandleTmp, qpPair, qpHandle);
    CHK_PRT_RETURN(ret != 0 || *qpHandle == NULL,
        hccp_err("[create][ra_loopback_qp]create qp failed, ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, ret));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaQpDestroy(void *qpHandle)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL, hccp_err("[destroy][ra_qp]qp_handle is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raQpDestroy == NULL,
        hccp_err("[destroy][ra_qp]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_qp_destroy is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: qpn[%u], phyId[%u], rdevIndex[%u] qpMode[%d] flag[%d]", raQpHandle->qpn,
        raQpHandle->phyId, raQpHandle->rdevIndex, raQpHandle->qpMode, raQpHandle->flag);

    ret = raQpHandle->rdmaOps->raQpDestroy(raQpHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaQpConnectAsync(void *qpHandle, const void *fdHandle)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || fdHandle == NULL,
        hccp_err("[connect_async][ra_qp]ra_qp_handle or fd_handle is NULL, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raQpConnectAsync == NULL,
        hccp_err("[connect_async][ra_qp]rdma_ops or ra_qp_handle->rdma_ops->ra_qp_connect_async is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raQpConnectAsync(raQpHandle, fdHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetQpStatus(void *qpHandle, int *status)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || status == NULL,
        hccp_err("[get][ra_qp_status]ra_qp_handle or status is NULL, para error!"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raGetQpStatus == NULL,
        hccp_err("[get][ra_qp_status]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_get_qp_status is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raGetQpStatus(raQpHandle, status);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaMrReg(void *qpHandle, struct MrInfoT *info)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || info == NULL,
        hccp_err("[reg][ra_mr]qp_handle is NULL or info is NULL, para error!"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raMrReg == NULL,
        hccp_err("[reg][ra_mr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_mr_reg is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raMrReg(raQpHandle, info);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaMrDereg(void *qpHandle, struct MrInfoT *info)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || info == NULL || info->addr == NULL,
        hccp_err("[dereg][ra_mr]qp_handle or info or addr is NULL, para error!"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raMrDereg == NULL,
        hccp_err("[dereg][ra_mr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_mr_dereg is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raMrDereg(raQpHandle, info);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || wr->bufList == NULL || opRsp == NULL,
        hccp_err("[send][ra_wr]qp_handle or wr or buf_list or op_rsp is NULL, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(wr->bufList->len > MAX_SG_LIST_LEN_MAX,
        hccp_err("[send][ra_wr]sg list len is more than 2G, len(%u)", wr->bufList->len),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendWr == NULL,
        hccp_err("[send][ra_wr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_wr is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raSendWr(raQpHandle, wr, opRsp);
    RaRdevIncSendWrNum();
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSendWrV2(void *qpHandle, struct SendWrV2 *wr, struct SendWrRsp *opRsp)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || wr->bufList == NULL || opRsp == NULL,
        hccp_err("[send][ra_wr]qp_handle or wr or buf_list or op_rsp is NULL, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(wr->bufList->len > MAX_SG_LIST_LEN_MAX,
        hccp_err("[send][ra_wr]sg list len is more than 2G, len(%u)", wr->bufList->len),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendWrV2 == NULL,
        hccp_err("[send][ra_wr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_wr_v2 is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = raQpHandle->rdmaOps->raSendWrV2(raQpHandle, wr, opRsp);
    RaRdevIncSendWrNum();
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSendWrlist(void *qpHandle, struct SendWrlistData wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum)
{
    int ret;
    unsigned int i;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    struct WrlistSendCompleteNum wrlistNum = {0};

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || opRsp == NULL || sendNum == 0 || completeNum == NULL,
        hccp_err("[send][ra_wrlist]qp_handle or wr or op_rsp or complete_num is NULL or send_num is zero, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    for (i = 0; i < sendNum; i++) {
        if (wr[i].memList.len > MAX_SG_LIST_LEN_MAX) {
            hccp_err("[send][ra_wrlist]sg list len is more than 2G, len(%u)", wr[i].memList.len);
            return ConverReturnCode(RDMA_OP, -EINVAL);
        }
    }

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendWrlist == NULL,
        hccp_err("[send][ra_wrlist]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_wr is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    wrlistNum.sendNum = sendNum;
    wrlistNum.completeNum = completeNum;
    ret = raQpHandle->rdmaOps->raSendWrlist(raQpHandle, wr, opRsp, wrlistNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSendWrlistExt(void *qpHandle, struct SendWrlistDataExt wr[], struct SendWrRsp opRsp[],
    unsigned int sendNum, unsigned int *completeNum)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    struct WrlistSendCompleteNum wrlistNum = {0};
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(qpHandle == NULL || wr == NULL || opRsp == NULL || sendNum == 0 || completeNum == NULL,
        hccp_err("[send][ra_wrlist]qp_handle or wr or op_rsp or complete_num is NULL "
                 "or send_num is zero, para error!"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    for (i = 0; i < sendNum; i++) {
        if (wr[i].memList.len > MAX_SG_LIST_LEN_MAX) {
            hccp_err("[send][ra_wrlist]sg list len is more than 2G, len(%u)", wr[i].memList.len);
            return ConverReturnCode(RDMA_OP, -EINVAL);
        }
    }

    CHK_PRT_RETURN(raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSendWrlistExt == NULL,
        hccp_err("[send][ra_wrlist]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_send_wr is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    wrlistNum.sendNum = sendNum;
    wrlistNum.completeNum = completeNum;
    ret = raQpHandle->rdmaOps->raSendWrlistExt(raQpHandle, wr, opRsp, wrlistNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetNotifyBaseAddr(void *rdevHandle, unsigned long long *va, unsigned long long *size)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || va == NULL || size == NULL,
        hccp_err("[get][ra_notify_base_addr]rdev_handle or va or size is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(rdevHandleTmp->rdmaOps == NULL || rdevHandleTmp->rdmaOps->raGetNotifyBaseAddr == NULL,
        hccp_err("[get][ra_notify_base_addr]rdma_ops is NULL or ra_get_notify_base_addr is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdevHandleTmp->rdmaOps->raGetNotifyBaseAddr(rdevHandleTmp, va, size);
    RaRdevSaveNotifyMr(rdevHandleTmp, ret, *va, *size);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetNotifyMrInfo(void *rdevHandle, struct MrInfoT *info)
{
    struct RaRdmaHandle *rdevHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || info == NULL,
        hccp_err("[get][ra_notify_mr_info]rdev_handle or info is NULL, invalid"), ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(rdevHandleTmp->rdmaOps == NULL || rdevHandleTmp->rdmaOps->raGetNotifyMrInfo == NULL,
        hccp_err("[get][ra_notify_mr_info]rdma_ops is NULL or ra_get_notify_mr_info is NULL, invalid"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = rdevHandleTmp->rdmaOps->raGetNotifyMrInfo(rdevHandleTmp, info);
    RaRdevSaveNotifyMr(rdevHandleTmp, ret, (uint64_t)(uintptr_t)info->addr, info->size);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketGetWhiteListStatus(unsigned int *enable)
{
    CHK_PRT_RETURN(enable == NULL, hccp_err("[get][ra_socket_white_list_status]white list switch enable is NULL"),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    *enable = gWhiteListSwitch;
    hccp_info_socket("white list status: enable[%u]", *enable);
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSocketSetWhiteListStatus(unsigned int enable)
{
    hccp_run_info("Input parameters: enable[%u]", enable);

    CHK_PRT_RETURN(enable != WHITE_LIST_DISABLE && enable != WHITE_LIST_ENABLE,
        hccp_err("[set][ra_socket_white_list_status]white list switch is invalid, enable(%u)", enable),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    gWhiteListSwitch = enable;
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaSocketWhiteListAdd(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    struct RaSocketHandle *socketHandleTmp = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(whiteList == NULL || num > MAX_WLIST_NUM || num == 0,
        hccp_err("[add][ra_socket_white_list]white_list is NULL, or num(%u) > %u or = 0, invalid", num, MAX_WLIST_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    socketHandleTmp = (struct RaSocketHandle *)socketHandle;
    if (socketHandleTmp == NULL || socketHandleTmp->socketOps == NULL ||
        socketHandleTmp->socketOps->raSocketWhiteListAdd == NULL) {
        hccp_err("[add][ra_socket_white_list]socket_handle or func is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    ret = RaInetPton(socketHandleTmp->rdevInfo.family, socketHandleTmp->rdevInfo.localIp, localIp, MAX_IP_LEN);
    CHK_PRT_RETURN(ret, hccp_err("[add][ra_socket_white_list]ra_inet_pton for local_ip failed, ret(%d)", ret),
        ConverReturnCode(SOCKET_OP, ret));

    phyId = socketHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[add][ra_socket_white_list]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(SOCKET_OP, -EINVAL));

    hccp_info_socket("Input parameters: phyId[%u], localIp[%s], num[%u]", phyId, localIp, num);

    ret = socketHandleTmp->socketOps->raSocketWhiteListAdd(socketHandleTmp->rdevInfo, whiteList, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketWhiteListDel(void *socketHandle, struct SocketWlistInfoT whiteList[], unsigned int num)
{
    struct RaSocketHandle *socketHandleTmp = NULL;
    char localIp[MAX_IP_LEN] = {0};
    unsigned int phyId;
    int ret;

    if (whiteList == NULL || num > MAX_WLIST_NUM || num == 0) {
        hccp_err("[del][ra_socket_white_list]white_list is NULL, or num (%u) > %u or = 0, invalid", num, MAX_WLIST_NUM);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    socketHandleTmp = (struct RaSocketHandle *)socketHandle;
    if (socketHandleTmp == NULL || socketHandleTmp->socketOps == NULL ||
        socketHandleTmp->socketOps->raSocketWhiteListDel == NULL) {
        hccp_err("[del][ra_socket_white_list]socket_handle or func is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    ret = RaInetPton(socketHandleTmp->rdevInfo.family, socketHandleTmp->rdevInfo.localIp, localIp, MAX_IP_LEN);
    if (ret) {
        hccp_err("[del][ra_socket_white_list]ra_inet_pton for local_ip failed, ret(%d)", ret);
        return ConverReturnCode(SOCKET_OP, ret);
    }

    phyId = socketHandleTmp->rdevInfo.phyId;
    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_err("[del][ra_socket_white_list]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    hccp_info_socket("Input parameters: phyId[%u], localIp[%s], num[%u]", phyId, localIp, num);

    ret = socketHandleTmp->socketOps->raSocketWhiteListDel(socketHandleTmp->rdevInfo, whiteList, num);
    return ConverReturnCode(SOCKET_OP, ret);
}

STATIC int RaIfaddrInfoConverter(unsigned int phyId, bool isAll, struct InterfaceInfo interfaceInfos[],
    unsigned int *num)
{
    struct IfaddrInfo *ifaddrInfos = NULL;
    unsigned int interfaceVersionV2 = 0;
    unsigned int interfaceVersion = 0;
    unsigned int i;
    int ret;

    ret = RaGetInterfaceVersion(phyId, RA_RS_GET_IFADDRS, &interfaceVersion);
    CHK_PRT_RETURN(ret != 0 || interfaceVersion == 0,
        hccp_err("[converter][ra_ifaddr]get interface version failed, ret(%d), phyId(%u), interfaceVersion(%u)", ret,
            phyId, interfaceVersion),
        -EINVAL);

    ret = RaGetInterfaceVersion(phyId, RA_RS_GET_IFADDRS_V2, &interfaceVersionV2);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[converter][ra_ifaddr]get interface version failed, ret(%d), phyId(%u), interfaceVersion(%u)", ret,
            phyId, interfaceVersion),
        -EINVAL);

    CHK_PRT_RETURN(interfaceVersionV2 < GET_IFADDRS_VERSION_3 && isAll,
        hccp_err("[converter][ra_ifaddr]current version do not support get all device ip addr, interfaceVersion(%u), "
                 "interfaceVersionV2(%u), isAll(%d)",
            interfaceVersion, interfaceVersionV2, isAll),
        -EPROTONOSUPPORT);

    if (interfaceVersion == GET_IFADDRS_VERSION_1) {
        ifaddrInfos = calloc(*num, sizeof(struct IfaddrInfo));
        if (ifaddrInfos == NULL) {
            hccp_err("[converter][ra_ifaddr]calloc for ifaddr_infos failed");
            return -EINVAL;
        }

        ret = RaHdcGetIfaddrs(phyId, ifaddrInfos, num);
        if (ret) {
            hccp_err("[converter][ra_ifaddr]ra_hdc_get_ifaddrs failed, ret(%d), phyId(%u), num(%u)", ret, phyId, *num);
            free(ifaddrInfos);
            return ret;
        }

        for (i = 0; i < *num; i++) {
            // device 网卡只支持IPv4
            interfaceInfos[i].family = AF_INET;
            interfaceInfos[i].scopeId = 0;
            ret = memcpy_s(&(interfaceInfos[i].ifaddr.ip), sizeof(union HccpIpAddr), &(ifaddrInfos[i].ip),
                sizeof(union HccpIpAddr));
            if (ret) {
                hccp_err("[converter][ra_ifaddr]memcpy_s interface[%u] ip failed, ret(%d)", i, ret);
                free(ifaddrInfos);
                return -ESAFEFUNC;
            }

            ret = memcpy_s(&(interfaceInfos[i].ifaddr.mask), sizeof(struct in_addr), &(ifaddrInfos[i].mask),
                sizeof(struct in_addr));
            if (ret) {
                hccp_err("[converter][ra_ifaddr]memcpy_s interface[%u] mask failed, ret(%d)", i, ret);
                free(ifaddrInfos);
                return -ESAFEFUNC;
            }
        }

        free(ifaddrInfos);
    } else if (interfaceVersion == GET_IFADDRS_VERSION_2) { /* version 2 support IPV6 and IPV4 */
        ret = RaHdcGetIfaddrsV2(phyId, isAll, interfaceInfos, num);
        CHK_PRT_RETURN(ret,
            hccp_err("[converter][ra_ifaddr]ra_hdc_get_ifaddrs_v2 failed, ret(%d), phyId(%u), isAll(%d), num(%u)", ret,
                phyId, isAll, *num),
            ret);
    } else {
        hccp_err("[converter][ra_ifaddr]interface version not support, interfaceVersion(%u)", interfaceVersion);
        return -EINVAL;
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaGetIfnum(struct RaGetIfattr *config, unsigned int *num)
{
    unsigned int interfaceVersion = 0;
    int ret;

    CHK_PRT_RETURN(config == NULL || num == NULL, hccp_err("[get][ra_ifnum]config or num is NULL"),
        ConverReturnCode(OTHERS, -EINVAL));

    CHK_PRT_RETURN(config->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_ifnum]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!",
            config->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%u], isAll:[%d]", config->phyId, config->nicPosition,
        config->isAll);

    ret = RaGetInterfaceVersion(config->phyId, RA_RS_GET_IFADDRS_V2, &interfaceVersion);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[get][ra_ifnum]get interface version failed, ret(%d), phyId(%u), interfaceVersion(%u)", ret,
            config->phyId, interfaceVersion),
        ConverReturnCode(OTHERS, -EINVAL));

    CHK_PRT_RETURN(interfaceVersion < GET_IFADDRS_VERSION_3 && config->nicPosition == NETWORK_OFFLINE && config->isAll,
        hccp_err("[get][ra_ifnum]current version do not support get all ip num, interfaceVersion(%u), isAll(%d)",
            interfaceVersion, config->isAll),
        ConverReturnCode(OTHERS, -ENOTSUPP));

    if (config->nicPosition == NETWORK_OFFLINE) {
        ret = RaHdcGetIfnum(config->phyId, config->isAll, num);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_ifnum]ra_hdc_get_ifnum failed, ret(%d)", ret),
            ConverReturnCode(OTHERS, ret));
    } else if (config->nicPosition == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetIfnum(config->phyId, num);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_ifnum]ra_peer_get_ifnum failed, ret(%d)", ret),
            ConverReturnCode(OTHERS, ret));
    } else {
        hccp_err("[get][ra_ifnum]Wrong mode, do not support online mode");
        return ConverReturnCode(OTHERS, -EPROTONOSUPPORT);
    }

    CHK_PRT_RETURN((*num) > MAX_SUPPORT_IFNUM,
        hccp_err("[get][ra_ifnum]get interface num(%d)! It must greater or equal to 0 and less or equal to %d", (*num),
            MAX_SUPPORT_IFNUM),
        ConverReturnCode(OTHERS, -EINVAL));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaGetIfaddrs(struct RaGetIfattr *config, struct InterfaceInfo interfaceInfos[],
    unsigned int *num)
{
    int ret;

    CHK_PRT_RETURN(config == NULL || interfaceInfos == NULL || num == NULL,
        hccp_err("[get][ra_ifaddrs]config or interface_infos or num is NULL"), ConverReturnCode(OTHERS, -EINVAL));

    CHK_PRT_RETURN(config->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_ifaddrs]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!",
            config->phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(OTHERS, -EINVAL));

    CHK_PRT_RETURN(*num == 0,
        hccp_err("[get][ra_ifaddrs]interface num(%u) is invalid! it must be greater than 0", *num),
        ConverReturnCode(OTHERS, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u], nicPosition:[%u], isAll[%d], interface num[%u]", config->phyId,
        config->nicPosition, config->isAll, *num);

    if (config->nicPosition == NETWORK_OFFLINE) {
        CHK_PRT_RETURN(*num > MAX_INTERFACE_NUM,
            hccp_err("[get][ra_ifaddrs]interface num(%u) is invalid! it must be less than %d!", *num,
                MAX_INTERFACE_NUM),
            ConverReturnCode(OTHERS, -EINVAL));

        ret = RaIfaddrInfoConverter(config->phyId, config->isAll, interfaceInfos, num);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_ifaddrs]ra_hdc_get_ifaddrs failed, ret(%d)", ret),
            ConverReturnCode(OTHERS, ret));
    } else if (config->nicPosition == NETWORK_PEER_ONLINE) {
        ret = RaPeerGetIfaddrs(config->phyId, interfaceInfos, num);
        CHK_PRT_RETURN(ret, hccp_err("[get][ra_ifaddrs]ra_peer_get_ifaddrs failed, ret(%d)", ret),
            ConverReturnCode(OTHERS, ret));
    } else {
        hccp_err("[get][ra_ifaddrs]Wrong mode, do not support online mode");
        return ConverReturnCode(OTHERS, -EPROTONOSUPPORT);
    }
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaGetInterfaceVersion(unsigned int phyId, unsigned int interfaceOpcode,
    unsigned int *interfaceVersion)
{
    int ret;

    if (interfaceVersion == NULL || phyId >= RA_MAX_PHY_ID_NUM || interfaceOpcode >= RA_RS_EXTER_OP_MAX_NUM) {
        hccp_err("[get][ra_interface_version]para is invalid! interface_version is NULL or phyId(%u) is "
                 "greater than [%u] or interface_opcode(%u) more than [%u]",
            phyId, RA_MAX_PHY_ID_NUM, interfaceOpcode, RA_RS_EXTER_OP_MAX_NUM);
        return ConverReturnCode(OTHERS, -EINVAL);
    }

    ret = RaHdcGetInterfaceVersion(phyId, interfaceOpcode, interfaceVersion);
    return ConverReturnCode(OTHERS, ret);
}

int ConverReturnCode(enum ModuleType module, int erroCode)
{
    unsigned int i;
    unsigned int num = sizeof(gErrcodeInfoList) / sizeof(gErrcodeInfoList[0]);
    int ret = CONVER_ERROR_CODE(module, DEFAULT_ERRCODE_TYPE, DEFAULT_MODULE_ERRCODE);

    if (erroCode == 0) {
        return 0;
    }

    if (erroCode / ACL_ERRCODE_DIGIT) { /* ACL error codes are transparently transmitted */
        return erroCode;
    }

    for (i = 0; i < num; i++) {
        if (erroCode == gErrcodeInfoList[i].origErrcode) {
            ret = CONVER_ERROR_CODE(module, gErrcodeInfoList[i].errType, gErrcodeInfoList[i].moduleErrcode);
            break;
        }
    }

    if (erroCode != -EAGAIN) { // 防止刷屏
        hccp_info_others("ConverReturnCode: orig_errcode[%d] curr_errcode[%d]", erroCode, ret);
    }
    return ret;
}

HCCP_ATTRI_VISI_DEF int RaRecvWrlist(void *qpHandle, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    int ret;
    unsigned int i;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || wr == NULL || recvNum == 0 || completeNum == NULL) {
        hccp_err("[recv][ra_wrlist]qp_handle or wr or complete_num is NULL or recv_num[%u] is zero, para error!",
            recvNum);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    for (i = 0; i < recvNum; i++) {
        if (wr[i].memList.len > MAX_SG_LIST_LEN_MAX) {
            hccp_err("[recv][ra_wrlist]sg list len is more than 2G, len(%u)", wr[i].memList.len);
            return ConverReturnCode(RDMA_OP, -EINVAL);
        }
    }
    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raRecvWrlist == NULL) {
        hccp_err("[recv][ra_wrlist]rdma_ops or ra_recv_wrlist is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    ret = raQpHandle->rdmaOps->raRecvWrlist(raQpHandle, wr, recvNum, completeNum);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetQpContext(void *qpHandle, void **qp, void **sendCq, void **recvCq)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || qp == NULL || sendCq == NULL || recvCq == NULL) {
        hccp_err("[request][ra_get_qp_context]qp_handle or qp or sendCq or recvCq is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raGetQpContext == NULL) {
        hccp_err("[get][ra_get_qp_context]rdma_ops is NULL or ra_get_qp_context is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    CHK_PRT_RETURN(raQpHandle->phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][rs_get_qp_context]qpPeer->phyId[%u] >= max_dev_num[%u], input is invalid", raQpHandle->phyId,
            RA_MAX_PHY_ID_NUM),
        -EINVAL);

    ret = raQpHandle->rdmaOps->raGetQpContext(raQpHandle, qp, sendCq, recvCq);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCqCreate(void *rdevHandle, struct CqAttr *attr)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raCqCreate == NULL ||
                       attr == NULL || attr->ibSendCq == NULL || attr->ibRecvCq == NULL || attr->qpContext == NULL,
        hccp_err("[create][ra_cq]para is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[create][ra_cq]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u]", phyId);

    ret = rdmaHandleTmp->rdmaOps->raCqCreate(rdmaHandleTmp, attr);
    CHK_PRT_RETURN(ret != 0 || *attr->ibSendCq == NULL || *attr->ibRecvCq == NULL || *attr->qpContext == NULL,
        hccp_err("[create][ra_cq]create cq failed, ret(%d) phyId(%u)", ret, phyId), ConverReturnCode(RDMA_OP, -EINVAL));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCqDestroy(void *rdevHandle, struct CqAttr *attr)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL ||
                       rdmaHandleTmp->rdmaOps->raCqDestroy == NULL || attr == NULL || attr->ibSendCq == NULL ||
                       attr->ibRecvCq == NULL || attr->qpContext == NULL,
        hccp_err("[destroy][ra_cq]para is NULL or func is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[destroy][ra_cq]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    hccp_run_info("Input parameters: phyId[%u]", phyId);

    ret = rdmaHandleTmp->rdmaOps->raCqDestroy(rdmaHandleTmp, attr);
    CHK_PRT_RETURN(ret != 0 || *attr->ibSendCq == NULL || *attr->ibRecvCq == NULL || *attr->qpContext == NULL,
        hccp_err("[destroy][ra_cq]destroy cq failed, ret(%d) phyId(%u)", ret, phyId),
        ConverReturnCode(RDMA_OP, -EINVAL));

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaNormalQpCreate(void *rdevHandle, struct ibv_qp_init_attr *qpInitAttr, void **qpHandle,
    void **qp)
{
    struct RaRdmaHandle *rdmaHandleTmp = (struct RaRdmaHandle *)rdevHandle;
    unsigned int phyId;
    int ret;

    if (rdevHandle == NULL || rdmaHandleTmp->rdmaOps == NULL || rdmaHandleTmp->rdmaOps->raNormalQpCreate == NULL ||
        qpInitAttr == NULL || qp == NULL) {
        hccp_err("[create][ra_normal_qp]para is NULL or func is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (qpHandle == NULL) {
        hccp_err("[create][ra_normal_qp]qp_handle is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    phyId = rdmaHandleTmp->rdevInfo.phyId;
    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_err("[create][ra_normal_qp]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    hccp_run_info("Input parameters: phyId[%u]", phyId);

    ret = rdmaHandleTmp->rdmaOps->raNormalQpCreate(rdmaHandleTmp, qpInitAttr, qpHandle, qp);
    if (ret != 0 || *qpHandle == NULL) {
        hccp_err("[create][ra_normal_qp]create qp failed, ret(%d) phyId(%u)", ret, phyId);
        return ConverReturnCode(RDMA_OP, ret);
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaNormalQpDestroy(void *qpHandle)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL) {
        hccp_err("[destroy][ra_normal_qp]qp_handle is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raNormalQpDestroy == NULL) {
        hccp_err("[destroy][ra_qp]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_normal_qp_destroy is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    hccp_run_info("Input parameters: qpn[%u], phyId[%u], rdevIndex[%u]", raQpHandle->qpn, raQpHandle->phyId,
        raQpHandle->rdevIndex);

    ret = raQpHandle->rdmaOps->raNormalQpDestroy(raQpHandle);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetQpAttrQos(void *qpHandle, struct QosAttr *attr)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || attr == NULL) {
        hccp_err("[set][ra_qp_attr_qos]qp_handle or attr is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSetQpAttrQos == NULL) {
        hccp_err("[set][ra_qp_attr_qos]rdma_ops is NULL or rdma_ops->ra_set_qp_attr_qos is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raQpHandle->rdmaOps->raSetQpAttrQos(raQpHandle, attr);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetQpAttrTimeout(void *qpHandle, unsigned int *timeout)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || timeout == NULL) {
        hccp_err("[set][ra_qp_attr_timeout]qp_handle or timeout is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSetQpAttrTimeout == NULL) {
        hccp_err("[set][ra_qp_attr_timeout]rdma_ops is NULL or rdma_ops->ra_set_qp_attr_timeout is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raQpHandle->rdmaOps->raSetQpAttrTimeout(raQpHandle, timeout);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSetQpAttrRetryCnt(void *qpHandle, unsigned int *retryCnt)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || retryCnt == NULL) {
        hccp_err("[set][ra_qp_attr_retry_cnt]qp_handle or retry_cnt is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raSetQpAttrRetryCnt == NULL) {
        hccp_err("[set][ra_qp_attr_retry_cnt]rdma_ops is NULL or rdma_ops->ra_set_qp_attr_retry_cnt is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raQpHandle->rdmaOps->raSetQpAttrRetryCnt(raQpHandle, retryCnt);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCreateCompChannel(const void *rdmaHandle, void **compChannel)
{
    int ret;
    struct RaRdmaHandle *raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;

    if (rdmaHandle == NULL) {
        hccp_err("[ra_create_comp_channel]rdma_handle(%p) is NULL, para error!", rdmaHandle);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (compChannel == NULL) {
        hccp_err("[ra_create_comp_channel]comp_channel is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raRdmaHandle->rdmaOps == NULL || raRdmaHandle->rdmaOps->raCreateCompChannel == NULL) {
        hccp_err("[ra_create_comp_channel]rdma_ops is NULL or ra_rdma_handle->rdma_ops->ra_create_comp_channel "
                 "is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raRdmaHandle->rdmaOps->raCreateCompChannel(raRdmaHandle, compChannel);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaDestroyCompChannel(const void *rdmaHandle, void *compChannel)
{
    int ret;
    struct RaRdmaHandle *raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;

    if (rdmaHandle == NULL) {
        hccp_err("[ra_destroy_comp_channel]rdma_handle(%p) is NULL, para error!", rdmaHandle);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (compChannel == NULL) {
        hccp_err("[ra_destroy_comp_channel]comp_channel is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raRdmaHandle->rdmaOps == NULL || raRdmaHandle->rdmaOps->raDestroyCompChannel == NULL) {
        hccp_err("[ra_destroy_comp_channel]rdma_ops is NULL or ra_rdma_handle->rdma_ops->ra_destroy_comp_channel "
                 "is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raRdmaHandle->rdmaOps->raDestroyCompChannel(compChannel);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetCqeErrInfo(unsigned int phyId, struct CqeErrInfo *info)
{
    int ret;

    if (info == NULL) {
        hccp_err("[ra_get_cqe_err_info]cqe_err_info is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_err("[ra_get_cqe_err_info]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = RaHdcGetCqeErrInfo(phyId, info);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaRdevGetCqeErrInfoList(void *rdmaHandle, struct CqeErrInfo *infoList, unsigned int *num)
{
    struct RaRdmaHandle *raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;
    int ret;

    if (rdmaHandle == NULL || infoList == NULL || num == NULL) {
        hccp_err("[get][cqe_err_info_list]rdma_handle or info_list or num is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (*num == 0 || *num > CQE_ERR_INFO_MAX_NUM) {
        hccp_err("[get][cqe_err_info_list]num(%u) is invalid!", *num);
        return ConverReturnCode(OTHERS, -EINVAL);
    }

    ret = RaHdcGetCqeErrInfoList(raRdmaHandle, infoList, num);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaGetQpAttr(void *qpHandle, struct QpAttr *attr)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret = 0;

    if (qpHandle == NULL || attr == NULL) {
        hccp_err("[get][GetQpAttr]qpHandle or attr is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    attr->qpn = raQpHandle->qpn;
    attr->udpSport = raQpHandle->typicalQpAttr.udpSport;
    attr->pathMtu = raQpHandle->typicalQpAttr.pathMtu;
    attr->vendorPrivInfo = raQpHandle->typicalQpAttr.vendorPrivInfo;
    attr->psn = raQpHandle->psn;
    attr->gidIdx = raQpHandle->gidIdx;
    (void)memcpy_s(attr->gid, HCCP_GID_RAW_LEN, raQpHandle->rdmaHandle->gid, HCCP_GID_RAW_LEN);

    if (raQpHandle->rdmaOps != NULL && raQpHandle->rdmaOps->raGetQpHyperFeature != NULL) {
        ret = raQpHandle->rdmaOps->raGetQpHyperFeature(raQpHandle, &attr->feature);
        if (ret != 0) {
            hccp_warn_rma("[get][GetQpAttr]raGetQpHyperFeature unsuccessful, ret:%d phyId:%u qpn:%u", ret,
                raQpHandle->phyId, raQpHandle->qpn);
        }
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCreateSrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    int ret;
    struct RaRdmaHandle *raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;

    if (rdmaHandle == NULL) {
        hccp_err("[ra_create_srq]rdma_handle(%p) is NULL, para error!", rdmaHandle);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (attr == NULL) {
        hccp_err("[ra_create_srq]srq_attr is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    // 创建srq&srq cq
    if (raRdmaHandle->rdmaOps == NULL || raRdmaHandle->rdmaOps->raCreateSrq == NULL) {
        hccp_err("[ra_create_srq]rdma_ops is NULL or ra_rdma_handle->rdma_ops->ra_create_srq "
                 "is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raRdmaHandle->rdmaOps->raCreateSrq(raRdmaHandle, attr);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaDestroySrq(const void *rdmaHandle, struct SrqAttr *attr)
{
    int ret;
    struct RaRdmaHandle *raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;

    if (rdmaHandle == NULL) {
        hccp_err("[ra_destroy_srq]rdma_handle(%p) is NULL, para error!", rdmaHandle);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (attr == NULL) {
        hccp_err("[ra_destroy_srq]srq_handle is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    // 销毁srq&srq cq
    if (raRdmaHandle->rdmaOps == NULL || raRdmaHandle->rdmaOps->raDestroySrq == NULL) {
        hccp_err("[ra_destroy_srq]rdma_ops is NULL or ra_rdma_handle->rdma_ops->ra_destroy_srq "
                 "is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raRdmaHandle->rdmaOps->raDestroySrq(raRdmaHandle, attr);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaCreateEventHandle(int *eventHandle)
{
    int ret;

    if (eventHandle == NULL) {
        hccp_err("[ra_create_event_handle]event_handle is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    ret = RaPeerCreateEventHandle(eventHandle);
    if (ret) {
        hccp_err("[ra_create_event_handle]ra_peer_create_event_handle failed ret(%d)", ret);
        return ConverReturnCode(SOCKET_OP, ret);
    }
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaCtlEventHandle(int eventHandle, const void *fdHandle, int opcode, enum RaEpollEvent event)
{
    int ret;

    if (eventHandle < 0) {
        hccp_err("[ra_ctl_event_handle]event_handle[%d] is invalid", eventHandle);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    if (fdHandle == NULL) {
        hccp_err("[ra_ctl_event_handle]fd_handle is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    if (opcode != EPOLL_CTL_ADD && opcode != EPOLL_CTL_DEL && opcode != EPOLL_CTL_MOD) {
        hccp_err("[ra_ctl_event_handle]opcode[%d] invalid, valid opcode includes {%d, %d, %d}", opcode, EPOLL_CTL_ADD,
            EPOLL_CTL_DEL, EPOLL_CTL_MOD);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }
    if (event >= RA_EPOLLINVALD) {
        hccp_err("[ra_ctl_event_handle]event[%d] invalid, valid range [0, %d)", event, RA_EPOLLINVALD);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    ret = RaPeerCtlEventHandle(eventHandle, fdHandle, opcode, event);
    if (ret) {
        hccp_err("[ra_ctl_event_handle]ra_peer_ctl_event_handle failed ret(%d)", ret);
        return ConverReturnCode(SOCKET_OP, ret);
    }
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaWaitEventHandle(int eventHandle, struct SocketEventInfoT *eventInfos, int timeout,
    unsigned int maxevents, unsigned int *eventsNum)
{
    int ret;

    if (eventHandle < 0) {
        hccp_err("[ra_wait_event_handle]event_handle[%d] is invalid", eventHandle);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    if (eventInfos == NULL) {
        hccp_err("[ra_wait_event_handle]event_infos is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    if (timeout < -1) {
        hccp_err("[ra_wait_event_handle]timeout[%d ms] is invalid", timeout);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    if (maxevents > MAX_SOCKET_EVENT_NUM) {
        hccp_err("[ra_wait_event_handle]maxevents[%u] exceeds %u", maxevents, MAX_SOCKET_EVENT_NUM);
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    if (eventsNum == NULL) {
        hccp_err("[ra_wait_event_handle]events_num is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    ret = RaPeerWaitEventHandle(eventHandle, eventInfos, timeout, maxevents, eventsNum);
    if (ret) {
        hccp_err("[ra_wait_event_handle]ra_peer_wait_event_handle failed ret(%d)", ret);
        return ConverReturnCode(SOCKET_OP, ret);
    }
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaDestroyEventHandle(int *eventHandle)
{
    int ret;

    if (eventHandle == NULL) {
        hccp_err("[ra_destroy_event_handle]event_handle is NULL");
        return ConverReturnCode(SOCKET_OP, -EINVAL);
    }

    ret = RaPeerDestroyEventHandle(eventHandle);
    if (ret) {
        hccp_err("[ra_destroy_event_handle]ra_peer_destroy_event_handle failed ret(%d)", ret);
        return ConverReturnCode(SOCKET_OP, ret);
    }
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPollCq(void *qpHandle, bool isSendCq, unsigned int numEntries, void *wc)
{
    int ret;
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;

    if (qpHandle == NULL || wc == NULL) {
        hccp_err("[ra_poll]qp_handle is NULL or wc is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raPollCq == NULL) {
        hccp_err("[ra_poll]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_poll_cq is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raQpHandle->rdmaOps->raPollCq(raQpHandle, isSendCq, numEntries, wc);
    if (ret < 0) {
        return ConverReturnCode(RDMA_OP, ret);
    }
    return ret;
}

HCCP_ATTRI_VISI_DEF int RaTypicalQpModify(void *qpHandle, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    unsigned int phyId;
    int ret;

    if (qpHandle == NULL || raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raTypicalQpModify == NULL) {
        hccp_err("[modify][ra_qp]qp_handle is NULL or func is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (localQpInfo == NULL || remoteQpInfo == NULL) {
        hccp_err("[modify][ra_qp]local_qp_info is NULL or remote_qp_info is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    phyId = raQpHandle->phyId;
    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_err("[modify][ra_qp]phyId(%u) must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    hccp_run_info("Input parameters: phyId[%u] local_qpn[%u] remote_qpn[%u]", phyId, localQpInfo->qpn,
        remoteQpInfo->qpn);

    ret = raQpHandle->rdmaOps->raTypicalQpModify(raQpHandle, localQpInfo, remoteQpInfo);
    if (ret != 0) {
        hccp_err("[modify][ra_qp]modify qp failed, ret(%d) phyId(%u)", ret, phyId);
        return ConverReturnCode(RDMA_OP, ret);
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaTypicalSendWr(void *qpHandle, struct SendWr *wr, struct SendWrRsp *opRsp)
{
    struct RaQpHandle *raQpHandle = (struct RaQpHandle *)qpHandle;
    int ret;

    if (qpHandle == NULL || wr == NULL || wr->bufList == NULL || opRsp == NULL) {
        hccp_err("[send][ra_wr]qp_handle or wr or buf_list or op_rsp is NULL, para error!");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (wr->bufList->len > MAX_SG_LIST_LEN_MAX) {
        hccp_err("[send][ra_wr]sg list len is more than 2G, len(%u)", wr->bufList->len);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    if (raQpHandle->rdmaOps == NULL || raQpHandle->rdmaOps->raTypicalSendWr == NULL) {
        hccp_err("[send][ra_wr]rdma_ops is NULL or ra_qp_handle->rdma_ops->ra_typical_send_wr is NULL, invalid");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    ret = raQpHandle->rdmaOps->raTypicalSendWr(raQpHandle, wr, opRsp);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaSocketGetVnicIpInfos(unsigned int phyId, enum IdType type, unsigned int ids[],
    unsigned int num, struct IpInfo infos[])
{
    int ret;

    if (ids == NULL || num == 0 || infos == NULL) {
        hccp_err("[get][vnic_ip]ids is NULL or num:%u == 0 or infos is NULL", num);
        return ConverReturnCode(OTHERS, -EINVAL);
    }

    if (phyId >= RA_MAX_PHY_ID_NUM) {
        hccp_err("[get][vnic_ip]phyId(%u) is invalid! it must be greater than or equal to 0 and less than %d!", phyId,
            RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(OTHERS, -EINVAL);
    }

    if (type != PHY_ID_VNIC_IP && type != SDID_VNIC_IP) {
        hccp_err("[get][vnic_ip]type[%u] invalid", type);
        return ConverReturnCode(OTHERS, -EINVAL);
    }

    ret = RaHdcGetVnicIpInfos(phyId, type, ids, num, infos);
    if (ret) {
        hccp_err("[get][vnic_ip]ra_hdc_get_vnic_ip_infos failed, ret(%d)", ret);
        return ConverReturnCode(OTHERS, ret);
    }
    return 0;
}

int RaQpBatchModifyCheckParam(void *rdmaHandle, void *qpHandle[], unsigned int num, int expectStatus)
{
    unsigned int i;

    if (rdmaHandle == NULL || num == 0 || qpHandle == NULL ||
        ((expectStatus != RA_QP_STATUS_CONNECTED) && (expectStatus != RA_QP_STATUS_PAUSE))) {
        hccp_err("[batch_modify][check param]expect_status is %d or rdma_handle is NULL or num[%u] is 0!", expectStatus,
            num);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    for (i = 0; i < num; i++) {
        if (qpHandle[i] == NULL) {
            hccp_err("[modify][ra_qp]qp_handle[%u] is NULL", i);
            return ConverReturnCode(RDMA_OP, -EINVAL);
        }
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaQpBatchModify(void *rdmaHandle, void *qpHandle[], unsigned int num, int expectStatus)
{
    struct RaRdmaHandle *raRdmaHandle = NULL;
    unsigned int phyId;
    int ret;

    ret = RaQpBatchModifyCheckParam(rdmaHandle, qpHandle, num, expectStatus);
    CHK_PRT_RETURN(ret, hccp_err("ra_qp_batch_modify_check_param invalid[%d]", ret), ret);

    raRdmaHandle = (struct RaRdmaHandle *)rdmaHandle;
    phyId = raRdmaHandle->rdevInfo.phyId;
    if (phyId >= RA_MAX_PHY_ID_NUM || raRdmaHandle->rdmaOps == NULL || raRdmaHandle->rdmaOps->raQpBatchModify == NULL) {
        hccp_err("[modify][ra_qp]phyId(%u) must be greater than or equal to 0 and less than %d or ops is NULL or "
                 "ra_rdma_handle->rdma_ops->ra_qp_batch_modify is NULL!",
            phyId, RA_MAX_PHY_ID_NUM);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    hccp_run_info("Input parameters: phyId[%u] num[%u] expect_status[%d]", phyId, num, expectStatus);

    // avoid poll_cq thread to poll cq
    if ((raRdmaHandle->supportLite != 0) && expectStatus == RA_QP_STATUS_PAUSE) {
        RA_PTHREAD_MUTEX_LOCK(&raRdmaHandle->rdevMutex);
    }
    ret = raRdmaHandle->rdmaOps->raQpBatchModify(rdmaHandle, qpHandle, num, expectStatus);
    if (ret != 0) {
        hccp_err("[modify][ra_qp_batch_modify]modify qp to [%d] failed, ret[%d] phyId[%u]", expectStatus, ret, phyId);
    }
    if ((raRdmaHandle->supportLite != 0) && expectStatus == RA_QP_STATUS_PAUSE) {
        RA_PTHREAD_MUTEX_UNLOCK(&raRdmaHandle->rdevMutex);
    }

    return ret;
}
