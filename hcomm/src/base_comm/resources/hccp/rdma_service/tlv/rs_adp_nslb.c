/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "dl_netco_function.h"
#include "file_opt.h"
#include "ra_rs_err.h"
#include "network_comm.h"
#include "rs_adp_nslb.h"

STATIC int RsGetNetcoCfg(unsigned int phyId, NetCoIpPortArg *netcoArg)
{
    char cfgVal[CFG_VAL_LEN] = {0};
    bool nslbSupport = false;
    int ret = 0;

    ret = FileReadCfg(NETCO_CFGFILE_PATH, (int)phyId, "udp_port_mode", cfgVal, CFG_VAL_LEN);
    // file not exist or item not found, degrade log level
    CHK_PRT_RETURN(ret == FILE_OPT_INNER_PARAM_ERR || ret == FILE_OPT_SYS_RD_FILE_NOT_FOUND,
        hccp_run_warn("file_read_cfg udp_port_mode unsuccessful, ret(%d)", ret), -ENOTSUPP);
    CHK_PRT_RETURN(ret != 0, hccp_err("file_read_cfg udp_port_mode failed, ret(%d)", ret), ret);

    nslbSupport = (strncmp(cfgVal, "nslb_dp", strlen("nslb_dp") + 1) == 0) ? true : false;
    CHK_PRT_RETURN(!nslbSupport, hccp_run_warn("phy_id(%u) not support nslb", phyId), -ENOTSUPP);

    (void)memset_s(cfgVal, CFG_VAL_LEN, 0, CFG_VAL_LEN);
    ret = FileReadCfg(NETCO_CFGFILE_PATH, (int)phyId, "nslb_dp_listen_port", cfgVal, CFG_VAL_LEN);
    // file not exist or item not found, degrade log level
    CHK_PRT_RETURN(ret == FILE_OPT_INNER_PARAM_ERR || ret == FILE_OPT_SYS_RD_FILE_NOT_FOUND,
        hccp_run_warn("file_read_cfg nslb_dp_listen_port unsuccessful, ret(%d)", ret), -ENOTSUPP);
    CHK_PRT_RETURN(ret != 0, hccp_err("file_read_cfg nslb_dp_listen_port failed, ret(%d)", ret), ret);

    netcoArg->listenPort = (unsigned short)strtoul(cfgVal, NULL, NETCO_PORT_NUM_BASE);
    netcoArg->gatewayPort = netcoArg->listenPort;
    return 0;
}

STATIC int RsNetcoInitArg(unsigned int phyId, NetCoIpPortArg *netcoArg)
{
    struct IfaddrInfo ifaddrInfos = {0};
    unsigned int gwAddr = 0;
    unsigned int num = 1;
    int ret = 0;

    ret = RsGetNetcoCfg(phyId, netcoArg);
    CHK_PRT_RETURN(ret == -ENOTSUPP, hccp_run_warn("get netco cfg unsuccessful, ret(%d) phyId(%u)", ret, phyId), ret);
    CHK_PRT_RETURN(ret != 0, hccp_err("get netco cfg failed, ret(%d) phyId(%u)", ret, phyId), ret);

    ret = RsGetIfaddrs(&ifaddrInfos, &num, phyId);
    CHK_PRT_RETURN(ret != 0 || num != 1, hccp_err("rs get ifaddr failed, ret(%d) or num(%u) != 1", ret, num), -EINVAL);
    netcoArg->localIp = ntohl(ifaddrInfos.ip.addr.s_addr);

    ret = NetGetGatewayAddress(phyId, &gwAddr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs get gateway failed ret %d", ret), ret);
    netcoArg->gatewayIp = ntohl(gwAddr);

    // 0 indicates port will be assigned randomly
    netcoArg->localNetPort = htons(0);
    return 0;
}

STATIC int RsNslbNetcoInit(unsigned int phyId, struct RsNslbCb *nslbCb)
{
    NetCoIpPortArg netcoArg = {0};
    struct rs_cb *rsCb = NULL;
    void *netcoCb = NULL;
    int ret = 0;

    ret = RsNetcoInitArg(phyId, &netcoArg);
    CHK_PRT_RETURN(ret == -ENOTSUPP, hccp_warn_others("get netco init arg unsuccessful, ret(%d)", ret), ret);
    CHK_PRT_RETURN(ret != 0, hccp_err("get netco init arg failed, ret(%d)", ret), -EINVAL);

    ret = RsNslbApiInit();
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_nslb_api_init[%d]", ret), ret);

    ret = RsGetRsCb(phyId, &rsCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_rs_cb failed, phyId(%u) invalid, ret(%d)", phyId, ret), ret);

    netcoCb = RsNetcoInit(rsCb->connCb.epollfd, netcoArg);
    CHK_PRT_RETURN(netcoCb == NULL, hccp_err("netco init failed"), -EINVAL);

    ret = pthread_mutex_init(&nslbCb->mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_nslb mutex_init failed, phyId(%u), ret(%d)", phyId, ret), ret);

    nslbCb->netcoCb = netcoCb;
    nslbCb->initFlag = true;
    return 0;
}

STATIC void RsNslbNetcoDeinit(struct RsNslbCb *nslbCb)
{
    RS_PTHREAD_MUTEX_LOCK(&nslbCb->mutex);
    RsNetcoDeinit(nslbCb->netcoCb);
    nslbCb->initFlag = false;
    nslbCb->netcoCb = NULL;

    RsNslbApiDeinit();
    RS_PTHREAD_MUTEX_ULOCK(&nslbCb->mutex);
    pthread_mutex_destroy(&nslbCb->mutex);
    return;
}

STATIC int RsNslbNetcoTblRequest(struct RsNslbCb *nslbCb, unsigned int type, char *data, unsigned int dataLen)
{
    int ret = 0;

    RS_PTHREAD_MUTEX_LOCK(&nslbCb->mutex);
    ret = RsNetcoTblAddUpd(nslbCb->netcoCb, type, data, dataLen);
    RS_PTHREAD_MUTEX_ULOCK(&nslbCb->mutex);

    return ret;
}

int RsNslbNetcoRequest(unsigned int phyId, struct RsNslbCb *nslbCb, unsigned int type, char *data, unsigned int dataLen)
{
    int ret = 0;

    switch (type) {
        case NETCO_REQ_TYPE_INIT:
            ret = RsNslbNetcoInit(phyId, nslbCb);
            CHK_PRT_RETURN(ret == -ENOTSUPP, hccp_warn_others("netco init unsuccessful ret(%d)", ret), -ENOTSUPP);
            break;
        case NETCO_REQ_TYPE_DEINIT:
            RsNslbNetcoDeinit(nslbCb);
            break;
        default:
            ret = RsNslbNetcoTblRequest(nslbCb, type, data, dataLen);
            break;
    }

    ret = (ret > 0) ? -ret : ret;
    CHK_PRT_RETURN(ret != 0, hccp_err("netco request failed, type(%u) ret(%d)", type, ret), ret);

    return 0;
}

int RsEpollNslbEventHandle(struct RsNslbCb *nslbCb, int fd, unsigned int events)
{
    unsigned int ret = 0;
    int retVal = 0;

    // netco is not initialized, no epoll event need to handle
    if (!nslbCb->initFlag) {
        return -ENODEV;
    }

    RS_PTHREAD_MUTEX_LOCK(&nslbCb->mutex);
    ret = RsNetcoEventDispatch(nslbCb->netcoCb, fd, events);
    retVal = (ret == NET_CO_PROCED) ? (int)ret : -ENODEV;
    RS_PTHREAD_MUTEX_ULOCK(&nslbCb->mutex);
    return retVal;
}
