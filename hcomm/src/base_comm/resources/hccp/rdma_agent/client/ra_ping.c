/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include "hccp_ping.h"
#include "ra.h"
#include "ra_client_host.h"
#include "user_log.h"
#include "config_log.h"
#include "ra_hdc_ping.h"
#include "ra_rs_comm.h"
#include "ra_ping.h"
#include "ra_rs_err.h"

struct RaPingOps gRaHdcPingOps = {
    .raPingInit = RaHdcPingInit,
    .raPingTargetAdd = RaHdcPingTargetAdd,
    .raPingTaskStart = RaHdcPingTaskStart,
    .raPingGetResults = RaHdcPingGetResults,
    .raPingTargetDel = RaHdcPingTargetDel,
    .raPingTaskStop = RaHdcPingTaskStop,
    .raPingDeinit = RaHdcPingDeinit,
};

STATIC int RaUdevInitCheck(unsigned int phyId, void *pingHandle)
{
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[check][ra_ping_init]phyId(%u) is invalid! "
                 "it must be greater than or equal to 0 and less than %d!",
            phyId, RA_MAX_PHY_ID_NUM),
        -EINVAL);
    CHK_PRT_RETURN(pingHandle == NULL, hccp_err("[check][ra_ping_init]phyId(%u) ping_handle is null!", phyId), -EINVAL);
    return 0;
}

STATIC int RaPingInitGetHandle(struct PingInitAttr *initAttr, struct PingInitInfo *initInfo,
    struct RaPingHandle *pingHandle)
{
    char localIp[MAX_IP_LEN] = {0};
    int ret;

    CHK_PRT_RETURN(initAttr == NULL || initInfo == NULL, hccp_err("[init][ra_ping]initAttr or init_info is NULL"),
        -EINVAL);
    CHK_PRT_RETURN(initAttr->mode != NETWORK_OFFLINE, hccp_err("[init][ra_ping]mode:%d do not support", initAttr->mode),
        -EINVAL);

    if (initAttr->protocol == PROTOCOL_RDMA) {
        CHK_PRT_RETURN(initAttr->commInfo.rdma.udpSport > MAX_PORT_NUM,
            hccp_err("[init][ra_ping]udp_sport %u invalid", initAttr->commInfo.rdma.udpSport), -EINVAL);
        ret = RaRdevInitCheck(initAttr->mode, initAttr->dev.rdma, localIp, MAX_IP_LEN, pingHandle);
        CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_ping]ra_rdev_init_check failed, ret(%d)", ret), -EINVAL);

        pingHandle->phyId = initAttr->dev.rdma.phyId;
        hccp_run_info("Input parameters: phyId[%u], nicPosition[%d] family[%d] ip[%s] bufferSize[0x%x]",
            initAttr->dev.rdma.phyId, initAttr->mode, initAttr->dev.rdma.family, localIp, initAttr->bufferSize);
    } else if (initAttr->protocol == PROTOCOL_UDMA) {
        ret = RaUdevInitCheck(initAttr->ub.phyId, pingHandle);
        CHK_PRT_RETURN(ret != 0, hccp_err("[init][ra_ping]ra_ub_dev_init_check failed, ret(%d)", ret), -EINVAL);

        pingHandle->phyId = initAttr->ub.phyId;
        hccp_run_info("Input parameters: phyId[%u], nicPosition[%d], eidIndex[%u] bufferSize[0x%x]", initAttr->ub.phyId,
            initAttr->mode, initAttr->dev.ub.eidIndex, initAttr->bufferSize);
    } else {
        hccp_err("[init][ra_ping]protocol:%d do not support", initAttr->protocol);
        return -ENOTSUPP;
    }
    pingHandle->protocol = initAttr->protocol;

    pingHandle->pingOps = &gRaHdcPingOps;
    CHK_PRT_RETURN(pingHandle->pingOps->raPingInit == NULL, hccp_err("[init][ra_ping]ra_ping_init is NULL"), -EINVAL);
    CHK_PRT_RETURN(initAttr->bufferSize == 0 || initAttr->bufferSize % (unsigned int)RA_RS_4K_PAGE_SIZE != 0,
        hccp_err("[init][ra_ping]initAttr->buffer_size:0x%x not 0x%xB aligned", initAttr->bufferSize,
            (unsigned int)RA_RS_4K_PAGE_SIZE),
        -EINVAL);
    pingHandle->bufferSize = initAttr->bufferSize;

    ret = pthread_mutex_init(&pingHandle->mutex, NULL);
    CHK_PRT_RETURN(ret, hccp_err("[init][ra_ping]init mutex failed, ret:%d", ret), ret);

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPingInit(struct PingInitAttr *initAttr, struct PingInitInfo *initInfo, void **pingHandle)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    int ret;

    CHK_PRT_RETURN(pingHandle == NULL, hccp_err("[init][ra_ping]ping_handle is NULL"), -EINVAL);

    pingHandleTmp = calloc(1, sizeof(struct RaPingHandle));
    CHK_PRT_RETURN(pingHandleTmp == NULL, hccp_err("[init][ra_ping]calloc for ping_handle failed"),
        ConverReturnCode(HCCP_INIT, -ENOMEM));

    ret = RaPingInitGetHandle(initAttr, initInfo, pingHandleTmp);
    if (ret != 0) {
        hccp_err("[init][ra_ping]ra_ping_init_get_handle failed, ret(%d)", ret);
        goto err;
    }

    ret = pingHandleTmp->pingOps->raPingInit(pingHandleTmp, initAttr, initInfo);
    if (ret != 0) {
        (void)pthread_mutex_destroy(&pingHandleTmp->mutex);
        goto err;
    }

    *pingHandle = (void *)pingHandleTmp;

    return 0;

err:
    free(pingHandleTmp);
    pingHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}

HCCP_ATTRI_VISI_DEF int RaPingTargetAdd(void *pingHandle, struct PingTargetInfo target[], uint32_t num)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(pingHandle == NULL || target == NULL || num == 0,
        hccp_err("[add][ra_ping]ping_handle or target "
                 "is NULL, or num is 0"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    pingHandleTmp = (struct RaPingHandle *)pingHandle;
    CHK_PRT_RETURN(pingHandleTmp->pingOps == NULL || pingHandleTmp->pingOps->raPingTargetAdd == NULL,
        hccp_err("[add][ra_ping]ping_ops or ra_ping_target_add is NULL"), ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = pingHandleTmp->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[add][ra_ping]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));
    CHK_PRT_RETURN(pingHandleTmp->targetCnt + num < num,
        hccp_err("[add][ra_ping]pingHandleTmp->targetCnt + num is out of range"), ConverReturnCode(RDMA_OP, -EINVAL));

    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    // disallow add target when task is running
    if (pingHandleTmp->taskCnt != 0) {
        hccp_err("[add][ra_ping]task_cnt:%u != 0 invalid, task already running", pingHandleTmp->taskCnt);
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -EEXIST);
    }
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);

    ret = pingHandleTmp->pingOps->raPingTargetAdd(pingHandleTmp, target, num);
    if (ret != 0) {
        return ConverReturnCode(RDMA_OP, ret);
    }

    // increase target cnt
    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    pingHandleTmp->targetCnt += num;
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPingTaskStart(void *pingHandle, struct PingTaskAttr *attr)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(pingHandle == NULL || attr == NULL, hccp_err("[start][ra_ping]ping_handle is NULL or attr is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    CHK_PRT_RETURN(attr->packetCnt == 0 || attr->packetInterval == 0 || attr->timeoutInterval == 0,
        hccp_err("[start][ra_ping]packet_cnt:%u or packet_interval:%u or timeout_interval:%u is 0", attr->packetCnt,
            attr->packetInterval, attr->timeoutInterval),
        ConverReturnCode(RDMA_OP, -EINVAL));

    pingHandleTmp = (struct RaPingHandle *)pingHandle;
    CHK_PRT_RETURN(pingHandleTmp->pingOps == NULL || pingHandleTmp->pingOps->raPingTaskStart == NULL,
        hccp_err("[start][ra_ping]ping_ops is NULL or ping_ops->ra_ping_task_start is NULL"),
        ConverReturnCode(RDMA_OP, -EINVAL));

    phyId = pingHandleTmp->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[start][ra_ping]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    // disallow multi task running or no target to start
    if (pingHandleTmp->taskCnt != 0) {
        hccp_warn_rma("[start][ra_ping]task_cnt:%u != 0 invalid, task already running", pingHandleTmp->taskCnt);
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -EEXIST);
    }
    if (pingHandleTmp->targetCnt == 0) {
        hccp_warn_rma("[start][ra_ping]target_cnt is 0 invalid, no target exist");
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -ENODEV);
    }

    // increase task cnt to avoid lock contention, trigger task running
    pingHandleTmp->taskCnt++;
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);

    hccp_run_info("Input parameters: phyId[%u], packetCnt[%u] interval[%u] timeoutInterval[%u], targetCnt[%u]", phyId,
        attr->packetCnt, attr->packetInterval, attr->timeoutInterval, pingHandleTmp->targetCnt);

    ret = pingHandleTmp->pingOps->raPingTaskStart(pingHandleTmp, attr);
    if (ret != 0) {
        // trigger failed, decrease task cnt
        hccp_err("[start][ra_ping]ping_task_start failed, ret(%d)", ret);
        RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
        pingHandleTmp->taskCnt--;
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, ret);
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPingGetResults(void *pingHandle, struct PingTargetResult target[], uint32_t *num)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    if (pingHandle == NULL || target == NULL || num == NULL || *num == 0) {
        hccp_err("[get][ra_ping]ping_handle is NULL or target is NULL or num is NULL or *num is 0");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    pingHandleTmp = (struct RaPingHandle *)pingHandle;
    if (pingHandleTmp->pingOps == NULL || pingHandleTmp->pingOps->raPingGetResults == NULL) {
        hccp_err("[get][ra_ping]ping_ops is NULL or ping_ops->ra_ping_get_results is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    phyId = pingHandleTmp->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[get][ra_ping]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    // num invalid, bigger than target exist
    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    if (pingHandleTmp->targetCnt < *num) {
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        hccp_err("[get][ra_ping]target_cnt:%u < num:%u, invalid", pingHandleTmp->targetCnt, *num);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);

    ret = pingHandleTmp->pingOps->raPingGetResults(pingHandleTmp, target, num);
    return ConverReturnCode(RDMA_OP, ret);
}

HCCP_ATTRI_VISI_DEF int RaPingTargetDel(void *pingHandle, struct PingTargetCommInfo target[], uint32_t num)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    if (pingHandle == NULL || target == NULL || num == 0) {
        hccp_err("[del][ra_ping]ping_handle is NULL or target is NULL or num:%u is 0", num);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    pingHandleTmp = (struct RaPingHandle *)pingHandle;
    if (pingHandleTmp->pingOps == NULL || pingHandleTmp->pingOps->raPingTargetDel == NULL) {
        hccp_err("[del][ra_ping]ping_ops is NULL or ping_ops->ra_ping_target_del is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    // disallow del target when task is running
    if (pingHandleTmp->taskCnt != 0) {
        hccp_err("[del][ra_ping]task_cnt:%u != 0 invalid, task already running", pingHandleTmp->taskCnt);
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -EEXIST);
    }
    // num invalid, bigger than target exist
    if (pingHandleTmp->targetCnt < num) {
        hccp_err("[del][ra_ping]target_cnt:%u < num:%u, invalid", pingHandleTmp->targetCnt, num);
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }
    // decrease target cnt to avoid lock contention, trigger target del
    pingHandleTmp->targetCnt -= num;
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);

    phyId = pingHandleTmp->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[del][ra_ping]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    ret = pingHandleTmp->pingOps->raPingTargetDel(pingHandleTmp, target, num);
    if (ret != 0) {
        // trigger del failed, increase target cnt
        RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
        pingHandleTmp->targetCnt += num;
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, ret);
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPingTaskStop(void *pingHandle)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    unsigned int phyId;
    int ret;

    if (pingHandle == NULL) {
        hccp_err("[stop][ra_ping]ping_handle is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    pingHandleTmp = (struct RaPingHandle *)pingHandle;
    if (pingHandleTmp->pingOps == NULL || pingHandleTmp->pingOps->raPingTaskStop == NULL) {
        hccp_err("[stop][ra_ping]ping_ops is NULL or ping_ops->ra_ping_task_stop is NULL");
        return ConverReturnCode(RDMA_OP, -EINVAL);
    }

    phyId = pingHandleTmp->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[stop][ra_ping]phyId(%u) must be less than %d!", phyId, RA_MAX_PHY_ID_NUM),
        ConverReturnCode(RDMA_OP, -EINVAL));

    // no task to stop
    RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
    if (pingHandleTmp->taskCnt == 0) {
        hccp_warn_rma("[stop][ra_ping]task_cnt is 0 invalid, no task running");
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, -ENODEV);
    }
    // decrease task cnt to avoid lock contention, trigger task stop
    pingHandleTmp->taskCnt--;
    RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);

    hccp_run_info("Input parameters: phyId[%u], targetCnt[%u]", phyId, pingHandleTmp->targetCnt);

    ret = pingHandleTmp->pingOps->raPingTaskStop(pingHandleTmp);
    if (ret != 0) {
        // trigger failed, increase task cnt
        RA_PTHREAD_MUTEX_LOCK(&pingHandleTmp->mutex);
        pingHandleTmp->taskCnt++;
        RA_PTHREAD_MUTEX_UNLOCK(&pingHandleTmp->mutex);
        return ConverReturnCode(RDMA_OP, ret);
    }

    return 0;
}

STATIC int RaPingDeinitParaCheck(struct RaPingHandle *pingHandle)
{
    char localIp[MAX_IP_LEN] = {0};
    union PingDev devInfo;
    unsigned int phyId;
    int ret;

    CHK_PRT_RETURN(pingHandle->pingOps == NULL || pingHandle->pingOps->raPingDeinit == NULL,
        hccp_err("[deinit][ra_ping]ping_ops is NULL or ra_ping_deinit is NULL"), -EINVAL);

    phyId = pingHandle->phyId;
    CHK_PRT_RETURN(phyId >= RA_MAX_PHY_ID_NUM,
        hccp_err("[deinit][ra_ping]phyId(%u) must be smaller than %u", phyId, RA_MAX_PHY_ID_NUM), -EINVAL);

    devInfo = pingHandle->dev;
    if (pingHandle->protocol == PROTOCOL_RDMA) {
        ret = RaInetPton(devInfo.rdma.family, devInfo.rdma.localIp, localIp, MAX_IP_LEN);
        CHK_PRT_RETURN(ret != 0, hccp_err("[deinit][ra_ping]ra_inet_pton for local_ip failed, ret(%d)", ret), ret);
        hccp_run_info("Input parameters: phyId[%u] dev_index[%u] family[%d] local_ip[%s] target_cnt[%u] task_cnt[%u]",
            phyId, pingHandle->devIndex, devInfo.rdma.family, localIp, pingHandle->targetCnt, pingHandle->taskCnt);
    } else if (pingHandle->protocol == PROTOCOL_UDMA) {
        hccp_run_info("Input parameters: eid_index[%u] eid[0x%016llx%016llx] target_cnt[%u] task_cnt[%u]",
            devInfo.ub.eidIndex, devInfo.ub.eid.in6.subnetPrefix, devInfo.ub.eid.in6.interfaceId, pingHandle->targetCnt,
            pingHandle->taskCnt);
    } else {
        hccp_err("[deinit][ra_ping]protocol:%d do not support", pingHandle->protocol);
        return -ENOTSUPP;
    }

    return 0;
}

HCCP_ATTRI_VISI_DEF int RaPingDeinit(void *pingHandle)
{
    struct RaPingHandle *pingHandleTmp = NULL;
    int ret;

    if (pingHandle == NULL) {
        hccp_err("[deinit][ra_ping]ping_handle is NULL");
        return ConverReturnCode(HCCP_INIT, -EINVAL);
    }
    pingHandleTmp = (struct RaPingHandle *)pingHandle;

    ret = RaPingDeinitParaCheck(pingHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_ping]ra_ping_deinit_para_check failed, ret(%d)", ret);
        return ConverReturnCode(HCCP_INIT, ret);
    }

    ret = pingHandleTmp->pingOps->raPingDeinit(pingHandleTmp);
    if (ret != 0) {
        hccp_err("[deinit][ra_ping]ra_ping_deinit failed, ret(%d)", ret);
    }

    pingHandleTmp->pingOps = NULL;
    (void)pthread_mutex_destroy(&pingHandleTmp->mutex);
    free(pingHandleTmp);
    pingHandleTmp = NULL;
    return ConverReturnCode(HCCP_INIT, ret);
}
