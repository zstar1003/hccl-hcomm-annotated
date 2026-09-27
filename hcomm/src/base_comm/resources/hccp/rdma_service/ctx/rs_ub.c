/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <string.h>
#include <urma_opcode.h>
#include <udma_u_ctl.h>
#include "securec.h"
#include "user_log.h"
#include "config_log.h"
#include "dl_urma_function.h"
#include "ra_rs_err.h"
#include "ra_rs_ctx.h"
#include "rs_inner.h"
#include "rs_epoll.h"
#include "rs_ctx_inner.h"
#include "rs_ctx.h"
#include "rs_ub_jetty.h"
#include "rs_ub_jfc.h"
#include "rs_ub.h"

int RsUbGetDevEidInfoNum(unsigned int phyId, unsigned int *num)
{
    (void)phyId;
    urma_eid_info_t *eidList = NULL;
    urma_device_t **devList = NULL;
    unsigned int totalNum = 0;
    unsigned int eidNum = 0;
    int devNum = 0;
    int ret = 0;
    int i = 0;

    ret = RsUbApiInit();
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_api_init failed, ret:%d", ret), ret);

    devList = RsUrmaGetDeviceList(&devNum);
    if (devList == NULL) {
        hccp_err("rs_urma_get_device_list failed, errno:%d", errno);
        ret = -EINVAL;
        goto ub_api_deinit;
    }

    for (i = 0; i < devNum; i++) {
        eidList = RsUrmaGetEidList(devList[i], &eidNum);
        // normal case, should continue to get eid_list from the rest of device
        if (eidList == NULL) {
            hccp_warn_rma("rs_urma_get_eid_list i=%u unsuccessful, eidList is NULL, errno:%d", i, errno);
            continue;
        }

        totalNum += eidNum;
        RsUrmaFreeEidList(eidList);
    }

    *num = totalNum;

    RsUrmaFreeDeviceList(devList);
ub_api_deinit:
    RsUbApiDeinit();
    return ret;
}

STATIC int RsUbCreateCtx(urma_device_t *urmaDev, unsigned int eidIndex, urma_context_t **urmaCtx)
{
    *urmaCtx = RsUrmaCreateContext(urmaDev, eidIndex);
    CHK_PRT_RETURN(*urmaCtx == NULL, hccp_err("rs_urma_create_context failed! errno:%d, eidIndex:%u", errno, eidIndex),
        -ENODEV);

    return 0;
}

int RsUbGetUeInfo(urma_context_t *urmaCtx, struct DevBaseAttr *devAttr)
{
    (void)urmaCtx;
    (void)devAttr;
#ifdef CUSTOM_INTERFACE
    struct udma_u_ue_info ueInfo = {0};
    urma_user_ctl_out_t out = {0};
    urma_user_ctl_in_t in = {0};
    int ret;

    in.opcode = UDMA_U_USER_CTL_QUERY_UE_INFO;
    out.addr = (uint64_t)(uintptr_t)&ueInfo;
    out.len = sizeof(struct udma_u_ue_info);
    ret = RsUrmaUserCtl(urmaCtx, &in, &out);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_user_ctl query ue info failed! ret:%d errno:%d", ret, errno), -EOPENSRC);

    devAttr->ub.dieId = ueInfo.die_id;
    devAttr->ub.chipId = ueInfo.chip_id;
    devAttr->ub.funcId = ueInfo.ue_id;
    hccp_info_rma("func_id:%u, chipId:%u, dieId:%u", devAttr->ub.funcId, devAttr->ub.chipId, devAttr->ub.dieId);
#endif
    return 0;
}

STATIC int RsUbFillDevEidInfoList(struct HccpDevEidInfo *totalList, unsigned int index, urma_device_t *device,
    urma_eid_info_t *eidInfo)
{
    struct DevBaseAttr devAttr = {0};
    urma_context_t *urmaCtx = NULL;
    int ret = 0;

    totalList[index].eidIndex = eidInfo->eid_index;
    totalList[index].type = device->type;

    (void)memcpy_s(totalList[index].eid.raw, sizeof(totalList[index].eid.raw), eidInfo->eid.raw,
        sizeof(eidInfo->eid.raw));

    ret = strcpy_s(totalList[index].name, DEV_EID_INFO_MAX_NAME, device->name);
    CHK_PRT_RETURN(ret != 0, hccp_err("strcpy device name failed, ret:%d", ret), -ESAFEFUNC);

    ret = RsUbCreateCtx(device, eidInfo->eid_index, &urmaCtx);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_create_ctx failed, ret:%d, eidIndex:%u", ret, eidInfo->eid_index), ret);

    ret = RsUbGetUeInfo(urmaCtx, &devAttr);
    if (ret != 0) {
        hccp_err("rs_ub_get_ue_info failed, ret:%d", ret);
        goto free_ctx;
    }

    totalList[index].dieId = devAttr.ub.dieId;
    totalList[index].chipId = devAttr.ub.chipId;
    totalList[index].funcId = devAttr.ub.funcId;

free_ctx:
    (void)RsUrmaDeleteContext(urmaCtx);
    return ret;
}

STATIC int RsUbFillInfoByEidList(urma_device_t *currDev, unsigned int *index, struct HccpDevEidInfo *totalList,
    unsigned int totalNum)
{
    urma_eid_info_t *eidList = NULL;
    urma_device_attr_t attr = {0};
    unsigned int eidNum = 0;
    unsigned int j;
    int ret = 0;

    ret = RsUrmaQueryDevice(currDev, &attr);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsUrmaQueryDevice failed, ret:%d, errno:%d", ret, errno), -EOPENSRC);

    eidList = RsUrmaGetEidList(currDev, &eidNum);
    // normal case, should continue to get eid_list from the rest of device
    if (eidList == NULL) {
        hccp_warn_rma("rs_urma_get_eid_list unsuccessful, eidList is NULL, errno:%d", errno);
        return 0;
    }

    for (j = 0; j < eidNum; j++) {
        if (*index >= totalNum) {
            hccp_err("index out of range, index:%u, totalNum:%u", *index, totalNum);
            ret = -EINVAL;
            goto free_eid_list;
        }
        ret = RsUbFillDevEidInfoList(totalList, *index, currDev, &eidList[j]);
        if (ret != 0) {
            hccp_err("rs_ub_fill_dev_eid_info_list failed, index:%u, ret:%d", *index, ret);
            goto free_eid_list;
        }
        totalList[*index].devFeature = attr.dev_cap.feature.value;
        *index += 1;
    }

free_eid_list:
    RsUrmaFreeEidList(eidList);
    return ret;
}

int RsUbGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
    unsigned int count)
{
    struct HccpDevEidInfo *totalList = NULL;
    urma_device_t **devList = NULL;
    unsigned int totalNum = 0;
    unsigned int index = 0;
    int devNum, ret, i;

    ret = RsUbApiInit();
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_api_init failed, ret:%d", ret), ret);

    ret = RsUbGetDevEidInfoNum(phyId, &totalNum);
    if ((ret != 0) || (startIndex > UINT_MAX - count) || (startIndex + count > totalNum)) {
        hccp_err("rs_ub_get_dev_eid_info_num failed ret:%d data size exceeds the max offset range "
                 "or start_index:%u + count:%u > total_num:%u",
            ret, startIndex, count, totalNum);
        ret = -EINVAL;
        goto ub_api_deinit;
    }

    totalList = calloc(totalNum, sizeof(struct HccpDevEidInfo));
    if (totalList == NULL) {
        hccp_err("calloc total_list failed, errno:%d", errno);
        ret = -ENOMEM;
        goto ub_api_deinit;
    }

    devList = RsUrmaGetDeviceList(&devNum);
    if (devList == NULL) {
        hccp_err("rs_urma_get_device_list failed, errno: %d", errno);
        ret = -EINVAL;
        goto free_total_list;
    }

    for (i = 0; i < devNum; i++) {
        ret = RsUbFillInfoByEidList(devList[i], &index, totalList, totalNum);
        if (ret != 0) {
            goto free_dev_list;
        }
    }

    // start_index + count <= total_num make sure count is valid
    (void)memcpy_s(infoList, sizeof(struct HccpDevEidInfo) * count, totalList + startIndex,
        sizeof(struct HccpDevEidInfo) * count);

free_dev_list:
    RsUrmaFreeDeviceList(devList);
free_total_list:
    free(totalList);
    totalList = NULL;
ub_api_deinit:
    RsUbApiDeinit();
    return ret;
}

int RsUbGetDevCb(struct rs_cb *rscb, unsigned int devIndex, struct RsUbDevCb **devCb)
{
    struct RsUbDevCb *devCbCurr = NULL;
    struct RsUbDevCb *devCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(devCbCurr, devCbNext, &rscb->udevList, list, struct RsUbDevCb);
    for (; (&devCbCurr->list) != &rscb->udevList;
         devCbCurr = devCbNext, devCbNext = list_entry(devCbNext->list.next, struct RsUbDevCb, list)) {
        if (devCbCurr->index == devIndex) {
            *devCb = devCbCurr;
            return 0;
        }
    }

    *devCb = NULL;
    hccp_err("dev_cb for devIndex:0x%x is not available!", devIndex);
    return -ENODEV;
}

STATIC int RsUbGetDevAttr(struct RsUbDevCb *devCb, struct DevBaseAttr *devAttr, unsigned int *devIndex)
{
    urma_device_attr_t attr = {0};
    int ret;
    int i;

    ret = RsUrmaQueryDevice(devCb->urmaDev, &attr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_query_device failed ret:%d", ret), -EOPENSRC);

    devAttr->sqMaxDepth = attr.dev_cap.max_jfs_depth;
    devAttr->rqMaxDepth = attr.dev_cap.max_jfr_depth;
    devAttr->sqMaxSge = attr.dev_cap.max_jfs_sge;
    devAttr->rqMaxSge = attr.dev_cap.max_jfr_sge;
    devAttr->ub.maxJfsInlineLen = attr.dev_cap.max_jfs_inline_len;
    devAttr->ub.maxJfsRsge = attr.dev_cap.max_jfs_rsge;
    devAttr->maxReadSize = attr.dev_cap.max_read_size;
    devAttr->maxWriteSize = attr.dev_cap.max_write_size;
    devAttr->maxMsgSize = attr.dev_cap.max_msg_size;
    devAttr->ub.rmTpCap.value = attr.dev_cap.rm_tp_cap.value;
    devAttr->ub.rcTpCap.value = attr.dev_cap.rc_tp_cap.value;
    devAttr->ub.umTpCap.value = attr.dev_cap.um_tp_cap.value;
    devAttr->ub.tpFeat.value = attr.dev_cap.tp_feature.value;
    for (i = 0; i < MAX_PRIORITY_CNT && i < URMA_MAX_PRIORITY_CNT; i++) {
        devAttr->ub.priorityInfo[i].SL = attr.dev_cap.priority_info[i].SL;
        devAttr->ub.priorityInfo[i].tpType.value = attr.dev_cap.priority_info[i].tp_type.value;
    }
    ret = RsUbGetUeInfo(devCb->urmaCtx, devAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_get_ue_info failed ret:%d", ret), -EOPENSRC);

    (void)memcpy_s(&devCb->devAttr, sizeof(struct DevBaseAttr), devAttr, sizeof(struct DevBaseAttr));
    devCb->rscb->devCnt++;
    *devIndex = RsGenerateDevIndex(devCb->rscb->devCnt, devAttr->ub.dieId, devAttr->ub.funcId);
    devCb->index = *devIndex;

    hccp_info_rma("max_jetty:%u, maxJfsInlineLen:%u, sqMaxDepth:%u, rqMaxDepth:%u, sqMaxSge:%u, rqMaxSge:%u "
                  "maxReadSize:%u maxWriteSize:%u maxMsgSize:%llu",
        attr.dev_cap.max_jetty, devAttr->ub.maxJfsInlineLen, devAttr->sqMaxDepth, devAttr->rqMaxDepth,
        devAttr->sqMaxSge, devAttr->rqMaxSge, devAttr->maxReadSize, devAttr->maxWriteSize, devAttr->maxMsgSize);
    return 0;
}

STATIC int RsUbDevShareJfrInit(struct RsUbDevCb *devCb)
{
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfr_cfg_t jfrCfg = {0};
    int randNum = 0;
    int ret;

    jfcCfg.depth = CQ_DEFAULT_MIN_RECV_DEPTH;
    devCb->shareJfc = RsUrmaCreateJfc(devCb->urmaCtx, &jfcCfg);
    CHK_PRT_RETURN(devCb->shareJfc == NULL, hccp_err("RsUrmaCreateJfc failed, errno:%d", errno), -ENOMEM);

    jfrCfg.depth = CQ_DEFAULT_MIN_RECV_DEPTH;
    jfrCfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    jfrCfg.trans_mode = URMA_TM_RM;
    jfrCfg.max_sge = QP_DEFAULT_MIN_CAP_RECV_SGE;
    jfrCfg.min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER;
    jfrCfg.jfc = devCb->shareJfc;
    ret = RsDrvGetRandomNum(&randNum);
    if (ret != 0) {
        hccp_err("RsDrvGetRandomNum failed, ret:%d", ret);
        goto delete_jfc;
    }
    jfrCfg.token_value.token = (uint32_t)randNum;
    devCb->shareJfr = RsUrmaCreateJfr(devCb->urmaCtx, &jfrCfg);
    if (devCb->shareJfr == NULL) {
        hccp_err("RsUrmaCreateJfr failed, errno:%d", errno);
        ret = -ENOMEM;
        goto delete_jfc;
    }

    return 0;

delete_jfc:
    RsUrmaDeleteJfc(devCb->shareJfc);
    devCb->shareJfc = NULL;
    return ret;
}

STATIC void RsUbDevShareJfrDeinit(struct RsUbDevCb *devCb)
{
    (void)RsUrmaDeleteJfr(devCb->shareJfr);
    devCb->shareJfr = NULL;
    (void)RsUrmaDeleteJfc(devCb->shareJfc);
    devCb->shareJfc = NULL;
}

STATIC int RsUbDevCbInit(struct CtxInitAttr *attr, struct RsUbDevCb *devCb, struct rs_cb *rscb, unsigned int *devIndex,
    struct DevBaseAttr *devAttr)
{
    int ret;

    devCb->rscb = rscb;
    devCb->phyId = attr->phyId;
    devCb->eidIndex = attr->ub.eidIndex;
    devCb->eid = attr->ub.eid;

    ret = pthread_mutex_init(&devCb->mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("mutex_init failed ret:%d", ret), -ESYSFUNC);

    RS_INIT_LIST_HEAD(&devCb->asyncEventList);
    RS_INIT_LIST_HEAD(&devCb->jfceList);
    RS_INIT_LIST_HEAD(&devCb->jfcList);
    RS_INIT_LIST_HEAD(&devCb->jettyList);
    RS_INIT_LIST_HEAD(&devCb->rjettyList);
    RS_INIT_LIST_HEAD(&devCb->tokenIdList);
    RS_INIT_LIST_HEAD(&devCb->lsegList);
    RS_INIT_LIST_HEAD(&devCb->rsegList);

    ret = RsUbCreateCtx(devCb->urmaDev, attr->ub.eidIndex, &(devCb->urmaCtx));
    if (ret != 0) {
        hccp_err("rs_ub_create_ctx failed, ret:%d", ret);
        goto destroy_mutex;
    }

    ret = RsUbDevShareJfrInit(devCb);
    if (ret != 0) {
        hccp_err("RsUbDevShareJfrInit failed, ret:%d", ret);
        goto close_dev;
    }

    ret = RsEpollCtl(devCb->rscb->connCb.epollfd, EPOLL_CTL_ADD, devCb->urmaCtx->async_fd, EPOLLIN | EPOLLRDHUP);
    if (ret != 0) {
        hccp_err("rs_epoll_ctl failed, ret:%d fd:%d", ret, devCb->urmaCtx->async_fd);
        goto free_share_jfr;
    }

    ret = RsUbGetDevAttr(devCb, devAttr, devIndex);
    if (ret != 0) {
        hccp_err("rs_ub_get_dev_attr failed, ret:%d", ret);
        goto epoll_del;
    }

    return 0;

epoll_del:
    (void)RsEpollCtl(devCb->rscb->connCb.epollfd, EPOLL_CTL_DEL, devCb->urmaCtx->async_fd, EPOLLIN | EPOLLRDHUP);
free_share_jfr:
    RsUbDevShareJfrDeinit(devCb);
close_dev:
    (void)RsUrmaDeleteContext(devCb->urmaCtx);
destroy_mutex:
    pthread_mutex_destroy(&devCb->mutex);
    return ret;
}

int RsUbCtxInit(struct rs_cb *rsCb, struct CtxInitAttr *attr, unsigned int *devIndex, struct DevBaseAttr *devAttr)
{
    struct RsUbDevCb *devCb = NULL;
    urma_eid_t eid;
    int ret;

    ret = RsUbApiInit();
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_api_init failed, ret:%d", ret), ret);

    devCb = calloc(1, sizeof(struct RsUbDevCb));
    if (devCb == NULL) {
        hccp_err("calloc for dev_cb failed, errno:%d", errno);
        ret = -ENOMEM;
        goto ub_api_deinit;
    }

    (void)memcpy_s(eid.raw, sizeof(eid.raw), attr->ub.eid.raw, sizeof(attr->ub.eid.raw));

    devCb->urmaDev = RsUrmaGetDeviceByEid(eid, URMA_TRANSPORT_UB);
    if (devCb->urmaDev == NULL) {
        hccp_err("rs_urma_get_device_by_eid failed, urmaDev is NULL, errno:%d eid:%016llx:%016llx", errno,
            (unsigned long long)be64toh(eid.in6.subnet_prefix), (unsigned long long)be64toh(eid.in6.interface_id));
        ret = -EINVAL;
        goto free_dev_cb;
    }

    ret = RsSensorNodeRegister(attr->phyId, rsCb);
    if (ret != 0) {
        hccp_err("rs_sensor_node_register failed, phyId(%u), ret(%d)", attr->phyId, ret);
        goto free_dev_cb;
    }

    ret = RsUbDevCbInit(attr, devCb, rsCb, devIndex, devAttr);
    if (ret != 0) {
        RsSensorNodeUnregister(devCb->rscb);
        hccp_err("rs_ub_dev_cb_init failed ret:%d, eidIndex:%u eid:%016llx:%016llx", ret, attr->ub.eidIndex,
            (unsigned long long)be64toh(eid.in6.subnet_prefix), (unsigned long long)be64toh(eid.in6.interface_id));
        goto free_dev_cb;
    }

    RS_PTHREAD_MUTEX_LOCK(&rsCb->mutex);
    RsListAddTail(&devCb->list, &rsCb->udevList);
    RS_PTHREAD_MUTEX_ULOCK(&rsCb->mutex);

    hccp_run_info("phy_id:%u, eidIndex:%u, devIndex:0x%x init success", attr->phyId, attr->ub.eidIndex, *devIndex);

    return 0;

free_dev_cb:
    free(devCb);
    devCb = NULL;
ub_api_deinit:
    RsUbApiDeinit();
    return ret;
}

int RsUbGetEidByIp(struct RsUbDevCb *devCb, struct IpInfo ip[], union HccpEid eid[], unsigned int *num)
{
    urma_net_addr_t netAddr = {0};
    unsigned int ipNum = *num;
    urma_eid_t urmaEid = {0};
    unsigned int i;
    int ret = 0;

    *num = 0;
    for (i = 0; i < ipNum; i++) {
        netAddr.sin_family = (sa_family_t)ip[i].family;
        if (netAddr.sin_family == AF_INET) {
            netAddr.in4 = ip[i].ip.addr;
        } else {
            netAddr.in6 = ip[i].ip.addr6;
        }
        ret = RsUrmaGetEidByIp(devCb->urmaCtx, &netAddr, &urmaEid);
        CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_get_eid_by_ip failed, ret:%d devIndex:0x%x", ret, devCb->index),
            -EOPENSRC);
        (void)memcpy_s(eid[i].raw, sizeof(union HccpEid), urmaEid.raw, sizeof(urma_eid_t));
        (*num)++;
    }

    return ret;
}

int RsUbGetIpByEid(struct RsUbDevCb *devCb, union HccpEid eid[], struct IpInfo ip[], unsigned int *num)
{
    urma_net_addr_t netAddr = {0};
    unsigned int eidNum = *num;
    urma_eid_t urmaEid = {0};
    unsigned int i;
    int ret = 0;

    *num = 0;
    for (i = 0; i < eidNum; i++) {
        (void)memcpy_s(urmaEid.raw, sizeof(urma_eid_t), eid[i].raw, sizeof(union HccpEid));
        ret = RsUrmaGetIpByEid(devCb->urmaCtx, &urmaEid, &netAddr);
        CHK_PRT_RETURN(ret != 0, hccp_err("RsUrmaGetIpByEid failed, ret:%d devIndex:0x%x", ret, devCb->index), ret);
        ip[i].family = (int)netAddr.sin_family;
        if (netAddr.sin_family == AF_INET) {
            ip[i].ip.addr = netAddr.in4;
        } else {
            ip[i].ip.addr6 = netAddr.in6;
        }
        (*num)++;
    }

    return ret;
}

STATIC int RsUbGetJfcCb(struct RsUbDevCb *devCb, unsigned long long addr, struct RsCtxJfcCb **jfcCb)
{
    struct RsCtxJfcCb **tempJfcCb = jfcCb;
    struct RsCtxJfcCb *jfcCbCurr = NULL;
    struct RsCtxJfcCb *jfcCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(jfcCbCurr, jfcCbNext, &devCb->jfcList, list, struct RsCtxJfcCb);
    for (; (&jfcCbCurr->list) != &devCb->jfcList;
         jfcCbCurr = jfcCbNext, jfcCbNext = list_entry(jfcCbNext->list.next, struct RsCtxJfcCb, list)) {
        if (jfcCbCurr->jfcAddr == addr) {
            *tempJfcCb = jfcCbCurr;
            return 0;
        }
    }

    *tempJfcCb = NULL;
    hccp_err("jfc_cb for jfc_addr:0x%llx is not available!", addr);

    return -ENODEV;
}

STATIC int RsUbFreeJfcCb(struct RsUbDevCb *devCb, struct RsCtxJfcCb *jfcCb)
{
    urma_jfc_t *urmaJfc = NULL;
    int ret = 0;

    RsListDel(&jfcCb->list);
    devCb->jfcCnt--;

    if (jfcCb->jfcType == JFC_MODE_STARS_POLL || jfcCb->jfcType == JFC_MODE_CCU_POLL ||
        jfcCb->jfcType == JFC_MODE_USER_CTL_NORMAL) {
        (void)RsUbDeleteJfcExt(devCb, jfcCb);
        hccp_info_rma("[deinit][rs_jfc]destroy success, dev jfcCnt:%u", devCb->jfcCnt);
    } else if (jfcCb->jfcType == JFC_MODE_NORMAL) {
        urmaJfc = (urma_jfc_t *)(uintptr_t)jfcCb->jfcAddr;
        (void)RsUrmaDeleteJfc(urmaJfc);
        hccp_info_rma("[deinit][rs_jfc]destroy success, dev jfcCnt:%u", devCb->jfcCnt);
    } else {
        hccp_err("jfc_type:%d is invalid, not support!", jfcCb->jfcType);
        ret = -EINVAL;
    }

    free(jfcCb);
    jfcCb = NULL;
    return ret;
}

int RsUbCtxJfcDestroy(struct RsUbDevCb *devCb, unsigned long long addr)
{
    struct RsCtxJfcCb *jfcCb = NULL;
    int ret;

    hccp_info_rma("[deinit][rs_jfc]destroy addr:0x%llx", addr);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    ret = RsUbGetJfcCb(devCb, addr, &jfcCb);
    if (ret != 0) {
        hccp_err("get jfc_cb failed, ret:%d, jfc addr:0x%llx", ret, addr);
        goto out;
    }

    ret = RsUbFreeJfcCb(devCb, jfcCb);
out:
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    return ret;
}

STATIC void RsUbFreeJfcCbList(struct RsUbDevCb *devCb, struct RsListHead *jfcList)
{
    struct RsCtxJfcCb *jfcCurr = NULL;
    struct RsCtxJfcCb *jfcNext = NULL;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    if (!RsListEmpty(jfcList)) {
        hccp_warn_rma("jfc list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(jfcCurr, jfcNext, jfcList, list, struct RsCtxJfcCb);
        for (; (&jfcCurr->list) != jfcList;
             jfcCurr = jfcNext, jfcNext = list_entry(jfcNext->list.next, struct RsCtxJfcCb, list)) {
            ret = RsUbFreeJfcCb(devCb, jfcCurr);
            if (ret != 0) {
                hccp_err("rs_ub_free_jfc_cb failed, ret:%d", ret);
            }
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

STATIC int RsUbFreeSegCb(struct RsUbDevCb *devCb, struct RsSegCb *segCb)
{
    int ret;

    ret = RsUrmaUnregisterSeg(segCb->segment);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_unregister_seg failed ret:%d", ret), -EOPENSRC);

    RsListDel(&segCb->list);
    devCb->lsegCnt--;
    free(segCb);
    segCb = NULL;

    return 0;
}

STATIC void RsUbFreeSegCbList(struct RsUbDevCb *devCb, struct RsListHead *lsegList, struct RsListHead *rsegList)
{
    struct RsSegCb *segCurr = NULL;
    struct RsSegCb *segNext = NULL;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    if (!RsListEmpty(lsegList)) {
        hccp_warn_rma("lseg list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(segCurr, segNext, lsegList, list, struct RsSegCb);
        for (; (&segCurr->list) != lsegList;
             segCurr = segNext, segNext = list_entry(segNext->list.next, struct RsSegCb, list)) {
            ret = RsUbFreeSegCb(devCb, segCurr);
            if (ret != 0) {
                hccp_err("rs_ub_free_seg_cb failed, ret:%d", ret);
            }
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    if (!RsListEmpty(rsegList)) {
        hccp_warn_rma("rseg list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(segCurr, segNext, rsegList, list, struct RsSegCb);
        for (; (&segCurr->list) != rsegList;
             segCurr = segNext, segNext = list_entry(segNext->list.next, struct RsSegCb, list)) {
            (void)RsUrmaUnimportSeg(segCurr->segment);
            RsListDel(&segCurr->list);
            free(segCurr);
            segCurr = NULL;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

STATIC void RsUbCtxFreeJettyCb(struct RsCtxJettyCb *jettyCb)
{
    struct RsCtxJettyCb *tmpJettyCb = jettyCb;

#ifdef CUSTOM_INTERFACE
    if (jettyCb->qpShareInfoAddr != NULL) {
        (void)DlHalBuffFree((void *)(uintptr_t)jettyCb->qpShareInfoAddr);
        jettyCb->qpShareInfoAddr = NULL;
    }
#endif

    pthread_mutex_destroy(&tmpJettyCb->crErrInfo.mutex);
    pthread_mutex_destroy(&tmpJettyCb->mutex);
    free(tmpJettyCb);
    tmpJettyCb = NULL;
}

STATIC void RsUbCtxDrvJettyDelete(struct RsCtxJettyCb *jettyCb)
{
    if (jettyCb->jettyMode == JETTY_MODE_URMA_NORMAL) {
        (void)RsUrmaDeleteJetty(jettyCb->jetty);
    } else {
        RsUbCtxExtJettyDelete(jettyCb);
    }

    // ccu jetty unreg db addr
    if (jettyCb->jettyMode == JETTY_MODE_CCU || jettyCb->jettyMode == JETTY_MODE_CCU_TA_CACHE) {
        (void)RsUbCtxLmemUnreg(jettyCb->devCb, jettyCb->dbSegHandle);
    }
}

STATIC int RsUbFreeRemJettyCb(struct RsUbDevCb *devCb, struct RsCtxRemJettyCb *rjettyCb)
{
    unsigned int remJettyId = rjettyCb->tjetty->id.id;
    unsigned int devIndex = devCb->index;
    int ret;

    RsListDel(&rjettyCb->list);
    devCb->rjettyCnt--;

    ret = RsUrmaUnimportJetty(rjettyCb->tjetty);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_unimport_jetty failed, ret:%d, devIndex:0x%x, remJettyId %u", ret, devIndex, remJettyId),
        -EOPENSRC);

    free(rjettyCb);
    rjettyCb = NULL;

    return 0;
}

STATIC void RsUbUnbindJettyCbList(struct RsUbDevCb *devCb, struct RsListHead *jettyList)
{
    struct RsCtxJettyCb *jettyCurr = NULL;
    struct RsCtxJettyCb *jettyNext = NULL;
    int ret;

    if (!RsListEmpty(jettyList)) {
        hccp_warn_rma("jetty list is not empty! start to unbind");
        RS_LIST_GET_HEAD_ENTRY(jettyCurr, jettyNext, jettyList, list, struct RsCtxJettyCb);
        for (; (&jettyCurr->list) != jettyList;
             jettyCurr = jettyNext, jettyNext = list_entry(jettyNext->list.next, struct RsCtxJettyCb, list)) {
            // no need to unbind
            if (jettyCurr->state != RS_JETTY_STATE_BIND) {
                continue;
            }
            hccp_info_rma("jetty_id[%u] will be unbind", jettyCurr->jetty->jetty_id.id);
            ret = RsUrmaUnbindJetty(jettyCurr->jetty);
            if (ret != 0) {
                hccp_err("rs_urma_unbind_jetty failed, ret:%d errno:%d devIndex:0x%x jetty_id:%u", ret, errno,
                    devCb->index, jettyCurr->jetty->jetty_id.id);
            }
            jettyCurr->state = RS_JETTY_STATE_CREATED;
        }
    }
}

STATIC void RsUbUnimportJettyCbList(struct RsUbDevCb *devCb, struct RsListHead *rjettyList)
{
    struct RsCtxRemJettyCb *remJettyCurr = NULL;
    struct RsCtxRemJettyCb *remJettyNext = NULL;
    int ret;

    if (!RsListEmpty(rjettyList)) {
        hccp_warn_rma("rjetty list is not empty! start to unimport");
        RS_LIST_GET_HEAD_ENTRY(remJettyCurr, remJettyNext, rjettyList, list, struct RsCtxRemJettyCb);
        for (; (&remJettyCurr->list) != rjettyList;
             remJettyCurr = remJettyNext,
             remJettyNext = list_entry(remJettyNext->list.next, struct RsCtxRemJettyCb, list)) {
            // no need to unimport
            if (remJettyCurr->state != RS_JETTY_STATE_IMPORTED) {
                continue;
            }
            hccp_info_rma("rjetty_id[%u] will be destroyed", remJettyCurr->tjetty->id.id);
            ret = RsUbFreeRemJettyCb(devCb, remJettyCurr);
            if (ret != 0) {
                hccp_err("rs_ub_ctx_jetty_unimport failed, ret:%d", ret);
            }
        }
    }
}

STATIC int RsUbCallocJettyBatchInfo(struct JettyDestroyBatchInfo *batchInfo, unsigned int num)
{
    batchInfo->jettyCbArr = calloc(num, sizeof(struct RsCtxJettyCb *));
    CHK_PRT_RETURN(batchInfo->jettyCbArr == NULL, hccp_err("calloc jetty_cb_arr failed"), -ENOMEM);
    batchInfo->jettyArr = calloc(num, sizeof(urma_jetty_t *));
    CHK_PRT_RETURN(batchInfo->jettyArr == NULL, hccp_err("calloc jetty_arr failed"), -ENOMEM);

    return 0;
}

STATIC void RsUbFreeJettyBatchInfo(struct JettyDestroyBatchInfo *batchInfo)
{
    if (batchInfo->jettyCbArr != NULL) {
        free(batchInfo->jettyCbArr);
        batchInfo->jettyCbArr = NULL;
    }
    if (batchInfo->jettyArr != NULL) {
        free(batchInfo->jettyArr);
        batchInfo->jettyArr = NULL;
    }
}

STATIC void RsUbFreeJettyCbBatch(struct JettyDestroyBatchInfo *batchInfo, unsigned int *num, urma_jetty_t *badJetty,
    urma_jfr_t *badJfr)
{
    (void)badJfr;
    unsigned int jettyDestroyNum = *num;
    unsigned int i;

    for (i = 0; i < *num; ++i) {
        // ccu jetty unreg db addr
        if (batchInfo->jettyCbArr[i]->jettyMode == JETTY_MODE_CCU ||
            batchInfo->jettyCbArr[i]->jettyMode == JETTY_MODE_CCU_TA_CACHE) {
            (void)RsUbCtxLmemUnreg(batchInfo->jettyCbArr[i]->devCb, batchInfo->jettyCbArr[i]->dbSegHandle);
        }
        RsUbCtxFreeJettyCb(batchInfo->jettyCbArr[i]);

        if (batchInfo->jettyArr[i] == badJetty) {
            jettyDestroyNum = i;
        }
    }
    *num = jettyDestroyNum;
}

STATIC int RsUbDestroyJettyCbBatch(struct JettyDestroyBatchInfo *batchInfo, unsigned int *num)
{
    urma_jetty_t *badJetty = NULL;
    urma_jfr_t *badJfr = NULL;
    int jettyDestroyRet = 0;

    RsUbVaMunmapBatch(batchInfo->jettyCbArr, *num);
    jettyDestroyRet = RsUrmaDeleteJettyBatch(batchInfo->jettyArr, (int)*num, &badJetty);
    if (jettyDestroyRet != 0) {
        hccp_err("rs_urma_delete_jetty_batch failed, jettyDestroyRet:%d, num:%u", jettyDestroyRet, *num);
    }

    RsUbFreeJettyIdBatch(batchInfo->jettyCbArr, *num);
    RsUbFreeJettyCbBatch(batchInfo, num, badJetty, badJfr);
    return jettyDestroyRet;
}

STATIC void RsUbDestroyJettyCbList(struct RsUbDevCb *devCb, struct RsListHead *jettyList)
{
    struct JettyDestroyBatchInfo batchInfo = {0};
    struct RsCtxJettyCb *jettyCurr = NULL;
    struct RsCtxJettyCb *jettyNext = NULL;
    unsigned int num = 0, i = 0;
    int ret;

    if (RsListEmpty(jettyList)) {
        return;
    }

    hccp_warn_rma("jetty list is not empty! start to delete");
    ret = RsUbCallocJettyBatchInfo(&batchInfo, devCb->jettyCnt);
    if (ret != 0) {
        hccp_err("rs_ub_calloc_jetty_batch_info failed, ret:%d", ret);
        goto free_batch_info;
    }

    RS_LIST_GET_HEAD_ENTRY(jettyCurr, jettyNext, jettyList, list, struct RsCtxJettyCb);
    for (; (&jettyCurr->list) != jettyList;
         jettyCurr = jettyNext, jettyNext = list_entry(jettyNext->list.next, struct RsCtxJettyCb, list)) {
        // no need to destroy
        if (jettyCurr->state != RS_JETTY_STATE_CREATED) {
            continue;
        }
        hccp_info_rma("jetty_id[%u] will be destroyed", jettyCurr->jetty->jetty_id.id);
        batchInfo.jettyCbArr[i] = jettyCurr;
        batchInfo.jettyArr[i] = jettyCurr->jetty;

        RsListDel(&jettyCurr->list);
        devCb->jettyCnt--;
        i++;
    }

    num = i;
    if (num != 0) {
        ret = RsUbDestroyJettyCbBatch(&batchInfo, &num);
        if (ret != 0) {
            hccp_err("rs_ub_ctx_jetty_destroy_batch failed, ret:%d, need to be destroyed:%u, actually destroyed:%u",
                ret, i, num);
        }
    }

free_batch_info:
    RsUbFreeJettyBatchInfo(&batchInfo);
}

void RsUbFreeJettyCbList(struct RsUbDevCb *devCb, struct RsListHead *jettyList, struct RsListHead *rjettyList)
{
    // free jetty step: unbind -> unimport -> delete
    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsUbUnbindJettyCbList(devCb, jettyList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsUbUnimportJettyCbList(devCb, rjettyList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsUbDestroyJettyCbList(devCb, jettyList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

int RsUbCtxChanCreate(struct RsUbDevCb *devCb, union DataPlaneCstmFlag dataPlaneFlag, unsigned long long *addr, int *fd)
{
    struct RsCtxJfceCb *jfceCb = NULL;
    urma_jfce_t *outJfce = NULL;
    int ret = 0;

    jfceCb = calloc(1, sizeof(struct RsCtxJfceCb));
    CHK_PRT_RETURN(jfceCb == NULL, hccp_err("calloc jfce_cb failed"), -ENOMEM);

    jfceCb->devCb = devCb;
    outJfce = RsUrmaCreateJfce(devCb->urmaCtx);
    if (outJfce == NULL) {
        hccp_err("rs_urma_create_jfce failed, errno:%d", errno);
        ret = -EOPENSRC;
        goto free_ctx_jfce_cb;
    }

    jfceCb->jfceAddr = (uint64_t)(uintptr_t)outJfce; // urma_jfce_t *
    *addr = jfceCb->jfceAddr;

    if (dataPlaneFlag.bs.pollCqCstm == 0) {
        ret = RsEpollCtl(devCb->rscb->connCb.epollfd, EPOLL_CTL_ADD, outJfce->fd, EPOLLIN | EPOLLRDHUP);
        if (ret != 0) {
            hccp_err("rs_epoll_ctl failed ret:%d, fd:%d", ret, devCb->rscb->connCb.epollfd);
            goto delete_jfce;
        }
    }
    jfceCb->dataPlaneFlag = dataPlaneFlag;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListAddTail(&jfceCb->list, &devCb->jfceList);
    jfceCb->devCb->jfceCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    *fd = outJfce->fd;
    hccp_info_rma("dev_index:0x%x jfce addr:0x%llx fd:%d", devCb->index, jfceCb->jfceAddr, outJfce->fd);
    return ret;

delete_jfce:
    (void)RsUrmaDeleteJfce(outJfce);
free_ctx_jfce_cb:
    free(jfceCb);
    jfceCb = NULL;
    return ret;
}

STATIC int RsUbGetJfceCb(struct RsUbDevCb *devCb, unsigned long long addr, struct RsCtxJfceCb **jfceCb)
{
    struct RsCtxJfceCb **tempJfceCb = jfceCb;
    struct RsCtxJfceCb *jfceCbCurr = NULL;
    struct RsCtxJfceCb *jfceCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(jfceCbCurr, jfceCbNext, &devCb->jfceList, list, struct RsCtxJfceCb);
    for (; (&jfceCbCurr->list) != (&devCb->jfceList);
         jfceCbCurr = jfceCbNext, jfceCbNext = list_entry(jfceCbNext->list.next, struct RsCtxJfceCb, list)) {
        if (jfceCbCurr->jfceAddr == addr) {
            *tempJfceCb = jfceCbCurr;
            return 0;
        }
    }

    *tempJfceCb = NULL;
    hccp_err("jfce_cb for jfce_addr:0x%llx is not available!", addr);

    return -ENODEV;
}

int RsUbCtxChanDestroy(struct RsUbDevCb *devCb, unsigned long long addr)
{
    struct RsCtxJfceCb *jfceCb = NULL;
    urma_jfce_t *jfce = NULL;
    int ret;

    hccp_info_rma("[rs_ctx_chan]jfce destroy addr:0x%llx", addr);

    ret = RsUbGetJfceCb(devCb, addr, &jfceCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get jfce_cb failed, ret:%d, jfce addr:0x%llx", ret, addr), ret);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&jfceCb->list);
    jfceCb->devCb->jfceCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    jfce = (urma_jfce_t *)(uintptr_t)jfceCb->jfceAddr;
    if (jfceCb->dataPlaneFlag.bs.pollCqCstm == 0) {
        (void)RsEpollCtl(devCb->rscb->connCb.epollfd, EPOLL_CTL_DEL, jfce->fd, EPOLLIN | EPOLLRDHUP);
    }

    ret = RsUrmaDeleteJfce(jfce);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_delete_jfce failed, ret:%d errno:%d jfce addr:0x%llx", ret, errno, jfceCb->jfceAddr),
        -EOPENSRC);

    free(jfceCb);
    jfceCb = NULL;

    hccp_info_rma("rs ctx jfce destroy success, dev jfce num is %u", devCb->jfceCnt);
    return ret;
}

STATIC int RsUbFreeJfceCb(struct RsUbDevCb *devCb, struct RsCtxJfceCb *jfceCb)
{
    int ret = 0;

    RsListDel(&jfceCb->list);
    devCb->jfceCnt--;

    ret = RsUrmaDeleteJfce((urma_jfce_t *)(uintptr_t)jfceCb->jfceAddr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[rs_ctx_chan]rs_ub_delete_jfce failed, ret:%d, jfce addr:0x%llx", ret, jfceCb->jfceAddr), -EOPENSRC);

    free(jfceCb);
    jfceCb = NULL;

    return 0;
}

STATIC void RsUbFreeJfceCbList(struct RsUbDevCb *devCb, struct RsListHead *jfceList)
{
    struct RsCtxJfceCb *jfceCurr = NULL;
    struct RsCtxJfceCb *jfceNext = NULL;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    if (!RsListEmpty(jfceList)) {
        hccp_warn_rma("jfce list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(jfceCurr, jfceNext, jfceList, list, struct RsCtxJfceCb);
        for (; (&jfceCurr->list) != jfceList;
             jfceCurr = jfceNext, jfceNext = list_entry(jfceNext->list.next, struct RsCtxJfceCb, list)) {
            ret = RsUbFreeJfceCb(devCb, jfceCurr);
            if (ret != 0) {
                hccp_err("rs_ub_free_jfce_cb failed, ret:%d", ret);
            }
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

int RsUbCtxTokenIdAlloc(struct RsUbDevCb *devCb, unsigned long long *addr, unsigned int *tokenId)
{
    struct RsTokenIdCb *tokenIdCb = NULL;

    tokenIdCb = calloc(1, sizeof(struct RsTokenIdCb));
    CHK_PRT_RETURN(tokenIdCb == NULL, hccp_err("calloc token_id_cb failed"), -ENOMEM);

    tokenIdCb->devCb = devCb;
    tokenIdCb->tokenId = RsUrmaAllocTokenId(devCb->urmaCtx);
    if (tokenIdCb->tokenId == NULL) {
        hccp_err("rs_urma_alloc_token_id failed, errno:%d, devIndex:0x%x", errno, devCb->index);
        goto free_ctx_token_id_cb;
    }

    *addr = (uint64_t)(uintptr_t)tokenIdCb->tokenId; // urma_token_id_t *
    *tokenId = tokenIdCb->tokenId->token_id;
    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListAddTail(&tokenIdCb->list, &devCb->tokenIdList);
    tokenIdCb->devCb->tokenIdCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    hccp_info_rma("alloc success, tokenId addr:0x%llx, devIndex:0x%x", *addr, devCb->index);
    return 0;

free_ctx_token_id_cb:
    free(tokenIdCb);
    tokenIdCb = NULL;
    return -EOPENSRC;
}

STATIC int RsUbGetTokenIdCb(struct RsUbDevCb *devCb, unsigned long long addr, struct RsTokenIdCb **tokenIdCb)
{
    struct RsTokenIdCb **tempTokenIdCb = tokenIdCb;
    struct RsTokenIdCb *tokenIdCbCurr = NULL;
    struct RsTokenIdCb *tokenIdCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(tokenIdCbCurr, tokenIdCbNext, &devCb->tokenIdList, list, struct RsTokenIdCb);
    for (; (&tokenIdCbCurr->list) != (&devCb->tokenIdList);
         tokenIdCbCurr = tokenIdCbNext,
         tokenIdCbNext = list_entry(tokenIdCbNext->list.next, struct RsTokenIdCb, list)) {
        if ((uint64_t)(uintptr_t)tokenIdCbCurr->tokenId == addr) {
            *tempTokenIdCb = tokenIdCbCurr;
            return 0;
        }
    }

    *tempTokenIdCb = NULL;
    hccp_err("token_id_cb for token_id addr:0x%llx is not available! devIndex:0x%x", addr, devCb->index);

    return -ENODEV;
}

STATIC int RsUbFreeTokenIdCb(struct RsUbDevCb *devCb, struct RsTokenIdCb *tokenIdCb)
{
    int ret = 0;

    RsListDel(&tokenIdCb->list);
    devCb->tokenIdCnt--;

    ret = RsUrmaFreeTokenId(tokenIdCb->tokenId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_free_token_id failed, ret:%d, devIndex:0x%x, tokenId addr:0x%llx", ret, devCb->index,
            (uint64_t)(uintptr_t)tokenIdCb->tokenId),
        -EOPENSRC);

    free(tokenIdCb);
    tokenIdCb = NULL;
    return 0;
}

int RsUbCtxTokenIdFree(struct RsUbDevCb *devCb, unsigned long long addr)
{
    struct RsTokenIdCb *tokenIdCb = NULL;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    ret = RsUbGetTokenIdCb(devCb, addr, &tokenIdCb);
    if (ret != 0) {
        hccp_err("get token_id_cb failed! ret %d, devIndex:0x%x, tokenId addr:0x%llx", ret, devCb->index, addr);
        goto free_lock;
    }
    ret = RsUbFreeTokenIdCb(devCb, tokenIdCb);
    if (ret != 0) {
        hccp_err("free_token_id_cb failed, ret:%d, devIndex:0x%x, tokenId addr:0x%llx", ret, devCb->index, addr);
        goto free_lock;
    }
    hccp_info_rma("rs token id free success, dev tokenId num is %u, devIndex:0x%x, tokenId addr:0x%llx",
        devCb->tokenIdCnt, devCb->index, addr);

free_lock:
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    return ret;
}

STATIC void RsUbFreeTokenIdCbList(struct RsUbDevCb *devCb, struct RsListHead *tokenIdList)
{
    struct RsTokenIdCb *tokenIdCurr = NULL;
    struct RsTokenIdCb *tokenIdNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    if (!RsListEmpty(tokenIdList)) {
        hccp_warn_rma("token_id list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(tokenIdCurr, tokenIdNext, tokenIdList, list, struct RsTokenIdCb);
        for (; (&tokenIdCurr->list) != tokenIdList;
             tokenIdCurr = tokenIdNext, tokenIdNext = list_entry(tokenIdNext->list.next, struct RsTokenIdCb, list)) {
            (void)RsUbFreeTokenIdCb(devCb, tokenIdCurr);
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    return;
}

void RsUbFreeAsyncEventCb(struct RsUbDevCb *devCb, struct RsCtxAsyncEventCb *asyncEventCb)
{
    RsListDel(&asyncEventCb->list);
    devCb->asyncEventCnt--;

    free(asyncEventCb);
    asyncEventCb = NULL;
}

STATIC void RsUbFreeAsyncEventCbList(struct RsUbDevCb *devCb, struct RsListHead *asyncEventList)
{
    struct RsCtxAsyncEventCb *asyncEventCurr = NULL;
    struct RsCtxAsyncEventCb *asyncEventNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    (void)RsEpollCtl(devCb->rscb->connCb.epollfd, EPOLL_CTL_DEL, devCb->urmaCtx->async_fd, EPOLLIN | EPOLLRDHUP);
    if (!RsListEmpty(asyncEventList)) {
        hccp_run_warn("async_event list is not empty!");
        RS_LIST_GET_HEAD_ENTRY(asyncEventCurr, asyncEventNext, asyncEventList, list, struct RsCtxAsyncEventCb);
        for (; (&asyncEventCurr->list) != asyncEventList;
             asyncEventCurr = asyncEventNext,
             asyncEventNext = list_entry(asyncEventNext->list.next, struct RsCtxAsyncEventCb, list)) {
            RsUbFreeAsyncEventCb(devCb, asyncEventCurr);
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

int RsUbCtxDeinit(struct RsUbDevCb *devCb)
{
    int ret;

    hccp_info_rma("[deinit][rs_ctx]start deinit, phyId:%u, devIndex:0x%x", devCb->phyId, devCb->index);

    RsUbFreeSegCbList(devCb, &devCb->lsegList, &devCb->rsegList);
    RsUbFreeJettyCbList(devCb, &devCb->jettyList, &devCb->rjettyList);

    /* Destroy shared JFR and JFC after all jettys are freed */
    RsUbDevShareJfrDeinit(devCb);

    RsUbFreeJfcCbList(devCb, &devCb->jfcList);
    RsUbFreeJfceCbList(devCb, &devCb->jfceList);
    RsUbFreeTokenIdCbList(devCb, &devCb->tokenIdList);
    RsUbFreeAsyncEventCbList(devCb, &devCb->asyncEventList);

    ret = RsUrmaDeleteContext(devCb->urmaCtx);
    if (ret != 0) {
        hccp_err("rs_urma_delete_context failed, ret:%d", ret);
    }

    pthread_mutex_destroy(&devCb->mutex);

    RS_PTHREAD_MUTEX_LOCK(&devCb->rscb->mutex);
    RsListDel(&devCb->list);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->rscb->mutex);
    RsSensorNodeUnregister(devCb->rscb);

    RsUbApiDeinit();

    hccp_run_info("phyId:%u, devIndex:0x%x deinit success", devCb->phyId, devCb->index);
    free(devCb);
    devCb = NULL;
    return ret;
}

STATIC int RsUbQuerySegCb(struct RsUbDevCb *devCb, uint64_t addr, struct RsSegCb **segCb, struct RsListHead *segList)
{
    struct RsSegCb *segCurr = NULL;
    struct RsSegCb *segNext = NULL;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RS_LIST_GET_HEAD_ENTRY(segCurr, segNext, segList, list, struct RsSegCb);
    for (; (&segCurr->list) != segList;
         segCurr = segNext, segNext = list_entry(segNext->list.next, struct RsSegCb, list)) {
        if ((segCurr->segInfo.addr <= addr) && (addr < segCurr->segInfo.addr + segCurr->segInfo.len)) {
            *segCb = segCurr;
            RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
            return 0;
        }
    }

    *segCb = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    hccp_info_rma("cannot find seg_cb for addr@0x%lx", addr);
    return -ENODEV;
}

STATIC int RsUbInitSegCb(struct MemRegAttrT *memAttr, struct RsUbDevCb *devCb, struct RsSegCb *segCb)
{
    struct RsTokenIdCb *tokenIdCb = NULL;
    urma_seg_cfg_t segCfg = {0};
    int ret = 0;

    segCfg.flag.value = memAttr->ub.flags.value;
    segCfg.va = memAttr->mem.addr;
    segCfg.len = memAttr->mem.size;
    segCfg.token_value = segCb->tokenValue;
    segCfg.user_ctx = (uintptr_t)NULL;
    segCfg.iova = 0;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);

    // token id in cfg is valid, get token id by mem_attr->ub.token_id_addr
    if (segCfg.flag.bs.token_id_valid == URMA_TOKEN_ID_VALID) {
        ret = RsUbGetTokenIdCb(devCb, memAttr->ub.tokenIdAddr, &tokenIdCb);
        if (ret != 0) {
            hccp_err("get token_id_cb failed! ret %d, devIndex:0x%x, tokenId addr:0x%llx", ret, devCb->index,
                memAttr->ub.tokenIdAddr);
            goto free_lock;
        }
        segCfg.token_id = tokenIdCb->tokenId;
    }

    segCb->segment = RsUrmaRegisterSeg(devCb->urmaCtx, &segCfg);
    if (segCb->segment == NULL) {
        hccp_err("[init][rs_ctx_lmem]rs_urma_register_seg len[0x%llx] failed, errno:%d", segCfg.len, errno);
        ret = -EOPENSRC;
        goto free_lock;
    }

    segCb->segInfo.seg = segCb->segment->seg;
    // resv len as 1 to save addr for later unreg to query
    segCb->segInfo.addr = (uint64_t)(uintptr_t)segCb->segment;
    segCb->segInfo.len = 1;
    RsListAddTail(&segCb->list, &devCb->lsegList);
    devCb->lsegCnt++;

free_lock:
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    return ret;
}

STATIC void RsUbDeinitSegCb(struct RsUbDevCb *devCb, struct RsSegCb *segCb)
{
    (void)RsUrmaUnregisterSeg(segCb->segment);
    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&segCb->list);
    devCb->lsegCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
}

int RsUbCtxLmemReg(struct RsUbDevCb *devCb, struct MemRegAttrT *memAttr, struct MemRegInfoT *memInfo)
{
    struct RsSegCb *lsegCb = NULL;
    int ret;

    CHK_PRT_RETURN(memAttr->mem.size == 0, hccp_err("mem_attr->mem.size is 0"), -EINVAL);

    lsegCb = calloc(1, sizeof(struct RsSegCb));
    CHK_PRT_RETURN(lsegCb == NULL, hccp_err("calloc lseg_cb failed"), -ENOMEM);
    lsegCb->devCb = devCb;
    lsegCb->tokenValue.token = memAttr->ub.tokenValue;

    ret = RsUbInitSegCb(memAttr, devCb, lsegCb);
    if (ret != 0) {
        hccp_err("RsUbInitSegCb failed, ret:%d devIndex:0x%x addr:0x%llx, len[0x%llx], flag[0x%x]", ret, devCb->index,
            memAttr->mem.addr, memAttr->mem.size, memAttr->ub.flags.value);
        goto init_err;
    }

    ret = memcpy_s(memInfo->key.value, sizeof(memInfo->key.value), &lsegCb->segInfo.seg, sizeof(urma_seg_t));
    if (ret != 0) {
        hccp_err("memcpy_s seg failed ret:%d sizeof(memInfo->key.value):%zu sizeof(urma_seg_t):%zu", ret,
            sizeof(memInfo->key.value), sizeof(urma_seg_t));
        ret = -ESAFEFUNC;
        goto reg_err;
    }
    memInfo->key.size = sizeof(urma_seg_t);
    memInfo->ub.tokenId = lsegCb->segment->token_id->token_id;
    memInfo->ub.targetSegHandle = (uintptr_t)lsegCb->segment;

    hccp_run_info("devIndex:0x%x addr:0x%llx, len[0x%llx], flag[0x%x] segment:0x%llx register success", devCb->index,
        memAttr->mem.addr, memAttr->mem.size, memAttr->ub.flags.value, memInfo->ub.targetSegHandle);
    return 0;

reg_err:
    RsUbDeinitSegCb(devCb, lsegCb);
init_err:
    free(lsegCb);
    lsegCb = NULL;
    return ret;
}

int RsUbCtxLmemUnreg(struct RsUbDevCb *devCb, unsigned long long addr)
{
    struct RsSegCb *lsegCb = NULL;
    int ret;

    ret = RsUbQuerySegCb(devCb, addr, &lsegCb, &devCb->lsegList);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsUbQuerySegCb failed ret:%d devIndex:0x%x segment:0x%llx", ret, devCb->index, addr), ret);

    ret = RsUrmaUnregisterSeg(lsegCb->segment);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("urma_unregister_seg failed ret:%d devIndex:0x%x segment:0x%llx", ret, devCb->index, addr), ret);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&lsegCb->list);
    devCb->lsegCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);
    hccp_run_info("devIndex:0x%x segment:0x%llx unregister success", devCb->index, addr);
    free(lsegCb);
    lsegCb = NULL;
    return 0;
}

int RsUbCtxRmemImport(struct RsUbDevCb *devCb, struct MemImportAttrT *memAttr, struct MemImportInfoT *memInfo)
{
    struct RsSegCb *remSegCb = NULL;
    urma_import_seg_flag_t flag = {0};
    urma_token_t tokenValue = {0};
    uint64_t mappingAddr = 0;
    int ret;

    remSegCb = calloc(1, sizeof(struct RsSegCb));
    CHK_PRT_RETURN(remSegCb == NULL, hccp_err("[init][rs_ctx_rmem]calloc rem_seg_cb failed"), -ENOMEM);

    ret = memcpy_s(&remSegCb->segInfo.seg, sizeof(urma_seg_t), &memAttr->key.value, memAttr->key.size);
    if (ret != 0) {
        hccp_err("[init][rs_ctx_rmem]memcpy_s failed, ret:%d", ret);
        ret = -ESAFEFUNC;
        goto free_rem_seg_cb;
    }

    tokenValue.token = memAttr->ub.tokenValue;
    flag.value = memAttr->ub.flags.value;
    // mapping_addr is needed if flag mapping set value
    if (memAttr->ub.flags.bs.mapping == 1) {
        mappingAddr = memAttr->ub.mappingAddr;
    }
    remSegCb->segment = RsUrmaImportSeg(devCb->urmaCtx, &remSegCb->segInfo.seg, &tokenValue, mappingAddr, flag);
    if (remSegCb->segment == NULL) {
        hccp_err("[init][rs_ctx_rmem]rs_urma_import_seg failed, errno:%d", errno);
        ret = -EOPENSRC;
        goto free_rem_seg_cb;
    }

    memInfo->ub.targetSegHandle = (uintptr_t)remSegCb->segment;
    // resv len as 1 to save addr for later unimport to query
    remSegCb->segInfo.addr = memInfo->ub.targetSegHandle;
    remSegCb->segInfo.len = 1;

    hccp_run_info("devIndex:0x%x segment:0x%llx import success", devCb->index, remSegCb->segInfo.addr);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListAddTail(&remSegCb->list, &devCb->rsegList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    return 0;

free_rem_seg_cb:
    free(remSegCb);
    remSegCb = NULL;
    return ret;
}

int RsUbCtxRmemUnimport(struct RsUbDevCb *devCb, unsigned long long addr)
{
    struct RsSegCb *remSegCb = NULL;
    int ret;

    ret = RsUbQuerySegCb(devCb, addr, &remSegCb, &devCb->rsegList);
    CHK_PRT_RETURN(ret != 0, hccp_warn_rma("[deinit][rs_ctx_rmem]can not find rem seg cb for addr:0x%llx", addr), 0);

    RsUrmaUnimportSeg(remSegCb->segment);
    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&remSegCb->list);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    hccp_run_info("devIndex:0x%x segment:0x%llx unimport success", devCb->index, addr);

    free(remSegCb);
    remSegCb = NULL;
    return 0;
}

STATIC int RsUbCtxJfcCreateNormal(struct RsUbDevCb *devCb, urma_jfc_cfg_t *jfcCfg, urma_jfc_t **outJfc)
{
    uint64_t jfceAddr = (uint64_t)(uintptr_t)jfcCfg->jfce;
    struct RsCtxJfceCb *jfceCb = NULL;
    int ret = 0;

    if (jfcCfg->jfce != NULL) {
        ret = RsUbGetJfceCb(devCb, jfceAddr, &jfceCb);
        CHK_PRT_RETURN(ret != 0, hccp_err("get jfce_cb failed, ret:%d, jfce addr:0x%llx", ret, jfceAddr), ret);
    }

    *outJfc = RsUrmaCreateJfc(devCb->urmaCtx, jfcCfg);
    CHK_PRT_RETURN(*outJfc == NULL, hccp_err("rs_urma_create_jfc failed, errno:%d", errno), -EOPENSRC);

    ret = RsUrmaRearmJfc(*outJfc, false);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_rearm_jfc failed, ret:%d errno:%d", ret, errno), -EOPENSRC);

    return ret;
}

STATIC void RsUbFillJfcInfo(struct RsCtxJfcCb *jfcCb, struct CtxCqInfo *info)
{
    info->addr = jfcCb->jfcAddr;
    info->ub.id = jfcCb->jfcId;
    info->ub.cqeSize = WQE_BB_SIZE;
    info->ub.bufAddr = jfcCb->bufAddr;
    info->ub.swdbAddr = jfcCb->swdbAddr;
}

int RsUbCtxJfcCreate(struct RsUbDevCb *devCb, struct CtxCqAttr *attr, struct CtxCqInfo *info)
{
    struct RsCtxJfcCb *jfcCb = NULL;
    urma_jfc_cfg_t jfcCfg = {0};
    urma_jfc_t *outJfc = NULL;
    int ret = 0;

    jfcCb = (struct RsCtxJfcCb *)calloc(1, sizeof(struct RsCtxJfcCb));
    CHK_PRT_RETURN(jfcCb == NULL, hccp_err("calloc jfc_cb failed"), -ENOMEM);

    jfcCb->devCb = devCb;
    jfcCb->jfcType = attr->ub.mode;
    jfcCb->depth = attr->depth;
    jfcCb->ccuExCfg.valid = attr->ub.ccuExCfg.valid;
    jfcCb->ccuExCfg.cqeFlag = attr->ub.ccuExCfg.cqeFlag;
    jfcCfg.depth = attr->depth;
    jfcCfg.flag.value = attr->ub.flag.value;
    jfcCfg.user_ctx = attr->ub.userCtx;
    jfcCfg.ceqn = attr->ub.ceqn;
    jfcCfg.jfce = attr->chanAddr == 0 ? NULL : (urma_jfce_t *)(uintptr_t)attr->chanAddr;
    if (attr->ub.mode == JFC_MODE_STARS_POLL || attr->ub.mode == JFC_MODE_CCU_POLL ||
        attr->ub.mode == JFC_MODE_USER_CTL_NORMAL) {
        ret = RsUbCtxJfcCreateExt(jfcCb, &jfcCfg, &outJfc);
        if (ret != 0) {
            hccp_err("rs_ub_ctx_jfc_create_ext jfc_mode:%d failed, ret:%d", attr->ub.mode, ret);
            goto jfc_cb_init_err;
        }
    } else if (attr->ub.mode == JFC_MODE_NORMAL) {
        jfcCfg.jfce = (attr->chanAddr == 0) ? NULL : (urma_jfce_t *)(uintptr_t)attr->chanAddr;
        ret = RsUbCtxJfcCreateNormal(devCb, &jfcCfg, &outJfc);
        if (ret != 0) {
            hccp_err("rs_ub_ctx_jfc_create_normal failed, jfcMode:%d ret:%d", attr->ub.mode, ret);
            goto jfc_cb_init_err;
        }
    } else {
        hccp_err("jfc_type %d is invalid, not support!", attr->ub.mode);
        ret = -EINVAL;
        goto jfc_cb_init_err;
    }
    jfcCb->jfcAddr = (uint64_t)(uintptr_t)outJfc; // urma_jfc_t *
    RsUbFillJfcInfo(jfcCb, info);

    hccp_info_rma("jfc addr:0x%llx mode:%d", jfcCb->jfcAddr, jfcCb->jfcType);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    jfcCb->devCb->jfcCnt++;
    RsListAddTail(&jfcCb->list, &devCb->jfcList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    return 0;

jfc_cb_init_err:
    free(jfcCb);
    jfcCb = NULL;

    return ret;
}

STATIC bool RsUbIsJettyModeValid(int jettyMode, bool lockFlag)
{
    if (jettyMode < 0 || jettyMode >= JETTY_MODE_MAX || (jettyMode == JETTY_MODE_CCU_TA_CACHE && !lockFlag)) {
        return false;
    }

    return true;
}

STATIC int RsUbJettyCbInit(struct RsUbDevCb *devCb, struct CtxQpAttr *jettyAttr, struct RsCtxJettyCb *jettyCb)
{
    bool lockFlag = jettyAttr->ub.taCacheMode.lockFlag;
    int jettyMode = jettyAttr->ub.mode;

    if (!RsUbIsJettyModeValid(jettyMode, lockFlag)) {
        hccp_err("unsupported jetty_mode:%d, lockFlag:%d", jettyMode, lockFlag);
        return -EINVAL;
    }

    jettyCb->txDepth = jettyAttr->sqDepth;
    jettyCb->rxDepth = jettyAttr->rqDepth;
    jettyCb->devCb = devCb;
    jettyCb->jettyMode = jettyMode;
    jettyCb->jettyId = jettyAttr->ub.jettyId;
    jettyCb->transportMode = jettyAttr->transportMode;
    jettyCb->state = RS_QP_STATUS_DISCONNECT;
    jettyCb->flag.value = jettyAttr->ub.flag.value;
    jettyCb->jfsFlag.value = jettyAttr->ub.jfsFlag.value;
    jettyCb->tokenIdAddr = jettyAttr->ub.tokenIdAddr;
    jettyCb->tokenValue = jettyAttr->ub.tokenValue;
    jettyCb->priority = jettyAttr->ub.priority;
    jettyCb->rnrRetry = jettyAttr->ub.rnrRetry;
    jettyCb->errTimeout = jettyAttr->ub.errTimeout;

    if (jettyCb->jettyMode == JETTY_MODE_CCU_TA_CACHE) {
        jettyCb->taCacheMode.lockFlag = jettyAttr->ub.taCacheMode.lockFlag;
        jettyCb->taCacheMode.sqeBufIdx = jettyAttr->ub.taCacheMode.sqeBufIdx;
    } else {
        jettyCb->extMode.sq = jettyAttr->ub.extMode.sq;
        jettyCb->extMode.piType = jettyAttr->ub.extMode.piType;
        jettyCb->extMode.cstmFlag = jettyAttr->ub.extMode.cstmFlag;
        jettyCb->extMode.sqebbNum = jettyAttr->ub.extMode.sqebbNum;
    }
    return 0;
}

#ifdef CUSTOM_INTERFACE
STATIC int RsUbJettyCbBuffAlloc(struct RsUbDevCb *devCb, struct RsCtxJettyCb *jettyCb, enum JfcMode jfcType)
{
    unsigned int logicDevid = 0;
    unsigned long flag = 0;
    int ret = 0;

    if (jfcType != JFC_MODE_NORMAL) {
        jettyCb->qpShareInfoAddr = NULL;
        return 0;
    }

    ret = rsGetLocalDevIDByHostDevID(devCb->phyId, &logicDevid);
    CHK_PRT_RETURN(ret != 0, hccp_err("rsGetLocalDevIDByHostDevID failed, phyId(%u), ret(%d)", devCb->phyId, ret), ret);

    flag = ((unsigned long)logicDevid << BUFF_FLAGS_DEVID_OFFSET) | BUFF_SP_SVM;
    ret = DlHalBuffAllocAlignEx(sizeof(struct CtxQpShareInfo), CI_ADDR_BUFFER_ALIGN_4K_PAGE_SIZE, flag,
        (int)devCb->rscb->grpId, (void **)&jettyCb->qpShareInfoAddr);
    if (ret != 0) {
        hccp_err("dl_hal_buff_alloc_align_ex failed, length:0x%llx, dev_id:0x%x, flag:0x%lx, grp_id:%u, ret:%d",
            sizeof(struct CtxQpShareInfo), logicDevid, flag, devCb->rscb->grpId, ret);
    }

    return ret;
}
#endif

STATIC int RsUbCtxInitJettyCb(struct RsUbDevCb *devCb, struct CtxQpAttr *attr, struct RsCtxJettyCb **jettyCb)
{
    struct RsCtxJettyCb *tmpJettyCb = NULL;
    int ret;

    tmpJettyCb = calloc(1, sizeof(struct RsCtxJettyCb));
    CHK_PRT_RETURN(tmpJettyCb == NULL, hccp_err("calloc tmp_jetty_cb failed, errno:%d", errno), -ENOMEM);

    ret = pthread_mutex_init(&tmpJettyCb->mutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init failed, ret:%d", ret);
        goto pthread_mutex_init_err;
    }

    ret = pthread_mutex_init(&tmpJettyCb->crErrInfo.mutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init failed, ret:%d", ret);
        goto cr_err_mutex_init_err;
    }

    ret = RsUbJettyCbInit(devCb, attr, tmpJettyCb);
    if (ret != 0) {
        hccp_err("jetty_cb init failed ret:%d", ret);
        goto jetty_cb_init_err;
    }

    *jettyCb = tmpJettyCb;
    return 0;

jetty_cb_init_err:
    pthread_mutex_destroy(&tmpJettyCb->crErrInfo.mutex);
cr_err_mutex_init_err:
    pthread_mutex_destroy(&tmpJettyCb->mutex);
pthread_mutex_init_err:
    free(tmpJettyCb);
    tmpJettyCb = NULL;

    return ret;
}

STATIC void RsUbCtxFillJfsCfg(struct RsCtxJettyCb *jettyCb, struct RsCtxJfcCb *sendJfcCb, urma_jfs_cfg_t *jfsCfg)
{
    jfsCfg->depth = (uint32_t)jettyCb->txDepth;
    jfsCfg->flag.value = jettyCb->jfsFlag.value;
    jfsCfg->trans_mode = jettyCb->transportMode;
    jfsCfg->priority = jettyCb->priority;
    jfsCfg->max_sge = (uint8_t)jettyCb->devCb->devAttr.sqMaxSge;
    jfsCfg->max_rsge = (uint8_t)jettyCb->devCb->devAttr.ub.maxJfsRsge;
    jfsCfg->max_inline_data = jettyCb->devCb->devAttr.ub.maxJfsInlineLen;
    jfsCfg->rnr_retry = jettyCb->rnrRetry;
    jfsCfg->err_timeout = jettyCb->errTimeout;
    jfsCfg->jfc = (urma_jfc_t *)(uintptr_t)sendJfcCb->jfcAddr;
    jfsCfg->user_ctx = (uint64_t)NULL;
}

int RsUbCtxRegJettyDb(struct RsCtxJettyCb *jettyCb, struct udma_u_jetty_info *jettyInfo)
{
    struct MemRegAttrT memAttr = {0};
    struct MemRegInfoT memInfo = {0};
    int ret;

    // register dwqe_addr with page size 4096, return db_addr to use, specify ub to alloc token id
    hccp_dbg("jetty_info->dwqe_addr:%pK, jettyInfo->dbAddr:%pK", jettyInfo->dwqe_addr, jettyInfo->db_addr);
    memAttr.mem.addr = (uint64_t)(uintptr_t)jettyInfo->dwqe_addr;
    memAttr.mem.size = 4096U;
    memAttr.ub.flags.value = 0;
    memAttr.ub.flags.bs.tokenPolicy = URMA_TOKEN_PLAIN_TEXT;
    memAttr.ub.flags.bs.cacheable = URMA_NON_CACHEABLE;
    memAttr.ub.flags.bs.access = MEM_SEG_ACCESS_READ | MEM_SEG_ACCESS_WRITE;
    memAttr.ub.flags.bs.nonPin = 1;
    // use user specified token id to register
    if (jettyCb->tokenIdAddr != 0) {
        memAttr.ub.flags.bs.tokenIdValid = URMA_TOKEN_ID_VALID;
        memAttr.ub.tokenIdAddr = jettyCb->tokenIdAddr;
    }
    memAttr.ub.tokenValue = jettyCb->tokenValue;
    ret = RsUbCtxLmemReg(jettyCb->devCb, &memAttr, &memInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsUbCtxLmemReg failed, ret:%d flags:0x%x", ret, memAttr.ub.flags.value), ret);

    jettyCb->dbTokenId = memInfo.ub.tokenId;
    jettyCb->dbSegHandle = memInfo.ub.targetSegHandle;
    return 0;
}

STATIC int RsUbCtxDrvJettyCreate(struct RsCtxJettyCb *jettyCb, struct RsCtxJfcCb *sendJfcCb,
    struct RsCtxJfcCb *recvJfcCb)
{
    urma_jetty_cfg_t jettyInitCfg = {0};
    urma_jfs_cfg_t jfsCfg = {0};

    jettyInitCfg.id = jettyCb->jettyId;
    jettyInitCfg.flag = jettyCb->flag;
    RsUbCtxFillJfsCfg(jettyCb, sendJfcCb, &jfsCfg);
    jettyInitCfg.jfs_cfg = jfsCfg;

    jettyInitCfg.shared.jfr = jettyCb->devCb->shareJfr;
    jettyInitCfg.shared.jfc = (urma_jfc_t *)(uintptr_t)recvJfcCb->jfcAddr;

    if (jettyCb->jettyMode == JETTY_MODE_URMA_NORMAL) {
        jettyCb->jetty = RsUrmaCreateJetty(jettyCb->devCb->urmaCtx, &jettyInitCfg);
        if (jettyCb->jetty == NULL) {
            hccp_err("rs_urma_create_jetty failed, errno=%d", errno);
        }
    } else {
        RsUbCtxExtJettyCreate(jettyCb, &jettyInitCfg);
    }

    if (jettyCb->jetty == NULL) {
        return -ENOMEM;
    }

    jettyCb->state = RS_JETTY_STATE_CREATED;
    return 0;
}

STATIC int RsUbFillJettyInfo(struct RsCtxJettyCb *jettyCb, struct QpCreateInfo *jettyInfo)
{
    struct RsJettyKeyInfo jettyKeyInfo = {0};
    int ret;

    jettyKeyInfo.jettyId = jettyCb->jetty->jetty_id;
    jettyKeyInfo.transMode = jettyCb->transportMode;
    ret = memcpy_s(jettyInfo->key.value, DEV_QP_KEY_SIZE, &jettyKeyInfo, sizeof(struct RsJettyKeyInfo));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy jetty_key_info failed, ret:%d", ret), -ESAFEFUNC);

    jettyInfo->key.size = (uint8_t)sizeof(struct RsJettyKeyInfo);
    jettyInfo->ub.uasid = jettyCb->jetty->jetty_id.uasid;
    jettyInfo->ub.id = jettyCb->jetty->jetty_id.id;
    jettyInfo->ub.dbAddr = jettyCb->dbAddr;
    jettyInfo->ub.sqBuffVa = jettyCb->sqBuffVa;
    jettyInfo->ub.wqebbSize = WQE_BB_SIZE;
    jettyInfo->ub.dbTokenId = jettyCb->dbTokenId;
    jettyInfo->va = (uint64_t)(uintptr_t)jettyCb->jetty;
    jettyInfo->ub.shareInfoAddr = (uint64_t)(uintptr_t)jettyCb->qpShareInfoAddr;
    jettyInfo->ub.shareInfoLen = sizeof(struct CtxQpShareInfo);

    return 0;
}

STATIC int RsUbQueryJfcCb(struct RsUbDevCb *devCb, unsigned long long scqIndex, unsigned long long rcqIndex,
    struct RsCtxJfcCb **sendJfcCb, struct RsCtxJfcCb **recvJfcCb)
{
    int ret;

    ret = RsUbGetJfcCb(devCb, scqIndex, sendJfcCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get send_jfc_cb failed, ret:%d scqIndex:0x%llx", ret, scqIndex), ret);

    ret = RsUbGetJfcCb(devCb, rcqIndex, recvJfcCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get recv_jfc_cb failed, ret:%d rcqIndex:0x%llx", ret, rcqIndex), ret);

    return 0;
}

int RsUbCtxJettyCreate(struct RsUbDevCb *devCb, struct CtxQpAttr *attr, struct QpCreateInfo *info)
{
    struct RsCtxJfcCb *sendJfcCb = NULL;
    struct RsCtxJfcCb *recvJfcCb = NULL;
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret;

    ret = RsUbCtxInitJettyCb(devCb, attr, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("alloc mem for jetty_cb failed, ret:%d", ret), ret);

    ret = RsUbQueryJfcCb(devCb, attr->scqIndex, attr->rcqIndex, &sendJfcCb, &recvJfcCb);
    if (ret != 0) {
        hccp_err("query jfc_cb scq_index:0x%llx rcq_index:0x%llx failed, ret:%d", attr->scqIndex, attr->rcqIndex, ret);
        goto free_jetty_cb;
    }

#ifdef CUSTOM_INTERFACE
    ret = RsUbJettyCbBuffAlloc(devCb, jettyCb, sendJfcCb->jfcType);
    if (ret != 0) {
        hccp_err("jetty_cb buff alloc failed ret:%d", ret);
        goto free_jetty_cb;
    }
#endif

    ret = RsUbCtxDrvJettyCreate(jettyCb, sendJfcCb, recvJfcCb);
    if (ret != 0) {
        hccp_err("rs_ub_ctx_drv_jetty_create failed, ret:%d", ret);
        goto free_jetty_cb;
    }

    ret = RsUbFillJettyInfo(jettyCb, info);
    if (ret != 0) {
        hccp_err("rs_ub_fill_jetty_info failed, ret:%d", ret);
        goto fill_jetty_info_err;
    }

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    jettyCb->scqIndex = attr->scqIndex;
    jettyCb->rcqIndex = attr->rcqIndex;
    RsListAddTail(&jettyCb->list, &devCb->jettyList);
    devCb->jettyCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    hccp_run_info("devIndex:0x%x jettyId:%u mode:%d create success, jettyCnt:%u", devCb->index, info->ub.id,
        jettyCb->jettyMode, devCb->jettyCnt);

    return 0;

fill_jetty_info_err:
    RsUbCtxDrvJettyDelete(jettyCb);
free_jetty_cb:
    RsUbCtxFreeJettyCb(jettyCb);

    return ret;
}

int RsUbGetJettyCb(struct RsUbDevCb *devCb, unsigned int jettyId, struct RsCtxJettyCb **jettyCb)
{
    struct RsCtxJettyCb **tempJettyCb = jettyCb;
    struct RsCtxJettyCb *jettyCbCurr = NULL;
    struct RsCtxJettyCb *jettyCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(jettyCbCurr, jettyCbNext, &devCb->jettyList, list, struct RsCtxJettyCb);
    for (; (&jettyCbCurr->list) != &devCb->jettyList;
         jettyCbCurr = jettyCbNext, jettyCbNext = list_entry(jettyCbNext->list.next, struct RsCtxJettyCb, list)) {
        if (jettyCbCurr->jetty != NULL && jettyCbCurr->jetty->jetty_id.id == jettyId) {
            *tempJettyCb = jettyCbCurr;
            return 0;
        }
    }

    *tempJettyCb = NULL;
    return -ENODEV;
}

int RsUbCtxJettyDestroy(struct RsUbDevCb *devCb, unsigned int jettyId)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret;

    ret = RsUbGetJettyCb(devCb, jettyId, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_run_warn("get jetty_cb unsuccessful, ret:%d, jettyId %u", ret, jettyId), ret);
    if (jettyCb->state != RS_JETTY_STATE_CREATED) {
        hccp_err("jetty_cb->state:%u not support to destroy, jettyId:%u", jettyCb->state, jettyId);
        return -EINVAL;
    }

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&jettyCb->list);
    devCb->jettyCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    RsUbCtxDrvJettyDelete(jettyCb);

    RsUbCtxFreeJettyCb(jettyCb);

    hccp_run_info("devIndex:0x%x jettyId:%u destroy success, jettyCnt:%u", devCb->index, jettyId, devCb->jettyCnt);

    return 0;
}

int RsUbCtxJettyFree(struct rs_cb *rscb, unsigned int ueInfo, unsigned int jettyId)
{
    struct RsUbDevCb *devCbCurr = NULL;
    struct RsUbDevCb *devCbNext = NULL;
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret = 0;

    RS_LIST_GET_HEAD_ENTRY(devCbCurr, devCbNext, &rscb->udevList, list, struct RsUbDevCb);
    for (; (&devCbCurr->list) != &rscb->udevList;
         devCbCurr = devCbNext, devCbNext = list_entry(devCbNext->list.next, struct RsUbDevCb, list)) {
        if ((devCbCurr->index & DEV_INDEX_UE_INFO_MASK) != ueInfo) {
            continue;
        }

        RS_PTHREAD_MUTEX_LOCK(&devCbCurr->mutex);
        ret = RsUbGetJettyCb(devCbCurr, jettyId, &jettyCb);
        if (ret == 0) {
            goto jetty_found;
        }
        RS_PTHREAD_MUTEX_ULOCK(&devCbCurr->mutex);
    }

    hccp_run_warn("get jetty_cb unsuccessful, ueInfo:0x%x, jettyId:%u", ueInfo, jettyId);
    return -ENODEV;

jetty_found:
    RsListDel(&jettyCb->list);
    devCbCurr->jettyCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&devCbCurr->mutex);

    if (jettyCb->state == RS_JETTY_STATE_BIND) {
        hccp_info_rma("jetty_id:%u will be unbind, devIndex:0x%x", jettyId, devCbCurr->index);
        (void)RsUrmaUnbindJetty(jettyCb->jetty);
    }

    if (jettyCb->state == RS_JETTY_STATE_BIND || jettyCb->state == RS_JETTY_STATE_CREATED) {
        hccp_info_rma("jetty_id:%u will be destroyed, devIndex:0x%x", jettyId, devCbCurr->index);
        RsUbCtxDrvJettyDelete(jettyCb);
    }

    RsUbCtxFreeJettyCb(jettyCb);
    return 0;
}

STATIC int RsUbGetRemJettyCb(struct RsUbDevCb *devCb, urma_jetty_id_t *remJettyId, struct RsCtxRemJettyCb **rjettyCb)
{
    struct RsCtxRemJettyCb **tempJettyCb = rjettyCb;
    struct RsCtxRemJettyCb *jettyCbCurr = NULL;
    struct RsCtxRemJettyCb *jettyCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(jettyCbCurr, jettyCbNext, &devCb->rjettyList, list, struct RsCtxRemJettyCb);
    for (; (&jettyCbCurr->list) != &devCb->rjettyList;
         jettyCbCurr = jettyCbNext, jettyCbNext = list_entry(jettyCbNext->list.next, struct RsCtxRemJettyCb, list)) {
        if (jettyCbCurr->tjetty == NULL) {
            hccp_warn_rma("devIndex:0x%x remJettyId:%u tjetty is NULL", devCb->index, remJettyId->id);
            continue;
        }
        if (memcmp(&jettyCbCurr->tjetty->id, remJettyId, sizeof(urma_jetty_id_t)) == 0) {
            *tempJettyCb = jettyCbCurr;
            return 0;
        }
    }

    *tempJettyCb = NULL;
    hccp_warn_rma("devIndex:0x%x remJettyId:%u is not available!", devCb->index, remJettyId->id);
    return -ENODEV;
}

STATIC int RsUbCtxInitRjettyCb(struct RsUbDevCb *devCb, struct RsJettyImportAttr *importAttr,
    struct RsCtxRemJettyCb **rjettyCb)
{
    struct RsCtxRemJettyCb *tmpRjettyCb = NULL;
    urma_jetty_id_t remJettyId = {0};
    int ret;

    ret = memcpy_s(&remJettyId, sizeof(remJettyId), importAttr->key.value, REM_JETTY_ID_SIZE);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("memcpy_s remJettyId failed, size:%u sizeof(remJettyId):%zu", REM_JETTY_ID_SIZE, sizeof(remJettyId)),
        -ESAFEFUNC);
    ret = RsUbGetRemJettyCb(devCb, &remJettyId, &tmpRjettyCb);
    CHK_PRT_RETURN(ret == 0,
        hccp_err("devIndex:0x%x remJettyId:%u not support to import again", ret, devCb->index, remJettyId.id), -EEXIST);

    tmpRjettyCb = calloc(1, sizeof(struct RsCtxRemJettyCb));
    CHK_PRT_RETURN(tmpRjettyCb == NULL, hccp_err("calloc tmpRjettyCb failed, errno:%d", errno), -ENOMEM);

    tmpRjettyCb->devCb = devCb;
    tmpRjettyCb->jettyKey = importAttr->key;
    tmpRjettyCb->mode = importAttr->attr.mode;
    tmpRjettyCb->tokenValue = importAttr->attr.tokenValue;
    tmpRjettyCb->policy = importAttr->attr.policy;
    tmpRjettyCb->type = importAttr->attr.type;
    tmpRjettyCb->flag = importAttr->attr.flag;
    tmpRjettyCb->tpType = importAttr->attr.tpType;
    tmpRjettyCb->expImportCfg = importAttr->attr.expImportCfg;

    *rjettyCb = tmpRjettyCb;
    return 0;
}

STATIC void RsUbCtxExpJettyImport(struct RsCtxRemJettyCb *rjettyCb, urma_rjetty_t *rjetty, urma_token_t *tokenValue)
{
    urma_import_jetty_ex_cfg_t expCfg = {0};

    expCfg.tp_handle = rjettyCb->expImportCfg.tpHandle;
    expCfg.peer_tp_handle = rjettyCb->expImportCfg.peerTpHandle;
    expCfg.tag = rjettyCb->expImportCfg.tag;
    expCfg.tp_attr.tx_psn = rjettyCb->expImportCfg.txPsn;
    expCfg.tp_attr.rx_psn = rjettyCb->expImportCfg.rxPsn;

    rjettyCb->tjetty = RsUrmaImportJettyEx(rjettyCb->devCb->urmaCtx, rjetty, tokenValue, &expCfg);
}

STATIC int RsUbCtxDrvJettyImport(struct RsCtxRemJettyCb *rjettyCb)
{
    struct RsJettyKeyInfo *jettyKeyInfo;
    urma_token_t tokenValue = {0};
    urma_rjetty_t rjetty = {0};

    tokenValue.token = rjettyCb->tokenValue;
    jettyKeyInfo = (struct RsJettyKeyInfo *)rjettyCb->jettyKey.value;
    rjetty.jetty_id = jettyKeyInfo->jettyId;
    rjetty.trans_mode = jettyKeyInfo->transMode;
    rjetty.policy = (urma_jetty_grp_policy_t)rjettyCb->policy;
    rjetty.type = (urma_target_type_t)rjettyCb->type;
    rjetty.flag.value = rjettyCb->flag.value;
    rjetty.tp_type = rjettyCb->tpType;

    if (rjettyCb->mode == JETTY_IMPORT_MODE_NORMAL) {
        rjettyCb->tjetty = RsUrmaImportJetty(rjettyCb->devCb->urmaCtx, &rjetty, &tokenValue);
    } else { // rjetty_cb->mode == JETTY_IMPORT_MODE_EXP
        RsUbCtxExpJettyImport(rjettyCb, &rjetty, &tokenValue);
    }
    CHK_PRT_RETURN(rjettyCb->tjetty == NULL, hccp_err("import_jetty failed, mode:%d errno:%d", rjettyCb->mode, errno),
        -EOPENSRC);
    return 0;
}

int RsUbCtxJettyImport(struct RsUbDevCb *devCb, struct RsJettyImportAttr *importAttr,
    struct RsJettyImportInfo *importInfo)
{
    struct RsCtxRemJettyCb *rjettyCb = NULL;
    int ret;

    ret = RsUbCtxInitRjettyCb(devCb, importAttr, &rjettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsUbCtxInitRjettyCb failed, ret:%d devIndex:0x%x", ret, devCb->index), ret);

    ret = RsUbCtxDrvJettyImport(rjettyCb);
    if (ret != 0) {
        hccp_err("RsUbCtxDrvJettyImport failed, ret:%d devIndex:0x%x", ret, devCb->index);
        goto free_rjetty_cb;
    }

    importInfo->remJettyId = rjettyCb->tjetty->id.id;
    importInfo->info.tjettyHandle = (uint64_t)(uintptr_t)rjettyCb->tjetty;
    importInfo->info.tpn = rjettyCb->tjetty->tp.tpn;

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListAddTail(&rjettyCb->list, &devCb->rjettyList);
    devCb->rjettyCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    rjettyCb->state = RS_JETTY_STATE_IMPORTED;

    hccp_run_info("devIndex:0x%x remJettyId:%u rjettyHandle:0x%llx mode:%d import success, rjettyCnt:%u", devCb->index,
        importInfo->remJettyId, rjettyCb->tjetty->handle, importAttr->attr.mode, devCb->rjettyCnt);

    return 0;

free_rjetty_cb:
    free(rjettyCb);
    rjettyCb = NULL;

    return ret;
}

STATIC int RsUbGetRemJettyCbDeprecated(struct RsUbDevCb *devCb, unsigned int remJettyId,
    struct RsCtxRemJettyCb **rjettyCb)
{
    struct RsCtxRemJettyCb **tempJettyCb = rjettyCb;
    struct RsCtxRemJettyCb *jettyCbCurr = NULL;
    struct RsCtxRemJettyCb *jettyCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(jettyCbCurr, jettyCbNext, &devCb->rjettyList, list, struct RsCtxRemJettyCb);
    for (; (&jettyCbCurr->list) != &devCb->rjettyList;
         jettyCbCurr = jettyCbNext, jettyCbNext = list_entry(jettyCbNext->list.next, struct RsCtxRemJettyCb, list)) {
        if (jettyCbCurr->tjetty == NULL) {
            hccp_warn_rma("rem_jetty_id:%u jetty_cb_curr->tjetty is NULL", remJettyId);
            continue;
        }
        if (jettyCbCurr->tjetty->id.id == remJettyId) {
            *tempJettyCb = jettyCbCurr;
            return 0;
        }
    }

    *tempJettyCb = NULL;
    hccp_err("rjetty_cb for rem_jetty %u is not available!", remJettyId);

    return -ENODEV;
}

int RsUbCtxJettyUnimport(struct RsUbDevCb *devCb, unsigned char rawRemJettyId[], unsigned int size)
{
    struct RsCtxRemJettyCb *rjettyCb = NULL;
    urma_jetty_id_t remJettyId = {0};
    unsigned int rjettyCnt = 0;
    uint64_t rjettyHandle = 0;
    int ret;

    if (size == REM_JETTY_ID_DEPRECATED_SIZE) {
        (void)memcpy_s(&remJettyId.id, size, rawRemJettyId, size);
        ret = RsUbGetRemJettyCbDeprecated(devCb, remJettyId.id, &rjettyCb);
    } else { // size == REM_JETTY_ID_SIZE
        ret = memcpy_s(&remJettyId, sizeof(remJettyId), rawRemJettyId, size);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("memcpy_s remJettyId failed, size:%u sizeof(remJettyId):%u", size, sizeof(remJettyId)),
            -ESAFEFUNC);
        ret = RsUbGetRemJettyCb(devCb, &remJettyId, &rjettyCb);
    }
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsUbGetRemJettyCb failed, ret:%d devIndex:0x%x remJettyId:%u size:%u", ret, devCb->index,
            remJettyId.id, size),
        ret);

    rjettyHandle = rjettyCb->tjetty->handle;
    CHK_PRT_RETURN(rjettyCb->state != RS_JETTY_STATE_IMPORTED,
        hccp_err("unimport failed, remJettyId:%u "
                 "rjettyHandle:0x%llx state:%u",
            remJettyId.id, rjettyHandle, rjettyCb->state),
        -EINVAL);

    RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
    RsListDel(&rjettyCb->list);
    devCb->rjettyCnt--;
    rjettyCnt = devCb->rjettyCnt;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

    ret = RsUrmaUnimportJetty(rjettyCb->tjetty);
    if (ret != 0) {
        hccp_err("urma_unimport_jetty failed, ret:%d devIndex:0x%x remJettyId:%u rjettyHandle:0x%llx", ret,
            devCb->index, remJettyId.id, rjettyHandle);
        goto out;
    }

    hccp_run_info("devIndex:0x%x remJettyId:%u rjettyHandle:0x%llx unimport success, rjettyCnt:%u", devCb->index,
        remJettyId.id, rjettyHandle, rjettyCnt);
out:
    free(rjettyCb);
    rjettyCb = NULL;
    return ret;
}

int RsUbCtxJettyBind(struct RsUbDevCb *devCb, struct RsCtxQpInfo *jettyInfo, struct RsCtxQpInfo *rjettyInfo)
{
    struct RsCtxRemJettyCb *rjettyCb = NULL;
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret;

    ret = RsUbGetJettyCb(devCb, jettyInfo->id, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get jetty_cb failed, ret:%d, jettyId %u", ret, jettyInfo->id), ret);

    ret = RsUbGetRemJettyCbDeprecated(devCb, rjettyInfo->id, &rjettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rjetty_cb failed, ret:%d, remJettyId %u", ret, rjettyInfo->id), ret);

    if (jettyCb->state != RS_JETTY_STATE_CREATED || rjettyCb->state != RS_JETTY_STATE_IMPORTED) {
        hccp_err("local jetty id:%u state:%u or remote jetty id:%u state:%u not support to bind", jettyInfo->id,
            jettyCb->state, rjettyInfo->id, rjettyCb->state);
        return -EINVAL;
    }

    ret = RsUrmaBindJetty(jettyCb->jetty, rjettyCb->tjetty);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_bind_jetty failed, ret:%d errno:%d", ret, errno), -EOPENSRC);

    jettyCb->state = RS_JETTY_STATE_BIND;

    hccp_run_info("rs ctx local jetty %u bind rem_jetty %u success, devIndex:0x%x", jettyInfo->id, rjettyInfo->id,
        devCb->index);

    return 0;
}

int RsUbCtxJettyUnbind(struct RsUbDevCb *devCb, unsigned int jettyId)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    int ret;

    ret = RsUbGetJettyCb(devCb, jettyId, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_run_warn("get jetty_cb unsuccessful, ret:%d, jettyId %u", ret, jettyId), ret);
    if (jettyCb->state != RS_JETTY_STATE_BIND) {
        hccp_err("jetty_cb->state:%u not support to unbind, jettyId:%u", jettyCb->state, jettyId);
        return -EINVAL;
    }

    ret = RsUrmaUnbindJetty(jettyCb->jetty);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_unbind_jetty failed, ret:%d jettyId %u", ret, jettyId), -EOPENSRC);

    jettyCb->state = RS_JETTY_STATE_CREATED;

    hccp_run_info("rs ctx local jetty %u unbind rem jetty success, devIndex:0x%x", jettyId, devCb->index);

    return 0;
}

STATIC int RsUbCtxFillLsge(struct RsUbDevCb *devCb, urma_sge_t *lsge, struct BatchSendWrData *wrData,
    unsigned int *totalLen, bool isInline)
{
    struct RsSegCb *segCb = NULL;
    unsigned int totalLenTmp = 0;
    unsigned int i;
    int ret;

    if (isInline == false) {
        for (i = 0; i < wrData->numSge; i++) {
            lsge[i].addr = wrData->sges[i].addr;
            lsge[i].len = wrData->sges[i].len;
            totalLenTmp += lsge[i].len;
            ret = RsUbQuerySegCb(devCb, wrData->sges[i].devLmemHandle, &segCb, &devCb->lsegList);
            CHK_PRT_RETURN(ret != 0,
                hccp_err("[send][rs_ub_ctx]can not find lmem seg cb for addr:0x%llx", lsge[i].addr), ret);
            lsge[i].tseg = segCb->segment;
        }
    } else {
        lsge[0].addr = (uint64_t)(uintptr_t)wrData->inlineData;
        lsge[0].len = wrData->inlineSize;
        lsge[0].tseg = NULL;
    }

    *totalLen = totalLenTmp;
    return 0;
}

STATIC int RsUbCtxFillRsge(struct RsUbDevCb *devCb, urma_sge_t *rsge, struct BatchSendWrData *wrData,
    unsigned int totalLen, urma_opcode_t opcode)
{
    struct RsSegCb *segCb = NULL;
    int ret;

    rsge[0].addr = wrData->remoteAddr;
    rsge[0].len = totalLen;
    ret = RsUbQuerySegCb(devCb, wrData->devRmemHandle, &segCb, &devCb->rsegList);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][rs_ub_ctx]can not find rmem seg cb for addr:0x%llx", rsge[0].addr), ret);
    rsge[0].tseg = segCb->segment;

    if (opcode == URMA_OPC_WRITE_NOTIFY) {
        rsge[1].addr = wrData->ub.notifyInfo.notifyAddr;
        rsge[1].len = 8; /* notify data is fixed 8 bytes */
        ret = RsUbQuerySegCb(devCb, wrData->ub.notifyInfo.notifyHandle, &segCb, &devCb->rsegList);
        CHK_PRT_RETURN(ret != 0, hccp_err("[send][rs_ub_ctx]can not find rmem seg cb for addr:0x%llx", rsge[1].addr),
            ret);
        rsge[1].tseg = segCb->segment;
    }

    return 0;
}

STATIC int RsUbCtxInitRwWr(struct RsUbDevCb *devCb, urma_jfs_wr_t *ubWr, struct BatchSendWrData *wrData,
    urma_sge_t *lsge, urma_sge_t *rsge)
{
    unsigned int lsgeNum, rsgeNum;
    unsigned int totalLen = 0;
    int ret;

    lsgeNum = (ubWr->flag.bs.inline_flag == 0) ? wrData->numSge : 1;
    ret = RsUbCtxFillLsge(devCb, lsge, wrData, &totalLen, ubWr->flag.bs.inline_flag);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][rs_ub_ctx]fill lsge failed, ret:%d", ret), ret);

    /* write with notify have 2 dst sge, sge[0] is data sge, sge[1] is notify sge */
    rsgeNum = (ubWr->opcode == URMA_OPC_WRITE_NOTIFY) ? 2 : 1;
    ret = RsUbCtxFillRsge(devCb, rsge, wrData, totalLen, ubWr->opcode);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][rs_ub_ctx]fill rsge failed, ret:%d", ret), ret);

    if (ubWr->opcode == URMA_OPC_READ) {
        ubWr->rw.src.sge = rsge;
        ubWr->rw.src.num_sge = rsgeNum;
        ubWr->rw.dst.sge = lsge;
        ubWr->rw.dst.num_sge = lsgeNum;
    } else {
        ubWr->rw.src.sge = lsge;
        ubWr->rw.src.num_sge = lsgeNum;
        ubWr->rw.dst.sge = rsge;
        ubWr->rw.dst.num_sge = rsgeNum;
    }

    // assign notify data & imm data
    if (ubWr->opcode == URMA_OPC_WRITE_NOTIFY) {
        ubWr->rw.notify_data = wrData->ub.notifyInfo.notifyData;
    } else {
        ubWr->rw.notify_data = (uint64_t)wrData->immData;
    }

    return 0;
}

STATIC int RsUbCtxInitJfsWr(struct RsCtxJettyCb *jettyCb, struct udma_u_jfs_wr_ex *ubWr, struct BatchSendWrData *wrData,
    urma_sge_t *lsge, urma_sge_t *rsge)
{
    urma_opcode_t opcode = (urma_opcode_t)wrData->ub.opcode;
    struct RsCtxRemJettyCb *rjettyCb = NULL;
    int ret;

    if (wrData->numSge > MAX_SGE_NUM) {
        hccp_err("[send][rs_ub_ctx]num_sge is invalid, numSge[%d]", wrData->numSge);
        return -EINVAL;
    }

    ubWr->wr.opcode = opcode;
    ubWr->wr.flag.value = wrData->ub.flags.value;
    ubWr->wr.user_ctx = wrData->ub.userCtx;

    if (opcode == URMA_OPC_NOP) {
        return 0;
    }

    // only write & write with notify & read op support inline reduce
    if (opcode == URMA_OPC_READ || opcode == URMA_OPC_WRITE || opcode == URMA_OPC_WRITE_NOTIFY) {
        ubWr->reduce_en = wrData->ub.reduceInfo.reduceEn;
        ubWr->reduce_opcode = wrData->ub.reduceInfo.reduceOpcode;
        ubWr->reduce_data_type = wrData->ub.reduceInfo.reduceDataType;
        hccp_dbg("[send][rs_ub_ctx]opcode[%d] reduce_en[%d] reduce_opcode[%d] reduce_data_type[%d]", opcode,
            ubWr->reduce_en, ubWr->reduce_opcode, ubWr->reduce_data_type);
    }

    if (opcode == URMA_OPC_READ || opcode == URMA_OPC_WRITE || opcode == URMA_OPC_WRITE_NOTIFY ||
        opcode == URMA_OPC_WRITE_IMM) {
        ret = RsUbGetRemJettyCbDeprecated(jettyCb->devCb, (unsigned int)wrData->ub.remJetty, &rjettyCb);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("[send][rs_ub_ctx]get rjetty_cb failed, ret:%d remJettyId:%llu", ret, wrData->ub.remJetty), ret);

        ubWr->wr.tjetty = rjettyCb->tjetty;
        return RsUbCtxInitRwWr(jettyCb->devCb, &ubWr->wr, wrData, lsge, rsge);
    }

    hccp_err("[send][rs_ub_ctx]invalid opcode[%d]", opcode);
    return -EINVAL;
}

STATIC int RsUbCtxBatchSendWrExt(struct RsCtxJettyCb *jettyCb, struct BatchSendWrData *wrData,
    struct SendWrResp *wrResp, struct WrlistSendCompleteNum *wrlistNum)
{
#define UB_DWQE_BB_NUM 2U
#define UB_DWQE_BB_SIZE 64U
    unsigned int sendNum = wrlistNum->sendNum;
    struct udma_u_jfs_wr_ex *badWr = NULL;
    struct udma_u_post_info wrOut = {0};
    urma_sge_t rsge[MAX_RSGE_NUM] = {0};
    urma_sge_t lsge[MAX_SGE_NUM] = {0};
    struct udma_u_jfs_wr_ex ubWr = {0};
    struct udma_u_wr_ex wrIn = {0};
    urma_user_ctl_out_t out = {0};
    urma_user_ctl_in_t in = {0};
    unsigned int i;
    int ret = 0;

    for (i = 0; i < sendNum; i++) {
        ret = RsUbCtxInitJfsWr(jettyCb, &ubWr, &wrData[i], &lsge[0], &rsge[0]);
        if (ret != 0) {
            hccp_err("[send][rs_ub_ctx]init jfs wr failed, ret:%d, curr_num[%u], sendNum[%u]", ret, i, sendNum);
            break;
        }

        wrIn.is_jetty = true;
        wrIn.jetty = jettyCb->jetty;
        wrIn.wr = &ubWr;
        wrIn.bad_wr = &badWr;

        in.addr = (uint64_t)(uintptr_t)&wrIn;
        in.len = (uint32_t)sizeof(struct udma_u_wr_ex);
        in.opcode = UDMA_U_USER_CTL_POST_WR;

        out.addr = (uint64_t)(uintptr_t)&wrOut;
        out.len = sizeof(struct udma_u_post_info);

        ret = RsUrmaUserCtl(jettyCb->devCb->urmaCtx, &in, &out);
        if (ret != 0) {
            hccp_err("rs_urma_user_ctl batch send wr failed, ret:%d, wr[%u], sendNum[%u] errno:%d", ret, i, sendNum,
                errno);
            ret = -EOPENSRC;
            break;
        }

        wrResp[i].doorbellInfo.dieId = (uint16_t)jettyCb->devCb->devAttr.ub.dieId;
        wrResp[i].doorbellInfo.funcId = (uint16_t)jettyCb->devCb->devAttr.ub.funcId;
        wrResp[i].doorbellInfo.jettyId = (uint16_t)jettyCb->jetty->jetty_id.id;
        wrResp[i].doorbellInfo.piVal = (uint16_t)wrOut.pi;
        // prepare dwqe doorbell info, only support 2BB, each BB size is 64B
        if (wrOut.pi - jettyCb->lastPi <= UB_DWQE_BB_NUM) {
            wrResp[i].doorbellInfo.dwqeSize = (uint16_t)(wrOut.pi - jettyCb->lastPi) * UB_DWQE_BB_SIZE;
            ret = memcpy_s(wrResp[i].doorbellInfo.dwqe, wrResp[i].doorbellInfo.dwqeSize, wrOut.ctrl,
                wrResp[i].doorbellInfo.dwqeSize);
            if (ret != 0) {
                hccp_err("[send][rs_ub_ctx]memcpy_s failed, ret:%d, wr[%u], sendNum[%u]", ret, i, sendNum);
                ret = -ESAFEFUNC;
                break;
            }
        }
        jettyCb->lastPi = wrOut.pi;

        // record doorbell info
        hccp_dbg("jetty_id %u post info: dwqe_addr:%pK, dbAddr:%pK, ctrl:%pK, pi:%u", jettyCb->jetty->jetty_id.id,
            wrOut.dwqe_addr, wrOut.db_addr, wrOut.ctrl, wrOut.pi);
    }

    *wrlistNum->completeNum = i;
    return ret;
}

int RsUbCtxBatchSendWr(struct rs_cb *rsCb, struct WrlistBaseInfo *baseInfo, struct BatchSendWrData *wrData,
    struct SendWrResp *wrResp, struct WrlistSendCompleteNum *wrlistNum)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    struct RsUbDevCb *devCb = NULL;
    int ret;

    if (wrlistNum->sendNum > MAX_CTX_WR_NUM || wrlistNum->sendNum == 0) {
        hccp_err("[send][rs_ub_ctx] send_num[%u] is invalid", wrlistNum->sendNum);
        return -EINVAL;
    }

    ret = RsUbGetDevCb(rsCb, baseInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[send][rs_ub_ctx]get dev_cb failed, ret:%d, devIndex:0x%x", ret, baseInfo->devIndex), ret);

    ret = RsUbGetJettyCb(devCb, baseInfo->qpn, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("[send][rs_ub_ctx]get jetty_cb failed, ret:%d, jettyId[%u]", ret, baseInfo->qpn),
        ret);

    ret = RsUbCtxBatchSendWrExt(jettyCb, wrData, wrResp, wrlistNum);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[send][rs_ub_ctx]send wr ext failed, ret:%d, sendNum[%u], completeNum[%u]", ret, wrlistNum->sendNum,
            *wrlistNum->completeNum),
        ret);

    return 0;
}

int RsUbCtxJettyUpdateCi(struct RsUbDevCb *devCb, unsigned int jettyId, uint16_t ci)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    struct udma_u_update_ci ciData = {0};
    urma_user_ctl_out_t out = {0};
    urma_user_ctl_in_t in = {0};
    int ret;

    ret = RsUbGetJettyCb(devCb, jettyId, &jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get jetty_cb failed, ret:%d, jettyId:%u", ret, jettyId), ret);

    ciData.is_jetty = true;
    ciData.ci = ci;
    ciData.jetty = jettyCb->jetty;
    in.addr = (uint64_t)(uintptr_t)&ciData;
    in.len = (uint32_t)sizeof(struct udma_u_update_ci);
    in.opcode = UDMA_U_USER_CTL_UPDATE_CI;
    ret = RsUrmaUserCtl(jettyCb->devCb->urmaCtx, &in, &out);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_user_ctl update ci failed, ret:%d errno:%d jettyId:%u ci:%u", ret, errno, jettyId, ci),
        -EOPENSRC);

    hccp_info_rma("[update_ci]devIndex:0x%x jetty_id:%u update ci:%u success", devCb->index, jettyId, ci);

    return 0;
}

STATIC int RsUbGetJettyDestroyBatchInfo(struct RsUbDevCb *devCb, unsigned int jettyIds[],
    struct JettyDestroyBatchInfo *batchInfo, unsigned int *num)
{
    unsigned int i;
    int ret = 0;

    for (i = 0; i < *num; ++i) {
        ret = RsUbGetJettyCb(devCb, jettyIds[i], &batchInfo->jettyCbArr[i]);
        CHK_PRT_RETURN(ret != 0, hccp_err("get jetty_cb[%u] failed, jettyId:%u, ret:%d", i, jettyIds[i], ret), ret);
        CHK_PRT_RETURN(batchInfo->jettyCbArr[i]->state != RS_JETTY_STATE_CREATED,
            hccp_err("jetty_cb[%u]->state:%u "
                     "not support to destroy, jettyId:%u",
                i, batchInfo->jettyCbArr[i]->state, jettyIds[i]),
            -EINVAL);

        RS_PTHREAD_MUTEX_LOCK(&devCb->mutex);
        RsListDel(&batchInfo->jettyCbArr[i]->list);
        devCb->jettyCnt--;
        RS_PTHREAD_MUTEX_ULOCK(&devCb->mutex);

        batchInfo->jettyArr[i] = batchInfo->jettyCbArr[i]->jetty;
    }

    return ret;
}

int RsUbCtxJettyDestroyBatch(struct RsUbDevCb *devCb, unsigned int jettyIds[], unsigned int *num)
{
    struct JettyDestroyBatchInfo batchInfo = {0};
    int ret;

    CHK_PRT_RETURN(*num == 0, hccp_err("num(%u) = 0, no need to destroy batch", *num), -EINVAL);

    ret = RsUbCallocJettyBatchInfo(&batchInfo, *num);
    if (ret != 0) {
        *num = 0;
        hccp_err("rs_ub_calloc_jetty_batch_info failed, ret:%d", ret);
        goto free_batch_info;
    }

    ret = RsUbGetJettyDestroyBatchInfo(devCb, jettyIds, &batchInfo, num);
    if (ret != 0) {
        *num = 0;
        hccp_err("get jetty destroy batch info failed, ret:%d", ret);
        goto free_batch_info;
    }

    ret = RsUbDestroyJettyCbBatch(&batchInfo, num);
    if (ret != 0) {
        hccp_err("rs_ub_destroy_jetty_cb_batch failed, ret:%d", ret);
    }

free_batch_info:
    RsUbFreeJettyBatchInfo(&batchInfo);
    return ret;
}

int RsUbCtxQueryJettyBatch(struct RsUbDevCb *devCb, unsigned int jettyIds[], struct JettyAttr attr[], unsigned int *num)
{
    struct RsCtxJettyCb *jettyCb = NULL;
    urma_jetty_attr_t attrTmp = {0};
    urma_jetty_cfg_t cfg = {0};
    unsigned int i = 0;
    int ret = 0;

    for (i = 0; i < *num; ++i) {
        (void)memset_s(&attrTmp, sizeof(urma_jetty_attr_t), 0, sizeof(urma_jetty_attr_t));
        ret = RsUbGetJettyCb(devCb, jettyIds[i], &jettyCb);
        if (ret != 0) {
            hccp_err("get jetty_cb failed, ret:%d, jettyId[%u]:%u", ret, i, jettyIds[i]);
            break;
        }

        ret = RsUrmaQueryJetty(jettyCb->jetty, &cfg, &attrTmp);
        if (ret != 0) {
            hccp_err("rs_urma_query_jetty failed, ret:%d, jettyId[%u]:%u", ret, i, jettyIds[i]);
            break;
        }

        (void)memcpy_s(&attr[i], sizeof(urma_jetty_attr_t), &attrTmp, sizeof(urma_jetty_attr_t));
    }

    *num = i;
    return ret;
}
