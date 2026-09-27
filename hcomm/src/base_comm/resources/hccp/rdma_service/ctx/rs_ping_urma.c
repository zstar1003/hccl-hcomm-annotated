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
#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <errno.h>
#include <urma_opcode.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "dl_urma_function.h"
#include "dl_net_function.h"
#include "hccp_common.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "ra_rs_comm.h"
#include "rs_inner.h"
#include "rs_epoll.h"
#include "rs_ub.h"
#include "rs_ping_inner.h"
#include "rs_ping_urma.h"

urma_cr_t gPingJettyRecvCr[RS_PING_URMA_RECV_WC_NUM] = {0};
urma_cr_t gPongJettyRecvCr[RS_PING_URMA_RECV_WC_NUM] = {0};

STATIC bool RsPingUrmaCheckFd(struct RsPingCtxCb *pingCb, int fd)
{
    if (pingCb->pingJetty.jfce != NULL && pingCb->pingJetty.jfce->fd == fd) {
        hccp_dbg("ping_jetty jfce->fd:%d poll jfc", fd);
        return true;
    }
    return false;
}

STATIC bool RsPongUrmaCheckFd(struct RsPingCtxCb *pingCb, int fd)
{
    if (pingCb->pongJetty.jfce != NULL && pingCb->pongJetty.jfce->fd == fd) {
        hccp_dbg("pong_jetty jfce->fd:%d poll jfc", fd);
        return true;
    }
    return false;
}

STATIC void RsGetJettyInfo(struct PingQpInfo *qpInfo, urma_jetty_id_t *jettyId, urma_eid_t *eid)
{
    urma_jetty_id_t jettyKeyInfo = {0};
    int ret = 0;

    ret = memcpy_s(&jettyKeyInfo, sizeof(urma_jetty_id_t), qpInfo->ub.key, qpInfo->ub.size);
    if (ret != 0) {
        hccp_err("memcpy jetty_key_info failed, ret:%d", ret);
        return;
    }

    if (jettyId != NULL) {
        *jettyId = jettyKeyInfo;
    }
    if (eid != NULL) {
        *eid = jettyKeyInfo.eid;
    }
}

STATIC bool RsPingCommonCompareUbInfo(struct PingQpInfo *a, struct PingQpInfo *b)
{
    if (a->ub.size != b->ub.size) {
        return false;
    }
    if (memcmp(&a->ub.key, &b->ub.key, sizeof(a->ub.key)) != 0) {
        return false;
    }
    return true;
}

STATIC int RsPingCbGetUrmaContextAndIndex(struct RsPingCtxCb *pingCb, struct PingInitAttr *attr)
{
    struct DevBaseAttr devAttr = {0};
    int ret;

    pingCb->udevCb.urmaCtx = RsUrmaCreateContext(pingCb->udevCb.urmaDev, attr->dev.ub.eidIndex);
    CHK_PRT_RETURN(pingCb->udevCb.urmaCtx == NULL,
        hccp_err("urma_create_context failed, errno:%d, "
                 "eidIndex:%u",
            errno, attr->dev.ub.eidIndex),
        -ENODEV);

    ret = RsUbGetUeInfo(pingCb->udevCb.urmaCtx, &devAttr);
    if (ret != 0) {
        hccp_err("rs_ub_get_ue_info failed, ret:%d errno:%d", ret, errno);
        ret = -EOPENSRC;
        goto free_urma_ctx;
    }

    pingCb->devIndex = RsGenerateDevIndex(PING_URMA_DEV_CNT, devAttr.ub.dieId, devAttr.ub.funcId);
    return 0;

free_urma_ctx:
    (void)RsUrmaDeleteContext(pingCb->udevCb.urmaCtx);
    pingCb->udevCb.urmaCtx = NULL;
    return ret;
}

STATIC int RsPingCommonInitJfce(struct RsPingCtxCb *pingCb, struct RsPingLocalJettyCb *jettyCb)
{
    jettyCb->jfce = RsUrmaCreateJfce(pingCb->udevCb.urmaCtx);
    CHK_PRT_RETURN(jettyCb->jfce == NULL, hccp_err("urma_create_jfce failed, errno:%d", errno), -EOPENSRC);

    hccp_run_info("eid:%016llx:%016llx init jfce success, fd:%d", pingCb->udevCb.eidInfo.eid.in6.subnetPrefix,
        pingCb->udevCb.eidInfo.eid.in6.interfaceId, jettyCb->jfce->fd);
    return 0;
}

STATIC int RsPingCommonInitSendJfcWithAttr(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, union PingQpAttr *attr,
    struct RsPingLocalJettyCb *jettyCb)
{
    (void)rscb;
    urma_jfc_cfg_t sendJfcCfg = {
        .depth = attr->ub.cqAttr.sendCqDepth,
        .flag = {.value = 0},
        .jfce = NULL,
        .user_ctx = 0,
    };
    jettyCb->sendJfc.depth = attr->ub.cqAttr.sendCqDepth;
    jettyCb->sendJfc.numEvents = 0;
    jettyCb->sendJfc.maxRecvWcNum = RS_PING_URMA_RECV_WC_NUM;
    jettyCb->sendJfc.jfc = RsUrmaCreateJfc(pingCb->udevCb.urmaCtx, &sendJfcCfg);
    CHK_PRT_RETURN(jettyCb->sendJfc.jfc == NULL, hccp_err("urma_create_jfc failed, errno:%d", errno), -EOPENSRC);

    hccp_run_info("eid:%016llx:%016llx init send jfc success, jfc_id:%u", pingCb->udevCb.eidInfo.eid.in6.subnetPrefix,
        pingCb->udevCb.eidInfo.eid.in6.interfaceId, jettyCb->sendJfc.jfc->jfc_id.id);
    return 0;
}

STATIC int RsPingCommonInitRecvJfcWithAttr(struct RsPingCtxCb *pingCb, union PingQpAttr *attr,
    struct RsPingLocalJettyCb *jettyCb)
{
    urma_jfc_cfg_t recvJfcCfg = {
        .depth = attr->ub.cqAttr.recvCqDepth,
        .flag = {.value = 0},
        .jfce = jettyCb->jfce,
        .user_ctx = 0,
    };
    jettyCb->recvJfc.depth = attr->ub.cqAttr.recvCqDepth;
    jettyCb->recvJfc.numEvents = 0;
    jettyCb->recvJfc.maxRecvWcNum = RS_PING_URMA_RECV_WC_NUM;
    jettyCb->recvJfc.jfc = RsUrmaCreateJfc(pingCb->udevCb.urmaCtx, &recvJfcCfg);
    CHK_PRT_RETURN(jettyCb->recvJfc.jfc == NULL, hccp_err("create recv jfc failed, errno:%d", errno), -EOPENSRC);

    hccp_run_info("eid:%016llx:%016llx init recv jfc success, jfc_id:%u", pingCb->udevCb.eidInfo.eid.in6.subnetPrefix,
        pingCb->udevCb.eidInfo.eid.in6.interfaceId, jettyCb->recvJfc.jfc->jfc_id.id);
    return 0;
}

STATIC int RsPingCommonInitJettyWithAttr(struct RsPingCtxCb *pingCb, union PingQpAttr *attr,
    struct RsPingLocalJettyCb *jettyCb)
{
    urma_jetty_cfg_t jettyCfg = {0};
    urma_jfs_cfg_t jfsCfg = {0};
    urma_jfr_cfg_t jfrCfg = {0};
    int ret;

    jfsCfg.depth = attr->ub.qpAttr.cap.maxSendWr;
    jfsCfg.trans_mode = URMA_TM_UM;
    jfsCfg.max_sge = (uint8_t)attr->ub.qpAttr.cap.maxSendSge;
    jfsCfg.max_inline_data = attr->ub.qpAttr.cap.maxInlineData;
    jfsCfg.rnr_retry = URMA_TYPICAL_RNR_RETRY;
    jfsCfg.jfc = jettyCb->sendJfc.jfc;
    jfsCfg.user_ctx = 0;

    jfrCfg.depth = attr->ub.qpAttr.cap.maxRecvWr;
    jfrCfg.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    jfrCfg.trans_mode = URMA_TM_UM;
    jfrCfg.max_sge = (uint8_t)attr->ub.qpAttr.cap.maxRecvSge;
    jfrCfg.min_rnr_timer = URMA_TYPICAL_MIN_RNR_TIMER;
    jfrCfg.jfc = jettyCb->recvJfc.jfc;
    jfrCfg.token_value.token = attr->ub.qpAttr.tokenValue;

    jettyCb->jfr = RsUrmaCreateJfr(pingCb->udevCb.urmaCtx, &jfrCfg);
    CHK_PRT_RETURN(jettyCb->jfr == NULL, hccp_err("urma_create_jfr failed, errno:%d", errno), -ENOMEM);

    jettyCfg.flag.bs.share_jfr = URMA_SHARE_JFR;
    jettyCfg.jfs_cfg = jfsCfg;
    jettyCfg.shared.jfr = jettyCb->jfr;
    jettyCfg.shared.jfc = jettyCb->recvJfc.jfc;

    jettyCb->jetty = RsUrmaCreateJetty(pingCb->udevCb.urmaCtx, &jettyCfg);
    if (jettyCb->jetty == NULL) {
        hccp_err("urma_create_jetty failed, errno:%d", errno);
        ret = -ENOMEM;
        goto create_jetty_fail;
    }

    jettyCb->tokenValue = attr->ub.qpAttr.tokenValue;
    hccp_run_info("eid:%016llx:%016llx init jetty success, jetty_id:%u", pingCb->udevCb.eidInfo.eid.in6.subnetPrefix,
        pingCb->udevCb.eidInfo.eid.in6.interfaceId, jettyCb->jetty->jetty_id.id);
    return 0;

create_jetty_fail:
    (void)RsUrmaDeleteJfr(jettyCb->jfr);
    jettyCb->jfr = NULL;
    return ret;
}

STATIC int RsPingCommonInitLocalJetty(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, union PingQpAttr *attr,
    struct RsPingLocalJettyCb *jettyCb)
{
    int ret;

    hccp_info_rma("eid:%016llx:%016llx cap{%u %u %u %u %u} start init local jettys",
        pingCb->udevCb.eidInfo.eid.in6.subnetPrefix, pingCb->udevCb.eidInfo.eid.in6.interfaceId,
        attr->ub.qpAttr.cap.maxSendWr, attr->ub.qpAttr.cap.maxRecvWr, attr->ub.qpAttr.cap.maxSendSge,
        attr->ub.qpAttr.cap.maxRecvSge, attr->ub.qpAttr.cap.maxInlineData);

    ret = RsPingCommonInitJfce(pingCb, jettyCb);
    if (ret != 0) {
        hccp_err("init jfce failed, ret:%d", ret);
        goto init_jfce_fail;
    }

    ret = RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_ADD, jettyCb->jfce->fd, EPOLLIN | EPOLLRDHUP);
    if (ret != 0) {
        hccp_err("RsEpollCtl failed! epollfd:%d fd:%d ret:%d", rscb->connCb.epollfd, jettyCb->jfce->fd, ret);
        goto epoll_ctl_fail;
    }

    ret = RsPingCommonInitSendJfcWithAttr(rscb, pingCb, attr, jettyCb);
    if (ret != 0) {
        hccp_err("init send jfc failed, ret:%d", ret);
        goto init_send_jfc_fail;
    }

    ret = RsPingCommonInitRecvJfcWithAttr(pingCb, attr, jettyCb);
    if (ret != 0) {
        hccp_err("init recv jfc failed, ret:%d", ret);
        goto init_recv_jfc_fail;
    }

    ret = RsPingCommonInitJettyWithAttr(pingCb, attr, jettyCb);
    if (ret != 0) {
        hccp_err("init jetty failed, ret:%d", ret);
        goto init_jetty_fail;
    }
    return 0;

init_jetty_fail:
    (void)RsUrmaDeleteJfc(jettyCb->recvJfc.jfc);
    jettyCb->recvJfc.jfc = NULL;
init_recv_jfc_fail:
    (void)RsUrmaDeleteJfc(jettyCb->sendJfc.jfc);
    jettyCb->sendJfc.jfc = NULL;
init_send_jfc_fail:
    (void)RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, jettyCb->jfce->fd, EPOLLIN | EPOLLRDHUP);
epoll_ctl_fail:
    (void)RsUrmaDeleteJfce(jettyCb->jfce);
    jettyCb->jfce = NULL;
init_jfce_fail:
    return ret;
}

STATIC void RsPingCommonDeinitLocalJetty(struct rs_cb *rscb, struct RsPingCtxCb *pingCb,
    struct RsPingLocalJettyCb *jettyCb)
{
    (void)pingCb;
    (void)RsUrmaDeleteJetty(jettyCb->jetty);
    jettyCb->jetty = NULL;

    (void)RsUrmaDeleteJfr(jettyCb->jfr);
    jettyCb->jfr = NULL;

    (void)RsUrmaDeleteJfc(jettyCb->recvJfc.jfc);
    jettyCb->recvJfc.jfc = NULL;

    (void)RsUrmaDeleteJfc(jettyCb->sendJfc.jfc);
    jettyCb->sendJfc.jfc = NULL;

    (void)RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, jettyCb->jfce->fd, EPOLLIN | EPOLLRDHUP);

    (void)RsUrmaDeleteJfce(jettyCb->jfce);
    jettyCb->jfce = NULL;

    return;
}

STATIC int RsPingCommonInitSegCb(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, struct RsPingSegCb *segCb)
{
    unsigned long flag = 0;
    uint32_t idx = 0;
    int ret;

    urma_reg_seg_flag_t segFlag = {.bs.token_policy = URMA_TOKEN_PLAIN_TEXT,
        .bs.cacheable = URMA_NON_CACHEABLE,
        .bs.access = URMA_ACCESS_LOCAL_ONLY,
        .bs.non_pin = (RaRsHasCapability(RA_CAP_DRV_SHAREPOOL_NON_PIN, 0, RsNetGetApiVersion())) ? 1 : 0,
        .bs.token_id_valid = URMA_TOKEN_ID_INVALID,
        .bs.reserved = 0};
    urma_seg_cfg_t segCfg = {.va = 0,
        .len = segCb->len,
        .token_value = segCb->tokenValue,
        .flag = segFlag,
        .user_ctx = (uintptr_t)NULL,
        .iova = 0};

    hccp_info_rma("payload_offset:%u len:0x%llx sge_num:%u grp_id:%u", segCb->payloadOffset, segCb->len, segCb->sgeNum,
        rscb->grpId);

    ret = pthread_mutex_init(&segCb->mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("pthread_mutex_init seg_cb mutex failed, ret:%d", ret), ret);

    flag = ((unsigned long)pingCb->logicDevid << BUFF_FLAGS_DEVID_OFFSET) | BUFF_SP_SVM;
    ret = DlHalBuffAllocAlignEx(segCb->len, (unsigned int)RA_RS_4K_PAGE_SIZE, flag, (int)rscb->grpId,
        (void **)&segCb->addr);
    if (ret != 0) {
        hccp_err("DlHalBuffAllocAlignEx failed, length:0x%llx, dev_id:0x%x, flag:0x%lx, grp_id:%u, ret:%d", segCb->len,
            pingCb->logicDevid, flag, rscb->grpId, ret);
        goto alloc_fail;
    }

    segCfg.va = segCb->addr;
    segCb->segment = RsUrmaRegisterSeg(pingCb->udevCb.urmaCtx, &segCfg);
    if (segCb->segment == NULL) {
        ret = -errno;
        hccp_err("urma_register_seg failed, ret:%d addr:0x%llx len:0x%llx", ret, segCb->addr, segCb->len);
        goto segment_reg_fail;
    }

    // init sge list
    segCb->sgeList = calloc(segCb->sgeNum, sizeof(urma_sge_t));
    if (segCb->sgeList == NULL) {
        ret = -errno;
        hccp_err("calloc failed, ret:%d sgeNum:%u", ret, segCb->sgeNum);
        goto calloc_fail;
    }

    for (idx = 0; idx < segCb->sgeNum; idx++) {
        segCb->sgeList[idx].tseg = segCb->segment;
        segCb->sgeList[idx].len = segCb->payloadOffset;
        if (idx == 0) {
            segCb->sgeList[idx].addr = segCb->addr;
        } else {
            segCb->sgeList[idx].addr = segCb->sgeList[idx - 1].addr + segCb->payloadOffset;
        }
    }
    segCb->sgeIdx = 0;

    hccp_info_rma("eid:%016llx:%016llx segment register success, addr:0x%llx len:%u",
        pingCb->udevCb.eidInfo.eid.in6.subnetPrefix, pingCb->udevCb.eidInfo.eid.in6.interfaceId, segCb->addr,
        segCb->len);

    return 0;

calloc_fail:
    (void)RsUrmaUnregisterSeg(segCb->segment);
    segCb->segment = NULL;
segment_reg_fail:
    (void)DlHalBuffFree((void *)(uintptr_t)segCb->addr);
alloc_fail:
    (void)pthread_mutex_destroy(&segCb->mutex);
    return ret;
}

STATIC void RsPingCommonDeinitSegCb(struct RsPingSegCb *segCb)
{
    hccp_dbg("addr:0x%llx len:%llu", segCb->addr, segCb->len);

    free(segCb->sgeList);
    segCb->sgeList = NULL;

    (void)RsUrmaUnregisterSeg(segCb->segment);
    segCb->segment = NULL;

    (void)DlHalBuffFree((void *)(uintptr_t)segCb->addr);

    (void)pthread_mutex_destroy(&segCb->mutex);
}

STATIC int RsPingPongInitLocalJettyBuffer(struct rs_cb *rscb, struct PingInitAttr *attr, struct PingInitInfo *info,
    struct RsPingCtxCb *pingCb)
{
    int ret;

    // prepare ping_jetty send segment
    pingCb->pingJetty.sendSegCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pingJetty.sendSegCb.len = attr->client.ub.qpAttr.cap.maxSendWr * pingCb->pingJetty.sendSegCb.payloadOffset;
    pingCb->pingJetty.sendSegCb.sgeNum = attr->client.ub.qpAttr.cap.maxSendWr;
    pingCb->pingJetty.sendSegCb.tokenValue.token = attr->client.ub.segAttr.tokenValue;
    ret = RsPingCommonInitSegCb(rscb, pingCb, &pingCb->pingJetty.sendSegCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ping_common_init_seg_cb ping_jetty send_seg_cb failed, ret %d", ret), ret);

    // prepare ping_jetty recv segment
    pingCb->pingJetty.recvSegCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pingJetty.recvSegCb.len = attr->client.ub.qpAttr.cap.maxRecvWr * pingCb->pingJetty.recvSegCb.payloadOffset;
    pingCb->pingJetty.recvSegCb.sgeNum = attr->client.ub.qpAttr.cap.maxRecvWr;
    pingCb->pingJetty.recvSegCb.tokenValue.token = attr->client.ub.segAttr.tokenValue;
    ret = RsPingCommonInitSegCb(rscb, pingCb, &pingCb->pingJetty.recvSegCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_seg_cb ping_jetty recv_seg_cb failed, ret %d", ret);
        goto init_ping_jetty_recv_seg_fail;
    }

    // prepare pong_jetty send segment
    pingCb->pongJetty.sendSegCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pongJetty.sendSegCb.len = attr->server.ub.qpAttr.cap.maxSendWr * pingCb->pongJetty.sendSegCb.payloadOffset;
    pingCb->pongJetty.sendSegCb.sgeNum = attr->server.ub.qpAttr.cap.maxSendWr;
    pingCb->pongJetty.sendSegCb.tokenValue.token = attr->server.ub.segAttr.tokenValue;
    ret = RsPingCommonInitSegCb(rscb, pingCb, &pingCb->pongJetty.sendSegCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_seg_cb pong_jetty send_seg_cb failed, ret %d", ret);
        goto init_pong_jetty_send_seg_fail;
    }
    // prepare pong_jetty recv segment
    pingCb->pongJetty.recvSegCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pongJetty.recvSegCb.len = attr->bufferSize;
    pingCb->pongJetty.recvSegCb.sgeNum = attr->bufferSize / pingCb->pongJetty.recvSegCb.payloadOffset;
    pingCb->pongJetty.recvSegCb.tokenValue.token = attr->server.ub.segAttr.tokenValue;
    ret = RsPingCommonInitSegCb(rscb, pingCb, &pingCb->pongJetty.recvSegCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_seg_cb pong_jetty recv_seg_cb failed, ret %d", ret);
        goto init_pong_jetty_recv_seg_fail;
    }
    info->result.bufferVa = pingCb->pongJetty.recvSegCb.addr;
    info->result.bufferSize = attr->bufferSize;
    info->result.payloadOffset = pingCb->pongJetty.recvSegCb.payloadOffset;
    info->result.headerSize = RS_PING_PAYLOAD_HEADER_RESV_CUSTOM;
    return 0;

init_pong_jetty_recv_seg_fail:
    RsPingCommonDeinitSegCb(&pingCb->pongJetty.sendSegCb);
init_pong_jetty_send_seg_fail:
    RsPingCommonDeinitSegCb(&pingCb->pingJetty.recvSegCb);
init_ping_jetty_recv_seg_fail:
    RsPingCommonDeinitSegCb(&pingCb->pingJetty.sendSegCb);
    return ret;
}

STATIC void RsPingCommonDeinitLocalJettyBuffer(struct RsPingCtxCb *pingCb)
{
    RsPingCommonDeinitSegCb(&pingCb->pongJetty.recvSegCb);
    RsPingCommonDeinitSegCb(&pingCb->pongJetty.sendSegCb);
    RsPingCommonDeinitSegCb(&pingCb->pingJetty.recvSegCb);
    RsPingCommonDeinitSegCb(&pingCb->pingJetty.sendSegCb);
}

STATIC int RsPingCommonJfrPostRecv(struct RsPingLocalJettyCb *jettyCb)
{
    urma_jfr_wr_t *jfrBadWr = NULL;
    urma_jfr_wr_t jfrWr = {0};
    urma_sge_t list = {0};
    uint32_t sgeIdx;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&jettyCb->recvSegCb.mutex);
    sgeIdx = jettyCb->recvSegCb.sgeIdx;
    (void)memcpy_s(&list, sizeof(urma_sge_t), &jettyCb->recvSegCb.sgeList[sgeIdx], sizeof(urma_sge_t));
    jettyCb->recvSegCb.sgeIdx = (sgeIdx + 1) % jettyCb->recvSegCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&jettyCb->recvSegCb.mutex);

    jfrWr.user_ctx = (uintptr_t)sgeIdx;
    jfrWr.next = NULL;
    jfrWr.src.sge = &list;
    jfrWr.src.num_sge = 1;

    ret = RsUrmaPostJettyRecvWr(jettyCb->jetty, &jfrWr, &jfrBadWr);
    if (ret != 0) {
        hccp_err("urma_post_jetty_recv_wr failed, ret:%d", ret);
        return ret;
    }

    return 0;
}

STATIC int RsPingCommonInitJettyPostRecvAll(struct RsPingLocalJettyCb *jettyCb)
{
    int ret = 0;
    uint32_t i;

    // reset recv jfc notify
    (void)RsUrmaRearmJfc(jettyCb->recvJfc.jfc, false);

    // prepare jfr wqe
    for (i = jettyCb->recvSegCb.sgeIdx; i < jettyCb->recvSegCb.sgeNum && i < jettyCb->jfr->jfr_cfg.depth; i++) {
        ret = RsPingCommonJfrPostRecv(jettyCb);
        if (ret != 0) {
            hccp_err("rs_ping_common_jfr_post_recv %u-th rqe failed, ret:%d", i, ret);
            break;
        }
    }

    return ret;
}

STATIC int RsPingPongInitLocalUbResources(struct rs_cb *rscb, struct PingInitAttr *attr, struct PingInitInfo *info,
    struct RsPingCtxCb *pingCb)
{
    urma_jetty_id_t jettyKey = {0};
    int ret;

    ret = RsPingCommonInitLocalJetty(rscb, pingCb, &attr->client, &pingCb->pingJetty);
    CHK_PRT_RETURN(ret != 0, hccp_err("init ping_jetty failed, ret:%d", ret), ret);
    info->client.version = 0;
    info->client.ub.tokenValue = attr->client.ub.qpAttr.tokenValue;
    info->client.ub.size = (uint8_t)sizeof(urma_jetty_id_t);
    jettyKey = pingCb->pingJetty.jetty->jetty_id;

    ret = memcpy_s(info->client.ub.key, sizeof(info->client.ub.key), &jettyKey, sizeof(urma_jetty_id_t));
    if (ret != 0) {
        hccp_err("memcpy_s urma_jetty_id_t to PingQpInfo.ub.key failed, ret:%d", ret);
        goto init_pong_jetty_fail;
    }

    ret = RsPingCommonInitLocalJetty(rscb, pingCb, &attr->server, &pingCb->pongJetty);
    if (ret != 0) {
        hccp_err("init pong_jetty failed, ret:%d", ret);
        goto init_pong_jetty_fail;
    }
    info->server.version = 0;
    info->server.ub.tokenValue = attr->server.ub.qpAttr.tokenValue;
    info->server.ub.size = (uint8_t)sizeof(urma_jetty_id_t);
    jettyKey = pingCb->pongJetty.jetty->jetty_id;
    (void)memcpy_s(info->server.ub.key, sizeof(info->server.ub.key), &jettyKey, sizeof(urma_jetty_id_t));

    ret = RsPingPongInitLocalJettyBuffer(rscb, attr, info, pingCb);
    if (ret != 0) {
        hccp_err("init jetty buffer failed, ret:%d", ret);
        goto init_buffer_fail;
    }

    ret = RsPingCommonInitJettyPostRecvAll(&pingCb->pingJetty);
    if (ret != 0) {
        hccp_err("ping_jetty post recv failed, ret:%d", ret);
        goto post_recv_fail;
    }
    ret = RsPingCommonInitJettyPostRecvAll(&pingCb->pongJetty);
    if (ret != 0) {
        hccp_err("pong_jetty post recv failed, ret:%d", ret);
        goto post_recv_fail;
    }
    return 0;

post_recv_fail:
    RsPingCommonDeinitLocalJettyBuffer(pingCb);
init_buffer_fail:
    RsPingCommonDeinitLocalJetty(rscb, pingCb, &pingCb->pongJetty);
init_pong_jetty_fail:
    RsPingCommonDeinitLocalJetty(rscb, pingCb, &pingCb->pingJetty);
    return ret;
}

STATIC int RsPingUrmaPingCbInit(unsigned int phyId, struct PingInitAttr *attr, struct PingInitInfo *info,
    unsigned int *devIndex, struct RsPingCtxCb *pingCb)
{
    struct rs_cb *rscb = NULL;
    union urma_eid eid;
    int ret;

    ret = RsGetRsCb(phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret), ret);

    // prepare input attr
    pingCb->udevCb.eidInfo.eidIndex = attr->dev.ub.eidIndex;
    pingCb->udevCb.eidInfo.eid = attr->dev.ub.eid;
    (void)memcpy_s(&pingCb->commInfo, sizeof(struct PingLocalCommInfo), &attr->commInfo,
        sizeof(struct PingLocalCommInfo));

    (void)memcpy_s(eid.raw, sizeof(eid.raw), attr->dev.ub.eid.raw, sizeof(attr->dev.ub.eid.raw));
    pingCb->udevCb.urmaDev = RsUrmaGetDeviceByEid(eid, URMA_TRANSPORT_UB);
    if (pingCb->udevCb.urmaDev == NULL) {
        hccp_err("urma_get_device_by_eid failed, urmaDev is NULL, errno:%d eid:%016llx:%016llx", errno,
            eid.in6.subnet_prefix, eid.in6.interface_id);
        ret = -ENODEV;
        goto get_urma_dev_fail;
    }

    ret = RsPingCbGetUrmaContextAndIndex(pingCb, attr);
    if (ret != 0) {
        hccp_err("rs_ping_cb_get_urma_context_and_index failed, ret:%d", ret);
        goto get_urma_dev_fail;
    }

    info->version = 0;
    ret = RsPingPongInitLocalUbResources(rscb, attr, info, pingCb);
    if (ret != 0) {
        hccp_err("rs_ping_pong_init_local_ub_resources failed, ret:%d phyId:%u", ret, phyId);
        goto init_local_resources_fail;
    }
    *devIndex = pingCb->devIndex;
    return 0;

init_local_resources_fail:
    (void)RsUrmaDeleteContext(pingCb->udevCb.urmaCtx);
    pingCb->udevCb.urmaCtx = NULL;
get_urma_dev_fail:
    (void)pthread_mutex_destroy(&pingCb->pingMutex);
    (void)pthread_mutex_destroy(&pingCb->pongMutex);
    return ret;
}

STATIC int RsPingUrmaFindTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target,
    struct RsPingTargetInfo **node)
{
    struct RsPingTargetInfo *targetNext = NULL;
    struct RsPingTargetInfo *targetCurr = NULL;
    urma_jetty_id_t targetJettyId = {0};
    urma_eid_t targetEid = {0};

    RsGetJettyInfo(target, &targetJettyId, &targetEid);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &pingCb->pingList, list, struct RsPingTargetInfo);
    for (; (&targetCurr->list) != &pingCb->pingList;
         targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPingTargetInfo, list)) {
        if (RsPingCommonCompareUbInfo(&targetCurr->qpInfo, target)) {
            *node = targetCurr;
            RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    hccp_info_rma("ping target node for jetty_id:%u eid:%016llx:%016llx not found", targetJettyId.id,
        targetEid.in6.subnet_prefix, targetEid.in6.interface_id);
    return -ENODEV;
}

STATIC int RsPingCommonImportJetty(urma_context_t *urmaCtx, struct PingQpInfo *target,
    urma_target_jetty_t **importTjetty)
{
    urma_token_t tokenValue = {0};
    urma_eid_t remoteEid = {0};
    urma_rjetty_t rjetty = {0};

    RsGetJettyInfo(target, &rjetty.jetty_id, &remoteEid);

    tokenValue.token = target->ub.tokenValue;
    rjetty.trans_mode = URMA_TM_UM;
    rjetty.type = URMA_JETTY;
    rjetty.flag.bs.token_policy = URMA_TOKEN_PLAIN_TEXT;
    rjetty.tp_type = URMA_UTP;

    *importTjetty = RsUrmaImportJetty(urmaCtx, &rjetty, &tokenValue);
    if (*importTjetty == NULL) {
        hccp_err("urma_import_jetty failed, errno:%d remote eid:%016llx:%016llx", errno, remoteEid.in6.subnet_prefix,
            remoteEid.in6.interface_id);
        return -EOPENSRC;
    }
    return 0;
}

STATIC int RsPingUrmaAllocTargetNode(struct RsPingCtxCb *pingCb, struct PingTargetInfo *target,
    struct RsPingTargetInfo **node)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    int ret;

    targetInfo = (struct RsPingTargetInfo *)calloc(1, sizeof(struct RsPingTargetInfo));
    CHK_PRT_RETURN(targetInfo == NULL, hccp_err("calloc target_info failed! errno:%d", errno), -ENOMEM);

    ret = pthread_mutex_init(&targetInfo->tripMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init tripMutex failed, ret:%d", ret);
        goto free_target_info;
    }

    targetInfo->payloadSize = target->payload.size;
    if (target->payload.size > 0) {
        targetInfo->payloadBuffer = (char *)calloc(1, target->payload.size);
        if (targetInfo->payloadBuffer == NULL) {
            hccp_err("calloc payloadBuffer failed! size:%u errno:%d", target->payload.size, errno);
            ret = -ENOMEM;
            goto free_trip_mutex;
        }
        (void)memcpy_s(targetInfo->payloadBuffer, target->payload.size, target->payload.buffer, target->payload.size);
    }

    (void)memcpy_s(&targetInfo->qpInfo, sizeof(struct PingQpInfo), &target->remoteInfo.qpInfo,
        sizeof(struct PingQpInfo));

    ret = RsPingCommonImportJetty(pingCb->udevCb.urmaCtx, &target->remoteInfo.qpInfo, &targetInfo->importTjetty);
    if (ret != 0) {
        hccp_err("rs_ping_import_jetty failed, ret:%d", ret);
        goto free_payload_buffer;
    }

    targetInfo->resultSummary.rttMin = ~0;
    targetInfo->state = RS_PING_PONG_TARGET_READY;
    *node = targetInfo;
    return 0;

free_payload_buffer:
    if (target->payload.size > 0 && targetInfo->payloadBuffer != NULL) {
        free(targetInfo->payloadBuffer);
        targetInfo->payloadBuffer = NULL;
    }
free_trip_mutex:
    (void)pthread_mutex_destroy(&targetInfo->tripMutex);
free_target_info:
    free(targetInfo);
    targetInfo = NULL;
    return ret;
}

STATIC void RsPingUrmaResetRecvBuffer(struct RsPingCtxCb *pingCb)
{
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongJetty.recvSegCb.mutex);
    (void)memset_s((void *)(uintptr_t)pingCb->pongJetty.recvSegCb.addr, pingCb->pongJetty.recvSegCb.len, 0,
        pingCb->pongJetty.recvSegCb.len);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongJetty.recvSegCb.mutex);
}

STATIC void RsPingFillSendHeader(struct RsPingPayloadHeader *header, urma_jetty_id_t *serverJettyKey,
    struct RsPingLocalJettyCb *pongJetty, struct RsPingTargetInfo *target)
{
    header->type = RS_PING_TYPE_URMA_DETECT;
    (void)memcpy_s(header->server.ub.key, sizeof(header->server.ub.key), serverJettyKey, sizeof(urma_jetty_id_t));
    header->server.ub.size = sizeof(urma_jetty_id_t);
    header->server.ub.tokenValue = pongJetty->tokenValue;
    (void)memcpy_s(&header->target, sizeof(struct PingQpInfo), &target->qpInfo, sizeof(struct PingQpInfo));
}

STATIC void RsPingJettyBuildUpWr(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target, urma_sge_t *list,
    urma_jfs_wr_t *wr)
{
    (void)pingCb;
    wr->opcode = URMA_OPC_SEND;
    wr->flag.bs.complete_enable = 1;
    wr->tjetty = target->importTjetty;
    wr->user_ctx = target->uuid;
    wr->send.src.sge = list;
    wr->send.src.num_sge = 1;
    wr->send.imm_data = 0;
    wr->next = NULL;
}

STATIC int RsPingUrmaPostSend(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target)
{
    urma_jetty_id_t serverJettyKey = {0};
    struct RsPingPayloadHeader *header = NULL;
    urma_jetty_id_t targetJettyId = {0};
    struct timeval timestamp = {0};
    urma_jfs_wr_t *badWr = NULL;
    urma_eid_t targetEid = {0};
    urma_jfs_wr_t wr = {0};
    urma_sge_t list = {0};
    uint32_t sgeIdx;
    int ret = 0;

    RsGetJettyInfo(&target->qpInfo, &targetJettyId, &targetEid);
    hccp_dbg("target uuid:0x%llx state:%d payload_size:%u jetty_id:%u eid:%016llx:%016llx", target->uuid, target->state,
        target->payloadSize, targetJettyId.id, targetEid.in6.subnet_prefix, targetEid.in6.interface_id);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingJetty.sendSegCb.mutex);
    sgeIdx = pingCb->pingJetty.sendSegCb.sgeIdx;
    (void)memcpy_s(&list, sizeof(urma_sge_t), &pingCb->pingJetty.sendSegCb.sgeList[sgeIdx], sizeof(urma_sge_t));
    pingCb->pingJetty.sendSegCb.sgeIdx = (sgeIdx + 1) % pingCb->pingJetty.sendSegCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingJetty.sendSegCb.mutex);

    // prepare ping_jetty send buffer
    serverJettyKey = pingCb->pongJetty.jetty->jetty_id;
    (void)memset_s((void *)(uintptr_t)list.addr, list.len, 0, list.len);
    header = (struct RsPingPayloadHeader *)(uintptr_t)list.addr;
    RsPingFillSendHeader(header, &serverJettyKey, &pingCb->pongJetty, target);

    if (target->payloadSize > 0) {
        ret = memcpy_s((void *)(uintptr_t)(list.addr + RS_PING_PAYLOAD_HEADER_RESV_CUSTOM),
            (list.len - RS_PING_PAYLOAD_HEADER_RESV_CUSTOM), (void *)target->payloadBuffer, target->payloadSize);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("memcpy_s buffer payload_size:%u list.len:%u failed, ret:%d", target->payloadSize,
                (list.len - RS_PING_PAYLOAD_HEADER_RESV_CUSTOM), ret),
            -ESAFEFUNC);
    }
    list.len = RS_PING_PAYLOAD_HEADER_RESV_CUSTOM + target->payloadSize;

    RsPingJettyBuildUpWr(pingCb, target, &list, &wr);

    // record timestamp t1
    (void)gettimeofday(&timestamp, NULL);
    header->timestamp.tvSec1 = (uint64_t)timestamp.tv_sec;
    header->timestamp.tvUsec1 = (uint64_t)timestamp.tv_usec;
    header->taskId = pingCb->taskId;
    header->magic = 0x55AA;

    ret = RsUrmaPostJettySendWr(pingCb->pingJetty.jetty, &wr, &badWr);
    if (ret != 0) {
        hccp_err("rs_urma_post_jetty_send_wr jetty_id:%u failed, ret:%d", serverJettyKey.id, ret);
        RS_PTHREAD_MUTEX_LOCK(&target->tripMutex);
        target->state = RS_PING_PONG_TARGET_ERROR;
        RS_PTHREAD_MUTEX_ULOCK(&target->tripMutex);
    }
    return ret;
}

STATIC int RsPingUrmaPollScq(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target)
{
    urma_cr_t sendCr = {0};
    int polledCnt;

    polledCnt = RsUrmaPollJfc(pingCb->pingJetty.sendJfc.jfc, 1, &sendCr);
    if (polledCnt != 1) {
        hccp_err("uuid:0x%llx rs_urma_poll_jfc polled_cnt:%d", target->uuid, polledCnt);
        target->state = RS_PING_PONG_TARGET_ERROR;
        return -ENODATA;
    }
    if (sendCr.status != URMA_CR_SUCCESS) {
        target->state = RS_PING_PONG_TARGET_ERROR;
        hccp_err("wr_id:0x%llx error cqe cr_status(%d)", sendCr.user_ctx, sendCr.status);
        return -EOPENSRC;
    }
    return 0;
}

STATIC int RsPingUrmaPollRcq(struct RsPingCtxCb *pingCb, int *polledCnt, struct timeval *timestamp2)
{
    urma_jfc_t *evJfc = NULL;
    uint32_t ackCnt = 1;
    int waitCnt;

    // record timestamp t2
    (void)gettimeofday(timestamp2, NULL);

    waitCnt = RsUrmaWaitJfc(pingCb->pingJetty.jfce, 1, 0, &evJfc);
    if (waitCnt == 0) {
        return -EAGAIN;
    }
    if (waitCnt != 1) {
        hccp_err("urma_wait_jfc failed, ret:%d", waitCnt);
        return -EOPENSRC;
    }
    RsUrmaAckJfc((urma_jfc_t **)&evJfc, &ackCnt, 1);

    if (evJfc != pingCb->pingJetty.recvJfc.jfc) {
        hccp_err("urma_wait_jfc returned unknown jfc");
        return -EOPENSRC;
    }
    pingCb->pingJetty.recvJfc.numEvents++;

    *polledCnt = RsUrmaPollJfc(evJfc, pingCb->pingJetty.recvJfc.maxRecvWcNum, gPingJettyRecvCr);
    CHK_PRT_RETURN(*polledCnt > pingCb->pingJetty.recvJfc.maxRecvWcNum || *polledCnt < 0,
        hccp_err("urma_poll_jfc failed, ret:%d", *polledCnt), -EOPENSRC);

    return 0;
}

STATIC int RsPingCommonPollSendJfc(struct RsPingLocalJettyCb *jettyCb)
{
    urma_cr_t cr = {0};
    int polledCnt;

    polledCnt = RsUrmaPollJfc(jettyCb->sendJfc.jfc, 1, &cr);
    if (polledCnt < 0) {
        hccp_warn_rma("urma_poll_jfc unsuccessful, polledCnt:%d", polledCnt);
    } else if (polledCnt > 0) {
        if (cr.status != URMA_CR_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe status(%d)", cr.user_ctx, cr.status);
            return -EOPENSRC;
        }
    }

    return 0;
}

STATIC int RsPongJettyFindTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target,
    struct RsPongTargetInfo **node)
{
    struct RsPongTargetInfo *targetNext = NULL;
    struct RsPongTargetInfo *targetCurr = NULL;
    urma_jetty_id_t targetJettyId = {0};
    urma_eid_t targetEid = {0};

    RsGetJettyInfo(target, &targetJettyId, &targetEid);

    RS_CHECK_POINTER_NULL_WITH_RET(pingCb);
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongMutex);
    RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &pingCb->pongList, list, struct RsPongTargetInfo);
    for (; (&targetCurr->list) != &pingCb->pongList;
         targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPongTargetInfo, list)) {
        if (RsPingCommonCompareUbInfo(&targetCurr->qpInfo, target)) {
            *node = targetCurr;
            RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);

    hccp_info_rma("pong target node for jetty_id:%u eid:%016llX:%016llX not found", targetJettyId.id,
        targetEid.in6.subnet_prefix, targetEid.in6.interface_id);
    return -ENODEV;
}

STATIC int RsPongJettyFindAllocTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target,
    struct RsPongTargetInfo **node)
{
    struct RsPongTargetInfo *targetInfo = NULL;
    int ret;

    ret = RsPongJettyFindTargetNode(pingCb, target, node);
    if (ret == 0 && (*node)->state == RS_PING_PONG_TARGET_READY) {
        return 0;
    } else if (ret == 0) {
        targetInfo = *node;
        hccp_info_rma("delete pong target uuid:0x%llx state:%d, realloc again", targetInfo->uuid, targetInfo->state);
        RsListDel(&targetInfo->list);
        if (targetInfo->importTjetty != NULL) {
            (void)RsUrmaUnimportJetty(targetInfo->importTjetty);
            targetInfo->importTjetty = NULL;
        }
        free(targetInfo);
        targetInfo = NULL;
    }

    targetInfo = (struct RsPongTargetInfo *)calloc(1, sizeof(struct RsPongTargetInfo));
    CHK_PRT_RETURN(targetInfo == NULL, hccp_err("calloc target_info failed! errno:%d", errno), -ENOMEM);

    (void)memcpy_s(&targetInfo->qpInfo, sizeof(struct PingQpInfo), target, sizeof(struct PingQpInfo));

    ret = RsPingCommonImportJetty(pingCb->udevCb.urmaCtx, target, &targetInfo->importTjetty);
    if (ret != 0) {
        hccp_err("rs_pong_import_jetty failed, ret:%d", ret);
        goto free_target_info;
    }

    targetInfo->state = RS_PING_PONG_TARGET_READY;
    *node = targetInfo;

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongMutex);
    targetInfo->uuid = (uint64_t)pingCb->pongNum << 32U;
    RsListAddTail(&targetInfo->list, &pingCb->pongList);
    pingCb->pongNum++;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);

    return 0;

free_target_info:
    free(targetInfo);
    return ret;
}

STATIC int RsPongJettyPostSend(struct RsPingCtxCb *pingCb, urma_cr_t *cr, struct timeval *timestamp2)
{
    struct RsPongTargetInfo *targetInfo = NULL;
    struct RsPingPayloadHeader *header = NULL;
    struct timeval timestamp3 = {0};
    urma_sge_t recvList = {0};
    urma_sge_t sendList = {0};
    urma_jfs_wr_t *badWr = NULL;
    urma_jfs_wr_t wr = {0};
    uint32_t recvSgeIdx;
    uint32_t sendSgeIdx;
    int ret = 0;

    // poll send jfc
    (void)RsPingCommonPollSendJfc(&pingCb->pongJetty);

    // handle detect packet & send response packet
    recvSgeIdx = (uint32_t)cr->user_ctx;
    if (recvSgeIdx >= pingCb->pingJetty.recvSegCb.sgeNum) {
        hccp_err("param err recv_sge_idx:%u > sge_num:%u", recvSgeIdx, pingCb->pingJetty.recvSegCb.sgeNum);
        return -EIO;
    }
    (void)memcpy_s(&recvList, sizeof(urma_sge_t), &pingCb->pingJetty.recvSegCb.sgeList[recvSgeIdx], sizeof(urma_sge_t));

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongJetty.sendSegCb.mutex);
    sendSgeIdx = pingCb->pongJetty.sendSegCb.sgeIdx;
    (void)memcpy_s(&sendList, sizeof(urma_sge_t), &pingCb->pongJetty.sendSegCb.sgeList[sendSgeIdx], sizeof(urma_sge_t));
    pingCb->pongJetty.sendSegCb.sgeIdx = (sendSgeIdx + 1) % pingCb->pongJetty.sendSegCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongJetty.sendSegCb.mutex);

    ret = memcpy_s((void *)(uintptr_t)sendList.addr, sendList.len, (void *)(uintptr_t)recvList.addr,
        cr->completion_len);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("memcpy_s buffer cr->completion_len:%u send_list.length:%u failed, ret:%d", cr->completion_len,
            sendList.len, ret),
        -ESAFEFUNC);
    sendList.len = cr->completion_len;
    header = (struct RsPingPayloadHeader *)(uintptr_t)sendList.addr;
    header->type = RS_PING_TYPE_URMA_RESPONSE;

    ret = RsPongJettyFindAllocTargetNode(pingCb, &header->server, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_pong_jetty_find_alloc_target_node failed, ret:%d", ret);
        return ret;
    }

    wr.opcode = URMA_OPC_SEND;
    wr.flag.bs.complete_enable = 1;
    wr.tjetty = targetInfo->importTjetty;
    wr.user_ctx = targetInfo->uuid;
    wr.send.src.sge = &sendList;
    wr.send.src.num_sge = 1;
    wr.send.imm_data = 0;
    wr.next = NULL;

    // record timestamp t3
    (void)gettimeofday(&timestamp3, NULL);
    header->timestamp.tvSec2 = (uint64_t)timestamp2->tv_sec;
    header->timestamp.tvUsec2 = (uint64_t)timestamp2->tv_usec;
    header->timestamp.tvSec3 = (uint64_t)timestamp3.tv_sec;
    header->timestamp.tvUsec3 = (uint64_t)timestamp3.tv_usec;
    header->magic = 0xAA55;

    ret = RsUrmaPostJettySendWr(pingCb->pongJetty.jetty, &wr, &badWr);
    if (ret != 0) {
        targetInfo->state = RS_PING_PONG_TARGET_ERROR;
        hccp_err("urma_post_jetty_send_wr failed, ret:%d", ret);
        return ret;
    }

    return ret;
}

STATIC void RsPongUrmaHandleSend(struct RsPingCtxCb *pingCb, int polledCnt, struct timeval *timestamp2)
{
    urma_cr_t *cr = NULL;
    int ret, i;

    cr = gPingJettyRecvCr;
    for (i = 0; i < polledCnt; i++) {
        if (cr[i].status != URMA_CR_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe status(%d)", cr[i].user_ctx, cr[i].status);
            continue;
        }

        ret = RsPongJettyPostSend(pingCb, &cr[i], timestamp2);
        if (ret != 0) {
            hccp_err("rs_pong_jetty_post_send failed, wrId:0x%llx", cr[i].user_ctx);
            continue;
        }

        ret = RsPingCommonJfrPostRecv(&pingCb->pingJetty);
        if (ret != 0) {
            hccp_err("rs_ping_common_jfr_post_recv failed, ret:%d", ret);
            continue;
        }
    }

    ret = RsUrmaRearmJfc(pingCb->pingJetty.recvJfc.jfc, false);
    if (ret != 0) {
        hccp_err("urma_rearm_jfc failed, ret:%d", ret);
    }

    return;
}

STATIC int RsPongJettyResolveResponsePacket(struct RsPingCtxCb *pingCb, uint32_t sgeIdx, struct timeval *timestamp4)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    struct RsPingPayloadHeader *header = NULL;
    urma_jetty_id_t targetJettyId = {0};
    urma_eid_t targetEid = {0};
    urma_sge_t *recvList = NULL;
    uint32_t rtt;
    int ret;

    recvList = &pingCb->pongJetty.recvSegCb.sgeList[sgeIdx];
    header = (struct RsPingPayloadHeader *)(uintptr_t)(recvList->addr);
    if (header->taskId != pingCb->taskId) {
        hccp_warn_rma("drop received packet, recv_taskId:%u, curr_taskId:%u", header->taskId, pingCb->taskId);
        return 0;
    }
    RsGetJettyInfo(&header->target, &targetJettyId, &targetEid);

    header->timestamp.tvSec4 = (uint64_t)timestamp4->tv_sec;
    header->timestamp.tvUsec4 = (uint64_t)timestamp4->tv_usec;
    rtt = RsPingGetTripTime(&header->timestamp);
    ret = RsPingUrmaFindTargetNode(pingCb, &header->target, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_ping_urma_find_target_node failed, ret:%d jettyId:%u eid:%016llX:%016llX rtt:%u", ret,
            targetJettyId.id, targetEid.in6.subnet_prefix, targetEid.in6.interface_id, rtt);
        return ret;
    }

    (void)memset_s((void *)header, RS_PING_PAYLOAD_HEADER_MASK_SIZE, 0, RS_PING_PAYLOAD_HEADER_MASK_SIZE);
    RS_PTHREAD_MUTEX_LOCK(&targetInfo->tripMutex);
    targetInfo->resultSummary.recvCnt++;
    targetInfo->resultSummary.taskId = header->taskId;
    // rtt timeout, increase timeoutCnt
    if (((uint64_t)targetInfo->resultSummary.taskAttr.timeoutInterval * RS_PING_MSEC_TO_USEC) < rtt) {
        targetInfo->resultSummary.timeoutCnt++;
        hccp_dbg("recvCnt:%u timeoutInterval:%u rtt:%u timeoutCnt:%u", targetInfo->resultSummary.recvCnt,
            targetInfo->resultSummary.taskAttr.timeoutInterval, rtt, targetInfo->resultSummary.timeoutCnt);
        RS_PTHREAD_MUTEX_ULOCK(&targetInfo->tripMutex);
        return 0;
    }

    // handle rtt_min, rtt_max, rtt_avg
    if (targetInfo->resultSummary.rttMin > rtt) {
        targetInfo->resultSummary.rttMin = rtt;
    }
    if (targetInfo->resultSummary.rttMax < rtt) {
        targetInfo->resultSummary.rttMax = rtt;
    }
    if (targetInfo->resultSummary.rttAvg == 0) {
        targetInfo->resultSummary.rttAvg = rtt;
    }
    targetInfo->resultSummary.rttAvg = (targetInfo->resultSummary.rttAvg + rtt) / 2U;
    RS_PTHREAD_MUTEX_ULOCK(&targetInfo->tripMutex);
    return 0;
}

STATIC void RsPongUrmaPollRcq(struct RsPingCtxCb *pingCb)
{
    struct timeval timestamp = {0};
    urma_jfc_t *evJfc = NULL;
    uint32_t recvSgeIdx;
    urma_cr_t *cr = NULL;
    uint32_t ackCnt = 1;
    int polledCnt, i;
    int waitCnt;
    int ret;

    // record timestamp t4
    (void)gettimeofday(&timestamp, NULL);

    waitCnt = RsUrmaWaitJfc(pingCb->pongJetty.jfce, 1, 0, &evJfc);
    if (waitCnt == 0) {
        return;
    }
    if (waitCnt != 1) {
        hccp_err("urma_wait_jfc failed, ret:%d", waitCnt);
        return;
    }
    RsUrmaAckJfc((urma_jfc_t **)&evJfc, &ackCnt, 1);

    if (evJfc != pingCb->pongJetty.recvJfc.jfc) {
        hccp_err("urma_wait_jfc returned unknown jfc");
        return;
    }
    pingCb->pongJetty.recvJfc.numEvents++;

    polledCnt = RsUrmaPollJfc(evJfc, pingCb->pongJetty.recvJfc.maxRecvWcNum, gPongJettyRecvCr);
    if (polledCnt > pingCb->pongJetty.recvJfc.maxRecvWcNum || polledCnt < 0) {
        hccp_err("urma_poll_jfc failed, ret:%d", polledCnt);
        goto rearm_jfc;
    }

    cr = gPongJettyRecvCr;
    for (i = 0; i < polledCnt; i++) {
        if (cr[i].status != URMA_CR_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe status(%d)", cr[i].user_ctx, cr[i].status);
            continue;
        }
        recvSgeIdx = (uint32_t)cr[i].user_ctx;
        if (recvSgeIdx >= pingCb->pongJetty.recvSegCb.sgeNum) {
            hccp_err("param err recv_sge_idx:%u > sge_num:%u", recvSgeIdx, pingCb->pongJetty.recvSegCb.sgeNum);
            continue;
        }

        // handle response packet result
        ret = RsPongJettyResolveResponsePacket(pingCb, recvSgeIdx, &timestamp);
        if (ret != 0) {
            continue;
        }

        ret = RsPingCommonJfrPostRecv(&pingCb->pongJetty);
        if (ret != 0) {
            continue;
        }
    }

rearm_jfc:
    ret = RsUrmaRearmJfc(evJfc, false);
    if (ret != 0) {
        hccp_err("urma_rearm_jfc failed, ret:%d", ret);
    }

    return;
}

STATIC int RsPingUrmaGetTargetResult(struct RsPingCtxCb *pingCb, struct PingTargetCommInfo *target,
    struct PingResultInfo *result)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    urma_jetty_id_t targetJettyId = {0};
    urma_eid_t targetEid = {0};
    int ret;

    RsGetJettyInfo(&target->qpInfo, &targetJettyId, &targetEid);

    ret = RsPingUrmaFindTargetNode(pingCb, &target->qpInfo, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_ping_urma_find_target_node failed, ret:%d jettyId:%u eid:%016llx:%016llx", ret, targetJettyId.id,
            targetEid.in6.subnet_prefix, targetEid.in6.interface_id);
        return ret;
    }

    (void)memcpy_s(&result->summary, sizeof(struct PingResultSummary), &targetInfo->resultSummary,
        sizeof(struct PingResultSummary));
    if (targetInfo->state == RS_PING_PONG_TARGET_FINISH) {
        result->state = PING_RESULT_STATE_VALID;
    } else {
        result->state = PING_RESULT_STATE_INVALID;
    }
    hccp_dbg("eid:%016llx:%016llx jetty_id:%u, state:%d send_cnt:%u recv_cnt:%u timeout_cnt:%u rtt_min:%u rtt_max:%u "
             "rtt_avg:%u",
        targetEid.in6.subnet_prefix, targetEid.in6.interface_id, targetJettyId.id, result->state,
        result->summary.sendCnt, result->summary.recvCnt, result->summary.timeoutCnt, result->summary.rttMin,
        result->summary.rttMax, result->summary.rttAvg);

    return 0;
}

STATIC void RsPingUrmaFreeTargetNode(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *targetInfo)
{
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RsListDel(&targetInfo->list);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    if (targetInfo->payloadSize > 0 && targetInfo->payloadBuffer != NULL) {
        free(targetInfo->payloadBuffer);
        targetInfo->payloadBuffer = NULL;
    }

    if (targetInfo->importTjetty != NULL) {
        (void)RsUrmaUnimportJetty(targetInfo->importTjetty);
    }
    return;
}

STATIC void RsPingPongJettyDelTargetList(struct RsPingCtxCb *pingCb)
{
    struct RsPongTargetInfo *pongNext = NULL;
    struct RsPingTargetInfo *pingNext = NULL;
    struct RsPongTargetInfo *pongCurr = NULL;
    struct RsPingTargetInfo *pingCurr = NULL;

    // del ping_list
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RS_LIST_GET_HEAD_ENTRY(pingCurr, pingNext, &pingCb->pingList, list, struct RsPingTargetInfo);
    for (; (&pingCurr->list) != &pingCb->pingList;
         pingCurr = pingNext, pingNext = list_entry(pingNext->list.next, struct RsPingTargetInfo, list)) {
        RsListDel(&pingCurr->list);
        if (pingCurr->payloadSize > 0 && pingCurr->payloadBuffer != NULL) {
            free(pingCurr->payloadBuffer);
            pingCurr->payloadBuffer = NULL;
        }
        if (pingCurr->importTjetty != NULL) {
            (void)RsUrmaUnimportJetty(pingCurr->importTjetty);
        }
        (void)pthread_mutex_destroy(&pingCurr->tripMutex);
        free(pingCurr);
        pingCurr = NULL;
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    // del pong_list
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongMutex);
    RS_LIST_GET_HEAD_ENTRY(pongCurr, pongNext, &pingCb->pongList, list, struct RsPongTargetInfo);
    for (; (&pongCurr->list) != &pingCb->pongList;
         pongCurr = pongNext, pongNext = list_entry(pongNext->list.next, struct RsPongTargetInfo, list)) {
        RsListDel(&pongCurr->list);
        if (pongCurr->importTjetty != NULL) {
            (void)RsUrmaUnimportJetty(pongCurr->importTjetty);
        }
        free(pongCurr);
        pongCurr = NULL;
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);
}

STATIC void RsPingUrmaPingCbDeinit(unsigned int phyId, struct RsPingCtxCb *pingCb)
{
    struct rs_cb *rscb = NULL;
    int ret;

    ret = RsGetRsCb(phyId, &rscb);
    if (ret != 0) {
        hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret);
        return;
    }

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    pingCb->taskStatus = RS_PING_TASK_RESET;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    RsPingPongJettyDelTargetList(pingCb);

    RsPingCommonDeinitLocalJettyBuffer(pingCb);
    RsPingCommonDeinitLocalJetty(rscb, pingCb, &pingCb->pongJetty);
    RsPingCommonDeinitLocalJetty(rscb, pingCb, &pingCb->pingJetty);

    (void)RsUrmaDeleteContext(pingCb->udevCb.urmaCtx);
    pingCb->udevCb.urmaCtx = NULL;
}

// ping_pong_dfx function
STATIC void RsPingUrmaAddTargetSuccess(struct PingTargetInfo *target, struct RsPingTargetInfo *targetInfo)
{
    urma_jetty_id_t jettyId = {0};
    RsGetJettyInfo(&targetInfo->qpInfo, &jettyId, NULL);
    hccp_info_rma("target eid:%016llx:%016llx payload_size:%u add success, jettyId:%u uuid:0x%llx",
        target->remoteInfo.eid.in6.subnetPrefix, target->remoteInfo.eid.in6.interfaceId, target->payload.size,
        jettyId.id, targetInfo->uuid);
}

STATIC void RsPingUrmaPingCbInitSuccess(unsigned int phyId, struct PingInitAttr *attr, unsigned int rdevIndex)
{
    hccp_run_info("ping_cb init success, phyId:%u, eid:%016llx:%016llx, rdevIndex:%u", phyId,
        attr->dev.ub.eid.in6.subnetPrefix, attr->dev.ub.eid.in6.interfaceId, rdevIndex);
}

STATIC void RsPingUrmaCannotFindTargetNode(unsigned int i, int ret, struct PingTargetCommInfo target,
    unsigned int phyId)
{
    urma_jetty_id_t jettyId = {0};
    RsGetJettyInfo(&target.qpInfo, &jettyId, NULL);

    hccp_err("rs_ping_urma_find_target_node i:%u failed, ret:%d eid:%016llx:%016llx jettyId:%u phyId:%u", i, ret,
        target.eid.in6.subnetPrefix, target.eid.in6.interfaceId, jettyId.id, phyId);
}

struct RsPingPongOps gRsPingUrmaOps = {
    .checkPingFd = RsPingUrmaCheckFd,
    .checkPongFd = RsPongUrmaCheckFd,
    .initPingCb = RsPingUrmaPingCbInit,
    .pingFindTargetNode = RsPingUrmaFindTargetNode,
    .pingAllocTargetNode = RsPingUrmaAllocTargetNode,
    .resetRecvBuffer = RsPingUrmaResetRecvBuffer,
    .pingPostSend = RsPingUrmaPostSend,
    .pingPollScq = RsPingUrmaPollScq,
    .pingPollRcq = RsPingUrmaPollRcq,
    .pongHandleSend = RsPongUrmaHandleSend,
    .pongPollRcq = RsPongUrmaPollRcq,
    .getTargetResult = RsPingUrmaGetTargetResult,
    .pingFreeTargetNode = RsPingUrmaFreeTargetNode,
    .deinitPingCb = RsPingUrmaPingCbDeinit,
};

struct RsPingPongDfx gRsPingUrmaDfx = {
    .addTargetSuccess = RsPingUrmaAddTargetSuccess,
    .initPingCbSuccess = RsPingUrmaPingCbInitSuccess,
    .pingCannotFindTargetNode = RsPingUrmaCannotFindTargetNode,
};

struct RsPingPongOps *RsPingUrmaGetOps(void)
{
    return &gRsPingUrmaOps;
}

struct RsPingPongDfx *RsPingUrmaGetDfx(void)
{
    return &gRsPingUrmaDfx;
}
