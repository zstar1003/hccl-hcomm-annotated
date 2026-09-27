/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define _GNU_SOURCE
#include "config_log.h"
#include <unistd.h>
#include <stdlib.h>
#include <dlfcn.h>
#include <fcntl.h>

#include "securec.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "rs_rdma_inner.h"
#include "rs_inner.h"
#include "verbs_exp.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "rs_drv_rdma.h"

int gMaxCqeNum = 0;
#define INVALID_EVENT -1

struct ibv_wc gWcBuf[RS_WC_NUM];

struct RsCqeErrInfo gRsCqeErr;

int RsDrvInitCqeErrInfo(void)
{
    struct RsCqeErrInfo *errInfo = &gRsCqeErr;
    int ret;

    ret = pthread_mutex_init(&errInfo->mutex, NULL);
    CHK_PRT_RETURN(ret, hccp_err("rscb mutex_init failed ret %d!, normal ret 0", ret), -ESYSFUNC);
    ret = memset_s(&errInfo->info, sizeof(struct CqeErrInfo), 0, sizeof(struct CqeErrInfo));
    if (ret != 0) {
        pthread_mutex_destroy(&errInfo->mutex);
    }
    CHK_PRT_RETURN(ret, hccp_err("memset_s failed ret[%d]", ret), -ESAFEFUNC);
    return 0;
}

void RsDrvDeinitCqeErrInfo(void)
{
    struct RsCqeErrInfo *errInfo = &gRsCqeErr;

    pthread_mutex_destroy(&errInfo->mutex);
    return;
}

STATIC void RsDrvSaveCqeErrInfo(uint32_t status, struct RsQpCb *qpCb)
{
    struct RsCqeErrInfo *errInfo = &gRsCqeErr;
    struct CqeErrInfo *tempInfo = &errInfo->info;

    RS_PTHREAD_MUTEX_LOCK(&errInfo->mutex);
    if (tempInfo->status != 0) {
        RS_PTHREAD_MUTEX_ULOCK(&errInfo->mutex);
        hccp_run_info("over status=[0x%x], drop qpn[0x%x] err cqe status=[0x%x]", tempInfo->status, qpCb->qpInfoLo.qpn,
            status);
        return;
    }
    tempInfo->status = status;
    tempInfo->qpn = (uint32_t)qpCb->qpInfoLo.qpn;
    RsGetCurTime(&tempInfo->time);
    RS_PTHREAD_MUTEX_ULOCK(&errInfo->mutex);

    return;
}

STATIC void RsDrvSaveQpCqeErrInfo(uint32_t status, struct RsQpCb *qpCb)
{
    RS_PTHREAD_MUTEX_LOCK(&qpCb->cqeErrInfo.mutex);
    if (qpCb->cqeErrInfo.info.status != 0) {
        RS_PTHREAD_MUTEX_ULOCK(&qpCb->cqeErrInfo.mutex);
        return;
    }
    qpCb->cqeErrInfo.info.status = status;
    qpCb->cqeErrInfo.info.qpn = (uint32_t)qpCb->qpInfoLo.qpn;
    RsGetCurTime(&qpCb->cqeErrInfo.info.time);
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->cqeErrInfo.mutex);

    RS_PTHREAD_MUTEX_LOCK(&qpCb->rdevCb->cqeErrCntMutex);
    qpCb->rdevCb->cqeErrCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->rdevCb->cqeErrCntMutex);

    return;
}

int RsDrvGetCqeErrInfo(struct CqeErrInfo *info)
{
    struct RsCqeErrInfo *errInfo = &gRsCqeErr;
    struct CqeErrInfo *tempInfo = &errInfo->info;
    int ret;

    CHK_PRT_RETURN(info == NULL, hccp_err("info is NULL"), -EINVAL);

    RS_PTHREAD_MUTEX_LOCK(&errInfo->mutex);
    if (tempInfo->status == 0) {
        RS_PTHREAD_MUTEX_ULOCK(&errInfo->mutex);
        return 0;
    }
    hccp_run_info("status=[%u]", tempInfo->status);
    ret = memcpy_s(info, sizeof(struct CqeErrInfo), &errInfo->info, sizeof(struct CqeErrInfo));
    if (ret) {
        hccp_err("memcpy_s failed");
        RS_PTHREAD_MUTEX_ULOCK(&errInfo->mutex);
        return -ESAFEFUNC;
    }

    ret = memset_s(&errInfo->info, sizeof(struct CqeErrInfo), 0, sizeof(struct CqeErrInfo));
    RS_PTHREAD_MUTEX_ULOCK(&errInfo->mutex);
    CHK_PRT_RETURN(ret, hccp_err("memset_s failed ret[%d], buf len:%u", ret, sizeof(struct CqeErrInfo)), -ESAFEFUNC);
    return 0;
}

STATIC void RsRdmaRetryTimeoutExceptionCheck(struct SensorNode *sensorNode, struct ibv_wc *wc)
{
    int ret = 0;

    if (wc->status != IBV_WC_RETRY_EXC_ERR) {
        return;
    }

    ret = RsRetryTimeoutExceptionCheck(sensorNode);

    hccp_warn_rma("update sensor state logic_devid(%u), qpn(%u), sensorUpdateCnt(%d), ret(%d)\n",
        sensorNode->logicDevid, wc->qp_num, sensorNode->sensorUpdateCnt, ret);
}

STATIC void RsCqeCallbackProcess(struct RsQpCb *qpCb, struct ibv_wc *wc, struct ibv_cq *evCq)
{
    (void)evCq;
    if (wc->status != IBV_WC_SUCCESS && wc->status != IBV_WC_WR_FLUSH_ERR) {
        hccp_err("Failed status [%s] [%u], wr[%llu]", RsIbvWcStatusStr(wc->status), wc->status, wc->wr_id);
        RsDrvSaveCqeErrInfo(wc->status, qpCb);
        RsDrvSaveQpCqeErrInfo(wc->status, qpCb);
        RsRdmaRetryTimeoutExceptionCheck(&qpCb->rdevCb->rsCb->sensorNode, wc);
    }

    return;
}

void RsDrvPollSrqCqHandle(struct RsQpCb *qpCb)
{
    struct ibv_cq *evCq = NULL;
    void *evCtx = NULL;

    if (RsIbvGetCqEvent(qpCb->srqContext->channel, &evCq, &evCtx)) {
        hccp_err("Failed to get cq_event");
        return;
    }

    if (evCq != qpCb->ibRecvCq || evCtx == NULL) {
        hccp_err("CQ event for unknown CQ");
        return;
    }

    ++qpCb->srqContext->numRecvCqEvents;

    struct event_summary *evCtxTmp = (struct event_summary *)evCtx;
    if ((int)evCtxTmp->event_id != INVALID_EVENT) {
        hccp_info_rma("SubmitEvent: event id:%d, pid:%d, grp id:%u, dev id:%u", evCtxTmp->event_id, evCtxTmp->pid,
            evCtxTmp->grp_id, qpCb->rdevCb->rsCb->chipId);
        int ret = DlHalEschedSubmitEvent(qpCb->rdevCb->rsCb->chipId, evCtx);
        if (ret) {
            hccp_warn_rma("halEschedSubmitEvent unsuccessful, ret:%d", ret);
        }
    }

    return;
}

void RsDrvPollCqHandle(struct RsQpCb *qpCb)
{
    struct ibv_cq *evCq = NULL;
    void *evCtx = NULL;
    int ne, i;

    if (RsIbvGetCqEvent(qpCb->channel, &evCq, &evCtx)) {
        hccp_err("Failed to get cq_event");
        return;
    }

    if (evCq != qpCb->ibSendCq && evCq != qpCb->ibRecvCq) {
        hccp_err("CQ event for unknown CQ");
        return;
    }

    if (evCq == qpCb->ibRecvCq) {
        ++qpCb->numRecvCqEvents;
    } else {
        ++qpCb->numSendCqEvents;
    }

    if (evCtx != NULL) {
        struct event_summary *evCtxTmp = (struct event_summary *)evCtx;
        if ((int)(evCtxTmp->event_id) != INVALID_EVENT) {
            hccp_info_rma("SubmitEvent: event id:%d, pid:%d, grp id:%u, dev id:%u", evCtxTmp->event_id, evCtxTmp->pid,
                evCtxTmp->grp_id, qpCb->rdevCb->rsCb->chipId);
            int ret = DlHalEschedSubmitEvent(qpCb->rdevCb->rsCb->chipId, evCtx);
            if (ret) {
                hccp_warn_rma("halEschedSubmitEvent unsuccessful, ret:%d", ret);
            }
        }
        return;
    }

    ne = ibv_poll_cq(evCq, RS_WC_NUM, gWcBuf);
    if (ne > RS_WC_NUM || ne < 0) {
        hccp_err("poll CQ failed %d", ne);
        return;
    }
    if (gMaxCqeNum < ne) {
        gMaxCqeNum = ne;
        hccp_run_info("rs_drv_poll_cq_handle: max_cqe_num=[%d]", gMaxCqeNum);
    }

    if (ibv_req_notify_cq(evCq, 0)) {
        hccp_err("Couldn't request CQ notification");
        return;
    }
    qpCb->rdevCb->pollCqeNum += ne;
    for (i = 0; i < ne; ++i) {
        RsCqeCallbackProcess(qpCb, &(gWcBuf[i]), evCq);
    }
    return;
}

int RsDrvCompareIpGid(int family, union HccpIpAddr localIp, union ibv_gid *gid)
{
    unsigned int gidV4[RS_GID_SEQ_NUM];

    if (family == AF_INET6) {
        if (memcmp(gid, &(localIp.addr6), sizeof(union ibv_gid)) == 0) {
            return 0;
        }
    } else {
        gidV4[RS_GID_SEQ_ZERO] = 0;
        gidV4[RS_GID_SEQ_ONE] = 0;
        /* The gid format generated by ipv4 is filled with 0xFFFF in [33, 48] */
        gidV4[RS_GID_SEQ_TWO] = htonl(0x0000FFFF);
        gidV4[RS_GID_SEQ_THREE] = localIp.addr.s_addr;
        if (memcmp(gid, &(gidV4), sizeof(union ibv_gid)) == 0) {
            return 0;
        }
    }

    return -ENODEV;
}

int RsDrvGetGidIndex(struct RsRdevCb *rdevCb, struct ibv_port_attr *attr, int *idx)
{
    static const char *portStates[] = {"Nop", "Down", "Init", "Armed", "", "Active Defer"};
    enum ibv_gid_type_sysfs type;
    union ibv_gid gidTmp;
    int gidIdx = -1;
    int ret;
    int i;

    ret = RsIbvQueryPort(rdevCb->ibCtx, rdevCb->ibPort, attr);
    CHK_PRT_RETURN(ret, hccp_err("ibv_query_port failed ret[%d]", ret), -EOPENSRC);

    // link maybe suffer from intermittent disconnection, should continue process
    if (attr->state != IBV_PORT_ACTIVE) {
        hccp_warn_rma("port number %u state is %s", rdevCb->ibPort, portStates[attr->state]);
    }

    for (i = 0; i < attr->gid_tbl_len; i++) {
        ret = RsIbvQueryGidType(rdevCb->ibCtx, rdevCb->ibPort, (unsigned int)i, &type);
        CHK_PRT_RETURN(ret, hccp_err("query gid type failed i %d, ret %d", i, ret), -EOPENSRC);
        if (type != IBV_GID_TYPE_SYSFS_ROCE_V2) {
            continue;
        }

        ret = RsIbvQueryGid(rdevCb->ibCtx, rdevCb->ibPort, i, &gidTmp);
        CHK_PRT_RETURN(ret, hccp_err("query gid failed i %d, ret %d", i, ret), -EOPENSRC);

        ret = RsDrvCompareIpGid((int)rdevCb->localIp.family, rdevCb->localIp.binAddr, &gidTmp);
        if (ret == 0) {
            gidIdx = i;
            break;
        }
    }

    if (gidIdx == -1) {
        hccp_err("get idx failed, attr->gid_tbl_len:%d", attr->gid_tbl_len);
        return -ENODEV;
    }

    hccp_dbg("GID index is %d", gidIdx);
    *idx = gidIdx;
    return 0;
}

STATIC int RsDrvGetSupportLite(struct RsRdevCb *rdevCb, int qpMode, unsigned int aiOpSupport)
{
    // bypass rdma lite when ai_op_support was set
    if (aiOpSupport == 1) {
        return 0;
    }

    if (qpMode == RA_RS_OP_QP_MODE || qpMode == RA_RS_OP_QP_MODE_EXT) {
        return rdevCb->supportLite;
    }

    return 0;
}

int RsDrvCreateCqWithAttrs(struct RsQpCb *qpCb, int isExt, struct CqExtAttr *cqAttr)
{
    int sendEqNum = cqAttr->sendCqCompVector;
    int recvEqNum = cqAttr->recvCqCompVector;
    struct rdma_lite_device_cq_init_attr attr = {
        .cq_type = qpCb->qpMode,
        .part_id = 0,
        .lite_op_supported = 0,
        .mem_align = qpCb->memAlign,
        .mem_idx = qpCb->memResp.memData.mem_idx,
        .ai_op_support = qpCb->aiOpSupport,
        .grp_id = qpCb->grpId,
        .cq_cstm_flag = qpCb->cqCstmFlag,
        .use_resv_mem = qpCb->useResvMem,
        .resv_mem_pool_id = qpCb->resvMemPoolId,
    };
    struct ibv_comp_channel *channel = qpCb->channel;

    attr.lite_op_supported = RsDrvGetSupportLite(qpCb->rdevCb, qpCb->qpMode, qpCb->aiOpSupport);
    // caller poll cq
    if (attr.lite_op_supported != 0 || attr.cq_cstm_flag != 0) {
        channel = NULL;
        sendEqNum = 0;
        recvEqNum = 0;
    }

    hccp_dbg("create cq start");
    if (isExt == 1) {
        qpCb->ibSendCq = RsIbvExpCreateCq(qpCb->rdevCb->ibCtx, cqAttr->sendCqDepth, NULL, channel, sendEqNum, &attr,
            &qpCb->qpResp.sendCqData);
        hccp_info_rma("rs_ibv_exp_create_cq ibSendCq");
    } else {
        qpCb->ibSendCq = RsIbvCreateCq(qpCb->rdevCb->ibCtx, cqAttr->sendCqDepth, NULL, channel, sendEqNum);
    }

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    CHK_PRT_RETURN(qpCb->ibSendCq == NULL, hccp_err("ibv create send cq failed"), -ENOMEM);

    if (isExt == 1) {
        qpCb->ibRecvCq = RsIbvExpCreateCq(qpCb->rdevCb->ibCtx, cqAttr->recvCqDepth, NULL, channel, recvEqNum, &attr,
            &qpCb->qpResp.recvCqData);
        hccp_info_rma("rs_ibv_exp_create_cq ibRecvCq");
    } else {
        qpCb->ibRecvCq = RsIbvCreateCq(qpCb->rdevCb->ibCtx, cqAttr->recvCqDepth, NULL, channel, recvEqNum);
    }

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    if (qpCb->ibRecvCq == NULL) {
        hccp_err("ibv create recv cq failed");
        (void)RsIbvDestroyCq(qpCb->ibSendCq);
        return -ENOMEM;
    }

    hccp_info_rma("create cq success");
    return 0;
}

#define RS_DRV_CQ_DEPTH 16384
#define RS_DRV_CQ_128_DEPTH 128
#define RS_DRV_CQ_32K_DEPTH 32768

int RsDrvCreateCq(struct RsQpCb *qpCb, int isExt)
{
    struct CqExtAttr cqAttr = {0};

    cqAttr.sendCqDepth = qpCb->sendCqDepth;
    cqAttr.sendCqCompVector = qpCb->eqNum;
    cqAttr.recvCqDepth = qpCb->recvCqDepth;
    cqAttr.recvCqCompVector = qpCb->eqNum;

    return RsDrvCreateCqWithAttrs(qpCb, isExt, &cqAttr);
}

void RsDrvEventDestroy(struct event_summary *event)
{
    if (event != NULL) {
        free(event);
        event = NULL;
    }
}

void RsDrvDestroyCq(struct RsQpCb *qpCb)
{
    (void)RsIbvDestroyCq(qpCb->ibRecvCq);
    (void)RsIbvDestroyCq(qpCb->ibSendCq);
    (void)RsDrvEventDestroy(qpCb->recvEvent);
    (void)RsDrvEventDestroy(qpCb->sendEvent);
}

int RsDrvQpStateModifytoReset(struct RsQpCb *qpCb)
{
    struct ibv_qp_init_attr qpInitAttr = {0};
    struct ibv_qp_attr attr = {0};
    int ret;

    attr.qp_state = IBV_QPS_RESET;
    ret = RsIbvModifyQp(qpCb->ibQp, &attr, IBV_QP_STATE);
    CHK_PRT_RETURN(ret, hccp_err("[modify]qpn[%d] modify to reset failed, ret %d", qpCb->qpInfoLo.qpn, ret), ret);

    (void)memset_s(&attr, sizeof(struct ibv_qp_attr), 0, sizeof(struct ibv_qp_attr));

    ret = RsIbvQueryQp(qpCb->ibQp, &attr, IBV_QP_STATE, &qpInitAttr);
    if ((ret != 0) || attr.qp_state != IBV_QPS_RESET) {
        hccp_err("query qpn[%d] attr failed, ret %d or qp state %d is not RESET", qpCb->qpInfoLo.qpn, ret,
            attr.qp_state);
        return ret;
    }

    hccp_info_rma("qpn[%d] modify to reset success", qpCb->qpInfoLo.qpn);
    return 0;
}

int RsDrvQpStateModifytoInit(struct RsQpCb *qpCb, struct ibv_qp_attr *attr)
{
    int ret;

    attr->qp_state = IBV_QPS_INIT;
    attr->pkey_index = 0;
    attr->port_num = qpCb->rdevCb->ibPort;
    attr->qp_access_flags = DEFAULT_ACCESS_FLAG;
    ret = RsIbvModifyQp(qpCb->ibQp, attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    CHK_PRT_RETURN(ret, hccp_err("[modify]qpn[%d] modify to init failed, ret %d", qpCb->qpInfoLo.qpn, ret), ret);

    hccp_info_rma("qpn[%d] modify to init success", qpCb->qpInfoLo.qpn);
    return 0;
}

enum ibv_mtu RsDrvSetMtu(struct RsQpCb *qpCb)
{
    int ret;
    struct ibv_port_attr portAttr = {0};
    enum ibv_mtu currMtu;

    ret = RsIbvQueryPort(qpCb->rdevCb->ibCtx, qpCb->rdevCb->ibPort, &portAttr);
    CHK_PRT_RETURN(ret, hccp_err("Error when trying to query port, ret[%d]", ret), -EOPENSRC);
#ifndef CA_CONFIG_LLT
    currMtu = portAttr.active_mtu;
#else
    currMtu = IBV_MTU_1024;
#endif

    return currMtu;
}

int RsDrvQpStateModifytoRtr(struct RsQpCb *qpCb, struct ibv_qp_attr *attr)
{
    struct ibv_port_attr portAttr = {0};
    int ret;

    attr->qp_state = IBV_QPS_RTR;
    attr->dest_qp_num = (uint32_t)qpCb->qpInfoRem.qpn;
    attr->rq_psn = (uint32_t)qpCb->qpInfoRem.psn;
    attr->min_rnr_timer = RS_QP_ATTR_MIN_RNR_TIMER;
    (attr->ah_attr).is_global = 0;
    (attr->ah_attr).dlid = qpCb->qpInfoRem.lid;
    (attr->ah_attr).sl = qpCb->qosAttr.sl;
    (attr->ah_attr).src_path_bits = 0;
    (attr->ah_attr).port_num = qpCb->rdevCb->ibPort;

    attr->path_mtu = RsDrvSetMtu(qpCb);
    if (qpCb->rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
        attr->max_dest_rd_atomic = RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE;
    } else {
        attr->max_dest_rd_atomic = RS_MAX_RD_ATOMIC_NUM;
        CHK_PRT_RETURN(attr->path_mtu < IBV_MTU_1024,
            hccp_err("qpn[%d] failed to set mtu, mtu[%d] < [%d]", qpCb->qpInfoLo.qpn, attr->path_mtu, IBV_MTU_1024),
            -EPERM);
    }

    // get gid_idx dynamically
    ret = RsDrvGetGidIndex(qpCb->rdevCb, &portAttr, &qpCb->qpInfoLo.gidIdx);
    CHK_PRT_RETURN(ret,
        hccp_err("rs_drv_get_gid_index failed ret[%d], qpn[%d] gidIdx[%d]", ret, qpCb->qpInfoLo.qpn,
            qpCb->qpInfoLo.gidIdx),
        ret);

    if (qpCb->qpInfoRem.gid.global.interface_id) {
        attr->ah_attr.is_global = 1;
        attr->ah_attr.grh.hop_limit = 1;
        attr->ah_attr.grh.dgid = qpCb->qpInfoRem.gid;
        attr->ah_attr.grh.sgid_index = qpCb->qpInfoLo.gidIdx;
    }

    (attr->ah_attr).grh.traffic_class = qpCb->qosAttr.tc;
    ret = RsIbvModifyQp(qpCb->ibQp, attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER);
    CHK_PRT_RETURN(ret,
        hccp_err("qpn[%d] failed to modify QP to RTR, ibv_modify_qp failed ret[%d], errno[%d]", qpCb->qpInfoLo.qpn, ret,
            errno),
        -EOPENSRC);
    hccp_info_rma("qp qos attr: qpn[%d] tc[%u] sl[%u]", qpCb->qpInfoLo.qpn, qpCb->qosAttr.tc, qpCb->qosAttr.sl);
    return 0;
}

int RsDrvQpStateModifytoRts(struct RsQpCb *qpCb, struct ibv_qp_attr *attr)
{
    int ret;
    attr->qp_state = IBV_QPS_RTS;
    attr->timeout = (uint8_t)qpCb->timeout;
    attr->retry_cnt = (uint8_t)qpCb->retryCnt;
    attr->rnr_retry = RS_QP_ATTR_RNR_RETRY;
    attr->sq_psn = (uint32_t)qpCb->qpInfoLo.psn;
    if (qpCb->rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
        attr->max_rd_atomic = RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE;
    } else {
        attr->max_rd_atomic = RS_MAX_RD_ATOMIC_NUM;
    }
    ret = RsIbvModifyQp(qpCb->ibQp, attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    CHK_PRT_RETURN(ret,
        hccp_err("qpn[%d] failed to modify QP to RTS, ibv_modify_qp failed ret[%d]", qpCb->qpInfoLo.qpn, ret),
        -EOPENSRC);
    hccp_info_rma("qp rdma attr: qpn[%d] timeout[%u] retrycnt[%u]", qpCb->qpInfoLo.qpn, qpCb->timeout, qpCb->retryCnt);
    return 0;
}

struct ibv_mr *RsDrvMrReg(struct ibv_pd *pd, char *addr, size_t length, int access)
{
    return RsIbvRegMr(pd, addr, length, access);
}

struct ibv_mr *RsDrvExpMrReg(struct ibv_pd *pd, char *addr, size_t length, int access,
    struct roce_process_sign roceSign)
{
    return RsIbvExpRegMr(pd, addr, length, access, roceSign);
}

int RsDrvMrDereg(struct ibv_mr *ibMr)
{
    return RsIbvDeregMr(ibMr);
}

#ifdef CUSTOM_INTERFACE
STATIC int RsOpenBackupIbCtx(struct RsRdevCb *rdevCb)
{
    struct ibv_context *ibCtxTmp = NULL;
    struct rdev rdevInfo = {0};
    int gidIndex = -1;
    int ret;
    int i;

    (void)memcpy_s(&rdevInfo, sizeof(struct rdev), &rdevCb->backupInfo.rdevInfo, sizeof(struct rdev));
    for (i = 0; (i < rdevCb->devNum) && (rdevCb->devList[i] != NULL); ++i) {
        ibCtxTmp = RsIbvOpenDevice(rdevCb->devList[i]);
        CHK_PRT_RETURN(ibCtxTmp == NULL, hccp_err("ibv_open_device with backup failed, errno:%d", errno), -ENODEV);
        ret = RsQueryGid(rdevInfo, ibCtxTmp, rdevCb->ibPort, &gidIndex);
        if (ret == 0) {
            rdevCb->backupInfo.ibCtx = ibCtxTmp;
            return 0;
        } else if (ret == -EEXIST) {
            RsIbvCloseDevice(ibCtxTmp);
        } else {
            RsIbvCloseDevice(ibCtxTmp);
            hccp_err("rs_query_gid failed, ret:%d", ret);
            return ret;
        }
    }

    CHK_PRT_RETURN(i == rdevCb->devNum,
        hccp_err("can not find ib_ctx for phyId[%u] local_ip[0x%x] in dev_list!", rdevInfo.phyId,
            rdevInfo.localIp.addr.s_addr),
        -EEXIST);
    return 0;
}

void RsCloseBackupIbCtx(struct RsRdevCb *rdevCb)
{
    // no need to close backup device
    if (!rdevCb->backupInfo.backupFlag || rdevCb->backupInfo.ibCtx == NULL) {
        return;
    }

    RsIbvCloseDevice(rdevCb->backupInfo.ibCtx);
}
#endif

STATIC int RsDrvQueryNotify(struct RsRdevCb *rdevCb)
{
    int ret = 0;

    if (rdevCb->notifyType == NO_USE) {
        return 0;
    }

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        if (rdevCb->backupInfo.backupFlag) {
            // open backup device to get ib_ctx to get backup notify va and size
            ret = RsOpenBackupIbCtx(rdevCb);
            CHK_PRT_RETURN(ret, hccp_err("rs_open_backup_ib_ctx failed, ret:%d", ret), ret);

            ret = RsIbvExpQueryNotify(rdevCb->backupInfo.ibCtx, &rdevCb->notifyVaBase, &rdevCb->notifySize);
            if (ret != 0) {
                RsCloseBackupIbCtx(rdevCb);
                hccp_err("rs_ibv_exp_query_notify with backup ctx failed, ret:%d", ret);
                return ret;
            }
        } else {
            ret = RsIbvExpQueryNotify(rdevCb->ibCtx, &rdevCb->notifyVaBase, &rdevCb->notifySize);
            CHK_PRT_RETURN(ret, hccp_err("rs_ibv_exp_query_notify failed, ret:%d", ret), ret);
        }
    }
#endif
    hccp_info_rma("chip_id:%u, RsDrvQueryNotify ok, notify va:0x%llx, size:%llu", rdevCb->rsCb->chipId,
        rdevCb->notifyVaBase, rdevCb->notifySize);
    return ret;
}

int RsDrvQueryNotifyAndAllocPd(struct RsRdevCb *rdevCb)
{
    int ret = 0;

    ret = RsDrvQueryNotify(rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_query_notify failed, ret %d", ret), ret);

    rdevCb->ibPd = RsIbvAllocPd(rdevCb->ibCtx);
    if (rdevCb->ibPd == NULL) {
#ifdef CUSTOM_INTERFACE
        if (RsIsCustomInterfaceSupported()) {
            RsCloseBackupIbCtx(rdevCb);
        }
#endif
        hccp_err("rs_ibv_alloc_pd failed, errno:%d", errno);
        return -ENOMEM;
    }

    return 0;
}

int RsDrvRegNotifyMr(struct RsRdevCb *rdevCb)
{
    struct roce_process_sign roceSign = {0};
    int access = DEFAULT_ACCESS_FLAG;
    (void)roceSign;
    rdevCb->notifyAccess = access;
    switch (rdevCb->notifyType) {
        case NO_USE:
            return 0;
        case NOTIFY: {
            rdevCb->notifyMr = RsIbvRegMr(rdevCb->ibPd, (void *)(uintptr_t)rdevCb->notifyVaBase, rdevCb->notifySize,
                access);
            break;
        }
        case EVENTID: {
#ifdef CUSTOM_INTERFACE
            if (RsIsCustomInterfaceSupported()) {
                rdevCb->notifyMr = RsIbvExpRegMr(rdevCb->ibPd, (void *)(uintptr_t)rdevCb->notifyVaBase,
                    rdevCb->notifySize, access, roceSign);
            } else {
                return 0;
            }
            break;
#else
            return 0;
#endif
        }
        default: {
            hccp_err("rs_drv_reg_notify_mr failed! notify_type = %u", rdevCb->notifyType);
            return -EINVAL;
        }
    }

    CHK_PRT_RETURN(rdevCb->notifyMr == NULL,
        hccp_err("ibv_reg_mr addr[0x%llx] len[%llu] errno[%d] failed", rdevCb->notifyVaBase, rdevCb->notifySize, errno),
        -EACCES);

    hccp_info_rma("ibv_reg_mr ok");
    return 0;
}

STATIC void RsBuildRecvWr(struct RecvWrlistData *wr, struct ibv_sge *list, struct ibv_recv_wr *ibWr)
{
    list->addr = (uintptr_t)wr->memList.addr;
    list->length = wr->memList.len;
    list->lkey = wr->memList.lkey;

    ibWr->sg_list = list;
    ibWr->wr_id = wr->wrId;
    ibWr->num_sge = 1; /* only support one sge */
    return;
}

int RsDrvPostRecv(struct RsQpCb *qpCb, struct RecvWrlistData *wr, unsigned int recvNum, unsigned int *completeNum)
{
    int ret = 0;
    unsigned int i, index;
    struct ibv_recv_wr *badWr = NULL;

    CHK_PRT_RETURN(recvNum == 0, hccp_err("recv_num[%u] is invalid!", recvNum), -EINVAL);

    struct ibv_recv_wr *recvWr = (struct ibv_recv_wr *)calloc(recvNum, sizeof(struct ibv_recv_wr));
    CHK_PRT_RETURN(recvWr == NULL, hccp_err("calloc recv_wr failed!"), -ENOSPC);

    struct ibv_sge *list = (struct ibv_sge *)calloc(recvNum, sizeof(struct ibv_sge));
    if (list == NULL) {
        hccp_err("calloc list failed!");
        ret = -ENOSPC;
        goto alloc_sge_fail;
    }

    for (i = 0; i < recvNum; i++) {
        RsBuildRecvWr(&wr[i], &list[i], &recvWr[i]);
        index = i + 1; // for fix pclint warning
        recvWr[i].next = (i < recvNum - 1) ? &(recvWr[index]) : NULL;
    }

    ret = RsIbvPostRecv(qpCb->ibQp, recvWr, &badWr);
    if (ret == 0) {
        *completeNum = recvNum;
    } else if (ret == -ENOMEM) {
        *completeNum = (unsigned int)((void *)badWr - (void *)recvWr) / sizeof(struct ibv_recv_wr);
        hccp_dbg("post recv wqe overflow, completeNum[%d]", *completeNum);
    } else {
        hccp_err("ibv_post_recv failed, ret[%d]", ret);
        *completeNum = 0;
    }
    qpCb->recvWrNum = qpCb->recvWrNum + (*completeNum);

    free(list);
    list = NULL;

alloc_sge_fail:
    free(recvWr);
    recvWr = NULL;
    return (ret == -ENOMEM) ? 0 : ret;
}

int RsDrvSendExp(struct RsQpCb *qpCb, struct RsMrCb *mrCb, struct RsMrCb *remMrCb, struct SendWr *wr,
    struct SendWrRsp *wrRsp)
{
    int i;
    int ret;
    struct ibv_sge list[RS_SGLIST_MAX];
    struct ibv_send_wr ibWr = {
        .sg_list = list,
        .opcode = wr->op,
        .send_flags = wr->sendFlag,
    };
    struct ibv_send_wr *badWr = NULL;
    struct wr_exp_rsp expRsp = {0};

    for (i = 0; i < wr->bufNum && i < RS_SGLIST_MAX; i++) {
        list[i].addr = (uintptr_t)wr->bufList[i].addr;
        list[i].length = wr->bufList[i].len;
        list[i].lkey = mrCb->ibMr->lkey;
    }

    ibWr.num_sge = i;
    ibWr.wr_id = mrCb->wrId;
    // send op has no rem_mr, no need to assign
    if (wr->op != RA_WR_SEND && wr->op != RA_WR_SEND_WITH_IMM) {
        ibWr.wr.rdma.rkey = remMrCb->mrInfo.rkey;
        ibWr.wr.rdma.remote_addr = wr->dstAddr;
    }

    ret = RsIbvExpPostSend(qpCb->ibQp, &ibWr, &badWr, &expRsp);
    if (ret) {
        hccp_err("rs_ibv_exp_post_send failed ret %d", ret);
    }

    if (qpCb->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
        wrRsp->wqeTmp.sqIndex = (unsigned int)qpCb->sqIndex;
        wrRsp->wqeTmp.wqeIndex = expRsp.wqe_index;
    } else if (qpCb->qpMode == RA_RS_OP_QP_MODE || qpCb->qpMode == RA_RS_OP_QP_MODE_EXT ||
               qpCb->qpMode == RA_RS_GDR_ASYN_QP_MODE) {
        wrRsp->db.dbIndex = (unsigned int)qpCb->dbIndex;
        wrRsp->db.dbInfo = expRsp.db_info;
    }

    return ret;
}

STATIC int RsDrvQpQueryInfo(struct RsQpCb *qpCb, struct RsRdevCb *rdevCb, struct ibv_port_attr *attr,
    struct ibv_qp_attr *qpAttr, union ibv_gid *gidTmp)
{
    int ret;
    /* modify qp state */
    ret = memset_s(qpAttr, sizeof(struct ibv_qp_attr), 0, sizeof(struct ibv_qp_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s failed ret[%d], buf len:%u", ret, sizeof(struct ibv_qp_attr)), -ESAFEFUNC);
    qpAttr->qp_state = IBV_QPS_INIT;
    qpAttr->pkey_index = 0;
    qpAttr->port_num = rdevCb->ibPort;
    qpAttr->qp_access_flags = DEFAULT_ACCESS_FLAG;
    ret = RsIbvModifyQp(qpCb->ibQp, qpAttr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    CHK_PRT_RETURN(ret, hccp_err("ibv_modify_qp failed ret[%d]", ret), -EOPENSRC);

    /* prepare qp info for exchange */
    ret = RsDrvGetGidIndex(rdevCb, attr, &qpCb->qpInfoLo.gidIdx);
    CHK_PRT_RETURN(ret,
        hccp_err("rs_drv_get_gid_index failed ret[%d] qpn[%u] gid_idx[%d]", ret, qpCb->ibQp->qp_num,
            qpCb->qpInfoLo.gidIdx),
        ret);

    ret = RsIbvQueryGid(rdevCb->ibCtx, rdevCb->ibPort, qpCb->qpInfoLo.gidIdx, gidTmp);
    CHK_PRT_RETURN(ret, hccp_err("ibv_query_gid failed ret[%d]", ret), -EOPENSRC);
    return 0;
}

RS_ATTRI_VISI_DEF int RsDrvGetRandomNum(int *randNum)
{
    int randFd;
    int ret = 0;
    int retClose = 0;

    randFd = open("/dev/urandom", O_RDONLY, S_IRUSR);
    CHK_PRT_RETURN(randFd < 0, hccp_err("open random failed ret[%d] rand_fd[%d]", errno, randFd), -EFILEOPER);
    do {
        ret = read(randFd, randNum, sizeof(int));
    } while ((ret < 0) && (errno == EINTR));

    if (ret < 0) {
        hccp_err("get random failed ret[%d]", ret);
        RS_CLOSE_RETRY_FOR_EINTR(retClose, randFd);
        return -EFILEOPER;
    }

    RS_CLOSE_RETRY_FOR_EINTR(retClose, randFd);
    return 0;
}

int RsDrvQpInfoRelated(struct RsQpCb *qpCb, struct RsRdevCb *rdevCb, struct ibv_port_attr *attr,
    struct ibv_qp_attr *qpAttr)
{
    int ret;
    union ibv_gid gidTmp;
    int randNum;

    ret = RsDrvQpQueryInfo(qpCb, rdevCb, attr, qpAttr, &gidTmp);
    CHK_PRT_RETURN(ret, hccp_err("query qp info failed, ret %d", ret), ret);

    ret = RsDrvGetRandomNum(&randNum);
    CHK_PRT_RETURN(ret, hccp_err("get random num failed, ret %d", ret), ret);

    qpCb->qpInfoLo.cmd = (unsigned int)RS_CMD_QP_INFO;
    qpCb->qpInfoLo.lid = attr->lid;
    qpCb->qpInfoLo.psn = (unsigned int)randNum & 0xffffff;
    qpCb->qpInfoLo.qpn = (int)qpCb->ibQp->qp_num;
    if (rdevCb->notifyType != NO_USE && rdevCb->notifyMr != NULL) {
        qpCb->qpInfoLo.notifyMr.addr = rdevCb->notifyVaBase;
        qpCb->qpInfoLo.notifyMr.len = rdevCb->notifySize;
        qpCb->qpInfoLo.notifyMr.rkey = rdevCb->notifyMr->rkey;
    } else {
        qpCb->qpInfoLo.notifyMr.addr = 0;
        qpCb->qpInfoLo.notifyMr.len = 0;
        qpCb->qpInfoLo.notifyMr.rkey = 0;
    }

    ret = memcpy_s(qpCb->qpInfoLo.gid.raw, sizeof(qpCb->qpInfoLo.gid.raw), gidTmp.raw, sizeof(gidTmp.raw));
    CHK_PRT_RETURN(ret,
        hccp_err("memcpy_s raw failed, ret:%d, dst_len:%u, src_len:%d", ret, sizeof(qpCb->qpInfoLo.gid.raw),
            RS_QP_ATTR_GID_LEN),
        -ESAFEFUNC);
    return 0;
}

STATIC int RsDrvExpQpCreateInit(struct ibv_exp_qp_init_attr *qpInitAttr, struct RsQpCb *qpCb,
    struct ibv_port_attr *attr)
{
    int ret;
    ret = memset_s(attr, sizeof(struct ibv_port_attr), 0, sizeof(struct ibv_port_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for attr failed, ret:%d", ret), -ESAFEFUNC);
    ret = memset_s(qpInitAttr, sizeof(struct ibv_exp_qp_init_attr), 0, sizeof(struct ibv_exp_qp_init_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for qp_init_attr failed, ret:%d", ret), -ESAFEFUNC);

    qpInitAttr->attr.qp_type = IBV_QPT_RC;
    qpInitAttr->attr.send_cq = qpCb->ibSendCq;
    qpInitAttr->attr.recv_cq = qpCb->ibRecvCq;
    qpInitAttr->attr.cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    qpInitAttr->attr.cap.max_send_wr = qpCb->txDepth;
    qpInitAttr->attr.cap.max_send_sge = qpCb->sendSgeNum;
    qpInitAttr->attr.cap.max_recv_wr = qpCb->rxDepth;
    qpInitAttr->attr.cap.max_recv_sge = qpCb->recvSgeNum;

    return 0;
}

STATIC int RsDrvExpQpCreate(struct RsQpCb *qpCb, int qpMode)
{
    int ret;
    struct ibv_qp_attr qpAttr;
    struct ibv_exp_qp_init_attr qpInitAttr;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_port_attr attr;

    hccp_dbg("qp exp create begin..");
    rdevCb = qpCb->rdevCb;
    qpCb->qpMode = qpMode;
    ret = RsDrvExpQpCreateInit(&qpInitAttr, qpCb, &attr);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_qp_create_init failed, ret:%d", ret), ret);

    qpInitAttr.gdr_enable = qpMode;
    qpInitAttr.lite_op_support = RsDrvGetSupportLite(rdevCb, qpCb->qpMode, qpCb->aiOpSupport);
    qpInitAttr.mem_align = qpCb->memAlign;
    qpInitAttr.mem_idx = qpCb->memResp.memData.mem_idx;
    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    qpCb->ibQp = RsIbvExpCreateQp(qpCb->ibPd, &qpInitAttr, &qpCb->qpResp.qpData);
    CHK_PRT_RETURN(qpCb->ibQp == NULL, hccp_err("rs_ibv_exp_create_qp failed"), -ENOMEM);

    qpCb->dbIndex = (qpMode == RA_RS_OP_QP_MODE || qpMode == RA_RS_GDR_ASYN_QP_MODE) ? qpCb->qpResp.qpData.qp_info : 0;
    qpCb->sqIndex = (qpMode == RA_RS_GDR_TMPL_QP_MODE) ? qpCb->qpResp.qpData.qp_info : 0;
    hccp_info_rma("db index is [%d], sq index is [%d]", qpCb->dbIndex, qpCb->sqIndex);

    /* query qp attr */
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, &qpInitAttr.attr);
    if (ret) {
        hccp_err("query qp attr failed ret %d", ret);
        ret = -EOPENSRC;
        goto exp_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret) {
        hccp_err("qp info related failed %d", ret);
        goto exp_init_qp_err;
    }

    hccp_info_rma("chip_id %u, rdevIndex:%u, qp[%d] create succ.", qpCb->rdevCb->rsCb->chipId, qpCb->rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);

    return 0;

exp_init_qp_err:
    (void)RsIbvDestroyQp(qpCb->ibQp);
    return ret;
}

STATIC int RsDrvNormalQpCreateInit(struct ibv_qp_init_attr *qpInitAttr, struct RsQpCb *qpCb, struct ibv_port_attr *attr)
{
    int ret;
    ret = memset_s(attr, sizeof(struct ibv_port_attr), 0, sizeof(struct ibv_port_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for attr failed, ret:%d", ret), -ESAFEFUNC);
    ret = memset_s(qpInitAttr, sizeof(struct ibv_qp_init_attr), 0, sizeof(struct ibv_qp_init_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for qp_init_attr failed, ret:%d", ret), -ESAFEFUNC);

    qpInitAttr->qp_type = IBV_QPT_RC;
    qpInitAttr->send_cq = qpCb->ibSendCq;
    qpInitAttr->recv_cq = qpCb->ibRecvCq;
    qpInitAttr->cap.max_inline_data = QP_DEFAULT_MAX_CAP_INLINE_DATA;
    qpInitAttr->cap.max_send_wr = qpCb->txDepth;
    qpInitAttr->cap.max_send_sge = qpCb->sendSgeNum;
    qpInitAttr->cap.max_recv_wr = qpCb->rxDepth;
    qpInitAttr->cap.max_recv_sge = qpCb->recvSgeNum;
    return 0;
}

STATIC int RsDrvQpNormal(struct RsQpCb *qpCb, int qpMode)
{
    int ret;
    struct ibv_qp_attr qpAttr;
    struct ibv_qp_init_attr qpInitAttr;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_port_attr attr;

    hccp_dbg("qp normal create begin..");
    rdevCb = qpCb->rdevCb;
    qpCb->qpMode = qpMode;
    ret = RsDrvNormalQpCreateInit(&qpInitAttr, qpCb, &attr);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_normal_qp_create_init failed, ret:%d", ret), ret);

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    qpCb->ibQp = RsIbvCreateQp(qpCb->ibPd, &qpInitAttr);
    CHK_PRT_RETURN(qpCb->ibQp == NULL, hccp_err("rs_ibv_create_qp failed, errno=%d", errno), -ENOMEM);

    /* query qp attr */
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, &qpInitAttr);
    if (ret) {
        hccp_err("query qp attr failed ret %d", ret);
        ret = -EOPENSRC;
        goto normal_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret) {
        hccp_err("qp info related failed %d", ret);
        goto normal_init_qp_err;
    }

    hccp_info_rma("chip_id %u, rdevIndex:%u, qp[%d] create succ.", qpCb->rdevCb->rsCb->chipId, qpCb->rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);

    return 0;

normal_init_qp_err:
    (void)RsIbvDestroyQp(qpCb->ibQp);
    return ret;
}

int RsDrvQpCreate(struct RsQpCb *qpCb, struct RsQpNorm *qpNorm)
{
    int ret;
    if (qpNorm->isExp != 0 && qpNorm->qpMode != RA_RS_NOR_QP_MODE) {
        ret = RsDrvExpQpCreate(qpCb, qpNorm->qpMode);
    } else {
        ret = RsDrvQpNormal(qpCb, qpNorm->qpMode);
    }

    return ret;
}

STATIC int RsDrvExpQpCreateInitWithAttrs(struct ibv_exp_qp_init_attr *qpInitAttr, struct RsQpCb *qpCb,
    struct ibv_port_attr *attr, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;

    ret = memset_s(attr, sizeof(struct ibv_port_attr), 0, sizeof(struct ibv_port_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for attr failed, ret:%d", ret), -ESAFEFUNC);
    ret = memset_s(qpInitAttr, sizeof(struct ibv_exp_qp_init_attr), 0, sizeof(struct ibv_exp_qp_init_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for qp_init_attr failed, ret:%d", ret), -ESAFEFUNC);

    qpInitAttr->attr.send_cq = qpCb->ibSendCq;
    qpInitAttr->attr.recv_cq = qpCb->ibRecvCq;

    ret = memcpy_s(&qpInitAttr->attr.cap, sizeof(struct ibv_qp_cap), &qpNorm->extAttrs.qpAttr.cap,
        sizeof(struct ibv_qp_cap));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for qp_init_attr failed, ret:%d", ret), -ENOMEM);
    qpInitAttr->attr.qp_type = qpNorm->extAttrs.qpAttr.qp_type;
    qpInitAttr->attr.sq_sig_all = qpNorm->extAttrs.qpAttr.sq_sig_all;
    qpInitAttr->udp_sport = qpCb->udpSport;
    qpInitAttr->use_resv_mem = qpNorm->extAttrs.cstmFlag.bs.useResvMem;
    qpInitAttr->resv_mem_pool_id = qpNorm->extAttrs.resvMemPoolId;

    return 0;
}

STATIC int RsDrvExpQpCreateWithAttrs(struct RsQpCb *qpCb, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;
    struct ibv_port_attr attr;
    struct ibv_qp_attr qpAttr;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_exp_qp_init_attr qpInitAttr;

    hccp_dbg("qp exp create begin..");
    rdevCb = qpCb->rdevCb;
    ret = RsDrvExpQpCreateInitWithAttrs(&qpInitAttr, qpCb, &attr, qpNorm);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_qp_create_init failed, ret:%d", ret), ret);

    qpInitAttr.gdr_enable = qpCb->qpMode;
    qpInitAttr.lite_op_support = RsDrvGetSupportLite(rdevCb, qpCb->qpMode, qpCb->aiOpSupport);
    qpInitAttr.mem_align = qpCb->memAlign;
    qpInitAttr.mem_idx = qpCb->memResp.memData.mem_idx;
    qpInitAttr.ai_op_support = qpCb->aiOpSupport;
    qpInitAttr.grp_id = qpCb->grpId;
    qpInitAttr.qp_cstm_flag = qpCb->aiOpSupport;

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    qpCb->ibQp = RsIbvExpCreateQp(qpCb->ibPd, &qpInitAttr, &qpCb->qpResp.qpData);
    CHK_PRT_RETURN(qpCb->ibQp == NULL, hccp_err("rs_ibv_exp_create_qp failed"), -ENOMEM);

    qpCb->dbIndex = (qpCb->qpMode == RA_RS_OP_QP_MODE || qpCb->qpMode == RA_RS_GDR_ASYN_QP_MODE)
                        ? qpCb->qpResp.qpData.qp_info
                        : 0;
    qpCb->sqIndex = (qpCb->qpMode == RA_RS_GDR_TMPL_QP_MODE) ? qpCb->qpResp.qpData.qp_info : 0;
    hccp_info_rma("db index is [%d], sq index is [%d]", qpCb->dbIndex, qpCb->sqIndex);

    /* query qp attr */
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, &qpInitAttr.attr);
    if (ret) {
        hccp_err("query qp attr failed ret %d", ret);
        ret = -EOPENSRC;
        goto exp_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret) {
        hccp_err("qp info related failed %d", ret);
        goto exp_init_qp_err;
    }

    hccp_info_rma("chip_id %u, rdevIndex:%u, qp[%d] create succ.", qpCb->rdevCb->rsCb->chipId, qpCb->rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);

    return 0;

exp_init_qp_err:
    (void)RsIbvDestroyQp(qpCb->ibQp);
    return ret;
}

STATIC int RsDrvNormalQpCreateInitWithAttrs(struct ibv_qp_init_attr *qpInitAttr, struct RsQpCb *qpCb,
    struct ibv_port_attr *attr, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;

    ret = memset_s(attr, sizeof(struct ibv_port_attr), 0, sizeof(struct ibv_port_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for attr failed, ret:%d", ret), -ESAFEFUNC);
    ret = memset_s(qpInitAttr, sizeof(struct ibv_qp_init_attr), 0, sizeof(struct ibv_qp_init_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s for qp_init_attr failed, ret:%d", ret), -ESAFEFUNC);

    qpInitAttr->send_cq = qpCb->ibSendCq;
    qpInitAttr->recv_cq = qpCb->ibRecvCq;

    ret = memcpy_s(&qpInitAttr->cap, sizeof(struct ibv_qp_cap), &qpNorm->extAttrs.qpAttr.cap,
        sizeof(struct ibv_qp_cap));
    CHK_PRT_RETURN(ret, hccp_err("memcpy_s for qp_init_attr failed, ret:%d", ret), -ENOMEM);
    qpInitAttr->qp_type = qpNorm->extAttrs.qpAttr.qp_type;
    qpInitAttr->sq_sig_all = qpNorm->extAttrs.qpAttr.sq_sig_all;
    return ret;
}

STATIC int RsDrvQpNormalWithAttrs(struct RsQpCb *qpCb, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;
    struct ibv_port_attr attr;
    struct ibv_qp_attr qpAttr;
    struct ibv_qp_init_attr qpInitAttr;
    struct RsRdevCb *rdevCb = NULL;

    hccp_dbg("qp normal create begin..");
    rdevCb = qpCb->rdevCb;
    ret = RsDrvNormalQpCreateInitWithAttrs(&qpInitAttr, qpCb, &attr, qpNorm);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_normal_qp_create_init failed, ret:%d", ret), ret);

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    qpCb->ibQp = RsIbvCreateQp(qpCb->ibPd, &qpInitAttr);
    CHK_PRT_RETURN(qpCb->ibQp == NULL, hccp_err("rs_ibv_create_qp failed, errno=%d", errno), -ENOMEM);

    /* query qp attr */
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, &qpInitAttr);
    if (ret) {
        hccp_err("query qp attr failed ret %d", ret);
        ret = -EOPENSRC;
        goto normal_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret) {
        hccp_err("qp info related failed ret %d", ret);
        goto normal_init_qp_err;
    }

    hccp_info_rma("chip_id %u, rdevIndex:%u, qp[%d] create succ.", qpCb->rdevCb->rsCb->chipId, qpCb->rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);

    return 0;

normal_init_qp_err:
    (void)RsIbvDestroyQp(qpCb->ibQp);
    return ret;
}

int RsDrvQpCreateWithAttrs(struct RsQpCb *qpCb, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;

    // ignore qp_mode when ai_op_support set
    if ((qpNorm->isExp != 0 && qpNorm->extAttrs.qpMode != RA_RS_NOR_QP_MODE) || (qpNorm->aiOpSupport != 0)) {
        ret = RsDrvExpQpCreateWithAttrs(qpCb, qpNorm);
    } else {
        ret = RsDrvQpNormalWithAttrs(qpCb, qpNorm);
    }

    return ret;
}

void RsDrvQpDestroy(struct RsQpCb *qpCb)
{
    (void)RsIbvDestroyQp(qpCb->ibQp);
}

int RsQueryEvent(int cqEventId, struct event_summary **event)
{
    *event = calloc(1, sizeof(struct event_summary));
    if ((*event) == NULL) {
        return -ENOMEM;
    }

    (*event)->pid = getpid();
    (*event)->grp_id = 6; // 6: 通信库使用的事件调度group
    (*event)->event_id = cqEventId;
    (*event)->subevent_id = 0;
    (*event)->msg_len = 0;
    (*event)->msg = NULL;
    (*event)->dst_engine = ACPU_DEVICE;
    (*event)->policy = ONLY;
    hccp_info_rma("pid:%d, cqEventId:%d", (*event)->pid, cqEventId);
    unsigned int i;
    for (i = 0; i < EVENT_SUMMARY_RSV; i++) {
        (*event)->rsv[i] = 0;
    }

    return 0;
}

int RsDrvCreateCqEvent(struct RsCqContext *cqContext, struct CqAttr *attr)
{
    int ret;

    hccp_info_rma("create cq event start cq_create_mode [%d]", cqContext->cqCreateMode);

    if (cqContext->cqCreateMode == RS_NORMAL_CQ_CREATE || cqContext->cqCreateMode == RS_SQ_CQ_CREATE) {
        ret = RsQueryEvent(attr->sendCqEventId, &(cqContext->sendEvent));
        CHK_PRT_RETURN(ret, hccp_err("rs_query_event send_event failed! ret:%d", ret), ret);

        cqContext->ibSendCq = RsIbvCreateCq(cqContext->rdevCb->ibCtx, attr->sendCqDepth, cqContext->sendEvent,
            cqContext->channel, cqContext->eqNum);
        if (cqContext->ibSendCq == NULL) {
            hccp_err("rs_drv_create_cq_event ibv create send cq failed, sendCqEventId:%d", attr->sendCqEventId);
            goto create_cq_even_err;
        }

        ret = ibv_req_notify_cq(cqContext->ibSendCq, 0);
        if (ret) {
            hccp_err("Couldn't request send CQ notification, ret:%d", ret);
            goto create_cq_even_err;
        }
        *attr->ibSendCq = cqContext->ibSendCq;
    }

    if (cqContext->cqCreateMode == RS_NORMAL_CQ_CREATE || cqContext->cqCreateMode == RS_SRQ_CQ_CREATE) {
        ret = RsQueryEvent(attr->recvCqEventId, &(cqContext->recvEvent));
        if (ret) {
            hccp_err("rs_query_event send_event failed! ret:%d", ret);
            goto create_cq_even_err;
        }

        cqContext->ibRecvCq = RsIbvCreateCq(cqContext->rdevCb->ibCtx, attr->recvCqDepth, cqContext->recvEvent,
            cqContext->channel, cqContext->eqNum);
        if (cqContext->ibRecvCq == NULL) {
            hccp_err("rs_drv_create_cq_event ibv create recv cq failed, recvCqEventId:%d", attr->recvCqEventId);
            goto create_cq_even_err;
        }

        ret = ibv_req_notify_cq(cqContext->ibRecvCq, 0);
        if (ret) {
            hccp_err("Couldn't request recv CQ notification, ret:%d", ret);
            goto create_cq_even_err;
        }
        *attr->ibRecvCq = cqContext->ibRecvCq;
    }

    hccp_info_rma("create cq event success");
    return 0;

create_cq_even_err:
    if (cqContext->ibRecvCq != NULL) {
        (void)RsIbvDestroyCq(cqContext->ibRecvCq);
    }

    if (cqContext->ibSendCq != NULL) {
        (void)RsIbvDestroyCq(cqContext->ibSendCq);
    }

    (void)RsDrvEventDestroy(cqContext->recvEvent);

    (void)RsDrvEventDestroy(cqContext->sendEvent);

    return -EOPENSRC;
}

int RsDrvCreateCqWithChannel(struct RsCqContext *cqContext, struct CqAttr *attr)
{
    hccp_dbg("create cq with channel start");

    cqContext->ibSendCq = RsIbvCreateCq(cqContext->rdevCb->ibCtx, attr->sendCqDepth, NULL, attr->sendChannel, 1);
    if (cqContext->ibSendCq == NULL) {
        hccp_err("rs_drv_create_cq_with_channel ibv create send cq failed.");
        return -EOPENSRC;
    }

    cqContext->ibRecvCq = RsIbvCreateCq(cqContext->rdevCb->ibCtx, attr->recvCqDepth, NULL, attr->recvChannel, 1);
    if (cqContext->ibRecvCq == NULL) {
        hccp_err("rs_drv_create_cq_with_channel ibv create recv cq failed.");
        goto create_recv_cq_err;
    }

    *attr->ibSendCq = cqContext->ibSendCq;
    *attr->ibRecvCq = cqContext->ibRecvCq;
    hccp_info_rma("create cq with channel success");
    return 0;

create_recv_cq_err:
    (void)RsIbvDestroyCq(cqContext->ibSendCq);
    return -EOPENSRC;
}

int RsDrvTypicalCqCreate(struct RsRdevCb *rdevCb, unsigned int cqDepth, unsigned int *cqn, struct ibv_cq **ibCq,
    struct rdma_lite_device_cq_attr *deviceCqAttr)
{
    struct rdma_lite_device_cq_init_attr initAttr = {
        .cq_type = RA_RS_OP_QP_MODE,
        .lite_op_supported = rdevCb->supportLite,
    };

    *ibCq = RsIbvExpCreateCq(rdevCb->ibCtx, (int)cqDepth, NULL, NULL, 0, &initAttr, deviceCqAttr);
    if (*ibCq == NULL) {
        hccp_err("rs_ibv_exp_create_cq failed, cqDepth[%u]", cqDepth);
        return -ENOMEM;
    }

    *cqn = deviceCqAttr->cqn;
    hccp_info_rma("drv typical cq create success, cqn[%u] cqDepth[%u]", *cqn, cqDepth);
    return 0;
}

int RsDrvDestroyCqEvent(struct RsCqContext *cqContext)
{
    int ret;

    if (cqContext->cqCreateMode == RS_NORMAL_CQ_CREATE || cqContext->cqCreateMode == RS_SRQ_CQ_CREATE) {
        ret = RsIbvDestroyCq(cqContext->ibRecvCq);
        if (ret) {
            hccp_err("rs_ibv_destroy_cq(recv) failed, ret %d", ret);
        }
        (void)RsDrvEventDestroy(cqContext->recvEvent);
    }

    if (cqContext->cqCreateMode == RS_NORMAL_CQ_CREATE || cqContext->cqCreateMode == RS_SQ_CQ_CREATE) {
        ret = RsIbvDestroyCq(cqContext->ibSendCq);
        if (ret) {
            hccp_err("rs_ibv_destroy_cq(send) failed, ret %d", ret);
        }
        (void)RsDrvEventDestroy(cqContext->sendEvent);
    }

    return 0;
}

int RsDrvNormalQpCreate(struct RsQpCb *qpCb, struct ibv_qp_init_attr *qpInitAttr)
{
    int ret;
    struct ibv_qp_attr qpAttr;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_port_attr attr;

    hccp_dbg("rs_drv_normal_qp_create begin..");
    rdevCb = qpCb->rdevCb;

    /* A return value of NULL indicates an OutOfMemoryError(OOM) has occurred */
    qpCb->ibQp = RsIbvCreateQp(qpCb->ibPd, qpInitAttr);
    CHK_PRT_RETURN(qpCb->ibQp == NULL, hccp_err("rs_ibv_create_qp failed, errno=%d", errno), -ENOMEM);

    /* query qp attr */
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, qpInitAttr);
    if (ret) {
        hccp_err("query qp attr failed ret %d", ret);
        ret = -EOPENSRC;
        goto normal_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret) {
        hccp_err("qp info related failed %d", ret);
        goto normal_init_qp_err;
    }

    hccp_info_rma("chip_id %u, rdevIndex:%u, qp[%d] create succ.", qpCb->rdevCb->rsCb->chipId, qpCb->rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);

    return 0;

normal_init_qp_err:
    (void)RsIbvDestroyQp(qpCb->ibQp);
    return ret;
}
