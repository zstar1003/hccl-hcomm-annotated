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
#include <errno.h>
#include <infiniband/verbs.h>
#include "securec.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "hccp_common.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "rs_epoll.h"
#include "rs_drv_socket.h"
#include "rs_drv_rdma.h"
#include "rs_ping_inner.h"
#ifndef HNS_ROCE_LLT
#include <dlog_pub.h>
#endif
#include "rs_ping_roce.h"

#define RS_PING_ROCE_RECV_WC_NUM 16

struct ibv_wc gPingQpRecvWc[RS_PING_ROCE_RECV_WC_NUM] = {0};
struct ibv_wc gPongQpRecvWc[RS_PING_ROCE_RECV_WC_NUM] = {0};

STATIC bool RsPingRoceCheckFd(struct RsPingCtxCb *pingCb, int fd)
{
    if (pingCb->pingQp.channel != NULL && pingCb->pingQp.channel->fd == fd) {
        hccp_dbg("ping_qp rq, channel->fd:%d poll cq", fd);
        return true;
    }
    return false;
}

STATIC bool RsPongRoceCheckFd(struct RsPingCtxCb *pingCb, int fd)
{
    if (pingCb->pongQp.channel != NULL && pingCb->pongQp.channel->fd == fd) {
        hccp_dbg("pong_qp rq, channel->fd:%d poll cq", fd);
        return true;
    }
    return false;
}

STATIC int RsPingCbGetDevRdevIndex(struct RsPingCtxCb *pingCb, int index)
{
    (void)pingCb;
    (void)index;
#ifdef CUSTOM_INTERFACE
    struct roce_dev_data rdevData = {0};
    int ret;

    if (RsIsCustomInterfaceSupported()) {
        RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
        pingCb->rdevCb.devName = RsIbvGetDeviceName(pingCb->rdevCb.devList[index]);
        ret = RsRoceGetRoceDevData(pingCb->rdevCb.devName, &rdevData);
        if (ret != 0) {
            hccp_err("rs_roce_get_roce_dev_data failed, ret:%d, devName:%s", ret, pingCb->rdevCb.devName);
            RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);
            return ret;
        }
        pingCb->devIndex = rdevData.rdev_index; // rdev_index is same to port_id
        RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);
    }
#endif
    return 0;
}

STATIC int RsPingCbGetIbCtxAndIndex(struct rdev *rdevInfo, struct RsPingCtxCb *pingCb)
{
    struct ibv_context *ibCtx = NULL;
    int ret;
    int i;

    for (i = 0; (i < pingCb->rdevCb.devNum) && (pingCb->rdevCb.devList[i] != NULL); ++i) {
        ibCtx = RsIbvOpenDevice(pingCb->rdevCb.devList[i]);
        CHK_PRT_RETURN(ibCtx == NULL, hccp_err("ibv_open_device failed!"), -ENODEV);
        ret = RsQueryGid(*rdevInfo, ibCtx, pingCb->rdevCb.ibPort, &pingCb->rdevCb.gidIdx);
        if (ret == 0) {
            ret = RsPingCbGetDevRdevIndex(pingCb, i);
            if (ret != 0) {
                hccp_err("rs_ping_cb_get_dev_rdev_index failed, ret:%d", ret);
                RsIbvCloseDevice(ibCtx);
                return ret;
            }
            pingCb->rdevCb.ibCtx = ibCtx;
            ret = RsIbvQueryGid(ibCtx, pingCb->rdevCb.ibPort, pingCb->rdevCb.gidIdx, &pingCb->rdevCb.gid);
            if (ret != 0) {
                RsIbvCloseDevice(ibCtx);
                hccp_err("query gid failed gid_idx %d, ret %d", pingCb->rdevCb.gidIdx, ret);
                return -EOPENSRC;
            }
            return 0;
        } else if (ret == -EEXIST) {
            RsIbvCloseDevice(ibCtx);
        } else {
            hccp_err("rs_query_gid failed, ret:%d", ret);
            RsIbvCloseDevice(ibCtx);
            return ret;
        }
    }

    CHK_PRT_RETURN(i == pingCb->rdevCb.devNum,
        hccp_err("can not find ib_ctx for phyId[%u] local_ip[0x%x] "
                 "in dev_list!",
            rdevInfo->phyId, rdevInfo->localIp.addr.s_addr),
        -ENODEV);
    return 0;
}

STATIC int RsPingCommonModifyLocalQp(struct RsPingCtxCb *pingCb, struct RsPingLocalQpCb *qpCb)
{
    struct ibv_qp_init_attr initAttr;
    struct ibv_qp_attr attr = {0};
    int ret;

    ret = RsIbvQueryQp(qpCb->ibQp, &attr, IBV_QP_STATE, &initAttr);
    CHK_PRT_RETURN(ret != 0 || attr.qp_state != IBV_QPS_RESET,
        hccp_err("rs_ibv_query_qp qpn:%u fail, ret:%d attr.qp_state:%d != %d", qpCb->ibQp->qp_num, ret, attr.qp_state,
            IBV_QPS_RESET),
        -EOPENSRC);

    attr.qp_state = IBV_QPS_INIT;
    attr.pkey_index = 0;
    attr.port_num = pingCb->rdevCb.ibPort;
    attr.qkey = qpCb->qkey;
    ret = RsIbvModifyQp(qpCb->ibQp, &attr, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_QKEY);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_ibv_modify_qp qpn:%u to init fail, ret:%d, errno:%d", qpCb->ibQp->qp_num, ret, errno), -EOPENSRC);

    attr.qp_state = IBV_QPS_RTR;
    ret = RsIbvModifyQp(qpCb->ibQp, &attr, IBV_QP_STATE);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_ibv_modify_qp qpn:%u to rtr fail, ret:%d, errno:%d", qpCb->ibQp->qp_num, ret, errno), -EOPENSRC);

    attr.qp_state = IBV_QPS_RTS;
    attr.sq_psn = 0;
    ret = RsIbvModifyQp(qpCb->ibQp, &attr, IBV_QP_STATE | IBV_QP_SQ_PSN);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_ibv_modify_qp qpn:%u to rts fail, ret:%d, errno:%d", qpCb->ibQp->qp_num, ret, errno), -EOPENSRC);

    return 0;
}

STATIC int RsPingCommonInitLocalQp(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, union PingQpAttr *attr,
    struct RsPingLocalQpCb *qpCb)
{
    struct ibv_exp_qp_init_attr qpInitAttr = {0};
    struct rdma_lite_device_qp_attr qpResp = {0};
    int randNum;
    int ret;

    hccp_info_rma("cq_attr{%d %d, %d %d}", attr->rdma.cqAttr.sendCqDepth, attr->rdma.cqAttr.sendCqCompVector,
        attr->rdma.cqAttr.recvCqDepth, attr->rdma.cqAttr.recvCqCompVector);

    // create send cq with attr
    qpCb->sendCq.depth = attr->rdma.cqAttr.sendCqDepth;
    qpCb->sendCq.compVector = attr->rdma.cqAttr.sendCqCompVector;
    qpCb->sendCq.ibCq = RsIbvCreateCq(pingCb->rdevCb.ibCtx, qpCb->sendCq.depth, NULL, NULL, qpCb->sendCq.compVector);
    qpCb->sendCq.maxRecvWcNum = RS_PING_ROCE_RECV_WC_NUM;
    ret = -errno;
    CHK_PRT_RETURN(qpCb->sendCq.ibCq == NULL, hccp_err("rs_ibv_create_cq send cq fail, ret:%d", ret), ret);

    // create channel & create recv cq with attr
    qpCb->channel = RsIbvCreateCompChannel(pingCb->rdevCb.ibCtx);
    if (qpCb->channel == NULL) {
        ret = -errno;
        hccp_err("rs_ibv_create_comp_channel failed! ret:%d", ret);
        goto create_channel_fail;
    }
    ret = RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_ADD, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
    if (ret != 0) {
        hccp_err("RsEpollCtl failed! epollfd:%d fd:%d ret:%d", rscb->connCb.epollfd, qpCb->channel->fd, ret);
        goto epoll_ctl_fail;
    }
    qpCb->recvCq.depth = attr->rdma.cqAttr.recvCqDepth;
    qpCb->recvCq.compVector = attr->rdma.cqAttr.recvCqCompVector;
    qpCb->recvCq.ibCq = RsIbvCreateCq(pingCb->rdevCb.ibCtx, qpCb->recvCq.depth, NULL, qpCb->channel,
        qpCb->recvCq.compVector);
    qpCb->recvCq.maxRecvWcNum = RS_PING_ROCE_RECV_WC_NUM;
    if (qpCb->recvCq.ibCq == NULL) {
        ret = -errno;
        hccp_err("rs_ibv_create_cq recv cq fail, ret:%d", ret);
        goto create_rcq_fail;
    }

    // create qp with attr
    (void)RsDrvGetRandomNum(&randNum);
    // clear bit IB_QP_SET_QKEY to avoid modify_qp to INIT failed
    qpCb->qkey = (uint32_t)(((uint32_t)randNum) & (~(1U << 31U)));
    (void)memcpy_s(&qpCb->qpCap, sizeof(struct ibv_qp_cap), &attr->rdma.qpAttr.cap, sizeof(struct ibv_qp_cap));
    qpCb->udpSport = attr->rdma.qpAttr.udpSport;
    qpInitAttr.attr.send_cq = qpCb->sendCq.ibCq;
    qpInitAttr.attr.recv_cq = qpCb->recvCq.ibCq;
    (void)memcpy_s(&qpInitAttr.attr.cap, sizeof(struct ibv_qp_cap), &qpCb->qpCap, sizeof(struct ibv_qp_cap));
    qpInitAttr.attr.qp_type = IBV_QPT_UD;
    qpInitAttr.udp_sport = attr->rdma.qpAttr.udpSport;

    hccp_info_rma("qkey:%u udp_sport:%u qp_cap{%u %u %u %u %u}", qpCb->qkey, qpCb->udpSport,
        attr->rdma.qpAttr.cap.maxSendWr, attr->rdma.qpAttr.cap.maxRecvWr, attr->rdma.qpAttr.cap.maxSendSge,
        attr->rdma.qpAttr.cap.maxRecvSge, attr->rdma.qpAttr.cap.maxInlineData);
    qpCb->ibQp = RsIbvExpCreateQp(pingCb->rdevCb.ibPd, &qpInitAttr, &qpResp);
    if (qpCb->ibQp == NULL) {
        ret = -errno;
        hccp_err("rs_ibv_exp_create_qp qp fail, ret:%d", ret);
        goto create_qp_fail;
    }

    ret = RsPingCommonModifyLocalQp(pingCb, qpCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_modify_local_qp failed, ret:%d", ret);
        goto modify_qp_fail;
    }

    ret = RsIbvReqNotifyCq(qpCb->recvCq.ibCq, 0);
    if (ret != 0) {
        hccp_err("rs_ibv_req_notify_cq failed, ret:%d", ret);
        goto modify_qp_fail;
    }

    hccp_run_info("qpn:%u create success, cqAttr{%d %d, %d %d} qkey:%u udpSport:%u qpCap{%u %u %u %u %u}",
        qpCb->ibQp->qp_num, attr->rdma.cqAttr.sendCqDepth, attr->rdma.cqAttr.sendCqCompVector,
        attr->rdma.cqAttr.recvCqDepth, attr->rdma.cqAttr.recvCqCompVector, qpCb->qkey, qpCb->udpSport,
        attr->rdma.qpAttr.cap.maxSendWr, attr->rdma.qpAttr.cap.maxRecvWr, attr->rdma.qpAttr.cap.maxSendSge,
        attr->rdma.qpAttr.cap.maxRecvSge, attr->rdma.qpAttr.cap.maxInlineData);

    return 0;

modify_qp_fail:
    (void)RsIbvDestroyQp(qpCb->ibQp);
create_qp_fail:
    (void)RsIbvDestroyCq(qpCb->recvCq.ibCq);
create_rcq_fail:
    (void)RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
epoll_ctl_fail:
    (void)RsIbvDestroyCompChannel(qpCb->channel);
create_channel_fail:
    (void)RsIbvDestroyCq(qpCb->sendCq.ibCq);
    return ret;
}

STATIC int RsPingCommonInitMrCb(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, struct RsPingMrCb *mrCb)
{
    unsigned long flag = 0;
    uint32_t idx = 0;
    int ret;

    hccp_info_rma("payload_offset:%u len:0x%llx sge_num:%u grp_id:%u", mrCb->payloadOffset, mrCb->len, mrCb->sgeNum,
        rscb->grpId);

    ret = pthread_mutex_init(&mrCb->mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("pthread_mutex_init mr_cb mutex failed, ret:%d", ret), ret);

    flag = ((unsigned long)pingCb->logicDevid << BUFF_FLAGS_DEVID_OFFSET) | BUFF_SP_SVM;
    ret = DlHalBuffAllocAlignEx(mrCb->len, (unsigned int)RA_RS_4K_PAGE_SIZE, flag, (int)rscb->grpId,
        (void **)&mrCb->addr);
    if (ret != 0) {
        hccp_err("DlHalBuffAllocAlignEx failed, length:0x%llx, dev_id:0x%x, flag:0x%lx, grpId:%u, ret:%d", mrCb->len,
            pingCb->logicDevid, flag, rscb->grpId, ret);
        goto alloc_fail;
    }

    mrCb->ibMr = RsDrvMrReg(pingCb->rdevCb.ibPd, (char *)(uintptr_t)mrCb->addr, mrCb->len,
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ);
    if (mrCb->ibMr == NULL) {
        ret = -errno;
        hccp_err("rs_ibv_reg_mr fail, ret:%d addr:0x%llx len:0x%llx", ret, mrCb->addr, mrCb->len);
        goto mr_reg_fail;
    }

    // init sge list
    mrCb->sgeList = calloc(mrCb->sgeNum, sizeof(struct ibv_sge));
    if (mrCb->sgeList == NULL) {
        ret = -errno;
        hccp_err("calloc fail, ret:%d sgeNum:%u", ret, mrCb->sgeNum);
        goto calloc_fail;
    }
    for (idx = 0; idx < mrCb->sgeNum; idx++) {
        mrCb->sgeList[idx].lkey = mrCb->ibMr->lkey;
        mrCb->sgeList[idx].length = mrCb->payloadOffset;
        if (idx == 0) {
            mrCb->sgeList[idx].addr = mrCb->addr;
        } else {
            mrCb->sgeList[idx].addr = mrCb->sgeList[idx - 1].addr + mrCb->payloadOffset;
        }
    }
    mrCb->sgeIdx = 0;

    hccp_info_rma("addr:0x%llx lkey:%u ", mrCb->addr, mrCb->ibMr->lkey);

    return 0;

calloc_fail:
    (void)RsDrvMrDereg(mrCb->ibMr);
mr_reg_fail:
    (void)DlHalBuffFree((void *)(uintptr_t)mrCb->addr);
alloc_fail:
    (void)pthread_mutex_destroy(&mrCb->mutex);
    return ret;
}

STATIC void RsPingCommonDeinitMrCb(struct RsPingMrCb *mrCb)
{
    hccp_dbg("addr:0x%llx len:%llu", mrCb->addr, mrCb->len);
    free(mrCb->sgeList);
    mrCb->sgeList = NULL;
    (void)RsDrvMrDereg(mrCb->ibMr);
    (void)DlHalBuffFree((void *)(uintptr_t)mrCb->addr);
    (void)pthread_mutex_destroy(&mrCb->mutex);
}

STATIC int RsPingPongInitLocalBuffer(struct rs_cb *rscb, struct PingInitAttr *attr, struct PingInitInfo *info,
    struct RsPingCtxCb *pingCb)
{
    int ret;

    // prepare ping_qp send mr
    pingCb->pingQp.sendMrCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pingQp.sendMrCb.len = pingCb->pingQp.qpCap.max_send_wr * pingCb->pingQp.sendMrCb.payloadOffset;
    pingCb->pingQp.sendMrCb.sgeNum = pingCb->pingQp.qpCap.max_send_wr;
    ret = RsPingCommonInitMrCb(rscb, pingCb, &pingCb->pingQp.sendMrCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ping_common_init_mr_cb ping_qp send_mr_cb failed, ret %d", ret), ret);
    // prepare ping_qp recv mr
    pingCb->pingQp.recvMrCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pingQp.recvMrCb.len = pingCb->pingQp.qpCap.max_recv_wr * pingCb->pingQp.recvMrCb.payloadOffset;
    pingCb->pingQp.recvMrCb.sgeNum = pingCb->pingQp.qpCap.max_recv_wr;
    ret = RsPingCommonInitMrCb(rscb, pingCb, &pingCb->pingQp.recvMrCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_mr_cb ping_qp recv_mr_cb failed, ret %d", ret);
        goto init_ping_qp_recv_mr_fail;
    }

    // prepare pong_qp send mr
    pingCb->pongQp.sendMrCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pongQp.sendMrCb.len = pingCb->pongQp.qpCap.max_send_wr * pingCb->pongQp.sendMrCb.payloadOffset;
    pingCb->pongQp.sendMrCb.sgeNum = pingCb->pongQp.qpCap.max_send_wr;
    ret = RsPingCommonInitMrCb(rscb, pingCb, &pingCb->pongQp.sendMrCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_mr_cb pong_qp send_mr_cb failed, ret %d", ret);
        goto init_pong_qp_send_mr_fail;
    }
    // prepare pong_qp recv mr
    pingCb->pongQp.recvMrCb.payloadOffset = PING_TOTAL_PAYLOAD_MAX_SIZE;
    pingCb->pongQp.recvMrCb.len = attr->bufferSize;
    pingCb->pongQp.recvMrCb.sgeNum = attr->bufferSize / pingCb->pongQp.recvMrCb.payloadOffset;
    ret = RsPingCommonInitMrCb(rscb, pingCb, &pingCb->pongQp.recvMrCb);
    if (ret != 0) {
        hccp_err("rs_ping_common_init_mr_cb pong_qp recv_mr_cb failed, ret %d", ret);
        goto init_pong_qp_recv_mr_fail;
    }
    info->result.bufferVa = pingCb->pongQp.recvMrCb.addr;
    info->result.bufferSize = attr->bufferSize;
    info->result.payloadOffset = pingCb->pongQp.recvMrCb.payloadOffset;
    info->result.headerSize = RS_PING_PAYLOAD_HEADER_RESV_GRH + RS_PING_PAYLOAD_HEADER_RESV_CUSTOM;

    return 0;

init_pong_qp_recv_mr_fail:
    RsPingCommonDeinitMrCb(&pingCb->pongQp.sendMrCb);
init_pong_qp_send_mr_fail:
    RsPingCommonDeinitMrCb(&pingCb->pingQp.recvMrCb);
init_ping_qp_recv_mr_fail:
    RsPingCommonDeinitMrCb(&pingCb->pingQp.sendMrCb);
    return ret;
}

STATIC int RsPingCommonPostRecv(struct RsPingLocalQpCb *qpCb)
{
    struct ibv_recv_wr *badWr = NULL;
    struct ibv_recv_wr wr = {0};
    struct ibv_sge list = {0};
    uint32_t sgeIdx;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->recvMrCb.mutex);
    sgeIdx = qpCb->recvMrCb.sgeIdx;
    (void)memcpy_s(&list, sizeof(struct ibv_sge), &qpCb->recvMrCb.sgeList[sgeIdx], sizeof(struct ibv_sge));
    qpCb->recvMrCb.sgeIdx = (sgeIdx + 1) % qpCb->recvMrCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->recvMrCb.mutex);

    wr.wr_id = (uint64_t)sgeIdx;
    wr.next = NULL;
    wr.sg_list = &list;
    wr.num_sge = 1;

    ret = RsIbvPostRecv(qpCb->ibQp, &wr, &badWr);
    if (ret != 0) {
        hccp_err("rs_ibv_post_recv failed, ret:%d", ret);
        return ret;
    }

    return 0;
}

STATIC int RsPingCommonInitPostRecvAll(struct RsPingLocalQpCb *qpCb)
{
    int ret = 0;
    uint32_t i;

    // prepare RQ wqe
    for (i = qpCb->recvMrCb.sgeIdx; i < qpCb->recvMrCb.sgeNum && i < qpCb->qpCap.max_recv_wr; i++) {
        ret = RsPingCommonPostRecv(qpCb);
        if (ret != 0) {
            hccp_err("rs_ping_common_post_recv %u-th rqe failed, ret:%d", i, ret);
            break;
        }
    }

    return ret;
}

STATIC void RsPingCommonDeinitLocalBuffer(struct RsPingCtxCb *pingCb)
{
    RsPingCommonDeinitMrCb(&pingCb->pongQp.recvMrCb);
    RsPingCommonDeinitMrCb(&pingCb->pongQp.sendMrCb);
    RsPingCommonDeinitMrCb(&pingCb->pingQp.recvMrCb);
    RsPingCommonDeinitMrCb(&pingCb->pingQp.sendMrCb);
}

STATIC void RsPingCommonDeinitLocalQp(struct rs_cb *rscb, struct RsPingCtxCb *pingCb, struct RsPingLocalQpCb *qpCb)
{
    (void)pingCb;
    if (qpCb == NULL || qpCb->channel == NULL) {
        hccp_err("qp_cb is NULL or qp_cb->channel is NULL");
        return;
    }

    (void)RsIbvDestroyQp(qpCb->ibQp);
    RsIbvAckCqEvents(qpCb->recvCq.ibCq, qpCb->recvCq.numEvents);
    qpCb->recvCq.numEvents = 0;
    (void)RsIbvDestroyCq(qpCb->recvCq.ibCq);
    (void)RsEpollCtl(rscb->connCb.epollfd, EPOLL_CTL_DEL, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
    (void)RsIbvDestroyCompChannel(qpCb->channel);
    qpCb->channel = NULL;
    (void)RsIbvDestroyCq(qpCb->sendCq.ibCq);
}

STATIC int RsPingPongInitLocalInfo(struct rs_cb *rscb, struct PingInitAttr *attr, struct PingInitInfo *info,
    struct RsPingCtxCb *pingCb)
{
    int ret;

    ret = RsPingCommonInitLocalQp(rscb, pingCb, &attr->client, &pingCb->pingQp);
    CHK_PRT_RETURN(ret != 0, hccp_err("init ping_qp failed, ret:%d", ret), ret);
    info->client.version = 0;
    (void)memcpy_s(&info->client.rdma.gid, sizeof(union HccpGid), &pingCb->rdevCb.gid, sizeof(union ibv_gid));
    info->client.rdma.qpn = pingCb->pingQp.ibQp->qp_num;
    info->client.rdma.qkey = pingCb->pingQp.qkey;

    ret = RsPingCommonInitLocalQp(rscb, pingCb, &attr->server, &pingCb->pongQp);
    if (ret != 0) {
        hccp_err("init pong_qp failed, ret:%d", ret);
        goto init_pong_qp_fail;
    }
    info->server.version = 0;
    (void)memcpy_s(&info->server.rdma.gid, sizeof(union HccpGid), &pingCb->rdevCb.gid, sizeof(union ibv_gid));
    info->server.rdma.qpn = pingCb->pongQp.ibQp->qp_num;
    info->server.rdma.qkey = pingCb->pongQp.qkey;

    ret = RsPingPongInitLocalBuffer(rscb, attr, info, pingCb);
    if (ret != 0) {
        hccp_err("init buffer failed, ret:%d", ret);
        goto init_buffer_fail;
    }

    ret = RsPingCommonInitPostRecvAll(&pingCb->pingQp);
    if (ret != 0) {
        hccp_err("ping_qp post recv failed, ret:%d", ret);
        goto post_recv_fail;
    }
    ret = RsPingCommonInitPostRecvAll(&pingCb->pongQp);
    if (ret != 0) {
        hccp_err("pong_qp post recv failed, ret:%d", ret);
        goto post_recv_fail;
    }

    return 0;

post_recv_fail:
    RsPingCommonDeinitLocalBuffer(pingCb);
init_buffer_fail:
    RsPingCommonDeinitLocalQp(rscb, pingCb, &pingCb->pongQp);
init_pong_qp_fail:
    RsPingCommonDeinitLocalQp(rscb, pingCb, &pingCb->pingQp);
    return ret;
}

STATIC int RsPingRocePingCbInit(unsigned int phyId, struct PingInitAttr *attr, struct PingInitInfo *info,
    unsigned int *devIndex, struct RsPingCtxCb *pingCb)
{
    struct rdev *rdevInfo = &attr->dev.rdma;
    struct rs_cb *rscb = NULL;
    int ret;

    ret = RsGetRsCb(phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, phyId[%u] invalid, ret %d", phyId, ret), ret);

    // prepare input attr
    pingCb->rdevCb.ip.family = (uint32_t)rdevInfo->family;
    pingCb->rdevCb.ip.binAddr = rdevInfo->localIp;
    ret = RsInetNtop(rdevInfo->family, &rdevInfo->localIp, pingCb->rdevCb.ip.readAddr, RS_MAX_IP_LEN);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_inet_ntop failed, ret %d", ret), -EINVAL);
    (void)memcpy_s(&pingCb->commInfo, sizeof(struct PingLocalCommInfo), &attr->commInfo,
        sizeof(struct PingLocalCommInfo));

    // open device & alloc global pd
    pingCb->rdevCb.devList = RsIbvGetDeviceList(&pingCb->rdevCb.devNum);
    if (pingCb->rdevCb.devList == NULL || pingCb->rdevCb.devNum == 0) {
        hccp_err("dev_list is NULL or dev_num[%d] is 0", pingCb->rdevCb.devNum);
        ret = -ENODEV;
        goto get_device_list_fail;
    }

    pingCb->rdevCb.ibPort = RS_PORT_DEF;
    ret = RsPingCbGetIbCtxAndIndex(rdevInfo, pingCb);
    if (ret != 0) {
        hccp_err("rs_ping_cb_get_ib_ctx_and_index failed, ret:%d", ret);
        goto get_ib_ctx_and_index_fail;
    }

    pingCb->rdevCb.ibPd = RsIbvAllocPd(pingCb->rdevCb.ibCtx);
    if (pingCb->rdevCb.ibPd == NULL) {
        hccp_err("rs_ibv_alloc_pd failed, errno:%d", errno);
        ret = -ENOMEM;
        goto alloc_pd_fail;
    }

    // init cq & qp & mr info, prepare output info
    info->version = 0;
    ret = RsPingPongInitLocalInfo(rscb, attr, info, pingCb);
    if (ret != 0) {
        hccp_err("rs_ping_pong_init_local_info failed, ret=%d phyId:%u", ret, rdevInfo->phyId);
        goto init_local_info_fail;
    }

    *devIndex = pingCb->devIndex;
    return 0;

init_local_info_fail:
    (void)RsIbvDeallocPd(pingCb->rdevCb.ibPd);
alloc_pd_fail:
    (void)RsIbvCloseDevice(pingCb->rdevCb.ibCtx);
get_ib_ctx_and_index_fail:
    RsIbvFreeDeviceList(pingCb->rdevCb.devList);
get_device_list_fail:
    (void)pthread_mutex_destroy(&pingCb->pingMutex);
    (void)pthread_mutex_destroy(&pingCb->pongMutex);
    return ret;
}

STATIC bool RsPingCommonCompareRdmaInfo(struct PingQpInfo *a, struct PingQpInfo *b)
{
    if (a->rdma.qpn != b->rdma.qpn) {
        return false;
    }
    if (a->rdma.qkey != b->rdma.qkey) {
        return false;
    }
    if (memcmp(&a->rdma.gid, &b->rdma.gid, sizeof(union HccpGid)) != 0) {
        return false;
    }
    return true;
}

STATIC int RsPingRoceFindTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target,
    struct RsPingTargetInfo **node)
{
    struct RsPingTargetInfo *targetNext = NULL;
    struct RsPingTargetInfo *targetCurr = NULL;

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &pingCb->pingList, list, struct RsPingTargetInfo);
    for (; (&targetCurr->list) != &pingCb->pingList;
         targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPingTargetInfo, list)) {
        if (RsPingCommonCompareRdmaInfo(&targetCurr->qpInfo, target)) {
            *node = targetCurr;
            RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    hccp_info_rma("ping target node for qpn:%u gid:%016llx:%016llx not found", target->rdma.qpn,
        target->rdma.gid.global.subnetPrefix, target->rdma.gid.global.interfaceId);
    return -ENODEV;
}

STATIC int RsPingCommonCreateAh(struct RsPingCtxCb *pingCb, struct PingLocalCommInfo *localInfo,
    struct PingQpInfo *remoteInfo, struct ibv_ah **ah)
{
    struct ibv_exp_ah_attr attrx = {0};
    struct ibv_global_route grh = {0};
    struct ibv_ah_attr attr = {0};
    struct ibv_ah *ahTmp = NULL;
    int ret = 0;

    (void)memcpy_s(&grh.dgid, sizeof(union ibv_gid), &remoteInfo->rdma.gid, sizeof(union HccpGid));
    grh.flow_label = localInfo->rdma.flowLabel;
    grh.sgid_index = (uint8_t)pingCb->rdevCb.gidIdx;
    grh.hop_limit = localInfo->rdma.hopLimit;
    grh.traffic_class = localInfo->rdma.qosAttr.tc;

    attr.grh = grh;
    attr.sl = localInfo->rdma.qosAttr.sl;
    attr.is_global = 1;
    attr.port_num = pingCb->rdevCb.ibPort;
    attrx.attr = attr;
    attrx.udp_sport = localInfo->rdma.udpSport;

    hccp_dbg("remote_qpn:%u flow_label:%u sgid_index:%u hop_limit:%u traffic_class:%u sl:%u is_global:%u "
             "port_num:%u udp_sport:%u",
        remoteInfo->rdma.qpn, grh.flow_label, grh.sgid_index, grh.hop_limit, grh.traffic_class, attr.sl, attr.is_global,
        attr.port_num, attrx.udp_sport);

    ahTmp = RsIbvExpCreateAh(pingCb->rdevCb.ibPd, &attrx);
    if (ahTmp == NULL) {
        ret = -EFAULT;
        hccp_err("rs_ibv_exp_create_ah failed, errno:%d", errno);
        return ret;
    }

    *ah = ahTmp;
    return ret;
}

STATIC int RsPingRoceAllocTargetNode(struct RsPingCtxCb *pingCb, struct PingTargetInfo *target,
    struct RsPingTargetInfo **node)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    int ret;

    targetInfo = (struct RsPingTargetInfo *)calloc(1, sizeof(struct RsPingTargetInfo));
    CHK_PRT_RETURN(targetInfo == NULL, hccp_err("calloc target_info fail! errno:%d", errno), -ENOMEM);

    ret = pthread_mutex_init(&targetInfo->tripMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init trip_mutex failed, ret:%d", ret);
        goto free_target_info;
    }

    targetInfo->payloadSize = target->payload.size;
    if (target->payload.size > 0) {
        targetInfo->payloadBuffer = (char *)calloc(1, target->payload.size);
        if (targetInfo->payloadBuffer == NULL) {
            hccp_err("calloc payload_buffer fail! size:%u errno:%d", target->payload.size, errno);
            ret = -ENOMEM;
            goto free_trip_mutex;
        }
        (void)memcpy_s(targetInfo->payloadBuffer, target->payload.size, target->payload.buffer, target->payload.size);
    }

    (void)memcpy_s(&targetInfo->qpInfo, sizeof(struct PingQpInfo), &target->remoteInfo.qpInfo,
        sizeof(struct PingQpInfo));
    ret = RsPingCommonCreateAh(pingCb, &target->localInfo, &target->remoteInfo.qpInfo, &targetInfo->ah);
    if (ret != 0) {
        hccp_err("rs_ping_common_create_ah fail! ret:%d", ret);
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

STATIC void RsPingRoceResetRecvBuffer(struct RsPingCtxCb *pingCb)
{
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongQp.recvMrCb.mutex);
    (void)memset_s((void *)(uintptr_t)pingCb->pongQp.recvMrCb.addr, pingCb->pongQp.recvMrCb.len, 0,
        pingCb->pongQp.recvMrCb.len);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongQp.recvMrCb.mutex);
}

STATIC void RsPingQpBuildUpWr(struct RsPingTargetInfo *target, struct ibv_sge *list, struct ibv_send_wr *wr)
{
    wr->wr_id = target->uuid;
    wr->next = NULL;
    wr->sg_list = list;
    wr->num_sge = 1;
    wr->opcode = IBV_WR_SEND;
    wr->send_flags = IBV_SEND_SIGNALED;
    wr->wr.ud.ah = target->ah;
    wr->wr.ud.remote_qpn = target->qpInfo.rdma.qpn;
    wr->wr.ud.remote_qkey = target->qpInfo.rdma.qkey;
}

STATIC int RsPingRocePostSend(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target)
{
    struct RsPingPayloadHeader *header = NULL;
    struct ibv_send_wr *badWr = NULL;
    struct timeval timestamp = {0};
    struct ibv_send_wr wr = {0};
    struct ibv_sge list = {0};
    uint32_t sgeIdx;
    int ret = 0;

    hccp_dbg("target uuid:0x%llx state:%d payload_size:%u qpn:%u gid:%016llx:%016llx", target->uuid, target->state,
        target->payloadSize, target->qpInfo.rdma.qpn, target->qpInfo.rdma.gid.global.subnetPrefix,
        target->qpInfo.rdma.gid.global.interfaceId);

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingQp.sendMrCb.mutex);
    sgeIdx = pingCb->pingQp.sendMrCb.sgeIdx;
    (void)memcpy_s(&list, sizeof(struct ibv_sge), &pingCb->pingQp.sendMrCb.sgeList[sgeIdx], sizeof(struct ibv_sge));
    pingCb->pingQp.sendMrCb.sgeIdx = (sgeIdx + 1) % pingCb->pingQp.sendMrCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingQp.sendMrCb.mutex);

    // prepare ping_qp send buffer
    (void)memset_s((void *)(uintptr_t)list.addr, list.length, 0, list.length);
    header = (struct RsPingPayloadHeader *)(uintptr_t)list.addr;
    header->type = RS_PING_TYPE_ROCE_DETECT;
    (void)memcpy_s(&header->server.rdma.gid, sizeof(union HccpGid), &pingCb->rdevCb.gid, sizeof(union ibv_gid));
    header->server.rdma.qpn = pingCb->pongQp.ibQp->qp_num;
    header->server.rdma.qkey = pingCb->pongQp.qkey;
    (void)memcpy_s(&header->target, sizeof(struct PingQpInfo), &target->qpInfo, sizeof(struct PingQpInfo));

    if (target->payloadSize > 0) {
        ret = memcpy_s((void *)(uintptr_t)(list.addr + RS_PING_PAYLOAD_HEADER_RESV_CUSTOM),
            (list.length - RS_PING_PAYLOAD_HEADER_RESV_CUSTOM), (void *)target->payloadBuffer, target->payloadSize);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("memcpy_s buffer payload_size:%u list.length:%u failed, ret:%d", target->payloadSize,
                (list.length - RS_PING_PAYLOAD_HEADER_RESV_CUSTOM), ret),
            -ESAFEFUNC);
    }
    list.length = RS_PING_PAYLOAD_HEADER_RESV_CUSTOM + target->payloadSize;

    RsPingQpBuildUpWr(target, &list, &wr);

    // record timestamp t1
    (void)gettimeofday(&timestamp, NULL);
    header->timestamp.tvSec1 = (uint64_t)timestamp.tv_sec;
    header->timestamp.tvUsec1 = (uint64_t)timestamp.tv_usec;
    header->taskId = pingCb->taskId;
    header->magic = 0x55AA;

    ret = RsIbvPostSend(pingCb->pingQp.ibQp, &wr, &badWr);
    if (ret != 0) {
        hccp_err("rs_ibv_post_send qpn:%u failed, ret:%d", pingCb->pingQp.ibQp->qp_num, ret);
        RS_PTHREAD_MUTEX_LOCK(&target->tripMutex);
        target->state = RS_PING_PONG_TARGET_ERROR;
        RS_PTHREAD_MUTEX_ULOCK(&target->tripMutex);
    }
    return ret;
}

STATIC int RsPingRocePollScq(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *target)
{
    struct ibv_wc wc = {0};
    int polledCnt;

    polledCnt = RsIbvPollCq(pingCb->pingQp.sendCq.ibCq, 1, &wc);
    if (polledCnt != 1) {
        hccp_err("uuid:0x%llx rs_ibv_poll_cq polled_cnt:%d", target->uuid, polledCnt);
        target->state = RS_PING_PONG_TARGET_ERROR;
        return -ENODATA;
    }
    if (wc.status != IBV_WC_SUCCESS) {
        target->state = RS_PING_PONG_TARGET_ERROR;
        hccp_err("wr_id:0x%llx error cqe %s(%d)", wc.wr_id, RsIbvWcStatusStr(wc.status), wc.status);
        return -EOPENSRC;
    }
    return 0;
}

STATIC int RsPingRocePollRcq(struct RsPingCtxCb *pingCb, int *polledCnt, struct timeval *timestamp2)
{
    struct ibv_cq *evCq = NULL;
    void *evCtx = NULL;
    int ret;

    // record timestamp t2
    (void)gettimeofday(timestamp2, NULL);

    ret = RsIbvGetCqEvent(pingCb->pingQp.channel, &evCq, &evCtx);
    if (ret != 0) {
        hccp_err("rs_ibv_get_cq_event ping_qp.channel failed, ret:%d", ret);
        return -EOPENSRC;
    }

    if (evCq != pingCb->pingQp.recvCq.ibCq) {
        hccp_err("CQ event for unknown CQ");
        return -EOPENSRC;
    }
    pingCb->pingQp.recvCq.numEvents++;

    *polledCnt = RsIbvPollCq(evCq, pingCb->pingQp.recvCq.maxRecvWcNum, gPingQpRecvWc);
    CHK_PRT_RETURN(*polledCnt > pingCb->pingQp.recvCq.maxRecvWcNum || *polledCnt < 0,
        hccp_err("ping_poll_rcq failed, ret:%d", *polledCnt), -EOPENSRC);

    return 0;
}

STATIC int RsPingCommonPollScq(struct RsPingLocalQpCb *qpCb)
{
    struct ibv_wc wc = {0};
    int polledCnt;

    polledCnt = RsIbvPollCq(qpCb->sendCq.ibCq, 1, &wc);
    if (polledCnt < 0) {
        hccp_warn_rma("rs_ibv_poll_cq unsuccessful, polledCnt:%d", polledCnt);
    } else if (polledCnt > 0) {
        if (wc.status != IBV_WC_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe %s(%d)", wc.wr_id, RsIbvWcStatusStr(wc.status), wc.status);
            return -EOPENSRC;
        }
    }

    return 0;
}

STATIC int RsPongFindTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target, struct RsPongTargetInfo **node)
{
    struct RsPongTargetInfo *targetNext = NULL;
    struct RsPongTargetInfo *targetCurr = NULL;

    RS_CHECK_POINTER_NULL_WITH_RET(pingCb);
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongMutex);
    RS_LIST_GET_HEAD_ENTRY(targetCurr, targetNext, &pingCb->pongList, list, struct RsPongTargetInfo);
    for (; (&targetCurr->list) != &pingCb->pongList;
         targetCurr = targetNext, targetNext = list_entry(targetNext->list.next, struct RsPongTargetInfo, list)) {
        if (RsPingCommonCompareRdmaInfo(&targetCurr->qpInfo, target)) {
            *node = targetCurr;
            RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);
            return 0;
        }
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);

    hccp_info_rma("pong target node for qpn:%u gid:%016llx:%016llx not found", target->rdma.qpn,
        target->rdma.gid.global.subnetPrefix, target->rdma.gid.global.interfaceId);
    return -ENODEV;
}

STATIC int RsPongFindAllocTargetNode(struct RsPingCtxCb *pingCb, struct PingQpInfo *target,
    struct RsPongTargetInfo **node)
{
    struct RsPongTargetInfo *targetInfo = NULL;
    int ret;

    ret = RsPongFindTargetNode(pingCb, target, node);
    if (ret == 0 && (*node)->state == RS_PING_PONG_TARGET_READY) {
        return 0;
    } else if (ret == 0) {
        targetInfo = *node;
        hccp_info_rma("delete pong target uuid:0x%llx state:%d, realloc again", targetInfo->uuid, targetInfo->state);
        RsListDel(&targetInfo->list);
        if (targetInfo->ah) {
            (void)RsIbvDestroyAh(targetInfo->ah);
        }
        free(targetInfo);
        targetInfo = NULL;
    }

    targetInfo = (struct RsPongTargetInfo *)calloc(1, sizeof(struct RsPongTargetInfo));
    CHK_PRT_RETURN(targetInfo == NULL, hccp_err("calloc target_info fail! errno:%d", errno), -ENOMEM);

    (void)memcpy_s(&targetInfo->qpInfo, sizeof(struct PingQpInfo), target, sizeof(struct PingQpInfo));
    ret = RsPingCommonCreateAh(pingCb, &pingCb->commInfo, target, &targetInfo->ah);
    if (ret != 0) {
        hccp_err("rs_ping_common_create_ah fail! ret:%d", ret);
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

STATIC int RsPongPostSend(struct RsPingCtxCb *pingCb, struct ibv_wc *wc, struct timeval *timestamp2)
{
    struct RsPongTargetInfo *targetInfo = NULL;
    struct RsPingPayloadHeader *header = NULL;
    struct ibv_send_wr *badWr = NULL;
    struct timeval timestamp3 = {0};
    struct ibv_sge recvList = {0};
    struct ibv_sge sendList = {0};
    struct ibv_send_wr wr = {0};
    uint32_t recvSgeIdx;
    uint32_t sendSgeIdx;
    int ret = 0;

    // poll send cq
    (void)RsPingCommonPollScq(&pingCb->pongQp);

    // handle detect packet & send response packet
    recvSgeIdx = (uint32_t)wc->wr_id;
    if (recvSgeIdx >= pingCb->pingQp.recvMrCb.sgeNum) {
        hccp_err("param err recv_sge_idx:%u >= sge_num:%u", recvSgeIdx, pingCb->pingQp.recvMrCb.sgeNum);
        return -EIO;
    }
    (void)memcpy_s(&recvList, sizeof(struct ibv_sge), &pingCb->pingQp.recvMrCb.sgeList[recvSgeIdx],
        sizeof(struct ibv_sge));

    RS_PTHREAD_MUTEX_LOCK(&pingCb->pongQp.sendMrCb.mutex);
    sendSgeIdx = pingCb->pongQp.sendMrCb.sgeIdx;
    (void)memcpy_s(&sendList, sizeof(struct ibv_sge), &pingCb->pongQp.sendMrCb.sgeList[sendSgeIdx],
        sizeof(struct ibv_sge));
    pingCb->pongQp.sendMrCb.sgeIdx = (sendSgeIdx + 1) % pingCb->pongQp.sendMrCb.sgeNum;
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongQp.sendMrCb.mutex);

    // UD consume 40 Bytes for GRH
    if (wc->byte_len < RS_PING_PAYLOAD_HEADER_RESV_GRH || wc->byte_len > PING_TOTAL_PAYLOAD_MAX_SIZE) {
        hccp_err("param err wc->byte_len:%u < %u or wc->byte_len:%u > %u", wc->byte_len,
            RS_PING_PAYLOAD_HEADER_RESV_GRH, wc->byte_len, PING_TOTAL_PAYLOAD_MAX_SIZE);
        return -EIO;
    }
    ret = memcpy_s((void *)(uintptr_t)sendList.addr, sendList.length,
        (void *)(uintptr_t)(recvList.addr + RS_PING_PAYLOAD_HEADER_RESV_GRH),
        wc->byte_len - RS_PING_PAYLOAD_HEADER_RESV_GRH);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("memcpy_s buffer wc->byte_len:%u send_list.length:%u failed, ret:%d", wc->byte_len, sendList.length,
            ret),
        -ESAFEFUNC);
    sendList.length = wc->byte_len - RS_PING_PAYLOAD_HEADER_RESV_GRH;
    header = (struct RsPingPayloadHeader *)(uintptr_t)sendList.addr;
    header->type = RS_PING_TYPE_ROCE_RESPONSE;

    ret = RsPongFindAllocTargetNode(pingCb, &header->server, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_pong_find_alloc_target_node failed, ret:%d", ret);
        return ret;
    }

    wr.wr_id = targetInfo->uuid;
    wr.next = NULL;
    wr.sg_list = &sendList;
    wr.num_sge = 1;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    wr.wr.ud.ah = targetInfo->ah;
    wr.wr.ud.remote_qpn = targetInfo->qpInfo.rdma.qpn;
    wr.wr.ud.remote_qkey = targetInfo->qpInfo.rdma.qkey;

    // record timestamp t3
    (void)gettimeofday(&timestamp3, NULL);
    header->timestamp.tvSec2 = (uint64_t)timestamp2->tv_sec;
    header->timestamp.tvUsec2 = (uint64_t)timestamp2->tv_usec;
    header->timestamp.tvSec3 = (uint64_t)timestamp3.tv_sec;
    header->timestamp.tvUsec3 = (uint64_t)timestamp3.tv_usec;
    header->magic = 0xAA55;

    ret = RsIbvPostSend(pingCb->pongQp.ibQp, &wr, &badWr);
    if (ret != 0) {
        targetInfo->state = RS_PING_PONG_TARGET_ERROR;
        hccp_err("rs_ibv_post_send failed, ret:%d", ret);
        return ret;
    }

    return ret;
}

STATIC void RsPongRoceHandleSend(struct RsPingCtxCb *pingCb, int polledCnt, struct timeval *timestamp2)
{
    struct ibv_wc *wc = NULL;
    int ret, i;

    wc = gPingQpRecvWc;
    for (i = 0; i < polledCnt; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe %s(%d)", wc[i].wr_id, RsIbvWcStatusStr(wc[i].status), wc[i].status);
            continue;
        }

        ret = RsPongPostSend(pingCb, &wc[i], timestamp2);
        if (ret != 0) {
            hccp_err("rs_pong_post_send failed, wrId:0x%llx", wc[i].wr_id);
            continue;
        }

        ret = RsPingCommonPostRecv(&pingCb->pingQp);
        if (ret != 0) {
            hccp_err("rs_ping_common_post_recv failed, ret:%d", ret);
            continue;
        }
    }

    ret = RsIbvReqNotifyCq(pingCb->pingQp.recvCq.ibCq, 0);
    if (ret != 0) {
        hccp_err("rs_ibv_req_notify_cq failed, ret:%d", ret);
    }

    return;
}

STATIC int RsPongResolveResponsePacket(struct RsPingCtxCb *pingCb, uint32_t sgeIdx, struct timeval *timestamp4)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    struct RsPingPayloadHeader *header = NULL;
    struct ibv_sge *recvList = NULL;
    uint32_t rtt;
    int ret;

    recvList = &pingCb->pongQp.recvMrCb.sgeList[sgeIdx];
    // UD consume 40 Bytes for GRH
    header = (struct RsPingPayloadHeader *)(uintptr_t)(recvList->addr + RS_PING_PAYLOAD_HEADER_RESV_GRH);
    if (header->taskId != pingCb->taskId) {
        hccp_warn_rma("drop received packet, recv_task_id:%u, curr_task_id:%u", header->taskId, pingCb->taskId);
        return 0;
    }

    header->timestamp.tvSec4 = (uint64_t)timestamp4->tv_sec;
    header->timestamp.tvUsec4 = (uint64_t)timestamp4->tv_usec;
    rtt = RsPingGetTripTime(&header->timestamp);
    ret = RsPingRoceFindTargetNode(pingCb, &header->target, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_ping_roce_find_target_node failed, ret:%d qpn:%u gid:%016llx:%016llx rtt:%u", ret,
            header->target.rdma.qpn, header->target.rdma.gid.global.subnetPrefix,
            header->target.rdma.gid.global.interfaceId, rtt);
        return ret;
    }

    (void)memset_s((void *)header, RS_PING_PAYLOAD_HEADER_MASK_SIZE, 0, RS_PING_PAYLOAD_HEADER_MASK_SIZE);
    RS_PTHREAD_MUTEX_LOCK(&targetInfo->tripMutex);
    targetInfo->resultSummary.recvCnt++;
    targetInfo->resultSummary.taskId = header->taskId;
    // rtt timeout, increase timeout_cnt
    if ((targetInfo->resultSummary.taskAttr.timeoutInterval * RS_PING_MSEC_TO_USEC) < rtt) {
        targetInfo->resultSummary.timeoutCnt++;
        hccp_dbg("recv_cnt:%u timeout_interval:%u rtt:%u timeout_cnt:%u", targetInfo->resultSummary.recvCnt,
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

STATIC void RsPongRocePollRcq(struct RsPingCtxCb *pingCb)
{
    struct timeval timestamp = {0};
    struct ibv_cq *evCq = NULL;
    struct ibv_wc *wc = NULL;
    uint32_t recvSgeIdx;
    void *evCtx = NULL;
    int polledCnt, i;
    int ret;

    // record timestamp t4
    (void)gettimeofday(&timestamp, NULL);

    ret = RsIbvGetCqEvent(pingCb->pongQp.channel, &evCq, &evCtx);
    if (ret != 0) {
        hccp_err("rs_ibv_get_cq_event pong_qp.channel failed, ret:%d", ret);
        return;
    }

    if (evCq != pingCb->pongQp.recvCq.ibCq) {
        hccp_err("CQ event for unknown CQ");
        return;
    }
    pingCb->pongQp.recvCq.numEvents++;

    polledCnt = RsIbvPollCq(evCq, pingCb->pongQp.recvCq.maxRecvWcNum, gPongQpRecvWc);
    if (polledCnt > pingCb->pongQp.recvCq.maxRecvWcNum || polledCnt < 0) {
        hccp_err("rs_ibv_poll_cq failed, ret:%d", polledCnt);
        return;
    }

    wc = gPongQpRecvWc;
    for (i = 0; i < polledCnt; i++) {
        if (wc[i].status != IBV_WC_SUCCESS) {
            hccp_err("wr_id:0x%llx error cqe %s(%d)", wc[i].wr_id, RsIbvWcStatusStr(wc[i].status), wc[i].status);
            continue;
        }
        recvSgeIdx = (uint32_t)wc[i].wr_id;
        if (recvSgeIdx >= pingCb->pongQp.recvMrCb.sgeNum) {
            hccp_err("param err recv_sge_idx:%u > sge_num:%u", recvSgeIdx, pingCb->pongQp.recvMrCb.sgeNum);
            continue;
        }

        // handle response packet result
        ret = RsPongResolveResponsePacket(pingCb, recvSgeIdx, &timestamp);
        if (ret != 0) {
            continue;
        }

        ret = RsPingCommonPostRecv(&pingCb->pongQp);
        if (ret != 0) {
            continue;
        }
    }

    ret = RsIbvReqNotifyCq(evCq, 0);
    if (ret != 0) {
        hccp_err("rs_ibv_req_notify_cq failed, ret:%d", ret);
    }

    return;
}

STATIC int RsPingRoceGetTargetResult(struct RsPingCtxCb *pingCb, struct PingTargetCommInfo *target,
    struct PingResultInfo *result)
{
    struct RsPingTargetInfo *targetInfo = NULL;
    int ret;

    ret = RsPingRoceFindTargetNode(pingCb, &target->qpInfo, &targetInfo);
    if (ret != 0) {
        hccp_err("rs_ping_roce_find_target_node failed, ret:%d qpn:%u gid:%016llx:%016llx", ret,
            target->qpInfo.rdma.qpn, target->qpInfo.rdma.gid.global.subnetPrefix,
            target->qpInfo.rdma.gid.global.interfaceId);
        return ret;
    }

    (void)memcpy_s(&result->summary, sizeof(struct PingResultSummary), &targetInfo->resultSummary,
        sizeof(struct PingResultSummary));
    if (targetInfo->state == RS_PING_PONG_TARGET_FINISH) {
        result->state = PING_RESULT_STATE_VALID;
    } else {
        result->state = PING_RESULT_STATE_INVALID;
    }

    hccp_dbg("ip:0x%llx qpn:%u, state:%d sendCnt:%u recvCnt:%u timeoutCnt:%u rttMin:%u rttMax:%u rttAvg:%u",
        target->ip.addr.s_addr, target->qpInfo.rdma.qpn, result->state, result->summary.sendCnt,
        result->summary.recvCnt, result->summary.timeoutCnt, result->summary.rttMin, result->summary.rttMax,
        result->summary.rttAvg);

    return 0;
}

STATIC void RsPingRoceFreeTargetNode(struct RsPingCtxCb *pingCb, struct RsPingTargetInfo *targetInfo)
{
    RS_PTHREAD_MUTEX_LOCK(&pingCb->pingMutex);
    RsListDel(&targetInfo->list);
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pingMutex);

    if (targetInfo->payloadSize > 0 && targetInfo->payloadBuffer != NULL) {
        free(targetInfo->payloadBuffer);
        targetInfo->payloadBuffer = NULL;
    }

    if (targetInfo->ah) {
        (void)RsIbvDestroyAh(targetInfo->ah);
    }
    return;
}

STATIC void RsPingPongDelTargetList(struct RsPingCtxCb *pingCb)
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
        if (pingCurr->ah) {
            (void)RsIbvDestroyAh(pingCurr->ah);
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
        if (pongCurr->ah) {
            (void)RsIbvDestroyAh(pongCurr->ah);
        }
        free(pongCurr);
        pongCurr = NULL;
    }
    RS_PTHREAD_MUTEX_ULOCK(&pingCb->pongMutex);
}

STATIC void RsPingRocePingCbDeinit(unsigned int phyId, struct RsPingCtxCb *pingCb)
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

    RsPingPongDelTargetList(pingCb);

    RsPingCommonDeinitLocalQp(rscb, pingCb, &pingCb->pongQp);
    RsPingCommonDeinitLocalQp(rscb, pingCb, &pingCb->pingQp);
    RsPingCommonDeinitLocalBuffer(pingCb);
    (void)RsIbvDeallocPd(pingCb->rdevCb.ibPd);
    (void)RsIbvCloseDevice(pingCb->rdevCb.ibCtx);
    RsIbvFreeDeviceList(pingCb->rdevCb.devList);
}

STATIC void RsPingRoceAddTargetSuccess(struct PingTargetInfo *target, struct RsPingTargetInfo *targetInfo)
{
    hccp_info_rma("target ip:0x%llx payload_size:%u add success, qpn:%u uuid:0x%llx", target->remoteInfo.ip.addr.s_addr,
        target->payload.size, targetInfo->qpInfo.rdma.qpn, targetInfo->uuid);
}

STATIC void RsPingRocePingCbInitSuccess(unsigned int phyId, struct PingInitAttr *attr, unsigned int devIndex)
{
    hccp_run_info("ping_cb init success, phyId:%u, localIp:0x%x, devIndex:%u", phyId,
        attr->dev.rdma.localIp.addr.s_addr, devIndex);
}

STATIC void RsPingRoceCannotFindTargetNode(unsigned int i, int ret, struct PingTargetCommInfo target,
    unsigned int phyId)
{
    hccp_err("rs_ping_roce_find_target_node i:%u failed, ret:%d ip:0x%llx qpn:%u phyId:%u", i, ret,
        target.ip.addr.s_addr, target.qpInfo.rdma.qpn, phyId);
}

struct RsPingPongOps gRsPingRoceOps = {
    .checkPingFd = RsPingRoceCheckFd,
    .checkPongFd = RsPongRoceCheckFd,
    .initPingCb = RsPingRocePingCbInit,
    .pingFindTargetNode = RsPingRoceFindTargetNode,
    .pingAllocTargetNode = RsPingRoceAllocTargetNode,
    .resetRecvBuffer = RsPingRoceResetRecvBuffer,
    .pingPostSend = RsPingRocePostSend,
    .pingPollScq = RsPingRocePollScq,
    .pingPollRcq = RsPingRocePollRcq,
    .pongHandleSend = RsPongRoceHandleSend,
    .pongPollRcq = RsPongRocePollRcq,
    .getTargetResult = RsPingRoceGetTargetResult,
    .pingFreeTargetNode = RsPingRoceFreeTargetNode,
    .deinitPingCb = RsPingRocePingCbDeinit,
};

struct RsPingPongDfx gRsPingRoceDfx = {
    .addTargetSuccess = RsPingRoceAddTargetSuccess,
    .initPingCbSuccess = RsPingRocePingCbInitSuccess,
    .pingCannotFindTargetNode = RsPingRoceCannotFindTargetNode,
};

struct RsPingPongOps *RsPingRoceGetOps(void)
{
    return &gRsPingRoceOps;
}

struct RsPingPongDfx *RsPingRoceGetDfx(void)
{
    return &gRsPingRoceDfx;
}
