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
#include <unistd.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/epoll.h>
#include <sys/eventfd.h>
#include <sys/socket.h>
#include <errno.h>
#include "securec.h"
#include "config_log.h"
#include "rs.h"
#include "ra_rs_err.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "rs_rdma_inner.h"
#include "rs_epoll.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "dl_ibv_extend_function.h"
#include "rs_drv_socket.h"
#include "rs_drv_rdma.h"
#include "rs_rdma.h"

unsigned int gRsSendWrNum = 0;

STATIC struct RsListHead gRsTypicalCqList;
STATIC pthread_mutex_t gRsTypicalCqMutex = PTHREAD_MUTEX_INITIALIZER;

STATIC void RsBufPrint(char *addr, int len)
{
    int i;

    for (i = 0; i < len; i++) {
        hccp_info_rma("0x%02x ", *(addr + i));
    }
}

STATIC int RsGetQpcb(struct RsRdevCb *rdevCb, uint32_t qpn, struct RsQpCb **qpCb)
{
    struct RsQpCb *qpCbTmp = NULL;
    struct RsQpCb *qpCbTmp2 = NULL;

    RS_LIST_GET_HEAD_ENTRY(qpCbTmp, qpCbTmp2, &rdevCb->qpList, list, struct RsQpCb);
    for (; (&qpCbTmp->list) != &rdevCb->qpList;
         qpCbTmp = qpCbTmp2, qpCbTmp2 = list_entry(qpCbTmp2->list.next, struct RsQpCb, list)) {
        if (qpCbTmp->ibQp->qp_num == qpn) {
            *qpCb = qpCbTmp;
            return 0;
        }
    }

    *qpCb = NULL;
    hccp_err("qp_cb for qp %u is not available", qpn);

    return -ENODEV;
}

int RsQpn2qpcb(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct RsQpCb **qpCb)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;
    struct RsRdevCb *rdevCb = NULL;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error! phyId:%u", phyId), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_qpn2qpcb rsGetLocalDevIDByHostDevID phyId[%u] invalid, ret:%d", phyId, ret), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("rs_qpn2qpcb get rs_cb failed, ret:%d", ret), -ENODEV);

    ret = RsGetRdevCb(rsCb, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_rdev_cb failed! ret:%d, rdevIndex:%u", ret, rdevIndex), ret);

    ret = RsGetQpcb(rdevCb, qpn, qpCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_qpcb failed! ret:%d, qpn:%u", ret, qpn), ret);

    return 0;
}

STATIC int RsGetMrcb(struct RsQpCb *qpCb, uint64_t addr, struct RsMrCb **mrCb, struct RsListHead *mrList)
{
    struct RsMrCb *mrTmp = NULL;
    struct RsMrCb *mrTmp2 = NULL;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RS_LIST_GET_HEAD_ENTRY(mrTmp, mrTmp2, mrList, list, struct RsMrCb);
    for (; (&mrTmp->list) != mrList; mrTmp = mrTmp2, mrTmp2 = list_entry(mrTmp2->list.next, struct RsMrCb, list)) {
        if ((mrTmp->mrInfo.addr <= addr) && (addr < mrTmp->mrInfo.addr + mrTmp->mrInfo.len)) {
            *mrCb = mrTmp;
            RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
            return 0;
        }
    }

    *mrCb = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);

    hccp_info_rma("cannot find mrcb for addr@0x%lx !", addr);

    return -ENODEV;
}

STATIC void *RsNotifyMrListAdd(struct RsQpCb *qpCb, const char *buf)
{
    int ret;
    struct RsMrCb *notifyMrCb;

    notifyMrCb = calloc(1, sizeof(struct RsMrCb));
    CHK_PRT_RETURN(notifyMrCb == NULL, hccp_err("notify_mr_cb calloc failed"), NULL);
    ret = memcpy_s(&notifyMrCb->mrInfo, sizeof(struct RsMrInfo), &((const struct RsQpInfo *)buf)->notifyMr,
        sizeof(struct RsMrInfo));
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, src_len:%u, dst_len:%u", ret, sizeof(struct RsMrInfo),
            sizeof(struct RsMrInfo));
        free(notifyMrCb);
        notifyMrCb = NULL;
        return NULL;
    }

    hccp_info_rma("qpn is %d, rdevIndex:%u, chipId %u, recv notify va is 0x%llx, notify size is %llu",
        qpCb->qpInfoLo.qpn, qpCb->rdevCb->rdevIndex, qpCb->rdevCb->rsCb->chipId, notifyMrCb->mrInfo.addr,
        notifyMrCb->mrInfo.len);

    RsListAddTail(&notifyMrCb->list, &qpCb->remMrList);

    return notifyMrCb;
}

STATIC int RsQpStateModify(struct RsQpCb *qpCb)
{
    struct ibv_qp_init_attr initAttr = {0};
    struct ibv_qp_attr attr = {0};
    enum ibv_qp_state state;
    int ret;

    // see ib_modify_qp_is_ok for status modify, only support modify qp from INIT to RTR
    ret = RsIbvQueryQp(qpCb->ibQp, &attr, IBV_QP_STATE, &initAttr);
    if (ret != 0) {
        hccp_warn_rma("rs_ibv_query_qp qpn:%d unsuccessful, ret:%d", qpCb->qpInfoLo.qpn, ret);
        state = IBV_QPS_UNKNOWN;
    } else {
        state = attr.qp_state;
    }

    // disallow modify qp from IBV_QPS_RTS to IBV_QPS_RTS
    if (state == IBV_QPS_RTS) {
        hccp_err("qpn:%d disallow modify from %d", qpCb->qpInfoLo.qpn, state);
        return -EINVAL;
    }

    hccp_info_rma("qpn:%d state:%d start modify", qpCb->qpInfoLo.qpn, state);

    // modify qp from others to RESET
    if (state != IBV_QPS_RESET && state != IBV_QPS_INIT && state != IBV_QPS_RTR) {
        ret = RsDrvQpStateModifytoReset(qpCb);
        CHK_PRT_RETURN(ret, hccp_err("qpn:%d modify %d to reset failed, ret:%d", qpCb->qpInfoLo.qpn, state, ret), ret);
        state = IBV_QPS_RESET;
    }

    // modify qp from RESET to INIT
    if (state == IBV_QPS_RESET) {
        ret = RsDrvQpStateModifytoInit(qpCb, &attr);
        CHK_PRT_RETURN(ret, hccp_err("qpn:%d modify %d to init failed, ret %d", qpCb->qpInfoLo.qpn, state, ret), ret);
        state = IBV_QPS_INIT;
    }

    // modify qp from INIT to RTR
    if (state == IBV_QPS_INIT) {
        ret = RsDrvQpStateModifytoRtr(qpCb, &attr);
        CHK_PRT_RETURN(ret, hccp_err("qpn:%d modify %d to rtr failed, ret %d", qpCb->qpInfoLo.qpn, state, ret), ret);
        state = IBV_QPS_RTR;
    }

    // modify qp from RTR to RTS
    if (state == IBV_QPS_RTR) {
        ret = RsDrvQpStateModifytoRts(qpCb, &attr);
        CHK_PRT_RETURN(ret, hccp_err("qpn:%d modify %d to rts failed, ret %d", qpCb->qpInfoLo.qpn, state, ret), ret);
    }

    hccp_info_rma("local qpn[%d] remote qpn[%d] modify succ", qpCb->qpInfoLo.qpn, qpCb->qpInfoRem.qpn);

    return 0;
}

STATIC int RsEpollRecvQpHandle(struct RsQpCb *qpCb, const char *bufTmp)
{
    int ret;
    float timeCost = 0.0;

    ret = memcpy_s(&qpCb->qpInfoRem, sizeof(struct RsQpInfo), bufTmp, sizeof(struct RsQpInfo));
    CHK_PRT_RETURN(ret,
        hccp_err("memcpy_s failed[%d], dest size:%d, src size:%d", ret, sizeof(struct RsQpInfo),
            sizeof(struct RsQpInfo)),
        -ENOMEM);

    /* modify qp state to RTR/RTS */
    ret = RsQpStateModify(qpCb);
    CHK_PRT_RETURN(ret,
        hccp_err("rs_qp_state_modify local qpn[%d] remote qpn[%d] failed ret[%d]", qpCb->qpInfoLo.qpn,
            qpCb->qpInfoRem.qpn, ret),
        ret);

    RsGetCurTime(&qpCb->endTime);
    HccpTimeInterval(&qpCb->endTime, &qpCb->startTime, &timeCost);
    if (timeCost > RS_EXPECT_TIME_MAX) {
        hccp_warn_rma("local qpn[%d] remote qpn [%d] connect success cost[%f] more than[%f]ms!", qpCb->qpInfoLo.qpn,
            qpCb->qpInfoRem.qpn, timeCost, RS_EXPECT_TIME_MAX);
    } else {
        hccp_info_rma("local qpn[%d] remote qpn [%d] connect success! cost [%f] ms", qpCb->qpInfoLo.qpn,
            qpCb->qpInfoRem.qpn, timeCost);
    }

    hccp_info_rma("qp [%d] state has been migrate to RTS!, qpCb state is %d", qpCb->qpInfoLo.qpn, qpCb->state);

    return 0;
}

STATIC void *RsEpollRecvMrHandle(struct RsQpCb *qpCb, const char *bufTmp)
{
    int ret;
    struct RsMrCb *mrCb;

    mrCb = calloc(1, sizeof(struct RsMrCb));
    CHK_PRT_RETURN(mrCb == NULL, hccp_err("mr_cb calloc failed"), NULL);
    ret = memcpy_s(&mrCb->mrInfo, sizeof(struct RsMrInfo), bufTmp, sizeof(struct RsMrInfo));
    if (ret) {
        hccp_err("memcpy_s failed[%d], dest size:%u, src size:%u", ret, sizeof(struct RsMrInfo),
            sizeof(struct RsMrInfo));
        free(mrCb);
        mrCb = NULL;
        return NULL;
    }

    RsListAddTail(&mrCb->list, &qpCb->remMrList);

    hccp_info_rma("recv mr addr is 0x%llx", mrCb->mrInfo.addr);
    hccp_info_rma("recv mr len is %llu", mrCb->mrInfo.len);

    return mrCb;
}

STATIC int RsCmdQpInfoHandle(struct RsQpCb *qpCb, unsigned int totalSize, const char *bufTmp, unsigned int curSize,
    bool *flag)
{
    int ret;
    CHK_PRT_RETURN((totalSize - curSize) < sizeof(struct RsQpInfo),
        hccp_info_rma("qp_info remain size"
                      "[%u] < size [%u], wait for next recv",
            totalSize - curSize, sizeof(struct RsQpInfo)),
        -EINVAL);

    ret = RsEpollRecvQpHandle(qpCb, bufTmp);
    CHK_PRT_RETURN(ret, hccp_err("rs_epoll_recv_qp_handle failed! ret[%d]", ret), ret);

    RsNotifyMrListAdd(qpCb, bufTmp);
    hccp_info_rma("rs_notify_mr_list_add");

    *flag = true;
    hccp_info_rma("qp_info cur_size(%u) len(%u) !", curSize, sizeof(struct RsQpInfo));

    return 0;
}

STATIC int RsCmdMrInfoHandle(struct RsQpCb *qpCb, unsigned int totalSize, const char *bufTmp, unsigned int curSize,
    bool *flag)
{
    CHK_PRT_RETURN((totalSize - curSize) < sizeof(struct RsMrInfo),
        hccp_info_rma("mr_info remain size"
                      "[%u] < size [%u], wait for next recv",
            totalSize - curSize, sizeof(struct RsMrInfo)),
        -EINVAL);

    (void)RsEpollRecvMrHandle(qpCb, bufTmp);

    *flag = true;

    hccp_info_rma("mr_info cur_size(%u) len(%u) !", curSize, sizeof(struct RsMrInfo));

    return 0;
}

STATIC int RsCmdLenInfoHandle(struct RsQpCb *qpCb, unsigned int totalSize, const char *bufTmp, unsigned int curSize,
    bool *flag)
{
    CHK_PRT_RETURN((totalSize - curSize) < sizeof(struct RsQpLenInfo),
        hccp_info_rma("len_info remain size"
                      "[%u] < size [%u], wait for next recv",
            totalSize - curSize, sizeof(struct RsQpLenInfo)),
        -EINVAL);

    qpCb->expectLen = *((const uint32_t *)(bufTmp + sizeof(uint32_t)));

    *flag = true;

    return 0;
}

STATIC void RsEpollRecvHandleRemain(struct RsQpCb *qpCb, unsigned int totalSize, unsigned int curSize, bool flag,
    const char *bufTmp)
{
    int ret = 0;

    qpCb->remainSize = totalSize - curSize;
    if ((qpCb->remainSize > 0) && (flag == true)) {
        ret = memcpy_s(qpCb->qpMrBuf, RS_BUF_SIZE, bufTmp, qpCb->remainSize);
        if (ret) {
            hccp_err("memcpy_s failed, ret:%d, remainSize:%u", ret, qpCb->remainSize);
            return;
        }
    }

    return;
}

STATIC void RsEpollRecvHandle(struct RsQpCb *qpCb, char *buf, int size)
{
    unsigned int totalSize = qpCb->remainSize + (unsigned int)size;
    char *bufTmp = (char *)qpCb->qpMrBuf;
    unsigned int curSize = 0;
    bool flag = false;
    uint32_t cmd;
    int ret;

    hccp_info_rma("Message for qp:%d, qpCb->remainSize:%u, size:%d", qpCb->qpInfoLo.qpn, qpCb->remainSize, size);
    ret = memcpy_s(qpCb->qpMrBuf + qpCb->remainSize, RS_BUF_SIZE - qpCb->remainSize, buf, size);
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, remainSize:%u, size:%d", ret, qpCb->remainSize, size);
        return;
    }

    do {
        cmd = *((uint32_t *)bufTmp);
        switch (cmd) {
            case RS_CMD_QP_INFO:
                ret = RsCmdQpInfoHandle(qpCb, totalSize, bufTmp, curSize, &flag);
                if (ret) {
                    goto out;
                }

                curSize += sizeof(struct RsQpInfo);
                bufTmp = qpCb->qpMrBuf + curSize;
                break;
            case RS_CMD_MR_INFO:
                ret = RsCmdMrInfoHandle(qpCb, totalSize, bufTmp, curSize, &flag);
                if (ret) {
                    goto out;
                }

                curSize += sizeof(struct RsMrInfo);
                bufTmp = qpCb->qpMrBuf + curSize;
                break;
            case RS_CMD_LEN_INFO:
                ret = RsCmdLenInfoHandle(qpCb, totalSize, bufTmp, curSize, &flag);
                if (ret) {
                    goto out;
                }
                curSize += sizeof(struct RsQpLenInfo);
                bufTmp = qpCb->qpMrBuf + curSize;
                break;
            default:
                hccp_warn_rma("qp %d, unknown cmd(0x%x)!", qpCb->qpInfoLo.qpn, cmd);
                RsBufPrint(buf, size);
                return;
        }
    } while (curSize < totalSize);

out:
    RsEpollRecvHandleRemain(qpCb, totalSize, curSize, flag, bufTmp);
}

STATIC void RsQpMrRecvHandle(int fd, struct RsQpCb *qpCb)
{
    char buf[RS_BUF_SIZE];
    int size;
    int ret;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);

    size = RsSocketRecv(fd, buf, RS_BUF_SIZE - qpCb->remainSize);
    hccp_dbg("fd %d qpn %d read size = %d, qpCb->remainSize:%u", fd, qpCb->qpInfoLo.qpn, size, qpCb->remainSize);

    if (size > 0) {
        qpCb->recvLen += (uint32_t)size;
        RsEpollRecvHandle(qpCb, buf, size);
    } else if (size == 0) {
        hccp_dbg("fd %d read size = %d, remote fd has been closed, fd cannot use !", fd, size);
#ifdef CA_CONFIG_LLT
        qpCb->state = RS_QP_STATUS_REM_FD_CLOSE;
#endif
    } else {
        ret = errno;
        hccp_dbg("no data available, errno:%d", ret);
    }
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);

    return;
}

STATIC int RsHandleQpMrEpollEvent(struct RsRdevCb *rdevCb, int fd)
{
    struct RsQpCb *qpCb;
    struct RsQpCb *qpCb2 = NULL;

    /* QP event, QP info exchange */
    RS_LIST_GET_HEAD_ENTRY(qpCb, qpCb2, &rdevCb->qpList, list, struct RsQpCb);
    for (; (&qpCb->list) != &rdevCb->qpList; qpCb = qpCb2, qpCb2 = list_entry(qpCb2->list.next, struct RsQpCb, list)) {
        if (qpCb->channel == NULL) {
            continue;
        }
        if (qpCb->srqContext != NULL && qpCb->srqContext->channel->fd == fd) {
            hccp_dbg("fd %d poll cq!", fd);
            RsDrvPollSrqCqHandle(qpCb);
            return 0;
        }
        if (fd == qpCb->channel->fd) {
            hccp_dbg("fd %d poll cq!", fd);
            RsDrvPollCqHandle(qpCb);
            return 0;
        }
    }
    return -ENODEV;
}

int RsEpollEventQpMrInHandle(struct rs_cb *rsCb, int fd)
{
    int ret;
    struct RsRdevCb *rdevCbTmp = NULL;
    struct RsRdevCb *rdevCbTmp2 = NULL;

    if (rsCb->protocol != PROTOCOL_RDMA) {
        return -ENODEV;
    }

    RS_LIST_GET_HEAD_ENTRY(rdevCbTmp, rdevCbTmp2, &rsCb->rdevList, list, struct RsRdevCb);
    for (; (&rdevCbTmp->list) != &rsCb->rdevList;
         rdevCbTmp = rdevCbTmp2, rdevCbTmp2 = list_entry(rdevCbTmp2->list.next, struct RsRdevCb, list)) {
        RS_PTHREAD_MUTEX_LOCK(&rdevCbTmp->rdevMutex);
        ret = RsHandleQpMrEpollEvent(rdevCbTmp, fd);
        RS_PTHREAD_MUTEX_ULOCK(&rdevCbTmp->rdevMutex);
        if (ret == 0) {
            return 0;
        }
    }
    return -ENODEV;
}

STATIC int RsMrInfoSync(struct RsMrCb *mrCb)
{
    int ret;

    hccp_info_rma("mr state:%d, addr:0x%lx", mrCb->state, mrCb->mrInfo.addr);

    CHK_PRT_RETURN(mrCb->state & RS_MR_STATE_SYNCED,
        hccp_warn_rma("mr synced ! mr_cb->flag[%d] & [%d] != 0", mrCb->state, RS_MR_STATE_SYNCED), 0);

    /*
     * no socket available for MR_INFO exchange if allowed
     * need exchange when socket available
     */
    CHK_PRT_RETURN(mrCb->qpCb->connInfo == NULL, hccp_warn_rma("no conn available !"), 0);

    CHK_PRT_RETURN(mrCb->qpCb->state == RS_QP_STATUS_REM_FD_CLOSE,
        hccp_warn_rma("remote qp fd closed, "
                      "can not use it anymore! status[%d](RS_QP_STATUS_REM_FD_CLOSE)",
            mrCb->qpCb->state),
        -EFAULT);

    CHK_PRT_RETURN(mrCb->qpCb->connInfo->connfd == RS_FD_INVALID,
        hccp_warn_rma("mr info sync failed! fd not ready! connfd[%d](RS_FD_INVALID)", mrCb->qpCb->connInfo->connfd),
        -ENETUNREACH);

    mrCb->mrInfo.cmd = (unsigned int)RS_CMD_MR_INFO;
    ret = RsSocketSend(mrCb->qpCb->connInfo->connfd, &mrCb->mrInfo, sizeof(struct RsMrInfo));
    CHK_PRT_RETURN(ret != sizeof(struct RsMrInfo),
        hccp_err("mr_info send %d/%ld incomplete", ret, sizeof(struct RsMrInfo)), -EAGAIN);

    mrCb->qpCb->sendLen += (uint32_t)ret;
    mrCb->state |= RS_MR_STATE_SYNCED;
    hccp_info_rma("after send mr state:%d, addr:0x%lx", mrCb->state, mrCb->mrInfo.addr);

    return 0;
}

STATIC int RsMrPrepareRoceSign(unsigned int phyId, struct RsRdevCb *devCb, struct roce_process_sign *roceSign)
{
    unsigned int tmpPhyId = phyId;
    unsigned int chipId;
    int ret;

    // reg mr with backup phyId
    if (devCb->backupInfo.backupFlag) {
        tmpPhyId = devCb->backupInfo.rdevInfo.phyId;
        ret = DlDrvGetLocalDevIdByHostDevId(tmpPhyId, &chipId);
    } else {
        ret = rsGetLocalDevIDByHostDevID(tmpPhyId, &chipId);
    }
    CHK_PRT_RETURN(ret != 0,
        hccp_err("get chipId failed, ret %d, phyid[%u] backupFlag[%d]", ret, tmpPhyId, devCb->backupInfo.backupFlag),
        -EACCES);
    roceSign->tgid = devCb->rsCb->pRsSign.tgid;
    roceSign->devid = chipId;
    roceSign->vfid = 0;
    ret = strcpy_s(roceSign->sign, PROCESS_RS_SIGN_LENGTH, devCb->rsCb->pRsSign.sign);
    CHK_PRT_RETURN(ret != 0, hccp_err("Invalid pid sign, ret(%d)", ret), -ESAFEFUNC);
    return 0;
}

STATIC int RsMrPreReg(unsigned int phyId, struct RsQpCb *qpCb, struct RsMrCb *mrCb, struct RdmaMrRegInfo *mrRegInfo)
{
    unsigned long long len = mrRegInfo->len;
    struct roce_process_sign roceSign = {0};
    int access = mrRegInfo->access;
    char *addr = mrRegInfo->addr;
    int ret;

    if (qpCb->rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE || qpCb->rdevCb->rsCb->hccpMode == NETWORK_ONLINE ||
        qpCb->isExp == RS_NOT_EXP) {
        mrCb->ibMr = RsDrvMrReg(qpCb->ibPd, addr, len, access);
        CHK_PRT_RETURN(mrCb->ibMr == NULL, hccp_err("rs_drv_mr_reg addr is NULL len[%lld] failed ", len), -EACCES);
    } else {
        ret = RsMrPrepareRoceSign(phyId, qpCb->rdevCb, &roceSign);
        CHK_PRT_RETURN(ret != 0, hccp_err("RsMrPrepareRoceSign failed, ret(%d)", ret), ret);
        mrCb->ibMr = RsDrvExpMrReg(qpCb->ibPd, addr, len, access, roceSign);
        CHK_PRT_RETURN(mrCb->ibMr == NULL, hccp_err("rs_drv_exp_mr_reg addr is NULL len[%lld] failed ", len), -EACCES);
    }

    mrCb->mrInfo.cmd = (unsigned int)RS_CMD_MR_INFO;
    mrCb->mrInfo.addr = (uintptr_t)addr;
    mrCb->mrInfo.len = len;
    mrCb->mrInfo.rkey = mrCb->ibMr->rkey;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RsListAddTail(&mrCb->list, &qpCb->mrList);
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);

    qpCb->mrNum++;
    return 0;
}

STATIC int RsCallocMr(int num, struct RsMrCb **mrCb)
{
    CHK_PRT_RETURN(num <= 0, hccp_err("invalid num for mr calloc"), -EINVAL);

    *mrCb = calloc(num, sizeof(struct RsMrCb));
    CHK_PRT_RETURN((*mrCb) == NULL, hccp_err("calloc mr_cb failed"), -ENOMEM);
    return 0;
}

STATIC int RsCallocQpcb(int num, struct RsQpCb **qpCb)
{
    if (num <= 0) {
        return -EINVAL;
    }

    *qpCb = calloc(num, sizeof(struct RsQpCb));
    if ((*qpCb) == NULL) {
        return -ENOMEM;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsMrReg(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct RdmaMrRegInfo *mrRegInfo)
{
    int ret;
    struct RsQpCb *qpCb = NULL;
    struct RsMrCb *mrCb = NULL;

    CHK_PRT_RETURN(mrRegInfo == NULL || mrRegInfo->addr == NULL || mrRegInfo->len == 0 || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    hccp_info_rma("qpn[%u], len[0x%llx], access[%d]", qpn, mrRegInfo->len, mrRegInfo->access);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_qpn2qpcb qpn[%d] ret[%d] failed ", qpn, ret), ret);

    CHK_PRT_RETURN(qpCb->mrNum >= RS_MR_NUM_MAX, hccp_err("Exceeded the maximum MR limit %d", qpCb->mrNum), -EINVAL);

    ret = RsGetMrcb(qpCb, (uintptr_t)mrRegInfo->addr, &mrCb, &qpCb->mrList);
    if (ret == 0) {
        hccp_warn_rma("mr already registered");
        goto found;
    }

    ret = RsCallocMr(1, &mrCb);
    CHK_PRT_RETURN(ret, hccp_err("calloc mr failed"), ret);

    mrCb->qpCb = qpCb;

    ret = RsMrPreReg(phyId, qpCb, mrCb, mrRegInfo);
    if (ret) {
        hccp_err("pre reg mr failed, qpn %u, ret %d", qpn, ret);
        goto reg_err;
    }

found:
    mrRegInfo->lkey = mrCb->ibMr->lkey;
    mrRegInfo->rkey = mrCb->ibMr->rkey;

    hccp_info_rma("rs_mr_reg succ, state:%u", mrCb->state);
    return 0;

reg_err:
    free(mrCb);
    mrCb = NULL;

    return ret;
}

RS_ATTRI_VISI_DEF int RsMrDereg(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, char *addr)
{
    int ret;
    struct RsQpCb *qpCb = NULL;
    struct RsMrCb *mrCb = NULL;

    hccp_dbg("start rs_mr_dereg");
    RS_CHECK_POINTER_NULL_RETURN_INT(addr);
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_qpn2qpcb failed ret[%d]", ret), ret);

    CHK_PRT_RETURN(RsGetMrcb(qpCb, (uintptr_t)addr, &mrCb, &qpCb->mrList),
        hccp_err("rs_get_mrcb failed, qpn[%u] addr[0x%llx]", qpn, (unsigned long long)(uintptr_t)addr), -EFAULT);

    ret = RsDrvMrDereg(mrCb->ibMr);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_mr_dereg failed ret[%d] ", ret), -EACCES);

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RsListDel(&mrCb->list);
    free(mrCb);
    mrCb = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
    qpCb->mrNum--;

    hccp_dbg("qpn[%u] succ", qpn);

    return 0;
}

STATIC int RsRegisterUbSegment(int directFlag, uint64_t addr, uint64_t len)
{
    struct DVattribute attr = {0};
    int ret = 0;

    if (directFlag != DIRECT_FLAG_UB) {
        return 0;
    }

    ret = DlDrvMemGetAttribute(addr, &attr);
    CHK_PRT_RETURN(ret != 0, hccp_err("DlDrvMemGetAttribute failed, ret:%d", ret), ret);

    if (attr.memType != DV_MEM_LOCK_DEV) { // not support host memory
        return ret;
    }

    ret = DlHalMemRegUbSegment(attr.devId, addr, len);
    CHK_PRT_RETURN(ret != 0, hccp_err("DlHalMemRegUbSegment failed, ret:%d devId:%u len:%u", ret, attr.devId, len),
        ret);

    return ret;
}

STATIC int RsUnRegisterUbSegment(int directFlag, uint64_t addr)
{
    struct DVattribute attr = {0};
    int ret = 0;

    if (directFlag != DIRECT_FLAG_UB) {
        return 0;
    }

    ret = DlDrvMemGetAttribute(addr, &attr);
    CHK_PRT_RETURN(ret != 0, hccp_err("DrvMemGetAttribute failed, ret:%d", ret), ret);

    if (attr.memType != DV_MEM_LOCK_DEV) { // not support host memory
        return ret;
    }

    ret = DlHalMemUnRegUbSegment(attr.devId, addr);
    if (ret != 0) {
        hccp_err("DlHalMemUnRegUbSegment failed, ret:%d devId:%u", ret, attr.devId);
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsRegisterMr(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle)
{
    RS_CHECK_POINTER_NULL_RETURN_INT(mrHandle);

    int ret;
    unsigned int chipId;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_mr *rsMrHandle = NULL;

    CHK_PRT_RETURN(mrRegInfo == NULL || mrRegInfo->addr == NULL || mrRegInfo->len == 0 || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    hccp_info_rma("[rs_register_mr] len[0x%llx], access[%d]", mrRegInfo->len, mrRegInfo->access);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_register_mr rsGetLocalDevIDByHostDevID phyId[%u] invalid, ret %d", phyId, ret),
        ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    ret = RsRegisterUbSegment(rdevCb->directFlag, (uint64_t)(uintptr_t)mrRegInfo->addr, mrRegInfo->len);
    if (ret != 0) {
        hccp_err("RsRegisterUbSegment failed, ret[%d] vendor_id[0x%x] part_id[0x%x] directFlag[%u] "
                 "addr[0x%llx] len[0x%llx]",
            ret, rdevCb->deviceAttr.vendor_id, rdevCb->deviceAttr.vendor_part_id, rdevCb->directFlag,
            (uint64_t)(uintptr_t)mrRegInfo->addr, mrRegInfo->len);
        goto mem_reg_err;
    }

    *mrHandle = (void *)RsDrvMrReg(rdevCb->ibPd, mrRegInfo->addr, mrRegInfo->len, mrRegInfo->access);
    if (*mrHandle == NULL) {
        hccp_warn_rma("rs_drv_mr_reg addr is NULL len[0x%llx] access[%d] unsuccessful errno[%d]", mrRegInfo->len,
            mrRegInfo->access, errno);
        goto mr_reg_err;
    }

    rsMrHandle = (struct ibv_mr *)*mrHandle;
    mrRegInfo->lkey = rsMrHandle->lkey;
    mrRegInfo->rkey = rsMrHandle->rkey;

    hccp_info_rma("rs_register_mr succ");
    return ret;
mr_reg_err:
    (void)RsUnRegisterUbSegment(rdevCb->directFlag, (uint64_t)(uintptr_t)mrRegInfo->addr);
mem_reg_err:
    mrRegInfo->lkey = 0;

    return ret;
}

STATIC int RsInitTypicalMrCb(unsigned int phyId, struct RdmaMrRegInfo *mrRegInfo, struct RsRdevCb *devCb,
    struct RsMrCb *mrCb)
{
    unsigned long long len = mrRegInfo->len;
    struct roce_process_sign roceSign = {0};
    char *addr = (char *)mrRegInfo->addr;
    int access = mrRegInfo->access;
    int ret;

    if (devCb->rsCb->hccpMode == NETWORK_PEER_ONLINE || devCb->rsCb->hccpMode == NETWORK_ONLINE) {
        mrCb->ibMr = RsDrvMrReg(devCb->ibPd, addr, len, access);
        CHK_PRT_RETURN(mrCb->ibMr == NULL,
            hccp_err("rs_drv_mr_reg ibMr is NULL len[%lld] failed, errno:%d", len, errno), -EACCES);
    } else {
        ret = RsMrPrepareRoceSign(phyId, devCb, &roceSign);
        CHK_PRT_RETURN(ret != 0, hccp_err("RsMrPrepareRoceSign failed, ret(%d)", ret), ret);
        mrCb->ibMr = RsDrvExpMrReg(devCb->ibPd, addr, len, access, roceSign);
        CHK_PRT_RETURN(mrCb->ibMr == NULL,
            hccp_err("rs_drv_exp_mr_reg ibMr is NULL len[%lld] failed, errno:%d", len, errno), -EACCES);
    }

    mrCb->mrInfo.addr = (uintptr_t)addr;
    mrCb->mrInfo.len = len;
    mrCb->mrInfo.rkey = mrCb->ibMr->rkey;

    RS_PTHREAD_MUTEX_LOCK(&devCb->rdevMutex);
    RsListAddTail(&mrCb->list, &devCb->typicalMrList);
    RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);

    return 0;
}

RS_ATTRI_VISI_DEF int RsTypicalRegisterMrV1(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle)
{
    RS_CHECK_POINTER_NULL_RETURN_INT(mrHandle);

    struct RsMrCb *typicalMrCb = NULL;
    struct RsRdevCb *rdevCb = NULL;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(mrRegInfo == NULL || mrRegInfo->addr == NULL || mrRegInfo->len == 0 || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    hccp_info_rma("[rs_typical_register_mr] len[0x%llx], access[%d]", mrRegInfo->len, mrRegInfo->access);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_typical_register_mr rsGetLocalDevIDByHostDevID phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    ret = RsQueryMrCb(rdevCb, (uint64_t)(uintptr_t)mrRegInfo->addr, &typicalMrCb, &rdevCb->typicalMrList);
    if (ret == 0) {
        hccp_warn_rma("typical mr already registered");
        goto found;
    }

    typicalMrCb = calloc(1, sizeof(struct RsMrCb));
    CHK_PRT_RETURN(typicalMrCb == NULL, hccp_err("calloc typical_mr_cb failed"), -ENOMEM);
    typicalMrCb->devCb = rdevCb;

    ret = RsInitTypicalMrCb(phyId, mrRegInfo, rdevCb, typicalMrCb);
    if (ret != 0) {
        hccp_err("rs_init_typical_mr_cb failed, devIndex[%u], ret[%d]", rdevIndex, ret);
        goto reg_err;
    }

found:
    *mrHandle = typicalMrCb->ibMr;
    mrRegInfo->lkey = typicalMrCb->ibMr->lkey;
    mrRegInfo->rkey = typicalMrCb->ibMr->rkey;
    hccp_info_rma("rs_typical_register_mr succ, state:%d", typicalMrCb->state);
    return 0;

reg_err:
    free(typicalMrCb);
    typicalMrCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsTypicalRegisterMr(unsigned int phyId, unsigned int rdevIndex, struct RdmaMrRegInfo *mrRegInfo,
    void **mrHandle)
{
    RS_CHECK_POINTER_NULL_RETURN_INT(mrHandle);

    struct RsMrCb *typicalMrCb = NULL;
    struct RsRdevCb *rdevCb = NULL;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(mrRegInfo == NULL || mrRegInfo->addr == NULL || mrRegInfo->len == 0 || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    hccp_info_rma("start register len[0x%llx], access[%d]", mrRegInfo->len, mrRegInfo->access);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rsGetLocalDevIDByHostDevID phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    typicalMrCb = calloc(1, sizeof(struct RsMrCb));
    CHK_PRT_RETURN(typicalMrCb == NULL, hccp_err("calloc typical_mr_cb failed"), -ENOMEM);
    typicalMrCb->devCb = rdevCb;

    ret = RsInitTypicalMrCb(phyId, mrRegInfo, rdevCb, typicalMrCb);
    if (ret != 0) {
        hccp_err("rs_init_typical_mr_cb failed, devIndex[%u], ret[%d]", rdevIndex, ret);
        goto reg_err;
    }

    // resv len as 1 to save addr for later unreg to query
    typicalMrCb->mrInfo.addr = (uint64_t)(uintptr_t)typicalMrCb->ibMr;
    typicalMrCb->mrInfo.len = 1U;
    *mrHandle = typicalMrCb->ibMr;
    mrRegInfo->lkey = typicalMrCb->ibMr->lkey;
    mrRegInfo->rkey = typicalMrCb->ibMr->rkey;
    hccp_info_rma("register succ, state:%d", typicalMrCb->state);
    return 0;

reg_err:
    free(typicalMrCb);
    typicalMrCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsRemapMr(unsigned int phyId, unsigned int rdevIndex, struct MemRemapInfo memList[],
    unsigned int memNum)
{
    struct RsRdevCb *devCb = NULL;
    struct RsMrCb *mrCurr = NULL;
    struct RsMrCb *mrNext = NULL;
    unsigned long long addr = 0;
    bool isMemMatched = false;
    unsigned int chipId;
    unsigned int i;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= %d, is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rsGetLocalDevIDByHostDevID failed, phyId:%u invalid, ret:%d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &devCb);
    CHK_PRT_RETURN(devCb == NULL, hccp_err("rs_rdev2rdev_cb failed, chipId:%u, ret:%d", chipId, ret), -ENODEV);

    for (i = 0; i < memNum; i++) {
        isMemMatched = false;
        addr = (uint64_t)(uintptr_t)memList[i].addr;
        RS_PTHREAD_MUTEX_LOCK(&devCb->rdevMutex);
        RS_LIST_GET_HEAD_ENTRY(mrCurr, mrNext, &devCb->typicalMrList, list, struct RsMrCb);
        for (; (&mrCurr->list) != &devCb->typicalMrList;
             mrCurr = mrNext, mrNext = list_entry(mrNext->list.next, struct RsMrCb, list)) {
            // mem is out range of mr, continue to find next matching mr
            if ((addr < (uint64_t)(uintptr_t)mrCurr->ibMr->addr) || (memList[i].size > mrCurr->ibMr->length) ||
                (addr + memList[i].size < addr) ||
                (addr + memList[i].size > (uint64_t)(uintptr_t)mrCurr->ibMr->addr + mrCurr->ibMr->length)) {
                continue;
            }

            // each mr remap each corresponding mem
            ret = RsRoceRemapMr(mrCurr->ibMr, (struct hns_roce_mr_remap_info *)(void *)&memList[i], 1);
            if (ret != 0) {
                hccp_err("remap %u-th mem failed, ret:%d addr:0x%llx size:0x%llx", i, ret, addr, memList[i].size);
                RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);
                return ret;
            }
            isMemMatched = true;
        }
        RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);

        if (!isMemMatched) {
            hccp_err("find %u-th mem failed, addr:0x%llx size:0x%llx", i, addr, memList[i].size);
            return -ENODEV;
        }
        hccp_dbg("remap %u-th mem success, addr:0x%llx size:0x%llx", i, addr, memList[i].size);
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsTypicalDeregisterMr(unsigned int phyId, unsigned int devIndex, unsigned long long addr)
{
    struct RsMrCb *typicalMrCb = NULL;
    struct RsRdevCb *devCb = NULL;
    unsigned int chipId;
    int ret;

    hccp_info_rma("typical mr unreg start, addr[%llu]", addr);
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= %d, is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0 || devCb == NULL,
        hccp_err("rs_rdev2rdev_cb get dev_cb failed for chip_id[%u], ret[%d]", chipId, ret), -ENODEV);

    ret = RsQueryMrCb(devCb, addr, &typicalMrCb, &devCb->typicalMrList);
    CHK_PRT_RETURN(ret, hccp_err("rs_query_mr_cb failed ret[%d]", ret), ret);

    ret = RsDrvMrDereg(typicalMrCb->ibMr);
    CHK_PRT_RETURN(ret, hccp_err("rs_drv_mr_dereg failed ret[%d]", ret), -EACCES);

    RS_PTHREAD_MUTEX_LOCK(&devCb->rdevMutex);
    RsListDel(&typicalMrCb->list);
    free(typicalMrCb);
    typicalMrCb = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&devCb->rdevMutex);

    hccp_info_rma("devIndex[%u] succ", devIndex);

    return 0;
}

RS_ATTRI_VISI_DEF int RsDeregisterMr(unsigned int phyId, unsigned int rdevIndex, void *mrHandle)
{
    RS_CHECK_POINTER_NULL_RETURN_INT(mrHandle);

    struct ibv_mr *rsMrHandle = (struct ibv_mr *)mrHandle;
    uint64_t addr = (uint64_t)(uintptr_t)rsMrHandle->addr;
    struct RsRdevCb *devCb = NULL;
    unsigned int chipId;
    int ret = 0;

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret != 0, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &devCb);
    CHK_PRT_RETURN(ret != 0 || devCb == NULL,
        hccp_err("rs_rdev2rdev_cb get dev_cb failed for chip_id[%u], ret[%d]", chipId, ret), -ENODEV);

    ret = RsDrvMrDereg(rsMrHandle);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_drv_mr_dereg failed ret[%d]", ret), -EACCES);

    ret = RsUnRegisterUbSegment(devCb->directFlag, addr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsUnRegisterUbSegment failed ret[%d], vendor_id[0x%x] part_id[0x%x]"
                 " directFlag[%u]",
            ret, devCb->deviceAttr.vendor_id, devCb->deviceAttr.vendor_part_id, devCb->directFlag),
        ret);

    hccp_info_rma("rs_deregister_mr succ");
    return 0;
}

RS_ATTRI_VISI_DEF int RsSendWr(unsigned int phyId, unsigned int rdevIndex, uint32_t qpn, struct SendWr *wr,
    struct SendWrRsp *wrRsp)
{
    int ret;
    struct RsQpCb *qpCb = NULL;
    struct RsMrCb *mrCb = NULL;
    struct RsMrCb *remMrCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(wr);
    RS_CHECK_POINTER_NULL_RETURN_INT(wr->bufList);
    RS_CHECK_POINTER_NULL_RETURN_INT(wrRsp);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    CHK_PRT_RETURN(wr->bufNum > MAX_SGE_NUM || wr->bufNum == 0, hccp_err("invalid buf_num[%u]!", wr->bufNum), -EINVAL);

    CHK_PRT_RETURN(wr->bufList->len > RS_SGLIST_LEN_MAX || wr->bufList->len == 0,
        hccp_err("sg list "
                 "len is more than 2G, len[%u]",
            wr->bufList->len),
        -EINVAL);

    if (RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb)) {
        return -EACCES;
    }

    qpCb->sendWrNum++;

    hccp_info_rma("qpn %d, bufList[0].addr is 0x%llx", qpn, wr->bufList[0].addr);
    if (RsGetMrcb(qpCb, wr->bufList[0].addr, &mrCb, &qpCb->mrList)) {
        hccp_err("qpn %d, bufList[0].addr[0x%llx] len[0x%x] is invalid.", qpn, wr->bufList[0].addr, wr->bufList[0].len);
        return -EFAULT;
    }

    // send op no need to check & get remote mr
    if (wr->op != RA_WR_SEND && wr->op != RA_WR_SEND_WITH_IMM) {
        hccp_info_rma("remote wr dst addr is 0x%llx", wr->dstAddr);
        if (RsGetMrcb(qpCb, wr->dstAddr, &remMrCb, &qpCb->remMrList)) {
            hccp_err("qpn %d, remote wr dst addr[0x%llx] len[0x%x] is invalid.", qpn, wr->dstAddr, wr->bufList[0].len);
            return -ENOENT;
        }
    }

    ret = RsDrvSendExp(qpCb, mrCb, remMrCb, wr, wrRsp);
    if (ret) {
        hccp_err("send exp failed qpn %u, ret %d", qpn, ret);
    }
    gRsSendWrNum++;
    return ret;
}

STATIC void BuildUpWrWithKey(struct WrInfo *wr, struct ibv_sge *list, struct ibv_send_wr *ibWr)
{
    list->addr = (uintptr_t)wr->memList.addr;
    list->length = wr->memList.len;
    list->lkey = wr->memList.lkey;

    ibWr->sg_list = list;
    ibWr->opcode = wr->op;
    ibWr->send_flags = (unsigned int)wr->sendFlags;
    ibWr->imm_data = htobe32(wr->immData);

    ibWr->num_sge = 1; /* only support one sge */
    ibWr->wr_id = wr->wrId;
    if (wr->op != IBV_WR_SEND && wr->op != IBV_WR_SEND_WITH_IMM) {
        ibWr->wr.rdma.rkey = wr->rkey;
        ibWr->wr.rdma.remote_addr = wr->dstAddr;
    }
}

STATIC void RsSendBuildUpWr(struct RsMrCb *mrCb, struct WrInfo *wr, struct ibv_sge *list, struct ibv_send_wr *ibWr)
{
    list->addr = (uintptr_t)wr->memList.addr;
    list->lkey = mrCb->ibMr->lkey;
    list->length = wr->memList.len;

    ibWr->sg_list = list;
    ibWr->opcode = wr->op;
    ibWr->imm_data = htobe32(wr->immData);
    ibWr->send_flags = (unsigned int)wr->sendFlags;

    ibWr->num_sge = 1; /* only support one sge */
    ibWr->wr_id = wr->wrId;
}

STATIC void RsWirteAndReadBuildUpWr(struct RsMrCb *mrCb, struct RsMrCb *remMrCb, struct WrInfo *wr,
    struct ibv_sge *list, struct ibv_send_wr *ibWr)
{
    list->addr = (uintptr_t)wr->memList.addr;
    list->length = wr->memList.len;
    list->lkey = mrCb->ibMr->lkey;

    ibWr->sg_list = list;
    ibWr->opcode = wr->op;
    ibWr->send_flags = (unsigned int)wr->sendFlags;
    ibWr->imm_data = htobe32(wr->immData);

    ibWr->num_sge = 1; /* only support one sge */
    ibWr->wr_id = wr->wrId;
    ibWr->wr.rdma.rkey = remMrCb->mrInfo.rkey;
    ibWr->wr.rdma.remote_addr = wr->dstAddr;
}

STATIC int RsBuildUpWrList(struct WrInfo *wrList, struct RsQpCb *qpCb, struct ibv_sge *list, struct ibv_send_wr *ibWr,
    unsigned int i)
{
    struct RsMrCb *mrCb = NULL;
    struct RsMrCb *remMrCb = NULL;
    CHK_PRT_RETURN(wrList[i].memList.len > RS_SGLIST_LEN_MAX,
        hccp_err("sg list len is more than 2G, len[%u]", wrList[i].memList.len), -EINVAL);

    hccp_dbg("qpn %d, bufList[0].addr is 0x%llx", qpCb->ibQp->qp_num, wrList[i].memList.addr);
    if (RsGetMrcb(qpCb, wrList[i].memList.addr, &mrCb, &qpCb->mrList)) {
        hccp_err("qpn %d, bufList[0].addr[0x%llx] len[0x%x] is invalid.", qpCb->ibQp->qp_num, wrList[i].memList.addr,
            wrList[i].memList.len);
        return -EFAULT;
    }

    // send op no need to check & get remote mr
    if (wrList[i].op != IBV_WR_SEND && wrList[i].op != IBV_WR_SEND_WITH_IMM) {
        hccp_dbg("remote wr dst addr is 0x%llx", wrList[i].dstAddr);
        if (RsGetMrcb(qpCb, wrList[i].dstAddr, &remMrCb, &qpCb->remMrList)) {
            hccp_err("qpn %d, remote wr dst addr[0x%llx] len[0x%x] is invalid.", qpCb->ibQp->qp_num, wrList[i].dstAddr,
                wrList[i].memList.len);
            return -ENOENT;
        }
        RsWirteAndReadBuildUpWr(mrCb, remMrCb, &wrList[i], &list[i], &ibWr[i]);
    } else {
        RsSendBuildUpWr(mrCb, &wrList[i], &list[i], &ibWr[i]);
    }

    return 0;
}

STATIC int RsBuildUpWrListWithKey(struct WrInfo *wrList, struct ibv_sge *list, struct ibv_send_wr *ibWr, unsigned int i)
{
    CHK_PRT_RETURN(wrList[i].memList.len > RS_SGLIST_LEN_MAX,
        hccp_err("sg list len is more than 2G, len[%u]", wrList[i].memList.len), -EINVAL);

    BuildUpWrWithKey(&wrList[i], &list[i], &ibWr[i]);
    return 0;
}

STATIC int RsSendNormalWrlist(struct RsQpCb *qpCb, struct WrInfo *wrList, unsigned int sendNum,
    unsigned int *completeNum, unsigned int keyFlag)
{
    int ret;
    unsigned int i, j;

    struct ibv_send_wr *badWr = NULL;
    CHK_PRT_RETURN(sendNum > MAX_WR_NUM || sendNum == 0, hccp_err("send num[%u] is invalid!", sendNum), -EINVAL);
    struct ibv_send_wr *ibWr = (struct ibv_send_wr *)calloc(sendNum, sizeof(struct ibv_send_wr));
    CHK_PRT_RETURN(ibWr == NULL, hccp_err("calloc ib_wr failed!"), -ENOSPC);

    struct ibv_sge *list = (struct ibv_sge *)calloc(sendNum, sizeof(struct ibv_sge));
    if (list == NULL) {
        hccp_err("calloc list failed!");
        ret = -ENOSPC;
        goto alloc_fail;
    }

    for (i = 0; i < sendNum; i++) {
        ret = (keyFlag == 0) ? RsBuildUpWrList(wrList, qpCb, list, ibWr, i)
                             : RsBuildUpWrListWithKey(wrList, list, ibWr, i);
        if (ret) {
            goto input_err;
        }
        j = i + 1;
        ibWr[i].next = (i < sendNum - 1) ? &ibWr[j] : NULL;
    }

    ret = RsIbvPostSend(qpCb->ibQp, &ibWr[0], &badWr);
    if (ret == 0) {
        *completeNum = sendNum;
    } else if (ret == -ENOMEM) {
        *completeNum = (unsigned int)((void *)badWr - (void *)ibWr) / sizeof(struct ibv_send_wr);
        hccp_dbg("post send wqe overflow, completeNum[%d]", *completeNum);
    } else {
        hccp_err("ibv_post_send failed, ret[%d]", ret);
        *completeNum = 0;
    }
    qpCb->sendWrNum = qpCb->sendWrNum + (*completeNum);

input_err:
    free(list);
    list = NULL;
alloc_fail:
    free(ibWr);
    ibWr = NULL;
    return (ret == -ENOMEM) ? 0 : ret;
}

STATIC int RsSendExpWrlist(struct RsQpCb *qpCb, struct WrInfo *wrList, unsigned int sendNum, struct SendWrRsp *wrRsp,
    unsigned int *completeNum, unsigned int keyFlag)
{
    struct ibv_post_send_ext_attr extAttr = {0};
    struct ibv_post_send_ext_resp extRsp = {0};
    struct ibv_send_wr *badWr = NULL;
    struct wr_exp_rsp expRsp = {0};
    struct ibv_send_wr ibWr = {0};
    struct ibv_sge list = {0};
    unsigned int i;
    int ret = 0;

    for (i = 0; i < sendNum; i++) {
        // reuse code: only need to build up one wr once a time
        ret = (keyFlag == 0) ? RsBuildUpWrList(&wrList[i], qpCb, &list, &ibWr, 0)
                             : RsBuildUpWrListWithKey(&wrList[i], &list, &ibWr, 0);
        if (ret != 0) {
            hccp_err("qpn:%u key_flag:%u build_up_wr i:%u failed, ret:%d", qpCb->ibQp->qp_num, keyFlag, i, ret);
            break;
        }

        if (wrList[i].op == RA_WR_RDMA_WRITE_WITH_NOTIFY || wrList[i].op == RA_WR_RDMA_REDUCE_WRITE ||
            wrList[i].op == RA_WR_RDMA_REDUCE_WRITE_WITH_NOTIFY) {
            ibWr.imm_data = htobe32((wrList[i].aux.notifyOffset & WRITE_NOTIFY_OFFSET_MASK) |
                                    WRITE_NOTIFY_VALUE_RECORD);
            extAttr.reduce_op = wrList[i].aux.reduceType;
            extAttr.reduce_type = wrList[i].aux.dataType;
            ret = RsIbvExtPostSend(qpCb->ibQp, &ibWr, &badWr, &extAttr, &extRsp);
            expRsp.wqe_index = extRsp.wqe_index;
            expRsp.db_info = extRsp.db_info;
            hccp_dbg("rs_ibv_ext_post_send, op = [%x], immData = [0x%lx], reduce_op = [%d],reduceType = [%d]",
                ibWr.opcode, ibWr.imm_data, extAttr.reduce_op, extAttr.reduce_type);
        } else {
            ret = RsIbvExpPostSend(qpCb->ibQp, &ibWr, &badWr, &expRsp);
            hccp_dbg("rs_ibv_exp_post_send, op = [%x], remoteAddr = [0x%llx], size = [%d]", ibWr.opcode,
                ibWr.wr.rdma.remote_addr, ibWr.sg_list->length);
        }

        if (ret != 0) {
            if (ret == -ENOMEM) {
                hccp_warn_rma("qpn:%u rs_ibv_exp_post_send i:%u unsuccessful, ret %d", qpCb->ibQp->qp_num, i, ret);
            } else {
                hccp_err("qpn:%u rs_ibv_exp_post_send i:%u failed, ret %d", qpCb->ibQp->qp_num, i, ret);
            }
            break;
        }

        qpCb->sendWrNum++;

        if (qpCb->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
            wrRsp[i].wqeTmp.sqIndex = (unsigned int)qpCb->sqIndex;
            wrRsp[i].wqeTmp.wqeIndex = expRsp.wqe_index;
        } else if (qpCb->qpMode == RA_RS_OP_QP_MODE || qpCb->qpMode == RA_RS_GDR_ASYN_QP_MODE) {
            wrRsp[i].db.dbIndex = (unsigned int)qpCb->dbIndex;
            wrRsp[i].db.dbInfo = expRsp.db_info;
        }
    }

    hccp_dbg("complete_num[%d], ret[%d]", i, ret);
    *completeNum = i;
    return (ret == -ENOMEM) ? 0 : ret;
}

RS_ATTRI_VISI_DEF int RsSendWrlist(struct RsWrlistBaseInfo baseInfo, struct WrInfo *wrList, unsigned int sendNum,
    struct SendWrRsp *wrRsp, unsigned int *completeNum)
{
    int ret;
    unsigned int phyId, rdevIndex, qpn;
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(wrList);
    RS_CHECK_POINTER_NULL_RETURN_INT(wrRsp);
    CHK_PRT_RETURN(sendNum > MAX_WR_NUM || sendNum == 0 || baseInfo.phyId >= RS_MAX_DEV_NUM,
        hccp_err("send_num[%u] or phyId:%u >= [%d], is invalid", sendNum, baseInfo.phyId, RS_MAX_DEV_NUM), -EINVAL);

    phyId = baseInfo.phyId;
    rdevIndex = baseInfo.rdevIndex;
    qpn = baseInfo.qpn;

    CHK_PRT_RETURN(RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb), hccp_err("rs_qpn2qpcb failed, physical id[%u]", phyId),
        -EACCES);

    // only allow normal qp to call this func when ai_op_support not set
    if (qpCb->qpMode == RA_RS_NOR_QP_MODE && qpCb->aiOpSupport == 0) {
        ret = RsSendNormalWrlist(qpCb, wrList, sendNum, completeNum, baseInfo.keyFlag);
    } else {
        ret = RsSendExpWrlist(qpCb, wrList, sendNum, wrRsp, completeNum, baseInfo.keyFlag);
    }
    return ret;
}

RS_ATTRI_VISI_DEF int RsRecvWrlist(struct RsWrlistBaseInfo baseInfo, struct RecvWrlistData *wr, unsigned int recvNum,
    unsigned int *completeNum)
{
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(wr);
    CHK_PRT_RETURN(recvNum > MAX_WR_NUM || recvNum == 0 || baseInfo.phyId >= RS_MAX_DEV_NUM,
        hccp_err("recv_num[%u] or phyId:%u >= [%d], is invalid", recvNum, baseInfo.phyId, RS_MAX_DEV_NUM), -EINVAL);

    CHK_PRT_RETURN(RsQpn2qpcb(baseInfo.phyId, baseInfo.rdevIndex, baseInfo.qpn, &qpCb),
        hccp_err("rs_qpn2qpcb failed, physical id[%u]", baseInfo.phyId), -EACCES);

    return RsDrvPostRecv(qpCb, wr, recvNum, completeNum);
}

RS_ATTRI_VISI_DEF int RsSetHostPid(uint32_t phyId, pid_t hostPid, const char *pidSign)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(pidSign);
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs_set_host_pid rs set param error ! phyId:%u", phyId), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_set_host_pid rsGetLocalDevIDByHostDevID phyId invalid, ret %d", ret), ret);

    hccp_info_rma("phyId[%u] host_pid[%d]", chipId, hostPid);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs cb failed, chipId:%u", chipId), ret);

    rsCb->pRsSign.tgid = hostPid;
    ret = strcpy_s(rsCb->pRsSign.sign, PROCESS_RS_SIGN_LENGTH, pidSign);
    CHK_PRT_RETURN(ret, hccp_err("copy sign failed, ret %d", ret), -ESAFEFUNC);

    return 0;
}

RS_ATTRI_VISI_DEF int RsRdevGetPortStatus(unsigned int phyId, unsigned int rdevIndex, enum PortStatus *status)
{
    struct ibv_port_attr portAttr = {0};
    struct RsRdevCb *rdevCb = NULL;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);
    CHK_PRT_RETURN(status == NULL, hccp_err("param err! status is NULL"), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rsGetLocalDevIDByHostDevID failed, phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    ret = RsIbvQueryPort(rdevCb->ibCtx, rdevCb->ibPort, &portAttr);
    CHK_PRT_RETURN(ret, hccp_err("ibv_query_port failed ret[%d]", ret), -EOPENSRC);

    *status = portAttr.state == IBV_PORT_ACTIVE ? PORT_STATUS_ACTIVE : PORT_STATUS_DOWN;

    hccp_dbg("phyId:%u port_attr.state:%u status:%u", phyId, portAttr.state, *status);
    return 0;
}

RS_ATTRI_VISI_DEF int RsGetNotifyMrInfo(unsigned int phyId, unsigned int rdevIndex, struct MrInfoT *info)
{
    struct RsRdevCb *rdevCb = NULL;
    struct rs_cb *rsCb = NULL;
    unsigned int chipId;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    CHK_PRT_RETURN(info == NULL, hccp_err("param err! info is NULL"), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret:%d", phyId, ret), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs_cb failed, ret:%d", ret), -ENODEV);

    ret = RsGetRdevCb(rsCb, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_rdev_cb failed!, ret:%d, rdevIndex:%u", ret, rdevIndex), ret);

    info->addr = (void *)(uintptr_t)rdevCb->notifyVaBase;
    info->size = rdevCb->notifySize;
    info->access = rdevCb->notifyAccess;
    info->lkey = rdevCb->notifyMr->lkey;

    return 0;
}

RS_ATTRI_VISI_DEF int RsNotifyCfgSet(unsigned int phyId, unsigned long long va, unsigned long long size)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT((void *)(uintptr_t)va);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM ||
                       (size != MAX_NOTIFY_SIZE_CLOUD && size != NOTIFY_NUM_MAX_V2 && size != NOTIFY_NUM_MAX_V3),
        hccp_err("rs_notify_cfg_set rs set param error ! phyId[%u] size[%llu]", phyId, size), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_notify_cfg_set phyId invalid, ret %d, phyId:%u", ret, phyId), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs cb failed, chipId:%u", chipId), ret);

    rsCb->notifyVaBase = va;
    rsCb->notifySize = size;

    return 0;
}

RS_ATTRI_VISI_DEF int RsNotifyCfgGet(unsigned int phyId, unsigned long long *va, unsigned long long *size)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(va);
    RS_CHECK_POINTER_NULL_RETURN_INT(size);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs_notify_cfg_get rs set param error ! phyId:%u", phyId),
        -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_notify_cfg_get phyId invalid, ret %d, phyId:%u", ret, phyId), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("get rs cb failed, chipId:%u", chipId), ret);

    *va = rsCb->notifyVaBase;
    *size = rsCb->notifySize;

    return 0;
}

RS_ATTRI_VISI_DEF int RsSetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int tempDepth,
    unsigned int *qpNum)
{
    (void)phyId;
    (void)rdevIndex;
    (void)tempDepth;
    (void)qpNum;
#ifdef CUSTOM_INTERFACE
    struct RsRdevCb *rdevCb = NULL;
    unsigned int sqDepth = 0;
    unsigned int chipId = 0;
    int ret;

    if (!RsIsCustomInterfaceSupported()) {
        return 0;
    }
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs_set_tsqp_depth param error ! phyId:%d", phyId), -EINVAL);

    CHK_PRT_RETURN(qpNum == NULL, hccp_err("rs_set_tsqp_depth qp_num is NULL, param error!"), -EINVAL);

    CHK_PRT_RETURN(tempDepth < RS_MIN_TEMPTH_DEPTH || tempDepth > RS_MAX_TEMPTH_DEPTH,
        hccp_err("param error!"
                 "temp_depth[%u] can not be smaller than [%d] or bigger than [%d]",
            tempDepth, RS_MIN_TEMPTH_DEPTH, RS_MAX_TEMPTH_DEPTH),
        -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret || rdevCb == NULL,
        hccp_err("rs_set_tsqp_depth rs_rdev2rdev_cb for chip_id[%u] "
                 "failed, ret %d",
            chipId, ret),
        ret);

    ret = RsRoceSetTsqpDepth(rdevCb->devName, rdevIndex, tempDepth, qpNum, &sqDepth);
    CHK_PRT_RETURN(ret, hccp_err("rs_roce_set_tsqp_depth failed, ret %d, devName[%s]", ret, rdevCb->devName), ret);

    rdevCb->txDepth = sqDepth;
    rdevCb->rxDepth = sqDepth;
    rdevCb->qpMaxNum = *qpNum;
#endif
    return 0;
}

RS_ATTRI_VISI_DEF int RsGetTsqpDepth(unsigned int phyId, unsigned int rdevIndex, unsigned int *tempDepth,
    unsigned int *qpNum)
{
    (void)phyId;
    (void)rdevIndex;
    (void)tempDepth;
    (void)qpNum;
#ifdef CUSTOM_INTERFACE
    struct RsRdevCb *rdevCb = NULL;
    unsigned int sqDepth = 0;
    unsigned int chipId = 0;
    int ret;

    if (!RsIsCustomInterfaceSupported()) {
        return 0;
    }
    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("param error ! phyId:%d", phyId), -EINVAL);

    CHK_PRT_RETURN(tempDepth == NULL || qpNum == NULL,
        hccp_err("temp_depth or qp_num is NULL,"
                 "param error!"),
        -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL,
        hccp_err("rs_get_tsqp_depth rs_rdev2rdev_cb for chip_id[%u] "
                 "failed, ret %d",
            chipId, ret),
        ret);

    ret = RsRoceGetTsqpDepth(rdevCb->devName, rdevIndex, tempDepth, qpNum, &sqDepth);
    CHK_PRT_RETURN(ret, hccp_err("rs_roce_get_tsqp_depth failed, ret %d, devName[%s]", ret, rdevCb->devName), ret);
#endif
    return 0;
}

STATIC void RsSetQpDepthAttr(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb, struct RsQpNorm *qpNorm)
{
    if (qpCb->qpMode == RA_RS_GDR_TMPL_QP_MODE) {
        qpCb->txDepth = rdevCb->txDepth;
        qpCb->rxDepth = rdevCb->rxDepth;
    } else {
        if (rdevCb->rsCb->hccpMode == NETWORK_OFFLINE) {
            qpCb->txDepth = RS_QP_TX_DEPTH_OFFLINE;
            qpCb->rxDepth = RS_QP_RX_DEPTH_OFFLINE;
        } else {
            qpCb->txDepth = RS_QP_TX_DEPTH_ONLINE;
            qpCb->rxDepth = RS_QP_RX_DEPTH_ONLINE;
        }
    }

    if (qpNorm->isExp != 0 && qpNorm->qpMode != RA_RS_NOR_QP_MODE) {
        if (rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
            qpCb->txDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH_PEER_ONLINE : qpCb->txDepth;
            qpCb->rxDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH_PEER_ONLINE : qpCb->rxDepth;
        } else {
            qpCb->txDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE && qpCb->qpMode != RA_RS_GDR_ASYN_QP_MODE)
                                ? RS_QP_32K_DEPTH
                                : qpCb->txDepth;
        }
        qpCb->sendSgeNum = 1;
        qpCb->recvSgeNum = 1;
    } else {
        if (rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
            qpCb->txDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH_PEER_ONLINE : qpCb->txDepth;
            qpCb->rxDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH_PEER_ONLINE : qpCb->rxDepth;
        } else {
            qpCb->txDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH : qpCb->txDepth;
            qpCb->rxDepth = (qpCb->qpMode != RA_RS_GDR_TMPL_QP_MODE) ? RS_QP_TX_DEPTH : qpCb->rxDepth;
        }
        qpCb->sendSgeNum = RS_QP_ATTR_MAX_SEND_SGE;
        qpCb->recvSgeNum = 1;
    }
}

STATIC int RsQpcbInit(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb, struct RsQpNorm *qpNorm)
{
#define RS_DRV_CQ_DEPTH 16384
#define RS_DRV_CQ_128_DEPTH 128
#define RS_DRV_CQ_8K_DEPTH 8192
#define RS_DRV_CQ_32K_DEPTH 32768
    int qpMode = qpNorm->qpMode;
    int ret;

    qpCb->rdevCb = rdevCb;
    RS_INIT_LIST_HEAD(&qpCb->mrList);
    RS_INIT_LIST_HEAD(&qpCb->remMrList);

    qpCb->qpMode = qpMode;
    qpCb->eqNum = 0;
    qpCb->numRecvCqEvents = 0;
    qpCb->numSendCqEvents = 0;
    qpCb->state = RS_QP_STATUS_DISCONNECT;
    qpCb->ibPd = rdevCb->ibPd;

    // cq attr
    if (qpNorm->isExt == 1) {
        // update TEMP & ASYN mode cq depth from 32K to 8K due to memory issue
        qpCb->sendCqDepth = (qpMode != RA_RS_GDR_TMPL_QP_MODE && qpMode != RA_RS_GDR_ASYN_QP_MODE) ? RS_DRV_CQ_32K_DEPTH
                                                                                                   : RS_DRV_CQ_8K_DEPTH;
        qpCb->recvCqDepth = RS_DRV_CQ_128_DEPTH;
    } else {
        qpCb->sendCqDepth = RS_DRV_CQ_DEPTH;
        qpCb->recvCqDepth = RS_DRV_CQ_DEPTH;
    }

    // qp attr
    RsSetQpDepthAttr(rdevCb, qpCb, qpNorm);

    qpCb->memAlign = qpNorm->memAlign;

    qpCb->channel = RsIbvCreateCompChannel(rdevCb->ibCtx);
    CHK_PRT_RETURN(qpCb->channel == NULL, hccp_err("ibv_create_comp_channel failed! errno(%d)", errno), -EINVAL);
    qpCb->qosAttr.tc = (RS_ROCE_DSCP_33 & RS_DSCP_MASK) << RS_DSCP_OFF;
    qpCb->qosAttr.sl = RS_ROCE_4_SL;
    qpCb->timeout = RS_QP_ATTR_TIMEOUT;
    qpCb->retryCnt = RS_QP_ATTR_RETRY_CNT;

    ret = RsEpollCtl(rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_ADD, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
    if (ret) {
        RsIbvDestroyCompChannel(qpCb->channel);
        hccp_err("add channel fd failed ret %d", ret);
        return ret;
    }
#endif
    return 0;
}

STATIC int RsQpcbDeinit(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb)
{
    int ret;

    if (qpCb == NULL || qpCb->channel == NULL) {
        hccp_err("qp_cb or qp_cb->channel is NULL!");
        return -EINVAL;
    }

    ret = RsEpollCtl(rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_DEL, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
    if (ret) {
        hccp_err("del channel fd failed ret %d", ret);
    }
#endif

    if (qpCb->channel != NULL) {
        RsIbvDestroyCompChannel(qpCb->channel);
        qpCb->channel = NULL;
    }
#ifndef CA_CONFIG_LLT
    return ret;
#else
    return 0;
#endif
}

STATIC int RsQpNotifyMr(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb, uint32_t *qpn)
{
    int ret;
    struct RsMrCb *notifyMrNode = NULL;

    ret = RsCallocMr(1, &notifyMrNode);
    CHK_PRT_RETURN(ret, hccp_err("notify_mr_cb malloc failed"), ret);

    RS_PTHREAD_MUTEX_LOCK(&rdevCb->rdevMutex);
    RsListAddTail(&qpCb->list, &rdevCb->qpList);
    RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rdevMutex);

    if (rdevCb->notifyType != NO_USE) {
        notifyMrNode->qpCb = qpCb;
        notifyMrNode->ibMr = rdevCb->notifyMr;
        notifyMrNode->mrInfo.addr = rdevCb->notifyVaBase;
        notifyMrNode->mrInfo.len = rdevCb->notifySize;
        notifyMrNode->mrInfo.rkey = notifyMrNode->ibMr->rkey;
    } else {
        notifyMrNode->qpCb = qpCb;
        notifyMrNode->ibMr = NULL;
    }

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RsListAddTail(&notifyMrNode->list, &qpCb->mrList);
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
    rdevCb->qpCnt++;
    *qpn = qpCb->ibQp->qp_num;

    hccp_info_rma("rs qp %d create OK!", *qpn);

    return 0;
}

STATIC int RsQpQueryInfo(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb **rdevCb, int qpMode)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs_qp_query_info rs set param error! phyId:%u", phyId), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_qp_query_info phyId[%u] invalid, ret:%d", phyId, ret), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("rs_qp_query_info get rs_cb failed, ret:%d", ret), -ENODEV);

    ret = RsGetRdevCb(rsCb, rdevIndex, rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_rdev_cb failed! ret:%d, rdevIndex:%u", ret, rdevIndex), ret);

    if (qpMode == RA_RS_GDR_TMPL_QP_MODE) {
        CHK_PRT_RETURN((*rdevCb)->qpCnt >= (*rdevCb)->qpMaxNum,
            hccp_err("Exceeded the maximum QP limit(%u)", (*rdevCb)->qpMaxNum), -EINVAL);
    } else {
        CHK_PRT_RETURN((*rdevCb)->qpCnt >= RS_QP_NUM_MAX,
            hccp_err("Exceeded the maximum QP limit(%u)", (*rdevCb)->qpCnt), -EINVAL);
    }

    return 0;
}

STATIC int RsInitMemPool(struct RsQpCb *qpCb)
{
    struct roce_mem_cq_qp_attr memAttr = {0};
    int ret;

    if ((qpCb->qpMode != RA_RS_OP_QP_MODE && qpCb->qpMode != RA_RS_OP_QP_MODE_EXT) ||
        qpCb->memAlign != LITE_ALIGN_2MB) {
        return 0;
    }

    // init mem_pool and store mem_data in mem_resp
    memAttr.mem_align = qpCb->memAlign;
    memAttr.send_qp_depth = qpCb->txDepth;
    memAttr.send_cq_depth = (unsigned int)qpCb->sendCqDepth;
    memAttr.send_sge_num = qpCb->sendSgeNum;
    memAttr.recv_qp_depth = qpCb->rxDepth;
    memAttr.recv_cq_depth = (unsigned int)qpCb->recvCqDepth;
    memAttr.recv_sge_num = qpCb->recvSgeNum;
    memAttr.use_resv_mem = qpCb->useResvMem;
    memAttr.resv_mem_pool_id = qpCb->resvMemPoolId;
    memAttr.ctx = qpCb->rdevCb->ibCtx;

    ret = RsRoceInitMemPool(&memAttr, &qpCb->memResp.memData, qpCb->rdevCb->rsCb->chipId);
    if (ret != 0) {
        hccp_err("rs_roce_init_mem_pool failed, ret=%d, chipId=%u", ret, qpCb->rdevCb->rsCb->chipId);
    }
    return ret;
}

STATIC void RsDeinitMemPool(struct RsQpCb *qpCb)
{
    if ((qpCb->qpMode != RA_RS_OP_QP_MODE && qpCb->qpMode != RA_RS_OP_QP_MODE_EXT) ||
        qpCb->memAlign != LITE_ALIGN_2MB) {
        return;
    }

    (void)RsRoceDeinitMemPool(qpCb->memResp.memData.mem_idx);
}

STATIC int RsAllocQpcb(struct RsRdevCb *rdevCb, struct RsQpCb **qpCb, struct RsQpNorm *qpNorm)
{
    int ret;

    ret = RsCallocQpcb(1, qpCb);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d errno:%d", ret, errno), -ENOMEM);

    ret = pthread_mutex_init(&(*qpCb)->qpMutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto qp_mutex_init_err;
    }

    ret = pthread_mutex_init(&(*qpCb)->cqeErrInfo.mutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto cqe_mutex_init_err;
    }

    ret = RsQpcbInit(rdevCb, *qpCb, qpNorm);
    if (ret) {
        hccp_err("create qp tx rx failed ret %d", ret);
        goto rs_qpcb_init_err;
    }

    ret = RsInitMemPool(*qpCb);
    if (ret) {
        hccp_err("init mem pool failed ret %d", ret);
        goto rs_init_mem_err;
    }

    ret = RsDrvCreateCq(*qpCb, qpNorm->isExt);
    if (ret) {
        hccp_err("create cq failed ret %d", ret);
        goto create_cq_err;
    }

    return 0;

create_cq_err:
    RsDeinitMemPool(*qpCb);

rs_init_mem_err:
    RsQpcbDeinit(rdevCb, *qpCb);

rs_qpcb_init_err:
    pthread_mutex_destroy(&(*qpCb)->cqeErrInfo.mutex);

cqe_mutex_init_err:
    pthread_mutex_destroy(&(*qpCb)->qpMutex);

qp_mutex_init_err:
    free(*qpCb);
    *qpCb = NULL;

    return ret;
}

STATIC void RsFreeQpcb(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb)
{
    RsDrvDestroyCq(qpCb);
    RsDeinitMemPool(qpCb);
    (void)RsQpcbDeinit(rdevCb, qpCb);
    pthread_mutex_destroy(&qpCb->cqeErrInfo.mutex);
    pthread_mutex_destroy(&qpCb->qpMutex);
    free(qpCb);
    qpCb = NULL;
}

RS_ATTRI_VISI_DEF int RsQpCreate(unsigned int phyId, unsigned int rdevIndex, struct RsQpNorm qpNorm,
    struct RsQpResp *qpResp)
{
    struct RsRdevCb *rdevCb = NULL;
    struct RsQpCb *qpCb = NULL;
    int ret;

    RS_QP_PARA_CHECK(phyId);
    CHK_PRT_RETURN(qpResp == NULL, hccp_err("qp_resp is NULL!"), -EINVAL);

    ret = RsQpQueryInfo(phyId, rdevIndex, &rdevCb, qpNorm.qpMode);
    CHK_PRT_RETURN(ret, hccp_err("query qp info failed:%d", ret), ret);

    ret = RsAllocQpcb(rdevCb, &qpCb, &qpNorm);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d", ret), ret);

    ret = RsDrvQpCreate(qpCb, &qpNorm);
    if (ret) {
        hccp_err("create drv qp create failed:%d", ret);
        goto create_qp_err;
    }

    ret = ibv_req_notify_cq(qpCb->ibSendCq, 0);
    if (ret) {
        hccp_err("Couldn't request send CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = ibv_req_notify_cq(qpCb->ibRecvCq, 0);
    if (ret) {
        hccp_err("Couldn't request recv CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = RsQpNotifyMr(rdevCb, qpCb, &qpResp->qpn); // alloc mr
    if (ret) {
        hccp_err("store qp notify mr failed:%d", ret);
        goto ret_noritfy_cq;
    }

    if (qpNorm.isExp) {
        qpCb->isExp = RS_IS_EXP;
    } else {
        qpCb->isExp = RS_NOT_EXP;
    }

    qpResp->qpn = (unsigned int)qpCb->qpInfoLo.qpn;
    qpResp->gidIdx = (unsigned int)qpCb->qpInfoLo.gidIdx;
    qpResp->psn = (unsigned int)qpCb->qpInfoLo.psn;
    qpResp->gid = qpCb->qpInfoLo.gid;

    return 0;

ret_noritfy_cq:
    RsDrvQpDestroy(qpCb);

create_qp_err:
    RsFreeQpcb(rdevCb, qpCb);
    return ret;
}

STATIC int RsQpcbInitWithAttrs(struct RsRdevCb *rdevCb, struct RsQpCb *qpCb, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;

    qpCb->rdevCb = rdevCb;
    RS_INIT_LIST_HEAD(&qpCb->mrList);
    RS_INIT_LIST_HEAD(&qpCb->remMrList);

    qpCb->qpMode = qpNorm->extAttrs.qpMode;
    qpCb->numRecvCqEvents = 0;
    qpCb->numSendCqEvents = 0;
    qpCb->state = RS_QP_STATUS_DISCONNECT;
    qpCb->ibPd = rdevCb->ibPd;

    qpCb->txDepth = qpNorm->extAttrs.qpAttr.cap.max_send_wr;
    qpCb->rxDepth = qpNorm->extAttrs.qpAttr.cap.max_send_wr;
    qpCb->sendSgeNum = qpNorm->extAttrs.qpAttr.cap.max_send_sge;
    qpCb->recvSgeNum = qpNorm->extAttrs.qpAttr.cap.max_recv_sge;
    qpCb->sendCqDepth = qpNorm->extAttrs.cqAttr.sendCqDepth;
    qpCb->recvCqDepth = qpNorm->extAttrs.cqAttr.recvCqDepth;
    qpCb->memAlign = qpNorm->extAttrs.memAlign;

    qpCb->channel = RsIbvCreateCompChannel(rdevCb->ibCtx);
    CHK_PRT_RETURN(qpCb->channel == NULL, hccp_err("ibv_create_comp_channel failed! errno(%d)", errno), -EINVAL);
    qpCb->qosAttr.tc = (RS_ROCE_DSCP_33 & RS_DSCP_MASK) << RS_DSCP_OFF;
    qpCb->qosAttr.sl = RS_ROCE_4_SL;
    qpCb->timeout = RS_QP_ATTR_TIMEOUT;
    qpCb->retryCnt = RS_QP_ATTR_RETRY_CNT;

    qpCb->udpSport = qpNorm->extAttrs.udpSport;

    qpCb->aiOpSupport = qpNorm->aiOpSupport;
    qpCb->grpId = rdevCb->rsCb->grpId;
    qpCb->cqCstmFlag = qpNorm->extAttrs.dataPlaneFlag.bs.cqCstm;
    qpCb->useResvMem = qpNorm->extAttrs.cstmFlag.bs.useResvMem;
    qpCb->resvMemPoolId = qpNorm->extAttrs.resvMemPoolId;

    ret = RsEpollCtl(rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_ADD, qpCb->channel->fd, EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
    if (ret) {
        RsIbvDestroyCompChannel(qpCb->channel);
        hccp_err("add channel fd failed ret %d", ret);
        return ret;
    }
#endif
    return 0;
}

STATIC int RsAllocQpcbWithAttrs(struct RsRdevCb *rdevCb, struct RsQpCb **qpCb, struct RsQpNormWithAttrs *qpNorm)
{
    int ret;

    ret = RsCallocQpcb(1, qpCb);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d errno:%d", ret, errno), -ENOMEM);

    ret = pthread_mutex_init(&(*qpCb)->qpMutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto qp_mutex_init_err;
    }

    ret = pthread_mutex_init(&(*qpCb)->cqeErrInfo.mutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto cqe_mutex_init_err;
    }

    ret = RsQpcbInitWithAttrs(rdevCb, *qpCb, qpNorm);
    if (ret) {
        hccp_err("create qp tx rx failed ret %d", ret);
        goto rs_qpcb_init_err;
    }

    ret = RsInitMemPool(*qpCb);
    if (ret) {
        hccp_err("init mem pool failed ret %d", ret);
        goto rs_init_mem_err;
    }

    ret = RsDrvCreateCqWithAttrs(*qpCb, qpNorm->isExt, &qpNorm->extAttrs.cqAttr);
    if (ret) {
        hccp_err("create cq failed ret %d", ret);
        goto create_cq_err;
    }

    return 0;

create_cq_err:
    RsDeinitMemPool(*qpCb);

rs_init_mem_err:
    RsQpcbDeinit(rdevCb, *qpCb);

rs_qpcb_init_err:
    pthread_mutex_destroy(&(*qpCb)->cqeErrInfo.mutex);

cqe_mutex_init_err:
    pthread_mutex_destroy(&(*qpCb)->qpMutex);

qp_mutex_init_err:
    free(*qpCb);
    *qpCb = NULL;

    return ret;
}

STATIC int RsQpCheckQpNorm(struct RsQpNormWithAttrs *qpNorm, int *qpMode)
{
    CHK_PRT_RETURN(qpNorm == NULL, hccp_err("qp_norm is NULL!"), -EINVAL);
    CHK_PRT_RETURN(qpNorm->extAttrs.version != QP_CREATE_WITH_ATTR_VERSION,
        hccp_err("attr version[%d] mismatch, expect [%d]", qpNorm->extAttrs.version, QP_CREATE_WITH_ATTR_VERSION),
        -EINVAL);

    *qpMode = qpNorm->extAttrs.qpMode;
    if (*qpMode < 0 || *qpMode >= RA_RS_ERR_QP_MODE) {
        hccp_err("qp_mode[%d] must be greater than or equal to 0 and less than %d", *qpMode, RA_RS_ERR_QP_MODE);
        return -EINVAL;
    }

    if (*qpMode == RA_RS_OP_QP_MODE_EXT) {
        *qpMode = RA_RS_OP_QP_MODE;
    }

    qpNorm->extAttrs.qpMode = *qpMode;
    return 0;
}

#ifdef CUSTOM_INTERFACE
STATIC void RsQpPrepareCqDataPlaneInfo(struct ibv_cq *ibCq, struct AiDataPlaneCq *dataPlaneCq)
{
    struct hns_roce_cq_data_plane_info cqInfo = {0};

    (void)RsRoceGetCqDataPlaneInfo(ibCq, &cqInfo);
    dataPlaneCq->cqn = cqInfo.cqn;
    dataPlaneCq->bufAddr = cqInfo.buf_addr;
    dataPlaneCq->cqeSize = cqInfo.cqe_size;
    dataPlaneCq->depth = cqInfo.depth;
    dataPlaneCq->headAddr = cqInfo.head_addr;
    dataPlaneCq->tailAddr = cqInfo.tail_addr;
    dataPlaneCq->swdbAddr = cqInfo.swdb_addr;
    dataPlaneCq->dbReg = cqInfo.db_reg;
    hccp_info_rma("cqn:%u buf_addr:0x%llx cqe_size:%u depth:%u head_addr:0x%llx tail_addr:0x%llx swdb_addr:0x%llx",
        dataPlaneCq->cqn, dataPlaneCq->bufAddr, dataPlaneCq->cqeSize, dataPlaneCq->depth, dataPlaneCq->headAddr,
        dataPlaneCq->tailAddr, dataPlaneCq->swdbAddr);
}

STATIC void RsQpPrepareWqDataPlaneInfo(struct hns_roce_wq_data_plane_info *wqInfo, struct AiDataPlaneWq *dataPlaneWq)
{
    dataPlaneWq->wqn = wqInfo->wqn;
    dataPlaneWq->bufAddr = wqInfo->buf_addr;
    dataPlaneWq->wqebbSize = wqInfo->wqebb_size;
    dataPlaneWq->depth = wqInfo->depth;
    dataPlaneWq->headAddr = wqInfo->head_addr;
    dataPlaneWq->tailAddr = wqInfo->tail_addr;
    dataPlaneWq->swdbAddr = wqInfo->swdb_addr;
    dataPlaneWq->dbReg = wqInfo->db_reg;
    hccp_info_rma("wqn:%u buf_addr:0x%llx wqebb_size:%u depth:%u head_addr:%u tail_addr:%u swdb_addr:0x%llx",
        dataPlaneWq->wqn, dataPlaneWq->bufAddr, dataPlaneWq->wqebbSize, dataPlaneWq->depth, dataPlaneWq->headAddr,
        dataPlaneWq->tailAddr, dataPlaneWq->swdbAddr);
}

STATIC void RsQpPrepareQpDataPlaneInfo(struct ibv_qp *ibQp, struct AiDataPlaneWq *dataPlaneSq,
    struct AiDataPlaneWq *dataPlaneRq)
{
    struct hns_roce_qp_data_plane_info qpInfo = {0};

    (void)RsRoceGetQpDataPlaneInfo(ibQp, &qpInfo);
    RsQpPrepareWqDataPlaneInfo(&qpInfo.sq, dataPlaneSq);
    RsQpPrepareWqDataPlaneInfo(&qpInfo.rq, dataPlaneRq);
}

STATIC void RsQpPrepareDataPlaneInfo(struct RsQpNormWithAttrs *qpNorm, struct RsQpCb *qpCb,
    struct RsQpRespWithAttrs *qpResp)
{
    // skip to prepare cq data plane info
    if (qpNorm->extAttrs.dataPlaneFlag.bs.cqCstm != 0) {
        qpResp->aiScqAddr = (unsigned long long)(uintptr_t)qpCb->ibSendCq;
        qpResp->aiRcqAddr = (unsigned long long)(uintptr_t)qpCb->ibRecvCq;
        RsQpPrepareCqDataPlaneInfo(qpCb->ibSendCq, &qpResp->dataPlaneInfo.scq);
        RsQpPrepareCqDataPlaneInfo(qpCb->ibRecvCq, &qpResp->dataPlaneInfo.rcq);
    }

    // skip to prepare qp data plane info
    if (qpNorm->aiOpSupport != 0) {
        RsQpPrepareQpDataPlaneInfo(qpCb->ibQp, &qpResp->dataPlaneInfo.sq, &qpResp->dataPlaneInfo.rq);
    }
}
#endif

STATIC void RsQpPrepareQpResp(struct RsQpNormWithAttrs *qpNorm, struct RsQpCb *qpCb, struct RsQpRespWithAttrs *qpResp)
{
    if (qpNorm->isExp != 0) {
        qpCb->isExp = RS_IS_EXP;
    } else {
        qpCb->isExp = RS_NOT_EXP;
    }

    qpResp->aiQpAddr = (unsigned long long)(uintptr_t)qpCb->ibQp;
    qpResp->sqIndex = (unsigned int)qpCb->sqIndex;
    qpResp->dbIndex = (unsigned int)qpCb->dbIndex;
    qpResp->gidIdx = (unsigned int)qpCb->qpInfoLo.gidIdx;
    qpResp->psn = (unsigned int)qpCb->qpInfoLo.psn;

#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        RsQpPrepareDataPlaneInfo(qpNorm, qpCb, qpResp);
    }
#endif

    return;
}

RS_ATTRI_VISI_DEF int RsQpCreateWithAttrs(unsigned int phyId, unsigned int rdevIndex, struct RsQpNormWithAttrs *qpNorm,
    struct RsQpRespWithAttrs *qpResp)
{
    struct RsRdevCb *rdevCb = NULL;
    struct RsQpCb *qpCb = NULL;
    int qpMode;
    int ret;

    RS_QP_PARA_CHECK(phyId);
    CHK_PRT_RETURN(qpResp == NULL, hccp_err("qp_resp is NULL!"), -EINVAL);

    ret = RsQpCheckQpNorm(qpNorm, &qpMode);
    CHK_PRT_RETURN(ret != 0, hccp_err("check qp mode failed, ret:%d", ret), ret);

    ret = RsQpQueryInfo(phyId, rdevIndex, &rdevCb, qpMode);
    CHK_PRT_RETURN(ret, hccp_err("query qp info failed:%d", ret), ret);

    ret = RsAllocQpcbWithAttrs(rdevCb, &qpCb, qpNorm);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d", ret), ret);

    ret = RsDrvQpCreateWithAttrs(qpCb, qpNorm);
    if (ret) {
        hccp_err("create drv qp create failed:%d", ret);
        goto create_qp_err;
    }

    ret = ibv_req_notify_cq(qpCb->ibSendCq, 0);
    if (ret) {
        hccp_err("Couldn't request send CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = ibv_req_notify_cq(qpCb->ibRecvCq, 0);
    if (ret) {
        hccp_err("Couldn't request recv CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = RsQpNotifyMr(rdevCb, qpCb, &qpResp->qpn); // alloc mr
    if (ret) {
        hccp_err("store qp notify mr failed:%d", ret);
        goto ret_noritfy_cq;
    }

    RsQpPrepareQpResp(qpNorm, qpCb, qpResp);

    return 0;

ret_noritfy_cq:
    RsDrvQpDestroy(qpCb);

create_qp_err:
    RsFreeQpcb(rdevCb, qpCb);
    return ret;
}

void RsMrRelease(struct RsQpCb *qpCb)
{
    struct RsMrCb *mrTmp2 = NULL;
    struct RsMrCb *mrTmp = NULL;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RS_LIST_GET_HEAD_ENTRY(mrTmp, mrTmp2, &qpCb->mrList, list, struct RsMrCb);
    for (; (&mrTmp->list) != &qpCb->mrList;
         mrTmp = mrTmp2, mrTmp2 = list_entry(mrTmp2->list.next, struct RsMrCb, list)) {
        if (mrTmp->ibMr != qpCb->rdevCb->notifyMr) {
            (void)RsDrvMrDereg(mrTmp->ibMr);
        }
        RsListDel(&mrTmp->list);
        free(mrTmp);
        mrTmp = NULL;
    }

    RS_LIST_GET_HEAD_ENTRY(mrTmp, mrTmp2, &qpCb->remMrList, list, struct RsMrCb);
    for (; (&mrTmp->list) != &qpCb->remMrList;
         mrTmp = mrTmp2, mrTmp2 = list_entry(mrTmp2->list.next, struct RsMrCb, list)) {
        RsListDel(&mrTmp->list);
        free(mrTmp);
        mrTmp = NULL;
    }
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
}

STATIC void RsQpRelease(struct RsQpCb *qpCb)
{
    RS_PTHREAD_MUTEX_LOCK(&qpCb->rdevCb->rdevMutex);
    RsListDel(&qpCb->list);
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->rdevCb->rdevMutex);
    RsIbvAckCqEvents(qpCb->ibSendCq, qpCb->numSendCqEvents);
    RsIbvAckCqEvents(qpCb->ibRecvCq, qpCb->numRecvCqEvents);

    // dereg mr
    RsMrRelease(qpCb);
}

RS_ATTRI_VISI_DEF int RsQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn)
{
    struct RsQpCb *qpCb = NULL;
    int ret;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed! qpn %u, ret %d", qpn, ret), ret);

    RsQpRelease(qpCb);

    // destroy qp
    RsDrvQpDestroy(qpCb);
    RsDrvDestroyCq(qpCb);
    RsDeinitMemPool(qpCb);

    qpCb->rdevCb->qpCnt--;
    ret = RsQpcbDeinit(qpCb->rdevCb, qpCb);
    if (ret) {
        hccp_err("rs_qpcb_deinit failed! ret[%d]", ret);
    }

    pthread_mutex_destroy(&qpCb->cqeErrInfo.mutex);
    pthread_mutex_destroy(&qpCb->qpMutex);
    hccp_info_rma("qp %d destroy qp, send wr[%u].", qpn, qpCb->sendWrNum);

    free(qpCb);
    qpCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsQpDestroyWithoutCQ(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn)
{
    struct RsQpCb *qpCb = NULL;
    int ret;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed! qpn %u, ret %d", qpn, ret), ret);

    RsQpRelease(qpCb);

    // destroy qp
    RsDrvQpDestroy(qpCb);
    RsDeinitMemPool(qpCb);

    qpCb->rdevCb->qpCnt--;
    ret = RsQpcbDeinit(qpCb->rdevCb, qpCb);
    if (ret) {
        hccp_err("rs_qpcb_deinit failed! ret[%d]", ret);
    }

    pthread_mutex_destroy(&qpCb->cqeErrInfo.mutex);
    pthread_mutex_destroy(&qpCb->qpMutex);
    hccp_info_rma("qp %d destroy qp without cq, send wr[%u].", qpn, qpCb->sendWrNum);

    free(qpCb);
    qpCb = NULL;
    return ret;
}

static void RsQpConnectAsyncMr(const struct RsQpCb *qpCb)
{
    int ret;
    struct RsMrCb *mrCb = NULL;
    struct RsMrCb *mrCb2 = NULL;

    RS_LIST_GET_HEAD_ENTRY(mrCb, mrCb2, &qpCb->mrList, list, struct RsMrCb);
    for (; (&mrCb->list) != &qpCb->mrList; mrCb = mrCb2, mrCb2 = list_entry(mrCb2->list.next, struct RsMrCb, list)) {
        ret = RsMrInfoSync(mrCb);
        if (ret) {
            hccp_warn_rma("rs_mr_info_sync unsuccessful, ret:%d", ret);
        }
    }
}

STATIC void RsQpConnectAsyncQpcbSet(int fd, struct RsQpCb *qpCb)
{
    int ret;
    ret = RsSocketSend(fd, &qpCb->qpInfoLo, sizeof(struct RsQpInfo));
    if (ret == sizeof(struct RsQpInfo)) {
        qpCb->sendLen += (uint32_t)ret;
        qpCb->state = RS_QP_STATUS_CONNECTING;
    } else {
        qpCb->state = RS_QP_STATUS_TIMEOUT;
    }
}

STATIC void RsQpConnectAsyncLength(int fd, struct RsQpCb *qpCb)
{
    int ret;
    struct RsQpLenInfo msg;

    msg.cmd = RS_CMD_LEN_INFO;
    msg.len = qpCb->sendLen;

    ret = RsSocketSend(fd, &msg, sizeof(struct RsQpLenInfo));
    if (ret != sizeof(struct RsQpLenInfo)) {
        qpCb->state = RS_QP_STATUS_TIMEOUT;
    }
}

static int RsQpConnectAsyncInitPara(struct RsQpConnPara qpConnPara, int fd, struct RsQpCb **qpCb,
    struct RsConnInfo **conn)
{
    int ret;

    CHK_PRT_RETURN(qpConnPara.phyId >= RS_MAX_DEV_NUM, hccp_err("param error ! phyId:%u", qpConnPara.phyId), -EINVAL);

    CHK_PRT_RETURN(fd < 0, hccp_err("param error ! fd:%d must be non-negative", fd), -EINVAL);

    ret = RsQpn2qpcb(qpConnPara.phyId, qpConnPara.rdevIndex, qpConnPara.qpn, qpCb);
    CHK_PRT_RETURN(ret, hccp_err("get qpcb failed, qpn %u, ret %d", qpConnPara.qpn, ret), ret);

    ret = RsFd2conn(fd, conn);
    CHK_PRT_RETURN(ret, hccp_err("get conn failed, fd %d, ret %d", fd, ret), ret);

    RsGetCurTime(&((*qpCb)->startTime));
    (*qpCb)->sendLen = 0;
    (*qpCb)->recvLen = 0;
    (*qpCb)->expectLen = 0;
    (*qpCb)->connInfo = *conn;

    return 0;
}

STATIC int RsTypicalQpStateModifytoRtr(struct RsQpCb *qpCb, struct TypicalQp *localQpInfo,
    struct TypicalQp *remoteQpInfo)
{
    struct ibv_port_attr portAttr = {0};
    union ibv_gid remoteInfoGid = {0};
    struct ibv_qp_attr attr = {0};
    int ret;

    attr.qp_state = IBV_QPS_RTR;
    attr.dest_qp_num = remoteQpInfo->qpn;
    attr.rq_psn = remoteQpInfo->psn;
    attr.min_rnr_timer = RS_QP_ATTR_MIN_RNR_TIMER;
    (attr.ah_attr).is_global = 0;
    (attr.ah_attr).sl = localQpInfo->sl;
    (attr.ah_attr).src_path_bits = 0;
    (attr.ah_attr).port_num = qpCb->rdevCb->ibPort;

    attr.path_mtu = RsDrvSetMtu(qpCb);
    CHK_PRT_RETURN(attr.path_mtu < IBV_MTU_1024,
        hccp_err("qpn[%u] failed to set mtu, mtu[%d] < [%d]", localQpInfo->qpn, attr.path_mtu, IBV_MTU_1024), -EPERM);
    if (qpCb->rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
        attr.max_dest_rd_atomic = RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE;
    } else {
        attr.max_dest_rd_atomic = RS_MAX_RD_ATOMIC_NUM;
    }
    (attr.ah_attr).grh.traffic_class = localQpInfo->tc;
    // get gid_idx dynamically to avoid gid_idx changed issue: refresh gid_idx when it changed
    ret = RsDrvGetGidIndex(qpCb->rdevCb, &portAttr, &qpCb->qpInfoLo.gidIdx);
    if (ret == 0 && localQpInfo->gidIdx != (uint32_t)qpCb->qpInfoLo.gidIdx) {
        hccp_warn_rma("qpn[%u] qp_mode[%d] refresh gid_idx[%u] to [%d]", localQpInfo->qpn, qpCb->qpMode,
            localQpInfo->gidIdx, qpCb->qpInfoLo.gidIdx);
        localQpInfo->gidIdx = (uint32_t)qpCb->qpInfoLo.gidIdx;
    }

    (void)memcpy_s(remoteInfoGid.raw, HCCP_GID_RAW_LEN, remoteQpInfo->gid, HCCP_GID_RAW_LEN);
    if (remoteInfoGid.global.interface_id) {
        attr.ah_attr.is_global = 1;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.dgid = remoteInfoGid;
        attr.ah_attr.grh.sgid_index = localQpInfo->gidIdx;
    }

    ret = RsIbvModifyQp(qpCb->ibQp, &attr,
        IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN | IBV_QP_MAX_DEST_RD_ATOMIC |
            IBV_QP_MIN_RNR_TIMER);
    CHK_PRT_RETURN(ret,
        hccp_err("[modifyto_rtr]local_qpn[%u] remote_qpn[%u] ibv_modify_qp failed ret[%d], errno[%d]", localQpInfo->qpn,
            remoteQpInfo->qpn, ret, errno),
        -EOPENSRC);
    hccp_info_rma("qp qos attr: qpn[%u] tc[%u] sl[%u]", localQpInfo->qpn, localQpInfo->tc, localQpInfo->sl);
    return 0;
}

STATIC int RsTypicalQpStateModifytoRts(struct RsQpCb *qpCb, struct TypicalQp *localQpInfo)
{
    struct ibv_qp_attr attr = {0};
    int ret;

    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = (uint8_t)localQpInfo->retryTime;
    attr.retry_cnt = (uint8_t)localQpInfo->retryCnt;
    attr.rnr_retry = RS_QP_ATTR_RNR_RETRY;
    attr.sq_psn = localQpInfo->psn;
    if (qpCb->rdevCb->rsCb->hccpMode == NETWORK_PEER_ONLINE) {
        attr.max_rd_atomic = RS_MAX_RD_ATOMIC_NUM_PEER_ONLINE;
    } else {
        attr.max_rd_atomic = RS_MAX_RD_ATOMIC_NUM;
    }

    ret = RsIbvModifyQp(qpCb->ibQp, &attr,
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[modifyto_rts]local_qpn[%u] ibv_modify_qp failed ret[%d], errno[%d]", localQpInfo->qpn, ret, errno),
        -EOPENSRC);

    hccp_info_rma("qp rdma attr: qpn[%u] timeout[%u] retrycnt[%u]", localQpInfo->qpn, localQpInfo->retryTime,
        localQpInfo->retryCnt);
    return 0;
}

STATIC void RsTypicalQpModifyInfoRelated(struct RsQpCb *qpCb, struct TypicalQp *localQpInfo,
    struct TypicalQp *remoteQpInfo)
{
    qpCb->state = RS_QP_STATUS_CONNECTED;
    // local qp info related: no need to relate qpn, psn, gid_idx, gid
    qpCb->qosAttr.tc = (unsigned char)localQpInfo->tc;
    qpCb->qosAttr.sl = (unsigned char)localQpInfo->sl;
    qpCb->retryCnt = localQpInfo->retryCnt;
    qpCb->timeout = localQpInfo->retryTime;
    // remote qp info related
    qpCb->qpInfoRem.qpn = (int)remoteQpInfo->qpn;
    qpCb->qpInfoRem.psn = (int)remoteQpInfo->psn;
    qpCb->qpInfoRem.gidIdx = (int)remoteQpInfo->gidIdx;
    (void)memcpy_s(qpCb->qpInfoRem.gid.raw, HCCP_GID_RAW_LEN, remoteQpInfo->gid, HCCP_GID_RAW_LEN);
}

STATIC void RsTypicalQpModifyExtend(struct RsQpCb *qpCb, struct TypicalQp *localQpInfo, struct TypicalQp *remoteQpInfo)
{
    struct ibv_hyroce_feature output = {0};
    struct ibv_hyroce_feature input = {0};
    struct ibv_qp_attr_extend attr = {0};
    uint32_t needMoreNego = 0;
    int ret = 0;

    if (qpCb->rdevCb->ibCtxEx == NULL) {
        return;
    }

    if (localQpInfo->tc != qpCb->qosAttr.tc || localQpInfo->sl != qpCb->qosAttr.sl) {
        hccp_warn_rma("localQpInfo tc:%u sl:%u is not equal to qpCb tc:%u sl:%u", localQpInfo->tc, localQpInfo->sl,
            qpCb->qosAttr.tc, qpCb->qosAttr.sl);
        return;
    }

    ret = memcpy_s(&input, sizeof(struct ibv_hyroce_feature), &remoteQpInfo->feature, sizeof(struct HyperFeature));
    if (ret != 0) {
        hccp_warn_rma("memcpy_s feature unsuccessful, ret:%d qpn:%u ibv_hyroce_feature len:%zu HyperFeature len:%zu",
            ret, qpCb->ibQp->qp_num, sizeof(struct ibv_hyroce_feature), sizeof(struct HyperFeature));
        return;
    }

    ret = RsIbvNegoQpHyroceFeature(qpCb->rdevCb->ibCtxEx, qpCb->ibQp, &input, &output, &needMoreNego);
    if (ret != 0) {
        hccp_warn_rma("RsIbvNegoQpHyroceFeature unsuccessful, ret:%d qpn:%u errno:%d", ret, qpCb->ibQp->qp_num, errno);
        return;
    }

    attr.qp = qpCb->ibQp;
    (void)memcpy_s(&attr.feature, sizeof(struct ibv_hyroce_feature), &output, sizeof(struct ibv_hyroce_feature));
    attr.udp_src_port = localQpInfo->udpSport;
    ret = RsIbvModifyQpExtend(qpCb->rdevCb->ibCtxEx, &attr,
        IBV_QP_ATTR_EXTEND_UDP_SRC_PORT | IBV_QP_ATTR_EXTEND_HYROCE_FEATURE);
    if (ret != 0) {
        hccp_warn_rma("RsIbvModifyQpExtend unsuccessful, ret:%d qpn:%u errno:%d", ret, qpCb->ibQp->qp_num, errno);
        return;
    }

    hccp_dbg("RsIbvModifyQpExtend successful, qpn:%u", qpCb->ibQp->qp_num);
    return;
}

STATIC int RsTypicalQueryQpAttr(struct RsQpCb *qpCb, struct TypicalQpAttr *qpAttr)
{
    unsigned int qpAttrMask = HNS_ROCE_AI_QPC_UDPSPN;
    struct hns_roce_qpc_attr_val qpAttrVal = {0};
    struct ibv_qp_init_attr initAttr = {0};
    struct ibv_qp_attr ibvQpAttr = {0};
    int ret = 0;

    (void)qpAttrMask;
    (void)qpAttrVal;
#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        ret = RsRoceQueryQpc(qpCb->ibQp, &qpAttrVal, qpAttrMask);
        if (ret != 0) {
            hccp_warn_rma("qpn:%d query qpc unsuccessful, ret %d", qpCb->qpInfoLo.qpn, ret);
        } else {
            qpCb->udpSport = qpAttrVal.udp_sport;
        }
    }
#endif

    ret = RsIbvQueryQp(qpCb->ibQp, &ibvQpAttr, IBV_QP_PATH_MTU, &initAttr);
    CHK_PRT_RETURN(ret, hccp_err("RsIbvQueryQp failed, ret:%d errno:%d", ret, errno), -EOPENSRC);

    if (qpCb->ibQpEx != NULL) {
        qpAttr->vendorPrivInfo = qpCb->ibQpEx->vendor_priv_info.value;
    }

    qpAttr->udpSport = qpCb->udpSport;
    qpAttr->pathMtu = (int)ibvQpAttr.path_mtu;
    return 0;
}

RS_ATTRI_VISI_DEF int RsTypicalQpModify(unsigned int phyId, unsigned int rdevIndex, struct TypicalQp localQpInfo,
    struct TypicalQp remoteQpInfo, struct TypicalQpAttr *qpAttr)
{
    struct ibv_qp_init_attr initAttr = {0};
    struct ibv_qp_attr attr = {0};
    struct RsQpCb *qpCb = NULL;
    int ret;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("[modify]phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM),
        -EINVAL);

    CHK_PRT_RETURN(RsQpn2qpcb(phyId, rdevIndex, localQpInfo.qpn, &qpCb),
        hccp_err("[modify]rs_qpn2qpcb qpn:%u failed, phyId[%u]", localQpInfo.qpn, phyId), -EACCES);

    CHK_PRT_RETURN(qpCb->state == RS_QP_STATUS_CONNECTED,
        hccp_info_rma("local_qpn:%u remote_qpn:%u already been connected, no need to modify again", localQpInfo.qpn,
            remoteQpInfo.qpn),
        0);

    // see ib_modify_qp_is_ok for status modify, only support modify qp from INIT to RTR
    ret = RsIbvQueryQp(qpCb->ibQp, &attr, IBV_QP_STATE, &initAttr);
    CHK_PRT_RETURN(ret != 0 || attr.qp_state != IBV_QPS_INIT,
        hccp_err("query qpn:%u failed, ret:%d or state:%d != %d", localQpInfo.qpn, ret, attr.qp_state, IBV_QPS_INIT),
        -EOPENSRC);

    RsTypicalQpModifyExtend(qpCb, &localQpInfo, &remoteQpInfo);

    ret = RsTypicalQpStateModifytoRtr(qpCb, &localQpInfo, &remoteQpInfo);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[modify]local_qpn:%u remote_qpn:%u modify to rtr failed, ret %d", localQpInfo.qpn, remoteQpInfo.qpn,
            ret),
        ret);

    ret = RsTypicalQpStateModifytoRts(qpCb, &localQpInfo);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[modify]local_qpn:%u remote_qpn:%u modify to rts failed, ret %d", localQpInfo.qpn, remoteQpInfo.qpn,
            ret),
        ret);

    ret = RsTypicalQueryQpAttr(qpCb, qpAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsTypicalQueryQpAttr failed, ret %d local_qpn:%u", ret, localQpInfo.qpn), ret);

    RsTypicalQpModifyInfoRelated(qpCb, &localQpInfo, &remoteQpInfo);

    hccp_info_rma("local_qpn:%u remote_qpn:%u modify succ, udpSport:%u", localQpInfo.qpn, remoteQpInfo.qpn,
        qpCb->udpSport);

    return 0;
}

STATIC int RsQpStateBatchModifytoPause(struct RsQpCb *qpCb)
{
    int ret;

    ret = RsDrvQpStateModifytoReset(qpCb);
    CHK_PRT_RETURN(ret, hccp_err("qp modify to reset failed, ret %d", ret), ret);

    hccp_info_rma("local qpn[%d] remote qpn[%d] modify to pause succ", qpCb->qpInfoLo.qpn, qpCb->qpInfoRem.qpn);
    return 0;
}

STATIC int RsQpStateBatchModifytoConnected(struct RsQpCb *qpCb)
{
    struct ibv_qp_attr attr;
    int ret;

    ret = memset_s(&attr, sizeof(struct ibv_qp_attr), 0, sizeof(struct ibv_qp_attr));
    CHK_PRT_RETURN(ret, hccp_err("memset_s attr failed ret %d", ret), -ESAFEFUNC);

    ret = RsDrvQpStateModifytoInit(qpCb, &attr);
    CHK_PRT_RETURN(ret, hccp_err("qp modify to init failed, ret %d", ret), ret);
    ret = RsDrvQpStateModifytoRtr(qpCb, &attr);
    CHK_PRT_RETURN(ret, hccp_err("qp modify to rtr failed, ret %d", ret), ret);
    ret = RsDrvQpStateModifytoRts(qpCb, &attr);
    CHK_PRT_RETURN(ret, hccp_err("qp modify to rts failed, ret %d", ret), ret);

    hccp_info_rma("local qpn[%d] remote qpn[%d] modify to rts succ", qpCb->qpInfoLo.qpn, qpCb->qpInfoRem.qpn);
    return 0;
}

RS_ATTRI_VISI_DEF int RsQpBatchModify(unsigned int phyId, unsigned int rdevIndex, int status, int qpn[], int qpnNum)
{
    struct RsQpCb *qpCb = NULL;
    int ret;
    int i;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("[modify]phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM),
        -EINVAL);

    for (i = 0; i < qpnNum; i++) {
        CHK_PRT_RETURN(RsQpn2qpcb(phyId, rdevIndex, (uint32_t)qpn[i], &qpCb),
            hccp_err("[modify]rs_qpn2qpcb failed, phyId[%u]", phyId), -EACCES);

        /*
         * see ib_modify_qp_is_ok for status modify
         * only support modify qp from STATUS_PAUSE(RESET) to STATUS_CONNECTED(INIT)
         */
        if (status == RS_QP_STATUS_CONNECTED && qpCb->state == RS_QP_STATUS_PAUSE) {
            ret = RsQpStateBatchModifytoConnected(qpCb);
            CHK_PRT_RETURN(ret,
                hccp_err("modify_qp qpn[%d]:%d to connected failed, ret[%d] phyId[%u]", i, qpn[i], ret, phyId), ret);
        } else if (status == RS_QP_STATUS_PAUSE) {
            ret = RsQpStateBatchModifytoPause(qpCb);
            CHK_PRT_RETURN(ret,
                hccp_err("modify_qp qpn[%d]:%d to pause failed, ret[%d] phyId[%u]", i, qpn[i], ret, phyId), ret);
        } else {
            hccp_err("modify_qp qpn[%d]:%d failed, not support to modify status[%d] to status[%d], phyId[%u]", i,
                qpn[i], qpCb->state, status, phyId);
            return -EINVAL;
        }

        qpCb->state = status;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsSetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int lbValue)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQpn2qpcb failed ret:%d", ret), ret);

    return RsRoceSetQpLbValue(qpCb->ibQp, lbValue);
}

RS_ATTRI_VISI_DEF int RsGetQpLbValue(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int *lbValue)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQpn2qpcb failed ret:%d", ret), ret);

    return RsRoceGetQpLbValue(qpCb->ibQp, lbValue);
}

RS_ATTRI_VISI_DEF int RsQpConnectAsync(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, int fd)
{
    int ret;
    struct RsQpCb *qpCb = NULL;
    struct RsConnInfo *conn = NULL;
    struct RsQpConnPara qpConnPara;
    hccp_info_rma("qp:%d, fd:%d", qpn, fd);

    qpConnPara.phyId = phyId;
    qpConnPara.rdevIndex = rdevIndex;
    qpConnPara.qpn = qpn;
    ret = RsQpConnectAsyncInitPara(qpConnPara, fd, &qpCb, &conn);
    CHK_PRT_RETURN(ret, hccp_err("rs_qp_connect_async_init_para failed, qpn %u, ret %d", qpn, ret), ret);

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);

    if (qpCb->state == RS_QP_STATUS_REM_FD_CLOSE) {
        hccp_warn_rma("remote qp fd close, can not use it anymore!");
        RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
        return -EFAULT;
    }

    if ((qpCb->state == RS_QP_STATUS_CONNECTED) || (qpCb->state == RS_QP_STATUS_CONNECTING)) {
        hccp_warn_rma("qp %d has already sync! state[%d]", qpCb->qpInfoLo.qpn, qpCb->state);
        RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
        return -EEXIST;
    }

    RsQpConnectAsyncQpcbSet(fd, qpCb);

    hccp_info_rma("after socket fd %d send QP %u, chipId %u, state:%d!", fd, qpn, qpCb->rdevCb->rsCb->chipId,
        qpCb->state);

    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);

    RsQpMrRecvHandle(fd, qpCb);

    RsQpConnectAsyncMr(qpCb);

    RsQpConnectAsyncLength(fd, qpCb);

    hccp_info_rma("QP %d async done, state:%d!", qpn, qpCb->state);

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetQpStatus(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct RsQpStatusInfo *qpInfo)
{
    unsigned int qpAttrMask = HNS_ROCE_AI_QPC_UDPSPN;
    struct hns_roce_qpc_attr_val qpAttrVal = {0};
    struct RsQpCb *qpCb = NULL;
    int ret;

    (void)qpAttrMask;
    (void)qpAttrVal;
    CHK_PRT_RETURN(qpInfo == NULL, hccp_err("param error, qpInfo is NULL"), -EINVAL);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("get qp cb failed, qpn:%u, ret %d", qpn, ret), ret);

    // qp state is CONNECTED, no need to handle
    if (qpCb->state == RS_QP_STATUS_CONNECTED) {
        goto update_qp_cb;
    }

    // modify state to CONNECTED
    if (qpCb->expectLen == qpCb->recvLen - sizeof(struct RsQpLenInfo)) {
        qpCb->state = RS_QP_STATUS_CONNECTED;
    } else {
        RsQpMrRecvHandle(qpCb->connInfo->connfd, qpCb);
        goto out;
    }

update_qp_cb:
#ifdef CUSTOM_INTERFACE
    if (RsIsCustomInterfaceSupported()) {
        ret = RsRoceQueryQpc(qpCb->ibQp, &qpAttrVal, qpAttrMask);
        if (ret != 0) {
            hccp_warn_rma("qpn:%d query qpc unsuccessful, ret %d", qpCb->qpInfoLo.qpn, ret);
        } else {
            qpCb->udpSport = qpAttrVal.udp_sport;
        }
    }
#endif
out:
    hccp_dbg("qp:%u, state:%d, udpSport:%u", qpn, qpCb->state, qpCb->udpSport);
    qpInfo->status = qpCb->state;
    qpInfo->udpSport = qpCb->udpSport;

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetQpContext(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn, void **qp,
    void **sendCq, void **recvCq)
{
    int ret;
    struct RsQpCb *qpCb = NULL;

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("phyId:%u >= [%d], is invalid", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_qpn2qpcb failed ret[%d]", ret), ret);

    *qp = qpCb->ibQp;
    *sendCq = qpCb->ibSendCq;
    *recvCq = qpCb->ibRecvCq;

    hccp_dbg("qpn[%u] succ", qpn);

    return 0;
}

int RsQueryRdevCb(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb **rdevCb)
{
    int ret;
    unsigned int chipId;
    struct rs_cb *rsCb = NULL;

    RS_QP_PARA_CHECK(phyId);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("rs_query_rdev_cb phyId[%u] invalid, ret:%d", phyId, ret), ret);

    ret = RsDev2rscb(chipId, &rsCb, false);
    CHK_PRT_RETURN(ret, hccp_err("rs_query_rdev_cb get rs_cb failed, ret:%d", ret), -ENODEV);

    ret = RsGetRdevCb(rsCb, rdevIndex, rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_get_rdev_cb failed! ret:%d, rdevIndex:%u", ret, rdevIndex), ret);

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLbMax(unsigned int phyId, unsigned int rdevIndex, int *lbMax)
{
    struct RsRdevCb *rdevCb = NULL;
    int ret = 0;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQueryRdevCb phyId:%u rdev_index:%u ret:%d", phyId, rdevIndex, ret), ret);

    return RsRoceGetQpNum(rdevCb->ibCtx, lbMax);
}

STATIC int RsBuildUpQpcb(struct RsCqContext *cqContext, struct ibv_qp_init_attr *qpInitAttr, struct RsQpCb **qpCb)
{
    int ret;

    ret = RsCallocQpcb(1, qpCb);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d errno:%d", ret, errno), -ENOMEM);

    ret = pthread_mutex_init(&(*qpCb)->qpMutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto pthread_mutex_init_err;
    }

    (*qpCb)->rdevCb = cqContext->rdevCb;
    RS_INIT_LIST_HEAD(&(*qpCb)->mrList);
    RS_INIT_LIST_HEAD(&(*qpCb)->remMrList);

    (*qpCb)->eqNum = cqContext->eqNum;
    (*qpCb)->channel = cqContext->channel;
    (*qpCb)->ibSendCq = cqContext->ibSendCq;
    (*qpCb)->ibRecvCq = cqContext->ibRecvCq;
    (*qpCb)->sendEvent = cqContext->sendEvent;
    (*qpCb)->recvEvent = cqContext->recvEvent;
    (*qpCb)->numRecvCqEvents = 0;
    (*qpCb)->numSendCqEvents = 0;
    (*qpCb)->srqContext = cqContext->srqContext;
    (*qpCb)->state = RS_QP_STATUS_DISCONNECT;
    (*qpCb)->ibPd = cqContext->rdevCb->ibPd;
    (*qpCb)->txDepth = qpInitAttr->cap.max_send_wr;
    (*qpCb)->rxDepth = qpInitAttr->cap.max_recv_wr;
    (*qpCb)->qosAttr.tc = (RS_ROCE_DSCP_33 & RS_DSCP_MASK) << RS_DSCP_OFF;
    (*qpCb)->qosAttr.sl = RS_ROCE_4_SL;
    (*qpCb)->timeout = RS_QP_ATTR_TIMEOUT;
    (*qpCb)->retryCnt = RS_QP_ATTR_RETRY_CNT;

    return 0;

pthread_mutex_init_err:
    free(*qpCb);
    (*qpCb) = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsCreateCqEvent(struct RsCqContext *cqContext, struct CqAttr *attr)
{
    int ret;
    cqContext->channel = RsIbvCreateCompChannel(cqContext->rdevCb->ibCtx);

    if (cqContext->channel == NULL) {
        hccp_err("ibv_create_comp_channel failed, ret %d, errno(%d)", -EINVAL, errno);
        return -EINVAL;
    }

    hccp_info_rma("comp channel fd[%d].", cqContext->channel->fd);
    ret = RsEpollCtl(cqContext->rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_ADD, cqContext->channel->fd,
        EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
    if (ret) {
        hccp_err("add channel fd failed ret %d", ret);
        goto rs_cq_epoll_ctl_err;
    }
#endif

    ret = RsDrvCreateCqEvent(cqContext, attr);
    if (ret) {
        hccp_err("create drv cq event failed:%d", ret);
        goto rs_cq_create_err;
    }

    return ret;
rs_cq_create_err:
    ret = RsEpollCtl(cqContext->rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_DEL, cqContext->channel->fd,
        EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
    if (ret) {
        hccp_err("del channel fd failed ret %d", ret);
    }
#endif
rs_cq_epoll_ctl_err:
    if (cqContext->channel != NULL) {
        RsIbvDestroyCompChannel(cqContext->channel);
        cqContext->channel = NULL;
    }
    return ret;
}

RS_ATTRI_VISI_DEF int RsCqCreate(unsigned int phyId, unsigned int rdevIndex, struct CqAttr *attr)
{
    int ret;
    struct RsRdevCb *rdevCb = NULL;
    struct RsCqContext *cqContext = NULL;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    if (ret) {
        hccp_err("rs_query_rdev_cb phyId[%u] rdev_index[%u], ret %d", phyId, rdevIndex, ret);
        return ret;
    }

    cqContext = calloc(1, sizeof(struct RsCqContext));
    if (cqContext == NULL) {
        return -ENOMEM;
    }
    cqContext->rdevCb = rdevCb;
    cqContext->eqNum = 0;
    if (attr->sendChannel == NULL && attr->recvChannel == NULL) {
        if (*attr->ibSendCq == NULL && *attr->ibRecvCq != NULL) {
            // 只创建sq cq
            cqContext->cqCreateMode = RS_SQ_CQ_CREATE;
            cqContext->ibRecvCq = *attr->ibRecvCq;
            cqContext->srqContext = attr->srqContext;
        } else {
            // 创建sq&rq cq
            cqContext->cqCreateMode = RS_NORMAL_CQ_CREATE;
        }
        ret = RsCreateCqEvent(cqContext, attr);
        if (ret) {
            hccp_err("create cq event failed:%d", ret);
            goto rs_cq_create_err;
        }
    } else if (attr->sendChannel != NULL && attr->recvChannel != NULL) {
        // 使用输入comp channel创建sq&rq
        ret = RsDrvCreateCqWithChannel(cqContext, attr);
        if (ret) {
            hccp_err("create drv cq with channel failed:%d", ret);
            goto rs_cq_create_err;
        }
    } else {
        hccp_err("rs create cq failed, sendChannel or recvChannel is NULL.");
        ret = -EPERM;
        goto rs_cq_create_err;
    }

    *attr->qpContext = cqContext;
    return 0;

rs_cq_create_err:
    free(cqContext);
    cqContext = NULL;

    return ret;
}

RS_ATTRI_VISI_DEF int RsTypicalCqCreate(unsigned int phyId, unsigned int rdevIndex, unsigned int cqDepth,
    unsigned int *cqn)
{
    struct RsTypicalCqEntry *entry;
    struct RsTypicalCqEntry *tmp;
    int ret;
    struct RsRdevCb *rdevCb = NULL;
    struct ibv_cq *ibCq = NULL;
    struct rdma_lite_device_cq_attr deviceCqAttr;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    if (ret) {
        hccp_err("rs_query_rdev_cb phyId[%u] rdevIndex[%u], ret %d", phyId, rdevIndex, ret);
        return ret;
    }

    ret = RsDrvTypicalCqCreate(rdevCb, cqDepth, cqn, &ibCq, &deviceCqAttr);
    if (ret) {
        hccp_err("rs_drv_typical_cq_create failed, cqDepth[%u] ret[%d]", cqDepth, ret);
        return ret;
    }

    pthread_mutex_lock(&gRsTypicalCqMutex);
    if (gRsTypicalCqList.next == NULL) {
        RS_INIT_LIST_HEAD(&gRsTypicalCqList);
    }

    RS_LIST_GET_HEAD_ENTRY(tmp, entry, &gRsTypicalCqList, list, struct RsTypicalCqEntry);
    for (; &tmp->list != &gRsTypicalCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct RsTypicalCqEntry, list)) {
        if (tmp->phyId == phyId && tmp->rdevIndex == rdevIndex && tmp->cqn == *cqn) {
            tmp->ibCq = ibCq;
            tmp->deviceCqAttr = deviceCqAttr;
            pthread_mutex_unlock(&gRsTypicalCqMutex);
            hccp_info_rma("RsTypicalCqCreate updated: phyId[%u] rdevIndex[%u] cqn[%u] cqDepth[%u]", phyId, rdevIndex,
                *cqn, cqDepth);
            return 0;
        }
    }

    entry = calloc(1, sizeof(struct RsTypicalCqEntry));
    if (entry == NULL) {
        pthread_mutex_unlock(&gRsTypicalCqMutex);
        hccp_err("RsTypicalCqCreate calloc failed, cqn[%u]", *cqn);
        return -ENOMEM;
    }
    entry->phyId = phyId;
    entry->rdevIndex = rdevIndex;
    entry->cqn = *cqn;
    entry->ibCq = ibCq;
    entry->deviceCqAttr = deviceCqAttr;
    RsListAddTail(&entry->list, &gRsTypicalCqList);
    pthread_mutex_unlock(&gRsTypicalCqMutex);

    hccp_info_rma("RsTypicalCqCreate success: phyId[%u] rdevIndex[%u] cqn[%u] cqDepth[%u]", phyId, rdevIndex, *cqn,
        cqDepth);

    return 0;
}

RS_ATTRI_VISI_DEF int RsTypicalCqDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn)
{
    struct RsTypicalCqEntry *entry;
    struct RsTypicalCqEntry *tmp;
    int ret;

    pthread_mutex_lock(&gRsTypicalCqMutex);
    if (gRsTypicalCqList.next == NULL) {
        pthread_mutex_unlock(&gRsTypicalCqMutex);
        hccp_err("rs_typical_cq_destroy: cqn[%u] not found phyId[%u] rdevIndex[%u]", cqn, phyId, rdevIndex);
        return -EINVAL;
    }

    RS_LIST_GET_HEAD_ENTRY(tmp, entry, &gRsTypicalCqList, list, struct RsTypicalCqEntry);
    for (; &tmp->list != &gRsTypicalCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct RsTypicalCqEntry, list)) {
        if (tmp->phyId == phyId && tmp->rdevIndex == rdevIndex && tmp->cqn == cqn) {
            RsListDel(&tmp->list);
            pthread_mutex_unlock(&gRsTypicalCqMutex);

            if (tmp->ibCq != NULL) {
                ret = RsIbvDestroyCq(tmp->ibCq);
                if (ret) {
                    hccp_err("rs_ibv_destroy_cq failed cqn[%u] ret[%d]", cqn, ret);
                    free(tmp);
                    return ret;
                }
            }
            free(tmp);
            hccp_info_rma("RsTypicalCqDestroy success: phyId[%u] rdevIndex[%u] cqn[%u]", phyId, rdevIndex, cqn);
            return 0;
        }
    }
    pthread_mutex_unlock(&gRsTypicalCqMutex);

    hccp_err("rs_typical_cq_destroy: cqn[%u] not found phyId[%u] rdevIndex[%u]", cqn, phyId, rdevIndex);
    return -EINVAL;
}

RS_ATTRI_VISI_DEF int RsGetLiteCqAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int cqn,
    struct rdma_lite_device_cq_attr *deviceCqAttr)
{
    struct RsTypicalCqEntry *entry;
    struct RsTypicalCqEntry *tmp;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(deviceCqAttr);

    pthread_mutex_lock(&gRsTypicalCqMutex);
    if (gRsTypicalCqList.next == NULL) {
        pthread_mutex_unlock(&gRsTypicalCqMutex);
        hccp_err("rs_get_lite_cq_attr: cqn[%u] not found phyId[%u] rdevIndex[%u]", cqn, phyId, rdevIndex);
        return -EINVAL;
    }

    RS_LIST_GET_HEAD_ENTRY(tmp, entry, &gRsTypicalCqList, list, struct RsTypicalCqEntry);
    for (; &tmp->list != &gRsTypicalCqList;
         tmp = entry, entry = list_entry(entry->list.next, struct RsTypicalCqEntry, list)) {
        if (tmp->phyId == phyId && tmp->rdevIndex == rdevIndex && tmp->cqn == cqn) {
            ret = memcpy_s(deviceCqAttr, sizeof(*deviceCqAttr), &tmp->deviceCqAttr, sizeof(tmp->deviceCqAttr));
            pthread_mutex_unlock(&gRsTypicalCqMutex);
            if (ret) {
                hccp_err("memcpy_s failed, ret:%d", ret);
                return ret;
            }
            hccp_info_rma("RsGetLiteCqAttr success: cqn[%u] depth[%u]", cqn, deviceCqAttr->depth);
            return 0;
        }
    }
    pthread_mutex_unlock(&gRsTypicalCqMutex);

    hccp_err("rs_get_lite_cq_attr: cqn[%u] not found phyId[%u] rdevIndex[%u]", cqn, phyId, rdevIndex);
    return -EINVAL;
}

RS_ATTRI_VISI_DEF int RsQpCreateWithCQWithAttrs(unsigned int phyId, unsigned int rdevIndex, unsigned int sendCqn,
    unsigned int recvCqn, struct RsQpNormWithAttrs *qpNorm, struct RsQpRespWithAttrs *qpResp)
{
    struct RsTypicalCqEntry *entry;
    struct RsTypicalCqEntry *tmp;
    struct RsRdevCb *rdevCb = NULL;
    struct RsQpCb *qpCb = NULL;
    struct ibv_cq *sendIbCq = NULL;
    struct ibv_cq *recvIbCq = NULL;
    struct rdma_lite_device_cq_attr sendDeviceCqAttr;
    struct rdma_lite_device_cq_attr recvDeviceCqAttr;
    int qpMode;
    int ret;
    bool sendFound = false;
    bool recvFound = false;

    RS_QP_PARA_CHECK(phyId);
    CHK_PRT_RETURN(qpResp == NULL, hccp_err("qp_resp is NULL!"), -EINVAL);

    ret = RsQpCheckQpNorm(qpNorm, &qpMode);
    CHK_PRT_RETURN(ret != 0, hccp_err("check qp mode failed, ret:%d", ret), ret);

    ret = RsQpQueryInfo(phyId, rdevIndex, &rdevCb, qpMode);
    CHK_PRT_RETURN(ret, hccp_err("query qp info failed:%d", ret), ret);

    pthread_mutex_lock(&gRsTypicalCqMutex);
    if (gRsTypicalCqList.next != NULL) {
        RS_LIST_GET_HEAD_ENTRY(tmp, entry, &gRsTypicalCqList, list, struct RsTypicalCqEntry);
        for (; &tmp->list != &gRsTypicalCqList;
             tmp = entry, entry = list_entry(entry->list.next, struct RsTypicalCqEntry, list)) {
            if (tmp->phyId == phyId && tmp->rdevIndex == rdevIndex && tmp->cqn == sendCqn) {
                sendIbCq = tmp->ibCq;
                sendDeviceCqAttr = tmp->deviceCqAttr;
                sendFound = true;
                break;
            }
        }
    }
    pthread_mutex_unlock(&gRsTypicalCqMutex);
    CHK_PRT_RETURN(!sendFound,
        hccp_err("send cq not found: sendCqn[%u] phyId[%u] rdevIndex[%u]", sendCqn, phyId, rdevIndex), -EINVAL);

    pthread_mutex_lock(&gRsTypicalCqMutex);
    if (gRsTypicalCqList.next != NULL) {
        RS_LIST_GET_HEAD_ENTRY(tmp, entry, &gRsTypicalCqList, list, struct RsTypicalCqEntry);
        for (; &tmp->list != &gRsTypicalCqList;
             tmp = entry, entry = list_entry(entry->list.next, struct RsTypicalCqEntry, list)) {
            if (tmp->phyId == phyId && tmp->rdevIndex == rdevIndex && tmp->cqn == recvCqn) {
                recvIbCq = tmp->ibCq;
                recvDeviceCqAttr = tmp->deviceCqAttr;
                recvFound = true;
                break;
            }
        }
    }
    pthread_mutex_unlock(&gRsTypicalCqMutex);
    CHK_PRT_RETURN(!recvFound,
        hccp_err("recv cq not found: recvCqn[%u] phyId[%u] rdevIndex[%u]", recvCqn, phyId, rdevIndex), -EINVAL);

    ret = RsCallocQpcb(1, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("alloc mem for qp_cb failed, ret:%d errno:%d", ret, errno), -ENOMEM);

    ret = pthread_mutex_init(&qpCb->qpMutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto qp_mutex_init_err;
    }

    ret = pthread_mutex_init(&qpCb->cqeErrInfo.mutex, NULL);
    if (ret) {
        hccp_err("pthread_mutex_init failed, ret %d", ret);
        goto cqe_mutex_init_err;
    }

    ret = RsQpcbInitWithAttrs(rdevCb, qpCb, qpNorm);
    if (ret) {
        hccp_err("create qp tx rx failed ret %d", ret);
        goto rs_qpcb_init_err;
    }

    ret = RsInitMemPool(qpCb);
    if (ret) {
        hccp_err("init mem pool failed ret %d", ret);
        goto rs_init_mem_err;
    }

    // Assign pre-existing CQs (instead of RsDrvCreateCqWithAttrs)
    qpCb->ibSendCq = sendIbCq;
    qpCb->ibRecvCq = recvIbCq;
    qpCb->qpResp.sendCqData = sendDeviceCqAttr;
    qpCb->qpResp.recvCqData = recvDeviceCqAttr;
    qpCb->sendCqDepth = sendDeviceCqAttr.depth;
    qpCb->recvCqDepth = recvDeviceCqAttr.depth;

    ret = RsDrvQpCreateWithAttrs(qpCb, qpNorm);
    if (ret) {
        hccp_err("Create drv qp create failed:%d", ret);
        goto create_qp_err;
    }

    ret = ibv_req_notify_cq(qpCb->ibSendCq, 0);
    if (ret) {
        hccp_err("Can't request send CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = ibv_req_notify_cq(qpCb->ibRecvCq, 0);
    if (ret) {
        hccp_err("Can't request recv CQ notification, ret:%d", ret);
        ret = -EOPENSRC;
        goto ret_noritfy_cq;
    }

    ret = RsQpNotifyMr(rdevCb, qpCb, &qpResp->qpn);
    if (ret) {
        hccp_err("Store qp notify mr failed:%d", ret);
        goto ret_noritfy_cq;
    }

    RsQpPrepareQpResp(qpNorm, qpCb, qpResp);

    return 0;

ret_noritfy_cq:
    RsDrvQpDestroy(qpCb);

create_qp_err:
    // Do NOT call RsDrvDestroyCq — CQs are not owned by this QP
    RsDeinitMemPool(qpCb);
    (void)RsQpcbDeinit(rdevCb, qpCb);

rs_init_mem_err:
rs_qpcb_init_err:
    pthread_mutex_destroy(&qpCb->cqeErrInfo.mutex);

cqe_mutex_init_err:
    pthread_mutex_destroy(&qpCb->qpMutex);

qp_mutex_init_err:
    free(qpCb);
    qpCb = NULL;

    return ret;
}

RS_ATTRI_VISI_DEF int RsCqDestroy(unsigned int phyId, unsigned int rdevIndex, struct CqAttr *attr)
{
    int ret;
    struct RsRdevCb *rdevCb = NULL;
    struct RsCqContext *cqContext = NULL;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_query_rdev_cb phyId[%u] rdev_index[%u], ret %d", phyId, rdevIndex, ret), ret);

    cqContext = *attr->qpContext;

    ret = RsDrvDestroyCqEvent(cqContext);
    if (ret) {
        hccp_err("rs_drv_destroy_cq_event failed ret %d", ret);
    }

    if (cqContext->channel != NULL) {
        ret = RsEpollCtl(rdevCb->rsCb->connCb.epollfd, EPOLL_CTL_DEL, cqContext->channel->fd, EPOLLIN | EPOLLRDHUP);
#ifndef CA_CONFIG_LLT
        if (ret) {
            hccp_err("del channel fd failed ret %d", ret);
        }
#endif
        RsIbvDestroyCompChannel(cqContext->channel);
        cqContext->channel = NULL;
    }

    free(cqContext);
    cqContext = NULL;

    return ret;
}

RS_ATTRI_VISI_DEF int RsNormalQpCreate(unsigned int phyId, unsigned int rdevIndex, struct ibv_qp_init_attr *qpInitAttr,
    struct RsQpResp *qpResp, void **qp)
{
    struct RsCqContext *cqContext = NULL;
    struct RsRdevCb *rdevCb = NULL;
    struct RsQpCb *qpCb = NULL;
    int ret;

    CHK_PRT_RETURN(qpResp == NULL, hccp_err("qp_resp is NULL!"), -EINVAL);
    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_query_rdev_cb phyId[%u] rdev_index[%u], ret %d", phyId, rdevIndex, ret), ret);

    CHK_PRT_RETURN(qpInitAttr == NULL, hccp_err("qp_init_attr is NULL!"), -EINVAL);

    cqContext = qpInitAttr->qp_context;
    CHK_PRT_RETURN(cqContext == NULL, hccp_err("cq_context is NULL!"), -EINVAL);
    CHK_PRT_RETURN(rdevCb != cqContext->rdevCb,
        hccp_err("rs_query_rdev_cb phyId[%u] rdev_index[%u],"
                 "rdevCb is invalid.",
            phyId, rdevIndex),
        -EINVAL);

    ret = RsBuildUpQpcb(cqContext, qpInitAttr, &qpCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_build_up_qpcb failed, ret:%d", ret), ret);

    ret = RsDrvNormalQpCreate(qpCb, qpInitAttr);
    if (ret) {
        hccp_err("create drv qp create failed:%d", ret);
        goto create_qp_err;
    }

    RS_PTHREAD_MUTEX_LOCK(&rdevCb->rdevMutex);
    RsListAddTail(&qpCb->list, &rdevCb->qpList);
    RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rdevMutex);
    rdevCb->qpCnt++;
    *qp = qpCb->ibQp;
    qpResp->qpn = (unsigned int)qpCb->qpInfoLo.qpn;
    qpResp->gidIdx = (unsigned int)qpCb->qpInfoLo.gidIdx;
    qpResp->psn = (unsigned int)qpCb->qpInfoLo.psn;
    qpResp->gid = qpCb->qpInfoLo.gid;

    hccp_info_rma("qp %d create qp.", qpResp->qpn);

    return 0;

create_qp_err:
    pthread_mutex_destroy(&qpCb->qpMutex);
    free(qpCb);
    qpCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsNormalQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn)
{
    struct RsQpCb *qpCb = NULL;
    int ret;

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    RsQpRelease(qpCb);

    // destroy qp
    RsDrvQpDestroy(qpCb);

    qpCb->rdevCb->qpCnt--;

    pthread_mutex_destroy(&qpCb->qpMutex);
    hccp_info_rma("qp %d destroy qp, send wr[%u].", qpn, qpCb->sendWrNum);

    free(qpCb);
    qpCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsCreateCompChannel(unsigned int phyId, unsigned int rdevIndex, void **compChannel)
{
    int ret;
    unsigned int chipId;

    struct RsRdevCb *rdevCb = NULL;

    CHK_PRT_RETURN(compChannel == NULL || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret,
        hccp_err("rs_create_comp_channel rsGetLocalDevIDByHostDevID phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    *compChannel = (void *)RsIbvCreateCompChannel(rdevCb->ibCtx);
    if (*compChannel == NULL) {
        hccp_err("rs_ibv_create_comp_channel failed, errno(%d)", errno);
        return -EOPENSRC;
    }
    hccp_info_rma("create comp channel success!");
    return 0;
}

RS_ATTRI_VISI_DEF int RsDestroyCompChannel(void *compChannel)
{
    int ret;
    struct ibv_comp_channel *rsCompChannel = (struct ibv_comp_channel *)compChannel;

    ret = RsIbvDestroyCompChannel(rsCompChannel);
    CHK_PRT_RETURN(ret, hccp_err("rs_destroy_comp_channel failed."), ret);
    hccp_info_rma("destroy comp channel success!");

    return 0;
}

RS_ATTRI_VISI_DEF int RsCreateSrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr *attr)
{
    int ret;
    struct RsRdevCb *rdevCb = NULL;
    struct RsCqContext *cqContext = NULL;

    CHK_PRT_RETURN(attr == NULL || attr->context == NULL || attr->ibRecvCq == NULL || attr->ibSrq == NULL ||
                       phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret, hccp_err("rs_query_rdev_cb phyId[%u] rdev_index[%u], ret %d", phyId, rdevIndex, ret), ret);

    cqContext = calloc(1, sizeof(struct RsCqContext));
    if (cqContext == NULL) {
        return -ENOMEM;
    }

    cqContext->rdevCb = rdevCb;
    cqContext->eqNum = 0;
    cqContext->cqCreateMode = RS_SRQ_CQ_CREATE;
    *attr->context = cqContext;

    struct CqAttr cqAttr = {0};
    cqAttr.recvCqDepth = attr->cqDepth;
    cqAttr.recvCqEventId = attr->srqEventId;
    cqAttr.ibRecvCq = attr->ibRecvCq;
    // 创建srq cq
    ret = RsCreateCqEvent(cqContext, &cqAttr);
    if (ret) {
        hccp_err("rs_create_cq_event create cq failed! ret:%d", ret);
        goto create_cq_event_err;
    }
    cqContext->ibSrqCq = *attr->ibRecvCq;

    struct ibv_srq_init_attr srqInitAttr = {.attr = {.max_wr = attr->srqDepth, .max_sge = attr->maxSge}};
    hccp_info_rma("max_wr [%u], max_sge[%u]", srqInitAttr.attr.max_wr, srqInitAttr.attr.max_sge);

    // 创建srq
    *attr->ibSrq = RsIbvCreateSrq(rdevCb->ibPd, &srqInitAttr);
    if (*attr->ibSrq == NULL) {
        hccp_err("rs_ibv_create_srq failed.");
        ret = -EOPENSRC;
        goto create_srq_err;
    }
    hccp_info_rma("create srq success!");

    return 0;
create_cq_event_err:
create_srq_err:
    cqAttr.qpContext = attr->context;
    RsCqDestroy(phyId, rdevIndex, &cqAttr);

    return ret;
}

RS_ATTRI_VISI_DEF int RsDestroySrq(unsigned int phyId, unsigned int rdevIndex, struct SrqAttr *attr)
{
    int ret;

    CHK_PRT_RETURN(*attr->context == NULL || *attr->ibSrq == NULL || phyId >= RS_MAX_DEV_NUM,
        hccp_err("param err, NULL pointer or phyId:%u >= [%d]", phyId, RS_MAX_DEV_NUM), -EINVAL);

    struct CqAttr cqAttr = {0};
    struct RsCqContext *cqContext = *attr->context;
    cqAttr.qpContext = attr->context;
    RsIbvAckCqEvents(cqContext->ibSrqCq, cqContext->numRecvCqEvents);

    // 销毁srq cq
    ret = RsCqDestroy(phyId, rdevIndex, &cqAttr);
    CHK_PRT_RETURN(ret, hccp_err("rs_cq_destroy destroy cq failed! ret:%d", ret), ret);

    ret = RsIbvDestroySrq(*attr->ibSrq);
    CHK_PRT_RETURN(ret, hccp_err("rs_ibv_destroy_srq failed."), ret);

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLiteSupport(unsigned int phyId, unsigned int rdevIndex, int *supportLite)
{
    int ret;
    unsigned int chipId;
    struct RsRdevCb *rdevCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(supportLite);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error ! phyId:%u", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    rdevCb->supportLite = 1;
    *supportLite = rdevCb->supportLite;

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLiteRdevCap(unsigned int phyId, unsigned int rdevIndex, struct LiteRdevCapResp *resp)
{
    int ret;
    unsigned int chipId;
    struct RsRdevCb *rdevCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(resp);

    CHK_PRT_RETURN(phyId >= RS_MAX_DEV_NUM, hccp_err("rs set param error ! phyId:%u", phyId), -EINVAL);
    ret = rsGetLocalDevIDByHostDevID(phyId, &chipId);
    CHK_PRT_RETURN(ret, hccp_err("phyId[%u] invalid, ret %d", phyId, ret), ret);

    ret = RsRdev2rdevCb(chipId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0 || rdevCb == NULL, hccp_err("rs_rdev2rdev_cb for chip_id[%u] failed, ret %d", chipId, ret),
        ret);

    ret = RsIbvExpQueryDevice(rdevCb->ibCtx, &resp->cap);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ibv_exp_query_device for phyId[%u] failed, ret %d", phyId, ret), ret);

    ret = memcpy_s(resp, sizeof(struct dev_cap_info), (void *)&resp->cap, sizeof(resp->cap));
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, src_len:%u, dst_len:%u", ret, (unsigned int)sizeof(resp->cap),
            (unsigned int)sizeof(struct dev_cap_info));
        return ret;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLiteQpCqAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteQpCqAttrResp *resp)
{
    int ret;
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(resp);

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    ret = memcpy_s(resp, sizeof(struct LiteQpCqAttrResp), (void *)&qpCb->qpResp, sizeof(qpCb->qpResp));
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, src_len:%u, dst_len:%u", ret, (unsigned int)sizeof(qpCb->qpResp),
            (unsigned int)sizeof(struct LiteQpCqAttrResp));
        return ret;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLiteQpAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteQpAttrResp *resp)
{
    int ret;
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(resp);

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    ret = memcpy_s(resp, sizeof(struct LiteQpAttrResp), (void *)&qpCb->qpResp.qpData, sizeof(qpCb->qpResp.qpData));
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, src_len:%u, dst_len:%u", ret, (unsigned int)sizeof(qpCb->qpResp.qpData),
            (unsigned int)sizeof(struct LiteQpAttrResp));
        return ret;
    }

    return 0;
}

RS_ATTRI_VISI_DEF int RsGetLiteMemAttr(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteMemAttrResp *resp)
{
    int ret;
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(resp);

    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    ret = memcpy_s(resp, sizeof(struct LiteMemAttrResp), (void *)&qpCb->memResp, sizeof(qpCb->memResp));
    if (ret) {
        hccp_err("memcpy_s failed, ret:%d, src_len:%u, dst_len:%u", ret, (unsigned int)sizeof(qpCb->memResp),
            (unsigned int)sizeof(struct LiteMemAttrResp));
        return ret;
    }

    return 0;
}

STATIC void RsGetMrInfo(struct RsQpCb *qpCb, struct LiteMrInfo *mr, uint32_t maxMrNum, struct RsListHead *mrList)
{
    struct RsMrCb *mrTmp = NULL;
    struct RsMrCb *mrTmp2 = NULL;
    uint32_t i = 0;

    RS_PTHREAD_MUTEX_LOCK(&qpCb->qpMutex);
    RS_LIST_GET_HEAD_ENTRY(mrTmp, mrTmp2, mrList, list, struct RsMrCb);
    for (; (&mrTmp->list) != mrList; mrTmp = mrTmp2, mrTmp2 = list_entry(mrTmp2->list.next, struct RsMrCb, list)) {
        if (i < maxMrNum) {
            mr[i].key = mrTmp->mrInfo.rkey;
            mr[i].addr = mrTmp->mrInfo.addr;
            mr[i].len = mrTmp->mrInfo.len;
            i++;
        } else {
            break;
        }
    }

    RS_PTHREAD_MUTEX_ULOCK(&qpCb->qpMutex);
}

RS_ATTRI_VISI_DEF int RsGetLiteConnectedInfo(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct LiteConnectedInfoResp *resp)
{
    int ret;
    struct RsQpCb *qpCb = NULL;

    RS_CHECK_POINTER_NULL_RETURN_INT(resp);
    RS_QP_PARA_CHECK(phyId);
    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed qpn %u, ret %d", qpn, ret), ret);

    resp->state = (unsigned int)qpCb->state;
    if (resp->state == RS_QP_STATUS_CONNECTED) {
        RsGetMrInfo(qpCb, &resp->localMr[0], RA_MR_MAX_NUM, &qpCb->mrList);
        RsGetMrInfo(qpCb, &resp->remMr[0], RA_MR_MAX_NUM, &qpCb->remMrList);
        resp->qosAttr.sl = qpCb->qosAttr.sl;
        resp->qosAttr.tc = qpCb->qosAttr.tc;
    }

    return 0;
}
