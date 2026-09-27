/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_DRV_RDMA_H
#define RS_DRV_RDMA_H

#define _GNU_SOURCE
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <pthread.h>
#include <infiniband/verbs.h>

#include "securec.h"
#include "rs.h"
#include "rs_inner.h"
#include "verbs_exp.h"
#include "hccp_common.h"
#include "ascend_hal_external.h"

#define DEFAULT_ACCESS_FLAG                                                                                            \
    (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_ATOMIC)
#define RS_SGLIST_MAX 16

enum RsCqCreateMode {
    RS_NORMAL_CQ_CREATE = 0,
    RS_SRQ_CQ_CREATE,
    RS_SQ_CQ_CREATE,
};

void RsDrvPollCqHandle(struct RsQpCb *qpCb);
void RsDrvPollSrqCqHandle(struct RsQpCb *qpCb);
int RsDrvGetGidIndex(struct RsRdevCb *rdevCb, struct ibv_port_attr *attr, int *idx);
int RsDrvCreateCq(struct RsQpCb *qpCb, int isExt);
int RsDrvCreateCqWithAttrs(struct RsQpCb *qpCb, int isExt, struct CqExtAttr *cqAttr);
int RsDrvQpStateModifytoReset(struct RsQpCb *qpCb);
int RsDrvQpStateModifytoInit(struct RsQpCb *qpCb, struct ibv_qp_attr *attr);
enum ibv_mtu RsDrvSetMtu(struct RsQpCb *qpCb);
int RsDrvQpStateModifytoRtr(struct RsQpCb *qpCb, struct ibv_qp_attr *attr);
int RsDrvQpStateModifytoRts(struct RsQpCb *qpCb, struct ibv_qp_attr *attr);
struct ibv_mr *RsDrvMrReg(struct ibv_pd *pd, char *addr, size_t length, int access);
struct ibv_mr *RsDrvExpMrReg(struct ibv_pd *pd, char *addr, size_t length, int access,
    struct roce_process_sign roceSign);
int RsDrvMrDereg(struct ibv_mr *ibMr);
void RsDrvDestroyCq(struct RsQpCb *qpCb);
int RsDrvOpenDevice(struct rs_cb *rscb, struct ibv_device *ibDev);
int RsDrvRegNotifyMr(struct RsRdevCb *rdevCb);
int RsDrvQueryNotifyAndAllocPd(struct RsRdevCb *rdevCb);
int RsDrvPostRecv(struct RsQpCb *qpCb, struct RecvWrlistData *wr, unsigned int recvNum, unsigned int *completeNum);
int RsDrvSendExp(struct RsQpCb *qpCb, struct RsMrCb *mrCb, struct RsMrCb *remMrCb, struct SendWr *wr,
    struct SendWrRsp *wrRsp);
int RsDrvSendIbv(struct RsQpCb *qpCb, struct RsMrCb *mrCb, struct RsMrCb *remMrCb, struct SendWr *wr, int immData);

int RsDrvQpInfoRelated(struct RsQpCb *qpCb, struct RsRdevCb *rdevCb, struct ibv_port_attr *attr,
    struct ibv_qp_attr *qpAttr);
int RsDrvQpCreate(struct RsQpCb *qpCb, struct RsQpNorm *qpNorm);
int RsDrvQpCreateWithAttrs(struct RsQpCb *qpCb, struct RsQpNormWithAttrs *qpNorm);
void RsDrvQpDestroy(struct RsQpCb *qpCb);
int RsDrvCreateCqEvent(struct RsCqContext *cqContext, struct CqAttr *attr);
int RsDrvCreateCqWithChannel(struct RsCqContext *cqContext, struct CqAttr *attr);
int RsDrvTypicalCqCreate(struct RsRdevCb *rdevCb, unsigned int cqDepth, unsigned int *cqn, struct ibv_cq **ibCq,
    struct rdma_lite_device_cq_attr *deviceCqAttr);
int RsDrvDestroyCqEvent(struct RsCqContext *cqContext);
int RsDrvNormalQpCreate(struct RsQpCb *qpCb, struct ibv_qp_init_attr *qpInitAttr);
int RsDrvInitCqeErrInfo(void);
void RsDrvDeinitCqeErrInfo(void);
int RsDrvGetCqeErrInfo(struct CqeErrInfo *info);
int RsQueryEvent(int cqEventId, struct event_summary **event);
void RsDrvEventDestroy(struct event_summary *event);
int RsDrvCompareIpGid(int family, union HccpIpAddr localIp, union ibv_gid *gid);
#ifdef CUSTOM_INTERFACE
void RsCloseBackupIbCtx(struct RsRdevCb *rdevCb);
#endif
#endif // RS_DRV_RDMA_H