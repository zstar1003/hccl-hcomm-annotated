/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include "ut_dispatch.h"
#include "ra_rs_err.h"
#include "rs_common_inner.h"
#include "rs_inner.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "rs_rdma.h"
#include "rs_drv_rdma.h"
#include "rs.h"
#include "rs_nda.h"
#include "tc_ut_rs_nda.h"

#define rs_nda_ut_msg(fmt, args...) fprintf(stderr, "\t>>>>> " fmt, ##args)

extern __thread struct rs_cb* gRsCb;
extern int RsQueryRdevCb(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb);
extern int RsNdaQpCreateEx(struct RsQpCb* qpCb, struct ibv_qp_init_attr_extend* qpInitAttrEx, struct NdaQpInfo* info);
extern int RsGetNdaPcieDbCb(struct RsNdaCb* ndaCb, uint64_t hva, struct NdaPcieDbCb** ndaDbCb);
extern int RsGetNdaUbDbCb(struct RsNdaCb* ndaCb, uint64_t guidL, uint64_t guidH, struct NdaUbDbCb** ndaDbCb);
extern int RsNdaCqCreateEx(
    struct RsRdevCb* rdevCb, struct ibv_cq_init_attr_extend* cqInitAttrEx, struct NdaCqInfo* info, void** ibvCqExt);
extern void* RsNdaUbAlloc(size_t size);
extern void* RsNdaDbMmap(struct doorbell_map_desc* desc);
extern int RsNdaDbUnmap(void* ptr, struct doorbell_map_desc* desc);

int RsQueryRdevCbNdaStub(unsigned int phyId, unsigned int rdevIndex, struct RsRdevCb** rdevCb)
{
    static struct RsRdevCb ndaRdevCb = {0};
    ndaRdevCb.rsCb = gRsCb;

    *rdevCb = &ndaRdevCb;
    return 0;
}

int DlDrvMemGetAttributeStub(DVdeviceptr vptr, struct DVattribute* attr)
{
    attr->memType = DV_MEM_LOCK_DEV;
    return 0;
}

int DlHalHostRegisterStub(void* srcPtr, UINT64 size, UINT32 flag, UINT32 devId, void** dstPtr)
{
    static void* sDstPtr = NULL;
    if (dstPtr == NULL) {
        return -1;
    }

    *dstPtr = &sDstPtr;
    return 0;
}

int DlHalHostUnRegisterExStub(void* srcPtr, UINT32 devId)
{
    uint64_t alignHva = AlignDown((uint64_t)1, RA_RS_4K_PAGE_SIZE);

    if ((uint64_t)(uintptr_t)srcPtr == alignHva) {
        return 0;
    }
    return -1;
}

void TcRsNdaGetDirectFlag()
{
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int directFlag = 0;
    int ret = 0;

    ret = RsNdaGetDirectFlag(RS_MAX_DEV_NUM, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, -EINVAL);

    mocker_clean();
    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    ret = RsNdaGetDirectFlag(phyId, rdevIndex, &directFlag);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();
}

void TcRsNdaQpCreateFailed()
{
    struct NdaQpInitAttr attr = {0};
    struct RsQpResp qpResp = {0};
    struct NdaQpInfo info = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    int ret = 0;
    struct RsNdaCb ndaCb = {0};
    struct NdaOps ops = {0};

    gRsCb = malloc(sizeof(struct rs_cb));
    attr.ops = &ops;
    gRsCb->ndaCb = &ndaCb;

    ret = RsNdaQpCreate(phyId, rdevIndex, NULL, &info, &qpResp);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_MAX;
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_INDEP_UB;
    mocker_clean();
    mocker(RsQueryRdevCb, 1, -1);
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    mocker(RsNdaQpCreateEx, 1, -1);
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &qpResp);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaQpCreate()
{
    struct NdaQpInitAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct NdaQpInfo info = {0};
    struct rdev rdevInfo = {0};
    uint32_t rdevIndex = 0;
    uint32_t phyId = 0;
    int ret = 0;

    struct RsQpResp resp = {0};
    struct NdaOps ops = {0};
    uint32_t qpn = 0;

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_nda_ut_msg("resource prepare begin..................\n");
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("RS INIT, ret:%d !\n", ret);

    ret = RsRdevInit(rdevInfo, NO_USE, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    attr.ops = &ops;
    ret = RsNdaQpCreate(phyId, rdevIndex, &attr, &info, &resp);
    EXPECT_INT_EQ(ret, 0);
    qpn = resp.qpn;
    rs_nda_ut_msg("RS CREATE NDA QP: QPN:%d, ret:%d\n", qpn, ret);

    ret = RsNdaQpDestroy(phyId, rdevIndex, qpn);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("RS destroy NDA QP: ret:%d\n", ret);

    ret = RsRdevDeinit(phyId, NO_USE, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
    rs_nda_ut_msg("resource free done..................\n");
}

void TcRsNdaQpCreateEx()
{
    struct ibv_qp_init_attr_extend qpInitAttrEx = {0};
    struct ibv_context_extend ibCtxEx = {0};
    struct RsRdevCb rdevCb = {0};
    struct NdaQpInfo info = {0};
    struct RsQpCb qpCb = {0};
    struct ibv_pd pd = {0};
    int ret = 0;

    qpInitAttrEx.pd = &pd;
    rdevCb.ibCtxEx = &ibCtxEx;
    qpCb.rdevCb = &rdevCb;
    mocker_clean();
    mocker(RsIbvQueryQp, 1, -1);
    ret = RsNdaQpCreateEx(&qpCb, &qpInitAttrEx, &info);
    EXPECT_INT_EQ(ret, -EOPENSRC);
    mocker_clean();

    mocker(RsDrvQpInfoRelated, 1, -1);
    ret = RsNdaQpCreateEx(&qpCb, &qpInitAttrEx, &info);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
}

void TcRsNdaUbAllocFailed()
{
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    size_t size = 32;
    void* ptr = NULL;

    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = &ndaCb;
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    ndaCbCur->ndaOps.alloc = malloc;
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);

    ndaCbCur->ndaOps.free = free;

    // rs_nda_ut_msg("resource free 1..................\n");
    // mocker(DlDrvMemGetAttribute, 10, 0);
    // ptr = RsNdaUbAlloc(size);
    // EXPECT_ADDR_NE(ptr, NULL);
    // rs_nda_ut_msg("ptr:0x%llx llt\n", (uint64_t)(uintptr_t)ptr);

    // rs_nda_ut_msg("resource free 2..................\n");
    // gRsCb->ndaOps.free = free;
    // RsNdaUbFree(ptr);
    // mocker_clean();
    // rs_nda_ut_msg("resource free 3..................\n");

    mocker_clean();
    mocker(DlDrvMemGetAttribute, 1, -1);
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();

    mocker_invoke(DlDrvMemGetAttribute, DlDrvMemGetAttributeStub, 10);
    mocker(DlHalMemRegUbSegment, 1, -1);
    ptr = RsNdaUbAlloc(size);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaDbMmapHostVa()
{
    struct doorbell_map_desc desc = {0};
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    void* ptr = NULL;
    int ret = 0;

    desc.hva = 1;
    desc.type = DB_MAP_MODE_HOST_VA;
    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = (void*)&ndaCb;

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    RS_INIT_LIST_HEAD(&ndaCbCur->ndaPcieCb.ndaDbList);
    mocker_clean();
    mocker(DlDrvDeviceGetIndexByPhyId, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    // build up NdaDbCb
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker_invoke_p5(
        "DlHalHostRegister", "DlHalHostRegisterStub", (stub_fn_t)DlHalHostRegister, (stub_fn_t)DlHalHostRegisterStub,
        10);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    ptr = NULL;

    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    mocker_clean();

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker_invoke_p5(
        "DlHalHostRegister", "DlHalHostRegisterStub", (stub_fn_t)DlHalHostRegister, (stub_fn_t)DlHalHostRegisterStub,
        10);
    mocker_invoke(DlHalHostUnRegisterEx, DlHalHostUnRegisterExStub, 10);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, -ENODEV);

    mocker(RsGetNdaPcieDbCb, 1, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    desc.type = DB_MAP_MODE_UB_RES + 1;
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaDbMmapUbRes()
{
    struct doorbell_map_desc desc = {0};
    struct RsNdaCb* ndaCbCur = NULL;
    struct RsNdaCb ndaCb = {0};
    void* ptr = NULL;
    int ret = 0;

    desc.ub_res.guid_h = 1;
    desc.ub_res.guid_l = 2;
    desc.ub_res.bits.offset = 3;
    desc.type = DB_MAP_MODE_UB_RES;
    gRsCb = malloc(sizeof(struct rs_cb));
    memset(gRsCb, 0, sizeof(struct rs_cb));
    gRsCb->ndaCb = &ndaCb;

    ndaCbCur = (struct RsNdaCb*)gRsCb->ndaCb;
    RS_INIT_LIST_HEAD(&ndaCbCur->ndaUbCb.ndaDbList);
    mocker_clean();
    mocker(DlDrvDeviceGetIndexByPhyId, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrMapV2, 10, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);
    mocker_clean();
    ptr = NULL;

    // build up NdaGuidCb
    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrMapV2, 10, 0);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    ptr = NULL;

    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_NE(ptr, NULL);
    mocker_clean();

    mocker(DlDrvDeviceGetIndexByPhyId, 10, 0);
    mocker(DlHalResAddrUnmapV2, 10, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, 0);
    mocker_clean();

    ret = RsNdaDbUnmap(ptr, &desc);
    EXPECT_INT_EQ(ret, -ENODEV);

    mocker(RsGetNdaUbDbCb, 1, -1);
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    desc.type = DB_MAP_MODE_UB_RES + 1;
    ptr = RsNdaDbMmap(&desc);
    EXPECT_ADDR_EQ(ptr, NULL);

    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaCqCreateFailed()
{
    struct NdaCqInitAttr attr = {0};
    struct NdaCqInfo info = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    void* ibvCqExt = NULL;
    int ret = 0;

    struct RsNdaCb ndaCb = {0};
    struct NdaOps ops = {0};

    gRsCb = malloc(sizeof(struct rs_cb));
    attr.ops = &ops;
    gRsCb->ndaCb = &ndaCb;

    ret = RsNdaCqCreate(phyId, rdevIndex, NULL, NULL, &ibvCqExt);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_MAX;
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -EINVAL);

    attr.dmaMode = QBUF_DMA_MODE_INDEP_UB;
    mocker_clean();
    mocker(RsQueryRdevCb, 1, -1);
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();

    mocker_invoke(RsQueryRdevCb, RsQueryRdevCbNdaStub, 1);
    mocker(RsNdaCqCreateEx, 1, -1);
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, -1);
    mocker_clean();
    free(gRsCb);
    gRsCb = NULL;
}

void TcRsNdaCqCreate()
{
    struct NdaCqInitAttr attr = {0};
    struct RsInitConfig cfg = {0};
    struct NdaCqInfo info = {0};
    struct rdev rdevInfo = {0};
    unsigned int rdevIndex = 0;
    unsigned int phyId = 0;
    void* ibvCqExt = NULL;
    int ret = 0;

    struct NdaOps ops = {0};

    rdevInfo.phyId = 0;
    rdevInfo.family = AF_INET;
    rdevInfo.localIp.addr.s_addr = inet_addr("127.0.0.1");

    rs_nda_ut_msg("resource prepare begin..................\n");
    /* +++++Resource Prepare+++++ */
    cfg.chipId = 0;
    cfg.hccpMode = NETWORK_OFFLINE;
    ret = RsInit(&cfg);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevInit(rdevInfo, NO_USE, &rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    attr.ops = &ops;
    ret = RsNdaCqCreate(phyId, rdevIndex, &attr, &info, &ibvCqExt);
    EXPECT_INT_EQ(ret, 0);

    ret = RsNdaCqDestroy(phyId, rdevIndex, ibvCqExt);
    EXPECT_INT_EQ(ret, 0);

    ret = RsRdevDeinit(phyId, NO_USE, rdevIndex);
    EXPECT_INT_EQ(ret, 0);

    ret = RsDeinit(&cfg);
    EXPECT_INT_EQ(ret, 0);
}
