/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include <stdint.h>
#include <errno.h>
#include "securec.h"
#include "config_log.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "dl_ibv_extend_function.h"
#include "hccp_nda.h"
#include "ra_rs_err.h"
#include "rs.h"
#include "rs_inner.h"
#include "rs_drv_rdma.h"
#include "rs_rdma.h"
#include "rs_nda.h"

RS_ATTRI_VISI_DEF int RsNdaGetDirectFlag(unsigned int phyId, unsigned int rdevIndex, int *directFlag)
{
    struct ibv_device_attr_extend extDevAttr = {0};
    struct RsRdevCb *rdevCb = NULL;
    int ret = 0;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQueryRdevCb phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret), ret);

    if (rdevCb->ibCtxEx == NULL) {
        hccp_warn_rma("ibCtxEx is null, phyId:%u rdevIndex:%u", phyId, rdevIndex);
        *directFlag = DIRECT_FLAG_NOTSUPP;
        return ret;
    }

    ret = RsIbvQueryDeviceExtend(rdevCb->ibCtxEx, &extDevAttr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsIbvQueryDeviceExtend failed, phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret), ret);

    if ((extDevAttr.ext_cap & IBV_EXTEND_DEV_NDA) == 0) {
        hccp_warn_rma("dev not support NDA, phyId:%u rdevIndex:%u ext_cap:0x%x", phyId, rdevIndex, extDevAttr.ext_cap);
        *directFlag = DIRECT_FLAG_NOTSUPP;
        return ret;
    }

    *directFlag = rdevCb->directFlag;
    return ret;
}

STATIC int RsNdaGetDirectFlagByDevAttr(struct ibv_device_attr *deviceAttr)
{
    struct {
        uint32_t vendorId;
        uint32_t vendorPartID;
    } vendorInfoList[] = {
        {0x19E5U, 0x0230U},
        {0x19E5U, 0x0231U},
        {0x19E5U, 0x0232U},
        {0xCC08U, 0x8200U},
        {0xCC08U, 0x8201U},
    };
    unsigned int listSize = sizeof(vendorInfoList) / sizeof(vendorInfoList[0]);
    unsigned int i;

    for (i = 0; i < listSize; i++) {
        if ((deviceAttr->vendor_id == vendorInfoList[i].vendorId) &&
            (deviceAttr->vendor_part_id == vendorInfoList[i].vendorPartID)) {
            return DIRECT_FLAG_UB;
        }
    }

    return DIRECT_FLAG_PCIE;
}

STATIC int RsGetNdaPcieDbCb(struct RsNdaCb *ndaCb, uint64_t hva, struct NdaPcieDbCb **ndaDbCb)
{
    struct NdaPcieDbCb *dbCbCurr = NULL;
    struct NdaPcieDbCb *dbCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(dbCbCurr, dbCbNext, &ndaCb->ndaPcieCb.ndaDbList, list, struct NdaPcieDbCb);
    for (; (&dbCbCurr->list) != &ndaCb->ndaPcieCb.ndaDbList;
         dbCbCurr = dbCbNext, dbCbNext = list_entry(dbCbNext->list.next, struct NdaPcieDbCb, list)) {
        if (dbCbCurr->hva == hva) {
            *ndaDbCb = dbCbCurr;
            return 0;
        }
    }

    *ndaDbCb = NULL;
    hccp_info_rma("ndaDbCb for hva:0x%llx does not exist", hva);
    return -ENODEV;
}

STATIC int RsGetNdaUbDbCb(struct RsNdaCb *ndaCb, uint64_t guidL, uint64_t guidH, struct NdaUbDbCb **ndaDbCb)
{
    struct NdaUbDbCb *dbCbCurr = NULL;
    struct NdaUbDbCb *dbCbNext = NULL;

    RS_LIST_GET_HEAD_ENTRY(dbCbCurr, dbCbNext, &ndaCb->ndaUbCb.ndaDbList, list, struct NdaUbDbCb);
    for (; (&dbCbCurr->list) != &ndaCb->ndaUbCb.ndaDbList;
         dbCbCurr = dbCbNext, dbCbNext = list_entry(dbCbNext->list.next, struct NdaUbDbCb, list)) {
        if (dbCbCurr->guidL == guidL && dbCbCurr->guidH == guidH) {
            *ndaDbCb = dbCbCurr;
            return 0;
        }
    }

    *ndaDbCb = NULL;
    hccp_info_rma("ndaDbCb for guidL:0x%llx guidH:0x%llx does not exist", guidL, guidH);
    return -ENODEV;
}

STATIC void *RsNdaPcieAlloc(size_t size)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;
    void *ptr = NULL;

    CHK_PRT_RETURN(ndaCb == NULL || ndaCb->ndaOps.alloc == NULL,
        hccp_err("ndaCb or ndaOps.alloc is NULL, chipId:%u", gRsCb->chipId), NULL);

    ptr = ndaCb->ndaOps.alloc(size);
    CHK_PRT_RETURN(ptr == NULL, hccp_err("ptr alloc failed"), NULL);

    return ptr;
}

STATIC void RsNdaPcieFree(void *ptr)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;

    if (ndaCb == NULL || ndaCb->ndaOps.free == NULL) {
        hccp_err("gRsCb->ndaCb or ndaOps.free is NULL, chipId:%u", gRsCb->chipId);
        return;
    }

    ndaCb->ndaOps.free(ptr);
    ptr = NULL;
}

STATIC void *RsNdaUbAlloc(size_t size)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;
    struct DVattribute attr = {0};
    void *ptr = NULL;
    int ret = 0;

    CHK_PRT_RETURN(ndaCb == NULL || ndaCb->ndaOps.alloc == NULL || ndaCb->ndaOps.free == NULL,
        hccp_err("ndaCb or alloc or free is NULL, chipId:%u", gRsCb->chipId), NULL);

    ptr = ndaCb->ndaOps.alloc(size);
    CHK_PRT_RETURN(ptr == NULL, hccp_err("ptr alloc failed"), NULL);

    ret = DlDrvMemGetAttribute((uint64_t)(uintptr_t)ptr, &attr);
    if (ret != 0) {
        hccp_err("DlDrvMemGetAttribute failed, ret:%d", ret);
        goto free_ptr;
    }

    if (attr.memType == DV_MEM_LOCK_DEV) {
        ret = DlHalMemRegUbSegment(attr.devId, (uint64_t)(uintptr_t)ptr, size);
        if (ret != 0) {
            hccp_err("DlHalMemRegUbSegment failed, ret:%d devId:%u size:%zu", ret, attr.devId, size);
            goto free_ptr;
        }
    }

    return ptr;

free_ptr:
    ndaCb->ndaOps.free(ptr);
    ptr = NULL;
    return NULL;
}

STATIC void RsNdaUbFree(void *ptr)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;
    struct DVattribute attr = {0};
    int ret = 0;

    if (ndaCb == NULL || ndaCb->ndaOps.free == NULL) {
        hccp_err("gRsCb->ndaCb or ndaOps.free is NULL, chipId:%u", gRsCb->chipId);
        return;
    }

    ret = DlDrvMemGetAttribute((uint64_t)(uintptr_t)ptr, &attr);
    if (ret != 0) {
        hccp_err("DlDrvMemGetAttribute failed, ret:%d", ret);
        goto free_ptr;
    }

    if (attr.memType == DV_MEM_LOCK_DEV) {
        (void)DlHalMemUnRegUbSegment(attr.devId, (uint64_t)(uintptr_t)ptr);
    }

free_ptr:
    ndaCb->ndaOps.free(ptr);
    ptr = NULL;
}

STATIC void RsNdaMemset(void *dst, int value, size_t count)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;

    if (ndaCb == NULL || ndaCb->ndaOps.memset_s == NULL) {
        hccp_err("gRsCb->ndaCb or ndaOps.memset_s is NULL, chipId:%u", gRsCb->chipId);
        return;
    }

    ndaCb->ndaOps.memset_s(dst, value, count);
}

STATIC int RsNdaMemcpy(void *dst, size_t dstSize, void *src, size_t srcSize, uint32_t direct)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;

    if (ndaCb == NULL || ndaCb->ndaOps.memcpy_s == NULL) {
        hccp_err("gRsCb->ndaCb or ndaOps.memcpy_s is NULL, chipId:%u", gRsCb->chipId);
        return -ENODEV;
    }

    return ndaCb->ndaOps.memcpy_s(dst, dstSize, src, srcSize, direct);
}

STATIC void RsNdaFreeUbDbCb(struct NdaUbDbCb *ndaDbCb)
{
    if (ndaDbCb->refCnt != 0) {
        return;
    }

    RsListDel(&ndaDbCb->list);
    free(ndaDbCb);
    ndaDbCb = NULL;
}

STATIC void *RsNdaDbMmapHostVa(struct RsNdaCb *ndaCb, struct doorbell_map_desc *desc)
{
    uint64_t alignHva = AlignDown(desc->hva, (uint64_t)RA_RS_4K_PAGE_SIZE);
    uint64_t alignSize = AlignUp(desc->size, (uint64_t)RA_RS_4K_PAGE_SIZE);
    uint64_t offset = desc->hva - alignHva;
    unsigned int logicId = gRsCb->logicId;
    struct NdaPcieDbCb *ndaDbCb = NULL;
    void *dbDva = NULL;
    int ret = 0;

    ret = RsGetNdaPcieDbCb(ndaCb, alignHva, &ndaDbCb);
    if (ret == 0) {
        ndaDbCb->refCnt++;
        return (void *)(uintptr_t)(ndaDbCb->dva + offset);
    }

    ndaDbCb = (struct NdaPcieDbCb *)calloc(1, sizeof(struct NdaPcieDbCb));
    CHK_PRT_RETURN(ndaDbCb == NULL, hccp_err("ndaDbCb calloc failed"), NULL);

    ret = DlHalHostRegister((void *)(uintptr_t)alignHva, alignSize, HOST_IO_MAP_DEV, logicId, &dbDva);
    if (ret != 0) {
        hccp_err("register host failed, chipId:%u logicId:%u ret:%d alignHva:0x%llx", gRsCb->chipId, logicId, ret,
            alignHva);
        free(ndaDbCb);
        ndaDbCb = NULL;
        return NULL;
    }

    ndaDbCb->hva = alignHva;
    ndaDbCb->dva = (uint64_t)(uintptr_t)dbDva;
    ndaDbCb->refCnt++;
    RsListAddTail(&ndaDbCb->list, &ndaCb->ndaPcieCb.ndaDbList);
    return (void *)(uintptr_t)(ndaDbCb->dva + offset);
}

STATIC void RsNdaMapPrivPrepare(struct doorbell_map_desc *desc, struct NdaUbResMapPrivInfo *resMapIn)
{
    resMapIn->guid_l = desc->ub_res.guid_l;
    resMapIn->guid_h = desc->ub_res.guid_h;
    resMapIn->db_idx = desc->ub_res.bits.offset / (uint64_t)RA_RS_4K_PAGE_SIZE;
    resMapIn->db_num = 1;
    return;
}

STATIC void *RsNdaDbMmapUbRes(struct RsNdaCb *ndaCb, struct doorbell_map_desc *desc)
{
    struct NdaUbResMapPrivInfo resMapIn = {0};
    struct res_map_info_out resInfoOut = {0};
    struct res_map_info_in resInfoIn = {0};
    unsigned int logicId = gRsCb->logicId;
    struct NdaUbDbCb *ndaDbCb = NULL;
    uint64_t dva = 0;
    int ret = 0;

    ret = RsGetNdaUbDbCb(ndaCb, desc->ub_res.guid_l, desc->ub_res.guid_h, &ndaDbCb);
    if (ret == 0) {
        goto map_db;
    }

    ndaDbCb = (struct NdaUbDbCb *)calloc(1, sizeof(struct NdaUbDbCb));
    CHK_PRT_RETURN(ndaDbCb == NULL, hccp_err("ndaDbCb calloc failed"), NULL);
    ndaDbCb->guidL = desc->ub_res.guid_l;
    ndaDbCb->guidH = desc->ub_res.guid_h;
    ndaDbCb->guidIdx = ndaCb->ndaUbCb.ndaDbGuidCnt;
    RsListAddTail(&ndaDbCb->list, &ndaCb->ndaUbCb.ndaDbList);
    ndaCb->ndaUbCb.ndaDbGuidCnt++;

map_db:
    RsNdaMapPrivPrepare(desc, &resMapIn);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = RES_ADDR_TYPE_NDA_URMA_DB;
    resInfoIn.res_id = RsNdaGenerateResId(resMapIn.db_idx, ndaDbCb->guidIdx);
    resInfoIn.priv_len = sizeof(struct NdaUbResMapPrivInfo);
    resInfoIn.priv = (void *)&resMapIn;
    ret = DlHalResAddrMapV2(logicId, &resInfoIn, &resInfoOut);
    if (ret != 0) {
        hccp_err("DlHalResAddrMapV2 failed, chipId:%u logicId:%u resId:0x%x ret:%d", gRsCb->chipId, logicId,
            resInfoIn.res_id, ret);
        RsNdaFreeUbDbCb(ndaDbCb);
        return NULL;
    }

    ndaDbCb->refCnt++;
    dva = resInfoOut.va + (desc->ub_res.bits.offset % (uint64_t)RA_RS_4K_PAGE_SIZE);
    return (void *)(uintptr_t)dva;
}

STATIC void *RsNdaDbMmap(struct doorbell_map_desc *desc)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;

    CHK_PRT_RETURN(desc == NULL, hccp_err("desc is null"), NULL);
    CHK_PRT_RETURN(ndaCb == NULL, hccp_err("ndaCb is null, chipId:%u", gRsCb->chipId), NULL);

    if (desc->type == DB_MAP_MODE_HOST_VA) {
        return RsNdaDbMmapHostVa(ndaCb, desc);
    } else if (desc->type == DB_MAP_MODE_UB_RES) {
        return RsNdaDbMmapUbRes(ndaCb, desc);
    } else {
        hccp_err("invalid desc->type:%u, chipId:%u", desc->type, gRsCb->chipId);
        return NULL;
    }
}

STATIC int RsNdaDbUnmapHostVa(struct RsNdaCb *ndaCb, void *ptr, struct doorbell_map_desc *desc)
{
    (void)ptr;
    uint64_t alignHva = AlignDown(desc->hva, (uint64_t)RA_RS_4K_PAGE_SIZE);
    unsigned int logicId = gRsCb->logicId;
    struct NdaPcieDbCb *ndaDbCb = NULL;
    int ret = 0;

    ret = RsGetNdaPcieDbCb(ndaCb, alignHva, &ndaDbCb);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsGetNdaPcieDbCb failed, hva:0x%llx chipId:%u ret:%d", alignHva, gRsCb->chipId, ret), ret);

    ndaDbCb->refCnt--;
    if (ndaDbCb->refCnt != 0) {
        return ret;
    }

    ret = DlHalHostUnRegisterEx((void *)(uintptr_t)alignHva, logicId, HOST_IO_MAP_DEV);
    if (ret != 0) {
        hccp_err("DlHalHostUnRegisterEx failed, chipId:%u logicId:%u ret:%d", gRsCb->chipId, logicId, ret);
    }

    RsListDel(&ndaDbCb->list);
    free(ndaDbCb);
    ndaDbCb = NULL;
    return ret;
}

STATIC int RsNdaDbUnmapUbRes(struct RsNdaCb *ndaCb, void *ptr, struct doorbell_map_desc *desc)
{
    (void)ptr;
    struct NdaUbResMapPrivInfo resMapIn = {0};
    struct res_map_info_in resInfoIn = {0};
    unsigned int logicId = gRsCb->logicId;
    struct NdaUbDbCb *ndaDbCb = NULL;
    int ret = 0;

    ret = RsGetNdaUbDbCb(ndaCb, desc->ub_res.guid_l, desc->ub_res.guid_h, &ndaDbCb);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsGetNdaUbDbCb failed, chipId:%u guidL:0x%llx guidH:0x%llx", gRsCb->chipId, desc->ub_res.guid_l,
            desc->ub_res.guid_h),
        ret);

    RsNdaMapPrivPrepare(desc, &resMapIn);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = RES_ADDR_TYPE_NDA_URMA_DB;
    resInfoIn.res_id = RsNdaGenerateResId(resMapIn.db_idx, ndaDbCb->guidIdx);
    resInfoIn.priv_len = sizeof(struct NdaUbResMapPrivInfo);
    resInfoIn.priv = (void *)&resMapIn;
    ret = DlHalResAddrUnmapV2(logicId, &resInfoIn);
    if (ret != 0) {
        hccp_err("DlHalResAddrUnmapV2 failed, chipId:%u logicId:%u resId:0x%x ret:%d", gRsCb->chipId, logicId,
            resInfoIn.res_id, ret);
    }

    ndaDbCb->refCnt--;
    RsNdaFreeUbDbCb(ndaDbCb);
    return ret;
}

STATIC int RsNdaDbUnmap(void *ptr, struct doorbell_map_desc *desc)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)gRsCb->ndaCb;

    CHK_PRT_RETURN(ptr == NULL || desc == NULL, hccp_err("ptr or desc is null"), -EINVAL);
    CHK_PRT_RETURN(ndaCb == NULL, hccp_err("ndaCb is null, chipId:%u", gRsCb->chipId), -ENODEV);

    if (desc->type == DB_MAP_MODE_HOST_VA) {
        return RsNdaDbUnmapHostVa(ndaCb, ptr, desc);
    } else if (desc->type == DB_MAP_MODE_UB_RES) {
        return RsNdaDbUnmapUbRes(ndaCb, ptr, desc);
    } else {
        hccp_err("invalid desc->type:%u, chipId:%u", desc->type, gRsCb->chipId);
        return -EINVAL;
    }
}

STATIC void RsNdaCbInitCb(struct RsNdaCb *ndaCb)
{
    ndaCb->ndaPcieCb.ibvExOps.alloc = RsNdaPcieAlloc;
    ndaCb->ndaPcieCb.ibvExOps.free = RsNdaPcieFree;
    ndaCb->ndaPcieCb.ibvExOps.db_mmap = RsNdaDbMmap;
    ndaCb->ndaPcieCb.ibvExOps.db_unmap = RsNdaDbUnmap;
    ndaCb->ndaPcieCb.ibvExOps.memset_s = RsNdaMemset;
    ndaCb->ndaPcieCb.ibvExOps.memcpy_s = RsNdaMemcpy;

    ndaCb->ndaUbCb.ibvExOps.alloc = RsNdaUbAlloc;
    ndaCb->ndaUbCb.ibvExOps.free = RsNdaUbFree;
    ndaCb->ndaUbCb.ibvExOps.db_mmap = RsNdaDbMmap;
    ndaCb->ndaUbCb.ibvExOps.db_unmap = RsNdaDbUnmap;
    ndaCb->ndaUbCb.ibvExOps.memset_s = RsNdaMemset;
    ndaCb->ndaUbCb.ibvExOps.memcpy_s = RsNdaMemcpy;

    RS_INIT_LIST_HEAD(&ndaCb->ndaPcieCb.ndaDbList);
    RS_INIT_LIST_HEAD(&ndaCb->ndaUbCb.ndaDbList);
}

int RsInitNdaCb(struct RsRdevCb *rdevCb)
{
    struct RsNdaCb *ndaCb = NULL;
    int count = 0;
    int ret = 0;

    rdevCb->directFlag = RsNdaGetDirectFlagByDevAttr(&rdevCb->deviceAttr);

    rdevCb->ibCtxEx = RsIbvOpenExtend(rdevCb->ibCtx);
    if (rdevCb->ibCtxEx == NULL) {
        hccp_warn_rma("ibCtxEx is null");
        return 0;
    }

    count = __sync_fetch_and_add(&rdevCb->rsCb->ndaCbRefCnt, 1);
    if (count > 0) {
        hccp_info_rma("RsNdaCb exist, ndaCbRefCnt:%d", rdevCb->rsCb->ndaCbRefCnt);
        return 0;
    }

    ndaCb = (struct RsNdaCb *)calloc(1, sizeof(struct RsNdaCb));
    if (ndaCb == NULL) {
        hccp_err("calloc for ndaCb failed");
        ret = -ENOMEM;
        goto calloc_err;
    }

    RsNdaCbInitCb(ndaCb);

    rdevCb->rsCb->ndaCb = (void *)ndaCb;
    return ret;

calloc_err:
    (void)__sync_fetch_and_sub(&rdevCb->rsCb->ndaCbRefCnt, 1);
    (void)RsIbvCloseExtend(rdevCb->ibCtxEx);
    rdevCb->ibCtxEx = NULL;
    return ret;
}

void RsDeinitNdaCb(struct RsRdevCb *rdevCb)
{
    if (rdevCb->ibCtxEx == NULL) {
        return;
    }

    (void)RsIbvCloseExtend(rdevCb->ibCtxEx);
    rdevCb->ibCtxEx = NULL;

    if (__sync_fetch_and_sub(&rdevCb->rsCb->ndaCbRefCnt, 1) > 1) {
        return;
    }

    free(rdevCb->rsCb->ndaCb);
    rdevCb->rsCb->ndaCb = NULL;
}

STATIC void RsNdaInitExOps(struct RsNdaCb *ndaCb, uint32_t dmaMode, struct NdaOps *ops, struct ibv_extend_ops **extOps)
{
    ndaCb->ndaOps.alloc = ops->alloc;
    ndaCb->ndaOps.free = ops->free;
    ndaCb->ndaOps.memset_s = ops->memset_s;
    ndaCb->ndaOps.memcpy_s = ops->memcpy_s;
    if (dmaMode == QBUF_DMA_MODE_DEFAULT) {
        *extOps = &ndaCb->ndaPcieCb.ibvExOps;
    } else if (dmaMode == QBUF_DMA_MODE_INDEP_UB) {
        *extOps = &ndaCb->ndaUbCb.ibvExOps;
    } else {
        *extOps = NULL;
    }
}

STATIC void RsNdaCqInitExPrepare(struct NdaCqInitAttr *attr, struct RsNdaCb *ndaCb,
    struct ibv_cq_init_attr_extend *cqInitAttrEx)
{
    (void)memcpy_s(&cqInitAttrEx->attr, sizeof(struct ibv_cq_init_attr_ex), &attr->attr,
        sizeof(struct ibv_cq_init_attr_ex));
    cqInitAttrEx->cq_cap_flag = attr->cqCapFlag;
    cqInitAttrEx->type = attr->dmaMode;
    RsNdaInitExOps(ndaCb, attr->dmaMode, attr->ops, &cqInitAttrEx->ops);
}

STATIC int RsNdaCqCreateEx(struct RsRdevCb *rdevCb, struct ibv_cq_init_attr_extend *cqInitAttrEx,
    struct NdaCqInfo *info, void **ibvCqExt)
{
    struct ibv_cq_extend *cqExt = NULL;
    int ret = 0;

    cqExt = RsIbvCreateCqExtend(rdevCb->ibCtxEx, cqInitAttrEx);
    CHK_PRT_RETURN(cqExt == NULL, hccp_err("RsNdaCreateCqExtend failed, errno:%d", errno), -ENOMEM);

    ret = memcpy_s(&info->cqInfo, sizeof(struct queueInfo), &cqExt->cq_info, sizeof(struct queue_info));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s queue_info failed, ret:%d", ret), -ESAFEFUNC);
    ret = memcpy_s(info->resv, sizeof(info->resv), cqExt->resv, sizeof(cqExt->resv));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s resv failed, ret:%d", ret), -ESAFEFUNC);
    info->cq = cqExt->cq;
    *ibvCqExt = cqExt;
    return 0;
}

RS_ATTRI_VISI_DEF int RsNdaCqCreate(unsigned int phyId, unsigned int rdevIndex, struct NdaCqInitAttr *attr,
    struct NdaCqInfo *info, void **ibvCqExt)
{
    struct ibv_cq_init_attr_extend cqInitAttrEx = {0};
    struct RsRdevCb *rdevCb = NULL;
    struct RsNdaCb *ndaCb = NULL;
    int ret = 0;

    CHK_PRT_RETURN(attr == NULL || info == NULL || ibvCqExt == NULL,
        hccp_err("attr or info or ibvCqExt is NULL, phyId:%u", phyId), -EINVAL);
    CHK_PRT_RETURN(attr->dmaMode >= QBUF_DMA_MODE_MAX,
        hccp_err("param err, dmaMode:%u >= %u, phyId:%u", attr->dmaMode, QBUF_DMA_MODE_MAX, phyId), -EINVAL);

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQueryRdevCb failed, phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret),
        ret);

    ndaCb = (struct RsNdaCb *)rdevCb->rsCb->ndaCb;
    CHK_PRT_RETURN(ndaCb == NULL, hccp_err("ndaCb is NULL, does not support nda, phyId:%u", phyId), -ENODEV);

    RsNdaCqInitExPrepare(attr, ndaCb, &cqInitAttrEx);

    ret = RsNdaCqCreateEx(rdevCb, &cqInitAttrEx, info, ibvCqExt);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsNdaCqCreateEx failed, phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsNdaCqDestroy(unsigned int phyId, unsigned int rdevIndex, void *ibvCqExt)
{
    struct RsRdevCb *rdevCb = NULL;
    int ret = 0;

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQueryRdevCb failed, phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret),
        ret);

    ret = RsIbvDestroyCqExtend(rdevCb->ibCtxEx, ibvCqExt);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("RsIbvDestroyCqExtend failed, phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret), ret);

    return ret;
}

STATIC int RsBuildUpNdaQpcb(struct RsRdevCb *rdevCb, struct ibv_qp_init_attr *qpInitAttr, struct RsQpCb **qpCb)
{
    int ret = 0;

    *qpCb = (struct RsQpCb *)calloc(1, sizeof(struct RsQpCb));
    CHK_PRT_RETURN(*qpCb == NULL, hccp_err("RsQpCb calloc failed"), -ENOMEM);

    ret = pthread_mutex_init(&(*qpCb)->qpMutex, NULL);
    if (ret != 0) {
        hccp_err("pthread_mutex_init failed, ret:%d", ret);
        goto qp_mutex_init_err;
    }

    (*qpCb)->rdevCb = rdevCb;
    RS_INIT_LIST_HEAD(&(*qpCb)->mrList);
    RS_INIT_LIST_HEAD(&(*qpCb)->remMrList);
    (*qpCb)->state = RS_QP_STATUS_DISCONNECT;
    (*qpCb)->ibPd = rdevCb->ibPd;
    (*qpCb)->txDepth = qpInitAttr->cap.max_send_wr;
    (*qpCb)->rxDepth = qpInitAttr->cap.max_recv_wr;

    (*qpCb)->numRecvCqEvents = 0;
    (*qpCb)->numSendCqEvents = 0;
    (*qpCb)->qosAttr.tc = (RS_ROCE_DSCP_33 & RS_DSCP_MASK) << RS_DSCP_OFF;
    (*qpCb)->qosAttr.sl = RS_ROCE_4_SL;
    (*qpCb)->timeout = RS_QP_ATTR_TIMEOUT;
    (*qpCb)->retryCnt = RS_QP_ATTR_RETRY_CNT;

    return ret;

qp_mutex_init_err:
    free(*qpCb);
    *qpCb = NULL;
    return ret;
}

STATIC void RsNdaQpInitExPrepare(struct RsRdevCb *rdevCb, struct NdaQpInitAttr *attr,
    struct ibv_qp_init_attr_extend *qpInitAttrEx)
{
    struct RsNdaCb *ndaCb = (struct RsNdaCb *)rdevCb->rsCb->ndaCb;

    qpInitAttrEx->pd = rdevCb->ibPd;
    (void)memcpy_s(&qpInitAttrEx->attr, sizeof(struct ibv_qp_init_attr), &attr->attr, sizeof(struct ibv_qp_init_attr));
    qpInitAttrEx->qp_cap_flag = attr->qpCapFlag;
    qpInitAttrEx->type = attr->dmaMode;
    RsNdaInitExOps(ndaCb, attr->dmaMode, attr->ops, &qpInitAttrEx->ops);
}

STATIC int RsNdaQpGetQpInfo(struct RsQpCb *qpCb, struct NdaQpInfo *info)
{
    int ret = 0;

    info->qp = qpCb->ibQp;
    ret = memcpy_s(&info->sqInfo, sizeof(struct queueInfo), &qpCb->ibQpEx->sq_info, sizeof(struct queue_info));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s sq_info failed, ret:%d", ret), -ESAFEFUNC);
    ret = memcpy_s(&info->rqInfo, sizeof(struct queueInfo), &qpCb->ibQpEx->rq_info, sizeof(struct queue_info));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s rq_info failed, ret:%d", ret), -ESAFEFUNC);
    ret = memcpy_s(info->resv, sizeof(info->resv), qpCb->ibQpEx->resv, sizeof(qpCb->ibQpEx->resv));
    CHK_PRT_RETURN(ret != 0, hccp_err("memcpy_s resv failed, ret:%d", ret), -ESAFEFUNC);

    return ret;
}

STATIC int RsNdaQpCreateEx(struct RsQpCb *qpCb, struct ibv_qp_init_attr_extend *qpInitAttrEx, struct NdaQpInfo *info)
{
    struct ibv_qp_attr qpAttr = {0};
    struct ibv_port_attr attr = {0};
    struct RsRdevCb *rdevCb = NULL;
    int ret = 0;

    rdevCb = qpCb->rdevCb;
    qpCb->ibQpEx = RsIbvCreateQpExtend(rdevCb->ibCtxEx, qpInitAttrEx);
    CHK_PRT_RETURN(qpCb->ibQpEx == NULL, hccp_err("RsNdaCreateQpExtend failed, errno:%d", errno), -ENOMEM);

    qpCb->ibQp = qpCb->ibQpEx->qp;
    ret = RsIbvQueryQp(qpCb->ibQp, &qpAttr, IBV_QP_CAP, &qpInitAttrEx->attr);
    if (ret != 0) {
        hccp_err("query qp attr failed ret:%d", ret);
        ret = -EOPENSRC;
        goto nda_init_qp_err;
    }

    ret = RsDrvQpInfoRelated(qpCb, rdevCb, &attr, &qpAttr);
    if (ret != 0) {
        hccp_err("qp info related failed ret:%d", ret);
        goto nda_init_qp_err;
    }

    ret = RsNdaQpGetQpInfo(qpCb, info);
    if (ret != 0) {
        hccp_err("RsNdaQpGetQpInfo failed ret:%d", ret);
        goto nda_init_qp_err;
    }

    hccp_info_rma("chip_id:%u, rdevIndex:%u, qp:%d create succ", rdevCb->rsCb->chipId, rdevCb->rdevIndex,
        qpCb->qpInfoLo.qpn);
    return ret;

nda_init_qp_err:
    (void)RsIbvDestroyQpExtend(rdevCb->ibCtxEx, qpCb->ibQpEx);
    qpCb->ibQpEx = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsNdaQpCreate(unsigned int phyId, unsigned int rdevIndex, struct NdaQpInitAttr *attr,
    struct NdaQpInfo *info, struct RsQpResp *qpResp)
{
    struct ibv_qp_init_attr_extend qpInitAttrEx = {0};
    struct RsRdevCb *rdevCb = NULL;
    struct RsNdaCb *ndaCb = NULL;
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    CHK_PRT_RETURN(attr == NULL || info == NULL || qpResp == NULL,
        hccp_err("attr or info or qpResp is NULL, phyId:%u", phyId), -EINVAL);
    CHK_PRT_RETURN(attr->dmaMode >= QBUF_DMA_MODE_MAX,
        hccp_err("param err, dmaMode:%u >= %u, phyId:%u", attr->dmaMode, QBUF_DMA_MODE_MAX, phyId), -EINVAL);

    ret = RsQueryRdevCb(phyId, rdevIndex, &rdevCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsQueryRdevCb phyId:%u rdevIndex:%u ret:%d", phyId, rdevIndex, ret), ret);

    ndaCb = (struct RsNdaCb *)rdevCb->rsCb->ndaCb;
    CHK_PRT_RETURN(ndaCb == NULL, hccp_err("ndaCb is NULL, phyId:%u rdevIndex:%u", phyId, rdevIndex), -ENODEV);

    ret = RsBuildUpNdaQpcb(rdevCb, &attr->attr, &qpCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsBuildUpNdaQpcb failed, ret:%d", ret), ret);

    RsNdaQpInitExPrepare(rdevCb, attr, &qpInitAttrEx);

    ret = RsNdaQpCreateEx(qpCb, &qpInitAttrEx, info);
    if (ret != 0) {
        hccp_err("create nda qp extend failed, ret:%d", ret);
        goto create_qp_err;
    }

    RS_PTHREAD_MUTEX_LOCK(&rdevCb->rdevMutex);
    RsListAddTail(&qpCb->list, &rdevCb->qpList);
    rdevCb->qpCnt++;
    RS_PTHREAD_MUTEX_ULOCK(&rdevCb->rdevMutex);
    qpResp->qpn = (unsigned int)qpCb->qpInfoLo.qpn;
    qpResp->gidIdx = (unsigned int)qpCb->qpInfoLo.gidIdx;
    qpResp->psn = (unsigned int)qpCb->qpInfoLo.psn;
    qpResp->gid = qpCb->qpInfoLo.gid;
    qpResp->directFlag = rdevCb->directFlag;
    return 0;

create_qp_err:
    pthread_mutex_destroy(&qpCb->qpMutex);
    free(qpCb);
    qpCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsNdaQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn)
{
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_err("get qp cb failed, qpn:%u, ret:%d", qpn, ret), ret);

    RS_PTHREAD_MUTEX_LOCK(&qpCb->rdevCb->rdevMutex);
    RsListDel(&qpCb->list);
    qpCb->rdevCb->qpCnt--;
    RS_PTHREAD_MUTEX_ULOCK(&qpCb->rdevCb->rdevMutex);

    RsMrRelease(qpCb);

    ret = RsIbvDestroyQpExtend(qpCb->rdevCb->ibCtxEx, qpCb->ibQpEx);
    if (ret != 0) {
        hccp_err("qp:%u destroy extend failed, ret:%d", qpn, ret);
    }
    qpCb->ibQpEx = NULL;

    pthread_mutex_destroy(&qpCb->qpMutex);

    free(qpCb);
    qpCb = NULL;
    return ret;
}

RS_ATTRI_VISI_DEF int RsGetQpHyperFeature(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct HyperFeature *hyperFeature)
{
    struct ibv_hyroce_feature feature = {0};
    struct RsQpCb *qpCb = NULL;
    int ret = 0;

    CHK_PRT_RETURN(hyperFeature == NULL, hccp_warn_rma("hyperFeature is NULL, phyId:%u", phyId), ret);

    ret = RsQpn2qpcb(phyId, rdevIndex, qpn, &qpCb);
    CHK_PRT_RETURN(ret != 0 || qpCb == NULL, hccp_warn_rma("get qp cb unsuccessful qpn %u, ret %d", qpn, ret), ret);

    if (qpCb->rdevCb->ibCtxEx == NULL) {
        return 0;
    }

    ret = RsIbvQueryQpSupportedHyroceFeature(qpCb->rdevCb->ibCtxEx, qpCb->ibQp, qpCb->qosAttr.sl, qpCb->qosAttr.tc,
        &feature);
    CHK_PRT_RETURN(ret != 0,
        hccp_warn_rma("RsIbvQueryQpSupportedHyroceFeature unsuccessful, qpn %u ret %d errno:%d", qpn, ret, errno), ret);

    ret = memcpy_s(hyperFeature, sizeof(struct HyperFeature), &feature, sizeof(struct ibv_hyroce_feature));
    CHK_PRT_RETURN(ret != 0,
        hccp_warn_rma("memcpy_s feature unsuccessful, qpn:%u ret:%d HyperFeature len:%zu "
                      "ibv_hyroce_feature len:%zu",
            qpn, ret, sizeof(struct HyperFeature), sizeof(struct ibv_hyroce_feature)),
        ret);

    hccp_dbg("get QpHyperFeature successful qpn %u", qpn);
    return ret;
}
