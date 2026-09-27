/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_NDA_H
#define RS_NDA_H

#include "ibv_extend.h"
#include "hccp_nda.h"
#include "rs_rdma_inner.h"
#include "rs_list.h"
#include "rs.h"

struct RsNdaPcieCb {
    struct ibv_extend_ops ibvExOps;
    struct RsListHead ndaDbList;
};

struct RsNdaUbCb {
    struct ibv_extend_ops ibvExOps;
    struct RsListHead ndaDbList;
    uint16_t ndaDbGuidCnt;
};

struct RsNdaCb {
    struct NdaOps ndaOps;
    struct RsNdaPcieCb ndaPcieCb;
    struct RsNdaUbCb ndaUbCb;
};

struct NdaPcieDbCb {
    uint64_t hva;
    uint64_t dva;
    uint32_t refCnt;
    uint32_t resv;

    struct RsListHead list;
};

struct NdaUbDbCb {
    uint64_t guidL;
    uint64_t guidH;
    uint16_t guidIdx;
    uint16_t resv;
    uint32_t refCnt;

    struct RsListHead list;
};

union DbUbResInfo {
    struct {
        uint32_t dbIdx : 12;
        uint32_t guidIdx : 16;
        uint32_t resv : 4;
    };
    uint32_t resId;
};

struct NdaUbResMapPrivInfo {
    unsigned long long guid_l;
    unsigned long long guid_h;
    unsigned int db_idx;
    unsigned int db_num;
    unsigned int resv[8];
};

static inline unsigned int RsNdaGenerateResId(uint32_t dbIdx, uint16_t guidIdx)
{
    union DbUbResInfo resInfo = {0};

    resInfo.dbIdx = dbIdx;
    resInfo.guidIdx = guidIdx;
    return resInfo.resId;
}

static inline uint64_t AlignDown(uint64_t val, uint64_t align)
{
    return (val) & ~(align - 1);
}

static inline uint64_t AlignUp(uint64_t val, uint64_t align)
{
    return (val + align - 1) & ~(align - 1);
}

int RsInitNdaCb(struct RsRdevCb *rdevCb);
void RsDeinitNdaCb(struct RsRdevCb *rdevCb);
RS_ATTRI_VISI_DEF int RsNdaGetDirectFlag(unsigned int phyId, unsigned int rdevIndex, int *directFlag);
RS_ATTRI_VISI_DEF int RsNdaCqCreate(unsigned int phyId, unsigned int rdevIndex, struct NdaCqInitAttr *attr,
    struct NdaCqInfo *info, void **ibvCqExt);
RS_ATTRI_VISI_DEF int RsNdaCqDestroy(unsigned int phyId, unsigned int rdevIndex, void *ibvCqExt);
RS_ATTRI_VISI_DEF int RsNdaQpCreate(unsigned int phyId, unsigned int rdevIndex, struct NdaQpInitAttr *attr,
    struct NdaQpInfo *info, struct RsQpResp *qpResp);
RS_ATTRI_VISI_DEF int RsNdaQpDestroy(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn);
RS_ATTRI_VISI_DEF int RsGetQpHyperFeature(unsigned int phyId, unsigned int rdevIndex, unsigned int qpn,
    struct HyperFeature *hyperFeature);
#endif // RS_NDA_H
