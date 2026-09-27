/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <urma_types.h>
#include "securec.h"
#include "user_log.h"
#include "dl_urma_function.h"
#include "hccp_ctx.h"
#include "ra_rs_err.h"
#include "rs_ctx_inner.h"
#include "rs_ub_tp.h"

int RsUbGetTpInfoList(struct RsUbDevCb *devCb, struct GetTpCfg *cfg, struct HccpTpInfo infoList[], unsigned int *num)
{
    urma_tp_info_t tpList[HCCP_MAX_TPID_INFO_NUM] = {0};
    urma_get_tp_cfg_t udmaCfg = {0};
    unsigned int tpCnt = *num;
    unsigned int i;
    int ret = 0;

    udmaCfg.flag.value = cfg->flag.value;
    udmaCfg.trans_mode = (urma_transport_mode_t)cfg->transMode;
    (void)memcpy_s(udmaCfg.local_eid.raw, sizeof(udmaCfg.local_eid.raw), cfg->localEid.raw, sizeof(cfg->localEid.raw));
    (void)memcpy_s(udmaCfg.peer_eid.raw, sizeof(udmaCfg.peer_eid.raw), cfg->peerEid.raw, sizeof(cfg->peerEid.raw));

    ret = RsUrmaGetTpList(devCb->urmaCtx, &udmaCfg, &tpCnt, tpList);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("[rs_ub_ctx]rs_urma_get_tp_list failed, ret[%d] transMode:%u flag:0x%x "
                 "tpCnt:%u localEid:%016llx:%016llx peerEid:%016llx:%016llx",
            ret, cfg->transMode, cfg->flag.value, tpCnt, (unsigned long long)be64toh(cfg->localEid.in6.subnetPrefix),
            (unsigned long long)be64toh(cfg->localEid.in6.interfaceId),
            (unsigned long long)be64toh(cfg->peerEid.in6.subnetPrefix),
            (unsigned long long)be64toh(cfg->peerEid.in6.interfaceId)),
        -EOPENSRC);

    *num = (tpCnt > *num) ? *num : tpCnt;
    for (i = 0; i < *num; ++i) {
        infoList[i].tpHandle = tpList[i].tp_handle;
    }

    return ret;
}

STATIC uint8_t RsGetBitmapCount(unsigned int attrBitmap)
{
    unsigned int bitmap = attrBitmap;
    uint8_t bitmapCnt = 0;

    while (bitmap != 0) {
        bitmap &= (bitmap - 1); // clear the last digit 1
        bitmapCnt++;
    }
    return bitmapCnt;
}

int RsUbGetTpAttr(struct RsUbDevCb *devCb, unsigned int *attrBitmap, const uint64_t tpHandle, struct TpAttr *attr)
{
    uint8_t tpAttrCnt = 0;
    int ret;

    tpAttrCnt = RsGetBitmapCount(*attrBitmap);
    ret = RsUrmaGetTpAttr(devCb->urmaCtx, tpHandle, &tpAttrCnt, attrBitmap, (urma_tp_attr_value_t *)attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_get_tp_attr failed, attrBitmap:%u ret:%d errno:%d", *attrBitmap, ret, errno), ret);

    return ret;
}

int RsUbSetTpAttr(struct RsUbDevCb *devCb, const unsigned int attrBitmap, const uint64_t tpHandle, struct TpAttr *attr)
{
    urma_net_addr_t dip = {0};
    urma_eid_t dEid = {0};
    uint8_t tpAttrCnt = 0;
    int ret;

    if ((attrBitmap & TP_ATTR_SIP_MASK) && (attrBitmap & TP_ATTR_SMAC_MASK)) {
        ret = RsUrmaGetSmac(devCb->urmaCtx, attr->sma);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("RsUrmaGetSmac failed, attrBitmap:%u ret:%d errno:%d", attrBitmap, ret, errno), -EOPENSRC);
    }

    if ((attrBitmap & TP_ATTR_DIP_MASK) && (attrBitmap & TP_ATTR_DMAC_MASK)) {
        (void)memcpy_s(&dEid.raw, sizeof(urma_eid_t), attr->dip, sizeof(urma_eid_t));
        dip.sin_family = AF_INET;
        dip.in4.s_addr = dEid.in4.addr;
        ret = RsUrmaGetDmac(devCb->urmaCtx, &dip, attr->dma);
        CHK_PRT_RETURN(ret != 0,
            hccp_err("RsUrmaGetDmac failed, attrBitmap:%u ret:%d errno:%d", attrBitmap, ret, errno), -EOPENSRC);
    }

    tpAttrCnt = RsGetBitmapCount(attrBitmap);
    ret = RsUrmaSetTpAttr(devCb->urmaCtx, tpHandle, tpAttrCnt, attrBitmap, (urma_tp_attr_value_t *)attr);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_set_tp_attr failed, attrBitmap:%u ret:%d errno:%d", attrBitmap, ret, errno), ret);

    return ret;
}
