/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((puber)->log)
#define BKF_MOD_NAME ((puber)->name)

#include "bkf_puber.h"
#include "bkf_msg.h"
#include "bkf_str.h"
#include "bkf_puber_data.h"
#include "bkf_puber_disp.h"
#include "bkf_puber_table_type.h"
#include "bkf_puber_conn.h"
#include "securec.h"
#include "v_stringlib.h"

#ifdef __cplusplus
extern "C" {
#endif

STATIC uint32_t BkfPuberInitChkArg(BkfPuberInitArg *arg)
{
    BOOL argIsInvalid = (arg == VOS_NULL) || (arg->name == VOS_NULL) || (arg->svcName == VOS_NULL) ||
                        (arg->memMng == VOS_NULL) || (arg->disp == VOS_NULL) || (arg->chMng == VOS_NULL) ||
                        (arg->pfm == VOS_NULL) || (arg->xMap == VOS_NULL) || (arg->tmrMng == VOS_NULL) ||
                        (arg->jobMng == VOS_NULL) || (arg->verifyMayAccelerate == VOS_NULL) ||
                        (arg->sysLogMng == VOS_NULL) || (arg->dc == VOS_NULL) || (arg->connCntMax <= 0);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    uint32_t strLen = VOS_StrLen(arg->name);
    argIsInvalid = (strLen == 0) || (VOS_StrpBrk(arg->name, " ") != VOS_NULL);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    strLen = VOS_StrLen(arg->svcName);
    argIsInvalid = (strLen == 0) || (strLen > BKF_TLV_NAME_LEN_MAX);
    if (argIsInvalid) {
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC uint32_t BkfPuberInitSelf(BkfPuber *puber, BkfPuberInitArg *arg)
{
    puber->argInit = *arg;
    puber->name = BkfStrNew(arg->memMng, "%s", arg->name);
    if (puber->name == VOS_NULL) {
        return BKF_ERR;
    }

    puber->argInit.name = puber->name;
    puber->log = arg->log;

    puber->svcName = BkfStrNew(arg->memMng, "%s", arg->svcName);
    if (puber->svcName == VOS_NULL) {
        return BKF_ERR;
    }

    puber->argInit.svcName = puber->svcName;
    return BKF_OK;
}

BkfPuber *BkfPuberInit(BkfPuberInitArg *arg)
{
    uint32_t ret = BkfPuberInitChkArg(arg);
    if (ret != BKF_OK) {
        return VOS_NULL;
    }

    uint32_t len = sizeof(BkfPuber);
    BkfPuber *puber = BKF_MALLOC(arg->memMng, len);
    if (puber == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(puber, len, 0, len);
    ret = BkfPuberInitSelf(puber, arg);
    if (ret != BKF_OK) {
        BkfPuberUninit(puber);
        return VOS_NULL;
    }

    puber->tableTypeMng = BkfPuberTableTypeInit(&puber->argInit);
    if (puber->tableTypeMng == VOS_NULL) {
        BkfPuberUninit(puber);
        return VOS_NULL;
    }

    puber->connMng = BkfPuberConnInit(&puber->argInit, puber->tableTypeMng);
    if (puber->connMng == VOS_NULL) {
        BkfPuberUninit(puber);
        return VOS_NULL;
    }

    BkfPuberDispInit(puber);
    ret = BkfBlackBoxRegType(arg->log, BKF_BLACKBOX_TYPE_SESS, BKF_256BYTE);
    if (ret != BKF_OK) {
        BkfPuberUninit(puber);
        return VOS_NULL;
    }
    BKF_LOG_INFO(BKF_LOG_HND, "puber(%#x, %s), init ok\n", BKF_MASK_ADDR(puber), puber->name);
    return puber;
}

void BkfPuberUninit(BkfPuber *puber)
{
    if (puber == VOS_NULL) {
        return;
    }

    BKF_LOG_INFO(BKF_LOG_HND, "puber(%#x, %s), uninit\n", BKF_MASK_ADDR(puber), puber->name);
    BkfBlackBoxDelRegType(puber->argInit.log, BKF_BLACKBOX_TYPE_SESS);
    BkfPuberConnUninit(puber->connMng);
    BkfPuberTableTypeUninit(puber->tableTypeMng);
    BkfPuberDispUninit(puber);
    BkfStrDel(puber->svcName);
    BkfStrDel(puber->name);
    BKF_FREE(puber->argInit.memMng, puber);
}

STATIC uint32_t BkfPuberAttachTableTypeExChkArg(BkfPuber *puber, BkfPuberTableTypeVTbl *vTbl, void *userData,
    uint16_t userDataLen)
{
    BOOL argIsInvalid = (puber == VOS_NULL) || (vTbl == VOS_NULL) || (vTbl->tupleUpdateCode == VOS_NULL) ||
                        (vTbl->tupleDeleteCode == VOS_NULL) || ((userDataLen > 0) && (userData == VOS_NULL));
    if (argIsInvalid) {
        return BKF_ERR;
    }

    BKF_LOG_DEBUG(BKF_LOG_HND, "puber(%#x, %s), tableTypeId(%u, %s)/(%#x, %#x), userData(%#x)/userDataLen(%u)\n",
        BKF_MASK_ADDR(puber), puber->name, vTbl->tableTypeId,
        BkfDcGetTableTypeIdStr(puber->argInit.dc, vTbl->tableTypeId), BKF_MASK_ADDR(vTbl->tupleUpdateCode),
        BKF_MASK_ADDR(vTbl->tupleDeleteCode), BKF_MASK_ADDR(userData), userDataLen);
    return BKF_OK;
}

uint32_t BkfPuberAttachTableTypeEx(BkfPuber *puber, BkfPuberTableTypeVTbl *vTbl, void *userData, uint16_t userDataLen)
{
    uint32_t ret = BkfPuberAttachTableTypeExChkArg(puber, vTbl, userData, userDataLen);
    if (ret != BKF_OK) {
        return ret;
    }

    return BkfPuberTableTypeAttachEx(puber->tableTypeMng, vTbl, userData, userDataLen);
}

void *BkfPuberGetTableTypeUserData(BkfPuber *puber, uint16_t tableTypeId)
{
    if (puber == VOS_NULL) {
        return VOS_NULL;
    }

    return BkfPuberTableTypeGetUserData(puber->tableTypeMng, tableTypeId);
}

void BkfPuberDeattachTableType(BkfPuber *puber, uint16_t tabTypeId)
{
    if (puber == VOS_NULL) {
        return;
    }
    BkfPuberTableTypeDeattach(puber->tableTypeMng, tabTypeId);
}

uint32_t BkfPuberEnable(BkfPuber *puber)
{
    if (puber == VOS_NULL) {
        return BKF_ERR;
    }

    uint32_t ret = BkfPuberConnEnable(puber->connMng);
    BKF_LOG_DEBUG(BKF_LOG_HND, "puber(%#x, %s), ret(%u)\n", BKF_MASK_ADDR(puber), puber->name, ret);
    return ret;
}

uint32_t BkfPuberSetSelfSvcInstId(BkfPuber *puber, uint32_t instId)
{
    if (puber == VOS_NULL) {
        return BKF_ERR;
    }

    puber->selfInstId = instId;
    puber->selfInstIdHasSet = VOS_TRUE;
    BKF_LOG_DEBUG(BKF_LOG_HND, "puber(%#x, %s)\n", BKF_MASK_ADDR(puber), puber->name);
    return BKF_OK;
}

STATIC uint32_t BkfPuberSetUnsetUrlChkArg(BkfPuber *puber, BkfUrl *url, BOOL isSet)
{
    if (puber == VOS_NULL || url == VOS_NULL) {
        return BKF_ERR;
    }

    uint8_t buf[BKF_LOG_LEN] = {0};
    BKF_LOG_DEBUG(BKF_LOG_HND, "puber(%#x, %s), url(%s), isSet(%u)\n", BKF_MASK_ADDR(puber), puber->name,
        BkfUrlGetStr(url, buf, sizeof(buf)), isSet);
    return BKF_OK;
}

uint32_t BkfPuberSetSelfUrl(BkfPuber *puber, BkfUrl *url)
{
    uint32_t ret = BkfPuberSetUnsetUrlChkArg(puber, url, VOS_TRUE);
    if (ret != BKF_OK) {
        return ret;
    }

    return BkfPuberConnSetSelfUrl(puber->connMng, url);
}

void BkfPuberUnsetSelfUrl(BkfPuber *puber, BkfUrl *url)
{
    uint32_t ret = BkfPuberSetUnsetUrlChkArg(puber, url, VOS_FALSE);
    if (ret != BKF_OK) {
        return;
    }

    BkfPuberConnUnsetSelfUrl(puber->connMng, url);
}

uint32_t BkfPuberSetConnUpLimit(BkfPuber *puber, uint32_t connMax)
{
    if (puber == VOS_NULL) {
        return BKF_ERR;
    }
    puber->argInit.connCntMax = (int32_t)connMax;
    return BKF_OK;
}
#ifdef __cplusplus
}
#endif
