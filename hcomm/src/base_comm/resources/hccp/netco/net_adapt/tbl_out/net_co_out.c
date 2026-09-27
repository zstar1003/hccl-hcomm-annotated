/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_out.h"
#include "net_co_out_data.h"
#include "net_co_main_data.h"
#include "net_co_in_out_comm.h"
#include "net_co_out_tbl_pri.h"
#include "net_co_out_disp.h"
#include "bkf_ch_ser.h"
#include "bkf_ch_ser_letcp.h"
#include "simpo_multi_layer_common_builder.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_PUBER_IDL_VER_MAJOR (0)
#define NET_CO_PUBER_IDL_VER_MINOR (0)
#define NET_CO_PUBER_CONN_CNT_MAX (BKF_1K)

STATIC uint32_t NetCoOutChSerInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    uint32_t nameLen = BKF_NAME_LEN_MAX + 1;
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_out", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfChSerMngInitArg arg = {
        .name = name,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .logCnt = co->logCnt,
        .log = co->log,
        .pfm = co->pfm,
        .xMap = co->xMap,
        .tmrMng = co->tmrMng,
        .jobMng = co->jobMng,
        .mux = co->mux,
    };
    co->out->chSerMng = BkfChSerInit(&arg);
    if (co->out->chSerMng == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "chSerMng(%#x), ng\n", BKF_MASK_ADDR(co->out->chSerMng));
        return BKF_ERR;
    }

    BkfChSerTypeVTbl vTbl = {0};
    uint32_t ret = BkfChSerLetcpBuildVTbl(name, &vTbl, nameLen);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }
    ret = BkfChSerRegType(co->out->chSerMng, &vTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }
    ret = BkfChSerSetSelfCid(co->out->chSerMng, co->selfCid);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoOutChSerUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "chSerMng(%#x)\n", BKF_MASK_ADDR(co->out->chSerMng));

    if (co->out->chSerMng != VOS_NULL) {
        BkfChSerUninit(co->out->chSerMng);
        co->out->chSerMng = VOS_NULL;
    }
}

STATIC uint32_t NetCoOutDcInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_out", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfDcInitArg arg = {.name = name,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .log = co->log,
        .pfm = co->pfm,
        .jobMng = co->jobMng,
        .jobTypeId = NET_CO_JOB_TYPE_OUT_DC,
        .jobPrio = NET_CO_JOB_PRIO_DFT,
        .sysLogMng = co->sysLogMng,
        .sliceVTbl.keyLen = sizeof(BifrostCncoiSliceKeyT),
        .sliceVTbl.keyCmp = (F_BKF_CMP)BifrostCncoiSliceKeyCmp,
        .sliceVTbl.keyGetStrOrNull = (F_BKF_GET_STR)BifrostCncoiSliceKeyGetStr,
        .sliceVTbl.keyCodec = (F_BKF_DO)BifrostCncoiSliceKeyCodec};
    co->out->dc = BkfDcInit(&arg);
    if (co->out->dc == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "dc(%#x), ng\n", BKF_MASK_ADDR(co->out->dc));
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoOutDcUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "dc(%#x)\n", BKF_MASK_ADDR(co->out->dc));

    if (co->out->dc != VOS_NULL) {
        BkfDcUninit(co->out->dc);
        co->out->dc = VOS_NULL;
    }
}

STATIC uint32_t NetCoOutSimpoInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    co->out->simpoBuilder = SimpoBuilderInit();
    if (co->out->simpoBuilder == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "simpoBuilder(%#x), ng\n", BKF_MASK_ADDR(co->out->simpoBuilder));
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoOutSimpoUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "simpoBuilder(%#x)\n", BKF_MASK_ADDR(co->out->simpoBuilder));

    if (co->out->simpoBuilder != VOS_NULL) {
        SimpoBuilderUninit(co->out->simpoBuilder);
        co->out->simpoBuilder = VOS_NULL;
    }
}

#if BKF_BLOCK("puber")
STATIC uint32_t NetCoOutPuberInitChkArg(NetCo *co)
{
    uint8_t buf1[BKF_URL_STR_LEN_MAX];
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "outConnLsnUrl(%s)\n", BkfUrlGetStr(&co->argInit.outConnLsnUrl, buf1, sizeof(buf1)));
    BOOL argIsInvalid = !BkfUrlIsValid(&co->argInit.outConnLsnUrl);
    if (argIsInvalid) {
        BKF_LOG_ERROR(BKF_LOG_HND, "argIsInvalid(%u)\n", argIsInvalid);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC BOOL NetCoOutOnPuberVerifyMayAccelerate(NetCo *co)
{
    BKF_LOG_DEBUG(BKF_LOG_HND, "\n");
    return VOS_TRUE;
}
STATIC void NetCoOutOnPuberConnect(NetCo *co, BkfUrl *connDestUrl, BkfUrl *connSrcUrl, BkfUrl *peerLsnUrl)
{
    uint8_t buf1[BKF_URL_STR_LEN_MAX + 1];
    uint8_t buf2[BKF_URL_STR_LEN_MAX + 1];
    uint8_t buf3[BKF_URL_STR_LEN_MAX + 1];
    BKF_LOG_DEBUG(BKF_LOG_HND, "connDestUrl(%s)/connSrcUrl(%s), peerLsnUrl(%s)\n",
        BkfUrlGetStr(connDestUrl, buf1, sizeof(buf1)), BkfUrlGetStr(connSrcUrl, buf2, sizeof(buf2)),
        BkfUrlGetStr(peerLsnUrl, buf3, sizeof(buf3)));
}
STATIC void NetCoOutOnPuberDisconnect(NetCo *co, BkfUrl *connDestUrl, BkfUrl *connSrcUrl)
{
    uint8_t buf1[BKF_URL_STR_LEN_MAX + 1];
    uint8_t buf2[BKF_URL_STR_LEN_MAX + 1];
    BKF_LOG_DEBUG(BKF_LOG_HND, "connDestUrl(%s)/connSrcUrl(%s)\n", BkfUrlGetStr(connDestUrl, buf1, sizeof(buf1)),
        BkfUrlGetStr(connSrcUrl, buf2, sizeof(buf2)));
}
STATIC uint32_t NetCoOutPuberInitPuber(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int32_t err = snprintf_truncated_s(name, sizeof(name), "%s_out", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BifrostCncoiPuberInitArg arg = {
        .name = name,
        .idlVersionMajor = NET_CO_PUBER_IDL_VER_MAJOR,
        .idlVersionMinor = NET_CO_PUBER_IDL_VER_MINOR,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .log = co->log,
        .pfm = co->pfm,
        .xMap = co->xMap,
        .tmrMng = co->tmrMng,
        .jobMng = co->jobMng,
        .jobTypeId = NET_CO_JOB_TYPE_OUT_PUBER,
        .jobPrio = NET_CO_JOB_PRIO_DFT,
        .cookie = co,
        .verifyMayAccelerate = (F_BKF_PUBER_VERIFY_MAY_ACCELERATE)NetCoOutOnPuberVerifyMayAccelerate,
        .sysLogMng = co->sysLogMng,
        .dc = co->out->dc,
        .chMng = co->out->chSerMng,
        .connCntMax = NET_CO_PUBER_CONN_CNT_MAX,
        .simpoBuilder = co->out->simpoBuilder,
        .onConnect = (F_PUBER_ON_CONNECT)NetCoOutOnPuberConnect,
        .onDisconnect = (F_PUBER_ON_DISCONNECT)NetCoOutOnPuberDisconnect,
    };
    co->out->puber = BifrostCncoiPuberInit(&arg);
    if (co->out->puber == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "puber(%#x), ng\n", BKF_MASK_ADDR(co->out->puber));
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC uint32_t NetCoOutPuberInit(NetCo *co)
{
    uint32_t ret = NetCoOutPuberInitChkArg(co);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }

    ret = NetCoOutPuberInitPuber(co);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberSetSelfSvcInstId(co->out->puber, NET_CO_DFT_INST_ID);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberSetSelfUrl(co->out->puber, &co->argInit.outConnLsnUrl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberCreateSlice(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberEnable(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC void NetCoOutPuberUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "puber(%#x)\n", BKF_MASK_ADDR(co->out->puber));

    if (co->out->puber != VOS_NULL) {
        BifrostCncoiPuberUninit(co->out->puber);
        co->out->puber = VOS_NULL;
    }
}
#endif

STATIC const BkfModVTbl g_NetCoOutModVTbl[] = {
    {(F_BKF_DO)NetCoOutDataNew, (F_BKF_DOV)NetCoOutDataDelete},
    {(F_BKF_DO)NetCoOutChSerInit, (F_BKF_DOV)NetCoOutChSerUninit},
    {(F_BKF_DO)NetCoOutDcInit, (F_BKF_DOV)NetCoOutDcUninit},
    {(F_BKF_DO)NetCoOutSimpoInit, (F_BKF_DOV)NetCoOutSimpoUninit},
    {(F_BKF_DO)NetCoOutPuberInit, (F_BKF_DOV)NetCoOutPuberUninit},
    {(F_BKF_DO)NetCoOutTblInit, (F_BKF_DOV)NetCoOutTblUninit},
    {(F_BKF_DO)NetCoOutDispInit, (F_BKF_DOV)NetCoOutDispUninit},
};
uint32_t NetCoOutInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    uint32_t ret = BkfModsInit(g_NetCoOutModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoOutModVTbl));
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoOutUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "out(%#x)\n", BKF_MASK_ADDR(co->out));
    if (co->out == VOS_NULL) {
        return;
    }

    BkfModsUninit(g_NetCoOutModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoOutModVTbl));
}

#ifdef __cplusplus
}
#endif
