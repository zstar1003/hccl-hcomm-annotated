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
#include "net_co_in.h"
#include "net_co_in_data.h"
#include "net_co_main_data.h"
#include "net_co_in_out_comm.h"
#include "net_co_in_disp.h"
#include "bkf_ch_cli.h"
#include "bkf_ch_cli_letcp.h"
#include "simpo_multi_layer_common_builder.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_SUBER_IDL_VER_MAJOR (0)
#define NET_CO_SUBER_IDL_VER_MINOR (0)

STATIC uint32_t NetCoInDataNew(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfMemMng *memMng = co->memMng;
    uint32_t len = sizeof(NetCoIn);
    NetCoIn *in = BKF_MALLOC(memMng, len);
    if (in == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(in, len, 0, len);

    co->in = in;
    return BKF_OK;
}
STATIC void NetCoInDataDelete(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfMemMng *memMng = co->memMng;
    BKF_FREE(memMng, co->in);
    co->in = VOS_NULL;
}

STATIC uint32_t NetCoInChCliInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    uint32_t nameLen = BKF_NAME_LEN_MAX + 1;
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_in", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfChCliMngInitArg arg = {
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
    co->in->chCliMng = BkfChCliInit(&arg);
    if (co->in->chCliMng == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "chCliMng(%#x), ng\n", BKF_MASK_ADDR(co->in->chCliMng));
        return BKF_ERR;
    }

    BkfChCliTypeVTbl vTbl = {0};
    uint32_t ret = BkfChCliLetcpBuildVTbl(name, &vTbl, nameLen);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }
    ret = BkfChCliRegType(co->in->chCliMng, &vTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }
    ret = BkfChCliSetSelfCid(co->in->chCliMng, co->selfCid);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoInChCliUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "chCliMng(%#x)\n", BKF_MASK_ADDR(co->in->chCliMng));

    if (co->in->chCliMng != VOS_NULL) {
        BkfChCliUninit(co->in->chCliMng);
        co->in->chCliMng = VOS_NULL;
    }
}

STATIC uint32_t NetCoInSimpoInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    co->in->simpoBuilder = SimpoBuilderInit();
    if (co->in->simpoBuilder == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "simpoBuilder(%#x), ng\n", BKF_MASK_ADDR(co->in->simpoBuilder));
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoInSimpoUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "simpoBuilder(%#x)\n", BKF_MASK_ADDR(co->in->simpoBuilder));

    if (co->in->simpoBuilder != VOS_NULL) {
        SimpoBuilderUninit(co->in->simpoBuilder);
        co->in->simpoBuilder = VOS_NULL;
    }
}

STATIC uint32_t NetCoInSuberInitChkArgs(NetCo *co)
{
    uint8_t buf1[BKF_URL_STR_LEN_MAX];
    uint8_t buf2[BKF_URL_STR_LEN_MAX];
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "inConnSelfUrl(%s)/inConnNetUrl(%s)\n",
        BkfUrlGetStr(&co->argInit.inConnSelfUrl, buf1, sizeof(buf1)),
        BkfUrlGetStr(&co->argInit.inConnNetUrl, buf2, sizeof(buf2)));
    BOOL argIsInvalid = !BkfUrlIsValid(&co->argInit.inConnSelfUrl) || !BkfUrlIsValid(&co->argInit.inConnNetUrl);
    if (argIsInvalid) {
        BKF_LOG_ERROR(BKF_LOG_HND, "argIsInvalid(%u)\n", argIsInvalid);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC uint32_t NetCoInSuberInitSuber(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int32_t err = snprintf_truncated_s(name, sizeof(name), "%s_in", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BifrostCncoiSuberInitArg arg = {
        .name = name,
        .idlVersionMajor = NET_CO_SUBER_IDL_VER_MAJOR,
        .idlVersionMinor = NET_CO_SUBER_IDL_VER_MINOR,
        .dbgOn = co->dbgOn,
        .memMng = co->memMng,
        .disp = co->disp,
        .log = co->log,
        .pfm = co->pfm,
        .xMap = co->xMap,
        .tmrMng = co->tmrMng,
        .jobMng = co->jobMng,
        .jobTypeId1 = NET_CO_JOB_TYPE_IN_SUBER,
        .jobPrioH = NET_CO_JOB_PRIO_DFT,
        .jobPrioL = NET_CO_JOB_PRIO_DFT - 1,
        .sysLogMng = co->sysLogMng,
        .chCliMng = co->in->chCliMng,
        .sliceKeyLen = sizeof(BifrostCncoiSliceKeyT),
        .subMod = BKF_SUBER_MODE_EX,
        .sliceKeyCmp = (F_BKF_CMP)BifrostCncoiSliceKeyCmp,
        .sliceKeyGetStrOrNull = (F_BKF_GET_STR)BifrostCncoiSliceKeyGetStr,
        .sliceKeyCodec = (F_BKF_DO)BifrostCncoiSliceKeyCodec,
        .simpoBuilder = co->in->simpoBuilder,
        .lsnUrl = co->argInit.outConnLsnUrl,
    };
    co->in->suber = BifrostCncoiSuberInit(&arg);
    if (co->in->suber == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "suber(%#x), ng\n", BKF_MASK_ADDR(co->in->suber));
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC uint32_t NetCoInSuberInit(NetCo *co)
{
    uint32_t ret = NetCoInSuberInitChkArgs(co);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }

    ret = NetCoInSuberInitSuber(co);
    if (ret != BKF_OK) {
        return BKF_ERR;
    }
    ret = BifrostCncoiSuberCreateSvcInstEx(co->in->suber, NET_CO_DFT_INST_ID);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiSuberSetSvcInstLocalUrl(co->in->suber, NET_CO_DFT_INST_ID, &co->argInit.inConnSelfUrl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiSuberSetSvcInstUrlEx(co->in->suber, NET_CO_DFT_INST_ID, &co->argInit.inConnNetUrl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiSuberBindSliceInstIdEx(co->in->suber, g_NetCoDftSliceKey, NET_CO_DFT_INST_ID);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiSuberEnable(co->in->suber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%#x), ng\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}
STATIC void NetCoInSuberUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "suber(%#x)\n", BKF_MASK_ADDR(co->in->suber));

    if (co->in->suber != VOS_NULL) {
        BifrostCncoiSuberUninit(co->in->suber);
        co->in->suber = VOS_NULL;
    }
}

STATIC const BkfModVTbl g_NetCoInModVTbl[] = {
    {(F_BKF_DO)NetCoInDataNew, (F_BKF_DOV)NetCoInDataDelete},
    {(F_BKF_DO)NetCoInChCliInit, (F_BKF_DOV)NetCoInChCliUninit},
    {(F_BKF_DO)NetCoInSimpoInit, (F_BKF_DOV)NetCoInSimpoUninit},
    {(F_BKF_DO)NetCoInSuberInit, (F_BKF_DOV)NetCoInSuberUninit},
    {(F_BKF_DO)NetCoInDispInit, (F_BKF_DOV)NetCoInDispUninit},
};
uint32_t NetCoInInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    uint32_t ret = BkfModsInit(g_NetCoInModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoInModVTbl));
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoInUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "in(%#x)\n", BKF_MASK_ADDR(co->in));
    if (co->in == VOS_NULL) {
        return;
    }

    BkfModsUninit(g_NetCoInModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoInModVTbl));
}

#ifdef __cplusplus
}
#endif
