/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unistd.h>
#include <stdlib.h>
#include <sys/types.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <dlfcn.h>
#include "urma_opcode.h"
#include "user_log.h"
#include "dl_hal_function.h"
#include "dl_urma_function.h"
#include "dl_net_function.h"
#include "dl_ccu_function.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "rs_ub_jfc.h"

struct ExtJfcAttr {
    urma_jfc_t *jfc;
    unsigned int jfcId;
    unsigned long long cqeBaseAddrVa;
};

STATIC int RsInitJfcAttr(struct RsCtxJfcCb *jfcCb, urma_jfc_cfg_t *jfcCfg, struct ExtJfcAttr *jfcAttr)
{
    unsigned int dieId = 0;
    int ret = 0;

    ret = RsUrmaAllocJfc(jfcCb->devCb->urmaCtx, jfcCfg, &jfcAttr->jfc);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_alloc_jfc failed, ret:%d errno:%d", ret, errno), -EOPENSRC);

    if (jfcCb->jfcType == JFC_MODE_USER_CTL_NORMAL) {
        return 0;
    }

    if (jfcCb->jfcType == JFC_MODE_CCU_POLL) {
        if (!jfcCb->ccuExCfg.valid || jfcCb->ccuExCfg.cqeFlag == 0) {
            dieId = jfcCb->devCb->devAttr.ub.dieId;
        } else {
            dieId = (jfcCb->devCb->devAttr.ub.dieId == 0) ? 1 : 0;
        }
        ret = RsCcuGetCqeBaseAddr(dieId, &jfcAttr->cqeBaseAddrVa);
        if (ret != 0 || jfcAttr->cqeBaseAddrVa == 0) {
            hccp_err("rs_ccu_get_cqe_base_addr failed, ret:%d, dieId:%u", ret, dieId);
            ret = -EOPENSRC;
            goto free_jfc;
        }
    } else {
        ret = RsNetGetCqeBaseAddr(jfcCb->devCb->devAttr.ub.dieId, &jfcAttr->cqeBaseAddrVa);
        if (ret != 0 || jfcAttr->cqeBaseAddrVa == 0) {
            hccp_err("rs_net_get_cqe_base_addr failed, ret:%d, dieId:%u", ret, jfcCb->devCb->devAttr.ub.dieId);
            ret = -EOPENSRC;
            goto free_jfc;
        }
    }

    ret = RsNetAllocJfcId(jfcCb->devCb->urmaDev->name, jfcCb->jfcType, &jfcAttr->jfcId);
    if (ret != 0) {
        hccp_err("rs_net_alloc_jfc_id failed, ret:%d", ret);
        goto free_jfc;
    }

    return 0;

free_jfc:
    (void)RsUrmaFreeJfc(jfcAttr->jfc);
    return ret;
}

STATIC void RsDeinitJfcAttr(struct RsCtxJfcCb *jfcCb, urma_jfc_cfg_t *jfcCfg, struct ExtJfcAttr *jfcAttr)
{
    (void)jfcCfg;
    (void)RsUrmaFreeJfc(jfcAttr->jfc);
    if (jfcCb->jfcType == JFC_MODE_USER_CTL_NORMAL) {
        return;
    }
    (void)RsNetFreeJfcId(jfcCb->devCb->urmaDev->name, jfcCb->jfcType, jfcAttr->jfcId);
}

STATIC int RsSetJfcOpt(struct RsCtxJfcCb *jfcCb, struct ExtJfcAttr *jfcAttr)
{
    int ret = 0;

    if (jfcCb->jfcType == JFC_MODE_USER_CTL_NORMAL) {
        return ret;
    }

    ret = RsUrmaSetJfcOpt(jfcAttr->jfc, URMA_JFC_ID, (void *)&jfcAttr->jfcId, sizeof(uint32_t));
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_set_jfc_opt URMA_JFC_ID failed, ret:%d, errno:%d", ret, errno),
        -EOPENSRC);

    ret = RsUrmaSetJfcOpt(jfcAttr->jfc, URMA_JFC_CQE_BASE_ADDR, (void *)&jfcAttr->cqeBaseAddrVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_set_jfc_opt URMA_JFC_CQE_BASE_ADDR failed, ret:%d, errno:%d", ret, errno), -EOPENSRC);

    return 0;
}

STATIC int RsJfcResAddrMunmap(struct RsCtxJfcCb *jfcCb, struct UdmaVaInfo *vaInfo)
{
    struct res_map_info_in resInfoIn = {0};
    int ret = 0;

    resInfoIn.res_id = RsGenerateMmapResId(jfcCb->jfcId, jfcCb->devCb->devAttr.ub.dieId,
        jfcCb->devCb->devAttr.ub.funcId);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = (enum res_addr_type)vaInfo->resType;
    resInfoIn.priv_len = sizeof(struct UdmaVaInfo);
    resInfoIn.priv = (void *)vaInfo;
    ret = DlHalResAddrUnmapV2(jfcCb->devCb->rscb->logicId, &resInfoIn);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("DlHalResAddrUnmapV2 failed, res_type:%d ret:%d, errno:%d", resInfoIn.res_type, ret, errno), ret);

    return ret;
}

STATIC int RsJfcResAddrMmap(struct RsCtxJfcCb *jfcCb, struct UdmaVaInfo *vaInfo, struct res_map_info_out *resInfoOut)
{
    struct res_map_info_in resInfoIn = {0};
    int ret = 0;

    resInfoIn.res_id = RsGenerateMmapResId(jfcCb->jfcId, jfcCb->devCb->devAttr.ub.dieId,
        jfcCb->devCb->devAttr.ub.funcId);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = (enum res_addr_type)vaInfo->resType;
    resInfoIn.priv_len = sizeof(struct UdmaVaInfo);
    resInfoIn.priv = (void *)vaInfo;
    ret = DlHalResAddrMapV2(jfcCb->devCb->rscb->logicId, &resInfoIn, resInfoOut);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("DlHalResAddrMapV2 failed, res_type:%d ret:%d, errno:%d", resInfoIn.res_type, ret, errno), ret);

    return ret;
}

STATIC void RsMunmapJfcVa(struct RsCtxJfcCb *jfcCb)
{
    struct UdmaVaInfo vaInfo = {0};

    if (jfcCb->jfcType != JFC_MODE_USER_CTL_NORMAL) {
        return;
    }

    vaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_JFC;
    vaInfo.va = jfcCb->bufAddr;
    vaInfo.len = WQE_BB_SIZE * jfcCb->depth;
    vaInfo.pid = getpid();
    (void)RsJfcResAddrMunmap(jfcCb, &vaInfo);

    vaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_DB;
    vaInfo.va = jfcCb->swdbAddr;
    vaInfo.len = sizeof(uint64_t);
    vaInfo.pid = getpid();
    (void)RsJfcResAddrMunmap(jfcCb, &vaInfo);
}

STATIC int RsMmapJfcVa(struct RsCtxJfcCb *jfcCb)
{
    struct res_map_info_out resInfoOut = {0};
    struct UdmaVaInfo jfcVaInfo = {0};
    struct UdmaVaInfo dbVaInfo = {0};
    int retTmp = 0;
    int ret = 0;

    jfcVaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_JFC;
    jfcVaInfo.va = jfcCb->bufAddr;
    jfcVaInfo.len = WQE_BB_SIZE * jfcCb->depth;
    jfcVaInfo.pid = getpid();
    ret = RsJfcResAddrMmap(jfcCb, &jfcVaInfo, &resInfoOut);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_jfc_res_addr_mmap failed, res_type:%u ret:%d", jfcVaInfo.resType, ret), ret);
    jfcCb->bufAddr = resInfoOut.va;

    dbVaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_DB;
    dbVaInfo.va = jfcCb->swdbAddr;
    dbVaInfo.len = sizeof(uint64_t);
    dbVaInfo.pid = getpid();
    ret = RsJfcResAddrMmap(jfcCb, &dbVaInfo, &resInfoOut);
    if (ret != 0) {
        hccp_err("rs_jfc_res_addr_mmap failed, res_type:%u ret:%d", dbVaInfo.resType, ret);
        goto munmap_jfc_buff_va;
    }

    jfcCb->swdbAddr = resInfoOut.va;
    return ret;

munmap_jfc_buff_va:
    jfcVaInfo.va = jfcCb->bufAddr;
    retTmp = RsJfcResAddrMunmap(jfcCb, &jfcVaInfo);
    CHK_PRT_RETURN(retTmp != 0,
        hccp_err("rs_jfc_res_addr_munmap failed, res_type:%u ret:%d", jfcVaInfo.resType, retTmp), retTmp);
    return ret;
}

STATIC int RsGetJfcOpt(struct RsCtxJfcCb *jfcCb, urma_jfc_t *jfc)
{
    uint64_t cqBuffVa = 0, dbVa = 0;
    int ret = 0;

    if (jfcCb->jfcType != JFC_MODE_USER_CTL_NORMAL) {
        return ret;
    }

    ret = RsUrmaGetJfcOpt(jfc, URMA_JFC_CQE_BASE_ADDR, &cqBuffVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_get_jfc_opt URMA_JFC_CQE_BASE_ADDR failed, ret:%d, errno:%d", ret, errno), -EOPENSRC);

    ret = RsUrmaGetJfcOpt(jfc, URMA_JFC_DB_ADDR, &dbVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_get_jfc_opt URMA_JFC_DB_ADDR failed, ret:%d, errno:%d", ret, errno),
        -EOPENSRC);

    jfcCb->bufAddr = cqBuffVa;
    jfcCb->swdbAddr = dbVa;

    ret = RsMmapJfcVa(jfcCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_mmap_jfc_va failed, ret:%d", ret), ret);

    return ret;
}

int RsUbCtxJfcCreateExt(struct RsCtxJfcCb *jfcCb, urma_jfc_cfg_t *jfcCfg, urma_jfc_t **jfc)
{
    struct ExtJfcAttr jfcAttr = {0};
    int ret = 0;

    ret = RsInitJfcAttr(jfcCb, jfcCfg, &jfcAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_init_jfc_attr failed, ret:%d", ret), ret);

    ret = RsSetJfcOpt(jfcCb, &jfcAttr);
    if (ret != 0) {
        hccp_err("rs_set_jfc_attr failed, ret:%d", ret);
        goto deinit_attr;
    }

    ret = RsUrmaActiveJfc(jfcAttr.jfc);
    if (ret != 0) {
        hccp_err("rs_urma_active_jfc failed, jfcId:%u, ret:%d, errno:%d", jfcAttr.jfc->jfc_id.id, ret, errno);
        ret = -EOPENSRC;
        goto deinit_attr;
    }
    jfcCb->jfcId = jfcAttr.jfc->jfc_id.id;

    ret = RsGetJfcOpt(jfcCb, jfcAttr.jfc);
    if (ret != 0) {
        hccp_err("rs_get_jfc_opt failed, jfcId:%u, ret:%d, errno:%d", jfcAttr.jfc->jfc_id.id, ret, errno);
        goto deactive_jfc;
    }

    *jfc = jfcAttr.jfc;
    return 0;

deactive_jfc:
    (void)RsUrmaDeactiveJfc(jfcAttr.jfc);
deinit_attr:
    (void)RsDeinitJfcAttr(jfcCb, jfcCfg, &jfcAttr);
    *jfc = NULL;
    return ret;
}

int RsUbDeleteJfcExt(struct RsUbDevCb *devCb, struct RsCtxJfcCb *jfcCb)
{
    urma_jfc_t *jfc = (urma_jfc_t *)(uintptr_t)(jfcCb->jfcAddr);
    unsigned int jfcId = jfc->jfc_id.id;
    int ret = 0;

    RsMunmapJfcVa(jfcCb);

    ret = RsUrmaDeactiveJfc(jfc);
    if (ret != 0) {
        hccp_err("rs_urma_deactive_jfc failed, jfcId:%u, ret:%d, errno:%d", jfcId, ret, errno);
        ret = -EOPENSRC;
    }

    ret = RsUrmaFreeJfc(jfc);
    if (ret != 0) {
        hccp_err("rs_urma_free_jfc failed, jfcId:%u, ret:%d, errno:%d", jfcId, ret, errno);
        ret = -EOPENSRC;
    }

    if (jfcCb->jfcType != JFC_MODE_USER_CTL_NORMAL) {
        ret = RsNetFreeJfcId(devCb->urmaDev->name, jfcCb->jfcType, jfcId);
        if (ret != 0) {
            hccp_err("rs_net_free_jfc_id failed, jfcId:%u, ret:%d", jfcId, ret);
        }
    }

    return ret;
}