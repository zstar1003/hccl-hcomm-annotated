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
#include <urma_opcode.h>
#include <udma_u_ctl.h>
#include "user_log.h"
#include "dl_urma_function.h"
#include "dl_net_function.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "rs_ub.h"
#include "rs_ub_jetty.h"

STATIC int RsResAddrMunmap(struct RsCtxJettyCb *jettyCb, struct UdmaVaInfo *vaInfo)
{
    struct res_map_info_in resInfoIn = {0};
    int ret = 0;

    resInfoIn.res_id = RsGenerateMmapResId(jettyCb->jetty->jetty_id.id, jettyCb->devCb->devAttr.ub.dieId,
        jettyCb->devCb->devAttr.ub.funcId);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = vaInfo->resType;
    resInfoIn.priv_len = sizeof(struct UdmaVaInfo);
    resInfoIn.priv = (void *)vaInfo;
    ret = DlHalResAddrUnmapV2(jettyCb->devCb->rscb->logicId, &resInfoIn);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("DlHalResAddrUnmapV2 failed, res_type:%d ret:%d, errno:%d", resInfoIn.res_type, ret, errno), ret);

    return ret;
}

STATIC int RsResAddrMmap(struct RsCtxJettyCb *jettyCb, struct UdmaVaInfo *vaInfo, struct res_map_info_out *resInfoOut)
{
    struct res_map_info_in resInfoIn = {0};
    int ret = 0;

    resInfoIn.res_id = RsGenerateMmapResId(jettyCb->jetty->jetty_id.id, jettyCb->devCb->devAttr.ub.dieId,
        jettyCb->devCb->devAttr.ub.funcId);
    resInfoIn.target_proc_type = PROCESS_CP1;
    resInfoIn.res_type = vaInfo->resType;
    resInfoIn.priv_len = sizeof(struct UdmaVaInfo);
    resInfoIn.priv = (void *)vaInfo;
    ret = DlHalResAddrMapV2(jettyCb->devCb->rscb->logicId, &resInfoIn, resInfoOut);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("DlHalResAddrMapV2 failed, res_type:%d ret:%d, errno:%d", resInfoIn.res_type, ret, errno), ret);

    return ret;
}

STATIC void RsMunmapJettyVa(struct RsCtxJettyCb *jettyCb)
{
    struct UdmaVaInfo vaInfo = {0};

    if ((jettyCb->jettyMode != JETTY_MODE_CACHE_LOCK_DWQE) && (jettyCb->jettyMode != JETTY_MODE_USER_CTL_NORMAL)) {
        return;
    }

    vaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_JETTY;
    vaInfo.va = jettyCb->sqBuffVa;
    vaInfo.len = WQE_BB_SIZE * jettyCb->txDepth * WQEBB_NUM_PER_SQE;
    vaInfo.pid = getpid();
    (void)RsResAddrMunmap(jettyCb, &vaInfo);

    vaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_DB;
    vaInfo.va = ALIGN_DOWN(jettyCb->dbAddr, RA_RS_4K_PAGE_SIZE);
    vaInfo.len = sizeof(uint64_t);
    vaInfo.pid = getpid();
    (void)RsResAddrMunmap(jettyCb, &vaInfo);
}

STATIC int RsMmapJettyVa(struct RsCtxJettyCb *jettyCb)
{
    struct res_map_info_out jettyVaInfoOut = {0};
    struct res_map_info_out dbVaInfoOut = {0};
    struct UdmaVaInfo jettyVaInfo = {0};
    struct UdmaVaInfo dbVaInfo = {0};
    uint64_t dbOffset = 0;
    int ret = 0;

    jettyVaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_JETTY;
    jettyVaInfo.va = jettyCb->sqBuffVa;
    jettyVaInfo.len = WQE_BB_SIZE * jettyCb->txDepth * WQEBB_NUM_PER_SQE;
    jettyVaInfo.pid = getpid();
    ret = RsResAddrMmap(jettyCb, &jettyVaInfo, &jettyVaInfoOut);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_res_addr_mmap failed, res_type:%u ret:%d", jettyVaInfo.resType, ret), ret);
    jettyCb->sqBuffVa = jettyVaInfoOut.va;

    dbVaInfo.resType = RES_ADDR_TYPE_HCCP_URMA_DB;
    dbVaInfo.va = ALIGN_DOWN(jettyCb->dbAddr, RA_RS_4K_PAGE_SIZE);
    dbOffset = jettyCb->dbAddr - dbVaInfo.va;
    dbVaInfo.len = sizeof(uint64_t);
    dbVaInfo.pid = getpid();
    ret = RsResAddrMmap(jettyCb, &dbVaInfo, &dbVaInfoOut);
    if (ret != 0) {
        hccp_err("rs_res_addr_mmap failed, res_type:%u ret:%d", dbVaInfo.resType, ret);
        goto munmap_sq_buff_va;
    }
    jettyCb->dbAddr = dbVaInfoOut.va + dbOffset;
    return ret;

munmap_sq_buff_va:
    jettyVaInfo.va = jettyVaInfoOut.va;
    ret += RsResAddrMunmap(jettyCb, &jettyVaInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_res_addr_munmap failed, res_type:%u ret:%d", jettyVaInfo.resType, ret), ret);
    return ret;
}

void RsUbCtxExtJettyDelete(struct RsCtxJettyCb *jettyCb)
{
    int ret = 0;

    RsMunmapJettyVa(jettyCb);
    ret = RsUrmaDeactiveJetty(jettyCb->jetty);
    if (ret != 0) {
        hccp_err("rs_urma_deactive_jetty failed, ret:%d errno:%d", ret, errno);
    }

    ret = RsUrmaFreeJetty(jettyCb->jetty);
    if (ret != 0) {
        hccp_err("rs_urma_free_jetty failed, ret:%d errno:%d", ret, errno);
    }

    if (jettyCb->jettyMode == JETTY_MODE_CACHE_LOCK_DWQE) {
        ret = RsNetFreeJettyId(jettyCb->devCb->urmaDev->name, jettyCb->jettyMode, jettyCb->jettyId);
        if (ret != 0) {
            hccp_err("rs_net_free_jetty_id failed, jettyId:%u ret:%d", jettyCb->jettyId, ret);
        }
    }

    return;
}

STATIC int RsSetCcuJettyOpt(struct RsCtxJettyCb *jettyCb)
{
    uint64_t ccuJettySqBuffVa = 0;
    int ret = 0;

    if (!RsIsCcuJetty(jettyCb->jettyMode)) {
        return ret;
    }

    if (jettyCb->jettyMode == JETTY_MODE_CCU) {
        ccuJettySqBuffVa = jettyCb->extMode.sq.buffVa;
    } else if (jettyCb->jettyMode == JETTY_MODE_CCU_TA_CACHE) {
        ccuJettySqBuffVa = (uint64_t)jettyCb->taCacheMode.sqeBufIdx;
    }

    ret = RsUrmaSetJettyOpt(jettyCb->jetty, URMA_JFS_SQE_BASE_ADDR, (void *)&ccuJettySqBuffVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_set_jetty_opt URMA_JFS_SQE_BASE_ADDR failed, ret:%d, errno:%d", ret, errno), -EOPENSRC);

    return ret;
}

STATIC int RsSetJettyOpt(struct RsCtxJettyCb *jettyCb)
{
    uint8_t dbCstm = jettyCb->extMode.cstmFlag.bs.dbCstm;
    uint16_t piType = jettyCb->extMode.piType;
    int ret = 0;

    hccp_dbg("sq.buff:0x%llx, sq.buffSize:%u, piType:%u, sqebbNum:%u, dbCstm:%u, txDepth:%u",
        jettyCb->extMode.sq.buffVa, jettyCb->extMode.sq.buffSize, piType, jettyCb->extMode.sqebbNum, dbCstm,
        jettyCb->txDepth);

    ret = RsUrmaSetJettyOpt(jettyCb->jetty, URMA_JFS_DB_STATUS, (void *)&dbCstm, sizeof(uint8_t));
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_set_jetty_opt URMA_JFS_DB_STATUS failed, ret:%d, errno:%d", ret, errno),
        -EOPENSRC);

    ret = RsUrmaSetJettyOpt(jettyCb->jetty, URMA_JFS_PI_TYPE, (void *)&piType, sizeof(uint16_t));
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_set_jetty_opt URMA_JFS_PI_TYPE failed, ret:%d, errno:%d", ret, errno),
        -EOPENSRC);

    ret = RsSetCcuJettyOpt(jettyCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsSetCcuJettyOpt failed, ret:%d, mode:%d", ret, jettyCb->jettyMode), ret);

    return ret;
}

STATIC int RsGetJettyOpt(struct RsCtxJettyCb *jettyCb)
{
    uint64_t sqBuffVa = 0, dbVa = 0;
    int ret = 0;

    ret = RsUrmaGetJettyOpt(jettyCb->jetty, URMA_JFS_SQE_BASE_ADDR, &sqBuffVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0,
        hccp_err("rs_urma_get_jetty_opt URMA_JFS_SQE_BASE_ADDR failed, ret:%d, errno:%d", ret, errno), -EOPENSRC);

    ret = RsUrmaGetJettyOpt(jettyCb->jetty, URMA_JFS_DB_ADDR, &dbVa, sizeof(uint64_t));
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_urma_get_jetty_opt URMA_JFS_DB_ADDR failed, ret:%d, errno:%d", ret, errno),
        -EOPENSRC);

    jettyCb->sqBuffVa = sqBuffVa;
    jettyCb->dbAddr = dbVa;
    if ((jettyCb->jettyMode == JETTY_MODE_CACHE_LOCK_DWQE) || (jettyCb->jettyMode == JETTY_MODE_USER_CTL_NORMAL)) {
        ret = RsMmapJettyVa(jettyCb);
        CHK_PRT_RETURN(ret != 0, hccp_err("rs_mmap_jetty_va failed, ret:%d", ret), ret);
    }

    return ret;
}

STATIC int RsFreeJettyId(const char *udevName, unsigned int jettyMode, unsigned int jettyId)
{
    int ret = 0;

    if (jettyMode != JETTY_MODE_CACHE_LOCK_DWQE) {
        return 0;
    }

    // only stars jetty need to free jetty id
    ret = RsNetFreeJettyId(udevName, jettyMode, jettyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_net_free_jetty_id failed, jettyId:%u ret:%d", jettyId, ret), ret);

    return ret;
}

STATIC int RsJettyAttrInit(struct RsCtxJettyCb *jettyCb, urma_jetty_cfg_t *jettyCfg)
{
    int ret = 0;

    CHK_PRT_RETURN(jettyCb->extMode.cstmFlag.bs.sqCstm == 1 && !RsIsCcuJetty(jettyCb->jettyMode),
        hccp_err("Non-CCU jetty cannot be created by specifying va, sqCstm:%u jettyMode:%u",
            jettyCb->extMode.cstmFlag.bs.sqCstm, jettyCb->jettyMode),
        -EINVAL);

    if (jettyCb->jettyMode == JETTY_MODE_CACHE_LOCK_DWQE) {
        ret = RsNetAllocJettyId(jettyCb->devCb->urmaDev->name, jettyCb->jettyMode, &jettyCfg->id);
        CHK_PRT_RETURN(ret != 0, hccp_err("rs_net_alloc_jetty_id failed, ret:%d", ret), ret);
        jettyCb->jettyId = jettyCfg->id;
    }

    ret = RsUrmaAllocJetty(jettyCb->devCb->urmaCtx, jettyCfg, &jettyCb->jetty);
    if (ret != 0) {
        ret = -EOPENSRC;
        RsFreeJettyId(jettyCb->devCb->urmaDev->name, jettyCb->jettyMode, jettyCb->jettyId);
        hccp_err("urma_alloc_jetty failed, ret:%d, errno:%d", ret, errno);
    }

    return ret;
}

STATIC int RsCcuJettyDbReg(struct RsCtxJettyCb *jettyCb)
{
    struct udma_u_jetty_info jettyInfo = {0};
    int ret = 0;

    if (!RsIsCcuJetty(jettyCb->jettyMode)) {
        return ret;
    }

    // only ccu jetty requires db registration
    jettyInfo.dwqe_addr = (void *)(ALIGN_DOWN(jettyCb->dbAddr, RA_RS_4K_PAGE_SIZE));
    ret = RsUbCtxRegJettyDb(jettyCb, &jettyInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_reg_jetty_db failed, ret:%d", ret), ret);

    return ret;
}

void RsUbCtxExtJettyCreate(struct RsCtxJettyCb *jettyCb, urma_jetty_cfg_t *jettyCfg)
{
    int ret = 0;

    ret = RsJettyAttrInit(jettyCb, jettyCfg);
    if (ret != 0) {
        jettyCb->jetty = NULL;
        return;
    }

    ret = RsSetJettyOpt(jettyCb);
    if (ret != 0) {
        hccp_err("rs_set_jetty_opt failed, ret:%d", ret);
        goto free_jetty;
    }

    ret = RsUrmaActiveJetty(jettyCb->jetty);
    if (ret != 0) {
        hccp_err("rs_urma_active_jetty failed, ret:%d, errno:%d", ret, errno);
        ret = -EOPENSRC;
        goto free_jetty;
    }

    ret = RsGetJettyOpt(jettyCb);
    if (ret != 0) {
        hccp_err("rs_get_jetty_opt failed, ret:%d", ret);
        goto deactive_jetty;
    }

    ret = RsCcuJettyDbReg(jettyCb);
    if (ret != 0) {
        goto deactive_jetty;
    }
    return;

deactive_jetty:
    ret = RsUrmaDeactiveJetty(jettyCb->jetty);
    if (ret != 0) {
        hccp_err("rs_urma_deactive_jetty failed, ret:%d errno:%d", ret, errno);
    }
free_jetty:
    ret = RsUrmaFreeJetty(jettyCb->jetty);
    if (ret != 0) {
        hccp_err("rs_urma_free_jetty failed, ret:%d errno:%d", ret, errno);
    }

    (void)RsFreeJettyId(jettyCb->devCb->urmaDev->name, jettyCb->jettyMode, jettyCb->jettyId);
    jettyCb->jetty = NULL;
}

void RsUbVaMunmapBatch(struct RsCtxJettyCb **jettyCbArr, unsigned int num)
{
    unsigned int i;

    for (i = 0; i < num; ++i) {
        RsMunmapJettyVa(jettyCbArr[i]);
    }
}

void RsUbFreeJettyIdBatch(struct RsCtxJettyCb **jettyCbArr, unsigned int num)
{
    unsigned int i;

    for (i = 0; i < num; ++i) {
        (void)RsFreeJettyId(jettyCbArr[i]->devCb->urmaDev->name, jettyCbArr[i]->jettyMode, jettyCbArr[i]->jettyId);
    }
}
