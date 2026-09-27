/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "securec.h"
#include "aubdfx_api.h"
#include "config_log.h"
#include "dl_hal_function.h"
#include "dl_ibverbs_function.h"
#include "dl_urma_function.h"
#include "dl_ccu_function.h"
#include "dl_net_function.h"
#include "dl_aubdfx_function.h"
#include "hccp_ctx.h"
#include "ra_rs_ctx.h"
#include "ra_rs_err.h"
#include "rs_inner.h"
#include "rs_ctx_inner.h"
#include "rs_ub.h"
#include "rs_ub_tp.h"
#include "rs_ub_dfx.h"
#include "rs_ctx.h"

int RsGetChipProtocol(unsigned int chipId, enum NetworkMode hccpMode, enum ProtocolTypeT *protocol,
    unsigned int logicId)
{
#define CHIP_NAME_950 "950"
#define CHIP_NAME_960 "960"
    halChipInfo chipInfo = {0};
    int ret;

    // set default protocol to RDMA for compatibility issue
    *protocol = PROTOCOL_RDMA;
    // other modes skip to get protocol
    if (hccpMode != NETWORK_OFFLINE) {
        return 0;
    }

    ret = DlHalGetChipInfo(logicId, &chipInfo);
    CHK_PRT_RETURN(ret != 0,
        hccp_warn_rma("hal get chip info unsuccessful, chipId[%u], logicId[%u], ret[%d]", chipId, logicId, ret), 0);

    if ((strncmp((char *)chipInfo.name, CHIP_NAME_950, sizeof(CHIP_NAME_950) - 1) == 0) ||
        (strncmp((char *)chipInfo.name, CHIP_NAME_960, sizeof(CHIP_NAME_960) - 1) == 0)) {
        *protocol = PROTOCOL_UDMA;
    }
    return 0;
}

STATIC int RsCtxUdmaApiInit(enum ProtocolTypeT protocol)
{
    int ret;

    ret = RsUbApiInit();
    if (ret != 0) {
        hccp_err("rsUbApiInit failed, protocol[%u], ret[%d]", protocol, ret);
        return ret;
    }
    ret = RsCcuApiInit();
    if (ret != 0) {
        hccp_err("rsCcuApiInit failed, protocol[%u], ret[%d]", protocol, ret);
        goto err_ccu;
    }
    ret = RsNetApiInit();
    if (ret != 0) {
        hccp_err("rsNetApiInit failed, protocol[%u], ret[%d]", protocol, ret);
        goto err_net;
    }
    ret = RsAubdfxApiInit();
    if (ret != 0) {
        // aubdfx is optional, failure does not affect the main flow, re-write ret to 0, and print warning log
        hccp_warn("rsAubdfxApiInit failed, protocol[%u], ret[%d], notify event will not be supported", protocol, ret);
        ret = 0;
    }
    return ret;

err_net:
    RsCcuApiDeinit();
err_ccu:
    RsUbApiDeinit();
    return ret;
}

int RsCtxApiInit(enum NetworkMode hccpMode, enum ProtocolTypeT protocol)
{
    int ret = 0;

    // other modes skip to init api
    if (hccpMode != NETWORK_OFFLINE) {
        return ret;
    }

    switch (protocol) {
        case PROTOCOL_RDMA:
            ret = RsApiInit();
            CHK_PRT_RETURN(ret != 0, hccp_err("RsApiInit failed, protocol[%u], ret[%d]", protocol, ret), ret);
            break;
        case PROTOCOL_UDMA:
            return RsCtxUdmaApiInit(protocol);
        default:
            hccp_err("unsupported protocol[%u]", protocol);
            return -EINVAL;
    }

    return ret;
}

int RsCtxApiDeinit(enum NetworkMode hccpMode, enum ProtocolTypeT protocol)
{
    // other modes skip to deinit api
    if (hccpMode != NETWORK_OFFLINE) {
        return 0;
    }

    switch (protocol) {
        case PROTOCOL_RDMA:
            RsApiDeinit();
            break;
        case PROTOCOL_UDMA:
            RsAubdfxApiDeinit();
            RsUbApiDeinit();
            RsCcuApiDeinit();
            RsNetApiDeinit();
            break;
        default:
            hccp_err("unsupported protocol[%u]", protocol);
            return -EINVAL;
    }
    return 0;
}

RS_ATTRI_VISI_DEF int RsGetDevEidInfoNum(unsigned int phyId, unsigned int *num)
{
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsUbGetDevEidInfoNum(phyId, num);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_get_dev_eid_info_num failed, phyId:%u, ret:%d", phyId, ret), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsGetDevEidInfoList(unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex,
    unsigned int count)
{
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(infoList);

    ret = RsUbGetDevEidInfoList(phyId, infoList, startIndex, count);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_get_dev_eid_info_list failed, phyId:%u, ret:%d", phyId, ret), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxInit(struct CtxInitAttr *attr, unsigned int *devIndex, struct DevBaseAttr *devAttr)
{
    struct rs_cb *rscb = NULL;
    unsigned int phyId;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(attr);
    RS_CHECK_POINTER_NULL_RETURN_INT(devIndex);
    RS_CHECK_POINTER_NULL_RETURN_INT(devAttr);
    phyId = attr->phyId;
    ret = RsGetRsCb(phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsGetRsCb failed, ret:%d", ret), ret);

#ifdef CUSTOM_INTERFACE
    // setup sharemem for pingmesh
    ret = RsSetupSharemem(rscb, false, phyId);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsSetupSharemem failed, phyId:%u, ret:%d", phyId, ret), ret);
#endif

    ret = RsUbCtxInit(rscb, attr, devIndex, devAttr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_init failed, devIndex:0x%x, ret:%d", *devIndex, ret), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxGetAsyncEvents(struct RaRsDevInfo *devInfo, struct AsyncEvent asyncEvents[],
    unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(asyncEvents);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    CHK_PRT_RETURN(*num == 0, hccp_err("num can not be 0"), -EINVAL);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    RsUbCtxGetAsyncEvents(devCb, asyncEvents, num);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxDeinit(struct RaRsDevInfo *devInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get devCb fail, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxDeinit(devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs ub ctx deinit failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsGetEidByIp(struct RaRsDevInfo *devInfo, struct IpInfo ip[], union HccpEid eid[],
    unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(ip);
    RS_CHECK_POINTER_NULL_RETURN_INT(eid);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbGetEidByIp(devCb, ip, eid, num);
    CHK_PRT_RETURN(ret != 0, hccp_err("get_eid_by_ip failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsGetIpByEid(struct RaRsDevInfo *devInfo, union HccpEid eid[], struct IpInfo ip[],
    unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(eid);
    RS_CHECK_POINTER_NULL_RETURN_INT(ip);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbGetIpByEid(devCb, eid, ip, num);
    CHK_PRT_RETURN(ret != 0, hccp_err("RsUbGetIpByEid failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsGetTpInfoList(struct RaRsDevInfo *devInfo, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(cfg);
    RS_CHECK_POINTER_NULL_RETURN_INT(infoList);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
    ret = RsUbGetTpInfoList(devCb, cfg, infoList, num);

    return ret;
}

RS_ATTRI_VISI_DEF int RsGetTpAttr(struct RaRsDevInfo *devInfo, unsigned int *attrBitmap, const uint64_t tpHandle,
    struct TpAttr *attr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(attrBitmap);
    RS_CHECK_POINTER_NULL_RETURN_INT(attr);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
    ret = RsUbGetTpAttr(devCb, attrBitmap, tpHandle, attr);

    return ret;
}

RS_ATTRI_VISI_DEF int RsSetTpAttr(struct RaRsDevInfo *devInfo, const unsigned int attrBitmap, const uint64_t tpHandle,
    struct TpAttr *attr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(attr);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
    ret = RsUbSetTpAttr(devCb, attrBitmap, tpHandle, attr);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxTokenIdAlloc(struct RaRsDevInfo *devInfo, unsigned long long *addr, unsigned int *tokenId)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(addr);
    RS_CHECK_POINTER_NULL_RETURN_INT(tokenId);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxTokenIdAlloc(devCb, addr, tokenId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_token_id_alloc failed, ret:%d, devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxTokenIdFree(struct RaRsDevInfo *devInfo, unsigned long long addr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxTokenIdFree(devCb, addr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_token_id_free failed, ret:%d, devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxLmemReg(struct RaRsDevInfo *devInfo, struct MemRegAttrT *memAttr,
    struct MemRegInfoT *memInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(memAttr);
    RS_CHECK_POINTER_NULL_RETURN_INT(memInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxLmemReg(devCb, memAttr, memInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_lmem_reg failed, ret:%d, devIndex:0x%x", ret, devInfo->devIndex), ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxLmemUnreg(struct RaRsDevInfo *devInfo, unsigned long long addr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxLmemUnreg(devCb, addr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_lmem_unreg failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxRmemImport(struct RaRsDevInfo *devInfo, struct MemImportAttrT *memAttr,
    struct MemImportInfoT *memInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(memAttr);
    RS_CHECK_POINTER_NULL_RETURN_INT(memInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxRmemImport(devCb, memAttr, memInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_rmem_import failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxRmemUnimport(struct RaRsDevInfo *devInfo, unsigned long long addr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxRmemUnimport(devCb, addr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_rmem_unimport failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxChanCreate(struct RaRsDevInfo *devInfo, union DataPlaneCstmFlag dataPlaneFlag,
    unsigned long long *addr, int *fd)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(addr);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxChanCreate(devCb, dataPlaneFlag, addr, fd);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_chan_create failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxChanDestroy(struct RaRsDevInfo *devInfo, unsigned long long addr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxChanDestroy(devCb, addr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_chan_destroy failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxCqCreate(struct RaRsDevInfo *devInfo, struct CtxCqAttr *attr, struct CtxCqInfo *info)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(attr);
    RS_CHECK_POINTER_NULL_RETURN_INT(info);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJfcCreate(devCb, attr, info);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jfc_create failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxCqDestroy(struct RaRsDevInfo *devInfo, unsigned long long addr)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJfcDestroy(devCb, addr);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jfc_destroy failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpCreate(struct RaRsDevInfo *devInfo, struct CtxQpAttr *qpAttr, struct QpCreateInfo *qpInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(qpAttr);
    RS_CHECK_POINTER_NULL_RETURN_INT(qpInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyCreate(devCb, qpAttr, qpInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_create failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpDestroy(struct RaRsDevInfo *devInfo, unsigned int id)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyDestroy(devCb, id);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_destroy failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpImport(struct RaRsDevInfo *devInfo, struct RsJettyImportAttr *importAttr,
    struct RsJettyImportInfo *importInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(importAttr);
    RS_CHECK_POINTER_NULL_RETURN_INT(importInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyImport(devCb, importAttr, importInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_import failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpUnimport(struct RaRsDevInfo *devInfo, unsigned char rawRemJettyId[], unsigned int size)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyUnimport(devCb, rawRemJettyId, size);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_unimport failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpBind(struct RaRsDevInfo *devInfo, struct RsCtxQpInfo *localQpInfo,
    struct RsCtxQpInfo *remoteQpInfo)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(localQpInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(remoteQpInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyBind(devCb, localQpInfo, remoteQpInfo);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_bind failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpUnbind(struct RaRsDevInfo *devInfo, unsigned int qpId)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxJettyUnbind(devCb, qpId);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_jetty_unbind failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxBatchSendWr(struct WrlistBaseInfo *baseInfo, struct BatchSendWrData *wrData,
    struct SendWrResp *wrResp, struct WrlistSendCompleteNum *wrlistNum)
{
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(baseInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(wrData);
    RS_CHECK_POINTER_NULL_RETURN_INT(wrResp);
    RS_CHECK_POINTER_NULL_RETURN_INT(wrlistNum);

    ret = RsGetRsCb(baseInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    switch (rscb->protocol) {
        case PROTOCOL_UDMA:
            ret = RsUbCtxBatchSendWr(rscb, baseInfo, wrData, wrResp, wrlistNum);
            break;
        default:
            hccp_err("protocol[%d] not support", rscb->protocol);
            return -EINVAL;
    }
    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxUpdateCi(struct RaRsDevInfo *devInfo, unsigned int qpId, uint16_t ci)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    switch (rscb->protocol) {
        case PROTOCOL_UDMA:
            ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
            CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
            ret = RsUbCtxJettyUpdateCi(devCb, qpId, ci);
            break;
        default:
            hccp_err("protocol[%d] not support", rscb->protocol);
            return -EINVAL;
    }
    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpDestroyBatch(struct RaRsDevInfo *devInfo, unsigned int ids[], unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(ids);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
    ret = RsUbCtxJettyDestroyBatch(devCb, ids, num);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxQpQueryBatch(struct RaRsDevInfo *devInfo, unsigned int ids[], struct JettyAttr attr[],
    unsigned int *num)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(ids);
    RS_CHECK_POINTER_NULL_RETURN_INT(attr);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);
    ret = RsUbCtxQueryJettyBatch(devCb, ids, attr, num);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxGetAuxInfo(struct RaRsDevInfo *devInfo, struct HccpAuxInfoIn *infoIn,
    struct HccpAuxInfoOut *infoOut)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(infoIn);
    RS_CHECK_POINTER_NULL_RETURN_INT(infoOut);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    ret = RsUbCtxGetAuxInfo(devCb, infoIn, infoOut);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_ub_ctx_get_aux_info failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex),
        ret);

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxGetCrErrInfoList(struct RaRsDevInfo *devInfo, struct CrErrInfo infoList[], unsigned int *num)
{
    struct RsCtxJettyCb *jettyCbCurr = NULL;
    struct RsCtxJettyCb *jettyCbNext = NULL;
    struct RsUbDevCb *devCb = NULL;
    unsigned int crErrIdx = 0;
    struct rs_cb *rscb = NULL;
    unsigned int numTmp = 0;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(infoList);
    RS_CHECK_POINTER_NULL_RETURN_INT(num);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get dev_cb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    if (RsListEmpty(&devCb->jettyList)) {
        *num = 0;
        return ret;
    }

    numTmp = *num;
    RS_LIST_GET_HEAD_ENTRY(jettyCbCurr, jettyCbNext, &devCb->jettyList, list, struct RsCtxJettyCb);
    for (; (&jettyCbCurr->list) != &devCb->jettyList;
         jettyCbCurr = jettyCbNext, jettyCbNext = list_entry(jettyCbNext->list.next, struct RsCtxJettyCb, list)) {
        if (jettyCbCurr->crErrInfo.info.status != 0) {
            RS_PTHREAD_MUTEX_LOCK(&jettyCbCurr->crErrInfo.mutex);
            infoList[crErrIdx].status = jettyCbCurr->crErrInfo.info.status;
            infoList[crErrIdx].jettyId = jettyCbCurr->crErrInfo.info.jettyId;
            infoList[crErrIdx].time = jettyCbCurr->crErrInfo.info.time;
            jettyCbCurr->crErrInfo.info.status = 0;
            RS_PTHREAD_MUTEX_ULOCK(&jettyCbCurr->crErrInfo.mutex);
            RS_PTHREAD_MUTEX_LOCK(&jettyCbCurr->devCb->cqeErrCntMutex);
            jettyCbCurr->devCb->cqeErrCnt--;
            RS_PTHREAD_MUTEX_ULOCK(&jettyCbCurr->devCb->cqeErrCntMutex);
            crErrIdx++;
            if (crErrIdx == numTmp) {
                break;
            }
        }
    }

    *num = crErrIdx;
    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxGetUbContext(struct RaRsDevInfo *devInfo, unsigned int id, unsigned int contextType,
    uint8_t context[], unsigned int *len)
{
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret = 0;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(context);
    RS_CHECK_POINTER_NULL_RETURN_INT(len);
    CHK_PRT_RETURN(*len < CONTEXT_MAX_LEN, hccp_err("len:%u less than %u", *len, CONTEXT_MAX_LEN), -EINVAL);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);

    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get devCb failed, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    if (contextType == CONTEXT_TYPE_JETTY) {
        ret = RsUbGetJettyContext(devCb, id, context, len);
    } else {
        hccp_err("invalid contextType:%u, devIndex:0x%x", contextType, devInfo->devIndex);
        ret = -EINVAL;
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxNotifyEvent(struct RaRsDevInfo *devInfo, struct CtxNotifyEvent *event)
{
    struct ub_service_errinfo errInfo = {0};
    struct RsUbDevCb *devCb = NULL;
    struct rs_cb *rscb = NULL;
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(devInfo);
    RS_CHECK_POINTER_NULL_RETURN_INT(event);

    hccp_info("[notify][event]serviceType:%u errorType:%u", event->serviceType, event->errorType);

    ret = RsGetRsCb(devInfo->phyId, &rscb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get rscb failed, ret:%d", ret), ret);
    ret = RsUbGetDevCb(rscb, devInfo->devIndex, &devCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("get devCb fail, ret:%d devIndex:0x%x", ret, devInfo->devIndex), ret);

    if (devCb->devAttr.ub.dieId > UINT8_MAX || devCb->devAttr.ub.funcId > UINT8_MAX) {
        hccp_warn("[notify][event]dieId:%u or funcId:%u out of range", devCb->devAttr.ub.dieId,
            devCb->devAttr.ub.funcId);
    }
    errInfo.dieid = (unsigned char)devCb->devAttr.ub.dieId;
    errInfo.ueid = (unsigned char)devCb->devAttr.ub.funcId;
    errInfo.servicetype = event->serviceType;
    errInfo.errortype = event->errorType;
    (void)memcpy_s(errInfo.srceid, sizeof(errInfo.srceid), event->srcEid.raw, sizeof(event->srcEid.raw));
    (void)memcpy_s(errInfo.dsteid, sizeof(errInfo.dsteid), event->dstEid.raw, sizeof(event->dstEid.raw));
    errInfo.value = (event->serviceType == UBMEM_TYPE) ? event->errorInfo.hpa : event->errorInfo.tpn;

    return RsAubdfxNotifyEvent(devCb->devAttr.ub.dieId, 0, &errInfo, sizeof(errInfo));
}
