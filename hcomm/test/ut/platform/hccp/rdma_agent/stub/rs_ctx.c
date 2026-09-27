/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccp_ctx.h"
#include "ra_rs_comm.h"
#include "ra_rs_ctx.h"
#include "ra_hdc_ctx.h"

int RsGetDevEidInfoNum(unsigned int phyId, unsigned int* num) { return 0; }

int RsGetDevEidInfoList(
    unsigned int phyId, struct HccpDevEidInfo infoList[], unsigned int startIndex, unsigned int count)
{
    return 0;
}

int RsGetEidByIp(struct RaRsDevInfo* devInfo, struct IpInfo ip[], unsigned int num, union HccpEid eid[]) { return 0; }

int RsGetIpByEid(struct RaRsDevInfo* devInfo, union HccpEid eid[], struct IpInfo ip[], unsigned int* num) { return 0; }

int RsCtxInit(struct CtxInitAttr* attr, unsigned int* devIndex, struct DevBaseAttr* devAttr) { return 0; }

int RsCtxDeinit(struct RaRsDevInfo* devInfo) { return 0; }

int RsCtxTokenIdAlloc(struct RaRsDevInfo* devInfo, unsigned long long* addr, unsigned int* tokenId) { return 0; }

int RsCtxTokenIdFree(struct RaRsDevInfo* devInfo, unsigned long long addr) { return 0; }

int RsCtxLmemReg(struct RaRsDevInfo* devInfo, struct MemRegAttrT* memAttr, struct MemRegInfoT* memInfo) { return 0; }

int RsCtxLmemUnreg(struct RaRsDevInfo* devInfo, unsigned long long addr) { return 0; }

int RsCtxRmemImport(struct RaRsDevInfo* devInfo, struct MemImportAttrT* memAttr, struct MemImportInfoT* memInfo)
{
    return 0;
}

int RsCtxRmemUnimport(struct RaRsDevInfo* devInfo, unsigned long long addr) { return 0; }

int RsCtxChanCreate(struct RaRsDevInfo* devInfo, unsigned long long* addr) { return 0; }

int RsCtxChanDestroy(struct RaRsDevInfo* devInfo, unsigned long long addr) { return 0; }

int RsCtxCqCreate(struct RaRsDevInfo* devInfo, struct CtxCqAttr* attr, struct CtxCqInfo* info) { return 0; }

int RsCtxCqDestroy(struct RaRsDevInfo* devInfo, unsigned long long addr) { return 0; }

int RsCtxQpCreate(struct RaRsDevInfo* devInfo, struct CtxQpAttr* qpAttr, struct QpCreateInfo* qpInfo) { return 0; }

int RsCtxQpDestroy(struct RaRsDevInfo* devInfo, unsigned int id) { return 0; }

int RsCtxQpImport(
    struct RaRsDevInfo* devInfo, struct RsJettyImportAttr* importAttr, struct RsJettyImportInfo* importInfo)
{
    return 0;
}

int RsCtxQpUnimport(struct RaRsDevInfo* devInfo, unsigned char rawRemJettyId[], unsigned int size) { return 0; }

int RsCtxQpBind(struct RaRsDevInfo* devInfo, struct RsCtxQpInfo* localQpInfo, struct RsCtxQpInfo* remoteQpInfo)
{
    return 0;
}

int RsCtxQpUnbind(struct RaRsDevInfo* devInfo, unsigned int qpId) { return 0; }

int RsCtxBatchSendWr(
    struct WrlistBaseInfo* baseInfo, struct BatchSendWrData* wrData, struct SendWrResp* wrResp,
    struct WrlistSendCompleteNum* wrlistNum)
{
    return 0;
}

int RsCtxUpdateCi(struct RaRsDevInfo* devInfo, unsigned int qpId, uint16_t ci) { return 0; }

int RsGetTpInfoList(struct RaRsDevInfo* devInfo, struct GetTpCfg* cfg, struct HccpTpInfo infoList[], unsigned int* num)
{
    return 0;
}

int RsCtxQpDestroyBatch(struct RaRsDevInfo* devInfo, unsigned int ids[], unsigned int* num) { return 0; }

int RsCtxQpQueryBatch(struct RaRsDevInfo* devInfo, unsigned int ids[], struct JettyAttr attr[], unsigned int* num)
{
    return 0;
}

int RsCtxGetAuxInfo(struct RaRsDevInfo* devInfo, struct HccpAuxInfoIn* infoIn, struct HccpAuxInfoOut* infoOut)
{
    return 0;
}

int RsGetTpAttr(struct RaRsDevInfo* devInfo, unsigned int* attrBitmap, const uint64_t tpHandle, struct TpAttr* attr)
{
    return 0;
}

int RsSetTpAttr(
    struct RaRsDevInfo* devInfo, const unsigned int attrBitmap, const uint64_t tpHandle, struct TpAttr* attr)
{
    return 0;
}

int RsCtxGetCrErrInfoList(struct RaRsDevInfo* devInfo, struct CqeErrInfo infoList[], unsigned int* num) { return 0; }

int RsCtxGetAsyncEvents(struct RaRsDevInfo* devInfo, struct AsyncEvent asyncEvents[], unsigned int* num) { return 0; }

int RsCtxNotifyEvent(struct RaRsDevInfo* devInfo, struct CtxNotifyEvent* event) { return 0; }

int RsCtxGetUbContext(
    struct RaRsDevInfo* devInfo, unsigned int id, unsigned int contextType, uint8_t context[], unsigned int* len)
{
    return 0;
}
