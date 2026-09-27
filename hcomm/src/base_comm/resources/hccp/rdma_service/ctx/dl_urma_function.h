/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_URMA_FUNCTION_H
#define DL_URMA_FUNCTION_H

#include <ccan/list.h>
#include <urma_types.h>

#ifdef CA_CONFIG_LLT
#define STATIC
#else
#define STATIC static
#endif

struct RsUrmaOps {
    urma_status_t (*rsUrmaInit)(urma_init_attr_t *conf);
    urma_status_t (*rsUrmaUninit)(void);
    urma_device_t **(*rsUrmaGetDeviceList)(int *numDevices);
    void (*rsUrmaFreeDeviceList)(urma_device_t **deviceList);
    urma_eid_info_t *(*rsUrmaGetEidList)(urma_device_t *dev, uint32_t *cnt);
    void (*rsUrmaFreeEidList)(urma_eid_info_t *eidList);
    urma_status_t (*rsUrmaQueryDevice)(urma_device_t *dev, urma_device_attr_t *devAttr);
    urma_status_t (*rsUrmaGetEidByIp)(const urma_context_t *ctx, const urma_net_addr_t *netAddr, urma_eid_t *eid);
    urma_status_t (*rsUrmaGetIpByEid)(const urma_context_t *ctx, const urma_eid_t *eid, urma_net_addr_t *netAddr);
    urma_status_t (*rsUrmaGetSmac)(const urma_context_t *ctx, uint8_t *mac);
    urma_status_t (*rsUrmaGetDmac)(const urma_context_t *ctx, const urma_net_addr_t *netaddr, uint8_t *mac);
    urma_context_t *(*rsUrmaCreateContext)(urma_device_t *dev, uint32_t eidIndex);
    urma_status_t (*rsUrmaDeleteContext)(urma_context_t *ctx);
    urma_jfr_t *(*rsUrmaCreateJfr)(urma_context_t *ctx, urma_jfr_cfg_t *jfrCfg);
    urma_status_t (*rsUrmaDeleteJfr)(urma_jfr_t *jfr);
    urma_status_t (*rsUrmaDeleteJfrBatch)(urma_jfr_t **jfrArr, int jfrNum, urma_jfr_t **badJfr);
    urma_jfc_t *(*rsUrmaCreateJfc)(urma_context_t *ctx, urma_jfc_cfg_t *jfcCfg);
    urma_status_t (*rsUrmaModifyJfc)(urma_jfc_t *jfc, urma_jfc_attr_t *attr);
    urma_status_t (*rsUrmaDeleteJfc)(urma_jfc_t *jfc);
    urma_jetty_t *(*rsUrmaCreateJetty)(urma_context_t *ctx, urma_jetty_cfg_t *jettyCfg);
    urma_status_t (*rsUrmaModifyJetty)(urma_jetty_t *jetty, urma_jetty_attr_t *attr);
    urma_status_t (*rsUrmaQueryJetty)(urma_jetty_t *jetty, urma_jetty_cfg_t *cfg, urma_jetty_attr_t *attr);
    urma_status_t (*rsUrmaDeleteJetty)(urma_jetty_t *jetty);
    urma_status_t (*rsUrmaDeleteJettyBatch)(urma_jetty_t **jettyArr, int jettyNum, urma_jetty_t **badJetty);
    urma_target_jetty_t *(*rsUrmaImportJetty)(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue);
    urma_status_t (*rsUrmaUnimportJetty)(urma_target_jetty_t *tjetty);
    urma_status_t (*rsUrmaBindJetty)(urma_jetty_t *jetty, urma_target_jetty_t *tjetty);
    urma_status_t (*rsUrmaUnbindJetty)(urma_jetty_t *jetty);
    int (*rsUrmaFlushJetty)(urma_jetty_t *jetty, int crCnt, urma_cr_t *cr);
    urma_jfce_t *(*rsUrmaCreateJfce)(urma_context_t *ctx);
    urma_status_t (*rsUrmaDeleteJfce)(urma_jfce_t *jfce);
    urma_status_t (*rsUrmaGetAsyncEvent)(urma_context_t *ctx, urma_async_event_t *event);
    void (*rsUrmaAckAsyncEvent)(urma_async_event_t *event);
    urma_token_id_t *(*rsUrmaAllocTokenId)(urma_context_t *ctx);
    urma_status_t (*rsUrmaFreeTokenId)(urma_token_id_t *tokenId);
    urma_target_seg_t *(*rsUrmaRegisterSeg)(urma_context_t *ctx, urma_seg_cfg_t *segCfg);
    urma_status_t (*rsUrmaUnregisterSeg)(urma_target_seg_t *targetSeg);
    urma_target_seg_t *(*rsUrmaImportSeg)(urma_context_t *ctx, urma_seg_t *seg, urma_token_t *tokenValue, uint64_t addr,
        urma_import_seg_flag_t flag);
    urma_status_t (*rsUrmaUnimportSeg)(urma_target_seg_t *tseg);
    urma_status_t (*rsUrmaPostJettySendWr)(urma_jetty_t *jetty, urma_jfs_wr_t *wr, urma_jfs_wr_t **badWr);
    urma_status_t (*rsUrmaPostJettyRecvWr)(urma_jetty_t *jetty, urma_jfr_wr_t *wr, urma_jfr_wr_t **badWr);
    int (*rsUrmaPollJfc)(urma_jfc_t *jfc, int crCnt, urma_cr_t *cr);
    urma_status_t (*rsUrmaRearmJfc)(urma_jfc_t *jfc, bool solicitedOnly);
    int (*rsUrmaWaitJfc)(urma_jfce_t *jfce, uint32_t jfcCnt, int timeOut, urma_jfc_t *jfc[]);
    void (*rsUrmaAckJfc)(urma_jfc_t *jfc[], uint32_t nevents[], uint32_t jfcCnt);
    urma_device_t *(*rsUrmaGetDeviceByEid)(urma_eid_t eid, urma_transport_type_t type);
    urma_status_t (*rsUrmaUserCtl)(urma_context_t *ctx, urma_user_ctl_in_t *in, urma_user_ctl_out_t *out);
    urma_status_t (*rsUrmaGetTpList)(urma_context_t *ctx, urma_get_tp_cfg_t *cfg, uint32_t *tpCnt,
        urma_tp_info_t *tpList);
    urma_status_t (*rsUrmaGetTpAttr)(const urma_context_t *ctx, const uint64_t tpHandle, uint8_t *tpAttrCnt,
        uint32_t *tpAttrBitmap, urma_tp_attr_value_t *tpAttr);
    urma_status_t (*rsUrmaSetTpAttr)(const urma_context_t *ctx, const uint64_t tpHandle, const uint8_t tpAttrCnt,
        const uint32_t tpAttrBitmap, const urma_tp_attr_value_t *tpAttr);
    urma_target_jetty_t *(*rsUrmaImportJettyEx)(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue,
        urma_import_jetty_ex_cfg_t *cfg);
    urma_status_t (*rsUrmaAllocJetty)(urma_context_t *urmaCtx, urma_jetty_cfg_t *cfg, urma_jetty_t **jetty);
    urma_status_t (*rsUrmaSetJettyOpt)(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len);
    urma_status_t (*rsUrmaGetJettyOpt)(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len);
    urma_status_t (*rsUrmaActiveJetty)(urma_jetty_t *jetty);
    urma_status_t (*rsUrmaDeactiveJetty)(urma_jetty_t *jetty);
    urma_status_t (*rsUrmaFreeJetty)(urma_jetty_t *jetty);
    urma_status_t (*rsUrmaAllocJfc)(urma_context_t *urmaCtx, urma_jfc_cfg_t *cfg, urma_jfc_t **jfc);
    urma_status_t (*rsUrmaSetJfcOpt)(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len);
    urma_status_t (*rsUrmaActiveJfc)(urma_jfc_t *jfc);
    urma_status_t (*rsUrmaGetJfcOpt)(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len);
    urma_status_t (*rsUrmaDeactiveJfc)(urma_jfc_t *jfc);
    urma_status_t (*rsUrmaFreeJfc)(urma_jfc_t *jfc);
};

void RsUbApiDeinit(void);
int RsUbApiInit(void);

int RsUrmaInit(urma_init_attr_t *conf);
int RsUrmaUninit(void);
urma_device_t **RsUrmaGetDeviceList(int *numDevices);
urma_device_t *RsUrmaGetDeviceByEid(urma_eid_t eid, urma_transport_type_t type);
void RsUrmaFreeDeviceList(urma_device_t **deviceList);
urma_eid_info_t *RsUrmaGetEidList(urma_device_t *dev, uint32_t *cnt);
void RsUrmaFreeEidList(urma_eid_info_t *eidList);
int RsUrmaQueryDevice(urma_device_t *dev, urma_device_attr_t *devAttr);
int RsUrmaGetEidByIp(const urma_context_t *ctx, const urma_net_addr_t *netAddr, urma_eid_t *eid);
int RsUrmaGetIpByEid(const urma_context_t *ctx, const urma_eid_t *eid, urma_net_addr_t *netAddr);
int RsUrmaGetSmac(const urma_context_t *ctx, uint8_t *mac);
int RsUrmaGetDmac(const urma_context_t *ctx, const urma_net_addr_t *netaddr, uint8_t *mac);
urma_context_t *RsUrmaCreateContext(urma_device_t *dev, uint32_t eidIndex);
int RsUrmaDeleteContext(urma_context_t *ctx);
urma_jfr_t *RsUrmaCreateJfr(urma_context_t *ctx, urma_jfr_cfg_t *jfrCfg);
int RsUrmaDeleteJfr(urma_jfr_t *jfr);
int RsUrmaDeleteJfrBatch(urma_jfr_t **jfrArr, int jfrNum, urma_jfr_t **badJfr);
urma_jfc_t *RsUrmaCreateJfc(urma_context_t *ctx, urma_jfc_cfg_t *jfcCfg);
int RsUrmaModifyJfc(urma_jfc_t *jfc, urma_jfc_attr_t *attr);
int RsUrmaDeleteJfc(urma_jfc_t *jfc);
urma_jetty_t *RsUrmaCreateJetty(urma_context_t *ctx, urma_jetty_cfg_t *jettyCfg);
int RsUrmaModifyJetty(urma_jetty_t *jetty, urma_jetty_attr_t *attr);
int RsUrmaQueryJetty(urma_jetty_t *jetty, urma_jetty_cfg_t *cfg, urma_jetty_attr_t *attr);
int RsUrmaDeleteJetty(urma_jetty_t *jetty);
int RsUrmaDeleteJettyBatch(urma_jetty_t **jettyArr, int jettyNum, urma_jetty_t **badJetty);
urma_target_jetty_t *RsUrmaImportJetty(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue);
int RsUrmaUnimportJetty(urma_target_jetty_t *tjetty);
int RsUrmaBindJetty(urma_jetty_t *jetty, urma_target_jetty_t *tjetty);
int RsUrmaUnbindJetty(urma_jetty_t *jetty);
int RsUrmaFlushJetty(urma_jetty_t *jetty, int crCnt, urma_cr_t *cr);
urma_jfce_t *RsUrmaCreateJfce(urma_context_t *ctx);
int RsUrmaDeleteJfce(urma_jfce_t *jfce);
int RsUrmaGetAsyncEvent(urma_context_t *ctx, urma_async_event_t *event);
void RsUrmaAckAsyncEvent(urma_async_event_t *event);
urma_target_seg_t *RsUrmaRegisterSeg(urma_context_t *ctx, urma_seg_cfg_t *segCfg);
int RsUrmaUnregisterSeg(urma_target_seg_t *targetSeg);
urma_token_id_t *RsUrmaAllocTokenId(urma_context_t *ctx);
int RsUrmaFreeTokenId(urma_token_id_t *tokenId);
urma_target_seg_t *RsUrmaImportSeg(urma_context_t *ctx, urma_seg_t *seg, urma_token_t *tokenValue, uint64_t addr,
    urma_import_seg_flag_t flag);
int RsUrmaUnimportSeg(urma_target_seg_t *tseg);
int RsUrmaPostJettySendWr(urma_jetty_t *jetty, urma_jfs_wr_t *wr, urma_jfs_wr_t **badWr);
int RsUrmaPostJettyRecvWr(urma_jetty_t *jetty, urma_jfr_wr_t *wr, urma_jfr_wr_t **badWr);
int RsUrmaPollJfc(urma_jfc_t *jfc, int crCnt, urma_cr_t *cr);
int RsUrmaRearmJfc(urma_jfc_t *jfc, bool solicitedOnly);
int RsUrmaWaitJfc(urma_jfce_t *jfce, uint32_t jfcCnt, int timeOut, urma_jfc_t *jfc[]);
void RsUrmaAckJfc(urma_jfc_t *jfc[], uint32_t nevents[], uint32_t jfcCnt);
int RsUrmaUserCtl(urma_context_t *ctx, urma_user_ctl_in_t *in, urma_user_ctl_out_t *out);
int RsUrmaGetTpList(urma_context_t *ctx, urma_get_tp_cfg_t *cfg, uint32_t *tpCnt, urma_tp_info_t *tpList);
int RsUrmaGetTpAttr(const urma_context_t *ctx, const uint64_t tpHandle, uint8_t *tpAttrCnt, uint32_t *tpAttrBitmap,
    urma_tp_attr_value_t *tpAttr);
int RsUrmaSetTpAttr(const urma_context_t *ctx, const uint64_t tpHandle, const uint8_t tpAttrCnt,
    const uint32_t tpAttrBitmap, const urma_tp_attr_value_t *tpAttr);
urma_target_jetty_t *RsUrmaImportJettyEx(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue,
    urma_import_jetty_ex_cfg_t *cfg);
int RsUrmaAllocJetty(urma_context_t *urmaCtx, urma_jetty_cfg_t *cfg, urma_jetty_t **jetty);
int RsUrmaSetJettyOpt(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len);
int RsUrmaActiveJetty(urma_jetty_t *jetty);
int RsUrmaGetJettyOpt(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len);
int RsUrmaDeactiveJetty(urma_jetty_t *jetty);
int RsUrmaFreeJetty(urma_jetty_t *jetty);
int RsUrmaAllocJfc(urma_context_t *urmaCtx, urma_jfc_cfg_t *cfg, urma_jfc_t **jfc);
int RsUrmaSetJfcOpt(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len);
int RsUrmaActiveJfc(urma_jfc_t *jfc);
int RsUrmaGetJfcOpt(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len);
int RsUrmaDeactiveJfc(urma_jfc_t *jfc);
int RsUrmaFreeJfc(urma_jfc_t *jfc);

#endif // DL_URMA_FUNCTION_H
