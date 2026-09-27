/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <errno.h>
#include <urma_api.h>
#include "hccp_dl.h"
#include "network_comm.h"
#include "config_log.h"
#include "perf.h"
#include "dl_urma_function.h"

static pthread_mutex_t gUrmaApiLock = PTHREAD_MUTEX_INITIALIZER;
static int gUrmaApiRefcnt = 0;
void *gUrmaApiHandle = NULL;
#ifndef CA_CONFIG_LLT
struct RsUrmaOps gUrmaOps;
#else
struct RsUrmaOps gUrmaOps = {
    .rsUrmaInit = urma_init,
    .rsUrmaUninit = urma_uninit,
    .rsUrmaGetDeviceList = urma_get_device_list,
    .rsUrmaGetDeviceByEid = urma_get_device_by_eid,
    .rsUrmaFreeDeviceList = urma_free_device_list,
    .rsUrmaGetEidList = urma_get_eid_list,
    .rsUrmaFreeEidList = urma_free_eid_list,
    .rsUrmaQueryDevice = urma_query_device,
    .rsUrmaGetEidByIp = urma_get_eid_by_ip,
    .rsUrmaGetIpByEid = urma_get_ip_by_eid,
    .rsUrmaGetSmac = urma_get_smac,
    .rsUrmaGetDmac = urma_get_dmac,
    .rsUrmaCreateContext = urma_create_context,
    .rsUrmaDeleteContext = urma_delete_context,
    .rsUrmaCreateJfr = urma_create_jfr,
    .rsUrmaDeleteJfr = urma_delete_jfr,
    .rsUrmaDeleteJfrBatch = urma_delete_jfr_batch,
    .rsUrmaCreateJfc = urma_create_jfc,
    .rsUrmaModifyJfc = urma_modify_jfc,
    .rsUrmaDeleteJfc = urma_delete_jfc,
    .rsUrmaCreateJetty = urma_create_jetty,
    .rsUrmaModifyJetty = urma_modify_jetty,
    .rsUrmaQueryJetty = urma_query_jetty,
    .rsUrmaDeleteJetty = urma_delete_jetty,
    .rsUrmaDeleteJettyBatch = urma_delete_jetty_batch,
    .rsUrmaImportJetty = urma_import_jetty,
    .rsUrmaUnimportJetty = urma_unimport_jetty,
    .rsUrmaBindJetty = urma_bind_jetty,
    .rsUrmaUnbindJetty = urma_unbind_jetty,
    .rsUrmaFlushJetty = urma_flush_jetty,
    .rsUrmaCreateJfce = urma_create_jfce,
    .rsUrmaDeleteJfce = urma_delete_jfce,
    .rsUrmaGetAsyncEvent = urma_get_async_event,
    .rsUrmaAckAsyncEvent = urma_ack_async_event,
    .rsUrmaAllocTokenId = urma_alloc_token_id,
    .rsUrmaFreeTokenId = urma_free_token_id,
    .rsUrmaRegisterSeg = urma_register_seg,
    .rsUrmaUnregisterSeg = urma_unregister_seg,
    .rsUrmaImportSeg = urma_import_seg,
    .rsUrmaUnimportSeg = urma_unimport_seg,
    .rsUrmaPostJettySendWr = urma_post_jetty_send_wr,
    .rsUrmaPostJettyRecvWr = urma_post_jetty_recv_wr,
    .rsUrmaPollJfc = urma_poll_jfc,
    .rsUrmaRearmJfc = urma_rearm_jfc,
    .rsUrmaWaitJfc = urma_wait_jfc,
    .rsUrmaAckJfc = urma_ack_jfc,
    .rsUrmaUserCtl = urma_user_ctl,
    .rsUrmaGetTpList = urma_get_tp_list,
    .rsUrmaGetTpAttr = urma_get_tp_attr,
    .rsUrmaSetTpAttr = urma_set_tp_attr,
    .rsUrmaImportJettyEx = urma_import_jetty_ex,
    .rsUrmaAllocJetty = urma_alloc_jetty,
    .rsUrmaSetJettyOpt = urma_set_jetty_opt,
    .rsUrmaActiveJetty = urma_active_jetty,
    .rsUrmaGetJettyOpt = urma_get_jetty_opt,
    .rsUrmaDeactiveJetty = urma_deactive_jetty,
    .rsUrmaFreeJetty = urma_free_jetty,
    .rsUrmaAllocJfc = urma_alloc_jfc,
    .rsUrmaSetJfcOpt = urma_set_jfc_opt,
    .rsUrmaActiveJfc = urma_active_jfc,
    .rsUrmaGetJfcOpt = urma_get_jfc_opt,
    .rsUrmaDeactiveJfc = urma_deactive_jfc,
    .rsUrmaFreeJfc = urma_free_jfc,
};
#endif

void RsUbApiDeinit(void)
{
    if (gUrmaApiHandle != NULL) {
        (void)HccpDlclose(gUrmaApiHandle);
        gUrmaApiHandle = NULL;
    }
    return;
}

STATIC int RsUrmaDeviceApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gUrmaOps.rsUrmaInit = (urma_status_t(*)(urma_init_attr_t *))HccpDlsym(gUrmaApiHandle, "urma_init");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaInit, "urma_init");

    gUrmaOps.rsUrmaUninit = (urma_status_t(*)(void))HccpDlsym(gUrmaApiHandle, "urma_uninit");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUninit, "urma_uninit");

    gUrmaOps.rsUrmaGetDeviceList = (urma_device_t * *(*)(int *)) HccpDlsym(gUrmaApiHandle, "urma_get_device_list");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetDeviceList, "urma_get_device_list");

    gUrmaOps.rsUrmaGetDeviceByEid = (urma_device_t * (*)(urma_eid_t, urma_transport_type_t))
        HccpDlsym(gUrmaApiHandle, "urma_get_device_by_eid");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetDeviceByEid, "urma_get_device_by_eid");

    gUrmaOps.rsUrmaFreeDeviceList = (void (*)(urma_device_t **))HccpDlsym(gUrmaApiHandle, "urma_free_device_list");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFreeDeviceList, "urma_free_device_list");

    gUrmaOps.rsUrmaGetEidList = (urma_eid_info_t * (*)(urma_device_t *, uint32_t *))
        HccpDlsym(gUrmaApiHandle, "urma_get_eid_list");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetEidList, "urma_get_eid_list");

    gUrmaOps.rsUrmaFreeEidList = (void (*)(urma_eid_info_t *))HccpDlsym(gUrmaApiHandle, "urma_free_eid_list");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFreeEidList, "urma_free_eid_list");

    gUrmaOps.rsUrmaQueryDevice = (urma_status_t(*)(urma_device_t *, urma_device_attr_t *))HccpDlsym(gUrmaApiHandle,
        "urma_query_device");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaQueryDevice, "urma_query_device");

    gUrmaOps.rsUrmaGetEidByIp = (urma_status_t(*)(const urma_context_t *, const urma_net_addr_t *,
        urma_eid_t *))HccpDlsym(gUrmaApiHandle, "urma_get_eid_by_ip");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetEidByIp, "urma_get_eid_by_ip");

    gUrmaOps.rsUrmaGetIpByEid = (urma_status_t(*)(const urma_context_t *, const urma_eid_t *,
        urma_net_addr_t *))HccpDlsym(gUrmaApiHandle, "urma_get_ip_by_eid");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetIpByEid, "urma_get_ip_by_eid");

    gUrmaOps.rsUrmaGetSmac = (urma_status_t(*)(const urma_context_t *, uint8_t *))HccpDlsym(gUrmaApiHandle,
        "urma_get_smac");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetSmac, "urma_get_smac");

    gUrmaOps.rsUrmaGetDmac = (urma_status_t(*)(const urma_context_t *, const urma_net_addr_t *,
        uint8_t *))HccpDlsym(gUrmaApiHandle, "urma_get_dmac");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetDmac, "urma_get_dmac");

    gUrmaOps.rsUrmaCreateContext = (urma_context_t * (*)(urma_device_t *, uint32_t))
        HccpDlsym(gUrmaApiHandle, "urma_create_context");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaCreateContext, "urma_create_context");

    gUrmaOps.rsUrmaDeleteContext = (urma_status_t(*)(urma_context_t *))HccpDlsym(gUrmaApiHandle, "urma_delete_context");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteContext, "urma_delete_context");
#endif
    return 0;
}

STATIC int RsUrmaJettyApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gUrmaOps.rsUrmaCreateJfr = (urma_jfr_t * (*)(urma_context_t *, urma_jfr_cfg_t *))
        HccpDlsym(gUrmaApiHandle, "urma_create_jfr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaCreateJfr, "urma_create_jfr");

    gUrmaOps.rsUrmaDeleteJfr = (urma_status_t(*)(urma_jfr_t *))HccpDlsym(gUrmaApiHandle, "urma_delete_jfr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJfr, "urma_delete_jfr");

    gUrmaOps.rsUrmaDeleteJfrBatch = (urma_status_t(*)(urma_jfr_t * *jfrArr, int jfrNum, urma_jfr_t **badJfr))
        HccpDlsym(gUrmaApiHandle, "urma_delete_jfr_batch");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJfrBatch, "urma_delete_jfr_batch");

    gUrmaOps.rsUrmaCreateJetty = (urma_jetty_t * (*)(urma_context_t *, urma_jetty_cfg_t *))
        HccpDlsym(gUrmaApiHandle, "urma_create_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaCreateJetty, "urma_create_jetty");

    gUrmaOps.rsUrmaModifyJetty = (urma_status_t(*)(urma_jetty_t *, urma_jetty_attr_t *))HccpDlsym(gUrmaApiHandle,
        "urma_modify_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaModifyJetty, "urma_modify_jetty");

    gUrmaOps.rsUrmaQueryJetty = (urma_status_t(*)(urma_jetty_t *, urma_jetty_cfg_t *,
        urma_jetty_attr_t *))HccpDlsym(gUrmaApiHandle, "urma_query_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaQueryJetty, "urma_query_jetty");

    gUrmaOps.rsUrmaDeleteJetty = (urma_status_t(*)(urma_jetty_t *))HccpDlsym(gUrmaApiHandle, "urma_delete_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJetty, "urma_delete_jetty");

    gUrmaOps.rsUrmaDeleteJettyBatch = (urma_status_t(*)(urma_jetty_t * *jettyArr, int jettyNum,
        urma_jetty_t **badJetty)) HccpDlsym(gUrmaApiHandle, "urma_delete_jetty_batch");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJettyBatch, "urma_delete_jetty_batch");

    gUrmaOps.rsUrmaImportJetty = (urma_target_jetty_t * (*)(urma_context_t *, urma_rjetty_t *, urma_token_t *))
        HccpDlsym(gUrmaApiHandle, "urma_import_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaImportJetty, "urma_import_jetty");

    gUrmaOps.rsUrmaUnimportJetty = (urma_status_t(*)(urma_target_jetty_t *))HccpDlsym(gUrmaApiHandle,
        "urma_unimport_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUnimportJetty, "urma_unimport_jetty");

    gUrmaOps.rsUrmaBindJetty = (urma_status_t(*)(urma_jetty_t *, urma_target_jetty_t *))HccpDlsym(gUrmaApiHandle,
        "urma_bind_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaBindJetty, "urma_bind_jetty");

    gUrmaOps.rsUrmaUnbindJetty = (urma_status_t(*)(urma_jetty_t *))HccpDlsym(gUrmaApiHandle, "urma_unbind_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUnbindJetty, "urma_unbind_jetty");

    gUrmaOps.rsUrmaFlushJetty = (int (*)(urma_jetty_t *, int, urma_cr_t *))HccpDlsym(gUrmaApiHandle,
        "urma_flush_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFlushJetty, "urma_flush_jetty");

    gUrmaOps.rsUrmaAllocJetty = (urma_status_t(*)(urma_context_t *, urma_jetty_cfg_t *,
        urma_jetty_t **))HccpDlsym(gUrmaApiHandle, "urma_alloc_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaAllocJetty, "urma_alloc_jetty");

    gUrmaOps.rsUrmaSetJettyOpt = (urma_status_t(*)(urma_jetty_t *, uint64_t, void *, uint32_t))HccpDlsym(gUrmaApiHandle,
        "urma_set_jetty_opt");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaSetJettyOpt, "urma_set_jetty_opt");

    gUrmaOps.rsUrmaActiveJetty = (urma_status_t(*)(urma_jetty_t *))HccpDlsym(gUrmaApiHandle, "urma_active_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaActiveJetty, "urma_active_jetty");

    gUrmaOps.rsUrmaGetJettyOpt = (urma_status_t(*)(urma_jetty_t *, uint64_t, void *, uint32_t))HccpDlsym(gUrmaApiHandle,
        "urma_get_jetty_opt");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetJettyOpt, "urma_get_jetty_opt");

    gUrmaOps.rsUrmaDeactiveJetty = (urma_status_t(*)(urma_jetty_t *))HccpDlsym(gUrmaApiHandle, "urma_deactive_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeactiveJetty, "urma_deactive_jetty");

    gUrmaOps.rsUrmaFreeJetty = (urma_status_t(*)(urma_jetty_t *))HccpDlsym(gUrmaApiHandle, "urma_free_jetty");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFreeJetty, "urma_free_jetty");
#endif
    return 0;
}

STATIC int RsUrmaJfcApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gUrmaOps.rsUrmaCreateJfc = (urma_jfc_t * (*)(urma_context_t *, urma_jfc_cfg_t *))
        HccpDlsym(gUrmaApiHandle, "urma_create_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaCreateJfc, "urma_create_jfc");

    gUrmaOps.rsUrmaModifyJfc = (urma_status_t(*)(urma_jfc_t *, urma_jfc_attr_t *))HccpDlsym(gUrmaApiHandle,
        "urma_modify_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaModifyJfc, "urma_modify_jfc");

    gUrmaOps.rsUrmaDeleteJfc = (urma_status_t(*)(urma_jfc_t *))HccpDlsym(gUrmaApiHandle, "urma_delete_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJfc, "urma_delete_jfc");

    gUrmaOps.rsUrmaCreateJfce = (urma_jfce_t * (*)(urma_context_t *)) HccpDlsym(gUrmaApiHandle, "urma_create_jfce");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaCreateJfce, "urma_create_jfce");

    gUrmaOps.rsUrmaDeleteJfce = (urma_status_t(*)(urma_jfce_t *))HccpDlsym(gUrmaApiHandle, "urma_delete_jfce");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeleteJfce, "urma_delete_jfce");

    gUrmaOps.rsUrmaGetAsyncEvent = (urma_status_t(*)(urma_context_t *, urma_async_event_t *))HccpDlsym(gUrmaApiHandle,
        "urma_get_async_event");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetAsyncEvent, "urma_get_async_event");

    gUrmaOps.rsUrmaAckAsyncEvent = (void (*)(urma_async_event_t *))HccpDlsym(gUrmaApiHandle, "urma_ack_async_event");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaAckAsyncEvent, "urma_ack_async_event");

    gUrmaOps.rsUrmaAllocJfc = (urma_status_t(*)(urma_context_t *, urma_jfc_cfg_t *,
        urma_jfc_t **))HccpDlsym(gUrmaApiHandle, "urma_alloc_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaAllocJfc, "urma_alloc_jfc");

    gUrmaOps.rsUrmaSetJfcOpt = (urma_status_t(*)(urma_jfc_t *, uint64_t, void *, uint32_t))HccpDlsym(gUrmaApiHandle,
        "urma_set_jfc_opt");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaSetJfcOpt, "urma_set_jfc_opt");

    gUrmaOps.rsUrmaActiveJfc = (urma_status_t(*)(urma_jfc_t *))HccpDlsym(gUrmaApiHandle, "urma_active_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaActiveJfc, "urma_active_jfc");

    gUrmaOps.rsUrmaGetJfcOpt = (urma_status_t(*)(urma_jfc_t *, uint64_t, void *, uint32_t))HccpDlsym(gUrmaApiHandle,
        "urma_get_jfc_opt");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetJfcOpt, "urma_get_jfc_opt");

    gUrmaOps.rsUrmaDeactiveJfc = (urma_status_t(*)(urma_jfc_t *))HccpDlsym(gUrmaApiHandle, "urma_deactive_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaDeactiveJfc, "urma_deactive_jfc");

    gUrmaOps.rsUrmaFreeJfc = (urma_status_t(*)(urma_jfc_t *))HccpDlsym(gUrmaApiHandle, "urma_free_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFreeJfc, "urma_free_jfc");
#endif
    return 0;
}

STATIC int RsUrmaSegmentApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gUrmaOps.rsUrmaAllocTokenId = (urma_token_id_t * (*)(urma_context_t *))
        HccpDlsym(gUrmaApiHandle, "urma_alloc_token_id");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaAllocTokenId, "urma_alloc_token_id");

    gUrmaOps.rsUrmaFreeTokenId = (urma_status_t(*)(urma_token_id_t *))HccpDlsym(gUrmaApiHandle, "urma_free_token_id");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaFreeTokenId, "urma_free_token_id");

    gUrmaOps.rsUrmaRegisterSeg = (urma_target_seg_t * (*)(urma_context_t *, urma_seg_cfg_t *))
        HccpDlsym(gUrmaApiHandle, "urma_register_seg");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaRegisterSeg, "urma_register_seg");

    gUrmaOps.rsUrmaUnregisterSeg = (urma_status_t(*)(urma_target_seg_t *))HccpDlsym(gUrmaApiHandle,
        "urma_unregister_seg");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUnregisterSeg, "urma_unregister_seg");

    gUrmaOps.rsUrmaImportSeg = (urma_target_seg_t *
                                (*)(urma_context_t *, urma_seg_t *, urma_token_t *, uint64_t, urma_import_seg_flag_t))
        HccpDlsym(gUrmaApiHandle, "urma_import_seg");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaImportSeg, "urma_import_seg");

    gUrmaOps.rsUrmaUnimportSeg = (urma_status_t(*)(urma_target_seg_t *))HccpDlsym(gUrmaApiHandle, "urma_unimport_seg");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUnimportSeg, "urma_unimport_seg");
#endif
    return 0;
}

STATIC int RsUrmaDataApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gUrmaOps.rsUrmaPostJettySendWr = (urma_status_t(*)(urma_jetty_t *, urma_jfs_wr_t *,
        urma_jfs_wr_t **))HccpDlsym(gUrmaApiHandle, "urma_post_jetty_send_wr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaPostJettySendWr, "urma_post_jetty_send_wr");

    gUrmaOps.rsUrmaPostJettyRecvWr = (urma_status_t(*)(urma_jetty_t *, urma_jfr_wr_t *,
        urma_jfr_wr_t **))HccpDlsym(gUrmaApiHandle, "urma_post_jetty_recv_wr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaPostJettyRecvWr, "urma_post_jetty_recv_wr");

    gUrmaOps.rsUrmaPollJfc = (int (*)(urma_jfc_t *, int, urma_cr_t *))HccpDlsym(gUrmaApiHandle, "urma_poll_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaPollJfc, "urma_poll_jfc");

    gUrmaOps.rsUrmaRearmJfc = (urma_status_t(*)(urma_jfc_t *, bool))HccpDlsym(gUrmaApiHandle, "urma_rearm_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaRearmJfc, "urma_rearm_jfc");

    gUrmaOps.rsUrmaWaitJfc = (int (*)(urma_jfce_t *, uint32_t, int, urma_jfc_t *[]))HccpDlsym(gUrmaApiHandle,
        "urma_wait_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaWaitJfc, "urma_wait_jfc");

    gUrmaOps.rsUrmaAckJfc = (void (*)(urma_jfc_t *[], uint32_t[], uint32_t))HccpDlsym(gUrmaApiHandle, "urma_ack_jfc");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaAckJfc, "urma_ack_jfc");

    gUrmaOps.rsUrmaUserCtl = (urma_status_t(*)(urma_context_t *, urma_user_ctl_in_t *,
        urma_user_ctl_out_t *))HccpDlsym(gUrmaApiHandle, "urma_user_ctl");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaUserCtl, "urma_user_ctl");

    gUrmaOps.rsUrmaGetTpList = (urma_status_t(*)(urma_context_t *, urma_get_tp_cfg_t *, uint32_t *,
        urma_tp_info_t *))HccpDlsym(gUrmaApiHandle, "urma_get_tp_list");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetTpList, "urma_get_tp_list");

    gUrmaOps.rsUrmaGetTpAttr = (urma_status_t(*)(const urma_context_t *, const uint64_t, uint8_t *, uint32_t *,
        urma_tp_attr_value_t *))HccpDlsym(gUrmaApiHandle, "urma_get_tp_attr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaGetTpAttr, "urma_get_tp_attr");

    gUrmaOps.rsUrmaSetTpAttr = (urma_status_t(*)(const urma_context_t *, const uint64_t, const uint8_t, const uint32_t,
        const urma_tp_attr_value_t *))HccpDlsym(gUrmaApiHandle, "urma_set_tp_attr");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaSetTpAttr, "urma_set_tp_attr");

    gUrmaOps.rsUrmaImportJettyEx = (urma_target_jetty_t * (*)(urma_context_t *, urma_rjetty_t *, urma_token_t *,
                                                              urma_import_jetty_ex_cfg_t *))
        HccpDlsym(gUrmaApiHandle, "urma_import_jetty_ex");
    DL_API_RET_IS_NULL_CHECK(gUrmaOps.rsUrmaImportJettyEx, "urma_import_jetty_ex");
#endif
    return 0;
}

STATIC int RsOpenUrmaSo(void)
{
    pthread_mutex_lock(&gUrmaApiLock);
#ifndef CA_CONFIG_LLT
    if (gUrmaApiHandle == NULL) {
        gUrmaApiHandle = HccpDlopen("liburma.so.0", RTLD_NOW);
        if (gUrmaApiHandle != NULL) {
            goto out;
        }

        gUrmaApiHandle = HccpDlopen("/lib64/liburma.so.0", RTLD_NOW);
        if (gUrmaApiHandle != NULL) {
            goto out;
        }
        pthread_mutex_unlock(&gUrmaApiLock);
        return -EINVAL;
    } else {
        hccp_run_info("urma_api dlopen again, gUrmaApiRefcnt:%d", gUrmaApiRefcnt + 1);
    }
out:
#endif
    gUrmaApiRefcnt++;
    pthread_mutex_unlock(&gUrmaApiLock);
    return 0;
}

STATIC void RsCloseUrmaSo(void)
{
    pthread_mutex_lock(&gUrmaApiLock);
#ifndef CA_CONFIG_LLT
    if (gUrmaApiHandle != NULL) {
        gUrmaApiRefcnt--;
        if (gUrmaApiRefcnt > 0) {
            goto out;
        }

        hccp_run_info("dlclose urma_api, gUrmaApiRefcnt:%d", gUrmaApiRefcnt);
        (void)HccpDlclose(gUrmaApiHandle);
        gUrmaApiHandle = NULL;
        gUrmaApiRefcnt = 0;
    }
out:
#endif
    pthread_mutex_unlock(&gUrmaApiLock);
    return;
}

STATIC int RsUrmaApiInit(void)
{
    int ret;

    ret = RsOpenUrmaSo();
    CHK_PRT_RETURN(ret,
        hccp_err("HccpDlopen[liburma.so] failed! ret=[%d], "
                 "Please check network adapter driver has been installed",
            ret),
        ret);

    ret = RsUrmaDeviceApiInit();
    if (ret != 0) {
        hccp_err("[rs_urma_device_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseUrmaSo();
        return ret;
    }

    ret = RsUrmaJettyApiInit();
    if (ret != 0) {
        hccp_err("[rs_urma_jetty_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseUrmaSo();
        return ret;
    }

    ret = RsUrmaJfcApiInit();
    if (ret != 0) {
        hccp_err("[rs_urma_jfc_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseUrmaSo();
        return ret;
    }

    ret = RsUrmaSegmentApiInit();
    if (ret != 0) {
        hccp_err("[rs_urma_segment_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseUrmaSo();
        return ret;
    }

    ret = RsUrmaDataApiInit();
    if (ret != 0) {
        hccp_err("[rs_urma_data_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseUrmaSo();
        return ret;
    }

    return 0;
}

int RsUbApiInit(void)
{
    int ret;

    ret = RsUrmaApiInit();
    CHK_PRT_RETURN(ret, hccp_err("rs_urma_api_init failed! ret=[%d]", ret), ret);

    return 0;
}

int RsUrmaInit(urma_init_attr_t *conf)
{
    if (gUrmaOps.rsUrmaInit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_init is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaInit(conf)));
}

int RsUrmaUninit(void)
{
    if (gUrmaOps.rsUrmaUninit == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_uninit is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUninit()));
}

urma_device_t **RsUrmaGetDeviceList(int *numDevices)
{
    if (gUrmaOps.rsUrmaGetDeviceList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_device_list is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaGetDeviceList(numDevices));
}

urma_device_t *RsUrmaGetDeviceByEid(urma_eid_t eid, urma_transport_type_t type)
{
    if (gUrmaOps.rsUrmaGetDeviceByEid == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_device_by_eid is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaGetDeviceByEid(eid, type));
}

void RsUrmaFreeDeviceList(urma_device_t **deviceList)
{
    if (gUrmaOps.rsUrmaFreeDeviceList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_free_device_list is null");
        return;
#endif
    }
    PERF_TRACE_VOID(RDMA_OP, gUrmaOps.rsUrmaFreeDeviceList(deviceList));
}

urma_eid_info_t *RsUrmaGetEidList(urma_device_t *dev, uint32_t *cnt)
{
    if (gUrmaOps.rsUrmaGetEidList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_eid_list is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaGetEidList(dev, cnt));
}

void RsUrmaFreeEidList(urma_eid_info_t *eidList)
{
    if (gUrmaOps.rsUrmaFreeEidList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_free_eid_list is null");
        return;
#endif
    }
    PERF_TRACE_VOID(RDMA_OP, gUrmaOps.rsUrmaFreeEidList(eidList));
}

int RsUrmaQueryDevice(urma_device_t *dev, urma_device_attr_t *devAttr)
{
    if (gUrmaOps.rsUrmaQueryDevice == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_query_device is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaQueryDevice(dev, devAttr)));
}

int RsUrmaGetEidByIp(const urma_context_t *ctx, const urma_net_addr_t *netAddr, urma_eid_t *eid)
{
    if (gUrmaOps.rsUrmaGetEidByIp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_eid_by_ip is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetEidByIp(ctx, netAddr, eid)));
}

int RsUrmaGetIpByEid(const urma_context_t *ctx, const urma_eid_t *eid, urma_net_addr_t *netAddr)
{
    if (gUrmaOps.rsUrmaGetIpByEid == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsUrmaGetIpByEid is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetIpByEid(ctx, eid, netAddr)));
}

int RsUrmaGetSmac(const urma_context_t *ctx, uint8_t *mac)
{
    if (gUrmaOps.rsUrmaGetSmac == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsUrmaGetSmac is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetSmac(ctx, mac)));
}

int RsUrmaGetDmac(const urma_context_t *ctx, const urma_net_addr_t *netaddr, uint8_t *mac)
{
    if (gUrmaOps.rsUrmaGetDmac == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rsUrmaGetDmac is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetDmac(ctx, netaddr, mac)));
}

urma_context_t *RsUrmaCreateContext(urma_device_t *dev, uint32_t eidIndex)
{
    if (gUrmaOps.rsUrmaCreateContext == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_create_context is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaCreateContext(dev, eidIndex));
}

int RsUrmaDeleteContext(urma_context_t *ctx)
{
    if (gUrmaOps.rsUrmaDeleteContext == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_context is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteContext(ctx)));
}

urma_jfr_t *RsUrmaCreateJfr(urma_context_t *ctx, urma_jfr_cfg_t *jfrCfg)
{
    if (gUrmaOps.rsUrmaCreateJfr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_create_jfr is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaCreateJfr(ctx, jfrCfg));
}

int RsUrmaDeleteJfr(urma_jfr_t *jfr)
{
    if (gUrmaOps.rsUrmaDeleteJfr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jfr is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJfr(jfr)));
}

urma_jfc_t *RsUrmaCreateJfc(urma_context_t *ctx, urma_jfc_cfg_t *jfcCfg)
{
    if (gUrmaOps.rsUrmaCreateJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_create_jfc is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaCreateJfc(ctx, jfcCfg));
}

int RsUrmaModifyJfc(urma_jfc_t *jfc, urma_jfc_attr_t *attr)
{
    if (gUrmaOps.rsUrmaModifyJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_modify_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaModifyJfc(jfc, attr)));
}

int RsUrmaDeleteJfc(urma_jfc_t *jfc)
{
    if (gUrmaOps.rsUrmaDeleteJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJfc(jfc)));
}

urma_jetty_t *RsUrmaCreateJetty(urma_context_t *ctx, urma_jetty_cfg_t *jettyCfg)
{
    if (gUrmaOps.rsUrmaCreateJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_create_jetty is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaCreateJetty(ctx, jettyCfg));
}

int RsUrmaModifyJetty(urma_jetty_t *jetty, urma_jetty_attr_t *attr)
{
    if (gUrmaOps.rsUrmaModifyJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_modify_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaModifyJetty(jetty, attr)));
}

int RsUrmaQueryJetty(urma_jetty_t *jetty, urma_jetty_cfg_t *cfg, urma_jetty_attr_t *attr)
{
    if (gUrmaOps.rsUrmaQueryJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_query_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaQueryJetty(jetty, cfg, attr)));
}

int RsUrmaDeleteJetty(urma_jetty_t *jetty)
{
    if (gUrmaOps.rsUrmaDeleteJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJetty(jetty)));
}

urma_target_jetty_t *RsUrmaImportJetty(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue)
{
    if (gUrmaOps.rsUrmaImportJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_import_jetty is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaImportJetty(ctx, rjetty, tokenValue));
}

int RsUrmaUnimportJetty(urma_target_jetty_t *tjetty)
{
    if (gUrmaOps.rsUrmaUnimportJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_unimport_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUnimportJetty(tjetty)));
}

int RsUrmaBindJetty(urma_jetty_t *jetty, urma_target_jetty_t *tjetty)
{
    if (gUrmaOps.rsUrmaBindJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_bind_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaBindJetty(jetty, tjetty)));
}

int RsUrmaUnbindJetty(urma_jetty_t *jetty)
{
    if (gUrmaOps.rsUrmaUnbindJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_unbind_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUnbindJetty(jetty)));
}

int RsUrmaFlushJetty(urma_jetty_t *jetty, int crCnt, urma_cr_t *cr)
{
    if (gUrmaOps.rsUrmaFlushJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_flush_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaFlushJetty(jetty, crCnt, cr));
}

urma_jfce_t *RsUrmaCreateJfce(urma_context_t *ctx)
{
    if (gUrmaOps.rsUrmaCreateJfce == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_create_jfce is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaCreateJfce(ctx));
}

int RsUrmaDeleteJfce(urma_jfce_t *jfce)
{
    if (gUrmaOps.rsUrmaDeleteJfce == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jfce is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJfce(jfce)));
}

int RsUrmaGetAsyncEvent(urma_context_t *ctx, urma_async_event_t *event)
{
    if (gUrmaOps.rsUrmaGetAsyncEvent == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_async_event is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetAsyncEvent(ctx, event)));
}

void RsUrmaAckAsyncEvent(urma_async_event_t *event)
{
    if (gUrmaOps.rsUrmaAckAsyncEvent == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_ack_async_event is null");
        return;
#endif
    }
    PERF_TRACE_VOID(RDMA_OP, gUrmaOps.rsUrmaAckAsyncEvent(event));
}

urma_token_id_t *RsUrmaAllocTokenId(urma_context_t *ctx)
{
    if (gUrmaOps.rsUrmaAllocTokenId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_alloc_token_id is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaAllocTokenId(ctx));
}

int RsUrmaFreeTokenId(urma_token_id_t *tokenId)
{
    if (gUrmaOps.rsUrmaFreeTokenId == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_free_token_id is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaFreeTokenId(tokenId)));
}

urma_target_seg_t *RsUrmaRegisterSeg(urma_context_t *ctx, urma_seg_cfg_t *segCfg)
{
    if (gUrmaOps.rsUrmaRegisterSeg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_register_seg is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaRegisterSeg(ctx, segCfg));
}

int RsUrmaUnregisterSeg(urma_target_seg_t *targetSeg)
{
    if (gUrmaOps.rsUrmaUnregisterSeg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_unregister_seg is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUnregisterSeg(targetSeg)));
}

urma_target_seg_t *RsUrmaImportSeg(urma_context_t *ctx, urma_seg_t *seg, urma_token_t *tokenValue, uint64_t addr,
    urma_import_seg_flag_t flag)
{
    if (gUrmaOps.rsUrmaImportSeg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_import_seg is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaImportSeg(ctx, seg, tokenValue, addr, flag));
}

int RsUrmaUnimportSeg(urma_target_seg_t *tseg)
{
    if (gUrmaOps.rsUrmaUnimportSeg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_unimport_seg is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUnimportSeg(tseg)));
}

int RsUrmaPostJettySendWr(urma_jetty_t *jetty, urma_jfs_wr_t *wr, urma_jfs_wr_t **badWr)
{
    if (gUrmaOps.rsUrmaPostJettySendWr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_post_jetty_send_wr is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaPostJettySendWr(jetty, wr, badWr)));
}

int RsUrmaPostJettyRecvWr(urma_jetty_t *jetty, urma_jfr_wr_t *wr, urma_jfr_wr_t **badWr)
{
    if (gUrmaOps.rsUrmaPostJettyRecvWr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_post_jetty_recv_wr is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaPostJettyRecvWr(jetty, wr, badWr)));
}

int RsUrmaPollJfc(urma_jfc_t *jfc, int crCnt, urma_cr_t *cr)
{
    if (gUrmaOps.rsUrmaPollJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_poll_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaPollJfc(jfc, crCnt, cr));
}

int RsUrmaRearmJfc(urma_jfc_t *jfc, bool solicitedOnly)
{
    if (gUrmaOps.rsUrmaRearmJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_rearm_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaRearmJfc(jfc, solicitedOnly)));
}

int RsUrmaWaitJfc(urma_jfce_t *jfce, uint32_t jfcCnt, int timeOut, urma_jfc_t *jfc[])
{
    if (gUrmaOps.rsUrmaWaitJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_wait_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaWaitJfc(jfce, jfcCnt, timeOut, jfc));
}

void RsUrmaAckJfc(urma_jfc_t *jfc[], uint32_t nevents[], uint32_t jfcCnt)
{
    if (gUrmaOps.rsUrmaAckJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_ack_jfc is null");
        return;
#endif
    }
    PERF_TRACE_VOID(RDMA_OP, gUrmaOps.rsUrmaAckJfc(jfc, nevents, jfcCnt));
}

int RsUrmaUserCtl(urma_context_t *ctx, urma_user_ctl_in_t *in, urma_user_ctl_out_t *out)
{
    if (gUrmaOps.rsUrmaUserCtl == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_user_ctl is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaUserCtl(ctx, in, out)));
}

int RsUrmaGetTpList(urma_context_t *ctx, urma_get_tp_cfg_t *cfg, uint32_t *tpCnt, urma_tp_info_t *tpList)
{
    if (gUrmaOps.rsUrmaGetTpList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_tp_list is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetTpList(ctx, cfg, tpCnt, tpList)));
}

int RsUrmaGetTpAttr(const urma_context_t *ctx, const uint64_t tpHandle, uint8_t *tpAttrCnt, uint32_t *tpAttrBitmap,
    urma_tp_attr_value_t *tpAttr)
{
    if (gUrmaOps.rsUrmaGetTpAttr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_tp_attr is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetTpAttr(ctx, tpHandle, tpAttrCnt, tpAttrBitmap, tpAttr)));
}

int RsUrmaSetTpAttr(const urma_context_t *ctx, const uint64_t tpHandle, const uint8_t tpAttrCnt,
    const uint32_t tpAttrBitmap, const urma_tp_attr_value_t *tpAttr)
{
    if (gUrmaOps.rsUrmaSetTpAttr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_set_tp_attr is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaSetTpAttr(ctx, tpHandle, tpAttrCnt, tpAttrBitmap, tpAttr)));
}

urma_target_jetty_t *RsUrmaImportJettyEx(urma_context_t *ctx, urma_rjetty_t *rjetty, urma_token_t *tokenValue,
    urma_import_jetty_ex_cfg_t *cfg)
{
    if (gUrmaOps.rsUrmaImportJettyEx == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_import_jetty_ex is null");
        return NULL;
#endif
    }
    return PERF_TRACE(RDMA_OP, gUrmaOps.rsUrmaImportJettyEx(ctx, rjetty, tokenValue, cfg));
}

int RsUrmaDeleteJettyBatch(urma_jetty_t **jettyArr, int jettyNum, urma_jetty_t **badJetty)
{
    if (gUrmaOps.rsUrmaDeleteJettyBatch == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jetty_batch is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJettyBatch(jettyArr, jettyNum, badJetty)));
}

int RsUrmaDeleteJfrBatch(urma_jfr_t **jfrArr, int jfrNum, urma_jfr_t **badJfr)
{
    if (gUrmaOps.rsUrmaDeleteJfrBatch == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_delete_jfr_batch is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeleteJfrBatch(jfrArr, jfrNum, badJfr)));
}

int RsUrmaAllocJetty(urma_context_t *urmaCtx, urma_jetty_cfg_t *cfg, urma_jetty_t **jetty)
{
    if (gUrmaOps.rsUrmaAllocJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_alloc_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaAllocJetty(urmaCtx, cfg, jetty)));
}

int RsUrmaSetJettyOpt(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len)
{
    if (gUrmaOps.rsUrmaSetJettyOpt == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_set_jetty_opt is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaSetJettyOpt(jetty, opt, buf, len)));
}

int RsUrmaActiveJetty(urma_jetty_t *jetty)
{
    if (gUrmaOps.rsUrmaActiveJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_active_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaActiveJetty(jetty)));
}

int RsUrmaGetJettyOpt(urma_jetty_t *jetty, uint64_t opt, void *buf, uint32_t len)
{
    if (gUrmaOps.rsUrmaGetJettyOpt == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_jetty_opt is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetJettyOpt(jetty, opt, buf, len)));
}

int RsUrmaDeactiveJetty(urma_jetty_t *jetty)
{
    if (gUrmaOps.rsUrmaDeactiveJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_deactive_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeactiveJetty(jetty)));
}

int RsUrmaFreeJetty(urma_jetty_t *jetty)
{
    if (gUrmaOps.rsUrmaFreeJetty == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_free_jetty is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaFreeJetty(jetty)));
}

int RsUrmaAllocJfc(urma_context_t *urmaCtx, urma_jfc_cfg_t *cfg, urma_jfc_t **jfc)
{
    if (gUrmaOps.rsUrmaAllocJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_alloc_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaAllocJfc(urmaCtx, cfg, jfc)));
}

int RsUrmaSetJfcOpt(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len)
{
    if (gUrmaOps.rsUrmaSetJfcOpt == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_set_jfc_opt is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaSetJfcOpt(jfc, opt, buf, len)));
}

int RsUrmaActiveJfc(urma_jfc_t *jfc)
{
    if (gUrmaOps.rsUrmaActiveJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_active_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaActiveJfc(jfc)));
}

int RsUrmaGetJfcOpt(urma_jfc_t *jfc, uint64_t opt, void *buf, uint32_t len)
{
    if (gUrmaOps.rsUrmaGetJfcOpt == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_get_jfc_opt is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaGetJfcOpt(jfc, opt, buf, len)));
}

int RsUrmaDeactiveJfc(urma_jfc_t *jfc)
{
    if (gUrmaOps.rsUrmaDeactiveJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_deactive_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaDeactiveJfc(jfc)));
}

int RsUrmaFreeJfc(urma_jfc_t *jfc)
{
    if (gUrmaOps.rsUrmaFreeJfc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_urma_free_jfc is null");
        return -EINVAL;
#endif
    }
    return PERF_TRACE(RDMA_OP, DlRetConvert(gUrmaOps.rsUrmaFreeJfc(jfc)));
}
