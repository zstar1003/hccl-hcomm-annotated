/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCP_ASYNC_CTX_H
#define HCCP_ASYNC_CTX_H

#include "hccp_common.h"
#include "hccp_ctx.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief register local mem async
 * @param ctx_handle [IN] ctx handle
 * @param lmem_info [IN/OUT] lmem reg info
 * @param lmem_handle [OUT] lmem handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_lmem_unregister_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxLmemRegisterAsync(void *ctxHandle, struct MrRegInfoT *lmemInfo, void **lmemHandle,
    void **reqHandle);

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief unregister local mem async
 * @param ctx_handle [IN] ctx handle
 * @param lmem_handle [IN] lmem handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_lmem_register_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxLmemUnregisterAsync(void *ctxHandle, void *lmemHandle, void **reqHandle);

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief create jetty/qp async
 * @param ctx_handle [IN] ctx handle
 * @param attr [IN] qp attr
 * @param info [OUT] qp info
 * @param qp_handle [OUT] qp handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_qp_destroy_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpCreateAsync(void *ctxHandle, struct QpCreateAttr *attr, struct QpCreateInfo *info,
    void **qpHandle, void **reqHandle);

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief destroy jetty/qp async
 * @param qp_handle [IN] qp handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_qp_create_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpDestroyAsync(void *qpHandle, void **reqHandle);

/**
 * @ingroup libudma
 * @brief batch destroy qp
 * @param ctx_handle [IN] ctx handle
 * @param qp_handle [IN] corresponding qp_handle list
 * @param num [IN/OUT] size of qp_handle list, max num is HCCP_MAX_QP_DESTROY_BATCH_NUM
 * @param req_handle [OUT] async request handle
 * @see ra_ctx_qp_create
 * @see ra_get_async_req_result
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpDestroyBatchAsync(void *ctxHandle, void *qpHandle[], unsigned int *num,
    void **reqHandle);

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief import jetty/prepare rem_qp_handle for modify qp async
 * @param ctx_handle [IN] ctx handle
 * @param info [IN/OUT] qp import info
 * @param rem_qp_handle [OUT] remote qp handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_qp_unimport_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpImportAsync(void *ctxHandle, struct QpImportInfoT *info, void **remQpHandle,
    void **reqHandle);

/**
 * @ingroup librdma
 * @ingroup libudma
 * @brief unimport jetty async
 * @param rem_qp_handle [IN] qp handle
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_qp_import_async
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaCtxQpUnimportAsync(void *remQpHandle, void **reqHandle);

/**
 * @ingroup libudma
 * @brief get corresponding eid by ip async
 * @param ctx_handle [IN] ctx handle
 * @param ip [IN] ip array, see struct IpInfo
 * @param eid [IN/OUT] eid array, see union HccpEid
 * @param num [IN/OUT] num of ip and eid array, max num is HCCP_EID_IP_QUERY_MAX_NUM
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetEidByIpAsync(void *ctxHandle, struct IpInfo ip[], union HccpEid eid[], unsigned int *num,
    void **reqHandle);

/**
 * @ingroup libudm
 * @brief get corresponding ip by eid async
 * @param ctx_handle [IN] ctx handle
 * @param eid [IN] eid array, see union HccpEid
 * @param ip [IN/OUT] ip array, see struct IpInfo
 * @param num [IN/OUT] num of ip and eid array, max num is HCCP_EID_IP_QUERY_MAX_NUM
 * @param req_handle [OUT] async request handle
 * @see ra_get_async_req_result
 * @see ra_ctx_init
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetIpByEidAsync(void *ctxHandle, union HccpEid eid[], struct IpInfo ip[], unsigned int *num,
    void **reqHandle);

/**
 * @ingroup libudma
 * @brief get tp info list
 * @param ctx_handle [IN] ctx handle
 * @param cfg [IN] get tp cfg
 * @param info_list [IN/OUT] corresponding tp info list
 * @param num [IN/OUT] size of info_list, max num is HCCP_MAX_TPID_INFO_NUM
 * @param req_handle [OUT] async request handle
 * @see ra_ctx_init
 * @see ra_get_async_req_result
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetTpInfoListAsync(void *ctxHandle, struct GetTpCfg *cfg, struct HccpTpInfo infoList[],
    unsigned int *num, void **reqHandle);

/**
 * @ingroup libudma
 * @brief get tp attr
 * @param ctx_handle [IN] ctx handle
 * @param tp_handle [IN] see struct tp_info
 * @param attr_bitmap [IN/OUT] see struct tp_attr
 * @param attr [IN/OUT] see struct tp_attr
 * @param req_handle [OUT] async request handle
 * @see ra_get_tp_info_list_async
 * @see ra_get_async_req_result
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaGetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t *attrBitmap, struct TpAttr *attr,
    void **reqHandle);

/**
 * @ingroup libudma
 * @brief set tp attr
 * @param ctx_handle [IN] ctx handle
 * @param tp_handle [IN] see struct tp_info
 * @param attr_bitmap [IN] see struct tp_attr
 * @param attr [IN] see struct tp_attr
 * @param req_handle [OUT] async request handle
 * @see ra_get_tp_info_list_async
 * @see ra_get_async_req_result
 * @retval #zero Success
 * @retval #non-zero Failure
 */
HCCP_ATTRI_VISI_DEF int RaSetTpAttrAsync(void *ctxHandle, uint64_t tpHandle, uint32_t attrBitmap, struct TpAttr *attr,
    void **reqHandle);

#ifdef __cplusplus
}
#endif

#endif // HCCP_ASYNC_CTX_H
