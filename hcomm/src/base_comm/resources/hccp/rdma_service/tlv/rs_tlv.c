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
#include "securec.h"
#include "config_log.h"
#include "ra_rs_err.h"
#include "rs_adp_nslb.h"
#include "rs_inner.h"
#include "dl_ccu_function.h"
#include "rs_ctx.h"
#include "rs_tlv.h"

STATIC int RsGetTlvCb(uint32_t phyId, struct RsTlvCb **tlvCb)
{
    struct rs_cb *rsCb = NULL;
    int ret;

    ret = RsGetRsCb(phyId, &rsCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_rs_cb failed, phyId(%u) invalid, ret(%d)", phyId, ret), ret);

    *tlvCb = &rsCb->tlvCb;
    return 0;
}

RS_ATTRI_VISI_DEF int RsTlvInit(unsigned int phyId, unsigned int *bufferSize)
{
    struct RsTlvCb *tlvCb = NULL;
    struct rs_cb *rsCb = NULL;
    int ret = 0;

    CHK_PRT_RETURN(bufferSize == NULL, hccp_err("param error, buffer_size is NULL"), -EINVAL);

    ret = RsGetRsCb(phyId, &rsCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_rs_cb failed, phyId(%u) invalid, ret(%d)", phyId, ret), ret);
    tlvCb = &rsCb->tlvCb;
    CHK_PRT_RETURN(tlvCb->initFlag, hccp_err("rs_tlv init repeat, phyId(%u)", phyId), -EINVAL);

    tlvCb->phyId = phyId;
    ret = pthread_mutex_init(&tlvCb->mutex, NULL);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_tlv mutex_init failed ret(%d)", ret), -ESYSFUNC);

    tlvCb->bufInfo.buf = (char *)calloc(RS_TLV_BUFFER_SIZE, sizeof(char));
    if (tlvCb->bufInfo.buf == NULL) {
        hccp_err("rs_tlv calloc buf failed errno(%d)", errno);
        (void)pthread_mutex_destroy(&tlvCb->mutex);
        return -ENOMEM;
    }

    tlvCb->initFlag = true;
    tlvCb->bufInfo.bufferSize = RS_TLV_BUFFER_SIZE;
    *bufferSize = RS_TLV_BUFFER_SIZE;

    hccp_run_info("rs_tlv_init successful, phyId(%u) bufferSize(%u)", phyId, *bufferSize);

    return ret;
}

RS_ATTRI_VISI_DEF int RsTlvDeinit(unsigned int phyId)
{
    struct RsTlvCb *tlvCb = NULL;
    int ret = 0;

    ret = RsGetTlvCb(phyId, &tlvCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_tlv_cb failed, ret(%d) phyId(%u)", ret, phyId), ret);

    CHK_PRT_RETURN(!tlvCb->initFlag, hccp_warn_init("rs_tlv not init or already deinit, phyId(%u)", phyId), 0);

    RS_PTHREAD_MUTEX_LOCK(&tlvCb->mutex);
    tlvCb->initFlag = false;
    free(tlvCb->bufInfo.buf);
    tlvCb->bufInfo.buf = NULL;
    RS_PTHREAD_MUTEX_ULOCK(&tlvCb->mutex);
    pthread_mutex_destroy(&tlvCb->mutex);

    hccp_run_info("rs_tlv_deinit successful, phyId(%u)", phyId);
    return ret;
}

STATIC int RsTlvAssembleSendData(struct TlvBufInfo *bufInfo, struct TlvRequestMsgHead *head, char *data,
    bool *isSendFinish, unsigned int dataMaxLength)
{
    int ret = 0;

    *isSendFinish = false;
    CHK_PRT_RETURN(head->offset >= bufInfo->bufferSize,
        hccp_err("[recv][rs_tlv]param error, offset(%u) >= bufferSize(%u), phyId(%u)", head->offset,
            bufInfo->bufferSize, head->phyId),
        -EINVAL);
    CHK_PRT_RETURN(head->sendBytes > dataMaxLength,
        hccp_err("[recv][rs_tlv]param error, sendBytes(%u) >= data size(%u), phyId(%u)", head->sendBytes, dataMaxLength,
            head->phyId),
        -EINVAL);
    CHK_PRT_RETURN((head->offset + head->sendBytes) > head->totalBytes,
        hccp_err("[recv][rs_tlv]data overflow, offset(%u) + sendBytes(%u) > totalBytes(%u), phyId(%u)", head->offset,
            head->sendBytes, head->totalBytes, head->phyId),
        -EINVAL);

    if (head->offset == 0) {
        (void)memset_s(bufInfo->buf, bufInfo->bufferSize, 0, bufInfo->bufferSize);
    }

    ret = memcpy_s(bufInfo->buf + head->offset, bufInfo->bufferSize - head->offset, data, head->sendBytes);
    CHK_PRT_RETURN(ret != 0, hccp_err("[recv][rs_tlv]memcpy_s data failed, ret(%d) phyId(%u)", ret, head->phyId),
        -ESAFEFUNC);

    if (head->offset + head->sendBytes == head->totalBytes) {
        *isSendFinish = true;
    }

    return 0;
}

STATIC int RsCcuRequest(struct TlvRequestMsgHead *head, char *dataIn, char *dataOut, unsigned int *bufferSize)
{
    int ret = 0;
    if (isCcuTlvReqExist()) {
        ret = RsCcuTlvRequest(head->type, dataIn, dataOut, head->totalBytes, bufferSize, MAX_TLV_MSG_DATA_LEN_V2);
        CHK_PRT_RETURN(ret != 0 && ret != -EUSERS,
            hccp_err("rs_ccu_tlv_request failed, ret(%d) msg_type(%u) phy_id(%u)", ret, head->type, head->phyId), ret);
        return ret;
    }

    switch (head->type) {
        case MSG_TYPE_CCU_INIT:
            ret = RsCcuInit();
            CHK_PRT_RETURN(ret != 0 && ret != -EUSERS,
                hccp_err("rs_ccu_init failed, ret(%d) module_type(%u) msg_type(%u) phy_id(%u)", ret, head->moduleType,
                    head->type, head->phyId),
                ret);
            break;
        case MSG_TYPE_CCU_UNINIT:
            ret = RsCcuUninit();
            CHK_PRT_RETURN(ret != 0,
                hccp_err("rs_ccu_uninit failed, ret(%d) module_type(%u) msg_type(%u) phyId(%u)", ret, head->moduleType,
                    head->type, head->phyId),
                ret);
            break;
        case MSG_TYPE_CCU_GET_MEM_INFO:
            ret = RsCcuGetMemInfo(dataIn, dataOut, bufferSize);
            CHK_PRT_RETURN(ret != 0,
                hccp_err("RsCcuGetMemInfo failed, ret(%d) module_type(%u) msg_type(%u) phyId(%u)", ret,
                    head->moduleType, head->type, head->phyId),
                ret);
            break;
        case MSG_TYPE_CCU_DISPATCH_CMD:
            ret = RsCcuCustomChannel((struct channel_info_in *)dataIn, (struct channel_info_out *)dataOut);
            *bufferSize = sizeof(struct channel_info_out);
            CHK_PRT_RETURN(ret != 0,
                hccp_err("RsCcuDispatchCmd failed, ret(%d) module_type(%u) msg_type(%u) phyId(%u)", ret,
                    head->moduleType, head->type, head->phyId),
                ret);
            break;
        default:
            hccp_err("[request][rs_ccu]msg type error, module_type(%u) msg_type(%u) phyId(%u)", head->moduleType,
                head->type, head->phyId);
            return -EINVAL;
    }

    return ret;
}

RS_ATTRI_VISI_DEF int RsTlvRequest(struct TlvRequestMsgHead *head, char *dataIn, char *dataOut,
    unsigned int *bufferSize, unsigned int dataMaxLength)
{
    struct RsTlvCb *tlvCb = NULL;
    bool isSendFinish = false;
    int ret = 0;

    CHK_PRT_RETURN(head == NULL || dataIn == NULL || dataOut == NULL || bufferSize == NULL,
        hccp_err("param error, head or dataIn or dataOut or bufferSize is NULL"), -EINVAL);

    ret = RsGetTlvCb(head->phyId, &tlvCb);
    CHK_PRT_RETURN(ret != 0, hccp_err("rs_get_tlv_cb failed, ret(%d) phyId(%u)", ret, head->phyId), ret);
    CHK_PRT_RETURN(tlvCb->bufInfo.buf == NULL, hccp_err("rs_tlv buf not initialized, phyId(%u)", head->phyId), -EINVAL);

    RS_PTHREAD_MUTEX_LOCK(&tlvCb->mutex);
    ret = RsTlvAssembleSendData(&tlvCb->bufInfo, head, dataIn, &isSendFinish, dataMaxLength);
    if (ret != 0) {
        hccp_err("rs_tlv_assemble_send_data failed, ret(%d) phyId(%u)", ret, tlvCb->phyId);
        goto tlv_request_release_lock;
    }

    if (!isSendFinish) {
        goto tlv_request_release_lock;
    }

    switch (head->moduleType) {
        case TLV_MODULE_TYPE_NSLB:
            ret = RsNslbNetcoRequest(head->phyId, &tlvCb->nslbCb, head->type, tlvCb->bufInfo.buf, head->totalBytes);
            break;
        case TLV_MODULE_TYPE_CCU:
            ret = RsCcuRequest(head, tlvCb->bufInfo.buf, dataOut, bufferSize);
            break;
        default:
            hccp_err("[request][rs_tlv]module type error, moduleType(%u) phyId(%u)", head->moduleType, head->phyId);
            ret = -EINVAL;
            break;
    }

tlv_request_release_lock:
    RS_PTHREAD_MUTEX_ULOCK(&tlvCb->mutex);
    return ret;
}

RS_ATTRI_VISI_DEF int RsCtxCustomChannel(const struct CustomChanInfoIn *in, struct CustomChanInfoOut *out)
{
    struct channel_info_out chanOut = {0};
    struct channel_info_in chanIn = {0};
    int ret;

    RS_CHECK_POINTER_NULL_RETURN_INT(in);
    RS_CHECK_POINTER_NULL_RETURN_INT(out);

    ret = memcpy_s(&chanIn, sizeof(struct channel_info_in), in, sizeof(struct CustomChanInfoIn));
    CHK_PRT_RETURN(ret != 0, hccp_err("[ccu]memcpy_s in failed, ret[%d]", ret), -ESAFEFUNC);

    ret = RsCcuCustomChannel(&chanIn, &chanOut);
    CHK_PRT_RETURN(ret != 0, hccp_err("[ccu]rs_ctx_ccu_custom_channel failed, ret[%d]", ret), ret);

    // prepare output data
    ret = memcpy_s(out, sizeof(struct CustomChanInfoOut), &chanOut, sizeof(struct channel_info_out));
    CHK_PRT_RETURN(ret != 0, hccp_err("[ccu]memcpy_s out failed, ret[%d]", ret), -ESAFEFUNC);

    return 0;
}
