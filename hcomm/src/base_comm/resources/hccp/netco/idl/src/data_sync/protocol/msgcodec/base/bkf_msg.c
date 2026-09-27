/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_msg.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

const char *BkfMsgGetIdStr(uint16_t msgId)
{
    static const char *msgStr[] = {
        "BKF_MSG_RESERVED",
        "BKF_MSG_HELLO",
        "BKF_MSG_HELLO_ACK",
        "BKF_MSG_SUB",
        "BKF_MSG_SUB_ACK",
        "BKF_MSG_UNSUB",
        "BKF_MSG_BATCH_BEGIN",
        "BKF_MSG_BATCH_END",
        "BKF_MSG_DATA",
        "BKF_MSG_DEL_SUB_NTF",
        "BKF_MSG_NOTIFY",
        "BKF_MSG_RPC",
        "BKF_MSG_SINGLE_SUB",
        "BKF_MSG_SINGLE_ACK",
        "BKF_MSG_SINGLE_UNSUB",
        "BKF_MSG_SINGLE_COND_BATCH_BEGIN",
        "BKF_MSG_SINGLE_COND_BATCH_END",
        "BKF_MSG_SINGLE_COND_DATA",
        "BKF_MSG_SINGLE_RESULT_DATA",
    };

    return BKF_GET_NUM_STR(msgId, msgStr);
}

const char *BkfTlvGetTypeStr(uint16_t typeId)
{
    static const char *tlvTypeStr[] = {
        "BKF_TLV_RESERVED",
        "BKF_TLV_TRANS_NUM",
        "BKF_TLV_PROJECT_NAME",
        "BKF_TLV_PROJECT_VERSION",
        "BKF_TLV_SERVICE_NAME",
        "BKF_TLV_PUBER_NAME",
        "BKF_TLV_SUBER_NAME",
        "BKF_TLV_IDL_VERSION",
        "BKF_TLV_REASON_CODE",
        "BKF_TLV_SLICE_KEY",
        "BKF_TLV_TABLE_TYPE",
        "BKF_TLV_TUPLE_IDL_DATA",
        "BKF_TLV_RPC",
        "BKF_TLV_COND_IDL_DATA",
        "BKF_TLV_RESULT_IDL_DATA",
        "BKF_TLV_SUBER_LSNURL",
    };

    return BKF_GET_NUM_STR(typeId, tlvTypeStr);
}

const char *BkfReasonCodeGetStr(uint8_t reasonCode)
{
    static const char *rcOkStr[] = {
        "BKF_RC_OK",
    };
    static const char *rcErrStr[] = {
        "BKF_ERR",
        "BKF_ERR_PROJECT_NAME",
        "BKF_ERR_PROJECT_VERSION",
        "BKF_ERR_SERVICE_NAME",
        "BKF_ERR_SUBER_NAME",
        "BKF_ERR_IDL_VERSION",
    };

    if (reasonCode < BKF_RC_OK_MAX) {
        return BKF_GET_NUM_STR(reasonCode, rcOkStr);
    } else {
        return BKF_GET_NUM_STR((uint8_t)(reasonCode - BKF_RC_ERR_MIN), rcErrStr);
    }
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
