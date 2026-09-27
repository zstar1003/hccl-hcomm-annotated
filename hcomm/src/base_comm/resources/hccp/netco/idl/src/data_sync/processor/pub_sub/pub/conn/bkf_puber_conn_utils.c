/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((connMng)->log)
#define BKF_MOD_NAME ((connMng)->name)
#define BKF_LINE (__LINE__ + 15000)

#include "bkf_puber_conn_utils.h"
#include "v_stringlib.h"

#ifdef __cplusplus
extern "C" {
#endif

uint32_t BkfPuberConnParseHello(BkfPuberConn *conn, BkfMsgDecoder *decoder, BkfWrapMsgHello *hello)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfTL *tl = VOS_NULL;
    uint32_t errCode = BKF_OK;
    while ((tl = BkfMsgDecodeTLV(decoder, &errCode)) != VOS_NULL) {
        switch (tl->typeId) {
            case BKF_TLV_PROJECT_NAME: {
                hello->projectName = (BkfTlvProjectName *)tl;
                break;
            }
            case BKF_TLV_PROJECT_VERSION: {
                hello->projectVersion = (BkfTlvProjectVersion *)tl;
                break;
            }
            case BKF_TLV_SERVICE_NAME: {
                hello->serviceName = (BkfTlvServiceName *)tl;
                break;
            }
            case BKF_TLV_SUBER_NAME: {
                hello->suberName = (BkfTlvSuberName *)tl;
                break;
            }
            case BKF_TLV_IDL_VERSION: {
                hello->idlVersion = (BkfTlvIdlVersion *)tl;
                break;
            }
            case BKF_TLV_SUBER_LSNURL: {
                hello->suberLsnUrl = (BkfTlvSuberLsnUrl *)tl;
                break;
            }
            default: {
                BKF_LOG_INFO(BKF_LOG_HND, "tlv type(%u, %s)\n", tl->typeId, BkfTlvGetTypeStr(tl->typeId));
                break;
            }
        }
    }
    return errCode;
}

uint32_t BkfPuberConnChkHello(BkfPuberConn *conn, BkfWrapMsgHello *hello)
{
    BOOL isValid = (hello->projectName != VOS_NULL) &&
                   (VOS_StrNCmp(hello->projectName->name, BKF_PROJECT_NAME, BKF_TLV_NAME_LEN_MAX + 1) == 0);
    if (!isValid) {
        return BKF_ERR;
    }

    isValid = (hello->projectVersion != VOS_NULL) && (hello->projectVersion->version <= BKF_PROJECT_VERSION);
    if (!isValid) {
        return BKF_ERR;
    }

    BkfPuberConnMng *connMng = conn->connMng;
    isValid = (hello->serviceName != VOS_NULL) &&
              (VOS_StrNCmp(hello->serviceName->name, connMng->argInit->svcName, BKF_TLV_NAME_LEN_MAX + 1) == 0);
    if (!isValid) {
        return BKF_ERR;
    }

    if (hello->suberName == VOS_NULL) {
        return BKF_ERR;
    }
    uint32_t strLen = VOS_StrLen(hello->suberName->name);
    isValid = (strLen > 0) && (strLen <= BKF_TLV_NAME_LEN_MAX);
    if (!isValid) {
        return BKF_ERR;
    }

    isValid = (hello->idlVersion != VOS_NULL) && (hello->idlVersion->major == connMng->argInit->idlVersionMajor);
    if (!isValid) {
        return BKF_ERR;
    }

    return BKF_OK;
}

uint32_t BkfPuberConnPackHelloAck(BkfPuberConn *conn, BkfMsgCoder *coder)
{
    BkfPuberConnMng *connMng = conn->connMng;
    BkfTlvReasonCode reasonCode = {.reasonCode = BKF_RC_OK};
    BkfTlvIdlVersion idlVersion = {
        .major = connMng->argInit->idlVersionMajor, .minor = connMng->argInit->idlVersionMinor};

    uint32_t ret = BkfMsgCodeMsgHead(coder, BKF_MSG_HELLO_ACK, 0);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_REASON_CODE, 0, &reasonCode.tl + 1, VOS_FALSE);
    ret |= BkfMsgCodeRawTLV(coder, BKF_TLV_PUBER_NAME, 0, (void *)connMng->argInit->name,
        VOS_StrLen(connMng->argInit->name) + 1, VOS_FALSE);
    ret |= BkfMsgCodeTLV(coder, BKF_TLV_IDL_VERSION, 0, &idlVersion.tl + 1, VOS_TRUE);
    return ret;
}

#ifdef __cplusplus
}
#endif
