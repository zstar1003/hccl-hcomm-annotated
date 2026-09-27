/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_cer_arg.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_assert.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

#define BKF_CERA_SIGN0 (0xab)
#define BKF_CERA_SIGN1 (0xaf)

#define BKF_CERA_SIGN_CLR(cera)                                                                                        \
    do {                                                                                                               \
        BKF_SIGN_CLR((cera)->sign[0]);                                                                                 \
        BKF_SIGN_CLR((cera)->sign[1]);                                                                                 \
    } while (0)

#define BKF_CERA_SIGN_SET(cera)                                                                                        \
    do {                                                                                                               \
        BKF_SIGN_SET((cera)->sign[0], BKF_CERA_SIGN0);                                                                 \
        BKF_SIGN_SET((cera)->sign[1], BKF_CERA_SIGN1);                                                                 \
    } while (0)

#define BKF_CERA_IS_VALID(cera)                                                                                        \
    (((cera) != VOS_NULL) && BKF_SIGN_IS_VALID((cera)->sign[0], BKF_CERA_SIGN0) &&                                     \
        BKF_SIGN_IS_VALID((cera)->sign[1], BKF_CERA_SIGN1) && BKF_CERA_TYPE_IS_VALID((cera)->type))

#define BKF_CERA_STR_INVALID "-"
#define BKF_CERA_V8TLS_STR_BUF_LEN_MIN (56)

uint32_t BkfCeraInit(BkfCera *cera)
{
    if (cera == VOS_NULL) {
        return BKF_ERR;
    }
    cera->type = BKF_CERA_TYPE_INVALID;
    BKF_CERA_SIGN_CLR(cera);

    return BKF_OK;
}

uint32_t BkfCeraV8Set(BkfCera *cera, uint32_t policyId, uint32_t vrId, int32_t role, int32_t verifyMod)
{
    if (cera == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(cera, sizeof(BkfCera), 0, sizeof(BkfCera));
    BKF_CERA_SIGN_SET(cera);
    cera->type = BKF_CERA_TYPE_V8;
    cera->v8.key.policyId = policyId;
    cera->v8.val.role = role;
    cera->v8.val.vrId = vrId;
    cera->v8.val.verifyMod = verifyMod;
    return 0;
}

BOOL BkfCeraIsValid(BkfCera *cera)
{
    return BKF_CERA_IS_VALID(cera);
}

int32_t BkfCeraV8Cmp(BkfCera *cera1Input, BkfCera *cera2InDs)
{
    int32_t ret = BKF_CMP_X(cera1Input->v8.key.policyId, cera2InDs->v8.key.policyId);
    if (ret != 0) {
        return ret;
    }
    ret = BKF_CMP_X(cera1Input->v8.val.role, cera2InDs->v8.val.role);
    if (ret != 0) {
        return ret;
    }
    return BKF_CMP_X(cera1Input->v8.val.vrId, cera2InDs->v8.val.vrId);
}
int32_t BkfCeraCmp(BkfCera *cera1Input, BkfCera *cera2InDs)
{
    int32_t ret;

    ret = BKF_CMP_X(cera1Input->type, cera2InDs->type);
    if (ret != 0) {
        return ret;
    }

    if (cera1Input->type == BKF_CERA_TYPE_V8) {
        return BkfCeraV8Cmp(cera1Input, cera2InDs);
    }
    /* 暂不支持 */
    BKF_ASSERT(0);
    return -1;
}

char *BkfCeraV8TlsVal2Str(BkfCera *cera, uint8_t *buf, int32_t bufLen)
{
    if (bufLen < BKF_CERA_V8TLS_STR_BUF_LEN_MIN) {
        return BKF_CERA_STR_INVALID;
    }

    int32_t ret = snprintf_truncated_s((char *)buf, bufLen, "V8Tls://policy %u vr %u role %d verifyMod %d",
        cera->v8.key.policyId, cera->v8.val.vrId, cera->v8.val.role, cera->v8.val.verifyMod);
    if (ret <= 0) {
        return BKF_CERA_STR_INVALID;
    }
    return (char *)buf;
}

char *BkfCeraGetStr(BkfCera *cera, uint8_t *buf, int32_t bufLen)
{
    if (cera == VOS_NULL) {
        return BKF_CERA_STR_INVALID;
    }
    if (cera->type == BKF_CERA_TYPE_V8) {
        return BkfCeraV8TlsVal2Str(cera, buf, bufLen);
    }
    /* 暂不支持 */
    BKF_ASSERT(0);
    return BKF_CERA_STR_INVALID;
}

uint32_t BkfCeraH2N(BkfCera *ceraH, BkfCera *ceraN)
{
    (void)ceraH;
    (void)ceraN;
    return 0;
}

uint32_t BkfCeraN2H(BkfCera *ceraN, BkfCera *ceraH)
{
    (void)ceraN;
    (void)ceraH;
    return 0;
}

#ifdef __cplusplus
}
#endif
