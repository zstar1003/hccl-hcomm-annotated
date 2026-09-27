/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_ch_ser_conn_id_base.h"
#include "bkf_url.h"
#include "securec.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#define BKF_CH_SER_CONN_ID_SIGN 0xAB56

uint32_t BkfChSerConnIdBaseInit(BkfChSerConnIdBase *base, uint8_t urlType)
{
    if ((base == VOS_NULL) || !BKF_URL_TYPE_IS_VALID(urlType)) {
        return BKF_ERR;
    }

    (void)memset_s(base, sizeof(BkfChSerConnIdBase), 0, sizeof(BkfChSerConnIdBase));
    BKF_SIGN_SET(base->sign, BKF_CH_SER_CONN_ID_SIGN);
    base->urlType = urlType;
    return BKF_OK;
}

void BkfChSerConnIdBaseUninit(BkfChSerConnIdBase *base)
{
    if ((base == VOS_NULL) || !BKF_SIGN_IS_VALID(base->sign, BKF_CH_SER_CONN_ID_SIGN)) {
        return;
    }

    (void)memset_s(base, sizeof(BkfChSerConnIdBase), 0, sizeof(BkfChSerConnIdBase));
    return;
}

BkfChSerConnIdBase *BkfChSerConnIdGetBase(BkfChSerConnId *connId)
{
    BkfChSerConnIdBase *base = (BkfChSerConnIdBase *)connId;

    if ((base == VOS_NULL) || !BKF_SIGN_IS_VALID(base->sign, BKF_CH_SER_CONN_ID_SIGN) ||
        !BKF_URL_TYPE_IS_VALID(base->urlType)) {
        return VOS_NULL;
    }

    return base;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
