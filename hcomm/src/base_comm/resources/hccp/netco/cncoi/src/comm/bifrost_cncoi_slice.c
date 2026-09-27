/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bifrost_cncoi_slice.h"
#include "securec.h"

#if __cplusplus
extern "C" {
#endif

int32_t BifrostCncoiSliceKeyCmp(const BifrostCncoiSliceKeyT *key1Input, const BifrostCncoiSliceKeyT *key2InDs)
{
    if (key1Input->sliceKey > key2InDs->sliceKey) {
        return 1;
    } else if (key1Input->sliceKey < key2InDs->sliceKey) {
        return -1;
    }
    return 0;
}

char *BifrostCncoiSliceKeyGetStr(const BifrostCncoiSliceKeyT *key, uint8_t *buf, int32_t bufLen)
{
    int32_t err;

    if ((key == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BifrostCncoiSliceKeyGetStrNgArg";
    }

    err = snprintf_truncated_s((char *)buf, bufLen, "bifrost_cncoi sliceKey = %u", key->sliceKey);
    return (err >= 0) ? (char *)buf : "__BifrostCncoiSliceKeyGetStrNgSnprintf";
}

uint32_t BifrostCncoiSliceKeyCodec(BifrostCncoiSliceKeyT *key)
{
    key->sliceKey = VOS_HTONL(key->sliceKey);
    return BKF_OK;
}
#if __cplusplus
}
#endif
