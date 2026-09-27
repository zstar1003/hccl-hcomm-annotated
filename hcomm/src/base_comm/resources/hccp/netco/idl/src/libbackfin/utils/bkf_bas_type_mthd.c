/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_bas_type_mthd.h"
#include "securec.h"
#include "v_stringlib.h"
#include "vos_base.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
int32_t BkfCharCmp(const char *key1Input, const char *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfCharGetStr(const char *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfCharGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%d", *input);
    if (writeLen <= 0) {
        return "__BkfCharGetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t BkfUCharCmp(const uint8_t *key1Input, const uint8_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfUCharGetStr(const uint8_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfUCharGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%u", *input);
    if (writeLen <= 0) {
        return "__BkfUCharGetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t BkfShortCmp(const int16_t *key1Input, const int16_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfShortGetStr(const int16_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfShortGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%d", *input);
    if (writeLen <= 0) {
        return "__BkfShortGetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t Bkfuint16_tCmp(const uint16_t *key1Input, const uint16_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *Bkfuint16_tGetStr(const uint16_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__Bkfuint16_tGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%u = %#x", *input, *input);
    if (writeLen <= 0) {
        return "__Bkfuint16_tGetStrSnprintfNg";
    }
    return (char *)buf;
}

uint32_t BkfXShortCodec(uint16_t *inOut)
{
    uint16_t temp;

    if (inOut == VOS_NULL) {
        return BKF_ERR;
    }

    temp = *inOut;
    *inOut = VOS_HTONS(temp);
    return BKF_OK;
}

int32_t BkfInt32Cmp(const int32_t *key1Input, const int32_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfInt32GetStr(const int32_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfInt32GetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%d", *input);
    if (writeLen <= 0) {
        return "__BkfInt32GetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t Bkfuint32_tCmp(const uint32_t *key1Input, const uint32_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *Bkfuint32_tGetStr(const uint32_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__Bkfuint32_tGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%u = %#x", *input, *input);
    if (writeLen <= 0) {
        return "__Bkfuint32_tGetStrSnprintfNg";
    }
    return (char *)buf;
}

uint32_t BkfXInt32Codec(uint32_t *inOut)
{
    uint32_t temp;

    if (inOut == VOS_NULL) {
        return BKF_ERR;
    }

    temp = *inOut;
    *inOut = VOS_HTONL(temp);
    return BKF_OK;
}

int32_t BkfInt64Cmp(const int64_t *key1Input, const int64_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfInt64GetStr(const int64_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfInt64GetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%" VOS_PRId64 "", *input);
    if (writeLen <= 0) {
        return "__BkfInt64GetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t BkfUInt64Cmp(const uint64_t *key1Input, const uint64_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfUInt64GetStr(const uint64_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfUInt64GetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%" VOS_PRIu64 " = 0x%" VOS_PRIx64 "", *input,
        *input);
    if (writeLen <= 0) {
        return "__BkfUInt64GetStrSnprintfNg";
    }
    return (char *)buf;
}

uint32_t BkfXInt64Codec(uint64_t *inOut)
{
    uint64_t temp;

    if (inOut == VOS_NULL) {
        return BKF_ERR;
    }

    temp = *inOut;
    *inOut = VOS_HTONLL(temp);
    return BKF_OK;
}

int32_t BkfIntptrCmp(const intptr_t *key1Input, const intptr_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfIntptrGetStr(const intptr_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfIntptrGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%" VOS_PRId64 "", (int64_t)(*input));
    if (writeLen <= 0) {
        return "__BkfIntptrGetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t BkfUIntptrCmp(const uintptr_t *key1Input, const uintptr_t *key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

char *BkfUIntptrGetStr(const uintptr_t *input, uint8_t *buf, int32_t bufLen)
{
    int32_t writeLen;

    if ((input == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfUIntptrGetStrArgNg";
    }
    writeLen = snprintf_truncated_s((char *)buf, (uint32_t)bufLen, "%" VOS_PRIu64 " = 0x%" VOS_PRIx64 "",
        (uint64_t)(*input), (uint64_t)(*input));
    if (writeLen <= 0) {
        return "__BkfBkfUIntptrGetStrSnprintfNg";
    }
    return (char *)buf;
}

int32_t BkfCmpPtrAddr(const void **key1Input, const void **key2InDs)
{
    return BKF_CMP_X(*key1Input, *key2InDs);
}

int32_t BkfIpAddrCmp(const uint32_t *key1IpAddrHInput, const uint32_t *key2IpAddrHInDs)
{
    return BKF_CMP_X(*key1IpAddrHInput, *key2IpAddrHInDs);
}

char *BkfIpAddrGetStr(const uint32_t *ipAddrH, uint8_t *buf, int32_t bufLen)
{
    char *addrStr = VOS_NULL;

    if ((ipAddrH == VOS_NULL) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return "__BkfIpAddrGetStrArgNg";
    }

    addrStr = VOS_IpAddrToStrEx(*ipAddrH, (char *)buf, bufLen);
    if (addrStr == VOS_NULL) {
        return "__VOS_IpAddrToStrEx_ng__";
    }

    return (char *)addrStr;
}

uint32_t BkfIpAddrCodec(uint32_t *ipAddrInOut)
{
    uint32_t temp;

    if (ipAddrInOut == VOS_NULL) {
        return BKF_ERR;
    }

    temp = *ipAddrInOut;
    *ipAddrInOut = VOS_HTONL(temp);
    return BKF_OK;
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
