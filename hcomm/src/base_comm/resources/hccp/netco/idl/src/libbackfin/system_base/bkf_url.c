/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_url.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_assert.h"
#include "securec.h"
#include "v_stringlib.h"
#include "vos_errno.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
#define BKF_URL_SIGN0 (0xab)
#define BKF_URL_SIGN1 (0xaf)

#define BKF_URL_IP_STR_BUF_LEN_MIN (48)
BKF_STATIC_ASSERT(BKF_URL_IP_STR_BUF_LEN_MIN >= BKF_URL_TCP_STR_BUF_LEN_MIN);
BKF_STATIC_ASSERT(BKF_URL_IP_STR_BUF_LEN_MIN >= BKF_URL_MESH_STR_BUF_LEN_MIN);
BKF_STATIC_ASSERT(BKF_URL_IP_STR_BUF_LEN_MIN >= BKF_URL_V8TLS_STR_BUF_LEN_MIN);

#define BKF_URL_SIGN_SET(url)                                                                                          \
    do {                                                                                                               \
        BKF_SIGN_SET((url)->sign, BKF_URL_SIGN0);                                                                      \
        BKF_SIGN_SET((url)->signH, BKF_URL_SIGN1);                                                                     \
    } while (0)
#define BKF_URL_SIGN_CLR(url)                                                                                          \
    do {                                                                                                               \
        BKF_SIGN_CLR((url)->sign);                                                                                     \
        BKF_SIGN_CLR((url)->signH);                                                                                    \
    } while (0)
#define BKF_URL_SIGN_CLONE(urlDest, urlSrc)                                                                            \
    do {                                                                                                               \
        (urlDest)->sign = (urlSrc)->sign;                                                                              \
        (urlDest)->signH = (urlSrc)->signH;                                                                            \
    } while (0)
#define BKF_URL_IS_VALID(url)                                                                                          \
    (((url) != VOS_NULL) && BKF_SIGN_IS_VALID((url)->sign, BKF_URL_SIGN0) &&                                           \
        BKF_SIGN_IS_VALID((url)->signH, BKF_URL_SIGN1) && BKF_URL_TYPE_IS_VALID((url)->type))

typedef uint32_t (*F_BKF_URL_STR_2VAL)(char *urlStr, BkfUrl *url);
typedef char *(*F_BKF_URL_VAL_2STR)(BkfUrl *url, uint8_t *buf, int32_t bufLen);
typedef uint32_t (*F_BKF_URL_H2N)(BkfUrl *urlH, BkfUrl *urlN);
typedef uint32_t (*F_BKF_URL_N2H)(BkfUrl *urlN, BkfUrl *urlH);
typedef struct TagBkfUrlVTbl {
    const char *urlStrPrefix;
    F_BKF_URL_STR_2VAL str2Val;
    F_BKF_URL_VAL_2STR val2Str;
    F_BKF_CMP cmp;
    int32_t urlStrBufLenMin;
    F_BKF_URL_H2N h2N;
    F_BKF_URL_N2H n2H;
} BkfUrlVTbl;

#pragma pack()

uint32_t BkfUrlInvalidStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlInvalidVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlInvalidCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlInvalidH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlInvalidN2H(BkfUrl *urlN, BkfUrl *urlH);

uint32_t BkfUrlDmsStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlDmsVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlDmsCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlDmsH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlDmsN2H(BkfUrl *urlN, BkfUrl *urlH);

uint32_t BkfUrlTcpStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlTcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlTcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlTcpH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlTcpN2H(BkfUrl *urlN, BkfUrl *urlH);

uint32_t BkfUrlMeshStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlMeshVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlMeshCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlMeshH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlMeshN2H(BkfUrl *urlN, BkfUrl *urlH);

uint32_t BkfUrlV8TlsStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlV8TlsVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlV8TlsCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlV8TlsH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlV8TlsN2H(BkfUrl *urlN, BkfUrl *urlH);
uint32_t BkfUrlV8TcpStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlV8TcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlV8TcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlV8TcpH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlV8TcpN2H(BkfUrl *urlN, BkfUrl *urlH);
uint32_t BkfUrlLeTcpStr2Val(char *urlStr, BkfUrl *url);
char *BkfUrlLeTcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen);
int32_t BkfUrlLeTcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs);
uint32_t BkfUrlLeTcpH2N(BkfUrl *urlH, BkfUrl *urlN);
uint32_t BkfUrlLeTcpN2H(BkfUrl *urlN, BkfUrl *urlH);

const BkfUrlVTbl g_BkfUrlVTbl[] = {
    {"invalid://", BkfUrlInvalidStr2Val, BkfUrlInvalidVal2Str, (F_BKF_CMP)BkfUrlInvalidCmp, 1, BkfUrlInvalidH2N,
        BkfUrlInvalidN2H},
    {"dms://", BkfUrlDmsStr2Val, BkfUrlDmsVal2Str, (F_BKF_CMP)BkfUrlDmsCmp, BKF_URL_DMS_STR_BUF_LEN_MIN, BkfUrlDmsH2N,
        BkfUrlDmsN2H},
    {"tcp://", BkfUrlTcpStr2Val, BkfUrlTcpVal2Str, (F_BKF_CMP)BkfUrlTcpCmp, BKF_URL_TCP_STR_BUF_LEN_MIN, BkfUrlTcpH2N,
        BkfUrlTcpN2H},
    {"mesh://", BkfUrlMeshStr2Val, BkfUrlMeshVal2Str, (F_BKF_CMP)BkfUrlMeshCmp, BKF_URL_MESH_STR_BUF_LEN_MIN,
        BkfUrlMeshH2N, BkfUrlMeshN2H},
    {"v8tls://", BkfUrlV8TlsStr2Val, BkfUrlV8TlsVal2Str, (F_BKF_CMP)BkfUrlV8TlsCmp, BKF_URL_V8TLS_STR_BUF_LEN_MIN,
        BkfUrlV8TlsH2N, BkfUrlV8TlsN2H},
    {"v8tcp://", BkfUrlV8TcpStr2Val, BkfUrlV8TcpVal2Str, (F_BKF_CMP)BkfUrlV8TcpCmp, BKF_URL_V8TCP_STR_BUF_LEN_MIN,
        BkfUrlV8TcpH2N, BkfUrlV8TcpN2H},
    {"letcp://", BkfUrlLeTcpStr2Val, BkfUrlLeTcpVal2Str, (F_BKF_CMP)BkfUrlLeTcpCmp, BKF_URL_LETCP_STR_BUF_LEN_MIN,
        BkfUrlLeTcpH2N, BkfUrlLeTcpN2H},
};
BKF_STATIC_ASSERT(BKF_GET_ARY_COUNT(g_BkfUrlVTbl) == BKF_URL_TYPE_CNT);

#define BKF_URL_STR_INVALID "-"
uint32_t BkfUrlInvalidStr2Val(char *urlStr, BkfUrl *url)
{
    (void)urlStr;
    (void)url;
    BKF_ASSERT(0);
    return BKF_ERR;
}
char *BkfUrlInvalidVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    (void)url;
    (void)buf;
    (void)bufLen;
    BKF_ASSERT(0);
    return BKF_URL_STR_INVALID;
}
int32_t BkfUrlInvalidCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    (void)url1Input;
    (void)url2InDs;
    BKF_ASSERT(0);
    return 0;
}
uint32_t BkfUrlInvalidH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    (void)urlH;
    (void)urlN;
    BKF_ASSERT(0);
    return BKF_ERR;
}
uint32_t BkfUrlInvalidN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    (void)urlN;
    (void)urlH;
    BKF_ASSERT(0);
    return BKF_ERR;
}

uint32_t BkfUrlDmsStr2Val(char *urlStr, BkfUrl *url)
{
    uint64_t dmsIdMax = (uint64_t)((uint32_t)-1);
    char *begin = VOS_NULL;
    uint64_t val;
    char *end = VOS_NULL;

    begin = urlStr + VOS_StrLen(g_BkfUrlVTbl[BKF_URL_TYPE_DMS].urlStrPrefix);
    if (*begin == ' ') {
        return BKF_ERR;
    }
    val = VOS_strtoull(begin, &end, 0);
    if ((end == VOS_NULL) || (*end != '\0') || (val > dmsIdMax)) {
        return BKF_ERR;
    }

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_DMS;
    url->dmsId = (uint32_t)val;
    return BKF_OK;
}
char *BkfUrlDmsVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    int32_t ret;
    if (bufLen < BKF_URL_DMS_STR_BUF_LEN_MIN) {
        return BKF_URL_STR_INVALID;
    }

    ret = snprintf_truncated_s((char *)buf, bufLen, "%s0x%X", g_BkfUrlVTbl[BKF_URL_TYPE_DMS].urlStrPrefix, url->dmsId);
    if (ret <= 0) {
        return BKF_URL_STR_INVALID;
    }

    return (char *)buf;
}
int32_t BkfUrlDmsCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BKF_CMP_X(url1Input->dmsId, url2InDs->dmsId);
}
uint32_t BkfUrlDmsH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    BKF_URL_SIGN_CLONE(urlN, urlH);
    urlN->type = urlH->type;
    urlN->dmsId = VOS_HTONL(urlH->dmsId);
    return BKF_OK;
}
uint32_t BkfUrlDmsN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    BKF_URL_SIGN_CLONE(urlH, urlN);
    urlH->type = urlN->type;
    urlH->dmsId = VOS_NTOHL(urlN->dmsId);
    return BKF_OK;
}

uint32_t BkfUrlIpStr2Val(char *urlStr, BkfUrl *url, uint8_t urlType)
{
    char *addrBegin = VOS_NULL;
    char *addrPortDivision = VOS_NULL;
    uint32_t addrH;
    uintptr_t portMax = (uintptr_t)((uint16_t)-1);
    uintptr_t portVal;
    char *end = VOS_NULL;

    addrBegin = urlStr + VOS_StrLen(g_BkfUrlVTbl[urlType].urlStrPrefix);
    addrPortDivision = VOS_StrChr(addrBegin, ':');
    if (addrPortDivision == VOS_NULL) {
        return BKF_ERR;
    }
    *addrPortDivision = '\0';
    int32_t ret = VOS_StrToIpAddr(addrBegin, &addrH);
    if (ret != VOS_OK) {
        return BKF_ERR;
    }
    portVal = VOS_strtoul(addrPortDivision + 1, &end, 10); /* 10进制 */
    if ((portVal > portMax) || (*end != '\0')) {
        return BKF_ERR;
    }

    BKF_URL_SIGN_SET(url);
    url->type = urlType;
    url->ip.addrH = addrH;
    url->ip.port = (uint16_t)portVal;
    return BKF_OK;
}
char *BkfUrlIpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen, uint8_t urlType)
{
    char *addrStr = VOS_NULL;
    char addrStrBuf[BKF_URL_IP_STR_BUF_LEN_MIN] = {0};
    int32_t ret;

    if (bufLen < g_BkfUrlVTbl[urlType].urlStrBufLenMin) {
        return BKF_URL_STR_INVALID;
    }

    addrStr = VOS_IpAddrToStrEx(url->ip.addrH, addrStrBuf, sizeof(addrStrBuf));
    if (addrStr == VOS_NULL) {
        return BKF_URL_STR_INVALID;
    }
    ret = snprintf_truncated_s((char *)buf, bufLen, "%s%s:%u", g_BkfUrlVTbl[urlType].urlStrPrefix, addrStr,
        url->ip.port);
    if (ret <= 0) {
        return BKF_URL_STR_INVALID;
    }

    return (char *)buf;
}
static inline int32_t BkfUrlIpCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    int32_t ret;

    ret = BKF_CMP_X(url1Input->ip.addrH, url2InDs->ip.addrH);
    if (ret != 0) {
        return ret;
    }

    return BKF_CMP_X(url1Input->ip.port, url2InDs->ip.port);
}
static inline uint32_t BkfUrlIpH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    BKF_URL_SIGN_CLONE(urlN, urlH);
    urlN->type = urlH->type;
    urlN->ip.addrH = VOS_HTONL(urlH->ip.addrH);
    urlN->ip.port = VOS_HTONS(urlH->ip.port);
    return BKF_OK;
}
static inline uint32_t BkfUrlIpN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    BKF_URL_SIGN_CLONE(urlH, urlN);
    urlH->type = urlN->type;
    urlH->ip.addrH = VOS_NTOHL(urlN->ip.addrH);
    urlH->ip.port = VOS_NTOHS(urlN->ip.port);
    return BKF_OK;
}

uint32_t BkfUrlTcpStr2Val(char *urlStr, BkfUrl *url)
{
    return BkfUrlIpStr2Val(urlStr, url, BKF_URL_TYPE_TCP);
}
char *BkfUrlTcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    return BkfUrlIpVal2Str(url, buf, bufLen, BKF_URL_TYPE_TCP);
}
int32_t BkfUrlTcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BkfUrlIpCmp(url1Input, url2InDs);
}
uint32_t BkfUrlTcpH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    return BkfUrlIpH2N(urlH, urlN);
}
uint32_t BkfUrlTcpN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    return BkfUrlIpN2H(urlN, urlH);
}

uint32_t BkfUrlMeshStr2Val(char *urlStr, BkfUrl *url)
{
    return BkfUrlIpStr2Val(urlStr, url, BKF_URL_TYPE_MESH);
}
char *BkfUrlMeshVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    return BkfUrlIpVal2Str(url, buf, bufLen, BKF_URL_TYPE_MESH);
}
int32_t BkfUrlMeshCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BkfUrlIpCmp(url1Input, url2InDs);
}
uint32_t BkfUrlMeshH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    return BkfUrlIpH2N(urlH, urlN);
}
uint32_t BkfUrlMeshN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    return BkfUrlIpN2H(urlN, urlH);
}

uint32_t BkfUrlV8TlsStr2Val(char *urlStr, BkfUrl *url)
{
    return BkfUrlIpStr2Val(urlStr, url, BKF_URL_TYPE_V8TLS);
}
char *BkfUrlV8TlsVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    return BkfUrlIpVal2Str(url, buf, bufLen, BKF_URL_TYPE_V8TLS);
}
int32_t BkfUrlV8TlsCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BkfUrlIpCmp(url1Input, url2InDs);
}
uint32_t BkfUrlV8TlsH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    return BkfUrlIpH2N(urlH, urlN);
}
uint32_t BkfUrlV8TlsN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    return BkfUrlIpN2H(urlN, urlH);
}

uint32_t BkfUrlV8TcpStr2Val(char *urlStr, BkfUrl *url)
{
    return BkfUrlIpStr2Val(urlStr, url, BKF_URL_TYPE_V8TCP);
}
char *BkfUrlV8TcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    return BkfUrlIpVal2Str(url, buf, bufLen, BKF_URL_TYPE_V8TCP);
}
int32_t BkfUrlV8TcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BkfUrlIpCmp(url1Input, url2InDs);
}
uint32_t BkfUrlV8TcpH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    return BkfUrlIpH2N(urlH, urlN);
}
uint32_t BkfUrlV8TcpN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    return BkfUrlIpN2H(urlN, urlH);
}

uint32_t BkfUrlLeTcpStr2Val(char *urlStr, BkfUrl *url)
{
    return BkfUrlIpStr2Val(urlStr, url, BKF_URL_TYPE_LETCP);
}
char *BkfUrlLeTcpVal2Str(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    return BkfUrlIpVal2Str(url, buf, bufLen, BKF_URL_TYPE_LETCP);
}
int32_t BkfUrlLeTcpCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    return BkfUrlIpCmp(url1Input, url2InDs);
}
uint32_t BkfUrlLeTcpH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    return BkfUrlIpH2N(urlH, urlN);
}
uint32_t BkfUrlLeTcpN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    return BkfUrlIpN2H(urlN, urlH);
}

uint32_t BkfUrlInit(BkfUrl *url)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }

    url->type = BKF_URL_TYPE_INVALID;
    BKF_URL_SIGN_CLR(url);
    return BKF_OK;
}

uint32_t BkfUrlSet(BkfUrl *url, char *urlStr)
{
    char *begin = VOS_NULL;
    uint8_t copyStr[BKF_URL_STR_LEN_MAX + 1];
    int32_t ret;
    uint32_t i;
    uint32_t cnt;
    char *find = VOS_NULL;
    BOOL findAtBegin = VOS_FALSE;

    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));
    if (urlStr == VOS_NULL) {
        return BKF_ERR;
    }
    if (VOS_StrLen(urlStr) > BKF_URL_STR_LEN_MAX) {
        return BKF_ERR;
    }
    ret = snprintf_truncated_s((char *)copyStr, sizeof(copyStr), "%s", urlStr);
    if (ret <= 0) {
        return BKF_ERR;
    }

    begin = VOS_StrTrim((char *)copyStr);
    if (begin == VOS_NULL) {
        return BKF_ERR;
    }
    VOS_StrToLower(begin);

    cnt = BKF_GET_ARY_COUNT(g_BkfUrlVTbl);
    for (i = 0; i < cnt; i++) {
        find = VOS_StrStr(begin, g_BkfUrlVTbl[i].urlStrPrefix);
        findAtBegin = (find == begin);
        if (findAtBegin) {
            return g_BkfUrlVTbl[i].str2Val(begin, url);
        }
    }

    return BKF_ERR;
}

uint32_t BkfUrlDmsSet(BkfUrl *url, uint32_t dmsId)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_DMS;
    url->dmsId = dmsId;
    return BKF_OK;
}

uint32_t BkfUrlTcpSet(BkfUrl *url, uint32_t addrH, uint16_t port)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_TCP;
    url->ip.addrH = addrH;
    url->ip.port = port;
    return BKF_OK;
}

uint32_t BkfUrlMeshSet(BkfUrl *url, uint32_t addrH, uint16_t port)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_MESH;
    url->ip.addrH = addrH;
    url->ip.port = port;
    return BKF_OK;
}

uint32_t BkfUrlTlsSet(BkfUrl *url, uint32_t addrH, uint16_t port)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_V8TLS;
    url->ip.addrH = addrH;
    url->ip.port = port;
    return BKF_OK;
}

uint32_t BkfUrlV8TcpSet(BkfUrl *url, uint32_t addrH, uint16_t port)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_V8TCP;
    url->ip.addrH = addrH;
    url->ip.port = port;
    return BKF_OK;
}

uint32_t BkfUrlLeTcpSet(BkfUrl *url, uint32_t addrH, uint16_t port)
{
    if (url == VOS_NULL) {
        return BKF_ERR;
    }
    (void)memset_s(url, sizeof(BkfUrl), 0, sizeof(BkfUrl));

    BKF_URL_SIGN_SET(url);
    url->type = BKF_URL_TYPE_LETCP;
    url->ip.addrH = addrH;
    url->ip.port = port;
    return BKF_OK;
}

BOOL BkfUrlIsValid(BkfUrl *url)
{
    return BKF_URL_IS_VALID(url);
}

int32_t BkfUrlCmp(BkfUrl *url1Input, BkfUrl *url2InDs)
{
    int32_t ret;

    if (!BKF_URL_IS_VALID(url1Input)) {
        BKF_ASSERT(0);
        return -1;
    }
    if (!BKF_URL_IS_VALID(url2InDs)) {
        BKF_ASSERT(0);
        return 1;
    }

    ret = BKF_CMP_X(url1Input->type, url2InDs->type);
    if (ret != 0) {
        return ret;
    }
    return g_BkfUrlVTbl[url1Input->type].cmp(url1Input, url2InDs);
}

char *BkfUrlGetStr(BkfUrl *url, uint8_t *buf, int32_t bufLen)
{
    if (!BKF_URL_IS_VALID(url) || (buf == VOS_NULL) || (bufLen <= 0)) {
        return BKF_URL_STR_INVALID;
    }
    (void)memset_s(buf, bufLen, 0, bufLen);
    return g_BkfUrlVTbl[url->type].val2Str(url, buf, bufLen);
}

uint32_t BkfUrlH2N(BkfUrl *urlH, BkfUrl *urlN)
{
    if (!BKF_URL_IS_VALID(urlH) || (urlN == VOS_NULL)) {
        return BKF_ERR;
    }

    return g_BkfUrlVTbl[urlH->type].h2N(urlH, urlN);
}

uint32_t BkfUrlN2H(BkfUrl *urlN, BkfUrl *urlH)
{
    if (!BKF_URL_IS_VALID(urlN) || (urlH == VOS_NULL)) {
        return BKF_ERR;
    }

    return g_BkfUrlVTbl[urlN->type].n2H(urlN, urlH);
}

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif
