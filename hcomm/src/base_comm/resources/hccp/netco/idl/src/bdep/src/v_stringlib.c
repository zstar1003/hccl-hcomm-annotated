/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <ctype.h>
#include "vos_typdef.h"
#include "vos_errno.h"
#include "vrp_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOS_TYPE_NUM10 (10)
#define VOS_TYPE_NUM100 (100)
#define VOS_TYPE_BITNUM24 (24)
#define VOS_TYPE_BITNUM16 (16)
#define VOS_TYPE_BITNUM8 (8)
#define VOS_TYPE_BITNUM32 (32)

char *VOS_StrStr(const char *pscStr1, const char *pscStr2)
{
    if ((pscStr1 == VOS_NULL_PTR) || (pscStr2 == VOS_NULL_PTR)) {
        return VOS_NULL_PTR;
    }

    return strstr(pscStr1, pscStr2);
}

char *vosUint8ToStr(uint8_t ucVal, char *pscStr)
{
    char *pscStrTmp = pscStr;
    if (ucVal < VOS_TYPE_NUM10) { /* 10以内个位数处理 */
        *pscStrTmp++ = (char)(ucVal + '0');
    } else if (ucVal < VOS_TYPE_NUM100) {                    /* 100以内十位数处理 */
        *pscStrTmp++ = (char)(ucVal / VOS_TYPE_NUM10 + '0'); /* 10位数转换 */
        *pscStrTmp++ = (char)(ucVal % VOS_TYPE_NUM10 + '0'); /* 10位数转换 */
    } else {
        *pscStrTmp++ = (char)(ucVal / VOS_TYPE_NUM100 + '0');                   /* 100位数转换 */
        *pscStrTmp++ = (char)((ucVal / VOS_TYPE_NUM10) % VOS_TYPE_NUM10 + '0'); /* 10位数转换 */
        *pscStrTmp++ = (char)(ucVal % VOS_TYPE_NUM10 + '0');                    /* 10位数转换 */
    }

    return pscStrTmp;
}

#define VOS_NS_IPV4ADDRSZ 16
char *VOS_IpAddrToStrEx(uint32_t uiAddr, char *pscStr, int32_t iBufSize)
{
    char *pscBuf = VOS_NULL_PTR;

    if ((pscStr == VOS_NULL_PTR) || (iBufSize < VOS_NS_IPV4ADDRSZ)) {
        return VOS_NULL_PTR;
    }

    pscBuf = pscStr;
    pscBuf = vosUint8ToStr((uint8_t)(uiAddr >> VOS_TYPE_BITNUM24), pscBuf); /* 右移24位 */
    *pscBuf++ = '.';
    pscBuf = vosUint8ToStr((uint8_t)(uiAddr >> VOS_TYPE_BITNUM16), pscBuf); /* 右移16位 */
    *pscBuf++ = '.';
    pscBuf = vosUint8ToStr((uint8_t)(uiAddr >> VOS_TYPE_BITNUM8), pscBuf); /* 右移8位 */
    *pscBuf++ = '.';
    pscBuf = vosUint8ToStr((uint8_t)(uiAddr >> 0), pscBuf);
    *pscBuf++ = '\0';

    return pscStr;
}

char *VOS_StrRChr(const char *pscStr, const char scChar)
{
    if (pscStr == VOS_NULL_PTR) {
        return VOS_NULL_PTR;
    }

    return strrchr(pscStr, scChar);
}

char *VOS_StrpBrk(const char *pscStr1, const char *pscStr2)
{
    if ((pscStr1 == VOS_NULL_PTR) || (pscStr2 == VOS_NULL_PTR)) {
        return VOS_NULL_PTR;
    }

    return strpbrk(pscStr1, pscStr2);
}

uintptr_t VOS_strtoul(const char *pscNptr, char **ppscEndptr, int32_t siBase)
{
    if ((pscNptr != VOS_NULL_PTR) && (ppscEndptr != VOS_NULL_PTR)) {
        return strtoul(pscNptr, ppscEndptr, siBase);
    } else {
        return 0;
    }
}

uint64_t VOS_strtoull(const char *pscNptr, char **ppscEndptr, int32_t siBase)
{
    if ((pscNptr != VOS_NULL_PTR) && (ppscEndptr != VOS_NULL_PTR)) {
        return strtoull(pscNptr, ppscEndptr, siBase);
    } else {
        return 0;
    }
}

char *VOS_StrChr(const char *pscStr, const char scChar)
{
    if (pscStr == VOS_NULL_PTR) {
        return VOS_NULL_PTR;
    }

    return strchr(pscStr, scChar);
}

BOOL VOS_StrToIpAddrCheckValid(const char *pscStr, uint32_t *pulIpAddr)
{
    if (pscStr == VOS_NULL_PTR || (strlen(pscStr) == 0)) {
        return FALSE;
    }

    if (pulIpAddr == VOS_NULL_PTR) {
        return FALSE;
    }

    char scLastChar = *(pscStr + strlen(pscStr) - 1);
    if ((!isdigit(*pscStr)) || (!isdigit(scLastChar))) {
        return FALSE;
    }

    return TRUE;
}

#define MAX_NO_OF_DOTS 3
#define MAX_SEG_VALUE 255
int32_t VOS_StrToIpAddr(const char *pscStr, uint32_t *pulIpAddr)
{
    uint32_t ulTempValue = 0;
    uint32_t ulCheck3dot = 0;
    uint32_t ulCountDots = 0;
    const char *pscTempStr = pscStr;

    if (!VOS_StrToIpAddrCheckValid(pscStr, pulIpAddr)) {
        return VOS_ERROR;
    }

    while (*pscTempStr != '\0') {
        if (isdigit(*pscTempStr)) {
            ulTempValue *= VOS_TYPE_NUM10;
            ulTempValue += (uint32_t)((int32_t)(*pscTempStr - '0'));
            ulCountDots = 0;

            if (ulTempValue > MAX_SEG_VALUE) {
                return VOS_ERROR;
            }
        } else {
            if (*pscTempStr != '.') {
                /* If any character other than digit or '.' is found.
                It's invalid. */
                return VOS_ERROR;
            }

            ulCheck3dot++;

            if (ulCheck3dot > MAX_NO_OF_DOTS) {
                return VOS_ERROR;
            }

            *pulIpAddr = (*pulIpAddr << VOS_TYPE_BITNUM8) | ulTempValue;
            ulTempValue = 0;
            ulCountDots++;

            /* Check if consecutive dots are found, it's invalid */
            if (ulCountDots > 1) {
                return VOS_ERROR;
            }
        }

        pscTempStr++;
    }

    if (ulCheck3dot == MAX_NO_OF_DOTS) {
        *pulIpAddr = (*pulIpAddr << VOS_TYPE_BITNUM8) | ulTempValue;
        return VOS_OK;
    }

    return VOS_ERROR;
}

char *VOS_StrTrim(char *pscStr)
{
    char *pscRetStr = VOS_NULL_PTR;
    char *pscTemp = VOS_NULL_PTR;
    uint32_t uiCount = 0;

    if ((pscStr == VOS_NULL_PTR) || ((uintptr_t)pscStr) == 0xcccccccc) {
        return VOS_NULL_PTR;
    }

    pscTemp = pscRetStr = pscStr;

    /* copy the string to pRetStr and get  the char number */
    for (; *pscTemp != '\0'; pscTemp++, uiCount++) {
    }

    /* delete the space behind the pRetStr */
    while (uiCount > 0 && ' ' == *--pscTemp) {
        uiCount--;
    }

    if (pscRetStr == pscTemp) {
        /* if pRetStr  is only space */
    } else {
        /* delete the space before pRetStr */
        while (' ' == *pscRetStr) {
            pscRetStr++;
            uiCount--;
        }

        /* copy the right string to scRet */
    }

    *(pscRetStr + uiCount) = '\0';
    return pscRetStr;
}

BOOL charIsUpper(char scChar)
{
    if ((scChar >= 'A') && (scChar <= 'Z')) {
        return VOS_TRUE;
    }
    return VOS_FALSE;
}

char charToLower(char scChar)
{
    char scCharTmp = scChar;

    if (charIsUpper(scCharTmp)) {
        scCharTmp = (char)(scCharTmp - ('A' - 'a'));
    }

    return scCharTmp;
}

void charToLowerCommon(char *pscStr, uint32_t uiStrLen)
{
    char *pscStrTmp = pscStr;
    uint32_t uiCount = 0;
    while (uiCount < uiStrLen) {
        *pscStrTmp = charToLower(*pscStrTmp);
        pscStrTmp++;
        uiCount++;
    }
    return;
}

void VOS_StrToLower(char *pscStr)
{
    uint32_t uiSrcStrLen;
    if (pscStr == VOS_NULL_PTR) {
        return;
    }
    uiSrcStrLen = strlen(pscStr);
    charToLowerCommon(pscStr, uiSrcStrLen);
    return;
}

#ifdef __cplusplus
}
#endif
