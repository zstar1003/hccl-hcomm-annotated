/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * @defgroup v_stringlib 字符及字符串操作接口
 * @ingroup v_syslib
 */
#ifndef __V_STRINGLIB_H__
#define __V_STRINGLIB_H__

#include <string.h>
#include "vos_typdef.h"
#include "vos_base.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOS_StrLen strlen
#define VOS_StrNCmp strncmp
#define VOS_StrCmp strcmp
#define VOS_MemCmp memcmp

char *VOS_StrStr(const char *pscStr1, const char *pscStr2);
char *VOS_IpAddrToStrEx(uint32_t uiAddr, char *pscStr, int32_t iBufSize);
char *VOS_StrRChr(const char *pscStr, const char scChar);
char *VOS_StrpBrk(const char *pscStr1, const char *pscStr2);
uintptr_t VOS_strtoul(const char *pscNptr, char **ppscEndptr, int32_t siBase);
uint64_t VOS_strtoull(const char *pscNptr, char **ppscEndptr, int32_t siBase);
char *VOS_StrChr(const char *pscStr, const char scChar);
int32_t VOS_StrToIpAddr(const char *pscStr, uint32_t *pulIpAddr);
char *VOS_StrTrim(char *pscStr);
void VOS_StrToLower(char *pscStr);

#ifdef __cplusplus
}
#endif

#endif /* __V_STRINGLIB_H__ */
