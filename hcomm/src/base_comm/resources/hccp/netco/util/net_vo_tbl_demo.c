/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl_demo.h"

#ifdef __cplusplus
extern "C" {
#endif

int32_t NetTblDemoKeyCmp(NetTblDemoKey *key1Input, NetTblDemoKey *key2InDs)
{
    (void)key1Input;
    (void)key2InDs;
    return 0;
}

uint32_t NetTblDemoKeyH2N(NetTblDemoKey *keyH, NetTblDemoKey *keyN)
{
    (void)keyH;
    (void)keyN;
    return 0;
}

uint32_t NetTblDemoKeyN2H(NetTblDemoKey *keyN, NetTblDemoKey *keyH)
{
    (void)keyN;
    (void)keyH;
    return 0;
}

char *NetTblDemoKeyGetStr(NetTblDemoKey *key, uint8_t *buf, int32_t bufLen)
{
    (void)key;
    (void)buf;
    (void)bufLen;
    return "__NetTblDemoKeyGetStr";
}

uint32_t NetTblDemoH2N(NetTblDemo *kvH, NetTblDemo *kvN)
{
    (void)kvH;
    (void)kvN;
    return 0;
}

uint32_t NetTblDemoN2H(NetTblDemo *kvN, NetTblDemo *kvH)
{
    (void)kvN;
    (void)kvH;
    return 0;
}

char *NetTblDemoGetStr(NetTblDemo *kv, uint8_t *buf, int32_t bufLen)
{
    (void)kv;
    (void)buf;
    (void)bufLen;
    return "__NetTblDemoGetStr";
}

char *NetTblDemoValGetStr(NetTblDemoVal *val, uint8_t *buf, int32_t bufLen)
{
    (void)val;
    (void)buf;
    (void)bufLen;
    return "__NetTblDemoValGetStr";
}
#ifdef __cplusplus
}
#endif
