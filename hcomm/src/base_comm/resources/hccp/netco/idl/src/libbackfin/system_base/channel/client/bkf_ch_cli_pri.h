/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_CH_CLI_PRI_H
#define BKF_CH_CLI_PRI_H

#include "bkf_ch_cli_adef.h"
#include "bkf_assert.h"
#include "v_avll.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#pragma pack(4)
#define BKF_CH_CLI_TYPE_HASH_MOD 0x0f
#define BKF_CH_CLI_GET_TYPE_HASH_IDX(typeId) BKF_GET_U8_FOLD4_VAL((uint8_t)(typeId))
typedef struct tagBkfChCliType BkfChCliType;
struct tagBkfChCliMng {
    BkfChCliMngInitArg argInit;
    char *name;

    uint8_t hasEnable : 1;
    uint8_t rsv1 : 7;
    uint8_t pad1[3];
    BkfChCliEnableArg argEnable;
    AVLL_TREE typeSet;
#ifndef BKF_CUT_AVL_CACHE
    BkfChCliType *typeCache[BKF_CH_CLI_TYPE_HASH_MOD + 1];
#endif
};

struct tagBkfChCliType {
    AVLL_NODE avlNode;
    BkfChCliTypeVTbl vTbl;
    BkfChCli *ch;
};

/* data op */
STATIC BkfChCliType *BkfChCliAddType(BkfChCliMng *chMng, BkfChCliTypeVTbl *vTbl);
STATIC void BkfChCliDelType(BkfChCliMng *chMng, BkfChCliType *chType);
STATIC void BkfChCliDelAllType(BkfChCliMng *chMng);
STATIC BkfChCliType *BkfChCliFindType(BkfChCliMng *chMng, uint8_t typeId);
STATIC BkfChCliType *BkfChCliFindNextType(BkfChCliMng *chMng, uint8_t typeId);
STATIC BkfChCliType *BkfChCliGetFirstType(BkfChCliMng *chMng, void **itorOutOrNull);
STATIC BkfChCliType *BkfChCliGetNextType(BkfChCliMng *chMng, void **itorInOut);

#pragma pack()

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
