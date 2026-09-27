/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_OUT_DATA_H
#define NET_CO_OUT_DATA_H

#include "net_co_main.h"
#include "net_vo_tbl.h"
#include "bkf_ch_ser_adef.h"
#include "bkf_dc.h"
#include "bifrost_cncoi_puber.h"
#include "v_avll.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
struct TagNetCoOut {
    AVLL_TREE tblTypeSet;
    BkfChSerMng *chSerMng;
    BkfDc *dc;
    void *simpoBuilder;
    BifrostCncoiPuber *puber;

    uint8_t rsv2[0x10];
};

typedef struct TagNetCoOutTblTypeVTbl {
    uint16_t typeId; /* key */
    uint8_t pad1[2];
    char *name;
    uint16_t keyLen;
    uint8_t pad2[2];
    F_BKF_CMP keyCmp;
    F_BKF_GET_STR keyGetStr;
    F_BKF_GET_STR valGetStr;
} NetCoOutTblTypeVTbl;
#define NET_CO_OUT_TBL_TYPE_SIGN (0xae95)
typedef struct TagNetCoOutTblType {
    uint16_t sign;
    uint8_t pad1[2];
    AVLL_NODE avlNode;
    NetCoOutTblTypeVTbl vTbl;

    AVLL_TREE tblSet;
    int32_t tblCnt;
} NetCoOutTblType;

#define NET_CO_OUT_TBL_SIGN (0x97)
typedef struct TagNetCoOutTbl {
    uint8_t sign;
    uint8_t pad1;
    uint16_t valLen;
    NetCoOutTblType *tblType;
    AVLL_NODE avlNode;

    void *val;
    uint8_t key[0];
} NetCoOutTbl;

/* disp */
typedef struct TagNetCoOutTblStatCnt {
    int32_t tblTypeCnt;
    int32_t tblCnt;
} NetCoOutTblStatCnt;

#define NET_CO_OUT_GET_TBL_CTX_SIGN (0xae98)
typedef struct TagNetCoOutGetTblCtx {
    uint16_t sign;
    uint8_t pad1[2];

    BOOL hasTblType;
    BOOL tblTypeIsNew;
    uint16_t tblTypeId;
    int32_t tblTypeCnt;

    NetCoOutTbl *tbl;
    uint8_t tblKey[NET_TBL_KEY_LEN_MAX];
    int32_t tblCntTotal;
    int32_t tblCntCurTblType;
} NetCoOutGetTblCtx;
#pragma pack()

uint32_t NetCoOutDataNew(NetCo *co);
void NetCoOutDataDelete(NetCo *co);

NetCoOutTblType *NetCoOutAddTblType(NetCo *co, NetCoOutTblTypeVTbl *vTbl);
void NetCoOutDelTblType(NetCo *co, NetCoOutTblType *tblType);
void NetCoOutDelAllTblType(NetCo *co);
NetCoOutTblType *NetCoOutFindTblType(NetCo *co, uint16_t tblTypeId);
NetCoOutTblType *NetCoOutFindNextTblType(NetCo *co, uint16_t tblTypeId);
NetCoOutTblType *NetCoOutGetFirstTblType(NetCo *co, void **itorOutOrNull);
NetCoOutTblType *NetCoOutGetNextTblType(NetCo *co, void **itorInOut);

NetCoOutTbl *NetCoOutAddTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey, void *tblValOrNull, uint16_t valLen);
uint32_t NetCoOutUpdTbl(NetCo *co, NetCoOutTbl *tbl, void *tblValOrNullNew, uint16_t valLenNew);
void NetCoOutDelOneTbl(NetCo *co, NetCoOutTbl *tbl);
void NetCoOutDelAllTbl(NetCo *co, NetCoOutTblType *tblType);
NetCoOutTbl *NetCoOutFindTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey);
NetCoOutTbl *NetCoOutFindNextTbl(NetCo *co, NetCoOutTblType *tblType, void *tblKey);
NetCoOutTbl *NetCoOutGetFirstTbl(NetCo *co, NetCoOutTblType *tblType, void **itorOutOrNull);
NetCoOutTbl *NetCoOutGetNextTbl(NetCo *co, NetCoOutTblType *tblType, void **itorInOut);

/* display使用 */
char *NetCoOutGetTblTypeStr(NetCoOutTblType *tblType, uint8_t *buf, int32_t bufLen);
char *NetCoOutGetTblStr(NetCoOutTbl *tbl, uint8_t *buf, int32_t bufLen);

void NetCoOutGetTblStatCnt(NetCo *co, NetCoOutTblStatCnt *cnt);
void NetCoOutGetTblStatCntOfTblType(NetCo *co, NetCoOutTblType *tblType, NetCoOutTblStatCnt *cnt);
char *NetCoOutGetTblStatCntStr(NetCoOutTblStatCnt *cnt, uint8_t *buf, int32_t bufLen);

NetCoOutTblType *NetCoOutGetFirstTblByCtx(NetCo *co, NetCoOutGetTblCtx *ctx);
NetCoOutTblType *NetCoOutGetNextTblByCtx(NetCo *co, NetCoOutGetTblCtx *ctx);

#ifdef __cplusplus
}
#endif

#endif
