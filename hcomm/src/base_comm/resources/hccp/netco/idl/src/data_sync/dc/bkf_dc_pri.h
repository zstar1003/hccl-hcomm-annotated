/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DC_PRI_H
#define BKF_DC_PRI_H

#include "bkf_comm.h"
#include "bkf_assert.h"
#include "v_avll.h"
#include "bkf_dl.h"
#include "bkf_dc.h"
#include "bkf_dc_itor.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
typedef struct tagBkfDcTableType BkfDcTableType;
typedef struct tagBkfDcSlice BkfDcSlice;
typedef struct tagBkfDcTable BkfDcTable;
typedef struct tagBkfDcTuple BkfDcTuple;

/* dc */
#define BKF_DC_TABLE_TYPE_HASH_MOD 0x0f
#define BKF_DC_GET_TABLE_TYPE_HASH_IDX(tableTypeId) BKF_GET_U8_FOLD4_VAL((uint8_t)(tableTypeId)) /* 强制取低8bit */
#define BKF_DC_SLICE_HASH_MOD 0x3f
#define BKF_DC_GET_SLICE_HASH_IDX(sliceKey) ((*(uint32_t *)(sliceKey)) & BKF_DC_SLICE_HASH_MOD)
struct tagBkfDc {
    BkfDcInitArg argInit;
    char *name;
    BkfLog *log;

    AVLL_TREE tableTypeSet;
    BkfDcTableType *tableTypeCache[BKF_DC_TABLE_TYPE_HASH_MOD + 1];
    AVLL_TREE sliceSet;
    BkfDcSlice *sliceCache[BKF_DC_SLICE_HASH_MOD + 1];
    uint64_t seq64Seed;
    void *cookieOfSpy;
};
BKF_STATIC_ASSERT(BKF_MBR_IS_FIRST(BkfDc, argInit));

#define BKF_DC_SEQ_INVALID 0

/* tableType */
struct tagBkfDcTableType {
    AVLL_NODE avlNode;
    BkfDcTableTypeVTbl vTbl;
    char *name;
    uint8_t complete : 1;
    uint8_t rsv : 7;
    uint8_t pad1[3];
    int32_t tupleCntAddUpd;
};

/* slice */
#define BKF_DC_TABLE_HASH_MOD 0x0f
#define BKF_DC_GET_TABLE_HASH_IDX(tableTypeId) BKF_GET_U8_FOLD4_VAL((uint8_t)(tableTypeId)) /* 强制取低8bit */
struct tagBkfDcSlice {
    BkfDc *dc;
    AVLL_NODE avlNode;
    AVLL_TREE tableSet;
    BkfDcTable *tableCache[BKF_DC_TABLE_HASH_MOD + 1];

    uint8_t key[0];
};

/* table */
#define BKF_DC_TUPLE_HASH_MOD 0x3f
#define BKF_DC_GET_TUPLE_HASH_IDX(tupleKey) ((*(uint32_t *)(tupleKey)) & BKF_DC_TUPLE_HASH_MOD)
struct tagBkfDcTable {
    BkfDcTableType *tableType;
    BkfDcSlice *slice;
    AVLL_NODE avlNode;
    uint16_t tableTypeId;
    uint8_t isRelease : 1;
    uint8_t rsv : 7;
    uint8_t pad1[1];

    AVLL_TREE tupleSet;
    BkfDcTuple *tupleCache[BKF_DC_TUPLE_HASH_MOD + 1];
    BkfDl tupleSetBySeq;
    BkfDl tupleSeqItorSet;

    BkfJobId *jobIdDelTuple;
    BkfDcTupleKeyItor *keyItorDelTuple;
};

/* tuple */
struct tagBkfDcTuple {
    uint64_t sign : 3;
    uint64_t isAddUpd : 1;
    uint64_t seq : 60;
    AVLL_NODE avlNode;
    BkfDlNode dlNodeBySeq;

    uint8_t keyVal[0];
};

#define BKF_DC_TUPLE_SEQ_MAX ((uint64_t)0x0fffffffffffffff)

#define BKF_DC_TUPLE_SIGN 5

#define BKF_DC_TUPLE_GET_VAL(tuple, tableType)                                                                         \
    (((tableType)->vTbl.tupleValLen > 0) ? (&(tuple)->keyVal[0] + (tableType)->vTbl.tupleKeyLen) : VOS_NULL)

/* func */
BkfDcTableType *BkfDcAddTableType(BkfDc *dc, BkfDcTableTypeVTbl *vTbl);
void BkfDcDelTableType(BkfDc *dc, BkfDcTableType *tableType);
void BkfDcDelAllTableType(BkfDc *dc);
BkfDcTableType *BkfDcFindTableType(BkfDc *dc, uint16_t tableTypeId);
BkfDcTableType *BkfDcFindNextTableType(BkfDc *dc, uint16_t tableTypeId);
BkfDcTableType *BkfDcGetFirstTableType(BkfDc *dc, void **itorOutOrNull);
BkfDcTableType *BkfDcGetNextTableType(BkfDc *dc, void **itorInOut);

BkfDcSlice *BkfDcAddSlice(BkfDc *dc, void *sliceKey);
void BkfDcDelSlice(BkfDc *dc, BkfDcSlice *slice);
void BkfDcDelAllSlice(BkfDc *dc);
BkfDcSlice *BkfDcFindSlice(BkfDc *dc, void *sliceKey);
BkfDcSlice *BkfDcFindNextSlice(BkfDc *dc, void *sliceKey);
BkfDcSlice *BkfDcGetFirstSlice(BkfDc *dc, void **itorOutOrNull);
BkfDcSlice *BkfDcGetNextSlice(BkfDc *dc, void **itorInOut);

BkfDcTable *BkfDcAddTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId);
void BkfDcDelTable(BkfDc *dc, BkfDcTable *table);
void BkfDcDelAllTable(BkfDc *dc, BkfDcSlice *slice);
BkfDcTable *BkfDcFindTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId);
BkfDcTable *BkfDcFindNextTable(BkfDc *dc, BkfDcSlice *slice, uint16_t tableTypeId);
BkfDcTable *BkfDcGetFirstTable(BkfDc *dc, BkfDcSlice *slice, void **itorOutOrNull);
BkfDcTable *BkfDcGetNextTable(BkfDc *dc, BkfDcSlice *slice, void **itorInOut);

BkfDcTuple *BkfDcAddTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey, void *tupleValOrNull);
uint32_t BkfDcUpdTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple, void *tupleValOrNull);
void BkfDcSetTupleDelState(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple);
BOOL BkfDcTry2FreeTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple);
BOOL BkfDcTry2FreeAllTuple(BkfDc *dc, BkfDcTable *table);
BkfDcTuple *BkfDcFindTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey);
BkfDcTuple *BkfDcFindNextTuple(BkfDc *dc, BkfDcTable *table, void *tupleKey);
BkfDcTuple *BkfDcGetFirstTuple(BkfDc *dc, BkfDcTable *table, void **itorOutOrNull);
BkfDcTuple *BkfDcGetNextTuple(BkfDc *dc, BkfDcTable *table, void **itorInOut);

uint32_t BkfDcTupleLimitInit(BkfDc *dc);
BOOL BkfDcTupleLimitMayAddTuple(BkfDc *dc, uint16_t tableTypeId);
uint32_t BkfDcTupleLimitSysLog(BkfDc *dc, uint16_t tableTypeId);

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
