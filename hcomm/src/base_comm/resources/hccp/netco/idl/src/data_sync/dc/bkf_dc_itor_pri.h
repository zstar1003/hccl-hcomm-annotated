/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DC_ITOR_PRI_H
#define BKF_DC_ITOR_PRI_H

#include "bkf_comm.h"
#include "v_avll.h"
#include "vos_base.h"
#include "bkf_dl.h"
#include "bkf_dc_pri.h"
#include "bkf_dc_itor.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)
/* tuple itor */
#define BKF_DC_TUPLE_KEY_ITOR_SIGN 0xAB898001
struct tagBkfDcTupleKeyItor {
    uint32_t sign;
    uint8_t getFirstTuple : 1;
    uint8_t rsv : 7;
    uint8_t pad1[3];
    BkfDcTableType *tableType;
    void *tupleKey;
    void *tupleVal;
    uint8_t *sliceKey;        /* 指向后面buf缓冲的数据key部分 */
    uint8_t *lastGetTupleKey; /* 指向后面buf缓冲的数据val部分 */
    uint8_t buf[0];
};

#define BKF_DC_TUPLE_SEQ_ITOR_SIGN 0xAB898002
struct tagBkfDcTupleSeqItor {
    uint32_t sign;
    BkfDcTable *table;
    BkfDlNode dlNode; /* in table */
    uint8_t lockDel : 1;
    uint8_t softDel : 1;
    uint8_t rsv : 6;
    uint8_t pad1[3];
    BkfDcTupleItorVTbl vTbl;
    char *name;

    BkfDcTuple *nextProcTuple; /* 如果为空，代表当前没有需要通过seq获取的节点 */
    uint64_t nextProcTupleSeq;
};

/* func */
BkfDcTupleKeyItor *BkfDcAddTupleKeyItor(BkfDc *dc, BkfDcTable *table);
void BkfDcDelTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor);

BkfDcTupleSeqItor *BkfDcAddTupleSeqItor(BkfDc *dc, BkfDcTable *table, BkfDcTupleItorVTbl *vTbl);
BOOL BkfDcDelTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor);
BkfDcTupleSeqItor *BkfDcGetFirstTupleSeqItor(BkfDc *dc, BkfDcTable *table, void **itorOutOrNull);
BkfDcTupleSeqItor *BkfDcGetNextTupleSeqItor(BkfDc *dc, BkfDcTable *table, void **itorInOut);
BkfDcTupleSeqItor *BkfDcGetNextTupleSeqItorBySeqItor(BkfDc *dc, BkfDcTable *table, BkfDcTupleSeqItor *seqItor);

void BkfDcUpdTableAllTupleSeqItorForTuple(BkfDc *dc, BkfDcTable *table, BkfDcTuple *tuple);
typedef void (*F_DC_TUPLE_SEQ_ITOR_PROC)(BkfDcTupleSeqItor *seqItor);
void BkfDcNtfTableAllSeqItor(BkfDc *dc, BkfDcTable *table, F_DC_TUPLE_SEQ_ITOR_PROC proc);

#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
