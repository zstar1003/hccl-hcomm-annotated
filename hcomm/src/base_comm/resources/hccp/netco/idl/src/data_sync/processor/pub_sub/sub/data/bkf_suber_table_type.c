/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_subscriber.h"
#include "bkf_mem.h"
#include "bkf_suber_env.h"
#include "v_avll.h"
#include "securec.h"
#include "bkf_suber_table_type.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)

#define BKF_PUBER_TABLE_TYPE_HASH_MOD 0x0f
#define BKF_PUBER_TABLE_TYPE_GET_HASH_IDX(tableTypeId) BKF_GET_U8_FOLD4_VAL((uint8_t)(tableTypeId))

typedef struct tagBkfSuberTableType {
    AVLL_NODE avlNode;
    uint8_t mode;
    uint8_t pad[3];
    union {
        BkfSuberTableTypeVTbl vTbl;
        BkfSuberTableTypeVTblEx vTblEx;
    };
    uint8_t userData[0];
} BkfSuberTableType;

struct tagBkfSuberTableTypeMng {
    BkfSuberEnv *env;
    AVLL_TREE tableTypeSet;
};
#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuberTableTypeMng *BkfSuberTableTypeMngInit(BkfSuberTableTypeMngInitArg *initArg)
{
    uint32_t len = sizeof(BkfSuberTableTypeMng);
    BkfSuberTableTypeMng *tableTypeMng = BKF_MALLOC(initArg->env->memMng, len);
    if (tableTypeMng == VOS_NULL) {
        return VOS_NULL;
    }
    VOS_AVLL_INIT_TREE(tableTypeMng->tableTypeSet, (AVLL_COMPARE)Bkfuint16_tCmp,
        BKF_OFFSET(BkfSuberTableType, vTbl) + BKF_OFFSET(BkfSuberTableTypeVTbl, tableTypeId),
        BKF_OFFSET(BkfSuberTableType, avlNode));
    tableTypeMng->env = initArg->env;
    return tableTypeMng;
}

void BkfSuberTableTypeMngUnInit(BkfSuberTableTypeMng *tableTypeMng)
{
    BkfSuberTableType *tableType = VOS_NULL;
    BkfSuberTableType *tableTypeNext = VOS_NULL;

    for (tableType = VOS_AVLL_FIRST(tableTypeMng->tableTypeSet); tableType != VOS_NULL; tableType = tableTypeNext) {
        tableTypeNext = VOS_AVLL_NEXT(tableTypeMng->tableTypeSet, tableType->avlNode);
        VOS_AVLL_DELETE(tableTypeMng->tableTypeSet, tableType->avlNode);
        BKF_FREE(tableTypeMng->env->memMng, tableType);
    }
    BKF_FREE(tableTypeMng->env->memMng, tableTypeMng);
}

uint32_t BkfSuberTableTypeAdd(BkfSuberTableTypeMng *tableTypeMng, uint8_t mode, void *vtbl, void *userData,
    uint16_t userDataLen)
{
    BkfSuberTableType *tableType = VOS_AVLL_FIND(tableTypeMng->tableTypeSet,
        &(((BkfSuberTableTypeVTbl *)vtbl)->tableTypeId));
    if (tableType != VOS_NULL) {
        return BKF_ERR;
    }

    uint32_t len = sizeof(BkfSuberTableType) + userDataLen;
    tableType = BKF_MALLOC(tableTypeMng->env->memMng, len);
    if (tableType == VOS_NULL) {
        return BKF_ERR;
    }

    if (userDataLen != 0) {
        if (memcpy_s(tableType->userData, userDataLen, userData, userDataLen) != 0) {
            goto error;
        }
    }
    if (mode == BKF_SUBER_MODE_DEFAULT) {
        tableType->vTbl = *(BkfSuberTableTypeVTbl *)vtbl;
    } else {
        tableType->vTblEx = *(BkfSuberTableTypeVTblEx *)vtbl;
    }
    tableType->mode = mode;
    VOS_AVLL_INIT_NODE(tableType->avlNode);
    BOOL isInsOk = VOS_AVLL_INSERT(tableTypeMng->tableTypeSet, tableType->avlNode);
    if (!isInsOk) {
        goto error;
    }

    return BKF_OK;

error:
    BKF_FREE(tableTypeMng->env->memMng, tableType);
    return BKF_ERR;
}

uint32_t BkfSuberTableTypeDel(BkfSuberTableTypeMng *tableTypeMng, uint16_t type)
{
    BkfSuberTableType *tableType = VOS_AVLL_FIND(tableTypeMng->tableTypeSet, &type);
    if (tableType == VOS_NULL) {
        return BKF_OK;
    }
    VOS_AVLL_DELETE(tableTypeMng->tableTypeSet, tableType->avlNode);
    BKF_FREE(tableTypeMng->env->memMng, tableType);
    return BKF_OK;
}

BkfSuberTableTypeVTbl *BkfSuberTableTypeGetVtbl(BkfSuberTableTypeMng *tableTypeMng, uint16_t tablTypeId)
{
    BkfSuberTableType *tableType = VOS_AVLL_FIND(tableTypeMng->tableTypeSet, &tablTypeId);
    if (tableType == VOS_NULL) {
        return VOS_NULL;
    }

    return &tableType->vTbl;
}
void *BkfSuberTableTypeGetUserData(BkfSuberTableTypeMng *tableTypeMng, uint16_t tablTypeId)
{
    BkfSuberTableType *tableType = VOS_AVLL_FIND(tableTypeMng->tableTypeSet, &tablTypeId);
    if (tableType == VOS_NULL) {
        return VOS_NULL;
    }

    return tableType->userData;
}

uint8_t BkfSuberDataTableTypeGetMode(BkfSuberTableTypeVTbl *vTbl)
{
    if (vTbl == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_SUBER_MODE_DEFAULT;
    }
    BkfSuberTableType *tableType = BKF_MBR_PARENT(BkfSuberTableType, vTbl, vTbl);
    return tableType->mode;
}
#endif

#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif