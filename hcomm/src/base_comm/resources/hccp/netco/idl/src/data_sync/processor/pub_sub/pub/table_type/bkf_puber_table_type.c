/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((tableTypeMng)->log)
#define BKF_MOD_NAME ((tableTypeMng)->name)

#include "bkf_puber_table_type.h"
#include "bkf_puber_table_type_data.h"
#include "bkf_puber_table_type_disp.h"

#ifdef __cplusplus
extern "C" {
#endif

BkfPuberTableTypeMng *BkfPuberTableTypeInit(BkfPuberInitArg *arg)
{
    BkfPuberTableTypeMng *tableTypeMng = BkfPuberTableTypeDataInit(arg);
    if (tableTypeMng != VOS_NULL) {
        BkfPuberTableTypeDispInit(tableTypeMng);
        BKF_LOG_INFO(BKF_LOG_HND, "tableTypeMng(%#x, %s), init ok\n", BKF_MASK_ADDR(tableTypeMng), arg->name);
    }

    return tableTypeMng;
}

void BkfPuberTableTypeUninit(BkfPuberTableTypeMng *tableTypeMng)
{
    if (tableTypeMng != VOS_NULL) {
        BKF_LOG_INFO(BKF_LOG_HND, "tableTypeMng(%#x, %s), uninit\n", BKF_MASK_ADDR(tableTypeMng),
            tableTypeMng->argInit->name);
        BkfPuberTableTypeDispUninit(tableTypeMng);
        BkfPuberTableTypeDataUninit(tableTypeMng);
    }
}

uint32_t BkfPuberTableTypeAttachEx(BkfPuberTableTypeMng *tableTypeMng, BkfPuberTableTypeVTbl *vTbl, void *userData,
    uint16_t userDataLen)
{
    BkfPuberTableType *tableType = BkfPuberTableTypeAdd(tableTypeMng, vTbl, userData, userDataLen);
    return (tableType != VOS_NULL) ? BKF_OK : BKF_ERR;
}

BkfPuberTableTypeVTbl *BkfPuberTableTypeGetVTbl(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId)
{
    BkfPuberTableType *tableType = BkfPuberTableTypeFind(tableTypeMng, tableTypeId);
    return (tableType != VOS_NULL) ? &tableType->vTbl : VOS_NULL;
}

void *BkfPuberTableTypeGetUserData(BkfPuberTableTypeMng *tableTypeMng, uint16_t tableTypeId)
{
    BkfPuberTableType *tableType = BkfPuberTableTypeFind(tableTypeMng, tableTypeId);
    return (tableType != VOS_NULL) ? tableType->userData : VOS_NULL;
}

void BkfPuberTableTypeDeattach(BkfPuberTableTypeMng *tableTypeMng, uint16_t tabType)
{
    BkfPuberTableType *tableType = BkfPuberTableTypeFind(tableTypeMng, tabType);
    if (tableType) {
        BkfPuberTableTypeDelete(tableType);
    }
}

#ifdef __cplusplus
}
#endif
