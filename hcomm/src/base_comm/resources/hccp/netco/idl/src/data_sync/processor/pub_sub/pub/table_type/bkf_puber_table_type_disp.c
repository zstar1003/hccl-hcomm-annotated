/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_puber_table_type_disp.h"
#include "bkf_puber_table_type_data.h"

#ifdef __cplusplus
extern "C" {
#endif

void BkfPuberTableTypeDispSummary(BkfPuberTableTypeMng *tableTypeMng, BkfDisp *disp)
{
    BkfPuberTableType *tableType = VOS_NULL;
    void *itor = VOS_NULL;
    int32_t typeCnt = 0;
    for (tableType = BkfPuberTableTypeGetFirst(tableTypeMng, &itor); tableType != VOS_NULL;
         tableType = BkfPuberTableTypeGetNext(tableTypeMng, &itor)) {
        typeCnt++;
    }

    BKF_DISP_PRINTF(disp, "----table type----\n");
    BKF_DISP_PRINTF(disp, "name(%s)/typeCnt(%d)\n", tableTypeMng->name, typeCnt);
}

void BkfPuberTableTypeDispType(BkfPuberTableTypeMng *tableTypeMng)
{
    int32_t cnt = 0;
    int32_t lastTableTypeId = 0;
    BkfPuberTableType *tableType = VOS_NULL;
    BkfDisp *disp = tableTypeMng->argInit->disp;
    if (!BKF_DISP_GET_LAST_3NUM(disp, &lastTableTypeId, &cnt, VOS_NULL)) {
        BkfPuberTableTypeDispSummary(tableTypeMng, disp);
        tableType = BkfPuberTableTypeGetFirst(tableTypeMng, VOS_NULL);
        cnt = 0;
    } else {
        tableType = BkfPuberTableTypeFindNext(tableTypeMng, (uint16_t)lastTableTypeId);
    }

    if (tableType == VOS_NULL) {
        BKF_DISP_PRINTF(disp, " ***total %d tableTyp(s), ***\n", cnt);
        return;
    }

    BKF_DISP_PRINTF(disp, "typeId[%d] = (%u)/cookie(%#x)/tupleUpdateCode(%#x)/tupleUpdateCode(%#x)\n", cnt,
        tableType->vTbl.tableTypeId, BKF_MASK_ADDR(tableType->vTbl.cookie),
        BKF_MASK_ADDR(tableType->vTbl.tupleUpdateCode), BKF_MASK_ADDR(tableType->vTbl.tupleUpdateCode));
    cnt++;
    BKF_DISP_SAVE_3NUM(disp, tableType->vTbl.tableTypeId, cnt, 0);
}

void BkfPuberTableTypeDispInit(BkfPuberTableTypeMng *tableTypeMng)
{
    BkfDisp *disp = tableTypeMng->argInit->disp;
    char *objName = tableTypeMng->name;
    uint32_t ret = BkfDispRegObj(disp, objName, tableTypeMng);
    if (ret != BKF_OK) {
        BKF_ASSERT(0);
        return;
    }

    (void)BKF_DISP_REG_FUNC(disp, BkfPuberTableTypeDispType, "disp puber table type info", objName, 0);
    tableTypeMng->dispInitOk = VOS_TRUE;
}

void BkfPuberTableTypeDispUninit(BkfPuberTableTypeMng *tableTypeMng)
{
    if (tableTypeMng->dispInitOk) {
        BkfDispUnregObj(tableTypeMng->argInit->disp, tableTypeMng->name);
    }
}

#ifdef __cplusplus
}
#endif
