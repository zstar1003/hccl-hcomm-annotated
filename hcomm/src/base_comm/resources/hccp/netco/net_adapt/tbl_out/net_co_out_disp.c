/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)
#include "net_co_out_disp.h"
#include "net_co_out_data.h"
#include "net_co_main_data.h"

#ifdef __cplusplus
extern "C" {
#endif
void NetCoOutDispSummary(NetCo *co)
{
    BkfDisp *disp = co->disp;
    NetCoOut *out = co->out;

    BKF_DISP_PRINTF(disp, "chSerMng(%#x)\n", BKF_MASK_ADDR(out->chSerMng));
    BKF_DISP_PRINTF(disp, "dc(%#x)\n", BKF_MASK_ADDR(out->dc));
    BKF_DISP_PRINTF(disp, "simpoBuilder(%#x)\n", BKF_MASK_ADDR(out->simpoBuilder));
    BKF_DISP_PRINTF(disp, "puber(%#x)\n", BKF_MASK_ADDR(out->puber));

    NetCoOutTblStatCnt cnt = {0};
    NetCoOutGetTblStatCnt(co, &cnt);
    uint8_t buf[BKF_1K];
    BKF_DISP_PRINTF(disp, "statCnt = %s \n", NetCoOutGetTblStatCntStr(&cnt, buf, sizeof(buf)));
}

void NetCoOutDispAllTbl(NetCo *co)
{
    BkfDisp *disp = co->disp;
    uint8_t buf[BKF_1K];
    uint8_t buf2[BKF_1K];
    NetCoOutGetTblCtx cur = {0};
    NetCoOutTblType *tblType;
    NetCoOutGetTblCtx *last = BKF_DISP_GET_LAST_CTX(disp, VOS_NULL);
    if (last == VOS_NULL) {
        NetCoOutTblStatCnt cnt = {0};
        NetCoOutGetTblStatCnt(co, &cnt);
        BKF_DISP_PRINTF(disp, "statCnt = %s\n", NetCoOutGetTblStatCntStr(&cnt, buf, sizeof(buf)));

        tblType = NetCoOutGetFirstTblByCtx(co, &cur);
    } else {
        cur = *last;
        tblType = NetCoOutGetNextTblByCtx(co, &cur);
    }

    if (tblType != VOS_NULL) {
        if (cur.tblTypeIsNew) {
            NetCoOutTblStatCnt cnt = {0};
            NetCoOutGetTblStatCntOfTblType(co, tblType, &cnt);
            BKF_DISP_PRINTF(disp, "======================================\n");
            BKF_DISP_PRINTF(disp, "tblType[%d] = [%s], statCnt = %s\n", cur.tblCntTotal - 1,
                NetCoOutGetTblTypeStr(tblType, buf, sizeof(buf)), NetCoOutGetTblStatCntStr(&cnt, buf2, sizeof(buf2)));
        }

        if (cur.tbl != VOS_NULL) {
            BKF_DISP_PRINTF(disp, "++++tbl[%d] = %s\n", cur.tblCntCurTblType - 1,
                NetCoOutGetTblStr(cur.tbl, buf, sizeof(buf)));
        }

        BKF_DISP_SAVE_CTX(disp, &cur, sizeof(cur), VOS_NULL, 0);
    } else {
        BKF_DISP_PRINTF(disp, " ***total %d tblType(s), %d tbl(s) ***\n", cur.tblTypeCnt, cur.tblCntTotal);
    }
}

uint32_t NetCoOutDispInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfDisp *disp = co->disp;
    char *objName = co->name;
    (void)BKF_DISP_REG_FUNC(disp, NetCoOutDispSummary, "disp out summary", objName, 0);
    (void)BKF_DISP_REG_FUNC(disp, NetCoOutDispAllTbl, "disp out tbl", objName, 0);

    return BKF_OK;
}

void NetCoOutDispUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");
}

#ifdef __cplusplus
}
#endif
