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
#include "net_co_in_disp.h"
#include "net_co_in_data.h"
#include "net_co_main_data.h"

#ifdef __cplusplus
extern "C" {
#endif
void NetCoInDispSummary(NetCo *co)
{
    BkfDisp *disp = co->disp;
    NetCoIn *in = co->in;

    BKF_DISP_PRINTF(disp, "chCliMng(%#x)\n", BKF_MASK_ADDR(in->chCliMng));
    BKF_DISP_PRINTF(disp, "simpoBuilder(%#x)\n", BKF_MASK_ADDR(in->simpoBuilder));
    BKF_DISP_PRINTF(disp, "suber(%#x)\n", BKF_MASK_ADDR(in->suber));
}

uint32_t NetCoInDispInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfDisp *disp = co->disp;
    char *objName = co->name;
    (void)BKF_DISP_REG_FUNC(disp, NetCoInDispSummary, "disp in summary", objName, 0);

    return BKF_OK;
}

void NetCoInDispUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");
}

#ifdef __cplusplus
}
#endif
