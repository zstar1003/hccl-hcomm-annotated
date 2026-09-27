/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_co_log_cnt.h"
#include "net_co_main_data.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif
uint32_t NetCoLogCntInit(NetCo *co)
{
    char name[BKF_NAME_LEN_MAX + 1];
    int err = snprintf_truncated_s(name, sizeof(name), "%s_logCnt", co->name);
    if (err <= 0) {
        return BKF_ERR;
    }
    BkfLogCntInitArg arg = {
        .name = name,
        .memMng = co->memMng,
        .disp = co->disp,
        .modCntMax = BKF_1K,
    };
    co->logCnt = BkfLogCntInit(&arg);
    return (co->logCnt != VOS_NULL) ? BKF_OK : BKF_ERR;
}

void NetCoLogCntUninit(NetCo *co)
{
    if (co->logCnt != VOS_NULL) {
        BkfLogCntUninit(co->logCnt);
        co->logCnt = VOS_NULL;
    }
}

#ifdef __cplusplus
}
#endif
