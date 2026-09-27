/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_DATA_H
#define BKF_PUBER_DATA_H

#include "bkf_puber_adef.h"
#include "bkf_puber_table_type.h"
#include "bkf_puber_conn.h"

#ifdef __cplusplus
extern "C" {
#endif

#pragma pack(4)
struct TagBkfPuber {
    BkfPuberInitArg argInit;
    char *name;
    BkfLog *log;
    char *svcName;
    uint8_t selfInstIdHasSet : 1;
    uint8_t dispInitOk : 1;
    uint8_t rsv : 6;
    uint8_t pad1[3];
    uint32_t selfInstId;
    BkfPuberTableTypeMng *tableTypeMng;
    BkfPuberConnMng *connMng;
};
#pragma pack()

#ifdef __cplusplus
}
#endif

#endif
