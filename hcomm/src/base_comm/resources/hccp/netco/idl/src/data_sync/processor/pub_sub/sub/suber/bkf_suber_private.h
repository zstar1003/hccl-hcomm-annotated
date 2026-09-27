/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBER_SUB_PRIVATE_H
#define BKF_SUBER_SUB_PRIVATE_H

#include "bkf_suber_conn.h"
#include "bkf_subscriber.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma pack(4)

struct tagBkfSuber {
    BkfSuberEnv env;
    BkfLog *log;
    BkfSuberDataMng *dataMng;
    BkfSuberConnMng *connMng;
};
#pragma pack()
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif