/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "net_vo_tbl.h"

#ifdef __cplusplus
extern "C" {
#endif
const char *NetTblTypeGetStr(uint16_t typeId)
{
    static const char *typeStr[] = {
        "NET_TBL_TYPE_DEMO",
        "NET_TBL_TYPE_COMM_INFO",
        "NET_TBL_TYPE_OPERATOR",
        "NET_TBL_TYPE_ADJACENCY",
        "NET_TBL_TYPE_RANK",
        "NET_TBL_TYPE_RANK_DISTRIBUTE",
        "NET_TBL_TYPE_ROOT_RANK",
    };
    return BKF_GET_NUM_STR(typeId, typeStr);
}

#ifdef __cplusplus
}
#endif
