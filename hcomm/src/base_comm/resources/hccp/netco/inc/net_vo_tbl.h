/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_H
#define NET_VO_TBL_H

#include "bkf_comm.h"
#include "net_vo_tbl_demo.h"
#include "net_vo_tbl_comm_info.h"
#include "net_vo_tbl_oper.h"
#include "net_vo_tbl_adj.h"
#include "net_vo_tbl_rank.h"
#include "net_vo_tbl_rank_dist.h"
#include "net_vo_tbl_root_rank.h"

#ifdef __cplusplus
extern "C" {
#endif

enum {
    NET_TBL_TYPE_COMM_INFO,
    NET_TBL_TYPE_OPER,
    NET_TBL_TYPE_ADJ,
    NET_TBL_TYPE_RANK,
    NET_TBL_TYPE_RANK_DIST,
    NET_TBL_TYPE_ROOT_RANK,

    NET_TBL_TYPE_CNT
};
#define NET_TBL_TYPE_IS_VALID(typeId) ((typeId) < NET_TBL_TYPE_CNT)

#define NET_TBL_KEY_LEN_MAX (512)
#define NET_TBL_KEY_LEN_IS_VALID(len) (((uint16_t)(len) != 0) && ((uint16_t)(len) <= NET_TBL_KEY_LEN_MAX))
/* val 长度用uint16_t标识，最长为uint16_t最大值 */

typedef uint32_t (*F_NET_TBL_CONVERT)(void *from, void *to);

const char *NetTblTypeGetStr(uint16_t typeId);

#ifdef __cplusplus
}
#endif

#endif
