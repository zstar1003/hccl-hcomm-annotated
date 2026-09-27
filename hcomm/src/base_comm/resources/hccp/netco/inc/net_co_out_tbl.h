/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_OUT_TBL_H
#define NET_CO_OUT_TBL_H

#include "net_co_main.h"
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
typedef uint32_t (*F_NET_CO_TBL_ADD_UPD)(NetCo *co, void *kv, uint32_t dataLen);
typedef void (*F_NET_CO_TBL_DEL)(NetCo *co, void *key);

/**
 * @brief 通知demo表添加更新
 *
 * @param[in] co 句柄
 * @param[in] kv 表数据
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetCoTblDemoAddUpd(NetCo *co, NetTblDemo *kv);

/**
 * @brief 通知demo表删除
 *
 * @param[in] co 句柄
 * @param[in] key 表key
 * @return 无
 */
void NetCoTblDemoDel(NetCo *co, NetTblDemoKey *key);

uint32_t NetCoTblCommInfoAddUpd(NetCo *co, NetTblCommInfo *kv, uint32_t dataLen);
void NetCoTblCommInfoDel(NetCo *co, NetTblCommInfoKey *key);

uint32_t NetCoTblOperAddUpd(NetCo *co, NetTblOperator *kv, uint32_t dataLen);
void NetCoTblOperDel(NetCo *co, NetTblOperatorKey *key);

uint32_t NetCoTblAdjAddUpd(NetCo *co, NetTblAdjacency *kv, uint32_t dataLen);
void NetCoTblAdjDel(NetCo *co, NetTblAdjacencyKey *key);

uint32_t NetCoTblRankAddUpd(NetCo *co, NetTblGlobalRank *kv, uint32_t dataLen);
void NetCoTblRankDel(NetCo *co, NetTblGlobalRankKey *key);

uint32_t NetCoTblRankDistAddUpd(NetCo *co, NetTblRankDistributeInfo *kv, uint32_t dataLen);
void NetCoTblRankDistDel(NetCo *co, NetTblRankDistributeInfoKey *key);

uint32_t NetCoTblRootRankAddUpd(NetCo *co, NetTblRootRank *kv, uint32_t dataLen);
void NetCoTblRootRankDel(NetCo *co, NetTblRootRankKey *key);

#ifdef __cplusplus
}
#endif

#endif
