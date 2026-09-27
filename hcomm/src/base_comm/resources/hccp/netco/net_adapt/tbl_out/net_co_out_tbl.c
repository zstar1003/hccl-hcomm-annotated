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

#include <sys/types.h>
#include <unistd.h>
#include "net_co_out_data.h"
#include "net_co_main_data.h"
#include "net_co_in_out_comm.h"
#include "net_vo_tbl.h"
#include "bifrost_cncoi_puber_comminfo.h"
#include "bifrost_cncoi_puber_operator.h"
#include "bifrost_cncoi_puber_adjacency.h"
#include "bifrost_cncoi_puber_rank_distribute.h"
#include "bifrost_cncoi_puber_rank.h"
#include "bifrost_cncoi_puber_root_rank.h"
#include "bifrost_cncoi_build_multi_layer.h"
#include "vos_assert.h"
#include "net_co_out_tbl.h"

#ifdef __cplusplus
extern "C" {
#endif
#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
#define NET_CO_OUT_TBL_CNT_MAX (0x10)

#pragma pack()
STATIC uint32_t NetCoOutAddUpdTbl(NetCo *co, uint16_t tblTypeId, void *key, void *val, uint16_t valLen);
STATIC void NetCoOutDelTbl(NetCo *co, uint16_t tblTypeId, void *key);

STATIC uint32_t NetCoOutBuildCommInfoTupleKey(NetTblCommInfoKey *from, BifrostCncoiComminfoKeyT *to);
STATIC uint32_t NetCoOutBuildCommInfoTblKey(BifrostCncoiComminfoKeyT *from, NetTblCommInfoKey *to);
STATIC uint32_t NetCoOutInitCommInfo(NetCo *co);
STATIC int32_t NetCoOutOnCommInfoTupleUpdCode(NetCo *co, SimpoBuilderT *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiComminfoKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

STATIC uint32_t NetCoOutBuildOperTupleKey(NetTblOperatorKey *from, BifrostCncoiOperatorKeyT *to);
STATIC uint32_t NetCoOutBuildOperTblKey(BifrostCncoiOperatorKeyT *from, NetTblOperatorKey *to);
STATIC uint32_t NetCoOutInitOper(NetCo *co);
STATIC int32_t NetCoOutOnOperTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiOperatorKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

STATIC uint32_t NetCoOutBuildAdjTupleKey(NetTblAdjacencyKey *from, BifrostCncoiAdjacencyKeyT *to);
STATIC uint32_t NetCoOutBuildAdjTblKey(BifrostCncoiAdjacencyKeyT *from, NetTblAdjacencyKey *to);
STATIC uint32_t NetCoOutInitAdj(NetCo *co);
STATIC int32_t NetCoOutOnAdjTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiAdjacencyKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

STATIC uint32_t NetCoOutBuildRankTupleKey(NetTblGlobalRankKey *from, BifrostCncoiRankKeyT *to);
STATIC uint32_t NetCoOutBuildRankTblKey(BifrostCncoiRankKeyT *from, NetTblGlobalRankKey *to);
STATIC uint32_t NetCoOutInitRank(NetCo *co);
STATIC int32_t NetCoOutOnRankTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

STATIC uint32_t NetCoOutBuildRankDistTupleKey(NetTblRankDistributeInfoKey *from, BifrostCncoiRankDistributeKeyT *to);
STATIC uint32_t NetCoOutBuildRankDistTblKey(BifrostCncoiRankDistributeKeyT *from, NetTblRankDistributeInfoKey *to);
STATIC uint32_t NetCoOutInitRankDist(NetCo *co);
STATIC int32_t NetCoOutOnRankDistTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankDistributeKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

STATIC uint32_t NetCoOutBuildRootRankTupleKey(NetTblRootRankKey *from, BifrostCncoiRootRankKeyT *to);
STATIC uint32_t NetCoOutBuildRootRankTblKey(BifrostCncoiRootRankKeyT *from, NetTblRootRankKey *to);
STATIC uint32_t NetCoOutInitRootRank(NetCo *co);
STATIC int32_t NetCoOutOnRootRankTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRootRankKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen);

#endif

#if BKF_BLOCK("公有函数定义")
STATIC const BkfModVTbl g_NetCoOutTblModVTbl[] = {
    {(F_BKF_DO)NetCoOutInitCommInfo, (F_BKF_DOV)VOS_NULL},
    {(F_BKF_DO)NetCoOutInitOper, (F_BKF_DOV)VOS_NULL},
    {(F_BKF_DO)NetCoOutInitAdj, (F_BKF_DOV)VOS_NULL},
    {(F_BKF_DO)NetCoOutInitRank, (F_BKF_DOV)VOS_NULL},
    {(F_BKF_DO)NetCoOutInitRankDist, (F_BKF_DOV)VOS_NULL},
    {(F_BKF_DO)NetCoOutInitRootRank, (F_BKF_DOV)VOS_NULL},
};
uint32_t NetCoOutTblInit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    uint32_t ret = BkfModsInit(g_NetCoOutTblModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoOutTblModVTbl));
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoOutTblUninit(NetCo *co)
{
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "\n");

    BkfModsUninit(g_NetCoOutTblModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoOutTblModVTbl));
}

uint32_t NetCoTblDemoAddUpd(NetCo *co, NetTblDemo *kv)
{
    (void)co;
    (void)kv;
    VOS_ASSERT(0);
    return BKF_ERR;
}

void NetCoTblDemoDel(NetCo *co, NetTblDemoKey *key)
{
    (void)co;
    (void)key;
    VOS_ASSERT(0);
}

uint32_t NetCoTblCommInfoAddUpd(NetCo *co, NetTblCommInfo *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "packNum %u, rankNum %u, dataLen %u\n", VOS_NTOHS(kv->val.packetNum),
        VOS_NTOHS(kv->val.rankNum), dataLen);
    if ((sizeof(NetTblCommInfo) + (VOS_NTOHS(kv->val.rankNum) * sizeof(NetTblRankInfo))) > dataLen) {
        BKF_LOG_ERROR(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblCommInfoN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblCommInfoGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblCommInfoVal) + sizeof(NetTblRankInfo) * kv->val.rankNum;
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_COMM_INFO, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiComminfoKeyT tupleKey;
    ret = NetCoOutBuildCommInfoTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberComminfoUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberComminfoNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblCommInfoDel(NetCo *co, NetTblCommInfoKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblCommInfoKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblCommInfoKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_COMM_INFO, key);
    BifrostCncoiComminfoKeyT tupleKey;
    uint32_t ret = NetCoOutBuildCommInfoTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberComminfoDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}

uint32_t NetCoTblOperAddUpd(NetCo *co, NetTblOperator *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dataLen %u\n", dataLen);
    if (sizeof(NetTblOperator) > dataLen) {
        BKF_LOG_ERROR(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblOperatorN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblOperatorGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblOperatorVal);
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_OPER, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiOperatorKeyT tupleKey;
    ret = NetCoOutBuildOperTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberOperatorUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberOperatorNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblOperDel(NetCo *co, NetTblOperatorKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblOperatorKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblOperatorKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_OPER, key);
    BifrostCncoiOperatorKeyT tupleKey;
    uint32_t ret = NetCoOutBuildOperTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberOperatorDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}

uint32_t NetCoTblAdjAddUpd(NetCo *co, NetTblAdjacency *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dstRankNum %u, dataLen %u\n", VOS_NTOHS(kv->val.dstRankNum), dataLen);
    if ((sizeof(NetTblAdjacency) + (VOS_NTOHS(kv->val.dstRankNum) * sizeof(NetTblAdjInfo))) > dataLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblAdjacencyN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblAdjacencyGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblAdjacencyVal) + sizeof(NetTblAdjInfo) * kv->val.dstRankNum;
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_ADJ, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiAdjacencyKeyT tupleKey;
    ret = NetCoOutBuildAdjTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberAdjacencyUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberAdjacencyNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblAdjDel(NetCo *co, NetTblAdjacencyKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblAdjacencyKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblAdjacencyKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_ADJ, key);
    BifrostCncoiAdjacencyKeyT tupleKey;
    uint32_t ret = NetCoOutBuildAdjTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberAdjacencyDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}

uint32_t NetCoTblRankAddUpd(NetCo *co, NetTblGlobalRank *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "packNum %u, rankNum %u, dataLen %u\n", VOS_NTOHS(kv->val.packetNum),
        VOS_NTOHS(kv->val.rankNum), dataLen);
    if ((sizeof(NetTblGlobalRank) + (VOS_NTOHS(kv->val.rankNum) * sizeof(NetTblGlobalRankInfo))) > dataLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblGlobalRankN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblGlobalRankGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblGlobalRankVal) + sizeof(NetTblGlobalRankInfo) * kv->val.rankNum;
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_RANK, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiRankKeyT tupleKey;
    ret = NetCoOutBuildRankTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRankUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRankNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblRankDel(NetCo *co, NetTblGlobalRankKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblGlobalRankKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblGlobalRankKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_RANK, key);
    BifrostCncoiRankKeyT tupleKey;
    uint32_t ret = NetCoOutBuildRankTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberRankDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}

uint32_t NetCoTblRankDistAddUpd(NetCo *co, NetTblRankDistributeInfo *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "dataLen %u\n", dataLen);
    if (sizeof(NetTblRankDistributeInfo) > dataLen) {
        BKF_LOG_ERROR(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblRankDistributeInfoN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblRankDistributeInfoGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblRankDistributeInfoVal);
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_RANK_DIST, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiRankDistributeKeyT tupleKey;
    ret = NetCoOutBuildRankDistTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRankDistributeUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRankDistributeNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblRankDistDel(NetCo *co, NetTblRankDistributeInfoKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblRankDistributeInfoKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblRankDistributeInfoKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_RANK_DIST, key);
    BifrostCncoiRankDistributeKeyT tupleKey;
    uint32_t ret = NetCoOutBuildRankDistTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberRankDistributeDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}

uint32_t NetCoTblRootRankAddUpd(NetCo *co, NetTblRootRank *kv, uint32_t dataLen)
{
    if ((co == VOS_NULL) || (kv == VOS_NULL)) {
        return BKF_ERR;
    }
    BKF_LOG_DEBUG(BKF_LOG_HND, "rootRankNum %u, dataLen %u\n", VOS_NTOHS(kv->val.rootRankNum), dataLen);
    if ((sizeof(NetTblRootRank) + (VOS_NTOHS(kv->val.rootRankNum) * sizeof(uint16_t))) > dataLen) {
        BKF_LOG_DEBUG(BKF_LOG_HND, "input data error\n");
        return BKF_ERR;
    }
    (void)NetTblRootRankN2H(kv, kv);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblRootRankGetStr(kv, buf, sizeof(buf)), getpid());

    uint16_t valLen = sizeof(NetTblRootRankVal) + sizeof(uint16_t) * kv->val.rootRankNum;
    uint32_t ret = NetCoOutAddUpdTbl(co, NET_TBL_TYPE_ROOT_RANK, &kv->key, &kv->val, valLen);
    if (ret != BKF_OK) {
        return ret;
    }
    BifrostCncoiRootRankKeyT tupleKey;
    ret = NetCoOutBuildRootRankTupleKey(&kv->key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRootRankUpdate(co->out->puber, g_NetCoDftSliceKey, &tupleKey, VOS_NULL);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }
    ret = BifrostCncoiPuberRootRankNotifyTableComplete(co->out->puber);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return ret;
    }

    return BKF_OK;
}

void NetCoTblRootRankDel(NetCo *co, NetTblRootRankKey *key)
{
    if ((co == VOS_NULL) || (key == VOS_NULL)) {
        return;
    }
    (void)NetTblRootRankKeyN2H(key, key);
    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s, pid(%d)\n", NetTblRootRankKeyGetStr(key, buf, sizeof(buf)), getpid());

    NetCoOutDelTbl(co, NET_TBL_TYPE_ROOT_RANK, key);
    BifrostCncoiRootRankKeyT tupleKey;
    uint32_t ret = NetCoOutBuildRootRankTupleKey(key, &tupleKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return;
    }
    BifrostCncoiPuberRootRankDelete(co->out->puber, g_NetCoDftSliceKey, &tupleKey);
}
#endif

#if BKF_BLOCK("私有函数定义")
STATIC uint32_t NetCoOutAddUpdTbl(NetCo *co, uint16_t tblTypeId, void *key, void *val, uint16_t valLen)
{
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, tblTypeId);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblTypeId(%u), tblType(%#x)\n", tblTypeId, BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, key);
    if (tbl == VOS_NULL) {
        tbl = NetCoOutAddTbl(co, tblType, key, val, valLen);
        if (tbl == VOS_NULL) {
            BKF_LOG_ERROR(BKF_LOG_HND, "tblTypeId(%u), tbl(%#x)\n", tblTypeId, BKF_MASK_ADDR(tbl));
            return BKF_ERR;
        }

        return BKF_OK;
    } else {
        uint32_t ret = NetCoOutUpdTbl(co, tbl, val, valLen);
        if (ret != BKF_OK) {
            BKF_LOG_ERROR(BKF_LOG_HND, "tblTypeId(%u), ret(%u)\n", tblTypeId, ret);
        }

        return ret;
    }
}

STATIC void NetCoOutDelTbl(NetCo *co, uint16_t tblTypeId, void *key)
{
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, tblTypeId);
    if (tblType == VOS_NULL) {
        return;
    }
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, key);
    if (tbl == VOS_NULL) {
        return;
    }
    NetCoOutDelOneTbl(co, tbl);
}

STATIC uint32_t NetCoOutBuildCommInfoTupleKey(NetTblCommInfoKey *from, BifrostCncoiComminfoKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiComminfoKeyT), 0, sizeof(BifrostCncoiComminfoKeyT));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->packetId = from->packetId;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildCommInfoTblKey(BifrostCncoiComminfoKeyT *from, NetTblCommInfoKey *to)
{
    (void)memset_s(to, sizeof(NetTblCommInfoKey), 0, sizeof(NetTblCommInfoKey));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->packetId = from->packetId;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitCommInfo(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_COMM_INFO,
        .name = "NET_TBL_TYPE_COMM_INFO",
        .keyLen = sizeof(NetTblCommInfoKey),
        .keyCmp = (F_BKF_CMP)NetTblCommInfoKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblCommInfoKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblCommInfoValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberComminfoVTbl dcVTbl = {.tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_COMMINFO_UPDATE_DATA)NetCoOutOnCommInfoTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberComminfoReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberComminfoCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnCommInfoTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiComminfoKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_COMM_INFO);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblCommInfoKey tblKey;
    uint32_t ret = NetCoOutBuildCommInfoTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblCommInfoKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnCommInfoTupleUpdCode(NetCo *co, SimpoBuilderT *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiComminfoKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnCommInfoTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblCommInfoVal *val = (NetTblCommInfoVal *)tbl->val;
    uint32_t i;
    (void)BifrostCncoiComminfoUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiComminfoUpdateKeyCreate(builder, tupleKey);

    (void)BifrostCncoiComminfoUpdateEntryStart(builder);
    (void)BifrostCncoiComminfoEntryPacketNumAdd(builder, val->packetNum);
    (void)BifrostCncoiComminfoEntryResvAdd(builder, val->resv);
    if (val->packetNum == 0) {
        (void)BifrostCncoiComminfoEntrySendRankInfoIgnored(builder);
    } else {
        (void)BifrostCncoiComminfoEntrySendRankInfoStart(builder);
        for (i = 0; i < val->packetNum; i++) {
            BifrostCncoiSendRankInfoT input = {0};
            input.sendRankInfo = val->sendRankInfo[i];
            (void)BifrostCncoiComminfoEntrySendRankInfoPush(builder, &input);
        }
        (void)BifrostCncoiComminfoEntrySendRankInfoEnd(builder);
    }
    BifrostCncoiCommMd5sumT md5Value;
    (void)memcpy_s(&md5Value, sizeof(BifrostCncoiCommMd5sumT), val->commMd5Sum, sizeof(val->commMd5Sum));
    (void)BifrostCncoiComminfoEntryMd5sumCreate(builder, &md5Value);
    (void)BifrostCncoiComminfoEntryRankTotalNumAdd(builder, val->rankTotalNum);
    (void)BifrostCncoiComminfoEntryResv2Add(builder, 0);

    if (val->rankNum == 0) {
        (void)BifrostCncoiComminfoEntryRankInfoIgnored(builder);
    } else {
        (void)BifrostCncoiComminfoEntryRankInfoStart(builder);
        for (i = 0; i < val->rankNum; i++) {
            BifrostCncoiCommInfoRankInfoT input = {0};
            input.deviceIp = val->rankInfo[i].deviceIp;
            input.serverIp = val->rankInfo[i].serverIp;
            input.podId = val->rankInfo[i].podId;
            (void)BifrostCncoiComminfoEntryRankInfoPush(builder, &input);
        }
        (void)BifrostCncoiComminfoEntryRankInfoEnd(builder);
    }

    (void)BifrostCncoiComminfoUpdateEntryEnd(builder);
    return BifrostCncoiComminfoUpdateEndAsRoot(builder);
}

STATIC uint32_t NetCoOutBuildOperTupleKey(NetTblOperatorKey *from, BifrostCncoiOperatorKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiOperatorKeyT), 0, sizeof(BifrostCncoiOperatorKeyT));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->operator= from->op;
    to->algorithm = from->algorithm;
    to->rootRank = from->rootRank;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildOperTblKey(BifrostCncoiOperatorKeyT *from, NetTblOperatorKey *to)
{
    (void)memset_s(to, sizeof(NetTblOperatorKey), 0, sizeof(NetTblOperatorKey));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->op = from->operator;
    to->algorithm = from->algorithm;
    to->rootRank = from->rootRank;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitOper(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_OPER,
        .name = "NET_TBL_TYPE_OPER",
        .keyLen = sizeof(NetTblOperatorKey),
        .keyCmp = (F_BKF_CMP)NetTblOperatorKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblOperatorKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblOperatorValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberOperatorVTbl dcVTbl = {.tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_OPERATOR_UPDATE_DATA)NetCoOutOnOperTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberOperatorReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberOperatorCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnOperTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiOperatorKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_OPER);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblOperatorKey tblKey;
    uint32_t ret = NetCoOutBuildOperTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblOperatorKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnOperTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiOperatorKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnOperTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblOperatorVal *val = (NetTblOperatorVal *)tbl->val;
    (void)BifrostCncoiOperatorUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiOperatorUpdateKeyCreate(builder, tupleKey);
    (void)BifrostCncoiOperatorUpdateEntryStart(builder);
    (void)BifrostCncoiOperatorEntryTrafficCntAdd(builder, val->trafficCnt);
    (void)BifrostCncoiOperatorEntryL4SPortIdAdd(builder, val->l4SPortId);
    (void)BifrostCncoiOperatorEntryMaskLenAdd(builder, val->maskLen);
    (void)BifrostCncoiOperatorEntryResvAdd(builder, 0);
    (void)BifrostCncoiOperatorUpdateEntryEnd(builder);
    return BifrostCncoiOperatorUpdateEndAsRoot(builder);
}

STATIC uint32_t NetCoOutBuildAdjTupleKey(NetTblAdjacencyKey *from, BifrostCncoiAdjacencyKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiAdjacencyKeyT), 0, sizeof(BifrostCncoiAdjacencyKeyT));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    err = memcpy_s(to->commMd5sum, sizeof(to->commMd5sum), from->commMd5Sum, sizeof(from->commMd5Sum));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->srcLocalRankId = from->srcLocalRankId;
    to->operator= from->op;
    to->algorithm = from->algorithm;
    to->rootRank = from->rootRank;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildAdjTblKey(BifrostCncoiAdjacencyKeyT *from, NetTblAdjacencyKey *to)
{
    (void)memset_s(to, sizeof(NetTblAdjacencyKey), 0, sizeof(NetTblAdjacencyKey));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    err = memcpy_s(to->commMd5Sum, sizeof(to->commMd5Sum), from->commMd5sum, sizeof(from->commMd5sum));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->srcLocalRankId = from->srcLocalRankId;
    to->op = from->operator;
    to->algorithm = from->algorithm;
    to->rootRank = from->rootRank;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitAdj(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_ADJ,
        .name = "NET_TBL_TYPE_ADJ",
        .keyLen = sizeof(NetTblAdjacencyKey),
        .keyCmp = (F_BKF_CMP)NetTblAdjacencyKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblAdjacencyKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblAdjacencyValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberAdjacencyVTbl dcVTbl = {.tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_ADJACENCY_UPDATE_DATA)NetCoOutOnAdjTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberAdjacencyReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberAdjacencyCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnAdjTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiAdjacencyKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_ADJ);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblAdjacencyKey tblKey;
    uint32_t ret = NetCoOutBuildAdjTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblAdjacencyKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnAdjTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiAdjacencyKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnAdjTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblAdjacencyVal *val = (NetTblAdjacencyVal *)tbl->val;
    uint32_t i;
    (void)BifrostCncoiAdjacencyUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiAdjacencyUpdateKeyCreate(builder, tupleKey);
    (void)BifrostCncoiAdjacencyUpdateEntryStart(builder);
    if (val->dstRankNum == 0) {
        (void)BifrostCncoiAdjacencyEntryAdjInfoIgnored(builder);
    } else {
        (void)BifrostCncoiAdjacencyEntryAdjInfoStart(builder);
        for (i = 0; i < val->dstRankNum; i++) {
            BifrostCncoiAdjInfoT input = {0};
            input.dstLocalRankId = val->adjinfo[i].dstLocalRankId;
            input.phaseId = val->adjinfo[i].phaseId;
            (void)BifrostCncoiAdjacencyEntryAdjInfoPush(builder, &input);
        }

        (void)BifrostCncoiAdjacencyEntryAdjInfoEnd(builder);
    }
    (void)BifrostCncoiAdjacencyUpdateEntryEnd(builder);
    return BifrostCncoiAdjacencyUpdateEndAsRoot(builder);
}

STATIC uint32_t NetCoOutBuildRankTupleKey(NetTblGlobalRankKey *from, BifrostCncoiRankKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiRankKeyT), 0, sizeof(BifrostCncoiRankKeyT));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->packetId = from->packetId;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildRankTblKey(BifrostCncoiRankKeyT *from, NetTblGlobalRankKey *to)
{
    (void)memset_s(to, sizeof(NetTblGlobalRankKey), 0, sizeof(NetTblGlobalRankKey));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->packetId = from->packetId;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitRank(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_RANK,
        .name = "NET_TBL_TYPE_RANK",
        .keyLen = sizeof(NetTblGlobalRankKey),
        .keyCmp = (F_BKF_CMP)NetTblGlobalRankKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblGlobalRankKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblGlobalRankValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberRankVTbl dcVTbl = {.tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_RANK_UPDATE_DATA)NetCoOutOnRankTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberRankReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberRankCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnRankTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiRankKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_RANK);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblGlobalRankKey tblKey;
    uint32_t ret = NetCoOutBuildRankTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblGlobalRankKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnRankTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnRankTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblGlobalRankVal *val = (NetTblGlobalRankVal *)tbl->val;
    uint32_t i;
    (void)BifrostCncoiRankUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiRankUpdateKeyCreate(builder, tupleKey);
    (void)BifrostCncoiRankUpdateEntryStart(builder);

    (void)BifrostCncoiRankEntryPacketNumAdd(builder, val->packetNum);
    (void)BifrostCncoiRankEntryResvAdd(builder, 0);
    if (val->packetNum == 0) {
        (void)BifrostCncoiRankEntrySendRankInfoIgnored(builder);
    } else {
        (void)BifrostCncoiRankEntrySendRankInfoStart(builder);

        for (i = 0; i < val->packetNum; i++) {
            BifrostCncoiSendRankInfoT input = {0};
            input.sendRankInfo = val->sendRankInfo[i];
            (void)BifrostCncoiRankEntrySendRankInfoPush(builder, &input);
        }
        (void)BifrostCncoiRankEntrySendRankInfoEnd(builder);
    }
    BifrostCncoiCommMd5sumT md5Value;
    (void)memcpy_s(&md5Value, sizeof(BifrostCncoiCommMd5sumT), val->commMd5Sum, sizeof(val->commMd5Sum));
    (void)BifrostCncoiRankEntryMd5sumCreate(builder, &md5Value);
    (void)BifrostCncoiRankEntryRankTotalNumAdd(builder, val->rankTotalNum);

    if (val->rankNum == 0) {
        (void)BifrostCncoiRankEntryRankInfoIgnored(builder);
    } else {
        (void)BifrostCncoiRankEntryRankInfoStart(builder);
        for (i = 0; i < val->rankNum; i++) {
            BifrostCncoiRankInfoT input = {0};
            input.deviceIp = val->rankInfo[i].deviceIp;
            input.serverIp = val->rankInfo[i].serverIp;
            (void)BifrostCncoiRankEntryRankInfoPush(builder, &input);
        }

        (void)BifrostCncoiRankEntryRankInfoEnd(builder);
    }

    (void)BifrostCncoiRankUpdateEntryEnd(builder);
    return BifrostCncoiRankUpdateEndAsRoot(builder);
}

STATIC uint32_t NetCoOutBuildRankDistTupleKey(NetTblRankDistributeInfoKey *from, BifrostCncoiRankDistributeKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiRankDistributeKeyT), 0, sizeof(BifrostCncoiRankDistributeKeyT));
    to->taskId = from->taskId;
    to->npuIp = from->npuIp;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildRankDistTblKey(BifrostCncoiRankDistributeKeyT *from, NetTblRankDistributeInfoKey *to)
{
    (void)memset_s(to, sizeof(NetTblRankDistributeInfoKey), 0, sizeof(NetTblRankDistributeInfoKey));
    to->taskId = from->taskId;
    to->npuIp = from->npuIp;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitRankDist(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_RANK_DIST,
        .name = "NET_TBL_TYPE_RANK_DIST",
        .keyLen = sizeof(NetTblRankDistributeInfoKey),
        .keyCmp = (F_BKF_CMP)NetTblRankDistributeInfoKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblRankDistributeInfoKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblRankDistributeInfoValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberRankDistributeVTbl dcVTbl = {
        .tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_RANK_DISTRIBUTE_UPDATE_DATA)NetCoOutOnRankDistTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberRankDistributeReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberRankDistributeCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnRankDistTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiRankDistributeKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_RANK_DIST);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblRankDistributeInfoKey tblKey;
    uint32_t ret = NetCoOutBuildRankDistTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblRankDistributeInfoKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnRankDistTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRankDistributeKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnRankDistTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblRankDistributeInfoVal *val = (NetTblRankDistributeInfoVal *)tbl->val;
    (void)BifrostCncoiRankDistributeUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiRankDistributeUpdateKeyCreate(builder, tupleKey);
    (void)BifrostCncoiRankDistributeUpdateEntryStart(builder);

    (void)BifrostCncoiRankDistributeEntryServerIpAdd(builder, val->serverIp);
    (void)BifrostCncoiRankDistributeEntryNodeIdAdd(builder, val->nodeId);
    (void)BifrostCncoiRankDistributeEntryLocalRankNumAdd(builder, val->localRankNum);
    (void)BifrostCncoiRankDistributeEntryPad1Add(builder, 0);
    (void)BifrostCncoiRankDistributeEntryPad2Add(builder, 0);
    (void)BifrostCncoiRankDistributeEntryRankTotalNumAdd(builder, val->rankTotalNum);

    (void)BifrostCncoiRankDistributeUpdateEntryEnd(builder);
    return BifrostCncoiRankDistributeUpdateEndAsRoot(builder);
}

STATIC uint32_t NetCoOutBuildRootRankTupleKey(NetTblRootRankKey *from, BifrostCncoiRootRankKeyT *to)
{
    (void)memset_s(to, sizeof(BifrostCncoiRootRankKeyT), 0, sizeof(BifrostCncoiRootRankKeyT));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->operator= from->op;
    to->algorithm = from->algorithm;
    return BKF_OK;
}

STATIC uint32_t NetCoOutBuildRootRankTblKey(BifrostCncoiRootRankKeyT *from, NetTblRootRankKey *to)
{
    (void)memset_s(to, sizeof(NetTblRootRankKey), 0, sizeof(NetTblRootRankKey));
    to->taskId = from->taskId;
    errno_t err = memcpy_s(to->commDesc, sizeof(to->commDesc), from->commDesc, sizeof(from->commDesc));
    if (err != EOK) {
        return BKF_ERR;
    }
    to->commInitTime = from->commInitTime;
    to->op = from->operator;
    to->algorithm = from->algorithm;
    return BKF_OK;
}

STATIC uint32_t NetCoOutInitRootRank(NetCo *co)
{
    NetCoOutTblTypeVTbl vTbl = {.typeId = NET_TBL_TYPE_ROOT_RANK,
        .name = "NET_TBL_TYPE_ROOT_RANK",
        .keyLen = sizeof(NetTblRootRankKey),
        .keyCmp = (F_BKF_CMP)NetTblRootRankKeyCmp,
        .keyGetStr = (F_BKF_GET_STR)NetTblRootRankKeyGetStr,
        .valGetStr = (F_BKF_GET_STR)NetTblRootRankValGetStr};
    NetCoOutTblType *tblType = NetCoOutAddTblType(co, &vTbl);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return BKF_ERR;
    }

    BifrostCncoiPuberRootRankVTbl dcVTbl = {.tupleValLen = 0, .tupleCntMax = NET_CO_OUT_TBL_CNT_MAX, .cookie = co};
    dcVTbl.onFillUpdateData = (F_BIFROST_CNCOI_PUBER_ON_FILL_ROOT_RANK_UPDATE_DATA)NetCoOutOnRootRankTupleUpdCode;
    uint32_t ret = BifrostCncoiPuberRootRankReg(co->out->puber, &dcVTbl);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }
    ret = BifrostCncoiPuberRootRankCreateTable(co->out->puber, g_NetCoDftSliceKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return BKF_ERR;
    }

    return BKF_OK;
}

STATIC NetCoOutTbl *NetCoOutOnRootRankTupleUpdCodeChkAndGetTbl(NetCo *co, SimpoBuilderT *builder,
    BifrostCncoiRootRankKeyT *tupleKey, void *codeBuf, int32_t bufLen)
{
    (void)builder;
    BOOL argIsInvalid = (co == VOS_NULL) || (tupleKey == VOS_NULL) || (codeBuf == VOS_NULL) || (bufLen <= 0);
    if (argIsInvalid) {
        return VOS_NULL;
    }
    NetCoOutTblType *tblType = NetCoOutFindTblType(co, NET_TBL_TYPE_ROOT_RANK);
    if (tblType == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tblType(%#x)\n", BKF_MASK_ADDR(tblType));
        return VOS_NULL;
    }
    NetTblRootRankKey tblKey;
    uint32_t ret = NetCoOutBuildRootRankTblKey(tupleKey, &tblKey);
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u)\n", ret);
        return VOS_NULL;
    }

    uint8_t buf[BKF_1K / 2];
    BKF_LOG_DEBUG(BKF_LOG_HND, "%s\n", NetTblRootRankKeyGetStr(&tblKey, buf, sizeof(buf)));
    NetCoOutTbl *tbl = NetCoOutFindTbl(co, tblType, &tblKey);
    if (tbl == VOS_NULL) {
        BKF_LOG_ERROR(BKF_LOG_HND, "tbl(%#x)\n", BKF_MASK_ADDR(tbl));
        return VOS_NULL;
    }

    return tbl;
}
STATIC int32_t NetCoOutOnRootRankTupleUpdCode(NetCo *co, void *builder, BifrostCncoiSliceKeyT *sliceKey,
    BifrostCncoiRootRankKeyT *tupleKey, void *notUse, void *codeBuf, int32_t bufLen)
{
    (void)sliceKey;
    (void)notUse;
    NetCoOutTbl *tbl = NetCoOutOnRootRankTupleUpdCodeChkAndGetTbl(co, builder, tupleKey, codeBuf, bufLen);
    if (tbl == VOS_NULL) {
        return 0;
    }

    NetTblRootRankVal *val = (NetTblRootRankVal *)tbl->val;
    (void)BifrostCncoiRootRankUpdateStartAsRoot(builder, (uint8_t *)codeBuf, (uint32_t)bufLen);
    (void)BifrostCncoiRootRankUpdateKeyCreate(builder, tupleKey);
    (void)BifrostCncoiRootRankUpdateEntryStart(builder);
    if (val->rootRankNum == 0) {
        (void)BifrostCncoiRootRankEntryRootRankInfoIgnored(builder);
    } else {
        (void)BifrostCncoiRootRankEntryRootRankInfoStart(builder);
        uint32_t i;
        for (i = 0; i < val->rootRankNum; i++) {
            BifrostCncoiRootRankInfoT input = {0};
            input.rootRankInfo = val->rootRankInfo[i];
            (void)BifrostCncoiRootRankEntryRootRankInfoPush(builder, &input);
        }
        (void)BifrostCncoiRootRankEntryRootRankInfoEnd(builder);
    }
    (void)BifrostCncoiRootRankUpdateEntryEnd(builder);
    return BifrostCncoiRootRankUpdateEndAsRoot(builder);
}

#endif
#ifdef __cplusplus
}
#endif
