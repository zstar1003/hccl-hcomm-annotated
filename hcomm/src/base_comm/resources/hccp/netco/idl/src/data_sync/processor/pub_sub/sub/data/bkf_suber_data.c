/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "bkf_suber_data.h"
#include "bkf_suber_table_type.h"
#include "bkf_url.h"
#include "bkf_dl.h"
#include "v_avll.h"
#include "securec.h"

#ifdef __cplusplus
extern "C" {
#endif

#if BKF_BLOCK("私有宏结构函数")
#pragma pack(4)
struct tagBkfSuberDataMng {
    BkfSuberEnv *env;
    AVLL_TREE instSet;
    uint32_t instCnt;
    AVLL_TREE sliceSet;
    uint32_t sliceCnt;
    AVLL_TREE appSubSet;
    uint32_t appSubCnt;
    void *cookie;
    F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER ntfPuberUrlAdd;
    F_BKF_SUBER_DATA_NTF_PUBER_URL_OPER ntfPuberUrlDel;
    F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER ntfSliceInstAdd;
    F_BKF_SUBER_DATA_NTF_SLICE_INST_OPER ntfSliceInstDel;
    F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER ntfAppSubAdd;
    F_BKF_SUBER_DATA_NTF_SLICE_APP_SUB_OPER ntfAppSubDel;
    BkfSuberTableTypeMng *tableTypeMng;
};
STATIC int32_t BkfSuberDataSubKeyCmp(BkfSuberAppSubKey *key1Input, BkfSuberAppSubKey *key2InDs);
STATIC BkfSuberInst *BkfSuberDataAddTmpInst(BkfSuberDataMng *dataMng, uint64_t instId);
STATIC BkfSuberSlice *BkfSuberDataAddTmpSlice(BkfSuberDataMng *dataMng, void *sliceKey);
void BkfSuberDataDelSliceInst(BkfSuberSlice *slice, BkfSuberSliceInst *sliceInst);
BkfSuberSliceInst *BkfSuberDataFindSliceInst(BkfSuberSlice *slice, uint64_t instId);
BkfSuberSliceInst *BkfSuberDataAddSliceInst(BkfSuberDataMng *dataMng, BkfSuberSlice *slice, uint64_t instId);

#pragma pack()
#endif

#if BKF_BLOCK("公有函数定义")
BkfSuberDataMng *BkfSuberDataMngInit(BkfSuberSubDataMngInitArg *initArg)
{
    uint32_t len = sizeof(BkfSuberDataMng);
    BkfSuberDataMng *dataMng = BKF_MALLOC(initArg->env->memMng, len);
    if (dataMng == VOS_NULL) {
        return VOS_NULL;
    }
    dataMng->env = initArg->env;
    dataMng->cookie = initArg->cookie;
    dataMng->ntfPuberUrlAdd = initArg->ntfPuberUrlAdd;
    dataMng->ntfPuberUrlDel = initArg->ntfPuberUrlDel;
    dataMng->ntfSliceInstAdd = initArg->ntfSliceInstAdd;
    dataMng->ntfSliceInstDel = initArg->ntfSliceInstDel;
    dataMng->ntfAppSubAdd = initArg->ntfAppSubAdd;
    dataMng->ntfAppSubDel = initArg->ntfAppSubDel;
    VOS_AVLL_INIT_TREE(dataMng->instSet, (AVLL_COMPARE)BkfUInt64Cmp, BKF_OFFSET(BkfSuberInst, instId),
        BKF_OFFSET(BkfSuberInst, avlNode));
    VOS_AVLL_INIT_TREE(dataMng->sliceSet, (AVLL_COMPARE)(initArg->env->sliceVTbl.keyCmp),
        BKF_OFFSET(BkfSuberSlice, sliceKey), BKF_OFFSET(BkfSuberSlice, avlNode));
    VOS_AVLL_INIT_TREE(dataMng->appSubSet, (AVLL_COMPARE)BkfSuberDataSubKeyCmp, BKF_OFFSET(BkfSuberAppSub, key),
        BKF_OFFSET(BkfSuberAppSub, avlNode));
    BkfSuberTableTypeMngInitArg tableTypeArg = {.env = initArg->env};
    dataMng->tableTypeMng = BkfSuberTableTypeMngInit(&tableTypeArg);
    if (dataMng->tableTypeMng == VOS_NULL) {
        BKF_FREE(initArg->env->memMng, dataMng);
        return VOS_NULL;
    }
    return dataMng;
}

void BkfSuberDataMngUnInit(BkfSuberDataMng *dataMng)
{
    BkfSuberDataDelAllAppSub(dataMng);
    BkfSuberDataDelAllSlice(dataMng);
    BkfSuberDataDelAllInst(dataMng);
    if (dataMng->tableTypeMng != VOS_NULL) {
        BkfSuberTableTypeMngUnInit(dataMng->tableTypeMng);
    }
    BKF_FREE(dataMng->env->memMng, dataMng);
}

#if BKF_BLOCK("inst")
BkfSuberInst *BkfSuberDataAddInst(BkfSuberDataMng *dataMng, uint64_t instId)
{
    BkfSuberInst *inst = BkfSuberDataFindInst(dataMng, instId);
    if (inst != VOS_NULL) {
        inst->isTemp = VOS_FALSE;
        return inst;
    }

    uint32_t len = sizeof(BkfSuberInst);
    inst = BKF_MALLOC(dataMng->env->memMng, len);
    if (inst == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(inst, len, 0, len);
    inst->instId = instId;
    inst->dataMng = dataMng;
    VOS_AVLL_INIT_NODE(inst->avlNode);
    BKF_DL_INIT(&inst->obSlice);
    BKF_DL_NODE_INIT(&inst->nodeObBySubConn);
    BOOL isInsOk = VOS_AVLL_INSERT(dataMng->instSet, inst->avlNode);
    if (!isInsOk) {
        BKF_FREE(dataMng->env->memMng, inst);
        return VOS_NULL;
    }

    if (dataMng->instCnt < 0xFFFFFFFF) {
        dataMng->instCnt++;
    }
    inst->isTemp = VOS_FALSE;
    return inst;
}

void BkfSuberDataDelInstByKey(BkfSuberDataMng *dataMng, uint64_t instId)
{
    BkfSuberInst *inst = VOS_AVLL_FIND(dataMng->instSet, &instId);
    if (inst == VOS_NULL) {
        return;
    }
    BkfSuberDataDelInst(inst);
}

void BkfSuberDataDelInst(BkfSuberInst *inst)
{
    BkfSuberDataMng *dataMng = inst->dataMng;
    BkfDlNode *temp = VOS_NULL;
    BkfDlNode *tempNext = VOS_NULL;
    BkfSuberSliceInst *sliceInstNode = VOS_NULL;

    for (temp = BKF_DL_GET_FIRST(&inst->obSlice); temp != VOS_NULL; temp = tempNext) {
        tempNext = BKF_DL_GET_NEXT(&inst->obSlice, temp);

        sliceInstNode = BKF_DL_GET_ENTRY(temp, BkfSuberSliceInst, nodeObByInst);
        BkfSuberDataDelSliceInst(sliceInstNode->slice, sliceInstNode);
    }
    // 通知删除地址
    if (BkfUrlIsValid(&inst->puberUrl)) {
        dataMng->ntfPuberUrlDel(dataMng->cookie, inst);
    }
    if (BKF_DL_NODE_IS_IN(&inst->nodeObBySubConn)) {
        // 关联关系应该已经摘除
        BKF_ASSERT(0);
        BKF_DL_REMOVE(&inst->nodeObBySubConn);
    }

    if (VOS_AVLL_IN_TREE(inst->avlNode)) {
        VOS_AVLL_DELETE(dataMng->instSet, inst->avlNode);
    }
    if (dataMng->instCnt > 0) {
        dataMng->instCnt--;
    }
    BKF_FREE(dataMng->env->memMng, inst);
}

void BkfSuberDataDelAllInst(BkfSuberDataMng *dataMng)
{
    BkfSuberInst *inst;
    void *itor;

    for (inst = BkfSuberDataGetFirstInst(dataMng, &itor); inst != VOS_NULL;
         inst = BkfSuberDataGetNextInst(dataMng, &itor)) {
        BkfSuberDataDelInst(inst);
    }
}

BkfSuberInst *BkfSuberDataFindInst(BkfSuberDataMng *dataMng, uint64_t instId)
{
    return VOS_AVLL_FIND(dataMng->instSet, &instId);
}

BkfSuberInst *BkfSuberDataFindNextInst(BkfSuberDataMng *dataMng, uint64_t instId)
{
    return VOS_AVLL_FIND_NEXT(dataMng->instSet, &instId);
}

BkfSuberInst *BkfSuberDataGetFirstInst(BkfSuberDataMng *dataMng, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    BkfSuberInst *inst = VOS_AVLL_FIRST(dataMng->instSet);
    if (inst != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(dataMng->instSet, inst->avlNode);
    }
    return inst;
}

BkfSuberInst *BkfSuberDataGetNextInst(BkfSuberDataMng *dataMng, void **itorInOut)
{
    BkfSuberInst *inst = *(BkfSuberInst **)itorInOut;
    if (inst == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(dataMng->instSet, inst->avlNode);
    }
    return inst;
}

uint32_t BkfSuberDataInstAddPuberUrl(BkfSuberInst *inst, BkfUrl *url)
{
    if (BkfUrlIsValid(&inst->puberUrl)) {
        if (BkfUrlCmp(&inst->puberUrl, url) == 0) {
            return BKF_OK;
        } else {
            inst->dataMng->ntfPuberUrlDel(inst->dataMng->cookie, inst);
        }
    }
    inst->puberUrl = *url;
    inst->dataMng->ntfPuberUrlAdd(inst->dataMng->cookie, inst);
    return BKF_OK;
}

uint32_t BkfSuberDataInstAddPuberUrlEx(BkfSuberInst *inst, BkfUrl *url)
{
    if (BkfUrlIsValid(&inst->puberUrl)) {
        if (BkfUrlCmp(&inst->puberUrl, url) == 0) {
            return BKF_OK;
        } else if (BkfUrlIsValid(&inst->localUrl)) { /* 两个url都有效，修改一个，则拆连接 */
            inst->dataMng->ntfPuberUrlDel(inst->dataMng->cookie, inst);
        }
    }
    inst->puberUrl = *url;

    BOOL ntfyPuberUrlAdd = BkfUrlIsValid(&inst->puberUrl) && BkfUrlIsValid(&inst->localUrl);
    /* V8TLS 需要配置两端的url */
    if (ntfyPuberUrlAdd) {
        inst->dataMng->ntfPuberUrlAdd(inst->dataMng->cookie, inst);
    }
    return BKF_OK;
}

uint32_t BkfSuberDataInstAddLocalUrl(BkfSuberInst *inst, BkfUrl *url)
{
    if (BkfUrlIsValid(&inst->localUrl)) {
        if (BkfUrlCmp(&inst->localUrl, url) == 0) {
            return BKF_OK;
        } else if (BkfUrlIsValid(&inst->puberUrl)) { /* puberurl，localurl都有效，拆除老连接 */
            inst->dataMng->ntfPuberUrlDel(inst->dataMng->cookie, inst);
        }
    }
    inst->localUrl = *url;
    BOOL ntfyPuberUrlAdd = BkfUrlIsValid(&inst->puberUrl) && BkfUrlIsValid(&inst->localUrl);
    /* V8TLS 需要配置两端的url */
    if (ntfyPuberUrlAdd) {
        inst->dataMng->ntfPuberUrlAdd(inst->dataMng->cookie, inst);
    }
    return BKF_OK;
}

void BkfSuberDataInstDelPuberUrl(BkfSuberInst *inst)
{
    if (BkfUrlIsValid(&inst->puberUrl)) {
        inst->dataMng->ntfPuberUrlDel(inst->dataMng->cookie, inst);
    }
    if (BKF_DL_NODE_IS_IN(&inst->nodeObBySubConn)) {
        BKF_ASSERT(0); // 残留
        BKF_DL_REMOVE(&inst->nodeObBySubConn);
    }
    BkfUrlInit(&inst->puberUrl);
}

void BkfSuberDataInstDelLocalUrl(BkfSuberInst *inst)
{
    if (BkfUrlIsValid(&inst->localUrl)) {
        inst->dataMng->ntfPuberUrlDel(inst->dataMng->cookie, inst);
    }
    if (BKF_DL_NODE_IS_IN(&inst->nodeObBySubConn)) {
        BKF_ASSERT(0); // 残留
        BKF_DL_REMOVE(&inst->nodeObBySubConn);
    }
    BkfUrlInit(&inst->localUrl);
}
#endif

#if BKF_BLOCK("slice")
BkfSuberSlice *BkfSuberDataAddSlice(BkfSuberDataMng *dataMng, void *sliceKey)
{
    BkfSuberSlice *slice = BkfSuberDataFindSlice(dataMng, sliceKey);
    if (slice != VOS_NULL) {
        slice->isTemp = VOS_FALSE;
        return slice;
    }

    uint32_t len = sizeof(BkfSuberSlice) + dataMng->env->sliceVTbl.keyLen;
    slice = BKF_MALLOC(dataMng->env->memMng, len);
    if (slice == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(slice, len, 0, len);
    uint32_t sliceKeyLen = dataMng->env->sliceVTbl.keyLen;
    if (memcpy_s(slice->sliceKey, sliceKeyLen, sliceKey, sliceKeyLen) != 0) {
        goto error;
    }
    slice->dataMng = dataMng;
    VOS_AVLL_INIT_NODE(slice->avlNode);
    BKF_DL_INIT(&slice->obAppSub);

    VOS_AVLL_INIT_TREE(slice->sliceInstSet, (AVLL_COMPARE)BkfUInt64Cmp, BKF_OFFSET(BkfSuberSliceInst, instId),
        BKF_OFFSET(BkfSuberSliceInst, avlNode));

    slice->instId = BKF_SUBER_INVALID_INST_ID;
    BOOL isInsOk = VOS_AVLL_INSERT(dataMng->sliceSet, slice->avlNode);
    if (!isInsOk) {
        goto error;
    }

    if (dataMng->sliceCnt < 0xFFFFFFFF) {
        dataMng->sliceCnt++;
    }
    slice->isTemp = VOS_FALSE;
    return slice;

error:
    BKF_FREE(dataMng->env->memMng, slice);
    return VOS_NULL;
}

void BkfSuberDataDelSliceByKey(BkfSuberDataMng *dataMng, void *sliceKey)
{
    BkfSuberSlice *slice = VOS_AVLL_FIND(dataMng->sliceSet, sliceKey);
    if (slice == VOS_NULL) {
        return;
    }
    BkfSuberDataDelSlice(slice);
}

void BkfSuberDataDelSlice(BkfSuberSlice *slice)
{
    BkfSuberDataMng *dataMng = slice->dataMng;
    BkfSuberSliceInst *sliceInst = VOS_NULL;
    void *itor = VOS_NULL;

    for (sliceInst = BkfSuberDataGetFirstSliceInst(slice, &itor); sliceInst != VOS_NULL;
         sliceInst = BkfSuberDataGetNextSliceInst(slice, &itor)) {
        BkfSuberDataDelSliceInst(slice, sliceInst);
    }

    if (VOS_AVLL_IN_TREE(slice->avlNode)) {
        VOS_AVLL_DELETE(dataMng->sliceSet, slice->avlNode);
    } else {
        BKF_ASSERT(0);
    }

    BkfDlNode *node = VOS_NULL;
    BkfDlNode *nextNode = VOS_NULL;
    BkfSuberAppSub *appSub = VOS_NULL;
    for (node = BKF_DL_GET_FIRST(&slice->obAppSub); node != VOS_NULL; node = nextNode) {
        nextNode = BKF_DL_GET_NEXT(&slice->obAppSub, node);
        BKF_DL_REMOVE(node);
        appSub = BKF_DL_GET_ENTRY(node, BkfSuberAppSub, nodeObBySlice);
        BkfSuberDataDelAppSub(appSub);
        slice->appSubCnt--;
    }

    if (dataMng->sliceCnt > 0) {
        dataMng->sliceCnt--;
    }
    BKF_FREE(dataMng->env->memMng, slice);
}

void BkfSuberDataDelSliceSpecInst(BkfSuberSlice *slice, uint64_t instId)
{
    if (slice == NULL) {
        return;
    }
    BkfSuberDataMng *dataMng = slice->dataMng;
    BkfSuberSliceInst *sliceInst = BkfSuberDataFindSliceInst(slice, instId);
    if (sliceInst) {
        BkfSuberDataDelSliceInst(slice, sliceInst);
    }
    if (slice->instId == instId) {
        slice->instId = BKF_SUBER_INVALID_INST_ID;
    }
    if (VOS_AVLL_FIRST(slice->sliceInstSet) == VOS_NULL) {
        BkfDlNode *node = VOS_NULL;
        BkfDlNode *nextNode = VOS_NULL;
        /* 此处不应该还有sub节点 */
        for (node = BKF_DL_GET_FIRST(&slice->obAppSub); node != VOS_NULL; node = nextNode) {
            nextNode = BKF_DL_GET_NEXT(&slice->obAppSub, node);
            BKF_DL_REMOVE(node);
            BkfSuberAppSub *appSub = BKF_DL_GET_ENTRY(node, BkfSuberAppSub, nodeObBySlice);
            BkfSuberDataDelAppSub(appSub);
            slice->appSubCnt--;
        }
        if (VOS_AVLL_IN_TREE(slice->avlNode)) {
            VOS_AVLL_DELETE(dataMng->sliceSet, slice->avlNode);
        }
        if (dataMng->sliceCnt > 0) {
            dataMng->sliceCnt--;
        }
        BKF_FREE(dataMng->env->memMng, slice);
    }
}

void BkfSuberDataDelAllSlice(BkfSuberDataMng *dataMng)
{
    BkfSuberSlice *slice;
    void *itor;

    for (slice = BkfSuberDataGetFirstSlice(dataMng, &itor); slice != VOS_NULL;
         slice = BkfSuberDataGetNextSlice(dataMng, &itor)) {
        BkfSuberDataDelSlice(slice);
    }
}

BkfSuberSlice *BkfSuberDataFindSlice(BkfSuberDataMng *dataMng, void *sliceKey)
{
    return VOS_AVLL_FIND(dataMng->sliceSet, sliceKey);
}

BkfSuberSlice *BkfSuberDataFindNextSlice(BkfSuberDataMng *dataMng, void *sliceKey)
{
    return VOS_AVLL_FIND_NEXT(dataMng->sliceSet, sliceKey);
}

BkfSuberSlice *BkfSuberDataGetFirstSlice(BkfSuberDataMng *dataMng, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    BkfSuberSlice *slice = VOS_AVLL_FIRST(dataMng->sliceSet);
    if (slice != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(dataMng->sliceSet, slice->avlNode);
    }
    return slice;
}

BkfSuberSlice *BkfSuberDataGetNextSlice(BkfSuberDataMng *dataMng, void **itorInOut)
{
    BkfSuberSlice *slice = *(BkfSuberSlice **)itorInOut;
    if (slice == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(dataMng->sliceSet, slice->avlNode);
    }
    return slice;
}

uint32_t BkfSuberDataSliceAddInstNormal(BkfSuberSlice *slice, BkfSuberInst *inst)
{
    if (slice->instId != BKF_SUBER_INVALID_INST_ID) {
        BKF_ASSERT(0); // 不允许切换inst
        slice->dataMng->ntfSliceInstDel(slice->dataMng->cookie, slice, inst->instId);
    }
    slice->instId = inst->instId;
    BkfSuberSliceInst *sliceInstNode = BkfSuberDataAddSliceInst(slice->dataMng, slice, inst->instId);
    if (sliceInstNode == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    slice->dataMng->ntfSliceInstAdd(slice->dataMng->cookie, slice, inst->instId);
    return BKF_OK;
}

/* 增量绑定,不会触发sub订阅，需要用户显式订阅 */
uint32_t BkfSuberDataSliceAddInstEx(BkfSuberSlice *slice, BkfSuberInst *inst)
{
    slice->instId = inst->instId;
    BkfSuberSliceInst *sliceInstNode = BkfSuberDataAddSliceInst(slice->dataMng, slice, inst->instId);
    if (sliceInstNode == VOS_NULL) {
        BKF_ASSERT(0);
        return BKF_ERR;
    }
    return BKF_OK;
}

uint32_t BkfSuberDataSliceAddInst(BkfSuberSlice *slice, uint64_t instId)
{
    if (slice->isTemp == VOS_TRUE) {
        return BKF_ERR;
    }

    BkfSuberInst *inst = BkfSuberDataFindInst(slice->dataMng, instId);
    if (inst == VOS_NULL) {
        inst = BkfSuberDataAddTmpInst(slice->dataMng, instId);
        if (inst == VOS_NULL) {
            return BKF_ERR;
        }
    }

    if (slice->instId == instId) {
        return BKF_OK;
    }
    /* 普通模式只支持每个slice一个inst */
    if (slice->dataMng->env->subMod == BKF_SUBER_MODE_DEFAULT) {
        return BkfSuberDataSliceAddInstNormal(slice, inst);
    }

    return BkfSuberDataSliceAddInstEx(slice, inst);
}

#endif

#if BKF_BLOCK("app sub")

/* 普通sub，slice只有一个inst，如果没绑定inst，则slice的inst是无效值 */
uint32_t BkfSuberDataAddAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId,
    BOOL isVerify)
{
    BkfSuberSlice *slice = BkfSuberDataFindSlice(dataMng, sliceKey);
    if (slice == VOS_NULL) {
        slice = BkfSuberDataAddTmpSlice(dataMng, sliceKey);
        if (slice == VOS_NULL) {
            return BKF_ERR;
        }
    }

    uint64_t subInstId = instId;
    if (instId == 0 || instId == BKF_SUBER_INVALID_INST_ID) {
        subInstId = slice->instId;
    }

    BkfSuberAppSub *appSub = BkfSuberDataFindAppSub(dataMng, sliceKey, tableTypeId, subInstId);
    if (appSub != VOS_NULL) {
        appSub->isVerify = BKF_COND_2BIT_FIELD(isVerify);
        dataMng->ntfAppSubAdd(dataMng->cookie, appSub);
        return BKF_OK;
    }

    uint32_t len = sizeof(BkfSuberAppSub) + dataMng->env->sliceVTbl.keyLen;
    appSub = BKF_MALLOC(dataMng->env->memMng, len);
    if (appSub == VOS_NULL) {
        return BKF_ERR;
    }

    (void)memset_s(appSub, len, 0, len);
    if (memcpy_s(appSub->sliceKey, dataMng->env->sliceVTbl.keyLen, sliceKey, dataMng->env->sliceVTbl.keyLen) != 0) {
        goto error;
    }
    appSub->dataMng = dataMng;
    appSub->key.tableTypeId = tableTypeId;
    appSub->key.sliceKeyCmp = (F_BKF_CMP)dataMng->env->sliceVTbl.keyCmp;
    appSub->key.sliceKey = appSub->sliceKey;
    appSub->key.instId = subInstId;
    appSub->isVerify = BKF_COND_2BIT_FIELD(isVerify);
    VOS_AVLL_INIT_NODE(appSub->avlNode);
    BKF_DL_NODE_INIT(&appSub->nodeObBySlice);
    if (!VOS_AVLL_INSERT(dataMng->appSubSet, appSub->avlNode)) {
        goto error;
    }

    if (dataMng->appSubCnt < 0xFFFFFFFF) {
        dataMng->appSubCnt++;
    }
    BKF_DL_ADD_LAST(&slice->obAppSub, &appSub->nodeObBySlice);
    slice->appSubCnt++;
    dataMng->ntfAppSubAdd(dataMng->cookie, appSub);
    return BKF_OK;
error:
    BKF_FREE(dataMng->env->memMng, appSub);
    return BKF_ERR;
}

void BkfSuberDataDelAppSubByKey(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId,
    BOOL isVerify)
{
    BkfSuberSlice *slice = BkfSuberDataFindSlice(dataMng, sliceKey);
    if (slice == VOS_NULL) {
        return;
    }
    uint64_t subInstId = instId;
    if (instId == 0 || instId == BKF_SUBER_INVALID_INST_ID) {
        subInstId = slice->instId;
    }
    BkfSuberAppSubKey subKey = {.tableTypeId = tableTypeId,
        .instId = subInstId,
        .sliceKey = sliceKey,
        .sliceKeyCmp = dataMng->env->sliceVTbl.keyCmp};
    BkfSuberAppSub *appSub = VOS_AVLL_FIND(dataMng->appSubSet, &subKey);
    if (appSub == VOS_NULL) {
        return;
    }

    /* 1.对账->去对账->发送unsub后切换为batok */
    /* 2.any->去订阅->发送unsub后删除sess */
    /* 3.非对账->去对账->不处理 */
    if (isVerify) {
        if (!(appSub->isVerify)) {
            return;
        }
        dataMng->ntfAppSubDel(dataMng->cookie, appSub);
    } else {
        BkfSuberDataDelAppSub(appSub);
        slice->appSubCnt--;
    }
}

void BkfSuberDataDelAppSub(BkfSuberAppSub *appSub)
{
    /* 设置为非对账，后续会将sess删除，否则带有verify标记的sees会切换为batok */
    appSub->isVerify = VOS_FALSE;
    BkfSuberDataMng *dataMng = appSub->dataMng;
    dataMng->ntfAppSubDel(dataMng->cookie, appSub);

    if (BKF_DL_NODE_IS_IN(&appSub->nodeObBySlice)) {
        BKF_DL_REMOVE(&appSub->nodeObBySlice);
    }

    if (VOS_AVLL_IN_TREE(appSub->avlNode)) {
        VOS_AVLL_DELETE(dataMng->appSubSet, appSub->avlNode);
    } else {
        BKF_ASSERT(0);
    }

    if (dataMng->appSubCnt > 0) {
        dataMng->appSubCnt--;
    }
    BKF_FREE(dataMng->env->memMng, appSub);
}

void BkfSuberDataDelAllAppSub(BkfSuberDataMng *dataMng)
{
    BkfSuberAppSub *appSub;
    void *itor;
    for (appSub = BkfSuberDataGetFirstAppSub(dataMng, &itor); appSub != VOS_NULL;
         appSub = BkfSuberDataGetNextAppSub(dataMng, &itor)) {
        BkfSuberDataDelAppSub(appSub);
    }
}

BkfSuberAppSub *BkfSuberDataFindAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId, uint64_t instId)
{
    BkfSuberAppSubKey subKey = {.tableTypeId = tableTypeId,
        .instId = instId,
        .sliceKey = sliceKey,
        .sliceKeyCmp = dataMng->env->sliceVTbl.keyCmp};
    return VOS_AVLL_FIND(dataMng->appSubSet, &subKey);
}

BkfSuberAppSub *BkfSuberDataFindNextAppSub(BkfSuberDataMng *dataMng, void *sliceKey, uint16_t tableTypeId,
    uint64_t instId)
{
    BkfSuberAppSubKey subKey = {.tableTypeId = tableTypeId,
        .instId = instId,
        .sliceKey = sliceKey,
        .sliceKeyCmp = dataMng->env->sliceVTbl.keyCmp};
    return VOS_AVLL_FIND_NEXT(dataMng->appSubSet, &subKey);
}

BkfSuberAppSub *BkfSuberDataGetFirstAppSub(BkfSuberDataMng *dataMng, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    BkfSuberAppSub *appSub = VOS_AVLL_FIRST(dataMng->appSubSet);
    if (appSub != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(dataMng->appSubSet, appSub->avlNode);
    }
    return appSub;
}

BkfSuberAppSub *BkfSuberDataGetNextAppSub(BkfSuberDataMng *dataMng, void **itorInOut)
{
    BkfSuberAppSub *appSub = *(BkfSuberAppSub **)itorInOut;
    if (appSub == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(dataMng->appSubSet, appSub->avlNode);
    }
    return appSub;
}
#endif
#endif

#if BKF_BLOCK("tableType")
uint32_t BkfSuberDataTableTypeAdd(BkfSuberDataMng *dataMng, uint8_t mode, void *vtbl, void *userData,
    uint16_t userDataLen)
{
    return BkfSuberTableTypeAdd(dataMng->tableTypeMng, mode, vtbl, userData, userDataLen);
}

uint32_t BkfSuberDataTableTypeDel(BkfSuberDataMng *dataMng, uint16_t tabType)
{
    return BkfSuberTableTypeDel(dataMng->tableTypeMng, tabType);
}

BkfSuberTableTypeVTbl *BkfSuberDataTableTypeGetVtbl(BkfSuberDataMng *dataMng, uint16_t tablTypeId)
{
    return BkfSuberTableTypeGetVtbl(dataMng->tableTypeMng, tablTypeId);
}

void *BkfSuberDataTableTypeGetUserData(BkfSuberDataMng *dataMng, uint16_t tablTypeId)
{
    return BkfSuberTableTypeGetUserData(dataMng->tableTypeMng, tablTypeId);
}
#endif

void BkfSuberDataScanAppSubBySlice(BkfSuberSlice *slice, uint64_t instId,
    F_BKF_SUBER_DATA_SCAN_APPSUB_BY_SLICE_PROC proc, void *usrData)
{
    BkfDlNode *node = VOS_NULL;
    BkfDlNode *nextNode = VOS_NULL;
    BkfSuberAppSub *appSub;
    BOOL needContinue = VOS_TRUE;
    for (node = BKF_DL_GET_FIRST(&slice->obAppSub); node != VOS_NULL; node = nextNode) {
        nextNode = BKF_DL_GET_NEXT(&slice->obAppSub, node);
        appSub = BKF_DL_GET_ENTRY(node, BkfSuberAppSub, nodeObBySlice);
        /* 普通模式slice只有一个inst，在未绑定切片时发起订阅，appSub填写的inst是无效值 */
        if (appSub->key.instId == BKF_SUBER_INVALID_INST_ID) {
            if (VOS_AVLL_IN_TREE(appSub->avlNode)) {
                VOS_AVLL_DELETE(slice->dataMng->appSubSet, appSub->avlNode);
            }
            appSub->key.instId = slice->instId;
            (void)VOS_AVLL_INSERT(slice->dataMng->appSubSet, appSub->avlNode);
        }
        if (appSub->key.instId == instId) {
            proc(appSub, usrData, &needContinue);
        }
        if (!needContinue) {
            return;
        }
    }
}

#if BKF_BLOCK("私有函数定义")
STATIC int32_t BkfSuberDataSubKeyCmp(BkfSuberAppSubKey *key1Input, BkfSuberAppSubKey *key2InDs)
{
    if (key1Input->tableTypeId > key2InDs->tableTypeId) {
        return 1;
    } else if (key1Input->tableTypeId < key2InDs->tableTypeId) {
        return -1;
    } else if (key1Input->instId > key2InDs->instId) {
        return 1;
    } else if (key1Input->instId < key2InDs->instId) {
        return -1;
    } else if (key1Input->sliceKeyCmp > key2InDs->sliceKeyCmp) {
        return 1;
    } else if (key1Input->sliceKeyCmp < key2InDs->sliceKeyCmp) {
        return -1;
    }
    return key1Input->sliceKeyCmp(key1Input->sliceKey, key2InDs->sliceKey);
}

STATIC BkfSuberInst *BkfSuberDataAddTmpInst(BkfSuberDataMng *dataMng, uint64_t instId)
{
    BkfSuberInst *inst = BkfSuberDataAddInst(dataMng, instId);
    if (inst == VOS_NULL) {
        return VOS_NULL;
    }
    inst->isTemp = VOS_TRUE;
    return inst;
}

STATIC BkfSuberSlice *BkfSuberDataAddTmpSlice(BkfSuberDataMng *dataMng, void *sliceKey)
{
    BkfSuberSlice *slice = BkfSuberDataAddSlice(dataMng, sliceKey);
    if (slice == VOS_NULL) {
        return VOS_NULL;
    }
    slice->isTemp = VOS_TRUE;
    return slice;
}
#endif

#if BKF_BLOCK("slice inst操作函数")
BkfSuberSliceInst *BkfSuberDataAddSliceInst(BkfSuberDataMng *dataMng, BkfSuberSlice *slice, uint64_t instId)
{
    BkfSuberSliceInst *sliceInst = VOS_AVLL_FIND(slice->sliceInstSet, &instId);
    if (sliceInst != VOS_NULL) {
        return sliceInst;
    }

    BkfSuberInst *inst = BkfSuberDataFindInst(dataMng, instId);
    if (inst == VOS_NULL) {
        return VOS_NULL;
    }

    uint32_t len = sizeof(BkfSuberSliceInst);
    sliceInst = BKF_MALLOC(dataMng->env->memMng, len);
    if (sliceInst == VOS_NULL) {
        BKF_ASSERT(0);
        return VOS_NULL;
    }
    (void)memset_s(sliceInst, len, 0, len);
    sliceInst->slice = slice;
    sliceInst->inst = inst;
    VOS_AVLL_INIT_NODE(sliceInst->avlNode);
    BKF_DL_NODE_INIT(&sliceInst->nodeObByInst);
    sliceInst->instId = instId;

    BOOL isInsOk = VOS_AVLL_INSERT(slice->sliceInstSet, sliceInst->avlNode);
    if (!isInsOk) {
        goto error;
    }
    slice->instCnt++;
    BKF_DL_ADD_LAST(&inst->obSlice, &sliceInst->nodeObByInst);
    inst->sliceCnt++;
    return sliceInst;
error:
    BKF_FREE(dataMng->env->memMng, sliceInst);
    return VOS_NULL;
}

BkfSuberSliceInst *BkfSuberDataFindSliceInst(BkfSuberSlice *slice, uint64_t instId)
{
    BkfSuberSliceInst *sliceInst = VOS_AVLL_FIND(slice->sliceInstSet, &instId);
    return sliceInst;
}

void BkfSuberDataDelSliceInst(BkfSuberSlice *slice, BkfSuberSliceInst *sliceInst)
{
    if (slice == VOS_NULL || sliceInst == VOS_NULL) {
        return;
    }
    BkfSuberDataMng *dataMng = slice->dataMng;

    if (VOS_AVLL_IN_TREE(sliceInst->avlNode)) {
        VOS_AVLL_DELETE(slice->sliceInstSet, sliceInst->avlNode);
    }
    slice->instCnt--;

    BkfSuberAppSub *appSub = VOS_NULL;
    BkfDlNode *node = VOS_NULL;
    BkfDlNode *nextNode = VOS_NULL;

    for (node = BKF_DL_GET_FIRST(&slice->obAppSub); node != VOS_NULL; node = nextNode) {
        nextNode = BKF_DL_GET_NEXT(&slice->obAppSub, node);
        appSub = BKF_DL_GET_ENTRY(node, BkfSuberAppSub, nodeObBySlice);
        if (appSub->key.instId == sliceInst->instId) {
            BKF_DL_REMOVE(node);
            BkfSuberDataDelAppSub(appSub);
            slice->appSubCnt--;
        }
    }
    BKF_DL_REMOVE(&sliceInst->nodeObByInst);
    sliceInst->inst->sliceCnt--;

    dataMng->ntfSliceInstDel(dataMng->cookie, sliceInst->slice, sliceInst->instId);
    BKF_FREE(dataMng->env->memMng, sliceInst);
}

BkfSuberSliceInst *BkfSuberDataGetFirstSliceInst(BkfSuberSlice *slice, void **itorOutOrNull)
{
    if (itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_NULL;
    }
    BkfSuberSliceInst *sliceInst = VOS_AVLL_FIRST(slice->sliceInstSet);
    if (sliceInst != VOS_NULL && itorOutOrNull != VOS_NULL) {
        *itorOutOrNull = VOS_AVLL_NEXT(slice->sliceInstSet, sliceInst->avlNode);
    }
    return sliceInst;
}

BkfSuberSliceInst *BkfSuberDataGetNextSliceInst(BkfSuberSlice *slice, void **itorInOut)
{
    BkfSuberSliceInst *sliceInst = *(BkfSuberSliceInst **)itorInOut;
    if (sliceInst == VOS_NULL) {
        *itorInOut = VOS_NULL;
    } else {
        *itorInOut = VOS_AVLL_NEXT(slice->sliceInstSet, sliceInst->avlNode);
    }
    return sliceInst;
}
#endif

#ifdef __cplusplus
}
#endif
