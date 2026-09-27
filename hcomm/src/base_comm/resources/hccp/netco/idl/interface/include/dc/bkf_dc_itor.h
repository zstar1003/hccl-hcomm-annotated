/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DC_ITOR_H
#define BKF_DC_ITOR_H

#include "bkf_comm.h"
#include "bkf_dc.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief dc库Key迭代器，用于批备
 */
typedef struct tagBkfDcTupleKeyItor BkfDcTupleKeyItor;

/**
 * @brief dc库seq迭代器，用于实备
 */
typedef struct tagBkfDcTupleSeqItor BkfDcTupleSeqItor;

/* add */
/**
 * @brief dc迭代器回调函数
 */
typedef void (*F_BKF_DC_NTF_BY_ITOR)(void *cookieAddItor);

/**
 * @brief dc迭代器虚表
 */
typedef struct tagBkfDcTupleItorVTbl {
    char *name; /**< 迭代器虚表名称 */
    void *cookie;
    F_BKF_DC_NTF_BY_ITOR afterTupleChangeOrNull;   /**< 在回调中不能有删当前表的操作 */
    F_BKF_DC_NTF_BY_ITOR afterTableReleaseOrNull;  /**< 在回调中不能有删当前表的操作 */
    F_BKF_DC_NTF_BY_ITOR afterTableCompleteOrNull; /**< 在回调中不能有删当前表的操作 */
} BkfDcTupleItorVTbl;

/* func */
/**
 * @brief dc库迭代器初始化
 *
 * @param[in] *dc dc库句柄
 * @return 初始化结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcItorInit(BkfDc *dc);

/**
 * @brief dc库迭代器反初始化
 *
 * @param[in] *dc dc库句柄
 * @return none
 */
void BkfDcItorUninit(BkfDc *dc);

/* keyItor */
/**
 * @brief 创建key类型的迭代器
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @return 创建结果
 *   @retval 非空 成功
 *   @retval 空 失败
 */
BkfDcTupleKeyItor *BkfDcNewTupleKeyItor(BkfDc *dc, void *sliceKey, uint16_t tableTypeId);

/**
 * @brief 删除key类型的迭代器
 *
 * @param[in] *dc dc库句柄
 * @param[in] *keyItor 迭代器
 * @return 无
 */
void BkfDcDeleteTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor);

/**
 * @brief 获取key类型迭代器指向的元组信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] *keyItor 迭代器
 * @param[in] info 元组信息
 * @return 获取结果
 *   @retval true 成功
 *   @retval false 失败
 */
BOOL BkfDcGetTupleByKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor, BkfDcTupleInfo *info);

/**
 * @brief 将key类型迭代器往后前进一步
 *
 * @param[in] *dc dc库句柄
 * @param[in] *keyItor 迭代器
 * @return 无
 */
void BkfDcForwordTupleKeyItor(BkfDc *dc, BkfDcTupleKeyItor *keyItor);

/* seqItor */
/**
 * @brief 创建sequence类型的迭代器
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *vTbl 虚表
 * @return 创建结果
 *   @retval 非空 成功
 *   @retval 空 失败
 */
BkfDcTupleSeqItor *BkfDcNewTupleSeqItor(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, BkfDcTupleItorVTbl *vTbl);

/**
 * @brief 删除sequence类型的迭代器
 *
 * @param[in] *dc dc库句柄
 * @param[in] *seqItor 迭代器
 * @return 无
 */
void BkfDcDeleteTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor);

/**
 * @brief 获取sequence类型迭代器指向的元组信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] *seqItor 迭代器
 * @param[in] info 元组信息
 * @return 获取结果
 *   @retval true 成功
 *   @retval false 失败
 */
BOOL BkfDcGetTupleBySeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor, BkfDcTupleInfo *info);

/**
 * @brief 将sequence类型迭代器往后前进一步
 *
 * @param[in] *dc dc库句柄
 * @param[in] *seqItor 迭代器
 * @return 无
 * @note 如果迭代器挪动前指向的元组可以删除（所有迭代器对此元组都已经移动过），并且元组是删除状态，dc内部会将删除元组
 */
void BkfDcForwordTupleSeqItor(BkfDc *dc, BkfDcTupleSeqItor *seqItor);

/**
 * @brief 获取sequence类型迭代器指向的元组信息，与普通流程分离
 *
 * @param[in] *dc dc库句柄
 * @param[in] *seqItor 迭代器
 * @param[in] info 元组信息
 * @return 获取结果
 *   @retval true 成功
 *   @retval false 失败
 */
BOOL BkfDcGetTupleByKeyItorFromDcOrApp(BkfDc *dc, BkfDcTupleKeyItor *keyItor, BkfDcTupleInfo *info);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
