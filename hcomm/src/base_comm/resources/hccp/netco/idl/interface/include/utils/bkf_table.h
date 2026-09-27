/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_TABLE_H
#define BKF_TABLE_H

#include "bkf_comm.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_mem.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)

/**
 * @brief table初始化参数
 */
typedef struct tagBkfTableInitArg {
    BkfMemMng *memMng; /**< 内存库句柄,@see bkf_mem.h,同一app内可复用 */
    uint32_t keyLen;   /**< 业务数据key长度 */
    uint32_t valLen;   /**< 业务数据val长度 */
    F_BKF_CMP keyCmp;  /**< 业务key比较接口 */
} BkfTableInitArg;

/**
 * @brief table句柄，具体实现后续放到到c文件中
 */
typedef struct tagBkfTableMng BkfTableMng;

/**
 * @brief 获取table存放信息
 */
typedef struct tagBkfTableInfo {
    void *key; /**< 业务数据key指针 */
    void *val; /**< 业务数据val指针 */
} BkfTableInfo;

/**
 * @brief table初始化
 *
 * @param[in] *arg 初始化参数
 * @return table句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfTableMng *BkfTableInit(BkfTableInitArg *arg);

/**
 * @brief table删除
 *
 * @param[in] tableMng table句柄
 * @return 无
 */
void BkfTableUnInit(BkfTableMng *tableMng);

/**
 * @brief 删除table中的全部数据
 *
 * @param[in] tableMng table句柄
 * @return 无
 */
void BkfTableDelAll(BkfTableMng *tableMng);

/**
 * @brief 获取业务数据的个数
 *
 * @param[in] tableMng table句柄
 * @return 业务数据个数，范围[0, 4294967295]
 */
uint32_t BkfTableGetCnt(BkfTableMng *tableMng);

/**
 * @brief table中添加业务数据，如果key已经存在，则更新val
 *
 * @param[in] tableMng table句柄
 * @param[in] *key 业务数据key
 * @param[in] *val 业务数据val
 * @param[out] *infoOutOrNull 用户key和val数据内存地址
 * @return 处理结果
 *   @retval BKF_OK 添加成功
 *   @retval 非BKF_OK 添加失败
 */
uint32_t BkfTableAdd(BkfTableMng *tableMng, void *key, void *val, BkfTableInfo *infoOutOrNull);

/**
 * @brief table中删除业务数据
 *
 * @param[in] *key 业务数据key
 * @return 无
 */
void BkfTableDel(BkfTableMng *tableMng, void *key);

/**
 * @brief 查找获取table中的业务数据
 *
 * @param[in] tableMng table句柄
 * @param[in] *key 业务数据key
 * @param[out] *infoOut 业务数据
 * @return 获取结果
 *   @retval 非BKF_OK 获取失败
 *   @retval BKF_OK 获取成功
 */
uint32_t BkfTableFind(BkfTableMng *tableMng, void *key, BkfTableInfo *infoOut);

/**
 * @brief 查找获取table中的下一个业务数据
 *
 * @param[in] tableMng table句柄
 * @param[in] *key 业务数据key
 * @param[out] *infoOut 业务数据
 * @return 获取结果
 *   @retval 非BKF_OK 获取失败
 *   @retval BKF_OK 获取成功
 */
uint32_t BkfTableFindNext(BkfTableMng *tableMng, void *key, BkfTableInfo *infoOut);

/**
 * @brief 获取table中的第一个业务数据
 *
 * @param[in] tableMng table句柄
 * @param[out] *infoOut 业务数据
 * @param[out] *itorOutOrNull 输出遍历迭代器
 * @return 获取结果
 *   @retval 非BKF_OK 获取失败
 *   @retval BKF_OK 获取成功
 */
uint32_t BkfTableGetFirst(BkfTableMng *tableMng, BkfTableInfo *infoOut, void **itorOutOrNull);

/**
 * @brief 获取table中的下一个业务数据
 *
 * @param[in] tableMng table句柄
 * @param[out] *infoOut 业务数据
 * @param[in] **itorInOut 遍历迭代器
 * @return 获取结果
 *   @retval 非BKF_OK 获取失败
 *   @retval BKF_OK 获取成功
 */
uint32_t BkfTableGetNext(BkfTableMng *tableMng, BkfTableInfo *infoOut, void **itorInOut);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif