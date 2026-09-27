/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_H
#define BKF_PUBER_H

#include "bkf_puber_adef.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC visibility push(default)
/**
 * @brief 发布者库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 发布者库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfPuber *BkfPuberInit(BkfPuberInitArg *arg);

/**
 * @brief 发布者库反初始化
 *
 * @param[in] *puber 发布者库句柄
 * @return none
 */
void BkfPuberUninit(BkfPuber *puber);

/**
 * @brief 发布者库注册虚表
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] *vTbl 虚表
 * @param[in] *userData 业务数据
 * @param[in] userDataLen 业务数据长度
 * @return 注册是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfPuberAttachTableTypeEx(BkfPuber *puber, BkfPuberTableTypeVTbl *vTbl, void *userData, uint16_t userDataLen);

/**
 * @brief 发布者库去注册虚表
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] tabTypeId 表id
 * @return none
 */
void BkfPuberDeattachTableType(BkfPuber *puber, uint16_t tabTypeId);

/**
 * @brief 发布者库获取业务数据
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] tableTypeId 数据类型
 * @return 业务数据
 *   @retval 非空 成功
 *   @retval 空 失败
 */
void *BkfPuberGetTableTypeUserData(BkfPuber *puber, uint16_t tableTypeId);

/**
 * @brief 发布者库使能
 *
 * @param[in] *puber 发布者库句柄
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfPuberEnable(BkfPuber *puber);

/**
 * @brief 设置自己的服务实例id
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] instId 自己服务实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfPuberSetSelfSvcInstId(BkfPuber *puber, uint32_t instId);

/**
 * @brief 设置自己的服务实例url
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] *url 自己的服务实例url
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfPuberSetSelfUrl(BkfPuber *puber, BkfUrl *url);

/**
 * @brief 清除自己的服务实例url
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] *url 自己的服务实例url
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
void BkfPuberUnsetSelfUrl(BkfPuber *puber, BkfUrl *url);

/**
 * @brief 控制会话发送数据是否继续调度。
 *
 * @param[in] *puberConn 连接
 * @return 是否继续调度
 *   @retval VOS_TRUE 继续调度
 *   @retval 其他 不调度
 */
BOOL BkfPuberMaySchedSess(void *puberConn);

/**
 * @brief 修改当前puber支持的最大连接数上限
 *
 * @param[in] *puber 发布者库句柄
 * @param[in] connMax 最大连接数
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfPuberSetConnUpLimit(BkfPuber *puber, uint32_t connMax);
#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif

#endif
