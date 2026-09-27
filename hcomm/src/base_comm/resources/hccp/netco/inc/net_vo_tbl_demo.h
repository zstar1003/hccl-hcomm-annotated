/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_DEMO_H
#define NET_VO_TBL_DEMO_H

#include "bkf_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
typedef struct TagNetTblDemoKey {
    uint32_t xxx;
} NetTblDemoKey;
typedef struct TagNetTblDemoVal {
    uint32_t yyy;
    int32_t zzzCnt;
    uint32_t zzz[0];
} NetTblDemoVal;
typedef struct TagNetTblDemo {
    NetTblDemoKey key;
    NetTblDemoVal val;
} NetTblDemo;

#pragma pack()

/**
 * @brief demo表key比较函数
 *
 * @param[in] key1Input key1
 * @param[in] key2InDs key2
 * @return 比较结果
 *   @retval >0 key1 > key2
 *   @retval =0 key1 = key2
 *   @retval <0 key1 < key2
 */
int32_t NetTblDemoKeyCmp(NetTblDemoKey *key1Input, NetTblDemoKey *key2InDs);

/**
 * @brief demo表key主机序转换到网络序
 *
 * @param[in] keyH key，主机序
 * @param[in] keyN key，网络序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblDemoKeyH2N(NetTblDemoKey *keyH, NetTblDemoKey *keyN);

/**
 * @brief demo表key网络序转换到主机序
 *
 * @param[in] keyN key，网络序
 * @param[in] keyH key，主机序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblDemoKeyN2H(NetTblDemoKey *keyN, NetTblDemoKey *keyH);

/**
 * @brief demo表key转换到字串，用于dfx
 *
 * @param[in] key 表键值
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblDemoKeyGetStr(NetTblDemoKey *key, uint8_t *buf, int32_t bufLen);

/**
 * @brief demo表主机序转换到网络序
 *
 * @param[in] kvH 表，主机序
 * @param[in] kvN 表，网络序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblDemoH2N(NetTblDemo *kvH, NetTblDemo *kvN);

/**
 * @brief demo表网络序转换到主机序
 *
 * @param[in] kvN 表，网络序
 * @param[in] kvH 表，主机序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblDemoN2H(NetTblDemo *kvN, NetTblDemo *kvH);

/**
 * @brief demo表转换到字串，用于dfx
 *
 * @param[in] kv 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblDemoGetStr(NetTblDemo *kv, uint8_t *buf, int32_t bufLen);

/**
 * @brief demo表val转换到字串，用于dfx
 *
 * @param[in] demo val 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblDemoValGetStr(NetTblDemoVal *val, uint8_t *buf, int32_t bufLen);

#ifdef __cplusplus
}
#endif

#endif
