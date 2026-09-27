/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_RANK_DIST_H
#define NET_VO_TBL_RANK_DIST_H

#include "bkf_comm.h"
#include "net_vo_tbl_comm_info.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

typedef struct TagNetTblRankDistributeInfoKey {
    uint64_t taskId;
    uint32_t npuIp;
} NetTblRankDistributeInfoKey;

typedef struct TagNetTblRankDistributeInfoVal {
    uint32_t serverIp;
    uint32_t nodeId;
    uint8_t localRankNum;
    uint8_t resv[3];
    uint32_t rankTotalNum;
} NetTblRankDistributeInfoVal;

typedef struct TagNetTblRankDistributeInfo {
    NetTblRankDistributeInfoKey key;
    NetTblRankDistributeInfoVal val;
} NetTblRankDistributeInfo;

#pragma pack()

/**
 * @brief rank分布式信息表key比较函数
 *
 * @param[in] key1Input key1
 * @param[in] key2InDs key2
 * @return 比较结果
 *   @retval >0 key1 > key2
 *   @retval =0 key1 = key2
 *   @retval <0 key1 < key2
 */
int32_t NetTblRankDistributeInfoKeyCmp(NetTblRankDistributeInfoKey *key1Input, NetTblRankDistributeInfoKey *key2InDs);

/**
 * @brief rank分布式信息表key主机序转换到网络序
 *
 * @param[in] keyH key，主机序
 * @param[in] keyN key，网络序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRankDistributeInfoKeyH2N(NetTblRankDistributeInfoKey *keyH, NetTblRankDistributeInfoKey *keyN);

/**
 * @brief rank分布式信息表key网络序转换到主机序
 *
 * @param[in] keyN key，网络序
 * @param[in] keyH key，主机序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRankDistributeInfoKeyN2H(NetTblRankDistributeInfoKey *keyN, NetTblRankDistributeInfoKey *keyH);

/**
 * @brief rank分布式信息表key转换到字串，用于dfx
 *
 * @param[in] key 表键值
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRankDistributeInfoKeyGetStr(NetTblRankDistributeInfoKey *key, uint8_t *buf, int32_t bufLen);

/**
 * @brief rank分布式信息表主机序转换到网络序
 *
 * @param[in] kvH 表，主机序
 * @param[in] kvN 表，网络序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRankDistributeInfoH2N(NetTblRankDistributeInfo *kvH, NetTblRankDistributeInfo *kvN);

/**
 * @brief rank分布式信息表网络序转换到主机序
 *
 * @param[in] kvN 表，网络序
 * @param[in] kvH 表，主机序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRankDistributeInfoN2H(NetTblRankDistributeInfo *kvN, NetTblRankDistributeInfo *kvH);

/**
 * @brief rank分布式信息表转换到字串，用于dfx
 *
 * @param[in] kv 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRankDistributeInfoGetStr(NetTblRankDistributeInfo *kv, uint8_t *buf, int32_t bufLen);

/**
 * @brief rank分布式信息表val转换到字串，用于dfx
 *
 * @param[in] rank分布式信息 val 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRankDistributeInfoValGetStr(NetTblRankDistributeInfoVal *val, uint8_t *buf, int32_t bufLen);

#ifdef __cplusplus
}
#endif

#endif
