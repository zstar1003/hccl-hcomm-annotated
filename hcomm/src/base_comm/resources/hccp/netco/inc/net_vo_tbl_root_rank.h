/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_ROOT_RANK_H
#define NET_VO_TBL_ROOT_RANK_H

#include "bkf_comm.h"
#include "net_vo_tbl_comm_info.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

typedef struct TagNetTblRootRankKey {
    uint64_t taskId;
    char commDesc[NET_COMMDESC_MAXLEN];
    uint64_t commInitTime;
    uint8_t op;
    uint8_t algorithm;
    uint16_t resv;
} NetTblRootRankKey;

typedef struct TagNetTblRootRankVal {
    uint16_t rootRankNum;
    uint16_t resv;
    uint16_t rootRankInfo[0];
} NetTblRootRankVal;

typedef struct TagNetTblRootRank {
    NetTblRootRankKey key;
    NetTblRootRankVal val;
} NetTblRootRank;

#pragma pack()

/**
 * @brief RootRank表key比较函数
 *
 * @param[in] key1Input key1
 * @param[in] key2InDs key2
 * @return 比较结果
 *   @retval >0 key1 > key2
 *   @retval =0 key1 = key2
 *   @retval <0 key1 < key2
 */
int32_t NetTblRootRankKeyCmp(NetTblRootRankKey *key1Input, NetTblRootRankKey *key2InDs);

/**
 * @brief RootRank表key主机序转换到网络序
 *
 * @param[in] keyH key，主机序
 * @param[in] keyN key，网络序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRootRankKeyH2N(NetTblRootRankKey *keyH, NetTblRootRankKey *keyN);

/**
 * @brief RootRank表key网络序转换到主机序
 *
 * @param[in] keyN key，网络序
 * @param[in] keyH key，主机序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRootRankKeyN2H(NetTblRootRankKey *keyN, NetTblRootRankKey *keyH);

/**
 * @brief RootRank表key转换到字串，用于dfx
 *
 * @param[in] key 表键值
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRootRankKeyGetStr(NetTblRootRankKey *key, uint8_t *buf, int32_t bufLen);

/**
 * @brief RootRank表主机序转换到网络序
 *
 * @param[in] kvH 表，主机序
 * @param[in] kvN 表，网络序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRootRankH2N(NetTblRootRank *kvH, NetTblRootRank *kvN);

/**
 * @brief RootRank表网络序转换到主机序
 *
 * @param[in] kvN 表，网络序
 * @param[in] kvH 表，主机序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblRootRankN2H(NetTblRootRank *kvN, NetTblRootRank *kvH);

/**
 * @brief RootRank表转换到字串，用于dfx
 *
 * @param[in] kv 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRootRankGetStr(NetTblRootRank *kv, uint8_t *buf, int32_t bufLen);

/**
 * @brief RootRank表val转换到字串，用于dfx
 *
 * @param[in] rootrank val 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblRootRankValGetStr(NetTblRootRankVal *val, uint8_t *buf, int32_t bufLen);

#ifdef __cplusplus
}
#endif

#endif
