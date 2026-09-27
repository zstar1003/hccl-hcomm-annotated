/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_RANK_H
#define NET_VO_TBL_RANK_H

#include "bkf_comm.h"
#include "net_vo_tbl_comm_info.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

#define NET_GLBRANKINFO_MAXSNDRANKINFOLEN 256
#define NET_GLBRANKINFO_MAXRANKNUM 1024

typedef struct TagNetTblGlobalRankInfo {
    uint32_t deviceIp;
    uint32_t serverIp;
} NetTblGlobalRankInfo;

typedef struct TagNetTblGlobalRankKey {
    uint64_t taskId;
    char commDesc[NET_COMMDESC_MAXLEN];
    uint64_t commInitTime;
    uint16_t packetId;
    uint16_t resv;
} NetTblGlobalRankKey;

typedef struct TagNetTblGlobalRankVal {
    uint16_t packetNum;
    uint16_t resv;
    uint32_t sendRankInfo[NET_GLBRANKINFO_MAXSNDRANKINFOLEN];
    uint8_t commMd5Sum[NET_COMMMD5SUM_MAXLEN];
    uint32_t rankTotalNum;
    uint16_t rankNum;
    uint16_t resv2;
    NetTblGlobalRankInfo rankInfo[0];
} NetTblGlobalRankVal;

typedef struct TagNetTblGlobalRank {
    NetTblGlobalRankKey key;
    NetTblGlobalRankVal val;
} NetTblGlobalRank;

#pragma pack()

/**
 * @brief 全局rank表key比较函数
 *
 * @param[in] key1Input key1
 * @param[in] key2InDs key2
 * @return 比较结果
 *   @retval >0 key1 > key2
 *   @retval =0 key1 = key2
 *   @retval <0 key1 < key2
 */
int32_t NetTblGlobalRankKeyCmp(NetTblGlobalRankKey *key1Input, NetTblGlobalRankKey *key2InDs);

/**
 * @brief 全局rank表key主机序转换到网络序
 *
 * @param[in] keyH key，主机序
 * @param[in] keyN key，网络序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblGlobalRankKeyH2N(NetTblGlobalRankKey *keyH, NetTblGlobalRankKey *keyN);

/**
 * @brief 全局rank表key网络序转换到主机序
 *
 * @param[in] keyN key，网络序
 * @param[in] keyH key，主机序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblGlobalRankKeyN2H(NetTblGlobalRankKey *keyN, NetTblGlobalRankKey *keyH);

/**
 * @brief 全局rank表key转换到字串，用于dfx
 *
 * @param[in] key 表键值
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblGlobalRankKeyGetStr(NetTblGlobalRankKey *key, uint8_t *buf, int32_t bufLen);

/**
 * @brief 全局rank表主机序转换到网络序
 *
 * @param[in] kvH 表，主机序
 * @param[in] kvN 表，网络序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblGlobalRankH2N(NetTblGlobalRank *kvH, NetTblGlobalRank *kvN);

/**
 * @brief 全局rank表网络序转换到主机序
 *
 * @param[in] kvN 表，网络序
 * @param[in] kvH 表，主机序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblGlobalRankN2H(NetTblGlobalRank *kvN, NetTblGlobalRank *kvH);

/**
 * @brief 全局rank表转换到字串，用于dfx
 *
 * @param[in] kv 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblGlobalRankGetStr(NetTblGlobalRank *kv, uint8_t *buf, int32_t bufLen);

/**
 * @brief 全局rank表val转换到字串，用于dfx
 *
 * @param[in] 全局rank val 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblGlobalRankValGetStr(NetTblGlobalRankVal *val, uint8_t *buf, int32_t bufLen);

#ifdef __cplusplus
}
#endif

#endif
