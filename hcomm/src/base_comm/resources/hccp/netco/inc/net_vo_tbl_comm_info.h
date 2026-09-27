/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_VO_TBL_COMM_INFO_H
#define NET_VO_TBL_COMM_INFO_H

#include "bkf_comm.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)

#define NET_COMMDESC_MAXLEN 128
#define NET_COMMMD5SUM_MAXLEN 16
#define NET_COMMINFO_MAXRANKINFOLEN 4
#define NET_TBL_MEM_LEN_MAX 20 // BKF_GET_MEM_STD_STR操作的允许的最大长度

typedef struct TagNetTblRankInfo {
    uint32_t deviceIp;
    uint32_t serverIp;
    uint16_t podId;
    uint16_t resv;
} NetTblRankInfo;

typedef struct TagNetTblCommInfoKey {
    uint64_t taskId;
    char commDesc[NET_COMMDESC_MAXLEN];
    uint64_t commInitTime;
    uint16_t packetId;
    uint16_t resv;
} NetTblCommInfoKey;

typedef struct TagNetTblCommInfoVal {
    uint16_t packetNum;
    uint16_t resv;
    uint32_t sendRankInfo[NET_COMMINFO_MAXRANKINFOLEN];
    uint8_t commMd5Sum[NET_COMMMD5SUM_MAXLEN];
    uint16_t rankTotalNum;
    uint16_t rankNum;
    NetTblRankInfo rankInfo[0];
} NetTblCommInfoVal;

typedef struct TagNetTblCommInfo {
    NetTblCommInfoKey key;
    NetTblCommInfoVal val;
} NetTblCommInfo;

#pragma pack()

/**
 * @brief 通信域信息表key比较函数
 *
 * @param[in] key1Input key1
 * @param[in] key2InDs key2
 * @return 比较结果
 *   @retval >0 key1 > key2
 *   @retval =0 key1 = key2
 *   @retval <0 key1 < key2
 */
int32_t NetTblCommInfoKeyCmp(NetTblCommInfoKey *key1Input, NetTblCommInfoKey *key2InDs);

/**
 * @brief 通信域信息表key主机序转换到网络序
 *
 * @param[in] keyH key，主机序
 * @param[in] keyN key，网络序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblCommInfoKeyH2N(NetTblCommInfoKey *keyH, NetTblCommInfoKey *keyN);

/**
 * @brief 通信域信息表key网络序转换到主机序
 *
 * @param[in] keyN key，网络序
 * @param[in] keyH key，主机序
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblCommInfoKeyN2H(NetTblCommInfoKey *keyN, NetTblCommInfoKey *keyH);

/**
 * @brief 通信域信息表key转换到字串，用于dfx
 *
 * @param[in] key 表键值
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblCommInfoKeyGetStr(NetTblCommInfoKey *key, uint8_t *buf, int32_t bufLen);

/**
 * @brief 通信域信息表主机序转换到网络序
 *
 * @param[in] kvH 表，主机序
 * @param[in] kvN 表，网络序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblCommInfoH2N(NetTblCommInfo *kvH, NetTblCommInfo *kvN);

/**
 * @brief 通信域信息表网络序转换到主机序
 *
 * @param[in] kvN 表，网络序
 * @param[in] kvH 表，主机序。调用者确保缓冲长度足够
 * @return 处理是否成功
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t NetTblCommInfoN2H(NetTblCommInfo *kvN, NetTblCommInfo *kvH);

/**
 * @brief 通信域信息表转换到字串，用于dfx
 *
 * @param[in] kv 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblCommInfoGetStr(NetTblCommInfo *kv, uint8_t *buf, int32_t bufLen);

/**
 * @brief 通信域信息表val转换到字串，用于dfx
 *
 * @param[in] 通信信息 val 表
 * @return 返回转换后的字串。不会返回空串
 *   @retval 非空 字串值
 */
char *NetTblCommInfoValGetStr(NetTblCommInfoVal *val, uint8_t *buf, int32_t bufLen);

#ifdef __cplusplus
}
#endif

#endif
