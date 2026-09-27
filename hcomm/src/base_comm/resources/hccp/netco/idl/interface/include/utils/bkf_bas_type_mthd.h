/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_BAS_TYPE_MTHD_H
#define BKF_BAS_TYPE_MTHD_H

#include "bkf_comm.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)

/*
1. cmp
2. codec
3. getStr，用于诊断
*/
typedef int32_t (*F_BKF_CMP)(void *key1Input, void *key2InDs);
typedef char *(*F_BKF_GET_STR)(void *x, uint8_t *buf, int32_t bufLen);

#define BKF_CMP_X(x1, x2) (((x1) >= (x2)) ? (((x1) > (x2)) ? 1 : 0) : -1)

/**
 * @brief char数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfCharCmp(const char *key1Input, const char *key2InDs);

/**
 * @brief char数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfCharGetStr(const char *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief UCHAR数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfUCharCmp(const uint8_t *key1Input, const uint8_t *key2InDs);

/**
 * @brief UCHAR数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfUCharGetStr(const uint8_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief int16_t数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfShortCmp(const int16_t *key1Input, const int16_t *key2InDs);

/**
 * @brief int16_t数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfShortGetStr(const int16_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief uint16_t数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t Bkfuint16_tCmp(const uint16_t *key1Input, const uint16_t *key2InDs);

/**
 * @brief uint16_t数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *Bkfuint16_tGetStr(const uint16_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief int16_t/uint16_t编解码
 *
 * @param[in] *inOut 数值地址，原地转换输出
 * @return 转码成功与否
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfXShortCodec(uint16_t *inOut);

/**
 * @brief INT32数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfInt32Cmp(const int32_t *key1Input, const int32_t *key2InDs);

/**
 * @brief INT32数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfInt32GetStr(const int32_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief uint32_t数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t Bkfuint32_tCmp(const uint32_t *key1Input, const uint32_t *key2InDs);

/**
 * @brief uint32_t数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *Bkfuint32_tGetStr(const uint32_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief INT32/uint32_t编解码
 *
 * @param[in] *inOut 数值地址，原地转换输出
 * @return 转码成功与否
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfXInt32Codec(uint32_t *inOut);

/**
 * @brief INT64数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfInt64Cmp(const int64_t *key1Input, const int64_t *key2InDs);

/**
 * @brief INT64数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfInt64GetStr(const int64_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief UINT64数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结构
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfUInt64Cmp(const uint64_t *key1Input, const uint64_t *key2InDs);

/**
 * @brief UINT64数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfUInt64GetStr(const uint64_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief INT64/UINT64编解码
 *
 * @param[in] *inOut 数值地址，原地转换输出
 * @return 转码成功与否
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfXInt64Codec(uint64_t *inOut);

/**
 * @brief INTPTR数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfIntptrCmp(const intptr_t *key1Input, const intptr_t *key2InDs);

/**
 * @brief INTPTR数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfIntptrGetStr(const intptr_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief UINTPTR数值比较
 *
 * @param[in] *key1Input 数值地址
 * @param[in] *key2InDs 另一个数值地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfUIntptrCmp(const uintptr_t *key1Input, const uintptr_t *key2InDs);

/**
 * @brief INTPTR数值转换成字串
 *
 * @param[in] *input 数值地址
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfUIntptrGetStr(const uintptr_t *input, uint8_t *buf, int32_t bufLen);

/**
 * @brief 地址比较
 *
 * @param[in] **key1Input 地址
 * @param[in] **key2InDs 另一地址
 * @return 比较的结果
 *   @retval 正数 *key1Input > key2InDs
 *   @retval 零 *key1Input == key2InDs
 *   @retval 负数 *key1Input < *key2InDs
 */
int32_t BkfCmpPtrAddr(const void **key1Input, const void **key2InDs);

/* 输入字串的地址(CHAR **), 然后解引用比对，容易误导保存字串的地址(认为是常量串)。因此，删除，避免使用 */

/**
 * @brief ip 地址比较
 *
 * @param[in] *key1IpAddrHInput ip地址,主机序
 * @param[in] *key2IpAddrHInDs 另一个ip地址
 * @return 比较的结构
 *   @retval 正数 *key1IpAddrHInput > key2IpAddrHInDs, 根据地址的大小比对，比如127.99.98.97 < 128.0.0.1
 *   @retval 零 *key1IpAddrHInput == key2IpAddrHInDs
 *   @retval 负数 *key1IpAddrHInput < *key2IpAddrHInDs
 */
int32_t BkfIpAddrCmp(const uint32_t *key1IpAddrHInput, const uint32_t *key2IpAddrHInDs);

/**
 * @brief ip 地址转换成字串
 *
 * @param[in] *ipAddrH ip地址,主机序
 * @param[in] *buf 转换成字串的缓冲
 * @param[in] bufLen 缓冲长度
 * @return 数值对应的字串。不会返回空串。如果参数不合法，会返回一个常量字串，用于错误提示。
 *   @retval 非空 数值对应的字串
 */
char *BkfIpAddrGetStr(const uint32_t *ipAddrH, uint8_t *buf, int32_t bufLen);

/**
 * @brief ip地址编解码
 *
 * @param[in] *ipAddrInOut 数值地址，原地转换输出
 * @return 转码成功与否
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfIpAddrCodec(uint32_t *ipAddrInOut);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
