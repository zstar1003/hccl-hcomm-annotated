/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_COMM_H
#define BKF_COMM_H

#include "vrp_typdef.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)

#ifndef STATIC
#define STATIC
#endif

#define BKF_STATIC_DEBUG
#define BKF_MEM_DEBUG_STR
#define BKF_MEM_STAT
#define BKF_PFM_ON

/* static assert */
#ifdef BKF_STATIC_DEBUG
#define BKF_STATIC_ASSERT(X) typedef uint8_t BkfStaticAssert__[(!!(X)) * 2 - 1]

#else

#define BKF_STATIC_ASSERT(X)

#endif

#define BKF_BLOCK(desp) 1

#define BKF_OK 0u
#define BKF_ERR (1900000000u + __LINE__)

#define BKF_BUILD_RET(ret, x) ((ret) + (uint8_t)(x) * 100000)

#define BKF_INVALID (-1)
#define BKF_1K (1024)
#define BKF_1M (BKF_1K * BKF_1K)
#define BKF_1G (BKF_1M * BKF_1K)
#define BKF_4BYTE (4)
#define BKF_8BYTE (8)
#define BKF_16BYTE (16)
#define BKF_NUM5 (5)
#define BKF_NUM3 (3)
#define BKF_NUM4 (4)
#define BKF_NUM2 (2)
#define BKF_NUM6 (6)
#define BKF_256BYTE (256)

#define BKF_LOG_LEN (512)

#define BKF_NS_PER_MS (1000000)
#define BKF_US_PER_MS (1000)
#define BKF_MS_PER_S (1000)

#define BKF_NAME_LEN_MAX 63

#define BKF_SIGN_SET(sign, val)                                                                                        \
    do {                                                                                                               \
        (sign) = (val);                                                                                                \
    } while (0)
#define BKF_SIGN_CLR(sign)                                                                                             \
    do {                                                                                                               \
        (sign) = 1;                                                                                                    \
    } while (0)
#define BKF_SIGN_IS_VALID(sign, val) ((sign) == (val))

#define BKF_SELF_ASSIGN(x)                                                                                             \
    do {                                                                                                               \
        (x) = (x);                                                                                                     \
    } while (0)

typedef uint32_t (*F_BKF_DO)(void *x);
typedef void (*F_BKF_DOV)(void *x);
typedef struct tagBkfModVTbl {
    F_BKF_DO init;
    F_BKF_DOV uninit;
} BkfModVTbl;

uint32_t BkfModsInit(const BkfModVTbl *modsVTbl, void *doArg, int32_t cnt);
void BkfModsUninit(const BkfModVTbl *modsVTbl, void *doArg, int32_t cnt);

#define BKF_GET_ARY_COUNT(a) (sizeof(a) / sizeof((a)[0]))

#define BKF_GET_ARY_SIM_MTRX_ITEM(aryBeginAddr, colCnt, row, col) ((aryBeginAddr) + (colCnt) * (row) + (col))

#define BKF_OFFSET(type, mbr) (uint16_t)((char *)(&(((type *)200)->mbr)) - (char *)200)
#define BKF_MBR_SIZE(type, mbr) sizeof(((type *)200)->mbr)
#define BKF_MBR_PARENT(type, mbr, mbrAddr) ((type *)((uint8_t *)(mbrAddr) - BKF_OFFSET(type, mbr)))
#define BKF_MBR_IS_FIRST(type, mbr) (offsetof(type, mbr) == 0)
#define BKF_MBR_IS_LAST(type, mbr) (offsetof(type, mbr) == (sizeof(type) - BKF_MBR_SIZE(type, mbr)))

#define BKF_MASK_ADDR(p) (((p) != VOS_NULL) ? ((((uint16_t)(uintptr_t)(p)) & 0xffff) | 0x1) : 0)

#define BKF_GET_MAX(a, b) (((a) >= (b)) ? (a) : (b))
#define BKF_GET_MIN(a, b) (((a) <= (b)) ? (a) : (b))

#define BKF_GET_BITS_MAX(bits) ((uint32_t)((((uint64_t)1) << (bits)) - 1)) /* bits范围[0, 32] */
#define BKF_GET_MASK(bits) BKF_GET_BITS_MAX(bits)

#define BKF_GET_ALIGN4_LEN(len) ((((uint32_t)(len) + 3) >> 2) << 2)
#define BKF_LEN_IS_ALIGN4(len) (((uint32_t)(len) & BKF_GET_MASK(2)) == 0)

#define BKF_BUILD_U16(H8, L8) (((uint16_t)(H8) << 8) | (uint16_t)(uint8_t)(L8))
#define BKF_GET_L8(v) ((uint8_t)(v))
#define BKF_GET_H8(v) ((uint8_t)((uint16_t)(v) >> 8))

#define BKF_BUILD_U32(H16, L16) (((uint32_t)(H16) << 16) | (uint32_t)(uint16_t)(L16))
#define BKF_GET_L16(v) ((uint16_t)(v))
#define BKF_GET_H16(v) ((uint16_t)((uint32_t)(v) >> 16))

#define BKF_GET_U8_FOLD4_VAL(x) ((((uint8_t)(x) >> 4) + ((uint8_t)(x))) & 0x0fu)
#define BKF_GET_U16_FOLD8_VAL(x) ((uint8_t)(((uint16_t)(x) >> 8) + ((uint16_t)(x))))
#define BKF_GET_U32_FOLD8_VAL(x)                                                                                       \
    ((uint8_t)(((uint32_t)(x) >> 24) + ((uint32_t)(x) >> 16) + ((uint32_t)(x) >> 8) + ((uint32_t)(x))))

static inline uint32_t BkfGetStrHashCode(char *str, uint32_t chCntMaxOfCalcHashCode)
{
    uint32_t hashCode = 0;
    uint32_t i = 0;
    while ((i < chCntMaxOfCalcHashCode) && (str[i] != '\0')) {
        hashCode = (hashCode << BKF_NUM5) - hashCode + str[i];
        i++;
    }
    return hashCode;
}

#define BKF_BIT_SET(f, b) ((f) |= (b))
#define BKF_BIT_CLR(f, b) ((f) &= ~(b))
#define BKF_BIT_TEST(f, b) ((f) & (b))

#define BKF_COND_2BIT_FIELD(cond) ((cond) ? VOS_TRUE : VOS_FALSE)

#define BKF_GET_NUM_STR(num, numStrAry) (((num) < BKF_GET_ARY_COUNT(numStrAry)) ? (numStrAry)[num] : "__num_unknown__")

#define BKF_GET_NEXT_VAL(valSeed) ((valSeed)++)

char *BkfGetMemStr(uint8_t *mem, uint32_t memLen, uint32_t newLenPerByte, uint8_t *buf, int32_t bufLen);
#define BKF_GET_MEM_STD_STR(mem, memLen, buf, bufLen) BkfGetMemStr((uint8_t *)(mem), (memLen), 16, (buf), (bufLen))

char *BkfTrimStrPath(char *str);

#define BKF_BUILD_X_AND_STR(x) (x), #x

#ifndef likely
#define likely(x) __builtin_expect(!!(x), 1) // gcc内置函数, 帮助编译器分支优化
#endif

#ifndef unlikely
#define unlikely(x) __builtin_expect(!!(x), 0)
#endif

#define BKF_RETURNVAL_IF(expr, val)                                                                                    \
    do {                                                                                                               \
        if (unlikely(expr)) {                                                                                          \
            return (val);                                                                                              \
        }                                                                                                              \
    } while (0)

#define BKF_RETURNNULL_IF(expr) BKF_RETURNVAL_IF(expr, VOS_NULL)

#define BKF_RETURNvoid_IF(expr)                                                                                        \
    do {                                                                                                               \
        if (unlikely(expr)) {                                                                                          \
            return;                                                                                                    \
        }                                                                                                              \
    } while (0)

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
