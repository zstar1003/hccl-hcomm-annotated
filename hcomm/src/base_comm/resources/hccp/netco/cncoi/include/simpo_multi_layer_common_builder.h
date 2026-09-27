/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIMPO_MULTI_LAYER_COMMON_BUILDER_H
#define SIMPO_MULTI_LAYER_COMMON_BUILDER_H

#include "flatbuffers_common_builder.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef SIMPO_INLINE
#define SIMPO_INLINE inline __attribute__((always_inline))
#endif

#define SIMPO_BUILD_OK 0              /* 操作成功 */
#define SIMPO_BUILD_INVAL (-1)        /* 非法输入，如数组传入NULL */
#define SIMPO_BUILD_NOMATCH (-2)      /* 序列化操作不匹配 */
#define SIMPO_BUILD_REPEATED_ADD (-3) /* 字段重复添加 */
#define SIMPO_BUILD_NOMEM (-4)        /* 用户序列化的buffer长度不足 */
#define SIMPO_BUILD_ORDER_ERR (-5)    /* 添加顺序错误 */
#define SIMPO_BUILD_ORDER_MISS (-6)   /* 添加字段有缺失 */

#define SIMPO_BUILD_MAX_STACK_LEN 20

#define SIMPO_BUILD_FLAG_SIZE 4

#define SIMPO_BUILD_FIXED_LENGTH_VEC_FLAG 3

#define SIMPO_BUILD_LENGTH_SIZE 4

#define SIMPO_TRANS_WORD(x)                                                                                            \
    do {                                                                                                               \
        (x) = SIMPO_SWAP_2(x);                                                                                         \
    } while (0)

#define SIMPO_TRANS_DWORD(x)                                                                                           \
    do {                                                                                                               \
        (x) = SIMPO_SWAP_4(x);                                                                                         \
    } while (0)

#define SIMPO_TRANS_QWORD(x)                                                                                           \
    do {                                                                                                               \
        (x) = SIMPO_SWAP_8(x);                                                                                         \
    } while (0)

#define SIMPO_TRANS_FLOAT(x)                                                                                           \
    do {                                                                                                               \
        (x) = FloatTransToLe(x);                                                                                       \
    } while (0)

#define SIMPO_TRANS_DOUBLE(x)                                                                                          \
    do {                                                                                                               \
        (x) = DoubleTransToLe(x);                                                                                      \
    } while (0)

#define SIMPO_TRANS_VALUE_DOUBLE(x) DoubleTransToLe(*(double *)(x))

#define SIMPO_TRANS_VALUE_FLOAT(x) FloatTransToLe(*(float *)(x))

#define SIMPO_TRANS_VALUE_8(x) SIMPO_SWAP_8(*(uint64_t *)(x))

#define SIMPO_TRANS_VALUE_4(x) SIMPO_SWAP_4(*(uint32_t *)(x))

#define SIMPO_TRANS_VALUE_2(x) SIMPO_SWAP_2(*(uint16_t *)(x))

#define SIMPO_TRANS_VALUE_1(x) *((uint8_t *)(x))

#define SIMPO_TABLE_LENGTH_SIZE 4

/* 反序列化用宏, 数组是否为定长 */
#define SIMPO_MULTI_CHECK_VEC_IS_FIXED(length) (((length) & (1 << 24)) != 0)

#define SIMPO_OPER_INIT(builder, index)                                                                                \
    do {                                                                                                               \
        if ((builder)->errorCode != SIMPO_BUILD_OK) {                                                                  \
            return (builder)->errorCode;                                                                               \
        }                                                                                                              \
        (builder)->id[(index)] = -1;                                                                                   \
    } while (0)

#define SIMPO_CHECK_OPER_ORDER(builder, index, oper, _id)                                                              \
    do {                                                                                                               \
        if (((builder)->errorCode == SIMPO_BUILD_OK) && ((builder)->id[(index)] + 1) != (_id)) {                       \
            (builder)->errorCode = SIMPO_BUILD_ORDER_ERR;                                                              \
            (builder)->errorOperName = #oper;                                                                          \
            return (builder)->errorCode;                                                                               \
        }                                                                                                              \
        (builder)->id[(index)] = (_id);                                                                                \
    } while (0)

#define SIMPO_CHECK_OPER_END(builder, index, oper, _id)                                                                \
    do {                                                                                                               \
        if (((builder)->errorCode == SIMPO_BUILD_OK) && (((builder)->id[(index)]) != (_id))) {                         \
            (builder)->errorCode = SIMPO_BUILD_ORDER_MISS;                                                             \
            (builder)->errorOperName = #oper;                                                                          \
            return (builder)->errorCode;                                                                               \
        }                                                                                                              \
    } while (0)

typedef struct tagSimpoBuilder {
    int32_t errorCode;                                 /* 错误码 */
    uint32_t temp;                                     /* 字节对齐 */
    uint8_t *curWriteAddr;                             /* 当前写的位置 */
    uint8_t *bufferEnd;                                /* buffer终点 */
    uint8_t *tempWriteAddr[SIMPO_BUILD_MAX_STACK_LEN]; /* 暂存上一层写的位置 */
    uint32_t cnt[SIMPO_BUILD_MAX_STACK_LEN];
    int32_t id[SIMPO_BUILD_MAX_STACK_LEN];
    char *errorOperName;
    uint8_t *buffer; /* buffer长度 */
    uint8_t *structVecLengthAddr;
    uint32_t structVecLength;
    uint32_t bufferLen; /* 待写入的buffer */
    uint32_t msgHead;
    uint32_t isBigEndian;
    uint32_t dopraEndian;
} SimpoBuilder;

typedef struct {
    union {
        uint32_t length;
        uint8_t flag[SIMPO_BUILD_FLAG_SIZE];
    };
    uint32_t sameLegth;
} SimpoLengthFlagU;

typedef SimpoBuilder SimpoBuilderT;

#if (VOS_BYTE_ORDER == VOS_BIG_ENDIAN)
#define SET_FIXED_LENGTH_VEC_HEAD(head, _length, _sameLength)                                                          \
    do {                                                                                                               \
        ((SimpoLengthFlagU *)(head))->length = (_length);                                                              \
        ((SimpoLengthFlagU *)(head))->flag[0] = 1;                                                                     \
        ((SimpoLengthFlagU *)(head))->sameLegth = (_sameLength);                                                       \
    } while (0)
#else /* VOS_BYTE_ORDER == VOS_LITTLE_ENDIAN */
#define SET_FIXED_LENGTH_VEC_HEAD(head, _length, _sameLength)                                                          \
    do {                                                                                                               \
        ((SimpoLengthFlagU *)(head))->length = (_length);                                                              \
        ((SimpoLengthFlagU *)(head))->flag[SIMPO_BUILD_FIXED_LENGTH_VEC_FLAG] = 1;                                     \
        ((SimpoLengthFlagU *)(head))->sameLegth = (_sameLength);                                                       \
    } while (0)
#endif /* VOS_BYTE_ORDER */

__attribute__((unused)) static const char *SimpoBuilderErrnoStr(int32_t err)
{
    if (!((err <= SIMPO_BUILD_OK) && (err >= SIMPO_BUILD_ORDER_MISS))) {
        return "unknow";
    }
    err *= -1;
    const char *strs[] = {"ok", "inval param", "opear not match", "field repeated add", "no memory",
        "add field order error", "add field order miss"};
    return strs[err];
}

__attribute__((unused)) static SimpoBuilderT *SimpoBuilderInit(void)
{
    SimpoBuilderT *builder = (SimpoBuilderT *)SIMPO_Alloc(sizeof(SimpoBuilderT));
    if (builder == NULL) {
        return NULL;
    }
    (void)memset_s(builder, sizeof(SimpoBuilderT), 0, sizeof(SimpoBuilderT));
    builder->isBigEndian = g_uiSimpoEndianIsBig;
    const uint8_t kEndianFlagByteOffset = 3;
    *(((uint8_t *)&(builder->msgHead)) +
        kEndianFlagByteOffset) = ((g_uiSimpoEndianIsBig != 0) ? 1 : 0); /* 第3个字节表示消息来自大端或小端 */
#if (VOS_BYTE_ORDER == VOS_BIG_ENDIAN)
    builder->dopraEndian = 1;
#else
    builder->dopraEndian = 0;
#endif
    return builder;
}

__attribute__((unused)) static void SimpoBuilderUninit(SimpoBuilderT *builder)
{
    if (builder != NULL) {
        SIMPO_Free(builder);
    }

    return;
}
#ifdef __cplusplus
}
#endif /* __cplusplus */

#endif
