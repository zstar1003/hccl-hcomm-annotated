/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef FLATGBUFFERS_COMMON_BUILDER_H
#define FLATGBUFFERS_COMMON_BUILDER_H
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include "securec.h"
#include "vos_base.h"

#if defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#pragma GCC diagnostic push
#endif

#ifndef UINT8_MAX
#include <stddef.h>
#include <limits.h>
#if ((defined(__STDC__) && __STDC__ && defined(__STDC_VERSION__) && __STDC_VERSION__ >= 199901L) ||                    \
     (defined(__WATCOMC__) && (defined(_STDINT_H_INCLUDED) || __WATCOMC__ >= 1250)) ||                                 \
     (defined(__GNUC__) &&                                                                                             \
         (__GNUC__ > 3 || defined(_STDINT_H) || defined(_STDINT_H_) || defined(__UINT_FAST64_TYPE__))))
#include <stdint.h>
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef __FILENAME__
#define SIMPO_DOLPHIN_FILENAME __FILENAME__
#else
#define SIMPO_DOLPHIN_FILENAME ""
#endif

#ifndef SIMPO_DOLPHIN_OSAL_LOG
#define SIMPO_DOLPHIN_OSAL_LOG
#define SIMPO_DOLPHIN_OSAL_LOG_DEFINE_OBJECT(pscModName, pcSubModName, uiMid)                                          \
    do {                                                                                                               \
    } while (0)

#define SIMPO_DOLPHIN_LogDbg(errid, args...)                                                                           \
    do {                                                                                                               \
    } while (0)

#define SIMPO_DOLPHIN_LogInfo(errid, args...)                                                                          \
    do {                                                                                                               \
    } while (0)

#define SIMPO_DOLPHIN_LogWarning(errid, args...)                                                                       \
    do {                                                                                                               \
    } while (0)

#define SIMPO_DOLPHIN_LogError(errid, args...)                                                                         \
    do {                                                                                                               \
    } while (0)

#define SIMPO_DOLPHIN_LogExcept(errid, args...)                                                                        \
    do {                                                                                                               \
    } while (0)
#endif

#ifndef SIMPO_RUNTIME_COMMON
#define SIMPO_RUNTIME_COMMON
typedef struct {
    uint8_t *pucBuffer;
    uint8_t *pucPackedBuffer;
    uint32_t uiBufferLen;
    uint32_t uiReserved;
} SIMPO_PACK_BUILDER_S;

typedef union {
    uint32_t uint32Val[2];
    double doubleVal;
} SIMPO_DOUBLE_TRANS_U;

typedef union {
    uint32_t uint32Val;
    float floatVal;
} SIMPO_FLOAT_TRANS_U;

typedef union {
    uint32_t uint32Val[2];
    uint64_t uint64Val;
} SIMPO_UINT64_TRANS_U;

#define SIMPO_BYTE_SIZE 1
#define SIMPO_WORD_SIZE 2
#define SIMPO_DWORD_SIZE 4
#define SIMPO_QWORD_SIZE 8
#define SIMPO_VEC_LEN_SIZE 4
#define SIMPO_REF_OFFSET_SIZE 4
#define SIMPO_VTABLE_OFFSET_SIZE 2
#define SIMPO_ENCODE_START_INDEX 0
#define SIMPO_DECODE_START_INDEX 0
#define SIMPO_VTABLE_HEAD_SIZE (2 * SIMPO_VTABLE_OFFSET_SIZE)
#define SIMPO_ROOT_VTABLE_LEN_OFFSET SIMPO_REF_OFFSET_SIZE
#define SIMPO_ROOT_TABLE_DATA_LEN_OFFSET (SIMPO_ROOT_VTABLE_LEN_OFFSET + SIMPO_VTABLE_OFFSET_SIZE)
#define SIMPO_ROOT_VTABLE_OFFSET (SIMPO_ROOT_TABLE_DATA_LEN_OFFSET + SIMPO_VTABLE_OFFSET_SIZE)
#define SIMPO_VEC_HEAD_SIZE (SIMPO_REF_OFFSET_SIZE + SIMPO_VEC_LEN_SIZE)

#define SIMPO_POINTER_STR(p) (((p) == VOS_NULL_PTR) ? "null" : "nonNull")
#define SIMPO_Alloc malloc
#define SIMPO_Free free
#define SIMPO_PACKED_BUFFER_OFFSET(pstBuilder) (uint32_t)((pstBuilder)->pucPackedBuffer - (pstBuilder)->pucBuffer)

#define SIMPO_GET_PACK_BUFFER_WRITE_ADDR(pSrc, type, pstBuilder, size)                                                 \
    do {                                                                                                               \
        (pSrc) = (type)((pstBuilder)->pucPackedBuffer);                                                                \
        (pstBuilder)->pucPackedBuffer += (uint32_t)(size);                                                             \
    } while (0)

static uint32_t g_uiSimpoEndianIsBig;

__attribute__((constructor)) static void SimpoCheckEndianness()
{
    uint32_t uiValue = 1;
    unsigned char *pucVaule = (unsigned char *)(&uiValue);
    g_uiSimpoEndianIsBig = ((*pucVaule) != 0) ? 0 : 1;
    return;
}
#define SIMPO_EndianIsBig() (g_uiSimpoEndianIsBig)

#define SIMPO_SWAP_1(x) (x)

#define SIMPO_SWAP_2(x) ((((x) & 0x00ff) << 8) | (((x) & 0xff00) >> 8))

#define SIMPO_SWAP_4(x)                                                                                                \
    ((((x) & 0x000000ffu) << 24) | (((x) & 0x0000ff00u) << 8) | (((x) & 0x00ff0000u) >> 8) |                           \
        (((x) & 0xff000000u) >> 24))

static inline uint64_t SIMPO_SWAP_8(uint64_t val)
{
    SIMPO_UINT64_TRANS_U temp;
    temp.uint64Val = val;
    uint32_t temp2 = temp.uint32Val[0];
    temp.uint32Val[0] = (uint32_t)SIMPO_SWAP_4(temp.uint32Val[1]);
    temp.uint32Val[1] = (uint32_t)SIMPO_SWAP_4(temp2);
    return temp.uint64Val;
}

#define SIMPO_TRANS_SCALAR_QWORD(pDesData, pSrcData)                                                                   \
    do {                                                                                                               \
        *((uint64_t *)(pDesData)) = (g_uiSimpoEndianIsBig != 0) ? (SIMPO_SWAP_8(*(uint64_t *)(pSrcData)))              \
                                                                : (*(uint64_t *)(pSrcData));                           \
    } while (0)

#define SIMPO_TRANS_SCALAR_DWORD(pDesData, pSrcData)                                                                   \
    do {                                                                                                               \
        *((uint32_t *)(pDesData)) = (g_uiSimpoEndianIsBig != 0) ? (SIMPO_SWAP_4(*(uint32_t *)(pSrcData)))              \
                                                                : (*(uint32_t *)(pSrcData));                           \
    } while (0)

#define SIMPO_TRANS_SCALAR_WORD(pDesData, pSrcData)                                                                    \
    do {                                                                                                               \
        *((uint16_t *)(pDesData)) = (g_uiSimpoEndianIsBig != 0) ? ((uint16_t)SIMPO_SWAP_2(*(uint16_t *)(pSrcData)))    \
                                                                : (*(uint16_t *)(pSrcData));                           \
    } while (0)

#define SIMPO_TRANS_SCALAR_BYTE(pDesData, pSrcData)                                                                    \
    do {                                                                                                               \
        *((uint8_t *)(pDesData)) = (*(uint8_t *)(pSrcData));                                                           \
    } while (0)

#define SIMPO_ENCODE_SCALAR(pDesData, pSrcData, size) SIMPO_TRANS_SCALAR_##size((pDesData), (pSrcData))

#define SIMPO_DECODE_SCALAR(pDesData, pSrcData, size) SIMPO_TRANS_SCALAR_##size((pDesData), (pSrcData))

#define SIMPO_DECODE_DATA_OFFSET(uiDataOffset)                                                                         \
    ((g_uiSimpoEndianIsBig != 0) ? (SIMPO_SWAP_4((uint32_t)(uiDataOffset))) : (uiDataOffset))
#define SIMPO_DECODE_VTABLE_OFFSET(usVtableOffset)                                                                     \
    ((g_uiSimpoEndianIsBig != 0) ? ((uint16_t)SIMPO_SWAP_2((uint16_t)(usVtableOffset))) : (usVtableOffset))
#define SIMPO_ENCODE_DATA_OFFSET(pDes, src)                                                                            \
    do {                                                                                                               \
        *(uint32_t *)(pDes) = ((g_uiSimpoEndianIsBig != 0) ? SIMPO_SWAP_4((uint32_t)(src)) : (uint32_t)(src));         \
    } while (0)
#define SIMPO_ENCODE_VTABLE_OFFSET(pDes, src)                                                                          \
    do {                                                                                                               \
        *(uint16_t *)(pDes) = ((g_uiSimpoEndianIsBig != 0) ? (uint16_t)SIMPO_SWAP_2((uint16_t)(src))                   \
                                                           : (uint16_t)(src));                                         \
    } while (0)

#define SIMPO_COPY_STRUCT_VECTOR(pDesVec, pSrcVec, uiLen, name, kind)                                                  \
    do {                                                                                                               \
        if (g_uiSimpoEndianIsBig != 0) {                                                                               \
            uint32_t uiStructIndexCopy;                                                                                \
            for (uiStructIndexCopy = 0; uiStructIndexCopy < (uiLen); ++uiStructIndexCopy) {                            \
                simpo##kind##Struct##name((name##_t *)(pDesVec) + uiStructIndexCopy,                                   \
                    (name##_t *)(pSrcVec) + uiStructIndexCopy);                                                        \
            }                                                                                                          \
        } else if ((uiLen) > 0) {                                                                                      \
            uint32_t desVecLen = (uint32_t)sizeof(name##_t) * (uiLen);                                                 \
            (void)memcpy_sp((name##_t *)(pDesVec), desVecLen, (name##_t *)(pSrcVec),                                   \
                (uint32_t)sizeof(name##_t) * (uiLen));                                                                 \
        }                                                                                                              \
    } while (0)

#define SIMPO_COPY_VECTOR(pDesData, pSrcData, size, uiVecLen, kind)                                                    \
    do {                                                                                                               \
        if (((SIMPO_##size##_SIZE) != 1) && (g_uiSimpoEndianIsBig)) {                                                  \
            uint32_t uiIndexCopy = SIMPO_##kind##_START_INDEX;                                                         \
            for (; uiIndexCopy < (uiVecLen); ++uiIndexCopy) {                                                          \
                SIMPO_TRANS_SCALAR_##size((uint8_t *)(pDesData) + uiIndexCopy * (SIMPO_##size##_SIZE),                 \
                    (uint8_t *)(pSrcData) + uiIndexCopy * (SIMPO_##size##_SIZE));                                      \
            }                                                                                                          \
        } else if ((uiVecLen) > 0) {                                                                                   \
            uint32_t desArrayLen = (SIMPO_##size##_SIZE) * (uiVecLen);                                                 \
            (void)memcpy_sp((uint8_t *)(pDesData), desArrayLen, (uint8_t *)(pSrcData),                                 \
                (SIMPO_##size##_SIZE) * (uiVecLen));                                                                   \
        }                                                                                                              \
    } while (0)

#define SIMPO_ENCODE_VECTOR(pDesBuff, pSrcBuff, size, uiVecLen)                                                        \
    do {                                                                                                               \
        SIMPO_ENCODE_DATA_OFFSET(pDesBuff, uiVecLen);                                                                  \
        SIMPO_COPY_VECTOR((uint8_t *)(pDesBuff) + SIMPO_VEC_LEN_SIZE, (uint8_t *)(pSrcBuff), size, (uiVecLen),         \
            ENCODE);                                                                                                   \
    } while (0)

#define SIMPO_GET_VTABLE_OFFSET(member_offset, vtable, vtable_len, id)                                                 \
    do {                                                                                                               \
        if (4 + (id) * 2 + 2 <= (vtable_len)) {                                                                        \
            (member_offset) = SIMPO_DECODE_VTABLE_OFFSET(*((uint16_t *)(vtable) + (id)));                              \
        } else {                                                                                                       \
            (member_offset) = 0;                                                                                       \
        }                                                                                                              \
    } while (0)

#define SIMPO_GET_REF_OFFSET(uiBaseOffset, uiNextOffset, uiRefOffset, uiBufferLen, pucBuffer, freeHook, msg)           \
    do {                                                                                                               \
        if ((uiBaseOffset) + SIMPO_REF_OFFSET_SIZE > (uiBufferLen)) {                                                  \
            (freeHook)((msg));                                                                                         \
            return VOS_NULL_PTR;                                                                                       \
        }                                                                                                              \
        (uiNextOffset) = SIMPO_DECODE_DATA_OFFSET(*(uint32_t *)((uint8_t *)(pucBuffer) + (uiBaseOffset)));             \
        (uiRefOffset) = (uiBaseOffset) + (uiNextOffset);                                                               \
        if ((uiRefOffset) < (uiBaseOffset)) {                                                                          \
            (freeHook)((msg));                                                                                         \
            return VOS_NULL_PTR;                                                                                       \
        }                                                                                                              \
    } while (0)

#define SIMPO_GET_VECTOR_LEN(uiVecLen, uiDataLen, uiRefOffset, pBuffer, uiBufferLen, uiMemberSize, freeHook, msg)      \
    do {                                                                                                               \
        if (((uiRefOffset) > (uiBufferLen)) || ((uiRefOffset) + SIMPO_VEC_LEN_SIZE > (uiBufferLen))) {                 \
            (freeHook)((msg));                                                                                         \
            return VOS_NULL_PTR;                                                                                       \
        }                                                                                                              \
        (uiVecLen) = SIMPO_DECODE_DATA_OFFSET(*(uint32_t *)((uint8_t *)(pBuffer) + (uiRefOffset)));                    \
        (uiDataLen) = (uiVecLen) * (uiMemberSize);                                                                     \
        if ((uiDataLen) / (uiMemberSize) != (uiVecLen)) {                                                              \
            (freeHook)((msg));                                                                                         \
            return VOS_NULL_PTR;                                                                                       \
        }                                                                                                              \
        if (((uiRefOffset) + SIMPO_VEC_LEN_SIZE + (uiDataLen) > (uiBufferLen)) ||                                      \
            ((uiRefOffset) + SIMPO_VEC_LEN_SIZE + (uiDataLen) < (uiRefOffset) + SIMPO_VEC_LEN_SIZE)) {                 \
            (freeHook)((msg));                                                                                         \
            return VOS_NULL_PTR;                                                                                       \
        }                                                                                                              \
    } while (0)

#define SIMPO_PACK_SPEEDING_ROOT_TABLE_OFFSET 0x00000004
static inline double DoubleTransToLe(double val)
{
    SIMPO_DOUBLE_TRANS_U temp;
    temp.doubleVal = val;
    uint32_t temp2 = temp.uint32Val[0];
    temp.uint32Val[0] = (uint32_t)SIMPO_SWAP_4(temp.uint32Val[1]);
    temp.uint32Val[1] = (uint32_t)SIMPO_SWAP_4(temp2);
    return temp.doubleVal;
}
static inline float FloatTransToLe(float val)
{
    SIMPO_FLOAT_TRANS_U temp;
    temp.floatVal = val;
    temp.uint32Val = (uint32_t)SIMPO_SWAP_4(temp.uint32Val);
    return temp.floatVal;
}
#define DOUBLE_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (double)(x) : DoubleTransToLe(x))
#define FLOAT_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (float)(x) : FloatTransToLe(x))
#define UINT64_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (uint64_t)(x) : (uint64_t)SIMPO_SWAP_8((uint64_t)(x)))
#define UINT32_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (uint32_t)(x) : (uint32_t)SIMPO_SWAP_4((uint32_t)(x)))
#define UINT16_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (uint16_t)(x) : (uint16_t)SIMPO_SWAP_2((uint16_t)(x)))
#define INT64_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (int64_t)(x) : (int64_t)SIMPO_SWAP_8((uint64_t)(x)))
#define INT32_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (int32_t)(x) : (int32_t)SIMPO_SWAP_4((uint32_t)(x)))
#define INT16_CAST_TO_LE(isBigEndian, x) (((isBigEndian) == 0) ? (int16_t)(x) : (int16_t)SIMPO_SWAP_2((uint16_t)(x)))
#define CHAR_CAST_TO_LE(isBigEndian, x) (x)
#define UINT8_CAST_TO_LE(isBigEndian, x) (x)
#define INT8_CAST_TO_LE(isBigEndian, x) (x)
#define BOOL_CAST_TO_LE(isBigEndian, x) (x)
#define SIMPO_SCALAR_TO_LE(isBigEndian, type, val) (type##_CAST_TO_LE((isBigEndian), (val)))
#endif

#ifndef SIMPO_DEFINE_SIMPO_DOLPHIN_LOG_OBJECT
#define SIMPO_DEFINE_SIMPO_DOLPHIN_LOG_OBJECT
__attribute__((unused)) static void SimpoDolphinOsalLogObjectNoUse(void)
{
    return;
}
#endif

#ifndef SIMPO_DEFINE_FLATBUFFERS_ALLOCATOR
#define SIMPO_DEFINE_FLATBUFFERS_ALLOCATOR
/* flatbuffers_allocator 用于全量源码构建兼容老代码 */
typedef struct flatbuffers_allocator {
    void *(*pfnAlloc)(size_t size); // default:malloc
    void (*pfnFree)(void *pointer); // default:free
} flatbuffers_allocator;

__attribute__((unused)) static flatbuffers_allocator g_stFbAllocator = {malloc, free};
#endif

#ifndef FLATCC_ALLOC
#define FLATCC_ALLOC(n) g_stFbAllocator.pfnAlloc(n)
#endif

#ifndef FLATCC_FREE
#define FLATCC_FREE(p) g_stFbAllocator.pfnFree(p)
#endif
#ifndef FLATBUFFERS_TRUE
#define FLATBUFFERS_TRUE 1
#endif
#ifndef FLATBUFFERS_FALSE
#define FLATBUFFERS_FALSE 0
#endif

#ifdef __cplusplus
}
#endif

#if defined(__GNUC__) && ((__GNUC__ > 4) || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6))
#endif

#endif
