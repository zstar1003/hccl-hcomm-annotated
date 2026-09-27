/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _VOS_BASE_H
#define _VOS_BASE_H

#include <inttypes.h>
#include <netinet/in.h>
#include "vos_typdef.h"

#ifdef __cplusplus
extern "C" {
#endif

#define VOS_PRId64 PRIi64
#define VOS_PRIu64 PRIu64
#define VOS_PRIX64 PRIX64
#define VOS_PRIx64 PRIx64
#define VOS_PRIdPTR PRIdPTR
#define VOS_PRIiPTR PRIiPTR
#define VOS_PRIuPTR PRIuPTR
#define VOS_PRIxPTR PRIxPTR
#define VOS_PRIXPTR PRIXPTR
#define VOS_PRIi64 PRIi64
#define VOS_PRIo64 PRIo64

#define VOS_NTOHL(x) ntohl(x)
#define VOS_HTONL(x) htonl(x)
#define VOS_NTOHS(x) ntohs(x)
#define VOS_HTONS(x) htons(x)
#define VOS_NTOHLL(x) ((((uint64_t)(VOS_NTOHL((uint32_t)(x)))) << 32) | (VOS_NTOHL((uint32_t)(((uint64_t)(x)) >> 32))))

#define VOS_HTONLL(x) ((((uint64_t)(VOS_NTOHL((uint32_t)(x)))) << 32) | (VOS_NTOHL((uint32_t)(((uint64_t)(x)) >> 32))))

/**
 * @ingroup vos_base
 * 基础时间数据结构。
 *
 * 时间范围的说明:
 * (1)DOPRA设定时间时年是没有上限的;
 * (2)DOPRA设定夏令时则年上限为2100;
 * (3)用户使用Linux系统年上限为2038;
 * (4)用户使用Vxworks系统年上限为2100;
 * (5)以上时间的下限均为1970-1-1 0:0:0.
 * 建议用户使用这个最小的交集，即年份的界限为 1970-1-1 0:0:0 ~ 2038-01-19 03:14:08.
 */
typedef struct tagVosSystm {
    uint16_t usYear;    /**< 年，取值范围为 >1970, <2038 */
    uint8_t ucMonth;    /**< 月，取值范围为[1,12] */
    uint8_t ucDate;     /**< 日，取值范围为[1, MAXDATE], 最大日期取决于年/月 */
    uint8_t ucHour;     /**< 小时, 取值范围为[0, 23] */
    uint8_t ucMinute;   /**< 分，取值范围为[0, 59] */
    uint8_t ucSecond;   /**< 秒，取值范围为[0,59] */
    uint8_t ucWeek;     /**< 星期，取值范围为[0, 6], 0代表星期天 */
    uint32_t uiMillSec; /**< 毫秒，取值范围为[0,999] */
} VOS_SYSTM_S;

/**
 * @ingroup vos_base
 * CPU TICK 时间。
 */
typedef struct tagCpuTick {
    uint32_t ulLow;  /**< 储存 CPU TICK 的寄存器的低 32 位 */
    uint32_t ulHigh; /**< 储存 CPU TICK 的寄存器的高 32 位 */
} CPU_TICK;

/**
 * @ingroup vos_base
 * CpuTick类型定义。
 */
typedef struct tagVosCpuTick {
    uint32_t uiLow;  /**< 低32位 */
    uint32_t uiHigh; /**< 高32位 */
} VOS_CPUTICK_S;

#ifdef __cplusplus
}
#endif

#endif /* _VOS_BASE_H */
