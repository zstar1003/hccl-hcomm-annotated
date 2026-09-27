/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_PUBER_H

#define BIFROST_CNCOI_PUBER_H

#include "bkf_comm.h"
#include "bkf_url.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_tmr.h"
#include "bkf_job.h"
#include "bkf_xmap.h"
#include "bkf_sys_log.h"
#include "bkf_puber_adef.h"
#include "bifrost_cncoi_slice.h"

#if __cplusplus
extern "C" {
#endif
#pragma pack(4)

/**
 * @brief 发布者句柄
 */
typedef struct tagBifrostCncoiPuber BifrostCncoiPuber;

/**
 * @brief 发布者初始化参数
 */
typedef struct tagBifrostCncoiPuberInitArg {
    char *name;               /**< 发布者库名称 */
    uint16_t idlVersionMajor; /**< 格式化编码主要版本号 */
    uint16_t idlVersionMinor; /**< 格式化编码次要版本号 */
    BOOL dbgOn;               /**< puber端dbg开关，关闭则无日志输出 */
    BkfMemMng *memMng;        /**< 内存库句柄,@see bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;            /**< 诊断显示库句柄,@see bkf_disp.h,同一app内可复用 */
    BkfLog *log;              /**< log库句柄,@see bkf_log.h,同一app内可复用 */
    BkfPfm *pfm;              /**< 性能测量库句柄,@see bkf_pfm.h,同一app内可复用 */
    BkfXMap
        *xMap; /**< 消息分发xmap句柄,见bkf_xmap.h,用于app将dms消息分发给bkf,配合BKF_MSG_DISPATCH使用,同一app内可复用 */
    BkfTmrMng *tmrMng;
    BkfJobMng *jobMng;
    uint32_t jobTypeId;
    uint32_t jobPrio;
    void *cookie;
    F_BKF_PUBER_VERIFY_MAY_ACCELERATE verifyMayAccelerate;
    BkfSysLogMng *sysLogMng;
    BkfDc *dc;
    BkfChSerMng *chMng;
    int32_t connCntMax;
    void *simpoBuilder;
    F_PUBER_ON_CONNECT onConnect;
    F_PUBER_ON_DISCONNECT onDisconnect;
    F_PUBER_ON_CONNECTOVERLOAD onConnectOver;
    uint8_t rsv1[0x10];
} BifrostCncoiPuberInitArg;

/**
 * @brief 发布者库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 发布者库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BifrostCncoiPuber *BifrostCncoiPuberInit(BifrostCncoiPuberInitArg *arg);
/**
 * @brief 发布者库销毁
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @return 无
 */
void BifrostCncoiPuberUninit(BifrostCncoiPuber *bifrostCncoiPuber);
/**
 * @brief 发布者库使能，使能前无法正常通信
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @return 使能结果
 *   @retval BKF_OK 使能成功
 *   @retval 非BKF_OK 使能失败
 */
uint32_t BifrostCncoiPuberEnable(BifrostCncoiPuber *bifrostCncoiPuber);
/**
 * @brief 设置发布者服务实例，用于服务端多实例场景
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiPuberSetSelfSvcInstId(BifrostCncoiPuber *bifrostCncoiPuber, uint32_t instId);
/**
 * @brief 设置发布者地址，用于订阅端发起连接
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @param[in] *url 地址
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiPuberSetSelfUrl(BifrostCncoiPuber *bifrostCncoiPuber, BkfUrl *url);
/**
 * @brief 删除设置发布者地址
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @return 无
 */
void BifrostCncoiPuberUnsetSelfUrl(BifrostCncoiPuber *bifrostCncoiPuber, BkfUrl *url);

/**
 * @brief 创建数据切片，即发布的数据表子集
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @param[in] *sliceKey 切片key
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiPuberCreateSlice(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey);
/**
 * @brief 删除数据切片
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @param[in] *sliceKey 切片key
 * @return 无
 */
void BifrostCncoiPuberDeleteSlice(BifrostCncoiPuber *bifrostCncoiPuber, BifrostCncoiSliceKeyT *sliceKey);

/**
 * @brief 设置puber支持的连接上限
 *
 * @param[in] *xxxPuber 发布者库句柄
 * @param[in] connMax 连接数上限
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiPuberSetConnUpLimit(BifrostCncoiPuber *bifrostCncoiPuber, uint32_t connMax);

#pragma pack()

#if __cplusplus
}
#endif

#endif
