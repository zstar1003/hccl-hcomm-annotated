/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BIFROST_CNCOI_SUBER_H

#define BIFROST_CNCOI_SUBER_H

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
#include "bkf_ch_cli_adef.h"
#include "bkf_subscriber.h"
#include "bifrost_cncoi_slice.h"

#if __cplusplus
extern "C" {
#endif
#pragma pack(4)

/**
 * @brief 订阅者句柄
 */
typedef struct tagBifrostCncoiSuber BifrostCncoiSuber;

/**
 * @brief 订阅者初始化参数
 */
typedef struct tagBifrostCncoiSuberInitArg {
    char *name;               /**< 订阅者库名称 */
    uint16_t idlVersionMajor; /**< 格式化编码主要版本号 */
    uint16_t idlVersionMinor; /**< 格式化编码次要版本号 */
    BOOL dbgOn;               /**< puber端dbg开关，关闭则无日志输出 */
    BkfMemMng *memMng;
    BkfDisp *disp;
    BkfLog *log;
    BkfPfm *pfm;
    BkfXMap *xMap;
    BkfTmrMng *tmrMng;
    BkfJobMng *jobMng;
    uint32_t jobTypeId1;
    uint32_t jobPrioH;
    uint32_t jobPrioL;
    BkfSysLogMng *sysLogMng;
    BkfChCliMng *chCliMng; /**< 传输层客户端句柄 */
    uint16_t sliceKeyLen;  /** < 切片key结构长度 */
    uint8_t subMod; /**< 默认值0,默认模式，回调APP只有cookie和slice，1 增强模式，回调APP除了原有参数再增加两端url */
    uint8_t pad2[1];
    F_BKF_CMP sliceKeyCmp;              /** < 切片key比较接口 */
    F_BKF_GET_STR sliceKeyGetStrOrNull; /** < 切片key获取字符串接口 */
    F_BKF_DO sliceKeyCodec;             /** < 切片key进行IDL编码接口 */
    void *simpoBuilder;
    BkfUrl lsnUrl;
    uint8_t rsv1[0x10];
} BifrostCncoiSuberInitArg;

/**
 * @brief 订阅者库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 发布者库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BifrostCncoiSuber *BifrostCncoiSuberInit(BifrostCncoiSuberInitArg *arg);
/**
 * @brief 订阅者库销毁
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @return 无
 */
void BifrostCncoiSuberUninit(BifrostCncoiSuber *bifrostCncoiSuber);
/**
 * @brief 订阅者库使能，使能前无法正常通信
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @return 使能结果
 *   @retval BKF_OK 使能成功
 *   @retval 非BKF_OK 使能失败
 */
uint32_t BifrostCncoiSuberEnable(BifrostCncoiSuber *bifrostCncoiSuber);
/**
 * @brief 设置订阅者地址，用于订阅端发起连接
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] *url 地址
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberSetSelfUrl(BifrostCncoiSuber *bifrostCncoiSuber, BkfUrl *url);

/**
 * @brief 创建服务实例，用于指定实例发起订阅
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id，由信息中心分配或本地自行指定
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberCreateSvcInst(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId);

/**
 * @brief 创建服务实例，用于指定实例发起订阅
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id，由信息中心分配或本地自行指定
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberCreateSvcInstEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId);
/**
 * @brief 删除服务实例
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 无
 */
void BifrostCncoiSuberDeleteSvcInst(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId);

/**
 * @brief 删除服务实例
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 无
 */
void BifrostCncoiSuberDeleteSvcInstEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId);

/**
 * @brief 设置服务实例地址，即对端地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] *url 地址
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberSetSvcInstUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId, BkfUrl *puberUrl);

/**
 * @brief 设置服务实例地址，即对端地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] *url 地址
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberSetSvcInstUrlEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId, BkfUrl *puberUrl);

/**
 * @brief 设置服务实例地址，即本地地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] *url 本地地址
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */

uint32_t BifrostCncoiSuberSetSvcInstLocalUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId, BkfUrl *localUrl);

/**
 * @brief 删除服务实例puber地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 无
 */
void BifrostCncoiSuberUnsetSvcInstUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint32_t instId);

/**
 * @brief 删除服务实例puber地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 无
 */
void BifrostCncoiSuberUnsetSvcInstUrlEx(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId);

/**
 * @brief 删除服务实例local地址
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 无
 */
void BifrostCncoiSuberUnsetSvcInstLocalUrl(BifrostCncoiSuber *bifrostCncoiSuber, uint64_t instId);

/**
 * @brief 数据切片绑定到服务实例，为订阅数据切片做准备
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] *sliceKey 切片key
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberBindSliceInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint32_t instId);

/**
 * @brief 数据切片绑定到服务实例，为订阅数据切片做准备
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] *sliceKey 切片key
 * @return 设置结果
 *   @retval BKF_OK 设置成功
 *   @retval 非BKF_OK 设置失败
 */
uint32_t BifrostCncoiSuberBindSliceInstIdEx(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint64_t instId);

/**
 * @brief 去绑定数据切片和所有服务实例
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @return 无
 */
void BifrostCncoiSuberUnbindSliceInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey);

/**
 * @brief 去绑定数据切片和指定服务实例
 *
 * @param[in] *xxxSuber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @return 无
 */
void BifrostCncoiSuberUnbindSliceSpecInstId(BifrostCncoiSuber *bifrostCncoiSuber, BifrostCncoiSliceKeyT *sliceKey,
    uint64_t instId);

#pragma pack()

#if __cplusplus
}
#endif

#endif
