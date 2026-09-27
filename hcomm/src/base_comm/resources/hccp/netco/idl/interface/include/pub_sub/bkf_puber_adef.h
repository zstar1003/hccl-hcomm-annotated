/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_PUBER_ADEF_H
#define BKF_PUBER_ADEF_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_xmap.h"
#include "bkf_tmr.h"
#include "bkf_job.h"
#include "bkf_url.h"
#include "bkf_dc.h"
#include "bkf_ch_ser_adef.h"
#include "bkf_sys_log.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/**
 * @brief 发布者句柄
 */
typedef struct TagBkfPuber BkfPuber;

typedef void (*F_PUBER_ON_CONNECT)(void *cookie, BkfUrl *connDestUrl, BkfUrl *connSrcUrl, BkfUrl *peerLsnUrl);

typedef void (*F_PUBER_ON_DISCONNECT)(void *cookie, BkfUrl *connDestUrl, BkfUrl *connSrcUrl);

typedef void (*F_PUBER_ON_CONNECTOVERLOAD)(void *cookie, BkfUrl *connSrcUrl);
/**
 * @brief 生产者判断是否加速对账虚接口
 * @param[in] *cookieInit 回调参数
 * @return 是否加速
 *   @retval TRUE 加速发送对账
 *   @retval FALSE 减速发送对账
 */
typedef BOOL (*F_BKF_PUBER_VERIFY_MAY_ACCELERATE)(void *cookieInit);

typedef struct {
    char *name;               /**< 模块名。不要有空格 */
    char *svcName;            /**< 服务名 */
    uint16_t idlVersionMajor; /**< 主版本号 */
    uint16_t idlVersionMinor; /**< 次要版本号 */
    BOOL dbgOn;               /**< 诊断开关 */
    BkfMemMng *memMng;        /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;            /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLog *log;              /**< log库句柄,见bkf_log.h,同一app内可复用 */
    BkfPfm *pfm;              /**< 性能测量库句柄,见bkf_pfm.h,同一app内可复用 */
    BkfXMap *xMap; /**< 消息分发xmap句柄,见bkf_xmap.h,用于app将dms消息分发给bkf,同一app内可复用 */
    BkfTmrMng *tmrMng;  /**< 定时器管理句柄,见bkf_tmr.h,同一app内可复用 */
    BkfJobMng *jobMng;  /**< job管理句柄,同一app内可复用,可能要配合xmap句柄使用,详见见bkf_job.h */
    uint32_t jobTypeId; /**< 发布者库使用的job类型id,不能与其他库复用 */
    uint32_t jobPrio;   /**< 发布者库使用的job优先级 */
    void *cookie;       /**< 回调cookie */
    F_BKF_PUBER_VERIFY_MAY_ACCELERATE verifyMayAccelerate; /**< 是否对账加速接口 */
    BkfSysLogMng *sysLogMng; /**< 系统日志句柄，见bkf_sys_log.h,同一app内可复用 */
    BkfDc *dc;               /**< dc句柄，见bkf_dc.h,同一app内可复用 */
    BkfChSerMng *chMng;      /**< 传输层服务端句柄，见bkf_ch_ser.h,puber实例独有，一般不可复用 */
    int32_t connCntMax;      /**< 连接最大数量限制 */
    uint64_t xSeed;          /**< 连接初始事务号，初始化为0即可 */
    F_PUBER_ON_CONNECT onConnect;             /**<业务如果关注连接UP可以注册本接口 */
    F_PUBER_ON_DISCONNECT onDisConnect;       /**<业务如果关注连接DOWN可以注册本接口 */
    F_PUBER_ON_CONNECTOVERLOAD onConnectOver; /**<业务如果需要感知连接超限被拒绝可以注册本接口 */
    uint8_t rsv1[0x10];
} BkfPuberInitArg;

/**
 * @brief 编码时缓冲不足，见F_BKF_PUBER_TUPLE_UPDATE_CODE与F_BKF_PUBER_TUPLE_DELETE_CODE接口
 */
#define BKF_PUBER_CODE_BUF_NOT_ENOUGH (0x7fffffff)

/**
 * @brief 元组更新消息编码虚接口
 * @param[in] *cookieReg 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *tupleKey 元组key
 * @param[in] *tupleVal 元组val
 * @param[in] *buf 码流写入的缓冲区
 * @param[in] bufLen 缓冲区长度
 * @return 编码后的码流长度
 *   @retval <=0 错误
 *   @retval BKF_PUBER_CODE_BUF_NOT_ENOUGH 缓冲不够
 *   @retval >0 写入缓冲区的长额度
 */
typedef int32_t (*F_BKF_PUBER_TUPLE_UPDATE_CODE)(void *cookieReg, void *sliceKey, void *tupleKey, void *tupleVal,
    void *buf, int32_t bufLen);

/**
 * @brief 元组删除消息编码虚接口
 * @param[in] *cookieReg 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *tupleKey 元组key
 * @param[in] *tupleVal 元组val
 * @param[in] *buf 码流写入的缓冲区
 * @param[in] bufLen 缓冲区长度
 * @return 编码后的码流长度
 *   @retval <=0 错误
 *   @retval BKF_PUBER_CODE_BUF_NOT_ENOUGH 缓冲不够
 *   @retval >0 写入缓冲区的长额度
 */
typedef int32_t (*F_BKF_PUBER_TUPLE_DELETE_CODE)(void *cookieReg, void *sliceKey, void *tupleKey, void *buf,
    int32_t bufLen);

/**
 * @brief 表订阅通知
 * @param[in] *cookieReg 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *usrData 业务数据
 * @param[in] *len 业务数据长度
 * @return none
 */
typedef void (*F_BKF_PUBER_TABLE_ONSUB)(void *cookieReg, void *sliceKey, void *usrData, int32_t len);

/**
 * @brief 表去订阅通知
 * @param[in] *cookieReg 回调cookie
 * @param[in] *sliceKey 切片key
 * @return none
 */
typedef void (*F_BKF_PUBER_TABLE_ONUNSUB)(void *cookieReg, void *sliceKey);

/**
 * @brief 基于表通知app平滑超时
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @return None
 */
typedef void (*F_BKF_PUBER_TABLE_ON_BATCH_TIMEOUT)(void *cookieOfRegister, void *sliceKey);

/**
 * @brief 表类型处理虚表，业务不同类型的表数据，注册不同的编码回调
 */
typedef struct {
    uint16_t tableTypeId;                                   /**< 表id */
    uint16_t batchTimeout;                                  /**< 平滑超时时间，设置为0则不生效 */
    void *cookie;                                           /**< 回调cookie */
    F_BKF_PUBER_TUPLE_UPDATE_CODE tupleUpdateCode;          /**< 元组更新编码接口 */
    F_BKF_PUBER_TUPLE_DELETE_CODE tupleDeleteCode;          /**< 元组删除编码接口 */
    F_BKF_PUBER_TABLE_ONSUB tableOnSub;                     /**< 表订阅通知 */
    F_BKF_PUBER_TABLE_ONUNSUB tableOnUnsub;                 /**< 表去订阅通知 */
    F_BKF_PUBER_TABLE_ON_BATCH_TIMEOUT tableOnBatchTimeout; /**< 平滑超时通知 */
} BkfPuberTableTypeVTbl;
#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif

#endif
