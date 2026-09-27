/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_SUBSCRIBER_H
#define BKF_SUBSCRIBER_H

#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_job.h"
#include "bkf_tmr.h"
#include "bkf_url.h"
#include "bkf_xmap.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_sys_log.h"
#include "bkf_ch_cli.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief 订阅者句柄
 */
typedef struct tagBkfSuber BkfSuber;

/**
 * @brief  0默认模式，回调APP只有cookie和slice，1 增强模式，回调APP除了原有参数再增加两端url
 */
#define BKF_SUBER_MODE_DEFAULT 0
#define BKF_SUBER_MODE_EX 1

/* suber会话断连原因枚举 */
enum {
    BKF_SUBER_DISCONNECT_REASON_LOCALFIN = 0,
    BKF_SUBER_DISCONNECT_REASON_PEERFIN = 1,
    BKF_SUBER_DISCONNECT_REASON_PEERERROR = 2,
    BKF_SUBER_DISCONNECT_REASON_CONNECTFAIL = 3,
    BKF_SUBER_DISCONNECT_REASON_MAX
};

typedef void (*F_SUBER_ON_CONNECT)(void *cookie, BkfUrl *connDestUrl, BkfUrl *connSrcUrl, uint16_t srcPort);

typedef void (*F_SUBER_ON_DISCONNECT)(void *cookie, BkfUrl *connDestUrl, BkfUrl *connSrcUrl, uint32_t disReason);

/* init */
/**
 * @brief 订阅者初始化参数
 */
typedef struct tagBkfSuberInitArg {
    char *name;               /**< 模块名。不要有空格 */
    char *svcName;            /**< 服务名 */
    uint16_t idlVersionMajor; /**< 主版本号 */
    uint16_t idlVersionMinor; /**< 次要版本号 */
    BOOL dbgOn;               /**< 诊断开关 */
    BkfMemMng *memMng;        /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;            /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLog *log;              /**< log库句柄,见bkf_log.h,同一app内可复用 */
    BkfPfm *pfm;              /**< 性能测量库句柄,见bkf_pfm.h,同一app内可复用 */
    BkfXMap
        *xMap; /**< 消息分发xmap句柄,见bkf_xmap.h,用于app将dms消息分发给bkf,配合BKF_MSG_DISPATCH使用,同一app内可复用 */
    BkfTmrMng *tmrMng; /**< 定时器管理句柄,见bkf_tmr.h,同一app内可复用 */
    BkfJobMng *jobMng; /**< job管理句柄,同一app内可复用,可能要配合xmap句柄使用,详见见bkf_job.h */
    BkfChCliMng *chCliMng;   /**< 传输层客户端句柄，见bkf_ch_ser.h,suber实例独有，一般不可复用 */
    uint32_t jobTypeId1;     /**< 发布者库使用的job类型id,不能与其他库复用 */
    uint32_t jobPrioH;       /**< 发布者库使用的job优先级:高优先级 */
    uint32_t jobPrioL;       /**< 发布者库使用的job优先级:低优先级 */
    BkfSysLogMng *sysLogMng; /**< 系统日志句柄，见bkf_sys_log.h,同一app内可复用 */

    uint16_t sliceKeyLen; /**< 切片key长度 */
    uint8_t subMod; /**< 默认值0,默认模式，回调APP只有cookie和slice，1 增强模式，回调APP除了原有参数再增加两端url */
    uint8_t hasLsnUrl : 1;
    uint8_t subConnState : 1;
    uint8_t pad2 : 6;
    F_BKF_CMP sliceKeyCmp;              /**< 切片key比较接口 */
    F_BKF_GET_STR sliceKeyGetStrOrNull; /**< 切片key获取定位诊断字符串接口 */
    F_BKF_DO sliceKeyCodec;             /**< 切片key编码接口 */
    BkfUrl lsnUrl;
    uint16_t reconnectDelayTime; /**<连接建立超时时间,单位s,不配置或填0则默认 5s同原先时间            */
    uint16_t pad3;
    void *cookie;
    F_SUBER_ON_CONNECT onConn;
    F_SUBER_ON_DISCONNECT onDisConn;
    uint8_t rsv1[0x10];
} BkfSuberInitArg;

/**
 * @brief 发起订阅参数
 */
typedef struct tagBkfSuberSubArg {
    void *sliceKey;       /**< 切片key */
    uint16_t tableTypeId; /**< 表类型id */
    uint8_t isVerify : 1; /**< 是否为对账订阅 */
    uint8_t rsv1 : 7;
    uint8_t reserv;
} BkfSuberSubArg;

/**
 * @brief 发起扩展型订阅参数
 */
typedef struct tagBkfSuberSubArgEx {
    void *sliceKey;       /**< 切片key */
    uint16_t tableTypeId; /**< 表类型id */
    uint8_t isVerify : 1; /**< 是否为对账订阅 */
    uint8_t rsv1 : 7;
    uint8_t reserv;
    uint64_t instId; /* 指定instId进行订阅or对账请求  */
} BkfSuberSubArgEx;

/**
 * @brief 基于表通知app虚函数
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @return None
 */
typedef void (*F_SUBER_ON_TABLE)(void *cookieOfRegister, void *sliceKey);

/**
 * @brief 基于数据通知app虚函数
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *data 数据指针
 * @param[in] len 数据长度
 * @param[in] dataNeedDecode 数据是否需要解码后使用
 * @return None
 */
typedef void (*F_SUBER_ON_DATA)(void *cookieOfRegister, void *sliceKey, void *data, int32_t len, BOOL dataNeedDecode);

/**
 * @brief 基于表通知app平滑超时
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @return None
 */
typedef void (*F_SUBER_ON_BATCH_TIMEOUT)(void *cookieOfRegister, void *sliceKey);

/**
 * @brief 基于表通知app虚函数
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *puber端url
 * @param[in] *本地url
 * @return None
 */
typedef void (*F_SUBER_ON_TABLEEX)(void *cookieOfRegister, void *sliceKey, BkfUrl *puberUrl, BkfUrl *localUrl);

/**
 * @brief 基于表通知app断开连接虚函数
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *puber端url
 * @param[in] *本地url
 * @param[in] 断连原因码
 * @return None
 */
typedef void (*F_SUBER_ON_TABLEDISCONNECTEX)(void *cookieOfRegister, void *sliceKey, BkfUrl *puberUrl, BkfUrl *localUrl,
    uint32_t reasonCode);

/**
 * @brief 基于数据通知app虚函数
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *data 数据指针
 * @param[in] len 数据长度
 * @param[in] dataNeedDecode 数据是否需要解码后使用
 * @param[in] *puber端url
 * @param[in] *本地url
 * @return None
 */
typedef void (*F_SUBER_ON_DATAEX)(void *cookieOfRegister, void *sliceKey, void *data, int32_t len, BOOL dataNeedDecode,
    BkfUrl *puberUrl, BkfUrl *localUrl);

/**
 * @brief 基于表通知app平滑超时
 *
 * @param[in] *cookieOfRegister 回调cookie
 * @param[in] *sliceKey 切片key
 * @param[in] *puber端url
 * @param[in] *本地url
 * @return None
 */
typedef void (*F_SUBER_ON_BATCH_TIMEOUTEX)(void *cookieOfRegister, void *sliceKey, BkfUrl *puberUrl, BkfUrl *localUrl);

/**
 * @brief 订阅者表类型处理虚表
 */
typedef struct tagBkfSuberTableTypeVTbl {
    char *name;                              /**< 虚表名称 */
    uint16_t tableTypeId;                    /**< 表类型id */
    uint16_t batchTimeout;                   /**< 平滑超时时间，设置为0则不生效 */
    BOOL needTableComplete;                  /**< 是否需要发布者表完整后才开始批备 */
    void *cookie;                            /**< 回调cookie */
    F_SUBER_ON_TABLE onTableBatchBegin;      /**< 表实例批备开始 */
    F_SUBER_ON_TABLE onTableBatchEnd;        /**< 表实例批备结束 */
    F_SUBER_ON_TABLE onTableVerifyBegin;     /**< 表实例对账开始 */
    F_SUBER_ON_TABLE onTableVerifyEnd;       /**< 表实例对账结束 */
    F_SUBER_ON_TABLE onTableDelete;          /**< 表实例删除 */
    F_SUBER_ON_TABLE onTableDisconn;         /**< 表实例连接中断 */
    F_SUBER_ON_DATA onDataUpdate;            /**< 数据元组更新 */
    F_SUBER_ON_DATA onDataDelete;            /**< 数据元组删除 */
    F_SUBER_ON_BATCH_TIMEOUT onBatchTimeout; /**< 平滑超时通知 */
} BkfSuberTableTypeVTbl;

/**
 * @brief 订阅者表类型处理扩展虚表
 */
typedef struct tagBkfSuberTableTypeVTblEx {
    char *name;                                    /**< 虚表名称 */
    uint16_t tableTypeId;                          /**< 表类型id */
    uint16_t batchTimeout;                         /**< 平滑超时时间，设置为0则不生效 */
    BOOL needTableComplete;                        /**< 是否需要发布者表完整后才开始批备 */
    void *cookie;                                  /**< 回调cookie */
    F_SUBER_ON_TABLEEX onTableBatchBeginEx;        /**< 表实例批备开始,扩展两端url */
    F_SUBER_ON_TABLEEX onTableBatchEndEx;          /**< 表实例批备结束,扩展两端url */
    F_SUBER_ON_TABLEEX onTableVerifyBeginEx;       /**< 表实例对账开始,扩展两端url */
    F_SUBER_ON_TABLEEX onTableVerifyEndEx;         /**< 表实例对账结束,扩展两端url */
    F_SUBER_ON_TABLEEX onTableDeleteEx;            /**< 表实例删除,扩展两端url */
    F_SUBER_ON_TABLEDISCONNECTEX onTableDisconnEx; /**< 表实例连接中断,扩展两端url */
    F_SUBER_ON_DATAEX onDataUpdateEx;              /**< 数据元组更新,扩展两端url */
    F_SUBER_ON_DATAEX onDataDeleteEx;              /**< 数据元组删除,扩展两端url */
    F_SUBER_ON_BATCH_TIMEOUTEX onBatchTimeoutEx;   /**< 平滑超时通知,扩展两端url */
} BkfSuberTableTypeVTblEx;
#pragma pack()

/* func */
/**
 * @brief 订阅者库初始化
 *
 * @param[in] *arg 初始化参数
 * @return 订阅者库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfSuber *BkfSuberInit(BkfSuberInitArg *arg);

/**
 * @brief 订阅者库反初始化
 *
 * @param[in] *suber 订阅者库句柄
 * @return none
 */
void BkfSuberUninit(BkfSuber *suber);

/**
 * @brief 注册表类型
 *
 * @param[in] *suber 订阅者管理库句柄
 * @param[in] *vTbl 表类型的虚表
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */
uint32_t BkfSuberRegisterTableTypeEx(BkfSuber *suber, BkfSuberTableTypeVTbl *vTbl, void *userData,
    uint16_t userDataLen);

/**
 * @brief 注册增强型表类型,回调APP携带两端url
 *
 * @param[in] *suber 订阅者管理库句柄
 * @param[in] *vTbl 增强型表类型的虚表
 * @return 注册结果
 *   @retval BKF_OK 注册成功
 *   @retval 其他 注册失败
 */

uint32_t BkfSuberRegisterTableTypeExVTbl(BkfSuber *suber, BkfSuberTableTypeVTblEx *vTbl, void *userData,
    uint16_t userDataLen);

/**
 * @brief 订阅者库获取业务数据
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] tableTypeId 数据类型
 * @return 业务数据
 *   @retval 非空 成功
 *   @retval 空 失败
 */
void *BkfSuberGetTableTypeUserData(BkfSuber *suber, uint16_t tableTypeId);

/**
 * @brief 去注册表类型
 *
 * @param[in] *suber 订阅者管理库句柄
 * @param[in] tabType 表类型ID
 * @return 注册结果
 *   @retval BKF_OK 去注册成功
 *   @retval 其他 去注册失败
 */
uint32_t BkfSuberUnRegisterTableType(BkfSuber *suber, uint16_t tabType);

/**
 * @brief 获取表类型的虚表
 *
 * @param[in] *suber 订阅者管理库句柄
 * @param[in] tableTypeId 表类型id
 * @return 表类型的虚表
 *   @retval 非空 获取成功
 *   @retval 其他 获取失败
 */
BkfSuberTableTypeVTbl *BkfSuberGetTableTypeVTbl(BkfSuber *suber, uint16_t tableTypeId);

/**
 * @brief 订阅者库使能
 *
 * @param[in] *suber 订阅者库句柄
 * @return 使能状态
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberEnable(BkfSuber *suber);

/**
 * @brief 描述：设置本地地址
 *
 * @param suber指针
 * @param url本端地址
 * @return 成功失败
 */
uint32_t BkfSuberSetSelfUrl(BkfSuber *suber, BkfUrl *selfUrl);

/**
 * @brief 描述：去设置本地地址
 *
 * @param suber指针
 * @param selfUrl对应地址
 * @return 成功失败
 */
void BkfSuberUnSetSelfUrl(BkfSuber *suber, BkfUrl *selfUrl);

/**
 * @brief 创建发布者服务实例
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 服务实例创建结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberCreateSvcInst(BkfSuber *suber, uint32_t instId);

/**
 * @brief 创建发布者服务实例，支持64位instId
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 服务实例创建结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberCreateSvcInstEx(BkfSuber *suber, uint64_t instId);

/**
 * @brief 删除发布者服务实例
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return none
 */
void BkfSuberDeleteSvcInst(BkfSuber *suber, uint32_t instId);

/**
 * @brief 删除发布者服务实例，支持64位instId
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return none
 */
void BkfSuberDeleteSvcInstEx(BkfSuber *suber, uint64_t instId);

/**
 * @brief 设置发布者服务实例远端url
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] url puber url
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSetSvcInstUrl(BkfSuber *suber, uint32_t instId, BkfUrl *puberUrl);

/**
 * @brief 设置发布者服务实例远端url增强API，支持64位instId
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] url puber url
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSetSvcInstUrlEx(BkfSuber *suber, uint64_t instId, BkfUrl *puberUrl);

/**
 * @brief
 * 设置发布者服务实例远端url和本地url,本地url必须在puberurl之前配置，puberurl会触发建联,已有puberurl，再修改localurl会先拆后建新连接
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @param[in] puberUrl 远端url
 * @param[in] localUrl  本地url
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSetSvcInstLocalUrl(BkfSuber *suber, uint64_t instId, BkfUrl *localUrl);

/**
 * @brief 去设置发布者服务实例远端url
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberUnSetSvcInstUrl(BkfSuber *suber, uint32_t instId);

/**
 * @brief 去设置发布者服务实例远端url,支持64位实例ID
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] instId 实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberUnSetSvcInstUrlEx(BkfSuber *suber, uint64_t instId);

/**
 * @brief 去设置发布者服务实例localurl，支持64位实例id
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in]  instId 实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberUnSetSvcInstLocalUrl(BkfSuber *suber, uint64_t instId);

/**
 * @brief 设置切片归属的服务实例
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] instId 实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSetSliceInstId(BkfSuber *suber, void *sliceKey, uint32_t instId);

/**
 * @brief 设置切片归属的服务实例，支持64位实例id
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] instId 实例id
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSetSliceInstIdEx(BkfSuber *suber, void *sliceKey, uint64_t instId);

/**
 * @brief 清除切片归属的服务所有inst实例,删除一个inst实例请使用BkfSuberUnsetSliceSpecInstId
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @return none
 */
void BkfSuberUnsetSliceInstId(BkfSuber *suber, void *sliceKey);

/**
 * @brief 清除切片指定的服务实例
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] instId 实例ID
 * @return none
 */

void BkfSuberUnsetSliceSpecInstId(BkfSuber *suber, void *sliceKey, uint64_t instId);

/**
 * @brief 订阅表，普通模式只支持slice拥有唯一inst
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *subArg 订阅参数（包含切片Key、tableTypeId、tblStateFlag）
 * @return 订阅结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSub(BkfSuber *suber, BkfSuberSubArg *subArg);

/**
 * @brief 去订阅表
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *subArg 订阅参数（包含切片Key、tableTypeId、tblStateFlag）
 * @return none
 */
void BkfSuberUnsub(BkfSuber *suber, BkfSuberSubArg *subArg);

/**
 * @brief 指定instId进行订阅or对账请求
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *subArg 订阅参数（包含切片Key、tableTypeId、tblStateFlag）
 * @return 订阅结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfSuberSubEx(BkfSuber *suber, BkfSuberSubArgEx *subArg);

/**
 * @brief 指定inst取消订阅or去对账请求
 *
 * @param[in] *suber 订阅者库句柄
 * @param[in] *subArg 订阅参数（包含切片Key、tableTypeId、tblStateFlag）
 * @return none
 */
void BkfSuberUnsubEx(BkfSuber *suber, BkfSuberSubArgEx *subArg);

#pragma GCC visibility pop
#define BKF_SUBER_CODE_BUF_NOT_ENOUGH (0x7fffffff) /* 缓冲不够 */
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
