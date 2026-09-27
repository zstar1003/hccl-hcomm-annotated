/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_DC_H
#define BKF_DC_H

#include "bkf_comm.h"
#include "bkf_bas_type_mthd.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log.h"
#include "bkf_pfm.h"
#include "bkf_job.h"
#include "bkf_sys_log.h"

#ifdef __cplusplus
#if __cplusplus
extern "C" {
#endif
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */

/**
 * @brief dc库句柄
 */
typedef struct tagBkfDc BkfDc;

/* init */
/**
 * @brief dc切片虚表,切片是dc中数据的子集,dc库采用策略模式适配不同app的切片，因此需要注入虚表, 每个dc注入一个切片虚表
 */
typedef struct tagBkfDcSliceVTbl {
    uint16_t keyLen; /**< 切片key长度 */
    uint8_t pad1[2];
    F_BKF_CMP keyCmp;              /**< 切片key比较函数*/
    F_BKF_GET_STR keyGetStrOrNull; /**< 切片key获取字符串提示信息  */
    F_BKF_DO keyCodec;             /**< key的转码函数 */
    uint8_t rsv[0x10];
} BkfDcSliceVTbl;

/**
 * @brief 切片 key最大长度
 */
#define BKF_DC_SLICE_KEY_LEN_MAX (BKF_1K / 4)

typedef void (*F_BKF_DC_DOBEFORE)(void *cookie);
typedef void (*F_BKF_DC_DOAFTER)(void *cookie);

/**
 * @brief dc初始化参数
 */
typedef struct tagBkfDcInitArg {
    char *name;              /**< 库名称 */
    BOOL dbgOn;              /**< 诊断开关 */
    BkfMemMng *memMng;       /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;           /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLog *log;             /**< log库句柄,见bkf_log.h,同一app内可复用 */
    BkfPfm *pfm;             /**< 性能测量库句柄,见bkf_pfm.h,同一app内可复用 */
    BkfJobMng *jobMng;       /**< job库句柄,见bkf_job.h,同一app内可复用 */
    uint32_t jobTypeId;      /**< dc库使用的job类型id,不能与其他库复用 */
    uint32_t jobPrio;        /**< dc库使用的job优先级 */
    BkfSysLogMng *sysLogMng; /**< 系统日志句柄，见bkf_sys_log.h,同一app内可复用 */

    BkfDcSliceVTbl sliceVTbl; /**< 切片虚表 */
    void *cookie;
    F_BKF_DC_DOBEFORE beforeProc;
    F_BKF_DC_DOAFTER afterProc;
    uint8_t rsv[0x10];
} BkfDcInitArg;

/**
 * @brief 获取切片虚表
 */
#define BKF_DC_GET_SLICE_VTBL(dc) (&((BkfDcInitArg *)(dc))->sliceVTbl)

/**
 * @brief 业务注册钩子函数，get指定表项第一个记录
 *
 * @param[in] *cookieReg 回调cookie
 * @param[in]  type  获取表项类型
 * @param[out]  outputKey 输出key缓存，bkf保证指针内存区大于业务注册长度
 * @param[out]  outputTupleVal 输出数据val缓存，bkf保证指针内存区大于业务注册长度
 * @return 0 OK
         其他 FAIL 获取数据失败
 */
typedef uint32_t (*F_BKF_DC_GET_FIRSTTUPLE)(void *cookie, uint32_t type, void *outputKey, void *outputTupleVal);

/**
 * @brief 业务注册钩子函数，get指定表项next记录
 *
 * @param[in] *cookieReg 回调cookie
 * @param[in]  type  获取表项类型
 * @param[in]  lastKey 上一条记录key，根据该key获取输出的next 记录key
 * @param[out]  outputKey 输出key缓存，bkf保证指针内存区大于业务注册长度
 * @param[out]  outputTupleVal 输出数据val缓存，bkf保证指针内存区大于业务注册长度
 * @return 0 OK
         其他 FAIL 获取数据失败
 */
typedef uint32_t (*F_BKF_DC_GET_NEXTTUPLE)(void *cookie, uint32_t type, void *lastKey, void *outputKey,
    void *outputTupleVal);

/* reg type */
/**
 * @brief
 * 表类型虚表，表类型指定的是dc中包含的数据类型，dc用策略模式处理不同类型数据，因此每种表类型都要注入一个虚表，用于处理某种数据类型
 */
typedef struct tagBkfDcTableTypeVTbl {
    char *name;           /**< 虚表名称 */
    uint16_t tableTypeId; /**< 表类型 */
    uint8_t noDcTuple;    /* 没有dc数据，依赖回调拉数据，实时变化处理完之后释放内存 */
    uint8_t pad[1];
    void *cookie;                       /**< 回调cookie */
    int32_t tupleCntMax;                /**< tuple最大数量 */
    uint16_t tupleKeyLen;               /**< tuple key长度 */
    uint16_t tupleValLen;               /**< tuple val长度 */
    F_BKF_CMP tupleKeyCmp;              /**< tuple key比较接口 */
    F_BKF_GET_STR tupleKeyGetStrOrNull; /**< tuple key获取诊断字符串接口 */
    F_BKF_GET_STR tupleValGetStrOrNull; /**< tuple val获取诊断字符串接口 */
    F_BKF_DC_GET_FIRSTTUPLE getFirst;
    F_BKF_DC_GET_NEXTTUPLE getNext;
} BkfDcTableTypeVTbl;

/**
 * @brief tuple key最大长度
 */
#define BKF_DC_TUPLE_KEY_LEN_MAX (BKF_1K / 2)

/* get */
/**
 * @brief 元组信息，用于业务获取元组,元组是dc中保存数据的原子单位，切片 + 表类型 = 表实例，每个表实例下可以有若干元组
 */
typedef struct tagBkfDcTupleInfo {
    void *key;       /**< key指针 */
    BOOL isAddUpd;   /**< 删除标记 */
    void *valOrNull; /**< val指针,实际与key指针指向连续空间 */
    uint64_t seq;    /**< 数据版本号 */
} BkfDcTupleInfo;

/* func */
/**
 * @brief dc库初始化
 *
 * @param[in] *arg 初始化参数
 * @return dc库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfDc *BkfDcInit(BkfDcInitArg *arg);

/**
 * @brief dc库反初始化
 *
 * @param[in] *dc dc库句柄
 * @return none
 */
void BkfDcUninit(BkfDc *dc);

/* table type */
/**
 * @brief dc库注册表类型
 *
 * @param[in] *dc dc库句柄
 * @param[in] *vTbl 注册参数
 * @return 注册结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcRegTableType(BkfDc *dc, BkfDcTableTypeVTbl *vTbl);

/* table type */
/**
 * @brief dc库去注册表类型
 *
 * @param[in] *dc dc库句柄
 * @param[in] tabTypeId 表id
 * @return none
 */
void BkfDcUnregTableType(BkfDc *dc, uint16_t tabTypeId);

/**
 * @brief dc库通知表类型完整
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型
 * @return 注册结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcNotifyTableTypeComplete(BkfDc *dc, uint16_t tableTypeId);

/**
 * @brief dc重置表完整
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型
 */

void BkfDcResetTableTypeComplete(BkfDc *dc, uint16_t tableTypeId);

/**
 * @brief dc库获取表类型完整
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型
 * @return 注册结果
 *   @retval true 完整
 *   @retval false 不完整
 */
BOOL BkfDcIsTableTypeComplete(BkfDc *dc, uint16_t tableTypeId);

/**
 * @brief 获取dc库注册过的表类型id字串诊断信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型id
 * @return 字串
 *   @retval 非空 字串信息。不会返回空，对非法的情况会返回一个常量串。
 */
const char *BkfDcGetTableTypeIdStr(BkfDc *dc, uint16_t tableTypeId);

/* slice */
/**
 * @brief 创建切片
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @return 创建结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcCreateSlice(BkfDc *dc, void *sliceKey);

/**
 * @brief 删除切片
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @return 无
 */
void BkfDcDeleteSlice(BkfDc *dc, void *sliceKey);

/**
 * @brief 切片是否存在
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @return 切片是否存在
 *   @retval true 存在
 *   @retval false 不存在
 */
BOOL BkfDcIsSliceExist(BkfDc *dc, void *sliceKey);

/**
 * @brief 获取切片的诊断信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] *buf 输出buf
 * @param[in] bufLen 输出buf长度
 * @return 诊断信息字串
 *   @retval 非空 诊断信息字串。不会返回空，对非法的情况会返回一个常量串。
 */
char *BkfDcGetSliceKeyStr(BkfDc *dc, void *sliceKey, uint8_t *buf, int32_t bufLen);

/* table */
/**
 * @brief 创建表
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @return 创建结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcCreateTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId);

/**
 * @brief 释放表
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @return 无
 * @attention 释放表，一方面会通知注册的迭代器；另一方面等待上层调用删元组和删表。
 * @see BkfDcNewTupleSeqItor
 * @see BkfDcDeleteTuple @see BkfDcDeleteTable
 */
void BkfDcReleaseTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId);

/**
 * @brief 删表
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @return 无
 * @attention 真正删除表。如果此表有剩余元组，会一并删除。
 */
void BkfDcDeleteTable(BkfDc *dc, void *sliceKey, uint16_t tableTypeId);

/**
 * @brief 查询表是否存在
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *isTableReleaseOrNull 输出表是否release状态。如果为空，不输出。
 * @return 表是否存在
 *   @retval true 存在, 不管表状态
 *   @retval false 不存在
 */
BOOL BkfDcIsTableExist(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, BOOL *isTableReleaseOrNull);

/* tuple */
/**
 * @brief 创建/更新元组
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @param[in] *tupleValOrNull 元组value。可以为空，对应注册时valLen填0
 * @param[out] *infoOrNull Tuple 元组的更新结果
 * @return 创建结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfDcUpdateTuple(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey, void *tupleValOrNull,
    BkfDcTupleInfo *infoOrNull);

/**
 * @brief 删除元组
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @return 无
 */
void BkfDcDeleteTuple(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey);

/**
 * @brief 获取一个元组信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @param[in] *info 输出元组信息
 * @return 是否存在
 *   @retval true 获取成功
 *   @retval false 获取失败，不存在
 */
BOOL BkfDcGetTupleInfo(BkfDc *dc, void *sliceKey, uint16_t tableTypeId, void *tupleKey, BkfDcTupleInfo *info);

/**
 * @brief 获取元组key的诊断信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @param[in] *buf 输出buf
 * @param[in] bufLen 输出buf长度
 * @return 诊断信息字串
 *   @retval 非空 诊断信息字串。不会返回空，对非法的情况会返回一个常量串。
 */
char *BkfDcGetTupleKeyStr(BkfDc *dc, uint16_t tableTypeId, void *tupleKey, uint8_t *buf, int32_t bufLen);

/**
 * @brief 获取元组val的诊断信息
 *
 * @param[in] *dc dc库句柄
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleVal 元组value
 * @param[in] *buf 输出buf
 * @param[in] bufLen 输出buf长度
 * @return 诊断信息字串
 *   @retval 非空 诊断信息字串。不会返回空，对非法的情况会返回一个常量串。
 */
char *BkfDcGetTupleValStr(BkfDc *dc, uint16_t tableTypeId, void *tupleVal, uint8_t *buf, int32_t bufLen);

/**
 * @brief 设置spy的cookie。用于测试。
 *
 * @param[in] *dc dc库句柄
 * @param[in] *cookie cookie
 * @return 无
 * @note 每dc只支持单cookie
 */
void BkfDcEnableSpy(BkfDc *dc, void *cookie);

/**
 * @brief spy元组更新。用于测试
 *
 * @param[in] *cookieOfSpy 需要传入dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @param[in] *tupleValOrNull 元组value。可以为空，对应注册时valLen填0
 * @return 无
 */
void BkfDcSpyUpdateTuple(void *cookieOfSpy, void *sliceKey, uint16_t tableTypeId, void *tupleKey, void *tupleValOrNull);

/**
 * @brief spy元组删除。用于测试
 *
 * @param[in] *cookieOfSpy 需要传入dc库句柄
 * @param[in] *sliceKey 切片key
 * @param[in] tableTypeId 表类型
 * @param[in] *tupleKey 元组key
 * @return 无
 */
void BkfDcSpyDeleteTuple(void *cookieOfSpy, void *sliceKey, uint16_t tableTypeId, void *tupleKey);

#pragma pack()
#pragma GCC visibility pop
#ifdef __cplusplus
#if __cplusplus
}
#endif
#endif

#endif
