/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BKF_LOG_H
#define BKF_LOG_H

#include "bkf_comm.h"
#include "bkf_mem.h"
#include "bkf_disp.h"
#include "bkf_log_comm.h"
#include "bkf_log_cnt.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma GCC visibility push(default)
#pragma pack(4)
/* common */
/**
 * @brief log库句柄,用于输出私有日志,输出文件的位置由outputOrNull实现决定
 */
typedef struct tagBkfLog BkfLog;

/**
 * @brief log输出到内存时，允许占用的最大长度，单位：字节
 */
#define BKF_LOG_MEM_BUF_LEN_MAX (BKF_1M * 100)

/* init */
/**
 * @brief log输出函数原型
 */
typedef void (*F_BKF_LOG_OUTPUT)(void *cookieInit, const char *outStr);

/* debug输出函数原型 */
typedef void (*F_BKF_DEBUG_OUTPUT)(void *cookieInit, const char *outStr);

/**
 * @brief log级别
 */
enum {
    BKF_LOG_LVL_DEBUG,
    BKF_LOG_LVL_INFO,
    BKF_LOG_LVL_WARN,
    BKF_LOG_LVL_ERROR,

    BKF_LOG_LVL_CNT
};

/**
 * @brief 检查log级别参数是否合法
 */
#define BKF_LOG_LVL_IS_VALID(lvl) ((lvl) < BKF_LOG_LVL_CNT)

/**
 * @brief log库初始化参数
 */
typedef struct tagBkfLogInitArg {
    char *name;                    /**< 名称 */
    BkfMemMng *memMng;             /**< 内存库句柄,见bkf_mem.h,同一app内可复用 */
    BkfDisp *disp;                 /**< 诊断显示库句柄,见bkf_disp.h,同一app内可复用 */
    BkfLogCnt *logCnt;             /**< log cnt库句柄,见bkf_log_cnt.h,日志记录次数，同一app内可复用 */
    void *cookie;                  /**< log输出接口回调cookie */
    F_BKF_LOG_OUTPUT outputOrNull; /**< 输出函数，可以为空 */
    BOOL outputEnable;             /**< output使能开关 */
    BOOL logFuncName;              /**< 是否允许日志中输出函数名 */
    BOOL logTime;                  /**< 是否允许日志中输出时间 */
    uint8_t moduleDefalutLvl;      /**< 模块默认的log level */
    uint8_t pad1[3];
    int32_t memBufLen; /**< 日志输出到内存允许占用的空间 */
    uint8_t rsv[0x10];
} BkfLogInitArg;

/* func */
/**
 * @brief log库初始化
 *
 * @param[in] *arg 参数
 * @return log库句柄
 *   @retval 非空 创建成功
 *   @retval 空 创建失败
 */
BkfLog *BkfLogInit(BkfLogInitArg *arg);

/**
 * @brief log库反初始化
 *
 * @param[in] *log log库句柄
 * @return none
 */
void BkfLogUninit(BkfLog *log);

/**
 * @brief 设置模块的log level。不设置的情况下，为log库初始化时候默认值。 @see struct tagBkfLogInitArg::moduleDefalutLvl
 *
 * @param[in] *log log库句柄
 * @param[in] *modName 模块名。一般在一份文件的开头定义一个模块名。 @see BKF_MOD_NAME
 * @param[in] lvl 模块的log level。
 * @return 设置结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfLogSetModuleLvl(BkfLog *log, char *modName, uint8_t lvl);

/**
 * @brief log前半部分处理。和后半部分合在一起是一个完整的处理流程。@see BkfLogSecondHalf
 * 该函数业务一般不直接调用，通过后面的宏封装来调用。@see BKF_LOG
 *
 * @param[in] *log log库句柄
 * @param[in] *modName 模块名。@see BKF_MOD_NAME
 * @param[in] line log位置标识。
 * @param[in] lvl log level。当此值不小于模块的log level 时,log 才会输出。@ see BkfLogSetModuleLvl
 * @return 处理结果
 *   @retval BKF_OK 成功，需要继续处理
 *   @retval 其他 失败
 */
uint32_t BkfLogFirstHalf(BkfLog *log, char *modName, uint16_t line, uint8_t lvl);

/**
 * @brief log后半部分处理。和前半部分合在一起是一个完整的处理流程。@see BkfLogFirstHalf
 * 该函数业务一般不直接调用，通过后面的宏封装来调用。@see BKF_LOG
 *
 * @param[in] *log log库句柄
 * @param[in] *modName 模块名。@see BKF_MOD_NAME
 * @param[in] line log位置标识。
 * @param[in] lvl log level。当此值不小于模块的log level 时,log 才会输出。@ see BkfLogSetModuleLvl
 * @param[in] *funcName 函数名。
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
void BkfLogSecondHalf(BkfLog *log, char *modName, uint16_t line, uint8_t lvl, char *funcName, const char *fmt, ...);

/**
 * @brief 设置是否可以输出。@see struct tagBkfLogInitArg::outputOrNull @see struct tagBkfLogInitArg::outputEnable
 *
 * @param[in] *log log库句柄
 * @param[in] enable 是否可以输出。true为可以；false为不可以。
 * @return none
 */
void BkfLogSetOutputEnable(BkfLog *log, BOOL enable);

/* 宏，便于使用 */
/**
 * @brief log标准封装。因为参数比较多，建议使用下面的封装宏。
 * @see BKF_LOG_DEBUG @see BKF_LOG_INFO @see BKF_LOG_WARN @see BKF_LOG_ERROR
 *
 * @param[in] *log log库句柄
 * @param[in] *modName 模块名。@see BKF_MOD_NAME
 * @param[in] line log位置标识。
 * @param[in] lvl log level。当此值不小于模块的log level 时,log 才会输出。@ see BkfLogSetModuleLvl
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
#define BKF_LOG(log, modName, line, lvl, fmt, ...)                                                                     \
    do {                                                                                                               \
        if (((log) != VOS_NULL) && (BkfLogFirstHalf((log), (char *)(modName), (line), (lvl)) == BKF_OK)) {             \
            BkfLogSecondHalf((log), (char *)(modName), (line), (lvl), (char *)(__func__), fmt, ##__VA_ARGS__);         \
        }                                                                                                              \
    } while (0)

/**
 * @brief debug log宏封装。记录细粒度的诊断信息。
 *
 * @param[in] *log log库句柄
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
#define BKF_LOG_DEBUG(log, fmt, ...)                                                                                   \
    do {                                                                                                               \
        BKF_LOG((log), (BKF_MOD_NAME_), (BKF_LINE_), (BKF_LOG_LVL_DEBUG), fmt, ##__VA_ARGS__);                         \
    } while (0)

/**
 * @brief info log宏封装。记录粗粒度的诊断信息，例如关键的输入输出点。
 *
 * @param[in] *log log库句柄
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
#define BKF_LOG_INFO(log, fmt, ...)                                                                                    \
    do {                                                                                                               \
        BKF_LOG((log), (BKF_MOD_NAME_), (BKF_LINE_), (BKF_LOG_LVL_INFO), fmt, ##__VA_ARGS__);                          \
    } while (0)

/**
 * @brief warn log宏封装。记录潜在的错误。
 *
 * @param[in] *log log库句柄
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
#define BKF_LOG_WARN(log, fmt, ...)                                                                                    \
    do {                                                                                                               \
        BKF_LOG((log), (BKF_MOD_NAME_), (BKF_LINE_), (BKF_LOG_LVL_WARN), fmt, ##__VA_ARGS__);                          \
    } while (0)

/**
 * @brief error log宏封装。记录错误。
 *
 * @param[in] *log log库句柄
 * @param[in] fmt 格式化字串
 * @param[in] ... 格式化字串相关的可变参数。
 * @return none
 */
#define BKF_LOG_ERROR(log, fmt, ...)                                                                                   \
    do {                                                                                                               \
        BKF_LOG((log), (BKF_MOD_NAME_), (BKF_LINE_), (BKF_LOG_LVL_ERROR), fmt, ##__VA_ARGS__);                         \
    } while (0)

#pragma pack()

/**
 * @brief 黑匣子类型，log库内唯一即可
 */
enum {
    BKF_BLACKBOX_TYPE_INVALID = 0,
    BKF_BLACKBOX_TYPE_SESS = 1,
    BKF_BLACKBOX_TYPE_V8TLSCLI,
    BKF_BLACKBOX_TYPE_V8TLSSER,
    BKF_BLACKBOX_TYPE_V8TCPSER,
    BKF_BLACKBOX_TYPE_V8TCPCLI
};

/**
 * @brief 历史日志类型注册
 *
 * @param[in] *log log库句柄
 * @param[in] type  日志类型
 * @param[in] logBuffLen  日志buff长度上限
 * @return 处理结果
 *   @retval BKF_OK 成功
 *   @retval 其他 失败
 */
uint32_t BkfBlackBoxRegType(BkfLog *log, uint16_t type, uint16_t logBuffLen);
/**
 * @brief 历史日志类型卸载
 *
 * @param[in] *log log库句柄
 * @param[in] type  日志类型
 * @return none
 */
void BkfBlackBoxDelRegType(BkfLog *log, uint16_t type);
/**
 * @brief 历史日志记录
 *
 * @param[in] *log log库句柄
 * @param[in] type  日志类型
 * @param[in] key  日志key,内存指针
 * @param[in] bid  二级key
 * @param[in] format  日志输出格式
 * @return none
 */
void BkfBlackBoxLog(BkfLog *log, uint16_t type, void *key, uint32_t bid, const char *format, ...);
/**
 * @brief 历史日志记录宏
 */
#define BKF_BLACKBOX_LOG(LOG, TYPE, KEYPTR, BID, fmt, ...)                                                             \
    BkfBlackBoxLog((LOG), (TYPE), (KEYPTR), (BID), (fmt), ##__VA_ARGS__)

/**
 * @brief 打印指定历史日志
 *
 * @param[in] *log log库句柄
 * @param[in] type  日志类型
 * @param[in] key  日志key指针
 * @param[in] bid   二级key
 * @return none
 */
void BkfBlackBoxDispOneInstLog(BkfLog *log, uint16_t type, void *key, uint32_t bid);
#define BKF_DEBUG_LEVEL_ALL 0x0f
/**
 * @brief 删除指定历史日志
 *
 * @param[in] *log log库句柄
 * @param[in] type  日志类型
 * @param[in] key  日志key指针
 * @return none
 */
void BkfBlackBoxDelLogInstNode(BkfLog *log, uint16_t type, void *key, uint32_t bid);
/**
 * @brief 历史日志删除操作宏
 */
#define BKF_BLACKBOX_DELLOGINST(LOG, TYPE, KEYPTR, BID) BkfBlackBoxDelLogInstNode((LOG), (TYPE), (KEYPTR), (BID))
/**
 * @brief 获取log日志是否打开的开关
 *
 * @param[in] *log log库句柄
 * @return VOS_TRUE log日志开
 *       VOS_FALSE 日志关
 */
BOOL BkfLogGetOutputEnable(BkfLog *log);
/**
 * @brief 对外指定级别的debug开关
 *
 * @param[in] *log log库句柄
 * @param[in] debugLevel  对应的级别，对标log级别，若指定BKF_LOG_LVL_CNT则代表指定所有级别
 * @param[in] enable 对应级别是否打开，TRUE：打开 FALSE：关闭
 * @return none
 */
void BkfLogSetDebugOutputEnable(BkfLog *log, uint32_t debugLevel, BOOL enable);
/**
 * @brief 获取debug开关
 *
 * @param[in] *log log库句柄
 * @return 各level的debug开关状态，每一个bit位对应一个级别的开关
 */
uint8_t BkfLogGetDebugOutputEnable(BkfLog *log);
/**
 * @brief 挂接用户侧debug输出函数
 *
 * @param[in] *log log库句柄
 * @param[in] fun debug输出函数
 * @return none
 */
void BkfLogSetdebugOutputFun(BkfLog *log, F_BKF_DEBUG_OUTPUT fun);

/**
 * @brief 以字节方式打印收到的报文码流
 *
 * @param[in] *log log库句柄
 * @param[in] *recBuf 存储数据流的缓存地址
 * @param[in] len  缓存数据流长度
 * @return none
 */
void BkfLogTraceRcvDataFlow(BkfLog *log, char *recBuf, int32_t len);
#pragma GCC visibility pop
#ifdef __cplusplus
}
#endif
#endif
