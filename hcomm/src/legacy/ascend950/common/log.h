/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <sstream>
#include <sys/syscall.h>
#include <unistd.h>
#include <securec.h>
#include <dlog_pub.h>
#include "hccl/base.h"

namespace Hccl {

#ifndef T_DESC
#define T_DESC(_msg, _y) ((_y) ? true : false)
#endif

#if T_DESC("日志处理适配", true)

constexpr u32 HCCL_LOG_DEBUG = 0x0;
constexpr u32 HCCL_LOG_INFO = 0x1;
constexpr u32 HCCL_LOG_WARN = 0x2;
constexpr u32 HCCL_LOG_ERROR = 0x3;
constexpr u32 HCCL_LOG_OPLOG = 0x6;
constexpr u32 HCCL_LOG_RUN_INFO = 0xff;
constexpr u32 HCCL_LOG_RUN_WARN = 0xfe;

enum class HcclSubModuleID {
    LOG_SUB_MODULE_ID_HCCL = 0,
    LOG_SUB_MODULE_ID_HCOM = 1,
};

/* 设置日志的commid和rankid */
#ifndef LIKELY
#define LIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 1)))
#define UNLIKELY(x) (static_cast<bool>(__builtin_expect(static_cast<bool>(x), 0)))
#endif

/* 每一条日志的长度,超过该长度会申请堆内存 */
constexpr s32 LOG_TMPBUF_SIZE = 512;

void CallDlogInvalidType(int level, int errCode, std::string file, int line);

void CallDlogNoSzFormat(int level, int errCode, std::string file, int line);

void CallDlogMemError(int level, std::string file, int line);

void CallDlogPrintError(int level, std::string file, int line);

void CallDlog(int level, int sysCallBack, const char* buffer, std::string file, int line);

bool CheckDebugLogLevel();

bool CheckInfoLogLevel();

int HcclCheckLogLevel(int logLevel);

#define LOG_FUNC(module, level, fmt, ...)              \
    do {                                               \
        DlogRecord(module, level, fmt, ##__VA_ARGS__); \
    } while (0)

#define HCCL_LOG_PRINT(moduleId, logType, format, ...)                                                              \
    do {                                                                                                            \
        LOG_FUNC(moduleId, logType, "[%s:%d] [%u]" format, __FILE__, __LINE__, syscall(SYS_gettid), ##__VA_ARGS__); \
    } while (0)

#define LOG_PRINT(logType, szFormat, ...)                                                                      \
    do {                                                                                                       \
        if (UNLIKELY(::Hccl::HcclCheckLogLevel(logType) == 1)) {                                               \
            char stackLogBuffer[::Hccl::LOG_TMPBUF_SIZE]; /* 使用栈中的buffer, 小而快 */               \
            if (szFormat == nullptr) {                                                                         \
                ::Hccl::CallDlogNoSzFormat(                                                                    \
                    ::Hccl::HCCL_LOG_ERROR, HCCL_ERROR_CODE(HcclResult::HCCL_E_INTERNAL), __FILE__, __LINE__); \
            } else {                                                                                           \
                if (memset_s(stackLogBuffer, ::Hccl::LOG_TMPBUF_SIZE, 0, sizeof(stackLogBuffer)) != EOK) {     \
                    ::Hccl::CallDlogMemError(::Hccl::HCCL_LOG_ERROR, __FILE__, __LINE__);                      \
                } else if (                                                                                    \
                    (snprintf_s(                                                                               \
                         stackLogBuffer, sizeof(stackLogBuffer), (sizeof(stackLogBuffer) - 1), szFormat,       \
                         ##__VA_ARGS__)                                                                        \
                     == -1)                                                                                    \
                    && (stackLogBuffer[0] == 0)) {                                                             \
                    ::Hccl::CallDlogPrintError(::Hccl::HCCL_LOG_ERROR, __FILE__, __LINE__);                    \
                } else {                                                                                       \
                    /* 如果collectiveID和rankID都为空，则默认输出为PID和TID */                   \
                    ::Hccl::CallDlog(logType, syscall(SYS_gettid), stackLogBuffer, __FILE__, __LINE__);        \
                }                                                                                              \
            }                                                                                                  \
        }                                                                                                      \
    } while (0)

/* 当前日志级别，为了优化性能，日志EVENT  判断在宏入口检查 */
/* 使用宏记录日志, 以便获取日志在代码中的位置 */
#define MODULE_DEBUG(format, ...)                                 \
    do {                                                          \
        LOG_PRINT(::Hccl::HCCL_LOG_DEBUG, format, ##__VA_ARGS__); \
    } while (0)

#define MODULE_INFO(format, ...)                                 \
    do {                                                         \
        LOG_PRINT(::Hccl::HCCL_LOG_INFO, format, ##__VA_ARGS__); \
    } while (0)

#define MODULE_WARNING(format, ...)                              \
    do {                                                         \
        LOG_PRINT(::Hccl::HCCL_LOG_WARN, format, ##__VA_ARGS__); \
    } while (0)

#define MODULE_ERROR(format, ...)                                 \
    do {                                                          \
        LOG_PRINT(::Hccl::HCCL_LOG_ERROR, format, ##__VA_ARGS__); \
    } while (0)

/* 运行日志，记录在run目录下 */
#define MODULE_RUN_INFO(format, ...)                                 \
    do {                                                             \
        LOG_PRINT(::Hccl::HCCL_LOG_RUN_INFO, format, ##__VA_ARGS__); \
    } while (0)

/* 运行警告日志，记录在run目录下 */
#define MODULE_RUN_WARNING(format, ...)                              \
    do {                                                             \
        LOG_PRINT(::Hccl::HCCL_LOG_RUN_WARN, format, ##__VA_ARGS__); \
    } while (0)

// 错误码
const u64 SYSTEM_RESERVE_ERROR = 0;
const u64 HCCL_MODULE_ID = 5;

/* 预定义日志宏, 便于使用 */
#define HCCL_DEBUG(...) MODULE_DEBUG(__VA_ARGS__)
#define HCCL_INFO(...) MODULE_INFO(__VA_ARGS__)
#define HCCL_WARNING(...) MODULE_WARNING(__VA_ARGS__)
#define HCCL_ERROR(...) MODULE_ERROR(__VA_ARGS__)
/* 运行日志 */
#define HCCL_RUN_INFO(...) MODULE_RUN_INFO(__VA_ARGS__)
#define HCCL_RUN_WARNING(...) MODULE_RUN_WARNING(__VA_ARGS__)

#define HCCL_ERROR_CODE(error)                                             \
    ((::Hccl::SYSTEM_RESERVE_ERROR << 32) + (::Hccl::HCCL_MODULE_ID << 24) \
     + ((static_cast<u64>(::Hccl::HcclSubModuleID::LOG_SUB_MODULE_ID_HCCL)) << 16) + static_cast<u64>(error))
#define HCOM_ERROR_CODE(error)                                             \
    ((::Hccl::SYSTEM_RESERVE_ERROR << 32) + (::Hccl::HCCL_MODULE_ID << 24) \
     + ((static_cast<u64>(::Hccl::HcclSubModuleID::LOG_SUB_MODULE_ID_HCOM)) << 16) + static_cast<u64>(error))
#endif

#if T_DESC("公共代码宏", true)

// 检查C++11的智能指针, 若为空, 则记录日志, 并返回错误
#define CHK_SMART_PTR_NULL(smart_ptr)                                                                                 \
    do {                                                                                                              \
        if (UNLIKELY(!(smart_ptr))) {                                                                                 \
            HCCL_ERROR(                                                                                               \
                "[%s]errNo[0x%016llx] ptr [%s] is nullptr, return HCCL_E_PTR", __func__, HCCL_ERROR_CODE(HCCL_E_PTR), \
                #smart_ptr);                                                                                          \
            return HCCL_E_PTR;                                                                                        \
        }                                                                                                             \
    } while (0)

/* 检查函数返回值, 记录指定日志, 并返回指定错误码 */
#define CHK_PRT_RET(result, exeLog, retCode) \
    do {                                     \
        if (UNLIKELY(result)) {              \
            exeLog;                          \
            return retCode;                  \
        }                                    \
    } while (0)

/* 检查函数返回值, 记录指定日志, 并直接返回 */
#define CHK_PRT_RET_NULL(result, exeLog) \
    do {                                 \
        if (UNLIKELY(result)) {          \
            exeLog;                      \
            return;                      \
        }                                \
    } while (0)

/* 检查函数返回值, 记录指定日志, 函数不返回 */
#define CHK_PRT_CONT(result, exeLog) \
    do {                             \
        if (UNLIKELY(result)) {      \
            exeLog;                  \
        }                            \
    } while (0)

// 检查C++11的智能指针, 若为空, 则记录日志, 并返回
#define CHK_SMART_PTR_RET_NULL(smart_ptr)                                                                   \
    do {                                                                                                    \
        if (UNLIKELY(!(smart_ptr))) {                                                                       \
            HCCL_ERROR("[%s]errNo[0x%016llx]smart_ptr is nullptr.", __func__, HCCL_ERROR_CODE(HCCL_E_PTR)); \
            return;                                                                                         \
        }                                                                                                   \
    } while (0)

/* 检查函数返回值, 并返回指定错误码 */
#define CHK_RET(call)                                                             \
    do {                                                                          \
        HcclResult hcclRet = static_cast<HcclResult>(call);                       \
        if (UNLIKELY(hcclRet != HcclResult::HCCL_SUCCESS)) {                      \
            if (hcclRet == HcclResult::HCCL_E_AGAIN) {                            \
                HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            } else {                                                              \
                HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet);   \
            }                                                                     \
            return hcclRet;                                                       \
        }                                                                         \
    } while (0)

/* 检查result. 若错误, 则设置错误并break */
#define CHK_PRT_BREAK(result, exeLog, exeCmd) \
    if (UNLIKELY(result)) {                   \
        exeLog;                               \
        exeCmd;                               \
        break;                                \
    }

/* 检查函数返回值，HCCL_E_UNAVAIL时给Warning, 并返回指定错误码 */
#define CHK_RET_UNAVAIL(call)                                                                   \
    do {                                                                                        \
        HcclResult hcclRet = static_cast<HcclResult>(call);                                     \
        if (UNLIKELY(hcclRet != HcclResult::HCCL_SUCCESS)) {                                    \
            if (hcclRet == HcclResult::HCCL_E_AGAIN || hcclRet == HcclResult::HCCL_E_UNAVAIL) { \
                HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet);               \
            } else {                                                                            \
                HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet);                 \
            }                                                                                   \
            return hcclRet;                                                                     \
        }                                                                                       \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回错误 */
#define CHK_PTR_NULL(ptr)                                                           \
    do {                                                                            \
        if (UNLIKELY((ptr) == nullptr)) {                                           \
            HCCL_ERROR(                                                             \
                "errNo[0x%016llx] ptr [%s] is NULL, return HcclResult::HCCL_E_PTR", \
                HCCL_ERROR_CODE(HcclResult::HCCL_E_PTR), #ptr);                     \
            return HcclResult::HCCL_E_PTR;                                          \
        }                                                                           \
    } while (0)

/* 检查函数返回值, 返错时打印函数名及通信域标识 */
#define CHK_RET_AND_PRINT_IDE(call, identifier)                                   \
    do {                                                                          \
        HcclResult hcclRet = static_cast<HcclResult>(call);                       \
        if (UNLIKELY(hcclRet != HCCL_SUCCESS)) {                                  \
            HCCL_RUN_INFO("[HCCL_TRACE]%s identifier[%s]", __func__, identifier); \
            if (hcclRet == HCCL_E_AGAIN) {                                        \
                HCCL_WARNING("[%s]call trace: hcclRet -> %d", __func__, hcclRet); \
            } else {                                                              \
                HCCL_ERROR("[%s]call trace: hcclRet -> %d", __func__, hcclRet);   \
            }                                                                     \
            return hcclRet;                                                       \
        }                                                                         \
    } while (0)

/* 检查函数返回值, 并返回HCCL_E_INTERNAL错误码 */
#define CHK_SAFETY_FUNC_RET(call)                                                   \
    do {                                                                            \
        s32 ret = call;                                                             \
        if (UNLIKELY(ret != EOK)) {                                                 \
            HCCL_ERROR("[%s]call trace: safety func err ret -> %d", __func__, ret); \
            return HCCL_E_INTERNAL;                                                 \
        }                                                                           \
    } while (0)

#define EXCEPTION_CATCH(expression, retExp)                                       \
    do {                                                                          \
        try {                                                                     \
            expression;                                                           \
        } catch (HcclException & e) {                                             \
            HCCL_ERROR("[%s]Failed, exception caught:%s", __func__, e.what());    \
            auto backTraces = e.GetBackTraceStrings();                            \
            std::for_each(backTraces.begin(), backTraces.end(), [](string item) { \
                HCCL_ERROR(item.c_str());                                         \
            });                                                                   \
            retExp;                                                               \
        } catch (std::exception & e) {                                            \
            HCCL_ERROR("[%s]Failed, exception caught:%s", __func__, e.what());    \
            retExp;                                                               \
        } catch (...) {                                                           \
            HCCL_ERROR("exception caught others");                                \
            retExp;                                                               \
        }                                                                         \
    } while (0)

#endif
} // namespace Hccl
#endif // HCCLV2_LOG_H
