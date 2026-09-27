/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_LOG_H
#define SIM_LOG_H

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE // global级别设置为最低, 只通过sink的级别控制日志输出

#include <iostream>
#include <memory>
#include <sstream>
#include <sys/types.h>

#include "spdlog/spdlog.h"

extern spdlog::logger* g_logger;

#ifdef HCCL_VM_MODULE
#define HCCL_VM_TAG "[HCCL-VM-" HCCL_VM_MODULE "]"
#else
#define HCCL_VM_TAG "[HCCL-VM]"
#endif

#define HCCL_VM_TRACE(...)                                          \
    do {                                                            \
        if (g_logger != nullptr) {                                  \
            SPDLOG_LOGGER_TRACE(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                           \
    } while (0)

#define HCCL_VM_DEBUG(...)                                          \
    do {                                                            \
        if (g_logger != nullptr) {                                  \
            SPDLOG_LOGGER_DEBUG(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                           \
    } while (0)

#define HCCL_VM_INFO(...)                                          \
    do {                                                           \
        if (g_logger != nullptr) {                                 \
            SPDLOG_LOGGER_INFO(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                          \
    } while (0)

#define HCCL_VM_WARN(...)                                          \
    do {                                                           \
        if (g_logger != nullptr) {                                 \
            SPDLOG_LOGGER_WARN(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                          \
    } while (0)

#define HCCL_VM_ERROR(...)                                          \
    do {                                                            \
        if (g_logger != nullptr) {                                  \
            SPDLOG_LOGGER_ERROR(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                           \
    } while (0)

#define HCCL_VM_CRITICAL(...)                                          \
    do {                                                               \
        if (g_logger != nullptr) {                                     \
            SPDLOG_LOGGER_CRITICAL(g_logger, HCCL_VM_TAG __VA_ARGS__); \
        }                                                              \
    } while (0)

struct LogConfig {
    int consoleLevel{2};                  // default info
    int fileLevel{1};                     // default debug
    size_t maxFileSize{50 * 1024 * 1024}; // 50MB
    size_t maxFiles{UINT16_MAX};
    std::string filePath{"logs"};
    std::string fileBaseName{"app_log"};
    std::string fileSuffix{".log"};
    bool enableCompress{false};
};

LogConfig LoadLogConfig(const std::string& process_name);
void InitLogger(const LogConfig& config);
void FlushLog();
void DeInitLogger();

#endif // SIM_LOG_H
