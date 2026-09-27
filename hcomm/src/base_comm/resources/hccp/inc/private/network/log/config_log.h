/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CONFIG_LOG_H
#define CONFIG_LOG_H

#include <stdint.h>
#include "network_comm.h"
#include "ra_rs_opcode.h"
#include "user_log.h"

extern uint64_t gDebugConfig;

static inline uint64_t HccpGetDebugConfig(void)
{
    return gDebugConfig;
}

static inline void HccpSetDebugConfig(uint64_t value)
{
    gDebugConfig = value;
}

#define hccp_info_log(moduleType, fmt, args...)                                                                        \
    do {                                                                                                               \
        if ((HccpGetDebugConfig() & (1ULL << moduleType)) != 0) {                                                      \
            hccp_run_info(fmt, ##args);                                                                                \
        } else {                                                                                                       \
            hccp_info(fmt, ##args);                                                                                    \
        }                                                                                                              \
    } while (0)

/* 各模块日志封装，调用时请按模块选择对应宏 */
#define hccp_info_init(fmt, args...) hccp_info_log(HCCP_INIT, fmt, ##args)

#define hccp_info_rma(fmt, args...) hccp_info_log(RDMA_OP, fmt, ##args)

#define hccp_info_socket(fmt, args...) hccp_info_log(SOCKET_OP, fmt, ##args)

#define hccp_info_others(fmt, args...) hccp_info_log(OTHERS, fmt, ##args)

#define hccp_warn_log(moduleType, fmt, args...)                                                                        \
    do {                                                                                                               \
        if ((HccpGetDebugConfig() & (1ULL << moduleType)) != 0) {                                                      \
            hccp_run_warn(fmt, ##args);                                                                                \
        } else {                                                                                                       \
            hccp_warn(fmt, ##args);                                                                                    \
        }                                                                                                              \
    } while (0)
/* 各模块日志封装，调用时请按模块选择对应宏 */
#define hccp_warn_init(fmt, args...) hccp_warn_log(HCCP_INIT, fmt, ##args)

#define hccp_warn_rma(fmt, args...) hccp_warn_log(RDMA_OP, fmt, ##args)

#define hccp_warn_socket(fmt, args...) hccp_warn_log(SOCKET_OP, fmt, ##args)

#define hccp_warn_others(fmt, args...) hccp_warn_log(OTHERS, fmt, ##args)

#endif /* CONFIG_LOG_H */
