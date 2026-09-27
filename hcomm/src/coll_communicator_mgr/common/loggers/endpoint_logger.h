/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ENDPOINT_LOGGER_H
#define ENDPOINT_LOGGER_H

#include <stdint.h>
#include "hccl/hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace hcomm {
    namespace logger {

        // 前向声明
        class CommAddrLogger;

        /**
         * @brief 端点日志记录器
         *
         * 职责：
         * - 打印端点位置信息（Device/Host）
         * - 打印端点的完整信息（包含 commAddr 和 loc）
         * - 协调 CommAddrLogger 完成地址打印
         *
         * 设计原则：
         * - 单一职责：专注于端点信息的日志记录
         * - 组合复用：通过组合 CommAddrLogger 实现代码复用
         * - 无状态：所有方法都是静态方法
         */
        class EndpointLogger {
        public:
            /**
             * @brief 打印端点位置信息
             * @param idx Channel 索引（用于日志上下文）
             * @param endpointName 端点名称（"localEndpoint" 或 "remoteEndpoint"）
             * @param loc 端点位置
             *
             * 根据 locType 打印不同的位置信息：
             * - DEVICE: 打印 devPhyId, superDevId, serverIdx, superPodIdx
             * - HOST: 打印 host.id
             * - 其他: 仅打印 locType
             */
            static void PrintLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc);

            /**
             * @brief 打印端点的完整信息（commAddr + loc）
             * @param idx Channel 索引（用于日志上下文）
             * @param endpointName 端点名称（"localEndpoint" 或 "remoteEndpoint"）
             * @param endpointDesc 端点描述符（包含 commAddr 和 loc）
             */
            static void Print(uint32_t idx, const char* endpointName, const EndpointDesc& endpointDesc);

        private:
            // 私有构造函数（静态工具类）
            EndpointLogger() = delete;
            EndpointLogger(const EndpointLogger&) = delete;
            EndpointLogger& operator=(const EndpointLogger&) = delete;

            /**
             * @brief 打印 Device 类型的位置信息
             */
            static void PrintDeviceLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc);

            /**
             * @brief 打印 Host 类型的位置信息
             */
            static void PrintHostLocation(uint32_t idx, const char* endpointName, const EndpointLoc& loc);
        };

    } // namespace logger
} // namespace hcomm

#ifdef __cplusplus
}
#endif

#endif // ENDPOINT_LOGGER_H
