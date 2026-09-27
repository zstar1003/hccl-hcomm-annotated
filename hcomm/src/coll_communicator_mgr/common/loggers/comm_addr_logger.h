/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_ADDR_LOGGER_H
#define COMM_ADDR_LOGGER_H

#include <stdint.h>
#include <string>
#include "hccl/hccl_res.h"

#ifdef __cplusplus
extern "C" {
#endif

namespace hcomm {
    namespace logger {

        /**
         * @brief 通信地址日志记录器
         *
         * 职责：
         * - CommAddr 类型到字符串的转换
         * - CommAddr 的格式化打印
         * - 支持 IPv4/IPv6/ID/EID 多种地址类型
         *
         * 设计原则：
         * - 单一职责：专注于通信地址的日志记录
         * - 无状态：所有方法都是静态方法，无需实例化
         * - 类型安全：对每种地址类型提供专门处理
         */
        class CommAddrLogger {
        public:
            /**
             * @brief 将 CommAddr 转换为带类型的字符串
             * @param commAddr 通信地址
             * @return 格式化的地址字符串
             *
             * 格式示例（完全模仿 IpAddress::Describe）：
             * - IPv4: "IpAddress[eid[0000000000000000:12345678], AF=v4, addr=192.168.1.1]"
             * - IPv6: "IpAddress[eid[1234567890abcdef:1234567890abcdef], AF=v6, addr=fe80::204:61ff:fe9d:f156,
             * scopeId=0x0]"
             * - ID:   "IpAddress[id:0x12345678]"
             * - EID:  "IpAddress[eid[1234567890abcdef:1234567890abcdef]]"
             */
            static std::string ToString(const CommAddr& commAddr);

            /**
             * @brief 打印 CommAddr 详情
             * @param idx Channel 索引（用于日志上下文）
             * @param endpointName 端点名称（"localEndpoint" 或 "remoteEndpoint"）
             * @param commAddr 通信地址
             */
            static void Print(uint32_t idx, const char* endpointName, const CommAddr& commAddr);

            /**
             * @brief 获取 CommAddr 类型的描述字符串
             * @param type 地址类型
             * @return 类型描述字符串（"IPv4", "IPv6", "ID", "EID"）
             */
            static std::string GetTypeString(CommAddrType type);

        private:
            // 私有构造函数（静态工具类）
            CommAddrLogger() = delete;
            CommAddrLogger(const CommAddrLogger&) = delete;
            CommAddrLogger& operator=(const CommAddrLogger&) = delete;

            /**
             * @brief IPv4 地址转换辅助函数
             */
            static std::string ConvertIPv4(const struct in_addr& addr);

            /**
             * @brief IPv6 地址转换辅助函数
             */
            static std::string ConvertIPv6(const struct in6_addr& addr6);

            /**
             * @brief ID 类型转换辅助函数
             */
            static std::string ConvertID(uint32_t id);

            /**
             * @brief EID 类型转换辅助函数
             */
            static std::string ConvertEID(const uint8_t eid[16]);
        };

    } // namespace logger
} // namespace hcomm

#ifdef __cplusplus
}
#endif

#endif // COMM_ADDR_LOGGER_H
