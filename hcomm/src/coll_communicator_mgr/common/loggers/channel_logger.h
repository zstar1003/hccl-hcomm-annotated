/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHANNEL_LOGGER_H
#define CHANNEL_LOGGER_H

#include <stdint.h>
#include <string>
#include "hccl/hccl_res.h"
#include "orion_adapter_hccp.h"
#include "hcomm_res_defs.h" // ChannelHandle

#ifdef __cplusplus
extern "C" {
#endif

namespace hcomm {
    namespace logger {

        // 前向声明
        class CommAddrLogger;
        class EndpointLogger;

        /**
         * @brief 通道日志记录器
         *
         * 职责：
         * - 打印 Channel 描述符的完整信息
         * - 打印 Channel 连接错误信息表格
         * - 协调 CommAddrLogger 和 EndpointLogger 完成信息打印
         *
         * 设计原则：
         * - 单一职责：专注于通道信息的日志记录
         * - 组合复用：通过组合其他 Logger 实现代码复用
         * - 无状态：所有方法都是静态方法
         */
        class ChannelLogger {
        public:
            /**
             * @brief 打印 HcclChannelDesc 详细信息
             * @param idx Channel 索引
             * @param channelDesc Channel 描述符
             *
             * 打印内容：
             * - 基本信息：remoteRank, channelProtocol, notifyNum, memHandleNum
             * - 本地端点：commAddr + loc
             * - 远端端点：commAddr + loc
             * - ROCE 协议特有属性（如果适用）
             */
            static void PrintDescInfo(uint32_t idx, const HcclChannelDesc& channelDesc);

            /**
             * @brief 打印 Channel 描述信息表格头部
             *
             * 表格列：idx | remoteRank | Proto | notifyNum | memHandleNum | localAddr | remoteAddr | ROCE Attr
             */
            static void PrintDescTableHeader();

            /**
             * @brief 打印单个 Channel 的描述信息（表格行）
             * @param idx Channel 索引
             * @param channelDesc Channel 描述符
             *
             * 表格列：idx | remoteRank | Proto | notifyNum | memHandleNum | localAddr | remoteAddr | ROCE Attr
             */
            static void PrintDescInfoRow(uint32_t idx, const HcclChannelDesc& channelDesc);

            /**
             * @brief 打印 Channel 连接错误信息表格头部
             * @param localRank 本地 Rank ID
             */
            static void PrintErrorTableHeader(uint32_t localRank);

            /**
             * @brief 打印单个 Channel 的错误状态（表格行）
             * @param idx Channel 索引
             * @param localRank 本地 Rank ID
             * @param channelDesc Channel 描述符
             * @param channelHandle Channel 句柄
             * @param status Channel 状态值
             * @param elapsedMs 已耗时（毫秒）
             * @param tlsStatus 当前 Channel 对应的 TLS 状态
             */
            static void PrintErrorInfo(
                uint32_t idx, uint32_t localRank, const HcclChannelDesc& channelDesc, ChannelHandle channelHandle,
                int32_t status, uint64_t elapsedMs, Hccl::TlsStatus tlsStatus);

            /**
             * @brief 打印 Channel 错误详情（批量）
             * @param localRank 本地 Rank ID
             * @param channelNum Channel 数量
             * @param channelDescs Channel 描述符数组
             * @param channelHandles Channel 句柄数组
             * @param statusList Channel 状态数组
             * @param elapsedMs 已耗时（毫秒）
             * @param tlsStatusList 逐通道 TLS 状态数组
             *
             * 功能：
             * - 打印错误详情表格（只打印异常状态的 Channel）
             * - 表格外单独打印详细信息（FAILED 或 TIMEOUT 状态）
             */
            static void PrintChannelErrorDetails(
                uint32_t localRank, uint32_t channelNum, const HcclChannelDesc* channelDescs,
                ChannelHandle* channelHandles, int32_t* statusList, int64_t elapsedMs,
                const Hccl::TlsStatus* tlsStatusList);

        private:
            // 私有构造函数（静态工具类）
            ChannelLogger() = delete;
            ChannelLogger(const ChannelLogger&) = delete;
            ChannelLogger& operator=(const ChannelLogger&) = delete;

            /**
             * @brief 打印 Channel 的基本信息字段
             */
            static void PrintBasicFields(uint32_t idx, const HcclChannelDesc& channelDesc);

            /**
             * @brief 打印 ROCE 协议特有属性
             */
            static void PrintRoceAttributes(uint32_t idx, const HcclChannelDesc& channelDesc);

            static void PrintUbMemAttributes(uint32_t idx, const HcclChannelDesc& channelDesc);

            static bool IsUbMemProtocol(const HcclChannelDesc& channelDesc);

            // ========== 新增：私有辅助函数（消除重复代码） ==========

            /**
             * @brief 判断是否为 ROCE 协议
             * @param channelDesc Channel 描述符
             * @return true 如果是 ROCE 协议，false 否则
             */
            static bool IsRoceProtocol(const HcclChannelDesc& channelDesc);

            /**
             * @brief 格式化 ROCE 属性为紧凑字符串（表格用）
             * @param channelDesc Channel 描述符
             * @return 紧凑格式字符串，如 "q:4 r:3 ri:20 tc:0 sl:0" 或 "-"
             */
            static std::string FormatRoceAttrCompact(const HcclChannelDesc& channelDesc);

            /**
             * @brief 格式化 ROCE 属性为详细字符串（日志用）
             * @param channelDesc Channel 描述符
             * @return 详细格式字符串，如 "queueNum[4], retryCnt[3], retryInterval[20], tc[0], sl[0]"
             */
            static std::string FormatRoceAttrDetail(const HcclChannelDesc& channelDesc);

            /**
             * @brief 判断状态是否异常（非 READY）
             * @param status Channel 状态值
             * @return true 如果状态异常，false 如果状态为 READY
             */
            static bool IsAbnormalStatus(int32_t status);

            /**
             * @brief 判断是否需要详细打印（FAILED 或 SOCKET_TIMEOUT）
             * @param status Channel 状态值
             * @return true 如果需要详细打印，false 否则
             */
            static bool NeedDetailPrint(int32_t status);

            /**
             * @brief 批量格式化端点地址（输出引用）
             * @param channelDesc Channel 描述符
             * @param outLocalAddr 输出本地地址字符串
             * @param outRemoteAddr 输出远端地址字符串
             */
            static void FormatEndpointAddresses(
                const HcclChannelDesc& channelDesc, std::string& outLocalAddr, std::string& outRemoteAddr);
        };

        /**
         * @brief Channel 状态工具类
         *
         * 职责：
         * - 将 ChannelStatus 状态值转换为可读字符串
         * - 提供状态枚举的字符串化能力
         */
        class ChannelStatusUtils {
        public:
            /**
             * @brief 将 ChannelStatus 状态值转换为可读字符串
             * @param status ChannelStatus 状态值
             * @return 状态描述字符串（如 "ChannelStatus::SOCKET_TIMEOUT"）
             */
            static std::string ToString(int32_t status);

        private:
            ChannelStatusUtils() = delete;
        };

    } // namespace logger
} // namespace hcomm

#ifdef __cplusplus
}
#endif

#endif // CHANNEL_LOGGER_H
