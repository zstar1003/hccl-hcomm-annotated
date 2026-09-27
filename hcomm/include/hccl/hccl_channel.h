/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CHANNEL_H
#define HCCL_CHANNEL_H

#include <stdint.h>
#include "hccl_types.h"
#include "hcomm_res_defs.h"
#include "hcomm_channel.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * @brief 内存句柄类型（不透明结构）
 */
typedef void* HcclMemHandle;

static const uint32_t HCCL_CHANNEL_MAGIC_WORD = 0x0f0f0f0fU;
static const uint32_t HCCL_CHANNEL_VERSION = 1U;            // HcclChannelDesc更新时，HCCL_CHANNEL_VERSION + 1
static const uint32_t HCCL_CHANNEL_DESC_RAW_MAX_LEN = 128U; ///< HcclChannelDesc union通用缓存最大长度（字节）

/**
 * @brief 通道描述参数
 * @note
 * 1、结构体需要将union内的raws完成初始化，union内及内部的结构体可以在末尾扩展新内容，但是总内存不能超过HCCL_CHANNEL_DESC_RAW_MAX_LEN字节。
 * union内扩展可以不需自增版本号，但是要增加双向兼容处理逻辑。
 * 2、结构体末尾允许扩展内容，但是每次扩展需要自增版本号，添加新增字段的初始化，且增加双向兼容处理逻辑。
 */
typedef struct {
    CommAbiHeader header;
    uint32_t remoteRank;          ///< 远端rankId
    CommProtocol channelProtocol; ///< 通信协议
    EndpointDesc localEndpoint;   ///< 本地网络设备端侧描述
    EndpointDesc remoteEndpoint;  ///< 远端网络设备端侧描述
    uint32_t notifyNum;           ///< channel上使用的通知消息数量
    HcclMemHandle* memHandles;    ///< 注册到通信域的待交换内存句柄
    uint32_t memHandleNum;        ///< 注册到通信域的待交换内存句柄数量
    union {
        uint8_t raws[HCCL_CHANNEL_DESC_RAW_MAX_LEN]; ///< 通用缓存
        struct {
            uint32_t queueNum;      ///< QP数量
            uint32_t retryCnt;      ///< 最大重传次数
            uint32_t retryInterval; ///< 重传间隔(ms)（对应协议计算公式）
            uint8_t tc;             ///< 流量类别(QoS)
            uint8_t sl;             ///< 服务等级(QoS)
        } roceAttr;
        struct {
            uint8_t
                pathMode; ///< UB_MEM访问路径模式：0表示默认路径(有单路径走单路径，无单路径走多路径)，1为单路径，2为多路径；0xFF表示使用默认值0
        } ubMemAttr;
    };
} HcclChannelDesc;

#ifndef LIKELY
#define LIKELY(x) (__builtin_expect(!!(x), 1))
#define UNLIKELY(x) (__builtin_expect(!!(x), 0))
#endif

/**
 * @brief 初始化HcclChannelDesc结构体
 *
 * @param[inout] channelDesc 返回的通道描述参数
 * @param[in] descNum 描述数量
 * @return HcclResult 执行结果状态码
 */
static inline HcclResult HcclChannelDescInit(HcclChannelDesc* channelDesc, uint32_t descNum)
{
    for (uint32_t idx = 0; idx < descNum; idx++) {
        if (channelDesc != NULL) {
            // 先用0xFF填充整个结构体
            (void)memset_s(channelDesc, sizeof(HcclChannelDesc), 0xFF, sizeof(HcclChannelDesc));

            // 初始化ABI头信息
            channelDesc->header.version = HCCL_CHANNEL_VERSION;
            channelDesc->header.magicWord = HCCL_CHANNEL_MAGIC_WORD;
            channelDesc->header.size = sizeof(HcclChannelDesc);
            channelDesc->header.reserved = 0;

            // 初始化关键字段
            channelDesc->remoteRank = ~0U; // 保持原始无效值标记
            channelDesc->channelProtocol = COMM_PROTOCOL_RESERVED;
            channelDesc->notifyNum = 0;
            channelDesc->memHandles = NULL;
            channelDesc->memHandleNum = 0;

            // 显式设置EndpointDesc相关字段为无效值
            if (UNLIKELY(EndpointDescInit(&channelDesc->localEndpoint, 1) != HCCL_SUCCESS)
                || UNLIKELY(EndpointDescInit(&channelDesc->remoteEndpoint, 1) != HCCL_SUCCESS)) {
                return HCCL_E_INTERNAL;
            }
            channelDesc++; // 移动到下一个描述符
        } else {
            return HCCL_E_PTR;
        }
    }
    return HCCL_SUCCESS;
}

/**
 * @defgroup 通信通道管理接口
 * @{
 */

/**
 * @brief 基于通信域和给定信息创建通信通道
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] channelDescs 通道描述列表
 * @param[in] channelNum channel数量
 * @param[out] channels 创建的通道句柄列表
 * @return HcclResult 执行结果状态码
 * @warning 重要约束：channelDescs必须使用HcclChannelDescInit进行初始化
 */
extern HcclResult HcclChannelAcquire(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels);

/**
 * @brief Channel 配置对象不透明句柄（HCCM 层）。
 *        通过 HcclChannelConfigCreate 创建，HcclChannelConfigDestroy 销毁。
 */
typedef void* HcclChannelConfig;

static const uint32_t HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN
    = 255U; ///< SHARED_QUEUE_TAG最大长度（字节），与HCCL_RES_TAG_MAX_LEN一致

/**
 * @brief Channel 配置属性类型枚举
 *        通过 HcclChannelConfigSetInt / HcclChannelConfigSetStr 设置。
 */
typedef enum {
    HCCL_CHANNEL_CONFIG_TYPE_INVALID = -1,
    /** 0: IS_SHARED_QUEUE (bool/int, 默认 0=false)。
     *    仅支持 AIV 引擎的 UB 网络语义协议（UB_CTP/UBC_TP/UB_RTP，不支持 UBMem/RoCE/UBOE）。
     *    设为 true 时，创建的多个 Channel 共享一个 Jetty。 */
    HCCL_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
    /** 1: SHARED_QUEUE_TAG (string)。
     *    仅 IS_SHARED_QUEUE=true 时有效，指定创建 channel 的 tag。
     *    重复调用时，使用相同 tag 创建出的 channel 共享一个 jetty。
     *    以'\0'结尾，最大字符长度为HCCL_CHANNEL_CONFIG_SHARED_QUEUE_TAG_MAX_LEN。 */
    HCCL_CHANNEL_CONFIG_TYPE_SHARED_QUEUE_TAG = 1,
} HcclChannelConfigType;

/**
 * @brief 创建 Channel 配置对象
 * @param[out] config 输出的配置对象指针（不透明句柄）
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclChannelConfigCreate(HcclChannelConfig* config);

/**
 * @brief 销毁 Channel 配置对象
 * @param[in] config 配置对象指针
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclChannelConfigDestroy(HcclChannelConfig config);

/**
 * @brief 设置 Channel 配置的整型属性
 * @param[in] config 配置对象指针
 * @param[in] type  属性类型，参见 HcclChannelConfigType
 * @param[in] value 属性值
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclChannelConfigSetInt(HcclChannelConfig config, HcclChannelConfigType type, uint32_t value);

/**
 * @brief 设置 Channel 配置的字符串型属性
 * @param[in] config 配置对象指针
 * @param[in] type  属性类型，参见 HcclChannelConfigType
 * @param[in] value 属性值字符串
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclChannelConfigSetStr(HcclChannelConfig config, HcclChannelConfigType type, const char* value);

/**
 * @brief 基于通信域和配置创建通信通道（支持共享 Jetty）
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in] channelDescs 通道描述列表
 * @param[in] channelNum channel数量
 * @param[in] config Channel 配置对象指针（可为 NULL，等价于 HcclChannelAcquire）
 * @param[out] channels 创建的通道句柄列表
 * @return HcclResult 执行结果状态码
 * @note 当 config 中 IS_SHARED_QUEUE=true 时：
 *       - 仅支持 AIV 引擎的 UB 网络语义协议（UB_CTP/UBC_TP/UB_RTP），不支持 UBMem/RoCE/UBOE。
 *       - channelDescs 中的源 localEndpoint 必须相同（复用的 jetty 只能关联一个 endpoint）。
 *       - 通信域基于 SHARED_QUEUE_TAG 指定的 tag 管理 channel 复用。
 *       - tag 之下，使用源目的 endpointPair 索引一组 channel。
 *       - 重复调用时，如果 tag->endpointPair 下 channel 数量不足则创建后再返回；如果足够直接按序返回。
 *       - 不同通信域使用不同 EndpointHandle，同一个 EndpointHandle 共享一个 Jetty。
 * @warning 重要约束：channelDescs必须使用HcclChannelDescInit进行初始化
 */
extern HcclResult HcclChannelAcquireWithConfig(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    HcclChannelConfig config, ChannelHandle* channels);

/**
 * @brief 查询通信通道是否已存在
 * @param[in] comm 通信域句柄
 * @param[in] engine 通信引擎类型
 * @param[in]
 * channelDescs 通道描述列表
 * @param[in] channelNum channel数量
 * @param[out] channels
 * 查询结果句柄列表：已存在返回handle，不存在返回0（空句柄）
 * @return HcclResult 执行结果状态码
 * @warning
 * channelDescs必须使用HcclChannelDescInit进行初始化
 */
extern HcclResult HcclChannelQuery(
    HcclComm comm, CommEngine engine, const HcclChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels);

/**
 * @brief 销毁指定通信通道
 * @param[in] comm 通信域句柄
 * @param[in] channels 待销毁的通道句柄列表
 * @param[in]
 * channelNum 通道数量
 * @return HcclResult 执行结果状态码
 * @warning
 * channels中的handle应为有效channel句柄，销毁后该handle不可再用。暂只支持CCU通信引擎创建的channel
 */
extern HcclResult HcclChannelDestroy(HcclComm comm, const ChannelHandle* channels, uint32_t channelNum);

/**
 * @brief 获取指定channel的Hccl通信缓存
 * @param[in] comm 通信域句柄
 * @param[in] channel 通信通道句柄
 * @param[out] buffer Hccl缓存地址
 * @param[out] size Hccl缓存大小
 * @return HcclResult 执行结果状态码
 */
extern HcclResult HcclChannelGetHcclBuffer(HcclComm comm, ChannelHandle channel, void** buffer, uint64_t* size);

/**
 * @brief 获取channel中全部交换获得的远端内存信息
 * @param[in] comm 通信域句柄
 * @param[in] channel 通道句柄
 * @param[out] memNum 内存数量
 * @param[out] remoteMems 远端内存列表
 * @param[out] memTags 远端内存字符串标签列表
 * @return HcclResult 执行结果状态码
 * @warning
 */
extern HcclResult
HcclChannelGetRemoteMems(HcclComm comm, ChannelHandle channel, uint32_t* memNum, CommMem** remoteMems, char*** memTags);

/** @} */ // 通信通道管理接口

#ifdef __cplusplus
}
#endif // __cplusplus
#endif // HCCL_CHANNEL_H
