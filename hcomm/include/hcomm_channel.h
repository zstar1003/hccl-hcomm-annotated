/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CHANNEL_H
#define HCOMM_CHANNEL_H

#include "hcomm_res_defs.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

static const uint32_t HCOMM_CHANNEL_MAGIC_WORD = 0x0fcf0f0fU;
static const uint32_t HCOMM_CHANNEL_VERSION_ONE = 1U;
/** ABI v3：相比 v1 增加 uint32_t qos 与 const char *channelName（channel 业务匹配标识） */
/** ABI v4：相比 v3 增加 roceAttr.srcPortList（QP 源端口号，union 内字段，sizeof 不变） */
static const uint32_t HCOMM_CHANNEL_VERSION = 4U;

// channelName标识最大长度（字节）
static const uint32_t HCOMM_CHANNEL_NAME_MAX_LEN = 191U;
static const uint32_t HCOMM_CHANNEL_DESC_RAW_MAX_LEN = 128U; ///< HcommChannelDesc union通用缓存最大长度（字节）

/**
 * @brief 通道描述参数
 * @note 结构体末尾扩展需要自增版本号，并补充兼容处理逻辑。
 *       ABI v1：HCOMM_CHANNEL_VERSION_ONE，无 union 之后的 qos 字段，见 HCOMM_CHANNEL_DESC_ABI_V1_SIZE。
 *       ABI v3：HCOMM_CHANNEL_VERSION，相比 v1 增加 uint32_t qos 与 const char *channelName（channel 业务匹配标识）。
 *       ABI v4：HCOMM_CHANNEL_VERSION，相比 v3 增加 roceAttr.srcPortList（QP 源端口号，union 内字段，sizeof 不变）。
 */
typedef struct {
    CommAbiHeader header;        ///< ABI头部，包含版本等信息
    EndpointDesc remoteEndpoint; ///< 远端网络设备端侧描述
    uint32_t notifyNum;          ///< channel上使用的通知消息数量
    bool exchangeAllMems;        ///< true表示无需显式传入memHandles
    HcommMemHandle* memHandles;  ///< 注册到通信域的待交换内存句柄
    uint32_t memHandleNum;       ///< 待交换内存句柄数量
    HcommSocket socket;          ///< 预创建socket句柄
    HcommSocketRole role;        ///< 本端角色(SERVER或CLIENT)
    uint16_t port;               ///< 监听端口或目标端口
    union {
        uint8_t raws[HCOMM_CHANNEL_DESC_RAW_MAX_LEN]; ///< 通用缓存
        struct {
            uint32_t queueNum;      ///< QP数量
            uint32_t retryCnt;      ///< 最大重传次数
            uint32_t retryInterval; ///< 重传间隔（ms）
            uint8_t tc;             ///< 流量类别（QoS)
            uint8_t sl;             ///< 服务等级（QoS)
            uint32_t qpThreshold;   ///< 多QP场景下，每个QP最小数据量(B)
            uint32_t cqAttrFlags; ///< CQ属性标志位，用于配置ibv_cq_init_attr_ex的flags标志位，默认0。
                                  ///< 备注：NPU网卡不支持该配置；第三方网卡场景下是否有效，与各自网卡能力相关
            uint16_t* srcPortList; ///< QP源端口号，用于哈希分流
            uint32_t sqDepth; ///< SQ队列深度，0/0xffffffff表示使用默认值（默认2K）；有效范围[64, 32K]且为2^n
            uint32_t scqDepth; ///< 发送CQ队列深度，0/0xffffffff表示使用默认值（默认2K）；有效范围[64, 32K]且为2^n
        } roceAttr;
        struct {
            uint32_t qos; ///< HCCS QoS
        } hccsAttr;
        struct {
            uint32_t sqDepth; ///< UB队列深度，0表示使用默认值, 0和0xffffffff表示使用默认值
        } ubAttr;
        struct {
            uint8_t
                pathMode; ///< UB_MEM访问路径模式：0表示默认路径(有单路径走单路径，无单路径走多路径)，1为单路径，2为多路径；0xFF表示使用默认值0
        } ubMemAttr;
    };
    uint32_t qos; ///< 通道 QoS（与协议解耦；集合通信场景通常来自通信域 hcclQos）；取值范围：[0, 7]，或 0xFFFFFFFF
                  ///< 表示未配置。未配置时：RoCE 不改写 roceAttr.sl/tc；UB 类协议由内部默认 QoS（4）决定。
    const char* channelName; ///< channel业务匹配标识，两端需相同；NULL表示匿名channel
} HcommChannelDesc;

/** v1 描述符在内存中的长度（不含 union 之后的 trailing qos），用于兼容校验 */
#define HCOMM_CHANNEL_DESC_ABI_V1_SIZE (offsetof(HcommChannelDesc, qos))

/**
 * @brief 初始化HcommChannelDesc结构体
 *
 * @param[inout] channelDesc 返回的通道描述参数
 * @param[in] descNum 描述数量
 * @return HcommResult 执行结果状态码
 */
static inline HcommResult HcommChannelDescInit(HcommChannelDesc* channelDesc, uint32_t descNum)
{
    const HcommResult hcommEPointer = (HcommResult)2;  // 对齐HCCL_E_PTR
    const HcommResult hcommEInternal = (HcommResult)4; // 对齐HCCL_E_INTERNAL

    for (uint32_t idx = 0; idx < descNum; ++idx) {
        if (channelDesc == NULL) {
            return hcommEPointer;
        }

        (void)memset_s(channelDesc, sizeof(HcommChannelDesc), 0xFF, sizeof(HcommChannelDesc));
        channelDesc->header.version = HCOMM_CHANNEL_VERSION;
        channelDesc->header.magicWord = HCOMM_CHANNEL_MAGIC_WORD;
        channelDesc->header.size = sizeof(HcommChannelDesc);
        channelDesc->header.reserved = 0;
        channelDesc->notifyNum = 0;
        channelDesc->exchangeAllMems = false;
        channelDesc->memHandles = NULL;
        channelDesc->memHandleNum = 0;
        channelDesc->socket = NULL;
        channelDesc->role = HCOMM_SOCKET_ROLE_RESERVED;
        channelDesc->port = 0;
        channelDesc->roceAttr.srcPortList = NULL;
        channelDesc->channelName = NULL;
        if (EndpointDescInit(&channelDesc->remoteEndpoint, 1) != 0) {
            return hcommEInternal;
        }
        ++channelDesc;
    }

    return 0;
}

/**
 * @brief Channel 配置对象不透明句柄（base_comm 层）。
 *        通过 HcommChannelConfigCreate 创建，HcommChannelConfigDestroy 销毁。
 */
typedef void* HcommChannelConfig;

/**
 * @brief Channel 配置属性类型枚举
 *        通过 HcommChannelConfigSetInt  设置。
 */
typedef enum {
    HCOMM_CHANNEL_CONFIG_TYPE_INVALID = -1,
    /** 0: IS_SHARED_QUEUE (bool/int, 默认 0=false)。
     *    仅支持 AIV 引擎的 UB 网络语义协议（UB_CTP/UBC_TP/UB_RTP，不支持 UBMem/RoCE/UBOE）。
     *    设为 true 时，使用 HcommChannelCreateWithConfig 创建的多个 Channel 共享一个 Jetty。 */
    HCOMM_CHANNEL_CONFIG_TYPE_IS_SHARED_QUEUE = 0,
} HcommChannelConfigType;

extern HcommResult HcommChannelCreate(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    ChannelHandle* channels);

/**
 * @brief 创建 Channel 配置对象
 * @param[out] config 输出的配置对象指针（不透明句柄）
 * @return HcommResult 执行结果状态码
 * @note 调用者获得 config 后通过 HcommChannelConfigSetInt/SetStr 设置属性，
 *       传入 HcommChannelCreateWithConfig 后由 HcommChannelConfigDestroy 销毁。
 */
extern HcommResult HcommChannelConfigCreate(HcommChannelConfig* config);

/**
 * @brief 销毁 Channel 配置对象
 * @param[in] config 配置对象指针
 * @return HcommResult 执行结果状态码
 */
extern HcommResult HcommChannelConfigDestroy(HcommChannelConfig config);

/**
 * @brief 设置 Channel 配置的整型属性
 * @param[in] config 配置对象指针
 * @param[in] type  属性类型，参见 HcommChannelConfigType
 * @param[in] value 属性值
 * @return HcommResult 执行结果状态码
 */
extern HcommResult HcommChannelConfigSetInt(HcommChannelConfig config, HcommChannelConfigType type, uint32_t value);

/**
 * @brief 通过配置创建通信通道
 * @param[in] endpointHandle 网络设备句柄
 * @param[in] engine 通信引擎类型
 * @param[in] channelDescs 通道描述参数数组
 * @param[in] channelNum 通道数量
 * @param[in] config Channel 配置对象指针（可为 NULL，等价于 HcommChannelCreate）
 * @param[out] channels 输出的通道句柄数组
 * @return HcommResult 执行结果状态码
 * @note 当 config 中 IS_SHARED_QUEUE=true 时：
 *       - 仅支持 UB 网络语义协议（UB_CTP/UBC_TP/UB_RTP），不支持 UBMem/RoCE/UBOE。
 *       - 使用相同 endpointHandle 多次调用本接口创建的 Channel 共享同一个 Jetty。
 *       - 使用不同 endpointHandle 调用创建的 Channel 不共享 Jetty。
 *       - 共享 Jetty 的不同 Channel 不支持并发使用，需由调用者按业务顺序串行调用。
 *       - 销毁 endpointHandle 前需确保所有共享 Jetty 的 Channel 已销毁。
 */
extern HcommResult HcommChannelCreateWithConfig(
    EndpointHandle endpointHandle, CommEngine engine, HcommChannelDesc* channelDescs, uint32_t channelNum,
    HcommChannelConfig config, ChannelHandle* channels);

extern HcommResult HcommChannelGetStatus(const ChannelHandle* channelList, uint32_t listNum, int32_t* statusList);

extern HcommResult HcommChannelDestroy(const ChannelHandle* channels, uint32_t channelNum);

#ifdef __cplusplus
}
#endif // __cplusplus

#endif // HCOMM_CHANNEL_H
