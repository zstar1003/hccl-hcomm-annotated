/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_MEM_TRANSPORT_H
#define HCCL_MEM_TRANSPORT_H

#include <stdint.h>
#include <acl/acl.h>
#include "hccl_net_dev.h"
#include "new/hccl_socket.h"
#include "hccl_mem.h"
#include "hccl_task_param.h"
#include "hccl_mem_transport_defs.h"
#include "hccl_network_pub.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* 通信队列句柄 */
typedef void* HcclQueue;

/**
 * @enum HcclQueueMode
 * @brief 通信队列工作模式枚举
 */
typedef enum {
    HCCL_QUEUE_MODE_RESERVED = -1,                 ///< 保留模式
    HCCL_QUEUE_MODE_CCU_POLL_CQ = 0,               ///< CCU轮询完成队列模式
    HCCL_QUEUE_MODE_SW_POLL_CQ_AICPU,              ///< 软件轮询CQ模式，AICPU算子展开，Jetty类型为Normal
    HCCL_QUEUE_MODE_STARS_POLL_CQ_DWQE_CACHE_LOCK, ///< Stars轮询CQ模式，支持DWQE缓存锁定
    HCCL_QUEUE_MODE_STARS_POLL_CQ_AICPU_NORMAL,    ///< Stars轮询CQ模式，AICPU算子常规模式
    HCCL_QUEUE_MODE_STARS_POLL_CQ_PI_TYPE_TRUE     ///< Stars轮询CQ模式，主机端算子下沉（pi_type=1）
} HcclQueueMode;

/**
 * @struct HcclQueueAttr
 * @brief 队列属性配置结构体
 * @var tc - 流量类别（Traffic Class），用于QoS配置
 * @var sl - 服务等级（Service Level），用于网络优先级
 * @var udpSport - UDP源端口号
 * @var mode - 队列工作模式，参考HcclQueueMode枚举
 */
typedef struct {
    uint8_t tc;
    uint8_t sl;
    uint16_t udpSport;
    HcclQueueMode mode;
} HcclQueueAttr;

/**
 * @brief 创建通信队列
 * @param[in] netDev 绑定的网络设备标识
 * @param[in] attr 队列属性配置
 * @param[out] queue 返回创建的队列句柄指针
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclQueueCreate(HcclNetDev netDev, const HcclQueueAttr* attr, HcclQueue* queue);

/**
 * @brief 销毁单个通信队列
 * @param[in] queue 要销毁的队列句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclQueueDestroy(HcclQueue queue);

/**
 * @brief 批量销毁通信队列
 * @param[in] queue 队列句柄数组指针
 * @param[in] queueNum 队列数量
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclQueueBatchDestroy(HcclQueue* queue, uint32_t queueNum);

/**
 * @brief 创建内存传输通道
 * @param[in] netDev 本端网络设备信息
 * @param[in] remoteDevInfo 远端设备信息指针
 * @param[in] queue 绑定的通信队列数组
 * @param[in] queueNum 通信队列数量
 * @param[in] buf 本端缓冲区描述数组
 * @param[in] bufNum 缓冲区数量
 * @param[in] binaryNotify 二进制通知对象指针数组
 * @param[in] binaryNotifyNum 通知对象数量
 * @param[in] socket 网络套接字描述符
 * @param[in] userExchangeInfo 用户自定义交换信息
 * @param[in] exchangeLen 交换信息长度
 * @param[out] memTransport 返回的传输上下文句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportCreate(
    HcclNetDev netDev, HcclNetDevInfos* remoteDevInfo, HcclQueue* queue, uint32_t queueNum, HcclBuf* buf,
    uint32_t bufNum, aclrtNotify* binaryNotify, uint32_t binaryNotifyNum, HcclSocket socket, char* userExchangeInfo,
    uint32_t exchangeLen, HcclMemTransport* memTransport);

/**
 * @brief 销毁内存传输通道
 * @param[in] memTransport 传输上下文句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportDestroy(HcclMemTransport memTransport);

/**
 * @brief 批量移除内存传输通道的队列
 * @param[in] memTransport 传输上下文数组指针
 * @param[in] memTransportNum 传输上下文数量
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportBatchDelQueue(HcclMemTransport* memTransport, uint32_t memTransportNum);

/**
 * @brief 为传输通道添加通信队列
 * @param[in] memTransport 传输上下文句柄
 * @param[in] queue 要添加的队列数组
 * @param[in] queueNum 队列数量
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportAddQueue(HcclMemTransport memTransport, HcclQueue* queue, uint32_t queueNum);

/**
 * @brief 获取传输通道状态
 * @param[in] memTransport 传输上下文句柄
 * @param[out] status 返回状态码指针
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportGetStatus(HcclMemTransport memTransport, int32_t* status);

/**
 * @brief 获取远端缓冲区地址信息
 * @param[in] memTransport 传输上下文句柄
 * @param[out] remoteBuf 返回远端缓冲区数组指针
 * @param[out] bufNum 返回缓冲区数量
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportGetRemoteAddr(HcclMemTransport memTransport, HcclBuf** remoteBuf, uint64_t* bufNum);

/**
 * @brief 获取远端交换信息
 * @param[in] memTransport 传输上下文句柄
 * @param[out] userRemoteExchangeInfo 返回的远端交换信息指针
 * @param[out] infoLen 返回的信息长度
 * @return 执行状态码 HcclResult
 */
extern HcclResult
HcclMemTransportGetRemoteExchangeInfo(HcclMemTransport memTransport, char** userRemoteExchangeInfo, uint32_t* infoLen);

/**
 * @brief 获取传输描述信息
 * @param[in] memTransport 传输上下文句柄
 * @param[out] outDesc 返回的描述信息指针（调用方不要释放）
 * @param[out] outDescLen 返回的描述信息长度
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportExport(HcclMemTransport memTransport, char** outDesc, uint64_t* outDescLen);

/**
 * @brief 通过描述信息重建传输通道
 * @param[in] description 序列化的描述信息
 * @param[in] descLen 描述信息长度
 * @param[out] memTransport 返回的传输上下文句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportImport(char* description, uint32_t descLen, HcclMemTransport* memTransport);

/**
 * @brief 绑定Dfx回调函数
 * @param[in] memTransport 传输上下文句柄
 * @param[in] callback Dfx回调函数指针
 * @param[in] args 回调参数指针
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclMemTransportBindDfxCallback(HcclMemTransport memTransport, HcclDfxCallback callback, void* args);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif
