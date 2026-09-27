/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SOCKET_H
#define HCCL_SOCKET_H

#include <stdint.h>
#include <arpa/inet.h>
#include "hccl_net_dev.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/* Socket通信句柄（不透明指针） */
typedef void* HcclSocket;

const uint32_t HCCL_SOCK_CONN_TAG_MAX_SIZE = 192; ///< 握手标识最大长度（含终止符）

/**
 * @brief 创建Socket通信实例
 * @param[in] nicDeployment 网卡部署位置（参考HcclNetDevDeployment枚举）
 * @param[in] devicePhyId 物理设备ID
 * @param[in] domain 协议域（如AF_INET/AF_INET6）
 * @param[in] addr 绑定的本地地址（支持IPv4/IPv6）
 * @param[in] addrlen 地址结构体长度
 * @param[out] socket 输出的socket句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketCreate(
    HcclNetDevDeployment nicDeployment, int32_t devicePhyId, int domain, const struct sockaddr* addr, socklen_t addrlen,
    HcclSocket* socket);

/**
 * @brief 关闭Socket句柄并释放资源
 * @param[in] socket 要关闭的socket句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketClose(HcclSocket socket);

/**
 * @brief 设置监听模式（服务端用）
 * @param[in] socket 已创建的socket句柄
 * @param[in] backlog 最大挂起连接数（参考系统SOMAXCONN）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketListen(HcclSocket socket, int32_t backlog);

/**
 * @brief 接受客户端连接（非阻塞操作）
 * @param[in] serverSocket 处于监听状态的服务器socket句柄
 * @param[in] handShakeTag 握手标识
 * @param[in] tagLen 标识长度（必须<=HCCL_SOCK_CONN_TAG_MAX_SIZE）
 * @param[out] socket 输出的新连接句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketAccept(void* serverSocket, char* handShakeTag, uint32_t tagLen, HcclSocket* socket);

/**
 * @brief 客户端发起连接请求（非阻塞操作）
 * @param[in] socket 已初始化的socket句柄
 * @param[in] addr 目标服务器地址
 * @param[in] addrlen 地址结构体长度
 * @param[in] handShakeTag 握手标识
 * @param[in] tagLen 标识长度（必须<=HCCL_SOCK_CONN_TAG_MAX_SIZE）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketConnect(
    HcclSocket socket, const struct sockaddr* addr, socklen_t addrlen, char* handShakeTag, uint32_t tagLen);

/**
 * @brief 获取Socket连接状态
 * @param[in] socket 目标socket句柄
 * @param[out] status 状态码输出（0表示正常）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketGetStatus(HcclSocket socket, int32_t* status);

/**
 * @brief 发送数据（非阻塞）
 * @param[in] socket 已连接的socket句柄
 * @param[in] data 待发送数据缓冲区
 * @param[in] len 待发送数据长度（字节）
 * @param[out] sendLen 实际发送数据长度（可能部分发送）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketISend(HcclSocket socket, void* data, uint64_t len, uint64_t* sendLen);

/**
 * @brief 接收数据（非阻塞）
 * @param[in] socket 已连接的socket句柄
 * @param[out] recvBuf 接收数据缓冲区
 * @param[in] len 缓冲区最大容量
 * @param[out] recvLen 实际接收数据长度
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketIRecv(HcclSocket socket, void* recvBuf, uint64_t len, uint64_t* recvLen);

/**
 * @struct SocketWlistInfo
 * @brief 白名单配置信息结构体
 * @var remoteAddr   - 允许连接的远端地址
 * @var connMaxNum   - 最大允许连接数（0表示无限制）
 * @var handShakeTag - 握手标识字符串（必须以'\0'结尾）
 */
typedef struct {
    HcclAddr remoteAddr;                            ///< 允许的远端地址
    uint32_t connMaxNum;                            ///< 最大连接数限制
    char handShakeTag[HCCL_SOCK_CONN_TAG_MAX_SIZE]; ///< 握手标识
} SocketWlistInfo;

/**
 * @brief 启用白名单认证功能
 * @param[in] socket 目标socket句柄（需处于未连接状态）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketEnableWhiteList(HcclSocket socket);

/**
 * @brief 禁用白名单认证功能
 * @param[in] socket 目标socket句柄
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketDisableWhiteList(HcclSocket socket);

/**
 * @brief 添加白名单规则
 * @param[in] socket 已启用白名单的socket句柄
 * @param[in] whitelists 规则数组（支持批量添加）
 * @param[in] num 规则数量（数组长度）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketAddWhiteList(HcclSocket socket, SocketWlistInfo* whitelists, uint32_t num);

/**
 * @brief 删除白名单规则
 * @param[in] socket 已启用白名单的socket句柄
 * @param[in] whitelists 待删除规则数组
 * @param[in] num 规则数量（数组长度）
 * @return 执行状态码 HcclResult
 */
extern HcclResult HcclSocketDelWhiteList(HcclSocket socket, SocketWlistInfo* whitelists, uint32_t num);

#ifdef __cplusplus
}
#endif // __cplusplus
#endif
