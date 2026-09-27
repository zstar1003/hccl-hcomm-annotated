/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef NET_CO_MAIN_H
#define NET_CO_MAIN_H

#include "bkf_comm.h"
#include "bkf_url.h"

#ifdef __cplusplus
extern "C" {
#endif
#pragma pack(4)
typedef struct TagNetCo NetCo;

typedef struct TagNetCoInitArg {
    int epFd;             /* epoll fd */
    BkfUrl inConnSelfUrl; /* 计算发起连接的计算地址。包括端口号，下同 */
    BkfUrl inConnNetUrl;  /* 计算发起连接的交换机地址 */
    BkfUrl outConnLsnUrl; /* 计算监听的地址 */
    uint8_t rsv1[0x10];
} NetCoInitArg;

#define NET_CO_PROCED (1987)
#pragma pack()

/**
 * @brief net co 模块初始化
 *
 * @param[in] *arg 参数
 * @return 初始化是否成功
 *   @retval 非空 成功
 *   @retval 空 失败
 * @note net co所有代码不可重入。调用函数时，如果涉及多线程，需要调用者确保调用线程安全（如加锁）。
 */
NetCo *NetCoInit(NetCoInitArg *arg);

/**
 * @brief net co 模块去初始化
 *
 * @param[in] co init返回的句柄
 * @return 无
 * @note 在处理流程中，会同步（调用阻塞式API）拆除计算和交换机的连接。
 */
void NetCoUninit(NetCo *co);

/**
 * @brief net co 模块fd消息派发处理函数
 *
 * @param[in] co 句柄
 * @param[in] fd 当前发生事件（如epoll_wait返回）的io fd
 * @param[in] curEvents 当前fd的发生的事件
 * @return net co模块是否已经处理
 *   @retval NET_CO_PROCED 已经处理，外部不需要再处理
 *   @retval 非NET_CO_PROCED 非net co模块创建的fd，未处理
 */
uint32_t NetCoFdEvtDispatch(NetCo *co, int fd, uint32_t curEvents);

#ifdef __cplusplus
}
#endif

#endif
