/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_SOCKET_H
#define RS_SOCKET_H
#include "rs_drv_socket.h"

#define RS_MAX_VNIC_NUM 16
#define RS_VNIC_MAX 128
#define RS_VNIC_FLAG 1
#define RS_MAX_SOCKET_NUM 16
#define RS_MAX_WLIST_NUM 16
#define RS_SOCK_LISTEN_PARALLEL_NUM 16384
#define RS_WLIST_VALID_FLAG_SIZE 6
#define RS_CLOSE_TIMEOUT 5

#define RS_SOCKET_PARA_CHECK(num, conn)                                                                                \
    do {                                                                                                               \
        if (((num) <= 0) || ((num) > RS_MAX_SOCKET_NUM) || ((conn) == NULL)) {                                         \
            hccp_err("rs socket param error ! number:%d", num);                                                        \
            return (-EINVAL);                                                                                          \
        }                                                                                                              \
    } while (0)

struct RsVnicInfo {
    uint32_t vnicFlag;
    uint32_t role;
};

int RsSocketNodeid2vnic(uint32_t nodeId, uint32_t *ipAddr);
int RsSocketConnectAsync(struct RsConnInfo *conn, struct rs_cb *rscb);
int RsGetSocketConnectState(struct RsConnInfo *conn);
void RsSocketSaveErrInfo(int action, int errNo, struct SocketErrInfo *errInfo);
#endif // RS_SOCKET_H
