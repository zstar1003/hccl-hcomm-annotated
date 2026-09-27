/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RS_ADP_NSLB_H
#define RS_ADP_NSLB_H

#include "rs_inner.h"

#define NET_CO_PROCED (1987)
#define NETCO_CFGFILE_PATH "/etc/hccl.cfg"
#define NETCO_PORT_NUM_BASE 10
#define CFG_VAL_LEN 16

#define NETCO_REQ_TYPE_INIT 9001
#define NETCO_REQ_TYPE_DEINIT 9002

int RsNslbNetcoRequest(unsigned int phyId, struct RsNslbCb *nslbCb, unsigned int type, char *data,
    unsigned int dataLen);
int RsEpollNslbEventHandle(struct RsNslbCb *nslbCb, int fd, unsigned int events);
#endif // RS_ADP_NSLB_H
