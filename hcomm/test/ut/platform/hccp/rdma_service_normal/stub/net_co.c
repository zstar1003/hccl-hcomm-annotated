/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdlib.h>
#include "netco_api.h"

#define STUB_NET_CO_PROCED (1987)

void* NetCoInitFactory(int epollfd, NetCoIpPortArg ipPortArg) { return NULL; }

void NetCoDestruct(void* co)
{
    co = NULL;
    return;
}

unsigned int NetCoFdEventDispatch(void* co, int fd, unsigned int curEvents) { return STUB_NET_CO_PROCED; }

int NetCoTblAddUpd(void* netco_handle, unsigned int type, char* data, unsigned int data_len) { return 0; }
