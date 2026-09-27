/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stddef.h>
#include <errno.h>
#include "ccu_u_api.h"

int ccu_init(void) { return 0; }

int ccu_uninit(void) { return 0; }

int ccu_custom_channel(const struct channel_info_in* in, struct channel_info_out* out)
{
    if (in == NULL || out == NULL) {
        return -EINVAL;
    }
    return 0;
}

unsigned long long ccu_get_cqe_base_addr(unsigned int die_id) { return 0; }

int ccu_get_mem_info(unsigned int die_id, unsigned long long mem_type_bitmap, struct ccu_mem_rsp* rsp) { return 0; }

int ccu_tlv_request(struct ccu_tlv_request_info* tlvRequestInfo) { return 0; }
