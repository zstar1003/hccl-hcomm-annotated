/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __HCCL_AICPU_TRANSPORT_INTERFACE_H__
#define __HCCL_AICPU_TRANSPORT_INTERFACE_H__

#include <cstdint>

extern "C" {
__attribute__((visibility("default"))) uint32_t RunTransportRoceTx(void* args);

__attribute__((visibility("default"))) uint32_t RunTransportRoceRx(void* args);
}
#endif // __MC2_AICPU_INTERFACE_HPP__
