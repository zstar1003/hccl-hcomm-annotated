/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "zero_copy_address_mgr.h"

namespace hccl {
HcclResult ZeroCopyAddressMgr::InitRingBuffer() { return HCCL_SUCCESS; }

HcclResult ZeroCopyAddressMgr::PushOne([[maybe_unused]] ZeroCopyRingBufferItem& item)
{
    HCCL_DEBUG("[ZeroCopyAddressMgr][PushOne] device");
    return HCCL_SUCCESS;
}
} // namespace hccl
