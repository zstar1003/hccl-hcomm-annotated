/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/**
 * Stub file: provides no-op replacements for functions that mockcpp can't hook,
 * allowing test coverage of hccl_group.cc without real communicator setup.
 * -fno-access-control is enabled, so we can set private members directly.
 */
#include "hccl_comm_pub.h"
#include "op_base.h"
#include "adapter_rts_common.h"

namespace hccl {

HcclResult hcclComm::SetGroupMode(bool isGroup)
{
    isGroupMode_ = isGroup;
    return HCCL_SUCCESS;
}

} // namespace hccl

HcclResult HcclBatchSendRecvGroup(HcclSendRecvItem*, uint32_t, HcclComm, aclrtStream) { return HCCL_SUCCESS; }

HcclResult hcclStreamSynchronize(HcclRtStream, s32) { return HCCL_SUCCESS; }
