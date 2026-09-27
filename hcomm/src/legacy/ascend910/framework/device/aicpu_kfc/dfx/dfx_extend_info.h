/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_
#define ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_

#include "aicpu_hccl_def.h"

namespace dfx {
class DfxExtendInfoHelper {
public:
    static void ResetTryRestartTimes(DfxExtendInfo& dfxExtendInfo);
    static void TryRestartOnceMore(DfxExtendInfo& dfxExtendInfo);
    static bool TryRestartTooManyTimes(const DfxExtendInfo& dfxExtendInfo);
};

} // namespace dfx
#endif // ASCEND_ACE_COMOP_HCCL_HCCL_AI_CPU_KERNEL_DFX_DFX_EXTEND_INFO_H_
