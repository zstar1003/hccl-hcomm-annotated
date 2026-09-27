/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_DFX_H
#define HCCL_CCU_DFX_H

#include <vector>
#include <string>
#include "hccl/base.h"
#include "hccl_types.h"
#include "task_param.h"
#include "ccu_error_info.h"
#include "ccu_jetty.h"

namespace Hccl {

HcclResult GetCcuErrorMsg(
    s32 deviceId, uint16_t status, const ParaCcu& ccuTaskParam, const std::string& groupRankContent,
    std::vector<CcuErrorInfo>& errorInfo);
HcclResult GetCcuJettys(s32 deviceLogicId, const ParaCcu& ccuTaskParam, std::vector<CcuJetty*>& ccuJettys);

} // namespace Hccl
#endif // HCCL_CCU_DFX_H
