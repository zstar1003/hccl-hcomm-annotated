/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_rep_v1.h"
#include "ccu_kernel.h"
#include "ccu_interface_assist_v1.h"

#include "string_util.h"
#include "ccu_api_exception.h"

namespace hcomm {
namespace CcuRep {

    FuncBlock::FuncBlock(CcuRepContext* context, std::string label, uint16_t callLayer)
        : context(context),
          label(label),
          callLayer(callLayer)
    {}

    FuncBlock::~FuncBlock() {}

}; // namespace CcuRep
}; // namespace hcomm
