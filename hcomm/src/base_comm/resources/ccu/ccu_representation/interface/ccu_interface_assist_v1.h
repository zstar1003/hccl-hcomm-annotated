/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INTERFACE_H
#define CCU_INTERFACE_H

#include <memory>

#include "ccu_rep_context_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_datatype_v1.h"

namespace hcomm {
namespace CcuRep {

    void AppendToContext(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBase> rep);
    std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock(CcuRepContext* context);
    void SetCurrentBlock(CcuRepContext* context, std::shared_ptr<CcuRep::CcuRepBlock> repBlock);
    Variable CreateVariable(CcuRepContext* context);

}; // namespace CcuRep
}; // namespace hcomm
#endif // _CCU_INTERFACE_H
