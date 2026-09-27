/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CONTEXT_RESOURCE_H
#define HCCL_CCU_CONTEXT_RESOURCE_H

#include <vector>
#include <array>
#include <unordered_map>

#include "ccu_datatype.h"
#include "ccu_device_manager.h"

namespace Hccl {

// Context资源
struct CcuRepResource {
    std::array<std::vector<CcuRep::CcuBuffer>, MAX_CCU_IODIE_NUM> ccubuffers;
    std::array<std::vector<CcuRep::CcuBuffer>, MAX_CCU_IODIE_NUM> blockCcubuffers;
    std::array<std::vector<CcuRep::Executor>, MAX_CCU_IODIE_NUM> executor;
    std::array<std::vector<CcuRep::Executor>, MAX_CCU_IODIE_NUM> blockExecutor;
    std::array<std::vector<CcuRep::MaskSignal>, MAX_CCU_IODIE_NUM> maskSignal;
    std::array<std::vector<CcuRep::MaskSignal>, MAX_CCU_IODIE_NUM> blockMaskSignal;
    std::array<std::vector<CcuRep::Address>, MAX_CCU_IODIE_NUM> address;
    std::array<std::vector<CcuRep::Address>, MAX_CCU_IODIE_NUM> blockAddress;
    std::array<std::vector<CcuRep::Variable>, MAX_CCU_IODIE_NUM> continuousVariable;
    std::array<std::vector<CcuRep::Variable>, MAX_CCU_IODIE_NUM> variable;
};

// Context共享资源
struct CcuSharedResource {
    std::unordered_map<std::string, CcuRep::Variable> sharedVar;
    std::unordered_map<std::string, CcuRep::MaskSignal> sharedSig;
};

}; // namespace Hccl

#endif // HCCL_CCU_CONTEXT_RESOURCE_H
