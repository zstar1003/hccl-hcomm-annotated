/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_RES_TYPE_CONVERTER_H
#define HCOMM_CCU_RES_TYPE_CONVERTER_H

#include <array>
#include <cstddef>

#include "ccu_res_defs.h"
#include "ccu_types.h"
#include "unified_platform/ccu/ccu_device/ccu_device_manager.h"

namespace hcomm {

inline CcuResult ConvertHcommCcuResTypeToHcclResType(HcommCcuResType ccuResType, ResType& hcclResType)
{
    static constexpr auto RES_TYPE_MAP = std::array{
        ResType::LOOP,    // <- HCOMM_CCU_RES_TYPE_LOOP
        ResType::MS,      // <- HCOMM_CCU_RES_TYPE_CCU_BUF
        ResType::XN,      // <- HCOMM_CCU_RES_TYPE_VARIABLE
        ResType::GSA,     // <- HCOMM_CCU_RES_TYPE_ADDRESS
        ResType::CKE,     // <- HCOMM_CCU_RES_TYPE_EVENT
        ResType::MISSION, // <- HCOMM_CCU_RES_TYPE_CCU_THREAD
        ResType::INS,     // <- HCOMM_CCU_RES_TYPE_INSTRUCTION
        ResType::CASC_CNT // <- HCOMM_CCU_RES_TYPE_CASC_CNT
    };
    const int32_t resTypeIndex = static_cast<int32_t>(ccuResType);
    if ((resTypeIndex < 0) || (static_cast<std::size_t>(resTypeIndex) >= RES_TYPE_MAP.size())) {
        return CcuResult::CCU_E_PARA;
    }
    hcclResType = RES_TYPE_MAP[static_cast<std::size_t>(resTypeIndex)];
    return CcuResult::CCU_SUCCESS;
}

inline CcuResult ConvertHcclResTypeToHcommCcuResType(ResType hcclResType, HcommCcuResType& ccuResType)
{
    static constexpr std::array<HcommCcuResType, static_cast<std::size_t>(ResType::__COUNT__)> RES_TYPE_MAP = {
        HCOMM_CCU_RES_TYPE_LOOP,        // <- ResType::LOOP
        HCOMM_CCU_RES_TYPE_CCU_BUF,     // <- ResType::MS
        HCOMM_CCU_RES_TYPE_EVENT,       // <- ResType::CKE
        HCOMM_CCU_RES_TYPE_VARIABLE,    // <- ResType::XN
        HCOMM_CCU_RES_TYPE_INVALID,     // <- ResType::COUNT_XN
        HCOMM_CCU_RES_TYPE_ADDRESS,     // <- ResType::GSA
        HCOMM_CCU_RES_TYPE_INSTRUCTION, // <- ResType::INS
        HCOMM_CCU_RES_TYPE_CCU_THREAD,  // <- ResType::MISSION
        HCOMM_CCU_RES_TYPE_CASC_CNT     // <- ResType::CASC_CNT
    };
    const auto resTypeIndex = static_cast<std::size_t>(static_cast<ResType::Value>(hcclResType));
    if ((resTypeIndex >= RES_TYPE_MAP.size()) || (RES_TYPE_MAP[resTypeIndex] == HCOMM_CCU_RES_TYPE_INVALID)) {
        return CcuResult::CCU_E_PARA;
    }
    ccuResType = RES_TYPE_MAP[resTypeIndex];
    return CcuResult::CCU_SUCCESS;
}

} // namespace hcomm

#endif // HCOMM_CCU_RES_TYPE_CONVERTER_H
