/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PROTOCOL_UTILS_H
#define PROTOCOL_UTILS_H

#include <string>
#include <unordered_map>
#include "hcomm_res_defs.h"
#include "comm_engine_utils.h"

inline const std::unordered_map<CommProtocol, std::string>& GetCommProtocolStrMap()
{
    static const std::unordered_map<CommProtocol, std::string> COMM_PROTOCOL_STR_MAP{
        {COMM_PROTOCOL_RESERVED, "RESERVED"},   {COMM_PROTOCOL_HCCS, "HCCS"},     {COMM_PROTOCOL_ROCE, "ROCE"},
        {COMM_PROTOCOL_PCIE, "PCIE"},           {COMM_PROTOCOL_SIO, "SIO"},       {COMM_PROTOCOL_UB_CTP, "UB_CTP"},
        {COMM_PROTOCOL_UBC_TP, "UBC_TP"},       {COMM_PROTOCOL_UB_MEM, "UB_MEM"}, {COMM_PROTOCOL_UBOE, "UBOE"},
        {COMM_PROTOCOL_HCCS_ONLY, "HCCS_ONLY"}, {COMM_PROTOCOL_UB_RTP, "UB_RTP"}};
    return COMM_PROTOCOL_STR_MAP;
}

#endif
