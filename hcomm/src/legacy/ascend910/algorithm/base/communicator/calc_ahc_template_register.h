/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include <functional>
#include <memory>

#include "comm_ahc_base_pub.h"

namespace hccl {

// 算子类型到 AHC 通信关系类型的map，新增支持算子类型需要此处添加
const std::map<TemplateType, AHCTemplateType> templateToAHCCalcTemplateMap = {
    {TemplateType::TEMPLATE_REDUCESCATTER_NB, AHCTemplateType::AHC_TEMPLATE_NB},
    {TemplateType::TEMPLATE_ALL_REDUCE_NB, AHCTemplateType::AHC_TEMPLATE_NB},
    {TemplateType::TEMPLATE_ALL_GATHER_NB, AHCTemplateType::AHC_TEMPLATE_NB},

    {TemplateType::TEMPLATE_REDUCESCATTER_RING, AHCTemplateType::AHC_TEMPLATE_RING},
    {TemplateType::TEMPLATE_ALL_REDUCE_RING, AHCTemplateType::AHC_TEMPLATE_RING},
    {TemplateType::TEMPLATE_ALL_GATHER_RING, AHCTemplateType::AHC_TEMPLATE_RING},

    {TemplateType::TEMPLATE_REDUCESCATTER_NHR, AHCTemplateType::AHC_TEMPLATE_NHR},
    {TemplateType::TEMPLATE_ALL_REDUCE_NHR, AHCTemplateType::AHC_TEMPLATE_NHR},
    {TemplateType::TEMPLATE_ALL_GATHER_NHR, AHCTemplateType::AHC_TEMPLATE_NHR},
};

// AHC通信关系注册
using AHCCommCalcFuncPtr = HcclResult (*)(const u32 rank, const std::vector<u32> commGroups, std::set<u32>& dstRanks);
class AHCCommCalcFuncRegistry {
public:
    AHCCommCalcFuncRegistry();
    static AHCCommCalcFuncRegistry& Instance();
    HcclResult Register(AHCTemplateType type, AHCCommCalcFuncPtr funPtr);
    AHCCommCalcFuncPtr GetCommCalcFunction(AHCTemplateType type);

private:
    std::vector<AHCCommCalcFuncPtr> commCalcFuncCreators_;
    mutable std::mutex mu_;
};

// 通信域建链方法函数注册
#define REGISTER_AHC_COMM_CALC_FUNC_HELPER(ctr, type, calcAlgName, calcFunc) \
    static HcclResult g_func_##calcAlgName##_##ctr = AHCCommCalcFuncRegistry::Instance().Register(type, calcFunc)
#define REGISTER_AHC_COMM_CALC_FUNC_HELPER_1(ctr, type, calcAlgName, calcFunc) \
    REGISTER_AHC_COMM_CALC_FUNC_HELPER(ctr, type, calcAlgName, calcFunc)
#define REGISTER_AHC_COMM_CALC_FUNC(type, calcAlgName, calcFunc) \
    REGISTER_AHC_COMM_CALC_FUNC_HELPER_1(__COUNTER__, type, calcAlgName, calcFunc)

} // namespace hccl
