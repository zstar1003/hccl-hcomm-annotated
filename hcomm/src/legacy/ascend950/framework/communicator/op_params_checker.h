/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef OP_PARAMS_CHECKER_H
#define OP_PARAMS_CHECKER_H

#include <bitset>
#include <unordered_map>
#include <hccl/hccl_types.h>
#include "op_type.h"
#include "hccl_params_pub.h"
#include "mc2_type.h"

namespace Hccl {

using DataTypeBitmap = std::bitset<static_cast<int>(HcclDataType::HCCL_DATA_TYPE_RESERVED)>;
using DataTypeSupportMap = std::unordered_map<OpType, DataTypeBitmap, std::EnumClassHash>;

class OpParamsChecker {
public:
    static HcclResult CheckOpDataTypeOpbase(const CollOpParams& opParams, bool ccuEnable, bool isDevUsed, bool isAiv);
    static HcclResult
    CheckOpDataTypeOffload(const CollOpParams& opParams, bool ccuEnable, bool isDevUsed, bool isAiv = false);
    static HcclResult CheckOpDataTypeMC2(const Mc2CommConfig& config);
    static HcclResult CheckOpDataTypeMC2V2(const Mc2CcTilingInner& config);

private:
    static DataType GetDataType(const CollOpParams& opParams);
    static HcclResult CheckOpDataTypeByMap(const CollOpParams& opParams, const DataTypeSupportMap& opData2TypeMap);

    static DataTypeBitmap dataTypeWithReduceAiv;
    static DataTypeBitmap dataTypeWithoutReduceAiv;
    static DataTypeBitmap dataTypeWithReduceCcu;
    static DataTypeBitmap dataTypeWithReduceAicpu;
    static DataTypeBitmap dataTypeWithoutReduce;
    static DataTypeBitmap dataTypeWithoutReduceCcuOpbase;
    static DataTypeBitmap dataTypeWithoutReduceCcuOffload;
    static DataTypeBitmap dataTypeWithReduceHost;
    static DataTypeBitmap dataTypeMC2HighP;
    static DataTypeBitmap inputDataTypeMC2LowP;
    static DataTypeBitmap OutputDataTypeMC2LowP;

    static DataTypeSupportMap opDataTypeSupportMapAivOpbase;
    static DataTypeSupportMap opDataTypeSupportMapAivOffload;
    static DataTypeSupportMap opDataTypeSupportMapCcuOpbase;
    static DataTypeSupportMap opDataTypeSupportMapCcuOffload;
    static DataTypeSupportMap opDataTypeSupportMapAicpuOpbase;
    static DataTypeSupportMap opDataTypeSupportMapAicpuOffload;
    static DataTypeSupportMap opDataTypeSupportMapHostOffload;
    static DataTypeSupportMap opDataTypeSupportMapMC2;
};

} // namespace Hccl

#endif // HCCLV2_OP_PARAMS_CHECKER_H
