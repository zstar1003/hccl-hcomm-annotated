/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLL_OPERATOR_CHECK_H
#define HCCLV2_COLL_OPERATOR_CHECK_H

#include <string>
#include "coll_operator.h"
namespace Hccl {

void ReportOpCheckFailed(const std::string& paraName, const std::string& localPara, const std::string& remotePara);

void ReportOpCheckFailed(const std::string& paraName, uint32_t localPara, uint32_t remotePara);

void CompareDataDesOp(const CollOperator& localOpData, const CollOperator& remoteOpData);

void CompareVDataDesOp(const CollOperator& localOpData, const CollOperator& remoteOpData);

void CompareAlltoAllOp(const CollOperator& localOpData, const CollOperator& remoteOpData);

void CompareAlltoAllVOp(const CollOperator& localOpData, const CollOperator& remoteOpData);

void CompareAlltoAllVCOp(const CollOperator& localOpData, const CollOperator& remoteOpData);

void CheckCollOperator(const CollOperator& localOpData, const CollOperator& remoteOpData);

} // namespace Hccl
#endif
