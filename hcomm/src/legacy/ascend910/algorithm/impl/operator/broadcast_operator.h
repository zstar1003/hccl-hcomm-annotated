/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef BROADCAST_OPERATOR_H
#define BROADCAST_OPERATOR_H

#include "coll_alg_operator.h"

namespace hccl {
class BroadCastOperator : public CollAlgOperator {
public:
    BroadCastOperator(
        AlgConfigurator* algConfigurator, CCLBufferManager& cclBufferManager, HcclDispatcher dispatcher,
        std::unique_ptr<TopoMatcher>& topoMatcher);
    ~BroadCastOperator() override;
    HcclResult
    SelectAlg(const std::string& tag, const OpParam& param, std::string& algName, std::string& newTag) override;

private:
    HcclResult SelectAlgforMix(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor310P3(const OpParam& param, std::string& algName) const;

    HcclResult SelectAlgfor310P(const OpParam& param, std::string& algName) const;

    HcclResult SelectAlgfor910A(const OpParam& param, std::string& algName) const;

    HcclResult SelectAlgfor910B(const OpParam& param, std::string& algName);

    HcclResult SelectAlgfor91093(const OpParam& param, std::string& algName);

    bool isAivMode_ = false;
};
} // namespace hccl

#endif /** _BROADCAST_OPERATOR_H__ */
