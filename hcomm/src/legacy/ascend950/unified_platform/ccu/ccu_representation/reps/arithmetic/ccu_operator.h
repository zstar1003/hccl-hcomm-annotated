/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_OPERATOR
#define HCCL_CCU_OPERATOR

#include "exception_util.h"
#include "ccu_api_exception.h"

namespace Hccl {
namespace CcuRep {

    template <typename lhsT, typename rhsT>
    class CcuOperator {
    public:
        CcuOperator(lhsT lhs, rhsT rhs) : lhs(lhs), rhs(rhs) {}
        lhsT lhs;
        rhsT rhs;
    };

    enum class CcuArithmeticOperatorType { ADDITION, INVALID };

    template <typename lhsT, typename rhsT>
    class CcuArithmeticOperator : public CcuOperator<lhsT, rhsT> {
    public:
        CcuArithmeticOperator(lhsT lhs, rhsT rhs, CcuArithmeticOperatorType type)
            : CcuOperator<lhsT, rhsT>(lhs, rhs),
              type(type)
        {
            Check();
        }
        void Check() const { THROW<CcuApiException>("Invalid Arithmetic Operator"); }

        CcuArithmeticOperatorType type{CcuArithmeticOperatorType::INVALID};
    };

    enum class CcuRelationalOperatorType { EQUAL, NOT_EQUAL, INVALID };

    template <typename lhsT, typename rhsT>
    class CcuRelationalOperator : public CcuOperator<lhsT, rhsT> {
    public:
        CcuRelationalOperator(lhsT lhs, rhsT rhs, CcuRelationalOperatorType type)
            : CcuOperator<lhsT, rhsT>(lhs, rhs),
              type(type)
        {
            Check();
        }
        void Check() const { THROW<CcuApiException>("Invalid Relational Operator"); }

        CcuRelationalOperatorType type{CcuRelationalOperatorType::INVALID};
    };

}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_OPERATOR
