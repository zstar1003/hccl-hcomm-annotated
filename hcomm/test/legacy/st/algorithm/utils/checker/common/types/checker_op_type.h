/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

/*
 * Description: op type header file
 * Author: zhangqingwei
 * Create: 2024-02-04
 */

#ifndef HCCLV1_OP_TYPE_H
#define HCCLV1_OP_TYPE_H

#include "enum_factory.h"
namespace hccl {

MAKE_ENUM(
    OpType, BROADCAST, ALLREDUCE, REDUCE, SEND, RECEIVE, ALLGATHER, ALLGATHERV, REDUCESCATTER, REDUCESCATTERV,
    ALLTOALLV, ALLTOALLVC, ALLTOALL, GATHER, SCATTER, BATCH_SEND_RECV, RESERVED)

inline std::string OpTypeToString(OpType type) { return type.Describe(); }

} // namespace hccl
#endif
