/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TRANSFORM_FUNS_TYPE_H
#define HCCLV2_TRANSFORM_FUNS_TYPE_H

#include <map>
#include "checker_def.h"
#include "op_type.h"
#include "reduce_op.h"
#include "data_type.h"
#include "port.h"
#include "op_mode.h"
#include "proto_stub.h"
#include "buffer_type.h"
#include "checker_buffer_type.h"
#include "dev_type.h"

using namespace checker;

namespace Hccl {

extern std::map<CheckerOpType, OpType> g_CheckerOpType2OpType_aicpu;
extern std::map<OpType, CheckerOpType> g_OpType2CheckerOpType_aicpu;
extern std::map<CheckerReduceOp, ReduceOp> g_CheckerReduceOp2ReduceOp_aicpu;
extern std::map<ReduceOp, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_aicpu;
extern std::map<CheckerDataType, DataType> g_CheckerDataType2DataType_aicpu;
extern std::map<DataType, CheckerDataType> g_DataType2CheckerDataType_aicpu;
extern std::map<LinkProtoType, LinkProtoStub> g_LinkProtoType2LinkProtoStub_aicpu;
extern std::map<CheckerOpMode, OpMode> g_CheckerOpMode2OpMode_aicpu;
extern std::map<Hccl::BufferType, checker::BufferType> g_HcclBufferType2CheckerBufferType_aicpu;
extern std::map<CheckerDevType, Hccl::DevType> g_CheckerDevType2HcclDevType_aicpu;

extern std::map<uint16_t, CheckerReduceOp> g_ReduceOp2CheckerReduceOp_ccu;
extern std::map<uint16_t, CheckerDataType> g_DataType2CheckerDataType_ccu;

} // namespace Hccl

#endif // HCCLV2_TRANSFORM_FUNS_TYPE_H
