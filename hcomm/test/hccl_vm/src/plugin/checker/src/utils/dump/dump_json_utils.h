/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_VM_DUMP_JSON_UTILS_H
#define HCCL_VM_DUMP_JSON_UTILS_H

#include <map>
#include <nlohmann_json/json.hpp>
#include <string>

#include "check_utils.h"

namespace HcclSim {
std::string DumpTaskTypeToString(TaskTypeStub taskType);
std::string DumpBufferTypeToString(BufferType bufferType);
std::string DumpReduceOpToString(HcclReduceOp reduceOp);

nlohmann::json DumpDataSliceToJson(const DataSlice& dataSlice);
nlohmann::json DumpSrcBufToJson(const SrcBufDes& srcBuf);
nlohmann::json DumpBufferSemanticToJson(
    const BufferSemantic& bufferSemantic, BufferType bufferType,
    const std::map<u32, u32>* globalStepToEventId = nullptr);
} // namespace HcclSim

#endif // HCCL_VM_DUMP_JSON_UTILS_H
