/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AIV_AIV_MODEL_INIT_H
#define AIV_AIV_MODEL_INIT_H

#include "ai_core_stub.h"
#include "ascendc_base_stub.h"

extern "C" void aiv_env_init(
    uint32_t rankId, size_t blockNum, const void* buffIn, uint32_t rankSize, uint64_t input, uint64_t inputSize,
    uint64_t output, uint64_t outputSize, uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase,
    uint64_t cclBufferSize, uint64_t aivCommInfoSize, AivSim::AivOpParam opParam);

extern "C" void aiv_set_block_idx(int64_t blockIdx);

extern "C" void aiv_dump_tasks(uint32_t launchIndex);

#endif // AIV_AIV_MODEL_INIT_H
