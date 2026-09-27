/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ERROR_CODES_H
#define ERROR_CODES_H

#include <cstdint>
#include <string>

namespace HcclSim {

enum class ErrorCode {
    GRAPH_TRANSLATE_FAILED = 101,
    GRAPH_DEADLOCK = 102,
    GRAPH_UNMATCHED = 103,
    GRAPH_MEMBER_MISSING = 104,
    GRAPH_STRUCTURE_INVALID = 105,
    GRAPH_SNAPSHOT_MISMATCH = 106,
    GRAPH_RESOURCE_NOT_FOUND = 107,
    GRAPH_REGISTER_UNINITIALIZED = 108,
    GRAPH_OUT_OF_RANGE = 109,
    GRAPH_ADDRESS_INVALID = 110,
    GRAPH_UNSUPPORTED = 111,
    GRAPH_REMOTE_RANK_MISMATCH = 112,
    GRAPH_LOOP_MERGE_ERROR = 113,

    SINGLETASK_SLICE_INVALID = 201,
    SINGLETASK_SLICE_CONFLICT = 202,
    SINGLETASK_SLAVE_STREAM_INVALID = 203,

    MEMCONFLICT_DAG_INVALID = 301,
    MEMCONFLICT_DETECTED = 302,

    SEMANTIC_BUFFER_EMPTY = 401,
    SEMANTIC_GAP = 402,
    SEMANTIC_REDUCE_ERROR = 403,
    SEMANTIC_SIMULATE_FAILED = 404,
    SEMANTIC_FINAL_CHECK_FAILED = 405,
    SEMANTIC_FINAL_MISSING = 406,
    SEMANTIC_FINAL_SRC_ERROR = 407,
    SEMANTIC_FINAL_SIZE_ERROR = 408,
    SEMANTIC_FINAL_REDUCE_ERROR = 409,

    DUMP_FAILED = 501,

    SYNC_DAG_INVALID = 601,
    SYNC_RESOURCE_CONFLICT = 602,

    CHECKER_RUNTIME_ERROR = 901,
    SETTING_WARNING = 902
};

constexpr uint32_t ToErrorCodeValue(ErrorCode code) { return static_cast<uint32_t>(code); }

inline std::string MakeErrorCodeText(ErrorCode code)
{
    return "[ErrorCode: " + std::to_string(ToErrorCodeValue(code)) + "]";
}

} // namespace HcclSim

#endif
