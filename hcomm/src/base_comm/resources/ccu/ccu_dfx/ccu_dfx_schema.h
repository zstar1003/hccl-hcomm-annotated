/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_DFX_SCHEMA_H
#define CCU_DFX_SCHEMA_H

#include <cstdint>
#include <sstream>
#include "hccl_types.h" // HcclResult

namespace hcomm {
// CCU mission 解码后的语义字段（V1/V2 共用输出布局）。
struct CcuMissionInfo {
    uint16_t currentIns;
    uint16_t endIns;
    uint16_t startIns;
};

// CCU loop 解码后的语义字段（V1/V2 共用输出布局）。
struct CcuLoopInfo {
    uint16_t currentCnt;
    uint32_t addrStride;
};

// V1/V2 schema 分派表项：按设备类型选取对应的一组解码/打印实现。
struct CcuVersionOps {
    const char* name;
    void (*printCcumDfxInfo)(const void* rawData, std::ostringstream& oss);
    HcclResult (*getMissionInfo)(const void* rawData, CcuMissionInfo* out);
    HcclResult (*getLoopInfo)(const void* rawData, CcuLoopInfo* out);
};

// 按当前设备类型解析出对应的 V1/V2 schema 操作集。
HcclResult GetCcuOps(const CcuVersionOps*& ops);
} // namespace hcomm

#endif // CCU_DFX_SCHEMA_H
