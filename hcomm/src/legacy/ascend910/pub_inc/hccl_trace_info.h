/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_TRACE_INFO_H
#define HCCL_TRACE_INFO_H

#include "hccl_common.h"
#include "hccl/base.h"
#include <string>
namespace hccl {
enum class AtraceOption { Opbasekey, Algtype };

class HcclTraceInfo {
public:
    enum class HcclTraceType { HostTraceType, DeviceTraceType };

    struct UtraceAttr {
        bool utraceStatusFlag;
        u32 pid;
        u32 deviceid;
    };

    HcclTraceInfo();
    HcclTraceInfo(const UtraceAttr& utraceAttr);
    ~HcclTraceInfo();
    HcclResult Init(std::string& logInfo);
    void DeInit();
    HcclResult Flush();
    HcclResult SaveTraceInfo(std::string& logInfo, AtraceOption op);
    HcclResult SavealgtypeTraceInfo(std::string& algtype, const std::string& tag);
    HcclTraceType hcclTraceType_ = HcclTraceType::HostTraceType;
    UtraceAttr utraceAttr_{};
    uint32_t index{0};
    HcclTraHandle handle{0};
};
} // namespace hccl
#endif // HCCL_TRACE_INFO_H
