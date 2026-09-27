/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_TRACE_SERIALIZER_H
#define HCCL_SIM_CCU_TRACE_SERIALIZER_H

#include <string>

#include "ccu_trace_types.h"

namespace CcuTrace {

class CcuTraceSerializer {
public:
    // 将 CcuTraceRun 序列化为 JSON 字符串
    static std::string SerializeToJson(const CcuTraceRun& traceRun);

    // 将 JSON 字符串写入文件
    static bool WriteToFile(const std::string& jsonStr, const std::string& outputPath);

    // 一步完成：序列化并写入文件
    static bool DumpToFile(const CcuTraceRun& traceRun, const std::string& outputPath);

private:
    CcuTraceSerializer() = delete;

    static std::string SerializeMetadata(const CcuRunMetadata& metadata);
    static std::string SerializeCcuRegistry(const std::vector<CcuIdentity>& registry);
    static std::string SerializeSqeTaskRegistry(const std::vector<CcuSqeTask>& registry);
    static std::string SerializeInstrSpaces(const std::vector<CcuInstrSpace>& instrSpaces);
    static std::string SerializeChannelSpaces(const std::vector<CcuChannelSpace>& channelSpaces);
    static std::string SerializeGlobalEntries(const std::vector<CcuTraceEntry>& entries);
    static std::string SerializeNonCcuEntries(const std::vector<CcuTraceNonCcuEntry>& entries);
    static std::string SerializeFinalSnapshots(const std::map<std::string, CcuResourceSnapshot>& snapshots);
    static std::string SerializeSummary(const CcuRunSummary& summary);

    static std::string SerializeResourceSnapshot(const CcuResourceSnapshot& snapshot);
    static std::string SerializeResourceDelta(const CcuResourceDelta& delta);
    static std::string SerializeContext(const CcuExecutionContext& ctx);
    static std::string SerializeWaitInfo(const CcuWaitInfo& waitInfo);
    static std::string SerializeErrorInfo(const CcuErrorInfo& errorInfo);
    static std::string SerializeCrossCcuChanges(const CcuCrossCcuChanges& changes);
    static std::string SerializeDetail(const CcuInstrTraceDetail& detail);

    // JSON 辅助函数
    static std::string EscapeJsonString(const std::string& s);
    static std::string ToHexString(uint64_t value);
    static std::string ToHexString(uint16_t value);
};

} // namespace CcuTrace

#endif // HCCL_SIM_CCU_TRACE_SERIALIZER_H
