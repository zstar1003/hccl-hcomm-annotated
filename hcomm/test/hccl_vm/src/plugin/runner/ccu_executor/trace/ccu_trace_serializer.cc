/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_trace_serializer.h"

#include <fstream>
#include <iomanip>
#include <sstream>

#include "sim_log.h"

namespace CcuTrace {

std::string CcuTraceSerializer::SerializeToJson(const CcuTraceRun& traceRun)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"runMetadata\": " << SerializeMetadata(traceRun.runMetadata) << ",\n";
    oss << "  \"ccuRegistry\": " << SerializeCcuRegistry(traceRun.ccuRegistry) << ",\n";
    oss << "  \"sqeTaskRegistry\": " << SerializeSqeTaskRegistry(traceRun.sqeTaskRegistry) << ",\n";
    oss << "  \"instrSpaces\": " << SerializeInstrSpaces(traceRun.instrSpaces) << ",\n";
    oss << "  \"channelSpaces\": " << SerializeChannelSpaces(traceRun.channelSpaces) << ",\n";
    oss << "  \"globalEntries\": " << SerializeGlobalEntries(traceRun.globalEntries) << ",\n";
    oss << "  \"nonCcuEntries\": " << SerializeNonCcuEntries(traceRun.nonCcuEntries) << ",\n";
    oss << "  \"ccuFinalSnapshots\": " << SerializeFinalSnapshots(traceRun.ccuFinalSnapshots) << ",\n";
    oss << "  \"runSummary\": " << SerializeSummary(traceRun.runSummary) << "\n";
    oss << "}";
    return oss.str();
}

bool CcuTraceSerializer::WriteToFile(const std::string& jsonStr, const std::string& outputPath)
{
    std::ofstream ofs(outputPath);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("Failed to open file: {}", outputPath);
        return false;
    }
    ofs << jsonStr;
    ofs.close();
    HCCL_VM_INFO("Trace written to: {}, size={} bytes", outputPath, jsonStr.size());
    return true;
}

bool CcuTraceSerializer::DumpToFile(const CcuTraceRun& traceRun, const std::string& outputPath)
{
    std::string json = SerializeToJson(traceRun);
    return WriteToFile(json, outputPath);
}

std::string CcuTraceSerializer::SerializeMetadata(const CcuRunMetadata& m)
{
    std::ostringstream oss;
    oss << "{\n";
    oss << "    \"traceFormatVersion\": " << m.traceFormatVersion << ",\n";
    oss << "    \"algorithmName\": \"" << EscapeJsonString(m.algorithmName) << "\",\n";
    oss << "    \"operatorName\": \"" << EscapeJsonString(m.operatorName) << "\",\n";
    oss << "    \"rankSize\": " << m.rankSize << ",\n";
    oss << "    \"diePerRank\": " << m.diePerRank << ",\n";
    oss << "    \"ccuVersion\": \"" << EscapeJsonString(m.ccuVersion) << "\",\n";
    oss << "    \"runTimestampNs\": " << m.runTimestampNs << ",\n";
    oss << "    \"totalSqeTaskCount\": " << m.totalSqeTaskCount << ",\n";
    oss << "    \"totalExecRounds\": " << m.totalExecRounds << "\n";
    oss << "  }";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeResourceSnapshot(const CcuResourceSnapshot& snap)
{
    std::ostringstream oss;
    oss << "{";

    // XN records
    oss << "\"xnRecords\": [";
    for (size_t i = 0; i < snap.xnRecords.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << snap.xnRecords[i].id << ", \"value\": \"" << ToHexString(snap.xnRecords[i].value)
            << "\"}";
    }
    oss << "], ";

    // GSA records
    oss << "\"gsaRecords\": [";
    for (size_t i = 0; i < snap.gsaRecords.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << snap.gsaRecords[i].id << ", \"value\": \"" << ToHexString(snap.gsaRecords[i].value)
            << "\"}";
    }
    oss << "], ";

    // CKE records
    oss << "\"ckeRecords\": [";
    for (size_t i = 0; i < snap.ckeRecords.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << snap.ckeRecords[i].id << ", \"value\": " << snap.ckeRecords[i].value << "}";
    }
    oss << "]";

    // 注：Channel 已移至 channelSpaces（静态配置，与 instrSpaces 并列）

    oss << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeCcuRegistry(const std::vector<CcuIdentity>& registry)
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < registry.size(); i++) {
        const auto& ccu = registry[i];
        oss << "    {\"rankId\": " << ccu.rankId << ", \"dieId\": " << ccu.dieId << ", \"ccuVersion\": \""
            << EscapeJsonString(RunnerCcuVersionToString(ccu.ccuVersion)) << "\""
            << ", \"initialSnapshot\": " << SerializeResourceSnapshot(ccu.initialSnapshot) << "}";
        if (i + 1 < registry.size())
            oss << ",";
        oss << "\n";
    }
    oss << "  ]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeSqeTaskRegistry(const std::vector<CcuSqeTask>& registry)
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < registry.size(); i++) {
        const auto& task = registry[i];
        oss << "    {\"sqeTaskId\": " << task.sqeTaskId << ", \"rankId\": " << task.rankId
            << ", \"dieId\": " << task.dieId << ", \"missionId\": " << static_cast<int>(task.missionId)
            << ", \"instStartId\": " << task.instStartId << ", \"instCnt\": " << task.instCnt
            << ", \"key\": " << task.key << ", \"args\": [";
        for (size_t j = 0; j < task.args.size(); j++) {
            if (j > 0)
                oss << ", ";
            oss << "\"" << ToHexString(task.args[j]) << "\"";
        }
        oss << "]"
            << ", \"simulatorPtr\": \"" << ToHexString(task.simulatorPtr) << "\""
            << ", \"firstExecRound\": " << task.firstExecRound << "}";
        if (i + 1 < registry.size())
            oss << ",";
        oss << "\n";
    }
    oss << "  ]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeInstrSpaces(const std::vector<CcuInstrSpace>& instrSpaces)
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < instrSpaces.size(); i++) {
        const auto& space = instrSpaces[i];
        oss << "    {\"rankId\": " << space.rankId << ", \"dieId\": " << space.dieId << ", \"instructions\": [\n";
        for (size_t j = 0; j < space.instructions.size(); j++) {
            const auto& instr = space.instructions[j];
            oss << "      {\"instrId\": " << instr.instrId << ", \"instrDescribe\": \""
                << EscapeJsonString(instr.instrDescribe) << "\"}";
            if (j + 1 < space.instructions.size())
                oss << ",";
            oss << "\n";
        }
        oss << "    ]}";
        if (i + 1 < instrSpaces.size())
            oss << ",";
        oss << "\n";
    }
    oss << "  ]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeChannelSpaces(const std::vector<CcuChannelSpace>& channelSpaces)
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < channelSpaces.size(); i++) {
        const auto& space = channelSpaces[i];
        oss << "    {\"rankId\": " << space.rankId << ", \"dieId\": " << space.dieId << ", \"channels\": [\n";
        for (size_t j = 0; j < space.channels.size(); j++) {
            const auto& ch = space.channels[j];
            oss << "      {\"channelId\": " << ch.channelId << ", \"remoteRankId\": " << ch.remoteRankId
                << ", \"remoteDieId\": " << ch.remoteDieId << "}";
            if (j + 1 < space.channels.size())
                oss << ",";
            oss << "\n";
        }
        oss << "    ]}";
        if (i + 1 < channelSpaces.size())
            oss << ",";
        oss << "\n";
    }
    oss << "  ]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeResourceDelta(const CcuResourceDelta& delta)
{
    std::ostringstream oss;
    oss << "{";

    oss << "\"xnChanges\": [";
    for (size_t i = 0; i < delta.xnChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << delta.xnChanges[i].id << ", \"valueBefore\": \""
            << ToHexString(delta.xnChanges[i].valueBefore) << "\""
            << ", \"valueAfter\": \"" << ToHexString(delta.xnChanges[i].valueAfter) << "\"}";
    }
    oss << "], ";

    oss << "\"gsaChanges\": [";
    for (size_t i = 0; i < delta.gsaChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << delta.gsaChanges[i].id << ", \"valueBefore\": \""
            << ToHexString(delta.gsaChanges[i].valueBefore) << "\""
            << ", \"valueAfter\": \"" << ToHexString(delta.gsaChanges[i].valueAfter) << "\"}";
    }
    oss << "], ";

    // Channel 表是静态配置，不会变化，无需序列化 channelChanges
    oss << "\"ckeChanges\": [";
    for (size_t i = 0; i < delta.ckeChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        oss << "{\"id\": " << delta.ckeChanges[i].id << ", \"valueBefore\": " << delta.ckeChanges[i].valueBefore
            << ", \"valueAfter\": " << delta.ckeChanges[i].valueAfter << "}";
    }
    oss << "]";

    oss << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeContext(const CcuExecutionContext& ctx)
{
    std::ostringstream oss;
    oss << "{\"inLoop\": " << (ctx.inLoop ? "true" : "false") << ", \"loopRound\": " << ctx.loopRound
        << ", \"loopExtendIndex\": " << ctx.loopExtendIndex << ", \"gsaOffset\": " << ctx.gsaOffset
        << ", \"iterStepGSA\": " << ctx.iterStepGSA << ", \"loopExtendNum\": " << ctx.loopExtendNum
        << ", \"curLoopCnt\": " << ctx.curLoopCnt << ", \"gsaAddrOffset\": " << ctx.gsaAddrOffset
        << ", \"addrExpandCoef\": " << ctx.addrExpandCoef << ", \"msOffset\": " << ctx.msOffset
        << ", \"ckeOffset\": " << ctx.ckeOffset << ", \"xnIdOffset\": " << ctx.xnIdOffset << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeWaitInfo(const CcuWaitInfo& info)
{
    std::ostringstream oss;
    oss << "{\"hadWait\": " << (info.hadWait ? "true" : "false") << ", \"waitRetryCount\": " << info.waitRetryCount
        << ", \"waitCKEId\": " << info.waitCKEId << ", \"waitCKEMask\": " << info.waitCKEMask
        << ", \"ckeValueOnFirstCheck\": " << info.ckeValueOnFirstCheck
        << ", \"ckeValueOnPass\": " << info.ckeValueOnPass << ", \"waitDurationNs\": " << info.waitDurationNs << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeErrorInfo(const CcuErrorInfo& info)
{
    std::ostringstream oss;
    oss << "{\"hasError\": " << (info.hasError ? "true" : "false") << ", \"errorMessage\": \""
        << EscapeJsonString(info.errorMessage) << "\""
        << ", \"failPhase\": \"" << EscapeJsonString(info.failPhase) << "\""
        << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeCrossCcuChanges(const CcuCrossCcuChanges& changes)
{
    std::ostringstream oss;
    oss << "{\"hasCrossCcuChange\": " << (changes.hasCrossCcuChange ? "true" : "false");

    oss << ", \"remoteCkeChanges\": [";
    for (size_t i = 0; i < changes.remoteCkeChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        const auto& c = changes.remoteCkeChanges[i];
        oss << "{\"remoteRankId\": " << c.remoteRankId << ", \"remoteDieId\": " << c.remoteDieId
            << ", \"ckeId\": " << c.ckeId << ", \"valueBefore\": " << c.valueBefore
            << ", \"valueAfter\": " << c.valueAfter << "}";
    }
    oss << "]";

    oss << ", \"remoteMsChanges\": [";
    for (size_t i = 0; i < changes.remoteMsChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        const auto& c = changes.remoteMsChanges[i];
        oss << "{\"remoteRankId\": " << c.remoteRankId << ", \"remoteDieId\": " << c.remoteDieId
            << ", \"msId\": " << c.msId << ", \"offset\": " << c.offset << ", \"length\": " << c.length << "}";
    }
    oss << "]";

    oss << ", \"remoteMemChanges\": [";
    for (size_t i = 0; i < changes.remoteMemChanges.size(); i++) {
        if (i > 0)
            oss << ", ";
        const auto& c = changes.remoteMemChanges[i];
        oss << "{\"remoteRankId\": " << c.remoteRankId << ", \"remoteAddr\": " << c.remoteAddr
            << ", \"length\": " << c.length << "}";
    }
    oss << "]";

    oss << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeDetail(const CcuInstrTraceDetail& detail)
{
    std::ostringstream oss;
    oss << "{\"typeName\": \"" << EscapeJsonString(detail.typeName) << "\""
        << ", \"args\": {";
    bool first = true;
    for (const auto& kv : detail.args) {
        if (!first)
            oss << ", ";
        oss << "\"" << EscapeJsonString(kv.first) << "\": \"" << EscapeJsonString(kv.second) << "\"";
        first = false;
    }
    oss << "}}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeGlobalEntries(const std::vector<CcuTraceEntry>& entries)
{
    std::ostringstream oss;
    oss << "[\n";
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        oss << "    {";
        oss << "\"globalSeqId\": " << e.globalSeqId << ", \"execRound\": " << e.execRound
            << ", \"rankId\": " << e.rankId << ", \"dieId\": " << e.dieId << ", \"instrId\": " << e.instrId
            << ", \"sqeTaskId\": " << e.sqeTaskId << ", \"category\": \"" << CcuInstrCategoryToString(e.category)
            << "\""
            << ", \"execState\": " << static_cast<int>(e.execState) << ", \"context\": " << SerializeContext(e.context)
            << ", \"resourceDelta\": " << SerializeResourceDelta(e.resourceDelta)
            << ", \"crossCcuChanges\": " << SerializeCrossCcuChanges(e.crossCcuChanges)
            << ", \"waitInfo\": " << SerializeWaitInfo(e.waitInfo)
            << ", \"errorInfo\": " << SerializeErrorInfo(e.errorInfo) << ", \"detail\": " << SerializeDetail(e.detail)
            << "}";
        if (i + 1 < entries.size())
            oss << ",";
        oss << "\n";
    }
    oss << "  ]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeNonCcuEntries(const std::vector<CcuTraceNonCcuEntry>& entries)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < entries.size(); i++) {
        const auto& e = entries[i];
        oss << "{\"globalSeqId\": " << e.globalSeqId << ", \"execRound\": " << e.execRound
            << ", \"rankId\": " << e.rankId << ", \"taskType\": " << static_cast<int>(e.taskType)
            << ", \"execResult\": " << static_cast<int>(e.execResult) << ", \"description\": \""
            << EscapeJsonString(e.description) << "\"}";
        if (i + 1 < entries.size())
            oss << ", ";
    }
    oss << "]";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeFinalSnapshots(const std::map<std::string, CcuResourceSnapshot>& snapshots)
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;
    for (const auto& kv : snapshots) {
        if (!first)
            oss << ", ";
        oss << "\"" << kv.first << "\": " << SerializeResourceSnapshot(kv.second);
        first = false;
    }
    oss << "}";
    return oss.str();
}

std::string CcuTraceSerializer::SerializeSummary(const CcuRunSummary& s)
{
    std::ostringstream oss;
    oss << "{";
    oss << "\"totalInstrExecuted\": " << s.totalInstrExecuted << ", \"totalFailedInstr\": " << s.totalFailedInstr
        << ", \"totalCkeWaitSpins\": " << s.totalCkeWaitSpins << ", \"totalHoldEvents\": " << s.totalHoldEvents;

    oss << ", \"instrCountByCategory\": {";
    bool first = true;
    for (const auto& kv : s.instrCountByCategory) {
        if (!first)
            oss << ", ";
        oss << "\"" << kv.first << "\": " << kv.second;
        first = false;
    }
    oss << "}";

    oss << ", \"instrCountByCcu\": {";
    first = true;
    for (const auto& kv : s.instrCountByCcu) {
        if (!first)
            oss << ", ";
        oss << "\"" << kv.first << "\": " << kv.second;
        first = false;
    }
    oss << "}";

    oss << "}";
    return oss.str();
}

// ===== JSON 辅助函数 =====

std::string CcuTraceSerializer::EscapeJsonString(const std::string& s)
{
    std::ostringstream oss;
    for (char c : s) {
        switch (c) {
            case '"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (static_cast<unsigned char>(c) < 0x20) {
                    oss << "\\u" << std::hex << std::setfill('0') << std::setw(4) << static_cast<int>(c);
                } else {
                    oss << c;
                }
                break;
        }
    }
    return oss.str();
}

std::string CcuTraceSerializer::ToHexString(uint64_t value)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << value;
    return oss.str();
}

std::string CcuTraceSerializer::ToHexString(uint16_t value)
{
    std::ostringstream oss;
    oss << "0x" << std::hex << value;
    return oss.str();
}

} // namespace CcuTrace
