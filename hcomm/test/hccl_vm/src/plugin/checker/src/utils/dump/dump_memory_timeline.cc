/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_memory_timeline.h"
#include <algorithm>
#include <cstdint>
#include <nlohmann_json/json.hpp>
#include <ostream>
#include <unordered_set>
#include <vector>

#include "dump/dump_manager.h"
#include "dump/dump_run_manifest.h"
#include "dump/msgpack_writer.h"
#include "sim_log.h"

namespace HcclSim {
using Json = nlohmann::json;
using EventIdToEventMap = std::unordered_map<u32, const TimelineEvent*>;
using GlobalStepToEventMap = std::unordered_map<u32, const TimelineEvent*>;
static constexpr size_t SNAPSHOT_CHUNK_SIZE = 128;

struct SourceEventLookupCache {
    const TimelineEvent* fallbackEvent = nullptr;
    std::unordered_map<RankId, const TimelineEvent*> matchedEventByRank;
};

static u32 ConvertBufferTypeToCode(BufferType bufferType) { return static_cast<u32>(bufferType); }

static u32 ConvertReduceOpToCode(HcclReduceOp reduceOp) { return static_cast<u32>(reduceOp); }

static std::string BuildRankSnapshotManifestPath(RankId rankId)
{
    return StringFormat("memory/rank_%u/manifest.msgpack", rankId);
}

static std::string BuildRankSnapshotChunkPath(RankId rankId, size_t chunkId)
{
    return StringFormat("memory/rank_%u/snapshots_%zu.msgpack", rankId, chunkId);
}

static Json CreateReservedArray(size_t reserveSize)
{
    Json arrayJson = Json::array();
    arrayJson.get_ref<Json::array_t&>().reserve(reserveSize);
    return arrayJson;
}

static EventIdToEventMap BuildEventIdToEventMap(const std::vector<TimelineEvent>& timelineEvents)
{
    EventIdToEventMap eventIdToEvent;
    eventIdToEvent.reserve(timelineEvents.size());
    for (const auto& timelineEvent : timelineEvents) {
        if (!timelineEvent.isMemoryEvent) {
            continue;
        }
        eventIdToEvent.emplace(timelineEvent.eventId, &timelineEvent);
    }
    return eventIdToEvent;
}

static GlobalStepToEventMap
BuildGlobalStepToEventMap(const std::map<u32, u32>& globalStepToEventId, const EventIdToEventMap& eventIdToEvent)
{
    GlobalStepToEventMap globalStepToEvent;
    globalStepToEvent.reserve(globalStepToEventId.size());
    for (const auto& stepToEventId : globalStepToEventId) {
        const auto eventIter = eventIdToEvent.find(stepToEventId.second);
        if (eventIter == eventIdToEvent.end()) {
            continue;
        }
        globalStepToEvent.emplace(stepToEventId.first, eventIter->second);
    }
    return globalStepToEvent;
}

static bool EventAffectsRank(const TimelineEvent* timelineEvent, RankId rankId)
{
    if (timelineEvent == nullptr) {
        return false;
    }
    if (timelineEvent->rankId == rankId) {
        return true;
    }
    return std::find(timelineEvent->affectedRanks.begin(), timelineEvent->affectedRanks.end(), rankId)
           != timelineEvent->affectedRanks.end();
}

static SourceEventLookupCache
BuildSourceEventLookupCache(const BufferSemantic& bufferSemantic, const GlobalStepToEventMap& globalStepToEvent)
{
    SourceEventLookupCache sourceEventLookupCache;
    std::unordered_set<RankId> pendingSourceRanks;
    pendingSourceRanks.reserve(bufferSemantic.srcBufs.size());
    for (const auto& srcBuf : bufferSemantic.srcBufs) {
        pendingSourceRanks.insert(srcBuf.rankId);
    }
    sourceEventLookupCache.matchedEventByRank.reserve(pendingSourceRanks.size());

    for (auto iter = bufferSemantic.affectedGlobalSteps.rbegin(); iter != bufferSemantic.affectedGlobalSteps.rend();
         ++iter) {
        const auto eventIter = globalStepToEvent.find(*iter);
        if (eventIter == globalStepToEvent.end()) {
            continue;
        }

        const TimelineEvent* timelineEvent = eventIter->second;
        if (sourceEventLookupCache.fallbackEvent == nullptr) {
            sourceEventLookupCache.fallbackEvent = timelineEvent;
        }

        for (auto rankIter = pendingSourceRanks.begin(); rankIter != pendingSourceRanks.end();) {
            if (EventAffectsRank(timelineEvent, *rankIter)) {
                sourceEventLookupCache.matchedEventByRank.emplace(*rankIter, timelineEvent);
                rankIter = pendingSourceRanks.erase(rankIter);
                continue;
            }
            ++rankIter;
        }

        if (pendingSourceRanks.empty()) {
            break;
        }
    }
    return sourceEventLookupCache;
}

static const TimelineEvent*
ResolveSourceEvent(const SrcBufDes& srcBuf, const SourceEventLookupCache& sourceEventLookupCache)
{
    const auto matchIter = sourceEventLookupCache.matchedEventByRank.find(srcBuf.rankId);
    if (matchIter != sourceEventLookupCache.matchedEventByRank.end()) {
        return matchIter->second;
    }
    return sourceEventLookupCache.fallbackEvent;
}

static void MsgpackWriteSourceRef(
    MsgpackWriter& writer, const SrcBufDes& srcBuf, const BufferSemantic& bufferSemantic,
    const SourceEventLookupCache& sourceEventLookupCache)
{
    writer.WriteMapHeader(3);

    writer.WriteString("buf");
    writer.WriteArrayHeader(4);
    writer.WriteUInt(srcBuf.rankId);
    writer.WriteUInt(ConvertBufferTypeToCode(srcBuf.bufType));
    writer.WriteUInt(srcBuf.srcAddr);
    writer.WriteUInt(bufferSemantic.size);

    const TimelineEvent* sourceEvent = ResolveSourceEvent(srcBuf, sourceEventLookupCache);
    writer.WriteString("snapshot");
    writer.WriteUInt(sourceEvent == nullptr ? 0 : sourceEvent->logicalEndStep);

    writer.WriteString("node");
    if (sourceEvent == nullptr) {
        writer.WriteString("");
    } else {
        writer.WriteString(sourceEvent->nodeId);
    }
}

static void MsgpackWriteBufferSemantic(
    MsgpackWriter& writer, const BufferSemantic& bufferSemantic, const GlobalStepToEventMap& globalStepToEvent)
{
    const size_t semanticMapSize = bufferSemantic.isReduce ? 5 : 4;
    writer.WriteMapHeader(semanticMapSize);

    writer.WriteString("start_addr");
    writer.WriteUInt(bufferSemantic.startAddr);

    writer.WriteString("size");
    writer.WriteUInt(bufferSemantic.size);

    writer.WriteString("is_reduce");
    writer.WriteUInt(bufferSemantic.isReduce ? 1 : 0);

    if (bufferSemantic.isReduce) {
        writer.WriteString("reduce_type");
        writer.WriteUInt(ConvertReduceOpToCode(bufferSemantic.reduceType));
    }

    writer.WriteString("srcs");
    writer.WriteArrayHeader(bufferSemantic.srcBufs.size());
    const SourceEventLookupCache sourceEventLookupCache
        = BuildSourceEventLookupCache(bufferSemantic, globalStepToEvent);
    for (const auto& srcBuf : bufferSemantic.srcBufs) {
        MsgpackWriteSourceRef(writer, srcBuf, bufferSemantic, sourceEventLookupCache);
    }
}

static void MsgpackWriteLayout(
    MsgpackWriter& writer, const RankMemorySemantics& rankMemorySemantics,
    const GlobalStepToEventMap& globalStepToEvent)
{
    writer.WriteArrayHeader(rankMemorySemantics.size());
    for (const auto& bufferEntry : rankMemorySemantics) {
        writer.WriteMapHeader(2);

        writer.WriteString("buffer_type");
        writer.WriteUInt(ConvertBufferTypeToCode(bufferEntry.first));

        writer.WriteString("semantics");
        writer.WriteArrayHeader(bufferEntry.second.size());
        for (const auto& bufferSemantic : bufferEntry.second) {
            MsgpackWriteBufferSemantic(writer, bufferSemantic, globalStepToEvent);
        }
    }
}

static void MsgpackWriteSnapshot(
    MsgpackWriter& writer, u32 logicalStep, const std::vector<std::string>& memoryTaskIds,
    const RankMemorySemantics& fullLayout, const GlobalStepToEventMap& globalStepToEvent)
{
    writer.WriteMapHeader(3);

    writer.WriteString("logical_step");
    writer.WriteUInt(logicalStep);

    writer.WriteString("memory_task_ids");
    writer.WriteArrayHeader(memoryTaskIds.size());
    for (const auto& taskId : memoryTaskIds) {
        writer.WriteString(taskId);
    }

    writer.WriteString("layout");
    MsgpackWriteLayout(writer, fullLayout, globalStepToEvent);
}

static HcclResult BuildSnapshotEntryMsgpack(
    std::vector<uint8_t>& snapshotEntries, u32 logicalStep, const std::vector<std::string>& memoryTaskIds,
    const RankMemorySemantics& fullLayout, const GlobalStepToEventMap& globalStepToEvent)
{
    MsgpackWriter writer(snapshotEntries);
    writer.WriteString(std::to_string(logicalStep));
    MsgpackWriteSnapshot(writer, logicalStep, memoryTaskIds, fullLayout, globalStepToEvent);
    return HcclResult::HCCL_SUCCESS;
}

static HcclResult WriteSnapshotChunkMsgpack(
    std::ostream& out, RankId rankId, size_t chunkId, size_t snapshotCount, const std::vector<uint8_t>& snapshotEntries)
{
    MsgpackWriter writer(out);
    writer.WriteMapHeader(4);

    writer.WriteString("rank_id");
    writer.WriteUInt(rankId);

    writer.WriteString("chunk_id");
    writer.WriteUInt(chunkId);

    writer.WriteString("snapshot_count");
    writer.WriteUInt(snapshotCount);

    writer.WriteString("snapshots");
    writer.WriteMapHeader(snapshotCount);
    writer.WriteRawBytes(snapshotEntries);

    if (!out.good()) {
        return HcclResult::HCCL_E_INTERNAL;
    }
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemoryTimelineDumpStream::Initialize(
    const std::vector<RankId>& rankIds, const std::vector<TimelineEvent>& timelineEvents,
    const std::map<u32, u32>& globalStepToEventId)
{
    m_enabled = DumpManager::GetInstance().IsMemorySnapshotEnabled();
    m_globalStepToEvent.clear();
    m_rankDumpStates.clear();

    if (!m_enabled) {
        return HcclResult::HCCL_SUCCESS;
    }

    const EventIdToEventMap eventIdToEvent = BuildEventIdToEventMap(timelineEvents);
    m_globalStepToEvent = BuildGlobalStepToEventMap(globalStepToEventId, eventIdToEvent);
    HCCL_VM_INFO(
        "start dump memory timeline, rank_count={}, event_count={}, step_event_count={}", rankIds.size(),
        eventIdToEvent.size(), m_globalStepToEvent.size());

    for (RankId rankId : rankIds) {
        RankDumpState rankDumpState;
        rankDumpState.manifestChunks = Json::array();
        m_rankDumpStates.emplace(rankId, std::move(rankDumpState));
    }

    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemoryTimelineDumpStream::FlushOpenChunk(RankId rankId, RankDumpState& rankDumpState)
{
    if (!m_enabled || rankDumpState.openChunkSnapshotCount == 0) {
        return HcclResult::HCCL_SUCCESS;
    }

    const size_t chunkId = rankDumpState.nextChunkId;
    const std::string chunkPath = BuildRankSnapshotChunkPath(rankId, chunkId);

    HcclResult ret = DumpManager::GetInstance().WriteMsgpackStream(
        chunkPath, [&rankDumpState, rankId, chunkId](std::ostream& out) -> HcclResult {
            return WriteSnapshotChunkMsgpack(
                out, rankId, chunkId, rankDumpState.openChunkSnapshotCount, rankDumpState.openChunkSnapshotEntries);
        });
    if (ret != HcclResult::HCCL_SUCCESS) {
        HCCL_VM_WARN("failed to dump rank [{}] snapshot chunk [{}].", rankId, chunkId);
        return ret;
    }

    Json chunkMetaJson = CreateReservedArray(5);
    chunkMetaJson.push_back(chunkId);
    chunkMetaJson.push_back(chunkPath);
    chunkMetaJson.push_back(rankDumpState.openChunkSnapshotCount);
    chunkMetaJson.push_back(rankDumpState.openChunkBeginStep);
    chunkMetaJson.push_back(rankDumpState.openChunkEndStep);
    rankDumpState.manifestChunks.push_back(std::move(chunkMetaJson));

    rankDumpState.nextChunkId++;
    rankDumpState.openChunkSnapshotCount = 0;
    rankDumpState.openChunkSnapshotEntries.clear();
    return HcclResult::HCCL_SUCCESS;
}

HcclResult MemoryTimelineDumpStream::AppendSnapshot(
    RankId rankId, u32 logicalStep, const std::vector<std::string>& memoryTaskIds,
    const RankMemorySemantics& fullLayout)
{
    if (!m_enabled) {
        return HcclResult::HCCL_SUCCESS;
    }

    auto rankStateIter = m_rankDumpStates.find(rankId);
    if (rankStateIter == m_rankDumpStates.end()) {
        RankDumpState rankDumpState;
        rankDumpState.manifestChunks = Json::array();
        rankStateIter = m_rankDumpStates.emplace(rankId, std::move(rankDumpState)).first;
    }
    RankDumpState& rankDumpState = rankStateIter->second;

    if (rankDumpState.openChunkSnapshotCount == 0) {
        rankDumpState.openChunkBeginStep = logicalStep;
        rankDumpState.openChunkSnapshotEntries.clear();
        rankDumpState.openChunkSnapshotEntries.reserve(8 * 1024 * 1024);
    }

    HcclResult ret = BuildSnapshotEntryMsgpack(
        rankDumpState.openChunkSnapshotEntries, logicalStep, memoryTaskIds, fullLayout, m_globalStepToEvent);
    if (ret != HcclResult::HCCL_SUCCESS) {
        return ret;
    }

    rankDumpState.openChunkSnapshotCount++;
    rankDumpState.totalSnapshotCount++;
    rankDumpState.openChunkEndStep = logicalStep;

    if (rankDumpState.openChunkSnapshotCount < SNAPSHOT_CHUNK_SIZE) {
        return HcclResult::HCCL_SUCCESS;
    }

    return FlushOpenChunk(rankId, rankDumpState);
}

HcclResult MemoryTimelineDumpStream::Finalize()
{
    if (!m_enabled) {
        return HcclResult::HCCL_SUCCESS;
    }

    Json memoryStatsJson = Json::object();
    memoryStatsJson["rank_count"] = m_rankDumpStates.size();
    memoryStatsJson["ranks"] = Json::array();
    size_t totalSnapshotCount = 0;
    size_t totalChunkCount = 0;

    for (auto& rankEntry : m_rankDumpStates) {
        const RankId rankId = rankEntry.first;
        RankDumpState& rankDumpState = rankEntry.second;

        HcclResult ret = FlushOpenChunk(rankId, rankDumpState);
        if (ret != HcclResult::HCCL_SUCCESS) {
            return ret;
        }

        Json snapshotManifestJson = Json::object();
        snapshotManifestJson["rank_id"] = rankId;
        snapshotManifestJson["chunk_count"] = rankDumpState.nextChunkId;
        snapshotManifestJson["chunks"] = std::move(rankDumpState.manifestChunks);

        const std::string snapshotManifestPath = BuildRankSnapshotManifestPath(rankId);
        ret = DumpManager::GetInstance().Write(snapshotManifestPath, snapshotManifestJson);
        if (ret != HcclResult::HCCL_SUCCESS) {
            HCCL_VM_WARN("failed to dump rank [{}] snapshot manifest.", rankId);
            return ret;
        }

        totalSnapshotCount += rankDumpState.totalSnapshotCount;
        totalChunkCount += rankDumpState.nextChunkId;
        Json rankStatsJson = Json::object();
        rankStatsJson["rank_id"] = rankId;
        rankStatsJson["snapshot_count"] = rankDumpState.totalSnapshotCount;
        rankStatsJson["chunk_count"] = rankDumpState.nextChunkId;
        memoryStatsJson["ranks"].push_back(std::move(rankStatsJson));
    }

    memoryStatsJson["snapshot_count"] = totalSnapshotCount;
    memoryStatsJson["chunk_count"] = totalChunkCount;
    DumpRunManifest::GetInstance().SetMemorySnapshotStats(memoryStatsJson);
    HCCL_VM_INFO("finish dump memory timeline, rank_count={}", m_rankDumpStates.size());
    return HcclResult::HCCL_SUCCESS;
}
} // namespace HcclSim
