/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "dump/dump_json_utils.h"

namespace HcclSim {
using Json = nlohmann::json;

std::string DumpTaskTypeToString(TaskTypeStub taskType)
{
    switch (taskType) {
        case TaskTypeStub::LOCAL_COPY:
            return "LOCAL_COPY";
        case TaskTypeStub::LOCAL_REDUCE:
            return "LOCAL_REDUCE";
        case TaskTypeStub::LOCAL_BATCH_REDUCE:
            return "LOCAL_BATCH_REDUCE";
        case TaskTypeStub::LOCAL_POST_TO:
            return "LOCAL_POST_TO";
        case TaskTypeStub::LOCAL_WAIT_FROM:
            return "LOCAL_WAIT_FROM";
        case TaskTypeStub::POST:
            return "POST";
        case TaskTypeStub::WAIT:
            return "WAIT";
        case TaskTypeStub::READ:
            return "READ";
        case TaskTypeStub::READ_REDUCE:
            return "READ_REDUCE";
        case TaskTypeStub::WRITE:
            return "WRITE";
        case TaskTypeStub::WRITE_REDUCE:
            return "WRITE_REDUCE";
        case TaskTypeStub::BEING_READ:
            return "BEING_READ";
        case TaskTypeStub::BEING_READ_REDUCE:
            return "BEING_READ_REDUCE";
        case TaskTypeStub::BEING_WRITTEN:
            return "BEING_WRITTEN";
        case TaskTypeStub::BEING_WRITTEN_REDUCE:
            return "BEING_WRITTEN_REDUCE";
        case TaskTypeStub::LOCAL_POST_TO_SHADOW:
            return "LOCAL_POST_TO_SHADOW";
        case TaskTypeStub::LOCAL_WAIT_FROM_SHADOW:
            return "LOCAL_WAIT_FROM_SHADOW";
        case TaskTypeStub::AIV_TASK:
            return "AIV_TASK";
        case TaskTypeStub::SET_VALUE:
            return "SET_VALUE";
        case TaskTypeStub::SET_FLAG:
            return "SET_FLAG";
        case TaskTypeStub::WAIT_FLAG:
            return "WAIT_FLAG";
        case TaskTypeStub::SEND_SYNC:
            return "SEND_SYNC";
        case TaskTypeStub::RECV_SYNC:
            return "RECV_SYNC";
        case TaskTypeStub::SEND_SYNC_REDUCE:
            return "SEND_SYNC_REDUCE";
        case TaskTypeStub::COMP_VALUE:
            return "COMP_VALUE";
        case TaskTypeStub::PIPE_BARRIER:
            return "PIPE_BARRIER";
        case TaskTypeStub::CCU_GRAPH:
            return "CCU_GRAPH";
        case TaskTypeStub::LOOP_START:
            return "LOOP_START";
        case TaskTypeStub::LOOP_END:
            return "LOOP_END";
        case TaskTypeStub::SUB_GRAPH_END:
            return "SUB_GRAPH_END";
        case TaskTypeStub::SET_FLAG_SHADOW:
            return "SET_FLAG_SHADOW";
        case TaskTypeStub::WAIT_FLAG_SHADOW:
            return "WAIT_FLAG_SHADOW";
        case TaskTypeStub::AIV_START:
            return "AIV_START";
        case TaskTypeStub::BLOCK_START:
            return "BLOCK_START";
        case TaskTypeStub::AIV_END:
            return "AIV_END";
        case TaskTypeStub::VIRTUAL_RANK_START:
            return "VIRTUAL_RANK_START";
        case TaskTypeStub::GRAPH_SEPARATE:
            return "GRAPH_SEPARATE";
        default:
            return "UNKNOWN";
    }
}

std::string DumpBufferTypeToString(BufferType bufferType)
{
    switch (bufferType) {
        case BufferType::INPUT:
            return "INPUT";
        case BufferType::OUTPUT:
            return "OUTPUT";
        case BufferType::CCL:
            return "CCL";
        case BufferType::SCRATCH:
            return "SCRATCH";
        case BufferType::INPUT_AIV:
            return "INPUT_AIV";
        case BufferType::OUTPUT_AIV:
            return "OUTPUT_AIV";
        case BufferType::AIV_COMMINFO:
            return "AIV_COMMINFO";
        case BufferType::USERBUF_AIV:
            return "USERBUF_AIV";
        case BufferType::MS:
            return "MS";
        case BufferType::RESERVED:
            return "RESERVED";
        default:
            return "UNKNOWN";
    }
}

std::string DumpReduceOpToString(HcclReduceOp reduceOp)
{
    switch (reduceOp) {
        case HCCL_REDUCE_SUM:
            return "SUM";
        case HCCL_REDUCE_PROD:
            return "PROD";
        case HCCL_REDUCE_MAX:
            return "MAX";
        case HCCL_REDUCE_MIN:
            return "MIN";
        case HCCL_REDUCE_RESERVED:
            return "RESERVED";
        default:
            return "UNKNOWN";
    }
}

Json DumpDataSliceToJson(const DataSlice& dataSlice)
{
    Json dataSliceJson = Json::object();
    dataSliceJson["buffer_type"] = DumpBufferTypeToString(dataSlice.GetType());
    dataSliceJson["offset"] = dataSlice.GetOffset();
    dataSliceJson["size"] = dataSlice.GetSize();
    return dataSliceJson;
}

Json DumpSrcBufToJson(const SrcBufDes& srcBuf)
{
    Json srcBufJson = Json::object();
    srcBufJson["rank_id"] = srcBuf.rankId;
    srcBufJson["buffer_type"] = DumpBufferTypeToString(srcBuf.bufType);
    srcBufJson["src_addr"] = srcBuf.srcAddr;
    return srcBufJson;
}

Json DumpBufferSemanticToJson(
    const BufferSemantic& bufferSemantic, BufferType bufferType, const std::map<u32, u32>* globalStepToEventId)
{
    Json semanticJson = Json::object();
    semanticJson["buffer_type"] = DumpBufferTypeToString(bufferType);
    semanticJson["start_addr"] = bufferSemantic.startAddr;
    semanticJson["size"] = bufferSemantic.size;
    semanticJson["is_reduce"] = bufferSemantic.isReduce;
    semanticJson["reduce_type"] = DumpReduceOpToString(bufferSemantic.reduceType);
    semanticJson["src_bufs"] = Json::array();
    for (const auto& srcBuf : bufferSemantic.srcBufs) {
        semanticJson["src_bufs"].push_back(DumpSrcBufToJson(srcBuf));
    }

    semanticJson["affected_global_steps"] = Json::array();
    semanticJson["affected_event_ids"] = Json::array();
    for (u32 globalStep : bufferSemantic.affectedGlobalSteps) {
        semanticJson["affected_global_steps"].push_back(globalStep);
        if (globalStepToEventId == nullptr) {
            continue;
        }
        auto eventIdIter = globalStepToEventId->find(globalStep);
        if (eventIdIter != globalStepToEventId->end()) {
            semanticJson["affected_event_ids"].push_back(eventIdIter->second);
        }
    }
    return semanticJson;
}
} // namespace HcclSim
