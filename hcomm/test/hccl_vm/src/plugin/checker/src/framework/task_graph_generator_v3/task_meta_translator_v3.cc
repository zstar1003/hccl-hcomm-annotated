/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_meta_translator_v3.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <sstream>
#include <utility>

#include "data_slice.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {
        ProtocolType ConvertProtocol(uint8_t commProtocol)
        {
            if (commProtocol == static_cast<uint8_t>(CommProtocol::COMM_PROTOCOL_ROCE)) {
                return ProtocolType::RDMA;
            }
            return ProtocolType::SDMA;
        }

        MemType ConvertMemType(BufferType inputType)
        {
            switch (inputType) {
                case ::INPUT:
                    return MemType::INPUT;
                case ::OUTPUT:
                    return MemType::OUTPUT;
                case ::CCL:
                case ::SCRATCH:
                    return MemType::CCL;
                case ::MS:
                    return MemType::MS_CCU;
                case ::INPUT_AIV:
                case ::OUTPUT_AIV:
                case ::AIV_COMMINFO:
                case ::USERBUF_AIV:
                    return MemType::AIV_UB;
                default:
                    return MemType::INVALID;
            }
        }

        MemSlice MakeMemSlice(RankId rankId, const DataSlice& slice)
        {
            MemSlice memSlice;
            memSlice.rankId = rankId;
            memSlice.memType = ConvertMemType(slice.GetType());
            memSlice.offset = slice.GetOffset();
            memSlice.len = slice.GetSize();
            memSlice.rawAddr = slice.GetRawAddr();
            return memSlice;
        }

        HcclResult MakeTaskPosition(const HcclTaskMetaData& taskMeta, OperatorId operatorId, TaskPosition& position)
        {
            if (taskMeta.streamId > std::numeric_limits<StreamId>::max()) {
                return HCCL_E_PARA;
            }

            position.operatorId = operatorId;
            position.rankId = taskMeta.rankId;
            position.streamId = static_cast<StreamId>(taskMeta.streamId);
            return HCCL_SUCCESS;
        }

        HcclResult GetNotifyId(uint64_t notifyId, uint32_t& out)
        {
            if (notifyId > std::numeric_limits<uint32_t>::max()) {
                return HCCL_E_PARA;
            }
            out = static_cast<uint32_t>(notifyId);
            return HCCL_SUCCESS;
        }

        HcclResult EnsureStream(AllRankNodeQueues& taskQueues, RankId rankId, StreamId streamId)
        {
            if (streamId == INVALID_STREAM_ID) {
                return HCCL_E_PARA;
            }

            auto& rankStreams = taskQueues[rankId];
            if (rankStreams.size() <= streamId) {
                rankStreams.resize(static_cast<size_t>(streamId) + 1);
            }
            return HCCL_SUCCESS;
        }

        CcuSqeParam MakeCcuSqeParam(const CcuTask& ccuTask)
        {
            CcuSqeParam param;
            param.dieId = ccuTask.dieId;
            param.missionId = ccuTask.missionId;
            param.timeout = ccuTask.timeout;
            param.instStartId = ccuTask.instStartId;
            param.instCnt = ccuTask.instCnt;
            param.key = ccuTask.key;
            param.argSize = ccuTask.argSize;

            const uint32_t argCount = std::min<uint32_t>(ccuTask.argSize, CCU_SQE_ARGS_LEN);
            for (uint32_t i = 0; i < argCount; ++i) {
                param.args[i] = ccuTask.args[i];
            }
            return param;
        }

        std::string DescribeTaskMetaForLog(const HcclTaskMetaData& taskMeta)
        {
            std::ostringstream os;
            os << "taskType=" << static_cast<int32_t>(taskMeta.taskType) << ", rankId=" << taskMeta.rankId
               << ", streamId=" << taskMeta.streamId;
            switch (taskMeta.taskType) {
                case HccLTaskMetaType::MEM_CPY:
                    os << ", srcRankId=" << taskMeta.taskData.transMem.srcRankId
                       << ", dstRankId=" << taskMeta.taskData.transMem.dstRankId << ", src=[0x" << std::hex
                       << taskMeta.taskData.transMem.srcOffset << ",0x"
                       << (taskMeta.taskData.transMem.srcOffset + taskMeta.taskData.transMem.len) << ")"
                       << ", dst=[0x" << taskMeta.taskData.transMem.dstOffset << ",0x"
                       << (taskMeta.taskData.transMem.dstOffset + taskMeta.taskData.transMem.len) << ")" << std::dec
                       << ", protocol=" << static_cast<uint32_t>(taskMeta.taskData.transMem.protocol);
                    break;
                case HccLTaskMetaType::REDUCE:
                    // 此处的 datacount 实际为 size
                    os << ", srcRankId=" << taskMeta.taskData.reduce.srcRankId
                       << ", dstRankId=" << taskMeta.taskData.reduce.dstRankId << ", src=[0x" << std::hex
                       << taskMeta.taskData.reduce.srcOffset << ",0x"
                       << (taskMeta.taskData.reduce.srcOffset + taskMeta.taskData.reduce.dataCount) << ")"
                       << ", dst=[0x" << taskMeta.taskData.reduce.dstOffset << ",0x"
                       << (taskMeta.taskData.reduce.dstOffset + taskMeta.taskData.reduce.dataCount) << ")" << std::dec
                       << ", dataType=" << static_cast<uint32_t>(taskMeta.taskData.reduce.dataType)
                       << ", reduceOp=" << static_cast<uint32_t>(taskMeta.taskData.reduce.reduceOp);
                    break;
                case HccLTaskMetaType::NOTIFY_RECORD:
                case HccLTaskMetaType::NOTIFY_WAIT:
                    os << ", srcRankId=" << taskMeta.taskData.notify.srcRankId
                       << ", dstRankId=" << taskMeta.taskData.notify.dstRankId
                       << ", notifyId=" << taskMeta.taskData.notify.notifyId
                       << ", notifyCount=" << taskMeta.taskData.notify.notifyCount
                       << ", protocol=" << static_cast<uint32_t>(taskMeta.taskData.notify.protocol);
                    break;
                case HccLTaskMetaType::CCU_GRAPH:
                    os << ", dieId=" << taskMeta.taskData.ccu.dieId << ", missionId=" << taskMeta.taskData.ccu.missionId
                       << ", instStartId=" << taskMeta.taskData.ccu.instStartId
                       << ", instCnt=" << taskMeta.taskData.ccu.instCnt << ", argSize=" << taskMeta.taskData.ccu.argSize
                       << ", key=" << taskMeta.taskData.ccu.key;
                    break;
                case HccLTaskMetaType::AIV_GRAPH:
                    os << ", launchId=" << taskMeta.taskData.aiv.launchIdx;
                    break;
                default:
                    break;
            }
            return os.str();
        }

        bool IsContiguousCcuSqe(const TaskCcuGraph& missionNode, const CcuSqeParam& sqe)
        {
            constexpr size_t DEFAULT_CCU_QUEUE_ID = 0U;
            constexpr uint32_t LEGACY_CCU_CONTINUATION_INST_CNT = 13U;
            const CcuSubGraphDesc& desc = missionNode.GetCcuDesc();
            if (desc.ccuParams.size() <= DEFAULT_CCU_QUEUE_ID || desc.ccuParams[DEFAULT_CCU_QUEUE_ID].empty()) {
                return false;
            }

            const CcuSqeParam& lastSqe = desc.ccuParams[DEFAULT_CCU_QUEUE_ID].back();
            if (lastSqe.instCnt != LEGACY_CCU_CONTINUATION_INST_CNT) {
                return false;
            }
            if (lastSqe.instCnt > std::numeric_limits<uint32_t>::max() - lastSqe.instStartId) {
                return false;
            }

            return sqe.instStartId == lastSqe.instStartId + lastSqe.instCnt;
        }

        HcclResult GetTaskDataSlice(
            StorageManager& storage, RankId expectedRank, uint64_t addr, uint64_t size, DataSlice& dataSlice)
        {
            RankId actualRank = 0;
            HcclResult ret = storage.GetSlice(addr, size, dataSlice, &actualRank);
            if (ret != HCCL_SUCCESS) {
                return ret;
            }
            if (actualRank != expectedRank) {
                HCCL_VM_ERROR(
                    "{} Resolved data slice rank mismatch, expectedRank={}, actualRank={}, addr={}, size={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_ADDRESS_INVALID), expectedRank, actualRank, addr, size);
                return HCCL_E_MEMORY;
            }
            return HCCL_SUCCESS;
        }
    } // namespace

    void TaskMetaTranslatorV3::Reset()
    {
        nodes_.clear();
        taskQueues_.clear();
        ccuMissionNodes_.clear();
    }

    std::vector<std::unique_ptr<TaskNode>> TaskMetaTranslatorV3::TakeNodes()
    {
        std::vector<std::unique_ptr<TaskNode>> result = std::move(nodes_);
        nodes_.clear();
        return result;
    }

    AllRankNodeQueues TaskMetaTranslatorV3::TakeTaskQueues()
    {
        AllRankNodeQueues result = std::move(taskQueues_);
        taskQueues_.clear();
        return result;
    }

    HcclResult TaskMetaTranslatorV3::Translate(StorageManager& storage, OperatorId operatorId)
    {
        Reset();

        const HcclVmTaskMetaData& taskMetaData = storage.GetHvmTaskMetaData();
        const auto& taskMetaVec = taskMetaData.task_meta;
        HCCL_VM_INFO("Start converting task metadata into graph nodes, taskMetaCount={}", taskMetaVec.size());
        for (uint32_t i = 0; i < taskMetaVec.size(); ++i) {
            NodeId nodeId = INVALID_NODE_ID;
            const HcclResult ret = TranslateOneTaskMeta(taskMetaVec[i], storage, i, operatorId, nodeId);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to convert one task into a graph node, taskIndex={}, "
                    "ret={}, taskMeta={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_TRANSLATE_FAILED), i, static_cast<uint32_t>(ret),
                    DescribeTaskMetaForLog(taskMetaVec[i]));
                return ret;
            }
        }

        size_t streamCount = 0;
        for (const auto& rankEntry : taskQueues_) {
            for (const auto& stream : rankEntry.second) {
                if (!stream.empty()) {
                    ++streamCount;
                }
            }
        }
        HCCL_VM_INFO(
            "Finished converting task metadata into graph nodes, taskMetaCount={}, nodeCount={}, "
            "rankCount={}, nonEmptyStreamCount={}",
            taskMetaVec.size(), nodes_.size(), taskQueues_.size(), streamCount);
        return HCCL_SUCCESS;
    }

    HcclResult
    TaskMetaTranslatorV3::AddTaskNode(const TaskPosition& position, std::unique_ptr<TaskNode> node, NodeId& nodeId)
    {
        if (node == nullptr) {
            return HCCL_E_MEMORY;
        }
        if (nodes_.size() >= MAX_NODE_COUNT) {
            return HCCL_E_MEMORY;
        }

        HcclResult ret = EnsureStream(taskQueues_, position.rankId, position.streamId);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        nodeId = static_cast<NodeId>(nodes_.size());
        node->SetNodeId(nodeId);
        node->SetPosition(position);
        nodes_.emplace_back(std::move(node));
        taskQueues_[position.rankId][position.streamId].push_back(nodeId);
        return HCCL_SUCCESS;
    }

    HcclResult TaskMetaTranslatorV3::TranslateOneTaskMeta(
        const HcclTaskMetaData& taskMeta, StorageManager& storage, uint32_t taskIndex, OperatorId operatorId,
        NodeId& nodeId)
    {
        TaskPosition position;
        HcclResult ret = MakeTaskPosition(taskMeta, operatorId, position);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }

        switch (taskMeta.taskType) {
            case HccLTaskMetaType::MEM_CPY: {
                const auto& transMem = taskMeta.taskData.transMem;
                const RankId srcRank = transMem.srcRankId;
                const RankId dstRank = transMem.dstRankId;
                DataSlice srcSlice;
                ret = GetTaskDataSlice(storage, srcRank, transMem.srcOffset, transMem.len, srcSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DataSlice dstSlice;
                ret = GetTaskDataSlice(storage, dstRank, transMem.dstOffset, transMem.len, dstSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol
                    = (srcRank == dstRank) ? ProtocolType::SDMA : ConvertProtocol(transMem.protocol);
                auto node = std::make_unique<TaskTransMem>(
                    MakeMemSlice(srcRank, srcSlice), MakeMemSlice(dstRank, dstSlice), protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::REDUCE: {
                const auto& reduce = taskMeta.taskData.reduce;
                const RankId srcRank = reduce.srcRankId;
                const RankId dstRank = reduce.dstRankId;
                // 此处的 datacount 实际为 size
                DataSlice srcSlice;
                ret = GetTaskDataSlice(storage, srcRank, reduce.srcOffset, reduce.dataCount, srcSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                DataSlice dstSlice;
                ret = GetTaskDataSlice(storage, dstRank, reduce.dstOffset, reduce.dataCount, dstSlice);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                const ProtocolType protocol
                    = (srcRank == dstRank) ? ProtocolType::SDMA : ConvertProtocol(taskMeta.taskData.transMem.protocol);
                auto node = std::make_unique<TaskReduce>(
                    MakeMemSlice(srcRank, srcSlice), MakeMemSlice(dstRank, dstSlice), reduce.dataType, reduce.reduceOp,
                    protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::NOTIFY_RECORD: {
                AicpuNotify notify;
                notify.recordRankId = taskMeta.rankId;
                notify.waitRankId = taskMeta.taskData.notify.dstRankId;
                ret = GetNotifyId(taskMeta.taskData.notify.notifyId, notify.notifyId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol = (notify.recordRankId == notify.waitRankId) ?
                                                  ProtocolType::INVALID :
                                                  ConvertProtocol(taskMeta.taskData.notify.protocol);
                auto node = std::make_unique<TaskRecordAICPU>(notify, protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::NOTIFY_WAIT: {
                AicpuNotify notify;
                notify.recordRankId = taskMeta.taskData.notify.srcRankId;
                notify.waitRankId = taskMeta.rankId;
                ret = GetNotifyId(taskMeta.taskData.notify.notifyId, notify.notifyId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }

                const ProtocolType protocol = (notify.recordRankId == notify.waitRankId) ?
                                                  ProtocolType::INVALID :
                                                  ConvertProtocol(taskMeta.taskData.notify.protocol);
                auto node = std::make_unique<TaskWaitAICPU>(notify, protocol);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::CCU_GRAPH: {
                if (taskMeta.taskData.ccu.argSize > CCU_SQE_ARGS_LEN) {
                    return HCCL_E_PARA;
                }

                const CcuSqeParam sqe = MakeCcuSqeParam(taskMeta.taskData.ccu);
                CcuMissionKey key;
                key.rankId = position.rankId;
                key.dieId = sqe.dieId;
                key.missionId = sqe.missionId;
                auto missionIter = ccuMissionNodes_.find(key);
                if (missionIter != ccuMissionNodes_.end()) {
                    const NodeId missionNodeId = missionIter->second;
                    if (missionNodeId < 0 || static_cast<size_t>(missionNodeId) >= nodes_.size()) {
                        return HCCL_E_INTERNAL;
                    }
                    auto* missionNode = dynamic_cast<TaskCcuGraph*>(nodes_[static_cast<size_t>(missionNodeId)].get());
                    if (missionNode == nullptr) {
                        return HCCL_E_INTERNAL;
                    }
                    if (IsContiguousCcuSqe(*missionNode, sqe)) {
                        missionNode->AddCcuParam(0, sqe);
                        nodeId = missionNodeId;
                        return HCCL_SUCCESS;
                    }
                }

                CcuSubGraphDesc desc;
                desc.rankId = taskMeta.rankId;
                desc.ccuParams.resize(1);
                desc.ccuParams[0].push_back(sqe);

                auto node = std::make_unique<TaskCcuGraph>(std::move(desc));
                ret = AddTaskNode(position, std::move(node), nodeId);
                if (ret != HCCL_SUCCESS) {
                    return ret;
                }
                ccuMissionNodes_[key] = nodeId;
                return HCCL_SUCCESS;
            }
            case HccLTaskMetaType::AIV_GRAPH: {
                position.launchIdx = taskMeta.taskData.aiv.launchIdx;
                auto node = std::make_unique<TaskAivGraph>(
                    taskMeta.rankId, taskMeta.taskData.aiv.launchIdx, taskMeta.streamId);
                return AddTaskNode(position, std::move(node), nodeId);
            }
            case HccLTaskMetaType::EVENT_WAIT:
            case HccLTaskMetaType::EVENT_RECORD:
            default:
                HCCL_VM_WARN(
                    "{} This task type is not supported for CheckerV3 graph generation, "
                    "taskIndex={}, taskMeta={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED), taskIndex, DescribeTaskMetaForLog(taskMeta));
                return HCCL_E_NOT_SUPPORT;
        }
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
