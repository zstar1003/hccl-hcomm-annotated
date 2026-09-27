/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "task_def_v3.h"

#include <algorithm>
#include <cstdint>
#include <sstream>

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    uint64_t g_checkerAivUbBufferSize = 0;
    uint64_t g_checkerAivCommInfoSize = 0;

    namespace {
        const char* ToString(MemType type)
        {
            switch (type) {
                case MemType::INPUT:
                    return "INPUT";
                case MemType::OUTPUT:
                    return "OUTPUT";
                case MemType::CCL:
                    return "CCL";
                case MemType::AIV_UB:
                    return "AIV_UB";
                case MemType::AIV_COMM:
                    return "AIV_COMM";
                case MemType::MS_CCU:
                    return "MS_CCU";
                default:
                    return "INVALID";
            }
        }

        const char* ToString(BoundaryType type)
        {
            switch (type) {
                case BoundaryType::MAIN_GRAPH:
                    return "MAIN_GRAPH";
                case BoundaryType::CCU_SUB_GRAPH:
                    return "CCU_SUB_GRAPH";
                case BoundaryType::AIV_SUB_GRAPH:
                    return "AIV_SUB_GRAPH";
                case BoundaryType::LOOP:
                    return "LOOP";
                default:
                    return "INVALID";
            }
        }

        const char* ToString(ProtocolType protocol)
        {
            switch (protocol) {
                case ProtocolType::SDMA:
                    return "SDMA";
                case ProtocolType::RDMA:
                    return "RDMA";
                case ProtocolType::CCU:
                    return "CCU";
                default:
                    return "INVALID";
            }
        }

        std::string DescribePosition(const TaskPosition& position)
        {
            std::ostringstream os;
            os << "rank=" << position.rankId << ", stream=" << position.streamId;
            if (position.queueId != INVALID_QUEUE_ID) {
                os << ", queue=" << position.queueId;
            }
            if (position.launchIdx != std::numeric_limits<uint64_t>::max()) {
                os << ", launch=" << position.launchIdx;
            }
            if (position.blockId != std::numeric_limits<uint32_t>::max()) {
                os << ", block=" << position.blockId;
            }
            if (position.pipe != std::numeric_limits<uint32_t>::max()) {
                os << ", pipe=" << position.pipe;
            }
            if (position.taskId != std::numeric_limits<uint32_t>::max()) {
                os << ", taskId=" << position.taskId;
            }
            return os.str();
        }

        std::string DescribeMemSlice(const MemSlice& slice)
        {
            std::ostringstream os;
            os << "rank " << slice.rankId << ", " << ToString(slice.memType) << ", [0x" << std::hex << slice.offset
               << ",0x" << (slice.offset + slice.len) << ")" << std::dec;
            return os.str();
        }

        std::string DescribeAivLocation(const TaskPosition& location)
        {
            std::ostringstream os;
            os << "rank=" << location.rankId << ", launch=" << location.launchIdx << ", block=" << location.blockId
               << ", pipe=" << location.pipe << ", taskId=" << location.taskId;
            return os.str();
        }

        std::string DescribeTaskIds(const std::vector<uint32_t>& taskIds)
        {
            std::ostringstream os;
            os << "[";
            for (size_t index = 0; index < taskIds.size(); ++index) {
                if (index != 0U) {
                    os << ",";
                }
                os << taskIds[index];
            }
            os << "]";
            return os.str();
        }

        std::string DescribeAicpuNotify(const AicpuNotify& notify)
        {
            std::ostringstream os;
            os << "{recordRank=" << notify.recordRankId << ", waitRank=" << notify.waitRankId
               << ", notifyId=" << notify.notifyId << "}";
            return os.str();
        }

        std::string DescribeCcuTrace(const CcuTraceInfo& trace)
        {
            std::ostringstream os;
            os << ", ccuTrace={op=" << trace.opName << ", role=" << trace.nodeRole << ", instrId=" << trace.instrId;
            if (trace.dieId != INVALID_DIE_ID) {
                os << ", dieId=" << trace.dieId;
            }
            if (trace.missionId != INVALID_MISSION_ID) {
                os << ", missionId=" << trace.missionId;
            }
            os << "}";
            return os.str();
        }

        bool AddNodeOnce(std::vector<TaskNode*>& nodes, TaskNode* node)
        {
            if (node == nullptr || std::find(nodes.begin(), nodes.end(), node) != nodes.end()) {
                return false;
            }
            nodes.push_back(node);
            return true;
        }

        bool RemoveNode(std::vector<TaskNode*>& nodes, const TaskNode* node)
        {
            const auto iter = std::find(nodes.begin(), nodes.end(), node);
            if (iter == nodes.end()) {
                return false;
            }
            nodes.erase(iter);
            return true;
        }
    } // namespace

    bool TaskNode::AddParent(TaskNode* parentNode) { return AddNodeOnce(parents_, parentNode); }

    bool TaskNode::AddChild(TaskNode* childNode) { return AddNodeOnce(children_, childNode); }

    bool TaskNode::RemoveParent(const TaskNode* parentNode) { return RemoveNode(parents_, parentNode); }

    bool TaskNode::RemoveChild(const TaskNode* childNode) { return RemoveNode(children_, childNode); }

    std::string TaskTransMem::Describe() const
    {
        std::ostringstream os;
        os << "[TaskTransMem] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", src=" << DescribeMemSlice(src_)
           << ", dst=" << DescribeMemSlice(dst_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskTransMem::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskTransMem] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << ", src=" << DescribeMemSlice(src_) << "\n"
           << "dst=" << DescribeMemSlice(dst_);
        return os.str();
    }

    std::string TaskBatchTransMem::Describe() const
    {
        std::ostringstream os;
        os << "[TaskBatchTransMem] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", pairCount=" << std::min(srcs_.size(), dsts_.size())
           << ", mergedPairCount=" << std::min(mergedSrcs_.size(), mergedDsts_.size());
        if (!srcs_.empty()) {
            os << ", src0=" << DescribeMemSlice(srcs_.front());
        }
        if (!dsts_.empty()) {
            os << ", dst0=" << DescribeMemSlice(dsts_.front());
        }
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskBatchTransMem::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskBatchTransMem] node=" << nodeId_ << "\n" << DescribePosition(loc_) << "\n";
        if (!srcs_.empty()) {
            os << ", src0=" << DescribeMemSlice(srcs_.front()) << "\n";
        }
        if (!dsts_.empty()) {
            os << ", dst0=" << DescribeMemSlice(dsts_.front()) << "\n";
        }
        return os.str();
    }

    std::string TaskReduce::Describe() const
    {
        std::ostringstream os;
        os << "[TaskReduce] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", protocol=" << ToString(protocol_)
           << ", dataType=" << static_cast<uint32_t>(dataType_) << ", reduceOp=" << static_cast<uint32_t>(reduceOp_);
        if (srcs_.size() <= 1U) {
            os << ", src=" << DescribeMemSlice(src_);
        } else {
            os << ", srcs=[";
            for (size_t index = 0; index < srcs_.size(); ++index) {
                if (index != 0U) {
                    os << ", ";
                }
                os << DescribeMemSlice(srcs_[index]);
            }
            os << "]";
        }
        os << ", dst=" << DescribeMemSlice(dst_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskReduce::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskReduce] node=" << nodeId_ << "\n" << DescribePosition(loc_) << "\n";
        if (srcs_.size() <= 1U) {
            os << ", src=" << DescribeMemSlice(src_) << "\n";
        } else {
            os << ", srcCount=" << srcs_.size() << ", src0=" << DescribeMemSlice(srcs_.front()) << "\n";
        }
        os << ", dst=" << DescribeMemSlice(dst_);
        return os.str();
    }

    std::string TaskBatchReduce::Describe() const
    {
        std::ostringstream os;
        os << "[TaskBatchReduce] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", dataType=" << static_cast<uint32_t>(dataType_)
           << ", reduceOp=" << static_cast<uint32_t>(reduceOp_)
           << ", groupCount=" << std::min(srcs_.size(), dsts_.size())
           << ", mergedGroupCount=" << std::min(mergedSrcs_.size(), mergedDsts_.size());
        if (!srcs_.empty() && !srcs_.front().empty()) {
            os << ", src0=" << DescribeMemSlice(srcs_.front().front());
        }
        if (!dsts_.empty()) {
            os << ", dst0=" << DescribeMemSlice(dsts_.front());
        }
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskBatchReduce::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskBatchReduce] node=" << nodeId_ << "\n" << DescribePosition(loc_) << "\n";
        if (!srcs_.empty() && !srcs_.front().empty()) {
            os << ", src0=" << DescribeMemSlice(srcs_.front().front()) << "\n";
        }
        if (!dsts_.empty()) {
            os << ", dst0=" << DescribeMemSlice(dsts_.front());
        }
        return os.str();
    }

    std::string TaskRecordAICPU::Describe() const
    {
        std::ostringstream os;
        os << "[TaskRecordAICPU] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", notify=" << DescribeAicpuNotify(notify_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskWaitAICPU::Describe() const
    {
        std::ostringstream os;
        os << "[TaskWaitAICPU] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", notify=" << DescribeAicpuNotify(notify_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskRecordCCU::Describe() const
    {
        std::ostringstream os;
        os << "[TaskRecordCCU] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", protocol=" << ToString(protocol_) << ", channelId=" << notify_.channelId
           << ", recordRank=" << notify_.recordRankId << ", waitRank=" << notify_.waitRankId
           << ", dieId=" << notify_.dieId << ", ckeId=" << notify_.ckeId << ", ckeMask=0x" << std::hex
           << notify_.ckeMask << std::dec;
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskWaitCCU::Describe() const
    {
        std::ostringstream os;
        os << "[TaskWaitCCU] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", protocol=" << ToString(protocol_)
           << ", channelId=" << notify_.channelId << ", recordRank=" << notify_.recordRankId
           << ", waitRank=" << notify_.waitRankId << ", dieId=" << notify_.dieId << ", ckeId=" << notify_.ckeId
           << ", ckeMask=0x" << std::hex << notify_.ckeMask << std::dec;
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskCcuGraph::Describe() const
    {
        std::ostringstream os;
        os << "[TaskCcuGraph] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", rank=" << ccuDesc_.rankId
           << ", queueNum=" << ccuDesc_.ccuParams.size();
        for (size_t i = 0; i < ccuDesc_.ccuParams.size(); ++i) {
            os << ", q" << i << "SqeNum=" << ccuDesc_.ccuParams[i].size();
        }
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskAivGraph::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivGraph] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", rank=" << rankId_
           << ", launch=" << launchIdx_ << ", hostStream=" << hostStreamId_;
        return os.str();
    }

    std::string TaskAivGraph::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivGraph] node=" << nodeId_ << "\n" << DescribePosition(loc_);
        return os.str();
    }

    std::string TaskAivSetFlag::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivSetFlag] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", rank=" << event_.rankId
           << ", launch=" << event_.launchIdx << ", block=" << event_.blockId << ", taskId=" << event_.taskId
           << ", curPipe=" << event_.curPipe << ", srcPipe=" << event_.srcPipe << ", dstPipe=" << event_.dstPipe
           << ", eventId=" << event_.eventId;
        return os.str();
    }
    std::string TaskAivSetFlag::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivSetFlag] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << "srcPipe=" << event_.srcPipe << ", dstPipe=" << event_.dstPipe << "\n"
           << "eventId=" << event_.eventId;
        return os.str();
    }

    std::string TaskAivWaitFlag::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivWaitFlag] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", rank=" << event_.rankId
           << ", launch=" << event_.launchIdx << ", block=" << event_.blockId << ", taskId=" << event_.taskId
           << ", curPipe=" << event_.curPipe << ", srcPipe=" << event_.srcPipe << ", dstPipe=" << event_.dstPipe
           << ", eventId=" << event_.eventId;
        return os.str();
    }
    std::string TaskAivWaitFlag::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivWaitFlag] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << "srcPipe=" << event_.srcPipe << ", dstPipe=" << event_.dstPipe << "\n"
           << "eventId=" << event_.eventId;
        return os.str();
    }

    std::string TaskAivPipeBarrier::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivPipeBarrier] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", "
           << DescribeAivLocation(info_.taskLoc) << ", pipeType=" << info_.pipeType << ", merged=" << info_.merged
           << ", memberTaskIds=" << DescribeTaskIds(info_.memberTaskIds);
        return os.str();
    }
    std::string TaskAivPipeBarrier::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivPipeBarrier] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << "memberTaskIds=" << DescribeTaskIds(info_.memberTaskIds);
        return os.str();
    }

    std::string TaskAivSyncAll::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivSyncAll] node=" << nodeId_ << ", " << DescribePosition(loc_) << ", "
           << DescribeAivLocation(info_.taskLoc) << ", syncRound=" << info_.syncRound << ", merged=" << info_.merged
           << ", memberTaskIds=" << DescribeTaskIds(info_.memberTaskIds);
        return os.str();
    }
    std::string TaskAivSyncAll::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivSyncAll] node=" << nodeId_ << "\n " << DescribePosition(loc_) << "\n"
           << "syncRound=" << info_.syncRound << "\n"
           << "memberTaskIds=" << DescribeTaskIds(info_.memberTaskIds);
        return os.str();
    }

    std::string TaskAivSendFlag::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivSendFlag] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", currentRank=" << flag_.currentRank << ", flagOwnerRank=" << flag_.flagOwnerRank
           << ", launch=" << flag_.launchIdx << ", block=" << flag_.blockId << ", pipe=" << flag_.curPipe
           << ", taskId=" << flag_.taskId << ", commInfoOffset=" << flag_.commInfoOffset << ", value=" << flag_.value;
        return os.str();
    }
    std::string TaskAivSendFlag::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivSendFlag] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << "flagOwnerRank=" << flag_.flagOwnerRank << "\n"
           << "commInfoOffset=" << flag_.commInfoOffset << ", value=" << flag_.value;
        return os.str();
    }

    std::string TaskAivRecvFlag::Describe() const
    {
        std::ostringstream os;
        os << "[TaskAivRecvFlag] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", currentRank=" << flag_.currentRank << ", flagOwnerRank=" << flag_.flagOwnerRank
           << ", launch=" << flag_.launchIdx << ", block=" << flag_.blockId << ", pipe=" << flag_.curPipe
           << ", taskId=" << flag_.taskId << ", commInfoOffset=" << flag_.commInfoOffset << ", value=" << flag_.value;
        return os.str();
    }
    std::string TaskAivRecvFlag::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskAivRecvFlag] node=" << nodeId_ << "\n"
           << DescribePosition(loc_) << "\n"
           << "flagOwnerRank=" << flag_.flagOwnerRank << "\n"
           << "commInfoOffset=" << flag_.commInfoOffset << ", value=" << flag_.value;
        return os.str();
    }

    std::string TaskStart::Describe() const
    {
        std::ostringstream os;
        os << "[TaskStart] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", boundary=" << ToString(boundaryType_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskStart::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskStart] boundary=" << ToString(boundaryType_);
        return os.str();
    }

    std::string TaskEnd::Describe() const
    {
        std::ostringstream os;
        os << "[TaskEnd] node=" << nodeId_ << ", " << DescribePosition(loc_)
           << ", boundary=" << ToString(boundaryType_);
        if (HasCcuTrace()) {
            os << DescribeCcuTrace(GetCcuTrace());
        }
        return os.str();
    }

    std::string TaskEnd::DescribeShort() const
    {
        std::ostringstream os;
        os << "[TaskEnd] boundary=" << ToString(boundaryType_);
        return os.str();
    }
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
