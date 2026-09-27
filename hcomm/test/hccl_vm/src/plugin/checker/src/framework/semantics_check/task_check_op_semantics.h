/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TASK_CHECK_OP_SEMANTICS_H
#define TASK_CHECK_OP_SEMANTICS_H

#include <map>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <vector>

#include "binary_data_type_pub.h"
#include "check_utils.h"
#include "sim_task.h"
#include "task_def.h"
#include "task_graph_generator.h"

namespace HcclSim {
enum class SliceOp {
    OVERRIDE,
    REDUCE,
};

struct SliceOpPair {
    RankId srcRank;
    RankId dstRank;
    DataSlice srcSlice;
    DataSlice dstSlice;
    SliceOp sliceOp;

    std::string Describe() const
    {
        std::stringstream ret;
        ret << "src rank is " << srcRank << ", ";
        ret << "dst rank is " << dstRank << ", ";
        ret << "src slice is " << srcSlice.Describe() << ", ";
        ret << "dst slice is " << dstSlice.Describe() << ", ";
        if (sliceOp == SliceOp::OVERRIDE) {
            ret << "sliceOp is "
                << "SliceOp::OVERRIDE" << std::endl;
        } else {
            ret << "sliceOp is "
                << "SliceOp::REDUCE" << std::endl;
        }

        return ret.str();
    }
};

struct LocalStep {
    RankId rankId;
    u32 localStep;
};

struct TimelineEvent {
    TaskNode* node = nullptr;
    u32 eventId = 0;
    std::string nodeId;
    u32 simGlobalStep = 0;
    u32 simLocalStep = 0;
    u32 logicalStartStep = 0;
    u32 logicalEndStep = 0;
    RankId rankId = 0;
    u32 queueId = 0;
    u32 pos = 0;
    TaskTypeStub taskType;
    std::vector<SliceOpPair> sliceOpPairs;
    std::vector<RankId> affectedRanks;
    bool isMemoryEvent = false;
};

struct RankMemoryTimeline {
    RankId rankId = 0;
    std::string timelineId;
    std::string timelineType;
};

class TaskCheckOpSemantics {
public:
    TaskCheckOpSemantics(u32 rankSize, HcclCMDType opType, HcclDataType dataType, u64 dataCount)
        : rankSize_(rankSize),
          opType_(opType),
          dataType_(dataType),
          dataCount_(dataCount)
    {
        CalcDataSize(opType, dataCount, dataType, dataSize_);
    }
    HcclResult Execute();
    void SetGraphHead(TaskNode* graphHead) { graphHead_ = graphHead; };
    void SetReduceType(HcclReduceOp reduceType) { reduceType_ = reduceType; };
    void SetSrcRank(RankId srcRank) { srcRank_ = srcRank; };
    void SetDstRank(RankId dstRank) { dstRank_ = dstRank; };
    void SetRoot(RankId root) { root_ = root; };
    void SetVDataDes(VDataDesTagInner vDataDes) { vDataDes_ = vDataDes; }
    void SetAll2AllDataDes(All2AllDataDesTagInner all2AllDataDes) { all2AllDataDes_ = all2AllDataDes; }

private:
    void InitInputBuffer();
    void InitInputBuffer(RankId root);
    void UpdateStep(TaskNode* simNode);
    HcclResult RecordNodeSemantics(TaskNode* simNode, const std::vector<SliceOpPair>& sliceOpPairs);

    void GetSrcIntersectionAddr(
        SliceOpPair& slicePair, const BufferSemantic& srcBufSemantic, u64& srcStartAddr, u64& srcEndAddr) const;
    HcclResult CheckBufSemantics(
        std::vector<BufferSemantic*>& bufSemantics, u64 startAddr, u64 size, bool ignoreError = false) const;
    void RemoveAffectedBufSemantics(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
        std::vector<BufferSemantic*>& affectedDstBufSemantics);
    void ApplyOverrideSrcBufSemantic(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics, u32 effectGlobalStep,
        const BufferSemantic srcBufSemantic);
    HcclResult ReduceToAffectedBufSemantic(
        const BufferSemantic& srcBufSemantic, std::vector<BufferSemantic*> toAddReduceInfoSemantics, u64 srcStartAddr);
    HcclResult ApplyReduceSrcBufSemantic(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics, u32 effectGlobalStep,
        const BufferSemantic& srcBufSemantic, std::vector<BufferSemantic*>& affectedDstBufSemantics);
    void GetAffectedBufSemantics(
        SliceOpPair& slicePair, const BufferSemantic& srcBufSemantic,
        std::map<RankId, RankMemorySemantics>& rankMemSemantics, std::vector<BufferSemantic*>& affectedDstBufSemantics);
    void GetAffectedBufSemantics(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
        std::vector<BufferSemantic*>& affectedDstBufSemantics);
    HcclResult ApplySrcBufSemanticsToDst(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
        std::map<RankId, bool>& memSemanticsChange, u32 effectGlobalStep, std::vector<BufferSemantic*> srcBufSemantics);

    HcclResult ProcessSliceOpPair(
        SliceOpPair& slicePair, std::map<RankId, RankMemorySemantics>& rankMemSemantics,
        std::map<RankId, bool>& memSemanticsChange, u32 effectGlobalStep);
    void GetSliceOpPair(TaskNode* simNodes, std::vector<SliceOpPair>& sliceOpPairs) const;
    HcclResult ProcessNodeSemantics(TaskNode* simNode);

    bool IsReadyForSimulate(const TaskNode* node, std::set<TaskNode*>& simulatedNodes) const;

    void AddChildrenToQueue(
        TaskNode* node, std::set<TaskNode*>& visitedNodes, std::queue<TaskNode*>& walkQue,
        std::set<TaskNode*>& simulatedNodes) const;
    void PrepareTimelineContext();
    void EnsureRankMemoryEntries(std::map<RankId, RankMemorySemantics>& rankMemSemantics) const;
    HcclResult GenerateMemoryTimeline();
    HcclResult BuildLogicalSchedule();
    HcclResult ReplayTimelineEvents();
    void BuildRankTimelineEvents();
    HcclResult GenMemSemantics();

    HcclCMDType opType_;
    HcclReduceOp reduceType_;
    HcclDataType dataType_;
    u64 dataCount_;
    RankId srcRank_;
    RankId dstRank_;
    RankId root_;
    VDataDesTagInner vDataDes_;
    All2AllDataDesTagInner all2AllDataDes_;

    TaskNodePtr graphHead_;
    u64 dataSize_ = 0;
    u64 inputDataSize_ = 0;
    u64 outputDataSize_ = 0;
    std::map<RankId, RankMemorySemantics> allRankMemSemantics_;
    u32 rankSize_;

    // 语义信息dump相关的环境变量
    u32 globalStep_ = 0;
    std::map<RankId, u32> localStep_;
    std::map<RankId, std::map<u32, u32>> localStep2GlobalStep_;
    std::map<u32, LocalStep> globalStep2LocalStep_;
    std::map<RankId, bool> memSemanticsChange_;
    std::vector<TaskNode*> graphNodes_;
    std::map<TaskNode*, std::string> nodeIdMap_;
    std::vector<TimelineEvent> timelineEvents_;
    std::map<u32, u32> globalStepToEventId_;
    std::map<RankId, RankMemorySemantics> initialRankMemSemantics_;
    std::map<RankId, RankMemoryTimeline> rankMemoryTimelines_;
};
} // namespace HcclSim

#endif
