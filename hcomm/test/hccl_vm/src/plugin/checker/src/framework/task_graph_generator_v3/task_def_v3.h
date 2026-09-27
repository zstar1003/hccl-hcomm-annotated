/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_DEF_V3_H
#define CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_DEF_V3_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <string>
#include <utility>
#include <vector>

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    using NodeId = int32_t;
    using OperatorId = uint32_t;
    using RankId = uint32_t;
    using StreamId = uint32_t;
    using QueueId = uint32_t;
    using ChannelId = uint16_t; // for CCU mode

    constexpr NodeId MAIN_START_NODE_ID = -1;
    constexpr NodeId INVALID_NODE_ID = std::numeric_limits<NodeId>::min();
    constexpr OperatorId INVALID_OPERATOR_ID = std::numeric_limits<OperatorId>::max();
    constexpr size_t MAX_NODE_COUNT = static_cast<size_t>(std::numeric_limits<NodeId>::max());
    constexpr RankId INVALID_RANK_ID = std::numeric_limits<RankId>::max();
    constexpr StreamId INVALID_STREAM_ID = std::numeric_limits<StreamId>::max();
    constexpr QueueId INVALID_QUEUE_ID = std::numeric_limits<QueueId>::max();
    constexpr ChannelId INVALID_CHANNEL_ID = std::numeric_limits<ChannelId>::max();
    constexpr uint32_t INVALID_DIE_ID = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t INVALID_MISSION_ID = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t INVALID_NOTIFY_ID = std::numeric_limits<uint32_t>::max();
    constexpr uint16_t INVALID_CCU_CKE = std::numeric_limits<uint16_t>::max();
    constexpr uint16_t INVALID_CCU_CKE_MASK = std::numeric_limits<uint16_t>::max();
    constexpr uint32_t CCU_SQE_ARGS_LEN = 13;

    extern uint64_t g_checkerAivUbBufferSize;
    extern uint64_t g_checkerAivCommInfoSize;

    enum class MemType : uint8_t {
        INPUT = 0,
        OUTPUT,
        CCL,
        AIV_UB,
        AIV_COMM,
        MS_CCU,
        INVALID = UINT8_MAX,
    };

    inline const char* DescribeMemType(MemType memType)
    {
        switch (memType) {
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
                return "invalid";
        }
    }

    enum class BoundaryType : uint8_t { MAIN_GRAPH = 0, CCU_SUB_GRAPH, AIV_SUB_GRAPH, LOOP, INVALID = UINT8_MAX };

    enum class TaskType : uint8_t {
        TRANS_MEM = 0,
        BATCH_TRANS_MEM,
        REDUCE,
        BATCH_REDUCE,
        RECORD,
        WAIT,
        CCU_GRAPH,
        AIV_GRAPH,
        AIV_SET_FLAG,
        AIV_WAIT_FLAG,
        AIV_PIPE_BARRIER,
        AIV_SYNC_ALL,
        AIV_SEND_FLAG,
        AIV_RECV_FLAG,
        START,
        END,
        INVALID = UINT8_MAX
    };

    enum class ProtocolType : uint8_t { SDMA = 0, RDMA, CCU, INVALID = UINT8_MAX };

    struct TaskPosition {
        OperatorId operatorId{INVALID_OPERATOR_ID};
        RankId rankId{INVALID_RANK_ID};
        StreamId streamId{INVALID_STREAM_ID};
        QueueId queueId{INVALID_QUEUE_ID};
        uint64_t launchIdx{std::numeric_limits<uint64_t>::max()};
        uint32_t blockId{std::numeric_limits<uint32_t>::max()};
        uint32_t pipe{std::numeric_limits<uint32_t>::max()};
        uint32_t taskId{std::numeric_limits<uint32_t>::max()};
    };

    using TaskLocation = TaskPosition;

    struct MemSlice {
        RankId rankId{INVALID_RANK_ID};
        MemType memType{MemType::INVALID};
        uint64_t offset{0};
        uint64_t len{0};
        uint64_t rawAddr{0};
    };

    struct AivPipeEvent {
        RankId rankId{INVALID_RANK_ID};
        uint64_t launchIdx{0};
        uint32_t blockId{std::numeric_limits<uint32_t>::max()};
        uint32_t curPipe{std::numeric_limits<uint32_t>::max()};
        uint32_t srcPipe{std::numeric_limits<uint32_t>::max()};
        uint32_t dstPipe{std::numeric_limits<uint32_t>::max()};
        int32_t eventId{0};
        uint32_t taskId{std::numeric_limits<uint32_t>::max()};
    };

    struct AivFlagSync {
        RankId currentRank{INVALID_RANK_ID};
        RankId flagOwnerRank{INVALID_RANK_ID};
        uint64_t launchIdx{0};
        uint32_t blockId{std::numeric_limits<uint32_t>::max()};
        uint32_t curPipe{std::numeric_limits<uint32_t>::max()};
        uint32_t taskId{std::numeric_limits<uint32_t>::max()};
        uint64_t commInfoOffset{0};
        int32_t value{0};
    };

    struct AivBarrierInfo {
        TaskPosition taskLoc;
        uint32_t pipeType{std::numeric_limits<uint32_t>::max()};
        bool merged{false};
        std::vector<NodeId> memberNodeIds;
        std::vector<uint32_t> memberTaskIds;
        std::vector<uint32_t> memberPipes;
    };

    struct AivSyncAllInfo {
        TaskPosition taskLoc;
        uint32_t syncRound{std::numeric_limits<uint32_t>::max()};
        bool merged{false};
        std::vector<NodeId> memberNodeIds;
        std::vector<uint32_t> memberTaskIds;
        std::vector<uint32_t> memberPipes;
    };

    struct AicpuNotify {
        RankId recordRankId{INVALID_RANK_ID};
        RankId waitRankId{INVALID_RANK_ID};
        uint32_t notifyId{INVALID_NOTIFY_ID};
    };

    struct CcuNotify {
        ChannelId channelId{INVALID_CHANNEL_ID};
        RankId recordRankId{INVALID_RANK_ID};
        RankId waitRankId{INVALID_RANK_ID};
        uint32_t dieId{INVALID_DIE_ID};
        uint16_t ckeId{INVALID_CCU_CKE};
        uint16_t ckeMask{INVALID_CCU_CKE_MASK};
    };

    struct CcuSqeParam {
        uint8_t dieId{0};
        uint8_t missionId{0};
        uint16_t timeout{0};
        uint32_t instStartId{0};
        uint32_t instCnt{0};
        uint32_t key{0};
        uint32_t argSize{0};
        std::array<uint64_t, CCU_SQE_ARGS_LEN> args{};
    };

    struct CcuSubGraphDesc {
        RankId rankId{INVALID_RANK_ID};
        std::vector<std::vector<CcuSqeParam>> ccuParams;
    };

    struct CcuRegisterSnapshot {
        std::map<uint16_t, uint64_t> xn;
        std::map<uint16_t, uint64_t> gsa;
        std::map<uint16_t, uint16_t> cke;
    };

    struct CcuTraceInfo {
        bool valid{false};
        std::string opName;
        std::string nodeRole;
        uint32_t instrId{0};
        TaskPosition position;
        TaskLocation taskLoc;
        QueueId queueId{INVALID_QUEUE_ID};
        uint32_t dieId{INVALID_DIE_ID};
        uint32_t missionId{INVALID_MISSION_ID};
        CcuRegisterSnapshot registerState;
        // loop start 节点的专属元数据
        uint16_t loopInstrIdStart{UINT16_MAX};
        uint16_t loopInstrIdEnd{UINT16_MAX};
        uint64_t loopCnt{UINT64_MAX};
        uint64_t loopExpandCnt{UINT64_MAX};
        NodeId loopEndNodeId{INVALID_NODE_ID};
        // loop 体内节点：所属 loop start 的 nodeId（INVALID_NODE_ID 表示不在 loop 内）
        NodeId loopStartNodeId{INVALID_NODE_ID};
    };

    class TaskNode {
    public:
        explicit TaskNode(TaskType type) : type_(type) {}
        TaskNode(TaskType type, const TaskPosition& position) : type_(type), loc_(position) {}
        virtual ~TaskNode() = default;

        TaskNode(TaskNode&&) = default;
        TaskNode& operator=(TaskNode&&) = default;
        TaskNode(const TaskNode&) = delete;
        TaskNode& operator=(const TaskNode&) = delete;

        virtual std::string Describe() const = 0;
        virtual std::string DescribeShort() const { return Describe(); }

        TaskType GetType() const { return type_; }
        OperatorId GetOperatorId() const { return operatorId_; }
        void SetOperatorId(OperatorId operatorId)
        {
            operatorId_ = operatorId;
            loc_.operatorId = operatorId;
        }
        NodeId GetNodeId() const { return nodeId_; }
        void SetNodeId(NodeId id) { nodeId_ = id; }

        const TaskPosition& GetPosition() const { return loc_; }
        void SetPosition(const TaskPosition& position)
        {
            loc_ = position;
            operatorId_ = position.operatorId;
            if (ccuTraceValid_) {
                ccuTrace_.position = position;
                ccuTrace_.taskLoc = position;
                ccuTrace_.queueId = position.queueId;
            }
        }
        const TaskLocation& GetLocation() const { return loc_; }
        void SetLocation(const TaskLocation& location) { SetPosition(location); }
        bool HasCcuTrace() const { return ccuTraceValid_; }
        const CcuTraceInfo& GetCcuTrace() const { return ccuTrace_; }
        void SetCcuTrace(CcuTraceInfo trace)
        {
            if (trace.position.rankId == INVALID_RANK_ID && trace.taskLoc.rankId != INVALID_RANK_ID) {
                trace.position = trace.taskLoc;
            }
            if (trace.taskLoc.rankId == INVALID_RANK_ID && trace.position.rankId != INVALID_RANK_ID) {
                trace.taskLoc = trace.position;
            }
            if (trace.position.queueId == INVALID_QUEUE_ID && trace.queueId != INVALID_QUEUE_ID) {
                trace.position.queueId = trace.queueId;
            }
            if (trace.taskLoc.queueId == INVALID_QUEUE_ID && trace.queueId != INVALID_QUEUE_ID) {
                trace.taskLoc.queueId = trace.queueId;
            }
            if (trace.queueId == INVALID_QUEUE_ID) {
                trace.queueId = trace.position.queueId;
            }
            if (trace.position.queueId == INVALID_QUEUE_ID) {
                trace.position.queueId = trace.taskLoc.queueId;
            }
            if (trace.taskLoc.queueId == INVALID_QUEUE_ID) {
                trace.taskLoc.queueId = trace.position.queueId;
            }
            ccuTrace_ = std::move(trace);
            ccuTraceValid_ = ccuTrace_.valid;
        }
        void ClearCcuTrace()
        {
            ccuTrace_ = CcuTraceInfo{};
            ccuTraceValid_ = false;
        }

        const std::vector<TaskNode*>& GetParents() const { return parents_; }
        const std::vector<TaskNode*>& GetChildren() const { return children_; }
        bool AddParent(TaskNode* parentNode);
        bool AddChild(TaskNode* childNode);
        bool RemoveParent(const TaskNode* parentNode);
        bool RemoveChild(const TaskNode* childNode);

    protected:
        TaskType type_{TaskType::INVALID};
        OperatorId operatorId_{INVALID_OPERATOR_ID};
        NodeId nodeId_{INVALID_NODE_ID};
        TaskPosition loc_;
        CcuTraceInfo ccuTrace_;
        bool ccuTraceValid_{false};

    private:
        std::vector<TaskNode*> parents_;
        std::vector<TaskNode*> children_;
    };

    class TaskTransMem : public TaskNode {
    public:
        TaskTransMem(const MemSlice& src, const MemSlice& dst, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::TRANS_MEM),
              src_(src),
              dst_(dst),
              protocol_(protocol)
        {}

        ~TaskTransMem() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        const MemSlice& GetSrc() const { return src_; }
        const MemSlice& GetDst() const { return dst_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        MemSlice src_;
        MemSlice dst_;
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskBatchTransMem : public TaskNode {
    public:
        explicit TaskBatchTransMem(ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::BATCH_TRANS_MEM),
              protocol_(protocol)
        {}

        ~TaskBatchTransMem() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        const std::vector<MemSlice>& GetSrcs() const { return srcs_; }
        const std::vector<MemSlice>& GetDsts() const { return dsts_; }
        const std::vector<MemSlice>& GetMergedSrcs() const { return mergedSrcs_; }
        const std::vector<MemSlice>& GetMergedDsts() const { return mergedDsts_; }
        void AddSrcMemSlice(const MemSlice& slice) { srcs_.push_back(slice); }
        void AddDstMemSlice(const MemSlice& slice) { dsts_.push_back(slice); }
        void AddMergedSrcMemSlice(const MemSlice& slice) { mergedSrcs_.push_back(slice); }
        void AddMergedDstMemSlice(const MemSlice& slice) { mergedDsts_.push_back(slice); }
        void SetSrcMemSlices(const std::vector<MemSlice>& srcs) { srcs_ = srcs; }
        void SetSrcMemSlices(std::vector<MemSlice>&& srcs) { srcs_ = std::move(srcs); }
        void SetDstMemSlices(const std::vector<MemSlice>& dsts) { dsts_ = dsts; }
        void SetDstMemSlices(std::vector<MemSlice>&& dsts) { dsts_ = std::move(dsts); }
        void SetMergedSrcMemSlices(const std::vector<MemSlice>& mergedSrcs) { mergedSrcs_ = mergedSrcs; }
        void SetMergedSrcMemSlices(std::vector<MemSlice>&& mergedSrcs) { mergedSrcs_ = std::move(mergedSrcs); }
        void SetMergedDstMemSlices(const std::vector<MemSlice>& mergedDsts) { mergedDsts_ = mergedDsts; }
        void SetMergedDstMemSlices(std::vector<MemSlice>&& mergedDsts) { mergedDsts_ = std::move(mergedDsts); }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        std::vector<MemSlice> srcs_{};
        std::vector<MemSlice> dsts_{};
        std::vector<MemSlice> mergedSrcs_{};
        std::vector<MemSlice> mergedDsts_{};
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskReduce : public TaskNode {
    public:
        TaskReduce(
            const MemSlice& src, const MemSlice& dst, uint8_t dataType, uint8_t reduceOp,
            ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::REDUCE),
              src_(src),
              srcs_{src},
              dst_(dst),
              dataType_(dataType),
              reduceOp_(reduceOp),
              protocol_(protocol)
        {}
        TaskReduce(
            const std::vector<MemSlice>& srcs, const MemSlice& dst, uint8_t dataType, uint8_t reduceOp,
            ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::REDUCE),
              src_(srcs.empty() ? MemSlice{} : srcs.front()),
              srcs_(srcs),
              dst_(dst),
              dataType_(dataType),
              reduceOp_(reduceOp),
              protocol_(protocol)
        {}

        ~TaskReduce() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        const MemSlice& GetSrc() const { return src_; }
        const std::vector<MemSlice>& GetSrcs() const { return srcs_; }
        const MemSlice& GetDst() const { return dst_; }
        uint8_t GetDataType() const { return dataType_; }
        uint8_t GetReduceOp() const { return reduceOp_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        MemSlice src_;
        std::vector<MemSlice> srcs_;
        MemSlice dst_;
        uint8_t dataType_{0};
        uint8_t reduceOp_{0};
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskBatchReduce : public TaskNode {
    public:
        TaskBatchReduce(uint8_t dataType, uint8_t reduceOp, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::BATCH_REDUCE),
              dataType_(dataType),
              reduceOp_(reduceOp),
              protocol_(protocol)
        {}

        ~TaskBatchReduce() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        const std::vector<std::vector<MemSlice>>& GetSrcs() const { return srcs_; }
        const std::vector<MemSlice>& GetDsts() const { return dsts_; }
        const std::vector<std::vector<MemSlice>>& GetMergedSrcs() const { return mergedSrcs_; }
        const std::vector<MemSlice>& GetMergedDsts() const { return mergedDsts_; }
        void AddSrcMemSlice(const std::vector<MemSlice>& slices) { srcs_.push_back(slices); }
        void AddDstMemSlice(const MemSlice& slice) { dsts_.push_back(slice); }
        void AddMergedSrcMemSlice(const std::vector<MemSlice>& slices) { mergedSrcs_.push_back(slices); }
        void AddMergedDstMemSlice(const MemSlice& slice) { mergedDsts_.push_back(slice); }
        void SetSrcMemSlices(const std::vector<std::vector<MemSlice>>& srcs) { srcs_ = srcs; }
        void SetSrcMemSlices(std::vector<std::vector<MemSlice>>&& srcs) { srcs_ = std::move(srcs); }
        void SetDstMemSlices(const std::vector<MemSlice>& dsts) { dsts_ = dsts; }
        void SetDstMemSlices(std::vector<MemSlice>&& dsts) { dsts_ = std::move(dsts); }
        void SetMergedSrcMemSlices(const std::vector<std::vector<MemSlice>>& mergedSrcs) { mergedSrcs_ = mergedSrcs; }
        void SetMergedSrcMemSlices(std::vector<std::vector<MemSlice>>&& mergedSrcs)
        {
            mergedSrcs_ = std::move(mergedSrcs);
        }
        void SetMergedDstMemSlices(const std::vector<MemSlice>& mergedDsts) { mergedDsts_ = mergedDsts; }
        void SetMergedDstMemSlices(std::vector<MemSlice>&& mergedDsts) { mergedDsts_ = std::move(mergedDsts); }
        uint8_t GetDataType() const { return dataType_; }
        uint8_t GetReduceOp() const { return reduceOp_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        std::vector<std::vector<MemSlice>> srcs_{};
        std::vector<MemSlice> dsts_{};
        std::vector<std::vector<MemSlice>> mergedSrcs_{};
        std::vector<MemSlice> mergedDsts_{};
        uint8_t dataType_{0};
        uint8_t reduceOp_{0};
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskRecordAICPU : public TaskNode {
    public:
        explicit TaskRecordAICPU(const AicpuNotify& notify, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::RECORD),
              notify_(notify),
              protocol_(protocol)
        {}

        ~TaskRecordAICPU() override = default;
        std::string Describe() const override;

        const AicpuNotify& GetNotify() const { return notify_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        AicpuNotify notify_;
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskWaitAICPU : public TaskNode {
    public:
        explicit TaskWaitAICPU(const AicpuNotify& notify, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::WAIT),
              notify_(notify),
              protocol_(protocol)
        {}

        ~TaskWaitAICPU() override = default;
        std::string Describe() const override;

        const AicpuNotify& GetNotify() const { return notify_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        AicpuNotify notify_;
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskRecordCCU : public TaskNode {
    public:
        explicit TaskRecordCCU(const CcuNotify& notify, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::RECORD),
              notify_(notify),
              protocol_(protocol)
        {}

        ~TaskRecordCCU() override = default;
        std::string Describe() const override;

        const CcuNotify& GetNotify() const { return notify_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        CcuNotify notify_;
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskWaitCCU : public TaskNode {
    public:
        explicit TaskWaitCCU(const CcuNotify& notify, ProtocolType protocol = ProtocolType::INVALID)
            : TaskNode(TaskType::WAIT),
              notify_(notify),
              protocol_(protocol)
        {}

        ~TaskWaitCCU() override = default;
        std::string Describe() const override;

        const CcuNotify& GetNotify() const { return notify_; }
        ProtocolType GetProtocol() const { return protocol_; }

    private:
        CcuNotify notify_;
        ProtocolType protocol_{ProtocolType::INVALID};
    };

    class TaskCcuGraph : public TaskNode {
    public:
        explicit TaskCcuGraph(const CcuSubGraphDesc& ccuDesc) : TaskNode(TaskType::CCU_GRAPH), ccuDesc_(ccuDesc) {}
        explicit TaskCcuGraph(CcuSubGraphDesc&& ccuDesc) : TaskNode(TaskType::CCU_GRAPH), ccuDesc_(std::move(ccuDesc))
        {}

        ~TaskCcuGraph() override = default;
        std::string Describe() const override;

        const CcuSubGraphDesc& GetCcuDesc() const { return ccuDesc_; }
        size_t GetQueueNum() const { return ccuDesc_.ccuParams.size(); }
        RankId GetRankId() const { return ccuDesc_.rankId; }

        void AddCcuParam(size_t queueId, const CcuSqeParam& param)
        {
            if (ccuDesc_.ccuParams.size() <= queueId) {
                ccuDesc_.ccuParams.resize(queueId + 1);
            }
            ccuDesc_.ccuParams[queueId].push_back(param);
        }

        bool IsValidQueue(size_t queueId) const { return queueId < ccuDesc_.ccuParams.size(); }

        uint32_t GetMissionEndInstrId(size_t queueId) const
        {
            if (!IsValidQueue(queueId) || ccuDesc_.ccuParams[queueId].empty()) {
                return 0;
            }
            const auto& lastParam = ccuDesc_.ccuParams[queueId].back();
            return lastParam.instStartId + lastParam.instCnt;
        }

        uint32_t GetDieId(size_t queueId) const
        {
            if (!IsValidQueue(queueId) || ccuDesc_.ccuParams[queueId].empty()) {
                return INVALID_DIE_ID;
            }
            return ccuDesc_.ccuParams[queueId][0].dieId;
        }

        uint64_t GetSqeArg(size_t queueId, size_t sqeIndex, size_t sqeArgsId) const
        {
            if (!IsValidQueue(queueId) || sqeArgsId >= CCU_SQE_ARGS_LEN
                || sqeIndex >= ccuDesc_.ccuParams[queueId].size()) {
                return 0;
            }
            return ccuDesc_.ccuParams[queueId][sqeIndex].args[sqeArgsId];
        }

    private:
        CcuSubGraphDesc ccuDesc_;
    };

    class TaskAivGraph : public TaskNode {
    public:
        TaskAivGraph(RankId rankId, uint64_t launchIdx, uint64_t hostStreamId)
            : TaskNode(TaskType::AIV_GRAPH),
              rankId_(rankId),
              launchIdx_(launchIdx),
              hostStreamId_(hostStreamId)
        {}
        ~TaskAivGraph() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        RankId GetRankId() const { return rankId_; }
        uint64_t GetLaunchIdx() const { return launchIdx_; }
        uint64_t GetHostStreamId() const { return hostStreamId_; }

    private:
        RankId rankId_{INVALID_RANK_ID};
        uint64_t launchIdx_{0};
        uint64_t hostStreamId_{0};
    };

    class TaskAivSetFlag : public TaskNode {
    public:
        explicit TaskAivSetFlag(const AivPipeEvent& event) : TaskNode(TaskType::AIV_SET_FLAG), event_(event) {}
        ~TaskAivSetFlag() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivPipeEvent& GetEvent() const { return event_; }

    private:
        AivPipeEvent event_;
    };

    class TaskAivWaitFlag : public TaskNode {
    public:
        explicit TaskAivWaitFlag(const AivPipeEvent& event) : TaskNode(TaskType::AIV_WAIT_FLAG), event_(event) {}
        ~TaskAivWaitFlag() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivPipeEvent& GetEvent() const { return event_; }

    private:
        AivPipeEvent event_;
    };

    class TaskAivPipeBarrier : public TaskNode {
    public:
        explicit TaskAivPipeBarrier(AivBarrierInfo info) : TaskNode(TaskType::AIV_PIPE_BARRIER), info_(std::move(info))
        {}
        ~TaskAivPipeBarrier() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivBarrierInfo& GetInfo() const { return info_; }

    private:
        AivBarrierInfo info_;
    };

    class TaskAivSyncAll : public TaskNode {
    public:
        explicit TaskAivSyncAll(AivSyncAllInfo info) : TaskNode(TaskType::AIV_SYNC_ALL), info_(std::move(info)) {}
        ~TaskAivSyncAll() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivSyncAllInfo& GetInfo() const { return info_; }

    private:
        AivSyncAllInfo info_;
    };

    class TaskAivSendFlag : public TaskNode {
    public:
        explicit TaskAivSendFlag(const AivFlagSync& flag) : TaskNode(TaskType::AIV_SEND_FLAG), flag_(flag) {}
        ~TaskAivSendFlag() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivFlagSync& GetFlag() const { return flag_; }

    private:
        AivFlagSync flag_;
    };

    class TaskAivRecvFlag : public TaskNode {
    public:
        explicit TaskAivRecvFlag(const AivFlagSync& flag) : TaskNode(TaskType::AIV_RECV_FLAG), flag_(flag) {}
        ~TaskAivRecvFlag() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;
        const AivFlagSync& GetFlag() const { return flag_; }

    private:
        AivFlagSync flag_;
    };

    class TaskStart : public TaskNode {
    public:
        explicit TaskStart(BoundaryType boundaryType) : TaskNode(TaskType::START), boundaryType_(boundaryType) {}
        ~TaskStart() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        BoundaryType GetBoundaryType() const { return boundaryType_; }

    private:
        BoundaryType boundaryType_{BoundaryType::INVALID};
    };

    class TaskEnd : public TaskNode {
    public:
        explicit TaskEnd(BoundaryType boundaryType) : TaskNode(TaskType::END), boundaryType_(boundaryType) {}
        ~TaskEnd() override = default;
        std::string Describe() const override;
        std::string DescribeShort() const override;

        BoundaryType GetBoundaryType() const { return boundaryType_; }

    private:
        BoundaryType boundaryType_{BoundaryType::INVALID};
    };
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // CHECKER_TASK_GRAPH_GENERATOR_V3_TASK_DEF_V3_H
