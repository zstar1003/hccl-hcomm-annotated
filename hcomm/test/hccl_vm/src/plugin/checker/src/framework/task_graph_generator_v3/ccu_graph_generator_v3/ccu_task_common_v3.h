/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_COMMON_V3_H
#define HCCLV2_CCU_COMMON_V3_H

#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <array>
#include <unordered_map>
#include <hccl_types.h>

#include "base.h"
#include "log.h"
#include "data_slice.h"
#include "ccu_instr_info.h"
#include "data_type.h"
#include "../task_graph_generator_v3.h"

namespace HcclSim {
class StorageManager;

namespace TaskGraphGeneratorV3 {

    constexpr uint32_t MAX_LOOPGROUP_NUM_V3 = 1024;
    constexpr uint32_t INVALID_QID_V3_V3 = 0xffffffff;

    enum class CcuNodeRoleV3 : uint8_t {
        HEAD = 0,
        LOCAL_COPY,
        LOCAL_REDUCE,
        LOCAL_BATCH_REDUCE,
        READ,
        READ_REDUCE,
        WRITE,
        WRITE_REDUCE,
        POST,
        WAIT,
        LOCAL_POST_TO,
        LOCAL_WAIT_FROM,
        LOOP_START,
        LOOP_END,
        SUB_GRAPH_END,
        UNKNOWN,
    };

    struct CcuNodeMetaV3 {
        CcuNodeRoleV3 role{CcuNodeRoleV3::UNKNOWN};
        RankId peerRank{INVALID_RANK_ID};
        uint32_t dieId{INVALID_DIE_ID};
        uint16_t ckeId{INVALID_CCU_CKE};
        uint16_t remainingCkeMask{0};
        bool invalidPost{false};
    };

    struct LoopInfoV3 {
        LoopInfoV3(TaskNode* start, TaskNode* end) : loopStart(start), loopEnd(end) {}
        TaskNode* loopStart{nullptr};
        TaskNode* loopEnd{nullptr};
        uint16_t instrIdStart{UINT16_MAX};
        uint16_t instrIdEnd{UINT16_MAX};
        uint64_t loopCnt{UINT64_MAX};
        uint64_t expandCnt{1};
        NodeId endNodeId{INVALID_NODE_ID};
    };

    struct BilateralWaitInfoV3 {
        std::vector<TaskNode*> waitNodes;
    };

    struct BilateralPostInfoV3 {
        std::vector<TaskNode*> asyncNodes;
    };

    struct BilateralNodeV3 {
        BilateralNodeV3(TaskNode* asyncNodeIn, TaskNode* wait, TaskNode* post)
            : backwardWait(wait),
              forwardPost(post),
              asyncNode(asyncNodeIn)
        {}
        TaskNode* backwardWait{nullptr};
        TaskNode* forwardPost{nullptr};
        TaskNode* asyncNode{nullptr};
    };

    class CcuGraphStateV3 {
    public:
        CcuGraphStateV3(TaskGraphGeneratorV3& graphIn, TaskCcuGraph& ccuGraphIn);

        HcclResult InitInstrInfo(StorageManager& storage);
        HcclResult CreateHeadNode();
        HcclResult AppendGeneratedNode(
            std::unique_ptr<TaskNode> node, RankId rankId, uint32_t queId, CcuNodeRoleV3 role, TaskNode*& outNode,
            RankId peerRank = INVALID_RANK_ID, uint16_t remainingCkeMask = 0, uint32_t dieId = INVALID_DIE_ID,
            uint16_t ckeId = INVALID_CCU_CKE, bool invalidPost = false);
        HcclResult AppendGeneratedNode(
            std::unique_ptr<TaskNode> node, const TaskPosition& position, CcuNodeRoleV3 role, TaskNode*& outNode,
            RankId peerRank = INVALID_RANK_ID, uint16_t remainingCkeMask = 0, uint32_t dieId = INVALID_DIE_ID,
            uint16_t ckeId = INVALID_CCU_CKE, bool invalidPost = false);

        void SetNodeMeta(TaskNode* node, const CcuNodeMetaV3& meta);
        const CcuNodeMetaV3* GetNodeMeta(const TaskNode* node) const;
        CcuNodeRoleV3 GetNodeRole(const TaskNode* node) const;
        RankId GetNodePeerRank(const TaskNode* node) const;
        uint16_t GetNodeRemainingCkeMask(const TaskNode* node) const;
        void SetNodeRemainingCkeMask(TaskNode* node, uint16_t remainingCkeMask);
        void SetNodePeerRank(TaskNode* node, RankId peerRank);
        std::string Describe() const;
        std::string DescribeNextInstructionByQueue() const;

        void GetSqe(uint32_t queId, uint16_t sqeArgsId, uint64_t& argVal);
        void GetDieId(uint32_t queId, uint32_t& dieId) const;
        StorageManager& GetStorageManager() const { return storage_; }
        uint32_t GetMissionEndInstrId(uint32_t queId) const;
        RankId GetRankId() const { return rankId; }

        TaskGraphGeneratorV3& graph;
        TaskCcuGraph& ccuGraph;
        RankId rankId{INVALID_RANK_ID};
        std::vector<hcomm::CcuRep::CcuInstrInfo> instrInfo;
        std::vector<std::vector<CcuSqeParam>> ccuParams;
        std::vector<u32> ccuParamIndexs;
        std::vector<u32> microCodePosInQue;
        std::vector<u32> startInstrIdInQue;
        std::vector<bool> instrQueStatus;
        std::vector<TaskNode*> tailNodes;
        TaskNode* ccuHeadTaskNode{nullptr};
        TaskNode* ccuSubGraphEndNode{nullptr};
        bool isGenGraphed{false};
        uint32_t queueNum_{0};
        uint32_t loopGroupIdx{0};
        std::array<uint32_t, MAX_LOOPGROUP_NUM_V3> loopIdx{};
        std::vector<std::vector<LoopInfoV3>> loopGroupInfo_;
        std::vector<NodeId> loopNodeIdStack_; // 嵌套 loop 栈：栈顶为当前最内层 loop start 节点 ID
        std::map<TaskNode*, TaskNode*> localPostWaitPairs_;
        std::vector<TaskNode*> parallelNodes_;
        std::vector<BilateralWaitInfoV3> waitInfoTmp_;
        std::vector<BilateralPostInfoV3> postInfoTmp_;
        std::vector<std::map<TaskNode*, TaskNode*>> bilateralPart1_;
        std::vector<std::map<TaskNode*, TaskNode*>> bilateralPart2_;
        std::vector<std::vector<BilateralNodeV3>> bilateralNodes_;
        size_t internalNodeCount{0};
        StorageManager& storage_;

    private:
        std::map<const TaskNode*, CcuNodeMetaV3> nodeMetas_;
    };

    void PrintCcuGraph(TaskNode* dummyStart);
    HcclResult GetPeerRankByTaskNode(CcuGraphStateV3* curCcuTask, TaskNode* currNode, RankId& peerRank);

    HcclResult CollectBilateralWaitInfo(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node);
    HcclResult
    CollectBilateralPostInfo(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node, bool isLast = false);

    HcclResult AppendTailNode(CcuGraphStateV3* curCcuTask, uint32_t queId, TaskNode* node);

    MemSlice MakeCcuMemSlice(RankId rankId, const DataSlice& slice);
    std::vector<MemSlice> MakeCcuMemSlices(RankId rankId, const std::vector<DataSlice>& slices);

    void AddLocalCopy(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice);
    void AddLocalReduce(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
    void AddLocalBatchReduce(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const std::vector<DataSlice>& srcSlices,
        const DataSlice& dstSlice, DataType hcclDataType);
    HcclResult AddBatchTransMem(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, std::vector<MemSlice> srcs,
        std::vector<MemSlice> dsts, std::vector<MemSlice> mergedSrcs, std::vector<MemSlice> mergedDsts,
        CcuNodeRoleV3 role, RankId peerRank = INVALID_RANK_ID);
    HcclResult AddBatchReduce(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, std::vector<std::vector<MemSlice>> srcGroups,
        std::vector<MemSlice> dsts, std::vector<std::vector<MemSlice>> mergedSrcGroups,
        std::vector<MemSlice> mergedDsts, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp,
        CcuNodeRoleV3 role, RankId peerRank = INVALID_RANK_ID);
    void AddWrite(
        uint32_t rankId, uint32_t rmtRankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice);
    void AddRead(
        uint32_t rankId, uint32_t rmtRankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice);
    void AddWriteReduce(
        uint32_t rankId, uint32_t rmtRankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
    void AddReadReduce(
        uint32_t rankId, uint32_t rmtRankId, uint32_t queId, CcuGraphStateV3* curCcuTask, const DataSlice& srcSlice,
        const DataSlice& dstSlice, HcclDataType checkerDataType, HcclReduceOp checkerReduceOp);
    TaskNode* AddLoopStartTask(
        uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, CcuGraphStateV3* curCcuTask,
        uint16_t instrIdStart = UINT16_MAX, uint16_t instrIdEnd = UINT16_MAX, uint64_t loopCnt = UINT64_MAX,
        uint64_t expandCnt = 1);
    TaskNode* AddLoopEndTask(uint32_t queId, uint32_t loopIdx, uint32_t loopGroupIdx, CcuGraphStateV3* curCcuTask);
    void AddPost(
        uint32_t rankId, uint32_t rmtRankId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t rmtDieId,
        uint16_t rmtCKEId, uint16_t setRmtCKEMask);
    TaskNode* AddWait(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t dieId, uint16_t waitCKEId,
        uint16_t remoteWaitMask);
    void AddLocalPost(
        uint32_t rankId, uint32_t queId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint16_t setCKEId,
        uint16_t setCKEMask, bool invalidPost = false);
    TaskNode* AddLocalWait(
        uint32_t rankId, uint32_t queId, CcuGraphStateV3* curCcuTask, uint32_t dieId, uint16_t waitCKEId,
        uint16_t localWaitMask);
    void AddCcuSubGraphEnd(TaskNode* node, CcuGraphStateV3* curCcuTask);
    void AddNodeRelation(TaskNode* parent, TaskNode* child);

    HcclResult ClearWaitMask(RankId rankId, uint32_t dieId, uint16_t waitCKEId, uint16_t waitCKEMask);
    HcclResult ProcessSetMask(
        RankId rankId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t setCKEId,
        uint16_t setCKEMask, bool invalidPost = false);

    uint16_t
    UpdateId(uint16_t CKEId, uint32_t curLoopIdx, uint32_t expandOffset, uint32_t ckOffset, uint32_t curExpandCnt);
    HcclResult GenSliceFromMs(uint16_t msId, uint64_t len, DataSlice& slice);
} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // HCCLV2_CCU_COMMON_V3_H
