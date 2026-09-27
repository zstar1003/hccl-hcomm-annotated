/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ccu_loop_merge_v3.h"

#include <algorithm>
#include <new>
#include <sstream>
#include <utility>

#include "ccu_task_transform_v3.h"
#include "sim_log.h"
#include "utils/error_codes.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    namespace {

        // 将src中的CKE操作合并到dst中，相同ckeId的操作通过OR合并mask
        void AppendCkeOps(std::vector<CcuLoopCkeOpV3>& dst, const std::vector<CcuLoopCkeOpV3>& src)
        {
            for (const auto& op : src) {
                if (op.mask == 0) {
                    continue;
                }
                auto iter = std::find_if(dst.begin(), dst.end(), [&op](const CcuLoopCkeOpV3& exist) {
                    return exist.ckeId == op.ckeId;
                });
                if (iter == dst.end()) {
                    dst.push_back(op);
                } else {
                    iter->mask = static_cast<uint16_t>(iter->mask | op.mask);
                }
            }
        }

        // 判断所有内存切片是否都属于指定的rank
        bool SameRankGroup(const std::vector<MemSlice>& slices, RankId rankId)
        {
            return std::all_of(slices.begin(), slices.end(), [rankId](const MemSlice& slice) {
                return slice.rankId == rankId;
            });
        }

        // 判断所有切片是否都属于同一个对端rank（非本端），并输出该对端rankId
        bool SamePeerGroup(const std::vector<MemSlice>& slices, RankId localRank, RankId& peerRank)
        {
            peerRank = INVALID_RANK_ID;
            for (const auto& slice : slices) {
                if (slice.rankId == localRank) {
                    return false;
                }
                if (peerRank == INVALID_RANK_ID) {
                    peerRank = slice.rankId;
                    continue;
                }
                if (peerRank != slice.rankId) {
                    return false;
                }
            }
            return peerRank != INVALID_RANK_ID;
        }

        // 根据src/dst的rank归属，推断传输指令的角色：LOCAL_COPY（本地拷贝）、WRITE（写远端）、READ（读远端）
        // 要求所有切片对的角色和对端rank必须一致，否则返回不支持
        HcclResult ResolveTransRole(
            RankId rankId, const std::vector<MemSlice>& srcs, const std::vector<MemSlice>& dsts, CcuNodeRoleV3& role,
            RankId& peerRank)
        {
            if (srcs.empty() || srcs.size() != dsts.size()) {
                return HCCL_E_NOT_SUPPORT;
            }

            role = CcuNodeRoleV3::UNKNOWN;
            peerRank = INVALID_RANK_ID;
            for (size_t index = 0; index < srcs.size(); ++index) {
                CcuNodeRoleV3 curRole = CcuNodeRoleV3::UNKNOWN;
                RankId curPeer = INVALID_RANK_ID;
                if (srcs[index].rankId == rankId && dsts[index].rankId == rankId) {
                    curRole = CcuNodeRoleV3::LOCAL_COPY;
                } else if (srcs[index].rankId == rankId && dsts[index].rankId != rankId) {
                    curRole = CcuNodeRoleV3::WRITE;
                    curPeer = dsts[index].rankId;
                } else if (srcs[index].rankId != rankId && dsts[index].rankId == rankId) {
                    curRole = CcuNodeRoleV3::READ;
                    curPeer = srcs[index].rankId;
                } else {
                    return HCCL_E_NOT_SUPPORT;
                }

                if (role == CcuNodeRoleV3::UNKNOWN) {
                    role = curRole;
                    peerRank = curPeer;
                    continue;
                }
                if (role != curRole || peerRank != curPeer) {
                    return HCCL_E_NOT_SUPPORT;
                }
            }
            return HCCL_SUCCESS;
        }

        // 根据srcGroups/dst的rank归属，推断Reduce指令的角色：
        // LOCAL_BATCH_REDUCE（本地归约）、WRITE_REDUCE（写远端归约）、READ_REDUCE（读远端归约）
        HcclResult ResolveReduceRole(
            RankId rankId, const std::vector<std::vector<MemSlice>>& srcGroups, const std::vector<MemSlice>& dsts,
            CcuNodeRoleV3& role, RankId& peerRank)
        {
            if (srcGroups.empty() || srcGroups.size() != dsts.size()) {
                return HCCL_E_NOT_SUPPORT;
            }

            role = CcuNodeRoleV3::UNKNOWN;
            peerRank = INVALID_RANK_ID;
            for (size_t index = 0; index < srcGroups.size(); ++index) {
                CcuNodeRoleV3 curRole = CcuNodeRoleV3::UNKNOWN;
                RankId curPeer = INVALID_RANK_ID;
                if (dsts[index].rankId == rankId && SameRankGroup(srcGroups[index], rankId)) {
                    curRole = CcuNodeRoleV3::LOCAL_BATCH_REDUCE;
                } else if (dsts[index].rankId != rankId && SameRankGroup(srcGroups[index], rankId)) {
                    curRole = CcuNodeRoleV3::WRITE_REDUCE;
                    curPeer = dsts[index].rankId;
                } else if (dsts[index].rankId == rankId && SamePeerGroup(srcGroups[index], rankId, curPeer)) {
                    curRole = CcuNodeRoleV3::READ_REDUCE;
                } else {
                    return HCCL_E_NOT_SUPPORT;
                }

                if (role == CcuNodeRoleV3::UNKNOWN) {
                    role = curRole;
                    peerRank = curPeer;
                    continue;
                }
                if (role != curRole || peerRank != curPeer) {
                    return HCCL_E_NOT_SUPPORT;
                }
            }
            return HCCL_SUCCESS;
        }

        // 处理指令的wait操作列表：对每个CKE等待信号执行掩码等待，若条件不满足则提前返回
        HcclResult ProcessWaitOps(
            CcuGraphStateV3* curCcuTask, uint32_t queId, const std::vector<CcuLoopCkeOpV3>& waitOps, bool& isContinue)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            for (const auto& wait : waitOps) {
                CHK_RET(ProcessWaitMask(rankId, dieId, curCcuTask, queId, wait.ckeId, wait.mask, isContinue));
                if (!isContinue) {
                    return HCCL_SUCCESS;
                }
            }
            return HCCL_SUCCESS;
        }

        // 处理指令的set操作列表：对每个CKE设置信号掩码
        HcclResult ProcessSetOps(CcuGraphStateV3* curCcuTask, uint32_t queId, const std::vector<CcuLoopCkeOpV3>& setOps)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            for (const auto& set : setOps) {
                CHK_RET(ProcessSetMask(rankId, dieId, curCcuTask, queId, set.ckeId, set.mask));
            }
            return HCCL_SUCCESS;
        }

        // 清除已处理的wait操作对应的CKE掩码，避免重复等待
        HcclResult ClearWaitOps(CcuGraphStateV3* curCcuTask, uint32_t queId, const std::vector<CcuLoopCkeOpV3>& waitOps)
        {
            RankId rankId = curCcuTask->GetRankId();
            uint32_t dieId = INVALID_DIE_ID;
            curCcuTask->GetDieId(queId, dieId);
            for (const auto& wait : waitOps) {
                CHK_RET(ClearWaitMask(rankId, dieId, wait.ckeId, wait.mask));
            }
            return HCCL_SUCCESS;
        }

    } // namespace

    bool CcuLoopInstrV3::MemMerge(bool allowOverlap)
    {
        (void)allowOverlap;
        return true;
    }

    std::shared_ptr<CcuLoopInstrV3>
    CcuLoopInstrV3::InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const
    {
        (void)instrs;
        return nullptr;
    }

    std::set<uint16_t> CcuLoopInstrV3::GetUsedMS() const { return {}; }

    std::set<uint16_t> CcuLoopInstrV3::GetUsedCKE() const
    {
        std::set<uint16_t> used = GetUsedCKEFromOps(waitOps);
        const auto setUsed = GetUsedCKEFromOps(setOps);
        used.insert(setUsed.begin(), setUsed.end());
        return used;
    }

    std::string CcuLoopInstrV3::Describe() const
    {
        std::ostringstream os;
        os << "{rankId=" << rankId << ", dieId=" << dieId << ", instrId=" << instrId << "}";
        return os.str();
    }

    void CcuLoopInstrV3::AddWait(uint16_t ckeId, uint16_t mask)
    {
        AppendCkeOps(waitOps, {CcuLoopCkeOpV3{ckeId, mask}});
    }

    void CcuLoopInstrV3::AddSet(uint16_t ckeId, uint16_t mask) { AppendCkeOps(setOps, {CcuLoopCkeOpV3{ckeId, mask}}); }

    // 将同一指令的多个内存切片按rank/memType/offset排序后合并为连续区间
    // MS_CCU类型不参与合并；若allowOverlap为false且检测到重叠则返回失败
    bool CcuLoopInstrV3::MergeMemSlices(std::vector<MemSlice>& slices, bool allowOverlap) const
    {
        if (slices.size() <= 1) {
            return true;
        }

        std::sort(slices.begin(), slices.end(), [](const MemSlice& lhs, const MemSlice& rhs) {
            if (lhs.rankId != rhs.rankId) {
                return lhs.rankId < rhs.rankId;
            }
            if (lhs.memType != rhs.memType) {
                return lhs.memType < rhs.memType;
            }
            return lhs.offset < rhs.offset;
        });

        std::vector<MemSlice> merged;
        merged.push_back(slices.front());
        for (size_t index = 1; index < slices.size(); ++index) {
            MemSlice& last = merged.back();
            const MemSlice& cur = slices[index];
            if (last.rankId == cur.rankId && last.memType == cur.memType && cur.memType != MemType::MS_CCU) {
                const uint64_t lastEnd = last.offset + last.len;
                const uint64_t curEnd = cur.offset + cur.len;
                if (lastEnd > cur.offset && !allowOverlap) {
                    HCCL_VM_WARN(
                        "Skip merged-loop optimization because memory slices from different "
                        "loop expansions overlap, loopInstrId={}",
                        instrId);
                    return false;
                }
                // 相邻或重叠的切片合并为一个，取并集区间
                if (lastEnd >= cur.offset) {
                    last.len = std::max(lastEnd, curEnd) - last.offset;
                    continue;
                }
            }
            merged.push_back(cur);
        }
        slices = std::move(merged);
        return true;
    }

    void CcuLoopInstrV3::CopyBaseTo(CcuLoopInstrV3& dst) const
    {
        dst.rankId = rankId;
        dst.dieId = dieId;
        dst.instrId = instrId;
        dst.waitOps = waitOps;
        dst.setOps = setOps;
    }

    std::set<uint16_t> CcuLoopInstrV3::GetUsedCKEFromOps(const std::vector<CcuLoopCkeOpV3>& ops)
    {
        std::set<uint16_t> used;
        for (const auto& op : ops) {
            if (op.mask != 0 && op.ckeId != INVALID_CCU_CKE) {
                used.insert(op.ckeId);
            }
        }
        return used;
    }

    bool CcuLoopTransMemV3::MemMerge(bool allowOverlap)
    {
        mergedSrcs = srcs;
        mergedDsts = dsts;
        return MergeMemSlices(mergedSrcs, allowOverlap) && MergeMemSlices(mergedDsts, allowOverlap);
    }

    // 将多条TransMem指令合并为一条：收集所有src/dst切片和msId，要求参与合并的指令类型一致
    std::shared_ptr<CcuLoopInstrV3>
    CcuLoopTransMemV3::InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const
    {
        auto result = std::make_shared<CcuLoopTransMemV3>();
        if (result == nullptr) {
            return nullptr;
        }
        CopyBaseTo(*result);
        result->srcs = srcs;
        result->dsts = dsts;
        result->msIds = msIds;
        for (const auto& instr : instrs) {
            auto trans = std::dynamic_pointer_cast<CcuLoopTransMemV3>(instr);
            if (trans == nullptr) {
                return nullptr;
            }
            result->srcs.insert(result->srcs.end(), trans->srcs.begin(), trans->srcs.end());
            result->dsts.insert(result->dsts.end(), trans->dsts.begin(), trans->dsts.end());
            result->msIds.insert(trans->msIds.begin(), trans->msIds.end());
        }
        return result;
    }

    std::set<uint16_t> CcuLoopTransMemV3::GetUsedMS() const { return msIds; }

    std::set<uint16_t> CcuLoopTransMemV3::GetUsedCKE() const { return CcuLoopInstrV3::GetUsedCKE(); }

    std::string CcuLoopTransMemV3::Describe() const
    {
        std::ostringstream os;
        os << "{rankId=" << rankId << ", dieId=" << dieId << ", instrId=" << instrId << ", srcs=" << srcs.size()
           << ", dsts=" << dsts.size() << ", waitOps=" << waitOps.size() << ", setOps=" << setOps.size() << "}";
        return os.str();
    }

    bool CcuLoopReduceV3::MemMerge(bool allowOverlap)
    {
        (void)allowOverlap;
        mergedSrcs = srcs;
        mergedDsts = dsts;
        for (auto& srcGroup : mergedSrcs) {
            if (!MergeMemSlices(srcGroup, true)) {
                return false;
            }
        }
        return true;
    }

    // 将多条Reduce指令合并为一条：要求所有指令的dataType和reduceOp必须一致，否则无法合并
    std::shared_ptr<CcuLoopInstrV3>
    CcuLoopReduceV3::InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const
    {
        auto result = std::make_shared<CcuLoopReduceV3>();
        if (result == nullptr) {
            return nullptr;
        }
        CopyBaseTo(*result);
        result->srcs = srcs;
        result->dsts = dsts;
        result->msIds = msIds;
        result->dataType = dataType;
        result->reduceOp = reduceOp;
        for (const auto& instr : instrs) {
            auto reduce = std::dynamic_pointer_cast<CcuLoopReduceV3>(instr);
            if (reduce == nullptr || reduce->dataType != dataType || reduce->reduceOp != reduceOp) {
                return nullptr;
            }
            result->srcs.insert(result->srcs.end(), reduce->srcs.begin(), reduce->srcs.end());
            result->dsts.insert(result->dsts.end(), reduce->dsts.begin(), reduce->dsts.end());
            result->msIds.insert(reduce->msIds.begin(), reduce->msIds.end());
        }
        return result;
    }

    std::set<uint16_t> CcuLoopReduceV3::GetUsedMS() const { return msIds; }

    std::set<uint16_t> CcuLoopReduceV3::GetUsedCKE() const { return CcuLoopInstrV3::GetUsedCKE(); }

    std::string CcuLoopReduceV3::Describe() const
    {
        std::ostringstream os;
        os << "{rankId=" << rankId << ", dieId=" << dieId << ", instrId=" << instrId << ", srcGroups=" << srcs.size()
           << ", dsts=" << dsts.size() << ", dataType=" << static_cast<uint32_t>(dataType)
           << ", reduceOp=" << static_cast<uint32_t>(reduceOp) << ", waitOps=" << waitOps.size()
           << ", setOps=" << setOps.size() << "}";
        return os.str();
    }

    std::shared_ptr<CcuLoopInstrV3>
    CcuLoopClearCkeV3::InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const
    {
        (void)instrs;
        auto result = std::make_shared<CcuLoopClearCkeV3>();
        if (result == nullptr) {
            return nullptr;
        }
        CopyBaseTo(*result);
        result->clearCKEId = clearCKEId;
        result->clearMask = clearMask;
        result->clearType = clearType;
        return result;
    }

    std::set<uint16_t> CcuLoopClearCkeV3::GetUsedCKE() const
    {
        std::set<uint16_t> used = CcuLoopInstrV3::GetUsedCKE();
        if (clearMask != 0 && clearCKEId != INVALID_CCU_CKE) {
            used.insert(clearCKEId);
        }
        return used;
    }

    std::string CcuLoopClearCkeV3::Describe() const
    {
        std::ostringstream os;
        os << "{rankId=" << rankId << ", dieId=" << dieId << ", instrId=" << instrId << ", clearCKEId=" << clearCKEId
           << ", clearMask=0x" << std::hex << clearMask << std::dec << ", waitOps=" << waitOps.size() << "}";
        return os.str();
    }

    bool CcuLoopInstrsV3::MemMerge(bool allowOverlap)
    {
        for (auto& instr : instrs) {
            if (instr == nullptr || !instr->MemMerge(allowOverlap)) {
                return false;
            }
        }
        return true;
    }

    std::set<uint16_t> CcuLoopInstrsV3::GetUsedMS() const
    {
        std::set<uint16_t> used;
        for (const auto& instr : instrs) {
            if (instr == nullptr) {
                continue;
            }
            const auto instrUsed = instr->GetUsedMS();
            used.insert(instrUsed.begin(), instrUsed.end());
        }
        return used;
    }

    std::set<uint16_t> CcuLoopInstrsV3::GetUsedCKE() const
    {
        std::set<uint16_t> used;
        for (const auto& instr : instrs) {
            if (instr == nullptr) {
                continue;
            }
            const auto instrUsed = instr->GetUsedCKE();
            used.insert(instrUsed.begin(), instrUsed.end());
        }
        return used;
    }

    // 检查各loop展开之间是否存在CKE或MS资源冲突（同一资源不能被多个展开使用）
    bool CcuLoopV3::CheckResourceConflict() const
    {
        std::set<uint16_t> allCKEs;
        std::set<uint16_t> allMSs;
        for (const auto& loopInstrs : loopExpands) {
            for (uint16_t ckeId : loopInstrs.GetUsedCKE()) {
                if (!allCKEs.insert(ckeId).second) {
                    HCCL_VM_WARN(
                        "Skip merged-loop optimization because different loop expansions reuse "
                        "the same CKE resource, loopInstrId={}, ckeId={}",
                        instrId, ckeId);
                    return false;
                }
            }
            for (uint16_t msId : loopInstrs.GetUsedMS()) {
                if (!allMSs.insert(msId).second) {
                    HCCL_VM_WARN(
                        "Skip merged-loop optimization because different loop expansions reuse "
                        "the same MS resource, loopInstrId={}, msId={}",
                        instrId, msId);
                    return false;
                }
            }
        }
        return true;
    }

    bool CcuLoopV3::LoopIterationMerge()
    {
        for (auto& loopInstrs : loopExpands) {
            if (!loopInstrs.MemMerge(true)) {
                HCCL_VM_WARN(
                    "Skip merged-loop optimization because memory slices inside one loop "
                    "iteration cannot be merged, loopInstrId={}",
                    instrId);
                return false;
            }
        }
        return true;
    }

    // 将多个loop展开的指令序列按位置逐条合并为一条序列
    // 前提：各展开的指令数量一致且CKE/MS资源无冲突；合并后再做内存切片合并且不允许重叠
    bool CcuLoopV3::LoopExpandMerge()
    {
        if (loopExpands.size() <= 1) {
            return true;
        }
        const size_t instrCnt = loopExpands.front().instrs.size();
        for (const auto& expand : loopExpands) {
            if (expand.instrs.size() != instrCnt) {
                HCCL_VM_WARN(
                    "Skip merged-loop optimization because loop expansions contain different "
                    "numbers of instructions, loopInstrId={}",
                    instrId);
                return false;
            }
        }
        if (!CheckResourceConflict()) {
            return false;
        }

        CcuLoopInstrsV3 merged;
        merged.startInstrId = loopExpands.front().startInstrId;
        merged.endInstrId = loopExpands.front().endInstrId;
        merged.loopCnt = loopExpands.front().loopCnt;
        // 按指令位置收集各展开中的并行指令，逐条调用InstrMerge合并
        for (size_t instrIdx = 0; instrIdx < instrCnt; ++instrIdx) {
            std::vector<std::shared_ptr<CcuLoopInstrV3>> parallelInstrs;
            for (size_t expandIdx = 1; expandIdx < loopExpands.size(); ++expandIdx) {
                parallelInstrs.push_back(loopExpands[expandIdx].instrs[instrIdx]);
            }
            auto mergedInstr = loopExpands.front().instrs[instrIdx]->InstrMerge(parallelInstrs);
            if (mergedInstr == nullptr) {
                HCCL_VM_WARN(
                    "Skip merged-loop optimization because one instruction position cannot be "
                    "merged across loop expansions, loopInstrId={}, instructionIndex={}",
                    instrId, instrIdx);
                return false;
            }
            merged.instrs.push_back(std::move(mergedInstr));
        }
        if (!merged.MemMerge(false)) {
            HCCL_VM_WARN(
                "Skip merged-loop optimization because merged instruction memory ranges still "
                "overlap, loopInstrId={}",
                instrId);
            return false;
        }

        loopExpands.clear();
        loopExpands.push_back(std::move(merged));
        return true;
    }

    // 将合并后的循环指令序列放置到任务图中：依次处理wait、执行传输/归约操作、处理set和清除wait
    HcclResult EmitMergedLoopInstrsV3(
        CcuGraphStateV3* curCcuTask, uint32_t queId, const CcuLoopInstrsV3& mergedInstrs, bool& isContinue)
    {
        if (curCcuTask == nullptr) {
            HCCL_VM_ERROR(
                "{} Failed to emit merged loop instructions because loop graph state is null, "
                "queueId={}",
                MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), queId);
            return HCCL_E_PTR;
        }
        const RankId rankId = curCcuTask->GetRankId();
        for (const auto& instr : mergedInstrs.instrs) {
            if (instr == nullptr) {
                HCCL_VM_ERROR(
                    "{} Failed to emit one merged loop instruction because the merged instruction "
                    "entry is null, rankId={}, queueId={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId), queId);
                return HCCL_E_PTR;
            }
            HcclResult ret = ProcessWaitOps(curCcuTask, queId, instr->waitOps, isContinue);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to process wait conditions before emitting one merged loop "
                    "instruction, rankId={}, queueId={}, mergedLoopInstr={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId), queId,
                    instr->Describe());
                return ret;
            }
            if (!isContinue) {
                return HCCL_SUCCESS;
            }

            if (auto trans = std::dynamic_pointer_cast<CcuLoopTransMemV3>(instr)) {
                CcuNodeRoleV3 role = CcuNodeRoleV3::UNKNOWN;
                RankId peerRank = INVALID_RANK_ID;
                ret = ResolveTransRole(rankId, trans->srcs, trans->dsts, role, peerRank);
                if (ret != HCCL_SUCCESS) {
                    HCCL_VM_ERROR(
                        "{} Failed to determine the transfer direction for one merged loop "
                        "instruction, rankId={}, queueId={}, mergedLoopInstr={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId),
                        queId, trans->Describe());
                    return ret;
                }
                ret = AddBatchTransMem(
                    rankId, queId, curCcuTask, trans->srcs, trans->dsts, trans->mergedSrcs, trans->mergedDsts, role,
                    peerRank);
                if (ret != HCCL_SUCCESS) {
                    HCCL_VM_ERROR(
                        "{} Failed to emit one merged loop transfer task, rankId={}, queueId={}, "
                        "mergedLoopInstr={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId),
                        queId, trans->Describe());
                    return ret;
                }
            } else if (auto reduce = std::dynamic_pointer_cast<CcuLoopReduceV3>(instr)) {
                CcuNodeRoleV3 role = CcuNodeRoleV3::UNKNOWN;
                RankId peerRank = INVALID_RANK_ID;
                ret = ResolveReduceRole(rankId, reduce->srcs, reduce->dsts, role, peerRank);
                if (ret != HCCL_SUCCESS) {
                    HCCL_VM_ERROR(
                        "{} Failed to determine the reduce direction for one merged loop "
                        "instruction, rankId={}, queueId={}, mergedLoopInstr={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId),
                        queId, reduce->Describe());
                    return ret;
                }
                ret = AddBatchReduce(
                    rankId, queId, curCcuTask, reduce->srcs, reduce->dsts, reduce->mergedSrcs, reduce->mergedDsts,
                    reduce->dataType, reduce->reduceOp, role, peerRank);
                if (ret != HCCL_SUCCESS) {
                    HCCL_VM_ERROR(
                        "{} Failed to emit one merged loop reduce task, rankId={}, queueId={}, "
                        "mergedLoopInstr={}",
                        MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId),
                        queId, reduce->Describe());
                    return ret;
                }
            } else if (auto clear = std::dynamic_pointer_cast<CcuLoopClearCkeV3>(instr)) {
                if (clear->clearMask != 0) {
                    HCCL_VM_WARN(
                        "{} Skip merged-loop optimization because ClearCKE uses a non-zero "
                        "mask, instrId={}, clearMask=0x{:x}",
                        MakeErrorCodeText(ErrorCode::GRAPH_UNSUPPORTED), clear->instrId, clear->clearMask);
                    return HCCL_E_NOT_SUPPORT;
                }
            } else {
                HCCL_VM_ERROR(
                    "{} Failed to emit one merged loop instruction because its merged instruction "
                    "type is not supported, rankId={}, queueId={}, mergedLoopInstr={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId), queId,
                    instr->Describe());
                return HCCL_E_NOT_SUPPORT;
            }

            ret = ProcessSetOps(curCcuTask, queId, instr->setOps);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to apply set masks after emitting one merged loop instruction, "
                    "rankId={}, queueId={}, mergedLoopInstr={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId), queId,
                    instr->Describe());
                return ret;
            }
            ret = ClearWaitOps(curCcuTask, queId, instr->waitOps);
            if (ret != HCCL_SUCCESS) {
                HCCL_VM_ERROR(
                    "{} Failed to clear consumed wait masks after emitting one merged loop "
                    "instruction, rankId={}, queueId={}, mergedLoopInstr={}",
                    MakeErrorCodeText(ErrorCode::GRAPH_LOOP_MERGE_ERROR).c_str(), static_cast<uint32_t>(rankId), queId,
                    instr->Describe());
                return ret;
            }
        }
        return HCCL_SUCCESS;
    }

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
