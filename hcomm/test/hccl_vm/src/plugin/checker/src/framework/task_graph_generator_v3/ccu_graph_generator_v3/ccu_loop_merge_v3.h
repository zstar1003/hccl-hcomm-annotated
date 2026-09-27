/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_LOOP_MERGE_V3_H
#define HCCLV2_CCU_LOOP_MERGE_V3_H

#include <cstdint>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "ccu_task_common_v3.h"

namespace HcclSim {
namespace TaskGraphGeneratorV3 {

    struct CcuLoopCkeOpV3 {
        uint16_t ckeId{INVALID_CCU_CKE};
        uint16_t mask{0};
    };

    class CcuLoopInstrV3 {
    public:
        CcuLoopInstrV3() = default;
        CcuLoopInstrV3(const CcuLoopInstrV3&) = default;
        CcuLoopInstrV3& operator=(const CcuLoopInstrV3&) = delete;
        virtual ~CcuLoopInstrV3() = default;

        virtual bool MemMerge(bool allowOverlap);
        virtual std::shared_ptr<CcuLoopInstrV3>
        InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const;
        virtual std::set<uint16_t> GetUsedMS() const;
        virtual std::set<uint16_t> GetUsedCKE() const;
        virtual std::string Describe() const;

        void AddWait(uint16_t ckeId, uint16_t mask);
        void AddSet(uint16_t ckeId, uint16_t mask);

    protected:
        bool MergeMemSlices(std::vector<MemSlice>& slices, bool allowOverlap) const;
        void CopyBaseTo(CcuLoopInstrV3& dst) const;
        static std::set<uint16_t> GetUsedCKEFromOps(const std::vector<CcuLoopCkeOpV3>& ops);

    public:
        RankId rankId{INVALID_RANK_ID};
        uint32_t dieId{INVALID_DIE_ID};
        uint16_t instrId{UINT16_MAX};
        std::vector<CcuLoopCkeOpV3> waitOps{};
        std::vector<CcuLoopCkeOpV3> setOps{};
    };

    class CcuLoopTransMemV3 : public CcuLoopInstrV3 {
    public:
        CcuLoopTransMemV3() = default;
        CcuLoopTransMemV3(const CcuLoopTransMemV3&) = default;
        CcuLoopTransMemV3& operator=(const CcuLoopTransMemV3&) = delete;
        ~CcuLoopTransMemV3() override = default;

        bool MemMerge(bool allowOverlap) override;
        std::shared_ptr<CcuLoopInstrV3>
        InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const override;
        std::set<uint16_t> GetUsedMS() const override;
        std::set<uint16_t> GetUsedCKE() const override;
        std::string Describe() const override;

        std::vector<MemSlice> srcs{};
        std::vector<MemSlice> dsts{};
        std::vector<MemSlice> mergedSrcs{};
        std::vector<MemSlice> mergedDsts{};
        std::set<uint16_t> msIds{};
    };

    class CcuLoopReduceV3 : public CcuLoopInstrV3 {
    public:
        CcuLoopReduceV3() = default;
        CcuLoopReduceV3(const CcuLoopReduceV3&) = default;
        CcuLoopReduceV3& operator=(const CcuLoopReduceV3&) = delete;
        ~CcuLoopReduceV3() override = default;

        bool MemMerge(bool allowOverlap) override;
        std::shared_ptr<CcuLoopInstrV3>
        InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const override;
        std::set<uint16_t> GetUsedMS() const override;
        std::set<uint16_t> GetUsedCKE() const override;
        std::string Describe() const override;

        std::vector<std::vector<MemSlice>> srcs{};
        std::vector<MemSlice> dsts{};
        std::vector<std::vector<MemSlice>> mergedSrcs{};
        std::vector<MemSlice> mergedDsts{};
        std::set<uint16_t> msIds{};
        HcclDataType dataType{HCCL_DATA_TYPE_RESERVED};
        HcclReduceOp reduceOp{HCCL_REDUCE_RESERVED};
    };

    class CcuLoopClearCkeV3 : public CcuLoopInstrV3 {
    public:
        CcuLoopClearCkeV3() = default;
        CcuLoopClearCkeV3(const CcuLoopClearCkeV3&) = default;
        CcuLoopClearCkeV3& operator=(const CcuLoopClearCkeV3&) = delete;
        ~CcuLoopClearCkeV3() override = default;

        std::shared_ptr<CcuLoopInstrV3>
        InstrMerge(const std::vector<std::shared_ptr<CcuLoopInstrV3>>& instrs) const override;
        std::set<uint16_t> GetUsedCKE() const override;
        std::string Describe() const override;

        uint16_t clearCKEId{INVALID_CCU_CKE};
        uint16_t clearMask{0};
        uint16_t clearType{0};
    };

    struct CcuLoopInstrsV3 {
        std::vector<std::shared_ptr<CcuLoopInstrV3>> instrs{};
        uint16_t startInstrId{UINT16_MAX};
        uint16_t endInstrId{UINT16_MAX};
        uint64_t loopCnt{UINT64_MAX};

        bool MemMerge(bool allowOverlap);
        std::set<uint16_t> GetUsedMS() const;
        std::set<uint16_t> GetUsedCKE() const;
    };

    struct CcuLoopV3 {
        std::vector<CcuLoopInstrsV3> loopExpands{};
        uint64_t expandCnt{UINT64_MAX};
        uint16_t instrId{UINT16_MAX};

        bool LoopIterationMerge();
        bool LoopExpandMerge();
        bool CheckResourceConflict() const;
    };

    template <typename T>
    std::shared_ptr<T>
    EnsureCcuLoopInstr(std::shared_ptr<CcuLoopInstrV3>& instrInLoop, RankId rankId, uint32_t dieId, uint16_t instrId)
    {
        if (instrInLoop == nullptr) {
            auto newInstr = std::make_shared<T>();
            if (newInstr == nullptr) {
                return nullptr;
            }
            newInstr->rankId = rankId;
            newInstr->dieId = dieId;
            newInstr->instrId = instrId;
            instrInLoop = newInstr;
            return newInstr;
        }
        return std::dynamic_pointer_cast<T>(instrInLoop);
    }

    HcclResult EmitMergedLoopInstrsV3(
        CcuGraphStateV3* curCcuTask, uint32_t queId, const CcuLoopInstrsV3& mergedInstrs, bool& isContinue);

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim

#endif // HCCLV2_CCU_LOOP_MERGE_V3_H
