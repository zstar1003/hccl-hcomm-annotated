/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REPRESENTATION_LOOPGROUP_BUNDLE_H
#define HCOMM_CCU_REPRESENTATION_LOOPGROUP_BUNDLE_H

#include <vector>
#include "ccu_types.h"
#include "ccu_datatype_v1.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_loopblock_v1.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepLoopGroupBundle : public CcuRepBase {
    public:
        enum class Layout {
            Config,    // 结构体 config 构造
            PackedVar, // 旧打包变量构造（兼容路径）
            VersionV2, // 960 三变量直传
        };

        struct LoopEntry {
            CcuLoopCfg config;
            Executor executor;
            std::shared_ptr<CcuRepLoopBlock> repLoopBlock;
            Variable loopParamVar;
            Variable iterNumVar;
            Variable addrOffsetVar;
            Variable ctxIdVar;
            Layout layout{Layout::Config};
        };

        CcuRepLoopGroupBundle(
            CcuInsGeneratorBase* insGenPtr, const CcuLoopGroupCfg& config, const Variable& parallelVar,
            const Variable& offsetVar);
        explicit CcuRepLoopGroupBundle(
            CcuInsGeneratorBase* insGenPtr, const Variable& parallelVar, const Variable& offsetVar);

        void AddLoop(const LoopEntry& entry);
        void SetRepeatLoopIdx(uint64_t idx) { repeatLoopIdx_ = idx; }
        void SetTotalLoopNum(uint64_t num) { totalLoopNum_ = num; }
        void SetLayout(Layout layout) { layout_ = layout; }
        void SetXnOffsetVar(const Variable& xnOffsetVar) { xnOffsetVar_ = Variable(xnOffsetVar); }
        void SetCompatRemapVars(const Variable& newParallelVar, const Variable& scratchVar)
        {
            // Variable 的 const& operator= 是 DSL 赋值，须用临时量走移动赋值做纯拷贝
            newParallelVar_ = Variable(newParallelVar);
            scratchVar_ = Variable(scratchVar);
        }

        bool Translate(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId, const TransDep& dep) override;
        uint16_t InstrCount() override;
        std::string Describe() override;

        uint16_t GetStartLoopInstrId() const;
        const Variable& GetOffsetParam() const { return offsetVar_; }

        const std::vector<LoopEntry>& GetLoops() const { return loops_; }
        const CcuLoopGroupCfg& GetConfig() const { return config_; }
        const Variable& GetParallelVar() const { return parallelVar_; }
        uint64_t GetRepeatLoopIdx() const { return repeatLoopIdx_; }
        uint64_t GetTotalLoopNum() const { return totalLoopNum_; }
        Layout GetLayout() const { return layout_; }
        const Variable& GetNewParallelVar() const { return newParallelVar_; }
        const Variable& GetScratchVar() const { return scratchVar_; }
        const Variable& GetXnOffsetVar() const { return xnOffsetVar_; }

    private:
        uint16_t LoopGroupInstrOffsetInBundle() const;

        CcuInsGeneratorBase* insGenPtr_{nullptr};
        CcuLoopGroupCfg config_;
        Variable parallelVar_;
        Variable offsetVar_;
        uint64_t repeatLoopIdx_{0};
        uint64_t totalLoopNum_{0};
        std::vector<LoopEntry> loops_;
        Layout layout_{Layout::Config};
        Variable newParallelVar_;
        Variable scratchVar_;
        Variable xnOffsetVar_;
    };

}; // namespace CcuRep
}; // namespace hcomm
#endif // HCOMM_CCU_REPRESENTATION_LOOPGROUP_BUNDLE_H
