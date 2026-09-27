/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REP_TRANSLATOR_H
#define HCCL_CCU_REP_TRANSLATOR_H

#include <memory>
#include <vector>
#include <functional>

#include "ccu_instr_info.h"
#include "ccu_rep_block.h"
#include "ccu_rep_reference_manager.h"
#include "ccu_context_resource.h"
#include "ccu_rep_base.h"

namespace Hccl {
namespace CcuRep {

    class CcuRepTranslator {
    public:
        CcuRepTranslator(
            int32_t deviceLogicId, uint8_t dieId, std::shared_ptr<CcuRepReferenceManager> refManager,
            std::array<uint16_t, MAX_CCU_IODIE_NUM>& reserverChannalId, std::pair<uint64_t, uint64_t>& ccuTokenInfo,
            uint64_t hbmTokenInfo);
        CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep& transDep);
        static uint32_t GetInstrNum();
        static CcuResReq GetResReq(uint8_t dieId);
        void GetRes(CcuRepResource& res);
        CcuInstrInfo Translate(
            const std::vector<std::shared_ptr<CcuRepBase>>& repVec, uint16_t startInstrId, bool isFuncBlock = false);
        void Translate(
            const std::vector<std::shared_ptr<CcuRepBase>>& repVec, CcuInstr*& instr, uint16_t& instrId,
            std::function<bool(std::shared_ptr<CcuRepBase>)> filter);
        void DumpInstruction(const CcuInstrInfo& instrInfo) const;
        void SetTransDep(TransDep transDepIn) { transDep = transDepIn; }

    private:
        template <typename T1, typename T2>
        void BuildReference(const std::shared_ptr<CcuRepBase>& rep);
        void PreProcess(std::shared_ptr<CcuRepBase> rep);
        void CommonProcess(CcuInstr*& instr, uint16_t& instrId);
        void FinishMainBlock(CcuInstr*& instr, uint16_t& instrId);
        void DumpRep(const std::vector<std::shared_ptr<CcuRepBase>>& repVec, const CcuInstrInfo& instrInfo) const;
        void BindResource(bool isFuncBlock);

    private:
        static const int XN_NUM = 4;  // 4: Xn资源个数
        static const int GSA_NUM = 3; // 3: Xn资源个数
        static const int CKE_NUM = 2; // 2: Xn资源个数
        std::shared_ptr<CcuRepReferenceManager> refManager{nullptr};
        Variable var[XN_NUM];
        Address addr[GSA_NUM];
        MaskSignal signal[CKE_NUM];
        TransDep transDep{};
    };
}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_REP_TRANSLATOR_H
