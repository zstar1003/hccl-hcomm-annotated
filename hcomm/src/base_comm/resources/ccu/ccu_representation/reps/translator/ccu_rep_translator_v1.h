/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_REP_TRANSLATOR_H
#define HCOMM_CCU_REP_TRANSLATOR_H

#include <memory>
#include <vector>
#include <functional>

#include "ccu_instr_info_v1.h"
#include "ccu_rep_block_v1.h"
#include "ccu_rep_reference_manager_v1.h"
#include "ccu_kernel_resource.h"
#include "ccu_rep_base_v1.h"
#include "ccu_kernel.h"
#include "ccu_dev_mgr_imp.h"

namespace hcomm {
namespace CcuRep {

    class CcuRepTranslator {
    public:
        CcuRepTranslator(
            int32_t userDevId, uint8_t dieId, std::shared_ptr<CcuRepReferenceManager> refManager,
            std::array<uint16_t, CCU_MAX_IODIE_NUM>& reserverChannalId, std::pair<uint64_t, uint64_t>& ccuTokenInfo,
            uint64_t hbmTokenInfo);

        CcuRepTranslator(std::shared_ptr<CcuRepReferenceManager> refManager, const TransDep& transDep);
        static uint32_t GetInstrNum(const int32_t userDevId);
        static CcuResReq GetResReq(const int32_t userDevId, uint8_t dieId);
        void GetRes(CcuRepResource& res);
        CcuInstrInfo Translate(
            CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, uint16_t startInstrId,
            bool isFuncBlock = false);
        void Translate(
            CcuKernel* ccuKernel, const std::vector<std::shared_ptr<CcuRepBase>>& repVec, CcuInstr*& instr,
            uint16_t& instrId, std::function<bool(std::shared_ptr<CcuRepBase>)> filter);
        void DumpInstruction(const CcuInstrInfo& instrInfo) const;
        void SetTransDep(TransDep transDepIn) { transDep = transDepIn; }
        TransDep& GetTransDep() { return transDep; }

    private:
        template <typename T1, typename T2>
        void BuildReference(const std::shared_ptr<CcuRepBase>& rep);
        void PreProcess(std::shared_ptr<CcuRepBase> rep);
        void CommonProcess(CcuKernel* ccuKernel, CcuInstr*& instr, uint16_t& instrId);
        void FinishMainBlock(CcuInstr*& instr, uint16_t& instrId);
        void DumpRep(const std::vector<std::shared_ptr<CcuRepBase>>& repVec, const CcuInstrInfo& instrInfo) const;
        void BindResource(bool isFuncBlock);

    private:
        static const int XN_NUM = 4;                     // 4: Xn资源个数
        static const int GSA_NUM = 3;                    // 3: GSA资源个数
        static const int CKE_NUM = 2;                    // 2: CKE资源个数
        static const int V2_RELJMP_INSTR_NUM = 9;        // V2: RelJmp生成的指令数
        static const int V2_FINISH_BLOCK_INSTR_NUM = 10; // V2: FinishMainBlock占用的指令数(RelJmp+Jump或10xNop)
        std::shared_ptr<CcuRepReferenceManager> refManager{nullptr};
        Variable var[XN_NUM];
        Address addr[GSA_NUM];
        CompletedEvent signal[CKE_NUM]; // 声明为CompletedEvent，使用离散cke
        TransDep transDep{};
        CcuVersion ccuVersion{CcuVersion::CCU_INVALID};
    };
}; // namespace CcuRep
}; // namespace hcomm

#endif // HCCL_CCU_REP_TRANSLATOR_H
