/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_INS_PREPROCESSOR_H
#define CCU_INS_PREPROCESSOR_H

#include "ccu_ins.h"
#include "ins_queue.h"
#include "ccu_ctx_mgr.h"
#include "ccu_respack_mgr.h"
#include "ccu_communicator.h"

namespace Hccl {

class CcuInsPreprocessor {
public:
    using InsIterator = HierarchicalQueue<Instruction, InsQueue>::UnConstIterator;

    explicit CcuInsPreprocessor(CommunicatorImpl* comm) : ccuComm(comm) {}

    ~CcuInsPreprocessor();

    void Preprocess(std::shared_ptr<InsQueue>& insQueue, bool isMc2 = false);

    CcuCommunicator* GetCcuComm();
    HcclResult
    RecoverCcuTransportCtx(const std::vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair);
    HcclResult RecoverCcuTransportConfirm();

    void PrepareCcuCtx(std::shared_ptr<InsQueue>& insQueue, bool isMc2);
    void RegisterCtx(bool isFuncBlock);
    bool IsRollback() const;

private:
    CcuCommunicator ccuComm;
    vector<u32> resPackIdxs;               //  用于注册需要ins对应的ResPackMgr中resPack的索引
    vector<CcuCtxSignature> ctxSignatures; //  用于注册需要ins对应的CcuCtxSignature
    vector<InsIterator> insPtrs;           //  用于注册时需要setExecId提供ins迭代器
    unordered_map<CcuCtxSignature, unordered_map<u32, std::unique_ptr<CcuCtxGroup>>>
        ccuCtxGroups; //  用于注册需要ins对应的ccuCtxGroup

    bool needHandShake{false};  // true: 有没注册的ctx则需要握手
    bool resAllocSuccess{true}; // true: 本地资源申请成功(包括connection\transport\transportGrp\ctx)
    bool isRollback{false};     // true: 资源申请失败，尝试回退

    void InsPreprocess(InsIterator& insIter, u32 resPackIndex, bool isMc2);
    void
    CreateCcuCtxGroup(const CcuInstruction& ccuIns, std::unique_ptr<CcuCtxGroup>& ccuCtxGroupPtr, bool& createStatus);
    std::unique_ptr<CcuContext> CreateCcuCtx(const CcuInstruction& ccuInst, bool& createStatus);
    bool CheckCtxTransportStatus(bool resAllocSuccess);
    void Fallback();
    void Confirm();
    void ClearTmpResRecords();
};

} // namespace Hccl

#endif // CCU_INS_PREPROCESSOR_H
