/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_rmt_mem_to_loc_ms_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransRmtMemToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMEMTOLOCMS_CODE, TransRmtMemToLocMSExecutor);

void TransRmtMemToLocMSExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransRmtMemToLocMSExecutor");
    locMSId_ = instr_.v1.transRmtMemToLocMS.locMSId & 0x7FFFU;
    locDieId_ = instr_.v1.transRmtMemToLocMS.locMSId >> 15U;
    rmtGSAId_ = instr_.v1.transRmtMemToLocMS.rmtGSAId;
    rmtXnId_ = instr_.v1.transRmtMemToLocMS.rmtXnId;
    lengthXnId_ = instr_.v1.transRmtMemToLocMS.lengthXnId;
    channelId_ = instr_.v1.transRmtMemToLocMS.channelId;
    clearType_ = instr_.v1.transRmtMemToLocMS.clearType;
    lengthEn_ = instr_.v1.transRmtMemToLocMS.lengthEn;
    setCKEId_ = instr_.v1.transRmtMemToLocMS.setCKEId;
    setCKEMask_ = instr_.v1.transRmtMemToLocMS.setCKEMask;
    waitCKEId_ = instr_.v1.transRmtMemToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMemToLocMS.waitCKEMask;
}

// 本端MS的数据搬运到本端内存中
void TransRmtMemToLocMSExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 1.要搬运的远端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], Get gsa addr offset = [{:04x}], ms offset = [{:04x}], cke offset = [{:04x}]", rankId_,
            dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data "
        "from rmtGSAId[{}] rmtAddr[{:x}] to locMsId[{}], "
        "with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, rmtGSAId_, rmtAddr, locMSId_, lengthXnId_, transLength_);
    bool ret = ccuResMgr.TransMemToMS(rankId_, dieId_, locMSId_, reinterpret_cast<void*>(rmtAddr), transLength_);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMemToLocMSExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "RmtMemToLocMS"); }

std::string TransRmtMemToLocMSExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMS[%u:%u] With "
        "LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u]\n",
        waitCKEId_, waitCKEMask_, rmtGSAId_, rmtXnId_, locMSId_ / 0x8000, locMSId_ % 0x8000, lengthXnId_, channelId_,
        setCKEId_, setCKEMask_, clearType_, lengthEn_);
}

CcuTrace::CcuInstrTraceDetail TransRmtMemToLocMSExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransRmtMemToLocMS";
    return detail;
}
