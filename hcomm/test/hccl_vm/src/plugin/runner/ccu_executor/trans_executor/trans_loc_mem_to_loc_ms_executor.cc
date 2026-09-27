/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_loc_mem_to_loc_ms_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMS_CODE, TransLocMemToLocMSExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::TRANSLOCMEMTOLOCMS_CODE, TransLocMemToLocMSExecutor);

void TransLocMemToLocMSExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        locGSAId_ = instr_.v1.transLocMemToLocMS.locGSAId;
        locXnId_ = instr_.v1.transLocMemToLocMS.locXnId;
        locMSId_ = instr_.v1.transLocMemToLocMS.locMSId & 0x7FFF;
        locDieId_ = instr_.v1.transLocMemToLocMS.locMSId >> 15;
        lengthXnId_ = instr_.v1.transLocMemToLocMS.lengthXnId;
        channelId_ = instr_.v1.transLocMemToLocMS.channelId;
        clearType_ = instr_.v1.transLocMemToLocMS.clearType;
        lengthEn_ = instr_.v1.transLocMemToLocMS.lengthEn;
        setCKEId_ = instr_.v1.transLocMemToLocMS.setCKEId;
        setCKEMask_ = instr_.v1.transLocMemToLocMS.setCKEMask;
        waitCKEId_ = instr_.v1.transLocMemToLocMS.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMemToLocMS.waitCKEMask;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        msId_ = instr_.v2.transLocMemToLocMS.msId & 0x7FFF;
        xsId_ = instr_.v2.transLocMemToLocMS.xsId;
        xstId_ = instr_.v2.transLocMemToLocMS.xstId;
        xlId_ = instr_.v2.transLocMemToLocMS.xlId;
        xoId_ = instr_.v2.transLocMemToLocMS.xoId;
        setCKEId_ = instr_.v2.transLocMemToLocMS.setCKEId;
        setCKEMask_ = instr_.v2.transLocMemToLocMS.setCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMemToLocMSExecutor::Process(CcuResourceManager& ccuResMgr)
{
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        locAddr += addrOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], Get gsa "
            "addr offset = [{:04x}], ms offset = [{:04x}], cke offset = [{:04x}]",
            rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data "
        "from locGSAId_[{}] locAddr[{:x}] to locMsId[{:04x}], "
        "with lengthXnId[{}] transLength[{}], lengthEn_[{}].",
        rankId_, dieId_, locGSAId_, locAddr, locMSId_, lengthXnId_, transLength_, lengthEn_);
    bool ret = ccuResMgr.TransMemToMS(rankId_, dieId_, locMSId_, reinterpret_cast<void*>(locAddr), transLength_);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToLocMSExecutor::RunV1() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToLocMS"); }

void TransLocMemToLocMSExecutor::RunV2()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();

    uint16_t xsId = GetXnId(xsId_);
    uint64_t srcLocMemAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);

    uint16_t xstId = GetXnId(xstId_);
    uint64_t addrExpandInfo = ccuResMgr.GetXnValue(rankId_, dieId_, xstId);
    uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    srcLocMemAddr = UpdateAddress(srcLocMemAddr, addrExpandCoef);

    uint16_t xlId = GetXnId(xlId_);
    uint64_t len = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);
    if (len == 0) {
        HCCL_VM_ERROR("The size of data transfer is 0.");
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    uint16_t locMsId = UpdateMSId(msId_);

    HCCL_VM_DEBUG("Trans data from srcMemAddr[0x{:x}] to msId[{}], length[{}].", srcLocMemAddr, locMsId, len);

    bool ret = ccuResMgr.TransMemToMS(rankId_, dieId_, locMsId, reinterpret_cast<void*>(srcLocMemAddr), len);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

void TransLocMemToLocMSExecutor::Run()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        RunV1();
        return;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        RunV2();
        return;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

std::string TransLocMemToLocMSExecutor::Describe()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        return HcclSim::StringFormat(
            "[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMS[%u:%u] With LengthXn[%u] Use "
            "Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]\n",
            waitCKEId_, waitCKEMask_, locGSAId_, locXnId_, locMSId_ / 0x8000, locMSId_ % 0x8000, lengthXnId_,
            channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
    } else {
        return HcclSim::StringFormat(
            "[TransLocMemToLocMSExecutor] msId[%u] xsId[%u] xstId[%u] "
            "xlId[%u] xoId[%u] setCKEId[%u] setCKEMask[0x%04x]\n",
            msId_, xsId_, xstId_, xlId_, xoId_, setCKEId_, setCKEMask_);
    }
}

CcuTrace::CcuInstrTraceDetail TransLocMemToLocMSExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMemToLocMS";
    return detail;
}
