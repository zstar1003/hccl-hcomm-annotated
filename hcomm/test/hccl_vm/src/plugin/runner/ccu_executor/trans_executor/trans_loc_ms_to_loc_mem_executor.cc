/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_loc_ms_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTOLOCMEM_CODE, TransLocMSToLocMemExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::TRANSLOCMSTOLOCMEM_CODE, TransLocMSToLocMemExecutor);

void TransLocMSToLocMemExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        locGSAId_ = instr_.v1.transLocMSToLocMem.locGSAId;
        locXnId_ = instr_.v1.transLocMSToLocMem.locXnId;
        locMSId_ = instr_.v1.transLocMSToLocMem.locMSId & 0x7FFF;
        locDieId_ = instr_.v1.transLocMSToLocMem.locMSId >> 15;
        lengthXnId_ = instr_.v1.transLocMSToLocMem.lengthXnId;
        channelId_ = instr_.v1.transLocMSToLocMem.channelId;
        clearType_ = instr_.v1.transLocMSToLocMem.clearType;
        lengthEn_ = instr_.v1.transLocMSToLocMem.lengthEn;
        setCKEId_ = instr_.v1.transLocMSToLocMem.setCKEId;
        setCKEMask_ = instr_.v1.transLocMSToLocMem.setCKEMask;
        waitCKEId_ = instr_.v1.transLocMSToLocMem.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMSToLocMem.waitCKEMask;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        xdId_ = instr_.v2.transLocMSToLocMem.xdId;
        xdtId_ = instr_.v2.transLocMSToLocMem.xdtId;
        msId_ = instr_.v2.transLocMSToLocMem.msId & 0x7FFF;
        xlId_ = instr_.v2.transLocMSToLocMem.xlId;
        xoId_ = instr_.v2.transLocMSToLocMem.xoId;
        setCKEId_ = instr_.v2.transLocMSToLocMem.setCKEId;
        setCKEMask_ = instr_.v2.transLocMSToLocMem.setCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMSToLocMemExecutor::Process(CcuResourceManager& ccuResMgr)
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
            "locCcu[{}:{}], Get gsa addr offset = [{:04x}], ms offset = [{}], "
            "cke offset = [{:04x}]",
            rankId_, dieId_, addrOffset, msOffset, ckeOffset);
    }
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data "
        "from locMsId[{}] to locGSAId[{}] locAddr[{:x}], "
        "with lengthXnId[{}] transLength[{}], lengthEn_[{}].",
        rankId_, dieId_, locMSId_, locGSAId_, locAddr, lengthXnId_, transLength_, lengthEn_);
    bool ret = ccuResMgr.TransMSToMem(rankId_, dieId_, locMSId_, reinterpret_cast<void*>(locAddr), transLength_);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToLocMemExecutor::RunV1() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMSToLocMem"); }

void TransLocMSToLocMemExecutor::RunV2()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();

    uint16_t xdId = GetXnId(xdId_);
    uint64_t dstLocMemAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);

    uint16_t xdtId = GetXnId(xdtId_);
    uint64_t addrExpandInfo = ccuResMgr.GetXnValue(rankId_, dieId_, xdtId);
    uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    dstLocMemAddr = UpdateAddress(dstLocMemAddr, addrExpandCoef);

    uint16_t xlId = GetXnId(xlId_);
    uint64_t len = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);
    if (len == 0) {
        HCCL_VM_ERROR("The size of data transfer is 0.");
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    uint16_t locMsId = UpdateMSId(msId_);

    HCCL_VM_DEBUG("Trans data from msId[{}] to dstMemAddr[0x{:x}], length[{}].", locMsId, dstLocMemAddr, len);

    bool ret = ccuResMgr.TransMSToMem(rankId_, dieId_, locMsId, reinterpret_cast<void*>(dstLocMemAddr), len);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

void TransLocMSToLocMemExecutor::Run()
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

std::string TransLocMSToLocMemExecutor::Describe()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        return HcclSim::StringFormat(
            "ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To LocMem[%u:%u] With LengthXn[%u] Use "
            "Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
            waitCKEId_, waitCKEMask_, locMSId_ / 0x8000, locMSId_ % 0x8000, locGSAId_, locXnId_, lengthXnId_,
            channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
    } else {
        return HcclSim::StringFormat(
            "[TransLocMSToLocMemExecutor] xdId[%u] xdtId[%u] msId[%u] "
            "xlId[%u] xoId[%u] setCKEId[%u] setCKEMask[0x%04x]\n",
            xdId_, xdtId_, msId_, xlId_, xoId_, setCKEId_, setCKEMask_);
    }
}

CcuTrace::CcuInstrTraceDetail TransLocMSToLocMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMSToLocMem";
    return detail;
}
