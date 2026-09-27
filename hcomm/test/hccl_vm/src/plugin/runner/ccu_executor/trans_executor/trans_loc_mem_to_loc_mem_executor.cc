/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_loc_mem_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V1(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTOLOCMEM_CODE, TransLocMemToLocMemExecutor);
REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::TRANSLOCMEMTOLOCMEM_CODE, TransLocMemToLocMemExecutor);

void TransLocMemToLocMemExecutor::Parser()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        dstGSAId_ = instr_.v1.transLocMemToLocMem.dstGSAId;
        dstXnId_ = instr_.v1.transLocMemToLocMem.dstXnId;
        srcGSAId_ = instr_.v1.transLocMemToLocMem.srcGSAId;
        srcXnId_ = instr_.v1.transLocMemToLocMem.srcXnId;
        lengthXnId_ = instr_.v1.transLocMemToLocMem.lengthXnId;
        channelId_ = instr_.v1.transLocMemToLocMem.channelId;
        clearType_ = instr_.v1.transLocMemToLocMem.clearType;
        lengthEn_ = instr_.v1.transLocMemToLocMem.lengthEn;
        setCKEId_ = instr_.v1.transLocMemToLocMem.setCKEId;
        setCKEMask_ = instr_.v1.transLocMemToLocMem.setCKEMask;
        waitCKEId_ = instr_.v1.transLocMemToLocMem.waitCKEId;
        waitCKEMask_ = instr_.v1.transLocMemToLocMem.waitCKEMask;
    } else if (version_ == RunnerCcuVersion::CCU_V2) {
        xdId_ = instr_.v2.transLocMemToLocMem.xdId;
        xdtId_ = instr_.v2.transLocMemToLocMem.xdtId;
        xsId_ = instr_.v2.transLocMemToLocMem.xsId;
        xstId_ = instr_.v2.transLocMemToLocMem.xstId;
        xlId_ = instr_.v2.transLocMemToLocMem.xlId;
        usedMSId_ = instr_.v2.transLocMemToLocMem.usedMSId;
        msNum_ = instr_.v2.transLocMemToLocMem.msNum;
        setCKEId_ = instr_.v2.transLocMemToLocMem.setCKEId;
        setCKEMask_ = instr_.v2.transLocMemToLocMem.setCKEMask;
    } else {
        HCCL_VM_ERROR("Invalid ccu version:{}", RunnerCcuVersionToString(version_));
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
}

void TransLocMemToLocMemExecutor::Process(CcuResourceManager& ccuResMgr)
{
    uint64_t srcLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, srcGSAId_);
    uint64_t dstLocAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, dstGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint64_t gsaOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        uint16_t ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        srcLocAddr += gsaOffset;
        dstLocAddr += gsaOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], Get gsa addr offset = [{:x}], cke id offset = [{}]", rankId_, dieId_, gsaOffset, ckeOffset);
    }
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data from srcLocGSAId_[{}] srcLocAddr[{:x}] to dstLocGSAId_[{}] "
        "dstLocAddr[{:x}], with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, srcGSAId_, srcLocAddr, dstGSAId_, dstLocAddr, lengthXnId_, transLength_);
    ccuResMgr.TransMemToMem(
        reinterpret_cast<void*>(srcLocAddr), reinterpret_cast<void*>(dstLocAddr), transLength_, false, 0, 0);
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToLocMemExecutor::RunV1() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToLocMem"); }

void TransLocMemToLocMemExecutor::RunV2()
{
    auto& ccuResMgr = CcuResourceManager::GetInstance();

    uint16_t xsId = GetXnId(xsId_);
    uint64_t srcLocMemAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);

    uint16_t xdId = GetXnId(xdId_);
    uint64_t dstLocMemAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);

    uint16_t xstId = GetXnId(xstId_);
    uint64_t addrExpandInfo = ccuResMgr.GetXnValue(rankId_, dieId_, xstId);
    uint16_t addrExpandCoef = (addrExpandInfo >> 53) & 0x3;
    srcLocMemAddr = UpdateAddress(srcLocMemAddr, addrExpandCoef);
    dstLocMemAddr = UpdateAddress(dstLocMemAddr, addrExpandCoef);

    uint16_t xlId = GetXnId(xlId_);
    uint64_t len = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);
    if (len == 0) {
        HCCL_VM_ERROR("The size of data transfer is 0.");
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    HCCL_VM_DEBUG(
        "Trans data from srcMemAddr[0x{:x}] to dstMemAddr[0x{:x}], length[{}].", srcLocMemAddr, dstLocMemAddr, len);

    ccuResMgr.TransMemToMem(
        reinterpret_cast<void*>(srcLocMemAddr), reinterpret_cast<void*>(dstLocMemAddr), len, false, 0, 0);

    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

void TransLocMemToLocMemExecutor::Run()
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

std::string TransLocMemToLocMemExecutor::Describe()
{
    if (version_ == RunnerCcuVersion::CCU_V1) {
        return HcclSim::StringFormat(
            "[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To LocMem[%u:%u] With "
            "LengthXn[%u] Use Channel[%u], Set "
            "CKE[%u:%04x], clearType[%u], lengthEn[%u]\n",
            waitCKEId_, waitCKEMask_, srcGSAId_, srcXnId_, dstGSAId_, dstXnId_, lengthXnId_, channelId_, setCKEId_,
            setCKEMask_, clearType_, lengthEn_);
    } else {
        return HcclSim::StringFormat(
            "[TransLocMemToLocMemExecutor] xdId[%u] xdtId[%u] xsId[%u] xstId[%u] "
            "xlId[%u] usedMSId[%u] msNum[%u] setCKEId[%u] setCKEMask[0x%04x]\n",
            xdId_, xdtId_, xsId_, xstId_, xlId_, usedMSId_, msNum_, setCKEId_, setCKEMask_);
    }
}

CcuTrace::CcuInstrTraceDetail TransLocMemToLocMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMemToLocMem";
    return detail;
}
