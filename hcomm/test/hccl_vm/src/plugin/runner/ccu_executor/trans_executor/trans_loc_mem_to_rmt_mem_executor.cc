/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_loc_mem_to_rmt_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransLocMemToRmtMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMEMTORMTMEM_CODE, TransLocMemToRmtMemExecutor);

void TransLocMemToRmtMemExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransLocMemToRmtMemExecutor");
    rmtGSAId_ = instr_.v1.transLocMemToRmtMem.rmtGSAId;
    rmtXnId_ = instr_.v1.transLocMemToRmtMem.rmtXnId;
    locGSAId_ = instr_.v1.transLocMemToRmtMem.locGSAId;
    locXnId_ = instr_.v1.transLocMemToRmtMem.locXnId;
    lengthXnId_ = instr_.v1.transLocMemToRmtMem.lengthXnId;
    channelId_ = instr_.v1.transLocMemToRmtMem.channelId;
    reduceDataType_ = instr_.v1.transLocMemToRmtMem.reduceDataType;
    reduceOpCode_ = instr_.v1.transLocMemToRmtMem.reduceOpCode;
    clearType_ = instr_.v1.transLocMemToRmtMem.clearType;
    lengthEn_ = instr_.v1.transLocMemToRmtMem.lengthEn;
    reduceEn_ = instr_.v1.transLocMemToRmtMem.reduceEn;
    setCKEId_ = instr_.v1.transLocMemToRmtMem.setCKEId;
    setCKEMask_ = instr_.v1.transLocMemToRmtMem.setCKEMask;
    waitCKEId_ = instr_.v1.transLocMemToRmtMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMemToRmtMem.waitCKEMask;
}

// 本端mem的数据搬运到对端mem中
void TransLocMemToRmtMemExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 1.要搬运的本端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        uint64_t offset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr += offset;
        locAddr += offset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], Get gsa "
            "addr offset = [{:04x}], cke offset = [{:04x}]",
            rankId_, dieId_, offset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data "
        "from locGSAId_[{}] locAddr[{:x}] to rmtGSAId_[{}] rmtAddr[{:x}], "
        "with lengthXnId[{}] transLength[{}], reduceEn[{}], reduceOp[{}].",
        rankId_, dieId_, locGSAId_, locAddr, rmtGSAId_, rmtAddr, lengthXnId_, transLength_, reduceEn_, reduceOpCode_);
    ccuResMgr.TransMemToMem(
        reinterpret_cast<void*>(locAddr), reinterpret_cast<void*>(rmtAddr), transLength_, reduceEn_, reduceOpCode_,
        reduceDataType_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMemToRmtMemExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "LocMemToRmtMem"); }

std::string TransLocMemToRmtMemExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Trans LocMem[%u:%u] To RmtMem[%u:%u] With "
        "LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]\n",
        waitCKEId_, waitCKEMask_, locGSAId_, locXnId_, rmtGSAId_, rmtXnId_, lengthXnId_, channelId_, setCKEId_,
        setCKEMask_, clearType_, lengthEn_, reduceDataType_, reduceOpCode_, reduceEn_);
}

CcuTrace::CcuInstrTraceDetail TransLocMemToRmtMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMemToRmtMem";
    return detail;
}
