/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_rmt_mem_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransRmtMemToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMEMTOLOCMEM_CODE, TransRmtMemToLocMemExecutor);

void TransRmtMemToLocMemExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransRmtMemToLocMemExecutor");
    locGSAId_ = instr_.v1.transRmtMemToLocMem.locGSAId;
    locXnId_ = instr_.v1.transRmtMemToLocMem.locXnId;
    rmtGSAId_ = instr_.v1.transRmtMemToLocMem.rmtGSAId;
    rmtXnId_ = instr_.v1.transRmtMemToLocMem.rmtXnId;
    lengthXnId_ = instr_.v1.transRmtMemToLocMem.lengthXnId;
    channelId_ = instr_.v1.transRmtMemToLocMem.channelId;
    reduceDataType_ = instr_.v1.transRmtMemToLocMem.reduceDataType;
    reduceOpCode_ = instr_.v1.transRmtMemToLocMem.reduceOpCode;
    clearType_ = instr_.v1.transRmtMemToLocMem.clearType;
    lengthEn_ = instr_.v1.transRmtMemToLocMem.lengthEn;
    reduceEn_ = instr_.v1.transRmtMemToLocMem.reduceEn;
    setCKEId_ = instr_.v1.transRmtMemToLocMem.setCKEId;
    setCKEMask_ = instr_.v1.transRmtMemToLocMem.setCKEMask;
    waitCKEId_ = instr_.v1.transRmtMemToLocMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMemToLocMem.waitCKEMask;
}

// 对端Mem的数据搬运到本端的Mem中
void TransRmtMemToLocMemExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 1.要搬运的远端内存地址及数据长度
    uint64_t rmtAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, rmtGSAId_);
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto offset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtAddr += offset;
        locAddr += offset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "locCcu[{}:{}], Get gsa addr offset = [{:04x}], cke offset = [{:04x}]", rankId_, dieId_, offset, ckeOffset);
    }
    // 3.搬运动作
    HCCL_VM_DEBUG(
        "locCcu[{}:{}] Trans data "
        "from rmtGSAId[{}] rmtAddr[{:x}] to locGSAId[{}] locAddr[{:x}], "
        "with lengthXnId[{}] transLength[{}], reduceEn[{}], reduceOp[{}].",
        rankId_, dieId_, rmtGSAId_, rmtAddr, locGSAId_, locAddr, lengthXnId_, transLength_, reduceEn_, reduceOpCode_);
    ccuResMgr.TransMemToMem(
        reinterpret_cast<void*>(rmtAddr), reinterpret_cast<void*>(locAddr), transLength_, reduceEn_, reduceOpCode_,
        reduceDataType_);
    // 4.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMemToLocMemExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMemToLocMem"); }

std::string TransRmtMemToLocMemExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[Simulation Execute] Wait CKE[%u:%04x], Trans RmtMem[%u:%u] To LocMem[%u:%u] With "
        "LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u], DataType[%u], ReduceType[%u] reduceEn[%u]\n",
        waitCKEId_, waitCKEMask_, rmtGSAId_, rmtXnId_, locGSAId_, locXnId_, lengthXnId_, channelId_, setCKEId_,
        setCKEMask_, clearType_, lengthEn_, reduceDataType_, reduceOpCode_, reduceEn_);
}

CcuTrace::CcuInstrTraceDetail TransRmtMemToLocMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransRmtMemToLocMem";
    return detail;
}
