/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_rmt_ms_to_loc_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransRmtMSToLocMemExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMSTOLOCMEM_CODE, TransRmtMSToLocMemExecutor);

void TransRmtMSToLocMemExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransRmtMSToLocMemExecutor");
    locGSAId_ = instr_.v1.transRmtMSToLocMem.locGSAId;
    locXnId_ = instr_.v1.transRmtMSToLocMem.locXnId;
    rmtMSId_ = instr_.v1.transRmtMSToLocMem.rmtMSId & 0x7FFF;
    rmtDieId_ = instr_.v1.transRmtMSToLocMem.rmtMSId >> 15;
    lengthXnId_ = instr_.v1.transRmtMSToLocMem.lengthXnId;
    channelId_ = instr_.v1.transRmtMSToLocMem.channelId;
    clearType_ = instr_.v1.transRmtMSToLocMem.clearType;
    lengthEn_ = instr_.v1.transRmtMSToLocMem.lengthEn;
    setCKEId_ = instr_.v1.transRmtMSToLocMem.setCKEId;
    setCKEMask_ = instr_.v1.transRmtMSToLocMem.setCKEMask;
    waitCKEId_ = instr_.v1.transRmtMSToLocMem.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMSToLocMem.waitCKEMask;
}

// 对端的源MS的数据搬运到本端的目的MEM中
void TransRmtMSToLocMemExecutor::Process(CcuResourceManager& ccuResMgr)
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransRmtMSToLocMemExecutor");
    // 1.根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channelId_);
    if (rmtCcu.second != rmtDieId_) {
        HCCL_VM_WARN(
            "dieId[{}] from channel is not same as rmtDieId[{}]. curCcu[{}:{}], rmtCcu[{}:{}]", rmtCcu.second,
            rmtDieId_, rankId_, dieId_, rmtCcu.first, rmtCcu.second);
        return;
    }
    // 2.要搬运的远端内存地址及数据长度
    uint64_t locAddr = ccuResMgr.GetGsaValue(rankId_, dieId_, locGSAId_);
    // 3.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto addrOffset = ccuSimulator_->GetLoopGsaAddrOffset();
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        locAddr += addrOffset;
        rmtMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "ccuId=[{}:{}], Get gsa addr offset = [{:04x}], ms offset = [{:04x}], cke offset = [{:04x}]", rankId_,
            dieId_, addrOffset, msOffset, ckeOffset);
    }
    // 4.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 5.搬运动作
    HCCL_VM_DEBUG(
        "ccuId=[{}:{}-{}:{}] Trans data "
        "from rmtMsId[{}] to locGSAId[{}] locAddr[{:x}], "
        "with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, rmtCcu.first, rmtCcu.second, rmtMSId_, locGSAId_, locAddr, lengthXnId_, transLength_);
    bool ret
        = ccuResMgr.TransMSToMem(rmtCcu.first, rmtCcu.second, rmtMSId_, reinterpret_cast<void*>(locAddr), transLength_);
    if (!ret) {
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    // 6.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMSToLocMemExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMsToLocMem"); }

std::string TransRmtMSToLocMemExecutor::Describe()
{
    return HcclSim::StringFormat(
        "ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMem[%u:%u] "
        "With LengthXn[%u] Use Channel[%u], Set "
        "CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_, waitCKEMask_, rmtMSId_ / 0x8000, rmtMSId_ % 0x8000, locGSAId_, locXnId_, lengthXnId_, channelId_,
        setCKEId_, setCKEMask_, clearType_, lengthEn_);
}

CcuTrace::CcuInstrTraceDetail TransRmtMSToLocMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransRmtMSToLocMem";
    return detail;
}
