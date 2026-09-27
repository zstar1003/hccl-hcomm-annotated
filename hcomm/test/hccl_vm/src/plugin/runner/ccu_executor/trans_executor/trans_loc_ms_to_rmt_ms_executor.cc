/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_loc_ms_to_rmt_ms_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransLocMSToRmtMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSLOCMSTORMTMS_CODE, TransLocMSToRmtMSExecutor);

void TransLocMSToRmtMSExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransLocMSToRmtMSExecutor");
    rmtMSId_ = instr_.v1.transLocMSToRmtMS.rmtMSId & 0x7FFF;
    rmtDieId_ = instr_.v1.transLocMSToRmtMS.rmtMSId >> 15;
    locMSId_ = instr_.v1.transLocMSToRmtMS.locMSId & 0x7FFF;
    locDieId_ = instr_.v1.transLocMSToRmtMS.locMSId >> 15;
    lengthXnId_ = instr_.v1.transLocMSToRmtMS.lengthXnId;
    channelId_ = instr_.v1.transLocMSToRmtMS.channelId;
    setRmtCKEId_ = instr_.v1.transLocMSToRmtMS.setRmtCKEId;
    setRmtCKEMask_ = instr_.v1.transLocMSToRmtMS.setRmtCKEMask;
    clearType_ = instr_.v1.transLocMSToRmtMS.clearType;
    lengthEn_ = instr_.v1.transLocMSToRmtMS.lengthEn;
    setCKEId_ = instr_.v1.transLocMSToRmtMS.setCKEId;
    setCKEMask_ = instr_.v1.transLocMSToRmtMS.setCKEMask;
    waitCKEId_ = instr_.v1.transLocMSToRmtMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transLocMSToRmtMS.waitCKEMask;
}

// 本端的源MS的数据搬运到对端的目的MS中
void TransLocMSToRmtMSExecutor::Process(CcuResourceManager& ccuResMgr)
{
    // 1.根据channel id获取remote rank id
    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, channelId_);
    if (rmtCcu.second != rmtDieId_) {
        HCCL_VM_WARN(
            "dieId[{}] from channel is not same as rmtDieId[{}]. curCcu[{}:{}], rmtCcu[{}:{}]", rmtCcu.second,
            rmtDieId_, rankId_, dieId_, rmtCcu.first, rmtCcu.second);
        return;
    }
    // 2.判断是否在Loop循环内GSA地址需偏移
    if (ccuSimulator_->GetState() == CcuExecState::EXEC_LOOP_INSTR) {
        auto msOffset = ccuSimulator_->GetLoopMsOffset();
        auto ckeOffset = ccuSimulator_->GetLoopCKEOffset();
        rmtMSId_ += msOffset;
        locMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "curCcu[{}:{}], Get ms offset = [{:04x}], cke offset = [{:04x}]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 3.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 4.搬运动作
    HCCL_VM_DEBUG(
        "ccuId[{}:{}-{}:{}] Trans data "
        "from locSrcMsId[{}] to rmtDstMsId[{}], "
        "with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, rmtCcu.first, rmtCcu.second, locMSId_, rmtMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rankId_, dieId_, rmtCcu.first, rmtCcu.second, locMSId_, rmtMSId_, transLength_);
    // 5.设置对端的cke
    SetRmtCKESignal(ccuResMgr, rmtCcu.first, rmtCcu.second, setRmtCKEId_, setRmtCKEMask_);
    // 6.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransLocMSToRmtMSExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransLocMsToRmtMs"); }

std::string TransLocMSToRmtMSExecutor::Describe()
{
    return HcclSim::StringFormat(
        "ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans LocMS[%u:%u] To RmtMS[%u:%u] With "
        "LengthXn[%u] Use Channel[%u], Set "
        "RmtCKE[%u:%04x], Set CKE[%u:%04x], clearType[%u], lengthEn[%u]",
        waitCKEId_, waitCKEMask_, locMSId_ / 0x8000, locMSId_ % 0x8000, rmtMSId_ / 0x8000, rmtMSId_ % 0x8000,
        lengthXnId_, channelId_, setRmtCKEId_, setRmtCKEMask_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}

CcuTrace::CcuInstrTraceDetail TransLocMSToRmtMSExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransLocMSToRmtMS";
    return detail;
}
