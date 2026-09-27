/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_rmt_ms_to_loc_ms_executor.h"

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v1.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

// 注册TransRmtMSToLocMSExecutor create Func
REG_CCU_EXECUTOR_CREATE_FUNC(SimCcuV1::TRANS_TYPE, SimCcuV1::TRANSRMTMSTOLOCMS_CODE, TransRmtMSToLocMSExecutor);

void TransRmtMSToLocMSExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V1, "TransRmtMSToLocMSExecutor");
    locMSId_ = instr_.v1.transRmtMSToLocMS.locMSId & 0x7FFF;
    locDieId_ = instr_.v1.transRmtMSToLocMS.locMSId >> 15;
    rmtMSId_ = instr_.v1.transRmtMSToLocMS.rmtMSId & 0x7FFF;
    rmtDieId_ = instr_.v1.transRmtMSToLocMS.rmtMSId >> 15;
    lengthXnId_ = instr_.v1.transRmtMSToLocMS.lengthXnId;
    channelId_ = instr_.v1.transRmtMSToLocMS.channelId;
    clearType_ = instr_.v1.transRmtMSToLocMS.clearType;
    lengthEn_ = instr_.v1.transRmtMSToLocMS.lengthEn;
    setCKEId_ = instr_.v1.transRmtMSToLocMS.setCKEId;
    setCKEMask_ = instr_.v1.transRmtMSToLocMS.setCKEMask;
    waitCKEId_ = instr_.v1.transRmtMSToLocMS.waitCKEId;
    waitCKEMask_ = instr_.v1.transRmtMSToLocMS.waitCKEMask;
}

// 对端的源MS的数据搬运到本端的目的MS中
void TransRmtMSToLocMSExecutor::Process(CcuResourceManager& ccuResMgr)
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
        locMSId_ += msOffset;
        rmtMSId_ += msOffset;
        setCKEId_ += ckeOffset;
        HCCL_VM_DEBUG(
            "ccuId=[{}:{}], Get ms offset = [{:04x}], cke offset = [{:04x}]", rankId_, dieId_, msOffset, ckeOffset);
    }
    // 3.要搬运的本端内存地址及数据长度
    transLength_ = (lengthEn_ == 0) ? HcclSim::BYTE_NUM_4K : ccuResMgr.GetXnValue(rankId_, dieId_, lengthXnId_);
    // 4.搬运动作
    HCCL_VM_DEBUG(
        "ccuId=[{}:{}-{}:{}] Trans data "
        "from rmtMsId[{}] to locMsId[{}], "
        "with lengthXnId[{}] transLength[{}].",
        rankId_, dieId_, rmtCcu.first, rmtCcu.second, rmtMSId_, locMSId_, lengthXnId_, transLength_);
    ccuResMgr.TransMSToMS(rmtCcu.first, rmtCcu.second, rankId_, dieId_, rmtMSId_, locMSId_, transLength_);
    // 5.设置本端的cke
    SetCkeSignal(ccuResMgr, setCKEId_, setCKEMask_);
}

void TransRmtMSToLocMSExecutor::Run() { WaitCkeProcess(waitCKEId_, waitCKEMask_, clearType_, "TransRmtMsToLocMs"); }

std::string TransRmtMSToLocMSExecutor::Describe()
{
    return HcclSim::StringFormat(
        "ParseTransLocMSToLocMemInstr Wait CKE[%u:%04x], Trans RmtMS[%u:%u] To LocMS[%u:%u] With "
        "LengthXn[%u] Use Channel[%u], "
        "Set CKE[%u:%04x], "
        "clearType[%u], lengthEn[%u]",
        waitCKEId_, waitCKEMask_, rmtMSId_ / 0x8000, rmtMSId_ % 0x8000, locMSId_ / 0x8000, locMSId_ % 0x8000,
        lengthXnId_, channelId_, setCKEId_, setCKEMask_, clearType_, lengthEn_);
}

CcuTrace::CcuInstrTraceDetail TransRmtMSToLocMSExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransRmtMSToLocMS";
    return detail;
}
