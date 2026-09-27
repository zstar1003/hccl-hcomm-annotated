/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "trans_mem_executor.h"

#include <cstdint>

#include "ccu_executor_manager.h"
#include "ccu_microcode_common_v2.h"
#include "sim_log.h"
#include "ccu_string_util.h"

using namespace std;
using namespace hcomm::CcuRep;

REG_CCU_EXECUTOR_CREATE_FUNC_V2(SimCcuV2::TRANS_TYPE, SimCcuV2::TRANSMEM_CODE, TransMemExecutor);

void TransMemExecutor::Parser()
{
    ValidateVersionExclusive(RunnerCcuVersion::CCU_V2, "TransMemExecutor");
#ifdef BUILD_A6_CCU_INSTR
    xdId_ = instr_.v2.transMem.xdId;
    xdtId_ = instr_.v2.transMem.xdtId;
    xsId_ = instr_.v2.transMem.xsId;
    xstId_ = instr_.v2.transMem.xstId;
    xlId_ = instr_.v2.transMem.xlId;
    xcId_ = instr_.v2.transMem.xcId;
    xnId_ = instr_.v2.transMem.xnId;
    xntId_ = instr_.v2.transMem.xntId;
    value_ = instr_.v2.transMem.value;
    udfType_ = instr_.v2.transMem.udfType;
    reduceDataType_ = instr_.v2.transMem.reduceDataType;
    reduceOpCode_ = instr_.v2.transMem.reduceOpCode;
    dmaOpCode_ = instr_.v2.transMem.dmaOpCode;
    order_ = instr_.v2.transMem.order;
    fence_ = instr_.v2.transMem.fence;
    cqe_ = instr_.v2.transMem.cqe;
    nf_ = instr_.v2.transMem.nf;
    udfEnable_ = instr_.v2.transMem.udfEnable;
    splitMode_ = instr_.v2.transMem.splitMode;
    se_ = instr_.v2.transMem.se;
    rmtJettyType_ = instr_.v2.transMem.rmtJettyType;
    srcMode_ = instr_.v2.transMem.src_mode;
    dstMode_ = instr_.v2.transMem.dst_mode;
    msIdmode_ = instr_.v2.transMem.msIdMode;
    targetHint_ = instr_.v2.transMem.targetHint;
    setCKEId_ = instr_.v2.transMem.setCKEId;
    setCKEMask_ = instr_.v2.transMem.setCKEMask;
#endif
}

void TransMemExecutor::Run()
{
    if (udfType_ != 0) {
        HCCL_VM_ERROR("udfType is not supported. udfType:[{}]", udfType_);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    auto& ccuResMgr = CcuResourceManager::GetInstance();

    uint16_t xdId = GetXnId(xdId_);

    uint16_t xsId = GetXnId(xsId_);

    uint16_t xlId = GetXnId(xlId_);
    uint64_t length = ccuResMgr.GetXnValue(rankId_, dieId_, xlId);

    uint16_t xcId = GetXnId(xcId_);
    uint64_t xcValue = ccuResMgr.GetXnValue(rankId_, dieId_, xcId);

    auto rmtCcu = ccuResMgr.GetRmtCcu(rankId_, dieId_, static_cast<uint16_t>(xcValue));
    int rmtRankId = rmtCcu.first;
    int rmtDieId = rmtCcu.second;
    if (rmtRankId < 0 || rmtDieId < 0) {
        HCCL_VM_ERROR("RemoteCCU not exist. rankId:[{}], dieId:[{}], xcValue:[{}]", rankId_, dieId_, xcValue);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }
    if (dmaOpCode_ == 3 || dmaOpCode_ == 5) {
        // 写数据 dst为对端 src为本端
        uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);
        uint64_t dstAddr = (dstMode_ == 1) ? UpdateAddress(xdValue) : UpdateAddressWithoutStride(xdValue);
        if (msIdmode_ == 1) {
            // XsId寄存器中存的就是MSId
            uint16_t srcMsId = UpdateMSId(xsId_) & 0x7FFF;
            ccuResMgr.TransMSToMem(rankId_, dieId_, srcMsId, reinterpret_cast<void*>(dstAddr), length);
        } else {
            uint64_t xsValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
            uint64_t srcAddr = (srcMode_ == 1) ? UpdateAddress(xsValue) : UpdateAddressWithoutStride(xsValue);
            ccuResMgr.TransMemToMem(
                reinterpret_cast<void*>(srcAddr), reinterpret_cast<void*>(dstAddr), length, udfEnable_, reduceOpCode_,
                reduceDataType_);
        }
    } else if (dmaOpCode_ == 6) {
        uint64_t xsValue = ccuResMgr.GetXnValue(rankId_, dieId_, xsId);
        // 读数据 src为对端 dst为本端
        uint64_t srcAddr = (srcMode_ == 1) ? UpdateAddress(xsValue) : UpdateAddressWithoutStride(xsValue);
        if (msIdmode_ == 1) {
            uint16_t dstMsId = UpdateMSId(xdId_) & 0x7FFF;
            ccuResMgr.TransMemToMS(rankId_, dieId_, dstMsId, reinterpret_cast<void*>(srcAddr), length);
        } else {
            uint64_t xdValue = ccuResMgr.GetXnValue(rankId_, dieId_, xdId);
            uint64_t dstAddr = (dstMode_ == 1) ? UpdateAddress(xdValue) : UpdateAddressWithoutStride(xdValue);
            ccuResMgr.TransMemToMem(
                reinterpret_cast<void*>(srcAddr), reinterpret_cast<void*>(dstAddr), length, udfEnable_, reduceOpCode_,
                reduceDataType_);
        }
    } else {
        HCCL_VM_ERROR("dmaOpCode is not supported. dmaOpCode:[{}]", dmaOpCode_);
        ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
        return;
    }

    // 写notify
    if (dmaOpCode_ == 5) {
        uint16_t xnId = GetXnId(xnId_);
        uint64_t xnAddr = ccuResMgr.GetXnValue(rankId_, dieId_, xnId);
        uint16_t xnIdRmt = 0;
        if (!ccuResMgr.GetXnIdByAddr(dieId_, CcuComponerntType::XN_A6, xnAddr, xnIdRmt)) {
            ccuSimulator_->SetExecState(CcuExecState::EXEC_FAIL);
            return;
        }
        ccuResMgr.UpdateXnValue(rmtRankId, rmtDieId, xnIdRmt, value_);
    }

    uint16_t ckeId = UpdateCkeId(setCKEId_);
    SetCkeSignal(ccuResMgr, ckeId, setCKEMask_);
}

std::string TransMemExecutor::Describe()
{
    return HcclSim::StringFormat(
        "[TransMemExecutor] xdId[%u] xdtId[%u] xsId[%u] xstId[%u] "
        "xlId[%u] xcId[%u] xnId[%u] xntId[%u] value[0x%08x] "
        "udfType[%u] reduceDataType[%u] reduceOpCode[%u] dmaOpCode[0x%02x] "
        "order[%u] fence[%u] cqe[%u] nf[%u] udfEnable[%u] splitMode[%u] "
        "se[%u] rmtJettyType[%u] src_mode[%u] dst_mode[%u] msIdmode[%u] "
        "targetHint[%u] setCKEId[%u] setCKEMask[0x%04x]\n",
        xdId_, xdtId_, xsId_, xstId_, xlId_, xcId_, xnId_, xntId_, value_, udfType_, reduceDataType_, reduceOpCode_,
        dmaOpCode_, order_, fence_, cqe_, nf_, udfEnable_, splitMode_, se_, rmtJettyType_, srcMode_, dstMode_,
        msIdmode_, targetHint_, setCKEId_, setCKEMask_);
}

CcuTrace::CcuInstrTraceDetail TransMemExecutor::CollectTraceDetail()
{
    CcuTrace::CcuInstrTraceDetail detail;
    detail.typeName = "TransMem";
    return detail;
}
