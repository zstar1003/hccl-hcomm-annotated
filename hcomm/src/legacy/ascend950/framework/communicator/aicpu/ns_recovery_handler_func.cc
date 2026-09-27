/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ns_recovery_handler_func.h"
#include "kfc.h"
#include "drv_api_exception.h"
#include "exception_util.h"
#include "internal_exception.h"

namespace Hccl {
NsRecoveryHandlerFunc& NsRecoveryHandlerFunc::GetInstance()
{
    static NsRecoveryHandlerFunc func;
    return func;
}

void NsRecoveryHandlerFunc::Call()
{
    std::vector<CommunicatorImplLite*> commLites = CommunicatorImplLiteMgr::GetInstance().GetAll();
    for (auto& comm : commLites) {
        if (!comm->IsCommReady()) {
            continue;
        }
        HandleStopLaunch(comm);
        HandleClean(comm);
    }
}

void NsRecoveryHandlerFunc::HandleStopLaunch(CommunicatorImplLite* comm) const
{
    if (comm->IsSuspended()) {
        return;
    }

    KfcCommand cmd = comm->BackGroundGetCmd();
    if (cmd != KfcCommand::NS_STOP_LAUNCH) {
        return;
    }
    HCCL_INFO("[NsRecovery][BackGround] received KfcCommand[NS_STOP_LAUNCH]");
    comm->SetNeedClean(true);
    comm->SetIsSuspended(true);
    comm->BackGroundSetStatus(KfcStatus::STOP_LAUNCH_DONE);
    HCCL_INFO("[NsRecovery][BackGround] send KfcStatus[STOP_LAUNCH_DONE]");
}

void NsRecoveryHandlerFunc::HandleClean(CommunicatorImplLite* comm)
{
    if (!comm->IsNeedClean()) {
        return;
    }
    KfcCommand cmd = comm->BackGroundGetCmd();
    if (cmd != KfcCommand::NS_CLEAN) {
        return;
    }
    HCCL_INFO("[NsRecovery][BackGround] received KfcCommand[NS_CLEAN]");
    comm->GetTransportLiteMgr()->Reset();
    StreamClean(comm);
    comm->SetNeedClean(false);
    comm->BackGroundSetStatus(KfcStatus::CLEAN_DONE);
    comm->ResetErrorReported();
    HCCL_INFO("[NsRecovery][BackGround] send KfcStatus[CLEAN_DONE]");
}

void NsRecoveryHandlerFunc::StreamClean(CommunicatorImplLite* comm)
{
    // 查询停流是否完成
    u32 localDevId = 0;
    auto ret = drvGetLocalDevIDByHostDevID(comm->GetDevPhyId(), &localDevId);
    if (ret != DRV_ERROR_NONE) {
        std::string formatStr = StringFormat(
            "NsRecoveryHandlerFunc::%s call drvGetLocalDevIDByHostDevID failed, devPhyId %u, ret %d", __func__,
            comm->GetDevPhyId(), ret);
        THROW<DrvApiException>(formatStr);
    }
    if (DeviceQuery(localDevId, APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH, 0U) != HCCL_SUCCESS) {
        comm->BackGroundSetStatus(KfcStatus::ERROR, KfcErrType::EXEC);
        THROW<InternalException>("[NsRecovery][BackGround] Stream Stop failed");
    }
    // 清理资源
    auto streamLiteMgr = comm->GetStreamLiteMgr();
    CHECK_NULLPTR(streamLiteMgr->GetMaster(), "[StreamClean]master stream is nullptr!");
    streamLiteMgr->GetMaster()->GetRtsq()->Reset(false);
    for (u32 i = 0; i < streamLiteMgr->SizeOfSlaves(); ++i) {
        streamLiteMgr->GetSlave(i)->GetRtsq()->Reset(false);
    }
    HCCL_INFO("[NsRecovery][BackGround] StreamClean success.");
}

constexpr u64 NSEC_PER_SEC = 1000000000U;

inline u64 GetCurCpuTimestamp()
{
    struct timespec timestamp;
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
}

constexpr u32 FIVE_MILLISECOND_OF_USLEEP = 5000U;

HcclResult NsRecoveryHandlerFunc::DeviceQuery(const uint32_t devId, const uint32_t step, const uint64_t timeout)
{
    uint32_t status;
    uint64_t endTime;
    const uint64_t startTime = GetCurCpuTimestamp();
    bool flag = true;
    while (flag) {
        ts_ctrl_msg_body_t queryIn = {};
        ts_ctrl_msg_body_t queryAck = {};
        size_t ackCount = sizeof(ts_ctrl_msg_body_t);
        queryIn.type = OPERATION_TYPE::OP_QUERY_ABORT_STATUS;
        queryIn.u.query_task_info.choice = APP_ABORT_STS_QUERY_CHOICE::APP_ABORT_STS_QUERY_BY_PID;
        struct tsdrv_ctrl_msg para;
        para.tsid = 0;
        para.msg_len = sizeof(ts_ctrl_msg_body_t);
        para.msg = static_cast<void*>(&queryIn);
        const drvError_t ret = halTsdrvCtl(
            devId, TSDRV_CTL_CMD_CTRL_MSG, static_cast<void*>(&para), sizeof(tsdrv_ctrl_msg),
            static_cast<void*>(&queryAck), &ackCount);
        if ((ret != DRV_ERROR_NONE) || (ackCount != sizeof(ts_ctrl_msg_body_t))) {
            HCCL_ERROR("halTsdrvCtl failed. ret = %d", ret);
            return HcclResult::HCCL_E_DRV;
        }

        status = queryAck.u.query_task_ack_info.status;
        if (status >= step) {
            flag = false;
            break;
        }
        endTime = GetCurCpuTimestamp();
        if ((timeout != 0U) && ((endTime - startTime) > timeout)) {
            HCCL_ERROR("[DeviceQuery]kill query timeout.");
            return HcclResult::HCCL_E_TIMEOUT;
        }
        SaluSleep(FIVE_MILLISECOND_OF_USLEEP);
    }
    return HcclResult::HCCL_SUCCESS;
}

} // namespace Hccl
