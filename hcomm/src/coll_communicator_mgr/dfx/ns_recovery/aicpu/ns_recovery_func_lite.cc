/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "ns_recovery_func_lite.h"
#include "kfc.h"
#include "sal_pub.h"
#include "ns_recovery_lite.h"
#include "coll_comm_aicpu_mgr.h"

namespace hccl {
NsRecoveryFuncLite& NsRecoveryFuncLite::GetInstance()
{
    static NsRecoveryFuncLite func;
    return func;
}

void NsRecoveryFuncLite::Call()
{
    std::shared_lock<std::shared_mutex> rwlock(CollCommAicpuMgr::GetInstance().GetMutex());

    std::vector<std::pair<std::string, CollCommAicpu*>> aicpuCommInfo;
    auto ret = CollCommAicpuMgr::GetInstance().GetAllComms(aicpuCommInfo);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("[NsRecovery][BackGround] AicpuGetCommAll failed, errNo[0x%016llx]", ret);
        return;
    }
    for (auto& commInfo : aicpuCommInfo) {
        CollCommAicpu* deviceComm = commInfo.second;
        if (deviceComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_INVALID) {
            continue;
        }
        HandleStopLaunch(deviceComm);
        HandleClean(deviceComm);
    }
}

void NsRecoveryFuncLite::HandleStopLaunch(CollCommAicpu* deviceComm) const
{
    if (deviceComm->GetCommmStatus() == HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING) {
        return;
    }

    Hccl::KfcCommand cmd = deviceComm->GetNsRecoveryLitePtr()->BackGroundGetCmd();
    if (cmd != Hccl::KfcCommand::NS_STOP_LAUNCH) {
        return;
    }
    HCCL_INFO("[NsRecovery][BackGround] received KfcCommand[NS_STOP_LAUNCH]");
    deviceComm->GetNsRecoveryLitePtr()->SetNeedClean(true);
    deviceComm->SetCommmStatus(HcclCommStatus::HCCL_COMM_STATUS_SUSPENDING);
    deviceComm->GetNsRecoveryLitePtr()->BackGroundSetStatus(Hccl::KfcStatus::STOP_LAUNCH_DONE);
    HCCL_INFO(
        "[NsRecovery][BackGround] commId[%s] send KfcStatus[STOP_LAUNCH_DONE]", deviceComm->GetIdentifier().c_str());
}

void NsRecoveryFuncLite::HandleClean(CollCommAicpu* deviceComm)
{
    if (!deviceComm->GetNsRecoveryLitePtr()->IsNeedClean()) {
        return;
    }
    Hccl::KfcCommand cmd = deviceComm->GetNsRecoveryLitePtr()->BackGroundGetCmd();
    if (cmd != Hccl::KfcCommand::NS_CLEAN) {
        return;
    }
    HCCL_INFO("[NsRecovery][BackGround] received KfcCommand[NS_CLEAN]");
    StreamClean(deviceComm);
    deviceComm->Clean();
    deviceComm->GetNsRecoveryLitePtr()->SetNeedClean(false);
    deviceComm->GetNsRecoveryLitePtr()->BackGroundSetStatus(Hccl::KfcStatus::CLEAN_DONE);
    HCCL_INFO("[NsRecovery][BackGround] commId[%s] send KfcStatus[CLEAN_DONE]", deviceComm->GetIdentifier().c_str());
}

void NsRecoveryFuncLite::StreamClean(CollCommAicpu* deviceComm)
{
    // 查询停流是否完成
    u32 localDevId{0};
    auto ret = drvGetLocalDevIDByHostDevID(deviceComm->GetTopoInfo().devicePhyId, &localDevId);
    if (ret != DRV_ERROR_NONE) {
        HCCL_ERROR(
            "NsRecoveryFuncLite::%s call drvGetLocalDevIDByHostDevID failed, devPhyId %u, ret %d", __func__,
            deviceComm->GetTopoInfo().devicePhyId, ret);
        return;
    }
    if (DeviceQuery(localDevId, APP_ABORT_STAUTS::APP_ABORT_KILL_FINISH, 0U) != HCCL_SUCCESS) {
        deviceComm->GetNsRecoveryLitePtr()->BackGroundSetStatus(Hccl::KfcStatus::ERROR, Hccl::KfcErrType::EXEC);
        HCCL_ERROR("[NsRecovery][BackGround] Stream Stop failed");
        return;
    }
    HCCL_INFO("[NsRecovery][BackGround] commId[%s] streamClean success.", deviceComm->GetIdentifier().c_str());
}

constexpr u64 NSEC_PER_SEC = 1000000000U;
inline u64 GetCurCpuTimestamp()
{
    struct timespec timestamp {
        0, 0
    };
    (void)clock_gettime(CLOCK_MONOTONIC_RAW, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * NSEC_PER_SEC) + (timestamp.tv_nsec));
}

constexpr u32 FIVE_MILLISECOND_OF_USLEEP = 5000U;

HcclResult NsRecoveryFuncLite::DeviceQuery(const uint32_t devId, const uint32_t step, const uint64_t timeout)
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

} // namespace hccl
