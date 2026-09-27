/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_MGR_H
#define COLL_COMM_MGR_H

#include <unordered_set>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#include <array>
#include "coll_comm.h"
#include "cluster_monitor.h"
#include "ns_recovery/task_abort_handler.h"
#include "legacy_op_hcom_info.h"
#include "order_launch_thread_mgr.h"

namespace hccl {
/**
 * @note 职责：实现多个集合通信通信域上下文的创建、销毁管理，及多通信域资源、信息的共享等。
 */
class CollCommMgr {
public:
    static CollCommMgr& GetInstance();
    void RegisterCollComm(CollComm* collComm);
    void UnregisterCollComm(CollComm* collComm);
    const std::unordered_map<std::string, CollComm*>& GetAllCollComms() const;
    hcomm::ClusterMonitor& GetClusterMonitor(s32 deviceLogicId);
    HcclResult TryReserveCcuMsComm(s32 deviceLogicId, const std::string& commId, bool& reserved);
    void ReleaseCcuMsComm(s32 deviceLogicId, const std::string& commId);
    OrderLaunchThreadMgr& GetOrderLaunchThreadMgr(s32 deviceLogicId);
    HcclTaskAbortHandler& GetTaskAbortHandler() { return taskAbortHandler_; }
    void InitBaseCommRes(uint32_t devId) const;
    ~CollCommMgr();

    // 以下接口以 Legacy 前缀标记，表示用于兼容历史老接口，仅做 bug 修复与兼容维护，不再承接新特性、不再继续演进
    HcclOpInfoCtx& LegacyGetOpHcomInfo(uint32_t devId);
    HcclOpInfoCtx& LegacyGetHcclExistDeviceOpInfoCtx(s32& devId);
    HcclOpInfoCtx& LegacyGetHcclOpInfoCtx(s32& devId);

private:
    std::unordered_map<std::string, CollComm*> allCollComms_;
    std::array<hcomm::ClusterMonitor, MAX_MODULE_DEVICE_NUM> clusterMonitor_;
    std::array<std::string, MAX_MODULE_DEVICE_NUM> ccuMsCommIds_{};
    std::array<OrderLaunchThreadMgr, MAX_MODULE_DEVICE_NUM> orderLaunchThreadMgrs_;
    HcclTaskAbortHandler taskAbortHandler_;

    std::array<HcclOpInfoCtx, MAX_MODULE_DEVICE_NUM + 1> opHcomInfos_;
    std::array<bool, MAX_MODULE_DEVICE_NUM + 1> baseCommInited_{};
    std::mutex opHcomInfosMutex_;

    std::mutex mutex_;
    std::mutex ccuMsCommMutex_;
};
} // namespace hccl
#endif // COLL_COMM_MGR_H
