/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_COMM_AICPU_H
#define COLL_COMM_AICPU_H

#include "common.h"
#include "aicpu_init_param.h"
#include "topo_matcher.h"
#include "hcomm_primitives.h"
#include "transport_pub.h"
#include "thread.h"
#include "local_notify.h"
#include "ub_transport_lite_impl.h"
#include "p2p_transport_lite_impl.h"
#include "task_exception.h"
#include "aicpu_launch_manager.h"
#include "channel_param.h"
#include "hdc_pub.h"
#include "ns_recovery/aicpu/ns_recovery_lite.h"
#include <atomic>
#include "hcclCommDfxLite.h"
#include "error_message_v2.h"
#include "kfc.h"
#include "aicpu_hdc.h"
#include "roce_transport_lite_impl.h"
#include "hccl/hccl_types.h"
#include "comm_engine_res_aicpu_mgr.h"
#include "channel_aicpu_mgr.h"

namespace hccl {
class HcclCommAicpu;
}

class CollCommAicpu {
public:
    ~CollCommAicpu();
    HcclResult InitAicpuIndOp(CommAicpuParam* commAicpuParam);

    // 资源管理 — 通过mgr指针暴露，调用者通过mgr操作资源
    CommEngineResAicpuMgr* GetCommEngineResMgr() { return commEngineResMgr_.get(); }
    ChannelAicpuMgr* GetChannelMgr() { return channelMgr_.get(); }

    // 910B legacy 通信域管理
    hccl::HcclCommAicpu* GetLegacy910CollComm();
    void SetLegacy910CollComm(std::shared_ptr<hccl::HcclCommAicpu> comm);
    bool IsLegacy910CollCommBusy();
    void SetLegacy910CollCommBusy(bool busy);

    const hccl::HcclTopoInfo& GetTopoInfo() { return topoInfo_; }
    const std::string& GetIdentifier() { return identifier_; }

    // taskException
    bool IsErrorReported() { return isErrorReported_; }
    void SetErrorReported(bool isErrorReported) { isErrorReported_ = isErrorReported; }
    HcclResult SendErrorMessageReportToHost(Hccl::ErrorMessageReport& errMsgInfo);
    HcclResult RegisterProfCallBack();
    hccl::HcclCommDfxLite* GetHcclCommDfxLite() { return &dfx_; };
    u32 GetDevId() { return devId_; }

    // h2d - d2h通道信息交互
    HcclResult BackGroundGetCmd(Hccl::KfcCommand& cmd);
    HcclResult BackGroundSetStatus(Hccl::KfcStatus state);
    u32 UpdateIndex();

    HcclCommStatus GetCommmStatus() { return commStatus_; }
    void SetCommmStatus(HcclCommStatus status);

    // N秒快恢
    hccl::NsRecoveryLitePtr GetNsRecoveryLitePtr();
    HcclResult Clean();
    HcclResult Resume(HcclChannelUrmaRes* commParam);

    HcclResult CheckIndOpExecStatus(bool timeout);

    // DFX — 单通信域粒度操作
    HcclResult InitDfxOpInfo(HcclDfxOpInfo* aicpuDfxInfo);
    HcclResult ProfilingReportDeviceOp();
    HcclResult UpdateTask();

private:
    HcclResult InitHDCommunicate(CommAicpuParam* commAicpuParam);

    u32 devId_{0};
    // 通用的通道
    std::shared_ptr<hccl::HDCommunicate> kfcControlTransferH2D_{nullptr};
    std::shared_ptr<hccl::HDCommunicate> kfcStatusTransferD2H_{nullptr};

    std::string identifier_;
    HcclCommStatus commStatus_{HcclCommStatus::HCCL_COMM_STATUS_INVALID};
    hccl::HcclTopoInfo topoInfo_;

    // dfx — 必须在 commEngineResMgr_/channelMgr_ 之前声明，确保后析构
    bool isErrorReported_{false};
    hccl::HcclCommDfxLite dfx_;

    // 资源管理 — 通过mgr持有（持有 dfx_ 引用）
    std::unique_ptr<CommEngineResAicpuMgr> commEngineResMgr_;
    std::unique_ptr<ChannelAicpuMgr> channelMgr_;

    // 910B legacy 通信域 — CollCommAicpu 作为 wrapper，通过 shared_ptr 共享所有权
    // std::atomic_bool 用于标记使用中
    std::pair<std::shared_ptr<hccl::HcclCommAicpu>, std::atomic_bool> legacy910CollComm_;

    // N秒快恢相关
    hccl::NsRecoveryLitePtr nsRecoveryLitePtr_{nullptr};

    u32 index_{0};
};

#endif // COLL_COMM_AICPU_H
