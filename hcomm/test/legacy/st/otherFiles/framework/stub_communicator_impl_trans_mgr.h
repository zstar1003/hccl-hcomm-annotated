/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_STUB_COMMUNICATOR_IMPL_TRANS_MGR_H
#define HCCLV2_STUB_COMMUNICATOR_IMPL_TRANS_MGR_H

#define private public
#define protected public
#include "communicator_impl.h"
#undef private
#undef protected

namespace Hccl {

class StubCommunicatorImplTransMgr : public CommunicatorImpl {
public:
    StubCommunicatorImplTransMgr()
    {
        myRank = 0;
        dataBufferManager = std::make_unique<DataBufManager>();

        localRmaBufManager = std::make_unique<LocalRmaBufManager>(*this);

        remoteRmaBufManager = std::make_unique<RemoteRmaBufManager>(*this);

        rmaConnectionManager = std::make_unique<RmaConnManager>(*this);

        connLocalNotifyManager = std::make_unique<ConnLocalNotifyManager>(this);

        connLocalCntNotifyManager = std::make_unique<ConnLocalCntNotifyManager>(this);

        aicpuQueueNotifyManager_ = std::make_unique<QueueNotifyManager>(*this);

        ccuQueueNotifyManager_ = std::make_unique<QueueNotifyManager>(*this);

        queueWaitGroupCntNotifyManager = std::make_unique<QueueWaitGroupCntNotifyManager>();

        queueBcastPostCntNotifyManager = std::make_unique<QueueBcastPostCntNotifyManager>();

        socketManager = std::make_unique<SocketManager>(*this, 0, 0, 60001);

        currentCollOperator = std::make_unique<CollOperator>();
        currentCollOperator->opMode = OpMode::OPBASE;
        currentCollOperator->opTag = "op_base";

        mirrorTaskManager = std::make_unique<MirrorTaskManager>(0, &GlobalMirrorTasks::Instance(), 0);
        std::shared_ptr<DfxOpInfo> dfxOpInfo = std::make_shared<DfxOpInfo>();
        CollOperator op;
        op.opType = OpType::ALLREDUCE;
        op.staticAddr = false;
        dfxOpInfo->op_ = op;

        mirrorTaskManager->SetCurrDfxOpInfo(dfxOpInfo);
    }

    void SetOp(OpMode opMode, string tag)
    {
        currentCollOperator->opMode = opMode;
        currentCollOperator->opTag = tag;
    }

    DataBufManager& GetDataBufferManager() const override { return *dataBufferManager.get(); }

    LocalRmaBufManager& GetLocalRmaBufManager() const override { return *localRmaBufManager.get(); }

    RemoteRmaBufManager& GetRemoteRmaBufManager() const override { return *remoteRmaBufManager.get(); }

    QueueNotifyManager& GetAicpuQueueNotifyManager() const override { return *aicpuQueueNotifyManager_.get(); }

    QueueNotifyManager& GetCcuQueueNotifyManager() const override { return *ccuQueueNotifyManager_.get(); }

    RmaConnManager& GetRmaConnManager() const override { return *rmaConnectionManager.get(); }

    CollOperator* GetCurrentCollOperator() const override { return currentCollOperator.get(); }

    ConnLocalNotifyManager& GetConnLocalNotifyManager() const override { return *connLocalNotifyManager; }

    ConnLocalCntNotifyManager& GetConnLocalCntNotifyManager() const override { return *connLocalCntNotifyManager; }

    SocketManager& GetSocketManager() const override { return *socketManager; }

    const string& GetEstablishLinkSocketTag() const override { return socketTag; }

    MirrorTaskManager& GetMirrorTaskManager() const override { return *mirrorTaskManager.get(); }

private:
    string socketTag = "tag";

    std::unique_ptr<DataBufManager> dataBufferManager;
    std::unique_ptr<LocalRmaBufManager> localRmaBufManager;
    std::unique_ptr<RemoteRmaBufManager> remoteRmaBufManager;
    unique_ptr<QueueNotifyManager> aicpuQueueNotifyManager_;
    unique_ptr<QueueNotifyManager> ccuQueueNotifyManager_;
    std::unique_ptr<ConnLocalNotifyManager> connLocalNotifyManager;
    std::unique_ptr<ConnLocalCntNotifyManager> connLocalCntNotifyManager;
    std::unique_ptr<StreamManager> streamManager;
    std::unique_ptr<SocketManager> socketManager;
    std::unique_ptr<RmaConnManager> rmaConnectionManager;
    std::unique_ptr<CollServiceBase> collService;
    std::unique_ptr<CollOperator> currentCollOperator;
    std::unique_ptr<MirrorTaskManager> mirrorTaskManager;
};
} // namespace Hccl
#endif
