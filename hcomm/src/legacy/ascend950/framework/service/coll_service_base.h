/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_COLLSERVICEBASE_H
#define HCCLV2_COLLSERVICEBASE_H

#include <string>
#include <unordered_map>
#include "dev_buffer.h"
#include "coll_operator.h"
#include "stream/stream.h"
#include "data_buf_manager.h"
#include "hccl_params_pub.h"
#include "virtual_topo.h"
#include "ccu_rank_group.h"
#include "ins_queue.h"
#include "host_buffer.h"

namespace Hccl {
class CollServiceBase {
public:
    explicit CollServiceBase(CommunicatorImpl* comm);
    virtual void Init() = 0;
    virtual void LoadWithOpBasedMode(CollOperator& op, std::unique_ptr<Stream> stream) = 0;
    virtual void LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream) = 0;
    virtual ~CollServiceBase();

    virtual void AllocCommResource(void* mc2Tiling, void** commContext, const AcceleratorState& tilingAccelerator);
    virtual HcclResult AllocCollOpResource(CollOperator& op, const std::string& opAlgTag, void** addr);
    virtual void GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup);
    virtual u32 GetCcuMc2ServerNum();
    virtual void RecoverTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair) = 0;

    virtual bool IsAllTransportRecoveredReady(const std::string& opTag);

    DevBuffer* GetOpCounterBuf();

    virtual HcclResult GetSnapShotDynamicBuf(CollOperator& op, BinaryStream& buf);

    void WaitTransportReady(const std::string& opTag) const;

    virtual void Resume();

    virtual void ReLoadWithOpBasedMode(CollOperator& op);
    virtual void ReLoadWithOffloadMode(CollOperator& op);

    virtual HcclResult GetAlgExecParam(bool clearEnable, u32 numBlocks, void*& commContext, u64& len);

protected:
    void RegisterOpBufToBufMgr(CollOperator& op);

    void RegisterCclLocRmaBuffer() const;
    void RegisterCclBuffer(const std::vector<LinkData>& links) const;

    void RegisterOpbasedStream(std::unique_ptr<Stream> stream);

    void RegisterOpbasedLocalRmaBuf(const std::string& opTag) const;
    void RegisterOffloadLocalRmaBuf(const std::string& opTag) const;
    void RegisterOffloadMasterStream(const std::string& opTag, std::unique_ptr<Stream> stream) const;

    void WaitOpbasedTransportReady() const;
    void WaitOffloadTransportReady(const std::string& opTag) const;

    void AddOpCounterMems();
    void SaveMirrorDfxOpInfo();

    std::pair<u32, u32> GetOpCount();

    void AddCountTask(bool isHead);

    std::shared_ptr<DevBuffer> counterBuf{nullptr};

    CommunicatorImpl* comm{nullptr};

    virtual void AllocQueueNotify(const InsQueue& insQueue);

    virtual void AllocQNotifyForSingleQ(const InsQueue& insQueue) const;
};
} // namespace Hccl
#endif // HCCL_COLLSERVICEBASE_H
