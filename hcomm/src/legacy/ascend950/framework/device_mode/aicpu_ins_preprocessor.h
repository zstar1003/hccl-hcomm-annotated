/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_INS_PREPROCESSOR_H
#define AICPU_INS_PREPROCESSOR_H

#include <unordered_map>
#include <memory>
#include "aicpu_ins.h"
#include "ins_queue.h"
#include "dev_buffer.h"
#include "communicator_impl.h"
#include "virtual_topo.h"
#include "connections_builder.h"

namespace Hccl {

class AicpuInsPreprocessor {
public:
    using InsIterator = HierarchicalQueue<Instruction, InsQueue>::Iterator;

    explicit AicpuInsPreprocessor(CommunicatorImpl* comm) : comm(comm) {}

    void Preprocess(std::shared_ptr<InsQueue>& insQueue);

    // AlltoallV算子使用
    void SetAicpuKernelLaunchParam(HcclKernelLaunchParam& param);
    bool IsAicpuResExisted(const std::string& algName);
    void SetAicpuResExisted(const std::string& algName);
    DevBuffer* GetAicpuResBuffer(const std::string& algName);

private:
    CommunicatorImpl* comm;
    std::unordered_map<std::string, std::shared_ptr<DevBuffer>> aicpuResMap; // 集合通信算子资源加载到device侧的内存
    std::unordered_map<std::string, bool> aicpuResExistedMap;

    std::set<LinkData> availableLinks;
    unordered_map<std::string, unique_ptr<ConnectionsBuilder>> connectionsBuilders;

    // AlltoallV算子使用
    std::vector<std::shared_ptr<DevBuffer>> sendCountsMem{};
    std::vector<std::shared_ptr<DevBuffer>> recvCountsMem{};
    std::vector<std::shared_ptr<DevBuffer>> sdisplsMem{};
    std::vector<std::shared_ptr<DevBuffer>> rdisplsMem{};
    bool isCountMemInited{false};
    u32 resIndex{0};
    u32 launchResIndex{0};

    void InsPreprocess(InsIterator& insIter);

    void AllocQueueNotify(std::vector<std::tuple<QId, QId, u32>>& queueNotifyReq) const;
    void AllocBcastPostCntNotify(std::vector<std::pair<QId, u32>>& bcastPostCntNotifyReq) const;
    void AllocWaitGroupCntNotify(std::vector<std::pair<QId, u32>>& waitGroupCntNotifyReq) const;
    void AllocWorkStream(u32 primQueueNum) const;
    void AllocInterRankNotifies(const vector<LinkData>& links);
    void BatchBuildTransports(const vector<LinkData>& links);

    void PackResAndCopyToDev(const std::string& algName, const CollAlgResReq& collAlgResReq);

    // AlltoallV算子使用
    void AllocAlltoallVOpMem();

    std::vector<char> PackOpData(const std::string& opTag, const std::string& algName, const CollAlgResReq& resReq);
};

} // namespace Hccl

#endif // AICPU_INS_PREPROCESSOR_H
