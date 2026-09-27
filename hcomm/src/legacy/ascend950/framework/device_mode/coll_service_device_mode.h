/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COLL_SERVICE_DEVICE_MODE_H
#define COLL_SERVICE_DEVICE_MODE_H

#include <memory>
#include "coll_service_base.h"
#include "communicator_impl.h"
#include "ins_queue.h"
#include "ccu_communicator.h"
#include "ccu_ins_preprocessor.h"
#include "aicpu_ins_preprocessor.h"
#include "mc2_compont.h"
#include "aicpu_stream_manager.h"
#include "aiv_mc2_compont.h"
#include "aiv_ins_preprocessor.h"
#include "aiv_ins.h"
#include "hccl_aiv_utils.h"

namespace Hccl {

class CollServiceDeviceMode : public CollServiceBase {
public:
    explicit CollServiceDeviceMode(CommunicatorImpl* comm)
        : CollServiceBase(comm),
          ccuInsPreprocessor(comm),
          aivInsPreprocessor(comm),
          aicpuInsPreprocessor(comm),
          mc2Compont(comm),
          aivMc2Compont(comm)
    {}

    void Init() override;
    // 单算子
    void LoadWithOpBasedMode(CollOperator& op, std::unique_ptr<Stream> stream) override;
    // 图模式
    void LoadWithOffloadMode(CollOperator& op, std::unique_ptr<Stream> stream) override;

    // MC2 CCU用
    void AllocCommResource(void* mc2Tiling, void** commContext, const AcceleratorState& tilingAccelerator) override;
    void GetCcuTaskInfo(void* tilingData, void* ccuTaskGroup) override;
    u32 GetCcuMc2ServerNum() override;
    CcuInsPreprocessor* GetCcuInsPreprocessor();
    AivInsPreprocessor* GetAivInsPreprocessor();
    std::vector<LinkData> GetUniqueLinks(std::shared_ptr<InsQueue>& insQueue) const;
    // Aicpu用
    AicpuInsPreprocessor* GetAicpuInsPreprocessor();
    bool IsAicpuResExisted(std::string algName);
    DevBuffer* GetAicpuResBuffer(std::string algName);

    // 快照保存和恢复场景使用
    void RecoverTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair) override;
    void RecoverCcuTransport(vector<LinkData>& links, vector<std::pair<LinkGroup, u32>> linkGroupPair);
    HcclResult GetSnapShotDynamicBuf(CollOperator& op, BinaryStream& buf) override;
    bool IsAllTransportRecoveredReady(const std::string& opTag) override;

    const Mc2Compont& GetMc2Compont() const { return mc2Compont; }
    const AivMc2Compont& GetAivMc2Compont() const { return aivMc2Compont; }
    // N秒快恢场景使用
    void Resume() override;

    HcclResult GetAlgExecParam(bool clearEnable, u32 numBlocks, void*& commContext, u64& len) override;

private:
    CcuInsPreprocessor ccuInsPreprocessor;
    AivInsPreprocessor aivInsPreprocessor;
    AicpuInsPreprocessor aicpuInsPreprocessor;
    AicpuStreamManager aicpuStreamManager;

    Mc2Compont mc2Compont;
    AivMc2Compont aivMc2Compont;
    std::unordered_set<u32> captureModelIds;

    std::shared_ptr<InsQueue> Orchestrate(const CollAlgOperator& op) const;

    // AIV场景Acl Graph专用
    HcclResult HandleAclGraphFirstOpAivBuff(rtStream_t mainStream);
    // 生成AivOpArgs，AIV superKernel
    HcclResult GenerateAivOpArgs(const AivInstruction& aivInstruction, AivOpArgs& aivOpArgs) const;
    void GeneratorAivSuperKernelArgs(
        const AivOpArgs& aivOpArgs, bool clearEnable, u32 numBlocks, AivSuperKernelArgs& superArgs) const;
};

} // namespace Hccl

#endif // COLL_SERVICE_DEVICE_MODE_H
