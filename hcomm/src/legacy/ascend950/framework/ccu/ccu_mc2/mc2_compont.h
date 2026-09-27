/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MC2_COMPONT_H
#define MC2_COMPONT_H

#include <memory>
#include "communicator_impl.h"
#include "ins_exe_que.h"
#include "mc2_type.h"
#include "task_param.h"

namespace Hccl {

struct HcclAlgoInfo {
    uint32_t opType;
    uint8_t algorithmType;
};

class Mc2Compont {
public:
    explicit Mc2Compont(CommunicatorImpl* comm) : comm(comm) {}
    ~Mc2Compont();
    Mc2Compont(const Mc2Compont& that) = delete;
    Mc2Compont& operator=(const Mc2Compont& that) = delete;
    Mc2Compont(Mc2Compont&& that) = delete;
    Mc2Compont& operator=(Mc2Compont&& that) = delete;

    void AllocCommResource(void* mc2Tiling, void** commContext);
    std::vector<CcuTaskParam> GetCcuTaskInfo(void* tilingData);
    std::vector<CcuTaskParam> GetAlgoCcuTaskInfo(InsExeQue::ExtInsExeEntityId execId) const;
    u32 GetCcuMc2ServerNum();
    void DisplayRPCMsg() const;

private:
    void Alloc();
    void AllocV2();
    void GenerateCcuServer(const std::unordered_set<uint64_t>& algoTemplateRequire);
    bool
    FindCcuServer(const std::unordered_set<uint64_t>& algoTemplateRequire, InsExeQue::ExtInsExeEntityId& execId) const;
    void GenerateAlgoTemplates(Mc2Tiling* mc2TilingPtr, std::unordered_set<uint64_t>& algoTemplateRequire);
    void
    GenerateAlgoTemplatesV2(const Mc2InitTilingInner* mc2TilingPtr, std::unordered_set<uint64_t>& algoTemplateRequire);
    void FillCollOperator(const Mc2CommConfig& config);
    void FillCollOperatorV2(const Mc2CcTilingInner& config);
    uint64_t GetTemplateSignature(const Mc2CommConfig& config) const;
    uint64_t GetTemplateSignatureV2(const Mc2CcTilingInner& config) const;
    void SaveMc2DfxTaskInfo(const CcuTaskParam& ccuTaskParam, uint64_t execId) const;
    bool CompareMissionMap(
        const std::map<uint8_t, std::map<uint32_t, uint32_t>>& mapA,
        const std::map<uint8_t, std::map<uint32_t, uint32_t>>& mapB) const;
    void MC2Orchestrate(const CollAlgParams& params, std::shared_ptr<InsQueue>& insQueue, uint8_t commEngine) const;
    void MC2AllocCommRes(const CollAlgParams& params, std::shared_ptr<InsQueue>& insQueue, uint8_t commEngine) const;
    void SaveAlgoInfo(uint32_t index, uint64_t templateSign, uint32_t opType, uint8_t algorithmType);

private:
    const uint32_t dataCount = 1024;
    CommunicatorImpl* comm;
    // algoTemplateMap已经生成的算子集合; key:签名, value: taskParam
    std::unordered_map<uint64_t, std::vector<std::vector<CcuTaskParam>>> algoTemplateMap;
    std::unordered_map<uint64_t, HcclAlgoInfo> algoInfoMap_;
    // ccuServer已经生成的server集合; key:execId, value:该server支持的算子签名
    std::unordered_map<InsExeQue::ExtInsExeEntityId, std::unordered_set<uint64_t>> ccuServerMap;

    std::shared_ptr<DevBuffer> workspaceBuffer{nullptr};
    std::shared_ptr<DevBuffer> combinOpParamBuffer{nullptr};
    std::shared_ptr<DevBuffer> comParamBuffer{nullptr};
    std::shared_ptr<DevBuffer> comSyncBuffer{nullptr};
    CcuResPack ccuResPack;
    InsExeQue::ExtInsExeEntityId curExecId{0};
    uint64_t tokenInfo{0};
    std::shared_ptr<DevBuffer> inputMem{nullptr};

    std::vector<u64> dataCounts;
    std::vector<u64> displs;

    HcclCombinOpParam combinOpParam{};
    bool ccuResourceAlloced{false}; // 标记通信域粒度资源已经申请过了
};
} // namespace Hccl
#endif // MC2_COMPONT_H
