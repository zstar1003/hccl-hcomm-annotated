/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TRANSFORM_TASK_V3_H
#define HCCLV2_CCU_TRANSFORM_TASK_V3_H

#include <cstddef>
#include <cstdint>
#include <map>
#include <set>
#include <queue>
#include <vector>
#include <memory>
#include <unordered_map>
#include <hccl_types.h>

#include "base.h"
#include "log.h"
#include "data_slice.h"
#include "ccu_microcode_v1.h"
#include "ccu_instr_info.h"
#include "data_type.h"
#include "ccu_task_common_v3.h"

using namespace hcomm;
namespace HcclSim {
namespace TaskGraphGeneratorV3 {
    // 几个常量定义
    constexpr uint16_t LOAD_TYPE = 0x0;
    constexpr uint16_t CTRL_TYPE = 0x1;
    constexpr uint16_t TRANS_TYPE = 0x2;
    constexpr uint16_t REDUCE_TYPE = 0x3;

    struct CcuGraphsGenerateInputV3 {
        TaskGraphGeneratorV3* graph{nullptr};
        std::vector<TaskCcuGraph*> ccuGraphs;
        StorageManager* storage{nullptr};
    };

    struct CcuGraphGenerateOutputV3 {
        NodeId startNodeId{INVALID_NODE_ID};
        NodeId endNodeId{INVALID_NODE_ID};
        size_t internalNodeCount{0};
        size_t recordWaitEdgeCount{0};
        uint64_t expandNs{0};
    };

    HcclResult ExpandCcuGraphsV3(const CcuGraphsGenerateInputV3& input, CcuGraphGenerateOutputV3& output);

    HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType);

    uint32_t GetPostRemainingCkeMask(TaskNode* post);

    void SetPostRemainingCkeMask(TaskNode* post, u32 remainingCkeMask);

    HcclResult ProcessWaitMask(
        RankId rankId, uint32_t dieId, CcuGraphStateV3* curCcuTask, uint32_t queId, uint16_t waitCKEId,
        uint16_t waitCKEMask, bool& isContinue);

    // 转换指令的函数指针
    using TransformInstrFunc = HcclResult (*)(const CcuRep::CcuInstr*, CcuGraphStateV3*, uint32_t, bool&, void*);

    // CCU指令版本
    enum class CcuInstrVersion : uint16_t { VERSION_A5 = 1, VERSION_A6 = 2 };

    class InstructMapBase {
    public:
        virtual ~InstructMapBase() = default;
        // 获取指令版本
        virtual CcuInstrVersion GetVersion() = 0;
        // 是否支持该指令
        virtual bool IsSupported(uint16_t header) = 0;
        // 指令转换函数
        virtual HcclResult Transform(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* task, uint32_t rankId, bool& isContinue, void* loopParam)
            = 0;
    };

    class InstructMapFactory {
    public:
        static std::unique_ptr<InstructMapBase> Create(CcuInstrVersion version);
    };

    class InstructMapA5 : public InstructMapBase {
    public:
        InstructMapA5();
        // 获取指令版本
        CcuInstrVersion GetVersion() override { return CcuInstrVersion::VERSION_A5; };
        // 是否支持该指令
        bool IsSupported(uint16_t header) override
        {
            return transformInstrSqeMap.find(header) != transformInstrSqeMap.end();
        };
        // 指令转换函数
        HcclResult Transform(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam);

    private:
        // 指令转换函数
        std::unordered_map<uint16_t, TransformInstrFunc> transformInstrSqeMap;
    };

    class InstructMapA6 : public InstructMapBase {
    public:
        InstructMapA6();
        // 获取指令版本
        CcuInstrVersion GetVersion() override { return CcuInstrVersion::VERSION_A6; };
        // 是否支持该指令,放基类的话无法访问到
        bool IsSupported(uint16_t header) override
        {
            return transformInstrSqeMap.find(header) != transformInstrSqeMap.end();
        };
        // 指令转换函数
        HcclResult Transform(
            const CcuRep::CcuInstr* instr, CcuGraphStateV3* curCcuTask, uint32_t queId, bool& isContinue,
            void* loopParam);

    private:
        // 指令转换函数
        std::unordered_map<uint16_t, TransformInstrFunc> transformInstrSqeMap;
    };

} // namespace TaskGraphGeneratorV3
} // namespace HcclSim
#endif
