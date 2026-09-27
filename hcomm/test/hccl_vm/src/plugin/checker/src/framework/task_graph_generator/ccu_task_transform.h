/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_CCU_TRANSFORM_TASK_H
#define HCCLV2_CCU_TRANSFORM_TASK_H

#include <cstdint>
#include <hccl_types.h>
#include <map>
#include <memory>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include "base.h"
#include "ccu_instr_info.h"
#include "ccu_microcode_v1.h"
#include "data_slice.h"
#include "data_type.h"
#include "log.h"
#include "sim_task.h"
#include "task_ccu.h"
#include "task_def.h"
#include "task_graph_generator.h"

using namespace hcomm;
namespace HcclSim {
// 几个常量定义
constexpr uint16_t LOAD_TYPE = 0x0;
constexpr uint16_t CTRL_TYPE = 0x1;
constexpr uint16_t TRANS_TYPE = 0x2;
constexpr uint16_t REDUCE_TYPE = 0x3;

HcclResult GenCcuGraph(TaskNode* dummyStart);

HcclResult
TransformInstr(const CcuRep::CcuInstr* instr, uint32_t rankId, uint32_t queId, TaskNode*& preNode, bool& isContinue);

HcclResult GetHcclDataTypeFromCCUDataType(uint16_t ccuDataType, uint16_t ccuReduceType, DataType& dataType);

uint32_t GetTopicId(HcclSim::TaskNode* post);

void SetTopicId(HcclSim::TaskNode* post, u32 topicId);

HcclResult ProcessWaitMask(
    RankId rankId, uint32_t dieId, TaskStubCcuGraph* curCcuTask, uint32_t queId, uint16_t waitCKEId,
    uint16_t waitCKEMask, bool& isContinue);

// 转换指令的函数指针
using TransformInstrFunc = HcclResult (*)(const CcuRep::CcuInstr*, TaskStubCcuGraph*, uint32_t, bool&, void*);

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
    virtual HcclResult
    Transform(const CcuRep::CcuInstr* instr, TaskStubCcuGraph* task, uint32_t rankId, bool& isContinue, void* loopParam)
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
    CcuInstrVersion GetVersion() override { return CcuInstrVersion::VERSION_A5; }
    // 是否支持该指令
    bool IsSupported(uint16_t header) override
    {
        return transformInstrSqeMap.find(header) != transformInstrSqeMap.end();
    }
    // 指令转换函数
    HcclResult Transform(
        const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam);

private:
    // 指令转换函数
    std::unordered_map<uint16_t, TransformInstrFunc> transformInstrSqeMap;
};

class InstructMapA6 : public InstructMapBase {
public:
    InstructMapA6();
    // 获取指令版本
    CcuInstrVersion GetVersion() override { return CcuInstrVersion::VERSION_A6; }
    // 是否支持该指令,放基类的话无法访问到
    bool IsSupported(uint16_t header) override
    {
        return transformInstrSqeMap.find(header) != transformInstrSqeMap.end();
    }
    // 指令转换函数
    HcclResult Transform(
        const CcuRep::CcuInstr* instr, TaskStubCcuGraph* curCcuTask, uint32_t queId, bool& isContinue, void* loopParam);

private:
    // 指令转换函数
    std::unordered_map<uint16_t, TransformInstrFunc> transformInstrSqeMap;
};
} // namespace HcclSim
#endif
