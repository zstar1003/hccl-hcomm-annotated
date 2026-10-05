/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "selector_engine.h"

#include <cstring>
#include <algorithm>
#include <set>

#include "alg_env_config.h"
#include "alg_parse.h"
#include "algo_name_mapper.h"
#include "config_log.h"
#include "cost_model.h"
#include "cost_table.h"
#include "alg_attrs.h"
#include "alg_attrs_registry.h"
#include "hccl_algo_dims.h"
#include "hccl_common.h"
#include "tuner_setup.h"
#include "auto_selector_base.h"

namespace ops_hccl {

static constexpr const char* COST_MODEL_TAG = "costmodel";
static constexpr const char* TUNER_INIT_TAG = "tuner_init";

// [中文导读] [AllReduce逐行 S35] 定义 Global 入口：返回进程内 SelectorEngine 单例。
SelectorEngine* SelectorEngine::Global()
// [中文导读] [AllReduce逐行 S36] 进入 Global 的实现作用域；返回进程内 SelectorEngine 单例。
{
    // [中文导读] [AllReduce逐行 S37] 设置 static SelectorEngine* globalSelectorEngine 为 new SelectorEngine；该值供下方当前分支使用。
    static SelectorEngine* globalSelectorEngine = new SelectorEngine;
    // [中文导读] [AllReduce逐行 S38] 直接返回 globalSelectorEngine，调用者取得本分支结果。
    return globalSelectorEngine;
// [中文导读] [AllReduce逐行 S39] 结束 Global 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S41] 定义 IsOpSupported 入口：查询新成本选择器支持的算子集合；AllReduce 在支持列表中。
bool SelectorEngine::IsOpSupported(HcclCMDType opType)
// [中文导读] [AllReduce逐行 S42] 进入 IsOpSupported 的实现作用域；查询新成本选择器支持的算子集合；AllReduce 在支持列表中。
{
    // 本迭代新选择器仅支持以下算子, 其他算子走老流程
    // [中文导读] [AllReduce逐行 S44] 设置 static const std::set<HcclCMDType> supportedOps 为 {；该值供下方当前分支使用。
    static const std::set<HcclCMDType> supportedOps = {
        // [中文导读] [AllReduce逐行 S45] 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_ALLREDUCE, HcclCMDType::HCCL_CMD_REDUCE_SCATTER, HcclCMDType::HCCL_CMD_ALLGATHER,；由其完整表达式完成参数组装、检查或结果写回。
        HcclCMDType::HCCL_CMD_ALLREDUCE,  HcclCMDType::HCCL_CMD_REDUCE_SCATTER,  HcclCMDType::HCCL_CMD_ALLGATHER,
        // [中文导读] [AllReduce逐行 S46] 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_REDUCE, HcclCMDType::HCCL_CMD_SCATTER, HcclCMDType::HCCL_CMD_BROADCAST,；由其完整表达式完成参数组装、检查或结果写回。
        HcclCMDType::HCCL_CMD_REDUCE,     HcclCMDType::HCCL_CMD_SCATTER,         HcclCMDType::HCCL_CMD_BROADCAST,
        // [中文导读] [AllReduce逐行 S47] 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_BARRIER, HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLTOALLV,；由其完整表达式完成参数组装、检查或结果写回。
        HcclCMDType::HCCL_CMD_BARRIER,    HcclCMDType::HCCL_CMD_BATCH_SEND_RECV, HcclCMDType::HCCL_CMD_ALLTOALLV,
        // [中文导读] [AllReduce逐行 S48] 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V, HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V,；由其完整表达式完成参数组装、检查或结果写回。
        HcclCMDType::HCCL_CMD_ALLTOALLVC, HcclCMDType::HCCL_CMD_ALLGATHER_V,     HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V,
        // [中文导读] [AllReduce逐行 S49] 续接 IsOpSupported 当前语句的具体实参/字段：HcclCMDType::HCCL_CMD_SEND, HcclCMDType::HCCL_CMD_RECEIVE, HcclCMDType::HCCL_CMD_ALLTOALL,；由其完整表达式完成参数组装、检查或结果写回。
        HcclCMDType::HCCL_CMD_SEND,       HcclCMDType::HCCL_CMD_RECEIVE,         HcclCMDType::HCCL_CMD_ALLTOALL,
    // [中文导读] [AllReduce逐行 S50] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    };
    // [中文导读] [AllReduce逐行 S51] 直接返回 调用 count 完成当前参数所指的子步骤 的结果，调用者取得本分支结果。
    return supportedOps.count(opType) > 0;
// [中文导读] [AllReduce逐行 S52] 结束 IsOpSupported 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S54] 定义 GetEngineByAlgName 入口：按注册的 Pascal 前缀反推算法执行配置，未知前缀默认 AICPU_TS。
OpExecuteConfig SelectorEngine::GetEngineByAlgName(const std::string& algName)
// [中文导读] [AllReduce逐行 S55] 进入 GetEngineByAlgName 的实现作用域；按注册的 Pascal 前缀反推算法执行配置，未知前缀默认 AICPU_TS。
{
    // [中文导读] [AllReduce逐行 S56] 设置 count 为 0；该值供下方当前分支使用。
    int count = 0;
    // [中文导读] [AllReduce逐行 S57] 设置 const EnginePrefixEntry* entries 为 GetEnginePrefixEntries(count)；该值供下方当前分支使用。
    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);
    // [中文导读] [AllReduce逐行 S58] 在 GetEngineByAlgName 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。
    for (int i = 0; i < count; ++i) {
        // [中文导读] [AllReduce逐行 S59] 设置 len 为 strlen(entries[i].pascal)；该值供下方当前分支使用。
        size_t len = strlen(entries[i].pascal);
        // [中文导读] [AllReduce逐行 S60] 分支条件为 algName.size() 至少 len 且 algName.substr(0, len) 等于 entries[i].pascal；成立进入本块，未成立继续后续分支。
        if (algName.size() >= len && algName.substr(0, len) == entries[i].pascal) {
            // [中文导读] [AllReduce逐行 S61] 直接返回 entries[i].engine，调用者取得本分支结果。
            return entries[i].engine;
        // [中文导读] [AllReduce逐行 S62] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S63] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S64] 直接返回 OpExecuteConfig::AICPU_TS，调用者取得本分支结果。
    return OpExecuteConfig::AICPU_TS;
// [中文导读] [AllReduce逐行 S65] 结束 GetEngineByAlgName 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S67] 定义 CandidateEnginesToPrefixes 入口：把可候选执行引擎转为算法名前缀，供 HCCL_ALGO 成本模型过滤。
std::vector<std::string> SelectorEngine::CandidateEnginesToPrefixes(const std::vector<OpExecuteConfig>& engines)
// [中文导读] [AllReduce逐行 S68] 进入 CandidateEnginesToPrefixes 的实现作用域；把可候选执行引擎转为算法名前缀，供 HCCL_ALGO 成本模型过滤。
{
    // [中文导读] [AllReduce逐行 S69] 调用 engineSet 完成当前参数所指的子步骤；本行实参为 std::set<OpExecuteConfig> engineSet(engines.begin(), engines.end())。
    std::set<OpExecuteConfig> engineSet(engines.begin(), engines.end());
    // [中文导读] [AllReduce逐行 S70] 建立本阶段局部对象 std::vector<std::string> prefixes，供 CandidateEnginesToPrefixes 下方参数组装和子调用使用。
    std::vector<std::string> prefixes;
    // [中文导读] [AllReduce逐行 S71] 设置 count 为 0；该值供下方当前分支使用。
    int count = 0;
    // [中文导读] [AllReduce逐行 S72] 设置 const EnginePrefixEntry* entries 为 GetEnginePrefixEntries(count)；该值供下方当前分支使用。
    const EnginePrefixEntry* entries = GetEnginePrefixEntries(count);
    // [中文导读] [AllReduce逐行 S73] 在 CandidateEnginesToPrefixes 中遍历 (int i = 0; i 小于 count; ++i 指定的集合或索引区间；边界/迭代规则为 (int i = 0; i 小于 count; ++i。
    for (int i = 0; i < count; ++i) {
        // [中文导读] [AllReduce逐行 S74] 分支条件为 engineSet.count(entries[i].engine) 不等于 0；成立进入本块，未成立继续后续分支。
        if (engineSet.count(entries[i].engine) != 0) {
            // [中文导读] [AllReduce逐行 S75] 对 prefixes 就地追加 entries[i].pascal，准备或更新本阶段列表。
            prefixes.emplace_back(entries[i].pascal);
        // [中文导读] [AllReduce逐行 S76] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S77] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S78] 直接返回 prefixes，调用者取得本分支结果。
    return prefixes;
// [中文导读] [AllReduce逐行 S79] 结束 CandidateEnginesToPrefixes 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S81] 定义 GetEnginePriority 入口：根据指定引擎生成可回退引擎候选列表，AIV_ONLY 列表仅 AIV。
std::vector<OpExecuteConfig> SelectorEngine::GetEnginePriority(OpExecuteConfig opExecuteConfig)
// [中文导读] [AllReduce逐行 S82] 进入 GetEnginePriority 的实现作用域；根据指定引擎生成可回退引擎候选列表，AIV_ONLY 列表仅 AIV。
{
    // [中文导读] [AllReduce逐行 S83] 续接 GetEnginePriority 当前语句的具体实参/字段：switch (opExecuteConfig) {；由其完整表达式完成参数组装、检查或结果写回。
    switch (opExecuteConfig) {
        // [中文导读] [AllReduce逐行 S84] 处理 OpExecuteConfig::CCU_MS 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::CCU_MS:
            // [中文导读] [AllReduce逐行 S85] 直接返回 {，调用者取得本分支结果。
            return {
                // [中文导读] [AllReduce逐行 S86] 续接 GetEnginePriority 当前语句的具体实参/字段：OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,；由其完整表达式完成参数组装、检查或结果写回。
                OpExecuteConfig::CCU_MS, OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS,
                // [中文导读] [AllReduce逐行 S87] 续接 GetEnginePriority 当前语句的具体实参/字段：OpExecuteConfig::HOSTCPU}；由其完整表达式完成参数组装、检查或结果写回。
                OpExecuteConfig::HOSTCPU};
        // [中文导读] [AllReduce逐行 S88] 处理 OpExecuteConfig::CCU_SCHED 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::CCU_SCHED:
            // [中文导读] [AllReduce逐行 S89] 直接返回 {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。
            return {OpExecuteConfig::CCU_SCHED, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        // [中文导读] [AllReduce逐行 S90] 处理 OpExecuteConfig::AIV 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::AIV:
            // [中文导读] [AllReduce逐行 S91] 直接返回 {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。
            return {OpExecuteConfig::AIV, OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        // [中文导读] [AllReduce逐行 S92] 处理 OpExecuteConfig::AIV_ONLY 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::AIV_ONLY:
            // [中文导读] [AllReduce逐行 S93] 直接返回 {OpExecuteConfig::AIV}，调用者取得本分支结果。
            return {OpExecuteConfig::AIV};
        // [中文导读] [AllReduce逐行 S94] 处理 OpExecuteConfig::AICPU_TS 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::AICPU_TS:
            // [中文导读] [AllReduce逐行 S95] 直接返回 {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
        // [中文导读] [AllReduce逐行 S96] 处理 OpExecuteConfig::HOSTCPU 的专用实现，不同类型或运算在其它 case 分开处理。
        case OpExecuteConfig::HOSTCPU:
            // [中文导读] [AllReduce逐行 S97] 直接返回 {OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。
            return {OpExecuteConfig::HOSTCPU};
        // [中文导读] [AllReduce逐行 S98] 未列出的类型或状态进入兜底分支；按下面返回码判为不支持或错误。
        default:
            // [中文导读] [AllReduce逐行 S99] 直接返回 {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU}，调用者取得本分支结果。
            return {OpExecuteConfig::AICPU_TS, OpExecuteConfig::HOSTCPU};
    // [中文导读] [AllReduce逐行 S100] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
// [中文导读] [AllReduce逐行 S101] 结束 GetEnginePriority 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S103] 定义 FilterCmByEngine 入口：把不在候选引擎集合中的成本项 count 清零，保留可参加成本计算的算法。
HcclResult SelectorEngine::FilterCmByEngine(CostModel& cm, const std::vector<OpExecuteConfig>& candidateEngines)
// [中文导读] [AllReduce逐行 S104] 进入 FilterCmByEngine 的实现作用域；把不在候选引擎集合中的成本项 count 清零，保留可参加成本计算的算法。
{
    // [中文导读] [AllReduce逐行 S105] 调用 engineSet 完成当前参数所指的子步骤；本行实参为 std::set<OpExecuteConfig> engineSet(candidateEngines.begin(), candidateEngines.end())。
    std::set<OpExecuteConfig> engineSet(candidateEngines.begin(), candidateEngines.end());
    // [中文导读] [AllReduce逐行 S106] 遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 cm.count; i++。
    for (int i = 0; i < cm.count; i++) {
        // [中文导读] [AllReduce逐行 S107] 分支条件为 cm.costAlgoParams[i].algName 等于 nullptr；成立进入本块，未成立继续后续分支。
        if (cm.costAlgoParams[i].algName == nullptr) {
            // [中文导读] [AllReduce逐行 S108] 跳过当前遍历项的剩余步骤，直接处理下一项。
            continue;
        // [中文导读] [AllReduce逐行 S109] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S110] 设置 OpExecuteConfig engine 为 GetEngineByAlgName(cm.costAlgoParams[i].algName)；该值供下方当前分支使用。
        OpExecuteConfig engine = GetEngineByAlgName(cm.costAlgoParams[i].algName);
        // [中文导读] [AllReduce逐行 S111] 分支条件为 engineSet.count(engine) 等于 0；成立进入本块，未成立继续后续分支。
        if (engineSet.count(engine) == 0) {
            // [中文导读] [AllReduce逐行 S112] 设置 cm.costAlgoParams[i].count 为 0；该值供下方当前分支使用。
            cm.costAlgoParams[i].count = 0;
        // [中文导读] [AllReduce逐行 S113] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S114] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S115] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S116] 结束 FilterCmByEngine 实现；其返回状态或已写回字段由调用者接收。
}

void SelectorEngine::LogAivOnlyNotMatch(const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo)
{
    HCCL_ERROR(
        "Failed to select AIV algorithm while configured as AIV_ONLY. "
        "Current topology: topoLevelNums=%u, level0Topo=%u, level0PcieMix=%d, level2UbRtp=%d, "
        "Level1Nhr=%d, userRankSize=%u. opType=%d, dataType=%d, dataSize=%llu, reduceOp=%d.",
        topoInfo->topoLevelNums, static_cast<uint32_t>(topoInfo->level0Topo), static_cast<int>(topoInfo->level0PcieMix),
        static_cast<int>(topoInfo->level2UbRtp), static_cast<int>(topoInfo->Level1Nhr), topoInfo->userRankSize,
        static_cast<int>(param.opType), static_cast<int>(param.DataDes.dataType), param.inputSize,
        static_cast<int>(param.reduceType));

    // 回溯检查所有 AIV 算法被过滤的原因
    HCCL_ERROR("[SelectorEngine] AIV algorithm filter details:");
    const AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return;
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        const AlgElement& alg = allAlgos->algElements[i];
        if (alg.opType != param.opType || alg.algName == nullptr) {
            continue;
        }
        const AlgAttrs* attrs = AlgAttrsRegistry::Instance().Get(alg.algName);
        if (attrs == nullptr || attrs->engine != OpExecuteConfig::AIV) {
            continue;
        }
        std::string algName = alg.algName;

        // topo 层检查
        auto topoResult = CheckAlgoMatchTopoWithReason(algName, topoInfo);
        if (!topoResult.matched) {
            HCCL_ERROR("[SelectorEngine] algName=%s filtered by topo: %s.", algName.c_str(), topoResult.reason.c_str());
            continue;
        }

        // op 层检查
        auto opResult = CheckAlgoMatchOpWithReason(*attrs, param, topoInfo);
        if (!opResult.matched) {
            HCCL_ERROR("[SelectorEngine] algName=%s filtered by op: %s.", algName.c_str(), opResult.reason.c_str());
        } else {
            HCCL_ERROR("[SelectorEngine] algName=%s passed all filters.", algName.c_str());
        }
    }
}

// [中文导读] [AllReduce逐行 S163] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
HcclResult
// [中文导读] [AllReduce逐行 S164] 定义 InitCostModel 入口：建立每通信域每执行引擎独立成本模型，上下文深拷贝参数，再按引擎和 HCCL_ALGO 配置过滤。
SelectorEngine::InitCostModel(HcclComm comm, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param, CostModel*& cm)
// [中文导读] [AllReduce逐行 S165] 进入 InitCostModel 的实现作用域；建立每通信域每执行引擎独立成本模型，上下文深拷贝参数，再按引擎和 HCCL_ALGO 配置过滤。
{
    // [中文导读] [AllReduce逐行 S166] 开始 HCCL_INFO 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] Initializing costModel for comm, engine=%d.", static_cast<int>(param.opExecuteConfig));

    // 全局一次性：构建算法名→3D 维度映射缓存
    // [中文导读] [AllReduce逐行 S169] 续接 InitCostModel 当前语句的具体实参/字段：static std::once_flag algoMapperFlag；由其完整表达式完成参数组装、检查或结果写回。
    static std::once_flag algoMapperFlag;
    // [中文导读] [AllReduce逐行 S170] 调用 std::call_once 完成当前参数所指的子步骤；本行实参为 std::call_once(algoMapperFlag, []() {。
    std::call_once(algoMapperFlag, []() {
        // [中文导读] [AllReduce逐行 S171] 调用 AlgoNameMapper::Global 完成当前参数所指的子步骤；本行实参为 AlgoNameMapper::Global()->Init(*GetAllAlgos())。
        AlgoNameMapper::Global()->Init(*GetAllAlgos());
    // [中文导读] [AllReduce逐行 S172] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    });

    // 调用 CostModelManager 初始化，costModel 作为出参返回（局部变量，无线程安全问题）
    // [中文导读] [AllReduce逐行 S175] 设置 CostModelManager* costModelMgr 为 CostModelManager::Global()；该值供下方当前分支使用。
    CostModelManager* costModelMgr = CostModelManager::Global();
    // [中文导读] [AllReduce逐行 S176] 续接 InitCostModel 当前语句的具体实参/字段：CostModel srcCm{nullptr, 0}；由其完整表达式完成参数组装、检查或结果写回。
    CostModel srcCm{nullptr, 0};
    // [中文导读] [AllReduce逐行 S177] 首次建立此通信域此引擎的成本模型；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(costModelMgr->InitCostModel(comm, topoInfo, srcCm, param));

    // CostModel 含指针(costAlgoParams),分段拷贝: [CostModel header][CostAlgoParams array]
    // [中文导读] [AllReduce逐行 S180] 设置 ctxPtr 为 nullptr；该值供下方当前分支使用。
    void* ctxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S181] 设置 uint64_t flatSize 为 sizeof(CostModel) + static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams)；该值供下方当前分支使用。
    uint64_t flatSize = sizeof(CostModel) + static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams);
    // [中文导读] [AllReduce逐行 S182] 设置 costModelTag 为 std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(当前执行配置)；该值供下方当前分支使用。
    std::string costModelTag = std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(param.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S183] 调用 HcclEngineCtxCreate 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(HcclEngineCtxCreate(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, flatSize, &ctxPtr));

    // [中文导读] [AllReduce逐行 S185] 设置 CostModel* storedCm 为 static_cast<CostModel*>(ctxPtr)；该值供下方当前分支使用。
    CostModel* storedCm = static_cast<CostModel*>(ctxPtr);
    // [中文导读] [AllReduce逐行 S186] 设置 storedCm->count 为 srcCm.count；该值供下方当前分支使用。
    storedCm->count = srcCm.count;
    // [中文导读] [AllReduce逐行 S187] 设置 storedCm->costAlgoParams 为 reinterpret_cast<CostAlgoParams*>(storedCm + 1)；该值供下方当前分支使用。
    storedCm->costAlgoParams = reinterpret_cast<CostAlgoParams*>(storedCm + 1);
    // [中文导读] [AllReduce逐行 S188] 分支条件为 srcCm.count 大于 0 且 srcCm.costAlgoParams 不等于 nullptr；成立进入本块，未成立继续后续分支。
    if (srcCm.count > 0 && srcCm.costAlgoParams != nullptr) {
        // [中文导读] [AllReduce逐行 S189] 调用 CHK_SAFETY_FUNC_RET 完成当前参数所指的子步骤；本行实参为 CHK_SAFETY_FUNC_RET(memcpy_s(。
        CHK_SAFETY_FUNC_RET(memcpy_s(
            // [中文导读] [AllReduce逐行 S190] 续接 InitCostModel 当前语句的具体实参/字段：storedCm->costAlgoParams, static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams), srcCm.costAlgoParams,；由其完整表达式完成参数组装、检查或结果写回。
            storedCm->costAlgoParams, static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams), srcCm.costAlgoParams,
            // [中文导读] [AllReduce逐行 S191] 按目标类型转换 static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams)))，保持接口参数的数值或地址语义。
            static_cast<uint64_t>(srcCm.count) * sizeof(CostAlgoParams)));
    // [中文导读] [AllReduce逐行 S192] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 深拷贝 param 指针指向的内存，comm ctx 持有独立所有权
    // [中文导读] [AllReduce逐行 S195] 遍历成本模型项，按候选引擎过滤算法条目；边界/迭代规则为 (int i = 0; i 小于 storedCm->count; ++i。
    for (int i = 0; i < storedCm->count; ++i) {
        // [中文导读] [AllReduce逐行 S196] 设置 n 为 storedCm->costAlgoParams[i].count；该值供下方当前分支使用。
        int n = storedCm->costAlgoParams[i].count;
        // [中文导读] [AllReduce逐行 S197] 设置 const CostModelParam* srcParam 为 storedCm->costAlgoParams[i].param；该值供下方当前分支使用。
        const CostModelParam* srcParam = storedCm->costAlgoParams[i].param;
        // [中文导读] [AllReduce逐行 S198] 分支条件为 n 大于 0 且 srcParam 不等于 nullptr；成立进入本块，未成立继续后续分支。
        if (n > 0 && srcParam != nullptr) {
            // [中文导读] [AllReduce逐行 S199] 设置 CostModelParam* owned 为 new (std::nothrow) CostModelParam[n]；该值供下方当前分支使用。
            CostModelParam* owned = new (std::nothrow) CostModelParam[n];
            // [中文导读] [AllReduce逐行 S200] 分支条件为 owned 等于 nullptr；成立进入本块，未成立继续后续分支。
            if (owned == nullptr) {
                // [中文导读] [AllReduce逐行 S201] 开始 HCCL_ERROR 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。
                HCCL_ERROR("[SelectorEngine] alloc param failed, i=%d count=%d.", i, n);
                // [中文导读] [AllReduce逐行 S202] 终止当前函数并向上返回 HcclResult::参数错误；调用者 CHK_RET 决定是否继续向上传播。
                return HcclResult::HCCL_E_PARA;
            // [中文导读] [AllReduce逐行 S203] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S204] 调用 CHK_SAFETY_FUNC_RET 完成当前参数所指的子步骤；本行实参为 CHK_SAFETY_FUNC_RET(memcpy_s(。
            CHK_SAFETY_FUNC_RET(memcpy_s(
                // [中文导读] [AllReduce逐行 S205] 续接 InitCostModel 当前语句的具体实参/字段：owned, static_cast<uint64_t>(n) * sizeof(CostModelParam), srcParam,；由其完整表达式完成参数组装、检查或结果写回。
                owned, static_cast<uint64_t>(n) * sizeof(CostModelParam), srcParam,
                // [中文导读] [AllReduce逐行 S206] 按目标类型转换 static_cast<uint64_t>(n) * sizeof(CostModelParam)))，保持接口参数的数值或地址语义。
                static_cast<uint64_t>(n) * sizeof(CostModelParam)));
            // [中文导读] [AllReduce逐行 S207] 设置 storedCm->costAlgoParams[i].param 为 owned；该值供下方当前分支使用。
            storedCm->costAlgoParams[i].param = owned;
        // [中文导读] [AllReduce逐行 S208] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S209] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 释放临时 srcCm（comm ctx 已有独立深拷贝副本）
    // [中文导读] [AllReduce逐行 S212] 调用 CostModelManager::FreeCostModel 完成当前参数所指的子步骤；本行实参为 CostModelManager::FreeCostModel(srcCm)。
    CostModelManager::FreeCostModel(srcCm);

    // 根据候选引擎过滤 costModel,再调用 algo 模块按 HCCL_ALGO 配置过滤
    // [中文导读] [AllReduce逐行 S215] 设置 std::vector<OpExecuteConfig> candidateEngines 为 GetEnginePriority(当前执行配置)；该值供下方当前分支使用。
    std::vector<OpExecuteConfig> candidateEngines = GetEnginePriority(param.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S216] 调用 FilterCmByEngine 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(FilterCmByEngine(*storedCm, candidateEngines));
    // [中文导读] [AllReduce逐行 S217] 设置 std::vector<std::string> candidatePrefixes 为 CandidateEnginesToPrefixes(candidateEngines)；该值供下方当前分支使用。
    std::vector<std::string> candidatePrefixes = CandidateEnginesToPrefixes(candidateEngines);
    // [中文导读] [AllReduce逐行 S218] 调用 FilterCmByHcclAlgo 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(FilterCmByHcclAlgo(comm, *storedCm, candidatePrefixes));

    // [中文导读] [AllReduce逐行 S220] 设置 cm 为 storedCm；该值供下方当前分支使用。
    cm = storedCm;

    // [中文导读] [AllReduce逐行 S222] 开始 HCCL_INFO 诊断输出，记录 InitCostModel 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] costModel initialized and stored in comm ctx, count=%d.", storedCm->count);
    // [中文导读] [AllReduce逐行 S223] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S224] 结束 InitCostModel 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S226] 定义 TunerEnrichCostTable 入口：把成本模型转换为当前数据量成本表，并在 tuner 已加载且表非空时允许插件改成本。
HcclResult SelectorEngine::TunerEnrichCostTable(
    // [中文导读] [AllReduce逐行 S227] 续接 TunerEnrichCostTable 的入口参数/基类初始化：HcclComm comm, CostModel* cm, CostTable& ct, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param)；引用参数按声明的 const 限制读写。
    HcclComm comm, CostModel* cm, CostTable& ct, TopoInfoWithNetLayerDetails* topoInfo, OpParam& param)
// [中文导读] [AllReduce逐行 S228] 进入 TunerEnrichCostTable 的实现作用域；把成本模型转换为当前数据量成本表，并在 tuner 已加载且表非空时允许插件改成本。
{
    // [中文导读] [AllReduce逐行 S229] 调用 CostTableManager::Global 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(CostTableManager::Global()->CostTableGen(*cm, ct, topoInfo, param));

    // [中文导读] [AllReduce逐行 S231] 分支条件为 ct.count 大于 0 且 HcclTunerIsLoaded(；成立进入本块，未成立继续后续分支。
    if (ct.count > 0 && HcclTunerIsLoaded()) {
        // tuner: Enrich 填 3D 名 + 调用插件改 cost
        // [中文导读] [AllReduce逐行 S233] 调用 AlgoNameMapper::Global 完成当前参数所指的子步骤；本行实参为 AlgoNameMapper::Global()->Enrich(ct.costs, ct.count)。
        AlgoNameMapper::Global()->Enrich(ct.costs, ct.count);
        // [中文导读] [AllReduce逐行 S234] 设置 tunerModified 为 false；该值供下方当前分支使用。
        bool tunerModified = false;
        // AllToAll(V/VC) 的 dataType 存在 all2AllVDataDes.sendType 中，而非 DataDes.dataType
        // [中文导读] [AllReduce逐行 S236] 设置 HcclDataType tunerDataType 为 输入元素类型；该值供下方当前分支使用。
        HcclDataType tunerDataType = param.DataDes.dataType;
        // [中文导读] [AllReduce逐行 S237] 分支条件为 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALL 或 param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLV；成立进入本块，未成立继续后续分支。
        if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALL || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
            // [中文导读] [AllReduce逐行 S238] 补充同一条件的 或者 子条件：param.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLVC。
            || param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) {
            // [中文导读] [AllReduce逐行 S239] 设置 tunerDataType 为 param.all2AllVDataDes.sendType；该值供下方当前分支使用。
            tunerDataType = param.all2AllVDataDes.sendType;
        // [中文导读] [AllReduce逐行 S240] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S241] 调用 HcclTunerCallGetCollInfo 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(HcclTunerCallGetCollInfo(
            // [中文导读] [AllReduce逐行 S242] 续接本次错误检查/子调用实参：comm, param.opType, 输入字节容量, tunerDataType, ct.costs, ct.count, &tunerModified；返回行为由所在完整宏决定。
            comm, param.opType, param.inputSize, tunerDataType, ct.costs, ct.count, &tunerModified));
        // [中文导读] [AllReduce逐行 S243] 分支条件为 tunerModified；成立进入本块，未成立继续后续分支。
        if (tunerModified) {
            // [中文导读] [AllReduce逐行 S244] 开始 HCCL_INFO 诊断输出，记录 TunerEnrichCostTable 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("[SelectorEngine] tuner modified cost table.");
        // [中文导读] [AllReduce逐行 S245] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S246] 开始 HCCL_INFO 诊断输出，记录 TunerEnrichCostTable 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("[SelectorEngine] tuner did not modify cost table, using CostModel selection.");
        // [中文导读] [AllReduce逐行 S247] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S248] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S249] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S250] 结束 TunerEnrichCostTable 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S252] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
HcclResult
// [中文导读] [AllReduce逐行 S253] 定义 Run 入口：选择器执行入口；ExecuteSelector 按优先级首个 MATCH，SelectorEngine 按成本模型/成本表选择。
SelectorEngine::Run(HcclComm comm, OpParam& param, TopoInfoWithNetLayerDetails* topoInfo, std::string& algName)
// [中文导读] [AllReduce逐行 S254] 进入 Run 的实现作用域；选择器执行入口；ExecuteSelector 按优先级首个 MATCH，SelectorEngine 按成本模型/成本表选择。
{
    // [中文导读] [AllReduce逐行 S255] 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S256] 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[SelectorEngine] Run start, opType=%d, opExecuteConfig=%d.", static_cast<int>(param.opType),
        // [中文导读] [AllReduce逐行 S257] 为 Run 的诊断/错误宏提供实参：static_cast<int>(当前执行配置，与前面的格式占位依次对应。
        static_cast<int>(param.opExecuteConfig));

    // [中文导读] [AllReduce逐行 S259] 分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY 且 AutoSelectorBase::IsRollBackAiv(param, topoInfo；成立进入本块，未成立继续后续分支。
    if (param.opExecuteConfig != OpExecuteConfig::AIV_ONLY && AutoSelectorBase::IsRollBackAiv(param, topoInfo)) {
        // [中文导读] [AllReduce逐行 S260] 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[SelectorEngine] Need to roll back AIV algo");
        // [中文导读] [AllReduce逐行 S261] 设置 当前执行配置 为 OpExecuteConfig::AIV_ONLY；该值供下方当前分支使用。
        param.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
    // [中文导读] [AllReduce逐行 S262] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // step 0: tuner 初始化（每通信域仅一次，独立于 costModel 引擎副本）
    // [中文导读] [AllReduce逐行 S265] 设置 tunerCtxPtr 为 nullptr；该值供下方当前分支使用。
    void* tunerCtxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S266] 设置 uint64_t tunerCtxSize 为 0；该值供下方当前分支使用。
    uint64_t tunerCtxSize = 0;
    // [中文导读] [AllReduce逐行 S267] 分支条件为 HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &tunerCtxPtr, &tunerCtxSize；成立进入本块，未成立继续后续分支。
    if (HcclEngineCtxGet(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, &tunerCtxPtr, &tunerCtxSize)
        // [中文导读] [AllReduce逐行 S268] 续接 Run 当前语句的具体实参/字段：不等于 成功状态) {；由其完整表达式完成参数组装、检查或结果写回。
        != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S269] 调用 HcclTunerInit 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(HcclTunerInit(comm, topoInfo));
        // [中文导读] [AllReduce逐行 S270] 调用 HcclEngineCtxCreate 完成当前参数所指的子步骤；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(HcclEngineCtxCreate(comm, TUNER_INIT_TAG, CommEngine::COMM_ENGINE_CPU, 1, &tunerCtxPtr));
    // [中文导读] [AllReduce逐行 S271] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // step 1: 从通信域 ctx 获取或初始化 costModel（按引擎区分 tag，回退时引擎变更会命中不同副本）
    // [中文导读] [AllReduce逐行 S274] 设置 CostModel* cm 为 nullptr；该值供下方当前分支使用。
    CostModel* cm = nullptr;
    // [中文导读] [AllReduce逐行 S275] 设置 ctxPtr 为 nullptr；该值供下方当前分支使用。
    void* ctxPtr = nullptr;
    // [中文导读] [AllReduce逐行 S276] 设置 uint64_t ctxSize 为 0；该值供下方当前分支使用。
    uint64_t ctxSize = 0;
    // [中文导读] [AllReduce逐行 S277] 设置 costModelTag 为 std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(当前执行配置)；该值供下方当前分支使用。
    std::string costModelTag = std::string(COST_MODEL_TAG) + "_" + ENGINE_STR_MAP.at(param.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S278] 分支条件为 HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) 等于 成功状态；成立进入本块，未成立继续后续分支。
    if (HcclEngineCtxGet(comm, costModelTag.c_str(), CommEngine::COMM_ENGINE_CPU, &ctxPtr, &ctxSize) == HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S279] 设置 cm 为 static_cast<CostModel*>(ctxPtr)；该值供下方当前分支使用。
        cm = static_cast<CostModel*>(ctxPtr);
        // [中文导读] [AllReduce逐行 S280] 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[SelectorEngine] costModel found in comm ctx, count=%d.", cm->count);
    // [中文导读] [AllReduce逐行 S281] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
    } else {
        // [中文导读] [AllReduce逐行 S282] 首次建立此通信域此引擎的成本模型；返回值非成功时立即从当前函数返回该错误。
        CHK_RET(InitCostModel(comm, topoInfo, param, cm));
    // [中文导读] [AllReduce逐行 S283] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // step 2: 生成 costTable 并调 tuner 改 cost
    // [中文导读] [AllReduce逐行 S286] 续接 Run 当前语句的具体实参/字段：CostTable ct{nullptr, 0}；由其完整表达式完成参数组装、检查或结果写回。
    CostTable ct{nullptr, 0};
    // [中文导读] [AllReduce逐行 S287] 建立本次数据量成本表并允许 tuner 改写；返回值非成功时立即从当前函数返回该错误。
    CHK_RET(TunerEnrichCostTable(comm, cm, ct, topoInfo, param));

    // step 3: min(ct)
    // [中文导读] [AllReduce逐行 S290] 设置 ret 为 SelectMinCost(ct, param, algName)；该值供下方当前分支使用。
    HcclResult ret = SelectMinCost(ct, param, algName);

    // [中文导读] [AllReduce逐行 S292] 续接 Run 当前语句的具体实参/字段：delete[] ct.costs；由其完整表达式完成参数组装、检查或结果写回。
    delete[] ct.costs;
    // [中文导读] [AllReduce逐行 S293] 设置 ct.costs 为 nullptr；该值供下方当前分支使用。
    ct.costs = nullptr;
    // [中文导读] [AllReduce逐行 S294] 设置 ct.count 为 0；该值供下方当前分支使用。
    ct.count = 0;

    // [中文导读] [AllReduce逐行 S296] 分支条件为 ret 不等于 成功状态；成立进入本块，未成立继续后续分支。
    if (ret != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S297] 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[SelectorEngine] Run failed, no algorithm selected.");
        // [中文导读] [AllReduce逐行 S298] 分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。
        if (param.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {
            // [中文导读] [AllReduce逐行 S299] 调用 LogAivOnlyNotMatch 完成当前参数所指的子步骤；本行实参为 LogAivOnlyNotMatch(param, topoInfo)。
            LogAivOnlyNotMatch(param, topoInfo);
        // [中文导读] [AllReduce逐行 S300] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S301] 直接返回 ret，调用者取得本分支结果。
        return ret;
    // [中文导读] [AllReduce逐行 S302] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S304] 调用 LogSelectedAlgo 完成当前参数所指的子步骤；本行实参为 LogSelectedAlgo(param, topoInfo, algName)。
    LogSelectedAlgo(param, topoInfo, algName);

    // [中文导读] [AllReduce逐行 S306] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S307] 结束 Run 实现；其返回状态或已写回字段由调用者接收。
}

void SelectorEngine::LogSelectedAlgo(
    const OpParam& param, const TopoInfoWithNetLayerDetails* topoInfo, const std::string& algName)
{
    HCCL_INFO(
        "[SelectorEngine] The opExecuteConfig is %s, the selected algo type is %s",
        ENGINE_STR_MAP.at(param.opExecuteConfig), algName.c_str());

    HCCL_CONFIG_INFO(
        HCCL_ALG,
        "op[%s] algName[%s] engine[%s] executor[%s] templates[%s] "
        "opMode[%d] deterministic[%u] isCapture[%d] "
        "dataSize[%llu] dataType[%s] reduceType[%s] root[%u] "
        "userRank[%u] rankSize[%u] serverNum[%u] superPodNum[%u] "
        "topoLevelNums[%u] level0Topo[%d] level0MeshType[%d] level0PcieMix[%d] "
        "deviceType[%d] commName[%s] enableDetour[%d] symMem[%d]",
        HcclCMDTypeToString(param.opType).c_str(), algName.c_str(), ENGINE_STR_MAP.at(param.opExecuteConfig),
        QueryExecutorName(algName).c_str(), QueryTemplateInfo(algName).c_str(), static_cast<int>(param.opMode),
        GetExternalInputHcclDeterministic(), static_cast<int>(param.isCapture), param.inputSize,
        GetDataTypeEnumStr(param.DataDes.dataType).c_str(), GetReduceOpEnumStr(param.reduceType).c_str(), param.root,
        topoInfo->userRank, topoInfo->userRankSize, topoInfo->serverNum, topoInfo->superPodNum, topoInfo->topoLevelNums,
        static_cast<int>(topoInfo->level0Topo), static_cast<int>(topoInfo->level0MeshType),
        static_cast<int>(topoInfo->level0PcieMix), static_cast<int>(param.deviceType), param.commName,
        static_cast<int>(param.enableDetour), static_cast<int>(param.supportSymmetricMemory));
}

std::string SelectorEngine::QueryTemplateInfo(const std::string& algName)
{
    std::string templateInfo;
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return templateInfo;
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        if (allAlgos->algElements[i].algName != nullptr && algName == allAlgos->algElements[i].algName) {
            for (int t = 0; t < allAlgos->algElements[i].templateNum; ++t) {
                if (t > 0) {
                    templateInfo += ",";
                }
                templateInfo += allAlgos->algElements[i].templateName[t];
            }
            break;
        }
    }
    return templateInfo;
}

std::string SelectorEngine::QueryExecutorName(const std::string& algName)
{
    AllAlgos* allAlgos = GetAllAlgos();
    if (allAlgos == nullptr) {
        return "";
    }
    for (int i = 0; i < allAlgos->count; ++i) {
        if (allAlgos->algElements[i].algName != nullptr && algName == allAlgos->algElements[i].algName) {
            return allAlgos->algElements[i].executorName != nullptr ? allAlgos->algElements[i].executorName : "";
        }
    }
    return "";
}

// [中文导读] [AllReduce逐行 S369] 定义 SelectMinCost 入口：忽略空算法名和负成本，保留首次最小项；同值项仅日志提示而不改首个胜者，写回算法名及执行配置。
HcclResult SelectorEngine::SelectMinCost(const CostTable& ct, OpParam& param, std::string& algName)
// [中文导读] [AllReduce逐行 S370] 进入 SelectMinCost 的实现作用域；忽略空算法名和负成本，保留首次最小项；同值项仅日志提示而不改首个胜者，写回算法名及执行配置。
{
    // [中文导读] [AllReduce逐行 S371] 分支条件为 ct.count 不超过 0；成立进入本块，未成立继续后续分支。
    if (ct.count <= 0) {
        // [中文导读] [AllReduce逐行 S372] 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S373] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[SelectorEngine] SelectMinCost: costTable is empty, opType=%d, dataSize=%llu.",
            // [中文导读] [AllReduce逐行 S374] 为 SelectMinCost 的诊断/错误宏提供实参：static_cast<int>(param.opType), 输入字节容量，与前面的格式占位依次对应。
            static_cast<int>(param.opType), param.inputSize);
        // [中文导读] [AllReduce逐行 S375] 终止当前函数并向上返回 不支持错误；调用者 CHK_RET 决定是否继续向上传播。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S376] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // 遍历找最小 cost 并打印 costTable 明细, 格式: | idx | algName | cost |
    // [中文导读] [AllReduce逐行 S379] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S380] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[SelectorEngine] SelectMinCost: costTable count=%d, opType=%d, dataSize=%llu.", ct.count,
        // [中文导读] [AllReduce逐行 S381] 为 SelectMinCost 的诊断/错误宏提供实参：static_cast<int>(param.opType), 输入字节容量，与前面的格式占位依次对应。
        static_cast<int>(param.opType), param.inputSize);
    // [中文导读] [AllReduce逐行 S382] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] "
              // [中文导读] [AllReduce逐行 S383] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
              "+-----+--------------------------------------------------+--------------+");
    // [中文导读] [AllReduce逐行 S384] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] | idx | algName                                          | cost         |");
    // [中文导读] [AllReduce逐行 S385] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] "
              // [中文导读] [AllReduce逐行 S386] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
              "+-----+--------------------------------------------------+--------------+");
    // [中文导读] [AllReduce逐行 S387] 设置 当前最低成本项索引 为 -1；该值供下方当前分支使用。
    int minIdx = -1;
    // [中文导读] [AllReduce逐行 S388] 设置 float 当前最低候选成本 为 0.0f；该值供下方当前分支使用。
    float minCost = 0.0f;
    // [中文导读] [AllReduce逐行 S389] 建立本阶段局部对象 std::vector<std::string> tiedAlgos，供 SelectMinCost 下方参数组装和子调用使用。
    std::vector<std::string> tiedAlgos;
    // [中文导读] [AllReduce逐行 S390] 逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (int i = 0; i 小于 ct.count; ++i。
    for (int i = 0; i < ct.count; ++i) {
        // [中文导读] [AllReduce逐行 S391] 设置 const char* name 为 ct.costs[i].algName；该值供下方当前分支使用。
        const char* name = ct.costs[i].algName;
        // [中文导读] [AllReduce逐行 S392] 设置 float cost 为 ct.costs[i].cost；该值供下方当前分支使用。
        float cost = ct.costs[i].cost;
        // [中文导读] [AllReduce逐行 S393] 建立本阶段局部对象 std::string nameStr = name 不等于 nullptr ? name : "-"，供 SelectMinCost 下方参数组装和子调用使用。
        std::string nameStr = name != nullptr ? name : "-";
        // [中文导读] [AllReduce逐行 S394] 续接 SelectMinCost 当前语句的具体实参/字段：char costBuf[32]；由其完整表达式完成参数组装、检查或结果写回。
        char costBuf[32];
        // [中文导读] [AllReduce逐行 S395] 建立本阶段局部对象 const char* fmt = (cost 至少 0.0f 且 cost 小于 1.0f) ? "%.6f" : "%.2f"，供 SelectMinCost 下方参数组装和子调用使用。
        const char* fmt = (cost >= 0.0f && cost < 1.0f) ? "%.6f" : "%.2f";
        // [中文导读] [AllReduce逐行 S396] 设置 costRet 为 sprintf_s(costBuf, sizeof(costBuf), fmt, cost)；该值供下方当前分支使用。
        int costRet = sprintf_s(costBuf, sizeof(costBuf), fmt, cost);
        // [中文导读] [AllReduce逐行 S397] 分支条件为 costRet 小于 0；成立进入本块，未成立继续后续分支。
        if (costRet < 0) {
            // [中文导读] [AllReduce逐行 S398] 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR("[SelectorEngine] SelectMinCost: sprintf_s failed.");
            // [中文导读] [AllReduce逐行 S399] 终止当前函数并向上返回 内部错误；调用者 CHK_RET 决定是否继续向上传播。
            return HCCL_E_INTERNAL;
        // [中文导读] [AllReduce逐行 S400] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S401] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[SelectorEngine] | %3d | %-48s | %12s |", i, nameStr.substr(0, 48).c_str(), costBuf);

        // [中文导读] [AllReduce逐行 S403] 分支条件为 name 等于 nullptr 或 cost 小于 0.0f；成立进入本块，未成立继续后续分支。
        if (name == nullptr || cost < 0.0f) {
            // [中文导读] [AllReduce逐行 S404] 跳过当前遍历项的剩余步骤，直接处理下一项。
            continue;
        // [中文导读] [AllReduce逐行 S405] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S406] 分支条件为 当前最低成本项索引 等于 -1 或 cost 小于 当前最低候选成本；成立进入本块，未成立继续后续分支。
        if (minIdx == -1 || cost < minCost) {
            // [中文导读] [AllReduce逐行 S407] 设置 当前最低成本项索引 为 i；该值供下方当前分支使用。
            minIdx = i;
            // [中文导读] [AllReduce逐行 S408] 设置 当前最低候选成本 为 cost；该值供下方当前分支使用。
            minCost = cost;
            // [中文导读] [AllReduce逐行 S409] 对 tiedAlgos 清空 ，准备或更新本阶段列表。
            tiedAlgos.clear();
            // [中文导读] [AllReduce逐行 S410] 对 tiedAlgos 就地追加 name，准备或更新本阶段列表。
            tiedAlgos.emplace_back(name);
        // [中文导读] [AllReduce逐行 S411] 分支条件为 cost 等于 当前最低候选成本；成立进入本块，未成立继续后续分支。
        } else if (cost == minCost) {
            // [中文导读] [AllReduce逐行 S412] 对 tiedAlgos 就地追加 name，准备或更新本阶段列表。
            tiedAlgos.emplace_back(name);
        // [中文导读] [AllReduce逐行 S413] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S414] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S415] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[SelectorEngine] "
              // [中文导读] [AllReduce逐行 S416] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
              "+-----+--------------------------------------------------+----------+--------------+----------+");

    // [中文导读] [AllReduce逐行 S418] 分支条件为 当前最低成本项索引 小于 0；成立进入本块，未成立继续后续分支。
    if (minIdx < 0) {
        // [中文导读] [AllReduce逐行 S419] 开始 HCCL_ERROR 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S420] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[SelectorEngine] SelectMinCost: no valid algorithm found, expansionMode=%d, costTable count=%d.",
            // [中文导读] [AllReduce逐行 S421] 为 SelectMinCost 的诊断/错误宏提供实参：static_cast<int>(param.commOpExpansionMode), ct.count，与前面的格式占位依次对应。
            static_cast<int>(param.commOpExpansionMode), ct.count);
        // [中文导读] [AllReduce逐行 S422] 终止当前函数并向上返回 不支持错误；调用者 CHK_RET 决定是否继续向上传播。
        return HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S423] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S425] 分支条件为 tiedAlgos.size() 大于 1；成立进入本块，未成立继续后续分支。
    if (tiedAlgos.size() > 1) {
        // [中文导读] [AllReduce逐行 S426] 建立本阶段局部对象 std::string algoNames，供 SelectMinCost 下方参数组装和子调用使用。
        std::string algoNames;
        // [中文导读] [AllReduce逐行 S427] 逐项比较当前数据量成本表中的算法成本；边界/迭代规则为 (size_t i = 0; i 小于 tiedAlgos.size(); ++i。
        for (size_t i = 0; i < tiedAlgos.size(); ++i) {
            // [中文导读] [AllReduce逐行 S428] 分支条件为 i 大于 0；成立进入本块，未成立继续后续分支。
            if (i > 0) {
                // [中文导读] [AllReduce逐行 S429] 设置 algoNames + 为 ", "；该值供下方当前分支使用。
                algoNames += ", ";
            // [中文导读] [AllReduce逐行 S430] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S431] 设置 algoNames + 为 tiedAlgos[i]；该值供下方当前分支使用。
            algoNames += tiedAlgos[i];
        // [中文导读] [AllReduce逐行 S432] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S433] 开始 HCCL_WARNING 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING(
            // [中文导读] [AllReduce逐行 S434] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[SelectorEngine] multiple algos with same cost=%f: [%s], selecting %s.", minCost, algoNames.c_str(),
            // [中文导读] [AllReduce逐行 S435] 为 SelectMinCost 的诊断/错误宏提供实参：tiedAlgos[0].c_str(，与前面的格式占位依次对应。
            tiedAlgos[0].c_str());
    // [中文导读] [AllReduce逐行 S436] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S438] 设置 algName 为 ct.costs[当前最低成本项索引].algName；该值供下方当前分支使用。
    algName = ct.costs[minIdx].algName;
    // [中文导读] [AllReduce逐行 S439] 设置 当前执行配置 为 GetEngineByAlgName(algName)；该值供下方当前分支使用。
    param.opExecuteConfig = GetEngineByAlgName(algName);
    // [中文导读] [AllReduce逐行 S440] 开始 HCCL_INFO 诊断输出，记录 SelectMinCost 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S441] 续接 SelectMinCost 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[SelectorEngine] SelectMinCost: selected algName=%s, engine=%s, cost=%f.", algName.c_str(),
        // [中文导读] [AllReduce逐行 S442] 为 SelectMinCost 的诊断/错误宏提供实参：ENGINE_STR_MAP.at(当前执行配置), 当前最低候选成本，与前面的格式占位依次对应。
        ENGINE_STR_MAP.at(param.opExecuteConfig), minCost);
    // [中文导读] [AllReduce逐行 S443] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S444] 结束 SelectMinCost 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
