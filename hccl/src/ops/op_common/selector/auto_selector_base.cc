/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "auto_selector_base.h"
#include "selector_registry.h"
#include "op_common.h"

namespace ops_hccl {

// [中文导读] [AllReduce逐行 S17] 声明返回类型 SelectorStatus，分别由错误码传播或候选匹配协议解释。
SelectorStatus
// [中文导读] [AllReduce逐行 S18] 定义 Select 入口：AutoSelectorBase 的旧引擎链：HostDPU 独占优先，CCU_MS→CCU_SCHED→CCU_FAIL，AIV 和 Stars/AICPU 分别匹配。
AutoSelectorBase::Select(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S19] 进入 Select 的实现作用域；AutoSelectorBase 的旧引擎链：HostDPU 独占优先，CCU_MS→CCU_SCHED→CCU_FAIL，AIV 和 Stars/AICPU 分别匹配。
{
    // [中文导读] [AllReduce逐行 S20] 开始 HCCL_DEBUG 诊断输出，记录 Select 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[AutoSelectorBase][%s] start, OpExecuteConfig is %d.", __func__, opParam.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S21] 设置 std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap 为 GetExternalInputHcclAlgoConfigAllType()；该值供下方当前分支使用。
    std::map<HcclCMDType, std::vector<HcclAlgoType>> configAlgMap = GetExternalInputHcclAlgoConfigAllType();
    // [中文导读] [AllReduce逐行 S22] 设置 ret 为 当前选择器不匹配；该值供下方当前分支使用。
    SelectorStatus ret = SelectorStatus::NOT_MATCH;
    // [中文导读] [AllReduce逐行 S23] 设置 hostDPUOnly 为 false；该值供下方当前分支使用。
    bool hostDPUOnly = false;
    // [中文导读] [AllReduce逐行 S24] 分支条件为 (CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) 等于 成功状态) 且 hostDPUOnly；成立进入本块，未成立继续后续分支。
    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        // [中文导读] [AllReduce逐行 S25] 设置 当前执行配置 为 OpExecuteConfig::HOSTCPU；该值供下方当前分支使用。
        opParam.opExecuteConfig = OpExecuteConfig::HOSTCPU;
        // [中文导读] [AllReduce逐行 S26] 设置 实际通信引擎 为 CommEngine::COMM_ENGINE_CPU；该值供下方当前分支使用。
        opParam.engine = CommEngine::COMM_ENGINE_CPU;
        // [中文导读] [AllReduce逐行 S27] 直接返回 尝试 HostDPU 专用候选 的结果，调用者取得本分支结果。
        return SelectDPUAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    // [中文导读] [AllReduce逐行 S28] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S29] 分支条件为 当前执行配置 等于 OpExecuteConfig::CCU_MS；成立进入本块，未成立继续后续分支。
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_MS) {
        // [中文导读] [AllReduce逐行 S30] 设置 ret 为 SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, 算法名输出参数)；该值供下方当前分支使用。
        ret = SelectCcuMsAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        // [中文导读] [AllReduce逐行 S31] 分支条件为 ret 等于 当前选择器不匹配；成立进入本块，未成立继续后续分支。
        if (ret == SelectorStatus::NOT_MATCH) {
            // [中文导读] [AllReduce逐行 S32] 设置 当前执行配置 为 OpExecuteConfig::CCU_SCHED；该值供下方当前分支使用。
            opParam.opExecuteConfig = OpExecuteConfig::CCU_SCHED;
        // [中文导读] [AllReduce逐行 S33] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S34] 直接返回 ret，调用者取得本分支结果。
            return ret;
        // [中文导读] [AllReduce逐行 S35] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S36] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S37] 分支条件为 当前执行配置 等于 OpExecuteConfig::CCU_SCHED；成立进入本块，未成立继续后续分支。
    if (opParam.opExecuteConfig == OpExecuteConfig::CCU_SCHED) {
        // [中文导读] [AllReduce逐行 S38] 设置 ret 为 SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, 算法名输出参数)；该值供下方当前分支使用。
        ret = SelectCcuScheduleAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        // [中文导读] [AllReduce逐行 S39] 分支条件为 ret 等于 当前选择器不匹配；成立进入本块，未成立继续后续分支。
        if (ret == SelectorStatus::NOT_MATCH) {
            // [中文导读] [AllReduce逐行 S40] 设置 当前执行配置 为 OpExecuteConfig::CCU_FAIL；该值供下方当前分支使用。
            opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        // [中文导读] [AllReduce逐行 S41] 前面的条件未满足时进入本分支，执行此处替代算法/协议/校验路径。
        } else {
            // [中文导读] [AllReduce逐行 S42] 直接返回 ret，调用者取得本分支结果。
            return ret;
        // [中文导读] [AllReduce逐行 S43] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S44] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S45] 分支条件为 ProcessAivConfig(opParam, topoInfo, configAlgMap, 算法名输出参数, ret；成立进入本块，未成立继续后续分支。
    if (ProcessAivConfig(opParam, topoInfo, configAlgMap, selectAlgName, ret)) {
        // [中文导读] [AllReduce逐行 S46] 直接返回 ret，调用者取得本分支结果。
        return ret;
    // [中文导读] [AllReduce逐行 S47] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S48] 分支条件为 IsStarsState(当前执行配置；成立进入本块，未成立继续后续分支。
    if (IsStarsState(opParam.opExecuteConfig)) {
        // 需要回退AIV的场景下，回退AIV算法
        // [中文导读] [AllReduce逐行 S50] 分支条件为 IsRollBackAiv(opParam, topoInfo；成立进入本块，未成立继续后续分支。
        if (IsRollBackAiv(opParam, topoInfo)) {
            // [中文导读] [AllReduce逐行 S51] 开始 HCCL_INFO 诊断输出，记录 Select 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO("[Algo][AutoSelectorBase] Need to roll back AIV algo");
            // [中文导读] [AllReduce逐行 S52] 设置 当前执行配置 为 OpExecuteConfig::AIV_ONLY；该值供下方当前分支使用。
            opParam.opExecuteConfig = OpExecuteConfig::AIV_ONLY;
            // [中文导读] [AllReduce逐行 S53] 显式标记 ProcessAivConfig(opParam, topoInfo, configAlgMap, 算法名输出参数, ret 在此兼容/default 分支未使用，避免编译器未使用参数警告。
            (void)ProcessAivConfig(opParam, topoInfo, configAlgMap, selectAlgName, ret);
            // [中文导读] [AllReduce逐行 S54] 开始 HCCL_INFO 诊断输出，记录 Select 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO(
                // [中文导读] [AllReduce逐行 S55] 续接 Select 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.", selectAlgName.c_str(),
                // [中文导读] [AllReduce逐行 S56] 为 Select 的诊断/错误宏提供实参：当前执行配置，与前面的格式占位依次对应。
                opParam.opExecuteConfig);
            // [中文导读] [AllReduce逐行 S57] 直接返回 ret，调用者取得本分支结果。
            return ret;
        // [中文导读] [AllReduce逐行 S58] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S59] 设置 ret 为 SelectAicpuAlgo(topoInfo, opParam, configAlgMap, 算法名输出参数)；该值供下方当前分支使用。
        ret = SelectAicpuAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
        // [中文导读] [AllReduce逐行 S60] 分支条件为 ret 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。
        if (ret == SelectorStatus::MATCH) {
            // [中文导读] [AllReduce逐行 S61] 设置 当前执行配置 为 OpExecuteConfig::AICPU_TS；该值供下方当前分支使用。
            opParam.opExecuteConfig = OpExecuteConfig::AICPU_TS;
        // [中文导读] [AllReduce逐行 S62] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S63] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S64] 开始 HCCL_INFO 诊断输出，记录 Select 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S65] 续接 Select 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[Algo][AutoSelectorBase] The selected algo is %s, OpExecuteConfig is %d.", selectAlgName.c_str(),
        // [中文导读] [AllReduce逐行 S66] 为 Select 的诊断/错误宏提供实参：当前执行配置，与前面的格式占位依次对应。
        opParam.opExecuteConfig);
    // [中文导读] [AllReduce逐行 S67] 直接返回 ret，调用者取得本分支结果。
    return ret;
// [中文导读] [AllReduce逐行 S68] 结束 Select 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S70] 定义 IsRollBackAiv 入口：检查特定 PCIe 混合 Mesh/P2P 场景是否需改用 AIV；普通 FP32 Mesh 示例通常不命中。
bool AutoSelectorBase::IsRollBackAiv(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo)
// [中文导读] [AllReduce逐行 S71] 进入 IsRollBackAiv 的实现作用域；检查特定 PCIe 混合 Mesh/P2P 场景是否需改用 AIV；普通 FP32 Mesh 示例通常不命中。
{
    // Mesh类算法场景，ATU资源受限，切换为AIV算法
    // [中文导读] [AllReduce逐行 S73] 续接 IsRollBackAiv 当前语句的具体实参/字段：bool isAllToAllOps = opParam.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALL；由其完整表达式完成参数组装、检查或结果写回。
    bool isAllToAllOps = opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALL
                         // [中文导读] [AllReduce逐行 S74] 补充同一条件的 或者 子条件：opParam.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLV。
                         || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLV
                         // [中文导读] [AllReduce逐行 S75] 补充同一条件的 或者 子条件：opParam.opType 等于 HcclCMDType::HCCL_CMD_ALLTOALLVC。
                         || opParam.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    // [中文导读] [AllReduce逐行 S76] 续接 IsRollBackAiv 当前语句的具体实参/字段：bool isInt64ReduceOps = 输入元素类型 等于 HcclDataType::有符号 64 位整数；由其完整表达式完成参数组装、检查或结果写回。
    bool isInt64ReduceOps = opParam.DataDes.dataType == HcclDataType::HCCL_DATA_TYPE_INT64
                            // [中文导读] [AllReduce逐行 S77] 补充同一条件的 并且 子条件：(opParam.opType 等于 HcclCMDType::HCCL_CMD_ALLREDUCE。
                            && (opParam.opType == HcclCMDType::HCCL_CMD_ALLREDUCE
                                // [中文导读] [AllReduce逐行 S78] 补充同一条件的 或者 子条件：opParam.opType 等于 HcclCMDType::HCCL_CMD_REDUCE_SCATTER。
                                || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE_SCATTER
                                // [中文导读] [AllReduce逐行 S79] 补充同一条件的 或者 子条件：opParam.opType 等于 HcclCMDType::HCCL_CMD_REDUCE。
                                || opParam.opType == HcclCMDType::HCCL_CMD_REDUCE);
    // [中文导读] [AllReduce逐行 S80] 续接 IsRollBackAiv 当前语句的具体实参/字段：bool isPcieMeshScene；由其完整表达式完成参数组装、检查或结果写回。
    bool isPcieMeshScene
        // [中文导读] [AllReduce逐行 S81] 设置  为 第零层 PCIe 混合标志 且 topoInfo->level0BigClosRange 且 (isAllToAllOps 或 isInt64ReduceOps)；该值供下方当前分支使用。
        = topoInfo->level0PcieMix && topoInfo->level0BigClosRange && (isAllToAllOps || isInt64ReduceOps);

    // P2P算子场景，ATU资源受限，使用PCIE链路的切换为AIV算法
    // [中文导读] [AllReduce逐行 S84] 续接 IsRollBackAiv 当前语句的具体实参/字段：bool isP2pOps = (opParam.opType 等于 HcclCMDType::HCCL_CMD_SEND) 或 (opParam.opType 等于 HcclCMDType::HCCL_CMD_RECEIVE)；由其完整表达式完成参数组装、检查或结果写回。
    bool isP2pOps = (opParam.opType == HcclCMDType::HCCL_CMD_SEND) || (opParam.opType == HcclCMDType::HCCL_CMD_RECEIVE);
    // [中文导读] [AllReduce逐行 S85] 设置 isLayer0AllConnetedWithMesh 为 IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH)；该值供下方当前分支使用。
    bool isLayer0AllConnetedWithMesh = IsLayerAllConnetedWithTopo(topoInfo, 0, CommTopo::COMM_TOPO_1DMESH);
    // [中文导读] [AllReduce逐行 S86] 续接 IsRollBackAiv 当前语句的具体实参/字段：bool isPcieP2pScene；由其完整表达式完成参数组装、检查或结果写回。
    bool isPcieP2pScene
        // [中文导读] [AllReduce逐行 S87] 续接 IsRollBackAiv 当前语句的具体实参/字段：= 第零层 PCIe 混合标志 且 topoInfo->serverNum 等于 1 且 isP2pOps 且 !isLayer0AllConnetedWithMesh；由其完整表达式完成参数组装、检查或结果写回。
        = topoInfo->level0PcieMix && topoInfo->serverNum == 1 && isP2pOps && !isLayer0AllConnetedWithMesh;

    // [中文导读] [AllReduce逐行 S89] 设置 isRollBackAiv 为 false；该值供下方当前分支使用。
    bool isRollBackAiv = false;
    // [中文导读] [AllReduce逐行 S90] 分支条件为 isPcieMeshScene；成立进入本块，未成立继续后续分支。
    if (isPcieMeshScene) {
        // [中文导读] [AllReduce逐行 S91] 开始 HCCL_INFO 诊断输出，记录 IsRollBackAiv 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S92] 续接 IsRollBackAiv 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AutoSelectorBase] Need to rollback aiv, isPcieMeshScene[%d]: isAllToAllOps[%d] isInt64ReduceOps[%d]",
            // [中文导读] [AllReduce逐行 S93] 为 IsRollBackAiv 的诊断/错误宏提供实参：isPcieMeshScene, isAllToAllOps, isInt64ReduceOps，与前面的格式占位依次对应。
            isPcieMeshScene, isAllToAllOps, isInt64ReduceOps);
        // [中文导读] [AllReduce逐行 S94] 设置 isRollBackAiv 为 true；该值供下方当前分支使用。
        isRollBackAiv = true;
    // [中文导读] [AllReduce逐行 S95] 分支条件为 isPcieP2pScene；成立进入本块，未成立继续后续分支。
    } else if (isPcieP2pScene) {
        // [中文导读] [AllReduce逐行 S96] 开始 HCCL_INFO 诊断输出，记录 IsRollBackAiv 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S97] 续接 IsRollBackAiv 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AutoSelectorBase] Need to rollback aiv, isPcieP2pScene[%d]: isP2pOps[%d] isLayer0AllConnetedWithMesh[%u]",
            // [中文导读] [AllReduce逐行 S98] 为 IsRollBackAiv 的诊断/错误宏提供实参：isPcieP2pScene, isP2pOps, isLayer0AllConnetedWithMesh，与前面的格式占位依次对应。
            isPcieP2pScene, isP2pOps, isLayer0AllConnetedWithMesh);
        // [中文导读] [AllReduce逐行 S99] 设置 isRollBackAiv 为 true；该值供下方当前分支使用。
        isRollBackAiv = true;
    // [中文导读] [AllReduce逐行 S100] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S101] 直接返回 isRollBackAiv，调用者取得本分支结果。
    return isRollBackAiv;
// [中文导读] [AllReduce逐行 S102] 结束 IsRollBackAiv 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S104] 定义 IsStarsState 入口：识别可进入旧 AICPU/HostCPU_TS 选择的执行配置，包括 CCU_FAIL。
bool AutoSelectorBase::IsStarsState(const OpExecuteConfig& opExecuteConfig) const
// [中文导读] [AllReduce逐行 S105] 进入 IsStarsState 的实现作用域；识别可进入旧 AICPU/HostCPU_TS 选择的执行配置，包括 CCU_FAIL。
{
    // [中文导读] [AllReduce逐行 S106] 直接返回 (，调用者取得本分支结果。
    return (
        // [中文导读] [AllReduce逐行 S107] 续接 IsStarsState 当前语句的具体实参/字段：opExecuteConfig 等于 OpExecuteConfig::AICPU_TS 或 opExecuteConfig 等于 OpExecuteConfig::HOSTCPU_TS；由其完整表达式完成参数组装、检查或结果写回。
        opExecuteConfig == OpExecuteConfig::AICPU_TS || opExecuteConfig == OpExecuteConfig::HOSTCPU_TS
        // [中文导读] [AllReduce逐行 S108] 补充同一条件的 或者 子条件：opExecuteConfig 等于 OpExecuteConfig::CCU_FAIL。
        || opExecuteConfig == OpExecuteConfig::CCU_FAIL);
// [中文导读] [AllReduce逐行 S109] 结束 IsStarsState 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S111] 定义 IsDefaultAlg 入口：判断显式算法配置是否 DEFAULT 或 NA。
bool AutoSelectorBase::IsDefaultAlg(const HcclAlgoType algoType) const
// [中文导读] [AllReduce逐行 S112] 进入 IsDefaultAlg 的实现作用域；判断显式算法配置是否 DEFAULT 或 NA。
{
    // [中文导读] [AllReduce逐行 S113] 直接返回 (algoType 等于 HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) 或 (algoType 等于 HcclAlgoType::HCCL_ALGO_TYPE_NA)，调用者取得本分支结果。
    return (algoType == HcclAlgoType::HCCL_ALGO_TYPE_DEFAULT) || (algoType == HcclAlgoType::HCCL_ALGO_TYPE_NA);
// [中文导读] [AllReduce逐行 S114] 结束 IsDefaultAlg 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S116] 调用 AutoSelectorBase::IsSmallData 完成当前参数所指的子步骤；本行实参为 bool AutoSelectorBase::IsSmallData(const u64 本 Rank 数据字节数) const { return 本 Rank 数据字节数 小于 SMALL_COUNT_512KB; }。
bool AutoSelectorBase::IsSmallData(const u64 dataSize) const { return dataSize < SMALL_COUNT_512KB; }

// [中文导读] [AllReduce逐行 S118] 调用 AutoSelectorBase::IsLargeData 完成当前参数所指的子步骤；本行实参为 bool AutoSelectorBase::IsLargeData(const u64 本 Rank 数据字节数) const { return 本 Rank 数据字节数 至少 LARGE_COUNT_1024KB; }。
bool AutoSelectorBase::IsLargeData(const u64 dataSize) const { return dataSize >= LARGE_COUNT_1024KB; }

// [中文导读] [AllReduce逐行 S120] 定义 IsSmallDataCCU 入口：CCU 公共并行阈值判断；rankSize 为零仅警告，判定仍以数据阈值为准。
bool AutoSelectorBase::IsSmallDataCCU(const u64 dataSize, const u64 rankSize) const
// [中文导读] [AllReduce逐行 S121] 进入 IsSmallDataCCU 的实现作用域；CCU 公共并行阈值判断；rankSize 为零仅警告，判定仍以数据阈值为准。
{
    // [中文导读] [AllReduce逐行 S122] 分支条件为 rankSize 等于 0；成立进入本块，未成立继续后续分支。
    if (rankSize == 0) {
        // [中文导读] [AllReduce逐行 S123] 开始 HCCL_WARNING 诊断输出，记录 IsSmallDataCCU 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING("the selector is not set RankSize");
    // [中文导读] [AllReduce逐行 S124] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S125] 直接返回 (本 Rank 数据字节数 不超过 CCU_PARALLEL_MAX_DATA_SIZE) ? true : false，调用者取得本分支结果。
    return (dataSize <= CCU_PARALLEL_MAX_DATA_SIZE) ? true : false;
// [中文导读] [AllReduce逐行 S126] 结束 IsSmallDataCCU 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S128] 定义 CalcFrameNum 入口：用 level0 实例规模的最大公约数推算框数。
u32 AutoSelectorBase::CalcFrameNum(const TopoInfoWithNetLayerDetails* topoInfo)
// [中文导读] [AllReduce逐行 S129] 进入 CalcFrameNum 的实现作用域；用 level0 实例规模的最大公约数推算框数。
{
    // [中文导读] [AllReduce逐行 S130] 设置 frameNum 为 0；该值供下方当前分支使用。
    u32 frameNum = 0;
    // [中文导读] [AllReduce逐行 S131] 分支条件为 topoInfo->topoLevelNums 不超过 1 或 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].empty(；成立进入本块，未成立继续后续分支。
    if (topoInfo->topoLevelNums <= 1 || topoInfo->netLayerDetails.instSizeListOfLayer[0].empty()) {
        // [中文导读] [AllReduce逐行 S132] 直接返回 frameNum，调用者取得本分支结果。
        return frameNum;
    // [中文导读] [AllReduce逐行 S133] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S134] 设置 gcd 为 topoInfo->当前网络层编号Details.instSizeListOfLayer[0][0]；该值供下方当前分支使用。
    u32 gcd = topoInfo->netLayerDetails.instSizeListOfLayer[0][0];
    // [中文导读] [AllReduce逐行 S135] 在 CalcFrameNum 中遍历 (size_t i = 1; i 小于 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].size(); ++i 指定的集合或索引区间；边界/迭代规则为 (size_t i = 1; i 小于 topoInfo->当前网络层编号Details.instSizeListOfLayer[0].size(); ++i。
    for (size_t i = 1; i < topoInfo->netLayerDetails.instSizeListOfLayer[0].size(); ++i) {
        // [中文导读] [AllReduce逐行 S136] 设置 a 为 gcd；该值供下方当前分支使用。
        u32 a = gcd;
        // [中文导读] [AllReduce逐行 S137] 设置 b 为 topoInfo->当前网络层编号Details.instSizeListOfLayer[0][i]；该值供下方当前分支使用。
        u32 b = topoInfo->netLayerDetails.instSizeListOfLayer[0][i];
        // [中文导读] [AllReduce逐行 S138] 在 (b 不等于 0 条件下继续重复当前处理。
        while (b != 0) {
            // [中文导读] [AllReduce逐行 S139] 设置 r 为 a % b；该值供下方当前分支使用。
            u32 r = a % b;
            // [中文导读] [AllReduce逐行 S140] 设置 a 为 b；该值供下方当前分支使用。
            a = b;
            // [中文导读] [AllReduce逐行 S141] 设置 b 为 r；该值供下方当前分支使用。
            b = r;
        // [中文导读] [AllReduce逐行 S142] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S143] 设置 gcd 为 a；该值供下方当前分支使用。
        gcd = a;
        // [中文导读] [AllReduce逐行 S144] 分支条件为 gcd 等于 1；成立进入本块，未成立继续后续分支。
        if (gcd == 1) {
            // [中文导读] [AllReduce逐行 S145] 退出当前 switch 或内层循环，保留此前选中的结果，不继续后续项。
            break;
        // [中文导读] [AllReduce逐行 S146] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S147] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S148] 设置 frameNum 为 (gcd 大于 0) ? 通信域 Rank 总数 / gcd : 0；该值供下方当前分支使用。
    frameNum = (gcd > 0) ? topoInfo->userRankSize / gcd : 0;
    // [中文导读] [AllReduce逐行 S149] 直接返回 frameNum，调用者取得本分支结果。
    return frameNum;
// [中文导读] [AllReduce逐行 S150] 结束 CalcFrameNum 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S152] 定义 SelectCcuMsAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
SelectorStatus AutoSelectorBase::SelectCcuMsAlgo(
    // [中文导读] [AllReduce逐行 S153] 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S154] 续接 SelectCcuMsAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S155] 进入 SelectCcuMsAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
{
    // [中文导读] [AllReduce逐行 S156] 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)opParam;
    // [中文导读] [AllReduce逐行 S157] 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)topoInfo;
    // [中文导读] [AllReduce逐行 S158] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S159] 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)selectAlgName;
    // [中文导读] [AllReduce逐行 S160] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S161] 结束 SelectCcuMsAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S163] 定义 SelectCcuScheduleAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
SelectorStatus AutoSelectorBase::SelectCcuScheduleAlgo(
    // [中文导读] [AllReduce逐行 S164] 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S165] 续接 SelectCcuScheduleAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S166] 进入 SelectCcuScheduleAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
{
    // [中文导读] [AllReduce逐行 S167] 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)opParam;
    // [中文导读] [AllReduce逐行 S168] 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)topoInfo;
    // [中文导读] [AllReduce逐行 S169] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S170] 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)selectAlgName;
    // [中文导读] [AllReduce逐行 S171] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S172] 结束 SelectCcuScheduleAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S174] 定义 SelectAicpuAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
SelectorStatus AutoSelectorBase::SelectAicpuAlgo(
    // [中文导读] [AllReduce逐行 S175] 续接 SelectAicpuAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S176] 续接 SelectAicpuAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S177] 进入 SelectAicpuAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
{
    // [中文导读] [AllReduce逐行 S178] 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)opParam;
    // [中文导读] [AllReduce逐行 S179] 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)topoInfo;
    // [中文导读] [AllReduce逐行 S180] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S181] 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)selectAlgName;
    // [中文导读] [AllReduce逐行 S182] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S183] 结束 SelectAicpuAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S185] 定义 SelectAivAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
SelectorStatus AutoSelectorBase::SelectAivAlgo(
    // [中文导读] [AllReduce逐行 S186] 续接 SelectAivAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S187] 续接 SelectAivAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S188] 进入 SelectAivAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
{
    // [中文导读] [AllReduce逐行 S189] 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)opParam;
    // [中文导读] [AllReduce逐行 S190] 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)topoInfo;
    // [中文导读] [AllReduce逐行 S191] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S192] 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)selectAlgName;
    // [中文导读] [AllReduce逐行 S193] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S194] 结束 SelectAivAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S196] 定义 SelectDPUAlgo 入口：基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
SelectorStatus AutoSelectorBase::SelectDPUAlgo(
    // [中文导读] [AllReduce逐行 S197] 续接 SelectDPUAlgo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam,
    // [中文导读] [AllReduce逐行 S198] 续接 SelectDPUAlgo 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数) const；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S199] 进入 SelectDPUAlgo 的实现作用域；基类默认候选实现，忽略三个参数并返回 NOT_MATCH；实际 AllReduce 选择由派生类 override 处理。
{
    // [中文导读] [AllReduce逐行 S200] 显式标记 opParam 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)opParam;
    // [中文导读] [AllReduce逐行 S201] 显式标记 topoInfo 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)topoInfo;
    // [中文导读] [AllReduce逐行 S202] 显式标记 configAlgMap 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)configAlgMap;
    // [中文导读] [AllReduce逐行 S203] 显式标记 算法名输出参数 在此兼容/default 分支未使用，避免编译器未使用参数警告。
    (void)selectAlgName;
    // [中文导读] [AllReduce逐行 S204] 当前类型、拓扑或配置不被此候选支持，返回 NOT_MATCH 供上层尝试其它引擎/选择器。
    return SelectorStatus::NOT_MATCH;
// [中文导读] [AllReduce逐行 S205] 结束 SelectDPUAlgo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S207] 定义 IsLayerAllConnetedWithTopo 入口：检查指定层是否有一个给定拓扑实例覆盖本层全部本地 Rank；缺少层或拓扑记录返回 false。
bool AutoSelectorBase::IsLayerAllConnetedWithTopo(
    // [中文导读] [AllReduce逐行 S208] 续接 IsLayerAllConnetedWithTopo 的入口参数/基类初始化：const TopoInfoWithNetLayerDetails* topoInfo, const u32 当前网络层编号, const CommTopo topoType)；引用参数按声明的 const 限制读写。
    const TopoInfoWithNetLayerDetails* topoInfo, const u32 netLayer, const CommTopo topoType)
// [中文导读] [AllReduce逐行 S209] 进入 IsLayerAllConnetedWithTopo 的实现作用域；检查指定层是否有一个给定拓扑实例覆盖本层全部本地 Rank；缺少层或拓扑记录返回 false。
{
    // [中文导读] [AllReduce逐行 S210] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S211] 调用 size 完成当前参数所指的子步骤；本行实参为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer.size() 不超过 当前网络层编号,。
        topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() <= netLayer,
        // [中文导读] [AllReduce逐行 S212] 开始 HCCL_WARNING 诊断输出，记录 IsLayerAllConnetedWithTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING(
            // [中文导读] [AllReduce逐行 S213] 续接 IsLayerAllConnetedWithTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[BaseSelector][IsLayerAllConnetedWithTopo] localNetInsSizeOfLayer size[%u] <= netLayer[%u]",
            // [中文导读] [AllReduce逐行 S214] 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：topoInfo->当前网络层编号Details.localNetInsSizeOfLayer.size(), 当前网络层编号，与前面的格式占位依次对应。
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size(), netLayer),
        // [中文导读] [AllReduce逐行 S215] 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。
        false);
    // [中文导读] [AllReduce逐行 S216] 设置 localRankSize 为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[当前网络层编号]；该值供下方当前分支使用。
    u32 localRankSize = topoInfo->netLayerDetails.localNetInsSizeOfLayer[netLayer];

    // [中文导读] [AllReduce逐行 S218] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S219] 调用 size 完成当前参数所指的子步骤；本行实参为 topoInfo->topoInstDetailsOfLayer.size() 不超过 当前网络层编号,。
        topoInfo->topoInstDetailsOfLayer.size() <= netLayer,
        // [中文导读] [AllReduce逐行 S220] 开始 HCCL_WARNING 诊断输出，记录 IsLayerAllConnetedWithTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_WARNING(
            // [中文导读] [AllReduce逐行 S221] 续接 IsLayerAllConnetedWithTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[BaseSelector][IsLayerAllConnetedWithTopo] topoInstDetailsOfLayer size[%u] <= netLayer[%u]",
            // [中文导读] [AllReduce逐行 S222] 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：topoInfo->topoInstDetailsOfLayer.size(), 当前网络层编号，与前面的格式占位依次对应。
            topoInfo->topoInstDetailsOfLayer.size(), netLayer),
        // [中文导读] [AllReduce逐行 S223] 为 IsLayerAllConnetedWithTopo 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。
        false);

    // [中文导读] [AllReduce逐行 S225] 设置 rankNumForTopoTypeItr 为 topoInfo->topoInstDetailsOfLayer[当前网络层编号].rankNumForTopoType.find(topoType)；该值供下方当前分支使用。
    auto rankNumForTopoTypeItr = topoInfo->topoInstDetailsOfLayer[netLayer].rankNumForTopoType.find(topoType);
    // [中文导读] [AllReduce逐行 S226] 分支条件为 rankNumForTopoTypeItr 等于 topoInfo->topoInstDetailsOfLayer[当前网络层编号].rankNumForTopoType.end(；成立进入本块，未成立继续后续分支。
    if (rankNumForTopoTypeItr == topoInfo->topoInstDetailsOfLayer[netLayer].rankNumForTopoType.end()) {
        // [中文导读] [AllReduce逐行 S227] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S228] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S230] 在 IsLayerAllConnetedWithTopo 中遍历 (auto topoRankNum : rankNumForTopoTypeItr->second 指定的集合或索引区间；边界/迭代规则为 (auto topoRankNum : rankNumForTopoTypeItr->second。
    for (auto topoRankNum : rankNumForTopoTypeItr->second) {
        // [中文导读] [AllReduce逐行 S231] 分支条件为 topoRankNum 等于 localRankSize；成立进入本块，未成立继续后续分支。
        if (topoRankNum == localRankSize) {
            // [中文导读] [AllReduce逐行 S232] 当前能力/拓扑/匹配检查满足，返回 true。
            return true;
        // [中文导读] [AllReduce逐行 S233] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S234] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S235] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
    return false;
// [中文导读] [AllReduce逐行 S236] 结束 IsLayerAllConnetedWithTopo 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S238] 定义 CheckMeshNumEqualToClosNum 入口：检查第零层 Mesh/CLOS 实例规模并比较首个实例 Rank 数；缺少数据返回内部错误。
HcclResult AutoSelectorBase::CheckMeshNumEqualToClosNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isEqual)
// [中文导读] [AllReduce逐行 S239] 进入 CheckMeshNumEqualToClosNum 的实现作用域；检查第零层 Mesh/CLOS 实例规模并比较首个实例 Rank 数；缺少数据返回内部错误。
{
    // [中文导读] [AllReduce逐行 S240] 设置 & topoInstDetails 为 topoInfo->topoInstDetailsOfLayer；该值供下方当前分支使用。
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;

    // 检查topoInstDetails是否为空
    // [中文导读] [AllReduce逐行 S243] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S244] 调用 empty 完成当前参数所指的子步骤；本行实参为 topoInstDetails.empty(),。
        topoInstDetails.empty(),
        // [中文导读] [AllReduce逐行 S245] 开始 HCCL_ERROR 诊断输出，记录 CheckMeshNumEqualToClosNum 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."),
        // [中文导读] [AllReduce逐行 S246] 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。
        HCCL_E_INTERNAL);

    // [中文导读] [AllReduce逐行 S248] 设置 & rankNumMap 为 topoInstDetails[0].rankNumForTopoType；该值供下方当前分支使用。
    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    // [中文导读] [AllReduce逐行 S249] 设置 closItr 为 rankNumMap.find(COMM_TOPO_CLOS)；该值供下方当前分支使用。
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    // [中文导读] [AllReduce逐行 S250] 设置 meshItr 为 rankNumMap.find(COMM_TOPO_1DMESH)；该值供下方当前分支使用。
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    // [中文导读] [AllReduce逐行 S251] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S252] 调用 end 完成当前参数所指的子步骤；本行实参为 closItr 等于 rankNumMap.end() 或 closItr->second.empty() 或 meshItr 等于 rankNumMap.end()。
        closItr == rankNumMap.end() || closItr->second.empty() || meshItr == rankNumMap.end()
            // [中文导读] [AllReduce逐行 S253] 调用 empty 完成当前参数所指的子步骤；本行实参为 或 meshItr->second.empty(),。
            || meshItr->second.empty(),
        // [中文导读] [AllReduce逐行 S254] 开始 HCCL_ERROR 诊断输出，记录 CheckMeshNumEqualToClosNum 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[BaseSelector][CheckMeshNumEqualToClosNum] topoInstDetailsOfLayer0 size is zero."),
        // [中文导读] [AllReduce逐行 S255] 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。
        HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量并比较是否相等
    // [中文导读] [AllReduce逐行 S258] 续接 CheckMeshNumEqualToClosNum 当前语句的具体实参/字段：isEqual = (closItr->second[0] 等于 meshItr->second[0])；由其完整表达式完成参数组装、检查或结果写回。
    isEqual = (closItr->second[0] == meshItr->second[0]);
    // [中文导读] [AllReduce逐行 S259] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S260] 结束 CheckMeshNumEqualToClosNum 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S262] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
HcclResult
// [中文导读] [AllReduce逐行 S263] 定义 CheckClosNumMultipleOfMeshNum 入口：检查第零层 CLOS Rank 数是否严格大于且整除 Mesh Rank 数，Mesh 本身还需大于 1。
AutoSelectorBase::CheckClosNumMultipleOfMeshNum(const TopoInfoWithNetLayerDetails* topoInfo, bool& isMultiple)
// [中文导读] [AllReduce逐行 S264] 进入 CheckClosNumMultipleOfMeshNum 的实现作用域；检查第零层 CLOS Rank 数是否严格大于且整除 Mesh Rank 数，Mesh 本身还需大于 1。
{
    // [中文导读] [AllReduce逐行 S265] 设置 & topoInstDetails 为 topoInfo->topoInstDetailsOfLayer；该值供下方当前分支使用。
    const auto& topoInstDetails = topoInfo->topoInstDetailsOfLayer;
    // 检查topoInstDetails是否为空
    // [中文导读] [AllReduce逐行 S267] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S268] 调用 empty 完成当前参数所指的子步骤；本行实参为 topoInstDetails.empty(),。
        topoInstDetails.empty(),
        // [中文导读] [AllReduce逐行 S269] 开始 HCCL_ERROR 诊断输出，记录 CheckClosNumMultipleOfMeshNum 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."),
        // [中文导读] [AllReduce逐行 S270] 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。
        HCCL_E_INTERNAL);

    // [中文导读] [AllReduce逐行 S272] 设置 & rankNumMap 为 topoInstDetails[0].rankNumForTopoType；该值供下方当前分支使用。
    const auto& rankNumMap = topoInstDetails[0].rankNumForTopoType;
    // [中文导读] [AllReduce逐行 S273] 设置 closItr 为 rankNumMap.find(COMM_TOPO_CLOS)；该值供下方当前分支使用。
    auto closItr = rankNumMap.find(COMM_TOPO_CLOS);
    // [中文导读] [AllReduce逐行 S274] 设置 meshItr 为 rankNumMap.find(COMM_TOPO_1DMESH)；该值供下方当前分支使用。
    auto meshItr = rankNumMap.find(COMM_TOPO_1DMESH);
    // [中文导读] [AllReduce逐行 S275] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S276] 调用 end 完成当前参数所指的子步骤；本行实参为 closItr 等于 rankNumMap.end() 或 closItr->second.empty() 或 meshItr 等于 rankNumMap.end()。
        closItr == rankNumMap.end() || closItr->second.empty() || meshItr == rankNumMap.end()
            // [中文导读] [AllReduce逐行 S277] 调用 empty 完成当前参数所指的子步骤；本行实参为 或 meshItr->second.empty(),。
            || meshItr->second.empty(),
        // [中文导读] [AllReduce逐行 S278] 开始 HCCL_ERROR 诊断输出，记录 CheckClosNumMultipleOfMeshNum 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[BaseSelector][CheckClosNumMultipleOfMeshNum] topoInstDetailsOfLayer0 size is zero."),
        // [中文导读] [AllReduce逐行 S279] 向条件返回宏提供 内部错误；上方检查成立才退出当前函数。
        HCCL_E_INTERNAL);

    // 获取CLOS和1DMESH拓扑的rank数量
    // [中文导读] [AllReduce逐行 S282] 设置 closRankNums 为 closItr->second[0]；该值供下方当前分支使用。
    const auto closRankNums = closItr->second[0];
    // [中文导读] [AllReduce逐行 S283] 设置 meshRankNums 为 meshItr->second[0]；该值供下方当前分支使用。
    const auto meshRankNums = meshItr->second[0];

    // 检查CLOS数量是否大于1DMESH数量且是1DMESH数量的倍数
    // [中文导读] [AllReduce逐行 S286] 续接 CheckClosNumMultipleOfMeshNum 当前语句的具体实参/字段：isMultiple = (meshRankNums 大于 1) 且 (closRankNums 大于 meshRankNums) 且 (closRankNums % meshRankNums 等于 0)；由其完整表达式完成参数组装、检查或结果写回。
    isMultiple = (meshRankNums > 1) && (closRankNums > meshRankNums) && (closRankNums % meshRankNums == 0);
    // [中文导读] [AllReduce逐行 S287] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S288] 结束 CheckClosNumMultipleOfMeshNum 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S290] 定义 IsTwoLevelNetLayer 入口：判定可用的二级网络：排除 HostDPUOnly，要求第二网络层含 CLOS 且第零层不止一个本地 Rank。
bool AutoSelectorBase::IsTwoLevelNetLayer(const TopoInfoWithNetLayerDetails* topoInfo, const OpParam& opParam)
// [中文导读] [AllReduce逐行 S291] 进入 IsTwoLevelNetLayer 的实现作用域；判定可用的二级网络：排除 HostDPUOnly，要求第二网络层含 CLOS 且第零层不止一个本地 Rank。
{
    // [中文导读] [AllReduce逐行 S292] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S293] 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo 等于 nullptr, HCCL_WARNING("[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr."), false，与前面的格式占位依次对应。
        topoInfo == nullptr, HCCL_WARNING("[AutoSelectorBase][IsTwoLevelNetLayer] topoInfo is nullptr."), false);
    // hostDPU场景不走二级网络算法
    // [中文导读] [AllReduce逐行 S295] 设置 hostDPUOnly 为 false；该值供下方当前分支使用。
    bool hostDPUOnly = false;
    // [中文导读] [AllReduce逐行 S296] 分支条件为 (CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) 等于 成功状态) 且 hostDPUOnly；成立进入本块，未成立继续后续分支。
    if ((CheckHostDPUOnly(opParam.hcclComm, topoInfo, hostDPUOnly) == HCCL_SUCCESS) && hostDPUOnly) {
        // [中文导读] [AllReduce逐行 S297] 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[AutoSelectorBase][IsTwoLevelNetLayer] host DPU only, not two level net layer.");
        // [中文导读] [AllReduce逐行 S298] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S299] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S300] 分支条件为 topoInfo->当前网络层编号Details.网络层数量 不超过 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->netLayerDetails.netLayerNum <= 1) {
        // [中文导读] [AllReduce逐行 S301] 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S302] 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AutoSelectorBase][IsTwoLevelNetLayer] netLayerNum[%u] <= 1, not two level net layer.",
            // [中文导读] [AllReduce逐行 S303] 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo->当前网络层编号Details.网络层数量，与前面的格式占位依次对应。
            topoInfo->netLayerDetails.netLayerNum);
        // [中文导读] [AllReduce逐行 S304] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S305] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S306] 设置 level1Idx 为 topoInfo->当前网络层编号Details.当前网络层编号s[1]；该值供下方当前分支使用。
    u32 level1Idx = topoInfo->netLayerDetails.netLayers[1];
    // [中文导读] [AllReduce逐行 S307] 设置 hasLevel1Clos 为 topoInfo->topoInstDetailsOfLayer.size() 大于 level1Idx；该值供下方当前分支使用。
    bool hasLevel1Clos = topoInfo->topoInstDetailsOfLayer.size() > level1Idx
                         // [中文导读] [AllReduce逐行 S308] 调用 find 完成当前参数所指的子步骤；本行实参为 且 topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS)。
                         && topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.find(COMM_TOPO_CLOS)
                                // [中文导读] [AllReduce逐行 S309] 调用 end 完成当前参数所指的子步骤；本行实参为 不等于 topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end()。
                                != topoInfo->topoInstDetailsOfLayer[level1Idx].rankNumForTopoType.end();
    // [中文导读] [AllReduce逐行 S310] 分支条件为 !hasLevel1Clos；成立进入本块，未成立继续后续分支。
    if (!hasLevel1Clos) {
        // [中文导读] [AllReduce逐行 S311] 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S312] 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AutoSelectorBase][IsTwoLevelNetLayer] level1[%u] has no CLOS topo, not two level net layer.", level1Idx);
        // [中文导读] [AllReduce逐行 S313] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S314] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S315] 分支条件为 topoInfo->当前网络层编号Details.localNetInsSizeOfLayer.size() 小于 1；成立进入本块，未成立继续后续分支。
    if (topoInfo->netLayerDetails.localNetInsSizeOfLayer.size() < 1
        // [中文导读] [AllReduce逐行 S316] 补充同一条件的 或者 子条件：topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[0] 不超过 1。
        || topoInfo->netLayerDetails.localNetInsSizeOfLayer[0] <= 1) {
        // [中文导读] [AllReduce逐行 S317] 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S318] 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[AutoSelectorBase][IsTwoLevelNetLayer] level0 localNetInsSizeOfLayer[%zu] <= 1, not two level net layer.",
            // [中文导读] [AllReduce逐行 S319] 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo->当前网络层编号Details.localNetInsSizeOfLayer.size(，与前面的格式占位依次对应。
            topoInfo->netLayerDetails.localNetInsSizeOfLayer.size());
        // [中文导读] [AllReduce逐行 S320] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S321] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }
    // [中文导读] [AllReduce逐行 S322] 开始 HCCL_INFO 诊断输出，记录 IsTwoLevelNetLayer 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S323] 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[AutoSelectorBase][IsTwoLevelNetLayer] topoLevelNums[%u], netLayerNum[%u], level0Topo[MESH_1D], "
        // [中文导读] [AllReduce逐行 S324] 续接 IsTwoLevelNetLayer 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "level1Idx[%u] has CLOS, level0LocalNetInsSize[%u], is two level net layer.",
        // [中文导读] [AllReduce逐行 S325] 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo->topoLevelNums, topoInfo->当前网络层编号Details.网络层数量, level1Idx，与前面的格式占位依次对应。
        topoInfo->topoLevelNums, topoInfo->netLayerDetails.netLayerNum, level1Idx,
        // [中文导读] [AllReduce逐行 S326] 为 IsTwoLevelNetLayer 的诊断/错误宏提供实参：topoInfo->当前网络层编号Details.localNetInsSizeOfLayer[0]，与前面的格式占位依次对应。
        topoInfo->netLayerDetails.localNetInsSizeOfLayer[0]);
    // [中文导读] [AllReduce逐行 S327] 当前能力/拓扑/匹配检查满足，返回 true。
    return true;
// [中文导读] [AllReduce逐行 S328] 结束 IsTwoLevelNetLayer 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S330] 定义 IsDevType960 入口：读取当前设备类型并判断是否 DEV_TYPE_960；代码未检查此处设备查询返回码。
bool AutoSelectorBase::IsDevType960()
// [中文导读] [AllReduce逐行 S331] 进入 IsDevType960 的实现作用域；读取当前设备类型并判断是否 DEV_TYPE_960；代码未检查此处设备查询返回码。
{
    // [中文导读] [AllReduce逐行 S332] 声明本阶段局部变量 HcclDevType deviceType，实际值由后续查询/计算填写。
    HcclDevType deviceType;
    // [中文导读] [AllReduce逐行 S333] 调用 HcclGetDeviceType 完成当前参数所指的子步骤；本行实参为 HcclGetDeviceType(deviceType)。
    HcclGetDeviceType(deviceType);
    // [中文导读] [AllReduce逐行 S334] 直接返回 deviceType 等于 HcclDevType::DEV_TYPE_960，调用者取得本分支结果。
    return deviceType == HcclDevType::DEV_TYPE_960;
// [中文导读] [AllReduce逐行 S335] 结束 IsDevType960 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S337] 定义 IsInputOutputOverlap 入口：按输入输出基址和字节容量计算闭区间，判断地址区间是否交叠；空地址/零大小判为不重叠。
bool AutoSelectorBase::IsInputOutputOverlap(const OpParam& opParam) const
// [中文导读] [AllReduce逐行 S338] 进入 IsInputOutputOverlap 的实现作用域；按输入输出基址和字节容量计算闭区间，判断地址区间是否交叠；空地址/零大小判为不重叠。
{
    // [中文导读] [AllReduce逐行 S339] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S340] 续接本次错误检查/子调用实参：opParam.inputPtr 等于 nullptr 或 opParam.outputPtr 等于 nullptr；返回行为由所在完整宏决定。
        opParam.inputPtr == nullptr || opParam.outputPtr == nullptr,
        // [中文导读] [AllReduce逐行 S341] 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer is null. Not overlap."),
        // [中文导读] [AllReduce逐行 S342] 为 IsInputOutputOverlap 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。
        false);

    // [中文导读] [AllReduce逐行 S344] 设置 inputDataSize 为 opParam.inputSize；该值供下方当前分支使用。
    u64 inputDataSize = opParam.inputSize;
    // [中文导读] [AllReduce逐行 S345] 设置 outputDataSize 为 opParam.outputSize；该值供下方当前分支使用。
    u64 outputDataSize = opParam.outputSize;

    // [中文导读] [AllReduce逐行 S347] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S348] 续接本次错误检查/子调用实参：inputDataSize 等于 0 或 outputDataSize 等于 0；返回行为由所在完整宏决定。
        inputDataSize == 0 || outputDataSize == 0,
        // 不存在overlap情况
        // [中文导读] [AllReduce逐行 S350] 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[Algo][AutoSelectorBase][IsInputOutputOverlap] The input or output buffer size is 0. Not overlap."),
        // [中文导读] [AllReduce逐行 S351] 为 IsInputOutputOverlap 的诊断/错误宏提供实参：false，与前面的格式占位依次对应。
        false);

    // [中文导读] [AllReduce逐行 S353] 设置 uintptr_t inputStart 为 reinterpret_cast<uintptr_t>(opParam.inputPtr)；该值供下方当前分支使用。
    uintptr_t inputStart = reinterpret_cast<uintptr_t>(opParam.inputPtr);
    // [中文导读] [AllReduce逐行 S354] 设置 uintptr_t outputStart 为 reinterpret_cast<uintptr_t>(opParam.outputPtr)；该值供下方当前分支使用。
    uintptr_t outputStart = reinterpret_cast<uintptr_t>(opParam.outputPtr);
    // [中文导读] [AllReduce逐行 S355] 设置 uintptr_t inputEnd 为 inputStart + inputDataSize - 1；该值供下方当前分支使用。
    uintptr_t inputEnd = inputStart + inputDataSize - 1;
    // [中文导读] [AllReduce逐行 S356] 设置 uintptr_t outputEnd 为 outputStart + outputDataSize - 1；该值供下方当前分支使用。
    uintptr_t outputEnd = outputStart + outputDataSize - 1;

    // [中文导读] [AllReduce逐行 S358] 开始 HCCL_DEBUG 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG(
        // [中文导读] [AllReduce逐行 S359] 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
        // [中文导读] [AllReduce逐行 S360] 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "outputEnd[%llu].",
        // [中文导读] [AllReduce逐行 S361] 为 IsInputOutputOverlap 的诊断/错误宏提供实参：inputStart, inputEnd, outputStart, outputEnd，与前面的格式占位依次对应。
        inputStart, inputEnd, outputStart, outputEnd);

    // [中文导读] [AllReduce逐行 S363] 开始条件错误返回宏：条件成立时打印下方诊断并返回所列错误码；判断对象由下一行展开。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S364] 续接本次错误检查/子调用实参：inputStart 不超过 outputEnd 且 outputStart 不超过 inputEnd；返回行为由所在完整宏决定。
        inputStart <= outputEnd && outputStart <= inputEnd,
        // [中文导读] [AllReduce逐行 S365] 开始 HCCL_INFO 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S366] 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[Algo][AutoSelectorBase][IsInputOutputOverlap] inputStart[%llu], inputEnd[%llu], outputStart[%llu], "
            // [中文导读] [AllReduce逐行 S367] 续接 IsInputOutputOverlap 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "outputEnd[%llu]. Overlap detected.",
            // [中文导读] [AllReduce逐行 S368] 为 IsInputOutputOverlap 的诊断/错误宏提供实参：inputStart, inputEnd, outputStart, outputEnd，与前面的格式占位依次对应。
            inputStart, inputEnd, outputStart, outputEnd),
        // [中文导读] [AllReduce逐行 S369] 为 IsInputOutputOverlap 的诊断/错误宏提供实参：true，与前面的格式占位依次对应。
        true);

    // [中文导读] [AllReduce逐行 S371] 开始 HCCL_DEBUG 诊断输出，记录 IsInputOutputOverlap 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[Algo][AutoSelectorBase][IsInputOutputOverlap]No overlap between input and output memory.");
    // [中文导读] [AllReduce逐行 S372] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
    return false;
// [中文导读] [AllReduce逐行 S373] 结束 IsInputOutputOverlap 实现；其返回状态或已写回字段由调用者接收。
}

// [中文导读] [AllReduce逐行 S375] 定义 ProcessAivConfig 入口：处理 AIV/AIV_ONLY 配置，普通 AIV 不匹配可回到 CCU_FAIL，AIV_ONLY 保留不匹配结果供上层报错。
bool AutoSelectorBase::ProcessAivConfig(
    // [中文导读] [AllReduce逐行 S376] 续接 ProcessAivConfig 的入口参数/基类初始化：OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,；引用参数按声明的 const 限制读写。
    OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo,
    // [中文导读] [AllReduce逐行 S377] 续接 ProcessAivConfig 的入口参数/基类初始化：const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& 算法名输出参数,；引用参数按声明的 const 限制读写。
    const std::map<HcclCMDType, std::vector<HcclAlgoType>>& configAlgMap, std::string& selectAlgName,
    // [中文导读] [AllReduce逐行 S378] 续接 ProcessAivConfig 的入口参数/基类初始化：SelectorStatus& ret) const；引用参数按声明的 const 限制读写。
    SelectorStatus& ret) const
// [中文导读] [AllReduce逐行 S379] 进入 ProcessAivConfig 的实现作用域；处理 AIV/AIV_ONLY 配置，普通 AIV 不匹配可回到 CCU_FAIL，AIV_ONLY 保留不匹配结果供上层报错。
{
    // [中文导读] [AllReduce逐行 S380] 分支条件为 当前执行配置 不等于 OpExecuteConfig::AIV 且 当前执行配置 不等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。
    if (opParam.opExecuteConfig != OpExecuteConfig::AIV && opParam.opExecuteConfig != OpExecuteConfig::AIV_ONLY) {
        // [中文导读] [AllReduce逐行 S381] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S382] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S384] 分支条件为 topoInfo->topLevelUboe；成立进入本块，未成立继续后续分支。
    if (topoInfo->topLevelUboe) {
        // [中文导读] [AllReduce逐行 S385] 设置 当前执行配置 为 OpExecuteConfig::CCU_FAIL；该值供下方当前分支使用。
        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        // [中文导读] [AllReduce逐行 S386] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S387] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S389] 设置 ret 为 SelectAivAlgo(topoInfo, opParam, configAlgMap, 算法名输出参数)；该值供下方当前分支使用。
    ret = SelectAivAlgo(topoInfo, opParam, configAlgMap, selectAlgName);
    // [中文导读] [AllReduce逐行 S390] 分支条件为 ret 等于 当前选择器不匹配；成立进入本块，未成立继续后续分支。
    if (ret == SelectorStatus::NOT_MATCH) {
        // [中文导读] [AllReduce逐行 S391] 分支条件为 当前执行配置 等于 OpExecuteConfig::AIV_ONLY；成立进入本块，未成立继续后续分支。
        if (opParam.opExecuteConfig == OpExecuteConfig::AIV_ONLY) {
            // [中文导读] [AllReduce逐行 S392] 开始 HCCL_WARNING 诊断输出，记录 ProcessAivConfig 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_WARNING(
                // [中文导读] [AllReduce逐行 S393] 续接 ProcessAivConfig 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[Algo][AutoSelectorBase] opType[%d] no aiv algorithm matched, current opExecuteConfig[%d].",
                // [中文导读] [AllReduce逐行 S394] 为 ProcessAivConfig 的诊断/错误宏提供实参：static_cast<int>(opParam.opType), static_cast<int>(当前执行配置，与前面的格式占位依次对应。
                static_cast<int>(opParam.opType), static_cast<int>(opParam.opExecuteConfig));
            // [中文导读] [AllReduce逐行 S395] 当前能力/拓扑/匹配检查满足，返回 true。
            return true;
        // [中文导读] [AllReduce逐行 S396] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S397] 设置 当前执行配置 为 OpExecuteConfig::CCU_FAIL；该值供下方当前分支使用。
        opParam.opExecuteConfig = OpExecuteConfig::CCU_FAIL;
        // [中文导读] [AllReduce逐行 S398] 当前能力/拓扑/匹配检查未满足，返回 false 供调用者走替代路径。
        return false;
    // [中文导读] [AllReduce逐行 S399] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S401] 当前能力/拓扑/匹配检查满足，返回 true。
    return true;
// [中文导读] [AllReduce逐行 S402] 结束 ProcessAivConfig 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
