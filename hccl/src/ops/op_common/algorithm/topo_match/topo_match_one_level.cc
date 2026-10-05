/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "topo_match_one_level.h"
#include <algorithm>
#include "log.h"

namespace ops_hccl {

TopoMatchOneLevel::TopoMatchOneLevel() {}

TopoMatchOneLevel::~TopoMatchOneLevel() {}

namespace {
    // 从 effIdx 中找 localRanks==userRankSize 的最低有效层；hostdpu 额外要求 locType==HOST
    // [中文导读] [AllReduce逐行 S23] 定义 PickFullLocalRanksLayer 入口：依有效层顺序寻找 localRanks 数量等于全域规模的层；Host 限定不满足时继续，找不到返回无效索引。
    u32 PickFullLocalRanksLayer(
        // [中文导读] [AllReduce逐行 S24] 续接 PickFullLocalRanksLayer 的入口参数/基类初始化：const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 userRankSize,；引用参数按声明的 const 限制读写。
        const std::vector<PhysicalLevelInfo>& physicalLevels, const std::vector<u32>& effIdx, u32 userRankSize,
        // [中文导读] [AllReduce逐行 S25] 续接 PickFullLocalRanksLayer 的入口参数/基类初始化：bool 仅允许 Host 位置拓扑层的条件)；引用参数按声明的 const 限制读写。
        bool requireHost)
    // [中文导读] [AllReduce逐行 S26] 进入 PickFullLocalRanksLayer 的实现作用域；依有效层顺序寻找 localRanks 数量等于全域规模的层；Host 限定不满足时继续，找不到返回无效索引。
    {
        // [中文导读] [AllReduce逐行 S27] 依有效物理层索引顺序查找覆盖全部 Rank 的层；边界/迭代规则为 (u32 idx : effIdx。
        for (u32 idx : effIdx) {
            // [中文导读] [AllReduce逐行 S28] 分支条件为 physicalLevels[idx].localRanks.size() 不等于 userRankSize；成立进入本块，未成立继续后续分支。
            if (physicalLevels[idx].localRanks.size() != userRankSize) {
                // [中文导读] [AllReduce逐行 S29] 跳过当前遍历项的剩余步骤，直接处理下一项。
                continue;
            // [中文导读] [AllReduce逐行 S30] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S31] 分支条件为 仅允许 Host 位置拓扑层的条件 且 physicalLevels[idx].locType 不等于 EndpointLocType::ENDPOINT_LOC_TYPE_HOST；成立进入本块，未成立继续后续分支。
            if (requireHost && physicalLevels[idx].locType != EndpointLocType::ENDPOINT_LOC_TYPE_HOST) {
                // [中文导读] [AllReduce逐行 S32] 跳过当前遍历项的剩余步骤，直接处理下一项。
                continue;
            // [中文导读] [AllReduce逐行 S33] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
            }
            // [中文导读] [AllReduce逐行 S34] 直接返回 idx，调用者取得本分支结果。
            return idx;
        // [中文导读] [AllReduce逐行 S35] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S36] 直接返回 INVALID_UINT，调用者取得本分支结果。
        return INVALID_UINT;
    // [中文导读] [AllReduce逐行 S37] 结束 PickFullLocalRanksLayer 实现；其返回状态或已写回字段由调用者接收。
    }
} // namespace

// [中文导读] [AllReduce逐行 S40] 定义 MatchTopo 入口：从有效物理层中选择覆盖全域 Rank 的最低一层，HostCPU 额外要求 Host 位置。
HcclResult TopoMatchOneLevel::MatchTopo(
    // [中文导读] [AllReduce逐行 S41] 续接 MatchTopo 的入口参数/基类初始化：TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)；引用参数按声明的 const 限制读写。
    TopoInfoWithNetLayerDetails* topoInfo, AlgHierarchyInfoForAllLevel& algHierarchyInfo, const AlgAttrs& algAttrs)
// [中文导读] [AllReduce逐行 S42] 进入 MatchTopo 的实现作用域；从有效物理层中选择覆盖全域 Rank 的最低一层，HostCPU 额外要求 Host 位置。
{
    // [中文导读] [AllReduce逐行 S43] 设置 & physicalLevels 为 topoInfo->physicalLevels；该值供下方当前分支使用。
    const auto& physicalLevels = topoInfo->physicalLevels;
    // [中文导读] [AllReduce逐行 S44] 分支条件为 physicalLevels.empty() 或 通信域 Rank 总数 等于 0；成立进入本块，未成立继续后续分支。
    if (physicalLevels.empty() || topoInfo->userRankSize == 0) {
        // [中文导读] [AllReduce逐行 S45] 开始 HCCL_ERROR 诊断输出，记录 MatchTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR(
            // [中文导读] [AllReduce逐行 S46] 续接 MatchTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[TopoMatchOneLevel] Rank [%u], physicalLevels empty or userRankSize 0. "
            // [中文导读] [AllReduce逐行 S47] 续接 MatchTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "physicalLevels.size[%zu], userRankSize[%u].",
            // [中文导读] [AllReduce逐行 S48] 为 MatchTopo 的诊断/错误宏提供实参：本地用户 Rank, physicalLevels.size(), 通信域 Rank 总数，与前面的格式占位依次对应。
            topoInfo->userRank, physicalLevels.size(), topoInfo->userRankSize);
        // [中文导读] [AllReduce逐行 S49] 终止当前函数并向上返回 HcclResult::内部错误；调用者 CHK_RET 决定是否继续向上传播。
        return HcclResult::HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S50] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S52] 设置 std::vector<u32> effIdx 为 CollectEffectiveIndices(physicalLevels, algAttrs.engine)；该值供下方当前分支使用。
    std::vector<u32> effIdx = CollectEffectiveIndices(physicalLevels, algAttrs.engine);
    // [中文导读] [AllReduce逐行 S53] 分支条件为 effIdx.empty(；成立进入本块，未成立继续后续分支。
    if (effIdx.empty()) {
        // [中文导读] [AllReduce逐行 S54] 开始 HCCL_INFO 诊断输出，记录 MatchTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO("[TopoMatchOneLevel] Rank [%u], no valid layer after engine filter.", topoInfo->userRank);
        // [中文导读] [AllReduce逐行 S55] 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。
        return HcclResult::HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S56] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S58] 续接 MatchTopo 当前语句的具体实参/字段：bool 仅允许 Host 位置拓扑层的条件 = (algAttrs.engine 等于 OpExecuteConfig::HOSTCPU)；由其完整表达式完成参数组装、检查或结果写回。
    bool requireHost = (algAttrs.engine == OpExecuteConfig::HOSTCPU);
    // [中文导读] [AllReduce逐行 S59] 设置 picked 为 PickFullLocalRanksLayer(physicalLevels, effIdx, 通信域 Rank 总数, 仅允许 Host 位置拓扑层的条件)；该值供下方当前分支使用。
    u32 picked = PickFullLocalRanksLayer(physicalLevels, effIdx, topoInfo->userRankSize, requireHost);
    // [中文导读] [AllReduce逐行 S60] 分支条件为 picked 等于 INVALID_UINT；成立进入本块，未成立继续后续分支。
    if (picked == INVALID_UINT) {
        // [中文导读] [AllReduce逐行 S61] 开始 HCCL_INFO 诊断输出，记录 MatchTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S62] 续接 MatchTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
            "[TopoMatchOneLevel] Rank [%u], no layer with localRanks == userRankSize[%u] (requireHost[%d]).",
            // [中文导读] [AllReduce逐行 S63] 为 MatchTopo 的诊断/错误宏提供实参：本地用户 Rank, 通信域 Rank 总数, static_cast<int32_t>(仅允许 Host 位置拓扑层的条件，与前面的格式占位依次对应。
            topoInfo->userRank, topoInfo->userRankSize, static_cast<int32_t>(requireHost));
        // [中文导读] [AllReduce逐行 S64] 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。
        return HcclResult::HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S65] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S67] 对 algHierarchyInfo 调整长度为 1，准备或更新本阶段列表。
    algHierarchyInfo.infos.resize(1);
    // [中文导读] [AllReduce逐行 S68] 对 algHierarchyInfo 调整长度为 1，准备或更新本阶段列表。
    algHierarchyInfo.infos[0].resize(1);
    // [中文导读] [AllReduce逐行 S69] 设置 匹配出的算法通信域层列表[0][0] 为 physicalLevels[picked].localRanks；该值供下方当前分支使用。
    algHierarchyInfo.infos[0][0] = physicalLevels[picked].localRanks;
    // [中文导读] [AllReduce逐行 S70] 设置 algHierarchyInfo.physicalIdxForAlgoLevels 为 {{static_cast<PhysicalLevelIndex>(picked)}}；该值供下方当前分支使用。
    algHierarchyInfo.physicalIdxForAlgoLevels = {{static_cast<PhysicalLevelIndex>(picked)}};
    // [中文导读] [AllReduce逐行 S71] 开始 HCCL_INFO 诊断输出，记录 MatchTopo 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S72] 续接 MatchTopo 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[TopoMatchOneLevel] Rank [%u], userRankSize [%u], physicalIdxForAlgoLevels: [%s].", topoInfo->userRank,
        // [中文导读] [AllReduce逐行 S73] 为 MatchTopo 的诊断/错误宏提供实参：通信域 Rank 总数, FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str(，与前面的格式占位依次对应。
        topoInfo->userRankSize, FormatPhysicalIdxForAlgoLevels(algHierarchyInfo.physicalIdxForAlgoLevels).c_str());
    // [中文导读] [AllReduce逐行 S74] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S75] 结束 MatchTopo 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
