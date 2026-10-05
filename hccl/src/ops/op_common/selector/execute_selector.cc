/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "execute_selector.h"
#include "auto_selector_base.h"
#include "selector_registry.h"

namespace ops_hccl {

ExecuteSelector::ExecuteSelector() {}

// [中文导读] [AllReduce逐行 S19] 声明返回类型 HcclResult，分别由错误码传播或候选匹配协议解释。
HcclResult
// [中文导读] [AllReduce逐行 S20] 定义 Run 入口：读取算子注册选择器，MC2 专用优先级 18；普通调用按 std::map 优先级顺序返回第一个 MATCH。
ExecuteSelector::Run(OpParam& opParam, TopoInfoWithNetLayerDetails* topoInfo, std::string& selectAlgName) const
// [中文导读] [AllReduce逐行 S21] 进入 Run 的实现作用域；读取算子注册选择器，MC2 专用优先级 18；普通调用按 std::map 优先级顺序返回第一个 MATCH。
{
    // [中文导读] [AllReduce逐行 S22] 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_DEBUG("[Algo][Selector] Run.");
    // [中文导读] [AllReduce逐行 S23] 设置 std::map<u32, AutoSelectorBase*> selectors 为 SelectorRegistry::Global()->GetAllSelectors()；该值供下方当前分支使用。
    std::map<u32, AutoSelectorBase*> selectors = SelectorRegistry::Global()->GetAllSelectors();

    // [中文导读] [AllReduce逐行 S25] 分支条件为 opParam.isMc2；成立进入本块，未成立继续后续分支。
    if (opParam.isMc2) {
        // [中文导读] [AllReduce逐行 S26] 设置 iter 为 selectors.find(18)；该值供下方当前分支使用。
        auto iter = selectors.find(18);
        // [中文导读] [AllReduce逐行 S27] 分支条件为 iter 等于 selectors.end(；成立进入本块，未成立继续后续分支。
        if (iter == selectors.end()) {
            // [中文导读] [AllReduce逐行 S28] 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_ERROR("[Algo][Selector] CCU selector is not registered.");
            // [中文导读] [AllReduce逐行 S29] 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。
            return HcclResult::HCCL_E_NOT_SUPPORT;
        // [中文导读] [AllReduce逐行 S30] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S31] 分支条件为 iter->second->Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。
        if (iter->second->Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {
            // [中文导读] [AllReduce逐行 S32] 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO(
                // [中文导读] [AllReduce逐行 S33] 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[Algo][Selector] The ccu selector[priority of %u] is matched, the selected algo type is %s",
                // [中文导读] [AllReduce逐行 S34] 为 Run 的诊断/错误宏提供实参：iter->first, 算法名输出参数.c_str(，与前面的格式占位依次对应。
                iter->first, selectAlgName.c_str());
            // [中文导读] [AllReduce逐行 S35] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
            return HcclResult::HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S36] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
        // [中文导读] [AllReduce逐行 S37] 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_ERROR("[Algo][Selector] CCU selector can not match for optype[%d].", opParam.opType);
        // [中文导读] [AllReduce逐行 S38] 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。
        return HcclResult::HCCL_E_NOT_SUPPORT;
    // [中文导读] [AllReduce逐行 S39] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S41] 设置 selectors 为 SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType)；该值供下方当前分支使用。
    selectors = SelectorRegistry::Global()->GetSelectorsByOpType(opParam.opType);
    // [中文导读] [AllReduce逐行 S42] 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_INFO("[Algo][Selector] The selector nums of optype[%d] is [%zu].", opParam.opType, selectors.size());
    // [中文导读] [AllReduce逐行 S43] 按旧选择器优先级顺序尝试已注册候选；边界/迭代规则为 (auto iter : selectors。
    for (auto iter : selectors) {
        // [中文导读] [AllReduce逐行 S44] 开始 HCCL_DEBUG 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
        HCCL_DEBUG("[Algo][Selector] The selector[priority of %llu] is running.", iter.first);
        // [中文导读] [AllReduce逐行 S45] 分支条件为 iter.second->Select(opParam, topoInfo, 算法名输出参数) 等于 当前选择器匹配；成立进入本块，未成立继续后续分支。
        if (iter.second->Select(opParam, topoInfo, selectAlgName) == SelectorStatus::MATCH) {
            // [中文导读] [AllReduce逐行 S46] 开始 HCCL_INFO 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
            HCCL_INFO(
                // [中文导读] [AllReduce逐行 S47] 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
                "[Algo][Selector] The selector[priority of %llu] is matched, the selected algo type is %s", iter.first,
                // [中文导读] [AllReduce逐行 S48] 为 Run 的诊断/错误宏提供实参：算法名输出参数.c_str(，与前面的格式占位依次对应。
                selectAlgName.c_str());
            // [中文导读] [AllReduce逐行 S49] 当前路径返回成功；异步原语的成功表示任务/协议提交成功，数据完成依赖相应通知或 Join。
            return HcclResult::HCCL_SUCCESS;
        // [中文导读] [AllReduce逐行 S50] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
        }
    // [中文导读] [AllReduce逐行 S51] 结束当前条件、循环或局部对象构造的作用域，恢复外层执行流程。
    }

    // [中文导读] [AllReduce逐行 S53] 开始 HCCL_ERROR 诊断输出，记录 Run 当前阶段的参数、候选或错误；日志本身不决定返回码。
    HCCL_ERROR(
        // [中文导读] [AllReduce逐行 S54] 续接 Run 的诊断格式串，展示此行列出的字段标签；它是编译期字符串拼接，不发起额外操作。
        "[Algo][Selector] No supported algorithm for optype[%d], opExecuteConfig[%d].",
        // [中文导读] [AllReduce逐行 S55] 为 Run 的诊断/错误宏提供实参：static_cast<int>(opParam.opType), static_cast<int>(当前执行配置，与前面的格式占位依次对应。
        static_cast<int>(opParam.opType), static_cast<int>(opParam.opExecuteConfig));
    // [中文导读] [AllReduce逐行 S56] 终止当前函数并向上返回 HcclResult::不支持错误；调用者 CHK_RET 决定是否继续向上传播。
    return HcclResult::HCCL_E_NOT_SUPPORT;
// [中文导读] [AllReduce逐行 S57] 结束 Run 实现；其返回状态或已写回字段由调用者接收。
}

} // namespace ops_hccl
