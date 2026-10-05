/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "coll_alg_v2_exec_registry.h"

namespace ops_hccl {

// [中文导读] [AllReduce逐行 S15] 声明CollAlgExecRegistryV2::Instance接口：取得全局执行器工厂注册表。
CollAlgExecRegistryV2& CollAlgExecRegistryV2::Instance()
// [中文导读] [AllReduce逐行 S16] 开始CollAlgExecRegistryV2::Instance的函数体。
{
    // [中文导读] [AllReduce逐行 S17] 函数内静态对象保存全局执行器工厂注册表。
    static CollAlgExecRegistryV2 globalExecRegistry;
    // [中文导读] [AllReduce逐行 S18] 返回全局执行器注册表引用。
    return globalExecRegistry;
// [中文导读] [AllReduce逐行 S19] 结束CollAlgExecRegistryV2::Instance函数体。
}

// [中文导读] [AllReduce逐行 S21] 声明CollAlgExecRegistryV2::Register接口：以算子类型和算法tag保存creator，禁止同一个键重复注册。
HcclResult CollAlgExecRegistryV2::Register(
    // [中文导读] [AllReduce逐行 S22] 函数参数包含算子类型、算法名字键，本行延续接口声明。
    const HcclCMDType type, const std::string& tag, const CollExecCreatorV2& collExecCreator)
// [中文导读] [AllReduce逐行 S23] 开始CollAlgExecRegistryV2::Register的函数体。
{
    // [中文导读] [AllReduce逐行 S24] 加锁保护算子类型/算法名到creator映射的注册写入。
    const std::lock_guard<std::mutex> lock(mu_);
    // [中文导读] [AllReduce逐行 S25] 当前算子类型已经注册同名tag时禁止覆盖。
    if (execCreators_[type].count(tag) != 0) {
        // [中文导读] [AllReduce逐行 S26] 输出错误日志，记录CollAlgExecRegistryV2::Register当前阶段和相关参数。
        HCCL_ERROR("[CollAlgExecRegistryV2]Exec tag[%s] already registered.", tag.c_str());
        // [中文导读] [AllReduce逐行 S27] 重复注册返回内部错误。
        return HcclResult::HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S28] 结束条件if (execCreators_[type].count(tag) != 0)。
    }
    // [中文导读] [AllReduce逐行 S29] 在该算子类型的子映射中插入算法tag及creator函数。
    execCreators_[type].emplace(tag, collExecCreator);
    // [中文导读] [AllReduce逐行 S30] 工厂注册成功返回。
    return HcclResult::HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S31] 结束CollAlgExecRegistryV2::Register函数体。
}

// [中文导读] [AllReduce逐行 S33] 声明CollAlgExecRegistryV2::GetAlgExec接口：按算子类型和算法名调用已注册creator新建执行器。
std::unique_ptr<InsCollAlgBase> CollAlgExecRegistryV2::GetAlgExec(const HcclCMDType type, const std::string& tag)
// [中文导读] [AllReduce逐行 S34] 开始CollAlgExecRegistryV2::GetAlgExec的函数体。
{
    // [中文导读] [AllReduce逐行 S35] 缺少该算子类型或该算法名的工厂时返回空对象。
    if (execCreators_.count(type) == 0 || execCreators_[type].count(tag) == 0) {
        // [中文导读] [AllReduce逐行 S36] 输出调试日志，记录CollAlgExecRegistryV2::GetAlgExec当前阶段和相关参数。
        HCCL_DEBUG("[CollAlgExecRegistryV2]Creator for executor tag[%s] has not registered.", tag.c_str());
        // [中文导读] [AllReduce逐行 S37] 查不到工厂返回nullptr，由调用方停止编排。
        return nullptr;
    // [中文导读] [AllReduce逐行 S38] 结束条件if (execCreators_.count(type) == 0 || execCreators_[type].count(tag) == 0)。
    }
    // [中文导读] [AllReduce逐行 S39] 输出调试日志，记录CollAlgExecRegistryV2::GetAlgExec当前阶段和相关参数。
    HCCL_DEBUG("[CollAlgExecRegistryV2][GetAlgExec]get executor by algName[%s].", tag.c_str());
    // [中文导读] [AllReduce逐行 S40] 调用对应creator创建具体InsCollAlgBase子类并以unique_ptr返回。
    return std::unique_ptr<InsCollAlgBase>(execCreators_[type][tag]());
// [中文导读] [AllReduce逐行 S41] 结束CollAlgExecRegistryV2::GetAlgExec函数体。
}

} // namespace ops_hccl
