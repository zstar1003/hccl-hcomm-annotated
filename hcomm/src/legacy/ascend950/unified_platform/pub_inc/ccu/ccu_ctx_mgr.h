/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_CTX_MGR_H
#define HCCL_CCU_CTX_MGR_H

#include <vector>
#include "ccu_ctx_arg.h"
#include "ccu_ctx.h"
#include "ccu_res_pack_legacy.h"
#include "ccu_task_arg.h"

namespace Hccl {

struct CcuCtxGroup {
    std::vector<std::unique_ptr<CcuContext>> ctxs;

    // 默认构造函数
    CcuCtxGroup() = default;
    // 移动构造函数
    CcuCtxGroup(CcuCtxGroup&& other) : ctxs(std::move(other.ctxs)) {}
    // 移动赋值操作符
    CcuCtxGroup& operator=(CcuCtxGroup&& other)
    {
        if (this == &other) {
            return *this;
        }

        // 清理当前的ctxs
        for (auto& ctx : ctxs) {
            ctx.reset();
        }
        ctxs.clear();

        // 将other的ctxs移动到当前对象
        ctxs.reserve(other.ctxs.size());
        for (auto& ctx : other.ctxs) {
            ctxs.push_back(std::move(ctx));
        }

        return *this;
    }

    CcuCtxSignature GetCtxSignature();
};

class CcuCtxMgr {
public:
    CcuCtxMgr();
    ~CcuCtxMgr();

    /**
     * @brief 分配资源
     *
     * @param deviceLogicId device逻辑ID
     * @param ctxGroup CCU上下文组
     * @param resPack 资源包
     *
     * @return 返回HcclResult，表示分配资源是否成功
     * @note 此函数用于分配资源
     */
    static HcclResult AllocRes(s32 deviceLogicId, CcuCtxGroup& ctxGroup, CcuResPack& resPack);

    /**
     * @brief 释放资源
     *
     * @param deviceLogicId device逻辑ID
     * @param ctxGroup CCU上下文组
     *
     * @return 返回HcclResult，表示释放资源是否成功
     *
     * @note 此函数用于释放指定device逻辑ID的CCU上下文组的资源
     */
    static HcclResult ReleaseRes(s32 deviceLogicId, CcuCtxGroup& ctxGroup);

    /**
     * @brief 获取任务参数
     *
     * @param deviceLogicId device逻辑ID
     * @param ccuTaskArg CCU 任务参数
     * @param executorId 执行器ID
     * @param taskParam 任务参数
     *
     * @return HcclResult 获取参数结果
     * @note 无
     */
    static HcclResult GetTaskParam(
        s32 deviceLogicId, CcuTaskArg& ccuTaskArg, const uint64_t executorId,
        std::vector<std::vector<CcuTaskParam>>& taskParam);

    /**
     * @brief 获取任务Profiling信息
     *
     * @param deviceLogicId device逻辑ID
     * @param ccuTaskArg CCU 任务参数
     * @param entityId 执行器ID
     * @param profilingInfo 任务profiling信息
     *
     * @return HcclResult 获取profiling结果
     * @note 无
     */
    static HcclResult GetProfilingInfo(
        s32 deviceLogicId, CcuTaskArg& ccuTaskArg, uint64_t entityId,
        std::vector<std::vector<CcuProfilingInfo>>& profilingInfo);

private:
};
}; // namespace Hccl

#endif // HCCL_CCU_CTX_MGR_H
