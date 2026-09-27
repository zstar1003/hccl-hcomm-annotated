/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_INS_EXE_QUE_H
#define HCCL_CCU_INS_EXE_QUE_H

#include "ccu_ctx_mgr.h"

namespace Hccl {
class InsExeQue {
public:
    InsExeQue();
    ~InsExeQue();

    struct ExtInsExeEntity {
        CcuCtxGroup ctxGroup;
        bool isFuncBlock{false};
    };

    using ExtInsExeEntityId = uint64_t;

    /**
     * @brief 注册扩展指令
     *
     * @param deviceLogicId device逻辑ID
     * @param entity 扩展指令执行实体
     * @param entityId 扩展指令执行实体ID
     * @return HcclResult 返回HcclResult类型的结果
     * @note 无
     */
    static HcclResult
    RegisterExtendInstruction(s32 deviceLogicId, ExtInsExeEntity& entity, ExtInsExeEntityId& entityId);

    /**
     * @brief 注销扩展指令
     *
     * @param deviceLogicId device逻辑ID
     * @param entityId 扩展指令执行实体ID
     * @return HcclResult 返回HcclResult类型的结果
     * @note 此函数用于注销扩展指令
     */
    static HcclResult DeregisterExtendInstruction(s32 deviceLogicId, const ExtInsExeEntityId& entityId);

private:
};
}; // namespace Hccl

#endif // HCCL_CCU_INS_EXE_QUE_H
