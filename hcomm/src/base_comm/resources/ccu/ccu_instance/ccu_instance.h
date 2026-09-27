/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_INSTANCE_H
#define HCOMM_CCU_INSTANCE_H

#include <array>
#include <memory>
#include <vector>

#include "ccu_types.h"
#include "ccu_res_pack.h"
#include "ccu_drv_handle.h"
#include "ccu_device_res.h"
#include "ccu_res_desc.h"
#include "ccu_res_desc_mgr.h"

namespace hcomm {

/**
 * @note 职责：管理通信域持有的CCU资源
 */
class CcuInstance {
public:
    explicit CcuInstance() {};
    ~CcuInstance();
    CcuResult InitByInsType(const CcuInstanceType insType);
    // 基于资源描述符数组初始化（新接口路径，资源数量由 resDesc 驱动）
    CcuResult InitByResDescs(const CcuResDesc* descs[], uint32_t descNum);
    // 使用当前 Device 上所有已使能 ioDie 的全部资源初始化
    CcuResult InitByAllRes();
    CcuResult Reset();
    CcuResPack* GetResPack();
    // 获取 ccuIns 在指定 die 上持有的资源描述符，用于查询接口
    const CcuResDesc& GetTotalResDescs(uint8_t dieId) const;
    CcuResult SaveKernel(const CcuKernelHandle kernelHandle);
    const std::vector<CcuKernelHandle>& GetUntranslatedKernels();
    void SetHandle(CcuInsHandle insHandle);

    // 是否使用固定资源数量创建（InitByInsType 路径置 true，其余路径保持 false）
    bool IsFixedResNum() const { return isFixedResNum_; }

    CcuResult BeginRegister();
    CcuResult CheckRegistering() const;
    CcuResult EndRegister();
    void AbortRegister();

    CcuResult CascCntHandleAlloc(uint8_t dieId, HcommCcuCascCntHandle handle);
    CcuResult GetCascCntBlock(HcommCcuCascCntHandle handle, CntXnBlock& cascCntBlock);

private:
    // 从 resPack_ 取 CcuResRepository，把各 die 各资源类型的占用数量写入 totalResDescs_
    CcuResult FillTotalResDescs();

    enum class RegisterState { IDLE, REGISTERING, REGISTER_ABORTED };
    RegisterState registerState_{RegisterState::IDLE};
    int32_t userDevId_{INT32_MAX};
    CcuInsHandle insHandle_{0};
    std::shared_ptr<hcomm::CcuDrvHandle> ccuDrvHandle_{};
    std::unique_ptr<CcuResPack> resPack_{};
    std::vector<CcuKernelHandle> kernelHandles_{};
    std::vector<CcuKernelHandle> untranslatedKernelHandles_{};
    std::array<CcuResDesc, CCU_MAX_IODIE_NUM> totalResDescs_{}; // ccuIns持有的全部资源描述符
    bool isFixedResNum_{false}; // InitByInsType 路径置 true（使用固定资源数量创建），其余保持 false
};

} // namespace hcomm
#endif // HCOMM_CCU_INSTANCE_H
