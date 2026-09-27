/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_RES_DESC_H
#define HCOMM_CCU_RES_DESC_H

#include <array>
#include <cstddef>

#include "ccu_common.h"
#include "ccu_res_defs.h"
#include "ccu_types.h"
#include "ccu_dev_mgr_imp.h"

namespace hcomm {

// CCU 资源类型数量，需与 ResType::__COUNT__ 保持一致
constexpr size_t CCU_RES_TYPE_COUNT = 9;

/**
 * @brief CCU 资源描述符，描述单个 ioDie 上各类资源的申请数量
 *
 * @note 该结构被 ccu_device 层（CcuAllocResHandleByResDescs）与
 *       ccu_instance 层（CcuResDescMgr/CcuInstance）共用，
 *       因此定义在 ccu_device 下避免 ccu_device 反向依赖 ccu_instance。
 */
class CcuResDesc {
public:
    uint32_t dieId{CCU_MAX_IODIE_NUM};
    std::array<uint32_t, CCU_RES_TYPE_COUNT> resNum{};

    CcuResult SetResNum(ResType resType, uint32_t num);
    CcuResult QueryResNum(ResType resType, uint32_t& num) const;

private:
    static bool IsValidResType(ResType resType);
    static size_t ResTypeIndex(ResType resType);
};

} // namespace hcomm

#endif // HCOMM_CCU_RES_DESC_H
