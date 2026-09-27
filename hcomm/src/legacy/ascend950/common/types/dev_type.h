/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_DEV_TYPE_H
#define HCCLV2_DEV_TYPE_H
#include "../utils/enum_factory.h"
namespace Hccl {

// 对内芯片类型
MAKE_ENUM(
    DevType, DEV_TYPE_910A, DEV_TYPE_V51_310_P3, DEV_TYPE_910A2, DEV_TYPE_V51_310_P1, DEV_TYPE_910A3, DEV_TYPE_950,
    DEV_TYPE_960, DEV_TYPE_NOSOC)

MAKE_ENUM(
    HcclMainboardId, MAINBOARD_POD, MAINBOARD_A_K_SERVER, MAINBOARD_A_X_SERVER, MAINBOARD_PCIE_STD, MAINBOARD_RSV,
    MAINBOARD_EQUIPMENT, MAINBOARD_EVB, MAINBOARD_OTHERS);

inline std::string DevTypeToString(DevType type) { return type.Describe(); }
} // namespace Hccl
#endif // HCCLV2_DEV_TYPE_H
