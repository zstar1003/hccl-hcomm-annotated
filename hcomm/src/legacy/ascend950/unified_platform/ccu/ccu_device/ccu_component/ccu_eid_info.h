/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_EID_INFO_H
#define CCU_EID_INFO_H

#include "types.h"
#include "orion_adapter_hccp.h"

namespace Hccl {

class CcuEidInfo {
public:
    explicit CcuEidInfo();

    // 进行资源清理
    virtual ~CcuEidInfo();

    CcuEidInfo(const CcuEidInfo& that) = delete;

    CcuEidInfo& operator=(const CcuEidInfo& that) = delete;

    static CcuEidInfo& GetInstance(int32_t logicDeviceId);
    HcclResult GetEidInfo(int32_t logicDeviceId, std::vector<HrtDevEidInfo>& eidInfo);

private:
    std::vector<HrtDevEidInfo> eidInfoList_;
    bool initflag_{false};
};

}; // namespace Hccl

#endif // CCU_EID_INFO_H
