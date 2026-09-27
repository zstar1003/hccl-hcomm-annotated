/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_COMMUNICATOR_MANAGER_LITE_H_
#define HCCL_AICPU_COMMUNICATOR_MANAGER_LITE_H_

#include "communicator_impl_lite.h"

namespace Hccl {
class CommunicatorImplLiteMgr {
public:
    CommunicatorImplLiteMgr();
    ~CommunicatorImplLiteMgr();
    static CommunicatorImplLiteMgr& GetInstance();
    void DestroyComm(u32 commIdIndex);
    CommunicatorImplLite* Get(const u32 commIdIndex);
    std::vector<CommunicatorImplLite*> GetAll();

    void SetEnvConfig(const HcclDeviceEnvConfigLite& envConfig) { envConfig_ = envConfig; }
    const HcclDeviceEnvConfigLite& GetEnvConfig() { return envConfig_; }

private:
    std::unordered_map<u32, std::unique_ptr<CommunicatorImplLite>> communicatorImplLites;
    std::mutex serialMutex;
    HcclDeviceEnvConfigLite envConfig_;
};

} // namespace Hccl

#endif // HCCL_AICPU_COMMUNICATOR_MANAGER_LITE_H_
