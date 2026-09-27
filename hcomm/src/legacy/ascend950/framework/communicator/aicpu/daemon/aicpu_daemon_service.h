/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_AICPU_DAEMON_SERVICE_H
#define HCCL_AICPU_DAEMON_SERVICE_H

#include <cstdint>
#include <vector>
#include <mutex>
#include "daemon_func.h"
#include "enum_factory.h"

namespace Hccl {

using Funcs = void (*)(void*);
extern "C" {
int32_t __attribute__((weak)) StartMC2MaintenanceThread(Funcs f1, void* p1, Funcs f2, void* p2);
};

MAKE_ENUM(CommandToBackGroud, Default, Stop)

class AicpuDaemonService {
public:
    static AicpuDaemonService& GetInstance();
    void ServiceRun(void* info);
    void ServiceStop(void* info) const;
    void Register(DaemonFunc* daemonFunc);
    void Unregister(DaemonFunc* daemonFunc);
    void Break(); // 强制停止

private:
    std::vector<DaemonFunc*> daemonFuncs;
    bool needBreak{false};
    static std::mutex mutexForFuncs_;
};
} // namespace Hccl
#endif // HCCL_AICPU_DAEMON_SERVICE_H
