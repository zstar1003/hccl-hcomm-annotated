/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef _SIM_RUNNER_OPS_H_
#define _SIM_RUNNER_OPS_H_

#include <thread>

#include "sim_models.h"
#include "db_sim_runner_db.h"

namespace sim {
bool GetCurrRunnerTls(uint64_t serverKey, Runner& runner);
bool SetCurrCtxTls(uint64_t ctx);
uint64_t GetCurrRankId();
uint64_t GetCurrDeviceKey();
uint64_t GetRankIdByCtxId(uint64_t ctxId);

void SetLastStreamIdTls(uint64_t streamId);
void SetLastTaskIdTls(uint64_t taskId);

uint64_t GetLastStreamIdTls();
uint64_t GetLastTaskIdTls();

void SetTsDevice(int tsId);
uint32_t GetRankSize();
uint32_t GetHostSize();
uint32_t GetCurrentStreamId(uint64_t streamKey);
uint64_t GetServerKeyById(uint32_t superPodIdx, uint32_t serverIdx);
} // namespace sim
#endif
