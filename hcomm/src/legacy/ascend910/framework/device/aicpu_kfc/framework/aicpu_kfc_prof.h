/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef AICPU_KFC_PROF_H
#define AICPU_KFC_PROF_H

#include "common/aicpu_hccl_def.h"

enum class KfcTimeLine : int64_t {
    HCC_EXEC_START_TIME,
    SEND_TASK_START_TIME,
    SEND_SQE_FINISH_TIME,
};
class AicpuKfcProf {
public:
    static void SetCurrentProf(uint64_t launchTime);
    static void SetKfcTimeLine(KfcTimeLine kfcTimeLine);
    static void SetDebugMode(uint8_t debugMode);
    static bool IsDebugModeEquals(const uint8_t mode);
    static bool NeedRecordTimeTaken();
    static AicpuComProf* GetCurrentAicpuProf();
    static void OutputProfLog(bool debugFlag, AicpuComProf* profInfo, AicpuComProf* backupProfInfo = nullptr);
    static void AddProfLoopCnt(u32 addCnt = 1U);
    static AicpuComProf& GetProInst(AicpuComContext& ctx);
    static AicpuComProf* GetaicpuProfInst();

private:
    static void SetProfLoopCnt(uint32_t setCnt);
    static uint8_t debugMode_;
};
#endif // __AICPU_PROF_H__
