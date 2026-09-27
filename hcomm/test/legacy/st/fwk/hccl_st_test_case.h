/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ST_TEST_CASE_H
#define HCCLV2_ST_TEST_CASE_H

#include "types.h"
#include "op_type.h"
#include "data_type.h"
#include "reduce_op.h"
#include "dev_type.h"
#include "env_config_v2.h"
#include "hccl_st_situation.h"
#include <utility>
#include <vector>
#include <mutex>
#include "context/st_ctx.h"

void PrepareCtx(ThreadContext* ctx);
bool VerifyCtx(ThreadContext* ctx);

class HcclStTestCase {
public:
    explicit HcclStTestCase(Situation& situation, string caseName = "")
        : situation(situation),
          testcaseName(std::move(caseName))
    {
        InitSituationEnv();
    }

    void Start();
    bool Verify();

private:
    Situation situation;

    const string testcaseName;

    void InitSituationEnv();

    void InternalProcess(ThreadContext* ctx);

    void SetEnv();

    void UnsetEnv();

    vector<ThreadContext*> contexts;
};

#endif
