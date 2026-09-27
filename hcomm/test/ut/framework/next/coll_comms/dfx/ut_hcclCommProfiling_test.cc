/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include "mockcpp/mokc.h"
#include <mockcpp/mockcpp.hpp>
#include <memory>

#define private public
#include "hcclCommProfiling.h"
#include "hcclCommProfilingLite.h"
#undef private

#include "mirror_task_manager.h"
#include "mirror_task_manager_lite.h"
#include "dfx_profiling_reporter.h"
#include "profiling_reporter_lite.h"

using namespace hccl;

class HcclCommProfilingTest : public testing::Test {
protected:
    std::unique_ptr<HcclCommProfiling> profiling_;

    void SetUp() override
    {
        Hccl::MirrorTaskManager* mgr = nullptr;
        profiling_ = std::make_unique<HcclCommProfiling>(0, mgr);
    }
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HcclCommProfilingTest, Constructor_Normal)
{
    Hccl::MirrorTaskManager* mgr = nullptr;
    auto prof = std::make_unique<HcclCommProfiling>(42, mgr);
    EXPECT_NE(prof, nullptr);
    EXPECT_FALSE(prof->initializedFlag_);
}

TEST_F(HcclCommProfilingTest, GetMirrorTaskManager_Null) { EXPECT_EQ(profiling_->GetMirrorTaskManager(), nullptr); }

TEST_F(HcclCommProfilingTest, GetMirrorTaskManager_NonNull)
{
    Hccl::MirrorTaskManager* mgr = reinterpret_cast<Hccl::MirrorTaskManager*>(0x1234);
    auto prof = std::make_unique<HcclCommProfiling>(0, mgr);
    EXPECT_EQ(prof->GetMirrorTaskManager(), mgr);
}

TEST_F(HcclCommProfilingTest, Init_NotInitialized) { HcclResult ret = profiling_->Init(); }

TEST_F(HcclCommProfilingTest, Init_AlreadyInitialized)
{
    profiling_->initializedFlag_ = true;
    HcclResult ret = profiling_->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommProfilingTest, ReportOp_NotInitialized) { EXPECT_NO_THROW(profiling_->ReportOp(1000, false, true)); }

TEST_F(HcclCommProfilingTest, SetCurrDfxOpInfo_Null) { EXPECT_NO_THROW(profiling_->SetCurrDfxOpInfo(nullptr)); }

TEST_F(HcclCommProfilingTest, UpdateProfStat_NotInitialized) { EXPECT_NO_THROW(profiling_->UpdateProfStat()); }

TEST_F(HcclCommProfilingTest, ReportAllTasks_NotInitialized) { EXPECT_NO_THROW(profiling_->ReportAllTasks(false)); }

TEST_F(HcclCommProfilingTest, ReportMc2CommInfo_Normal)
{
    Mc2CommInfo info{};
    info.FreeStreamId = 1;
    info.streamsId = {1, 2, 3};
    info.groupname = "test_group";
    info.myRankId = 0;
    info.rankSize = 4;
    info.parentRankId = 0;
    EXPECT_NO_THROW(profiling_->ReportMc2CommInfo(info));
}

TEST_F(HcclCommProfilingTest, ReportMc2CommInfo_Empty)
{
    Mc2CommInfo info{};
    EXPECT_NO_THROW(profiling_->ReportMc2CommInfo(info));
}

class HcclCommProfilingLiteTest : public testing::Test {
protected:
    std::unique_ptr<HcclCommProfilingLite> profilingLite_;

    void SetUp() override { profilingLite_ = std::make_unique<HcclCommProfilingLite>(0); }
    void TearDown() override { GlobalMockObject::verify(); }
};

TEST_F(HcclCommProfilingLiteTest, Constructor_Normal)
{
    auto prof = std::make_unique<HcclCommProfilingLite>(0);
    EXPECT_NE(prof, nullptr);
    EXPECT_FALSE(prof->initializedFlag_);
}

TEST_F(HcclCommProfilingLiteTest, Init_AlreadyInitialized)
{
    profilingLite_->initializedFlag_ = true;
    HcclResult ret = profilingLite_->Init();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcclCommProfilingLiteTest, UpdateProfStat_NotInitialized) { EXPECT_NO_THROW(profilingLite_->UpdateProfStat()); }

TEST_F(HcclCommProfilingLiteTest, ReportAllTasks_NotInitialized)
{
    std::vector<hccl::Thread*> threads;
    Hccl::DfxCommContext ctx{};
    EXPECT_NO_THROW(profilingLite_->ReportAllTasks(threads, ctx));
}
