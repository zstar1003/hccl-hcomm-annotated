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
#include <mockcpp/mockcpp.hpp>

#include "ccu_comp.h"
#include "hccl_common.h"

#include "unified_platform/ccu/ccu_device/ccu_component/ccu_component.h"

#include "ccu_pfe_cfg_mgr.h"

#define private public
#define protected public
using namespace hcomm;

class CcuPfeCfgMgrTest : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    static void TearDownTestCase()
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void SetUp() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
    void TearDown() override
    {
        mockcpp::GlobalMockObject::verify();
        mockcpp::GlobalMockObject::reset();
    }
};

TEST_F(CcuPfeCfgMgrTest, Ut_GetPfeJettyCtxCfg_When_DieIdInvalid_Expect_ReturnEmpty)
{
    auto& mgr = CcuPfeCfgMgr::GetInstance(0);
    // 传入无效 dieId，验证返回空向量以覆盖 warning 分支
    EXPECT_TRUE(mgr.GetPfeJettyCtxCfg(CCU_MAX_IODIE_NUM).empty());
}
