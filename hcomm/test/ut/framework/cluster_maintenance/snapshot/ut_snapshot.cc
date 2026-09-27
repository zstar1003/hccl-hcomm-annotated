/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <mockcpp/mockcpp.hpp>
#include "adapter_rts_common.h"
#include "adapter_hccp_common.h"

#define private public
#include "snapshot_control.h"
#undef private

using namespace hccl;

class SnapshotTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--SnapshotTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--SnapshotTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

TEST_F(SnapshotTest, ut_Register_and_PreProcess_success)
{
    MOCKER(SnapShotSaveAction).stubs().will(returnValue(HCCL_SUCCESS));

    u32 phyId = 0;
    MOCKER(SnapShotSaveAction)
        .stubs()
        .with(mockcpp::any(), outBound(phyId), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    SnapshotSetInvalidComm setInvalidCommCallback = [](bool isInvalid) {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPreProcess preProcessCallback = []() {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPostProcess postProcessCallback = []() {
        return HCCL_SUCCESS;
    };

    HcclResult ret = SnapshotControl::GetInstance(0).RegisterComm(
        "testComm", setInvalidCommCallback, preProcessCallback, postProcessCallback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).RegisterBackup("testComm", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).PreProcess();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SnapshotTest, ut_Register_and_PostProcess_success)
{
    MOCKER(SnapShotSaveAction).stubs().will(returnValue(HCCL_SUCCESS));

    u32 phyId = 0;
    MOCKER(SnapShotSaveAction)
        .stubs()
        .with(mockcpp::any(), outBound(phyId), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    SnapshotSetInvalidComm setInvalidCommCallback = [](bool isInvalid) {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPreProcess preProcessCallback = []() {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPostProcess postProcessCallback = []() {
        return HCCL_SUCCESS;
    };

    HcclResult ret = SnapshotControl::GetInstance(0).RegisterComm(
        "testComm", setInvalidCommCallback, preProcessCallback, postProcessCallback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).RegisterBackup("testComm", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).PostProcess();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

TEST_F(SnapshotTest, ut_Register_and_Restore_success)
{
    MOCKER(SnapShotRestoreAction).stubs().will(returnValue(HCCL_SUCCESS));

    u32 phyId = 0;
    MOCKER(SnapShotSaveAction)
        .stubs()
        .with(mockcpp::any(), outBound(phyId), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    SnapshotSetInvalidComm setInvalidCommCallback = [](bool isInvalid) {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPreProcess preProcessCallback = []() {
        return HCCL_SUCCESS;
    };
    SnapshotCheckPostProcess postProcessCallback = []() {
        return HCCL_SUCCESS;
    };

    HcclResult ret = SnapshotControl::GetInstance(0).RegisterComm(
        "testComm", setInvalidCommCallback, preProcessCallback, postProcessCallback);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).RegisterBackup("testComm", 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = SnapshotControl::GetInstance(0).Recovery();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
