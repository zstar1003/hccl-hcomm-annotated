/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "gtest/gtest.h"
#include <mockcpp/mockcpp.hpp>

#include "offload_stream_manager_pub.h"
#include "stream_pub.h"
#include "sal.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>

using namespace std;
using namespace hccl;

class OffloadStreamManagerTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--OffloadStreamManagerTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--OffloadStreamManagerTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown()
    {
        std::cout << "A Test TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    OffloadStreamManager manager;
};

TEST_F(OffloadStreamManagerTest, ut_register_and_get_one_slave)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));

    s32 deviceLogicID;
    /*  设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 获取资源 */
    outStream = manager.GetSlaves("tag", 1);
    EXPECT_EQ(outStream.size(), 1);

    /*  释放资源 */
    ret = manager.ClearSlaves("tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁公共资源 */
    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_register_and_get_muti_slaves)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    for (int i = 0; i < 10; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }

    /* 设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 获取资源 */
    outStream = manager.GetSlaves("tag", 9);
    EXPECT_EQ(outStream.size(), 9);

    outStream = manager.GetSlaves("tag", 1);
    EXPECT_EQ(outStream.size(), 1);

    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_register_and_get_zero_slave)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    /*  设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 获取资源 */
    outStream = manager.GetSlaves("tag", 0);
    EXPECT_EQ(outStream.size(), 0);

    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_get_zero_slave_no_register)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    /* 获取资源 */
    outStream = manager.GetSlaves("tag", 0);
    EXPECT_EQ(outStream.size(), 0);

    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_register_slave_fail)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));

    /*  设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 重复设置资源失败 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = manager.ClearSlaves("tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(OffloadStreamManagerTest, ut_get_slave_exceed_fail)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    for (int i = 0; i < 5; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    s32 deviceLogicID;
    /* 设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 获取资源 */
    outStream = manager.GetSlaves("tag", 8);
    EXPECT_EQ(outStream.size(), 0);

    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_get_slave_invalid_tag_fail)
{
    s32 ret = HCCL_SUCCESS;
    std::vector<Stream> outStream;

    std::vector<Stream> slaves;
    for (int i = 0; i < 5; i++) {
        slaves.push_back(Stream(StreamType::STREAM_TYPE_OFFLINE));
    }
    /* 设置资源 */
    ret = manager.RegisterSlaves("tag", slaves);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 获取资源 */
    outStream = manager.GetSlaves("tag1", 5);
    EXPECT_EQ(outStream.size(), 0);

    manager.ClearSlaves();
}

TEST_F(OffloadStreamManagerTest, ut_register_and_get_master)
{
    s32 ret = HCCL_SUCCESS;
    Stream master(StreamType::STREAM_TYPE_OFFLINE);
    Stream outStream;

    ret = manager.RegisterMaster("tag", master);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 重复注册 */
    ret = manager.RegisterMaster("tag", master);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    outStream = manager.GetMaster("tag");
    EXPECT_EQ(outStream.ptr(), master.ptr());
}

TEST_F(OffloadStreamManagerTest, ut_get_unregistered_master_fail)
{
    s32 ret = HCCL_SUCCESS;
    Stream master(StreamType::STREAM_TYPE_OFFLINE);
    Stream outStream;

    ret = manager.RegisterMaster("tag", master);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 未注册的tag */
    outStream = manager.GetMaster("tag1");
    EXPECT_TRUE(outStream.ptr() == nullptr);
}
