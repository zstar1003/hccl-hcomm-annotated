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
#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>
#define private public
#include "communicator_impl.h"
#include "data_buf_manager.h"
#include "internal_exception.h"
#include "rmt_data_buffer_mgr.h"
#include "mem_transport_lite.h"
#include "mem_transport_lite_mgr.h"
#include "port.h"
#undef private

using namespace Hccl;

class RmtDataBufferMgrTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RmtDataBufferMgr tests set up." << std::endl; }

    static void TearDownTestCase() { std::cout << "RmtDataBufferMgr tests tear down." << std::endl; }

    virtual void SetUp()
    {
        mirrorTaskMgrLite = new MirrorTaskManagerLite();
        memTransportLiteMgr = new MemTransportLiteMgr(mirrorTaskMgrLite);
        algInfo = new CollAlgInfo(mode, tag);
        std::cout << "A Test case in RmtDataBufferMgr SetUP" << std::endl;
    }

    virtual void TearDown()
    {
        delete memTransportLiteMgr;
        delete mirrorTaskMgrLite;
        delete algInfo;
        GlobalMockObject::verify();
        std::cout << "A Test case in RmtDataBufferMgr TearDown" << std::endl;
    }

    MemTransportLiteMgr* memTransportLiteMgr;
    MirrorTaskManagerLite* mirrorTaskMgrLite;
    CollAlgInfo* algInfo;
    std::string tag = "tag";
    OpMode mode{OpMode::OPBASE};
};

// 测试在opBase场景下，GetBuffer能够正确调用MemTransportLite的GetRmtBuffer并返回正确结果
TEST_F(RmtDataBufferMgrTest, get_GetBuffer_opbase_success)
{
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);

    // Mocker掉MemTransportLiteMgr::GetOpbase，使其返回一个非空的MemTransportLite指针
    MemTransportLite* fakeTransportPtr = reinterpret_cast<MemTransportLite*>(0x1000);
    MOCKER_CPP(&MemTransportLiteMgr::GetOpbase).stubs().with(mockcpp::any()).will(returnValue(fakeTransportPtr));

    // Mocker掉MemTransportLite::GetRmtBuffer，使其返回一个预设的Buffer对象
    Buffer expectedBuffer(0x300, 200);
    MOCKER_CPP(&MemTransportLite::GetRmtBuffer).stubs().with(mockcpp::any()).will(returnValue(expectedBuffer));

    // 构造RmtDataBufferMgr并调用GetBuffer，验证返回结果与预设的Buffer对象相同
    RmtDataBufferMgr rmtDataBufferMgr(memTransportLiteMgr, algInfo);
    DataBuffer result = rmtDataBufferMgr.GetBuffer(linkData, BufferType::INPUT);
    EXPECT_EQ(result.GetAddr(), 0x300);
    EXPECT_EQ(result.GetSize(), static_cast<size_t>(200));
}

TEST_F(RmtDataBufferMgrTest, get_GetBuffer_opbase_fail)
{
    LinkData linkData(BasePortType(PortDeploymentType::DEV_NET, ConnectProtoType::UB), 0, 1, 0, 1);

    // Mocker掉MemTransportLiteMgr::GetOpbase，使其返回一个空指针，模拟未找到对应的MemTransportLite的情况
    RmtDataBufferMgr rmtDataBufferMgr(memTransportLiteMgr, algInfo);
    EXPECT_THROW(rmtDataBufferMgr.GetBuffer(linkData, BufferType::INPUT), NullPtrException);
}
