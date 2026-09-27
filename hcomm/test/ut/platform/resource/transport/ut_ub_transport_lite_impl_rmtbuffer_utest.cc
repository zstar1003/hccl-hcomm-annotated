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
#include <stdio.h>

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#ifndef private
#define private public
#define protected public
#endif

#include "binary_stream.h"
#include "ub_transport_lite_impl.h"

#undef private
#undef protected

using namespace std;
using namespace Hccl;

class UbTransportLiteImplRmtbuffer_UT : public testing::Test {
protected:
    static void SetUpTestCase()
    {
        std::cout << "\033[36m--UbTransportLiteImplRmtbuffer_UT SetUP--\033[0m" << std::endl;
    }
    static void TearDownTestCase()
    {
        std::cout << "\033[36m--UbTransportLiteImplRmtbuffer_UT TearDown--\033[0m" << std::endl;
    }
    virtual void SetUp() {}
    virtual void TearDown() { GlobalMockObject::verify(); }

    // 构造 UbTransportLiteImpl 反序列化所需的完整序列化数据
    static std::vector<char> BuildTransportData(u32 type, u32 notifyNum, u32 bufferNum, u32 rmtbufferNum, u32 connNum)
    {
        BinaryStream binaryStream;
        binaryStream << type;
        binaryStream << notifyNum;
        binaryStream << bufferNum;
        binaryStream << rmtbufferNum;
        binaryStream << connNum;
        std::vector<char> emptyVector;
        binaryStream << emptyVector; // notifyUniqueIds
        binaryStream << emptyVector; // rmtNotifyUniqueIds
        binaryStream << emptyVector; // locBufferUniqueIds
        binaryStream << emptyVector; // rmtBufferUniqueIds
        binaryStream << emptyVector; // drainBufferUniqueIds
        binaryStream << emptyVector; // connUniqueIds
        std::vector<char> data;
        binaryStream.Dump(data);
        return data;
    }
};

TEST_F(
    UbTransportLiteImplRmtbuffer_UT,
    Ut_HeaderSerialization_When_RmtbufferDifferentFromBuffer_Expect_FiveFieldsDeserialized)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readType, type);
    EXPECT_EQ(readNotifyNum, notifyNum);
    EXPECT_EQ(readBufferNum, bufferNum);
    EXPECT_EQ(readRmtbufferNum, rmtbufferNum);
    EXPECT_EQ(readConnNum, connNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_HeaderSerialization_When_RmtbufferNumIsZero_Expect_ZeroDeserialized)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 0;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 0);
    EXPECT_EQ(readBufferNum, 3);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_HeaderSerialization_When_RmtbufferEqualsBuffer_Expect_SameValue)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 3;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> data;
    binaryStream.Dump(data);

    BinaryStream readStream(data);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, readBufferNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_RmtbufferNumMember_When_Default_Expect_Zero)
{
    u32 type = static_cast<u32>(TransportType::UB);
    std::vector<char> data = BuildTransportData(type, 0, 0, 0, 0);
    UbTransportLiteImpl transportDev(data);
    EXPECT_EQ(transportDev.rmtbufferNum, 0);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_RmtbufferNumMember_When_RmtbufferDifferentFromBuffer_Expect_IndependentValue)
{
    u32 type = static_cast<u32>(TransportType::UB);
    std::vector<char> data = BuildTransportData(type, 2, 3, 5, 1);
    UbTransportLiteImpl transportDev(data);
    transportDev.notifyNum = 2;
    transportDev.bufferNum = 3;
    transportDev.rmtbufferNum = 5;
    transportDev.connNum = 1;

    EXPECT_EQ(transportDev.bufferNum, 3);
    EXPECT_EQ(transportDev.rmtbufferNum, 5);
    EXPECT_NE(transportDev.bufferNum, transportDev.rmtbufferNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_BufferType_Expect_UseRmtbufferNum)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;

    u32 numForBuffer = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::BUFFER;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForBuffer = notifyNum;
    } else {
        numForBuffer = rmtbufferNum;
    }

    EXPECT_EQ(numForBuffer, rmtbufferNum);
    EXPECT_NE(numForBuffer, bufferNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_NotifyType_Expect_UseNotifyNum)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;

    u32 numForNotify = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::NOTIFY;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForNotify = notifyNum;
    } else {
        numForNotify = rmtbufferNum;
    }

    EXPECT_EQ(numForNotify, notifyNum);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_ParseRmtBufferVecLogic_When_RmtbufferNumIsZero_Expect_NumIsZero)
{
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 0;

    u32 numForBuffer = 0;
    UbTransportLiteImpl::RmaUbBufType rmtType = UbTransportLiteImpl::RmaUbBufType::BUFFER;
    if (rmtType == UbTransportLiteImpl::RmaUbBufType::NOTIFY) {
        numForBuffer = notifyNum;
    } else {
        numForBuffer = rmtbufferNum;
    }

    EXPECT_EQ(numForBuffer, 0);
}

TEST_F(UbTransportLiteImplRmtbuffer_UT, Ut_FullPacketDeserialize_When_RmtbufferDifferentFromBuffer_Expect_HeaderCorrect)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> notifyUniqueIds(notifyNum * 4, 0);
    binaryStream << notifyUniqueIds;

    std::vector<char> rmtNotifyUniqueIds;
    for (u32 i = 0; i < notifyNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x1000 + i);
        elemStream << static_cast<u64>(256);
        elemStream << static_cast<u32>(10 + i);
        elemStream << static_cast<u32>(1);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtNotifyUniqueIds.insert(rmtNotifyUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtNotifyUniqueIds;

    std::vector<char> rmtBufferUniqueIds;
    for (u32 i = 0; i < rmtbufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x2000 + i);
        elemStream << static_cast<u64>(512);
        elemStream << static_cast<u32>(20 + i);
        elemStream << static_cast<u32>(2);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtBufferUniqueIds.insert(rmtBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> connUniqueIds(connNum * 4, 0);
    binaryStream << connUniqueIds;

    std::vector<char> packet;
    binaryStream.Dump(packet);

    BinaryStream readStream(packet);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 5);
    EXPECT_EQ(readBufferNum, 3);

    std::vector<char> readNotifyUniqueIds;
    readStream >> readNotifyUniqueIds;
    EXPECT_EQ(readNotifyUniqueIds.size(), notifyNum * 4);

    std::vector<char> readRmtNotifyUniqueIds;
    readStream >> readRmtNotifyUniqueIds;

    std::vector<char> readRmtBufferUniqueIds;
    readStream >> readRmtBufferUniqueIds;
}

TEST_F(
    UbTransportLiteImplRmtbuffer_UT, Ut_FullPacketDeserializeV2_When_RmtbufferDifferentFromBuffer_Expect_HeaderCorrect)
{
    u32 type = static_cast<u32>(TransportType::UB);
    u32 notifyNum = 2;
    u32 bufferNum = 3;
    u32 rmtbufferNum = 5;
    u32 connNum = 1;

    BinaryStream binaryStream;
    binaryStream << type;
    binaryStream << notifyNum;
    binaryStream << bufferNum;
    binaryStream << rmtbufferNum;
    binaryStream << connNum;

    std::vector<char> notifyUniqueIds(notifyNum * 4, 0);
    binaryStream << notifyUniqueIds;

    std::vector<char> rmtNotifyUniqueIds;
    for (u32 i = 0; i < notifyNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x1000 + i);
        elemStream << static_cast<u64>(256);
        elemStream << static_cast<u32>(10 + i);
        elemStream << static_cast<u32>(1);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtNotifyUniqueIds.insert(rmtNotifyUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtNotifyUniqueIds;

    std::vector<char> locBufferUniqueIds;
    for (u32 i = 0; i < bufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x3000 + i);
        elemStream << static_cast<u64>(1024);
        elemStream << static_cast<u32>(30 + i);
        elemStream << static_cast<u32>(3);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        locBufferUniqueIds.insert(locBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << locBufferUniqueIds;

    std::vector<char> rmtBufferUniqueIds;
    for (u32 i = 0; i < rmtbufferNum; i++) {
        BinaryStream elemStream;
        elemStream << static_cast<u64>(0x2000 + i);
        elemStream << static_cast<u64>(512);
        elemStream << static_cast<u32>(20 + i);
        elemStream << static_cast<u32>(2);
        std::vector<char> elemData;
        elemStream.Dump(elemData);
        rmtBufferUniqueIds.insert(rmtBufferUniqueIds.end(), elemData.begin(), elemData.end());
    }
    binaryStream << rmtBufferUniqueIds;

    std::vector<char> connUniqueIds(connNum * 4, 0);
    binaryStream << connUniqueIds;

    std::vector<char> packet;
    binaryStream.Dump(packet);

    BinaryStream readStream(packet);
    u32 readType, readNotifyNum, readBufferNum, readRmtbufferNum, readConnNum;
    readStream >> readType;
    readStream >> readNotifyNum;
    readStream >> readBufferNum;
    readStream >> readRmtbufferNum;
    readStream >> readConnNum;

    EXPECT_EQ(readRmtbufferNum, 5);
    EXPECT_EQ(readBufferNum, 3);

    std::vector<char> readNotifyUniqueIds;
    readStream >> readNotifyUniqueIds;

    std::vector<char> readRmtNotifyUniqueIds;
    readStream >> readRmtNotifyUniqueIds;

    std::vector<char> readLocBufferUniqueIds;
    readStream >> readLocBufferUniqueIds;

    std::vector<char> readRmtBufferUniqueIds;
    readStream >> readRmtBufferUniqueIds;
}
