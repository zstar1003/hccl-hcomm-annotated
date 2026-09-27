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
#include <slog.h>
#include <nlohmann/json.hpp>
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>

#define private public
#define protected public
#include "checkcrc.h"
#include "log.h"
#include "orion_adapter_rts.h"
#undef private
#undef protected

using namespace std;
using namespace Hccl;

class CheckCrcTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--CheckCrcTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--CheckCrcTest TearDown--\033[0m" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(CheckCrcTest, utCheckCrc1)
{
    s32 ret = HCCL_SUCCESS;
    CheckCrc src;
    CheckCrc dst;
    char str[10000];
    for (u32 i = 0; i < 10000; i++) {
        str[i] = i % 100;
    }
    u32 zeroNum = 0;
    ret = src.GetCrcNum(&zeroNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 srcCrc1;
    u32 srcCrc2;
    u32 srcCrc3;
    u32 srcNum;
    char* str1 = "123456789";
    ret = src.CalcStringCrc(str1, &srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.Calc32Crc(str, 100, &srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.Calc32Crc(str, 1000, &srcCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.Calc32Crc(str, 10000, &srcCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = src.AddCrc(srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.AddCrc(srcCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.AddCrc(srcCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = src.GetCrcNum(&srcNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 dstCrc1;
    u32 dstCrc2;
    u32 dstCrc3;
    u32 dstNum;

    ret = dst.Calc32Crc(str, 100, &dstCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = dst.Calc32Crc(str, 1000, &dstCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = dst.Calc32Crc(str, 10000, &dstCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = dst.AddCrc(dstCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = dst.AddCrc(dstCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = dst.AddCrc(dstCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = dst.GetCrcNum(&dstNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u32 srcCrcValue[3];
    u32 srcErrorCrcValue[3] = {100, 200, 300};

    ret = src.GetCrc(3, srcCrcValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char* recvBuf = new char[256];
    MOCKER(HrtMallocHost).stubs().with(mockcpp::any()).will(returnValue(static_cast<void*>(recvBuf)));
    MOCKER(HrtFreeHost).stubs().with(mockcpp::any()).will(ignoreReturnValue());
    std::string srcStr = src.GetString();
    delete[] recvBuf;

    ret = src.AddCrc(100);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.AddCrc(200);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = src.AddCrc(300);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    string checkStr = src.GetString();

    ret = src.ClearCrcInfo();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    u32 clearNum = 0xFF;
    ret = src.GetCrcNum(&clearNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(clearNum, 0);
}

const std::string RankTable4p = R"(
    {
        "version": "2.0",
        "rank_count" : "4",
        "rank_list": [
            {
                "rank_id": 0,
                "local_id": 0,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0",
                        "fabric_type": "INNER",
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 1,
                "local_id": 1,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0",
                        "fabric_type": "INNER",
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 2,
                "local_id": 2,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0",
                        "fabric_type": "INNER",
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            },
            {
                "rank_id": 3,
                "local_id": 3,
                "level_list":  [
                    {
                        "level": 0,
                        "id" : "az0-rack0",
                        "fabric_type": "INNER",
                        "rank_addr_type": "",
                        "rank_addrs": []
                    }
                ]
            }
        ]
    }
    )";

TEST_F(CheckCrcTest, GetCrc_NormalCase)
{
    CheckCrc crcChecker;
    crcChecker.crcTable_.emplace_back(0x11111111);
    crcChecker.crcTable_.emplace_back(0x22222222);
    crcChecker.crcTable_.emplace_back(0x33333333);
    u32 num = crcChecker.crcTable_.size();
    u32* crcAddr = new u32[num];
    memset(crcAddr, 0, num * sizeof(u32));

    HcclResult result = crcChecker.GetCrc(num, crcAddr);
    EXPECT_EQ(result, HCCL_SUCCESS);
    for (u32 i = 0; i < num; i++) {
        EXPECT_EQ(crcAddr[i], crcChecker.crcTable_[i]);
    }

    delete[] crcAddr;
}

TEST_F(CheckCrcTest, GetCrc_NumZero)
{
    CheckCrc crcChecker;
    u32 num = 0;
    u32* crcAddr = new u32[1]; // 分配任意大小的内存，因为num为0

    HcclResult result = crcChecker.GetCrc(num, crcAddr);
    EXPECT_EQ(result, HCCL_E_PARA);

    delete[] crcAddr;
}

TEST_F(CheckCrcTest, GetCrc_NumMismatch)
{
    CheckCrc crcChecker;
    crcChecker.crcTable_ = {0x12345678, 0x87654321, 0xABCDEF12};
    u32 num = 4;
    u32* crcAddr = new u32[num];

    HcclResult result = crcChecker.GetCrc(num, crcAddr);
    EXPECT_EQ(result, HCCL_E_INTERNAL);

    delete[] crcAddr;
}

TEST_F(CheckCrcTest, AddCrc_MultipleAdds)
{
    CheckCrc crcChecker;
    u32 crcValues[] = {0x12345678, 0x87654321, 0xABCDEF12};
    for (u32 value : crcValues) {
        crcChecker.AddCrc(value);
    }
    EXPECT_EQ(crcChecker.crcTable_.size(), 3);
    for (size_t i = 0; i < 3; i++) {
        EXPECT_EQ(crcChecker.crcTable_[i], crcValues[i]);
    }
}

TEST_F(CheckCrcTest, GetCrcNum_NormalCase)
{
    CheckCrc crcChecker;
    u32 crcValues[] = {0x12345678, 0x87654321, 0xABCDEF12};
    for (u32 value : crcValues) {
        crcChecker.AddCrc(value);
    }
    u32 num = 0;
    crcChecker.GetCrcNum(&num);
    EXPECT_EQ(num, 3);
}

TEST_F(CheckCrcTest, GetString_EmptyTable)
{
    CheckCrc crcChecker;
    std::string result = crcChecker.GetString();
    EXPECT_EQ(result, "0 ");
}

TEST_F(CheckCrcTest, GetString_OneElement)
{
    CheckCrc crcChecker;
    u32 crcValue = 0x12345678;
    crcChecker.AddCrc(crcValue);
    std::string result = crcChecker.GetString();
    EXPECT_EQ(result, "1 305419896 ");
}

TEST_F(CheckCrcTest, GetString_MultipleElements)
{
    CheckCrc crcChecker;
    u32 crcValues[] = {305419896, 2236969153, 2882400562};
    for (u32 value : crcValues) {
        crcChecker.AddCrc(value);
    }
    std::string result = crcChecker.GetString();
    EXPECT_EQ(result, "3 305419896 2236969153 2882400562 ");
}
