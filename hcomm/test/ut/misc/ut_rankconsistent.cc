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
#include "mem_host_pub.h"
#include <string>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "llt_hccl_stub_sal_pub.h"
#include "calc_crc.h"

#define private public
#include "rank_consistentcy_checker.h"
#undef private

using namespace std;
using namespace hccl;

class RankConSistentTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--RankConsistentcyChecker SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--RankConsistentcyChecker TearDown--\033[0m" << std::endl; }
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

#if 1 // 这个用例其实没什么意义
TEST_F(RankConSistentTest, ut_Compare_Cann_Version)
{
    RankConsistentcyChecker* implObj = new RankConsistentcyChecker();
    implObj->SetCheckCannVersionSwitch(true);

    bool ret;
    HcclCheckInfo checkInfo;
    HcclCheckInfo checkInfoRecv;

    // 一端版本为"1.83.T6.0.B114", 另一端为""
    std::string cannVersion1 = "1.83.T6.0.B114";
    strncpy_s(checkInfo.version, MAX_CANN_VERSION_LEN, cannVersion1.c_str(), cannVersion1.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, false);

    // 一端版本为"1.83.T6.0.B114", 另一端为"1.83.T6.0.B115"
    std::string cannVersion2 = "1.83.T6.0.B115";
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, cannVersion2.c_str(), cannVersion2.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);

    // 两端均为"1.83.T6.0.B114"
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, cannVersion1.c_str(), cannVersion1.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, false);

    implObj->SetCheckCannVersionSwitch(false);

    delete implObj;
}

TEST_F(RankConSistentTest, ut_Compare_Toolkit_And_Ops_Version)
{
    RankConsistentcyChecker* implObj = new RankConsistentcyChecker();
    implObj->SetCheckCannVersionSwitch(true);

    bool ret;
    HcclCheckInfo checkInfo;
    HcclCheckInfo checkInfoRecv;
    std::string oldVersion = "9.0.0";
    std::string newVersion = "9.1.0";
    std::string delimiter = "_";

    // 一端版本为"9.1.0_9.1.0", 另一端为"9.0.0_9.1.0"
    std::string localVersion = newVersion + delimiter + newVersion;
    std::string remoteVersion = oldVersion + delimiter + newVersion;
    strncpy_s(checkInfo.version, MAX_CANN_VERSION_LEN, localVersion.c_str(), localVersion.size());
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, remoteVersion.c_str(), remoteVersion.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);

    // 一端版本为"9.1.0_9.1.0", 另一端为"9.1.0_9.0.0"
    remoteVersion = oldVersion + delimiter + newVersion;
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, remoteVersion.c_str(), remoteVersion.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);

    // 一端版本为"9.1.0_9.1.0", 另一端为"9.0.0_9.0.0"
    remoteVersion = oldVersion + delimiter + oldVersion;
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, remoteVersion.c_str(), remoteVersion.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);

    // 两端均为"9.1.0_9.1.0"
    remoteVersion = newVersion + delimiter + newVersion;
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, remoteVersion.c_str(), remoteVersion.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, false);

    // 一端版本为"9.1.0_9.1.0", 另一端为"9.1.0_9.1.0_9.1.0"
    remoteVersion = newVersion + delimiter + newVersion + delimiter + newVersion;
    strncpy_s(checkInfoRecv.version, MAX_CANN_VERSION_LEN, remoteVersion.c_str(), remoteVersion.size());
    ret = implObj->CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);

    implObj->SetCheckCannVersionSwitch(false);

    delete implObj;
}
#endif

TEST_F(RankConSistentTest, ut_Compare_check_error)
{
    RankConsistentcyChecker consistent;
    HcclCheckInfo checkInfo;
    HcclCheckInfo checkInfoRecv;

    HcclCRCInfo crcInfo;
    crcInfo.configFileExist_ = false;
    crcInfo.crcNum = 1;
    crcInfo.crcArray[0] = 0;
    HcclCMDInfo cmdInfo;
    cmdInfo.cmdType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    cmdInfo.count = 1024;
    cmdInfo.dataType = HCCL_DATA_TYPE_INT8;
    checkInfo.crcInfoGlobal = crcInfo;
    checkInfo.crcInfoOp = crcInfo;
    checkInfo.cmdInfo = cmdInfo;
    checkInfo.protocolType = ProtocolType::RDMA;

    crcInfo.configFileExist_ = true;
    crcInfo.crcNum = 1;
    crcInfo.crcArray[0] = 2;

    cmdInfo.cmdType = HcclCMDType::HCCL_CMD_BROADCAST;
    cmdInfo.count = 2048;
    cmdInfo.dataType = HCCL_DATA_TYPE_INT16;
    checkInfoRecv.crcInfoGlobal = crcInfo;
    checkInfoRecv.crcInfoOp = crcInfo;
    checkInfoRecv.cmdInfo = cmdInfo;
    checkInfoRecv.protocolType = ProtocolType::TCP;

    bool ret = consistent.CompareFrame(checkInfo, checkInfoRecv);
    EXPECT_EQ(ret, true);
}

TEST_F(RankConSistentTest, ut_crc)
{
    RankConsistentcyChecker consistentSrc;
    RankConsistentcyChecker consistentDst;
    char str[10000];
    for (u32 i = 0; i < 10000; i++) {
        str[i] = i % 100;
    }

    u32 srcCrc1;
    u32 srcCrc2;
    u32 srcCrc3;
    std::string str1(str, 100);
    HcclResult ret = CalcCrc::HcclCalcCrc(str1.c_str(), str1.length(), srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string str2(str, 1000);
    ret = CalcCrc::HcclCalcCrc(str2.c_str(), str2.length(), srcCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    std::string str3(str, 10000);
    ret = CalcCrc::HcclCalcCrc(str3.c_str(), str3.length(), srcCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = consistentSrc.AddCrc(srcCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentSrc.AddCrc(srcCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentSrc.AddCrc(srcCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("srcCrc1[%d], srcCrc2[%d], srcCrc3[%d]", srcCrc1, srcCrc2, srcCrc3);

    u32 dstCrc1;
    u32 dstCrc2;
    u32 dstCrc3;

    ret = CalcCrc::HcclCalcCrc(str1.c_str(), str1.length(), dstCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CalcCrc::HcclCalcCrc(str2.c_str(), str2.length(), dstCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = CalcCrc::HcclCalcCrc(str3.c_str(), str3.length(), dstCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = consistentDst.AddCrc(dstCrc1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentDst.AddCrc(dstCrc2);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentDst.AddCrc(dstCrc3);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("dstCrc1[%d], dstCrc2[%d], dstCrc3[%d]", dstCrc1, dstCrc2, dstCrc3);

    u32 srcCrcValue[3];
    u32 srcErrorCrcValue[3] = {100, 200, 300};

    ret = consistentSrc.GetCrc(3, srcCrcValue);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HCCL_INFO("Get src crc value: value1[%d], value2[%d], value3[%d]", srcCrcValue[0], srcCrcValue[1], srcCrcValue[2]);

    ret = consistentSrc.AddCrc(100);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentSrc.AddCrc(200);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = consistentSrc.AddCrc(300);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = consistentSrc.ClearCrcInfo();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}
