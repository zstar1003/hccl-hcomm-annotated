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
#include <iostream>

#define private public
#define protected public

#include "base_config_legacy.h"
#include "sal.h"
#include "orion_adapter_rts.h"
#undef private
#undef protected

using namespace Hccl;
using namespace std;

class EnvConfigValidateTest : public testing::Test {
protected:
    static void SetUpTestCase() { cout << "EnvConfigValidateTest SetUP" << endl; }

    static void TearDownTestCase() { cout << "EnvConfigValidateTest TearDown" << endl; }

    virtual void SetUp() { cout << "A Test case in EnvConfigValidateTest SetUP" << endl; }

    virtual void TearDown()
    {
        GlobalMockObject::verify();
        cout << "A Test case in EnvConfigValidateTest TearDown" << endl;
    }
};

const string LONG_STRING
    = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX"
      "YZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
      "MNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmn"
      "opqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
      "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
      "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMN"
      "OPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzAB"
      "CDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnop"
      "qrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcd"
      "efghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01"
      "23456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOP"
      "QRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCD"
      "EFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqr"
      "stuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdef"
      "ghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123"
      "456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQR"
      "STUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEF"
      "GHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrst"
      "uvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefgh"
      "ijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
      "6789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRST"
      "UVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGH"
      "IJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuv"
      "wxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghij"
      "klmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ01234567"
      "89abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
      "WXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJ"
      "KLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwx"
      "yzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijkl"
      "mnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
      "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWX"
      "YZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKL"
      "MNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyz"
      "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmn"
      "opqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789ab"
      "cdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789abcde";

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_shoudl_default)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    rtsCfg.execTimeOut.Parse();
    EXPECT_EQ(rtsCfg.execTimeOut.Get(), 1836);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_shoudl_fail_when_value_too_long)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000000000")));
    EXPECT_THROW(rtsCfg.execTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_shoudl_fail_when_value_not_number)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("123abc")));
    EXPECT_THROW(rtsCfg.execTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_shoudl_success_910A3)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("300")));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_910A3));
    rtsCfg.execTimeOut.Parse();
    EXPECT_EQ(rtsCfg.execTimeOut.Get(), 300);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_should_success_when_value_0_v950)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    rtsCfg.execTimeOut.Parse();
    EXPECT_EQ(rtsCfg.execTimeOut.Get(), 0);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_should_success_when_value_2147483647_v950)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("2147483587")));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    rtsCfg.execTimeOut.Parse();
    EXPECT_EQ(rtsCfg.execTimeOut.Get(), 2147483587);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_EXEC_TIMEOUT_shoudl_success_v950)
{
    EnvRtsConfig rtsCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("300")));
    MOCKER(HrtGetDeviceType).stubs().will(returnValue((DevType)DevType::DEV_TYPE_950));
    rtsCfg.execTimeOut.Parse();
    EXPECT_EQ(rtsCfg.execTimeOut.Get(), 300);
}

TEST_F(EnvConfigValidateTest, Ut_EnvConfig_When_HCCL_DFS_CONFIG_Set_TaskExceptionOff_Expect_GetDfsConfig_ReturnIsFalse)
{
    EnvLogConfig logCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("task_exception:off")));
    logCfg.dfsConfig.Parse();
    EXPECT_EQ(logCfg.dfsConfig.Get().taskExceptionEnable, false);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_BUFFSIZE_shoudl_default)
{
    EnvAlgoConfig algoCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    algoCfg.bufferSize.Parse();
    EXPECT_EQ(algoCfg.bufferSize.Get(), 200 * 1024 * 1024);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_BUFFSIZE_shoudl_fail_when_value_too_long)
{
    EnvAlgoConfig algoCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000000000")));
    EXPECT_THROW(algoCfg.bufferSize.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_BUFFSIZE_shoudl_fail_when_value_not_number)
{
    EnvAlgoConfig algoCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("123abc")));
    EXPECT_THROW(algoCfg.bufferSize.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_BUFFSIZE_should_fail_when_value_0)
{
    EnvAlgoConfig algoCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_THROW(algoCfg.bufferSize.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_BUFFSIZE_should_success)
{
    EnvAlgoConfig algoCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("300")));
    algoCfg.bufferSize.Parse();
    EXPECT_EQ(algoCfg.bufferSize.Get(), 300 * 1024 * 1024);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_FILE_should_default)
{
    EnvHostNicConfig hostNicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    hostNicCfg.hcclWhiteListFile.Parse();
    EXPECT_EQ(hostNicCfg.hcclWhiteListFile.Get(), "");
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_FILE_should_fail_when_too_long)
{
    EnvHostNicConfig hostNicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(LONG_STRING));
    EXPECT_THROW(hostNicCfg.hcclWhiteListFile.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_FILE_should_fail_when_file_not_exist)
{
    EnvHostNicConfig hostNicCfg{};
    MOCKER(SalGetEnv).stubs().will(
        returnValue(string(HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/env_config/not_exist_file")));
    EXPECT_THROW(hostNicCfg.hcclWhiteListFile.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_FILE_should_success_when_file_exist)
{
    EnvHostNicConfig hostNicCfg{};
    string filePath{HCOMM_CODE_ROOT_DIR "/test/legacy/ut/framework/env_config/test_file.txt"};
    MOCKER(SalGetEnv).stubs().will(returnValue(filePath));
    hostNicCfg.hcclWhiteListFile.Parse();
    string absPath = hostNicCfg.hcclWhiteListFile.Get();
    EXPECT_TRUE(absPath.size() >= filePath.size());
    EXPECT_EQ(
        absPath.compare(absPath.size() - filePath.size(), filePath.size(), filePath), 0); // check endwith filePath
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_SL_should_default)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    rdmaCfg.rdmaServerLevel.Parse();
    EXPECT_EQ(rdmaCfg.rdmaServerLevel.Get(), 4);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_SL_should_fail_when_value_too_long)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000000000")));
    EXPECT_THROW(rdmaCfg.rdmaServerLevel.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_SL_shoudl_fail_when_value_not_number)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string(" 5")));
    EXPECT_THROW(rdmaCfg.rdmaServerLevel.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_SL_shoudl_fail_when_value_too_big)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("8")));
    EXPECT_THROW(rdmaCfg.rdmaServerLevel.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_SL_should_success)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("5")));
    rdmaCfg.rdmaServerLevel.Parse();
    EXPECT_EQ(rdmaCfg.rdmaServerLevel.Get(), 5);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_should_default)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 20);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_should_fail_when_value_too_long)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000000000")));
    EXPECT_THROW(rdmaCfg.rdmaTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_shoudl_fail_when_value_not_number)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("123abc")));
    EXPECT_THROW(rdmaCfg.rdmaTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_shoudl_fail_when_value_is_negative)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("-1")));
    EXPECT_THROW(rdmaCfg.rdmaTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_should_success_when_value_25)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("25")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 25);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_should_success_when_value_32)
{
    // >=31时设置为0，表示永不超时
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("32")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 0);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TIMEOUT_should_success)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("15")));
    rdmaCfg.rdmaTimeOut.Parse();
    EXPECT_EQ(rdmaCfg.rdmaTimeOut.Get(), 15);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_should_default)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    rdmaCfg.rdmaRetryCnt.Parse();
    EXPECT_EQ(rdmaCfg.rdmaRetryCnt.Get(), 7);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_should_fail_when_value_too_long)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000000000")));
    EXPECT_THROW(rdmaCfg.rdmaRetryCnt.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_shoudl_fail_when_value_not_number)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("123abc")));
    EXPECT_THROW(rdmaCfg.rdmaRetryCnt.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_shoudl_fail_when_value_too_small)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_THROW(rdmaCfg.rdmaRetryCnt.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_shoudl_fail_when_value_too_big)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("8")));
    EXPECT_THROW(rdmaCfg.rdmaRetryCnt.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_RETRY_CNT_should_success)
{
    EnvRdmaConfig rdmaCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("5")));
    rdmaCfg.rdmaRetryCnt.Parse();
    EXPECT_EQ(rdmaCfg.rdmaRetryCnt.Get(), 5);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_IP_should_success_when_input_default)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.hcclIfIp.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_IP_should_success_with_valid_ipv4)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("127.0.0.1")));
    EXPECT_NO_THROW(nicCfg.hcclIfIp.Parse());
    EXPECT_EQ(nicCfg.hcclIfIp.Get().GetIpStr(), "127.0.0.1");
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_IP_should_fail_with_invalid_ipv4)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("127.0.0.1.1")));
    EXPECT_THROW(nicCfg.hcclIfIp.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_IP_should_success_with_valid_ipv6)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("2001:db8:85a3::8a2e:370:7334")));
    EXPECT_NO_THROW(nicCfg.hcclIfIp.Parse());
    EXPECT_EQ(nicCfg.hcclIfIp.Get().GetIpStr(), "2001:db8:85a3::8a2e:370:7334");
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_IP_should_fail_with_invalid_ipv4_range)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("999.0.0.1")));
    EXPECT_THROW(nicCfg.hcclIfIp.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_BASE_PORT_should_success_when_input_default)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.hcclIfBasePort.Parse());
    EXPECT_EQ(nicCfg.hcclIfBasePort.Get(), 65536);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_BASE_PORT_should_success_with_valid_value)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("10000")));
    EXPECT_NO_THROW(nicCfg.hcclIfBasePort.Parse());
    EXPECT_EQ(nicCfg.hcclIfBasePort.Get(), 10000);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_BASE_PORT_should_fail_with_invalid_zero)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_THROW(nicCfg.hcclIfBasePort.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_BASE_PORT_should_fail_with_invalid_string)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("test")));
    EXPECT_THROW(nicCfg.hcclIfBasePort.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_IF_BASE_PORT_should_fail_with_invalid_large_value)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("100000000000")));
    EXPECT_THROW(nicCfg.hcclIfBasePort.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_IFNAME_should_success_when_input_default)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.hcclSocketIfName.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_FAMILY_should_success_when_input_default)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.hcclSocketFamily.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_FAMILY_should_success_with_AF_INET)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("AF_INET")));
    EXPECT_NO_THROW(nicCfg.hcclSocketFamily.Parse());
    EXPECT_EQ(nicCfg.hcclSocketFamily.Get(), AF_INET);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_FAMILY_should_success_with_AF_INET6)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("AF_INET6")));
    EXPECT_NO_THROW(nicCfg.hcclSocketFamily.Parse());
    EXPECT_EQ(nicCfg.hcclSocketFamily.Get(), AF_INET6);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_FAMILY_should_fail_with_invalid_family)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("AF_INET1")));
    EXPECT_THROW(nicCfg.hcclSocketFamily.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_SOCKET_FAMILY_should_fail_with_invalid_value)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("123456")));
    EXPECT_THROW(nicCfg.hcclSocketFamily.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_CONNECT_TIMEOUT_should_success_when_input_default)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.linkTimeOut.Parse());
    EXPECT_EQ(nicCfg.linkTimeOut.Get(), 120);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_CONNECT_TIMEOUT_should_success_with_valid_value)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1000")));
    EXPECT_NO_THROW(nicCfg.linkTimeOut.Parse());
    EXPECT_EQ(nicCfg.linkTimeOut.Get(), 1000);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_CONNECT_TIMEOUT_should_success_with_another_valid_value)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1200")));
    EXPECT_NO_THROW(nicCfg.linkTimeOut.Parse());
    EXPECT_EQ(nicCfg.linkTimeOut.Get(), 1200);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_CONNECT_TIMEOUT_should_fail_with_invalid_string)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("test")));
    EXPECT_THROW(nicCfg.linkTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_CONNECT_TIMEOUT_should_fail_with_invalid_large_value)
{
    EnvSocketConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1234567891011")));
    EXPECT_THROW(nicCfg.linkTimeOut.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_DISABLE_should_success_when_input_default)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.whitelistDisable.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_DISABLE_should_success_with_1)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1")));
    EXPECT_NO_THROW(nicCfg.whitelistDisable.Parse());
    EXPECT_EQ(nicCfg.whitelistDisable.Get(), 1);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_DISABLE_should_success_with_0)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_NO_THROW(nicCfg.whitelistDisable.Parse());
    EXPECT_EQ(nicCfg.whitelistDisable.Get(), 0);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_DISABLE_should_fail_with_invalid_string)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("test")));
    EXPECT_THROW(nicCfg.whitelistDisable.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_WHITELIST_DISABLE_should_fail_with_invalid_large_value)
{
    EnvHostNicConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1234567891011")));
    EXPECT_THROW(nicCfg.whitelistDisable.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_ENTRY_LOG_ENABLE_should_success_when_input_default)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.entryLogEnable.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_ENTRY_LOG_ENABLE_should_success_with_1)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1")));
    EXPECT_NO_THROW(nicCfg.entryLogEnable.Parse());
    EXPECT_EQ(nicCfg.entryLogEnable.Get(), 1);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_ENTRY_LOG_ENABLE_should_success_with_0)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_NO_THROW(nicCfg.entryLogEnable.Parse());
    EXPECT_EQ(nicCfg.entryLogEnable.Get(), 0);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_ENTRY_LOG_ENABLE_should_fail_with_invalid_string)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("test")));
    EXPECT_THROW(nicCfg.entryLogEnable.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_ENTRY_LOG_ENABLE_should_fail_with_invalid_large_value)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("1234567891011")));
    EXPECT_THROW(nicCfg.entryLogEnable.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TC_should_success_when_input_default)
{
    EnvRdmaConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    EXPECT_NO_THROW(nicCfg.rdmaTrafficClass.Parse());
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TC_should_success_with_valid_value)
{
    EnvRdmaConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("4")));
    EXPECT_NO_THROW(nicCfg.rdmaTrafficClass.Parse());
    EXPECT_EQ(nicCfg.rdmaTrafficClass.Get(), 4);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TC_should_success_with_zero)
{
    EnvRdmaConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("0")));
    EXPECT_NO_THROW(nicCfg.rdmaTrafficClass.Parse());
    EXPECT_EQ(nicCfg.rdmaTrafficClass.Get(), 0);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TC_should_fail_with_invalid_string)
{
    EnvRdmaConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("test")));
    EXPECT_THROW(nicCfg.rdmaTrafficClass.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_RDMA_TC_should_fail_with_invalid_value)
{
    EnvRdmaConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("3")));
    EXPECT_THROW(nicCfg.rdmaTrafficClass.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_DFS_CONFIG_shoud_default)
{
    EnvLogConfig logCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("")));
    logCfg.dfsConfig.Parse();
    EXPECT_EQ(logCfg.dfsConfig.Get().taskExceptionEnable, true);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_DFS_CONFIG_task_exception_shoud_false)
{
    EnvLogConfig logCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("task_exception:off")));
    logCfg.dfsConfig.Parse();
    EXPECT_EQ(logCfg.dfsConfig.Get().taskExceptionEnable, false);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_DFS_CONFIG_should_fail_with_invalid_value)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("!")));
    EXPECT_THROW(nicCfg.dfsConfig.Parse(), InvalidParamsException);
}

TEST_F(EnvConfigValidateTest, test_parse_HCCL_DFS_CONFIG_should_fail_with_task_exception_invalid_value)
{
    EnvLogConfig nicCfg{};
    MOCKER(SalGetEnv).stubs().will(returnValue(string("task_exception:~")));
    EXPECT_THROW(nicCfg.dfsConfig.Parse(), InvalidParamsException);
}
