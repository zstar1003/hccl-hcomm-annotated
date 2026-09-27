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

#include <externalinput_pub.h>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <sal.h>

#include <iostream>
#include <fstream>
#include "comm.h"

#include "externalinput.h"
#include "adapter_rts_common.h"
#include "env_config.h"
#include "config_log.h"
#include "config_plf_log.h"

using namespace std;
using namespace hccl;

class ExternalInputTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "ExternalInputTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "ExternalInputTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = -1;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(ExternalInputTest, ut_external_input_env_variables_params_intra_comm_type)
{
    u32 intraRoce = 0;
    HcclResult ret;
    // 不初始化环境变量，为默认值，pcie:1 roce:0
    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:0 roce:0，报warning走pcie，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:0 roce:1，走roce，pcie:0 roce:1
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 1 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:1 roce:0，走pcie，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:1 roce:1，暂不支持，报错，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "1", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:-1 roce:2，异常值，报错，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "-1", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "2", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:2 roce:-1，异常值，报错，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "2", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "-1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:abc roce:1，异常值，报错，pcie:1 roce:0
    setenv("HCCL_INTRA_PCIE_ENABLE", "abc", 1);
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 取消环境变量，则默认为pcie:1, roce:0
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化roce:0，异常值，报错，pcie:1 roce:0
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:0，异常值，报错，pcie:1 roce:0
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    setenv("HCCL_INTRA_PCIE_ENABLE", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化roce:1，pcie:0 roce:1
    unsetenv("HCCL_INTRA_PCIE_ENABLE");
    setenv("HCCL_INTRA_ROCE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 1 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化pcie:1，异常值，报错，pcie:1 roce:0
    unsetenv("HCCL_INTRA_ROCE_ENABLE");
    setenv("HCCL_INTRA_PCIE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    intraRoce = GetExternalInputIntraRoceSwitch();
    HCCL_INFO("the intraRoce is %u", intraRoce);
    intraRoce == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    SetFftsSwitch(true);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    auto fftsSwitch = GetExternalInputHcclEnableFfts();
    EXPECT_EQ(true, fftsSwitch);
    SetFftsSwitch(false);
    InitEnvVarParam();
}
TEST_F(ExternalInputTest, ut_external_input_env_variables_taskExceptionSwitch)
{
    u32 taskExceptionSwitch = 0;
    HcclResult ret;
    // 不初始化环境变量，为默认值，taskExceptionSwitch:0

    taskExceptionSwitch = GetExternalInputTaskExceptionSwitch();
    HCCL_INFO("the taskExceptionSwitch is %u", taskExceptionSwitch);
    taskExceptionSwitch == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化taskExceptionSwitch:1，设置成功
    ResetInitState();
    setenv("HCCL_DIAGNOSE_ENABLE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    taskExceptionSwitch = GetExternalInputTaskExceptionSwitch();
    HCCL_INFO("the taskExceptionSwitch is %u", taskExceptionSwitch);
    taskExceptionSwitch == 1 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 初始化taskExceptionSwitch:abc, 报错
    ResetInitState();
    setenv("HCCL_DIAGNOSE_ENABLE", "abc", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_DIAGNOSE_ENABLE");
}

// 自定义端口hcclIfBasePort的st测试
TEST_F(ExternalInputTest, ut_external_input_env_variables_port)
{
    u32 baseport;
    HcclResult ret;

    // eg1：不初始化环境变量，为默认值，port=HCCL_INVALID_IF_BASE_PORT
    baseport = GetExternalInputHcclIfBasePort();
    HCCL_INFO("the base port is %u", baseport);
    ret = ((baseport == HCCL_INVALID_PORT) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // eg2：初始化port=0，异常值，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg3：初始化port=10000，走port=10000
    setenv("HCCL_IF_BASE_PORT", "10000", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    baseport = GetExternalInputHcclIfBasePort();
    HCCL_INFO("the base port is %u", baseport);
    ret = ((baseport == 10000) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // eg4：初始化port=30000，走port=30000
    setenv("HCCL_IF_BASE_PORT", "30000", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    baseport = GetExternalInputHcclIfBasePort();
    HCCL_INFO("the base port is %u", baseport);
    ret = ((baseport == 30000) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // eg5：初始化port=65520，走port=65520
    setenv("HCCL_IF_BASE_PORT", "65520", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    baseport = GetExternalInputHcclIfBasePort();
    HCCL_INFO("the base port is %u", baseport);
    ret = ((baseport == 65520) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // eg6：初始化port=65535，异常值，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "65535", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg7：初始化port=-1，异常值，报错，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "-1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg8：初始化port=0xfffffff，异常值，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "0xfffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg9：初始化port="test"，异常值，报错，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg10：初始化port=0xfff，走port=65520
    setenv("HCCL_IF_BASE_PORT", "0xfff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg11：初始化port=4294967295，走port=60000
    setenv("HCCL_IF_BASE_PORT", "4294967295", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg12：初始化port=1023，异常值，报错，走port=HOST_CONTROL_BASE_PORT
    setenv("HCCL_IF_BASE_PORT", "1023", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // eg13：取消环境变量，则默认走port=HCCL_INVALID_IF_BASE_PORT
    unsetenv("HCCL_IF_BASE_PORT");
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    baseport = GetExternalInputHcclIfBasePort();
    HCCL_INFO("the base port is %u", baseport);
    ret = ((baseport == HCCL_INVALID_PORT) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_params_exec_timeout)
{
    int timeout;
    HcclResult ret;

    timeout = GetExternalInputHcclExecTimeOut();
    HCCL_INFO("the timeout is [%d s]", timeout);
    timeout == NOTIFY_DEFAULT_WAIT_TIME ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_EXEC_TIMEOUT", "0xffffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_EXEC_TIMEOUT", "0x5a5a", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_EXEC_TIMEOUT", "this is a test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_EXEC_TIMEOUT", "-2555", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_EXEC_TIMEOUT", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_EXEC_TIMEOUT", "17341", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    timeout = GetExternalInputHcclExecTimeOut();
    HCCL_INFO("the timeout is [%d s]", timeout);
    timeout == HCCL_EXEC_TIME_OUT_S ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_EXEC_TIMEOUT", "68", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    timeout = GetExternalInputHcclExecTimeOut();
    HCCL_INFO("the timeout is [%d s]", timeout);
    timeout == 68 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_EXEC_TIMEOUT", "10000", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    timeout = GetExternalInputHcclExecTimeOut();
    HCCL_INFO("the timeout is [%d s]", timeout);
    timeout == 9996 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设回默认值，并取消环境变量，just for ut
    setenv("HCCL_EXEC_TIMEOUT", "17340", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    timeout = GetExternalInputHcclExecTimeOut();
    HCCL_INFO("the timeout is [%d s]", timeout);
    timeout == HCCL_EXEC_TIME_OUT_S ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_EXEC_TIMEOUT");
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_trafficClass)
{
    u32 rdmaTrafficClass;
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == HCCL_RDMA_TC_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TC", "0xffffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TC", "0x5a5a", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TC", "this is a test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TC", "-1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TC", "256", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == HCCL_RDMA_TC_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TC", "3", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == 3 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TC", "4", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == 4 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TC", "252", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == 252 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设回默认值，并取消环境变量，just for ut
    setenv("HCCL_RDMA_TC", "132", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTrafficClass = GetExternalInputRdmaTrafficClass();
    HCCL_INFO("the rdmaTrafficClass is %d", rdmaTrafficClass);
    rdmaTrafficClass == HCCL_RDMA_TC_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_TC");
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_serverLevel)
{
    u32 rdmaServerLevel;
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaServerLevel = GetExternalInputRdmaServerLevel();
    HCCL_INFO("the rdmaServerLevel is %d", rdmaServerLevel);
    rdmaServerLevel == HCCL_RDMA_SL_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_SL", "0xffffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_SL", "0x5a5a", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_SL", "this is a test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_SL", "-1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_SL", "8", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    rdmaServerLevel = GetExternalInputRdmaServerLevel();
    HCCL_INFO("the rdmaServerLevel is %d", rdmaServerLevel);
    rdmaServerLevel == HCCL_RDMA_SL_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_SL", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaServerLevel = GetExternalInputRdmaServerLevel();
    HCCL_INFO("the rdmaServerLevel is %d", rdmaServerLevel);
    rdmaServerLevel == 0 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_SL", "7", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaServerLevel = GetExternalInputRdmaServerLevel();
    HCCL_INFO("the rdmaServerLevel is %d", rdmaServerLevel);
    rdmaServerLevel == 7 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设回默认值，并取消环境变量，just for ut
    setenv("HCCL_RDMA_SL", "4", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaServerLevel = GetExternalInputRdmaServerLevel();
    HCCL_INFO("the rdmaServerLevel is %d", rdmaServerLevel);
    rdmaServerLevel == HCCL_RDMA_SL_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_SL");
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_RdmaTimeOut)
{
    u32 rdmaTimeOut;
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == HCCL_RDMA_TIMEOUT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TIMEOUT", "0xffffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TIMEOUT", "0x5a5a", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TIMEOUT", "this is a test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TIMEOUT", "4", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_TIMEOUT", "25", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == HCCL_RDMA_TIMEOUT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TIMEOUT", "5", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == 5 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TIMEOUT", "14", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == 14 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    DevType deviceType;
    deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_RDMA_TIMEOUT", "21", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == 21 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
    deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    setenv("HCCL_RDMA_TIMEOUT", "22", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 设回默认值，并取消环境变量，just for ut
    setenv("HCCL_RDMA_TIMEOUT", "20", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaTimeOut = GetExternalInputRdmaTimeOut();
    HCCL_INFO("the rdmaTimeOut is %d", rdmaTimeOut);
    rdmaTimeOut == HCCL_RDMA_TIMEOUT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_TIMEOUT");
    GlobalMockObject::verify();
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_RdmaRetryCnt)
{
    u32 rdmaRetryCnt;
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaRetryCnt = GetExternalInputRdmaRetryCnt();
    HCCL_INFO("the rdmaRetryCnt is %d", rdmaRetryCnt);
    rdmaRetryCnt == HCCL_RDMA_RETRY_CNT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_RETRY_CNT", "0xffffffff", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_RETRY_CNT", "0x5a5a", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_RETRY_CNT", "this is a test", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_RETRY_CNT", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_RDMA_RETRY_CNT", "8", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    rdmaRetryCnt = GetExternalInputRdmaRetryCnt();
    HCCL_INFO("the rdmaRetryCnt is %d", rdmaRetryCnt);
    rdmaRetryCnt == HCCL_RDMA_RETRY_CNT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_RETRY_CNT", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaRetryCnt = GetExternalInputRdmaRetryCnt();
    HCCL_INFO("the rdmaRetryCnt is %d", rdmaRetryCnt);
    rdmaRetryCnt == 1 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaRetryCnt = GetExternalInputRdmaRetryCnt();
    HCCL_INFO("the rdmaRetryCnt is %d", rdmaRetryCnt);
    rdmaRetryCnt == 7 ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 设回默认值，并取消环境变量，just for ut
    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rdmaRetryCnt = GetExternalInputRdmaRetryCnt();
    HCCL_INFO("the rdmaRetryCnt is %d", rdmaRetryCnt);
    rdmaRetryCnt == HCCL_RDMA_RETRY_CNT_DEFAULT ? ret = HCCL_SUCCESS : ret = HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_RETRY_CNT");
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_masterinfo)
{
    HcclResult ret;
    // masterIp error
    string masterIp = "1";
    string masterPort = "6000";
    string masterDeviceId = "0";
    string rankSize = "8";
    string rankIp = "192.168.1.1";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_E_PARA);

    // masterPort error
    masterIp = "192.168.1.1";
    masterPort = "asb";
    masterDeviceId = "0";
    rankSize = "8";
    rankIp = "192.168.1.1";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_E_PARA);
    // masterDeviceId error
    masterIp = "192.168.1.1";
    masterPort = "0";
    masterDeviceId = "abs";
    rankSize = "8";
    rankIp = "192.168.1.1";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_E_PARA);
    // rankSize error
    masterIp = "192.168.1.1";
    masterPort = "6000";
    masterDeviceId = "0";
    rankSize = "a";
    rankIp = "192.168.1.1";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_E_PARA);
    // rankIp error
    masterIp = "192.168.1.1";
    masterPort = "6000";
    masterDeviceId = "0";
    rankSize = "8";
    rankIp = "123";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_E_PARA);

    masterIp = "192.168.1.1";
    masterPort = "6000";
    masterDeviceId = "0";
    rankSize = "8";
    rankIp = "192.168.1.1";
    ret = SetMasterInfo(masterIp, masterPort, masterDeviceId, rankSize, rankIp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ResetInitState();
}

TEST_F(ExternalInputTest, ut_external_input_env_variables_HcclSocketIfName)
{
    std::vector<std::string> configIfNames;
    std::vector<std::string> ifNameList;
    bool searchNot;
    bool searchExact;
    bool socketNameMatch;
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // ret = ParseHcclSocketIfName();
    // 未设置环境变量，判断初始值
    searchNot = GetExternalInputHcclSocketIfName().searchNot;
    searchExact = GetExternalInputHcclSocketIfName().searchExact;
    configIfNames = GetExternalInputHcclSocketIfName().configIfNames;
    ret = (configIfNames.empty() && !searchNot && !searchExact) ? HCCL_SUCCESS : HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 环境变量格式错误，初始化环境变量失败
    setenv("HCCL_SOCKET_IFNAME", ",eth0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 环境变量格式错误，初始化环境变量失败
    setenv("HCCL_SOCKET_IFNAME", "eth0,eth1,", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    // 匹配eth和enp前缀的网卡
    setenv("HCCL_SOCKET_IFNAME", "eth,enp", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    configIfNames = GetExternalInputHcclSocketIfName().configIfNames;
    ifNameList = {"eth", "enp"};
    socketNameMatch = (configIfNames.size() == ifNameList.size());
    for (u32 innerIndex = 0; socketNameMatch && innerIndex < configIfNames.size(); innerIndex++) {
        if (configIfNames[innerIndex] != ifNameList[innerIndex]) {
            socketNameMatch = false;
        }
    }
    searchNot = GetExternalInputHcclSocketIfName().searchNot;
    searchExact = GetExternalInputHcclSocketIfName().searchExact;
    ret = (socketNameMatch && !searchNot && !searchExact) ? HCCL_SUCCESS : HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 精确匹配eth和enp的网卡
    setenv("HCCL_SOCKET_IFNAME", "=eth,enp", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configIfNames = GetExternalInputHcclSocketIfName().configIfNames;
    ifNameList = {"eth", "enp"};
    socketNameMatch = (configIfNames.size() == ifNameList.size());
    for (u32 innerIndex = 0; socketNameMatch && innerIndex < configIfNames.size(); innerIndex++) {
        if (configIfNames[innerIndex] != ifNameList[innerIndex]) {
            socketNameMatch = false;
        }
    }
    searchNot = GetExternalInputHcclSocketIfName().searchNot;
    searchExact = GetExternalInputHcclSocketIfName().searchExact;
    ret = (socketNameMatch && !searchNot && searchExact) ? HCCL_SUCCESS : HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 不匹配eth和enp前缀的网卡
    setenv("HCCL_SOCKET_IFNAME", "^eth,enp", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    configIfNames = GetExternalInputHcclSocketIfName().configIfNames;
    ifNameList = {"eth", "enp"};
    socketNameMatch = (configIfNames.size() == ifNameList.size());
    for (u32 innerIndex = 0; socketNameMatch && innerIndex < configIfNames.size(); innerIndex++) {
        if (configIfNames[innerIndex] != ifNameList[innerIndex]) {
            socketNameMatch = false;
        }
    }
    searchNot = GetExternalInputHcclSocketIfName().searchNot;
    searchExact = GetExternalInputHcclSocketIfName().searchExact;
    ret = (socketNameMatch && searchNot && !searchExact) ? HCCL_SUCCESS : HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 不匹配eth和enp的网卡
    setenv("HCCL_SOCKET_IFNAME", "^=eth,enp", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    configIfNames = GetExternalInputHcclSocketIfName().configIfNames;
    ifNameList = {"eth", "enp"};
    socketNameMatch = (configIfNames.size() == ifNameList.size());
    for (u32 innerIndex = 0; innerIndex < configIfNames.size(); innerIndex++) {
        if (configIfNames[innerIndex] != ifNameList[innerIndex]) {
            socketNameMatch = false;
        }
    }
    searchNot = GetExternalInputHcclSocketIfName().searchNot;
    searchExact = GetExternalInputHcclSocketIfName().searchExact;
    ret = (socketNameMatch && searchNot && searchExact) ? HCCL_SUCCESS : HCCL_E_PARA;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_SOCKET_IFNAME");
}

#if 1
TEST_F(ExternalInputTest, st_ParseCannVersion)
{
    int ret = HCCL_SUCCESS;

    std::string temp_filename = "version.info";

    std::vector<std::string> file_lines = {"Version=111", "hccl_running_version=[222", "timestamp=333"};

    {
        std::ofstream file(temp_filename);
        for (const auto& line : file_lines) {
            file << line << "\n";
        }
    }
    ret = LoadCannVersionInfoFile(temp_filename, "Version=");
    ret = LoadCannVersionInfoFile(temp_filename, "timestamp=");
    ret = LoadCannVersionInfoFile(temp_filename, "hccl_running_version=[2");

    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::remove(temp_filename.c_str());

    MOCKER(LoadCannVersionInfoFile).stubs().will(returnValue(HCCL_SUCCESS));

    char* path = getenv("LD_LIBRARY_PATH");

    setenv("LD_LIBRARY_PATH", "EmptyString", 1);
    ret = ParseCannVersion();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("LD_LIBRARY_PATH");

    setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/CANN-6.3/runtime", 1);
    ret = ParseCannVersion();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("LD_LIBRARY_PATH");

    setenv("LD_LIBRARY_PATH", "/usr/local/Ascend/latest", 1);
    ret = ParseCannVersion();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    unsetenv("LD_LIBRARY_PATH");

    setenv("LD_LIBRARY_PATH", path, 1);
    GlobalMockObject::verify();
}
#endif

// 测试cclbufsize
TEST_F(ExternalInputTest, ut_external_input_env_cclbufsize)
{
    u64 cclbufsize;
    HcclResult ret;

    cclbufsize = GetExternalInputCCLBuffSize();
    HCCL_INFO("the cclBufferSize is %llu", cclbufsize);
    ret = ((cclbufsize == HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE * (1 * 1024 * 1024)) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ResetInitState();
    setenv("HCCL_BUFFSIZE", "1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    cclbufsize = GetExternalInputCCLBuffSize();
    HCCL_INFO("the cclBufferSize is %llu", cclbufsize);
    ret = ((cclbufsize == HCCL_CCL_COMM_BUFFER_MIN * (1 * 1024 * 1024)) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ResetInitState();
    setenv("HCCL_BUFFSIZE", "200", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    cclbufsize = GetExternalInputCCLBuffSize();
    HCCL_INFO("the cclBufferSize is %llu", cclbufsize);
    ret = ((cclbufsize == HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE * (1 * 1024 * 1024)) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ResetInitState();
    setenv("HCCL_BUFFSIZE", "2048", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ResetInitState();
    setenv("HCCL_BUFFSIZE", "0", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    ResetInitState();
    unsetenv("HCCL_BUFFSIZE");
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    cclbufsize = GetExternalInputCCLBuffSize();
    HCCL_INFO("the cclBufferSize is %llu", cclbufsize);
    ret = ((cclbufsize == HCCL_CCL_COMM_DEFAULT_BUFFER_SIZE * (1 * 1024 * 1024)) ? HCCL_SUCCESS : HCCL_E_PARA);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

// 测试HCCL_INTER_HCCS_DISABLE
TEST_F(ExternalInputTest, ut_external_input_env_interHccs)
{
    bool interHccsDisable = false;
    HcclResult ret;

    ResetInitState();
    setenv("HCCL_INTER_HCCS_DISABLE", "true", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    interHccsDisable = GetExternalInputInterHccsDisable();
    EXPECT_EQ(interHccsDisable, true);

    ResetInitState();
    setenv("HCCL_INTER_HCCS_DISABLE", "false", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    interHccsDisable = GetExternalInputInterHccsDisable();
    EXPECT_EQ(interHccsDisable, false);

    ResetInitState();
    setenv("HCCL_INTER_HCCS_DISABLE", "test", 1);
    ret = InitEnvVarParam();
    EXPECT_NE(ret, HCCL_SUCCESS);

    unsetenv("HCCL_INTER_HCCS_DISABLE");
    ResetInitState();
}

// 测试HCCL_LOG_CONFIG
TEST_F(ExternalInputTest, ut_external_hccl_log_config)
{
    u32 logConfigValue = 0;
    HcclResult ret;
    ResetInitState();
    // 测试HCCL_LOG_CONFIG为无效变量-1的情况
    setenv("HCCL_ENTRY_LOG_ENABLE", "-1", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    logConfigValue = GetExternalInputHcclEnableEntryLog();
    EXPECT_EQ(logConfigValue, 0);

    unsetenv("HCCL_ENTRY_LOG_ENABLE");
    ResetInitState();
}

TEST_F(ExternalInputTest, ut_external_input_env_retryparams)
{
    u32 maxcnt = 0;
    u32 holdtime = 0;
    u32 intervaltime = 0;
    HcclResult ret = HCCL_E_PARA;

    /* 测试默认配置 */
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    maxcnt = GetExternalInputRetryMaxCnt();
    holdtime = GetExternalInputRetryHoldTime();
    intervaltime = GetExternalInputRetryIntervalTime();
    EXPECT_EQ(maxcnt, HCCL_RETRY_MAXCNT_DEFAULT);
    EXPECT_EQ(holdtime, HCCL_RETRY_HOLD_TIME_DEFAULT);
    EXPECT_EQ(intervaltime, HCCL_RETRY_INTERVAL_DEFAULT);

    /* 测试正常配置MaxCnt:2, HoldTime:100, IntervalTime:100 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:2, HoldTime:100, IntervalTime:100", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    maxcnt = GetExternalInputRetryMaxCnt();
    holdtime = GetExternalInputRetryHoldTime();
    intervaltime = GetExternalInputRetryIntervalTime();
    EXPECT_EQ(maxcnt, 2);
    EXPECT_EQ(holdtime, 100);
    EXPECT_EQ(intervaltime, 100);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置1 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:600001, HoldTime:100, IntervalTime:100", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置2 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:2, HoldTime:3600001, IntervalTime:100", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置3 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:2, HoldTime:100, IntervalTime:3600001", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置4 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:1.0, HoldTime:200.1, IntervalTime:300.3", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置5 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt:1s, HoldTime:2ms, IntervalTime:3us", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置6 */
    setenv("HCCL_OP_RETRY_PARAMS", "Max:1s, HoldTime:200, IntervalTime:200", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置7 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt1s, HoldTime:200, IntervalTime:200", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");

    /* 测试异常配置配置8 */
    setenv("HCCL_OP_RETRY_PARAMS", "MaxCnt: 0, HoldTime: 5000, IntervalTime: 1000", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_OP_RETRY_PARAMS");
}

TEST_F(ExternalInputTest, ut_external_input_env_hccl_algo)
{
    HcclResult ret;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_ALGO", "level0:NA/level1:pairwise", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "alltoall=level0:NA&level1:pairwise", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "alltoall=level0:NA;level1:pairwise&allreduce=level0:NA;level1:pipeline", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "level0:NA/level1:pairwise", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "allreduce=level0:Yes;level1:pairwise", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_ALGO", "not_op=level0:Yes;level1:pairwise", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    unsetenv("HCCL_ALGO");
}

TEST_F(ExternalInputTest, ut_external_input_env_hccl_op_expansion_mode)
{
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_OP_EXPANSION_MODE", "aix", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    unsetenv("HCCL_OP_EXPANSION_MODE");
}

TEST_F(ExternalInputTest, ut_external_input_env_hccl_deterministic)
{
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclDeterministicV2(), 0);
    EXPECT_EQ(GetExternalInputHcclDeterministic(), false);

    setenv("HCCL_DETERMINISTIC", "sdqawe", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    setenv("HCCL_DETERMINISTIC", "TRUE", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclDeterministicV2(), 1);
    EXPECT_EQ(GetExternalInputHcclDeterministic(), true);

    setenv("HCCL_DETERMINISTIC", "STRICT", 1);

    DevType devType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclDeterministicV2(), 2);
    EXPECT_EQ(GetExternalInputHcclDeterministic(), false);

    setenv("HCCL_DETERMINISTIC", "FALSE", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclDeterministicV2(), 0);
    EXPECT_EQ(GetExternalInputHcclDeterministic(), false);

    unsetenv("HCCL_DETERMINISTIC");
    GlobalMockObject::verify();
}

TEST_F(ExternalInputTest, ut_external_input_env_expansion_mode_hostts)
{
    ResetInitState();
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    setenv("HCCL_OP_EXPANSION_MODE", "HOST_TS", 1);
    HcclResult ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclEnableFfts(), false);

    unsetenv("HCCL_OP_EXPANSION_MODE");
    GlobalMockObject::verify();

    ResetInitState();
    deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    setenv("HCCL_OP_EXPANSION_MODE", "HOST_TS", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputHcclEnableFfts(), true);

    unsetenv("HCCL_OP_EXPANSION_MODE");
    GlobalMockObject::verify();
    ResetInitState();
}

TEST_F(ExternalInputTest, ut_external_input_env_expansion_mode_host)
{
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_OP_EXPANSION_MODE", "HOST", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_OP_EXPANSION_MODE");
}

TEST_F(ExternalInputTest, ut_external_input_env_rdma_high_perf_enable)
{
    HcclResult ret;
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ResetInitState();
    setenv("HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "TRUE", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "FALSE", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT", "ERROR", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    unsetenv("HCCL_RDMA_PCIE_DIRECT_POST_NOSTRICT");
}

TEST_F(ExternalInputTest, ut_external_input_env_socket_port_range_null)
{
    setenv("HCCL_NPU_SOCKET_PORT_RANGE", "    ", 1);
    HcclResult ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<HcclSocketPortRange> portRange = GetExternalInputHostSocketPortRange();
    std::vector<HcclSocketPortRange> npuPortRange = GetExternalInputNpuSocketPortRange();
    EXPECT_EQ(portRange.size(), 0);
    EXPECT_EQ(npuPortRange.size(), 0);
    EXPECT_EQ(CheckSocketPortRangeValid("HCCL_NPU_SOCKET_PORT_RANGE", npuPortRange), HCCL_SUCCESS);
    PrintSocketPortRange("HCCL_NPU_SOCKET_PORT_RANGE", npuPortRange);

    SocketLocation socketLoc = SOCKET_HOST;
    EXPECT_EQ(PortRangeSwitchOn(socketLoc), HCCL_SUCCESS);
    socketLoc = SOCKET_NPU;
    EXPECT_EQ(PortRangeSwitchOn(socketLoc), HCCL_SUCCESS);

    unsetenv("HCCL_NPU_SOCKET_PORT_RANGE");
}

TEST_F(ExternalInputTest, ut_external_input_env_socket_port_range)
{
    setenv("HCCL_HOST_SOCKET_PORT_RANGE", "60000,60001-60031", 1);
    HcclResult ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<HcclSocketPortRange> hostPortRange = GetExternalInputHostSocketPortRange();
    EXPECT_EQ(hostPortRange.size(), 2);

    unsetenv("HCCL_HOST_SOCKET_PORT_RANGE");
}

TEST_F(ExternalInputTest, ut_external_input_env_socket_port_range_310p)
{
    setenv("HCCL_HOST_SOCKET_PORT_RANGE", "60000,60001-60031", 1);
    MOCKER(hrtGetDeviceType).stubs().with(outBound(DevType::DEV_TYPE_310P1)).will(returnValue(HCCL_SUCCESS));
    HcclResult ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_HOST_SOCKET_PORT_RANGE");
    GlobalMockObject::verify();
}

TEST_F(ExternalInputTest, ut_external_input_env_HCCL_DEBUG_CONFIG_invalid)
{
    HcclResult ret = HCCL_SUCCESS;

    // 配置为无效值
    setenv("HCCL_DEBUG_CONFIG", "AAA", 1);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_E_PARA);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(ExternalInputTest, ut_external_input_env_HCCL_DEBUG_CONFIG_invert)
{
    u64 config = 0;
    HcclResult ret = HCCL_SUCCESS;

    // 配置为：^alg,task
    setenv("HCCL_DEBUG_CONFIG", "^alg,task", 1);
    config = (~0ULL) & (~HCCL_ALG) & (~HCCL_TASK);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");

    // 配置为：^alg
    setenv("HCCL_DEBUG_CONFIG", "^alg", 1);
    config = (~0ULL) & (~HCCL_ALG);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");

    // 配置为：^task
    setenv("HCCL_DEBUG_CONFIG", "^task", 1);
    config = (~0ULL) & (~HCCL_TASK);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");
}

TEST_F(ExternalInputTest, ut_external_input_env_HCCL_DEBUG_CONFIG)
{
    u64 config = 0;
    HcclResult ret = HCCL_SUCCESS;

    // 配置为：task
    setenv("HCCL_DEBUG_CONFIG", "task", 1);
    config = (0ULL) | (HCCL_TASK);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");

    // 配置为：alg
    setenv("HCCL_DEBUG_CONFIG", "alg", 1);
    config = (0ULL) | (HCCL_ALG);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");

    // 配置为：resource
    setenv("HCCL_DEBUG_CONFIG", "resource", 1);
    config = (0ULL) | (PLF_RES);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");

    // 配置为：alg,task
    setenv("HCCL_DEBUG_CONFIG", "alg,task", 1);
    config = (0ULL) | (HCCL_ALG) | (HCCL_TASK);
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    EXPECT_EQ(hccl::InitDebugConfigByEnv(), HCCL_SUCCESS);
    EXPECT_EQ(hccl::GetDebugConfig(), config);

    // 空配置
    setenv("HCCL_DEBUG_CONFIG", "", 1);
    config = 0;
    ret = InitEnvParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(GetExternalInputDebugConfig(), config);
    unsetenv("HCCL_DEBUG_CONFIG");
}
