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
#include <stdlib.h>
#include <pthread.h>
#include <assert.h>
#include <semaphore.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <unistd.h>
#include <errno.h>
#include <securec.h>
#include <sys/types.h>
#include <stddef.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <driver/ascend_hal.h>
#include "externalinput.h"
#include "hccl/base.h"
#include <hccl/hccl_types.h>
#include "hcom_pub.h"
#include "hccl/hcom.h"
#include "llt_hccl_stub_pub.h"
#include <sys/mman.h>
#include <fcntl.h>
#include "hccl_comm_pub.h"
#include "sal.h"
#include "dlra_function.h"
#include "hccl_comm_pub.h"
#include "config.h"
#include "topo/topoinfo_ranktableParser_pub.h"
#include "externalinput_pub.h"
#include "v80_rank_table.h"
#include "dltdt_function.h"
#include <iostream>
#include <fstream>
#include "opexecounter_pub.h"
#include "hccl_communicator.h"
#include "op_base.h"
#include "adapter_prof.h"
#include "param_check_pub.h"

using namespace std;
using namespace hccl;

class AllGatherMix_Opbase_Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AllGatherMix_Opbase_Test SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "AllGatherMix_Opbase_Test TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

#define HCCL_COM_DATA_SIZE 1024
#define DEV_NUM_8 8

TEST_F(AllGatherMix_Opbase_Test, ut_HcclAllGatherOutPlace_mix_ranksize_1)
{
    setenv("HCCL_OP_EXPANSION_MODE", "HOST", 1);
    ResetInitState();
    InitExternalInput();

    DevType deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCallbackTask::CallbackRegStream).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));
    s32 portNum = -1;
    MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::InitPreResource).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::RegisterToHeartBeat, HcclResult(HcclCommunicator::*)())
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    rtStream_t stream;
    s8* sendBuf;
    s8* recvBuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    ret = hrtSetDevice(0);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* comm;
    s32 ndev = DEV_NUM_8;

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sendBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendBuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvBuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvBuf, count * sizeof(s8), 0, count * sizeof(s8));

    HcclComm newcomm;
    HcclRootInfo id;
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::IsAtomicInit).stubs().will(returnValue(true));

    HcclCommunicator impl;
    HcclCommParams params;
    string commId = "AllGatherMixOpbase";
    memcpy_s(params.id.internal, HCCL_ROOT_INFO_BYTES, commId.c_str(), commId.length() + 1);
    params.rank = 0;
    params.totalRanks = 1;
    params.isHeterogComm = false;
    params.logicDevId = 0;
    params.deviceType = DevType::DEV_TYPE_910_93;

    RankTable_t rankTable;
    rankTable.collectiveId = "192.168.10.101-8000-8001";
    vector<RankInfo_t> rankVec(1);
    rankVec[0].rankId = 0;
    rankVec[0].deviceInfo.devicePhyId = 0;
    HcclIpAddress ipAddr1(1695197376);                 // 1,695,197,376
    rankVec[0].deviceInfo.deviceIp.push_back(ipAddr1); // 101.10.168.192
    rankVec[0].serverIdx = 0;
    rankVec[0].serverId = "192.168.0.101";
    rankTable.rankList.assign(rankVec.begin(), rankVec.end());
    rankTable.deviceNum = 1;
    rankTable.serverNum = 1;
    aclrtSetDevice(0);

    ret = impl.Init(params, rankTable);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    impl.userRankSize_ = 1;

    ret = impl.AllGatherOutPlace(commId, sendBuf, recvBuf, count, HCCL_DATA_TYPE_INT8, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);

    sal_free(sendBuf);
    sal_free(recvBuf);
    rt_ret = aclrtDestroyStream(stream);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    GlobalMockObject::verify();

    unsetenv("HCCL_OP_EXPANSION_MODE");
    ResetInitState();
    InitExternalInput();
}
