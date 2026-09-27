/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_api_base_test.h"

void Ut_Device_Set(int devId)
{
    HcclResult ret = hrtSetDevice(devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

rtError_t rtOpenNetService(rtNetServiceOpenArgs* openArgs)
{
    // hccpThreadStatus = 1;
    return ACL_RT_SUCCESS;
}

rtError_t rtCloseNetService()
{
    // hccpThreadStatus = 0;
    return ACL_RT_SUCCESS;
}

void Ut_Clusterinfo_File_Create(const char* filename, nlohmann::json rankTable)
{
    const char* file_name_t = filename;
    std::ofstream outfile(file_name_t, std::ios::out | std::ios::trunc | std::ios::binary);
    if (outfile.is_open()) {
        outfile << std::setw(1) << (rankTable) << std::endl;
        HCCL_INFO("Successfully wrote to %s", file_name_t);
    } else {
        HCCL_ERROR("Failed to open %s for writing", file_name_t);
    }
    outfile.close();
}

HcclRootInfo Ut_Get_Root_Info(int devId)
{
    HcclRootInfo id;
    HcclResult ret = hrtSetDevice(devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    return id;
}

void Ut_Comm_Destroy(void*& comm)
{
    HcclResult ret = HcclCommDestroy(comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

void Ut_Comm_Create(void*& comm, int devId, const char* rankTableFile, int rankId)
{
    HcclResult ret = hrtSetDevice(devId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommInitClusterInfo(rankTableFile, rankId, &comm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

void Ut_Buf_Create(s8*& buf, int len)
{
    buf = (s8*)sal_malloc(len * sizeof(s8));
    sal_memset(buf, len * sizeof(s8), 0, len * sizeof(s8));
}

void Ut_BufV_Create(s8*& buf, int bufLen, u64*& counts, int countsLen, int c, u64*& displs, int displsLen, int d)
{
    counts = (u64*)sal_malloc(countsLen * sizeof(u64));
    for (int i = 0; i < countsLen; i++)
        counts[i] = c;
    displs = (u64*)sal_malloc(displsLen * sizeof(u64));
    for (int i = 0; i < displsLen; i++)
        displs[i] = d * i;
    buf = (s8*)sal_malloc(bufLen * sizeof(s8));
    sal_memset(buf, bufLen * sizeof(s8), 0, bufLen * sizeof(s8));
}

void Ut_Stream_Create(rtStream_t& stream, int priority)
{
    rtError_t rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
}

void Ut_Stream_Synchronize(aclrtStream& stream)
{
    aclError rt_ret = aclrtSynchronizeStream(stream);
    EXPECT_EQ(rt_ret, ACL_SUCCESS);
}

void Ut_Stream_Destroy(rtStream_t& stream)
{
    rtError_t rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(rt_ret, ACL_SUCCESS);
}

void Ut_Stream_SynchronizeAndDestroy(rtStream_t& stream)
{
    Ut_Stream_Synchronize(stream);
    Ut_Stream_Destroy(stream);
}

void When_Need_HcclGetRootInfo(void)
{
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(5));

    MOCKER_CPP(&HcclSocket::Listen, HcclResult(HcclSocket::*)(u32 port))
        .stubs()
        .with(mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&TopoInfoExchangeAgent::Connect).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_INTERNAL));

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(1));
}

void Ut_MultiServer_MOCK_And_Clusterinfo_File_Create(const char* filename, nlohmann::json rankTable)
{
    aclmdlRICaptureStatus captureStatus = aclmdlRICaptureStatus::ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    int mockModel = 0;
    void* pmockModel = &mockModel;
    MOCKER(aclmdlRICaptureGetInfo)
        .stubs()
        .with(
            mockcpp::any(), outBoundP(&captureStatus, sizeof(captureStatus)),
            outBoundP(&pmockModel, sizeof(pmockModel)))
        .will(returnValue(0));

    MOCKER_CPP(&HcclCommunicator::StreamIsCapture).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER(GetExternalInputHcclEnableEntryLog).stubs().with(mockcpp::any()).will(returnValue(true));

    MOCKER_CPP(&TransportManager::Alloc).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::ExecOp).stubs().with(mockcpp::any()).will(returnValue(HCCL_SUCCESS));

    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaGetInterfaceVersion).stubs().will(returnValue(HCCL_SUCCESS));

    Ut_Clusterinfo_File_Create(filename, rankTable);
}

void BaseInit::SetUp()
{
    strcpy(rankTableFileName, "./ut_opbase_test.json");
    comm = nullptr;
    stream = 0;
    s32 portNum = 7;
    MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));
    static s32 call_cnt = 0;
    string name = std::to_string(call_cnt++) + "_" + __PRETTY_FUNCTION__;
    DlTdtFunction::GetInstance().DlTdtFunctionInit();
    rtNetServiceOpenArgs* openArgs;
    rtOpenNetService(openArgs);
    ra_set_shm_name(name.c_str());
    ResetInitState();

    MOCKER(hrtProfRegisterCtrlCallback).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(LoadBinaryFromFile).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(RptInputErr).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER(HcclNetInit).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::InitNetResource).stubs().will(returnValue(HCCL_SUCCESS));

    MOCKER_CPP(&HcclCommunicator::GetHDCommunicate).stubs().will(returnValue(HCCL_SUCCESS));

    // mock掉对heartbeat模块的依赖
    MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::RegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&Heartbeat::UnRegisterRanks).stubs().will(returnValue(HCCL_SUCCESS));
}
void BaseInit::TearDown() { rtCloseNetService(); }
