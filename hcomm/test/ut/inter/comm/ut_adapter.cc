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

#include <sys/epoll.h>
#include <driver/ascend_hal.h>
#include "network/hccp.h"
#include "adapter_rts.h"
#include "dispatcher_pub.h"
#include "externalinput.h"
#include "config.h"
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include "tsd/tsd_client.h"
#include "llt_hccl_stub_pub.h"
#include "llt_hccl_stub.h"
#include "sal.h"
#include "task_profiling_pub.h"
#include "dlra_function.h"
#include "dltdt_function.h"
#include "dlhal_function.h"
#include "externalinput_pub.h"
#include "task_overflow_pub.h"
#include "env_config.h"
#include "heartbeat.h"

#define private public
#define protected public
#include "topoinfo_detect.h"
#undef protected
#undef private

using namespace std;

class RuntimeTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "RuntimeTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "RuntimeTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        DlTdtFunction::GetInstance().DlTdtFunctionInit();
        std::cout << "A Test SetUP" << std::endl;
        MOCKER_CPP(&Heartbeat::Init).stubs().will(returnValue(HCCL_SUCCESS));
    }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

extern bool CompareDevType(DevType left, DevType right);
TEST_F(RuntimeTest, EventTest610)
{
    setenv("HCCL_DFS_CONFIG", "connection_fault_detection_time:0", 1);
    InitEnvParam();
    MOCKER(CompareDevType).stubs().with(mockcpp::any(), eq(DevType::DEV_TYPE_310P1)).will(returnValue(true));

    rtNotify_t notify;
    HcclResult ret = hrtNotifyCreate(0, &notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 offset = 0;
    ret = hrtNotifyGetOffset(notify, offset);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    Stream stream(StreamType::STREAM_TYPE_OFFLINE);
    ret = hrtNotifyWaitWithTimeOut(notify, stream.ptr(), 100);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

s32 fake_rtGetSocVersionV80(char* chipVer, const u32 maxLen)
{
    sal_memcpy(chipVer, sizeof("Ascend910"), "Ascend910", sizeof("Ascend910"));
    return DRV_ERROR_NONE;
}

s32 fake_rtGetSocVersionV81(char* chipVer, const u32 maxLen)
{
    sal_memcpy(chipVer, sizeof("Ascend910B1"), "Ascend910B1", sizeof("Ascend910B1"));
    return DRV_ERROR_NONE;
}

s32 fake_rtGetSocVersionV51(char* chipVer, const u32 maxLen)
{
    sal_memcpy(chipVer, sizeof("Ascend310P3"), "Ascend310P3", sizeof("Ascend310P3"));
    return DRV_ERROR_NONE;
}

TEST_F(RuntimeTest, test_NotifySize)
{
    HcclResult ret;
    u32 notifySize = 0;
    DevType deviceType;

    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV80));

    deviceType = DevType::DEV_TYPE_910;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    ret = hrtGetNotifySize(notifySize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifySize, 8);
    GlobalMockObject::verify();

    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV81));

    deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    ret = hrtGetNotifySize(notifySize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifySize, 4);
    GlobalMockObject::verify();

    MOCKER(rtGetSocVersion).stubs().will(invoke(fake_rtGetSocVersionV51));

    deviceType = DevType::DEV_TYPE_310P3;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    ret = hrtGetNotifySize(notifySize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(notifySize, 8);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_HrtDevFree)
{
    void* data = malloc(10);

    HcclResult ret = HrtDevFree(data);
    EXPECT_EQ(ret, HCCL_E_PTR);

    free(data);
}

TEST_F(RuntimeTest, test_HrtDevMallocAndFree)
{
    HcclResult ret;
    u64 size = 4096;
    void* devMemAddr{nullptr};
    ret = HrtDevMalloc(&devMemAddr, size);
    EXPECT_EQ(ret, 2);

    ret = HrtDevFree(devMemAddr);
    EXPECT_EQ(ret, 2);
}

TEST_F(RuntimeTest, test_SetQpAttrQos)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaSetQpAttrQos).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    setenv("HCCL_RDMA_TC", "4", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    QpHandle qpHandle = nullptr;
    ret = SetQpAttrQos(qpHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    setenv("HCCL_RDMA_TC", "132", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_TC");

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaGetNotifyMrInfo)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_E_NOT_SUPPORT));

    void* handle;
    mr_info* mrInfo;
    HcclResult ret = HrtRaGetNotifyMrInfo(0, handle, mrInfo);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(RuntimeTest, test_SetQpAttrRetryCnt)
{
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    MOCKER(hrtRaSetQpAttrRetryCnt).expects(atMost(1)).will(returnValue(HCCL_SUCCESS));

    HcclResult ret = HCCL_SUCCESS;
    setenv("HCCL_RDMA_RETRY_CNT", "2", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    QpHandle qpHandle = nullptr;
    ret = SetQpAttrRetryCnt(qpHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 将HCCL_RDMA_RETRY_CNT环境变量设置为默认值，防止影响其它用例
    setenv("HCCL_RDMA_RETRY_CNT", "7", 1);
    ret = InitEnvVarParam();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    unsetenv("HCCL_RDMA_RETRY_CNT");

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaSocketRecv)
{
    s64 ret = 0;
    void* fdHandle = nullptr;
    void* data = nullptr;
    u64 size = 0;
    u64 recvSize = 0;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct ra_init_config config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE};
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(ra_init).stubs().will(returnValue(328002));
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_E_PARA);

    ret = hrtRaSocketRecv(fdHandle, data, size, &recvSize);
    EXPECT_EQ(ret, 1);

    ret = HrtRaDeInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_ra_deinit_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_deinit).stubs().will(returnValue(-EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).expects(once()).will(returnValue(1));

    ra_init_config raConfig;
    raConfig.phy_id = 0;
    raConfig.nic_position = 0;
    ret = HrtRaDeInit(&raConfig);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_rt_ra_send_wr_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* handle = nullptr;
    struct send_wr wr;
    struct send_wr_rsp opRsp = {0};
    u32 wqeIndex;
    MOCKER(ra_send_wr).stubs().will(returnValue(-EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).expects(once()).will(returnValue(1));

    ret = HrtRaSendWr(handle, &wr, &opRsp);
    EXPECT_EQ(ret, HCCL_E_ROCE_TRANSFER);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaQpNonBlockConnectAsync)
{
    s64 ret = 0;
    void* fdHandle = nullptr;
    void* sockHandle = nullptr;

    MOCKER_CPP(&DlRaFunction::DlRaFunctionInit).expects(atMost(1)).will(returnValue(SOCK_EADDRINUSE));
    ret = HrtRaQpNonBlockConnectAsync(fdHandle, sockHandle);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaQpConnectAsync)
{
    s64 ret = 0;
    void* fdHandle = nullptr;
    void* sockHandle = nullptr;

    MOCKER_CPP(&DlRaFunction::DlRaFunctionInit).expects(atMost(1)).will(returnValue(SOCK_EADDRINUSE));

    ret = HrtRaQpConnectAsync(fdHandle, sockHandle);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaDeRegGlobalMr)
{
    s64 ret = 0;
    u32 temp = 10;
    void* rdmaHandle = &temp;
    void* mrHandle = &temp;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    struct ra_init_config config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE};
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = hrtRaDeRegGlobalMr(rdmaHandle, mrHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HrtRaDeInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtRaSocketNonBlockSend)
{
    HcclResult ret = HCCL_SUCCESS;
    void* fdHandle = nullptr;
    void* data = nullptr;
    u64 size = 0;
    u64 sentSize = 0;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 result = 0;
    MOCKER(ra_socket_send).expects(once()).will(returnValue(0));
    result = hrtRaSocketNonBlockSend(fdHandle, data, size, &sentSize);
    EXPECT_EQ(ret, 0);
    GlobalMockObject::verify();

    MOCKER(ra_socket_send).expects(once()).will(returnValue(SOCK_EAGAIN));
    result = hrtRaSocketNonBlockSend(fdHandle, data, size, &sentSize);
    EXPECT_EQ(ret, 0);
    GlobalMockObject::verify();

    MOCKER(ra_socket_send).expects(once()).will(returnValue(-1));
    result = hrtRaSocketNonBlockSend(fdHandle, data, size, &sentSize);
    EXPECT_EQ(ret, 0);
    GlobalMockObject::verify();

    size = SOCKET_SEND_MAX_SIZE + 1;
    result = hrtRaSocketNonBlockSend(fdHandle, data, size, &sentSize);
    EXPECT_EQ(ret, 0);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaRdmaDeInit)
{
    s32 ret = HCCL_SUCCESS;
    RdmaHandle rdmaHandle;

    MOCKER(ra_rdev_deinit).expects(atMost(1)).will(returnValue(-2));

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HrtRaRdmaDeInit(rdmaHandle, 1);
    EXPECT_EQ(ret, 19);
}

TEST_F(RuntimeTest, test_hrtRaRdmaDeInitRef)
{
    s32 ret = HCCL_SUCCESS;
    RdmaHandle rdmaHandle;

    MOCKER(ra_rdev_deinit).expects(atMost(1)).will(returnValue(-2));

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HrtRaRdmaDeInitRef(rdmaHandle, 1);
}

TEST_F(RuntimeTest, test_hrtRaSocketDeInitRef)
{
    s32 ret = HCCL_SUCCESS;
    SocketHandle socketHandle;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtRaSocketDeInitRef(socketHandle);
}

TEST_F(RuntimeTest, test_hrtRaSocketNonBlockRecv)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    s32 result = 0;
    void* FdHandle = nullptr;
    void* data = malloc(10);
    u64 size = 64;
    u64* recvSize = nullptr;

    result = hrtRaSocketNonBlockRecv(FdHandle, data, size, recvSize);
    free(data);
}

TEST_F(RuntimeTest, test_hrtRaSocketBlockSend)
{
    s64 ret = 0;
    void* fdHandle = nullptr;
    void* data = nullptr;
    u64 size = 0x8FFFFFFFFFFFFFFF;

    ret = hrtRaSocketBlockSend(fdHandle, data, size);
    EXPECT_EQ(ret, HCCL_E_PARA);

    u64 sendlen = 1;
    MOCKER(ra_socket_send)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any(), outBoundP(&sendlen))
        .will(returnValue(0))
        .then(returnValue(128201))
        .then(returnValue(1));
    ret = hrtRaSocketBlockSend(&fdHandle, data, sendlen);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    auto time_point = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration duration = std::chrono::seconds(999999);
    // 将time_point与duration相加
    auto new_time_point = time_point + duration;
    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));
    MOCKER((std::chrono::steady_clock::now)).stubs().will(returnValue(time_point)).then(returnValue(new_time_point));
    ret = hrtRaSocketBlockSend(&fdHandle, data, sendlen);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    ret = hrtRaSocketBlockSend(&fdHandle, data, sendlen);
    EXPECT_EQ(ret, HCCL_E_NETWORK);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_WaitTopoExchangeServerCompelte)
{
    HcclResult ret = HCCL_SUCCESS;
    std::shared_ptr<TopoInfoDetect> topoDetectAgent = std::make_shared<TopoInfoDetect>();

    auto time_point = std::chrono::steady_clock::now();
    std::chrono::steady_clock::duration duration = std::chrono::seconds(999999);
    // 将time_point与duration相加
    auto new_time_point = time_point + duration;

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));
    MOCKER((std::chrono::steady_clock::now)).stubs().will(returnValue(time_point)).then(returnValue(new_time_point));

    topoDetectAgent->g_topoExchangeServerStatus_.EmplaceAndUpdate(0, [](volatile u32& status) {
        status = 1;
    });
    ret = topoDetectAgent->WaitTopoExchangeServerCompelte(0);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaDestroyCq)
{
    s64 ret = 0;
    void* rdmaHandle = malloc(8);
    struct cq_attr* attr = (struct cq_attr*)malloc(16);
    attr->qp_context = nullptr;

    ret = hrtRaDestroyCq(rdmaHandle, attr);
    EXPECT_EQ(ret, 0);

    free(rdmaHandle);
    free(attr);
}

TEST_F(RuntimeTest, test_hrtRaQpDestroy)
{
    void* QpHandle = nullptr;
    HrtRaQpDestroy(QpHandle);
}

TEST_F(RuntimeTest, test_hrtRaRdmaInitWithAttr)
{
    int rdma = 0;
    struct RdevInitInfo init_info {
        0
    };
    struct rdev rdevInfo {};
    RdmaHandle rdmaHandle = &rdma;
    int ret = HrtRaRdmaInitWithAttr(init_info, rdevInfo, rdmaHandle);
    EXPECT_EQ(ret, 19);
}

TEST_F(RuntimeTest, test_hrtRaRdmaGetHandle)
{
    HcclResult ret;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    int rdma = 0;
    unsigned int phyId = 0;
    RdmaHandle rdmaHandle = &rdma;
    ret = HrtRaRdmaGetHandle(phyId, rdmaHandle);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaCreateCompChannel)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    struct ra_init_config config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE};
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 rdma = 0;
    RdmaHandle rdmaHandle = &rdma;
    void** compChannel = &rdmaHandle;

    ret = hrtRaCreateCompChannel(rdmaHandle, compChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HrtRaDeInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtRaDestroyCompChannel)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    struct ra_init_config config = {DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE};
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64 rdma = 0;
    RdmaHandle rdmaHandle = &rdma;
    void* compChannel = &rdma;

    ret = hrtRaDestroyCompChannel(rdmaHandle, compChannel);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HrtRaDeInit(&config);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtRaGetCqeErrInfo_001)
{
    HcclResult ret = HCCL_SUCCESS;
    struct cqe_err_info info;

    ret = hrtRaGetCqeErrInfo(0, &info);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtRaGetQpAttr_001)
{
    HcclResult ret = HCCL_SUCCESS;
    QpHandle qpHandle;
    struct qp_attr attr = {0};

    ret = hrtRaGetQpAttr(qpHandle, &attr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_hrtNotifyGetAddr)
{
    MOCKER(rtGetNotifyAddress).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(RT_ERROR_NONE));

    HcclRtNotify notify;
    u64* notifyAddr = new u64(0);

    auto ret = hrtNotifyGetAddr(notify, notifyAddr);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete notifyAddr;
}

TEST_F(RuntimeTest, ut_hrtGetNotifyID)
{
    MOCKER(aclrtGetNotifyId).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ACL_SUCCESS));

    HcclRtNotify signal = new u32(0);
    u32 notifyID;

    auto ret = hrtGetNotifyID(signal, &notifyID);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    delete signal;
}

TEST_F(RuntimeTest, ut_hrtGetDeviceInfo)
{
    MOCKER(aclrtGetDeviceInfo).stubs().with().will(returnValue(ACL_SUCCESS));

    u32 deviceId = 0;
    HcclRtDeviceModuleType moduleType = HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM;
    HcclRtDeviceInfoType infoType = HcclRtDeviceInfoType::HCCL_INFO_TYPE_CUST_OP_ENHANCE;
    s64 val = 1;

    auto ret = hrtGetDeviceInfo(deviceId, moduleType, infoType, val);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(RuntimeTest, ut_printMemoryAttr)
{
    MOCKER(HcclCheckLogLevel).stubs().will(returnValue(false));

    const void* memAddr = nullptr;
    auto ret = PrintMemoryAttr(memAddr);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(RuntimeTest, test_GetMsTimeFromExecTimeout)
{
    // 记录用户设置数值，用于用例执行结束后配置恢复
    s32 execTimeoutActual = 0;
    s32 execTimeoutTmp = GetExternalInputHcclExecTimeOut();
    std::string setTimeOutValue;
    HcclResult ret;
    s32 INPUT_MIN;
    s32 INPUT_MAX;
    s32 EXPECT_MIN;
    s32 EXPECT_MAX;

    DevType deviceType;
    ret = hrtGetDeviceType(deviceType); // 80和71要分开
    if (deviceType == DevType::DEV_TYPE_910_93
        || deviceType == DevType::DEV_TYPE_910B) { // 81/71 execTimeout范围[0, 2147483647]s
        INPUT_MIN = 0;
        INPUT_MAX = HCCL_EXEC_TIME_OUT_S_910_93;
        EXPECT_MIN = 5000;
        EXPECT_MAX = HCCL_EXEC_TIME_OUT_S_910_93;
    } else { // 非81/71 execTimeout范围[1, 17340]s
        INPUT_MIN = 1;
        INPUT_MAX = HCCL_EXEC_TIME_OUT_S;
        EXPECT_MIN = 73000; // execTimeout最小赋值为68s
        EXPECT_MAX = 17345000;
    }
    // case1: execTimeOut = INPUTMIN
    setTimeOutValue = to_string(INPUT_MIN);
    ret = SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    execTimeoutActual = GetMsTimeFromExecTimeout();
    EXPECT_EQ(EXPECT_MIN, execTimeoutActual);

    // case2: execTimeOut = INPUTMAX
    setTimeOutValue = to_string(INPUT_MAX);
    ret = SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    execTimeoutActual = GetMsTimeFromExecTimeout();
    EXPECT_EQ(EXPECT_MAX, execTimeoutActual);

    // case3: execTimeOut = [INPUT_MIN, INPUT_MAX)
    srand(time(0));
    setTimeOutValue = to_string(rand() % INPUT_MAX);
    ret = SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    execTimeoutActual = GetMsTimeFromExecTimeout();
    bool flag = false;
    if ((execTimeoutActual > 0) && (execTimeoutActual < 0x7FFFFFFF)) {
        flag = true;
    }
    EXPECT_EQ(true, flag);

    // 恢复用户设置数值
    setTimeOutValue = to_string(execTimeoutTmp);
    SetHccLExecTimeOut(setTimeOutValue.c_str(), HcclExecTimeoutSet::HCCL_EXEC_TIMEOUT_SET_BY_ENV);
}

TEST_F(RuntimeTest, test_adapter_interface_pre_exec)
{
    MOCKER(GetWorkflowMode).stubs().will(returnValue(HcclWorkflowMode::HCCL_WORKFLOW_MODE_OP_BASE));

    MOCKER(hrtRaGetSockets).stubs().will(returnValue(SOCK_EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).stubs().will(returnValue(0));

    struct SocketInfoT conn10[2];
    hrtRaBlockGetSockets(0, conn10, 2);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtRaSocketListenStop_nodev)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_socket_listen_stop).stubs().will(returnValue(SOCK_ENODEV));

    SocketListenInfoT sockListen;
    ret = hrtRaSocketListenStop(&sockListen, 1);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}

const unsigned int TOPOLOGY_CONVERT[4][4]
    = {TOPOLOGY_PIX,     TOPOLOGY_SIO,     TOPOLOGY_HCCS_SW, TOPOLOGY_HCCS_SW, TOPOLOGY_SIO, TOPOLOGY_PIX,
       TOPOLOGY_HCCS_SW, TOPOLOGY_HCCS_SW, TOPOLOGY_HCCS_SW, TOPOLOGY_HCCS_SW, TOPOLOGY_PIX, TOPOLOGY_SIO,
       TOPOLOGY_HCCS_SW, TOPOLOGY_HCCS_SW, TOPOLOGY_SIO,     TOPOLOGY_PIX};

HcclResult stub_hrtGetPairPhyDevicesInfo(u32 phyDevId, u32 otherPhyDevId, s64* pValue)
{
    if (phyDevId > 3 || otherPhyDevId > 3) {
        *pValue = 0;
    } else {
        *pValue = TOPOLOGY_CONVERT[phyDevId][otherPhyDevId];
    }

    return HCCL_SUCCESS;
}
#if 0 // 执行失败
TEST_F(RuntimeTest, test_hrtGetPairPhyDevicesInfo)
{
    MOCKER(hrtGetPairPhyDevicesInfo)
    .stubs()
    .will(invoke(stub_hrtGetPairPhyDevicesInfo));
    std::unordered_map<u32, u32> pairLinkCounter;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)] = 0;
    pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)] = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                continue;
            }
            LinkTypeInServer linkType;
            hrtGetPairDeviceLinkType(i, j, linkType);
            pairLinkCounter[static_cast<u32>(linkType)]++;
        }
    }

    EXPECT_EQ(pairLinkCounter[static_cast<u32>(LinkTypeInServer::SIO_TYPE)], 4);
    EXPECT_EQ(pairLinkCounter[static_cast<u32>(LinkTypeInServer::HCCS_SW_TYPE)], 8);
    GlobalMockObject::verify();
}
#endif
rtError_t stub_rtGetPairPhyDevicesInfo(uint32_t phyDevId, uint32_t otherPhyDevId, s64* pValue)
{
    if (phyDevId > 3 || otherPhyDevId > 3) {
        *pValue = 0;
    } else {
        *pValue = TOPOLOGY_CONVERT[phyDevId][otherPhyDevId];
    }

    return ACL_RT_SUCCESS;
}

TEST_F(RuntimeTest, test_hrtGetPairPhyDevicesInfo1)
{
    MOCKER(hrtGetPairPhyDevicesInfo).stubs().will(invoke(stub_rtGetPairPhyDevicesInfo));

    s64 linkTypeRaw = 0;
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {
            if (i == j) {
                continue;
            }
            hrtGetPairPhyDevicesInfo(i, j, &linkTypeRaw);
            if (i > 3 || j > 3) {
                EXPECT_EQ(linkTypeRaw, 0);
            } else {
                EXPECT_EQ(linkTypeRaw, TOPOLOGY_CONVERT[i][j]);
            }
        }
    }
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, test_hrtHalHostRegister)
{
    void* addr = new (std::nothrow) u64[10];
    u64 size;
    u32 flag;
    u32 devid;
    void* dev;
    HcclResult ret = hrtHalHostRegister(addr, size, flag, devid, dev);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    u64* addr1 = new (std::nothrow) u64[256 * 1024 * 1024 / sizeof(u64)];
    ret = hrtHalHostRegister(addr1, 256 * 1024 * 1024, flag, devid, dev);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete[] static_cast<u64*>(addr);
    delete[] addr1;
}

TEST_F(RuntimeTest, test_hrtHalHostUnregister)
{
    void* addr = new u32(0);
    u32 devid;
    HcclResult ret = hrtHalHostUnregister(addr, devid);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete addr;
}

TEST_F(RuntimeTest, test_hrtHalHostUnregisterExNull)
{
    void* addr = new u32(0);
    u32 devid;
    DlHalFunction::GetInstance().dlHalHostUnregisterEx = nullptr;
    HcclResult ret = hrtHalHostUnregisterEx(addr, devid, HOST_MEM_MAP_DEV_PCIE_TH);
    EXPECT_EQ(ret, HCCL_E_DRV);
    delete addr;
}

TEST_F(RuntimeTest, test_hrtHalHostUnregisterEx)
{
    void* addr = new u32(0);
    u32 devid;
    DlHalFunction::GetInstance().DlHalFunctionInit();
    HcclResult ret = hrtHalHostUnregisterEx(addr, devid, HOST_MEM_MAP_DEV_PCIE_TH);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete addr;
}

TEST_F(RuntimeTest, test_hrtHalSensorNodeRegister)
{
    DlHalFunction::GetInstance().DlHalFunctionInit();
    uint64_t handle = 0;
    auto ret = hrtHalSensorNodeRegister(0, &handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtHalSensorNodeUnregister)
{
    DlHalFunction::GetInstance().DlHalFunctionInit();
    uint64_t handle = 1;
    auto ret = hrtHalSensorNodeUnregister(0, handle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, test_hrtHalSensorNodeUpdateState)
{
    DlHalFunction::GetInstance().DlHalFunctionInit();
    uint64_t handle = 1;
    auto ret = hrtHalSensorNodeUpdateState(
        0, handle, HAL_GENERAL_SOFTWARE_FAULT_NORMAL_RESOURCE_RECYCLE_FAILED,
        HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_ONE_TIME);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, rt_ra_get_qp_depth)
{
    HcclResult ret;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    int rdma;
    RdmaHandle rdmaHandle = &rdma;
    unsigned int tempDepth;
    unsigned int qpNum;

    ret = HrtRaGetQpDepth(rdmaHandle, &tempDepth, &tempDepth);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, rt_ra_set_qp_depth)
{
    HcclResult ret;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    int rdma;
    RdmaHandle rdmaHandle = &rdma;
    unsigned int tempDepth{};
    unsigned int qpNum;

    ret = HrtRaSetQpDepth(rdmaHandle, tempDepth, &tempDepth);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_rt_ra_epoll)
{
    HcclResult ret;

    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    std::vector<SocketEventInfo> eventInfos(1);
    struct socket_peer_info info;
    int sockfd = socket(AF_INET, SOCK_STREAM, 0);
    info.fd = sockfd;
    unsigned int events_num = 0;
    int epollFd;
    FdHandle fd = (void*)&info;

    ret = hrtRaCreateEventHandle(epollFd);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtRaCtlEventHandle(epollFd, fd, EPOLL_CTL_ADD, HcclEpollEvent::HCCL_EPOLLIN);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtRaWaitEventHandle(epollFd, eventInfos, 1, 1, events_num);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = hrtRaDestroyEventHandle(epollFd);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    close(sockfd);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_rt_hrt_notify_reset_test)
{
    int nty = 1;
    HcclRtNotify notify = &nty;
    auto ret = hrtNotifyReset(notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_rt_hrt_task_abort_callback_test)
{
    int32_t ProcessTaskAbortHandleCallback(
        int32_t devId, aclrtDeviceTaskAbortStage stage, uint32_t timeout, void* args);
    void* ptr = nullptr;
    auto ret = hrtTaskAbortHandleCallback(ProcessTaskAbortHandleCallback, nullptr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_rt_hrt_Resource_Clean_test)
{
    auto ret = hrtResourceClean();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_hrtRaRdevGetPortStatus)
{
    DlRaFunction::GetInstance().DlRaFunctionInit();
    int a = 0;
    RdmaHandle handle = &a;
    enum port_status* status;
    hrtRaRdevGetPortStatus(handle, status);
}

TEST_F(RuntimeTest, ut_hrtRaGetCqeErrInfoList)
{
    DlRaFunction::GetInstance().DlRaFunctionInit();
    int a = 0;
    RdmaHandle handle = &a;
    struct cqe_err_info err_info_list[1] = {};
    u32 num = 0;
    hrtRaGetCqeErrInfoList(handle, err_info_list, &num);
}

TEST_F(RuntimeTest, ut_hrtGetPairDevicePhyId_0)
{
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    LinkTypeInServer linkType = LinkTypeInServer::SIO_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkType))
        .will(returnValue(HCCL_SUCCESS));

    u32 localPhyId = 0;
    u32 pairPhyId = 0;
    auto ret = hrtGetPairDevicePhyId(localPhyId, pairPhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(pairPhyId, 1);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_hrtGetPairDevicePhyId_16)
{
    DevType devType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(devType)).will(returnValue(HCCL_SUCCESS));

    LinkTypeInServer linkTypeHCCS = LinkTypeInServer::HCCS_TYPE;
    MOCKER(hrtGetPairDeviceLinkType)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBound(linkTypeHCCS))
        .will(returnValue(HCCL_SUCCESS));

    u32 localPhyId = 16;
    u32 pairPhyId = 0;
    auto ret = hrtGetPairDevicePhyId(localPhyId, pairPhyId);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();
}

aclError aclrtGetDeviceInfo_stub(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
    static std::array<int64_t, 7> vals = {0x0, 0x1c, 0x1d, 0x18, 0x19, 0x14, 0x15};
    static int flag = 0;
    if (!flag) {
        ++flag;
        return 0x07110001;
    }
    *value = vals[flag++ - 1];
    return ACL_SUCCESS;
}

TEST_F(RuntimeTest, ut_hrtGetHccsPortNum)
{
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    s32 portNum = 0;
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_E_NOT_SUPPORT);
    GlobalMockObject::verify();

    deviceType = DevType::DEV_TYPE_910_93;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));

    MOCKER(aclrtGetDeviceInfo).stubs().will(invoke(aclrtGetDeviceInfo_stub));
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_E_RUNTIME);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, -1);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 6);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 6);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 7);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 7);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 7);
    EXPECT_EQ(hrtGetHccsPortNum(0, portNum), HCCL_SUCCESS);
    EXPECT_EQ(portNum, 7);
}

TEST_F(RuntimeTest, ut_ra_socket_listen_stop_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_socket_listen_stop).stubs().will(returnValue(SOCK_EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).expects(once()).will(returnValue(0));

    SocketListenInfoT sockListen;
    ret = hrtRaSocketListenStop(&sockListen, 1);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_hrtRaSocketNonBlockBatchConnect)
{
    HcclResult ret = HCCL_SUCCESS;
    SocketConnectInfoT conn;
    u32 num = 1;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(ra_socket_batch_connect).expects(once()).will(returnValue(0));
    ret = hrtRaSocketNonBlockBatchConnect(&conn, num);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER(ra_socket_batch_connect).expects(once()).will(returnValue(SOCK_EAGAIN));
    ret = hrtRaSocketNonBlockBatchConnect(&conn, num);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
    GlobalMockObject::verify();

    MOCKER(ra_socket_batch_connect).expects(once()).will(returnValue(-1));
    ret = hrtRaSocketNonBlockBatchConnect(&conn, num);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_ra_socket_batch_close_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_socket_batch_close).stubs().will(returnValue(SOCK_EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).expects(once()).will(returnValue(0));

    SocketCloseInfoT sockConn;
    ret = hrtRaSocketBatchClose(&sockConn, 1, 1);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_hrtRaNonBlockGetSockets)
{
    HcclResult ret = HCCL_SUCCESS;
    u32 role = 0;
    SocketInfoT conn;
    u32 num = 1;
    u32 connectedNum = 0;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(ra_get_sockets).expects(once()).will(returnValue(0));
    ret = hrtRaNonBlockGetSockets(role, &conn, num, &connectedNum);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();

    MOCKER(ra_get_sockets).expects(once()).will(returnValue(SOCK_EAGAIN));
    ret = hrtRaNonBlockGetSockets(role, &conn, num, &connectedNum);
    EXPECT_EQ(ret, HCCL_E_AGAIN);
    GlobalMockObject::verify();

    MOCKER(ra_get_sockets).expects(once()).will(returnValue(-1));
    ret = hrtRaNonBlockGetSockets(role, &conn, num, &connectedNum);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_ra_init_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_init).stubs().will(returnValue(HCCP_EAGAIN));

    MOCKER(GetExternalInputHcclLinkTimeOut).expects(once()).will(returnValue(0));

    ra_init_config raConfig;
    raConfig.phy_id = 0;
    raConfig.nic_position = 0;
    ret = HrtRaInit(&raConfig);
    EXPECT_EQ(ret, HCCL_E_TIMEOUT);
    GlobalMockObject::verify();

    MOCKER(ra_init).stubs().will(returnValue(328002));
    struct ra_init_config config;
    ret = HrtRaInit(&config);
    EXPECT_EQ(ret, HCCL_E_PARA);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_ra_socket_listen_stop_error)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    MOCKER(ra_socket_listen_stop).stubs().will(returnValue(SOCK_EADDRINUSE));

    SocketListenInfoT sockListen;
    ret = hrtRaSocketListenStop(&sockListen, 1);
    EXPECT_EQ(ret, HCCL_E_TCP_CONNECT);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_hrtMemAsyncCopyWithoutCheckKind)
{
    HcclResult ret = HCCL_SUCCESS;

    int dstBuf = 1;
    int srcBuf = 2;
    u64 destMax = 1;
    u64 count = 1;
    HcclRtMemcpyKind kind = HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE;
    rtStream_t stream;

    MOCKER(rtsMemcpyAsync).stubs().will(returnValue(RT_ERROR_NONE));

    hrtStreamCreateWithFlags(&stream, HCCL_STREAM_PRIORITY_HIGH, ACL_STREAM_FAST_LAUNCH | ACL_STREAM_FAST_SYNC);
    ret = hrtMemAsyncCopyWithoutCheckKind(&dstBuf, destMax, &srcBuf, count, kind, stream);
    hrtStreamDestroy(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_hrtGetDeviceRefresh)
{
    s32 deviceLogicId = 0;
    HcclResult ret;
    MOCKER(aclrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(1));
    ret = hrtGetDeviceRefresh(&deviceLogicId);
    EXPECT_EQ(ret, HCCL_E_RUNTIME);
    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_hrtGetDeviceTypeBySocVersion)
{
    log_level_set_stub(3);
    DevType deviceType;
    std::string str{"Ascend310B1"};
    HcclResult cc = hrtGetDeviceTypeBySocVersion(str, deviceType);
    EXPECT_EQ(HCCL_SUCCESS, cc);
    EXPECT_EQ(deviceType, DevType::DEV_TYPE_310P3);
    log_level_set_stub(3);
}

const char* fake_rtGetSocVersionV91095()
{
    static std::string socName = "Ascend950";
    return socName.c_str();
}

TEST_F(RuntimeTest, ut_hrtGetDeviceType_950_return_ok)
{
    CallBackInitRts();
    MOCKER(aclrtGetSocName).stubs().will(invoke(fake_rtGetSocVersionV91095));

    DevType deviceType;
    HcclResult cc = hrtGetDeviceType(deviceType);
    EXPECT_EQ(cc, HCCL_SUCCESS);
    EXPECT_EQ(deviceType, DevType::DEV_TYPE_950);
}

TEST_F(RuntimeTest, ut_hrtGetDeviceTypeBySocVersion_950_return_ok)
{
    log_level_set_stub(3);
    DevType deviceType;
    std::string str{"Ascend950DT_9591"};
    HcclResult cc = hrtGetDeviceTypeBySocVersion(str, deviceType);
    EXPECT_EQ(cc, HCCL_SUCCESS);
    EXPECT_EQ(deviceType, DevType::DEV_TYPE_950);
    log_level_set_stub(3);
}

TEST_F(RuntimeTest, ut_hrtRaTypicalSendWr)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(ra_typical_send_wr).stubs().will(returnValue(SOCK_EAGAIN)).then(returnValue(0));

    ret = hrtRaTypicalSendWr(NULL, NULL, NULL);
}

TEST_F(RuntimeTest, ut_hrtRaQpDestroy)
{
    s32 ret = HCCL_SUCCESS;

    void* handle = nullptr;
    MOCKER(ra_qp_destroy).stubs().will(returnValue(ROCE_EAGAIN)).then(returnValue(0));

    ret = HrtRaQpDestroy(handle);
}

TEST_F(RuntimeTest, ut_hrtRaSendWr)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(ra_send_wr).stubs().will(returnValue(ROCE_EAGAIN)).then(returnValue(0));

    ret = HrtRaSendWr(NULL, NULL, NULL);
}

TEST_F(RuntimeTest, ut_hrtRaDeInit)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(ra_deinit).stubs().will(returnValue(HCCP_EAGAIN)).then(returnValue(0));

    struct ra_init_config raConfig;
    raConfig.phy_id = 0;
    raConfig.nic_position = 0;
    ret = HrtRaDeInit(&raConfig);
}

TEST_F(RuntimeTest, ut_hrtRaSocketListenStart)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(hrtRaSocketNonBlockListenStart).stubs().will(returnValue(HCCL_E_AGAIN)).then(returnValue(0));

    SocketListenInfoT sockListen;
    ret = hrtRaSocketListenStart(&sockListen, 1);
}

TEST_F(RuntimeTest, ut_rt_ra_socket_batch_connect)
{
    s32 ret = HCCL_SUCCESS;

    MOCKER(ra_socket_batch_connect).stubs().will(returnValue(SOCK_EAGAIN)).then(returnValue(0));

    SocketConnectInfoT sockConn;
    ret = hrtRaSocketBatchConnect(&sockConn, 1, 1);
}

HcclResult stub_hrtRaGetTlsEnable(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 2;
    return HCCL_SUCCESS;
}

TEST_F(RuntimeTest, st_test_HrtRaGetTlsEnable)
{
    bool tls_enable = false;
    struct ra_info info;
    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(invoke(stub_hrtRaGetTlsEnable));
    HcclResult ret = HrtRaGetTlsEnable(&info, &tls_enable);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

HcclResult
stub_hrtRaGetInterfaceVersion1(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 1;
    return HCCL_SUCCESS;
}

HcclResult
stub_hrtRaGetInterfaceVersion0(unsigned int phyId, unsigned int interfaceOpcode, unsigned int* interfaceVersion)
{
    *interfaceVersion = 0;
    return HCCL_SUCCESS;
}

TEST_F(RuntimeTest, ut_IsSupportRaSocketAbort_Support)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool isSupportRaSocketAbort;

    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(invoke(stub_hrtRaGetInterfaceVersion1));

    ret = IsSupportRaSocketAbort(isSupportRaSocketAbort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(isSupportRaSocketAbort, true);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_IsSupportRaSocketAbort_NoSupport_Version0)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool isSupportRaSocketAbort;

    MOCKER(hrtRaGetInterfaceVersion).expects(atMost(1)).will(invoke(stub_hrtRaGetInterfaceVersion0));

    ret = IsSupportRaSocketAbort(isSupportRaSocketAbort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(isSupportRaSocketAbort, false);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, ut_IsSupportRaSocketAbort_NoSupport)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);
    bool isSupportRaSocketAbort;

    MOCKER(hrtRaGetInterfaceVersion).stubs().will(returnValue(HCCL_E_NOT_SUPPORT));

    ret = IsSupportRaSocketAbort(isSupportRaSocketAbort);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(isSupportRaSocketAbort, false);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, st_test_HcclNetDevGetTlsStatus)
{
    HcclNetDevCtx netDevCtx;
    HcclResult ret = HcclNetOpenDev(&netDevCtx, NicType::DEVICE_NIC_TYPE, 0, 0, HcclIpAddress("6.6.6.6"));
    EXPECT_EQ(ret, HCCL_SUCCESS);
    TlsStatus status;
    ret = HcclNetDevGetTlsStatus(netDevCtx, &status);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclNetCloseDev(netDevCtx);
}

TEST_F(RuntimeTest, ut_ra_socket_listen_start_timeout)
{
    HcclResult ret;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    MOCKER(ra_socket_listen_start).stubs().will(returnValue(128205));
    SocketListenInfoT sockListen1;
    ret = hrtRaSocketNonBlockListenStart(&sockListen1, 1);
    EXPECT_EQ(ret, HCCL_E_UNAVAIL);

    GlobalMockObject::verify();
}

TEST_F(RuntimeTest, stt_hrtRaGetDevCapInfo)
{
    HcclResult ret = HCCL_SUCCESS;
    RdmaHandle rdmaHandle;
    SrqInfo srqInfo;

    ret = hrtRaCreateSrq(rdmaHandle, srqInfo);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_test_hrtRaNormalQpCreate)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* handle = (void*)0x01;
    struct ibv_qp_init_attr initAttr;
    initAttr.qp_type = IBV_QPT_RC;
    void* qpHandle = nullptr;
    struct ibv_qp* qp = nullptr;

    MOCKER(ra_normal_qp_create).expects(once()).will(returnValue(0));
    ret = hrtRaNormalQpCreate(handle, &initAttr, qpHandle, qp);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_test_hrtRaNormalQpDestroy)
{
    HcclResult ret = HCCL_SUCCESS;
    ret = DlRaFunction::GetInstance().DlRaFunctionInit();
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* qpHandle = (void*)0x01;

    ret = hrtRaNormalQpDestroy(qpHandle);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_hrtStreamCreate_and_destroy_test)
{
    s32 ret = HCCL_SUCCESS;
    HcclRtStream stream = NULL;

    s32 device_id = 0;
    ret = hrtSetDevice(device_id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 申请stream
    ret = hrtStreamCreate(&stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 销毁资源
    ret = hrtStreamDestroy(stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(RuntimeTest, ut_hrtCtxGetCurrent_null_ctx)
{
    MOCKER(aclrtGetCurrentContext).stubs().will(returnValue(ACL_ERROR_RT_CONTEXT_NULL));

    HcclResult ret = HCCL_SUCCESS;
    aclrtContext ctx = nullptr;
    ret = hrtCtxGetCurrent(&ctx);

    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_EQ(ctx, nullptr);
}

TEST_F(RuntimeTest, Ut_IsSupportRaSocketAsync_When_RaSocketSupportAsync_Expect_True)
{
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    u32 configVersion = 2;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&configVersion))
        .then(returnValue(HCCL_SUCCESS));
    bool isSupportHdcAsync;
    HcclResult ret = IsSupportHdcAsync(isSupportHdcAsync);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_TRUE(isSupportHdcAsync);
}

TEST_F(RuntimeTest, Ut_IsSupportRaSocketAsync_When_RaSocketNotSupportAsync_Expect_False)
{
    MOCKER(hrtGetDevice).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER(hrtGetDevicePhyIdByIndex).stubs().will(returnValue(HCCL_SUCCESS));

    u32 configVersion = 1;
    MOCKER(hrtRaGetInterfaceVersion)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), outBoundP(&configVersion))
        .will(returnValue(HCCL_E_NETWORK))
        .then(returnValue(HCCL_E_NOT_SUPPORT))
        .then(returnValue(HCCL_SUCCESS));

    bool isSupportHdcAsync;
    HcclResult ret = IsSupportHdcAsync(isSupportHdcAsync);
    EXPECT_EQ(ret, HCCL_E_NETWORK);
    EXPECT_FALSE(isSupportHdcAsync);

    ret = IsSupportHdcAsync(isSupportHdcAsync);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isSupportHdcAsync);

    ret = IsSupportHdcAsync(isSupportHdcAsync);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_FALSE(isSupportHdcAsync);
}
