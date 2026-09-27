/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_TEST_MOCK_SETUP_H
#define HCCLV2_TEST_MOCK_SETUP_H

#include <mockcpp/mokc.h>
#include <mockcpp/mockcpp.hpp>

/**
 * @brief 设置通信 Notify 相关的公共 mock。
 *
 * 在测试类的 SetUp() 中调用此宏，消除跨文件重复的 MOCKER 设置代码。
 * 前提条件：调用方需定义以下成员变量：
 *   u32 fakeDevPhyId; u64 fakeNotifyHandleAddr; u32 fakeNotifyId;
 *   u32 fakeOffset; char fakeName[65];
 * 以及已 #include DevType、DevId 等类型定义。
 */
#define SETUP_COMM_NOTIFY_MOCKS()                                                                         \
    do {                                                                                                  \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                                \
        MOCKER(HrtNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));                 \
        MOCKER(HrtNotifyCreateWithFlag).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));         \
        MOCKER(HrtGetNotifyID).stubs().will(returnValue(fakeNotifyId));                                   \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId))); \
        MOCKER(HrtIpcSetNotifyName)                                                                       \
            .stubs()                                                                                      \
            .with(mockcpp::any(), outBoundP(fakeName, sizeof(fakeName)), mockcpp::any());                 \
        MOCKER(HrtNotifyGetOffset).stubs().will(returnValue(fakeOffset));                                 \
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));               \
    } while (0)

/**
 * @brief 设置 CCU Transport 相关的公共 mock。
 *
 * 在测试 SetUp 或用例中调用此宏，消除跨文件重复的 CCU DeviceManager mock 设置代码。
 * 前提条件：调用方需定义局部变量 jfcHandle（void* 类型）。
 */
#define SETUP_CCU_TRANSPORT_MOCKS()                                                                     \
    do {                                                                                                \
        MOCKER(CcuDeviceManager::AllocXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));       \
        MOCKER(CcuDeviceManager::AllocCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));      \
        MOCKER(CcuDeviceManager::ConfigChannel).defaults().will(returnValue(HcclResult::HCCL_SUCCESS)); \
        MOCKER(CcuDeviceManager::ReleaseXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));     \
        MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));    \
        MOCKER(HrtGetDevice).defaults().will(returnValue(0));                                           \
        MOCKER(HrtRaUbUnimportJetty).defaults().will(returnValue(0));                                   \
        MOCKER(HrtGetDeviceType).defaults().will(returnValue(DevType::DEV_TYPE_950));                   \
        MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<DevId>(0)));       \
        MOCKER(HrtRaUbCreateJetty).defaults().will(returnValue(HrtRaUbJettyCreatedOutParam()));         \
        MOCKER(HraGetDieAndFuncId).defaults().will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));   \
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));                               \
        MOCKER(RaUbImportJetty).defaults().will(returnValue(HrtRaUbJettyImportedOutParam()));           \
        MOCKER(HrtRaUbLocalMemReg).defaults().will(returnValue(HrtRaUbLocalMemRegOutParam()));          \
    } while (0)

/**
 * @brief 设置 Cnt Notify 相关的公共 mock。
 *
 * 在测试用例中调用此宏，消除跨文件重复的 Cnt Notify MOCKER 设置代码。
 * 前提条件：调用方需定义以下变量：
 *   u64 fakeNotifyHandleAddr; u32 fakeNotifyId;
 */
#define SETUP_CNT_NOTIFY_MOCKS()                                                               \
    do {                                                                                       \
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType::DEV_TYPE_910A2));           \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                     \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1))); \
        MOCKER(HrtCntNotifyCreate).stubs().will(returnValue((void*)(fakeNotifyHandleAddr)));   \
        MOCKER(HrtGetCntNotifyId).stubs().will(returnValue(fakeNotifyId));                     \
    } while (0)

/**
 * @brief 设置 CCU Transport 相关的公共 mock（带 AllocCke 返回值参数）。
 *
 * 与 SETUP_CCU_TRANSPORT_MOCKS 类似，但 AllocCke 可返回自定义值。
 * 前提条件：调用方需定义局部变量 jfcHandle（void* 类型）。
 */
#define SETUP_CCU_TRANSPORT_MOCKS_EX(allocCkeRet)                                                       \
    do {                                                                                                \
        MOCKER(CcuDeviceManager::AllocXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));       \
        MOCKER(CcuDeviceManager::AllocCke).defaults().will(returnValue(allocCkeRet));                   \
        MOCKER(CcuDeviceManager::ConfigChannel).defaults().will(returnValue(HcclResult::HCCL_SUCCESS)); \
        MOCKER(CcuDeviceManager::ReleaseXn).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));     \
        MOCKER(CcuDeviceManager::ReleaseCke).defaults().will(returnValue(HcclResult::HCCL_SUCCESS));    \
        MOCKER(HrtGetDevice).defaults().will(returnValue(0));                                           \
        MOCKER(HrtRaUbUnimportJetty).defaults().will(returnValue(0));                                   \
        MOCKER(HrtGetDeviceType).defaults().will(returnValue(DevType::DEV_TYPE_950));                   \
        MOCKER(HrtGetDevicePhyIdByUserDevId).defaults().will(returnValue(static_cast<DevId>(0)));       \
        MOCKER(HrtRaUbCreateJetty).defaults().will(returnValue(HrtRaUbJettyCreatedOutParam()));         \
        MOCKER(HraGetDieAndFuncId).defaults().will(returnValue(std::pair<uint32_t, uint32_t>(0, 0)));   \
        MOCKER(HrtRaUbCreateJfc).defaults().will(returnValue(jfcHandle));                               \
        MOCKER(RaUbImportJetty).defaults().will(returnValue(HrtRaUbJettyImportedOutParam()));           \
        MOCKER(HrtRaUbLocalMemReg).defaults().will(returnValue(HrtRaUbLocalMemRegOutParam()));          \
    } while (0)

/**
 * @brief 设置 Stream 设备相关的公共 mock（st_ins_rules 中重复的打桩代码）。
 *
 * 消除 st_ins_rules.cc 中多次重复的 stream/device mock 设置。
 */
#define SETUP_INS_RULES_STREAM_MOCKS()                                                                        \
    do {                                                                                                      \
        void* ptr = nullptr;                                                                                  \
        MOCKER(HrtStreamCreateWithFlags).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(ptr)); \
        MOCKER(HrtGetStreamId).stubs().with(mockcpp::any()).will(returnValue(0));                             \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                                    \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(1)));                \
    } while (0)

/**
 * @brief 设置 Stream 相关的公共 mock（st_stream 中重复的打桩代码）。
 *
 * 前提条件：调用方需定义局部变量 fakePtr, fakeId, fakeDevLogId, fakeDevPhyId, fakeSqId。
 */
#define SETUP_STREAM_MOCKS()                                                                              \
    do {                                                                                                  \
        MOCKER(HrtGetStreamId).stubs().will(returnValue(fakeId));                                         \
        MOCKER(HrtGetDevice).stubs().will(returnValue(fakeDevLogId));                                     \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(fakeDevPhyId))); \
        MOCKER(HrtStreamCreateWithFlags).stubs().will(returnValue(fakePtr));                              \
        MOCKER(HrtStreamGetSqId).stubs().will(returnValue(fakeSqId));                                     \
        MOCKER(HrtStreamDestroy).stubs();                                                                 \
    } while (0)

/**
 * @brief 设置设备相关的基础 mock（HrtGetDevice + HrtGetDevicePhyIdByUserDevId）。
 *
 * 在需要简单设备打桩的测试用例中调用。
 */
#define SETUP_DEVICE_MOCKS(devPhyId)                                                                  \
    do {                                                                                              \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                            \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().will(returnValue(static_cast<DevId>(devPhyId))); \
    } while (0)

/**
 * @brief 设置 HccpPeerManager 测试的公共 mock。
 */
#define SETUP_HCCP_PEER_MGR_MOCKS(phyId1, phyId2) \
    do {                                          \
        DevId fakedevPhyId = phyId1;              \
        DevId fakedevPhyId1 = phyId2;             \
        MOCKER(HrtGetDevicePhyIdByUserDevId)      \
            .stubs()                              \
            .with(mockcpp::any())                 \
            .will(returnValue(fakedevPhyId))      \
            .then(returnValue(fakedevPhyId1));    \
        MOCKER(HrtRaInit).stubs().with();         \
        MOCKER(HrtRaDeInit).stubs().with();       \
    } while (0)

/**
 * @brief 设置 CommunicatorImpl::Init 相关的公共 mock。
 *
 * 在 MockCommunicatorImpl() 及 Init 相关的测试用例中调用，
 * 消除重复的 Hrt 接口与 CommunicatorImpl 子模块 Init mock 设置。
 */
#define SETUP_COMM_INIT_MOCKS()                                                                                     \
    do {                                                                                                            \
        MOCKER(HrtSetDevice).stubs().with(mockcpp::any()).will(ignoreReturnValue());                                \
        MOCKER(HrtGetDevice).stubs().will(returnValue(0));                                                          \
        MOCKER(HrtGetDeviceType).stubs().will(returnValue(DevType(DevType::DEV_TYPE_950)));                         \
        MOCKER(HrtOpenTsdProcess)                                                                                   \
            .stubs()                                                                                                \
            .with(mockcpp::any(), mockcpp::any())                                                                   \
            .will(returnValue(HcclResult::HCCL_SUCCESS));                                                           \
        MOCKER(HrtRaTlvInit).stubs().with(mockcpp::any()).will(returnValue(HcclResult::HCCL_SUCCESS));              \
        MOCKER(HrtGetDevicePhyIdByUserDevId).stubs().with(mockcpp::any()).will(returnValue(static_cast<DevId>(1))); \
        MOCKER(RaInit).stubs().with(mockcpp::any()).will(returnValue(0));                                           \
        MOCKER(RaTlvInit).stubs().with(mockcpp::any()).will(returnValue(0));                                        \
        MOCKER_CPP(&CommunicatorImpl::InitRankGraph, void (CommunicatorImpl::*)(const std::string&))                \
            .stubs()                                                                                                \
            .with(mockcpp::any())                                                                                   \
            .will(ignoreReturnValue());                                                                             \
        MOCKER_CPP(&CommunicatorImpl::InitNotifyManager).stubs().will(ignoreReturnValue());                         \
        MOCKER_CPP(&CommunicatorImpl::InitStreamManager).stubs().will(ignoreReturnValue());                         \
        MOCKER_CPP(&CommunicatorImpl::InitSocketManager).stubs().will(ignoreReturnValue());                         \
        MOCKER_CPP(&CommunicatorImpl::InitRmaConnManager).stubs().will(ignoreReturnValue());                        \
        MOCKER_CPP(&CommunicatorImpl::InitDataBufferManager).stubs().will(ignoreReturnValue());                     \
        MOCKER_CPP(&CommunicatorImpl::InitHostDeviceSyncNotifyManager).stubs().will(ignoreReturnValue());           \
        MOCKER_CPP(&CommunicatorImpl::InitCollService).stubs().will(ignoreReturnValue());                           \
        MOCKER_CPP(&CommunicatorImpl::InitMirrorTaskManager).stubs().will(ignoreReturnValue());                     \
        MOCKER_CPP(&CommunicatorImpl::InitProfilingReporter).stubs().will(returnValue(HcclResult::HCCL_SUCCESS));   \
        MOCKER_CPP(&CcuComponent::Init).stubs().will(ignoreReturnValue());                                          \
        MOCKER_CPP(&CcuResBatchAllocator::Init).stubs().will(ignoreReturnValue());                                  \
    } while (0)

/**
 * @brief 设置 Socket::GetStatus 的公共 mock。
 *
 * 消除 ut/st_ccu_transport_group.cc 中重复的 Socket 状态打桩代码。
 */
#define SETUP_SOCKET_STATUS_MOCKS()                                    \
    do {                                                               \
        MOCKER_CPP(&Socket::GetStatus)                                 \
            .stubs()                                                   \
            .will(returnValue((SocketStatus)SocketStatus::INIT))       \
            .then(returnValue((SocketStatus)SocketStatus::CONNECTING)) \
            .then(returnValue((SocketStatus)SocketStatus::OK))         \
            .then(returnValue((SocketStatus)SocketStatus::TIMEOUT));   \
    } while (0)

/**
 * @brief 打桩 MemTransportManager::GetOpbasedTransport。
 *
 * 消除 st_ins_rules.cc 中多次重复的 transport mock 代码。
 * 前提条件：调用方需定义局部变量 stubTransportPtr。
 */
#define MOCK_GET_OPBASED_TRANSPORT(stubTransportPtr)      \
    MOCKER_CPP(&MemTransportManager::GetOpbasedTransport) \
        .stubs()                                          \
        .with(mockcpp::any(), mockcpp::any())             \
        .will(returnValue(stubTransportPtr))

/**
 * @brief 设置 LocalRmaBuffer 相关的公共 mock。
 *
 * 消除 st_ins_rules.cc 中多次重复的 RMA buffer mock 代码。
 * 前提条件：调用方需定义局部变量 comm。
 */
#define SETUP_LOCAL_RMA_BUF_MOCK(comm)                                                                                 \
    std::shared_ptr<DevBuffer> rmaDevBuf = DevBuffer::Create(0x100, 0x100);                                            \
    StubLocalRmaBuffer stubLocalRmaBuf(rmaDevBuf, RmaType::UB);                                                        \
    LocalRmaBuffer* localRmaBuf = &stubLocalRmaBuf;                                                                    \
    LocalRmaBufManager localRmaBufMgr(comm);                                                                           \
    MOCKER_CPP(                                                                                                        \
        &LocalRmaBufManager::Get, LocalRmaBuffer* (LocalRmaBufManager::*)(const string&, const PortData&, BufferType)) \
        .stubs()                                                                                                       \
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())                                                          \
        .will(returnValue(localRmaBuf))

/**
 * @brief 设置 DataBufManager::Get 的公共 mock。
 */
#define SETUP_DATA_BUF_MOCK(buffer)                                                                              \
    do {                                                                                                         \
        DataBufManager dataBufMgr;                                                                               \
        MOCKER_CPP(&DataBufManager::Get).stubs().with(mockcpp::any(), mockcpp::any()).will(returnValue(buffer)); \
    } while (0)

/**
 * @brief 声明 Stream 测试变量并设置 Stream mock。
 *
 * 消除 st_stream.cc 中多次重复的变量声明 + SETUP_STREAM_MOCKS 代码。
 * 声明变量：fakePtr, fakeId, fakeDevLogId, fakeDevPhyId, fakeSqId, fakeStmMode。
 */
#define SETUP_STREAM_TEST()   \
    void* fakePtr = (void*)1; \
    u32 fakeId = 1;           \
    s32 fakeDevLogId = 1;     \
    s32 fakeDevPhyId = 1;     \
    u32 fakeSqId = 2;         \
    u64 fakeStmMode = 3;      \
    SETUP_STREAM_MOCKS()

#endif // HCCLV2_TEST_MOCK_SETUP_H
