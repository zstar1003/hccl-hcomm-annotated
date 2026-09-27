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
#include <algorithm>
#include <list>
#include <vector>
#include <string>
#include <securec.h>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#define private public
#include "hccl_communicator.h"
#include "orion_adapter_rts.h"
#include "comm_manager.h"
#include "hcom_v2.h"
#include "param_check_v2.h"
#include "log.h"
#include "hccl_common_v2.h"
#include "communicator_impl.h"
#undef private
using namespace std;
using namespace Hccl;
#define HCCL_COM_DATA_SIZE 1024
#define HUAWEI_SECC_RET_TRANSFORM(ret) ((ret == EOK) ? HCCL_SUCCESS : ((ret == EINVAL) ? HCCL_E_PARA : HCCL_E_INTERNAL))
#define HUAWEI_SECC_RET_CHECK_AND_RETURN(ret) \
    do {                                      \
        switch (ret) {                        \
            case EOK:                         \
                return HCCL_SUCCESS;          \
            case EINVAL:                      \
                return HCCL_E_PARA;           \
            default:                          \
                return HCCL_E_INTERNAL;       \
        }                                     \
    } while (0)

extern HcclResult HcomSetGradFusionByIndex(const char* group, u32 segmentNum, const u32* IdxList);
extern HcclResult HcomSetGradFusionBySize(const char* group, u32 segmentNum, const float* sizeList);
extern HcclResult HcomDestroyBackloggedGroup(const std::string& group);

class HcomTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcomTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcomTest TearDown" << std::endl; }
    virtual void SetUp() { std::cout << "A Test case in HcomTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in HcomTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }
};

void* sal_malloc(u32 size)
{
    if (size == 0) {
        HCCL_ERROR("errNo[0x%016llx] sal malloc fail, size[%u], return NULL", HCCL_ERROR_CODE(HCCL_E_MEMORY), size);
        return nullptr;
    }

    return malloc(size);
}

HcclResult sal_memset(void* dest, size_t destMaxSize, int c, size_t count)
{
    CHK_PTR_NULL(dest);
    s32 ret = memset_s(dest, destMaxSize, c, count);
    if (ret != EOK) {
        HCCL_ERROR(
            "errNo[0x%016llx] In sal_memset, memset_s failed. errorno[%d], params: dest[%p], "
            "destMaxSize[%d], c[%d], count[%d]",
            HCCL_ERROR_CODE(HUAWEI_SECC_RET_TRANSFORM(ret)), ret, dest, destMaxSize, c, count);
    }
    HUAWEI_SECC_RET_CHECK_AND_RETURN(ret);
}

void sal_free(void* ptr)
{
    if (ptr) {
        free(ptr);
    }
}

// 抽取公共初始化函数
void SetupCommonCommInfo()
{
    Hccl::CommParams commParams;
    auto hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;

    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    commInfoV2.hcclGroupMap = hcclGroupMap;
    commInfoV2.commParams = commParams;
    commInfoV2.isUsed = true;
    commInfoV2.pComm = hcclComm;
}

void SetupAndRunCollectiveOperation(
    const std::string& opName, std::function<int(const char*, s8*, s8*, s32, HcclDataType, s64, rtStream_t)> opFunc,
    s8 expectedValue)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator* hcclComm = new Hccl::HcclCommunicator(commParams);
    hcclComm->pimpl->rankSize = 4;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));

    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    // 设置 sendbuf 的值
    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    // 调用实际的通信操作
    ret = opFunc(opName.c_str(), sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, (s64)hcclComm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    // 验证结果
    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != expectedValue) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    delete hcclComm;
}

TEST_F(HcomTest, HcomGetRankIdV2_func)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;

    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    char group[] = "hccl_world_group";
    u32 rankId = 0;
    HcclResult ret = HcomGetRankIdV2(group, &rankId);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, HcomGetRankIdV2_func_3)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    char group[64] = "hccl_world_group";
    u32 rankId = 0;
    HcclResult ret = HcomGetRankIdV2(group, &rankId);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, HcomGetWorkspaceSubStreamNumV2_func)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::CalcCollOffloadOpRes).stubs().will(returnValue(HCCL_SUCCESS));

    char* group = nullptr;
    u64 streamNum = 1;
    u64 dataSize = 10;
    HcclCMDType optype = HcclCMDType::HCCL_CMD_ALLREDUCE;
    HcclResult ret = HcomGetWorkspaceSubStreamNumV2(group, streamNum, dataSize, HCCL_DATA_TYPE_INT8, optype);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char group1[64] = "testgroup";
    ret = HcomGetWorkspaceSubStreamNumV2(group1, streamNum, dataSize, HCCL_DATA_TYPE_INT8, optype);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclCMDType optype1 = HcclCMDType::HCCL_CMD_INVALID;
    ret = HcomGetWorkspaceSubStreamNumV2(group, streamNum, dataSize, HCCL_DATA_TYPE_INT8, optype1);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, HcomGetWorkspaceMemSizeV2_func)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::CalcCollOffloadOpRes).stubs().will(returnValue(HCCL_SUCCESS));
    std::string opType = "HcomAllReduce";
    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    char group[64] = "hccl_world_group";
    u64 memSize = 0;
    HcclResult ret = HcomGetWorkspaceMemSizeV2(opType, count, dataType, group, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    opType = "null";
    ret = HcomGetWorkspaceMemSizeV2(opType, count, dataType, group, memSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, HcomGetWorkspaceMemSizeV2_func_err)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::CalcCollOffloadOpRes).stubs().will(returnValue(HCCL_SUCCESS));
    std::string opType = "HcomAllReduce";
    u64 count = 0;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    char group[64] = "testGroup";
    u64 memSize = 0;
    HcclResult ret = HcomGetWorkspaceMemSizeV2(opType, count, dataType, group, memSize);
    EXPECT_EQ(ret, HCCL_E_NOT_FOUND);

    char worldGroup[64] = "hccl_world_group";
    opType = "null";
    ret = HcomGetWorkspaceMemSizeV2(opType, count, dataType, worldGroup, memSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, HcomSetWorkspaceResource_V2_func)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&CommunicatorImpl::SetCollOffloadSlaveStreams).stubs().will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::SetCollOffloadScratchBuf).stubs().will(returnValue(HCCL_SUCCESS));
    u64 memSize = 0;
    u64 stream_list_size = 0;
    vector<rtStream_t> streamList(stream_list_size);
    int a = 0;
    void* memptr = &a;
    char* group = nullptr;
    std::string tag = "testtag";
    HcclResult ret = HcomSetWorkspaceResourceV2(tag, group, streamList, memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    char group1[64] = "testgroup";
    ret = HcomSetWorkspaceResourceV2(tag, group1, streamList, memptr, memSize);
    EXPECT_EQ(ret, HCCL_E_PARA);

    HcclGroupParamsV2 hcclGroupParamsV2_1;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm1 = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2_1.pComm = nullptr;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap1 = {{"hccl_world_group", hcclGroupParamsV2_1}};
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = false;
    CommManager::GetInstance(1).GetCommInfoV2().hcclGroupMap = hcclGroupMap1;
    CommManager::GetInstance(1).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(1).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(1).GetCommInfoV2().pComm = hcclComm;
    ret = HcomSetWorkspaceResourceV2(tag, group, streamList, memptr, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, HcomAllGatherV2_func)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    ret = HcomAllGatherV2("testallgather", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_allreduce_v2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    ret = HcomAllReduceV2("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomReduceScatterV2_v2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomReduceScatterV2(
        "testReducescatter", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, st_hcom_allreduce_v2_2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    ret = HcomAllReduceV2("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_MAX, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_allreduce_v2_3)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }
    ret = HcomAllReduceV2("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_MIN, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomBroadcastV2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    hcclComm->pimpl->rankSize = 4;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomBroadcastV2("testBroadCast", sendbuf, count, HCCL_DATA_TYPE_INT8, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomAlltoAllV2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomAlltoAllV2(
        sendbuf, count, HCCL_DATA_TYPE_INT8, recvbuf, count, HCCL_DATA_TYPE_INT8, NULL, stream, "testAlltoall");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomReduceV2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    hcclComm->pimpl->rankSize = 4;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomReduceV2("testReduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, 0, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomAlltoAllVV2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    s32 rankSize = 1;
    vector<u64> sendCounts(rankSize, 0);
    vector<u64> recvCounts(rankSize, 0);
    vector<u64> sdispls(rankSize, 0);
    vector<u64> rdispls(rankSize, 0);

    ret = HcomAlltoAllVV2(
        sendbuf, sendCounts.data(), sdispls.data(), HCCL_DATA_TYPE_INT8, recvbuf, recvCounts.data(), rdispls.data(),
        HCCL_DATA_TYPE_INT8, NULL, stream, "testAlltoallv");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, St_HcomAlltoAllVCV2_When_Normal_Input_Expect_Success)
{
    void* sendBuf = nullptr;
    void* sendCountMatrix = (void*)0x1000000;
    void* recvBuf = nullptr;
    HcclDataType sendType = HCCL_DATA_TYPE_INT8;
    HcclDataType recvType = HCCL_DATA_TYPE_INT8;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclComm comm = static_cast<HcclComm>(hcclComm.get());
    int a = 0;
    rtStream_t stream = static_cast<rtStream_t>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAlltoAllVCV2(sendBuf, sendCountMatrix, sendType, recvBuf, recvType, NULL, stream, "testAlltoallvc");
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_hcom_HcomSend_v2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomSendV2("testSend", sendbuf, count, HCCL_DATA_TYPE_INT8, 1, 1, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_HcomRecv_v2)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    rtError_t rt_ret = RT_ERROR_NONE;
    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomReceiveV2("testRecv", sendbuf, count, HCCL_DATA_TYPE_INT8, 1, 1, NULL, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, ut_hcom_CheckOpParamV2) // 放最后
{
    char tag[] = "tag";
    u64 count = 64;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    char group[] = "group";
    u32 x = 1;
    void* stream = static_cast<void*>(&x);
    HcclResult ret = HcomCheckOpParamV2(tag, count, dataType, group, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_HcomGetAlltoAllStagedWorkSpaceMemSizeV2)
{
    u64 sendcounts = 64;
    u64 sdispls = 64;
    u64 recvCounts = 64;
    u64 rdispls = 64;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    char group[] = "hccl_world_group";
    u64 memSize = 10;

    MOCKER_CPP(&HcclCommunicator::CalcCollOffloadOpRes)
        .stubs()
        .with(mockcpp::any(), mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult ret = HcomGetAlltoAllStagedWorkSpaceMemSizeV2(
        group, &sendcounts, &sdispls, dataType, &recvCounts, &rdispls, dataType, memSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_Hcom_checkUserRankV2)
{
    const u32 totalRanks = 4;
    const u32 userRank = 8;

    HcclResult ret = HcomCheckUserRankV2(totalRanks, userRank);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, ut_Hcom_checkUserRankV2_2)
{
    const u32 totalRanks = 8;
    const u32 userRank = 4;

    HcclResult ret = HcomCheckUserRankV2(totalRanks, userRank);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, ut_Hcom_checkDataTypeV2)
{
    HcclDataType dataType = HCCL_DATA_TYPE_RESERVED;

    HcclResult ret = HcomCheckDataTypeV2(dataType);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}
TEST_F(HcomTest, st_Hcom_GetLocalRankSizeV2)
{
    char group[256] = "hccl_world_group";
    u32 localRankSize;

    HcclResult ret = HcomGetLocalRankSizeV2(group, &localRankSize);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}
TEST_F(HcomTest, st_Hcom_GetLocalRankIdV2)
{
    char group[256] = "hccl_world_group";
    u32 localRankId;

    HcclResult ret = HcomGetLocalRankIdV2(group, &localRankId);
    EXPECT_EQ(ret, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcomTest, st_HcomCalcTaskNumV2_When_Expect_ReturnlsHCCL_E_INTERNAL)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;

    HcomOpParam hcomOpParam;
    hcomOpParam.opType = "HcomAllReduce";
    hcomOpParam.count = 1;
    hcomOpParam.dataType = HCCL_DATA_TYPE_INT8;
    u32 taskNum = 0;
    hcclComm->GetCommImpl()->CollAlgComponentInit();
    auto ret = HcomCalcTaskNumV2(&hcomOpParam, taskNum);
    EXPECT_EQ(ret, HCCL_E_INTERNAL);
}

TEST_F(HcomTest, st_HcomCalcTaskNumV2_When_OpType_not_Find_Expect_ReturnlsHCCL_E_INTERNAL)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;

    HcomOpParam hcomOpParam;
    hcomOpParam.opType = "HcomAll";
    hcomOpParam.count = 1;
    hcomOpParam.dataType = HCCL_DATA_TYPE_INT8;
    u32 taskNum = 0;
    hcclComm->GetCommImpl()->CollAlgComponentInit();
    auto ret = HcomCalcTaskNumV2(&hcomOpParam, taskNum);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, HcomGraphAllGatherV2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupAndRunCollectiveOperation(
        "testallgather",
        [](const char* name, s8* sendbuf, s8* recvbuf, s32 count, HcclDataType type, s64 comm, rtStream_t stream) {
            return HcclCommGraphAllGatherV2(name, sendbuf, recvbuf, count, type, comm, stream);
        },
        2);
}

TEST_F(HcomTest, st_hcom_graph_allreduce_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupAndRunCollectiveOperation(
        "testallreduce",
        [](const char* name, s8* sendbuf, s8* recvbuf, s32 count, HcclDataType type, s64 comm, rtStream_t stream) {
            return HcomGraphAllReduceV2(name, sendbuf, recvbuf, count, type, HCCL_REDUCE_SUM, comm, stream);
        },
        2);
}

TEST_F(HcomTest, st_hcom_graph_HcomReduceV2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupAndRunCollectiveOperation(
        "testReduce",
        [](const char* name, s8* sendbuf, s8* recvbuf, s32 count, HcclDataType type, s64 comm, rtStream_t stream) {
            return HcomGraphReduceV2(name, sendbuf, recvbuf, count, type, HCCL_REDUCE_SUM, 0, comm, stream);
        },
        2);
}

TEST_F(HcomTest, st_hcom_HcomGraphReduceScatterV2_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupAndRunCollectiveOperation(
        "testReducescatter",
        [](const char* name, s8* sendbuf, s8* recvbuf, s32 count, HcclDataType type, s64 comm, rtStream_t stream) {
            return HcomGraphReduceScatterV2(name, sendbuf, recvbuf, count, type, HCCL_REDUCE_SUM, comm, stream);
        },
        2);
}

TEST_F(HcomTest, st_hcom_HcomGraphSendV2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator* hcclComm = new Hccl::HcclCommunicator(commParams);
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomGraphSendV2("testSend", sendbuf, count, HCCL_DATA_TYPE_INT8, 1, 1, (s64)hcclComm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    delete hcclComm;
}

TEST_F(HcomTest, st_hcom_HcomGraphReceiveV2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator* hcclComm = new Hccl::HcclCommunicator(commParams);
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));
    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp).stubs().will(returnValue(HCCL_SUCCESS));
    int ret = HCCL_SUCCESS;
    s8* sendbuf;
    s8* recvbuf;
    s32 rank = 0;
    s32 errors = 0;
    s32 count = HCCL_COM_DATA_SIZE;

    u32 a = 10;
    rtStream_t stream = reinterpret_cast<rtStream_t>(&a);
    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));

    for (int j = 0; j < count; j++) {
        sendbuf[j] = 2;
    }

    ret = HcomGraphReceiveV2("testRecv", sendbuf, count, HCCL_DATA_TYPE_INT8, 1, 1, (s64)hcclComm, stream);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    for (int j = 0; j < count; j++) {
        if (recvbuf[j] != 2) {
            errors++;
            break;
        }
    }

    sal_free(sendbuf);
    sal_free(recvbuf);
    delete hcclComm;
}

TEST_F(HcomTest, st_hcom_HcomGraphBroadcastV2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupAndRunCollectiveOperation(
        "testBroadCast",
        [](const char* name, s8* sendbuf, s8* recvbuf, s32 count, HcclDataType type, s64 comm, rtStream_t stream) {
            return HcomGraphBroadcastV2(name, sendbuf, count, type, 0, comm, stream);
        },
        2);
}

TEST_F(HcomTest, st_hcom_create_comm_buffer_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    hcclComm->GetCommImpl()->rankSize = 2;
    hcclComm->GetCommImpl()->InitDataBufferManager();
    int ret = HcomCreateCommCclBufV2(NULL);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* commInputPtr = nullptr;
    u64 commInputSize = 0;
    ret = HcomGetInCclBufV2(NULL, commInputPtr, commInputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* commOutputPtr = nullptr;
    u64 commOutputSize = 0;
    ret = HcomGetOutCclBufV2(NULL, commOutputPtr, commOutputSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* commIndirectInPtr = nullptr;
    u64 commIndirectInBufSize = 0;
    ret = HcomGetIndirectInCclBufV2(NULL, commIndirectInPtr, commIndirectInBufSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    void* commIndirectOutPtr = nullptr;
    u64 commIndirectOutBufSize = 0;
    ret = HcomGetIndirectOutCclBufV2(NULL, commIndirectOutPtr, commIndirectOutBufSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, st_hcom_graph_get_rank_id_and_size_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    HcclGroupParamsV2 hcclGroupParamsV2;
    Hccl::CommParams commParams;
    Hccl::HcclCommunicator* hcclComm = new Hccl::HcclCommunicator(commParams);
    uint32_t rankId;
    uint32_t rankSize;
    int ret = HcclCommGraphGetRankIdV2((s64)hcclComm, &rankId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclCommGraphGetRankSizeV2((s64)hcclComm, &rankSize);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    delete hcclComm;
}

TEST_F(HcomTest, st_hcom_get_comm_handle_by_group_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    HcclComm commHandle = nullptr;
    HcclResult ret = HcomGetCommHandleByGroupV2(group, &commHandle);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_get_topo_desc_v2_toposize_not_enough_When_ERROR_ReturnlsHCCL_E_PARA)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    HcclTopoDescs topoDescs;
    topoDescs.algSets = 0;
    topoDescs.rankSize = 1;
    uint32_t topoSize = 0;
    HcclResult ret = HcomGetTopoDescV2(group, &topoDescs, topoSize);
    EXPECT_EQ(ret, HCCL_E_PARA);
}

TEST_F(HcomTest, st_hcom_get_topo_desc_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    std::vector<HcclTopoDescs> topoDescs;
    char group[] = "hccl_world_group";
    const int size = 3;
    topoDescs.resize(size);
    for (int i = 0; i < size; i++) {
        topoDescs[i].algSets = 0;
        topoDescs[i].rankSize = 2;
    }
    uint32_t topoSize = 3;
    HcclResult ret = HcomGetTopoDescV2(group, topoDescs.data(), topoSize);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_get_dev_type_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    DevType devType;
    HcclResult ret = HcomGetDevTypeV2(devType);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_set_global_work_space_v2_When_Noraml_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    std::vector<void*> globalWorkSpaceAddr{};
    HcclResult ret = HcomSetGlobalWorkSpaceV2(group, globalWorkSpaceAddr);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_check_comm_validity_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    HcclResult ret = HcomCheckCommValidityV2(group);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_support_deteministic_optim_v2_When_Normal_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    bool isDeterministicOptim = false;
    HcclResult ret = HcomSupportDeterministicOptimV2(group, isDeterministicOptim);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_set_aiv_core_limit_v2_When_Noraml_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    u32 aivCoreLimit = 0;
    HcclResult ret = HcomSetAivCoreLimitV2(group, aivCoreLimit);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_set_qo_cfg_v2_When_Noraml_Expect_ReturnlsHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    char group[] = "hccl_world_group";
    const u32 qosCfg = 0;
    HcclResult ret = HcomSetQosCfgV2(group, qosCfg);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_hcom_unload_task_v2_When_Noraml_Expect_ReturnlsHCCL_SUCCESS)
{
    Hccl::CommParams commParams;
    auto hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    commInfoV2.hcclGroupMap = hcclGroupMap;
    commInfoV2.commParams = commParams;
    commInfoV2.isUsed = true;
    commInfoV2.pComm = hcclComm;
    hcclComm->pimpl->streamManager = std::make_unique<StreamManager>(hcclComm->pimpl.get());
    hcclComm->pimpl->dataBufferManager = std::make_unique<DataBufManager>();
    hcclComm->pimpl->localRmaBufManager = std::make_unique<LocalRmaBufManager>(*hcclComm->pimpl.get());
    hcclComm->pimpl->memTransportManager = std::make_unique<MemTransportManager>(*hcclComm->pimpl.get());
    hcclComm->pimpl->collServices[AcceleratorState::AICPU_TS]
        = std::make_shared<CollServiceAiCpuImpl>(hcclComm->pimpl.get());
    std::string group = "hccl_world_group";
    const char* tag = "1";
    HcclResult ret = HcomUnloadTaskV2(group, tag);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_Normal_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_DipHas0_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {0, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_OutputEq0_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 0;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {0, 0};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_CountNotFixCounts_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {0, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_CountTooLarge_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 0x7ffffffffff;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(HcomTest, St_HcomAllGatherVV2_When_DatatypeNotSurport_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCount = 1;
    void* recvBuf = FAKE_PTR;
    u64 recvCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 recvDispls[FAKE_RANK_SIZE] = {1, 1};
    HcclDataType sendType = FAKE_DATA_TYPE;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomAllGatherVV2(tag, sendBuf, sendCount, recvBuf, &recvCounts, &recvDispls, sendType, group, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_Normal_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_DipHas0_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {0, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_InuputCountEq0_Expect_Success)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {0, 0};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_SUCCESS);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_CountNotFixCounts_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_CountTooLarge_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 0xffffffff;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_E_PARA);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_DatatypeNotSurport_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_RESERVED;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcomTest, St_HcomReduceScatterVV2_When_ReduceOpNotSurport_Expect_Error)
{
    char tag[] = "tag";
    char group[256] = "hccl_world_group";
    constexpr u64 FAKE_RANK_SIZE = 2;
    constexpr void* FAKE_PTR = (void*)0x1000000;
    constexpr HcclDataType FAKE_DATA_TYPE = HCCL_DATA_TYPE_INT32;

    Hccl::CommParams commParams;
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = std::make_shared<Hccl::HcclCommunicator>(commParams);
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = FAKE_RANK_SIZE;
    HcclGroupParamsV2 hcclGroupParamsV2;
    hcclGroupParamsV2.pComm = hcclComm;
    std::map<std::string, HcclGroupParamsV2> hcclGroupMap = {{"hccl_world_group", hcclGroupParamsV2}};
    hcclGroupMap["hccl_world_group_1"] = hcclGroupParamsV2;
    CommManager::GetInstance(0).GetCommInfoV2().hcclGroupMap = hcclGroupMap;

    CommManager::GetInstance(0).GetCommInfoV2().commParams = commParams;
    CommManager::GetInstance(0).GetCommInfoV2().isUsed = true;
    CommManager::GetInstance(0).GetCommInfoV2().pComm = hcclComm;
    MOCKER(HrtGetDevice).stubs().with(mockcpp::any()).will(returnValue(0));

    void* sendBuf = FAKE_PTR;
    u64 sendCounts[FAKE_RANK_SIZE] = {1, 1};
    u64 sendDispls[FAKE_RANK_SIZE] = {1, 1};
    void* recvBuf = FAKE_PTR;
    u64 recvCount = 1;
    HcclDataType dataType = FAKE_DATA_TYPE;
    HcclReduceOp op = HCCL_REDUCE_PROD;
    int a = 1;
    aclrtStream stream = static_cast<aclrtStream>(&a);

    MOCKER_CPP(&HcclCommunicator::LoadOffloadCollOp)
        .stubs()
        .with(mockcpp::any(), mockcpp::any())
        .will(returnValue(HCCL_SUCCESS));
    HcclResult result
        = HcomReduceScatterVV2(tag, sendBuf, &sendCounts, &sendDispls, recvBuf, recvCount, dataType, op, group, stream);
    EXPECT_EQ(result, HCCL_E_NOT_SUPPORT);
}

TEST_F(HcomTest, St_HcomAllReduceV2_When_HcomCheckReductionOpV2_fail_Expect_HCCL_E_NOT_SUPPORT)
{
    // 前置条件
    rtStream_t stream;
    s8* sendbuf;
    s8* recvbuf;
    s32 count = HCCL_COM_DATA_SIZE;

    sendbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(sendbuf, count * sizeof(s8), 0, count * sizeof(s8));
    recvbuf = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(recvbuf, count * sizeof(s8), 0, count * sizeof(s8));
    MOCKER(HcomCheckReductionOpV2).stubs().with(mockcpp::any()).will(returnValue(HCCL_E_NOT_SUPPORT));

    // 执行条件
    auto res
        = HcomAllReduceV2("testallreduce", sendbuf, recvbuf, count, HCCL_DATA_TYPE_INT8, HCCL_REDUCE_SUM, NULL, stream);

    // 后置验证
    EXPECT_EQ(res, HCCL_E_NOT_SUPPORT);

    sal_free(sendbuf);
    sal_free(recvbuf);
}

TEST_F(HcomTest, st_HcomSetAivClearEnableV2_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    SetupCommonCommInfo();
    char group[256] = "hccl_world_group";
    bool aivClearEnable = true;
    HcclResult ret = HcomSetAivClearEnableV2(group, aivClearEnable);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = commInfoV2.pComm;
    EXPECT_EQ(hcclComm->pimpl->aivClearEnable, aivClearEnable);
}

TEST_F(HcomTest, St_HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2_Expect_Success)
{
    MOCKER_CPP(&CommunicatorImpl::CalcCollOffloadOpRes).stubs().will(returnValue(0));
    u64 menSize{};
    EXPECT_EQ(HcomGetAlltoAllvcStagedWorkSpaceMemSizeV2("hccl_world_group", menSize), HCCL_SUCCESS);
    EXPECT_EQ(menSize, 0);
}

TEST_F(HcomTest, St_HcomGetCommCCLBufferSizeV2_When_Call_Expect_Success)
{
    HcclResult ret = HcomGetCommCCLBufferSizeV2();
    EXPECT_EQ(ret, HCCL_SUCCESS);
}

TEST_F(HcomTest, st_HcomGraphSelectAlgV2_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = commInfoV2.pComm;
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = 2;
    hcclComm->pimpl->status = CommStatus::COMM_READY;

    s64 comm = reinterpret_cast<s64>(hcclComm.get());
    char group[256] = "hccl_world_group";
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int32_t aivCoreLimit = 0;
    bool ifAiv = false;
    std::string algName = "";

    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).defaults().with(mockcpp::any()).will(ignoreReturnValue());

    HcclResult ret = HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    ret = HcomSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_SUCCESS, ret);
}

TEST_F(HcomTest, st_HcomGraphSelectAlgV2_When_UnNormal_Expect_ReturnError)
{
    SetupCommonCommInfo();

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = commInfoV2.pComm;
    hcclComm->pimpl->status = CommStatus::COMM_READY;

    s64 comm = reinterpret_cast<s64>(hcclComm.get());
    char group[256] = "hccl_world_group";
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int32_t aivCoreLimit = 0;
    bool ifAiv = false;
    std::string algName = "";

    HcclCMDType opType = HcclCMDType::HCCL_CMD_INVALID;
    HcclResult ret = HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
    ret = HcomSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);

    opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    hcclComm->pimpl->status = CommStatus::COMM_ERROR;
    ret = HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_E_INTERNAL, ret);

    hcclComm->pimpl->status = CommStatus::COMM_READY;
    hcclComm->pimpl->isSuspended = true;
    ret = HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_E_SUSPENDING, ret);

    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    hcclComm->pimpl->isSuspended = false;
    opType = HcclCMDType::HCCL_CMD_REDUCE;
    ret = HcomGraphSelectAlgV2(comm, group, opType, count, dataType, op, aivCoreLimit, ifAiv, algName);
    EXPECT_EQ(HCCL_E_PARA, ret);
}

TEST_F(HcomTest, st_HcomCalcNumBlocksV2_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = commInfoV2.pComm;
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = 2;
    hcclComm->pimpl->status = CommStatus::COMM_READY;

    char group[256] = "hccl_world_group";
    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    u64 count = 1024;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    int32_t aivCoreLimit = 2;
    std::string algName = "";
    u32 numBlocks = 0;

    HcclResult ret = HcomCalcNumBlocksV2(group, opType, count, dataType, aivCoreLimit, algName, numBlocks);
    EXPECT_EQ(HCCL_SUCCESS, ret);
    EXPECT_EQ(numBlocks, aivCoreLimit);

    opType = HcclCMDType::HCCL_CMD_INVALID;
    ret = HcomCalcNumBlocksV2(group, opType, count, dataType, aivCoreLimit, algName, numBlocks);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);
}

TEST_F(HcomTest, st_HcclGetAlgExecParamV2_When_Normal_Expect_ReturnHCCL_SUCCESS)
{
    SetupCommonCommInfo();

    auto& commInfoV2 = CommManager::GetInstance(0).GetCommInfoV2();
    std::shared_ptr<Hccl::HcclCommunicator> hcclComm = commInfoV2.pComm;
    hcclComm->pimpl->myRank = 0;
    hcclComm->pimpl->rankSize = 2;
    hcclComm->pimpl->status = CommStatus::COMM_READY;

    std::string tag = "aivTag";
    char group[256] = "hccl_world_group";
    u64 count = 1024;

    s8* inputPtr = (s8*)sal_malloc(count * sizeof(s8));
    s8* outputPtr = (s8*)sal_malloc(count * sizeof(s8));
    sal_memset(inputPtr, count * sizeof(s8), 0, count * sizeof(s8));
    sal_memset(outputPtr, count * sizeof(s8), 0, count * sizeof(s8));

    HcclCMDType opType = HcclCMDType::HCCL_CMD_ALLREDUCE;
    bool clearEnable = true;
    HcclDataType dataType = HCCL_DATA_TYPE_INT8;
    HcclReduceOp op = HCCL_REDUCE_SUM;
    void* commContext = nullptr;
    u64 len = 0;
    int32_t aivCoreLimit = 2;

    MOCKER_CPP(&CommunicatorImpl::ExecAlgSelect)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));
    MOCKER_CPP(&CommunicatorImpl::ReportProfInfo).defaults().with(mockcpp::any()).will(ignoreReturnValue());
    MOCKER_CPP(&CommunicatorImpl::GetAlgExecParam)
        .defaults()
        .with(mockcpp::any())
        .will(returnValue(HcclResult::HCCL_SUCCESS));

    HcclResult ret = HcclGetAlgExecParamV2(
        tag, group, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, commContext, len, aivCoreLimit);
    EXPECT_EQ(HCCL_SUCCESS, ret);

    opType = HcclCMDType::HCCL_CMD_INVALID;
    ret = HcclGetAlgExecParamV2(
        tag, group, count, inputPtr, outputPtr, opType, clearEnable, dataType, op, commContext, len, aivCoreLimit);
    EXPECT_EQ(HCCL_E_NOT_SUPPORT, ret);

    sal_free(inputPtr);
    sal_free(outputPtr);
}

TEST_F(HcomTest, st_HcomGetDevIdV2_expect_ReturnHCCL_SUCCESS)
{
    SetupCommonCommInfo();
    auto hcclComm = CommManager::GetInstance(0).GetCommInfoV2().pComm;
    hcclComm->GetCommImpl()->devLogicId = 99;

    s32 devId{};
    EXPECT_EQ(HcomGetDevIdV2("hccl_world_group", &devId), HCCL_SUCCESS);
    EXPECT_EQ(devId, 99);
}
