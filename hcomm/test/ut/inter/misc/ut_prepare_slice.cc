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

#define private public
#define protected public
#include "hccl_communicator.h"
#include "hccl_impl.h"
#include "all_reduce_local_reduce_pub.h"
#include "alg_template_base_pub.h"
#undef protected
#undef private

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "gradient_segment.h"
#include "sal.h"

#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "rank_consistentcy_checker.h"
#include <iostream>
#include <fstream>
#include "ranktable/v80_rank_table.h"
#include "dlra_function.h"
#include <fcntl.h>
#include <unistd.h>
#include "llt_hccl_stub_profiling_plugin.h"
#include "task_profiling_pub.h"
#include "workflow_pub.h"
#include "dltdt_function.h"
#include "heartbeat.h"
#include "opexecounter_pub.h"
#include "param_check_pub.h"
#include "callback_thread_manager.h"
#include "dispatcher_pub.h"
#include "dispatcher_pub.h"
using namespace std;
using namespace hccl;

class PrepareSliceTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "PrepareSliceTest SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "PrepareSliceTest TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown()
    {
        GlobalMockObject::verify();
        std::cout << "A Test TearDown" << std::endl;
    }
};

TEST_F(PrepareSliceTest, ut_slice)
{
    Stream mainstream;
    void* dispatcherPtr = nullptr;

    HcclResult ret = HcclDispatcherInit(DispatcherType::DISPATCHER_NORMAL, 0, &dispatcherPtr);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    EXPECT_NE(dispatcherPtr, nullptr);
    DispatcherPub* dispatcher = reinterpret_cast<DispatcherPub*>(dispatcherPtr);
    std::vector<Stream> meshStreams;
    std::vector<std::shared_ptr<LocalNotify>> meshSignal;
    std::vector<std::shared_ptr<LocalNotify>> meshSignalAux;
    HcomCollOpInfo opInfo;

    DeviceMem inputMem = DeviceMem::create(0, 1024);
    DeviceMem outputMem = DeviceMem::create(0, 1024);

    opInfo.outputAddr = inputMem.ptr();

    std::vector<Slice> dataSlice;
    std::vector<Slice> startOffset;

    std::unique_ptr<AllReduceLocalReduce> tempAlg;
    tempAlg.reset(new (std::nothrow) AllReduceLocalReduce(dispatcher));
    tempAlg->Prepare(0, meshStreams, meshSignal, meshSignalAux, 0, 8, 0, &opInfo);
    tempAlg->Prepare(
        outputMem, outputMem, outputMem, 17, HcclDataType::HCCL_DATA_TYPE_FP16, mainstream,
        HcclReduceOp::HCCL_REDUCE_SUM, 0, dataSlice, 0);

    tempAlg->PrepareSlice(17, 2, 8, dataSlice, startOffset);
    for (const auto& slice : dataSlice) {
        HCCL_ERROR("offset: %llu, size: %llu", slice.offset, slice.size);
    }

    for (const auto& slice : startOffset) {
        HCCL_ERROR("offset: %llu, size: %llu", slice.offset, slice.size);
    }
    if (dispatcherPtr != nullptr) {
        ret = HcclDispatcherDestroy(dispatcherPtr);
        EXPECT_EQ(ret, HCCL_SUCCESS);
        dispatcherPtr = nullptr;
    }
}
