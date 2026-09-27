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

#define private public
#define protected public
#include "hcom_op_utils.h"
#include "hccl_impl.h"
#include "hccl_communicator.h"
#undef private
#undef protected

#include "hccl/base.h"
#include <hccl/hccl_types.h>

#include "stream_pub.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "hccl_comm_pub.h"
#include "sal.h"
#include "llt_hccl_stub_pub.h"
#include "externalinput.h"
#include "config.h"
#include "topoinfo_ranktableParser_pub.h"
#include "ranktable/v80_rank_table.h"
#include "hcom_op_utils.h"
#include "hcom_ops_kernel_info_store.h"
#include "external/ge/ge_api_types.h"  // ge对内options
#include "framework/common/ge_types.h" // ge对外options
#include "graph/ge_local_context.h"
#include "hcom_pub.h"
#include "hcom.h"

#include "graph/debug/ge_attr_define.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/attr_utils.h"
#include "graph/node.h"
#include "graph/op_desc.h"
#include "graph/utils/tensor_utils.h"
#include "graph/ge_tensor.h"
#include "register/op_tiling_info.h"
#include "kernel_tiling/kernel_tiling.h"

#include "evaluator.h"
#include "model.h"
#include "cluster.h"

#include <iostream>
#include <fstream>
#include "graph/ge_context.h"
#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "hcom_graph_mc2.h"
#include "adapter_rts.h"

using namespace std;
using namespace hccl;

class HcomGraphMc2Test : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "HcomGraphMc2Test SetUP" << std::endl; }
    static void TearDownTestCase() { std::cout << "HcomGraphMc2Test TearDown" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp()
    {
        s32 portNum = 7;
        MOCKER(hrtGetHccsPortNum).stubs().with(mockcpp::any(), outBound(portNum)).will(returnValue(HCCL_SUCCESS));

        std::cout << "A Test SetUP" << std::endl;
    }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(HcomGraphMc2Test, ut_mc2_creatComResourceErr)
{
    std::vector<void*> commContext{};
    ge::graphStatus ge_ret;
    HcclRootInfo id;
    HcclResult ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(newcomm);

    const ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>();
    opDescPtr->SetType("MatmulAllReduce");
    ge::AttrUtils::SetStr(opDescPtr, "group", hcclComm->GetIdentifier());

    ge_ret = hccl::HcomCreateComResource(opDescPtr, commContext);
    EXPECT_EQ(ge_ret, ge::GRAPH_FAILED);

    shared_ptr<std::vector<void*>> rt_resource_list = std::make_shared<std::vector<void*>>();
    opDescPtr->SetExtAttr("_rt_resource_list", rt_resource_list);
    ge_ret = hccl::HcomCreateComResource(opDescPtr, commContext);
    EXPECT_EQ(ge_ret, ge::GRAPH_FAILED);
}

TEST_F(HcomGraphMc2Test, ut_mc2_creatComResource)
{
    DevType deviceType = DevType::DEV_TYPE_910B;
    MOCKER(hrtGetDeviceType).stubs().with(outBound(deviceType)).will(returnValue(HCCL_SUCCESS));
    MOCKER_CPP(&HcclCommunicator::HcclGetCmdTimeout).stubs().will(returnValue(50));

    std::vector<void*> commContext{};
    ge::graphStatus ge_ret;
    rtStream_t stream;
    rtNotify_t notify;
    HcclRootInfo id;
    s32 devicePhyId = 0;
    rtError_t rt_ret = RT_ERROR_NONE;
    HcclResult ret = hrtGetDevice(&devicePhyId);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    ret = HcclGetRootInfo(&id);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    HcclComm newcomm;
    ret = HcclCommInitRootInfo(1, &id, 0, &newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    hccl::hcclComm* hcclComm = static_cast<hccl::hcclComm*>(newcomm);

    ge::OpDescPtr opDescPtr = std::make_shared<ge::OpDesc>();
    opDescPtr->SetType("MatmulAllReduce");
    ge::AttrUtils::SetStr(opDescPtr, "group", hcclComm->GetIdentifier());

    shared_ptr<std::vector<void*>> rt_resource_list = std::make_shared<std::vector<void*>>();

    rt_ret = aclrtCreateStream(&stream);
    EXPECT_EQ(rt_ret, RT_ERROR_NONE);
    ret = hrtNotifyCreate(0, &notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    rt_resource_list->push_back((void*)stream);
    opDescPtr->SetExtAttr("_rt_resource_list", rt_resource_list);

    ge_ret = hccl::HcomCreateComResource(opDescPtr, commContext);
    EXPECT_EQ(ge_ret, ge::GRAPH_SUCCESS);

    rt_ret = aclrtDestroyStream(stream);
    EXPECT_EQ(ret, RT_ERROR_NONE);
    ret = hrtNotifyDestroy(notify);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = HcclCommDestroy(newcomm);
    EXPECT_EQ(ret, HCCL_SUCCESS);
    GlobalMockObject::verify();
}
