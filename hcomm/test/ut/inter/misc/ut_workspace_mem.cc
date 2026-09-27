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

#include "workspace_mem.h"
#include "mem_host_pub.h"
#include "mem_device_pub.h"
#include "sal.h"

using namespace std;
using namespace hccl;

class WorkSpaceMemTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "\033[36m--WorkSpaceMemTest SetUP--\033[0m" << std::endl; }
    static void TearDownTestCase() { std::cout << "\033[36m--WorkSpaceMemTest TearDown--\033[0m" << std::endl; }
    // Some expensive resource shared by all tests.
    virtual void SetUp() { std::cout << "A Test SetUP" << std::endl; }
    virtual void TearDown() { std::cout << "A Test TearDown" << std::endl; }
};

TEST_F(WorkSpaceMemTest, ut_set_one_mem_resource)
{
    s32 ret = HCCL_SUCCESS;
    const u64 size = 1024;
    const u64 allocSize = 100;
    DeviceMem deviceMem = DeviceMem::alloc(size);

    WorkSpaceMem WorkSpaceMem;
    /*  设置资源 */
    ret = WorkSpaceMem.SetMemResource("tag", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr1 = WorkSpaceMem.AllocMem("tag", allocSize);
    (ptr1 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr2 = WorkSpaceMem.AllocMem("tag", allocSize);
    (ptr2 == ((char*)deviceMem.ptr() + allocSize)) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /*  释放资源 */
    ret = WorkSpaceMem.DestroyMemResource("tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁公共资源 */
    WorkSpaceMem.DestroyMemResource();
}

TEST_F(WorkSpaceMemTest, ut_set_muti_mem_resource)
{
    s32 ret = HCCL_SUCCESS;
    const u64 size = 1024;
    const u64 allocSize = 100;
    DeviceMem deviceMem = DeviceMem::alloc(size);

    WorkSpaceMem WorkSpaceMem;
    /*  设置资源 */
    ret = WorkSpaceMem.SetMemResource("tag", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = WorkSpaceMem.SetMemResource("tag1", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = WorkSpaceMem.SetMemResource("tag2", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr0 = WorkSpaceMem.AllocMem("tag", allocSize);
    (ptr0 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr1 = WorkSpaceMem.AllocMem("tag1", allocSize);
    (ptr1 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr2 = WorkSpaceMem.AllocMem("tag2", allocSize);
    (ptr2 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁公共资源 */
    WorkSpaceMem.DestroyMemResource();
}

TEST_F(WorkSpaceMemTest, ut_workspace_mem_resource_fail)
{
    s32 ret = HCCL_SUCCESS;
    const u64 size = 1024;
    const u64 allocSize = 100;
    DeviceMem deviceMem = DeviceMem::alloc(size);

    WorkSpaceMem WorkSpaceMem;
    /*  设置资源 */
    ret = WorkSpaceMem.SetMemResource("tag", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 设置空指针错误 */
    ret = WorkSpaceMem.SetMemResource("tag", NULL, size);
    EXPECT_EQ(ret, HCCL_E_PTR);

    /* 分配错误tag  异常*/
    void* ptr0 = WorkSpaceMem.AllocMem("tag1", allocSize);
    (ptr0 == NULL) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 分配正确的指针*/
    void* ptr1 = WorkSpaceMem.AllocMem("tag", allocSize);
    (ptr1 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 分配过大的内存*/
    void* ptr2 = WorkSpaceMem.AllocMem("tag", size);
    (ptr2 == NULL) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁错误的tag资源 */
    ret = WorkSpaceMem.DestroyMemResource("tag1");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁正确的tag资源*/
    ret = WorkSpaceMem.DestroyMemResource("tag");
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /*销毁全部管理资源 */
    WorkSpaceMem.DestroyMemResource();
}

TEST_F(WorkSpaceMemTest, ut_work_mem_resource_fun)
{
    s32 ret = HCCL_SUCCESS;
    const u64 size = 1024;
    const u64 allocSize = 100;
    DeviceMem deviceMem = DeviceMem::alloc(size);

    WorkSpaceMem WorkSpaceMem;
    /*  设置资源 */
    ret = WorkSpaceMem.SetMemResource("tag", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = WorkSpaceMem.SetMemResource("tag1", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    ret = WorkSpaceMem.SetMemResource("tag2", deviceMem.ptr(), size);
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr0 = WorkSpaceMem.AllocMem("tag", allocSize);
    (ptr0 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr1 = WorkSpaceMem.AllocMem("tag1", allocSize);
    (ptr1 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    void* ptr2 = WorkSpaceMem.AllocMem("tag2", allocSize);
    (ptr2 == deviceMem.ptr()) ? ret = HCCL_SUCCESS : ret = HCCL_E_MEMORY;
    EXPECT_EQ(ret, HCCL_SUCCESS);

    /* 销毁公共资源 */
    WorkSpaceMem.DestroyMemResource();
}
