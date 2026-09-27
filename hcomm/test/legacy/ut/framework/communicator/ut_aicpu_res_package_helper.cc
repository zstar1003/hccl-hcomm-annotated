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
#include "aicpu_res_package_helper.h"
#include <vector>

using namespace Hccl;

class AicpuResPackageHelperTest : public testing::Test {
protected:
    static void SetUpTestCase() { std::cout << "AicpuResPackageHelperTest SetUP" << std::endl; }

    static void TearDownTestCase() { std::cout << "AicpuResPackageHelperTest TearDown" << std::endl; }

    virtual void SetUp() { std::cout << "A Test case in AicpuResPackageHelperTest SetUp" << std::endl; }

    virtual void TearDown()
    {
        std::cout << "A Test case in AicpuResPackageHelperTest TearDown" << std::endl;
        GlobalMockObject::verify();
    }

    inline void Fill(char* data, const std::string& str, u32 maxSize) { strncpy(data, str.c_str(), maxSize); }

    inline void Fill(std::vector<char>& data, const std::string& str) { data.assign(str.begin(), str.end()); }
};

TEST_F(AicpuResPackageHelperTest, test_pack_and_parse)
{
    AicpuResPackageHelper tool;
    std::vector<ModuleData> dataVec;
    ModuleData module0;
    Fill(module0.name, "module0", sizeof(module0.name));
    Fill(module0.data, "data0");

    ModuleData module1;
    Fill(module1.name, "module1", sizeof(module1.name));
    Fill(module1.data, "data1");

    ModuleData module2;
    Fill(module2.name, "module2", sizeof(module2.name));
    Fill(module2.data, "data2");

    dataVec.push_back(module0);
    dataVec.push_back(module1);
    dataVec.push_back(module2);

    auto packedData = tool.GetPackedData(dataVec);
    auto parsedDataVec = tool.ParsePackedData(packedData);

    ASSERT_EQ(dataVec.size(), parsedDataVec.size());
    for (u32 i = 0; i < dataVec.size(); i++) {
        EXPECT_STREQ(dataVec[i].name, parsedDataVec[i].name);
        EXPECT_EQ(dataVec[i].data, parsedDataVec[i].data);
    }
}
