/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_st_situation.h"
#include "fake_pub_stub.h"
#include "ranktable/stub_rank_table.h"
#include "iostream"

using namespace Hccl;
std::string fakeSocVersion;

const char* fakePubStubGetSocVersion() { return fakeSocVersion.c_str(); }

void SetFakeSocVersion(DevType type)
{
    if (type == DevType::DEV_TYPE_910A) {
        fakeSocVersion = "Ascend910";
    } else if (type == DevType::DEV_TYPE_910A2) {
        fakeSocVersion = "Ascend910B1";
    } else if (type == DevType::DEV_TYPE_910A3) {
        fakeSocVersion = "Ascend910C1";
    } else {
        std::cout << "ST only support 910A/910A2/910A3 now" << std::endl;
        throw std::exception();
    }
}

Situation::Situation() : opType(OpType::ALLREDUCE), dataType(DataType::FP32), reduceOp(ReduceOp::SUM), count(1)
{
    SetClusterType(DevType::DEV_TYPE_910A, FakeClusterType::CLUSTER_1_SERVER_8_DEV);
}

Situation& Situation::SetDataType(DataType dataType1)
{
    this->dataType = dataType1;
    return *this;
}

Situation& Situation::SetReduceOp(ReduceOp reduceOp1)
{
    this->reduceOp = reduceOp1;
    return *this;
}

Situation& Situation::SetOpType(OpType opType1)
{
    this->opType = opType1;
    return *this;
}

Situation& Situation::SetCount(int dataCount)
{
    this->count = dataCount;
    return *this;
}

Situation& Situation::SetEnv(const std::string& name, const std::string& value)
{
    this->envConfigs[name] = value;
    return *this;
}

Situation& Situation::SetClusterType(DevType type, FakeClusterType cluster)
{
    clusterType = cluster;
    devType = type;

    SetFakeSocVersion(devType);
    if (cluster == FakeClusterType::CLUSTER_1_SERVER_8_DEV) { // 单机8卡
        serverNum = 1;
        deviceNum = 8;
    } else if (cluster == FakeClusterType::CLUSTER_1_SERVER_2_DEV) {
        serverNum = 1;
        deviceNum = 2;
    } else {
        std::cout << "ST only support 1 Server now" << std::endl;
        throw std::exception();
    }

    return *this;
}
