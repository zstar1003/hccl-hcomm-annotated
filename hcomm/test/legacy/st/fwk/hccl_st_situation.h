/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_ST_SITUATION_H
#define HCCLV2_ST_SITUATION_H

#include <map>
#include "types.h"
#include "op_type.h"
#include "data_type.h"
#include "reduce_op.h"
#include "dev_type.h"

using namespace Hccl;

MAKE_ENUM(FakeClusterType, CLUSTER_1_SERVER_8_DEV, CLUSTER_1_SERVER_2_DEV)

class Situation {
public:
    Situation();

    Situation& SetDataType(DataType dataType);

    Situation& SetReduceOp(ReduceOp reduceOp);

    Situation& SetOpType(OpType opType);

    Situation& SetCount(int count);

    Situation& SetEnv(const std::string& name, const std::string& value);

    Situation& SetClusterType(DevType type, FakeClusterType cluster);

    FakeClusterType GetClusterType() const { return clusterType; }

    inline int GetServerNum() { return serverNum; }

    inline OpType GetOpType() { return opType; }

    inline int GetDeviceNum() { return deviceNum; }

    inline DataType GetDataType() { return dataType; }

    inline ReduceOp GetReduceOp() { return reduceOp; }

    inline int GetDstRank() { return dstRank; }

    inline int GetCount() { return count; }

    inline int GetRoot() { return root; }

    inline bool GetStaticAddr() { return staticAddr; }

    inline bool GetStaticShape() { return staticShape; }

    inline DevType GetDevType() { return devType; }

    inline int GetRankSize() { return serverNum * deviceNum; }

    inline const std::map<std::string, std::string>& GetEnv() { return envConfigs; }

private:
    OpType opType;
    DataType dataType;
    ReduceOp reduceOp;
    DevType devType{};
    int count;
    int dstRank{};
    int root{};
    int serverNum{};
    int deviceNum{};
    bool staticAddr{};
    bool staticShape{};

    std::map<std::string, std::string> envConfigs;

    FakeClusterType clusterType{};
};

#endif
