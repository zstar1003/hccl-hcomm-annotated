/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CHECKER_BIG_GRAPH_DATA_LOADER_H
#define CHECKER_BIG_GRAPH_DATA_LOADER_H

#include <cstdint>
#include <map>
#include <vector>

#include "binary_data_type_pub.h"
#include "hccl_types.h"
#include "sim_loader.h"
#include "sim_op_db_types.h"
#include "task_graph_generator_v3/task_def_v3.h"

namespace HcclSim {
namespace BigGraphCheckV3 {

    struct OperatorRankData {
        uint32_t rankId{UINT32_MAX};
        sim::CompositeOpDetail op;
        std::vector<HcclTaskMetaData> taskMetas;
    };

    struct OpParam {
        TaskGraphGeneratorV3::OperatorId operatorId{TaskGraphGeneratorV3::INVALID_OPERATOR_ID};
        uint32_t syncIter{0};
        uint32_t opIter{0};
        std::vector<OperatorRankData> ranks;
    };

    struct BigGraphData {
        uint32_t syncIter{0};
        std::vector<sim::CcuChannelTab> channels;
        std::vector<sim::CcuInstrResTab> instrRes;
        std::vector<OpParam> operators;

        void Clear()
        {
            syncIter = 0;
            channels.clear();
            instrRes.clear();
            operators.clear();
        }
    };

    class BigGraphDataLoader {
    public:
        HcclResult Load(loader::Loader& loader, uint32_t syncIter, BigGraphData& data) const;

    private:
        static HcclResult DecodeTaskMeta(const sim::OpTaskTab& task, HcclTaskMetaData& taskMeta);
    };

} // namespace BigGraphCheckV3
} // namespace HcclSim

#endif // CHECKER_BIG_GRAPH_DATA_LOADER_H
