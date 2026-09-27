/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "big_graph_data_loader.h"

#include <algorithm>
#include <cstring>

#include "sim_log.h"

namespace HcclSim {
namespace BigGraphCheckV3 {

    HcclResult BigGraphDataLoader::DecodeTaskMeta(const sim::OpTaskTab& task, HcclTaskMetaData& taskMeta)
    {
        if (task.optaskMeta.size() < sizeof(HcclTaskMetaData)) {
            HCCL_VM_ERROR(
                "Cannot load operator task metadata because the payload is too small, taskSeq={}, "
                "actualSize={}, expectedSize={}",
                task.taskSeq, task.optaskMeta.size(), sizeof(HcclTaskMetaData));
            return HCCL_E_PARA;
        }
        std::memcpy(&taskMeta, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
        return HCCL_SUCCESS;
    }

    HcclResult BigGraphDataLoader::Load(loader::Loader& loader, uint32_t syncIter, BigGraphData& data) const
    {
        data.Clear();
        data.syncIter = syncIter;

        HcclResult ret = loader.GetCcuChannelInfo(data.channels);
        if (ret != HCCL_SUCCESS) {
            data.Clear();
            return ret;
        }
        ret = loader.GetInstrResInfo(data.instrRes);
        if (ret != HCCL_SUCCESS) {
            data.Clear();
            return ret;
        }

        std::map<uint32_t, std::vector<sim::CompositeOpDetail>> compositeData;
        ret = loader.LoadCompositeOpDetailBySyncIter(syncIter, compositeData);
        if (ret != HCCL_SUCCESS) {
            HCCL_VM_ERROR(
                "Failed to load operator data for a sync window, syncIter={}, ret={}", syncIter,
                static_cast<uint32_t>(ret));
            data.Clear();
            return ret;
        }

        size_t operatorCount = 0;
        for (const auto& rankEntry : compositeData) {
            operatorCount = std::max(operatorCount, rankEntry.second.size());
        }
        if (operatorCount > static_cast<size_t>(TaskGraphGeneratorV3::INVALID_OPERATOR_ID)) {
            HCCL_VM_ERROR(
                "Too many operators in one sync window, syncIter={}, operatorCount={}", syncIter, operatorCount);
            data.Clear();
            return HCCL_E_PARA;
        }

        data.operators.resize(operatorCount);
        for (size_t operatorIndex = 0; operatorIndex < operatorCount; ++operatorIndex) {
            OpParam& opParam = data.operators[operatorIndex];
            opParam.operatorId = static_cast<TaskGraphGeneratorV3::OperatorId>(operatorIndex);
            opParam.syncIter = syncIter;

            bool hasOp = false;
            for (const auto& rankEntry : compositeData) {
                if (operatorIndex >= rankEntry.second.size()) {
                    continue;
                }

                const sim::CompositeOpDetail& compositeOp = rankEntry.second[operatorIndex];
                if (!hasOp) {
                    opParam.opIter = compositeOp.detail.opIter;
                    hasOp = true;
                }

                OperatorRankData rankData;
                rankData.rankId = rankEntry.first;
                rankData.op = compositeOp;
                rankData.taskMetas.reserve(compositeOp.tasks.size());
                for (const sim::OpTaskTab& task : compositeOp.tasks) {
                    HcclTaskMetaData taskMeta;
                    const HcclResult decodeRet = DecodeTaskMeta(task, taskMeta);
                    if (decodeRet != HCCL_SUCCESS) {
                        data.Clear();
                        return decodeRet;
                    }
                    rankData.taskMetas.push_back(taskMeta);
                }
                opParam.ranks.push_back(std::move(rankData));
            }

            if (!hasOp) {
                data.operators.resize(operatorIndex);
                break;
            }
        }

        HCCL_VM_INFO(
            "Loaded multi-operator data, syncIter={}, operatorCount={}, rankCount={}", syncIter, data.operators.size(),
            compositeData.size());
        return HCCL_SUCCESS;
    }

} // namespace BigGraphCheckV3
} // namespace HcclSim
