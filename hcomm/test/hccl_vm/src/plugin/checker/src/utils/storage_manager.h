/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef STORAGE_MANAGER_H
#define STORAGE_MANAGER_H

#include <cstdint>
#include <map>
#include <mutex>
#include <vector>

#include "binary_data_type_pub.h"
#include "data_slice.h"
#include "dtype_common.h"
#include "sim_common.h"
#include "sim_op_db_types.h"
#include "task_meta_defs.h"
#include "framework/task_graph_generator_v3/task_def_v3.h"

namespace HcclSim {
struct MemBlock {
    uint64_t startAddr;    // 物理起始地址
    uint64_t size;         // 块大小
    BufferType bufferType; // 类型
    uint64_t globalOffset; // 该类型总偏移
};

// Raw V-operator parameters reported by one rank in the current op group.
struct VRankParam {
    uint64_t localCount = 0;
    std::vector<uint64_t> counts;
    std::vector<uint64_t> displs;
};

struct SendRecvPairParam {
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    bool sendSeen = false;
    bool recvSeen = false;
};

struct BatchSendRecvRankParam {
    uint32_t itemNum = 0;
    uint64_t peerCount = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint32_t sendPeer = 0;
    uint32_t recvPeer = 0;
};

struct CheckerParam {
    HcclCMDType cmdType = static_cast<HcclCMDType>(0);
    uint32_t rankSize = 0;
    HcclDataType dataType = static_cast<HcclDataType>(0);
    uint64_t dataCount = 0;

    // 以下是可能缺失的字段，初始化为安全值
    HcclReduceOp reduceType = static_cast<HcclReduceOp>(0);
    uint32_t srcRank = 0;
    uint32_t dstRank = 0;
    uint32_t root = 0;
    std::vector<SendRecvPairParam> sendRecvPairs;
    std::vector<BatchSendRecvRankParam> batchSendRecvRankParams;

    VDataDesTagInner vDataDes;
    std::vector<VRankParam> vRankParams;
    All2AllDataDesTagInner all2AllDataDes;
};

struct RemoteDieInfo {
    uint32_t dstRank;
    uint32_t remoteDieId;
};

using ChannelsPerDie = std::map<uint32_t, RemoteDieInfo>;

class StorageManager {
public:
    StorageManager() = default;
    static StorageManager& GetInstance()
    {
        static StorageManager instance;
        return instance;
    }
    // 禁用拷贝构造和赋值
    StorageManager(const StorageManager&) = delete;
    StorageManager& operator=(const StorageManager&) = delete;

    void SetDataId(const std::string& data_id)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_data_id = data_id;
    }

    std::string GetDataId() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_data_id;
    }

    CheckerParam GetCheckerParam() const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        return m_checker_param;
    }

    CheckerParam GetCheckerParam(TaskGraphGeneratorV3::OperatorId operatorId) const
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        const auto it = m_checker_params.find(operatorId);
        return it == m_checker_params.end() ? m_checker_param : it->second;
    }

    void SaveCheckerParam(TaskGraphGeneratorV3::OperatorId operatorId)
    {
        std::lock_guard<std::mutex> lock(m_mutex);
        m_checker_params[operatorId] = m_checker_param;
    }

    HcclResult
    LoadHcclVmSynthesisData(uint32_t rankId, sim::OpMemInfoTab memInfo, std::vector<sim::CcuChannelTab>& channels);
    HcclResult LoadHcclVmInstrData(std::vector<sim::CcuInstrResTab>& instrRes);
    HcclResult LoadHcclVmTaskMetaData(std::vector<std::vector<sim::OpTaskTab>>& allTasks);
    HcclResult LoadDecodedHcclVmTaskMetaData(const std::vector<std::vector<HcclTaskMetaData>>& allTaskMetas);
    void Reset(bool clearMemLayout = true);
    uint64_t GetBlockSize(uint32_t rankId, BufferType bufferType);
    HcclResult GetSlice(uint64_t addr, uint64_t len, DataSlice& dataSlice, uint32_t* rank = nullptr);
    uint32_t GetRankSize() const;

    HcclResult ReadHeader(FILE* fp, FileHeader& header);
    HcclResult ChannelWrite(FILE* fp, const ChannelInfo& chInfo);
    HcclResult ChannelRead(FILE* fp, ChannelInfo& chInfo);

    HcclVmInstrData GetHvmInstrData() const;
    HcclResult Trans2CheckerParam(sim::OpDetailTab& detailTab, ::OpDetails& detail);
    HcclVmTaskMetaData GetHvmTaskMetaData() const;
    void InitCcuInfo(DevType& devType, std::vector<uint64_t>& resourceBaseAddr);
    void BeginOpGroup();
    HcclResult FinalizeOpGroup();
    void MergeAll2AllVSendCountMatrix();

private:
    std::string FindRootPath();
    bool IsDirExists(const std::string& path);

    std::string m_data_id;
    mutable std::mutex m_mutex;

    // RankID -> Type -> StartAddr -> BlockInfo
    std::map<uint32_t, std::map<BufferType, std::map<uint64_t, MemBlock>>> m_mem_layout;
    std::map<RankId, std::map<uint32_t, ChannelsPerDie>> m_allRankChannelInfo;
    CheckerParam m_checker_param;
    std::map<TaskGraphGeneratorV3::OperatorId, CheckerParam> m_checker_params;
    // 用于收集每一轮算子中所有rank的发送矩阵数据
    std::map<uint32_t, std::vector<uint64_t>> m_all2AllvSendMatrices;

    HcclVmSynData m_synData;
    HcclVmInstrData m_instrData;
    HcclVmTaskMetaData m_taskMeataData;
    DevType devType_{DevType::DEV_TYPE_COUNT}; // 初始化无效值
};
} // namespace HcclSim

#endif
