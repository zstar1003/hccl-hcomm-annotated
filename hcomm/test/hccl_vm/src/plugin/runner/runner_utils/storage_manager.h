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
#include <queue>
#include <vector>

#include "sim_binary_data_type_pub.h"
#include "ccu_resource_common.h"
#include "dtype_common.h"
#include "sim_common_defs.h"
#include "sim_loader.h"
#include "sim_op_db_types.h"
#include "store_sim_shm_memory_common.h"

namespace HcclSim {
struct BufferInfo {
    uint64_t virtualAddr{0};
    char* phyAddr{nullptr};
    uint64_t size{0};
};

struct StreamTag {
    uint32_t rankId{0};
    uint32_t streamId{0};
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

    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
};

// 一个rank的任务队列： {streamId, {tasklist}}
using SingleTaskQueue = std::vector<std::queue<HcclTaskMetaData>>;
using AllRankTaskQueues = std::vector<SingleTaskQueue>;

class StorageManager {
public:
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

    std::string FindRootPath();

    HcclVmResult InitCcuResource(std::vector<sim::CcuInstrResTab>& instrRes);
    HcclVmResult InitAivResourceFromCompositeOpDetail(const sim::CompositeOpDetail& opDetail);
    void ResetAivResource();
    void ReleasePhyMem();

    HcclVmResult LoadHcclVmSynthesisData(sim::OpDetailTab& detail, std::vector<sim::CcuChannelTab>& channels);
    HcclVmResult LoadHcclVmInstrData(std::vector<sim::CcuInstrResTab>& instrRes);
    HcclVmResult LoadHcclVmTaskMetaData(std::vector<sim::OpTaskTab>& tasks);
    HcclVmResult ConvertTaskQueue(const HcclVmTaskMetaData& taskMeataData);
    uint32_t GetRankSize() const;

    HcclVmInstrData GetHvmInstrData() const;
    HcclVmResult Trans2CheckerParam(sim::OpDetailTab& detailTab, ::OpDetails& detail);
    HcclVmTaskMetaData GetHvmTaskMetaData() const;
    AllRankTaskQueues& GetAllRankTaskQueues();
    void DumpAllRankInputOutput(std::vector<std::map<uint32_t, sim::CompositeOpDetail>>& compositeDataMap);
    HcclVmResult DumpHcclVmFlagData(uint16_t finishFlag);
    HcclVmResult GetHcclVmFlagData(HcclSim::HcclVmFlagData& flagData);
    void Reset()
    {
        m_allRankTaskQueues.clear();
        m_allRankChannelInfo.clear();
        m_allPhyMem.clear();
        m_synData = {};
        m_instrData = {};
        m_checker_param = {};
        devType_ = DevType::DEV_TYPE_COUNT;
    }
    std::vector<RankChannelInfo>& GetAllRankChannelInfo();

private:
    StorageManager() = default;
    HcclVmResult PrintAllRankInputBuffer();
    void FlexiblePrintData(std::ofstream& ofs, const BufferInfo& buffer);
    void PrintOpData1(std::ofstream& ofs, const std::vector<BufferInfo>& allRankInput);
    void PrintOpData2(
        std::ofstream& ofs, const std::vector<BufferInfo>& allRankInput, const std::vector<BufferInfo>& allRankOutput);

    bool IsDirExists(const std::string& path);

    std::string m_data_id;
    mutable std::mutex m_mutex;

    std::vector<RankChannelInfo> m_allRankChannelInfo;
    CheckerParam m_checker_param;

    HcclVmSynData m_synData;
    HcclVmInstrData m_instrData;
    AllRankTaskQueues m_allRankTaskQueues;
    std::vector<sim::PhyMemBlock> m_allPhyMem;
    DevType devType_{DevType::DEV_TYPE_COUNT}; // 初始化无效值
};
} // namespace HcclSim

#endif
