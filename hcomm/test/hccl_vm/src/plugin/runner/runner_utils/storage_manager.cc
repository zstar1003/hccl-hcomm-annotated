/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "storage_manager.h"

#include <cstdint>
#include <cstdlib> // strtoull
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <nlohmann_json/json.hpp>
#include <string>
#include <sys/stat.h>
#include <unistd.h>
#include <vector>

#include "aiv_task_snapshot_loader.h"
#include "aiv_graph_executor_mgr.h"
#include "aiv_resource_manager.h"
#include "store_binary_data_operator.h"
#include "ccu_resource_manager.h"
#include "sim_log.h"

namespace HcclSim {
static const std::string PLUGIN_PATH = "/plugin";

static const std::string HCCLVM_FLAG_DATA_FILE = "/hcclvm_flag_data.bin";
static const std::string HCCLVM_TASK_DATA_FILE = "/%s_hcclvm_task_data.bin";
static const std::string HCCLVM_SYN_DATA_FILE = "/%s_hcclvm_syn_data.bin";
static const std::string HCCLVM_INSTR_DATA_FILE = "/%s_hcclvm_instr_data.bin";
static const std::string HCCLVM_ALL_RANK_INPUT_OUTPUT_FILE = "/all_rank_input_output.txt";

static const uint32_t PRINT_DATA_PREFIX = 512;          // 打印数据前缀
static const uint32_t PRINT_DATA_SUFFIX = 512;          // 打印数据后缀
static const uint32_t PRINT_DATA_SIZE_THRESHOLD = 1024; // 打印数据最大大小

std::string EidToHexString(uint8_t eid[])
{
    std::ostringstream oss;
    oss << std::hex << std::setfill('0');
    for (uint32_t i = 0; i < URMA_EID_LEN; i++) {
        oss << std::setw(2) << static_cast<uint32_t>(eid[i]);
    }
    return oss.str();
}

void StorageManager::ReleasePhyMem()
{
    for (const auto& phyMem : m_allPhyMem) {
        sim::ReleaseInNoHostProcess(phyMem);
    }
    m_allPhyMem.clear();
}

HcclVmResult StorageManager::PrintAllRankInputBuffer()
{
    std::vector<sim::PhyMemBlock> allPhyMem;
    for (uint32_t i = 0; i < m_synData.memory_info.count; i++) {
        if (m_synData.memory_info.data[i].buffer_type == 0) {
            auto startAddr = m_synData.memory_info.data[i].start_addr;

            sim::PhyMemBlock srcPhyMem{};
            auto srcAddr = sim::AcquireDevPtrInNoHostProcess((void*)startAddr, srcPhyMem);
            if (srcAddr == nullptr) {
                HCCL_VM_ERROR(
                    "无法获取startAddr的设备地址(addr= {:x})！", reinterpret_cast<uintptr_t>((void*)startAddr));
                return HcclVmResult::HCCL_SIM_E_INTERNAL;
            }
            m_allPhyMem.push_back(srcPhyMem);

            auto size = m_synData.memory_info.data[i].size;
            HCCL_VM_DEBUG(
                "rankId={}, input buffer= {:x}, size={}", m_synData.memory_info.data[i].rank_id, startAddr, size);
            for (uint32_t idx = 0; idx < size; idx++) {
                HCCL_VM_TRACE("idx={}, value={:x}", idx, *((char*)srcAddr + idx));
            }
        }
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::Trans2CheckerParam(sim::OpDetailTab& detailTab, ::OpDetails& detail)
{
    devType_ = static_cast<DevType>(detailTab.devType);
    m_checker_param.cmdType = static_cast<HcclCMDType>(detail.opType);
    m_checker_param.rankSize = detailTab.rankSize;
    m_checker_param.dataType = static_cast<HcclDataType>(detail.dataType);
    m_checker_param.dataCount = detail.opV1.count;
    m_checker_param.reduceType = static_cast<HcclReduceOp>(detail.reduceType);
    m_checker_param.srcRank = detailTab.srcRank;
    m_checker_param.dstRank = detailTab.dstRank;
    m_checker_param.root = detailTab.root;
    m_checker_param.all2AllDataDes.sendType = detail.opV2.sendDataType;
    m_checker_param.all2AllDataDes.recvType = detail.opV2.recvDataType;
    m_checker_param.all2AllDataDes.sendCount = detail.opV2.sendCount;
    m_checker_param.all2AllDataDes.recvCount = detail.opV2.recvCount;
    m_checker_param.all2AllDataDes.count = 0;
    const HcclCMDType cmdType = static_cast<HcclCMDType>(detail.opType);
    const bool isAll2AllOp = cmdType == HcclCMDType::HCCL_CMD_ALLTOALL || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV
                             || cmdType == HcclCMDType::HCCL_CMD_ALLTOALLVC;
    if (isAll2AllOp && detailTab.opExtInfo.size() >= sizeof(uint32_t)) {
        uint32_t count = 0;
        std::memcpy(&count, detailTab.opExtInfo.data(), sizeof(uint32_t));
        m_checker_param.all2AllDataDes.count = count;
        for (uint32_t i = 0; i < count; i++) {
            uint64_t val = 0;
            size_t offset = sizeof(uint32_t) + i * sizeof(uint64_t);
            if (offset + sizeof(uint64_t) <= detailTab.opExtInfo.size()) {
                std::memcpy(&val, detailTab.opExtInfo.data() + offset, sizeof(uint64_t));
            }
            m_checker_param.all2AllDataDes.sendCountMatrix.push_back(val);
        }
    }

    HCCL_VM_DEBUG("Success");

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult
StorageManager::LoadHcclVmSynthesisData(sim::OpDetailTab& detailTab, std::vector<sim::CcuChannelTab>& channels)
{
    if (detailTab.opDetail.size() < sizeof(::OpDetails)) {
        HCCL_VM_ERROR("opDetail BLOB too small");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    ::OpDetails opDetails{};
    std::memcpy(&opDetails, detailTab.opDetail.data(), sizeof(::OpDetails));

    Trans2CheckerParam(detailTab, opDetails);
    auto rankSize = GetRankSize();
    m_allRankChannelInfo.resize(rankSize);

    // 转换channel映射表
    for (auto& channel : channels) {
        CcuInfo rmtDieInfo1;
        rmtDieInfo1.rankId = channel.dstRankId;
        rmtDieInfo1.dieId = channel.dstDieId;
        HCCL_VM_INFO(
            "[Channel info] channelId={}, srcRank={}, srcDie={}, dstRank={}, dstDie={} srcEid={}, dstEid={}",
            channel.channelId, channel.srcRankId, static_cast<uint32_t>(channel.srcDieId), channel.dstRankId,
            channel.dstDieId, EidToHexString(channel.leid), EidToHexString(channel.reid));
        m_allRankChannelInfo[channel.srcRankId][channel.srcDieId][channel.channelId] = rmtDieInfo1;
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::LoadHcclVmInstrData(std::vector<sim::CcuInstrResTab>& instrRes)
{
    for (auto& ccuInstr : instrRes) {
        HCCL_VM_DEBUG("rankId={}, dieId={}, count={}", ccuInstr.rankId, ccuInstr.dieId, ccuInstr.instrCount);
    }

    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::LoadHcclVmTaskMetaData(std::vector<sim::OpTaskTab>& tasks)
{
    HcclVmTaskMetaData taskMeataData;
    for (const auto& task : tasks) {
        if (task.optaskMeta.size() >= sizeof(HcclTaskMetaData)) {
            HcclTaskMetaData metaData;
            std::memcpy(&metaData, task.optaskMeta.data(), sizeof(HcclTaskMetaData));
            taskMeataData.task_meta.push_back(metaData);
        } else {
            HCCL_VM_WARN(
                "optaskMeta too small, taskSeq={} src:{:d}, dst:{:d}", task.taskSeq, task.optaskMeta.size(),
                sizeof(HcclTaskMetaData));
        }
    }
    for (auto& taskMeta : taskMeataData.task_meta) {
        HCCL_VM_DEBUG(
            "rankId={}, dieId={}, instrCnt={}, argSize={}, streamId={}", taskMeta.rankId,
            static_cast<uint32_t>(taskMeta.taskData.ccu.dieId), taskMeta.taskData.ccu.instCnt,
            taskMeta.taskData.ccu.argSize, taskMeta.streamId);
    }
    ConvertTaskQueue(taskMeataData);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::GetHcclVmFlagData(HcclSim::HcclVmFlagData& waitFlag)
{
    // 1. 构造路径
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "rb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    auto ret = HcclVmFlagDataRead(fp, waitFlag, HCCLVM_FLAG_FILE_MAGIC);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataRead failed");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::DumpHcclVmFlagData(uint16_t status)
{
    HCCL_VM_DEBUG("Start dumping hccl vm flag data...");
    // 1. 构造完整路径
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    std::string fullPath = rootPath + DATA_FILE_PATH + HCCLVM_FLAG_DATA_FILE;
    // 2. 写文件，通知runner启动
    FILE* fp = fopen(fullPath.c_str(), "wb");
    if (!fp) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }

    HcclSim::HcclVmFlagData flagData{};
    flagData.header.magic = HCCLVM_FLAG_FILE_MAGIC;
    flagData.runner_status = status;
    auto ret = HcclVmFlagDataWrite(fp, flagData);
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR("HcclVmFlagDataWrite failed");
        fclose(fp);
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    fflush(fp);
    fclose(fp);
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::string StorageManager::FindRootPath()
{
    // 使用相对路径前缀：.  ./..  ./../..
    std::string current_search_path = ".";
    for (int i = 0; i <= 3; ++i) {
        char abs_path[PATH_MAX];
        // 尝试获取当前探测点的绝对路径
        if (realpath(current_search_path.c_str(), abs_path) != nullptr) {
            std::string check_target = std::string(abs_path) + PLUGIN_PATH;

            // 检查该绝对路径下的 plugin 目录是否存在
            if (IsDirExists(check_target)) {
                return std::string(abs_path);
            }
        } else {
            // 如果 realpath 失败（例如路径被删除或权限不足）
            HCCL_VM_WARN("Iteration {}: realpath failed for {}", i, current_search_path);
        }

        // 没找到，将探测路径向上推一级
        current_search_path += "/..";
    }

    HCCL_VM_WARN("RootPath NOT found");
    return "";
}

HcclVmResult StorageManager::ConvertTaskQueue(const HcclVmTaskMetaData& taskMeataData)
{
    auto rankSize = GetRankSize();
    m_allRankTaskQueues.resize(rankSize);
    HCCL_VM_INFO("rankSize={}", rankSize);

    auto taskMetaVec = taskMeataData.task_meta;
    uint32_t size = taskMetaVec.size();
    for (uint32_t i = 0; i < size; i++) {
        uint32_t rankId = taskMetaVec[i].rankId;
        if (rankId >= rankSize) {
            HCCL_VM_ERROR("rankId={} is out of range. rankSize={}", rankId, rankSize);
            return HcclVmResult::HCCL_SIM_E_PARA;
        }
        uint32_t streamId = (taskMetaVec[i].streamId);
        if (m_allRankTaskQueues[rankId].size() <= streamId) {
            m_allRankTaskQueues[rankId].resize(streamId + 1);
        }
        m_allRankTaskQueues[rankId][streamId].push(taskMetaVec[i]);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

bool StorageManager::IsDirExists(const std::string& path)
{
    struct stat info;
    if (stat(path.c_str(), &info) != 0) {
        return false; // 不存在
    }
    return (info.st_mode & S_IFDIR); // 存在且是目录
}

uint32_t StorageManager::GetRankSize() const { return m_checker_param.rankSize; }

std::vector<RankChannelInfo>& StorageManager::GetAllRankChannelInfo() { return m_allRankChannelInfo; }

HcclVmInstrData StorageManager::GetHvmInstrData() const { return m_instrData; }

AllRankTaskQueues& StorageManager::GetAllRankTaskQueues() { return m_allRankTaskQueues; }

HcclVmResult StorageManager::InitCcuResource(std::vector<sim::CcuInstrResTab>& instrRes)
{
    RunnerCcuVersion ccuVersion{RunnerCcuVersion::CCU_INVALID};
    if (devType_ == DevType::DEV_TYPE_950) {
        ccuVersion = RunnerCcuVersion::CCU_V1;
#ifdef BUILD_A6_CCU_INSTR
    } else if (devType_ == DevType::DEV_TYPE_960) {
        ccuVersion = RunnerCcuVersion::CCU_V2;
#endif
    } else {
        HCCL_VM_ERROR("Wrong devive type: {:d}", static_cast<int>(devType_));
        return HcclVmResult::HCCL_SIM_E_INTERNAL;
    }
    std::vector<uint64_t> ccuResourceBaseAddr{};
    ccuResourceBaseAddr.push_back(0x123123123);
    ccuResourceBaseAddr.push_back(0x456456456);

    auto rankSize = GetRankSize();
    auto& ccuResMgr = CcuResourceManager::GetInstance();
    for (uint32_t rankId = 0; rankId < rankSize; rankId++) {
        ccuResMgr.Init(rankId, rankSize, ccuVersion, ccuResourceBaseAddr);
        ccuResMgr.InitChannelInfo(rankId, m_allRankChannelInfo[rankId]);
    }
    for (auto& instr : instrRes) {
        CcuInstrData ccuInstr;
        ccuInstr.instrCnt = instr.instrCount;
        ccuInstr.instrData.reserve(instr.instrCount);
        for (uint32_t i = 0; i < instr.instrCount; ++i) {
            hcomm::CcuRep::CcuInstr tmp;
            std::memcpy(&tmp, instr.instrSpace[i], sizeof(hcomm::CcuRep::CcuInstr));
            ccuInstr.instrData.push_back(tmp);
        }
        ccuResMgr.InitInstrInfo(instr.rankId, instr.dieId, ccuInstr);
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult StorageManager::InitAivResourceFromCompositeOpDetail(const sim::CompositeOpDetail& opDetail)
{
    auto ret = AivResourceManager::GetInstance().Init(opDetail.rankId, opDetail.memInfo, GetRankSize());
    if (ret != HcclVmResult::HCCL_SIM_SUCCESS) {
        HCCL_VM_ERROR(
            "init aiv resource failed, "
            "rankId={}, opDetailId={}, memInfoId={}, ret={}",
            opDetail.rankId, opDetail.detail.id, opDetail.memInfo.id, static_cast<int>(ret));
        return ret;
    }
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

void StorageManager::ResetAivResource()
{
    AivGraphExecutorMgr::GetInstance().Reset();
    AivResourceManager::GetInstance().Reset();
}

void StorageManager::DumpAllRankInputOutput(std::vector<std::map<uint32_t, sim::CompositeOpDetail>>& compositeDataMap)
{
    const char* enableDumpData = std::getenv("HCCLVM_ENABLE_DUMP_DATA");
    if (enableDumpData == nullptr || std::string(enableDumpData).empty() || std::string(enableDumpData) == "0") {
        HCCL_VM_DEBUG("HCCLVM_ENABLE_DUMP_DATA is not set");
        return;
    }
    auto rankSize = GetRankSize();
    if (compositeDataMap.empty() || compositeDataMap[0].empty()) {
        HCCL_VM_ERROR("compositeDataMap is empty");
        return;
    }
    std::string rootPath = FindRootPath();
    if (rootPath.empty()) {
        HCCL_VM_ERROR("Failed to find root path");
        return;
    }
    std::string fullPath = rootPath + HCCLVM_ALL_RANK_INPUT_OUTPUT_FILE;
    std::ofstream ofs(fullPath);
    if (!ofs.is_open()) {
        HCCL_VM_ERROR("Open file failed: {}", fullPath);
        return;
    }

    HCCL_VM_INFO("Dump All Rank Input Output Data. rankSize={}", rankSize);
    for (size_t opIdx = 0; opIdx < compositeDataMap.size(); opIdx++) {
        auto& rankTask = compositeDataMap[opIdx];
        if (rankTask.empty()) {
            continue;
        }
        std::vector<BufferInfo> allRankInput(rankSize);
        std::vector<BufferInfo> allRankOutput(rankSize);
        for (auto& it : rankTask) {
            auto& memInfo = it.second.memInfo;
            if (memInfo.inputAddr != 0 && memInfo.inputSize > 0) {
                sim::PhyMemBlock srcPhyMem{};
                auto startAddr = sim::AcquireDevPtrInNoHostProcess((void*)(memInfo.inputAddr), srcPhyMem);
                if (startAddr == nullptr) {
                    HCCL_VM_ERROR(
                        "fail addr(type={}, addr={:x})!", static_cast<int>(BufferType::INPUT), memInfo.inputAddr);
                    return;
                }
                m_allPhyMem.push_back(srcPhyMem);
                allRankInput[it.first].virtualAddr = memInfo.inputAddr;
                allRankInput[it.first].phyAddr = (char*)startAddr;
                allRankInput[it.first].size = memInfo.inputSize;
                HCCL_VM_INFO(
                    "opIdx={}, rankId={}, inputAddr={:x}, size={}", opIdx, it.first, memInfo.inputAddr,
                    memInfo.inputSize);
            }

            if (memInfo.outputAddr != 0 && memInfo.outputSize > 0) {
                sim::PhyMemBlock srcPhyMem{};
                auto startAddr = sim::AcquireDevPtrInNoHostProcess((void*)(memInfo.outputAddr), srcPhyMem);
                if (startAddr == nullptr) {
                    HCCL_VM_ERROR(
                        "fail addr(type={}, addr={:x})!", static_cast<int>(BufferType::OUTPUT), memInfo.outputAddr);
                    return;
                }
                m_allPhyMem.push_back(srcPhyMem);
                allRankOutput[it.first].virtualAddr = memInfo.outputAddr;
                allRankOutput[it.first].phyAddr = (char*)startAddr;
                allRankOutput[it.first].size = memInfo.outputSize;
                HCCL_VM_INFO(
                    "opIdx={}, rankId={}, outputAddr={:x}, size={}", opIdx, it.first, memInfo.outputAddr,
                    memInfo.outputSize);
            }
        }

        const auto& firstComp = rankTask.begin()->second;
        if (firstComp.detail.opDetail.size() < sizeof(::OpDetails)) {
            HCCL_VM_ERROR("opDetail BLOB too small, opIdx={}, size={}", opIdx, firstComp.detail.opDetail.size());
            continue;
        }
        ::OpDetails opDetails{};
        std::memcpy(&opDetails, firstComp.detail.opDetail.data(), sizeof(::OpDetails));

        ofs << "\n=== Op " << std::dec << opIdx << " (opType=" << static_cast<int>(opDetails.opType) << ") ===\n";

        switch (static_cast<HcclCMDType>(opDetails.opType)) {
            case HcclCMDType::HCCL_CMD_BROADCAST:
                PrintOpData1(ofs, allRankInput);
                break;
            case HcclCMDType::HCCL_CMD_ALLGATHER:
            case HcclCMDType::HCCL_CMD_ALLREDUCE:
            case HcclCMDType::HCCL_CMD_SCATTER:
            case HcclCMDType::HCCL_CMD_REDUCE:
            case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            case HcclCMDType::HCCL_CMD_ALLTOALL:
            case HcclCMDType::HCCL_CMD_ALLTOALLV:
                PrintOpData2(ofs, allRankInput, allRankOutput);
                break;
        }
    }
    ofs.close();
    HCCL_VM_INFO("Dump All Rank Input Output Data success.");
}

void StorageManager::FlexiblePrintData(std::ofstream& ofs, const BufferInfo& buffer)
{
    if (buffer.size <= PRINT_DATA_SIZE_THRESHOLD) {
        for (uint32_t i = 0; i < buffer.size; i++) {
            ofs << std::hex << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
        }
        return;
    }

    // 大数据量，只打印数据的前后各1024字节
    for (uint32_t i = 0; i < PRINT_DATA_PREFIX; i++) {
        ofs << std::hex << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
    }
    ofs << "\n...[skipped " << std::dec << buffer.size - PRINT_DATA_SIZE_THRESHOLD << " bytes]...\n";
    for (uint32_t i = buffer.size - PRINT_DATA_SUFFIX; i < buffer.size; i++) {
        ofs << std::hex << static_cast<int>(static_cast<unsigned char>(buffer.phyAddr[i]));
    }
}

void StorageManager::PrintOpData1(std::ofstream& ofs, const std::vector<BufferInfo>& allRankInput)
{
    auto rankSize = GetRankSize();
    for (uint32_t idx = 0; idx < rankSize; idx++) {
        ofs << std::dec << "Rank " << idx << " Input: " << (uint64_t)(void*)allRankInput[idx].virtualAddr
            << ", size=" << allRankInput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankInput[idx]);
        ofs << std::endl;
    }
}

void StorageManager::PrintOpData2(
    std::ofstream& ofs, const std::vector<BufferInfo>& allRankInput, const std::vector<BufferInfo>& allRankOutput)
{
    auto rankSize = GetRankSize();
    for (uint32_t idx = 0; idx < rankSize; idx++) {
        ofs << std::dec << "Rank " << idx << " Input: " << (uint64_t)(void*)allRankInput[idx].virtualAddr
            << ", size=" << allRankInput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankInput[idx]);
        ofs << std::endl;
        ofs << std::dec << "Rank " << idx << " Output: " << (uint64_t)(void*)allRankOutput[idx].virtualAddr
            << ", size=" << allRankOutput[idx].size << std::endl;
        FlexiblePrintData(ofs, allRankOutput[idx]);
        ofs << "\n" << std::endl;
    }
}
} // namespace HcclSim
