/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_RESOURCE_MANAGER_H
#define HCCL_SIM_CCU_RESOURCE_MANAGER_H

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

#include "ccu_resource_v1.h"
#include "ccu_resource_v2.h"

constexpr uint16_t U16_INVALID = UINT16_MAX;
constexpr uint32_t U32_INVALID = UINT32_MAX;
constexpr uint32_t S32_INVALID = INT32_MAX;
constexpr uint64_t U64_INVALID = UINT64_MAX;

struct CcuResData {
    RunnerCcuVersion version{RunnerCcuVersion::CCU_V1};
    std::vector<std::unique_ptr<CcuResourceV1>> v1Res;
    std::vector<std::unique_ptr<CcuResourceV2>> v2Res;
};

// 记录CCU中偏移地址的类型
enum class CcuComponerntType : uint16_t {
    UNKNOWN = 0,
    XN_A6 = 1,
    CKE_A6 = 2,
    PFE_A6 = 3,
    CHANNEL_A6 = 4,
    JETTY_A6 = 5,
    MISSION_A6 = 6,
    LOOP_A6 = 7,
};

class CcuResourceManager {
public:
    CcuResourceManager(const CcuResourceManager&) = delete;
    CcuResourceManager& operator=(const CcuResourceManager&) = delete;

    static CcuResourceManager& GetInstance()
    {
        static CcuResourceManager instance;
        return instance;
    }

    void Reset();
    void Init(int rankId, int rankSize, RunnerCcuVersion version, const std::vector<uint64_t>& ccuResourceBaseAddr);
    void InitInstrInfo(int rankId, int dieId, const CcuInstrData& ccuInstrInfo);
    void InitChannelInfo(int rankId, const RankChannelInfo& channelInfo);
    void InitChannelId2RmtRankMap(int rankId, int dieId, uint16_t channelId, int rmtRank, uint16_t rmtDieId);

    void AddTaskInfo(int rankId, const HcclTaskMetaData& task); // 收集SQE参数信息
    uint64_t GetXnValue(int rankId, int dieId, uint16_t xnId) const;
    uint64_t GetGsaValue(int rankId, int dieId, uint16_t gsaId) const;
    uint16_t GetCkeValue(int rankId, int dieId, uint16_t ckeId) const;
    char* GetMsAddr(int rankId, int dieId, uint16_t msId) const;
    std::pair<int, int> GetRmtCcu(int rankId, int dieId, uint16_t channelId) const;
    uint64_t* GetXnAddr(int rankId, int dieId, uint16_t xnId) const;
    void UpdateXnValue(int rankId, int dieId, uint16_t xnId, uint64_t value);
    void UpdateGsaValue(int rankId, int dieId, uint16_t gsaId, uint64_t value);
    void UpdateCkeValue(int rankId, int dieId, uint16_t ckeId, uint16_t value);
    void
    TransMemToMem(void* srcBuf, void* dstBuf, uint64_t length, bool reduceEn, uint16_t reduceOp, uint16_t dataType);
    void
    TransMSToMS(int srcRank, int srcDie, int dstRank, int dstDie, uint16_t srcMsId, uint16_t dstMsId, uint16_t length);
    bool TransMemToXn(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length);
    bool TransXnToMem(int rankId, int dieId, uint16_t xnId, uint64_t buf, uint16_t length);
    bool TransMSToMem(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length);
    bool TransMemToMS(int rankId, int dieId, uint16_t msId, void* buf, uint16_t length);

    // 通过MS的地址找到MS的Id
    bool GetMSIdByAddr(uint32_t dieId, uint64_t addr, uint16_t& msId);
    // 通过XnId所在的地址值来找到XnId以及寄存器的类型
    bool GetXnAndTypeIdByAddr(uint32_t dieId, uint64_t xnAddr, CcuComponerntType& type, uint16_t& xnId);
    // 通过XnId所在的地址值来找到XnId
    bool GetXnIdByAddr(uint32_t dieId, CcuComponerntType type, uint64_t xnAddr, uint16_t& xnId);
    // 通过XnId所在的地址值来找到XnId
    bool GetAddrByXnId(uint32_t dieId, CcuComponerntType type, uint16_t xnId, uint64_t& xnAddr);

    uint64_t GetSqeArgValue(int rankId, int dieId, uint16_t argId) const;
    uint16_t GetInstrCnt(int rankId, int dieId) const;
    std::vector<hcomm::CcuRep::CcuInstr> GetInstrData(int rankId, int dieId) const;
    std::string GetInstrDescribe(int rankId, int dieId, int instrId) const;
    std::shared_ptr<CcuSimulator>
    InitSimulator(int rankId, int dieId, uint16_t instrStartId, uint16_t endInstrId, uint16_t instCnt);

    RunnerCcuVersion GetVersion() const { return ccuResData_.version; }

    void DumpChannelId2RmtRank(int rankId, int dieId) const;
    void DumpCcuInstructions(int rankId) const;
    void DumpCcuAllResouceInfo(int rankId) const;
    void DumpCcuXnResouceInfo(int rankId) const;
    void DumpCcuGsaResouceInfo(int rankId) const;
    void DumpCcuCkeResouceInfo(int rankId) const;
    void DumpCcuChannelResouceInfo(int rankId) const;

private:
    CcuResourceManager() = default;
    ~CcuResourceManager() = default;

private:
    bool enableDump_{false};
    CcuResData ccuResData_{}; // ccu资源数据
    std::mutex ccuExecutorMutex;
    std::vector<uint64_t> ccuResourceBaseAddr_{};
};

#endif // HCCL_SIM_CCU_RESOURCE_MANAGER_H
