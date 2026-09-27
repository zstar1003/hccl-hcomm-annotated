/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_REP_CTX_H
#define CCU_REP_CTX_H

#include <set>
#include <string>
#include <unordered_map>

#include "hcomm_primitives.h"
#include "ccu_rep_base_v1.h"
#include "ccu_rep_block_v1.h"

#include "ccu_common.h"
#include "task_param.h"

namespace hcomm {
constexpr uint16_t CCU_MAX_CHANNEL_NUM = 16;                    // 最多16条link
constexpr uint16_t INVALID_CKE_ID = 0xFFFF;                     // CKE ID非法值
constexpr uint16_t INVALID_VALUE_CHANNELID = 0xFFFF;            // channel id非法值
constexpr uint64_t INVALID_VALUE_NOTIFYID = 0xFFFFFFFFFFFFFFFF; // NOTIFY id非法值

enum class CcuProfilinType { CCU_TASK_PROFILING, CCU_WAITCKE_PROFILING, CCU_LOOPGROUP_PROFILING, CCU_MAP_PROFILING };

using CcuProfilingInfo = Hccl::CcuProfilingInfo;
namespace CcuRep {

    struct LoopGroupProfilingInfo {
        std::vector<CcuProfilingInfo> ccuProfilingInfos;
        std::unordered_map<std::shared_ptr<CcuRep::CcuRepBase>, uint32_t> loadRep2ArgIdxMap; // loadArg rep -> argIdx
        std::vector<std::shared_ptr<CcuRepBase>> assignProfilingReps;                        // assign rep
        std::vector<std::shared_ptr<CcuRepBase>> lgProfilingReps;                            // loopgroup rep
    };

    class CcuRepContext {
    public:
        explicit CcuRepContext();
        virtual ~CcuRepContext();

        // 平台层内部使用
        std::shared_ptr<CcuRep::CcuRepBlock> CurrentBlock();
        void SetCurrentBlock(std::shared_ptr<CcuRep::CcuRepBlock> repBlock);
        virtual void Append(std::shared_ptr<CcuRep::CcuRepBase> rep);
        const std::vector<std::shared_ptr<CcuRep::CcuRepBase>>& GetRepSequence() const;
        std::shared_ptr<CcuRep::CcuRepBase> GetRepByInstrId(uint16_t instrId);
        void DumpReprestation();

        void SetDieId(uint32_t dieId);
        uint32_t GetDieId() const;
        void SetMissionId(uint32_t missionId);
        uint32_t GetMissionId() const;
        void SetMissionKey(uint32_t missionKey);
        uint32_t GetMissionKey() const;

        // ccu profiling相关接口
        std::vector<CcuProfilingInfo>& GetProfilingInfo();
        CcuRep::LoopGroupProfilingInfo& GetLGProfilingInfo();
        const std::vector<std::shared_ptr<CcuRep::CcuRepBase>>& GetWaiteCkeProfilingReps() const;
        void CollectProfilingReps(std::shared_ptr<CcuRep::CcuRepBase> rep);

        void AddSqeProfiling(const std::string& kernelName);
        int32_t AddProfiling(const std::string& name, uint32_t mask);
        int32_t AddProfiling(const ChannelHandle channel, const std::string& name, uint32_t signalIndex, uint32_t mask);
        int32_t AddProfiling(const ChannelHandle* channels, uint32_t channelNum);
        int32_t AddProfiling(
            const ChannelHandle* channels, uint32_t channelNum, HcommDataType hcommDataType,
            HcommDataType hcommOutputDataType, HcommReduceOp hcommOpType);

        void SetDependencyInfo(uint32_t id, uint32_t mask, const std::shared_ptr<CcuRepBase>& rep);
        std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>> GetDependencyInfo(uint32_t id);
        void EraseDependencyInfo(uint32_t id);
        void ClearDependencyInfo();

    public:
        // CCU Profiling相关数据
        CcuProfilingInfo ccuProfilingInfoCache;
        std::vector<std::shared_ptr<CcuRepBase>> allLgProfilingReps;   // 当前所有的loopGroup Rep
        LoopGroupProfilingInfo lgProfilingInfo;                        // LoopGroup相关profiling缓存信息
        std::vector<std::shared_ptr<CcuRepBase>> waitCkeProfilingReps; // waitCKE相关REP缓存
        std::vector<CcuProfilingInfo> profilingInfo;                   // context全部profiling缓存信息
        // 需要校验返回值是否为nullptr
        CcuInsGeneratorBase* GetInsGenerator() { return insGenerator; }

        void SetInsGenerater(CcuInsGeneratorBase* insGeneratorBase) { insGenerator = insGeneratorBase; }

    protected:
        std::set<std::string> registeredLoop;
        CcuInsGeneratorBase* insGenerator{nullptr};
        std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>> depInfo;

    private:
        std::shared_ptr<CcuRep::CcuRepBlock> activeBlock{nullptr};
        std::shared_ptr<CcuRep::CcuRepBlock> mainBlock{nullptr};

        uint32_t dieId{UINT32_MAX};
        uint32_t missionId{UINT32_MAX};
        uint32_t missionKey{0};
    };

}; // namespace CcuRep
}; // namespace hcomm

#endif // _CCU_REP_CTX_H
