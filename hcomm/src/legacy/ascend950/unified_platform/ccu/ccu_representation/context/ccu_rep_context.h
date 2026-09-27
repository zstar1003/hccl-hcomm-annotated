/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_REP_CTX_H
#define HCCL_CCU_REP_CTX_H

#include <set>
#include <string>
#include <unordered_map>

#include "ccu_transport_group.h"
#include "ccu_rep_base.h"
#include "ccu_rep_block.h"
#include "task_param.h"
#include "ccu_ctx_arg.h"
#include "const_val.h"

namespace Hccl {
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
        void Append(std::shared_ptr<CcuRep::CcuRepBase> rep);
        const std::vector<std::shared_ptr<CcuRep::CcuRepBase>>& GetRepSequence();
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
        LoopGroupProfilingInfo& GetLGProfilingInfo();
        const std::vector<std::shared_ptr<CcuRepBase>>& GetWaiteCkeProfilingReps() const;
        void CollectProfilingReps(std::shared_ptr<CcuRep::CcuRepBase> rep);

        void AddSqeProfiling(const CcuCtxArg& arg);
        void AddProfiling(const std::string& name, uint32_t mask);
        void AddProfiling(const CcuTransport& transport, const std::string& name, uint32_t signalIndex, uint32_t mask);
        void AddProfiling(
            const CcuTransportGroup& transportGroup, const std::string& name, uint32_t signalIndex, uint32_t mask);
        void AddProfiling(const std::vector<CcuTransport*>& transports);
        void AddProfiling(
            const std::vector<CcuTransport*>& transports, DataType dataType, DataType outputDataType, ReduceOp opType);

        void SetDependencyInfo(uint32_t id, uint32_t mask, std::shared_ptr<CcuRepBase> rep);
        std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>> GetDependencyInfo(uint32_t id);
        void ClearDependencyInfo();

    protected:
        std::set<std::string> registeredLoop;

    private:
        std::shared_ptr<CcuRep::CcuRepBlock> activeBlock{nullptr};
        std::shared_ptr<CcuRep::CcuRepBlock> mainBlock{nullptr};

        uint32_t dieId{0};
        uint32_t missionId{INVALID_U32};
        uint32_t missionKey{0};

        // CCU Profiling相关数据
        CcuProfilingInfo ccuProfilingInfoCache;
        std::vector<std::shared_ptr<CcuRepBase>> allLgProfilingReps;   // 当前所有的loopGroup Rep
        LoopGroupProfilingInfo lgProfilingInfo;                        // LoopGroup相关profiling缓存信息
        std::vector<std::shared_ptr<CcuRepBase>> waitCkeProfilingReps; // waitCKE相关REP缓存
        std::vector<CcuProfilingInfo> profilingInfo;                   // context全部profiling缓存信息

        std::unordered_map<uint32_t, std::unordered_map<uint32_t, std::vector<std::shared_ptr<CcuRepBase>>>> depInfo;
    };

}; // namespace CcuRep
}; // namespace Hccl

#endif // HCCL_CCU_REP_CTX_H
