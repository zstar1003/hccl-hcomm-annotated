/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "microcode_optimizer.h"

#include <chrono>
#include <sstream>
#include <string>

#include "ccu_log.h"
#include "ccu_microcode_v1.h" // hcomm::CcuRep::CcuV2::ParseInstrV2, 供 opt_log 输出可读形式

namespace hcomm {
namespace CcuOpt {

    MicrocodeOptimizer::MicrocodeOptimizer() = default;

    OptimizerOptions MicrocodeOptimizer::DefaultOptions()
    {
        // 极简后端优化恒为 CkeOnly (只处理 CKE 写后读), 无其他档位, 运行时不可改.
        OptimizerOptions opts{};
        opts.schedLevel = SchedLevel::CkeOnly;
        return opts;
    }

    CcuRep::CcuInstrInfo MicrocodeOptimizer::Optimize(const CcuRep::CcuInstrInfo& input)
    {
        schedOpts_.level = opts_.schedLevel;

        using Clock = std::chrono::steady_clock;
        auto usSince = [](Clock::time_point startTime) -> uint64_t {
            return static_cast<uint64_t>(
                std::chrono::duration_cast<std::chrono::microseconds>(Clock::now() - startTime).count());
        };

        const auto totalStart = Clock::now();

        const auto pass1Start = Clock::now();
        InstructionScheduler sched(schedOpts_);
        CcuRep::CcuInstrInfo afterSched = sched.Schedule(input);
        stats_.sched = sched.Stats();
        stats_.pass1DurationUs = usSince(pass1Start);
        stats_.totalDurationUs = usSince(totalStart);

        HCCL_INFO(
            "[CcuMicrocodeOpt][timing] total=%lluus, pass1(sched)=%lluus",
            static_cast<unsigned long long>(stats_.totalDurationUs),
            static_cast<unsigned long long>(stats_.pass1DurationUs));

        DumpOptLog(input, afterSched);
        return afterSched;
    }

    namespace {

        // 打印输入序列 (用户视角的原始指令). 全局 id = startInstrId + local, 与硬件视角一致.
        void DumpOptLogInput(const CcuRep::CcuInstrInfo& input)
        {
            const uint16_t inStart = input.startInstrId;
            for (size_t i = 0; i < input.instrVec.size(); ++i) {
                HCCL_INFO(
                    "[CcuMicrocodeOpt][optlog] input   [%3zu] gid=%u: %s", i, static_cast<unsigned>(inStart + i),
                    CcuRep::CcuV2::ParseInstrV2(input.instrVec.data() + i).c_str());
            }
        }

        // 打印输出序列 + 后→前映射. runlog 里报"指令 gid=X 挂了", 用户在此段直接搜 gid=X 即可
        // 定位到 src 行, src=-1 表示是新插入的 NOP.
        void DumpOptLogOutput(
            const CcuRep::CcuInstrInfo& input, const CcuRep::CcuInstrInfo& output,
            const std::vector<int32_t>& originIndex, bool mapConsistent)
        {
            const uint16_t inStart = input.startInstrId;
            const uint16_t outStart = output.startInstrId;
            for (size_t i = 0; i < output.instrVec.size(); ++i) {
                std::string srcTag = "NEW  (inserted NOP)";
                if (mapConsistent) {
                    int32_t src = originIndex[i];
                    if (src >= 0) {
                        std::ostringstream oss;
                        oss << "src[" << src << "] gid=" << (inStart + static_cast<uint32_t>(src));
                        srcTag = oss.str();
                    }
                } else {
                    srcTag = "N/A (mapping omitted)";
                }
                HCCL_INFO(
                    "[CcuMicrocodeOpt][optlog] output  [%3zu] gid=%u <- %s : %s", i,
                    static_cast<unsigned>(outStart + i), srcTag.c_str(),
                    CcuRep::CcuV2::ParseInstrV2(output.instrVec.data() + i).c_str());
            }
        }

    } // namespace

    void MicrocodeOptimizer::DumpOptLog(const CcuRep::CcuInstrInfo& input, const CcuRep::CcuInstrInfo& output) const
    {
        const auto& originIndex = stats_.sched.originIndex;
        const bool mapConsistent = originIndex.size() == output.instrVec.size();
        if (!mapConsistent) {
            HCCL_WARNING(
                "[CcuMicrocodeOpt][optlog] origin index size mismatch: "
                "originIndex=%zu, output=%zu; mapping will be omitted.",
                originIndex.size(), output.instrVec.size());
        }

        HCCL_INFO(
            "[CcuMicrocodeOpt][optlog] === begin "
            "(inputInstrCount=%u, outputInstrCount=%u, "
            "missionStart before=%u after=%u) ===",
            input.instrCount, output.instrCount, input.missionStartInstrId, output.missionStartInstrId);

        DumpOptLogInput(input);
        DumpOptLogOutput(input, output, originIndex, mapConsistent);

        HCCL_INFO("[CcuMicrocodeOpt][optlog] === end ===");
    }

    CcuRep::CcuInstrInfo MicrocodeOptimizer::Run(
        const CcuRep::CcuInstrInfo& input, uint16_t reserveXnId, uint16_t reserveCkeId,
        const std::vector<PinnedGroup>& pinnedGroups)
    {
        OptimizerOptions opts = DefaultOptions(); // 恒 CkeOnly.
        HCCL_INFO(
            "[CcuMicrocodeOpt] active options: sched=CkeOnly, reserveXn=%u, reserveCke=%u, pinnedGroups=%zu",
            static_cast<unsigned>(reserveXnId), static_cast<unsigned>(reserveCkeId), pinnedGroups.size());

        MicrocodeOptimizer opt;
        opt.SetOptions(opts);
        opt.SetPinnedGroups(&pinnedGroups);
        return opt.Optimize(input);
    }

} // namespace CcuOpt
} // namespace hcomm
