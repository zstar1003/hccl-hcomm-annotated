/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef UT_CCU_MICRO_SIM_H
#define UT_CCU_MICRO_SIM_H

#include <array>
#include <cstdint>
#include <cstring>

#include "ccu_instr_info_v1.h" // 引入 CcuInstr / CcuInstrInfo / CcuInstrHeader

namespace hcomm {
namespace CcuRep {

    // V1 微码 opcode 常量。与 ccu_microcode.cc 匿名命名空间、SimCcuV1 (ccu_microcode_common_v1.h) 保持一致。
    namespace MicroSimV1 {
        constexpr uint16_t LOAD_TYPE = 0x0;
        constexpr uint16_t CTRL_TYPE = 0x1;

        constexpr uint16_t LOADSQEARGSTOXN_CODE = 0x1;
        constexpr uint16_t LOADIMDTOXN_CODE = 0x3;
        constexpr uint16_t LOADXX_CODE = 0x6;

        constexpr uint16_t JMP_CODE = 0x5;

        constexpr uint16_t XN_MAX = 3072; // CCU_RESOURCE_XN_MAX
    } // namespace MicroSimV1

    // CCU V1 微码解释器(精简实现)。载入 CcuKernel 翻译生成的 CcuInstrInfo 并解释执行,
    // 执行完毕后可读取 Xn 寄存器的最终值,供测试与预期结果对比。
    class CcuMicroSim {
    public:
        CcuMicroSim() = default;
        ~CcuMicroSim() = default;

        // 载入待执行的指令序列。PC 在 [startInstrId, startInstrId + instrCount) 范围内执行,
        // 通过 instrVec[pc - startInstrId] 索引指令。
        void Load(const CcuInstrInfo& info)
        {
            instr_ = info.instrVec.empty() ? nullptr : info.instrVec.data();
            start_ = info.startInstrId;
            end_ = static_cast<uint16_t>(info.startInstrId + info.instrCount);
            ResetXn();
        }

        void SetXn(uint16_t id, uint64_t v)
        {
            if (id < MicroSimV1::XN_MAX) {
                xn_[id] = v;
            }
        }

        uint64_t GetXn(uint16_t id) const { return (id < MicroSimV1::XN_MAX) ? xn_[id] : 0; }

        uint16_t GetStartInstrId() const { return start_; }
        uint16_t GetEndInstrId() const { return end_; }

        // 执行指令序列。正常跑完返回 true;PC 越界、寄存器 id 越界或空程序返回 false。
        bool Run()
        {
            if (instr_ == nullptr || start_ == end_) {
                return false;
            }
            uint16_t pc = start_;
            // 上限保护,防止跳转死循环(指令数远小于 64K)。
            constexpr uint32_t MAX_STEPS = 0x10000;
            for (uint32_t steps = 0; steps < MAX_STEPS; ++steps) {
                if (pc == end_) {
                    return true; // 正常结束
                }
                if (pc < start_ || pc >= end_) {
                    return false; // PC 越界
                }
                const CcuInstr& ins = instr_[pc - start_];
                uint16_t type = ins.header.type;
                uint16_t code = ins.header.code;
                if (type == MicroSimV1::LOAD_TYPE) {
                    if (code == MicroSimV1::LOADIMDTOXN_CODE) {
                        if (!CheckXnId(ins.v1.loadImdToXn.xnId)) {
                            return false;
                        }
                        xn_[ins.v1.loadImdToXn.xnId] = ins.v1.loadImdToXn.immediate;
                        pc++;
                    } else if (code == MicroSimV1::LOADSQEARGSTOXN_CODE) {
                        if (!CheckXnId(ins.v1.loadSqeArgsToXn.xnId) || !CheckXnId(ins.v1.loadSqeArgsToXn.sqeArgsId)) {
                            return false;
                        }
                        xn_[ins.v1.loadSqeArgsToXn.xnId] = sqeArgs_[ins.v1.loadSqeArgsToXn.sqeArgsId];
                        pc++;
                    } else if (code == MicroSimV1::LOADXX_CODE) {
                        // *xdId = *xmId + *xnId
                        if (!CheckXnId(ins.v1.loadXX.xdId) || !CheckXnId(ins.v1.loadXX.xmId)
                            || !CheckXnId(ins.v1.loadXX.xnId)) {
                            return false;
                        }
                        xn_[ins.v1.loadXX.xdId] = xn_[ins.v1.loadXX.xmId] + xn_[ins.v1.loadXX.xnId];
                        pc++;
                    } else {
                        pc++; // 其余 load 指令按 nop
                    }
                } else if (type == MicroSimV1::CTRL_TYPE && code == MicroSimV1::JMP_CODE) {
                    // 语义与 JumpExecutor::RunV1 一致:conditionXn != expectData 时,跳转到 dstInstrXn 指向的 PC
                    if (!CheckXnId(ins.v1.jmp.conditionXnId) || !CheckXnId(ins.v1.jmp.dstInstrXnId)) {
                        return false;
                    }
                    uint64_t condVal = xn_[ins.v1.jmp.conditionXnId];
                    if (condVal != ins.v1.jmp.expectData) {
                        pc = static_cast<uint16_t>(xn_[ins.v1.jmp.dstInstrXnId]);
                    } else {
                        pc++;
                    }
                } else {
                    pc++; // Nop / SetCKE / ClearCKE / Loop 等按 nop
                }
            }
            return false; // 超出步数上限
        }

        // 设置 SQE 参数值,供 LoadSqeArgsToXn 指令读取。
        // 当前测试用例不使用此接口(kernel 参数在注册阶段已固化为立即数)。
        void SetSqeArg(uint16_t argId, uint64_t v)
        {
            if (argId < MicroSimV1::XN_MAX) {
                sqeArgs_[argId] = v;
            }
        }

    private:
        // xn_/sqeArgs_ 访问前的越界守卫,与 SetXn/GetXn/SetSqeArg 保持一致。
        // 越界即返回 false,令 Run() 中止执行(避免栈上 OOB 读写);与同文件 PC 越界、
        // 步数上限等"静默返回失败、由测试断言发现"的风格一致。
        static bool CheckXnId(uint16_t id) { return id < MicroSimV1::XN_MAX; }

        void ResetXn()
        {
            xn_.fill(0);
            sqeArgs_.fill(0);
        }

        const CcuInstr* instr_{nullptr};
        uint16_t start_{0};
        uint16_t end_{0};
        std::array<uint64_t, MicroSimV1::XN_MAX> xn_{};
        std::array<uint64_t, MicroSimV1::XN_MAX> sqeArgs_{}; // sqeArgsId -> 值
    };

}; // namespace CcuRep
}; // namespace hcomm

#endif // UT_CCU_MICRO_SIM_H
