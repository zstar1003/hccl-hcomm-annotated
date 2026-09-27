/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_EXECUTOR_MANAGER_H
#define HCCL_SIM_CCU_EXECUTOR_MANAGER_H

#include <cstdint>
#include <functional>
#include <map>
#include <memory>

#include "ccu_executor_base.h"
#include "ccu_microcode_common_v1.h"

struct CcuExecutorKey {
    RunnerCcuVersion version;
    uint16_t header;

    bool operator<(const CcuExecutorKey& other) const
    {
        if (version != other.version) {
            return version < other.version;
        }
        return header < other.header;
    }
};

using CcuExecutorCreateFunc = std::function<std::unique_ptr<CcuExecutorBase>(
    int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)>;

// ccu executor实例创建接口的管理类
class CcuExecutorCreateFuncMgr {
public:
    static CcuExecutorCreateFuncMgr& Instance();
    CcuExecutorCreateFuncMgr(const CcuExecutorCreateFuncMgr&) = delete;
    CcuExecutorCreateFuncMgr& operator=(const CcuExecutorCreateFuncMgr&) = delete;

    void RegFunc(RunnerCcuVersion version, uint16_t instrType, const CcuExecutorCreateFunc& func)
    {
        CcuExecutorKey key{version, instrType};
        container[key] = func;
    }

    const CcuExecutorCreateFunc GetFunc(RunnerCcuVersion version, uint16_t instrType)
    {
        CcuExecutorKey key{version, instrType};
        auto res = container.find(key);
        if (res == container.end()) {
            return nullptr;
        }
        return res->second;
    }

private:
    CcuExecutorCreateFuncMgr() = default;
    ~CcuExecutorCreateFuncMgr() = default;

private:
    std::map<CcuExecutorKey, CcuExecutorCreateFunc> container{};
};

// 根据指令类型注册创建ccuExecutor实例的函数
class CcuExecutorCreateFuncRegister {
public:
    CcuExecutorCreateFuncRegister(
        RunnerCcuVersion version, uint16_t type, uint16_t code, const CcuExecutorCreateFunc& func)
    {
        hcomm::CcuRep::CcuInstrHeader ccuInstr = hcomm::CcuRep::InstrHeader(type, code);
        CcuExecutorCreateFuncMgr::Instance().RegFunc(version, ccuInstr.header, func);
    }
    ~CcuExecutorCreateFuncRegister() = default;
};

// 根据指令类型创建对应的ccuExecutor实例
class CcuExecutorFactory {
public:
    static std::unique_ptr<CcuExecutorBase> MakeCcuExecutorInstance(
        RunnerCcuVersion version, uint16_t instrType, int streamId, int rankId, int dieId,
        const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator)
    {
        auto createFunc = CcuExecutorCreateFuncMgr::Instance().GetFunc(version, instrType);
        if (createFunc == nullptr) {
            return nullptr;
        }
        return createFunc(streamId, rankId, dieId, instr, ccuSimulator);
    }
    ~CcuExecutorFactory() = default;
};

#define REG_CCU_EXECUTOR_CREATE_FUNC_V1(type, code, className)                                                      \
    static CcuExecutorCreateFuncRegister g_reg##className##_v1(                                                     \
        RunnerCcuVersion::CCU_V1, type, code,                                                                       \
        [](int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator) { \
            auto executor = std::make_unique<className>(streamId, rankId, dieId, instr, ccuSimulator);              \
            executor->SetVersion(RunnerCcuVersion::CCU_V1);                                                         \
            return executor;                                                                                        \
        })

// V2指令注册宏
#define REG_CCU_EXECUTOR_CREATE_FUNC_V2(type, code, className)                                                      \
    static CcuExecutorCreateFuncRegister g_reg##className##_v2(                                                     \
        RunnerCcuVersion::CCU_V2, type, code,                                                                       \
        [](int streamId, int rankId, int dieId, const hcomm::CcuRep::CcuInstr& instr, CcuSimulator* ccuSimulator) { \
            auto executor = std::make_unique<className>(streamId, rankId, dieId, instr, ccuSimulator);              \
            executor->SetVersion(RunnerCcuVersion::CCU_V2);                                                         \
            return executor;                                                                                        \
        })

// 兼容旧宏（默认V1）
#define REG_CCU_EXECUTOR_CREATE_FUNC REG_CCU_EXECUTOR_CREATE_FUNC_V1

#endif // HCCL_SIM_CCU_EXECUTOR_MANAGER_H
