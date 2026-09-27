/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "KERNEL_STUB"

#include <errno.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <dlfcn.h>
#include <filesystem>
#include <iostream>
#include <limits>
#include <map>
#include <sstream>
#include <set>
#include <string>
#include <unistd.h>
#include <vector>

#include "acl/acl_base.h"
#include "acl/acl_rt.h"
#include "aiv_kernel/aiv_mode_stub/aiv_mode_stub_base.h"
#include "hccl/hccl_types.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "store_sim_comm_pool_policy.h"
#include "store_sim_device_memory_manager.h"
#include "store_sim_memory_manager.h"
#include "store_sim_run_mode.h"
#include "store_sim_store_pub.h"
#include "db_sim_op_db_ops.h"
#include "db_sim_runner_ops.h"
#include "sim_common_api.h"
#include "hccl_proxy_common.h"
#include "sim_sub_process_manager.h"

namespace fs = std::filesystem;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

struct ArgsBuffer {
    void* data;
    uint64_t size;
};

thread_local pid_t g_devicePid = 0;

static bool CheckDeviceProcStatus()
{
    if (g_devicePid == 0) {
        return false;
    }
    int status = 0;
    pid_t result = waitpid(g_devicePid, &status, WNOHANG);
    if (result == 0) {
        return true;
    }
    if (result == g_devicePid) {
        if (WIFSIGNALED(status)) {
            HCCL_VM_ERROR("device process[{}] killed by signal {}", g_devicePid, WTERMSIG(status));
        } else {
            HCCL_VM_ERROR("device process[{}] exited with status {}", g_devicePid, WEXITSTATUS(status));
        }
    } else {
        HCCL_VM_ERROR("waitpid failed for pid {}, errno: {} ({})", g_devicePid, errno, strerror(errno));
    }
    FlushLog();
    exit(EXIT_FAILURE);
    g_devicePid = 0;
    return false;
}

namespace sim {
    constexpr size_t MAX_ARGS_BUFF_SIZE = 64 * 1024U;

    struct FuncArgsDetail {
        uint8_t* argsData{nullptr};
        size_t argsDataSize{0};
        bool isHold{false};
    };

    struct FuncArgs {
        uint8_t* argsBuff{nullptr};
        size_t argsBufferSize{0};
        bool isSysMem{true};
        size_t useOffset{0};
        std::vector<FuncArgsDetail*> argDetail;

        FuncArgs()
        {
            argsBuff = new uint8_t[MAX_ARGS_BUFF_SIZE];
            argsBufferSize = MAX_ARGS_BUFF_SIZE;
            isSysMem = true;
        }
        ~FuncArgs()
        {
            if (isSysMem && argsBuff) {
                delete[] argsBuff;
            }

            for (auto& arg : argDetail) {
                delete arg;
            }
        }

        void ResetArgsBuff()
        {
            if (isSysMem && argsBuff) {
                delete[] argsBuff;
                argsBuff = nullptr;
                argsBufferSize = 0;
            }
        }
    };

    struct FuncHandle {
        std::string funcName{""};
        std::string kernelName{""};
        std::string soName{""};
        std::vector<FuncArgs*> funArgs;
        ~FuncHandle()
        {
            for (auto& funcArg : funArgs) {
                delete funcArg;
            }
        }
    };

    struct DevBinary;
    struct Program {
        DevBinary* bin{nullptr};
        std::map<std::string, FuncHandle*> funcs;
        ~Program()
        {
            for (auto& func : funcs) {
                delete func.second;
            }
        }
    };

    struct DevBinary {
        std::string binPath{""};
        std::map<std::string, std::string> funcSoMap;
        void* data{nullptr};
        size_t dataLen{0};
        Program prog;

        ~DevBinary()
        {
            if (data != nullptr) {
                delete[] (char*)data;
            }
        }
    };

    std::set<DevBinary*> g_kernelBinary;
} // namespace sim

aclrtBinary aclrtCreateBinary(const void* data, size_t dataLen)
{
    sim::DevBinary* binPtr = new sim::DevBinary();

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("failed");
        return 0;
    }

    binPtr->data = reinterpret_cast<void*>(new char[dataLen]);
    memcpy(binPtr->data, data, dataLen);
    binPtr->dataLen = dataLen;
    HCCL_VM_INFO("dataLen{:d} binary{:p}", dataLen, (aclrtBinary)(binPtr));
    return (aclrtBinary)(binPtr);
}

aclError aclrtDestroyBinary(aclrtBinary binary)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    if (auto search = sim::g_kernelBinary.find(binPtr); search == sim::g_kernelBinary.end()) {
        HCCL_VM_ERROR("can not find this binary");
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    sim::g_kernelBinary.erase(binPtr);

    HCCL_VM_INFO(" binPtr{:p}", binary);
    delete binPtr;
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoad(const aclrtBinary binary, aclrtBinHandle* binHandle)
{
    sim::DevBinary* binPtr = (sim::DevBinary*)binary;
    // 需要解析binary
    *binHandle = (aclrtBinHandle) & (binPtr->prog);
    HCCL_VM_INFO(" binHandle:{:p}", *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle)
{
    (void)binHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    // 复用已有的
    for (auto* devBin : sim::g_kernelBinary) {
        if (devBin != nullptr && devBin->binPath == binPath) {
            *binHandle = (aclrtBinHandle) & (devBin->prog);
            HCCL_VM_INFO("binPath:{} reused binHandle:{:p}", binPath, *binHandle);
            return ACL_SUCCESS;
        }
    }

    sim::DevBinary* binPtr = new sim::DevBinary();
    binPtr->binPath = binPath;
    binPtr->prog.bin = binPtr;
    sim::ParseKernelJson(binPath, binPtr->funcSoMap);

    auto res = sim::g_kernelBinary.insert(binPtr);
    if (!res.second) {
        HCCL_VM_ERROR("file:{} insert failed", binPath);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    *binHandle = (aclrtBinHandle) & (binPtr->prog);
    HCCL_VM_INFO(" binPath:{} binHandle{:p}", binPath, *binHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryLoadFromData(
    const void* data, size_t length, const aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    (void)data;
    (void)length;
    (void)options;
    (void)binHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)binHandle;

    std::string kernelSoName{""};
    std::string funcName(kernelName);
    if (prog->bin != nullptr) {
        auto soIt = prog->bin->funcSoMap.find(funcName);
        if (soIt != prog->bin->funcSoMap.end()) {
            kernelSoName = soIt->second;
        }
    }

    auto funcIter = prog->funcs.find(funcName);
    if (funcIter == prog->funcs.end()) {
        HCCL_VM_WARN("kernelName:{} not register insert it", kernelName);

        sim::FuncHandle* func = new sim::FuncHandle;
        func->funcName = funcName;
        func->kernelName = kernelName;
        func->soName = kernelSoName;
        auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
        if (!res.second) {
            HCCL_VM_ERROR("func:{} kernelName:{} insert failed", funcName, kernelName);
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    } else {
        *funcHandle = reinterpret_cast<aclrtFuncHandle>(funcIter->second);
    }

    HCCL_VM_INFO(" funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunctionByEntry(aclrtBinHandle binHandle, uint64_t funcEntry, aclrtFuncHandle* funcHandle)
{
    (void)binHandle;
    (void)funcEntry;
    (void)funcHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void** aicAddr, void** aivAddr)
{
    (void)funcHandle;
    (void)aicAddr;
    (void)aivAddr;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionName(aclrtFuncHandle funcHandle, uint32_t maxLen, char* name)
{
    (void)maxLen;
    sim::FuncHandle* funcHandlePtr = (sim::FuncHandle*)(uintptr_t)funcHandle;

    memcpy(name, funcHandlePtr->funcName.data(), funcHandlePtr->funcName.length());
    HCCL_VM_INFO(" funcName{}", funcHandlePtr->funcName.data());
    return ACL_SUCCESS;
}

aclError aclrtRegisterCpuFunc(
    const aclrtBinHandle handle, const char* funcName, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    sim::Program* prog = (sim::Program*)(uintptr_t)handle;

    sim::FuncHandle* func = new sim::FuncHandle;
    func->funcName = funcName;
    func->kernelName = kernelName;
    auto res = prog->funcs.insert(std::pair<std::string, sim::FuncHandle*>(func->funcName, func));
    if (!res.second) {
        HCCL_VM_ERROR("func:{} kernelName:{} insert failed", funcName, kernelName);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    *funcHandle = reinterpret_cast<aclrtFuncHandle>(func);
    HCCL_VM_INFO(" funcHandle:{:p}", *funcHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle* argsHandle)
{
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    sim::FuncArgs* args = new sim::FuncArgs;
    func->funArgs.push_back(args);

    auto iter = func->funArgs.rbegin();
    *argsHandle = (aclrtArgsHandle)args;
    HCCL_VM_INFO("FuncHandle:{:p} ArgsHandle:{:p} ", funcHandle, *argsHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsInitByUserMem(
    aclrtFuncHandle funcHandle, aclrtArgsHandle argsHandle, void* userHostMem, size_t actualArgsSize)
{
    (void)funcHandle;
    HCCL_VM_INFO(" argsHandle:{:p} userHostMem:{:p},actualArgsSize:{:d}", argsHandle, userHostMem, actualArgsSize);
    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
    args->ResetArgsBuff();
    args->argsBuff = (uint8_t*)userHostMem;
    args->argsBufferSize = actualArgsSize;
    args->isSysMem = false;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetMemSize(aclrtFuncHandle funcHandle, size_t userArgsSize, size_t* actualArgsSize)
{
    (void)funcHandle;
    HCCL_VM_INFO("userArgsSize {:d}.", userArgsSize);
    *actualArgsSize = userArgsSize;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetHandleMemSize(aclrtFuncHandle funcHandle, size_t* memSize)
{
    HCCL_VM_INFO("funcHandle:{:p} userArgsSize 64k", funcHandle);
    // 句柄 + 参数的内存大小
    *memSize = sim::MAX_ARGS_BUFF_SIZE;
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void* param, size_t paramSize, aclrtParamHandle* paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;
    HCCL_VM_INFO(" argsHandle:{:p} paramSize:{:d}", argsHandle, paramSize);
    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    argsDetail->argsData = args->argsBuff + args->useOffset;
    argsDetail->argsDataSize = paramSize;

    memcpy(argsDetail->argsData, param, paramSize);
    args->useOffset += paramSize;
    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_INFO("argsHandle:{:p} paramHandle:{:p} ", argsHandle, *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle* paramHandle)
{
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* argsDetail = new sim::FuncArgsDetail;
    argsDetail->isHold = true;

    args->argDetail.push_back(argsDetail);
    *paramHandle = (aclrtParamHandle)argsDetail;
    HCCL_VM_INFO("paramHandle:{:p} ", *paramHandle);
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(
    aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void** bufferAddr)
{
    HCCL_VM_INFO("argsHandle:{:p} ParamHandle:{:p} dataSize:{:d}", argsHandle, paramHandle, dataSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    detail->argsData = args->argsBuff + args->useOffset;
    detail->argsDataSize = dataSize;
    *bufferAddr = reinterpret_cast<void*>(detail->argsData);
    HCCL_VM_INFO("paramHandle:{:p} ", *bufferAddr);
    return ACL_SUCCESS;
}

aclError
aclrtKernelArgsParaUpdate(aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, void* param, size_t paramSize)
{
    HCCL_VM_INFO("argsHandle:{:p} ParamHandle:{:p} paramSize:{:d}", argsHandle, paramHandle, paramSize);
    sim::FuncArgs* args = (sim::FuncArgs*)(uintptr_t)argsHandle;

    sim::FuncArgsDetail* detail = (sim::FuncArgsDetail*)(uintptr_t)paramHandle;
    if (detail->isHold || detail->argsDataSize != paramSize) {
        HCCL_VM_ERROR("invalid param handle type:{:d}", detail->isHold);
        return ACL_ERROR_INTERNAL_ERROR;
    }

    memcpy(detail->argsData, param, paramSize);
    HCCL_VM_INFO("argsData:{:p} ", reinterpret_cast<void*>(detail->argsData));
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle)
{
    (void)argsHandle;
    HCCL_VM_WARN("is empty.");
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernel(
    aclrtFuncHandle funcHandle, uint32_t blockDim, const void* argsData, size_t argsSize, aclrtStream stream)
{
    (void)funcHandle;
    (void)blockDim;
    (void)argsData;
    (void)argsSize;
    (void)stream;
    HCCL_VM_WARN("is empty.");

    return ACL_SUCCESS;
}

// 检查展开模式退化至AICPU模式
void CheckExpansionModeDegradeToAICPU()
{
    // 避免同一轮次多次kernel下发重复更新展开模式
    int curMode = sim::QueryLatestOpExpansionMode();
    if (curMode == static_cast<int>(sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU)) {
        return;
    }

    // CCU/AIV退化为AICPU模式时，更新模型中展开模式为AICPU
    const char* expanEnv = std::getenv("HCCL_OP_EXPANSION_MODE");
    std::string expanMode = expanEnv == nullptr ? "" : std::string(expanEnv);
    bool ccuEnabled = expanMode == "CCU_SCHED" || expanMode == "CCU_MS";
    bool aivEnabled = expanMode == "AIV";
    if (ccuEnabled || aivEnabled) {
        HCCL_VM_INFO("Switch the expansion mode[{} -> AICPU].", aivEnabled ? "AIV" : "CCU");
        constexpr uint8_t aicpuMode = static_cast<uint8_t>(sim::SimOpExpansionMode::SIM_OP_EXPANSION_MODE_AICPU);
        sim::UpdateOpExpansionMode(aicpuMode);
    }
}

void TryLaunchAicpuDevProcForRank(int32_t rankId, uint32_t deviceKey)
{
    auto& procMgr = sim::GetAicpuProcMgr();
    if (procMgr.IsAlive()) {
        return;
    }

    auto config = sim::CreateAicpuDeviceConfig(rankId, deviceKey);
    if (procMgr.CreateProcess(config) != 0) {
        HCCL_VM_ERROR("failed to create device process.");
        exit(EXIT_FAILURE);
    }

    g_devicePid = procMgr.GetPid();
    HCCL_VM_INFO("device process for rankId[{}] launched, pid={}", rankId, g_devicePid);
}

void LaunchAICPUKernelFunc(const std::string& kernelName, const std::string& soName, aclrtArgsHandle argsHandle)
{
    uint32_t rankId = (uint32_t)sim::GetCurrRankId();
    uint32_t devKey = (uint32_t)sim::GetCurrDeviceKey();
    HCCL_VM_INFO("rankId:{}, devKey:{}, kernelName:{}.", rankId, devKey, kernelName);

    // 检查退化逻辑并在退化场景启动device进程
    CheckExpansionModeDegradeToAICPU();
    TryLaunchAicpuDevProcForRank(rankId, devKey);

    sim::FuncArgs* args = (sim::FuncArgs*)argsHandle;
    uint64_t size = args->useOffset;
    void* ptr = nullptr;
    aclrtMalloc(&ptr, size, aclrtMemMallocPolicy::ACL_MEM_MALLOC_NORMAL_ONLY);
    if (ptr == nullptr) {
        HCCL_VM_ERROR("malloc device memory failed, size:{}", size);
        return;
    }

    aclrtMemcpy(ptr, size, args->argsBuff, size, aclrtMemcpyKind::ACL_MEMCPY_HOST_TO_DEVICE);

    // 填充Kernel执行所需参数通知Device侧执行，Proxy进入等待状态
    ExecKernelPayload payload{};
    memset(&payload, 0, sizeof(payload));
    strncpy(payload.kernelName, kernelName.c_str(), sizeof(payload.kernelName) - 1);
    strncpy(payload.soName, soName.c_str(), sizeof(payload.soName) - 1);
    payload.args = reinterpret_cast<uint64_t>(ptr);

    uint8_t rspCmd;
    RspExecKernelPayload rspPayload{};
    uint32_t rspLen = 0;
    if (sim::GetAicpuProcMgr().Request(
            PIPE_CMD_EXEC_KERNEL, &payload, sizeof(payload), rspCmd, &rspPayload, sizeof(rspPayload), rspLen)
        != 0) {
        HCCL_VM_ERROR("Request EXEC_KERNEL failed.");
        return;
    }

    if (rspCmd != PIPE_RSP_EXEC_KERNEL) {
        HCCL_VM_ERROR("unexpected response cmd: 0x{:02x}", rspCmd);
        return;
    }

    if (rspPayload.status != 0) {
        HCCL_VM_ERROR("kernel returned error status: {}", rspPayload.status);
    }
}

aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle funcHandle, uint32_t blockDim, aclrtStream stream, aclrtLaunchKernelCfg* cfg,
    aclrtArgsHandle argsHandle, void* reserve)
{
    (void)blockDim;
    (void)cfg;
    (void)reserve;
    sim::FuncHandle* func = (sim::FuncHandle*)(uintptr_t)funcHandle;
    HCCL_VM_INFO(
        "funcName:{}, kernelName:{}, func:{:p} stream:{:p} args:{:p}", func->funcName, func->kernelName,
        (void*)funcHandle, (void*)stream, (void*)argsHandle);
    if (argsHandle == nullptr || stream == nullptr) {
        HCCL_VM_ERROR("invalid input argsHandle or stream");
        return ACL_ERROR_INVALID_PARAM;
    }

    // AICPU或CCU退化为AICPU模式时调用
    LaunchAICPUKernelFunc(func->kernelName, func->soName, argsHandle);
    HCCL_VM_INFO("kernel:{}[{}] execute finished.", func->kernelName, func->soName);

    return ACL_SUCCESS;
}

#ifdef __cplusplus
}
#endif // __cplusplus

// ===== AIV virtual-kernel support scope begin =====
// AIV作用范围：这里保留HCCL AIV ExecuteKernelLaunch的C++ hook链路，并在真实
// aclrtLaunchKernelWithHostArgs launch点记录AIV_GRAPH任务、分配launchIndex、执行x86 AIV stub。
extern "C" bool GetPhyMemBlockByVirPtr(void* virPtr, uint32_t& offset, sim::PhyMemBlock& phyMem);

namespace {
constexpr uint32_t INVALID_AIV_LAUNCH_INDEX = std::numeric_limits<uint32_t>::max();
std::atomic<uint32_t> g_aivLaunchIndex{0};
thread_local uint32_t g_currentAivLaunchIndex = INVALID_AIV_LAUNCH_INDEX;
thread_local void* g_currentAivStream = nullptr;
thread_local bool g_currentAivContextActive = false;
} // namespace

namespace ops_hccl {
constexpr uint32_t AIV_STUB_MAX_RANK_SIZE = 512;
constexpr uint32_t AIV_STUB_MAX_RANK_SIZE_V = 256;
constexpr uint32_t AIV_STUB_MAX_NUM_BLOCKS = 48;
constexpr int32_t AIV_STUB_TOPO_LEN = AIV_STUB_MAX_RANK_SIZE;
constexpr uint64_t AIV_STUB_GM_IN_TABLE_OFFSET = AivCommInfoLayout::GM_IN_TABLE_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_TABLE_OFFSET = AivCommInfoLayout::GM_OUT_TABLE_OFFSET;
constexpr uint64_t AIV_STUB_TOPO_OFFSET = 32 * 1024;
constexpr uint64_t AIV_STUB_FLAG1_OFFSET = AivCommInfoLayout::FLAG1_OFFSET;
constexpr uint64_t AIV_STUB_FLAG2_OFFSET = AivCommInfoLayout::FLAG2_OFFSET;
constexpr uint64_t AIV_STUB_TAG_CLEAR_OFFSET = AivCommInfoLayout::TAG_OFFSET;
constexpr uint64_t AIV_STUB_BASE_FLAG_OFFSET = AivCommInfoLayout::BASE_FLAG_OFFSET;
constexpr uint64_t AIV_STUB_FLAG_EMPTY_OFFSET = AivCommInfoLayout::EMPTY_CLEAR_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_PING_OFFSET = AivCommInfoLayout::PING_OFFSET;
constexpr uint64_t AIV_STUB_GM_OUT_PONG_OFFSET = AivCommInfoLayout::PONG_OFFSET;
constexpr uint64_t AIV_STUB_COMM_INFO_SIZE = AivCommInfoLayout::SIZE_BYTES;
constexpr uint64_t AIV_STUB_UB_ALIGN_SIZE = 32;
constexpr uint64_t AIV_STUB_FLAG_SLOT_SIZE = 128;
constexpr uint32_t AIV_STUB_FLAG_SLOT_PRINT_NUM = 16;
constexpr uint32_t AIV_STUB_TAG_PRINT_NUM = 16;

enum class KernelArgsType { ARGS_TYPE_SERVER = 0, ARGS_TYPE_TWO_SHOT = 1, ARGS_TYPE_DEFAULT };

HcclCMDType g_currentAivCmdType = HcclCMDType::HCCL_CMD_MAX;
KernelArgsType g_currentAivArgsType = KernelArgsType::ARGS_TYPE_SERVER;

struct ExtraArgs {
    uint64_t sendCounts[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t sendDispls[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t recvCounts[AIV_STUB_MAX_RANK_SIZE_V] = {};
    uint64_t recvDispls[AIV_STUB_MAX_RANK_SIZE_V] = {};
};

struct OpCounterInfo {
    uint64_t headCountMem = 0;
    uint64_t tailCountMem = 0;
    uint64_t addOneMem = 0;
    uint32_t memSize = 0;
    bool isEnableCounter = false;
};

struct AivOpArgs {
    HcclCMDType cmdType = HcclCMDType::HCCL_CMD_MAX;
    std::string comm = {};
    HcclComm hcclComm = nullptr;
    uint32_t numBlocks = AIV_STUB_MAX_NUM_BLOCKS;
    void* stream = nullptr;
    uint64_t beginTime = 0;
    OpCounterInfo counter = {};
    void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t count = 0;
    HcclDataType dataType = HcclDataType::HCCL_DATA_TYPE_INT32;
    HcclReduceOp op = HcclReduceOp::HCCL_REDUCE_SUM;
    uint32_t root = 0;
    uint32_t sliceId = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    bool isOpBase = false;
    ExtraArgs extraArgs = {};
    uint64_t topo_[AIV_STUB_TOPO_LEN] = {0};
    KernelArgsType argsType = KernelArgsType::ARGS_TYPE_SERVER;
};

struct AivKernelArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
};

struct AivExtraKernelArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    ExtraArgs extraArgs = {};
};

struct AivHostLaunchArgs {
    const void* buffersIn = nullptr;
    uint64_t input = 0;
    uint64_t output = 0;
    uint32_t rank = 0;
    uint32_t sendRecvRemoteRank = 0;
    uint32_t rankSize = 0;
    uint64_t xRankSize = 0;
    uint64_t yRankSize = 0;
    uint64_t zRankSize = 0;
    uint64_t len = 0;
    uint32_t dataType = 0;
    uint32_t reduceOp = 0;
    uint32_t root = 0;
    uint32_t tag = 0;
    uint64_t inputSliceStride = 0;
    uint64_t outputSliceStride = 0;
    uint64_t repeatNum = 0;
    uint64_t inputRepeatStride = 0;
    uint64_t outputRepeatStride = 0;
    uint32_t numBlocks = 0;
    bool isOpBase = false;
    const void* headCountMem = nullptr;
    const void* tailCountMem = nullptr;
    const void* addOneMem = nullptr;
    uint32_t counterMemSize = 0;
    bool isEnableCounter = false;
    ExtraArgs extraArgs = {};
    bool hasExtraArgs = false;
};

struct CheckerFuncHandleView {
    std::string funcName{};
    std::string kernelName{};
};

using AivOpKernelFunc = void (*)(
    uint8_t* buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
    uint64_t xRankSize, uint64_t yRankSize, uint64_t zRankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp,
    uint32_t root, uint32_t sliceId, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum,
    uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint32_t numBlocks, bool isOpBase, uint8_t* headCountMem,
    uint8_t* tailCountMem, uint8_t* addOneMem, uint32_t counterMemSize, bool isEnableCounter);
using AivExtraOpKernelFunc = void (*)(
    uint8_t* buffIn, uint64_t input, uint64_t output, uint32_t rank, uint32_t sendRecvRemoteRank, uint32_t rankSize,
    uint64_t xRankSize, uint64_t yRankSize, uint64_t zRankSize, uint64_t len, uint32_t dataType, uint32_t reduceOp,
    uint32_t root, uint32_t sliceId, uint64_t inputSliceStride, uint64_t outputSliceStride, uint64_t repeatNum,
    uint64_t inputRepeatStride, uint64_t outputRepeatStride, uint32_t numBlocks, bool isOpBase, uint8_t* headCountMem,
    uint8_t* tailCountMem, uint8_t* addOneMem, uint32_t counterMemSize, bool isEnableCounter, ExtraArgs extraArgs);
using AivEnvInitFunc = void (*)(
    uint32_t rankId, size_t blockNum, const void* buffIn, uint32_t rankSize, uint64_t input, uint64_t inputSize,
    uint64_t output, uint64_t outputSize, uint64_t inputGlobalOffsetBase, uint64_t outputGlobalOffsetBase,
    uint64_t cclBufferSize, uint64_t aivCommInfoSize, AivSim::AivOpParam opParam);
using AivSetBlockIdxFunc = void (*)(int64_t blockIdx);
using AivDumpTasksFunc = void (*)(uint32_t launchIndex);

constexpr const char* AIV_STUB_ENV_INIT_SYMBOL = "aiv_env_init";
constexpr const char* AIV_STUB_SET_BLOCK_IDX_SYMBOL = "aiv_set_block_idx";
constexpr const char* AIV_STUB_DUMP_TASKS_SYMBOL = "aiv_dump_tasks";
constexpr const char* AIV_STUB_SO_NAME = "libhccl_aiv_kernel.so";
constexpr uint64_t INVALID_MEMORY_LAYOUT_SIZE = static_cast<uint64_t>(-1);

struct ResolvedHostPtrHandle {
    const uint8_t* hostPtr = nullptr;
    sim::PhyMemBlock phyMem = {};
    bool needRelease = false;
};

struct ResolvedKernelLaunchArgs {
    AivHostLaunchArgs args = {};
    ResolvedHostPtrHandle buffersInHandle = {};
};

struct OpMemInfoMatchInfo {
    uint64_t baseAddr = 0;
    uint64_t offsetInLayout = 0;
    uint64_t totalSize = 0;
};

struct CclOpMemInfoCacheEntry {
    uint64_t cclAddr = 0;
    uint64_t cclSize = 0;
    uint32_t opMemId = 0;
    uint32_t opDetailId = 0;
    bool valid = false;
};

struct CclOpMemInfoCache {
    bool initialized = false;
    uint32_t rankSize = 0;
    std::vector<CclOpMemInfoCacheEntry> entries;
};

static CclOpMemInfoCache g_cclOpMemInfoCache;

static ResolvedHostPtrHandle ResolveHostPtr(const void* devPtr)
{
    ResolvedHostPtrHandle handle{};
    if (devPtr == nullptr) {
        return handle;
    }

    sim::PhyMemBlock phyMem{};
    uint32_t offset = 0;
    if (!::GetPhyMemBlockByVirPtr(const_cast<void*>(devPtr), offset, phyMem)) {
        return handle;
    }

    handle.phyMem = phyMem;
    auto* devStartPtr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(devPtr) - offset);
    auto* hostBasePtr
        = static_cast<const uint8_t*>(sim::DeviceMemoryManager::GetInstance().GetHostPtrByDevPtr(devStartPtr));
    if (hostBasePtr == nullptr) {
        hostBasePtr = static_cast<const uint8_t*>(
            sim::DeviceMemoryManager::GetInstance().AcquirePhyMem(phyMem.name, phyMem.device_id, phyMem.size));
        if (hostBasePtr == nullptr) {
            return handle;
        }
        handle.phyMem = phyMem;
        handle.needRelease = true;
    }
    handle.hostPtr = hostBasePtr + offset;
    return handle;
}

static void ReleaseHostPtr(ResolvedHostPtrHandle& handle)
{
    if (!handle.needRelease) {
        return;
    }

    // needRelease 仅在走 AcquirePhyMem 时置位；按 size 判定是否走了复用区（与申请同一判据）
    if (sim::CommPoolPolicy::ShouldRedirect(handle.phyMem.size, sim::IsCheckOnlyMode())) {
        sim::MemoryManager::GetInstance().ReleaseMemByName(sim::CommPoolPolicy::kPoolName);
    } else {
        sim::DeviceMemoryManager::GetInstance().ReleasePhyMem(handle.phyMem.name, handle.phyMem.device_id);
    }
    handle.hostPtr = nullptr;
    handle.phyMem = {};
    handle.needRelease = false;
}

static void DumpAivExtraArgs(const ExtraArgs& extraArgs)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] extraArgs:\n";
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.sendCounts[" << i << "] = " << static_cast<unsigned long long>(extraArgs.sendCounts[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.sendDispls[" << i << "] = " << static_cast<unsigned long long>(extraArgs.sendDispls[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.recvCounts[" << i << "] = " << static_cast<unsigned long long>(extraArgs.recvCounts[i])
            << '\n';
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "  extraArgs.recvDispls[" << i << "] = " << static_cast<unsigned long long>(extraArgs.recvDispls[i])
            << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpAivTopo(const uint64_t topo[AIV_STUB_TOPO_LEN])
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] topo:\n";
    for (uint32_t i = 0; i < static_cast<uint32_t>(AIV_STUB_TOPO_LEN); ++i) {
        oss << "  opArgs.topo_[" << i << "] = " << static_cast<unsigned long long>(topo[i]) << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpFlagSlots(
    std::ostringstream& oss, const uint8_t* flagBase, uint64_t byteOffset, uint32_t slotCount, const char* label)
{
    const uint64_t slotBase = byteOffset / AIV_STUB_UB_ALIGN_SIZE;
    const uint64_t slotStride = AIV_STUB_FLAG_SLOT_SIZE / AIV_STUB_UB_ALIGN_SIZE;
    oss << "    " << label << " byteOffset=0x" << std::hex << static_cast<unsigned long long>(byteOffset) << std::dec
        << " slotBase=" << static_cast<unsigned long long>(slotBase)
        << " slotStride=" << static_cast<unsigned long long>(slotStride) << " slotCount=" << slotCount << '\n';
    for (uint32_t i = 0; i < slotCount; ++i) {
        const uint64_t slotIndex = slotBase + static_cast<uint64_t>(i) * slotStride;
        const auto* slotHead = reinterpret_cast<const int32_t*>(
            flagBase + byteOffset + static_cast<uint64_t>(i) * AIV_STUB_FLAG_SLOT_SIZE);
        oss << "      slot[" << static_cast<unsigned long long>(slotIndex)
            << "] addr=" << static_cast<const void*>(slotHead) << " words=[" << slotHead[0] << ", " << slotHead[1]
            << ", " << slotHead[2] << ", " << slotHead[3] << "]\n";
    }
}

static void DumpBuffersInParsedDeviceView(const void* buffersInDev, uint32_t rankSize, uint32_t numBlocks)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-ExecuteKernelLaunch] buffersIn-parse:\n";
    oss << "  [buffersIn-parse] aivCommInfoPtr base(dev) = " << buffersInDev << '\n';
    if (buffersInDev == nullptr) {
        oss << "  [buffersIn-parse] buffersIn is null, skip device-side parse dump.\n";
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    ResolvedHostPtrHandle buffersInHandle = ResolveHostPtr(buffersInDev);
    auto* buffersInHost = buffersInHandle.hostPtr;
    oss << "  [buffersIn-parse] aivCommInfoPtr base(host) = " << static_cast<const void*>(buffersInHost) << '\n';
    if (buffersInHost == nullptr) {
        oss << "  [buffersIn-parse] failed to translate device ptr to host ptr.\n";
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    const uint32_t parsedRankSize = (rankSize < AIV_STUB_MAX_RANK_SIZE) ? rankSize : AIV_STUB_MAX_RANK_SIZE;
    const auto* gmInTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_GM_IN_TABLE_OFFSET);
    const auto* gmOutTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_GM_OUT_TABLE_OFFSET);
    const auto* topoTable = reinterpret_cast<const uint64_t*>(buffersInHost + AIV_STUB_TOPO_OFFSET);
    const auto* flag1Base = buffersInHost + AIV_STUB_FLAG1_OFFSET;
    const auto* tagTable = reinterpret_cast<const int32_t*>(buffersInHost + AIV_STUB_TAG_CLEAR_OFFSET);
    const auto* emptyClearTable = reinterpret_cast<const int32_t*>(buffersInHost + AIV_STUB_FLAG_EMPTY_OFFSET);
    const uint64_t nonPingpongBaseFlagOffset = AIV_STUB_BASE_FLAG_OFFSET - AIV_STUB_FLAG1_OFFSET;

    oss << "  [buffersIn-parse] device-side parsed results from buffersIn (not a raw buffersIn pointer dump):\n";
    oss << "    AIV comm layout: commInfoSize=0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_COMM_INFO_SIZE) << ", GM_OUT_TABLE=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_TABLE_OFFSET) << ", TOPO=0x"
        << static_cast<unsigned long long>(AIV_STUB_TOPO_OFFSET) << ", TAG/CLEAR=0x"
        << static_cast<unsigned long long>(AIV_STUB_TAG_CLEAR_OFFSET) << ", FLAG1=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET) << ", FLAG2=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG2_OFFSET) << ", BASE_FLAG=0x"
        << static_cast<unsigned long long>(AIV_STUB_BASE_FLAG_OFFSET) << ", EMPTY_CLEAR=0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET) << ", PING=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PING_OFFSET) << ", PONG=0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PONG_OFFSET) << std::dec << '\n';
    oss << "    Checker AIV mode models the complete per-rank aivCommInfo region.\n";
    oss << "    GM_IN parsed entries count=" << parsedRankSize << " from +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_GM_IN_TABLE_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < parsedRankSize; ++i) {
        oss << "      GM_IN[" << i << "] dev=0x" << std::hex << static_cast<unsigned long long>(gmInTable[i])
            << std::dec << '\n';
    }
    if (parsedRankSize == 0) {
        oss << "      rankSize is 0, device would not populate GM_IN[].\n";
    }

    oss << "    GM_OUT parsed entries count=" << parsedRankSize << " from (+0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_TABLE_OFFSET) << " table) + FLAG1_OFFSET(0x"
        << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET) << ')' << std::dec << '\n';
    for (uint32_t i = 0; i < parsedRankSize; ++i) {
        const uint64_t commInfoDev = gmOutTable[i];
        const uint64_t flagDev = (commInfoDev == 0) ? 0 : (commInfoDev + AIV_STUB_FLAG1_OFFSET);
        oss << "      GM_OUT[" << i << "] dev=0x" << std::hex << static_cast<unsigned long long>(flagDev)
            << " (src commInfoDev=0x" << std::hex << static_cast<unsigned long long>(commInfoDev) << std::dec << ")\n";
    }
    if (parsedRankSize == 0) {
        oss << "      rankSize is 0, device would not populate GM_OUT[].\n";
    }

    oss << "    TOPO_ parsed entries [0.." << static_cast<unsigned>(AIV_STUB_TOPO_LEN - 1) << "] from +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_TOPO_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < static_cast<uint32_t>(AIV_STUB_TOPO_LEN); ++i) {
        oss << "      TOPO_[" << i << "] = " << static_cast<unsigned long long>(topoTable[i]) << '\n';
    }

    const uint32_t barrierSlotPrintNum = (rankSize < AIV_STUB_MAX_RANK_SIZE) ? rankSize : AIV_STUB_MAX_RANK_SIZE;
    oss << "  [buffersIn-parse] follow-up work areas derived from the same base:\n";
    oss << "    FLAG1 non-pingpong GM_OUT @ +0x" << std::hex << static_cast<unsigned long long>(AIV_STUB_FLAG1_OFFSET)
        << ", flagSlotSize=" << std::dec << AIV_STUB_FLAG_SLOT_SIZE << ", baseFlagOffset relative to GM_OUT=0x"
        << std::hex << static_cast<unsigned long long>(nonPingpongBaseFlagOffset) << ", absoluteBaseFlag=0x"
        << static_cast<unsigned long long>(AIV_STUB_BASE_FLAG_OFFSET) << std::dec << '\n';
    oss << "    FLAG2 pingpong alt @ +0x" << std::hex << static_cast<unsigned long long>(AIV_STUB_FLAG2_OFFSET)
        << ", PING data @ +0x" << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PING_OFFSET) << ", PONG data @ +0x"
        << static_cast<unsigned long long>(AIV_STUB_GM_OUT_PONG_OFFSET) << std::dec << '\n';
    oss << "    FLAG1 bytes=[0x0, 0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET - AIV_STUB_FLAG1_OFFSET) << ')'
        << ", flagSlotSize=" << std::dec << AIV_STUB_FLAG_SLOT_SIZE << std::dec << '\n';
    DumpFlagSlots(oss, flag1Base, 0, AIV_STUB_FLAG_SLOT_PRINT_NUM, "FLAG1 operator slots[0..15]");
    DumpFlagSlots(
        oss, flag1Base, nonPingpongBaseFlagOffset, barrierSlotPrintNum,
        "BASE_FLAG_OFFSET - FLAG1_OFFSET barrier slots");

    oss << "    TAG/CLEAR ints[0.." << (AIV_STUB_TAG_PRINT_NUM - 1) << "] @ +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_TAG_CLEAR_OFFSET) << std::dec << '\n';
    oss << "      Checker AIV mode keeps tag_ fixed at 1; ping-pong kernels therefore select FLAG2 and PONG.\n";
    for (uint32_t i = 0; i < AIV_STUB_TAG_PRINT_NUM; ++i) {
        oss << "      TAG_CLEAR[" << i << "] = " << tagTable[i] << '\n';
    }
    oss << "    EMPTY_CLEAR ints[0.." << (AIV_STUB_TAG_PRINT_NUM - 1) << "] @ +0x" << std::hex
        << static_cast<unsigned long long>(AIV_STUB_FLAG_EMPTY_OFFSET) << std::dec << '\n';
    for (uint32_t i = 0; i < AIV_STUB_TAG_PRINT_NUM; ++i) {
        oss << "      EMPTY_CLEAR[" << i << "] = " << emptyClearTable[i] << '\n';
    }

    HCCL_VM_DEBUG("{}", oss.str());
    ReleaseHostPtr(buffersInHandle);
}

static void DumpLaunchKernelCfg(const aclrtLaunchKernelCfg* cfg)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] cfg:\n";
    oss << "  cfg = " << cfg << '\n';
    if (cfg == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    oss << "  cfg->attrs = " << cfg->attrs << '\n';
    oss << "  cfg->numAttrs = " << cfg->numAttrs << '\n';
    for (size_t i = 0; i < cfg->numAttrs; ++i) {
        const auto& attr = cfg->attrs[i];
        oss << "    cfg->attrs[" << i << "].id = " << static_cast<int>(attr.id) << '\n';
        switch (attr.id) {
            case ACL_RT_LAUNCH_KERNEL_ATTR_SCHEM_MODE:
                oss << "      schemMode = " << static_cast<unsigned>(attr.value.schemMode) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_DYN_UBUF_SIZE:
                oss << "      dynUBufSize = " << attr.value.dynUBufSize << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_ENGINE_TYPE:
                oss << "      engineType = " << static_cast<unsigned>(attr.value.engineType) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_BLOCKDIM_OFFSET:
                oss << "      blockDimOffset = " << attr.value.blockDimOffset << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_BLOCK_TASK_PREFETCH:
                oss << "      isBlockTaskPrefetch = " << static_cast<unsigned>(attr.value.isBlockTaskPrefetch) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_DATA_DUMP:
                oss << "      isDataDump = " << static_cast<unsigned>(attr.value.isDataDump) << '\n';
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT:
                oss << "      timeout = " << static_cast<unsigned>(attr.value.timeout) << "(s)\n";
                break;
            case ACL_RT_LAUNCH_KERNEL_ATTR_TIMEOUT_US:
                oss << "      timeoutUs.low = " << attr.value.timeoutUs.timeoutLow << '\n';
                oss << "      timeoutUs.high = " << attr.value.timeoutUs.timeoutHigh << '\n';
                break;
            default:
                oss << "      raw rsv = [" << attr.value.rsv[0] << ", " << attr.value.rsv[1] << ", "
                    << attr.value.rsv[2] << ", " << attr.value.rsv[3] << "]\n";
                break;
        }
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static void DumpPlaceHolderArray(const aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] placeHolderArray:\n";
    oss << "  placeHolderArray = " << placeHolderArray << '\n';
    oss << "  placeHolderNum = " << placeHolderNum << '\n';
    if (placeHolderArray == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    for (size_t i = 0; i < placeHolderNum; ++i) {
        oss << "    placeHolderArray[" << i << "].addrOffset = " << placeHolderArray[i].addrOffset << '\n';
        oss << "    placeHolderArray[" << i << "].dataOffset = " << placeHolderArray[i].dataOffset << '\n';
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

static bool IsAivExtraArgsCmdType(HcclCMDType cmdType) { return cmdType == HcclCMDType::HCCL_CMD_ALLTOALLV; }

static void CopyCommonKernelArgsToHostArgs(const AivKernelArgs& src, AivHostLaunchArgs& dst)
{
    dst.buffersIn = src.buffersIn;
    dst.input = src.input;
    dst.output = src.output;
    dst.rank = src.rank;
    dst.sendRecvRemoteRank = src.sendRecvRemoteRank;
    dst.rankSize = src.rankSize;
    dst.xRankSize = src.xRankSize;
    dst.yRankSize = src.yRankSize;
    dst.zRankSize = src.zRankSize;
    dst.len = src.len;
    dst.dataType = src.dataType;
    dst.reduceOp = src.reduceOp;
    dst.root = src.root;
    dst.tag = src.tag;
    dst.inputSliceStride = src.inputSliceStride;
    dst.outputSliceStride = src.outputSliceStride;
    dst.repeatNum = src.repeatNum;
    dst.inputRepeatStride = src.inputRepeatStride;
    dst.outputRepeatStride = src.outputRepeatStride;
    dst.numBlocks = src.numBlocks;
    dst.isOpBase = src.isOpBase;
    dst.headCountMem = src.headCountMem;
    dst.tailCountMem = src.tailCountMem;
    dst.addOneMem = src.addOneMem;
    dst.counterMemSize = src.counterMemSize;
    dst.isEnableCounter = src.isEnableCounter;
}

static void CopyCommonKernelArgsToHostArgs(const AivExtraKernelArgs& src, AivHostLaunchArgs& dst)
{
    dst.buffersIn = src.buffersIn;
    dst.input = src.input;
    dst.output = src.output;
    dst.rank = src.rank;
    dst.sendRecvRemoteRank = src.sendRecvRemoteRank;
    dst.rankSize = src.rankSize;
    dst.xRankSize = src.xRankSize;
    dst.yRankSize = src.yRankSize;
    dst.zRankSize = src.zRankSize;
    dst.len = src.len;
    dst.dataType = src.dataType;
    dst.reduceOp = src.reduceOp;
    dst.root = src.root;
    dst.tag = src.tag;
    dst.inputSliceStride = src.inputSliceStride;
    dst.outputSliceStride = src.outputSliceStride;
    dst.repeatNum = src.repeatNum;
    dst.inputRepeatStride = src.inputRepeatStride;
    dst.outputRepeatStride = src.outputRepeatStride;
    dst.numBlocks = src.numBlocks;
    dst.isOpBase = src.isOpBase;
    dst.headCountMem = src.headCountMem;
    dst.tailCountMem = src.tailCountMem;
    dst.addOneMem = src.addOneMem;
    dst.counterMemSize = src.counterMemSize;
    dst.isEnableCounter = src.isEnableCounter;
}

static bool
ParseAivHostLaunchArgs(const void* hostArgs, size_t argsSize, HcclCMDType cmdType, AivHostLaunchArgs& parsedArgs)
{
    parsedArgs = {};
    if (hostArgs == nullptr) {
        HCCL_VM_ERROR("hostArgs is null, can not virtual execute kernel.");
        return false;
    }

    if (IsAivExtraArgsCmdType(cmdType)) {
        if (argsSize < sizeof(AivExtraKernelArgs)) {
            HCCL_VM_ERROR(
                "argsSize({}) < sizeof(AivExtraKernelArgs)({}) "
                "for cmdType={}, stop virtual execution.",
                argsSize, sizeof(AivExtraKernelArgs), static_cast<int>(cmdType));
            return false;
        }
        const auto& rawArgs = *static_cast<const AivExtraKernelArgs*>(hostArgs);
        CopyCommonKernelArgsToHostArgs(rawArgs, parsedArgs);
        parsedArgs.extraArgs = rawArgs.extraArgs;
        parsedArgs.hasExtraArgs = true;
        return true;
    }

    if (argsSize < sizeof(AivKernelArgs)) {
        HCCL_VM_ERROR(
            "argsSize({}) < sizeof(AivKernelArgs)({}) "
            "for cmdType={}, stop virtual execution.",
            argsSize, sizeof(AivKernelArgs), static_cast<int>(cmdType));
        return false;
    }
    const auto& rawArgs = *static_cast<const AivKernelArgs*>(hostArgs);
    CopyCommonKernelArgsToHostArgs(rawArgs, parsedArgs);
    parsedArgs.hasExtraArgs = false;
    return true;
}

static void DumpHostLaunchArgs(const AivHostLaunchArgs* hostArgs, size_t argsSize)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] hostArgs:\n";
    oss << "  hostArgs = " << hostArgs << '\n';
    oss << "  argsSize = " << argsSize << '\n';
    oss << "  sizeof(AivKernelArgs) = " << sizeof(AivKernelArgs) << '\n';
    oss << "  sizeof(AivExtraKernelArgs) = " << sizeof(AivExtraKernelArgs) << '\n';
    oss << "  sizeof(AivHostLaunchArgs normalized) = " << sizeof(AivHostLaunchArgs) << '\n';
    if (hostArgs == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return;
    }

    oss << "  hostArgs.buffersIn = " << hostArgs->buffersIn << '\n';
    oss << "  hostArgs.input = 0x" << std::hex << static_cast<unsigned long long>(hostArgs->input) << std::dec << '\n';
    oss << "  hostArgs.output = 0x" << std::hex << static_cast<unsigned long long>(hostArgs->output) << std::dec
        << '\n';
    oss << "  hostArgs.rank = " << hostArgs->rank << '\n';
    oss << "  hostArgs.sendRecvRemoteRank = " << hostArgs->sendRecvRemoteRank << '\n';
    oss << "  hostArgs.rankSize = " << hostArgs->rankSize << '\n';
    oss << "  hostArgs.xRankSize = " << static_cast<unsigned long long>(hostArgs->xRankSize) << '\n';
    oss << "  hostArgs.yRankSize = " << static_cast<unsigned long long>(hostArgs->yRankSize) << '\n';
    oss << "  hostArgs.zRankSize = " << static_cast<unsigned long long>(hostArgs->zRankSize) << '\n';
    oss << "  hostArgs.len = " << static_cast<unsigned long long>(hostArgs->len) << '\n';
    oss << "  hostArgs.dataType = " << hostArgs->dataType << '\n';
    oss << "  hostArgs.reduceOp = " << hostArgs->reduceOp << '\n';
    oss << "  hostArgs.root = " << hostArgs->root << '\n';
    oss << "  hostArgs.tag = " << hostArgs->tag << '\n';
    oss << "  hostArgs.inputSliceStride = " << static_cast<unsigned long long>(hostArgs->inputSliceStride) << '\n';
    oss << "  hostArgs.outputSliceStride = " << static_cast<unsigned long long>(hostArgs->outputSliceStride) << '\n';
    oss << "  hostArgs.repeatNum = " << static_cast<unsigned long long>(hostArgs->repeatNum) << '\n';
    oss << "  hostArgs.inputRepeatStride = " << static_cast<unsigned long long>(hostArgs->inputRepeatStride) << '\n';
    oss << "  hostArgs.outputRepeatStride = " << static_cast<unsigned long long>(hostArgs->outputRepeatStride) << '\n';
    oss << "  hostArgs.numBlocks = " << hostArgs->numBlocks << '\n';
    oss << "  hostArgs.isOpBase = " << static_cast<int>(hostArgs->isOpBase) << '\n';
    oss << "  hostArgs.headCountMem = " << hostArgs->headCountMem << '\n';
    oss << "  hostArgs.tailCountMem = " << hostArgs->tailCountMem << '\n';
    oss << "  hostArgs.addOneMem = " << hostArgs->addOneMem << '\n';
    oss << "  hostArgs.counterMemSize = " << hostArgs->counterMemSize << '\n';
    oss << "  hostArgs.isEnableCounter = " << static_cast<int>(hostArgs->isEnableCounter) << '\n';
    oss << "  hostArgs.hasExtraArgs = " << static_cast<int>(hostArgs->hasExtraArgs) << '\n';
    HCCL_VM_DEBUG("{}", oss.str());
    if (hostArgs->hasExtraArgs) {
        DumpAivExtraArgs(hostArgs->extraArgs);
    }
}

static const char* GetAivWideKernelTypeSuffix(HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return "half";
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return "int16_t";
        case HcclDataType::HCCL_DATA_TYPE_UINT16:
            return "uint16_t";
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return "float";
        case HcclDataType::HCCL_DATA_TYPE_FP64:
            return "uint64_t";
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return "int32_t";
        case HcclDataType::HCCL_DATA_TYPE_UINT32:
            return "uint32_t";
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return "int8_t";
        case HcclDataType::HCCL_DATA_TYPE_UINT8:
            return "uint8_t";
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return "bfloat16_t";
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return "int64_t";
        case HcclDataType::HCCL_DATA_TYPE_UINT64:
            return "uint64_t";
        case HcclDataType::HCCL_DATA_TYPE_HIF8:
            return "hifloat8_t";
        case HcclDataType::HCCL_DATA_TYPE_FP8E4M3:
            return "fp8_e4m3fn_t";
        case HcclDataType::HCCL_DATA_TYPE_FP8E5M2:
            return "fp8_e5m2_t";
        case HcclDataType::HCCL_DATA_TYPE_FP8E8M0:
            return "fp8_e8m0_t";
        default:
            return nullptr;
    }
}

static const char* GetAivReduceKernelTypeSuffix(HcclDataType dataType)
{
    switch (dataType) {
        case HcclDataType::HCCL_DATA_TYPE_FP16:
            return "half";
        case HcclDataType::HCCL_DATA_TYPE_INT16:
            return "int16_t";
        case HcclDataType::HCCL_DATA_TYPE_FP32:
            return "float";
        case HcclDataType::HCCL_DATA_TYPE_INT32:
            return "int32_t";
        case HcclDataType::HCCL_DATA_TYPE_INT8:
            return "int8_t";
        case HcclDataType::HCCL_DATA_TYPE_BFP16:
            return "bfloat16_t";
        case HcclDataType::HCCL_DATA_TYPE_INT64:
            return "int64_t";
        default:
            return nullptr;
    }
}

static std::string BuildAivKernelName(const char* kernelPrefix, const char* typeSuffix)
{
    if (kernelPrefix == nullptr || typeSuffix == nullptr) {
        return {};
    }

    return std::string(kernelPrefix) + typeSuffix;
}

static bool IsUnsupportedFallbackAivCmdType(HcclCMDType cmdType)
{
    switch (cmdType) {
        case HcclCMDType::HCCL_CMD_ALLGATHER_V:
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER_V:
        case HcclCMDType::HCCL_CMD_ALLTOALLVC:
            return true;
        default:
            return false;
    }
}

static std::string GetFallbackAivKernelName(HcclCMDType cmdType, HcclDataType dataType, KernelArgsType argsType)
{
    switch (cmdType) {
        case HcclCMDType::HCCL_CMD_ALLGATHER:
            return BuildAivKernelName("aiv_all_gather_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_ALLREDUCE:
            return BuildAivKernelName(
                argsType == KernelArgsType::ARGS_TYPE_TWO_SHOT ? "aiv_allreduce_mesh1d_twoshot_" : "aiv_allreduce_",
                GetAivReduceKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_REDUCE_SCATTER:
            return BuildAivKernelName("aiv_reduce_scatter_", GetAivReduceKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_BROADCAST:
            return BuildAivKernelName("aiv_broadcast_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_REDUCE:
            return BuildAivKernelName("aiv_reduce_", GetAivReduceKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_SCATTER:
            return BuildAivKernelName("aiv_scatter_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_ALLTOALL:
            return BuildAivKernelName("aiv_alltoall_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_ALLTOALLV:
            return BuildAivKernelName("aiv_alltoallv_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_SEND:
            return BuildAivKernelName("aiv_send_", GetAivWideKernelTypeSuffix(dataType));
        case HcclCMDType::HCCL_CMD_RECEIVE:
            return BuildAivKernelName("aiv_recv_", GetAivWideKernelTypeSuffix(dataType));
        default:
            return {};
    }
}

static std::string InferKernelNameFromFuncHandle(aclrtFuncHandle funcHandle)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-aclrtLaunchKernelWithHostArgs] InferKernelNameFromFuncHandle:\n";
    oss << "  funcHandle = " << funcHandle << '\n';
    if (funcHandle == nullptr) {
        HCCL_VM_DEBUG("{}", oss.str());
        return {};
    }

    char funcNameBuffer[256] = {0};
    aclError funcNameRet = aclrtGetFunctionName(funcHandle, sizeof(funcNameBuffer), funcNameBuffer);
    oss << "  aclrtGetFunctionName(funcHandle) ret = " << static_cast<int>(funcNameRet) << '\n';
    oss << "  aclrtGetFunctionName(funcHandle) funcName = " << (funcNameBuffer[0] == '\0' ? "<empty>" : funcNameBuffer)
        << '\n';

    std::string resolvedKernelName;

    if (funcNameRet == ACL_SUCCESS) {
        const auto* funcHandleView = reinterpret_cast<const CheckerFuncHandleView*>(funcHandle);
        oss << "  [InferKernelNameFromFuncHandle] current checker aclrtBinaryGetFunction stores funcHandle as "
            << "sim::FuncHandle*.\n";
        oss << "  [InferKernelNameFromFuncHandle] funcHandle->funcName = "
            << (funcHandleView->funcName.empty() ? "<empty>" : funcHandleView->funcName.c_str()) << '\n';
        oss << "  [InferKernelNameFromFuncHandle] funcHandle->kernelName = "
            << (funcHandleView->kernelName.empty() ? "<empty>" : funcHandleView->kernelName.c_str()) << '\n';
        if (!funcHandleView->kernelName.empty()) {
            resolvedKernelName = funcHandleView->kernelName;
        } else if (!funcHandleView->funcName.empty()) {
            resolvedKernelName = funcHandleView->funcName;
        }
    }

    if (resolvedKernelName.empty()) {
        resolvedKernelName = std::string(funcNameBuffer);
    }
    oss << "  resolvedKernelName = " << (resolvedKernelName.empty() ? "<empty>" : resolvedKernelName.c_str()) << '\n';
    HCCL_VM_DEBUG("{}", oss.str());
    return resolvedKernelName;
}

static std::string GetAivLibraryPath(const std::string& soName, const std::string& kernelName)
{
    const char* kernelNameCStr = kernelName.empty() ? "<empty>" : kernelName.c_str();
    const std::string& installDir = InstallPath::GetHcclVmInstallAbsPath();
    if (installDir.empty()) {
        HCCL_VM_ERROR("install root is empty, can not locate {} for kernel {}", soName, kernelNameCStr);
        return {};
    }

    std::error_code ec;
    const fs::path soPath = fs::path(installDir) / "lib" / GetArchStr() / soName;
    if (!fs::exists(soPath, ec)) {
        if (ec) {
            HCCL_VM_ERROR(
                "failed to stat aiv library path, kernel={}, installDir={}, so={}, err={}", kernelNameCStr, installDir,
                soPath.string(), ec.message());
            return {};
        }
        HCCL_VM_ERROR(
            "missing aiv stub shared library, kernel={}, installDir={}, expectedSo={}", kernelNameCStr, installDir,
            soPath.string());
        return {};
    }

    return soPath.string();
}

static ResolvedKernelLaunchArgs PrepareResolvedKernelLaunchArgs(const AivHostLaunchArgs& rawArgs)
{
    ResolvedKernelLaunchArgs resolvedArgs{};
    resolvedArgs.args = rawArgs;
    resolvedArgs.buffersInHandle = ResolveHostPtr(rawArgs.buffersIn);
    if (resolvedArgs.buffersInHandle.hostPtr == nullptr && rawArgs.buffersIn != nullptr) {
        HCCL_VM_ERROR(
            "buffersIn translation failed, translated host ptr is null, "
            "raw={:p}",
            rawArgs.buffersIn);
    }
    resolvedArgs.args.buffersIn = resolvedArgs.buffersInHandle.hostPtr;
    return resolvedArgs;
}

enum class OpMemInfoLookupStatus { RESOLVED, MISSING, INVALID };

static bool
TryMatchOpMemInfoRange(uint64_t queryVirtualAddr, uint64_t baseAddr, uint64_t size, OpMemInfoMatchInfo& matchInfo)
{
    if (size == 0 || queryVirtualAddr < baseAddr) {
        return false;
    }

    const uint64_t offsetInLayout = queryVirtualAddr - baseAddr;
    if (offsetInLayout >= size) {
        return false;
    }

    matchInfo.baseAddr = baseAddr;
    matchInfo.offsetInLayout = offsetInLayout;
    matchInfo.totalSize = size;
    return true;
}

static bool StartsWith(const std::string& value, const char* prefix) { return value.rfind(prefix, 0) == 0; }

static bool IsAivBroadcastKernel(const std::string& kernelName) { return StartsWith(kernelName, "aiv_broadcast_"); }

static bool IsAivScatterKernel(const std::string& kernelName) { return StartsWith(kernelName, "aiv_scatter_"); }

static OpMemInfoLookupStatus LookupOpMemInfoByVirtualAddr(
    uint32_t rankId, uint64_t queryVirtualAddr, BufferType bufType, OpMemInfoMatchInfo& matchInfo)
{
    matchInfo = {};
    if (queryVirtualAddr == 0) {
        return OpMemInfoLookupStatus::RESOLVED;
    }

    sim::OpMemInfoTab opMemInfo{};
    if (sim::QueryCurrentOpMemInfoByRank(rankId, opMemInfo) != 0) {
        return OpMemInfoLookupStatus::MISSING;
    }

    uint64_t baseAddr = 0;
    uint64_t size = 0;
    if (bufType == BufferType::INPUT) {
        baseAddr = opMemInfo.inputAddr;
        size = opMemInfo.inputSize;
    } else if (bufType == BufferType::OUTPUT) {
        baseAddr = opMemInfo.outputAddr;
        size = opMemInfo.outputSize;
    } else if (bufType == BufferType::CCL) {
        baseAddr = opMemInfo.cclAddr;
        size = opMemInfo.cclSize;
    } else {
        HCCL_VM_ERROR(
            "unsupported opMemInfo buffer type, rankId={}, baseAddr=0x{:x}, bufType={}", rankId, queryVirtualAddr,
            static_cast<uint32_t>(bufType));
        return OpMemInfoLookupStatus::INVALID;
    }

    if (!TryMatchOpMemInfoRange(queryVirtualAddr, baseAddr, size, matchInfo)) {
        HCCL_VM_DEBUG(
            "queryAddr=0x{:x} did not match opMemInfo range. "
            "rankId={}, bufType={}, opMemBase=0x{:x}, opMemSize={}",
            queryVirtualAddr, rankId, static_cast<uint32_t>(bufType), baseAddr, size);
        return OpMemInfoLookupStatus::MISSING;
    }

    return OpMemInfoLookupStatus::RESOLVED;
}

static void InitCclOpMemInfoCache(uint32_t rankSize)
{
    if (g_cclOpMemInfoCache.initialized) {
        return;
    }

    g_cclOpMemInfoCache.initialized = true;
    g_cclOpMemInfoCache.rankSize = rankSize;
    g_cclOpMemInfoCache.entries.assign(rankSize, CclOpMemInfoCacheEntry{});

    for (uint32_t rankId = 0; rankId < rankSize; ++rankId) {
        sim::OpMemInfoTab opMemInfo{};
        if (sim::QueryCurrentOpMemInfoByRank(rankId, opMemInfo) != 0) {
            HCCL_VM_ERROR("failed to cache CCL opMemInfo, rankId={}", rankId);
            continue;
        }

        auto& entry = g_cclOpMemInfoCache.entries[rankId];
        entry.cclAddr = opMemInfo.cclAddr;
        entry.cclSize = opMemInfo.cclSize;
        entry.opMemId = opMemInfo.id;
        entry.opDetailId = opMemInfo.opDetailId;
        entry.valid = opMemInfo.cclAddr != 0 && opMemInfo.cclSize != 0;
        HCCL_VM_INFO(
            "cache CCL opMemInfo, rankId={}, opMemId={}, opDetailId={}, cclAddr=0x{:x}, cclSize={}", rankId,
            entry.opMemId, entry.opDetailId, entry.cclAddr, entry.cclSize);
    }
}

static OpMemInfoLookupStatus LookupCachedCclOpMemInfoByVirtualAddr(
    uint32_t rankId, uint64_t queryVirtualAddr, uint32_t rankSize, OpMemInfoMatchInfo& matchInfo)
{
    matchInfo = {};
    if (queryVirtualAddr == 0) {
        return OpMemInfoLookupStatus::RESOLVED;
    }

    InitCclOpMemInfoCache(rankSize);

    if (rankId >= g_cclOpMemInfoCache.entries.size()) {
        HCCL_VM_ERROR(
            "cached CCL opMemInfo rank out of range, rankId={}, cachedRankSize={}, currentRankSize={}", rankId,
            g_cclOpMemInfoCache.rankSize, rankSize);
        return OpMemInfoLookupStatus::INVALID;
    }

    if (g_cclOpMemInfoCache.rankSize != rankSize) {
        HCCL_VM_ERROR(
            "cached CCL opMemInfo rankSize mismatch, cachedRankSize={}, currentRankSize={}",
            g_cclOpMemInfoCache.rankSize, rankSize);
        return OpMemInfoLookupStatus::INVALID;
    }

    const auto& entry = g_cclOpMemInfoCache.entries[rankId];
    if (!entry.valid) {
        HCCL_VM_ERROR(
            "cached CCL opMemInfo is invalid, rankId={}, opMemId={}, opDetailId={}, "
            "cclAddr=0x{:x}, cclSize={}",
            rankId, entry.opMemId, entry.opDetailId, entry.cclAddr, entry.cclSize);
        return OpMemInfoLookupStatus::MISSING;
    }

    if (!TryMatchOpMemInfoRange(queryVirtualAddr, entry.cclAddr, entry.cclSize, matchInfo)) {
        HCCL_VM_DEBUG(
            "queryAddr=0x{:x} did not match cached CCL opMemInfo range. "
            "rankId={}, opMemId={}, opDetailId={}, cclAddr=0x{:x}, cclSize={}",
            queryVirtualAddr, rankId, entry.opMemId, entry.opDetailId, entry.cclAddr, entry.cclSize);
        return OpMemInfoLookupStatus::MISSING;
    }

    return OpMemInfoLookupStatus::RESOLVED;
}

static void BackfillCurrentAivOpMemCclBuffer(const ResolvedKernelLaunchArgs& resolvedArgs)
{
    if (resolvedArgs.args.buffersIn == nullptr) {
        return;
    }

    const uint32_t rank = resolvedArgs.args.rank;
    const uint32_t rankSize = resolvedArgs.args.rankSize;
    if (rankSize == 0 || rank >= rankSize) {
        HCCL_VM_WARN(
            "skip backfilling current opMem CCL buffer, invalid rank/rankSize, rank={}, rankSize={}", rank, rankSize);
        return;
    }

    const auto* ipcBufferGlobal = static_cast<const uint64_t*>(resolvedArgs.args.buffersIn);
    const uint64_t cclBufferAddr = ipcBufferGlobal[rank];
    if (cclBufferAddr == 0) {
        HCCL_VM_DEBUG("skip backfilling current opMem CCL buffer, rank={}, cclBufferAddr is 0", rank);
        return;
    }

    OpMemInfoMatchInfo cclMatchInfo{};
    const OpMemInfoLookupStatus cclStatus
        = LookupCachedCclOpMemInfoByVirtualAddr(rank, cclBufferAddr, rankSize, cclMatchInfo);
    if (cclStatus != OpMemInfoLookupStatus::RESOLVED || cclMatchInfo.baseAddr == 0 || cclMatchInfo.totalSize == 0) {
        HCCL_VM_WARN(
            "skip backfilling current opMem CCL buffer, failed to resolve cached CCL, "
            "rank={}, rankSize={}, cclBufferAddr=0x{:x}, status={}, resolvedBase=0x{:x}, resolvedSize={}",
            rank, rankSize, cclBufferAddr, static_cast<int>(cclStatus), cclMatchInfo.baseAddr, cclMatchInfo.totalSize);
        return;
    }

    if (sim::UpdateOpMemCclBuffer(cclMatchInfo.baseAddr, cclMatchInfo.totalSize) != 0) {
        HCCL_VM_ERROR(
            "failed to backfill current opMem CCL buffer, rank={}, cclAddr=0x{:x}, cclSize={}", rank,
            cclMatchInfo.baseAddr, cclMatchInfo.totalSize);
        return;
    }

    HCCL_VM_INFO(
        "backfill current opMem CCL buffer, rank={}, cclAddr=0x{:x}, cclSize={}", rank, cclMatchInfo.baseAddr,
        cclMatchInfo.totalSize);
}

static void ResolveVirtualAivBufferSizes(
    const std::string& kernelName, ResolvedKernelLaunchArgs& resolvedArgs, uint64_t& inputSize, uint64_t& outputSize,
    uint64_t& inputGlobalOffsetBase, uint64_t& outputGlobalOffsetBase, uint64_t& cclBufferSize,
    uint64_t& aivCommInfoSize)
{
    inputGlobalOffsetBase = 0;
    outputGlobalOffsetBase = 0;
    const uint64_t inputAddr = resolvedArgs.args.input;
    OpMemInfoMatchInfo inputMatchInfo{};
    const OpMemInfoLookupStatus inputStatus
        = LookupOpMemInfoByVirtualAddr(resolvedArgs.args.rank, inputAddr, BufferType::INPUT, inputMatchInfo);
    if (inputStatus == OpMemInfoLookupStatus::MISSING) {
        if (IsAivScatterKernel(kernelName) && resolvedArgs.args.rank != resolvedArgs.args.root) {
            HCCL_VM_INFO(
                "skip missing input opMemInfo for kernel {}, rank={}, root={}, baseAddr=0x{:x}", kernelName,
                resolvedArgs.args.rank, resolvedArgs.args.root, inputAddr);
            resolvedArgs.args.input = 0;
            inputSize = 0;
        } else {
            HCCL_VM_ERROR(
                "expected exactly one opMemInfo range, rankId={}, baseAddr=0x{:x}, bufType={}, matchedCount={}",
                resolvedArgs.args.rank, inputAddr, static_cast<uint32_t>(BufferType::INPUT), 0);
            inputSize = INVALID_MEMORY_LAYOUT_SIZE;
        }
    } else if (inputStatus == OpMemInfoLookupStatus::INVALID) {
        inputSize = INVALID_MEMORY_LAYOUT_SIZE;
    } else {
        inputSize = inputMatchInfo.totalSize;
        inputGlobalOffsetBase = inputMatchInfo.offsetInLayout;
    }

    const uint64_t outputAddr = resolvedArgs.args.output;
    OpMemInfoMatchInfo outputMatchInfo{};
    const OpMemInfoLookupStatus outputStatus
        = LookupOpMemInfoByVirtualAddr(resolvedArgs.args.rank, outputAddr, BufferType::OUTPUT, outputMatchInfo);
    if (outputStatus == OpMemInfoLookupStatus::MISSING) {
        if (IsAivBroadcastKernel(kernelName)) {
            HCCL_VM_INFO(
                "skip missing output opMemInfo for kernel {}, rank={}, baseAddr=0x{:x}", kernelName,
                resolvedArgs.args.rank, outputAddr);
            resolvedArgs.args.output = 0;
            outputSize = 0;
        } else {
            HCCL_VM_ERROR(
                "expected exactly one opMemInfo range, rankId={}, baseAddr=0x{:x}, bufType={}, matchedCount={}",
                resolvedArgs.args.rank, outputAddr, static_cast<uint32_t>(BufferType::OUTPUT), 0);
            outputSize = INVALID_MEMORY_LAYOUT_SIZE;
        }
    } else if (outputStatus == OpMemInfoLookupStatus::INVALID) {
        outputSize = INVALID_MEMORY_LAYOUT_SIZE;
    } else {
        outputSize = outputMatchInfo.totalSize;
        outputGlobalOffsetBase = outputMatchInfo.offsetInLayout;
    }

    cclBufferSize = 0;
    aivCommInfoSize = AIV_STUB_COMM_INFO_SIZE;

    if (resolvedArgs.args.buffersIn == nullptr) {
        return;
    }

    const auto* ipcBufferGlobal = static_cast<const uint64_t*>(resolvedArgs.args.buffersIn);
    for (uint32_t i = 0; i < resolvedArgs.args.rankSize; ++i) {
        if (cclBufferSize == 0) {
            const uint64_t cclBufferAddr = ipcBufferGlobal[i];
            OpMemInfoMatchInfo cclMatchInfo{};
            const OpMemInfoLookupStatus finalCclBufferStatus
                = LookupCachedCclOpMemInfoByVirtualAddr(i, cclBufferAddr, resolvedArgs.args.rankSize, cclMatchInfo);
            if (finalCclBufferStatus == OpMemInfoLookupStatus::MISSING) {
                HCCL_VM_ERROR(
                    "expected exactly one opMemInfo range, rankId={}, baseAddr=0x{:x}, bufType={}, matchedCount={}", i,
                    cclBufferAddr, static_cast<uint32_t>(BufferType::CCL), 0);
                cclBufferSize = INVALID_MEMORY_LAYOUT_SIZE;
            } else if (finalCclBufferStatus == OpMemInfoLookupStatus::INVALID) {
                cclBufferSize = INVALID_MEMORY_LAYOUT_SIZE;
            } else {
                cclBufferSize = cclMatchInfo.totalSize;
            }
        }
        if (cclBufferSize != 0) {
            break;
        }
    }
}

static void DumpVirtualKernelExtraArgsWithSource(std::ostringstream& oss, const ExtraArgs& extraArgs)
{
    oss << "    kernelFunc.extraArgs <- aclrtLaunchKernelWithHostArgs(hostArgs->extraArgs)\n";
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.sendCounts[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.sendCounts[i])
            << " <- hostArgs->extraArgs.sendCounts[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.sendDispls[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.sendDispls[i])
            << " <- hostArgs->extraArgs.sendDispls[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.recvCounts[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.recvCounts[i])
            << " <- hostArgs->extraArgs.recvCounts[" << i << "]\n";
    }
    for (uint32_t i = 0; i < AIV_STUB_MAX_RANK_SIZE_V; ++i) {
        oss << "      kernelFunc.extraArgs.recvDispls[" << i
            << "] = " << static_cast<unsigned long long>(extraArgs.recvDispls[i])
            << " <- hostArgs->extraArgs.recvDispls[" << i << "]\n";
    }
}

static void DumpVirtualKernelFuncArgs(
    const std::string& kernelName, uint32_t numBlocks, const AivHostLaunchArgs& rawArgs,
    const ResolvedKernelLaunchArgs& resolvedArgs)
{
    std::ostringstream oss;
    oss << "[virtual-aiv-VirtualExecuteAivKernel] kernelFunc shared args:\n";
    oss << "    launchContext.kernelName = " << kernelName << " <- aclrtLaunchKernelWithHostArgs(funcHandle)\n";
    oss << "    launchContext.numBlocks = " << numBlocks << " <- aclrtLaunchKernelWithHostArgs(numBlocks)\n";
    oss << "    kernelFunc.buffIn = " << resolvedArgs.args.buffersIn
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->buffersIn=" << rawArgs.buffersIn
        << "), checker translates only buffersIn to host address\n";
    oss << "    kernelFunc.input = 0x" << std::hex << static_cast<unsigned long long>(resolvedArgs.args.input)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->input=0x" << static_cast<unsigned long long>(rawArgs.input)
        << "), checker keeps raw input address unchanged\n"
        << std::dec;
    oss << "    kernelFunc.output = 0x" << std::hex << static_cast<unsigned long long>(resolvedArgs.args.output)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->output=0x" << static_cast<unsigned long long>(rawArgs.output)
        << "), checker keeps raw output address unchanged\n"
        << std::dec;
    oss << "    kernelFunc.rank = " << resolvedArgs.args.rank << " <- aclrtLaunchKernelWithHostArgs(hostArgs->rank)\n";
    oss << "    kernelFunc.sendRecvRemoteRank = " << resolvedArgs.args.sendRecvRemoteRank
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->sendRecvRemoteRank)\n";
    oss << "    kernelFunc.rankSize = " << resolvedArgs.args.rankSize
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->rankSize)\n";
    oss << "    kernelFunc.xRankSize = " << static_cast<unsigned long long>(resolvedArgs.args.xRankSize)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->xRankSize)\n";
    oss << "    kernelFunc.yRankSize = " << static_cast<unsigned long long>(resolvedArgs.args.yRankSize)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->yRankSize)\n";
    oss << "    kernelFunc.zRankSize = " << static_cast<unsigned long long>(resolvedArgs.args.zRankSize)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->zRankSize)\n";
    oss << "    kernelFunc.len = " << static_cast<unsigned long long>(resolvedArgs.args.len)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->len)\n";
    oss << "    kernelFunc.dataType = " << resolvedArgs.args.dataType
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->dataType)\n";
    oss << "    kernelFunc.reduceOp = " << resolvedArgs.args.reduceOp
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->reduceOp)\n";
    oss << "    kernelFunc.root = " << resolvedArgs.args.root << " <- aclrtLaunchKernelWithHostArgs(hostArgs->root)\n";
    oss << "    kernelFunc.sliceId = " << resolvedArgs.args.tag << " <- aclrtLaunchKernelWithHostArgs(hostArgs->tag)\n";
    oss << "    kernelFunc.inputSliceStride = " << static_cast<unsigned long long>(resolvedArgs.args.inputSliceStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->inputSliceStride)\n";
    oss << "    kernelFunc.outputSliceStride = " << static_cast<unsigned long long>(resolvedArgs.args.outputSliceStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->outputSliceStride)\n";
    oss << "    kernelFunc.repeatNum = " << static_cast<unsigned long long>(resolvedArgs.args.repeatNum)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->repeatNum)\n";
    oss << "    kernelFunc.inputRepeatStride = " << static_cast<unsigned long long>(resolvedArgs.args.inputRepeatStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->inputRepeatStride)\n";
    oss << "    kernelFunc.outputRepeatStride = "
        << static_cast<unsigned long long>(resolvedArgs.args.outputRepeatStride)
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->outputRepeatStride)\n";
    oss << "    kernelFunc.numBlocks = " << resolvedArgs.args.numBlocks
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->numBlocks)\n";
    oss << "    kernelFunc.isOpBase = " << (resolvedArgs.args.isOpBase ? "true" : "false")
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->isOpBase)\n";
    oss << "    kernelFunc.headCountMem = " << resolvedArgs.args.headCountMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->headCountMem=" << rawArgs.headCountMem
        << "), checker keeps raw headCountMem pointer unchanged\n";
    oss << "    kernelFunc.tailCountMem = " << resolvedArgs.args.tailCountMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->tailCountMem=" << rawArgs.tailCountMem
        << "), checker keeps raw tailCountMem pointer unchanged\n";
    oss << "    kernelFunc.addOneMem = " << resolvedArgs.args.addOneMem
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->addOneMem=" << rawArgs.addOneMem
        << "), checker keeps raw addOneMem pointer unchanged\n";
    oss << "    kernelFunc.counterMemSize = " << resolvedArgs.args.counterMemSize
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->counterMemSize)\n";
    oss << "    kernelFunc.isEnableCounter = " << (resolvedArgs.args.isEnableCounter ? "true" : "false")
        << " <- aclrtLaunchKernelWithHostArgs(hostArgs->isEnableCounter)\n";
    if (resolvedArgs.args.hasExtraArgs) {
        DumpVirtualKernelExtraArgsWithSource(oss, resolvedArgs.args.extraArgs);
    } else {
        oss << "    kernelFunc.extraArgs <- not present in aclrtLaunchKernelWithHostArgs host args\n";
    }
    HCCL_VM_DEBUG("{}", oss.str());
}

struct VirtualAivLibrary {
    void* handle = nullptr;
    std::string soName{};
    std::string soPath{};
    AivEnvInitFunc envInit = nullptr;
    AivSetBlockIdxFunc setBlockIdx = nullptr;
    AivDumpTasksFunc dumpTasks = nullptr;
};

static void CloseVirtualAivLibrary(VirtualAivLibrary& lib)
{
    if (lib.handle != nullptr) {
        dlclose(lib.handle);
        lib.handle = nullptr;
    }

    lib.envInit = nullptr;
    lib.setBlockIdx = nullptr;
    lib.dumpTasks = nullptr;
}

static VirtualAivLibrary LoadVirtualAivLibrary(const std::string& soName, const std::string& kernelName)
{
    VirtualAivLibrary lib{};
    lib.soName = soName;
    if (lib.soName.empty()) {
        HCCL_VM_ERROR("empty soName for kernel {}", kernelName.empty() ? "<empty>" : kernelName.c_str());
        return lib;
    }

    lib.soPath = GetAivLibraryPath(lib.soName, kernelName);
    if (lib.soPath.empty()) {
        return lib;
    }

    dlerror();
    lib.handle = dlopen(lib.soPath.c_str(), RTLD_NOW | RTLD_LOCAL);
    const char* dlopenErr = dlerror();
    if (lib.handle == nullptr || dlopenErr != nullptr) {
        HCCL_VM_ERROR("dlopen {} failed, err = {}", lib.soPath, dlopenErr == nullptr ? "unknown" : dlopenErr);
        CloseVirtualAivLibrary(lib);
        return lib;
    }

    dlerror();
    lib.envInit = reinterpret_cast<AivEnvInitFunc>(dlsym(lib.handle, AIV_STUB_ENV_INIT_SYMBOL));
    const char* envInitErr = dlerror();
    if (lib.envInit == nullptr || envInitErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_ENV_INIT_SYMBOL, lib.soPath,
            envInitErr == nullptr ? "unknown" : envInitErr);
        lib.envInit = nullptr;
    }

    dlerror();
    lib.setBlockIdx = reinterpret_cast<AivSetBlockIdxFunc>(dlsym(lib.handle, AIV_STUB_SET_BLOCK_IDX_SYMBOL));
    const char* setBlockIdxErr = dlerror();
    if (lib.setBlockIdx == nullptr || setBlockIdxErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_SET_BLOCK_IDX_SYMBOL, lib.soPath,
            setBlockIdxErr == nullptr ? "unknown" : setBlockIdxErr);
        lib.setBlockIdx = nullptr;
    }

    dlerror();
    lib.dumpTasks = reinterpret_cast<AivDumpTasksFunc>(dlsym(lib.handle, AIV_STUB_DUMP_TASKS_SYMBOL));
    const char* dumpTasksErr = dlerror();
    if (lib.dumpTasks == nullptr || dumpTasksErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", AIV_STUB_DUMP_TASKS_SYMBOL, lib.soPath,
            dumpTasksErr == nullptr ? "unknown" : dumpTasksErr);
        lib.dumpTasks = nullptr;
    }

    return lib;
}

static aclError VirtualExecuteAivKernel(
    const std::string& kernelName, const std::string& soName, uint32_t numBlocks, const AivHostLaunchArgs& rawArgs,
    uint32_t launchIndex)
{
    VirtualAivLibrary lib = LoadVirtualAivLibrary(soName, kernelName);
    if (lib.handle == nullptr || lib.envInit == nullptr || lib.setBlockIdx == nullptr) {
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }
    if (kernelName.empty()) {
        HCCL_VM_ERROR("empty kernelName, can not virtual execute AIV kernel.");
        CloseVirtualAivLibrary(lib);
        return ACL_ERROR_INVALID_PARAM;
    }

    dlerror();
    void* kernelSymbol = dlsym(lib.handle, kernelName.c_str());
    const char* kernelErr = dlerror();
    if (kernelSymbol == nullptr || kernelErr != nullptr) {
        HCCL_VM_ERROR(
            "dlsym {} from {} failed, err = {}", kernelName, lib.soPath, kernelErr == nullptr ? "unknown" : kernelErr);
        CloseVirtualAivLibrary(lib);
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    ResolvedKernelLaunchArgs resolvedArgs = PrepareResolvedKernelLaunchArgs(rawArgs);
    uint64_t inputSize = 0;
    uint64_t outputSize = 0;
    uint64_t inputGlobalOffsetBase = 0;
    uint64_t outputGlobalOffsetBase = 0;
    uint64_t cclBufferSize = 0;
    uint64_t aivCommInfoSize = 0;
    ResolveVirtualAivBufferSizes(
        kernelName, resolvedArgs, inputSize, outputSize, inputGlobalOffsetBase, outputGlobalOffsetBase, cclBufferSize,
        aivCommInfoSize);
    DumpVirtualKernelFuncArgs(kernelName, numBlocks, rawArgs, resolvedArgs);
    if (inputSize == INVALID_MEMORY_LAYOUT_SIZE || outputSize == INVALID_MEMORY_LAYOUT_SIZE
        || cclBufferSize == INVALID_MEMORY_LAYOUT_SIZE) {
        HCCL_VM_ERROR("failed to resolve opMemInfo size for kernel {}, rank={}", kernelName, resolvedArgs.args.rank);
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        CloseVirtualAivLibrary(lib);
        return ACL_ERROR_INVALID_PARAM;
    }
    BackfillCurrentAivOpMemCclBuffer(resolvedArgs);

    AivSim::AivOpParam curOpParam{};
    curOpParam.dataType = resolvedArgs.args.dataType;
    curOpParam.len = resolvedArgs.args.len;
    curOpParam.reduceOp = resolvedArgs.args.reduceOp;
    curOpParam.root = resolvedArgs.args.root;
    curOpParam.sliceId = resolvedArgs.args.tag;
    curOpParam.inputStride = resolvedArgs.args.inputSliceStride;
    curOpParam.outputStride = resolvedArgs.args.outputSliceStride;
    size_t kernelNameCopyLen = kernelName.size();
    if (kernelNameCopyLen >= AivSim::AIV_OP_KERNEL_NAME_MAX_LEN) {
        kernelNameCopyLen = AivSim::AIV_OP_KERNEL_NAME_MAX_LEN - 1;
    }
    std::memcpy(curOpParam.kernelName, kernelName.data(), kernelNameCopyLen);
    curOpParam.kernelName[kernelNameCopyLen] = '\0';
    HCCL_VM_DEBUG(
        "aiv_env_init and curOp:\n"
        "  rank = {}\n"
        "  blockNum = {}\n"
        "  buffIn = {:p}\n"
        "  rankSize = {}\n"
        "  input = 0x{:x}\n"
        "  inputSize = {}\n"
        "  output = 0x{:x}\n"
        "  outputSize = {}\n"
        "  inputGlobalOffsetBase = {}\n"
        "  outputGlobalOffsetBase = {}\n"
        "  cclBufferSize = {}\n"
        "  aivCommInfoSize = {}\n"
        "  curOp.dataType = {}\n"
        "  curOp.len = {}\n"
        "  curOp.reduceOp = {}\n"
        "  curOp.root = {}\n"
        "  curOp.sliceId = {}\n"
        "  curOp.inputStride = {}\n"
        "  curOp.outputStride = {}\n"
        "  curOp.kernelName = {}",
        resolvedArgs.args.rank, numBlocks, resolvedArgs.args.buffersIn, resolvedArgs.args.rankSize,
        static_cast<unsigned long long>(resolvedArgs.args.input), static_cast<unsigned long long>(inputSize),
        static_cast<unsigned long long>(resolvedArgs.args.output), static_cast<unsigned long long>(outputSize),
        static_cast<unsigned long long>(inputGlobalOffsetBase), static_cast<unsigned long long>(outputGlobalOffsetBase),
        static_cast<unsigned long long>(cclBufferSize), static_cast<unsigned long long>(aivCommInfoSize),
        curOpParam.dataType, static_cast<unsigned long long>(curOpParam.len), curOpParam.reduceOp, curOpParam.root,
        curOpParam.sliceId, static_cast<unsigned long long>(curOpParam.inputStride),
        static_cast<unsigned long long>(curOpParam.outputStride), curOpParam.kernelName);
    lib.envInit(
        resolvedArgs.args.rank, numBlocks, resolvedArgs.args.buffersIn, resolvedArgs.args.rankSize,
        resolvedArgs.args.input, inputSize, resolvedArgs.args.output, outputSize, inputGlobalOffsetBase,
        outputGlobalOffsetBase, cclBufferSize, aivCommInfoSize, curOpParam);

    if (numBlocks == 0) {
        HCCL_VM_DEBUG("numBlocks is 0, skip kernel invocation.");
        ReleaseHostPtr(resolvedArgs.buffersInHandle);
        CloseVirtualAivLibrary(lib);
        return ACL_SUCCESS;
    }

    for (uint32_t blockIdx = 0; blockIdx < numBlocks; ++blockIdx) {
        HCCL_VM_DEBUG(
            "launch kernel {}, blockIdx={} <- "
            "aclrtLaunchKernelWithHostArgs(numBlocks) loop index; shared kernelFunc args are printed above",
            kernelName, blockIdx);
        lib.setBlockIdx(static_cast<int64_t>(blockIdx));
        if (resolvedArgs.args.hasExtraArgs) {
            auto kernelFunc = reinterpret_cast<AivExtraOpKernelFunc>(kernelSymbol);
            kernelFunc(
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.buffersIn)), resolvedArgs.args.input,
                resolvedArgs.args.output, resolvedArgs.args.rank, resolvedArgs.args.sendRecvRemoteRank,
                resolvedArgs.args.rankSize, resolvedArgs.args.xRankSize, resolvedArgs.args.yRankSize,
                resolvedArgs.args.zRankSize, resolvedArgs.args.len, resolvedArgs.args.dataType,
                resolvedArgs.args.reduceOp, resolvedArgs.args.root, resolvedArgs.args.tag,
                resolvedArgs.args.inputSliceStride, resolvedArgs.args.outputSliceStride, resolvedArgs.args.repeatNum,
                resolvedArgs.args.inputRepeatStride, resolvedArgs.args.outputRepeatStride, resolvedArgs.args.numBlocks,
                resolvedArgs.args.isOpBase,
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.headCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.tailCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.addOneMem)),
                resolvedArgs.args.counterMemSize, resolvedArgs.args.isEnableCounter, resolvedArgs.args.extraArgs);
        } else {
            auto kernelFunc = reinterpret_cast<AivOpKernelFunc>(kernelSymbol);
            kernelFunc(
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.buffersIn)), resolvedArgs.args.input,
                resolvedArgs.args.output, resolvedArgs.args.rank, resolvedArgs.args.sendRecvRemoteRank,
                resolvedArgs.args.rankSize, resolvedArgs.args.xRankSize, resolvedArgs.args.yRankSize,
                resolvedArgs.args.zRankSize, resolvedArgs.args.len, resolvedArgs.args.dataType,
                resolvedArgs.args.reduceOp, resolvedArgs.args.root, resolvedArgs.args.tag,
                resolvedArgs.args.inputSliceStride, resolvedArgs.args.outputSliceStride, resolvedArgs.args.repeatNum,
                resolvedArgs.args.inputRepeatStride, resolvedArgs.args.outputRepeatStride, resolvedArgs.args.numBlocks,
                resolvedArgs.args.isOpBase,
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.headCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.tailCountMem)),
                const_cast<uint8_t*>(static_cast<const uint8_t*>(resolvedArgs.args.addOneMem)),
                resolvedArgs.args.counterMemSize, resolvedArgs.args.isEnableCounter);
        }
    }
    if (lib.dumpTasks != nullptr) {
        lib.dumpTasks(launchIndex);
    }

    ReleaseHostPtr(resolvedArgs.buffersInHandle);
    CloseVirtualAivLibrary(lib);
    return ACL_SUCCESS;
}

HcclResult ExecuteKernelLaunch(const AivOpArgs& opArgs)
{
    HCCL_VM_DEBUG(
        "called with parameters:\n"
        "  cmdType = {}\n"
        "  comm = {}\n"
        "  hcclComm = {:p}\n"
        "  numBlocks = {}\n"
        "  stream = {:p}\n"
        "  beginTime = {}\n"
        "  counter.headCountMem = 0x{:x}\n"
        "  counter.tailCountMem = 0x{:x}\n"
        "  counter.addOneMem = 0x{:x}\n"
        "  counter.memSize = {}\n"
        "  counter.isEnableCounter = {}\n"
        "  input = 0x{:x}\n"
        "  output = 0x{:x}\n"
        "  rank = {}\n"
        "  sendRecvRemoteRank = {}\n"
        "  rankSize = {}\n"
        "  xRankSize = {}\n"
        "  yRankSize = {}\n"
        "  zRankSize = {}\n"
        "  count = {}\n"
        "  dataType = {}\n"
        "  op = {}\n"
        "  root = {}\n"
        "  sliceId = {}\n"
        "  inputSliceStride = {}\n"
        "  outputSliceStride = {}\n"
        "  repeatNum = {}\n"
        "  inputRepeatStride = {}\n"
        "  outputRepeatStride = {}\n"
        "  isOpBase = {}\n"
        "  argsType = {}\n"
        "  buffersIn(base aivCommInfoPtr, raw pointer only; parsed results below) = {:p}",
        static_cast<int>(opArgs.cmdType), opArgs.comm, opArgs.hcclComm, opArgs.numBlocks, opArgs.stream,
        static_cast<unsigned long long>(opArgs.beginTime), static_cast<unsigned long long>(opArgs.counter.headCountMem),
        static_cast<unsigned long long>(opArgs.counter.tailCountMem),
        static_cast<unsigned long long>(opArgs.counter.addOneMem), opArgs.counter.memSize,
        static_cast<int>(opArgs.counter.isEnableCounter), static_cast<unsigned long long>(opArgs.input),
        static_cast<unsigned long long>(opArgs.output), opArgs.rank, opArgs.sendRecvRemoteRank, opArgs.rankSize,
        static_cast<unsigned long long>(opArgs.xRankSize), static_cast<unsigned long long>(opArgs.yRankSize),
        static_cast<unsigned long long>(opArgs.zRankSize), static_cast<unsigned long long>(opArgs.count),
        static_cast<int>(opArgs.dataType), static_cast<int>(opArgs.op), opArgs.root, opArgs.sliceId,
        static_cast<unsigned long long>(opArgs.inputSliceStride),
        static_cast<unsigned long long>(opArgs.outputSliceStride), static_cast<unsigned long long>(opArgs.repeatNum),
        static_cast<unsigned long long>(opArgs.inputRepeatStride),
        static_cast<unsigned long long>(opArgs.outputRepeatStride), static_cast<int>(opArgs.isOpBase),
        static_cast<int>(opArgs.argsType), opArgs.buffersIn);
    if (IsAivExtraArgsCmdType(opArgs.cmdType)) {
        DumpAivExtraArgs(opArgs.extraArgs);
    } else {
        HCCL_VM_DEBUG("extraArgs are not used for cmdType={}.", static_cast<int>(opArgs.cmdType));
    }
    DumpAivTopo(opArgs.topo_);
    DumpBuffersInParsedDeviceView(opArgs.buffersIn, opArgs.rankSize, opArgs.numBlocks);

    const HcclCMDType prevCmdType = g_currentAivCmdType;
    const KernelArgsType prevArgsType = g_currentAivArgsType;
    const uint32_t prevLaunchIndex = g_currentAivLaunchIndex;
    void* prevStream = g_currentAivStream;
    const bool prevContextActive = g_currentAivContextActive;
    g_currentAivCmdType = opArgs.cmdType;
    g_currentAivArgsType = opArgs.argsType;
    g_currentAivLaunchIndex = INVALID_AIV_LAUNCH_INDEX;
    g_currentAivStream = opArgs.stream;
    g_currentAivContextActive = true;
    HCCL_VM_INFO(
        "prepared AIV launch context, cmdType={}, argsType={}, stream={:p}", static_cast<int>(g_currentAivCmdType),
        static_cast<int>(g_currentAivArgsType), g_currentAivStream);

    using ExecuteKernelLaunchFunc = HcclResult (*)(const AivOpArgs&);
    constexpr const char* executeKernelLaunchSymbol = "_ZN8ops_hccl19ExecuteKernelLaunchERKNS_9AivOpArgsE";

    dlerror();
    auto executeKernelLaunchFunc
        = reinterpret_cast<ExecuteKernelLaunchFunc>(dlsym(RTLD_NEXT, executeKernelLaunchSymbol));
    const char* dlsymErr = dlerror();
    if (executeKernelLaunchFunc == nullptr || dlsymErr != nullptr) {
        g_currentAivCmdType = prevCmdType;
        g_currentAivArgsType = prevArgsType;
        g_currentAivLaunchIndex = prevLaunchIndex;
        g_currentAivStream = prevStream;
        g_currentAivContextActive = prevContextActive;
        HCCL_VM_ERROR(
            "dlsym {} failed, err = {}", executeKernelLaunchSymbol, dlsymErr == nullptr ? "unknown" : dlsymErr);
        return HcclResult::HCCL_E_NOT_SUPPORT;
    }

    HcclResult ret = executeKernelLaunchFunc(opArgs);
    g_currentAivCmdType = prevCmdType;
    g_currentAivArgsType = prevArgsType;
    g_currentAivLaunchIndex = prevLaunchIndex;
    g_currentAivStream = prevStream;
    g_currentAivContextActive = prevContextActive;
    HCCL_VM_INFO("returned {}", static_cast<int>(ret));
    return ret;
}
} // namespace ops_hccl

extern "C" aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    (void)cfg;
    HCCL_VM_DEBUG(
        "called with parameters:\n"
        "  funcHandle = {:p}\n"
        "  numBlocks = {}\n"
        "  stream = {:p}\n"
        "  hostArgs = {:p}\n"
        "  argsSize = {}\n"
        "  placeHolderArray = {:p}\n"
        "  placeHolderNum = {}\n"
        "  currentCmdType = {}\n"
        "  currentArgsType = {}\n"
        "  currentLaunchIndex = {}\n"
        "  currentContextStream = {:p}\n"
        "  currentContextActive = {}",
        funcHandle, numBlocks, stream, hostArgs, argsSize, static_cast<const void*>(placeHolderArray), placeHolderNum,
        static_cast<int>(ops_hccl::g_currentAivCmdType), static_cast<int>(ops_hccl::g_currentAivArgsType),
        g_currentAivLaunchIndex, g_currentAivStream, static_cast<int>(g_currentAivContextActive));
    ops_hccl::DumpLaunchKernelCfg(cfg);
    ops_hccl::DumpPlaceHolderArray(placeHolderArray, placeHolderNum);

    if (!g_currentAivContextActive) {
        HCCL_VM_ERROR("no active HCCL AIV ExecuteKernelLaunch context, "
                      "can not record or virtual execute AIV kernel.");
        return ACL_ERROR_INTERNAL_ERROR;
    }

    ops_hccl::AivHostLaunchArgs parsedHostArgs{};
    if (!ops_hccl::ParseAivHostLaunchArgs(hostArgs, argsSize, ops_hccl::g_currentAivCmdType, parsedHostArgs)) {
        return ACL_ERROR_INVALID_PARAM;
    }
    ops_hccl::DumpHostLaunchArgs(&parsedHostArgs, argsSize);
    const std::string inferredKernelName = ops_hccl::InferKernelNameFromFuncHandle(funcHandle);
    const bool isUnsupportedFallbackCmdType = ops_hccl::IsUnsupportedFallbackAivCmdType(ops_hccl::g_currentAivCmdType);
    const std::string fallbackKernelName = ops_hccl::GetFallbackAivKernelName(
        ops_hccl::g_currentAivCmdType, static_cast<HcclDataType>(parsedHostArgs.dataType),
        ops_hccl::g_currentAivArgsType);

    std::string kernelName = inferredKernelName;
    if (!kernelName.empty()) {
        HCCL_VM_INFO(
            "use inferred kernelName from funcHandle = {} "
            "(cmdType={}, dataType={}, argsType={}, fallbackKernelName={})",
            kernelName, static_cast<int>(ops_hccl::g_currentAivCmdType), parsedHostArgs.dataType,
            static_cast<int>(ops_hccl::g_currentAivArgsType),
            fallbackKernelName.empty() ? "<empty>" : fallbackKernelName.c_str());
    } else {
        if (isUnsupportedFallbackCmdType) {
            HCCL_VM_ERROR(
                "cmdType={} fallback AIV kernel is not supported currently.",
                static_cast<int>(ops_hccl::g_currentAivCmdType));
            return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
        }
        kernelName = fallbackKernelName;
        if (!kernelName.empty()) {
            HCCL_VM_INFO(
                "fallback kernelName by cmdType/dataType/argsType = {} "
                "(cmdType={}, dataType={}, argsType={}, inferredKernelName=<empty>)",
                kernelName, static_cast<int>(ops_hccl::g_currentAivCmdType), parsedHostArgs.dataType,
                static_cast<int>(ops_hccl::g_currentAivArgsType));
        }
    }

    if (kernelName.empty()) {
        HCCL_VM_ERROR(
            "failed to resolve kernelName from funcHandle/hostArgs, "
            "cmdType={}, dataType={}, argsType={}, can not virtual execute kernel.",
            static_cast<int>(ops_hccl::g_currentAivCmdType), parsedHostArgs.dataType,
            static_cast<int>(ops_hccl::g_currentAivArgsType));
        return ACL_ERROR_RT_FEATURE_NOT_SUPPORT;
    }

    const std::string soName = ops_hccl::AIV_STUB_SO_NAME;
    HCCL_VM_INFO(
        "resolved kernelName = {}, soName = {}, cmdType = {}, "
        "argsType = {}",
        kernelName, soName, static_cast<int>(ops_hccl::g_currentAivCmdType),
        static_cast<int>(ops_hccl::g_currentAivArgsType));

    const uint32_t prevLaunchIndex = g_currentAivLaunchIndex;

    HcclTaskMetaData taskMetaData{};
    taskMetaData.taskType = HccLTaskMetaType::AIV_GRAPH;
    taskMetaData.commId = 0;
    taskMetaData.rankId = static_cast<uint32_t>(sim::GetCurrRankId());
    taskMetaData.jettyId = 0;
    void* recordStream = stream != nullptr ? stream : g_currentAivStream;
    if (recordStream != nullptr) {
        taskMetaData.streamId = sim::GetCurrentStreamId(reinterpret_cast<uint64_t>(recordStream));
    }

    const uint32_t launchIndex = g_aivLaunchIndex.fetch_add(1, std::memory_order_relaxed);
    taskMetaData.taskData.aiv.launchIdx = static_cast<uint64_t>(launchIndex);
    g_currentAivLaunchIndex = launchIndex;
    uint32_t unusedIndex = 0;
    auto insertRet = InsertTaskToCollection(&taskMetaData, &unusedIndex);
    if (insertRet != HcclSim::HcclVmResult::HCCL_SIM_SUCCESS) {
        g_currentAivLaunchIndex = prevLaunchIndex;
        HCCL_VM_ERROR(
            "failed to insert AIV launch task, ret={}, rankId={}", static_cast<uint32_t>(insertRet),
            taskMetaData.rankId);
        return ACL_ERROR_INTERNAL_ERROR;
    }
    HCCL_VM_INFO(
        "inserted AIV launch task, rankId={}, "
        "launchIndex={}, streamId={}",
        taskMetaData.rankId, launchIndex, taskMetaData.streamId);

    aclError ret = ops_hccl::VirtualExecuteAivKernel(kernelName, soName, numBlocks, parsedHostArgs, launchIndex);
    g_currentAivLaunchIndex = prevLaunchIndex;
    HCCL_VM_INFO("virtual execute ret = {}", static_cast<int>(ret));
    return ret;
}
// ===== AIV virtual-kernel support scope end =====
