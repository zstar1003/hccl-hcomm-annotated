/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_device_pub.h"

#include <cerrno>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <execinfo.h>
#include <fcntl.h>
#include <link.h>
#include <map>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include "sim_log.h"
#include "store_sim_memory_manager.h"
#include "db_sim_runner_db.h"
#include "sim_pipe_io.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

uint32_t g_rankId = 0;
uint32_t g_deviceKey = 0;

char g_crash_file_name[64] = "crash_default.log";

HcclVmResult SetCurRankId(uint32_t rankId)
{
    HCCL_VM_INFO("SetCurRankId, old rankId: {}, new rankId: {}", g_rankId, rankId);
    g_rankId = rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult GetCurRankId(uint32_t* rankId)
{
    *rankId = g_rankId;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

void SetCurDeviceKey(uint32_t deviceKey)
{
    HCCL_VM_INFO("SetCurDeviceKey, old deviceKey: {}, new deviceKey: {}", g_deviceKey, deviceKey);
    g_deviceKey = deviceKey;
}

uint32_t GetCurDeviceKey() { return g_deviceKey; }

uint8_t sqBuffer[HCCL_SQE_SIZE * HCCL_SQE_MAX_CNT];
HcclVmResult GetSqBufferAddr(uint8_t** sqBuff)
{
    *sqBuff = sqBuffer;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> jettyId2PiValMap;
HcclVmResult GetPiValByJettyId(uint32_t jettyId, uint32_t* piValue)
{
    *piValue = jettyId2PiValMap[jettyId];
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

HcclVmResult UpdatePiValByJettyId(uint32_t jettyId, uint32_t piValue)
{
    jettyId2PiValMap[jettyId] = piValue;
    return HcclVmResult::HCCL_SIM_SUCCESS;
}

std::map<uint32_t, uint32_t> sqTailMap;
uint32_t GetSqTail(uint32_t sqId) { return sqTailMap[sqId]; }

void UpdateSqTail(uint32_t sqId, uint32_t newTail) { sqTailMap[sqId] = newTail; }

// AICPU构造Task时任务转换，将本端设备地址(本rank对应host进程申请的)转换为虚拟地址
uint64_t TransLocalAddrToVirtual(uint64_t devAddr)
{
    uint64_t deviceKey = GetCurDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devAddr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return ((virMem.dev_mapped_ptr <= devAddr) && (devAddr < (virMem.dev_mapped_ptr + virMem.size))
                      && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV))
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
        return 0;
    }

    uint64_t diff = devAddr - virMemRes.first.dev_mapped_ptr;
    return virMemRes.first.start_ptr + diff;
}

// AICPU构造Task时任务转换，将对端设备地址(rankId对应host进程申请的)转换为虚拟地址
uint64_t TransRemoteAddrToVirtualByRank(uint64_t devAddr, uint32_t rankId)
{
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devAddr, rankId](const sim::VirtualMemBlock& virMem) {
              return ((virMem.dev_mapped_ptr <= devAddr) && (devAddr < (virMem.dev_mapped_ptr + virMem.size))
                      && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV))
                     && (virMem.rank_id == rankId);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
        return 0;
    }

    uint64_t diff = devAddr - virMemRes.first.dev_mapped_ptr;
    return virMemRes.first.start_ptr + diff;
}

uint64_t GetDevMapperAddrByDevAddrImpl(uint64_t devAddr, const char* file, int line)
{
    uint64_t deviceKey = GetCurDeviceKey();
    auto virMemRes
        = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devAddr, deviceKey](const sim::VirtualMemBlock& virMem) {
              return (virMem.dev_mapped_ptr <= devAddr) && (devAddr < virMem.dev_mapped_ptr + virMem.size)
                     && (virMem.src_type == static_cast<uint8_t>(sim::VIR_MEM_TYPE_DEV))
                     && (virMem.device_id == deviceKey);
          });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[0x{:x}], called from {}:{}", devAddr, file, line);
        return 0;
    }

    // 已经映射过的这里直接取值
    if (virMemRes.first.is_dev_access == 1) {
        HCCL_VM_INFO("device cann access this devAddr[0x{:x}], called from {}:{}", devAddr, file, line);
        return devAddr;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phyMemRes by phyMemId[{}], called from {}:{}", phyMemId, file, line);
        return 0;
    }

    std::string memName(phyMemRes->name);
    void* devMappedPtr = sim::MemoryManager::GetInstance().AcquireMemByName(memName.c_str());
    if (devMappedPtr == nullptr) {
        HCCL_VM_ERROR("acquire device shm for memName[{}] failed, called from {}:{}", memName, file, line);
        return 0;
    }

    uint64_t offset = devAddr - virMemRes.first.dev_mapped_ptr;
    uint64_t result = reinterpret_cast<uint64_t>(devMappedPtr) + offset;
    HCCL_VM_INFO(
        "devAddr[0x{:x}] offset[{}] devMappedPtr[0x{:x}] memName[{}]", devAddr, offset,
        reinterpret_cast<uint64_t>(devMappedPtr), memName);
    return result;
}

uint32_t GetRankIdByDevAddr(uint64_t devAddr)
{
    auto virMemRes = RunnerDB::GetOneByPred<sim::VirtualMemBlock>([devAddr](const sim::VirtualMemBlock& virMem) {
        return (
            (virMem.start_ptr <= devAddr) && (devAddr < (virMem.start_ptr + virMem.size))
            && (virMem.src_type == (uint8_t)sim::VIR_MEM_TYPE_DEV));
    });
    if (!virMemRes.second) {
        HCCL_VM_ERROR("cannot find virMemRes by devAddr[{}]", devAddr);
        return 0;
    }

    auto phyMemId = virMemRes.first.phy_mem_id;
    auto phyMemRes = RunnerDB::GetById<sim::PhyMemBlock>(phyMemId);
    if (!phyMemRes.has_value()) {
        HCCL_VM_ERROR("cannot find phyMemRes by phyMemId[{}]", phyMemId);
        return 0;
    };

    uint32_t device_id = phyMemRes->device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([device_id](const sim::Rank& rank) {
        return rank.device_id == device_id;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("cannot find rank by device_id[{}]", device_id);
        return 0;
    }

    return rank.first.rank_id;
}

uint32_t GetRankIdByIpAddr(std::string ipAddr)
{
    auto ret = RunnerDB::GetOneByPred<sim::EndPoint>([ipAddr](const sim::EndPoint& p) {
        return strcmp(p.ip_addr, ipAddr.c_str()) == 0;
    });
    if (!ret.second) {
        HCCL_VM_ERROR("cannot find Port by ipAddr[{}]", ipAddr);
        return 0;
    }

    uint32_t device_id = ret.first.device_id;
    auto rank = RunnerDB::GetOneByPred<sim::Rank>([device_id](const sim::Rank& rank) {
        return rank.device_id == device_id;
    });
    if (!rank.second) {
        HCCL_VM_ERROR("cannot find Rank by device_id[{}]", device_id);
        return 0;
    }

    return rank.first.rank_id;
}

void UpdataKfcStatus(uint64_t d2hAddr)
{
    if (d2hAddr == 0) {
        HCCL_VM_ERROR("d2hAddr is 0, cannot update kfc status.");
        return;
    }

    uint8_t devDoneStatus = 3;
    constexpr uint32_t headTailCnt = 66;
    constexpr uint32_t headCntOffset = 4088;
    constexpr uint32_t tailCntOffset = 4092;
    *reinterpret_cast<uint8_t*>(d2hAddr) = devDoneStatus;                // 更新设备状态
    *reinterpret_cast<uint32_t*>(d2hAddr + headCntOffset) = headTailCnt; // 更新headCnt
    *reinterpret_cast<uint32_t*>(d2hAddr + tailCntOffset) = headTailCnt; // 更新tailCnt
    HCCL_VM_INFO("update kfc status success.");
}

static int DumpModuleMap(struct dl_phdr_info* info, size_t size, void* data)
{
    int fd = *(int*)data;
    char buf[512];
    const char* name = info->dlpi_name[0] ? info->dlpi_name : "[main]";

    uint64_t vaddr_min = (uint64_t)-1;
    uint64_t vaddr_max = 0;
    for (int i = 0; i < info->dlpi_phnum; i++) {
        if (info->dlpi_phdr[i].p_type != PT_LOAD) {
            continue;
        }
        uint64_t seg_start = info->dlpi_addr + info->dlpi_phdr[i].p_vaddr;
        uint64_t seg_end = seg_start + info->dlpi_phdr[i].p_memsz;
        if (seg_start < vaddr_min) {
            vaddr_min = seg_start;
        }
        if (seg_end > vaddr_max) {
            vaddr_max = seg_end;
        }
    }

    int len = snprintf(
        buf, sizeof(buf), "  0x%016lx-0x%016lx  base=0x%016lx  %s\n", vaddr_min, vaddr_max, (uint64_t)info->dlpi_addr,
        name);
    if (len > 0) {
        write(fd, buf, len);
    }
    return 0;
}

static void DumpCallSites(int fd, void* const* stack_pointers, int nptrs)
{
    char buf[768];
    int len;

    len = snprintf(
        buf, sizeof(buf),
        "\n---- [Call Sites] (call_site = ret_addr - 4, aarch64 instruction size) ----\n"
        "     (addr2line -e <binary> -f -C -p <file_offset>)\n");
    if (len > 0) {
        write(fd, buf, len);
    }

    for (int i = 2; i < nptrs; i++) {
        uint64_t ret_addr = (uint64_t)stack_pointers[i];
        uint64_t call_addr = ret_addr - 4;

        Dl_info info;
        uint64_t base = 0;
        const char* module = "??";
        if (dladdr((void*)call_addr, &info) && info.dli_fname) {
            base = (uint64_t)info.dli_fbase;
            module = info.dli_fname;
        }

        uint64_t call_offset = call_addr - base;

        len = snprintf(
            buf, sizeof(buf), "  #%02d  call_site=0x%016lx  file_offset=0x%lx  base=0x%016lx  %s\n", i, call_addr,
            call_offset, base, module);
        if (len > 0) {
            write(fd, buf, len);
        }
    }
}

// 信号处理函数
void SignalHandler(int signum)
{
    const int MAX_STACK_FRAMES = 64;
    void* stack_pointers[MAX_STACK_FRAMES];

    int fd = open(g_crash_file_name, O_WRONLY | O_CREAT | O_APPEND, 0644);
    if (fd < 0) {
        _exit(1);
    }
    char msg[512] = {0};
    int len = snprintf(msg, sizeof(msg), "\n---- [Crash] Received signal:%d ----\n", signum);
    if (len > 0) {
        write(fd, msg, len);
    }

    int nptrs = backtrace(stack_pointers, MAX_STACK_FRAMES);
    backtrace_symbols_fd(stack_pointers, nptrs, fd);

    DumpCallSites(fd, stack_pointers, nptrs);

    len = snprintf(msg, sizeof(msg), "\n---- [Module Map] (file_offset = runtime_addr - base) ----\n");
    if (len > 0) {
        write(fd, msg, len);
    }
    dl_iterate_phdr(DumpModuleMap, &fd);

    close(fd);
    signal(signum, SIG_DFL);
    _exit(signum);
}

// 注册所有崩溃相关的信号
void RegisterSignalHandler()
{
    pid_t pid = getpid();
    sprintf(g_crash_file_name, "crash_%d.log", pid);
    signal(SIGSEGV, SignalHandler); // 段错误
    signal(SIGABRT, SignalHandler); // Abort (如 assert 失败)
    signal(SIGFPE, SignalHandler);  // 算术溢出/除零
    signal(SIGBUS, SignalHandler);  // 总线错误
    signal(SIGILL, SignalHandler);  // 非法指令
}

bool GetWqebufferByJettyId(uint64_t jettyId, uint64_t& wqeBuffer)
{
    auto raJetty = RunnerDB::GetById<sim::RaJetty>(jettyId);
    if (!raJetty.has_value()) {
        HCCL_VM_ERROR("RaJetty id:{:d} not found", jettyId);
        return false;
    }

    wqeBuffer = raJetty->sqBuffer;
    return true;
}

static int g_h2dReadFd = -1;
static int g_d2hWriteFd = -1;

void InitPipeFds(int h2dReadFd, int d2hWriteFd)
{
    g_h2dReadFd = h2dReadFd;
    g_d2hWriteFd = d2hWriteFd;
}

int DeviceSendMsg(uint8_t cmd, const void* data, uint32_t dataLen)
{
    if (g_d2hWriteFd < 0) {
        HCCL_VM_ERROR("pipe d2h write fd not initialized.");
        return -1;
    }

    return sim::PipeSendMsg(g_d2hWriteFd, cmd, data, dataLen);
}

int DeviceRecvMsg(uint8_t& outCmd, void* outData, uint32_t maxLen, uint32_t& outLen)
{
    if (g_h2dReadFd < 0) {
        HCCL_VM_ERROR("pipe h2d read fd not initialized.");
        return -1;
    }

    return sim::PipeRecvMsg(g_h2dReadFd, outCmd, outData, maxLen, outLen);
}

#ifdef __cplusplus
}
#endif
