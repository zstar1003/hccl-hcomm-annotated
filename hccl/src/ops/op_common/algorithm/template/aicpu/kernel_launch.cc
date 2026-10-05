/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <sstream>
#include <memory>
#include <cstring>
#include <sched.h>
#include <hccl/hccl_comm.h>
#include "alg_param.h"
#include "executor_base.h"
#include "coll_alg_exec_registry.h"
#include "coll_alg_v2_exec_registry.h"
#include "hcomm_primitives.h"
#include "hcomm_primitives_dl.h"
#include "dfx/task_exception_fun.h"
#include "kernel_launch.h"
#include "hcomm_diag_dl.h"
#include "hcomm_device_profiling_dl.h"
#include <unordered_map>
#include <mutex>
#include <atomic>
#include "hccl_device_comm_dl.h"
#include "exec_timeout_manager.h"
#include "alg_data_trans_wrapper.h"
#include "aicpu_task_cache_key.h"
#include "aicpu_task_cache_comm_manager.h"
#include "aicpu_task_cache_utils.h"
#include "aicpu_task_cache_policy.h"
#include "ins_send_executor.h"
#include "ins_recv_executor.h"

using namespace ops_hccl;
namespace {
// 统计缓存信息
struct CacheStats {
    std::atomic<uint64_t> hits{0};
    std::atomic<uint64_t> misses{0};

    // [中文导读] [AllReduce逐行 S47] 声明CacheStats::hitRate接口：由原子命中/未命中计数计算资源对象缓存命中比例。
    double hitRate() const
    // [中文导读] [AllReduce逐行 S48] 开始CacheStats::hitRate的函数体。
    {
        // [中文导读] [AllReduce逐行 S49] 将命中和未命中次数相加，得到当前总查询次数。
        uint64_t total = hits + misses;
        // [中文导读] [AllReduce逐行 S50] 查询次数非零时返回命中比例，否则返回0避免除零。
        return total > 0 ? static_cast<double>(hits) / total : 0.0;
    // [中文导读] [AllReduce逐行 S51] 结束CacheStats::hitRate函数体。
    }

    void Reset()
    {
        hits = 0;
        misses = 0;
    }
};

// 通信域缓存
class CommDomainCache {
public:
    explicit CommDomainCache(const std::string& commName) : commName_(commName) {}

    const std::string& GetCommName() const { return commName_; }

    // 获得缓存项，返回共享所有权保证使用期间对象稳定存活
    // [中文导读] [AllReduce逐行 S68] 声明CommDomainCache::Get接口：以algTag查设备资源对象缓存，返回shared_ptr保持对象存活。
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string& algTag)
    // [中文导读] [AllReduce逐行 S69] 开始CommDomainCache::Get的函数体。
    {
        // [中文导读] [AllReduce逐行 S70] 取得当前通信域资源缓存锁，保护algTag查找。
        std::lock_guard<std::mutex> lock(mutex_);
        // [中文导读] [AllReduce逐行 S71] 在当前域缓存中按算法tag查找资源对象。
        auto it = cache_.find(algTag);
        // [中文导读] [AllReduce逐行 S72] 命中时返回shared_ptr副本保证对象存活，未命中返回nullptr。
        return it != cache_.end() ? it->second : nullptr;
    // [中文导读] [AllReduce逐行 S73] 结束CommDomainCache::Get函数体。
    }

    // 缓存算法
    // [中文导读] [AllReduce逐行 S76] 声明CommDomainCache::Put接口：以algTag保存资源对象的独立拷贝。
    void Put(const std::string& algTag, const AlgResourceCtxSerializable& value)
    // [中文导读] [AllReduce逐行 S77] 开始CommDomainCache::Put的函数体。
    {
        // [中文导读] [AllReduce逐行 S78] 取得当前通信域资源缓存锁，保护缓存写入。
        std::lock_guard<std::mutex> lock(mutex_);
        // [中文导读] [AllReduce逐行 S79] 拷贝资源对象并以algTag替换/建立缓存项。
        cache_[algTag] = std::make_shared<AlgResourceCtxSerializable>(value);
    // [中文导读] [AllReduce逐行 S80] 结束CommDomainCache::Put函数体。
    }

    // 移除特定算法
    bool Remove(const std::string& algTag)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return cache_.erase(algTag) > 0;
    }

    // 清空所有缓存项
    void Clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        cache_.clear();
    }

    // [中文导读] [AllReduce逐行 S96] 返回可修改的CacheStats引用，调用方据此原子增加hits/misses。
    CacheStats& GetStats() { return stats_; }
    // [中文导读] [AllReduce逐行 S97] 返回只读CacheStats引用，调用方据此读取hits/misses。
    const CacheStats& GetStats() const { return stats_; }

    // [中文导读] [AllReduce逐行 S99] 声明CommDomainCache::GetCacheSize接口：加锁读取一个通信域的资源缓存条数。
    size_t GetCacheSize() const
    // [中文导读] [AllReduce逐行 S100] 开始CommDomainCache::GetCacheSize的函数体。
    {
        // [中文导读] [AllReduce逐行 S101] 取得域缓存锁保护条数读取。
        std::lock_guard<std::mutex> lock(mutex_);
        // [中文导读] [AllReduce逐行 S102] 返回当前通信域资源对象缓存项数。
        return cache_.size();
    // [中文导读] [AllReduce逐行 S103] 结束CommDomainCache::GetCacheSize函数体。
    }

private:
    std::string commName_;
    std::map<std::string, std::shared_ptr<const AlgResourceCtxSerializable>> cache_;
    CacheStats stats_;
    mutable std::mutex mutex_;
};

// 通信域缓存管理器
class CommDomainCacheManager {
public:
    // 获取算法缓存
    // [中文导读] [AllReduce逐行 S116] 声明CommDomainCacheManager::Get接口：从algTag解析通信域后查询资源对象缓存并统计命中率。
    std::shared_ptr<const AlgResourceCtxSerializable> Get(const std::string& algTag, const std::string& paramCommName)
    // [中文导读] [AllReduce逐行 S117] 开始CommDomainCacheManager::Get的函数体。
    {
        // [中文导读] [AllReduce逐行 S118] 从algTag第一/第二个下划线之间提取通信域名字。
        std::string commName = ExtractCommName(algTag);
        // 提取失败时使用参数中的commName
        // [中文导读] [AllReduce逐行 S120] 提取为空时使用调用参数中的通信域名字。
        if (commName.empty())
            // [中文导读] [AllReduce逐行 S121] 用paramCommName替代无法从tag解析出的名字。
            commName = paramCommName;

        // [中文导读] [AllReduce逐行 S123] 查找或创建该通信域的资源缓存容器。
        auto commCache = GetOrCreateComm(commName);
        // [中文导读] [AllReduce逐行 S124] 通信域缓存存在时尝试查找算法tag。
        if (commCache) {
            // [中文导读] [AllReduce逐行 S125] 获取该通信域命中/未命中统计引用。
            auto& stats = commCache->GetStats();
            // [中文导读] [AllReduce逐行 S126] 查询该算法tag对应的资源缓存对象。
            auto result = commCache->Get(algTag);
            // [中文导读] [AllReduce逐行 S127] 资源对象命中时统计命中并返回。
            if (result) {
                // [中文导读] [AllReduce逐行 S128] 原子增加资源对象缓存命中次数。
                stats.hits++;
                // [中文导读] [AllReduce逐行 S129] 返回缓存shared_ptr对象，保持资源对象存活。
                return result;
            // [中文导读] [AllReduce逐行 S130] 结束条件if (result)。
            }
            // [中文导读] [AllReduce逐行 S131] 当前域资源tag未命中，原子增加未命中次数。
            stats.misses++;
        // [中文导读] [AllReduce逐行 S132] 结束条件if (commCache)。
        }
        // [中文导读] [AllReduce逐行 S133] 未取得资源对象时返回空指针。
        return nullptr;
    // [中文导读] [AllReduce逐行 S134] 结束CommDomainCacheManager::Get函数体。
    }

    // 缓存算法结果
    // [中文导读] [AllReduce逐行 S137] 声明CommDomainCacheManager::Put接口：选择通信域缓存并存入反序列化资源对象。
    void Put(const std::string& algTag, const AlgResourceCtxSerializable& value, const std::string& paramCommName)
    // [中文导读] [AllReduce逐行 S138] 开始CommDomainCacheManager::Put的函数体。
    {
        // [中文导读] [AllReduce逐行 S139] 从算法tag解析通信域缓存名字。
        std::string commName = ExtractCommName(algTag);
        // [中文导读] [AllReduce逐行 S140] tag解析失败时使用显式传入的通信域名字。
        if (commName.empty())
            // [中文导读] [AllReduce逐行 S141] 把paramCommName作为目标缓存域。
            commName = paramCommName;

        // [中文导读] [AllReduce逐行 S143] 查找或创建该通信域资源缓存容器。
        auto commCache = GetOrCreateComm(commName);
        // [中文导读] [AllReduce逐行 S144] 取得域缓存后写入资源对象。
        if (commCache) {
            // [中文导读] [AllReduce逐行 S145] 调用域缓存Put，按算法tag拷贝保存资源对象。
            commCache->Put(algTag, value);
        // [中文导读] [AllReduce逐行 S146] 结束条件if (commCache)。
        }
    // [中文导读] [AllReduce逐行 S147] 结束CommDomainCacheManager::Put函数体。
    }

    // 释放通信域缓存
    bool ReleaseComm(const std::string& commName)
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        return commCaches_.erase(commName) > 0;
    }

    // 获得通信域统计信息
    // [中文导读] [AllReduce逐行 S157] 声明CommDomainCacheManager::GetCommStats接口：读取域的缓存统计值和缓存条数。
    bool GetCommStats(const std::string& commName, CacheStats& outStats, size_t& outCacheSize) const
    // [中文导读] [AllReduce逐行 S158] 开始CommDomainCacheManager::GetCommStats的函数体。
    {
        // [中文导读] [AllReduce逐行 S159] 加锁保护通信域缓存映射及统计读取。
        std::lock_guard<std::mutex> lock(mapMutex_);
        // [中文导读] [AllReduce逐行 S160] 查找需要统计的通信域缓存对象。
        auto it = commCaches_.find(commName);
        // [中文导读] [AllReduce逐行 S161] 该通信域存在时读取统计值和缓存条数。
        if (it != commCaches_.end()) {
            // [中文导读] [AllReduce逐行 S162] 原子读取命中次数并写入输出统计。
            outStats.hits = it->second->GetStats().hits.load();
            // [中文导读] [AllReduce逐行 S163] 原子读取未命中次数并写入输出统计。
            outStats.misses = it->second->GetStats().misses.load();
            // [中文导读] [AllReduce逐行 S164] 加锁读取域缓存条数并写入输出参数。
            outCacheSize = it->second->GetCacheSize();
            // [中文导读] [AllReduce逐行 S165] 读取成功返回true。
            return true;
        // [中文导读] [AllReduce逐行 S166] 结束条件if (it != commCaches_.end())。
        }
        // [中文导读] [AllReduce逐行 S167] 通信域未找到返回false。
        return false;
    // [中文导读] [AllReduce逐行 S168] 结束CommDomainCacheManager::GetCommStats函数体。
    }

    // 获得全局统计信息
    void GetGlobalStats(
        size_t& totalCommDomains, size_t& totalcacheEntries, uint64_t& totalHits, uint64_t& totalMisses) const
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        totalCommDomains = commCaches_.size();
        totalcacheEntries = 0;
        totalHits = 0;
        totalMisses = 0;
        for (const auto& pair : commCaches_) {
            const auto& commName = pair.first;
            const auto& commCache = pair.second;
            totalcacheEntries += commCache->GetCacheSize();
            totalHits += commCache->GetStats().hits.load();
            totalMisses += commCache->GetStats().misses.load();
        }
    }

    // 清空所有缓存
    void ClearAll()
    {
        std::lock_guard<std::mutex> lock(mapMutex_);
        commCaches_.clear();
    }

    // 从algTag中提取通信域名称
    // [中文导读] [AllReduce逐行 S196] 声明CommDomainCacheManager::ExtractCommName接口：提取algTag第一和第二个下划线之间的通信域名称，失败返回空串。
    std::string ExtractCommName(const std::string& algTag)
    // [中文导读] [AllReduce逐行 S197] 开始CommDomainCacheManager::ExtractCommName的函数体。
    {
        // [中文导读] [AllReduce逐行 S198] 查找algTag第一个下划线位置。
        size_t firstUnderscore = algTag.find('_');
        // [中文导读] [AllReduce逐行 S199] 没有第一个下划线时无法解析通信域名字。
        if (firstUnderscore == std::string::npos)
            // [中文导读] [AllReduce逐行 S200] 返回空串，由调用者使用显式通信域名字兜底。
            return "";

        // [中文导读] [AllReduce逐行 S202] 从第一个下划线后查第二个下划线位置。
        size_t secondUnderscore = algTag.find('_', firstUnderscore + 1);
        // [中文导读] [AllReduce逐行 S203] 没有第二个下划线时也无法解析通信域名字。
        if (secondUnderscore == std::string::npos)
            // [中文导读] [AllReduce逐行 S204] 返回空串，由调用者使用显式通信域名字兜底。
            return "";

        // [中文导读] [AllReduce逐行 S206] 提取两个下划线中间的子串作为通信域名字。
        return algTag.substr(firstUnderscore + 1, secondUnderscore - firstUnderscore - 1);
    // [中文导读] [AllReduce逐行 S207] 结束CommDomainCacheManager::ExtractCommName函数体。
    }

private:
    // 获取或创建通信域缓存
    // [中文导读] [AllReduce逐行 S211] 声明CommDomainCacheManager::GetOrCreateComm接口：加锁查找或创建通信域资源缓存。
    std::shared_ptr<CommDomainCache> GetOrCreateComm(const std::string& commName)
    // [中文导读] [AllReduce逐行 S212] 开始CommDomainCacheManager::GetOrCreateComm的函数体。
    {
        // [中文导读] [AllReduce逐行 S213] 加锁保护全局通信域到资源缓存对象的映射。
        std::lock_guard<std::mutex> lock(mapMutex_);
        // [中文导读] [AllReduce逐行 S214] 按通信域名字查找已有缓存容器。
        auto it = commCaches_.find(commName);
        // [中文导读] [AllReduce逐行 S215] 通信域缓存已经存在时直接返回。
        if (it != commCaches_.end()) {
            // [中文导读] [AllReduce逐行 S216] 返回已有通信域的缓存对象shared_ptr。
            return it->second;
        // [中文导读] [AllReduce逐行 S217] 结束条件if (it != commCaches_.end())。
        }
        // [中文导读] [AllReduce逐行 S218] 创建新的CommDomainCache并插入通信域映射。
        auto result = commCaches_.emplace(commName, std::make_shared<CommDomainCache>(commName));
        // [中文导读] [AllReduce逐行 S219] 返回刚创建或emplace返回的域缓存对象。
        return result.first->second;
    // [中文导读] [AllReduce逐行 S220] 结束CommDomainCacheManager::GetOrCreateComm函数体。
    }

    mutable std::mutex mapMutex_;
    std::map<std::string, std::shared_ptr<CommDomainCache>> commCaches_;
};

// 全局缓存管理器实例
static CommDomainCacheManager g_cacheManager;

// [中文导读] [AllReduce逐行 S229] 声明DeserializeResCtx接口：按param.resCtx/ctxSize恢复Host序列化资源对象。
std::unique_ptr<AlgResourceCtxSerializable> DeserializeResCtx(const OpParam* param)
// [中文导读] [AllReduce逐行 S230] 开始DeserializeResCtx的函数体。
{
    // [中文导读] [AllReduce逐行 S231] 创建新的可序列化资源对象接收字段恢复。
    std::unique_ptr<AlgResourceCtxSerializable> resCtx = std::make_unique<AlgResourceCtxSerializable>();
    // [中文导读] [AllReduce逐行 S232] 将Device参数中的资源上下文地址解释为字节指针。
    char* ctx = static_cast<char*>(param->resCtx);
    // [中文导读] [AllReduce逐行 S233] 按param.ctxSize复制资源序列化字节。
    std::vector<char> seq(ctx, ctx + param->ctxSize);
    // [中文导读] [AllReduce逐行 S234] 调用资源对象DeSerialize恢复线程、通道、拓扑等字段。
    resCtx->DeSerialize(seq);
    // [中文导读] [AllReduce逐行 S235] 返回独占持有的恢复资源对象。
    return resCtx;
// [中文导读] [AllReduce逐行 S236] 结束DeserializeResCtx函数体。
}
} // namespace

namespace ops_hccl {
constexpr u32 PERCENTAGE_MULTIPLIER = 100; // 命中率转百分比
// 选择走新（CollAlgExecRegistryV2）/老（CollAlgExecRegistry）算子流程
// A5芯片或者template名称前缀为"opv2_"（当前A2的HostNic Send/Recv使用）走新流程，其他芯片走老流程
// [中文导读] [AllReduce逐行 S243] 声明IsOpsV2接口：opv2_前缀或支持out-place芯片走新执行器流程。
bool IsOpsV2(const char* algName, HcclDevType deviceType)
// [中文导读] [AllReduce逐行 S244] 开始IsOpsV2的函数体。
{
    // 检查algName前缀是否为"opv2_"
    // [中文导读] [AllReduce逐行 S246] 算法名字非空时才检查opv2_前缀。
    if (algName != nullptr) {
        // [中文导读] [AllReduce逐行 S247] 定义新流程算法名的opv2_前缀。
        const char* prefix = "opv2_";
        // [中文导读] [AllReduce逐行 S248] 比较算法名开头是否为opv2_。
        if (strncmp(algName, prefix, strlen(prefix)) == 0) {
            // [中文导读] [AllReduce逐行 S249] opv2_前缀算法使用新执行器流程。
            return true;
        // [中文导读] [AllReduce逐行 S250] 结束条件if (strncmp(algName, prefix, strlen(prefix)) == 0)。
        }
    // [中文导读] [AllReduce逐行 S251] 结束条件if (algName != nullptr)。
    }

    // 根据deviceType判断
    // [中文导读] [AllReduce逐行 S254] 支持out-place的芯片也走新流程，包括本例950。
    if (shouldGoOutPlace(deviceType)) {
        // [中文导读] [AllReduce逐行 S255] 返回使用新执行器注册器及资源格式。
        return true;
    // [中文导读] [AllReduce逐行 S256] 结束条件if (shouldGoOutPlace(deviceType))。
    }

    // [中文导读] [AllReduce逐行 S258] 其余芯片/算法走旧执行器流程。
    return false;
// [中文导读] [AllReduce逐行 S259] 结束IsOpsV2函数体。
}
} // namespace ops_hccl

// [中文导读] [AllReduce逐行 S262] 声明EnforceLaunchTask接口：结束批模式强制提交当前任务，再恢复批模式。
inline HcclResult EnforceLaunchTask(const char* algTag)
// [中文导读] [AllReduce逐行 S263] 开始EnforceLaunchTask的函数体。
{
    // [中文导读] [AllReduce逐行 S264] 结束当前批模式，把当前线程待提交任务实际下发。
    if (HcommBatchModeEnd(algTag) != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S265] 输出错误日志，记录EnforceLaunchTask当前阶段和相关参数。
        HCCL_ERROR("failed set eager mode, tag is %s.", algTag);
        // [中文导读] [AllReduce逐行 S266] 批模式结束失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S267] 结束条件if (HcommBatchModeEnd(algTag) != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S268] 再次启动同一算法tag的批模式，使后续非缓存任务分开采集。
    if (HcommBatchModeStart(algTag) != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S269] 输出错误日志，记录EnforceLaunchTask当前阶段和相关参数。
        HCCL_ERROR("failed set batch mode, tag is %s.", algTag);
        // [中文导读] [AllReduce逐行 S270] 重新启动批模式失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S271] 结束条件if (HcommBatchModeStart(algTag) != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S272] 结束批模式强制提交当前任务，再恢复批模式处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S273] 结束EnforceLaunchTask函数体。
}

// [中文导读] 新流程的设备侧展开入口：设置超时和批传输能力，先排入等待Host输入就绪的通知任务。
// [中文导读] 随后按opType和algName创建executor并调用Orchestrate；算法模板在这里之后才生成搬运/同步任务。
// [中文导读] [AllReduce逐行 S277] 声明OpOrchestrate接口：配置超时/批传输能力，排入输入等待并按注册器调用executor::Orchestrate。
inline HcclResult
// [中文导读] [AllReduce逐行 S278] 声明OpOrchestrate接口：配置超时/批传输能力，排入输入等待并按注册器调用executor::Orchestrate。
OpOrchestrate(OpParam* param, const AlgResourceCtxSerializable* resCtxPtr, ThreadHandle thread, std::string& algName)
// [中文导读] [AllReduce逐行 S279] 开始OpOrchestrate的函数体。
{
    // RTSQ等待时间: 与算子展开无关, 但resCtx固定该设置不会再变更
    // [中文导读] 先应用资源上下文中的执行队列资源等待超时；是否可设置取决于运行时能力。
    // [中文导读] [AllReduce逐行 S282] 运行时支持时才设置线程资源获取超时。
    if (HcommIsSupportHcommThreadResAcquireTimeOut()) {
        // [中文导读] [AllReduce逐行 S283] 使用资源对象fullTimeout设置执行队列资源等待超时。
        CHK_RET(HcclThreadResAcquireTimeOut(resCtxPtr->fullTimeout));
    // [中文导读] [AllReduce逐行 S284] 结束条件if (HcommIsSupportHcommThreadResAcquireTimeOut())。
    }

    // NotifyWait等待时间: 只在算子展开过程中使用
    // [中文导读] [AllReduce逐行 S287] 运行时支持时才设置通知默认等待超时。
    if (HcommIsSupportHcommSetNotifyWaitTimeOut()) {
        // [中文导读] [AllReduce逐行 S288] 使用资源对象waitTimeout设置Device展开通知等待时长。
        CHK_RET(HcclSetNotifyWaitTimeOut(resCtxPtr->waitTimeout));
    // [中文导读] [AllReduce逐行 S289] 结束条件if (HcommIsSupportHcommSetNotifyWaitTimeOut())。
    }

    // 主thread等待Host stream的通知
    // [中文导读] 根据新旧 Thread 申请方式找出 Host 输入通知槽，旧接口需按各流最大通知数定位。
    // [中文导读] [AllReduce逐行 S293] 主线程内部通知需求作为Host输入通知槽的起点。
    u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
    // [中文导读] [AllReduce逐行 S294] 旧线程申请接口为所有线程申请同一最大通知容量，需重新求最大值定位槽。
    if (!resCtxPtr->isHcclThreadAcquireWithConfigSupported) {
        // [中文导读] [AllReduce逐行 S295] 遍历各从线程通知需求。
        for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
            // [中文导读] [AllReduce逐行 S296] 某个从线程需求更大时更新Host输入槽索引。
            if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
                // [中文导读] [AllReduce逐行 S297] 保存更大的通知需求，最后作为Host输入同步槽。
                maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
            // [中文导读] [AllReduce逐行 S298] 结束条件if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum)。
            }
        // [中文导读] [AllReduce逐行 S299] 结束循环for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++)。
        }
    // [中文导读] [AllReduce逐行 S300] 结束条件if (!resCtxPtr->isHcclThreadAcquireWithConfigSupported)。
    }
    // [中文导读] [AllReduce逐行 S301] 输出调试日志，记录OpOrchestrate当前阶段和相关参数。
    HCCL_DEBUG(
        // [中文导读] [AllReduce逐行 S302] 补充日志格式：[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u] s", __func__, thread, maxNotifyNum。
        "[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u] s", __func__, thread, maxNotifyNum,
        // [中文导读] [AllReduce逐行 S303] 提供上述日志的实参，涉及本次只读资源对象。
        resCtxPtr->waitTimeout);
    // [中文导读] 把输入就绪等待排入设备主 Thread，与 Host 在用户流中发送的通知配对。
    // [中文导读] [AllReduce逐行 S305] 在Device主线程排入Host输入就绪等待，匹配Host Record的最后一个槽。
    CHK_RET(HcclThreadNotifyWaitOnThreadDefault(thread, maxNotifyNum, resCtxPtr->waitTimeout));

    // 设置执行超时时间: 用于NotifyWait, 只在算子展开过程中使用
    // [中文导读] [AllReduce逐行 S308] 更新展开期间调用默认notify包装器使用的执行超时。
    ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);

    // 设置BatchTransfer是否可行: 只在算子展开过程中使用
    // [中文导读] 将 Host 记录的批传输能力应用到设备包装层，保证本次展开采用可用的传输接口。
    // [中文导读] [AllReduce逐行 S312] 根据Host探测到的批传输能力，初始化Device数据包装层能力标志。
    CHK_RET(InitHcommBatchTransferOnThreadSupported(resCtxPtr->isHcommBatchTransferOnThreadSupported));

    // 根据算法名字获取executor: 只用于算子展开
    // [中文导读] 用算子类型和算法名取得新 executor，再调用 Orchestrate 展开模板任务。
    // [中文导读] [AllReduce逐行 S316] 按算子ALLREDUCE和算法名获取新的具体执行器对象。
    std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
    // [中文导读] [AllReduce逐行 S317] 工厂返回空对象时无法编排，进入错误分支。
    if (executor.get() == nullptr) {
        // [中文导读] [AllReduce逐行 S318] 输出错误日志，记录OpOrchestrate当前阶段和相关参数。
        HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
        // [中文导读] [AllReduce逐行 S319] 缺失执行器返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S320] 结束条件if (executor.get() == nullptr)。
    }

    // 执行算法编排: 只用于算子展开
    // [中文导读] [AllReduce逐行 S323] 调用InsV2AllReduceSoleExecutor::Orchestrate，失败进入错误分支。
    if (executor->Orchestrate(*param, *resCtxPtr) != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S324] 输出错误日志，记录OpOrchestrate当前阶段和相关参数。
        HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
        // [中文导读] [AllReduce逐行 S325] 算法编排失败返回内部错误。
        return HCCL_E_INTERNAL;
    // [中文导读] [AllReduce逐行 S326] 结束条件if (executor->Orchestrate(*param, *resCtxPtr) != HCCL_SUCCESS)。
    }

    // [中文导读] [AllReduce逐行 S328] 配置超时/批传输能力，排入输入等待并按注册器调用executor::Orchestrate处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S329] 结束OpOrchestrate函数体。
}

// [中文导读] [AllReduce逐行 S331] 声明HcclOrderLaunchNotifyRecord接口：有效的Host/Device保序线程存在时记录入口已展开通知。
static HcclResult HcclOrderLaunchNotifyRecord(const OpParam* param)
// [中文导读] [AllReduce逐行 S332] 开始HcclOrderLaunchNotifyRecord的函数体。
{
    // [中文导读] [AllReduce逐行 S333] 读取Host保序线程导出到Device的句柄。
    ThreadHandle exportHostOrderThread = param->exportHostOrderThread;
    // [中文导读] [AllReduce逐行 S334] 读取Device展开保序线程句柄。
    ThreadHandle deviceOrderThread = param->deviceOrderThread;
    // [中文导读] [AllReduce逐行 S335] 输出运行日志，记录HcclOrderLaunchNotifyRecord当前阶段和相关参数。
    HCCL_INFO(
        // [中文导读] [AllReduce逐行 S336] 补充日志格式：[%s]. Before Notify1 Record, commName[%s], exportHostOrderThread is [0x%llx], deviceOrderThread is [0x%llx]。
        "[%s]. Before Notify1 Record, commName[%s], exportHostOrderThread is [0x%llx], deviceOrderThread is [0x%llx]",
        // [中文导读] [AllReduce逐行 S337] 提供上述日志的实参，涉及算子参数、通信域名字。
        __func__, param->commName, exportHostOrderThread, deviceOrderThread);

    // [中文导读] [AllReduce逐行 S339] 两个保序线程都有效时才记录保序通知。
    if (exportHostOrderThread != 0 && deviceOrderThread != 0) {
        // [中文导读] [AllReduce逐行 S340] 开始调用线程通知Record原语。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S341] 从Device保序线程发通知到Host保序线程的指定槽。
            HcommThreadNotifyRecordOnThread(deviceOrderThread, exportHostOrderThread, HOST_ORDER_THREAD_NOTIFY_IDX)));
        // [中文导读] [AllReduce逐行 S342] 输出运行日志，记录HcclOrderLaunchNotifyRecord当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S343] 补充日志格式：[%s]. After Notify1 Record deviceOrderThread is [0x%llx], exportHostOrderThread is [0x%llx]", __func__。
            "[%s]. After Notify1 Record deviceOrderThread is [0x%llx], exportHostOrderThread is [0x%llx]", __func__,
            // [中文导读] [AllReduce逐行 S344] 提供上述日志的实参：deviceOrderThread, exportHostOrderThread。
            deviceOrderThread, exportHostOrderThread);
    // [中文导读] [AllReduce逐行 S345] 结束条件if (exportHostOrderThread != 0 && deviceOrderThread != 0)。
    }

    // [中文导读] [AllReduce逐行 S347] 有效的Host/Device保序线程存在时记录入口已展开通知处理完成，返回成功。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S348] 结束HcclOrderLaunchNotifyRecord函数体。
}

// [中文导读] Host下发的AICPU入口，不是上层HcclAlltoAll API本身。param携带算子参数及设备可读的资源描述。
// [中文导读] 新流程依次取得通信域占用保护、恢复资源/变长参数、展开或回放任务，最后排入完成通知并清除占用标记。
// [中文导读] 下文还保留旧流程；不要将两个分支连成同一次算子的必经步骤。错误路径会提前返回。
// [中文导读] [AllReduce逐行 S353] 声明HcclLaunchAicpuKernel接口：Device入口：获取通信域占用标记、恢复资源、回放/展开任务、排入完成通知并清除占用标记。
extern "C" unsigned int HcclLaunchAicpuKernel(OpParam* param)
// [中文导读] [AllReduce逐行 S354] 开始HcclLaunchAicpuKernel的函数体。
{
    // 修改当前进程的调度策略和优先级
    // [中文导读] 入口先恢复普通调度策略，失败则结束本次 kernel 调用。
    // [中文导读] [AllReduce逐行 S357] 创建Linux调度参数结构。
    struct sched_param schedParam;
    // [中文导读] [AllReduce逐行 S358] 调度优先级设为0，恢复普通SCHED_OTHER。
    schedParam.sched_priority = 0; // 设置优先级为0
    // [中文导读] [AllReduce逐行 S359] 让当前AICPU进程使用普通调度策略，失败时结束入口。
    if (sched_setscheduler(0, SCHED_OTHER, &schedParam) == -1) {
        // [中文导读] [AllReduce逐行 S360] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_ERROR("%s sched_setscheduler to SCHED_OTHER failed", __func__);
        // [中文导读] [AllReduce逐行 S361] 调度策略设置失败返回kernel失败码1。
        return 1;
    // [中文导读] [AllReduce逐行 S362] 结束条件if (sched_setscheduler(0, SCHED_OTHER, &schedParam) == -1)。
    }
    // [中文导读] 参数非空后才能读取通信域和算法标识；通信域占用保护在算法工作开始前取得。
    // [中文导读] [AllReduce逐行 S364] 检查Host传入kernel参数地址非空。
    if (param == nullptr) {
        // [中文导读] [AllReduce逐行 S365] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_ERROR("%s param is nullptr", __func__);
        // [中文导读] [AllReduce逐行 S366] 参数地址为空返回kernel失败码1。
        return 1;
    // [中文导读] [AllReduce逐行 S367] 结束条件if (param == nullptr)。
    }
    // [中文导读] [AllReduce逐行 S368] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    // [中文导读] [AllReduce逐行 S369] 按通信域名字取得HCOMM设备侧域占用标记。
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S370] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        // [中文导读] [AllReduce逐行 S371] 取得通信域占用标记失败返回kernel失败码1。
        return 1;
    // [中文导读] [AllReduce逐行 S372] 结束条件if (HcommAcquireComm(param->commName) != HCCL_SUCCESS)。
    }

    // AICPU 按序下发
    // [中文导读] 通知 Host 保序 Thread 本次 kernel 已进入展开阶段，配合 Host 的两阶段保序逻辑。
    // [中文导读] [AllReduce逐行 S376] 通知Host保序线程本次Device kernel已进入展开阶段。
    CHK_RET(HcclOrderLaunchNotifyRecord(param));

    // [中文导读] [AllReduce逐行 S378] 从固定长度param.algName构造算法名字符串。
    std::string algName = std::string(param->algName);
    // [中文导读] 兼容流程需要 Scatter 维测描述；新流程稍后使用独立的算子信息转换。
    // [中文导读] [AllReduce逐行 S380] 非OpsV2兼容流程先生成Scatter维测对象；950主例跳过。
    if (!ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // [中文导读] [AllReduce逐行 S381] 声明旧流程Scatter维测结构。
        ScatterOpInfo opInfo;
        // [中文导读] [AllReduce逐行 S382] 从OpParam构造旧流程Scatter算子信息。
        if (CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S383] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("%s CreateScatter fail", __func__);
            // [中文导读] [AllReduce逐行 S384] 旧流程Scatter元信息创建失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S385] 结束条件if (CreateScatter(param, &opInfo) != HCCL_SUCCESS)。
        }

        // [中文导读] [AllReduce逐行 S387] 运行时有旧维测接口时登记Scatter算子信息。
        if (HcommIsSupportHcommRegOpInfo()
            // [中文导读] [AllReduce逐行 S388] 把ScatterOpInfo结构地址/长度注册到通信域。
            && HcommRegOpInfo(param->commName, reinterpret_cast<void*>(&opInfo), sizeof(ScatterOpInfo))
                   // [中文导读] [AllReduce逐行 S389] 维测注册失败进入错误分支。
                   != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S390] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S391] 补充日志格式：%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%zu]", __func__, param->commName, opInfo.algTag。
                "%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%zu]", __func__, param->commName, opInfo.algTag,
                // [中文导读] [AllReduce逐行 S392] 提供上述日志的实参：sizeof(ScatterOpInfo。
                sizeof(ScatterOpInfo));
            // [中文导读] [AllReduce逐行 S393] 旧流程维测注册失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S394] 结束代码块。
        }

        // [中文导读] [AllReduce逐行 S396] 运行时支持异常回调登记时注册旧流程Scatter异常信息查询。
        if (HcommIsSupportHcommRegOpTaskException()
            // [中文导读] [AllReduce逐行 S397] 将GetScatterOpInfo作为任务异常查询回调登记到通信域。
            && HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S398] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S399] 补充日志格式：%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag)。
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            // [中文导读] [AllReduce逐行 S400] 异常回调注册失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S401] 结束代码块。
        }
    // [中文导读] [AllReduce逐行 S402] 结束条件if (!ops_hccl::IsOpsV2(param->algName, param->deviceType))。
    }

    // [中文导读] [AllReduce逐行 S404] opv2_或out-place芯片采用新资源格式和执行器注册器。
    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        // [中文导读] 新流程在展开任务前查询通信域状态；暂停中清除本次占用标记并返回专用暂停错误。
        // [中文导读] [AllReduce逐行 S407] 通信域状态查询结果初始化为INVALID。
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        // [中文导读] [AllReduce逐行 S408] 运行时有状态查询能力时检查通信域状态。
        if (HcommIsSupportHcclCommGetStatus()) {
            // [中文导读] [AllReduce逐行 S409] 查询通信域当前状态并保留返回码。
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            // [中文导读] [AllReduce逐行 S410] 通信域状态查询失败时直接结束入口。
            if (statusRet != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S411] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                // [中文导读] [AllReduce逐行 S412] 状态查询失败返回kernel失败码1；本错误路径未走尾部ReleaseComm。
                return 1;
            // [中文导读] [AllReduce逐行 S413] 结束条件if (statusRet != HCCL_SUCCESS)。
            }
            // [中文导读] [AllReduce逐行 S414] 通信域正在暂停时走专用暂停处理。
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                // [中文导读] [AllReduce逐行 S415] 清除本次取得的通信域占用标记。
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    // [中文导读] [AllReduce逐行 S416] 输出警告日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                // [中文导读] [AllReduce逐行 S417] 上述条件不成立时进入替代分支。
                } else {
                    // [中文导读] [AllReduce逐行 S418] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                    HCCL_ERROR(
                        // [中文导读] [AllReduce逐行 S419] 补充日志格式：%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName)。
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                // [中文导读] [AllReduce逐行 S420] 结束条件} else。
                }
                // [中文导读] [AllReduce逐行 S421] 暂停状态返回301专用错误码。
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            // [中文导读] [AllReduce逐行 S422] 结束条件if (commStatus == HCCL_COMM_STATUS_SUSPENDING)。
            }
            // [中文导读] [AllReduce逐行 S423] 状态非READY时阻止算法展开。
            if (commStatus != HCCL_COMM_STATUS_READY) {
                // [中文导读] [AllReduce逐行 S424] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                // [中文导读] [AllReduce逐行 S425] 非就绪状态返回kernel失败码1；不能假定提前返回已经释放占用标记。
                return 1;
            // [中文导读] [AllReduce逐行 S426] 结束条件if (commStatus != HCCL_COMM_STATUS_READY)。
            }
        // [中文导读] [AllReduce逐行 S427] 结束条件if (HcommIsSupportHcclCommGetStatus())。
        }

        // [中文导读] 此缓存保存反序列化后的资源上下文，与下方缓存SQE/task的机制不同；命中仍需校验可复用性。
        // [中文导读] [AllReduce逐行 S430] 用shared_ptr持有缓存资源对象，确保本次使用期间缓存对象存在。
        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        // [中文导读] [AllReduce逐行 S431] 创建独占资源恢复对象指针，供cache miss时持有。
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        // [中文导读] [AllReduce逐行 S432] 初始有效资源对象指针为空，后面由缓存或反序列化设置。
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        // [中文导读] [AllReduce逐行 S433] 命中率日志按100倍转为百分数。
        u32 hitRateNum = 100;
        // [中文导读] [AllReduce逐行 S434] 非BatchSendRecv可尝试资源对象缓存；普通AllReduce满足。
        if (param->opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // 通过缓存实现反序列化优化
            // [中文导读] 按通信域与算法 tag 查反序列化缓存，并检查资源仍适用于当前参数。
            // [中文导读] [AllReduce逐行 S437] 按算法tag和通信域查询反序列化资源对象缓存。
            cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
            // [中文导读] [AllReduce逐行 S438] 对象存在且仍匹配本次通信域资源时才接受缓存。
            if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
                // [中文导读] [AllReduce逐行 S439] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
                // [中文导读] [AllReduce逐行 S440] 为命中率统计从tag提取通信域名字。
                std::string commName = g_cacheManager.ExtractCommName(param->algTag);
                // [中文导读] [AllReduce逐行 S441] tag无法解析通信域名字时使用显式域名。
                if (commName.empty())
                    // [中文导读] [AllReduce逐行 S442] 用本次param.commName作为统计域名字。
                    commName = param->commName;

                // [中文导读] [AllReduce逐行 S444] 创建接收缓存统计值的结构。
                CacheStats stats;
                // [中文导读] [AllReduce逐行 S445] 声明接收缓存条数的变量。
                size_t cacheSize;
                // [中文导读] [AllReduce逐行 S446] 取得该通信域统计成功时打印命中率和缓存条数。
                if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                    // [中文导读] [AllReduce逐行 S447] 输出调试日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                    HCCL_DEBUG(
                        // [中文导读] [AllReduce逐行 S448] 补充日志格式：[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str()。
                        "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                        // [中文导读] [AllReduce逐行 S449] 提供上述日志的实参：stats.hitRate() * hitRateNum, cacheSize。
                        stats.hitRate() * hitRateNum, cacheSize);
                // [中文导读] [AllReduce逐行 S450] 结束条件if (g_cacheManager.GetCommStats(commName, stats, cacheSize))。
                }
                // [中文导读] 命中后使用 shared_ptr 持有的只读资源对象，保证本次展开期间缓存对象存活。
                // [中文导读] [AllReduce逐行 S452] 将shared_ptr持有的只读资源对象设为本次有效资源。
                resCtxPtr = cachedResCtxHolder.get();
            // [中文导读] [AllReduce逐行 S453] 上述条件不成立时进入替代分支。
            } else {
                // [中文导读] [AllReduce逐行 S454] 记录是否有旧对象但已不可复用，用于陈旧缓存日志。
                bool isStaleCache = (cachedResCtxHolder != nullptr);
                // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
                // [中文导读] 未命中或陈旧缓存时恢复 Host 下发的序列化资源，并刷新设备侧资源对象缓存。
                // [中文导读] [AllReduce逐行 S457] 从Host下发的资源序列化地址/长度恢复新对象。
                resCtx = DeserializeResCtx(param);
                // [中文导读] [AllReduce逐行 S458] 用新资源对象刷新Device通信域资源对象缓存。
                g_cacheManager.Put(param->algTag, *resCtx, param->commName);
                // [中文导读] [AllReduce逐行 S459] 使用刚反序列化的对象作为本次资源。
                resCtxPtr = resCtx.get();
                // [中文导读] [AllReduce逐行 S460] 原来已有对象但无效时打印STALE日志。
                if (isStaleCache) {
                    // [中文导读] [AllReduce逐行 S461] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                    HCCL_INFO(
                        // [中文导读] [AllReduce逐行 S462] 补充日志格式：[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__。
                        "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                        // [中文导读] [AllReduce逐行 S463] 提供上述日志的实参，涉及算子参数、算法关联tag。
                        param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
                // [中文导读] [AllReduce逐行 S464] 上述条件不成立时进入替代分支。
                } else {
                    // [中文导读] [AllReduce逐行 S465] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                    HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
                // [中文导读] [AllReduce逐行 S466] 结束条件} else。
                }
            // [中文导读] [AllReduce逐行 S467] 结束条件} else。
            }
        // [中文导读] [AllReduce逐行 S468] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S469] BatchSendRecv不使用该普通资源对象缓存，每次恢复序列化资源。
            resCtx = DeserializeResCtx(param);
            // [中文导读] [AllReduce逐行 S470] 把新恢复对象设置为本次有效资源。
            resCtxPtr = resCtx.get();
        // [中文导读] [AllReduce逐行 S471] 结束条件} else。
        }

        // [中文导读] AllToAll系列恢复counts/displacements等描述的可用指针，不是在这里交换各Rank的用户数据。
        // 还原变长指针
        // [中文导读] [AllReduce逐行 S475] 初始化变长参数恢复返回码为成功。
        HcclResult ret = HCCL_SUCCESS;
        // [中文导读] 按算子类型恢复尾部描述的内部指针；BatchSendRecv 与变长集合通信采用不同布局。
        // [中文导读] [AllReduce逐行 S477] BatchSendRecv才重建sendRecvItems指针。
        if (param->opType == HCCL_CMD_BATCH_SEND_RECV) {
            // [中文导读] [AllReduce逐行 S478] 按BatchSendRecv尾部布局恢复items指针。
            ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
        // [中文导读] [AllReduce逐行 S479] 其它变长参数的算子继续检查对应类型。
        } else if (
            // [中文导读] [AllReduce逐行 S480] AllToAllV或VC需要恢复counts/displacements。
            param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
            // [中文导读] [AllReduce逐行 S481] 固定AllToAll也使用同一套指针恢复逻辑。
            || param->opType == HCCL_CMD_ALLTOALL) {
            // [中文导读] [AllReduce逐行 S482] 恢复AllToAll变长描述中的各个内部指针。
            ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
        // [中文导读] [AllReduce逐行 S483] ReduceScatterV恢复vDataDes指针。
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            // [中文导读] [AllReduce逐行 S484] 恢复ReduceScatterV的counts/displacements指针。
            ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
        // [中文导读] [AllReduce逐行 S485] AllGatherV恢复vDataDes指针。
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            // [中文导读] [AllReduce逐行 S486] 恢复AllGatherV的counts/displacements指针；AllReduce无需这些恢复。
            ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
        // [中文导读] [AllReduce逐行 S487] 结束条件} else if (param->opType == HCCL_CMD_ALLGATHER_V)。
        }
        // [中文导读] [AllReduce逐行 S488] 变长参数恢复失败时结束入口。
        if (ret != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S489] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
            // [中文导读] [AllReduce逐行 S490] 返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S491] 结束条件if (ret != HCCL_SUCCESS)。
        }
        // 获取Device测主thread
        // [中文导读] 取资源列表中的主 Thread，并开启当前算法 tag 的批量下发模式。
        // [中文导读] [AllReduce逐行 S494] 资源线程列表第0项就是算法Device主线程。
        ThreadHandle thread = resCtxPtr->threads[0];
        // [中文导读] [AllReduce逐行 S495] 开始本算法tag的HCOMM批提交模式。
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S496] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            // [中文导读] [AllReduce逐行 S497] 批模式启动失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S498] 结束条件if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS)。
        }

        // 要在下第一个task之前上报
        // [中文导读] 在第一个执行任务之前注册维测信息，使后续任务异常可以关联本次算子。
        // [中文导读] [AllReduce逐行 S502] 创建新流程算子维测信息结构。
        HcclDfxOpInfoCompat dfxOpInfo{};
        // [中文导读] [AllReduce逐行 S503] 将OpParam转换为维测描述，失败终止。
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S504] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            // [中文导读] [AllReduce逐行 S505] 元信息转换失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S506] 结束条件if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS)。
        }
        // [中文导读] [AllReduce逐行 S507] 在通信域登记本次维测描述。
        if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void*>(&dfxOpInfo)) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S508] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            // [中文导读] [AllReduce逐行 S509] 维测注册失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S510] 结束条件if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void*>(&dfxOpInfo)) != HCCL_SUCCESS)。
        }

        // 上报上报mainstream数据,第一个任务
        // [中文导读] [AllReduce逐行 S513] 上报Device kernel开始任务到profiling。
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S514] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S515] 补充日志格式：%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread。
                "%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread,
                // [中文导读] [AllReduce逐行 S516] 提供上述日志的实参，涉及算子参数、通信域名字。
                param->commName);
            // [中文导读] [AllReduce逐行 S517] 开始profiling上报失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S518] 结束条件if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS)。
        }

        // [中文导读] [AllReduce逐行 S520] 读取Host用户流导出到AICPU_TS的线程句柄，末尾向它发完成通知。
        ThreadHandle exportedAicpuTsThread = param->opThread;

        // 检查aicpu task cache使能约束
        // [中文导读] 只有运行时提供查询能力且缓存策略允许时才尝试任务缓存；资源缓存命中不等于任务缓存命中。
        // [中文导读] [AllReduce逐行 S524] 默认不启用AICPU任务缓存。
        bool enableCache = false;
        // [中文导读] [AllReduce逐行 S525] 有task lookup能力时才运行缓存策略判断。
        if (HcommIsSupportHcommAicpuTsTaskCacheLookup()) {
            // [中文导读] [AllReduce逐行 S526] 根据本次算子、资源和配置检查是否可启用AICPU task/SQE缓存。
            CHK_RET(AicpuTaskCachePolicy::IsAicpuTaskCacheEnable(*param, *resCtxPtr, enableCache));
        // [中文导读] [AllReduce逐行 S527] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheLookup())。
        }

        // 打印算子信息用于调试
        // [中文导读] [AllReduce逐行 S530] 声明全局原子展开序号，仅供调试日志使用。
        static std::atomic<uint64_t> opUnfoldIdx{0};
        // [中文导读] [AllReduce逐行 S531] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S532] 补充日志格式：[HcclLaunchAicpuKernel] opUnfoldIdx[%llu] commName[%s] opType[%u] inputPtr[0x%016llx] inputSize[%llu]。
            "[HcclLaunchAicpuKernel] opUnfoldIdx[%llu] commName[%s] opType[%u] inputPtr[0x%016llx] inputSize[%llu] "
            // [中文导读] [AllReduce逐行 S533] 补充日志格式：outputPtr[0x%016llx] outputSize[%llu] opMode[%u] algName[%s] isZeroCopy[%d] opExpanMode[%u]。
            "outputPtr[0x%016llx] outputSize[%llu] opMode[%u] algName[%s] isZeroCopy[%d] opExpanMode[%u] "
            // [中文导读] [AllReduce逐行 S534] 补充日志格式：enableCache[%d]。
            "enableCache[%d]",
            // [中文导读] [AllReduce逐行 S535] 原子取当前展开序号并加1，日志记录域名与算子类型。
            opUnfoldIdx.fetch_add(1, std::memory_order_relaxed), param->commName, static_cast<uint32_t>(param->opType),
            // [中文导读] [AllReduce逐行 S536] 日志记录当前用户输入输出地址及长度。
            param->inputPtr, param->inputSize, param->outputPtr, param->outputSize,
            // [中文导读] [AllReduce逐行 S537] 日志记录单算子/图模式、算法名和零拷贝标志。
            static_cast<uint32_t>(param->opMode), param->algName, param->isZeroCopy,
            // [中文导读] [AllReduce逐行 S538] 日志记录展开模式及是否启用任务缓存。
            static_cast<uint32_t>(param->commOpExpansionMode), enableCache);

        // [中文导读] task缓存未命中才调用OpOrchestrate生成任务；命中则更新本次用户地址并回放，可能看不到模板调用。
        // [中文导读] [AllReduce逐行 S541] 任务缓存策略允许时走lookup/采集/回放分支。
        if (enableCache) { // 使能aicpu task cache
            // 注意: OpOrchestrate尚未调用, 首个NotifyWait与算子展开相关的task尚未生成, AicpuTsThread中一定无SQE
            // 因此, 无需通过强制下发SQE, 来避免cache miss下缓存算法无关的task 或 cache hit下task下发乱序

            // 准备地址信息 (当前rank的userIn和userOut)
            // [中文导读] 任务缓存以本次输入和输出地址及有效容量更新重放任务，缓存不固定绑定首次用户地址。
            // [中文导读] [AllReduce逐行 S547] 本次任务重放需要更新两个地址：用户输入和用户输出。
            constexpr uint64_t ADDRS_COUNT = 2;
            // [中文导读] [AllReduce逐行 S548] 构造本次用户输入/输出地址数组。
            void* addrs[ADDRS_COUNT] = {param->inputPtr, param->outputPtr};
            // [中文导读] [AllReduce逐行 S549] 初始化输入有效字节长度为零。
            uint64_t inputSize = 0;
            // [中文导读] [AllReduce逐行 S550] 初始化输出有效字节长度为零。
            uint64_t outputSize = 0;
            // [中文导读] [AllReduce逐行 S551] 开始计算任务缓存使用的用户区有效长度。
            CHK_RET(static_cast<HcclResult>(AicpuTaskCacheUtils::GetInputOutputInfoForCache(
                // [中文导读] [AllReduce逐行 S552] 按算子类型和通信域rank数计算输入输出区长度。
                *param, resCtxPtr->topoInfo.userRankSize, inputSize, outputSize)));
            // [中文导读] [AllReduce逐行 S553] 构造与两个用户地址配对的有效长度数组。
            uint64_t sizes[ADDRS_COUNT] = {inputSize, outputSize};

            // [中文导读] [AllReduce逐行 S555] 创建本次AICPU任务缓存tag字符串。
            std::string cacheTag;
            // [中文导读] [AllReduce逐行 S556] 初始化任务缓存尚未命中标志。
            bool isCacheHit = false;
            // 组装aicpu task cache tag
            // [中文导读] 按本次参数及输入容量生成任务缓存键，再查询是否已有可重放任务。
            // [中文导读] [AllReduce逐行 S559] 根据算子参数和输入大小构造任务缓存key。
            CHK_RET(AicpuTaskCacheKey::GetAicpuTaskCacheTag(*param, inputSize, cacheTag));

            // 查询aicpu task cache
            // [中文导读] [AllReduce逐行 S562] 运行时有缓存lookup接口时查询已有任务缓存。
            if (HcommIsSupportHcommAicpuTsTaskCacheLookup()) {
                // [中文导读] [AllReduce逐行 S563] 查询该cacheTag是否存在可重放任务，写入isCacheHit。
                CHK_RET(static_cast<HcclResult>(HcommAicpuTsTaskCacheLookup(cacheTag.c_str(), &isCacheHit)));
            // [中文导读] [AllReduce逐行 S564] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheLookup())。
            }
            // [中文导读] [AllReduce逐行 S565] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_INFO("[HcclLaunchAicpuKernel] isCacheHit[%d] for cacheTag[%s]", isCacheHit, cacheTag.c_str());

            // [中文导读] 任务缓存未命中时围绕算法展开建立一次采集区间，命中则走后面的刷新重放分支。
            // [中文导读] [AllReduce逐行 S568] 任务缓存未命中时展开并采集本次算法任务。
            if (!isCacheHit) { // cache miss
                // [中文导读] [AllReduce逐行 S569] 初始化整个任务采集流程返回码。
                HcclResult cacheRet = HCCL_SUCCESS;
                // [中文导读] [AllReduce逐行 S570] 使用单次do区块，使失败宏break后集中清理当前cacheTag。
                do {
                    // 算子展开前, 通知aicpu task cache开始缓存task
                    // [中文导读] [AllReduce逐行 S572] 支持缓存采集接口时开启采集。
                    if (HcommIsSupportHcommAicpuTsTaskCacheStart()) {
                        // [中文导读] [AllReduce逐行 S573] 声明/更新缓存采集开始返回码，下一行实际调用。
                        cacheRet = static_cast<HcclResult>(
                            // [中文导读] [AllReduce逐行 S574] 启动任务采集并登记本次用户地址与有效容量。
                            HcommAicpuTsTaskCacheStart(cacheTag.c_str(), addrs, sizes, ADDRS_COUNT));
                        // [中文导读] [AllReduce逐行 S575] 开始条件检查宏，条件成立时输出错误并break退出单次采集区块。
                        CHK_PRT_BREAK(
                            // [中文导读] [AllReduce逐行 S576] 缓存采集开始失败时跳出单次do区块。
                            cacheRet != HCCL_SUCCESS,
                            // [中文导读] [AllReduce逐行 S577] 记录采集开始错误并通过break进入失败清理。
                            HCCL_ERROR("[%s] HcommAicpuTsTaskCacheStart error, ret[%d]", __func__, cacheRet), (void)0);
                    // [中文导读] [AllReduce逐行 S578] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheStart())。
                    }

                    // 设置算子展开相关的配置, 下发首个NotifyWait, 构造executor并执行算子展开
                    // [中文导读] 采集区间内完成输入等待和算法任务生成，确保缓存覆盖算子展开相关的任务。
                    // [中文导读] [AllReduce逐行 S582] 排入输入等待并调用具体执行器编排，当前任务属于采集区间。
                    cacheRet = OpOrchestrate(param, resCtxPtr, thread, algName);
                    // [中文导读] [AllReduce逐行 S583] 开始条件检查宏，条件成立时输出错误并break退出单次采集区块。
                    CHK_PRT_BREAK(
                        // [中文导读] [AllReduce逐行 S584] 编排失败时记录错误，跳出本次采集区块。
                        cacheRet != HCCL_SUCCESS, HCCL_ERROR("[%s] OpOrchestrate error, ret[%d]", __func__, cacheRet),
                        // [中文导读] [AllReduce逐行 S585] break宏的附加动作为空，本行结束该编排失败检查。
                        (void)0);

                    // 使用aicpu task cache后确保算子展开相关的SQE通过LaunchTask被缓存, 用于cache
                    // miss下避免缓存算法无关的task 注意: cache hit时, task刷新后直接下发, 这里无需强制下发 注意:
                    // hccl无法识别cache容量是否已满; 理论上如果cache容量满了, cache不使能, 无需强制下发
                    // (仅首次执行触发, 开销有限)
                    // [中文导读] 强制提交本次展开的待下发任务，使缓存采集不夹带后面的算法无关任务。
                    // [中文导读] [AllReduce逐行 S592] 结束再启动批模式，强制提交本次算法展开任务，使其被缓存采集。
                    cacheRet = EnforceLaunchTask(param->algTag);
                    // [中文导读] [AllReduce逐行 S593] 开始条件检查宏，条件成立时输出错误并break退出单次采集区块。
                    CHK_PRT_BREAK(
                        // [中文导读] [AllReduce逐行 S594] 强制下发失败时退出采集区间。
                        cacheRet != HCCL_SUCCESS,
                        // [中文导读] [AllReduce逐行 S595] 记录强制提交错误并跳到集中清理。
                        HCCL_ERROR("[%s] EnforceLaunchTask error, ret[%d]", __func__, cacheRet), (void)0);

                    // 算子展开后, 通知aicpu task cache停止缓存task
                    // [中文导读] 任务采集结束后关闭缓存区间；停止采集和设备任务执行完成是不同阶段。
                    // [中文导读] [AllReduce逐行 S599] 支持采集结束接口时关闭本次cacheTag采集区间。
                    if (HcommIsSupportHcommAicpuTsTaskCacheEnd()) {
                        // [中文导读] [AllReduce逐行 S600] 结束cacheTag采集并保留返回码。
                        cacheRet = static_cast<HcclResult>(HcommAicpuTsTaskCacheEnd(cacheTag.c_str()));
                        // [中文导读] [AllReduce逐行 S601] 开始条件检查宏，条件成立时输出错误并break退出单次采集区块。
                        CHK_PRT_BREAK(
                            // [中文导读] [AllReduce逐行 S602] 采集结束失败时跳出单次区块。
                            cacheRet != HCCL_SUCCESS,
                            // [中文导读] [AllReduce逐行 S603] 记录结束采集错误，执行break进入失败清理。
                            HCCL_ERROR("[%s] HcommAicpuTsTaskCacheEnd error, ret[%d]", __func__, cacheRet), (void)0);
                    // [中文导读] [AllReduce逐行 S604] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheEnd())。
                    }
                // [中文导读] [AllReduce逐行 S605] 单次do区块结束，不形成循环。
                } while (0);

                // [中文导读] 采集或提交失败时清除这个 tag 的任务缓存，避免下次重放不完整内容。
                // [中文导读] [AllReduce逐行 S608] 任意缓存采集/编排/强制提交步骤失败则清理不完整任务缓存。
                if (UNLIKELY(cacheRet != HCCL_SUCCESS)) {
                    // [中文导读] [AllReduce逐行 S609] 有Clear接口时清理本次cacheTag。
                    if (HcommIsSupportHcommAicpuTsTaskCacheClear()) {
                        // [中文导读] [AllReduce逐行 S610] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                        HCCL_ERROR("[%s] cache submit error, clear tag[%s]", __func__, cacheTag.c_str());
                        // [中文导读] [AllReduce逐行 S611] 显式忽略Clear返回码，仍向上传递原cacheRet错误。
                        (void)HcommAicpuTsTaskCacheClear(cacheTag.c_str());
                    // [中文导读] [AllReduce逐行 S612] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheClear())。
                    }
                    // [中文导读] [AllReduce逐行 S613] 返回原任务采集流程错误；此提前返回不经过尾部ReleaseComm。
                    return cacheRet;
                // [中文导读] [AllReduce逐行 S614] 结束缓存采集失败清理分支。
                }

                // 首次缓存记录通信域与tag关系
                // [中文导读] 建立通信域到缓存 tag 的关系，供后续通信域缓存清理时找到相关任务。
                // [中文导读] [AllReduce逐行 S618] 将通信域和cacheTag绑定，供域恢复/销毁时清除关联缓存。
                AicpuTaskCacheCommManager::Instance().AddCommTagMap(param->hcclComm, cacheTag);
            // [中文导读] [AllReduce逐行 S619] 任务缓存命中分支使用刷新并重放，跳过OpOrchestrate。
            } else { // cache hit
                // 刷新并下发task
                // [中文导读] 命中后刷新用户地址并重放任务，因此此次调用不再执行模板展开。
                // [中文导读] [AllReduce逐行 S622] 只有Execute接口存在时调用任务重放。
                if (HcommIsSupportHcommAicpuTsTaskCacheExecute()) {
                    // [中文导读] [AllReduce逐行 S623] 开始任务缓存重放调用并将结果转成HcclResult。
                    CHK_RET(static_cast<HcclResult>(
                        // [中文导读] [AllReduce逐行 S624] 使用本次用户输入输出地址与容量更新并执行缓存任务。
                        HcommAicpuTsTaskCacheExecute(cacheTag.c_str(), addrs, sizes, ADDRS_COUNT)));
                // [中文导读] [AllReduce逐行 S625] 结束条件if (HcommIsSupportHcommAicpuTsTaskCacheExecute())。
                }
            // [中文导读] [AllReduce逐行 S626] 结束循环do。
            }
        // [中文导读] [AllReduce逐行 S627] 未启用任务缓存时直接重新展开。
        } else { // 不使能aicpu task cache
            // 设置算子展开相关的配置, 下发首个NotifyWait, 构造executor并执行算子展开
            // [中文导读] 未启用任务缓存时直接展开本次算法，仍然包含同样的输入依赖与执行配置。
            // [中文导读] [AllReduce逐行 S630] 配置本次展开状态、排入输入等待并执行具体算法Orchestrate。
            CHK_RET(OpOrchestrate(param, resCtxPtr, thread, algName));
        // [中文导读] [AllReduce逐行 S631] 结束条件if (param->opType == HCCL_CMD_BATCH_SEND_RECV)。
        }

        // [中文导读] 算法任务之后排入对用户流所导出Thread的通知，与Host侧的等待配对；API返回不等于设备已执行完。
        // [中文导读] 在算法尾部向导出的用户流 Thread 记录通知，使用户流后续消费输出保持先后关系。
        // [中文导读] [AllReduce逐行 S635] 用户流结果通知使用第0个通知槽。
        constexpr u32 DEFAULT_NOTIFY_IDX = 0;
        // [中文导读] [AllReduce逐行 S636] 输出调试日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_DEBUG(
            // [中文导读] [AllReduce逐行 S637] 补充日志格式：[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread。
            "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
            // [中文导读] [AllReduce逐行 S638] 提供上述日志的实参：exportedAicpuTsThread, DEFAULT_NOTIFY_IDX。
            exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
        // [中文导读] [AllReduce逐行 S639] 开始向用户流导出线程记录完成通知。
        CHK_RET(static_cast<HcclResult>(
            // [中文导读] [AllReduce逐行 S640] 从算法Device主线程通知用户流，保护随后读取输出的任务顺序。
            HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

        // 上报mainstream数据,最后一个任务
        // [中文导读] [AllReduce逐行 S643] 上报Device kernel结束任务profiling。
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S644] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S645] 补充日志格式：%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread。
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread,
                // [中文导读] [AllReduce逐行 S646] 提供上述日志的实参，涉及算子参数、通信域名字。
                param->commName);
            // [中文导读] [AllReduce逐行 S647] 结束profiling上报失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S648] 结束条件if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS)。
        }

        // [中文导读] [AllReduce逐行 S650] 上报Device算子profiling信息。
        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S651] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            // [中文导读] [AllReduce逐行 S652] 算子profiling上报失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S653] 结束条件if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS)。
        }

        // [中文导读] 结束本tag的批量提交模式；这是执行控制，不能替代数据完成所需的流/通知依赖。
        // [中文导读] [AllReduce逐行 S656] 结束本算法tag的批提交模式，提交尚未下发任务。
        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S657] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            // [中文导读] [AllReduce逐行 S658] 批提交结束失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S659] 结束条件if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS)。
        }
    // [中文导读] [AllReduce逐行 S660] 非OpsV2进入旧执行器流程，950主例不走。
    } else {
        // [中文导读] [AllReduce逐行 S661] 按算法名从旧注册器取得ExecutorBase对象。
        std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
        // [中文导读] [AllReduce逐行 S662] 旧执行器不存在时结束入口。
        if (executor.get() == nullptr) {
            // [中文导读] [AllReduce逐行 S663] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            // [中文导读] [AllReduce逐行 S664] 缺少旧执行器返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S665] 结束条件if (executor.get() == nullptr)。
        }
        // [中文导读] [AllReduce逐行 S666] 把旧资源地址解释为AlgResourceCtx格式。
        AlgResourceCtx* resCtx = reinterpret_cast<AlgResourceCtx*>(param->resCtx);
        // 获取Device测主thread
        // [中文导读] [AllReduce逐行 S668] 声明旧资源结构尾部的ThreadHandle数组指针。
        ThreadHandle* threadHandlePtr
            // [中文导读] [AllReduce逐行 S669] 从旧资源结构末尾偏移sizeof(AlgResourceCtx)定位线程数组。
            = reinterpret_cast<ThreadHandle*>(reinterpret_cast<u8*>(resCtx) + sizeof(AlgResourceCtx));
        // [中文导读] [AllReduce逐行 S670] 取第0项为旧流程Device主线程。
        ThreadHandle thread = threadHandlePtr[0];
        // [中文导读] [AllReduce逐行 S671] 从旧资源结构读取用户流导出的AICPU_TS线程。
        ThreadHandle exportedAicpuTsThread = resCtx->opThread;
        // [中文导读] [AllReduce逐行 S672] 读取旧资源主线程通知容量。
        u32 notifyNumOnMainThread = resCtx->notifyNumOnMainThread;
        // [中文导读] [AllReduce逐行 S673] 开启旧流程算法tag的批提交模式。
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S674] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            // [中文导读] [AllReduce逐行 S675] 旧流程批模式启动失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S676] 结束条件if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS)。
        }

        // [中文导读] [AllReduce逐行 S678] 用户流已导出时使用跨线程通知和旧profiling流程。
        if (exportedAicpuTsThread != 0) {
            // [中文导读] [AllReduce逐行 S679] 初始化旧流程所有执行线程的profiling上下文。
            if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S680] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to init Profiling");
                // [中文导读] [AllReduce逐行 S681] profiling初始化失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S682] 结束条件if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS)。
            }

            // 上报主流和第一个task  wait之前
            // [中文导读] [AllReduce逐行 S685] 上报旧流程主流和第一个任务。
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S686] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to report MainStream And FirstTask");
                // [中文导读] [AllReduce逐行 S687] 旧流程首任务上报失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S688] 结束条件if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS)。
            }

            // 主thread等待Host stream的通知
            // [中文导读] [AllReduce逐行 S691] 输出调试日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_DEBUG(
                // [中文导读] [AllReduce逐行 S692] 补充日志格式：[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u] s", __func__, thread。
                "[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u] s", __func__, thread,
                // [中文导读] [AllReduce逐行 S693] 提供上述日志的实参，涉及Device主线程通知容量。
                notifyNumOnMainThread, CUSTOM_TIMEOUT);
            // [中文导读] [AllReduce逐行 S694] 开始把旧流程输入等待结果转换为HcclResult。
            CHK_RET(
                // [中文导读] [AllReduce逐行 S695] 在旧主线程等待Host输入就绪通知。
                static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyNumOnMainThread, CUSTOM_TIMEOUT)));
        // [中文导读] [AllReduce逐行 S696] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S697] 用户流未导出时改为等待旧ACL notify资源。
            if (HcommAclrtNotifyWaitOnThread(thread, resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S698] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to wait notify[%d] from host main stream", resCtx->notifyIds[0]);
                // [中文导读] [AllReduce逐行 S699] 旧ACL输入通知等待失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S700] 结束条件if (HcommAclrtNotifyWaitOnThread(thread, resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS)。
            }
        // [中文导读] [AllReduce逐行 S701] 结束条件} else。
        }

        // 执行算法编排
        // [中文导读] [AllReduce逐行 S704] 调用旧ExecutorBase::Orchestrate编排本次算子。
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
            // [中文导读] [AllReduce逐行 S705] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            // [中文导读] [AllReduce逐行 S706] 旧算法编排失败返回kernel失败码1。
            return 1;
        // [中文导读] [AllReduce逐行 S707] 结束条件if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS)。
        }

        // [中文导读] [AllReduce逐行 S709] 用户流已导出时上报旧profiling并发送结果通知。
        if (exportedAicpuTsThread != 0) {
            // 上报device侧的op 附加信息
            // [中文导读] [AllReduce逐行 S711] 声明旧算子profiling附加信息结构。
            HcomProInfoTmp profInfo;
            // [中文导读] [AllReduce逐行 S712] 从入口算法类型字符串构造profiling使用的字符串对象。
            std::string algTypeStr(param->algTypeStr);
            // [中文导读] [AllReduce逐行 S713] 安全复制算法类型到profiling结构，失败结束入口。
            if (strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()) != EOK) {
                // [中文导读] [AllReduce逐行 S714] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("[%s] strcpy_s profInfo.algType failed.", __func__);
                // [中文导读] [AllReduce逐行 S715] 算法类型字符串复制失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S716] 结束条件if (strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()) != EOK)。
            }
            // [中文导读] [AllReduce逐行 S717] 安全复制通信域名到profiling结构。
            if (strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName) != EOK) {
                // [中文导读] [AllReduce逐行 S718] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("[%s] strcpy_s profInfo.commName failed.", __func__);
                // [中文导读] [AllReduce逐行 S719] 通信域名字复制失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S720] 结束条件if (strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName) != EOK)。
            }
            // [中文导读] [AllReduce逐行 S721] 保存通信域名字符串实际长度。
            profInfo.commNameLen = strlen(param->commName);
            // [中文导读] [AllReduce逐行 S722] 保存旧流程输入数据元素数量。
            profInfo.dataCount = param->DataDes.count;
            // [中文导读] [AllReduce逐行 S723] 把输入数据类型转成profiling使用的8位类型字段。
            profInfo.dataType = static_cast<uint8_t>(param->DataDes.dataType);
            // [中文导读] [AllReduce逐行 S724] 保存旧流程通信域rank数量。
            profInfo.rankSize = resCtx->topoInfo.userRankSize;
            // [中文导读] [AllReduce逐行 S725] 上报旧流程Device算子profiling信息，本行未检查返回码。
            HcommProfilingReportDeviceHcclOpInfo(profInfo);

            // 主thread通知Host stream
            // [中文导读] [AllReduce逐行 S728] 旧流程完成通知也使用用户线程槽0。
            constexpr u32 DEFAULT_NOTIFY_IDX = 0;
            // [中文导读] [AllReduce逐行 S729] 输出调试日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
            HCCL_DEBUG(
                // [中文导读] [AllReduce逐行 S730] 补充日志格式：[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread。
                "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
                // [中文导读] [AllReduce逐行 S731] 提供上述日志的实参：exportedAicpuTsThread, DEFAULT_NOTIFY_IDX。
                exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
            // [中文导读] [AllReduce逐行 S732] 开始旧流程用户流结果通知记录调用。
            CHK_RET(static_cast<HcclResult>(
                // [中文导读] [AllReduce逐行 S733] 从旧算法Device主线程发完成通知到用户流导出线程。
                HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

            // 上报主流和最后一个task 在notify之后
            // [中文导读] [AllReduce逐行 S736] 上报旧主流和最后一个任务。
            if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S737] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to report MainStream And LastTask");
                // [中文导读] [AllReduce逐行 S738] 旧末任务profiling上报失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S739] 结束条件if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS)。
            }

            // [中文导读] [AllReduce逐行 S741] 结束旧流程批模式提交剩余任务。
            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S742] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                // [中文导读] [AllReduce逐行 S743] 旧批模式结束失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S744] 结束条件if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS)。
            }

            // [中文导读] [AllReduce逐行 S746] 结束旧流程所有执行线程profiling上下文。
            if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S747] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to End Profiling");
                // [中文导读] [AllReduce逐行 S748] profiling结束失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S749] 结束条件if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS)。
            }
        // [中文导读] [AllReduce逐行 S750] 上述条件不成立时进入替代分支。
        } else {
            // [中文导读] [AllReduce逐行 S751] 无用户流导出时发送旧ACL notify通知Host。
            if (HcommAclrtNotifyRecordOnThread(thread, resCtx->notifyIds[1]) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S752] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed to record host main stream");
                // [中文导读] [AllReduce逐行 S753] 旧ACL结果通知记录失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S754] 结束条件if (HcommAclrtNotifyRecordOnThread(thread, resCtx->notifyIds[1]) != HCCL_SUCCESS)。
            }

            // [中文导读] [AllReduce逐行 S756] 结束无用户流导出场景的批提交模式。
            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                // [中文导读] [AllReduce逐行 S757] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                // [中文导读] [AllReduce逐行 S758] 批提交结束失败返回kernel失败码1。
                return 1;
            // [中文导读] [AllReduce逐行 S759] 结束条件if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS)。
            }
        // [中文导读] [AllReduce逐行 S760] 结束条件} else。
        }
    // [中文导读] [AllReduce逐行 S761] 结束条件} else。
    }

    // [中文导读] [AllReduce逐行 S763] 成功路径清除通信域占用标记，使恢复/销毁流程可继续使用该域。
    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        // [中文导读] [AllReduce逐行 S764] 输出错误日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        // [中文导读] [AllReduce逐行 S765] 通信域占用标记释放失败返回kernel失败码1。
        return 1;
    // [中文导读] [AllReduce逐行 S766] 结束条件if (HcommReleaseComm(param->commName) != HCCL_SUCCESS)。
    }
    // [中文导读] [AllReduce逐行 S767] 输出运行日志，记录HcclLaunchAicpuKernel当前阶段和相关参数。
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    // [中文导读] [AllReduce逐行 S768] kernel正常结束返回0；队列实际完成仍由流通知保证。
    return 0;
// [中文导读] [AllReduce逐行 S769] 结束HcclLaunchAicpuKernel函数体。
}

extern "C" unsigned int HcclLaunchP2pAicpuKernel(void* args)
{
    if (args == nullptr) {
        HCCL_ERROR("%s args is nullptr", __func__);
        return 1;
    }
    HcclP2pKernelParam* params = static_cast<HcclP2pKernelParam*>(args);
    ThreadHandle sendRecvThread = params->sendRecvThread;
    void* paramPtr = static_cast<void*>(&params->opParams[0]);
    OpParam* param = static_cast<OpParam*>(paramPtr);

    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }

    if (param->opType != HcclCMDType::HCCL_CMD_SEND && param->opType != HcclCMDType::HCCL_CMD_RECEIVE) {
        HCCL_ERROR("%s only support SEND/RECV, opType[%d]", __func__, static_cast<int>(param->opType));
        return 1;
    }

    HCCL_INFO(
        "Entry-%s, commName[%s], tag[%s], algTag[%s], opType[%d]", __func__, param->commName, param->tag, param->algTag,
        static_cast<int>(param->opType));
    // 保留通信域管理 - 保证生命周期安全
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    std::string algName = std::string(param->algName);
    // 根据算法名字获取executor
    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                } else {
                    HCCL_ERROR(
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                }
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            }
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        }

        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        u32 hitRateNum = 100;

        // 通过缓存实现反序列化优化
        cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
        if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
            HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
            std::string commName = g_cacheManager.ExtractCommName(param->algTag);
            if (commName.empty())
                commName = param->commName;

            CacheStats stats;
            size_t cacheSize;
            if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                HCCL_DEBUG(
                    "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                    stats.hitRate() * hitRateNum, cacheSize);
            }
            resCtxPtr = cachedResCtxHolder.get();
        } else {
            bool isStaleCache = (cachedResCtxHolder != nullptr);
            // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
            resCtx = DeserializeResCtx(param);
            g_cacheManager.Put(param->algTag, *resCtx, param->commName);
            resCtxPtr = resCtx.get();
            if (isStaleCache) {
                HCCL_INFO(
                    "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                    param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
            } else {
                HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
            }
        }

        // 获取Device测主thread
        ThreadHandle thread = resCtxPtr->threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 要在下第一个task之前上报
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }
        if (HcclDfxRegOpInfoByCommId(param->commName, (&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }

        // 上报mainstream数据,第一个任务
        if (HcommProfilingReportKernelStartTask(sendRecvThread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__,
                sendRecvThread, param->commName);
            return 1;
        }

        std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR(
                "Fail to find executor for algName[%s], opType[%d]", algName.c_str(), static_cast<int>(param->opType));
            HcommReleaseComm(param->commName);
            return 1;
        }

        ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);
        HcclResult ret = HCCL_SUCCESS;
        ret = executor->OrchestrateWithThread(*param, *resCtxPtr, sendRecvThread);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s, opType[%d]", param->algName, static_cast<int>(param->opType));
            HcommReleaseComm(param->commName);
            return 1;
        }
        // 上报mainstream数据,最后一个任务
        if (HcommProfilingReportKernelEndTask(sendRecvThread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__,
                sendRecvThread, param->commName);
            return 1;
        }
        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            return 1;
        }
        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        HCCL_ERROR(
            "%s P2P only support OpsV2, algName[%s], deviceType[%d]", __func__, param->algName,
            static_cast<int>(param->deviceType));
        HcommReleaseComm(param->commName);
        return 1;
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

HcclResult ops_hccl::RestoreVarDataBatchSendRecv(OpParam& param)
{
    u64 sendRecvItemSize = static_cast<u64>(sizeof(HcclSendRecvItem));
    u64 itemNum = static_cast<u64>(param.batchSendRecvDataDes.itemNum);
    if (param.varMemSize != itemNum * sendRecvItemSize) {
        HCCL_ERROR(
            "param.varMemSize[%lu] is not equal to itemNum[%lu] multiply [HcclSendRecvItem] size[%lu]."
            "Failed to restore end recv info for BatchSendRecv!",
            param.varMemSize, itemNum, sendRecvItemSize);
        return HCCL_E_PARA;
    }
    param.batchSendRecvDataDes.sendRecvItemsPtr = reinterpret_cast<HcclSendRecvItem*>(param.varData);
    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataAlltoAllV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    // [中文导读] 根据 AllToAll 或 VC 布局检查描述区的最小和最大容量，防止按非法长度恢复数组指针。
    u64 minVectorNum = ALL_TO_ALL_V_VECTOR_NUM;
    u64 maxVectorNum
        = (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC) ? ALL_TO_ALL_VC_VECTOR_NUM : ALL_TO_ALL_V_VECTOR_NUM;
    CHK_PRT_RET(
        param.varMemSize < minVectorNum * rankSize * sizeof(u64)
            || param.varMemSize > maxVectorNum * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataAlltoAllV] param.varMemSize [%llu] is invalid,"
            " minVectorNum is [%llu], maxVectorNum is [%llu], rankSize is [%llu], sizeof(u64) is [%llu],",
            param.varMemSize, minVectorNum, maxVectorNum, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    constexpr u32 ALL_TO_ALL_V_OFFSET_SCOUNTS = 0;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RECV_COUNTS = 1;
    constexpr u32 ALL_TO_ALL_V_OFFSET_SDISPLS = 2;
    constexpr u32 ALL_TO_ALL_V_OFFSET_RDISPLS = 3;
    constexpr u32 ALL_TO_ALL_VC_OFFSET_PEER_RDISPLS = 4;

    // [中文导读] 按固定段偏移重建四个数组，段内每个元素对应一个通信域 Rank。
    u64* data = reinterpret_cast<u64*>(param.varData);
    param.all2AllVDataDes.sendCounts = data;
    param.all2AllVDataDes.recvCounts = data + ALL_TO_ALL_V_OFFSET_RECV_COUNTS * rankSize;
    param.all2AllVDataDes.sdispls = data + ALL_TO_ALL_V_OFFSET_SDISPLS * rankSize;
    param.all2AllVDataDes.rdispls = data + ALL_TO_ALL_V_OFFSET_RDISPLS * rankSize;

    // [中文导读] 只有 VC 的第五段实际存在时恢复对端接收位移；四段参数仍保留普通中转路径。
    if (param.opType == HcclCMDType::HCCL_CMD_ALLTOALLVC
        && param.varMemSize == ALL_TO_ALL_VC_VECTOR_NUM * rankSize * sizeof(u64)) {
        param.all2AllVDataDes.peerRdispls = data + ALL_TO_ALL_VC_OFFSET_PEER_RDISPLS * rankSize;
    }

    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataReduceScatterV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%llu", rankSize);
    CHK_PRT_RET(
        param.varMemSize != REDUCE_SCATTER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataReduceScatterV] param.varMemSize [%llu] is invalid,"
            "REDUCE_SCATTER_V_VECTOR_NUM is [%llu], rankSize is [%llu], sizeof(u64) is [%zu],",
            param.varMemSize, REDUCE_SCATTER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    param.vDataDes.displs = data + rankSize;
    return HCCL_SUCCESS;
}

HcclResult ops_hccl::RestoreVarDataAllGatherV(OpParam& param, const AlgResourceCtxSerializable& resCtx)
{
    u64 rankSize = resCtx.topoInfo.userRankSize;
    HCCL_INFO("rankSize:%llu", rankSize);
    CHK_PRT_RET(
        param.varMemSize != ALL_GATHER_V_VECTOR_NUM * rankSize * sizeof(u64),
        HCCL_ERROR(
            "[RestoreVarDataAllGatherV] param.varMemSize [%llu] is invalid,"
            "ALL_GATHER_V_VECTOR_NUM is [%llu], rankSize is [%llu], sizeof(u64) is [%zu],",
            param.varMemSize, ALL_GATHER_V_VECTOR_NUM, rankSize, sizeof(u64)),
        HCCL_E_PARA);

    u64* data = reinterpret_cast<u64*>(param.varData);
    param.vDataDes.counts = data;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.counts[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.counts)[i]);
    }
    param.vDataDes.displs = data + rankSize;
    for (u64 i = 0; i < rankSize; i++) {
        HCCL_INFO("param.vDataDes.displs[%u]:%u", i, reinterpret_cast<u64*>(param.vDataDes.displs)[i]);
    }
    return HCCL_SUCCESS;
}

extern "C" unsigned int HcclLaunchAicpuKernelA3(OpParam* param)
{
    if (param == nullptr) {
        HCCL_ERROR("%s param is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, commName[%s], tag[%s], algTag[%s]", __func__, param->commName, param->tag, param->algTag);
    if (HcommAcquireComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommAcquireComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }

    std::string algName = std::string(param->algName);
    if (!ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        ScatterOpInfo opInfo;
        if (CreateScatter(param, &opInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("%s CreateScatter fail", __func__);
            return 1;
        }

        if (HcommIsSupportHcommRegOpInfo()
            && HcommRegOpInfo(param->commName, reinterpret_cast<void*>(&opInfo), sizeof(ScatterOpInfo))
                   != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpInfo fail, commName[%s], algTag[%s], size[%zu]", __func__, param->commName, opInfo.algTag,
                sizeof(ScatterOpInfo));
            return 1;
        }

        if (HcommIsSupportHcommRegOpTaskException()
            && HcommRegOpTaskException(param->commName, ops_hccl::GetScatterOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s HcommRegOpTaskException fail, commName[%s], algTag[%s]", __func__, param->commName, param->algTag);
            return 1;
        }
    }

    // 根据算法名字获取executor
    if (ops_hccl::IsOpsV2(param->algName, param->deviceType)) {
        // 判断通信域状态
        HcclCommStatus commStatus = HCCL_COMM_STATUS_INVALID;
        if (HcommIsSupportHcclCommGetStatus()) {
            auto statusRet = HcclCommGetStatus(param->commName, &commStatus);
            if (statusRet != HCCL_SUCCESS) {
                HCCL_ERROR("%s HcclCommGetStatus fail, commName[%s], ret = %d", __func__, param->commName, statusRet);
                return 1;
            }
            if (commStatus == HCCL_COMM_STATUS_SUSPENDING) {
                if (HcommReleaseComm(param->commName) == HCCL_SUCCESS) {
                    HCCL_WARNING("%s commStatus is suspending, release commName[%s]", __func__, param->commName);
                } else {
                    HCCL_ERROR(
                        "%s commStatus is suspending, HcommReleaseComm fail, commName[%s]", __func__, param->commName);
                }
                return 301U; /* 301U: AICPUSUSPENDING_ERROR */
            }
            if (commStatus != HCCL_COMM_STATUS_READY) {
                HCCL_ERROR("%s commStatus is not ready!, commStatus = %d", __func__, static_cast<int>(commStatus));
                return 1;
            }
        }

        std::shared_ptr<const AlgResourceCtxSerializable> cachedResCtxHolder;
        std::unique_ptr<AlgResourceCtxSerializable> resCtx;
        const AlgResourceCtxSerializable* resCtxPtr{nullptr};
        if (param->opType != HcclCMDType::HCCL_CMD_BATCH_SEND_RECV) {
            // 通过缓存实现反序列化优化
            cachedResCtxHolder = g_cacheManager.Get(param->algTag, param->commName);
            if (cachedResCtxHolder != nullptr && IsResCtxCacheReusable(*cachedResCtxHolder, *param)) {
                HCCL_INFO("[%s] Cache HIT for algTag[%s]", __func__, param->algTag);
                std::string commName = g_cacheManager.ExtractCommName(param->algTag);
                if (commName.empty())
                    commName = param->commName;

                CacheStats stats;
                size_t cacheSize;
                if (g_cacheManager.GetCommStats(commName, stats, cacheSize)) {
                    HCCL_DEBUG(
                        "[%s] comm[%s] hitRate=%.2f%%, cacheSize=%zu", __func__, commName.c_str(),
                        stats.hitRate() * PERCENTAGE_MULTIPLIER, cacheSize);
                }
                resCtxPtr = cachedResCtxHolder.get();
            } else {
                bool isStaleCache = (cachedResCtxHolder != nullptr);
                // 未命中或者通信域恢复后缓存失效，进行反序列化并存入缓存
                resCtx = DeserializeResCtx(param);
                g_cacheManager.Put(param->algTag, *resCtx, param->commName);
                resCtxPtr = resCtx.get();
                if (isStaleCache) {
                    HCCL_INFO(
                        "[%s] Cache STALE and refreshed for algTag[%s], cachedComm[%p], currentComm[%p]", __func__,
                        param->algTag, cachedResCtxHolder->commInfoPtr, param->hcclComm);
                } else {
                    HCCL_INFO("[%s] Cache MISS and stored for algTag[%s]", __func__, param->algTag);
                }
            }
        } else {
            resCtx = DeserializeResCtx(param);
            resCtxPtr = resCtx.get();
        }

        // 还原变长指针
        HcclResult ret = HCCL_SUCCESS;
        if (param->opType == HCCL_CMD_BATCH_SEND_RECV) {
            ret = ops_hccl::RestoreVarDataBatchSendRecv(*param);
        } else if (
            param->opType == HCCL_CMD_ALLTOALLV || param->opType == HCCL_CMD_ALLTOALLVC
            || param->opType == HCCL_CMD_ALLTOALL) {
            ret = ops_hccl::RestoreVarDataAlltoAllV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_REDUCE_SCATTER_V) {
            ret = ops_hccl::RestoreVarDataReduceScatterV(*param, *resCtxPtr);
        } else if (param->opType == HCCL_CMD_ALLGATHER_V) {
            ret = ops_hccl::RestoreVarDataAllGatherV(*param, *resCtxPtr);
        }
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("failed to restore optype [%d] data and counts.", param->opType);
            return 1;
        }
        // 获取Device测主thread
        ThreadHandle thread = resCtxPtr->threads[0];
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        // 要在下第一个task之前上报
        HcclDfxOpInfoCompat dfxOpInfo{};
        if (ConvertToHcclDfxOpInfo(param, &dfxOpInfo) != HCCL_SUCCESS) {
            HCCL_ERROR("ConvertToHcclDfxOpInfo fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }
        if (HcclDfxRegOpInfoByCommId(param->commName, reinterpret_cast<void*>(&dfxOpInfo)) != HCCL_SUCCESS) {
            HCCL_ERROR("HcclDfxRegOpInfoByCommId fail, commName is %s, tag is %s", param->commName, param->algTag);
            return 1;
        }

        // 上报上报mainstream数据,第一个任务
        if (HcommProfilingReportKernelStartTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%sfailed to report MainStream And FirstTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        // 主thread等待Host stream的通知
        ThreadHandle exportedAicpuTsThread = param->opThread;
        u32 maxNotifyNum = resCtxPtr->notifyNumOnMainThread;
        for (u32 i = 0; i < resCtxPtr->notifyNumPerThread.size(); i++) {
            if (resCtxPtr->notifyNumPerThread[i] > maxNotifyNum) {
                maxNotifyNum = resCtxPtr->notifyNumPerThread[i];
            }
        }
        HCCL_DEBUG(
            "[%s]Notify wait on thread[%llu], maxNotifyNum[%u], timeout[%u] s", __func__, thread, maxNotifyNum,
            CUSTOM_TIMEOUT);
        CHK_RET(static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, maxNotifyNum, CUSTOM_TIMEOUT)));

        std::shared_ptr<InsCollAlgBase> executor = CollAlgExecRegistryV2::Instance().GetAlgExec(param->opType, algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }

        // 设置执行超时时间
        ExecTimeoutManager::Instance().SetExecTimeout(param->opConfig.execTimeout);
        // 设置BatchTransfer是否可行
        CHK_RET(InitHcommBatchTransferOnThreadSupported(resCtxPtr->isHcommBatchTransferOnThreadSupported));
        // 执行算法编排
        if (executor->Orchestrate(*param, *resCtxPtr) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }

        // 上报mainstream数据,最后一个任务
        if (HcommProfilingReportKernelEndTask(thread, param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR(
                "%s failed to report MainStream And LastTask, thread %lu, param->commName %s.", __func__, thread,
                param->commName);
            return 1;
        }

        constexpr u32 DEFAULT_NOTIFY_IDX = 0;
        HCCL_DEBUG(
            "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
            exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
        CHK_RET(static_cast<HcclResult>(
            HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

        if (HcommProfilingReportDeviceOp(param->commName) != HCCL_SUCCESS) {
            HCCL_ERROR("%s HcommProfilingReportDeviceOp fail, commName[%s]", __func__, param->commName);
            return 1;
        }

        if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
            return 1;
        }
    } else {
        std::unique_ptr<ExecutorBase> executor = CollAlgExecRegistry::Instance().GetAlgExec(algName);
        if (executor.get() == nullptr) {
            HCCL_ERROR("Fail to find executor for algName[%s]", algName.c_str());
            return 1;
        }
        AlgResourceCtx* resCtx = reinterpret_cast<AlgResourceCtx*>(param->resCtx);
        // 获取Device测主thread
        ThreadHandle* threadHandlePtr
            = reinterpret_cast<ThreadHandle*>(reinterpret_cast<u8*>(resCtx) + sizeof(AlgResourceCtx));
        ThreadHandle thread = threadHandlePtr[0];
        ThreadHandle exportedAicpuTsThread = resCtx->opThread;
        u32 notifyNumOnMainThread = resCtx->notifyNumOnMainThread;
        if (HcommBatchModeStart(param->algTag) != HCCL_SUCCESS) {
            HCCL_ERROR("failed set batch mode, tag is %s.", param->algTag);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            if (HcommProfilingInit(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to init Profiling");
                return 1;
            }

            // 上报主流和第一个task  wait之前
            if (HcommProfilingReportMainStreamAndFirstTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And FirstTask");
                return 1;
            }

            // 主thread等待Host stream的通知
            HCCL_DEBUG(
                "[%s]Notify wait on thread[%llu], notifyNumOnMainThread[%u], timeout[%u] s", __func__, thread,
                notifyNumOnMainThread, CUSTOM_TIMEOUT);
            CHK_RET(
                static_cast<HcclResult>(HcommThreadNotifyWaitOnThread(thread, notifyNumOnMainThread, CUSTOM_TIMEOUT)));
        } else {
            if (HcommAclrtNotifyWaitOnThread(thread, resCtx->notifyIds[0], CUSTOM_TIMEOUT) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to wait notify[%d] from host main stream", resCtx->notifyIds[0]);
                return 1;
            }
        }

        // 执行算法编排
        if (executor->Orchestrate(*param, resCtx) != HCCL_SUCCESS) {
            HCCL_ERROR("orchestrate failed for alg:%s", param->algName);
            return 1;
        }

        if (exportedAicpuTsThread != 0) {
            // 上报device侧的op 附加信息
            HcomProInfoTmp profInfo;
            std::string algTypeStr(param->algTypeStr);
            if (strcpy_s(profInfo.algType, sizeof(profInfo.algType), algTypeStr.c_str()) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.algType failed.", __func__);
                return 1;
            }
            if (strcpy_s(profInfo.commName, sizeof(profInfo.commName), param->commName) != EOK) {
                HCCL_ERROR("[%s] strcpy_s profInfo.commName failed.", __func__);
                return 1;
            }
            profInfo.commNameLen = strlen(param->commName);
            profInfo.dataCount = param->DataDes.count;
            profInfo.dataType = static_cast<uint8_t>(param->DataDes.dataType);
            profInfo.rankSize = resCtx->topoInfo.userRankSize;
            HcommProfilingReportDeviceHcclOpInfo(profInfo);

            // 主thread通知Host stream
            constexpr u32 DEFAULT_NOTIFY_IDX = 0;
            HCCL_DEBUG(
                "[%s]Notify record on srcThread[%llu], dstThread[%llu], notifyIdx[%u]", __func__, thread,
                exportedAicpuTsThread, DEFAULT_NOTIFY_IDX);
            CHK_RET(static_cast<HcclResult>(
                HcommThreadNotifyRecordOnThread(thread, exportedAicpuTsThread, DEFAULT_NOTIFY_IDX)));

            // 上报主流和最后一个task 在notify之后
            if (HcommProfilingReportMainStreamAndLastTask(thread) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to report MainStream And LastTask");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }

            if (HcommProfilingEnd(threadHandlePtr, resCtx->slaveThreadNum + 1) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to End Profiling");
                return 1;
            }
        } else {
            if (HcommAclrtNotifyRecordOnThread(thread, resCtx->notifyIds[1]) != HCCL_SUCCESS) {
                HCCL_ERROR("failed to record host main stream");
                return 1;
            }

            if (HcommBatchModeEnd(param->algTag) != HCCL_SUCCESS) {
                HCCL_ERROR("failed set eager mode, tag is %s.", param->algTag);
                return 1;
            }
        }
    }

    if (HcommReleaseComm(param->commName) != HCCL_SUCCESS) {
        HCCL_ERROR("%s HcommReleaseComm fail, commName[%s]", __func__, param->commName);
        return 1;
    }
    HCCL_INFO("%s success, tag[%s], algTag[%s], commName[%s]", __func__, param->tag, param->algTag, param->commName);
    return 0;
}

extern "C" unsigned int HcclLaunchAicpuCacheEvictKernel(HcclComm* comm)
{
    if (comm == nullptr) {
        HCCL_ERROR("%s comm is nullptr", __func__);
        return 1;
    }
    HCCL_INFO("Entry-%s, comm[%p]", __func__, *comm);
    if (*comm != nullptr) {
        if (AicpuTaskCacheCommManager::Instance().EvictTaskCache(*comm) != HCCL_SUCCESS) {
            HCCL_ERROR("%s EvictTaskCache fail, comm[%p]", __func__, *comm);
            return 1;
        }
    }

    return 0;
}
