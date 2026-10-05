/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "endpoint_mgr.h"
#include <algorithm>
#include "hcomm_c_adpt.h"

namespace hccl {

TaggedMemMap::~TaggedMemMap()
{
    if (handle_ == nullptr) {
        return;
    }
    // [中文导读] 遍历此 Endpoint 的所有标签注册并逐项注销；某项失败只记录日志，仍继续清理其他项。
    for (const auto& kv : tagToHandle_) {
        HcommResult ret = HcommMemUnreg(handle_, kv.second);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[TaggedMemMap::~TaggedMemMap] HcommMemUnreg failed, handle[%p] tag[%s] ret[%d]", handle_,
                kv.first.c_str(), ret);
        }
    }
}

MemHandle TaggedMemMap::FindHandle(const std::string& tag) const
{
    auto it = tagToHandle_.find(tag);
    return it != tagToHandle_.end() ? it->second : nullptr;
}

bool TaggedMemMap::HasTag(const std::string& tag) const { return tagToHandle_.find(tag) != tagToHandle_.end(); }

void TaggedMemMap::EmplaceHandle(const std::string& tag, MemHandle handle) { tagToHandle_.emplace(tag, handle); }

MemHandle TaggedMemMap::RemoveTag(const std::string& tag)
{
    auto it = tagToHandle_.find(tag);
    if (it == tagToHandle_.end()) {
        return nullptr;
    }
    MemHandle handle = it->second;
    tagToHandle_.erase(it);
    return handle;
}

// [中文导读] 先清理端点关联的内存注册记录，再销毁端点，保留注销操作所需的有效Endpoint句柄。
// [中文导读] 资源由管理器缓存并统一释放，单次算子返回不代表这些可复用资源立即销毁。
EndpointMgr::~EndpointMgr()
{
    endpointTagMemMap_.clear();
    for (const auto& kv : endpointMap_) {
        const EndpointHandle& endpointHandle = kv.second;
        (void)HcommEndpointDestroy(endpointHandle);
    }
    // 销毁共享 jetty 场景按 tag 创建的独立 Endpoint
    for (const auto& kv : taggedEndpointMap_) {
        (void)HcommEndpointDestroy(kv.second);
    }
    taggedEndpointMap_.clear();
}

// [中文导读] 按EndpointDesc查询缓存。命中直接返回同一Endpoint，未命中才调用HcommEndpointCreate。
// [中文导读] 这正是域级“申请通道”未必触发“创建端点”的原因，测创建需显式准备未命中的场景。
// [中文导读] [AllReduce逐行 S72] EndpointMgr::Get的接口声明：当前对象句柄；这些参数属于本函数调用边界。
HcclResult EndpointMgr::Get(EndpointDesc epDesc, EndpointHandle& handle)
// [中文导读] [AllReduce逐行 S73] 进入EndpointMgr::Get函数体：在互斥锁保护下按 EndpointDesc 缓存获取或创建本地 Endpoint。
{
    // [中文导读] 同一把互斥锁覆盖查缓存、创建和插入，防止并发请求为同一端点描述重复建立资源。
    // [中文导读] [AllReduce逐行 S75] 调用lock；保持声明的局部对象用于后续处理。
    std::lock_guard<std::mutex> lock(mutex_);
    // [中文导读] [AllReduce逐行 S76] 设置auto iterPtr为/按`endpointMap_.find(epDesc)`；调用find。
    auto iterPtr = endpointMap_.find(epDesc);
    // [中文导读] [AllReduce逐行 S77] 仅当`(iterPtr != endpointMap_.end())`成立时进入此分支；调用end。
    if (iterPtr != endpointMap_.end()) {
        // [中文导读] [AllReduce逐行 S78] 设置当前对象句柄为/按`iterPtr->second`。
        handle = iterPtr->second;
        // [中文导读] [AllReduce逐行 S79] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S80] 结束`if (iterPtr != endpointMap_.end())`分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S81] 记录EndpointMgr::Get的状态/性能诊断；日志本身不执行传输。
    HCCL_INFO("[EndpointMgr::Get] create Endpoint");
    // [中文导读] [AllReduce逐行 S82] 调用基础资源接口创建本地Endpoint；返回非成功时由检查宏立即向上传递。
    CHK_RET(static_cast<HcclResult>(HcommEndpointCreate(&epDesc, &handle)));

    // [中文导读] [AllReduce逐行 S84] 调用emplace，使用当前对象句柄；传入/处理当前对象句柄。
    endpointMap_.emplace(epDesc, handle);
    // [中文导读] [AllReduce逐行 S85] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S86] 结束EndpointMgr::Get函数体；控制流返回外层。
}

HcclResult EndpointMgr::GetWithTag(EndpointDesc epDesc, const std::string& sharedQueueTag, EndpointHandle& handle)
{
    // tag 为空：退化为默认 Get，兼容非共享路径或无 tag 场景
    if (sharedQueueTag.empty()) {
        return Get(epDesc, handle);
    }

    // [中文导读] 共享队列 tag 与端点描述共同构成缓存键，允许同一地址按不同 tag 隔离端点资源。
    EndpointDescTagKey key{epDesc, sharedQueueTag};

    // 快路径：持锁查缓存，命中直接返回
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto iter = taggedEndpointMap_.find(key);
        if (iter != taggedEndpointMap_.end()) {
            handle = iter->second;
            return HCCL_SUCCESS;
        }
    }

    // 慢路径：持锁创建 + 二次检查。
    // 不采用"无锁创建+失败销毁"乐观模式：HcommEndpointCreate 涉及 device context 分配等重操作，
    // 高并发同 key 多线程重复创建+销毁的代价高于锁内串行等待；且 create/destroy 非严格幂等时可能残留状态。
    std::lock_guard<std::mutex> lock(mutex_);
    // 二次检查：另一线程可能已在快路径后、本线程拿锁前完成创建
    auto iter = taggedEndpointMap_.find(key);
    if (iter != taggedEndpointMap_.end()) {
        handle = iter->second;
        return HCCL_SUCCESS;
    }
    // 锁内创建：同一 key 不会有并发的重复创建
    CHK_RET(static_cast<HcclResult>(HcommEndpointCreate(&epDesc, &handle)));
    taggedEndpointMap_.emplace(std::move(key), handle);
    HCCL_INFO("[EndpointMgr::GetWithTag] create tagged Endpoint, tag[%s], handle[%p].", sharedQueueTag.c_str(), handle);
    return HCCL_SUCCESS;
}

// [中文导读] 将域内CCL等内存登记到具体Endpoint，得到后续建链可携带的MemHandle。
// [中文导读] 先比较域内存版本，再按tag跳过已注册项；注册描述不等于分配或复制用户内存。
// [中文导读] 只有当前批次顺利处理完才更新记录版本，HCCL_E_AGAIN可表示底层已有对应注册。
// [中文导读] [AllReduce逐行 S128] EndpointMgr::RegisterMemory的接口声明：按内存版本和 tag 复用端点注册句柄，整批完成后提交版本；这些参数属于本函数调用边界。
HcclResult EndpointMgr::RegisterMemory(
    // [中文导读] [AllReduce逐行 S129] EndpointMgr::RegisterMemory的接口声明：本地端点句柄、内存标签数组、域内注册内存描述数组；这些参数属于本函数调用边界。
    EndpointHandle epHandle, const std::vector<std::string>& memTag, const std::vector<HcclMem>& memVec,
    // [中文导读] [AllReduce逐行 S130] EndpointMgr::RegisterMemory的接口声明：域内存登记版本；这些参数属于本函数调用边界。
    uint64_t commMemsVersion)
// [中文导读] [AllReduce逐行 S131] 进入EndpointMgr::RegisterMemory函数体：按内存版本和 tag 复用端点注册句柄，整批完成后提交版本。
{
    // [中文导读] [AllReduce逐行 S132] 调用lock；保持声明的局部对象用于后续处理。
    std::lock_guard<std::mutex> lock(mutex_);
    // [中文导读] 按 Endpoint 创建或取得注册账本；每个端点分别保存标签句柄和已同步的域内存版本。
    // [中文导读] [AllReduce逐行 S134] 设置当前Endpoint的tag注册账本为/按`endpointTagMemMap_.try_emplace(epHandle, epHandle).first->second`（本地端点句柄）；调用try_emplace，使用本地端点句柄。
    auto& taggedMap = endpointTagMemMap_.try_emplace(epHandle, epHandle).first->second;

    // 版本一致，CommMems 无变更，跳过注册
    // [中文导读] [AllReduce逐行 S137] 仅当`(taggedMap.GetVersion() == commMemsVersion)`（当前Endpoint的tag注册账本的GetVersion字段、域内存登记版本）成立时进入此分支；调用GetVersion，使用当前Endpoint的tag注册账本的GetVersion字段、域内存登记版本。
    if (taggedMap.GetVersion() == commMemsVersion) {
        // [中文导读] [AllReduce逐行 S138] 记录EndpointMgr::RegisterMemory的状态/性能诊断；日志本身不执行传输。
        HCCL_INFO(
            // [中文导读] [AllReduce逐行 S139] 为当前EndpointMgr::RegisterMemory诊断/异常表达式提供格式文本，将报告域内存登记版本；这一物理行没有数据搬运副作用。
            "[%s]commMemsVersion[%llu] unchanged, skip registration, epHandle[%p]", __FUNCTION__, commMemsVersion,
            // [中文导读] [AllReduce逐行 S140] 为前述多行表达式补入`epHandle)`（本地端点句柄）；本行是参数/结构化初始化续行。
            epHandle);
        // [中文导读] [AllReduce逐行 S141] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
        return HCCL_SUCCESS;
    // [中文导读] [AllReduce逐行 S142] 结束`if (taggedMap.GetVersion() == commMemsVersion)`（当前Endpoint的tag注册账本的GetVersion字段、域内存登记版本）分支/循环；控制流返回外层。
    }
    // [中文导读] 标签数组必须覆盖全部内存项，保证下面按同一下标配对时不会读到缺失标签。
    // [中文导读] [AllReduce逐行 S144] 开始带日志的条件错误处理：后续实参提供触发条件、诊断与返回状态；命中条件才提前返回。
    CHK_PRT_RET(
        // [中文导读] [AllReduce逐行 S145] 向条件错误检查提供`memTag.size() < memVec.size(),`（内存标签数组的size字段、域内注册内存描述数组的size字段），用于确定触发条件或形成对应诊断。
        memTag.size() < memVec.size(),
        // [中文导读] [AllReduce逐行 S146] 记录EndpointMgr::RegisterMemory的错误诊断，字段包含内存标签数组的size字段、域内注册内存描述数组的size字段；日志本身不执行传输。
        HCCL_ERROR("[%s] memTag.size()[%zu] < memVec.size()[%zu]", __FUNCTION__, memTag.size(), memVec.size()),
        // [中文导读] [AllReduce逐行 S147] 记录EndpointMgr::RegisterMemory的状态/性能诊断；日志本身不执行传输。
        HCCL_E_PARA);

    // [中文导读] [AllReduce逐行 S149] 设置条目下标为/按`0`。
    size_t index = 0;
    // [中文导读] [AllReduce逐行 S150] 按`(const auto& mem : memVec)`（域内注册内存描述数组）遍历本批条目/分片；各次处理保持数组对应关系。
    for (const auto& mem : memVec) {
        // [中文导读] [AllReduce逐行 S151] 设置内存标签为/按`memTag[index]`（内存标签数组、条目下标）。
        const std::string& tag = memTag[index];
        // [中文导读] [AllReduce逐行 S152] 推进/回退`index++`（条目下标），更新当前分片、槽位或状态重试的计数。
        index++;
        // 检查tag是否已注册，避免重复注册
        // [中文导读] [AllReduce逐行 S154] 仅当`(taggedMap.HasTag(tag))`（当前Endpoint的tag注册账本的HasTag字段、内存标签）成立时进入此分支；判断该tag是否已登记从而避免重复注册。
        if (taggedMap.HasTag(tag)) {
            // [中文导读] [AllReduce逐行 S155] 记录EndpointMgr::RegisterMemory的状态/性能诊断，字段包含内存标签的c_str字段；日志本身不执行传输。
            HCCL_INFO("[%s]tag already registered, reuse existing handle, tag=%s", __FUNCTION__, tag.c_str());
            // [中文导读] [AllReduce逐行 S156] 跳过本轮剩余代码并进入下一项/下一次状态查询。
            continue;
        // [中文导读] [AllReduce逐行 S157] 结束`if (taggedMap.HasTag(tag))`（当前Endpoint的tag注册账本的HasTag字段、内存标签）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S158] 设置底层注册内存句柄为/按`nullptr`。
        MemHandle memHandle = nullptr;
        // [中文导读] 将域内存的类型、地址和字节大小组装成底层注册描述，注册作用域是当前 Endpoint。
        // [中文导读] [AllReduce逐行 S160] 准备交给底层注册的内存描述的局部存储/结构描述，初始化方式以本行声明为准。
        CommMem commMem{static_cast<CommMemType>(mem.type), mem.addr, mem.size};
        // [中文导读] [AllReduce逐行 S161] 设置当前调用状态为/按`static_cast<HcclResult>(HcommMemReg(epHandle, tag.c_str(), &commMem, &memHandle))`（本地端点句柄、内存标签的c_str字段、交给底层注册的内存描述、底层注册内存句柄）；向当前Endpoint登记地址/长度/token资源并返回注册句柄。
        HcclResult ret = static_cast<HcclResult>(HcommMemReg(epHandle, tag.c_str(), &commMem, &memHandle));
        // [中文导读] 仅 SUCCESS 或 AGAIN 允许继续；即便底层报告已有注册，也要求拿到有效句柄后才能记入账本。
        // [中文导读] [AllReduce逐行 S163] 仅当`(ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN)`（当前调用状态）成立时进入此分支。
        if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN) {
            // [中文导读] [AllReduce逐行 S164] 记录EndpointMgr::RegisterMemory的错误诊断，字段包含当前调用状态；日志本身不执行传输。
            HCCL_ERROR("[%s]call trace: hcclRet -> %d", __FUNCTION__, ret);
            // [中文导读] [AllReduce逐行 S165] 返回ret，表示保留当前调用的状态（可能成功或失败）；此路径停止本函数的后续处理。
            return ret;
        // [中文导读] [AllReduce逐行 S166] 结束`if (ret != HCCL_SUCCESS && ret != HCCL_E_AGAIN)`（当前调用状态）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S167] 检查`memHandle`（底层注册内存句柄）不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_PTR_NULL(memHandle);
        // [中文导读] [AllReduce逐行 S168] 将成功/已有的注册句柄登记到当前tag账本；传入/处理当前Endpoint的tag注册账本的EmplaceHandle字段、内存标签、底层注册内存句柄。
        taggedMap.EmplaceHandle(tag, memHandle); // 记录到tag映射，后续相同tag直接命中
        // [中文导读] [AllReduce逐行 S169] 仅当`(ret == HCCL_E_AGAIN)`（当前调用状态）成立时进入此分支。
        if (ret == HCCL_E_AGAIN) {
            // [中文导读] [AllReduce逐行 S170] 记录EndpointMgr::RegisterMemory的警告诊断；日志本身不执行传输。
            HCCL_WARNING("This mem has already been registered, addr=%p, size=%llu", mem.addr, mem.size);
        // [中文导读] [AllReduce逐行 S171] 结束`if (ret == HCCL_E_AGAIN)`（当前调用状态）分支/循环；控制流返回外层。
        }
    // [中文导读] [AllReduce逐行 S172] 结束`for (const auto& mem : memVec)`（域内注册内存描述数组）分支/循环；控制流返回外层。
    }

    // [中文导读] 只有全部待处理内存注册完成才提交版本；中途失败保留旧版本，以便后续请求重新检查。
    // [中文导读] [AllReduce逐行 S175] 提交已完成处理的域内存版本；传入/处理当前Endpoint的tag注册账本的SetVersion字段、域内存登记版本。
    taggedMap.SetVersion(commMemsVersion);
    // [中文导读] [AllReduce逐行 S176] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S177] 结束EndpointMgr::RegisterMemory函数体；控制流返回外层。
}

// [中文导读] 把上层选择的内存tag转换成此Endpoint下的注册句柄，保留请求顺序；不存在则返回错误。
// [中文导读] 这是“选择要交换哪些已注册内存”，不是让所有注册内存无条件出现在每条Channel上。
// [中文导读] [AllReduce逐行 S181] EndpointMgr::GetMemHandlesByTags的接口声明：按请求 tag 顺序取得端点注册句柄；失败可留下部分输出；这些参数属于本函数调用边界。
HcclResult EndpointMgr::GetMemHandlesByTags(
    // [中文导读] [AllReduce逐行 S182] EndpointMgr::GetMemHandlesByTags的接口声明：本地端点句柄、本通道需要交换的内存标签、本通道端点注册句柄列表；这些参数属于本函数调用边界。
    EndpointHandle epHandle, const std::vector<std::string>& memTags, std::vector<MemHandle>& memHandleVec)
// [中文导读] [AllReduce逐行 S183] 进入EndpointMgr::GetMemHandlesByTags函数体：按请求 tag 顺序取得端点注册句柄；失败可留下部分输出。
{
    // [中文导读] [AllReduce逐行 S184] 调用lock；保持声明的局部对象用于后续处理。
    std::lock_guard<std::mutex> lock(mutex_);
    // [中文导读] 清空本次出参后查找指定端点账本，避免把前次查询结果混入当前 Channel 的内存列表。
    // [中文导读] [AllReduce逐行 S186] 调用clear，使用本通道端点注册句柄列表的clear字段；传入/处理本通道端点注册句柄列表的clear字段。
    memHandleVec.clear();
    // [中文导读] [AllReduce逐行 S187] 设置auto it为/按`endpointTagMemMap_.find(epHandle)`（本地端点句柄）；调用find，使用本地端点句柄。
    auto it = endpointTagMemMap_.find(epHandle);
    // [中文导读] [AllReduce逐行 S188] 仅当`(it == endpointTagMemMap_.end())`成立时进入此分支；调用end。
    if (it == endpointTagMemMap_.end()) {
        // [中文导读] [AllReduce逐行 S189] 记录EndpointMgr::GetMemHandlesByTags的错误诊断，字段包含本地端点句柄；日志本身不执行传输。
        HCCL_ERROR("[%s] epHandle[%p] not found in endpointTagMemMap_", __FUNCTION__, epHandle);
        // [中文导读] [AllReduce逐行 S190] 直接返回`HCCL_E_MEMORY`；将当前查询结果/句柄交给调用者。
        return HCCL_E_MEMORY;
    // [中文导读] [AllReduce逐行 S191] 结束`if (it == endpointTagMemMap_.end())`分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S192] 设置当前Endpoint的tag注册账本为/按`it->second`。
    const auto& taggedMap = it->second;
    // [中文导读] 按调用方标签顺序输出注册句柄；任一标签缺失立即报错，因此失败时列表可能只有前缀。
    // [中文导读] [AllReduce逐行 S194] 按`(const auto& tag : memTags)`（内存标签、本通道需要交换的内存标签）遍历本批条目/分片；各次处理保持数组对应关系。
    for (const auto& tag : memTags) {
        // [中文导读] [AllReduce逐行 S195] 设置当前对象句柄为/按`taggedMap.FindHandle(tag)`（当前Endpoint的tag注册账本的FindHandle字段、内存标签）；按内存tag查询当前Endpoint的注册句柄。
        MemHandle handle = taggedMap.FindHandle(tag);
        // [中文导读] [AllReduce逐行 S196] 仅当`(handle == nullptr)`（当前对象句柄）成立时进入此分支。
        if (handle == nullptr) {
            // [中文导读] [AllReduce逐行 S197] 记录EndpointMgr::GetMemHandlesByTags的错误诊断；日志本身不执行传输。
            HCCL_ERROR(
                // [中文导读] [AllReduce逐行 S198] 为当前EndpointMgr::GetMemHandlesByTags诊断/异常表达式提供格式文本，将报告内存标签的c_str字段；这一物理行没有数据搬运副作用。
                "[%s] tag[%s] not found in endpoint[%p], registration may have been skipped", __FUNCTION__, tag.c_str(),
                // [中文导读] [AllReduce逐行 S199] 为调用c_str，使用内存标签的c_str字段、本地端点句柄补入`epHandle)`（本地端点句柄）；本行是参数/结构化初始化续行。
                epHandle);
            // [中文导读] [AllReduce逐行 S200] 返回HCCL_E_NOT_FOUND，表示没有找到对应资源/实现；此路径停止本函数的后续处理。
            return HCCL_E_NOT_FOUND;
        // [中文导读] [AllReduce逐行 S201] 结束`if (handle == nullptr)`（当前对象句柄）分支/循环；控制流返回外层。
        }
        // [中文导读] [AllReduce逐行 S202] 将当前条目追加到对应数组/列表；传入/处理本通道端点注册句柄列表的push_back字段、当前对象句柄。
        memHandleVec.push_back(handle);
    // [中文导读] [AllReduce逐行 S203] 结束`for (const auto& tag : memTags)`（内存标签、本通道需要交换的内存标签）分支/循环；控制流返回外层。
    }
    // [中文导读] [AllReduce逐行 S204] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S205] 结束EndpointMgr::GetMemHandlesByTags函数体；控制流返回外层。
}

HcclResult EndpointMgr::UnregMemByTag(const std::string& tag)
{
    std::lock_guard<std::mutex> lock(mutex_);
    HcclResult lastErr = HCCL_SUCCESS;
    // [中文导读] 同一标签可能注册在多个 Endpoint，遍历所有账本分别注销，跳过未注册的端点。
    for (auto& kv : endpointTagMemMap_) {
        MemHandle handle = kv.second.FindHandle(tag);
        if (handle == nullptr) {
            continue;
        }
        HcommResult ret = HcommMemUnreg(kv.first, handle);
        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR(
                "[%s] HcommMemUnreg failed, epHandle[%p] tag[%s] ret[%d]", __FUNCTION__, kv.first, tag.c_str(), ret);
            lastErr = static_cast<HcclResult>(ret);
            continue;
        }
        // [中文导读] 底层注销成功后才移除标签；失败项保留账本供后续处理，最终返回最后一次注销错误。
        kv.second.RemoveTag(tag);
    }
    return lastErr;
}

bool EndpointMgr::IsDescExist(EndpointDesc epDesc) { return endpointMap_.find(epDesc) != endpointMap_.end(); }

} // namespace hccl
