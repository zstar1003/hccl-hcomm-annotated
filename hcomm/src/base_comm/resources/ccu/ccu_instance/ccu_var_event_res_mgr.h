/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCOMM_CCU_VAR_EVENT_RES_MGR_H
#define HCOMM_CCU_VAR_EVENT_RES_MGR_H

#include <shared_mutex>
#include <vector>
#include <unordered_map>

#include "ccu_types.h"

#include "ccu_res_repo.h"

namespace hcomm {

enum class CcuVarEventType {
    VARIABLE = 0,
    EVENT = 1,
};

struct CcuVarEventRes {
    CcuInsHandle insHandle{0};
    int32_t userDevId{-1};
    uint8_t dieId{0};
    CcuVarEventType type{CcuVarEventType::VARIABLE};
    std::vector<ResInfo> resInfos{};
    // 借用指针，指向发起预约的 CcuInstance 名下 CcuResPack 持有的资源仓，本类不持有所有权。
    // 有效性依赖既定时序：~CcuInstance 中 ReleaseByInstance 先于 resPack_ 置空执行，
    // CcuInstance::Reset 调 ExcludeAllocatedFromRepo 时资源仓亦仍存活。
    // 新增读写该指针的路径时，须确认调用点仍在 CcuResPack 析构之前。
    CcuResRepository* resRepo{nullptr};
    // 申请时对每个资源注册好的进程可访问 VA，下标与资源 index 一一对应
    std::vector<uint64_t> vaList{};
};

// 职责：管理通信域内 Variable(XN) / Event(CKE) 预约资源的完整生命周期，涵盖四个阶段——
// 从实例资源仓切出连续块、申请期为每个资源注册进程可访问 VA、按 handle 记账、随实例回收。
// 四者是同一份资源的先后阶段而非彼此独立的职责，故收敛在同一个类内：Acquire 完成
// 「切块 + 映射 + 记账」，ReleaseByHandle / ReleaseByInstance 完成对称的「解映射 + 归还」。
//
// 为何 Variable 与 Event 合用一个管理器：两类资源共享完全相同的使用模式——一次性预约连续块、
// 申请期暴露地址、句柄随通信域销毁统一回收；差异仅在资源池（blockXn / blockCke）与 runtime
// 资源类型（RT_RES_TYPE_CCU_XN / RT_RES_TYPE_CCU_CKE）两处，均由 CcuVarEventType 参数化。
// 拆成两个类会使上述四阶段流程与回滚逻辑重复一遍，故不拆。
//
// 与 CcuResIdAllocator 的边界：后者位于设备层，管理自持的 [0, capacity) 完整 id 空间，
// 其 resInfos_ 记录的是「已分配块」；本类位于实例层，从上游下发给本通信域的若干段「空闲块」
// 中做二次切分，无 capacity 概念。两者数据结构语义相反，不存在替代关系。
//
// 线程安全契约：
// 1) resMap_ 与 handleSeed_ 为 per-device、跨 CcuInstance 共享，全部读写均由 mapMutex_ 保护；
// 2) CcuVarEventRes::resRepo 指向的资源池由发起预约的 CcuInstance（经其 CcuResPack）持有，
//    不在 mapMutex_ 的保护范围内。CcuInstance 自身不含锁，约定同一 instance 只由单线程串行
//    访问，池的并发安全由该约定保证；不同 instance 的池互不相交，跨 instance 并发是安全的。
// 因此 ExcludeAllocatedFromRepo 取 shared_lock 是为遍历 resMap_，而非保护池。
class CcuVarEventResMgr {
public:
    static CcuVarEventResMgr& GetInstance(const int32_t userDevId);

    // 预约一段连续资源并在申请期完成地址映射；任一阶段失败均整笔回滚，仅全程成功才写 handle
    CcuResult Acquire(
        CcuInsHandle insHandle, CcuResRepository& resRepo, CcuVarEventType type, uint8_t dieId, uint32_t num,
        uint64_t& handle);
    CcuResult GetVariableXnId(uint64_t handle, uint32_t index, uint8_t& dieId, uint32_t& xnId) const;
    CcuResult GetEventCkeId(uint64_t handle, uint32_t index, uint8_t& dieId, uint32_t& ckeId) const;
    // 保存 Alloc 阶段为该 handle 注册好的全部资源 VA（下标与资源 index 一一对应）
    CcuResult SaveAddrs(CcuVarEventType type, uint64_t handle, const std::vector<uint64_t>& vaList);
    // 返回 Alloc 阶段已注册的第 index 个资源 VA
    CcuResult GetSavedAddr(CcuVarEventType type, uint64_t handle, uint32_t index, uint64_t& va) const;
    CcuResult ReleaseByHandle(uint64_t handle);
    CcuResult ReleaseByInstance(CcuInsHandle insHandle);
    CcuResult ExcludeAllocatedFromRepo(CcuInsHandle insHandle) const;

private:
    explicit CcuVarEventResMgr() = default;
    ~CcuVarEventResMgr() = default;

    CcuVarEventResMgr(const CcuVarEventResMgr& that) = delete;
    CcuVarEventResMgr& operator=(const CcuVarEventResMgr& that) = delete;

    // 校验参数、选池、切出连续块并登记到 resMap_，成功时经 newHandle 返回新句柄
    CcuResult AllocAndRecord(
        CcuInsHandle insHandle, CcuResRepository& resRepo, CcuVarEventType type, uint8_t dieId, uint32_t num,
        uint64_t& newHandle);
    // 为 handle 名下 num 个资源逐个注册进程可访问 VA 并缓存；失败时解除本次已完成的映射，
    // 资源池的归还由 Acquire 与切池动作配对完成
    CcuResult RegisterAddrs(CcuVarEventType type, uint64_t handle, uint32_t num);

    static CcuResult AllocFromPool(std::vector<ResInfo>& pool, uint32_t num, std::vector<ResInfo>& out);
    static void ReturnToPool(std::vector<ResInfo>& pool, const std::vector<ResInfo>& res);
    // 释放资源前，将 Alloc 阶段映射的进程可访问 VA 逐个 unmap（仅处理已保存 VA 的资源）；
    // 返回首个 unmap 失败的错误码，失败不中断，其余资源继续 unmap
    static CcuResult UnmapSavedAddrs(const CcuVarEventRes& res);

    int32_t userDevId_{-1};
    uint64_t handleSeed_{0};
    mutable std::shared_timed_mutex mapMutex_;
    std::unordered_map<uint64_t, CcuVarEventRes> resMap_{};
};

} // namespace hcomm

#endif // HCOMM_CCU_VAR_EVENT_RES_MGR_H
