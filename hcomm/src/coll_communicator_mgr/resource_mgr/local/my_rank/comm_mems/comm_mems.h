/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef COMM_MEMS_H
#define COMM_MEMS_H

#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>
#include "hccl_types.h"
#include "log.h"
#include "hccl_mem_defs.h"
#include "rma_buffer_mgr.h"
#include "hcomm_c_adpt.h"

namespace std {
template <>
struct hash<CommMemInfo> {
    size_t operator()(const CommMemInfo& memInfo) const { return std::hash<void*>()(memInfo.mem.addr); }
};
} // namespace std

namespace hccl {
struct CommMemInfoEqual {
    bool operator()(const CommMemInfo& lhs, const CommMemInfo& rhs) const { return lhs.mem.addr == rhs.mem.addr; }
};

CommMemType ConvertHcclToCommMemType(HcclMemType hcclType);
HcclMemType ConvertCommToHcclMemType(CommMemType commType);

/**
 * @note 职责：集合通信域内MyRank的通信内存管理，包括HCCL Buffer和其他待注册到EndPoint内存
 */
class CommMems {
public:
    using Handle = std::shared_ptr<CommMemInfo>;
    using MemKey = hccl::BufferKey<uintptr_t, uint64_t>;
    using Table = hcomm::RmaBufferMgr<MemKey, Handle>;

    explicit CommMems(uint64_t bufferSize);
    ~CommMems() = default;

    HcclResult Add(void* addr, uint64_t len);

    HcclResult GetHcclBuffer(void*& addr, uint64_t& len);

    HcclResult HcclBufferMemset(void*& addr, uint64_t& len, bool clearFlag) const;

    HcclResult Init(HcclMem cclBuffer);

    // 用户注册/反注册内存
    HcclResult CommRegMem(const std::string& tag, const CommMem& mem, void** rawHandle);
    HcclResult CommUnregMem(const std::string& tag, const void* rawHandle);
    // 从 CommMemInfo* 数组提取 tag 列表
    HcclResult GetTagsFromHandles(void** memHandles, uint32_t memHandleNum, std::vector<std::string>& memTags);
    /**
     * 获取域内全部待注册内存（cclBuffer + 所有用户绑定内存），用于 endpoint 粒度批量注册。
     * 约定：返回的 memVec/memTags 0号位固定为 cclBuffer（tag="HcclBuffer"），
     *       后续为 opBindings_ 全量，同下标一一对应。
     *       version 为当前 CommMems 内存集合变更版本号。
     */
    HcclResult GetAllMemory(std::vector<HcclMem>& memVec, std::vector<std::string>& memTags, uint64_t& version);

private:
    uint64_t bufferSize_{};
    CommMemInfo cclMemInfo_{};
    uint64_t memVersion_{0}; // opBindings_ 变更版本号，每次注册/反注册递增

    static inline MemKey MakeKey(void* addr, uint64_t size)
    {
        return MemKey(reinterpret_cast<uintptr_t>(addr), static_cast<uint64_t>(size));
    }
    struct TagRegistry {
        Table table; // 区间树 + ref 语义
    };
    // 用户绑定内存
    std::mutex memMutex_;
    // 每个 tag 一份 registry
    std::unordered_map<std::string, TagRegistry> tagRegs_;
    // 每个tag 1个 CommMemInfo
    std::unordered_map<std::string, std::shared_ptr<CommMemInfo>> opBindings_;
};
} // namespace hccl

#endif // COMM_MEMS_H
