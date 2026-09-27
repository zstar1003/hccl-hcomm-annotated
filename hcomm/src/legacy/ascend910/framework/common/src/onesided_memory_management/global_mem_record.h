/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <unordered_set>
#include <mutex>
#include <memory>
#include <string>

#include <hccl/hccl_comm.h>
#include <hccl/hccl_inner.h>
#include "hccl_common.h"
#include "hccl_ip_address.h"
#include "hccl_mem_defs.h"

namespace hccl {

// 进程粒度的内存记录
class GlobalMemRecord {
public:
    // 不使用默认构造函数
    GlobalMemRecord() = delete;

    explicit GlobalMemRecord(const HcclMem* mem);
    explicit GlobalMemRecord(const HcclMem& mem);

    // 因为成员中有mutex，单独实现移动构造函数，并禁用拷贝构造函数
    GlobalMemRecord(const GlobalMemRecord& other) = delete;
    GlobalMemRecord(GlobalMemRecord&& other) noexcept;

    GlobalMemRecord& operator=(const GlobalMemRecord& other) = delete;
    GlobalMemRecord& operator=(GlobalMemRecord&& other) = delete;
    inline HcclMemType GetMemType() const { return type_; }

    inline const void* GetAddr() const { return addr_; }

    inline u64 GetSize() const { return size_; }

    inline bool IsBeingBound() const { return !boundComm_.empty(); }

    inline std::vector<std::string> GetBoundComm() const
    {
        return std::vector<std::string>(boundComm_.cbegin(), boundComm_.cend());
    }

    bool operator<(const GlobalMemRecord& other) const
    {
        // 先比较type
        if (type_ != other.type_) {
            return type_ < other.type_;
        }
        // 再比较地址
        if (addr_ != other.addr_) {
            return addr_ < other.addr_;
        }
        // 最后比较size
        return size_ < other.size_;
    }

    bool operator==(const GlobalMemRecord& other) const
    {
        return (type_ == other.type_) && (addr_ == other.addr_) && (size_ == other.size_);
    }

    bool HasOverlap(const GlobalMemRecord& other) const;          // 判断传入的内存是否有重叠
    HcclResult BindToComm(const std::string& commIdentifier);     // 绑定一个通信域
    HcclResult UnbindFromComm(const std::string& commIdentifier); // 与一个通信域解绑
    std::string PrintInfo() const;                                // 获取内存信息字符串
    inline void SaveRegBufInfo(HcclNetDevCtx& ctx, HcclBuf& buf)
    {
        HCCL_INFO("[GlobalMemRecord][SaveRegBufInfo] ctx[%p], addr[%p], len[%llu].", ctx, buf.addr, buf.len);
        std::lock_guard<std::mutex> lock(regBufInfoMtx_);
        regBufInfo_.insert(std::make_pair(ctx, buf));
    }

    inline std::unordered_map<HcclNetDevCtx, HcclBuf> GetAllRegBufInfo() const { return regBufInfo_; }

private:
    const HcclMemType type_;                      // 内存块类型, host或device
    const void* addr_;                            // 内存块地址
    const u64 size_;                              // 内存块大小字节数
    std::unique_ptr<std::mutex> pLock_;           // 锁保证多线程访问安全
    std::unordered_set<std::string> boundComm_{}; // 绑定了的通信域
    std::mutex regBufInfoMtx_;
    std::unordered_map<HcclNetDevCtx, HcclBuf> regBufInfo_{};
};

} // namespace hccl
