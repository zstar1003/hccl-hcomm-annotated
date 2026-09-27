/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RMA_BUFFER_MGR_H
#define RMA_BUFFER_MGR_H

#include <map>
#include <mutex>
#include <shared_mutex>
#include <utility>
#include "buffer_key.h"
#include "log.h"

namespace hcomm {
template <typename KeyType, typename BufferType, template <typename...> class M = std::map, typename... MapArgs>
class RmaBufferMgr {
public:
    struct BufferWithRef {
        // BufferType可以是指针的类型
        BufferType buffer{};
        uint64_t ref{}; // 引用计数

        BufferWithRef(BufferType buf, uint64_t r) : buffer(buf), ref(r) {}
    };

    using AddrType = typename KeyType::AddrType;
    using SizeType = typename KeyType::SizeType;

    using MapType = M<KeyType, BufferWithRef, MapArgs...>;
    using Iterator = typename MapType::iterator;
    using ConstIterator = typename MapType::const_iterator;

    template <typename... BufferArgs>
    // 禁止直接调用,请使用加锁版本的 AddWithoutCheck 替代
    std::pair<Iterator, bool> AddToTree(const KeyType& key, BufferArgs&&... bufferArgs)
    {
        auto result = intervalTree_.emplace(
            std::piecewise_construct, std::forward_as_tuple(key),
            std::forward_as_tuple(BufferWithRef{BufferType{std::forward<BufferArgs>(bufferArgs)...}, 1}));
        if (!result.second) {
            result.first->second.ref++;
            // 翻转
            if (result.first->second.ref == 0) {
                HCCL_ERROR("Error: ref = 0, ref++ flipped");
                throw std::logic_error("ref++ = 0, ref++ flipped");
            } else if (result.first->second.ref > 1) {
                HCCL_INFO(
                    "Memory is already registered, just increase the reference count, "
                    "current memory reference count[%llu], %s.",
                    result.first->second.ref, key.ToString().c_str());
            }
        }

        return result;
    }

    template <typename... BufferArgs>
    std::pair<Iterator, bool> AddWithoutCheck(const KeyType& key, BufferArgs&&... bufferArgs)
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        return AddToTree(key, std::forward<BufferArgs>(bufferArgs)...);
    }

    // 1.添加成功：输入key是表中某一最相近key的空集。 计数+1，返回添加成功的迭代器，及true
    // 2.添加已存在：输入key是表中某一最相近key的全集。 计数+1，返回添加该key的迭代器，及false
    // 3.添加失败：输入key是表中某一个最相近key的交集、子集、超集。返回空迭代器，及false
    template <typename... BufferArgs>
    std::pair<Iterator, bool> Add(const KeyType& key, BufferArgs&&... bufferArgs)
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        auto overlapResult = CheckOverlap(key);
        if (overlapResult.second) {
            HCCL_ERROR("Error: Buffer key overlaps with existing buffer key.");
            return std::make_pair(intervalTree_.end(), false);
        }
        return AddToTree(key, std::forward<BufferArgs>(bufferArgs)...);
    }

    // 1.查询成功：输入key是表中某一最相近key的子集、全集。 返回true，最相近key的bufferType
    // 2.查询失败：输入key是表中某一个最相近key的空集、交集。返回false，空bufferType
    std::pair<bool, BufferType> Find(const KeyType& key) const
    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        auto it = intervalTree_.lower_bound(key);
        if (it != intervalTree_.end() && (it->first == key || it->first.IsSuperset(key))) {
            return std::make_pair(true, it->second.buffer);
        }

        if (it != intervalTree_.begin()) {
            auto prevIt = std::prev(it);
            if (prevIt->first.IsSuperset(key)) {
                return std::make_pair(true, prevIt->second.buffer);
            }
            if (it != intervalTree_.end()) {
                HCCL_WARNING(
                    "Key[%s] not found. The near key is [%s] or [%s].", key.ToString().c_str(),
                    it->first.ToString().c_str(), prevIt->first.ToString().c_str());
            } else {
                HCCL_WARNING(
                    "Key[%s] not found. The near key is [%s]", key.ToString().c_str(),
                    prevIt->first.ToString().c_str());
            }
        } else {
            if (it != intervalTree_.end()) {
                HCCL_WARNING(
                    "Key[%s] not found. The near key is [%s]", key.ToString().c_str(), it->first.ToString().c_str());
            } else {
                HCCL_WARNING("Key[%s] not found. There is no key in table.", key.ToString().c_str());
            }
        }

        return std::make_pair(false, BufferType{}); // 未找到
    }

    // 1.删除成功：输入key是表中某一最相近key的全集。 计数-1且之后为0。  返回true
    // 2.删除引用数-1但未删除：输入key是表中某一最相近key的全集。 计数-1且之后大于0。 返回false
    // 3.删除失败：输入key是表中某一个最相近key的交集、子集、超集、空集。——抛出NOT_FOUND异常
    bool Del(const KeyType& key)
    {
        std::unique_lock<std::shared_mutex> lock(mtx_);
        auto it = intervalTree_.find(key);
        if (it == intervalTree_.end()) {
            HCCL_ERROR("Error: Buffer key not found.");
            throw std::out_of_range("Del NOT_FOUND");
        }

        if (--(it->second.ref) == 0) {
            intervalTree_.erase(it);
            return true;
        }
        // 引用计数大于0，不删除
        HCCL_RUN_INFO(
            "Memory reference count is larger than 0, (used by other RemoteRank), do not deregister memory."
            "current memory reference count[%llu], %s.",
            it->second.ref, key.ToString().c_str());
        return false;
    }

    bool IsInTree(const KeyType& key)
    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        auto it = intervalTree_.find(key);
        if (it == intervalTree_.end()) {
            return false;
        }
        return true;
    }

    ConstIterator Begin() { return intervalTree_.begin(); }

    ConstIterator Next(ConstIterator it) const { return std::next(it); }

    ConstIterator End() { return intervalTree_.end(); }
    size_t size() const { return intervalTree_.size(); }
    void PrintContents() const
    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        for (const auto& pair : intervalTree_) {
            HCCL_INFO("Key: %s, Value: %p", pair.first.ToString().c_str(), pair.second.buffer.get());
        }
    }

    template <typename Fn>
    void ForEach(Fn&& fn) const
    {
        std::shared_lock<std::shared_mutex> lock(mtx_);
        for (const auto& pair : intervalTree_) {
            std::forward<Fn>(fn)(pair.first, pair.second.buffer);
        }
    }

private:
    MapType intervalTree_;
    mutable std::shared_mutex mtx_{};

    std::pair<Iterator, bool> CheckOverlap(const KeyType& key)
    {
        auto it = intervalTree_.lower_bound(key);
        if (it != intervalTree_.end()) {
            // 情况1：addr_ == it->first.addr_ && size_ == it->first.size_
            if (it->first == key) {
                return std::make_pair(it, false);
            }

            // 情况2：addr_ == it->first.addr_ && size_ < it->first.size_。it->first.IsSubset(key)非必须
            // 情况3：addr_ < it->first.addr_
            if (it->first.IsSuperset(key) || it->first.IsIntersect(key)) {
                return std::make_pair(it, true);
            }
        }

        // 剩下的是空集
        if (it != intervalTree_.begin()) {
            auto prevIt = std::prev(it);
            // 情况4：addr_ > prevIt->first.addr_
            // 情况5： 1) addr_ > prevIt->first.addr_的子集情况；
            // 2) addr_ == prevIt->first.addr_，size_ > prevIt->first.size
            if (prevIt->first.IsIntersect(key) || prevIt->first.IsSubset(key) || prevIt->first.IsSuperset(key)) {
                return std::make_pair(prevIt, true);
            }

            // 6. 剩下的是空集
            return std::make_pair(prevIt, false);
        }

        // 剩下的是空集
        return std::make_pair(it, false);
    }
};
} // namespace hcomm

#endif
