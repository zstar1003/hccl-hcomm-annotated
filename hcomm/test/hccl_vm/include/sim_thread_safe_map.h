/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_THREAD_SAFE_MAP_H
#define SIM_THREAD_SAFE_MAP_H

#include <map>
#include <mutex>

template <typename K, typename V>
class ThreadSafeMap {
public:
    V& operator[](const K& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_[key];
    }

    void insert(const K& key, const V& value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        data_[key] = value;
    }

    bool try_get(const K& key, V& output) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = data_.find(key);
        if (it != data_.end()) {
            output = it->second;
            return true;
        }
        return false;
    }

    bool erase(const K& key)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.erase(key) > 0;
    }

    bool contains(const K& key) const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.find(key) != data_.end();
    }

    size_t size() const
    {
        std::lock_guard<std::mutex> lock(mutex_);
        return data_.size();
    }

    void clear()
    {
        std::lock_guard<std::mutex> lock(mutex_);
        data_.clear();
    }

    // 遍历操作
    template <typename F>
    void for_each(F func)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pair : data_) {
            func(pair.first, pair.second);
        }
    }

    // 条件删除
    template <typename Predicate>
    size_t erase_if(Predicate pred)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        size_t count = 0;
        for (auto it = data_.begin(); it != data_.end();) {
            if (pred(it->first, it->second)) {
                it = data_.erase(it);
                ++count;
            } else {
                ++it;
            }
        }
        return count;
    }

private:
    std::map<K, V> data_;
    mutable std::mutex mutex_;
};

#endif // SIM_THREAD_SAFE_MAP_H
