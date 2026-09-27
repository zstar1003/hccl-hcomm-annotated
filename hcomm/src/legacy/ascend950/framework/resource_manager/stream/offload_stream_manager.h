/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_OFFLOAD_STREAM_MANAGER_H
#define HCCLV2_OFFLOAD_STREAM_MANAGER_H

#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include "hccl/base.h"
#include "stream.h"

namespace Hccl {

// 可继续优化为公共数据结构
template <typename T>
class CountSet {
private:
    // 底层存储：键=唯一元素，值=计数
    using MapType = std::unordered_map<T, int>;
    MapType count_map;

public:
    using iterator = typename MapType::iterator;
    // 1. 添加元素（计数+1）
    // 返回std::pair<iterator, bool>,与std::set::insert返回值语义完全一致
    std::pair<iterator, bool> insert(const T& elem)
    {
        auto it = count_map.find(elem);
        bool inserted = false;
        if (it == count_map.end()) {
            // 首次插入，计数初始化为1
            it = count_map.emplace(elem, 1).first;
            inserted = true;
        } else {
            // 元素已存在，计数+1
            it->second++;
            inserted = false;
        }
        return {it, inserted};
    }

    // 2. 删除元素（计数-1，计数为0时移除该元素）
    // 返回值：删除后剩余的计数（-1表示元素不存在）
    int erase(const T& elem)
    {
        auto it = count_map.find(elem);
        if (it == count_map.end()) {
            return -1; // 元素不存在
        }
        it->second--; // 计数-1
        if (it->second == 0) {
            count_map.erase(it); // 计数为0，移除键，避免枚举到空元素
            return 0;
        }
        return it->second;
    }
};

class OffloadStreamManager {
public:
    void RegisterMaster(const std::string& opTag, std::unique_ptr<Stream> stream);
    void RegisterSlaves(const std::string& opTag, const std::vector<void*>& slaveStreams);

    void Unregister(const std::string& opTag);

    Stream* GetMaster(const std::string& opTag);

    Stream* GetSlave(const std::string& opTag);

    void ResetIndex(const std::string& opTag, u32 index);

    u32 GetSlaveIndex(const std::string& opTag) const;

    Stream* GetSlave(const std::string& opTag, u32 index) const;
    HcclResult ClearOpStream(const std::string& opTag);

private:
    void ActivateSlaveStreams(const std::string& opTag, const Stream* masterStream);
    void CheckOpTag(const std::string& opTag) const;

    std::unordered_map<std::string, std::unique_ptr<Stream>> masters;
    std::unordered_map<std::string, std::vector<std::unique_ptr<Stream>>> slaves;
    u32 slaveIndex{0};
    std::string currOpTag{""};
    std::unordered_map<u32, CountSet<u32>> streamActiveManager_{}; // set中存放当前进程中以已由hccl激活的stream
};

} // namespace Hccl

#endif // HCCLV2_OFFLOAD_STREAM_MANAGER_H
