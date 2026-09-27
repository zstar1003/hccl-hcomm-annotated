/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_REFERENCE_MAP_H
#define HCCL_REFERENCE_MAP_H

#include <unordered_map>

#include "hccl/hccl_types.h"
#include "log.h"

namespace hccl {

template <typename keyType, typename valueType>
class ReferenceMap {
public:
    typename std::unordered_map<keyType, valueType>::iterator begin() { return data_.begin(); }

    typename std::unordered_map<keyType, valueType>::iterator end() { return data_.end(); }

    u32 insert(const keyType key, const valueType& value)
    {
        if (has(key)) {
            ref_[key]++;
        } else {
            data_.insert(std::make_pair(key, value));
            ref_[key] = 1;
        }
        return count(key);
    }

    u32 erase(const keyType key)
    {
        u32 refCount = count(key);
        if (refCount > 1) {
            ref_[key]--;
        } else if (refCount == 1) {
            data_.erase(key);
            ref_.erase(key);
        }
        return count(key);
    }

    void clear()
    {
        data_.clear();
        ref_.clear();
    }

    bool has(const keyType key) { return (data_.find(key) != data_.end() ? true : false); }

    u32 count(const keyType key) { return (has(key) ? ref_[key] : 0); }

    valueType& operator[](const keyType key) { return data_[key]; }

    HcclResult ref(const keyType key)
    {
        if (has(key)) {
            ref_[key]++;
        } else {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    HcclResult unref(const keyType key)
    {
        if (has(key)) {
            ref_[key]--;
        } else {
            return HCCL_E_PARA;
        }
        return HCCL_SUCCESS;
    }

    u32 Size() { return data_.size(); }

private:
    std::unordered_map<keyType, valueType> data_;
    std::unordered_map<keyType, u32> ref_;
};

} // namespace hccl
#endif // HCCL_REFERENCE_MAP_H
