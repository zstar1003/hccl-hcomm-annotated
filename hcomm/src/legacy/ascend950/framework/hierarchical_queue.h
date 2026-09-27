/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_HIERARCHICAL_QUEUE_H
#define HCCLV2_HIERARCHICAL_QUEUE_H

#include "exception_util.h"
#include "null_ptr_exception.h"
#include <string_util.h>
#include "iterator.h"
#include "iterator_unconst.h"
#include "types.h"

#include <memory>
#include <vector>

namespace Hccl {
using namespace std;

template <typename E, typename SubQueue>
class HierarchicalQueue {
public:
    using SlaveIterator = BaseConstIterator<vector, shared_ptr<SubQueue>>;

    using Iterator = BaseConstIterator<vector, unique_ptr<E>>;

    using UnConstIterator = BaseIterator<vector, unique_ptr<E>>;

    using UnConstSlaveIterator = BaseIterator<vector, shared_ptr<SubQueue>>;

    inline u32 GetId() const { return id; }

    inline bool IsMaster() const { return masterFlag; }

    virtual shared_ptr<SubQueue> Fork()
    {
        // 待修改 Fork() can only be called by master queue!;
        auto slave = make_shared<SubQueue>();
        slave->masterFlag = false;
        slave->id = slaves.size() + 1;
        slaves.push_back(slave);
        slave->master = static_cast<SubQueue*>(this)->shared_from_this();
        return slave;
    }

    virtual void Append(unique_ptr<E> elem)
    {
        if (elem == nullptr) {
            std::string msg = StringFormat("[%s] elem Get nullptr", __func__);
            THROW<NullPtrException>(msg);
        }
        elements.push_back(std::move(elem));
    }

    weak_ptr<HierarchicalQueue> GetMaster() { return master; }

    SlaveIterator IterSlaves() const { return SlaveIterator(slaves); };

    Iterator Iter() const { return Iterator(elements); };

    UnConstSlaveIterator UnConstIterSlaves() { return UnConstSlaveIterator(slaves); };

    UnConstIterator UnConstIter() { return UnConstIterator(elements); };

    inline u32 Size() const { return elements.size(); };

    const E* First() const
    {
        if (elements.empty()) {
            return nullptr;
        }
        return elements.front().get();
    };

    const E* Last() const
    {
        if (elements.empty()) {
            return nullptr;
        }
        return elements.back().get();
    };

    inline u32 SizeOfSlaves() const { return slaves.size(); };

protected:
    HierarchicalQueue() : id(0), masterFlag(true) {};

    QId id;
    bool masterFlag;
    weak_ptr<SubQueue> master;
    vector<shared_ptr<SubQueue>> slaves;
    // 使用指针保证E的多态
    vector<unique_ptr<E>> elements;
};
} // namespace Hccl

#endif // HCCLV2_HIERARCHICAL_QUEUE_H
