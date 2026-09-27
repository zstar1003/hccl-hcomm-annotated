/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_REFERENCED_H
#define HCCL_REFERENCED_H

namespace Hccl {

class Referenced {
public:
    // 初始化这个类，引用计数设为1，并且将p指向传入的地址
    Referenced() : refCount(0) {}

    // 引用计数加1
    int Ref() { return ++refCount; }

    // 引用计数减1
    int Unref() { return --refCount; }

    // 返回引用计数
    int Count() const { return refCount; }

    int Clear()
    {
        refCount = 0;
        return refCount;
    }
    bool IsZero() const { return refCount == 0; }
    ~Referenced() {}

private:
    int refCount; // 引用计数，表示有多少个变量引用这块内存
};

} // namespace Hccl

#endif // HCCL_REFERENCED_H
