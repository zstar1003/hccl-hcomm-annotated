/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_HOST_PUB_H
#define MEM_HOST_PUB_H

#include "hccl_common.h"

namespace hccl {
class HostMem {
public:
    /* * 默认构造函数, 只生成无效的HostMem对象 */
    explicit HostMem() : ptr_(nullptr), size_(0), owner_(false) {}

    /* * 拷贝构造函数, 用于HostMem::create的返回
        新实例对源实例的ptr无所有权, 析构时不释放ptr
        源实例保留原来对ptr的所有权, 析构时释放ptr */
    HostMem(const HostMem& that);

    /* * 移动构造函数, 用于HostMem::alloc的返回
        新实例对源实例的ptr有所有权, 析构时释放ptr
        源实例放弃原来对ptr的所有权, 析构时不释放ptr */
    HostMem(HostMem&& that);

    ~HostMem();

    /**
    通过静态成员函数来创建HostMem对象，目的如下:
    1)
    根据入参实例化，用create
    临时申请用，用alloc
    否则先调用底层函数申请memory，再用create会造成下层实现代码上移(rt_malloc)
    造成代码维护困难
    2)
    语义上类似C语言申请内存(malloc)的方式，好理解
    */
    static HostMem alloc(u64 size, bool isRtsMem = true);
    static HostMem create(void* ptr, u64 size);
    void free();

    /* * 部分操作符声明or重载, 期望达到类似memory指针操作那样来操作Mem对象 */
    /* * 重载move-assignment运算符, 用于alloc返回
        左值对象对右值对象的ptr有所有权, 析构时释放ptr
        右值对象放弃其原来对ptr的所有权, 析构时不释放ptr */
    HostMem operator=(HostMem&& that);

    /* * 重载copy-assignment运算符, 用于create返回和普通的HostMem对象拷贝
        左值对象对右值对象的ptr无所有权, 析构时释放ptr
        右值对象保留其原来对ptr的所有权, 析构时不释放ptr */
    HostMem& operator=(const HostMem& that);

    // "bool"运算符(可执行if(object){...}的操作判断该HostMem是否为空)
    operator bool() const { return ptr_ != nullptr; }

    // "=="运算符
    bool operator==(const HostMem& that) const { return (ptr_ == that.ptr()) && (size_ == that.size()); }

    bool operator!=(const HostMem& that) const { return (ptr_ != that.ptr()) || (size_ != that.size()); }

    // 取地址
    void* ptr() const { return ptr_; }

    /* * 内联成员函数 */
    u64 size() const { return size_; }

    /* * 在当前mem实例中截取一段形成新的Mem实例 */
    HostMem range(u64 offset, u64 size) const;

    void* ptr_; /* * memory地址 */
protected:
private:
    explicit HostMem(void* ptr, u64 size, bool owner, bool isRtsMem = false);
    explicit HostMem(u64 size);
    u64 size_;   /* * memory的size, 单位 : 字节 */
    bool owner_; /* * 类实例资源owner, 类似std::shared_ptr的做法 */
    bool isRtsMem_ = true;
};
} // namespace hccl

#endif /* MEM_HOST_PUB_H */
