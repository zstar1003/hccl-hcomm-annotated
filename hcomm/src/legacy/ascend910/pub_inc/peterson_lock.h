/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef PETERSON_LOCK_H
#define PETERSON_LOCK_H

#include <string>
#include <thread>
#include <atomic>
#include <hccl/hccl_types.h>
#include "hccl_common.h"
#include "mem_device_pub.h"

/**
 * 这里实现Peterson算法来实现host侧与device侧的互斥访问
 *
 * 背景：host侧与device可以通过H2D/D2H等API来进行共享内存访问，
 *       那么这两者之间无法对共享变量使用类似于CAS的机制，因此不能基于
 *       原子修改类API的方式实现锁，所以这里基于Peterson算法实现，
 *       paper参考:https://zoo.cs.yale.edu/classes/cs323/doc/Peterson.pdf
 *
 * 依赖：Peterson算法虽然不依赖原子操作，但是依赖读写内存序，因此需要添加内存屏障
 *       来避免编译器与CPU的乱序执行
 *
 * 限制：目前实现只支持单个Host线程与单个Device线程互斥访问，不支持Host/Device侧多线程访问，
 *       如需单侧多线程访问请先使用std::mutex或者pthread_spinlock_t保证互斥，再使用该锁
 *
 * 内存布局：
 *      [u32 turn] [u32 hostFlag] [u32 deviceFlag]
 */
namespace hccl {
class PetersonLock {
public:
    static constexpr u64 DEFAULT_LOCK_TIMEOUT_SEC = 60; /* 默认的超时时间, 60s */

    /* Host侧对象构造函数，在Init()中会进行Device侧内存申请 */
    explicit PetersonLock(u64 timeoutSec);
    /* Device侧对象构造函数，devPtr是在Host申请的设备侧内存地址 */
    PetersonLock(void* devPtr, u64 timeoutSec);
    ~PetersonLock();

    HcclResult Init();
    HcclResult DeInit();

    /* 目前该接口只服务于传递地址给Device侧，所以直接返回u64而不是指针 */
    u64 GetDevMemAddr() const;

    HcclResult Lock();
    HcclResult Unlock();

    /* 显式禁用所有copy、move构造函数，因为锁不可复制，不可移动 */
    PetersonLock(const PetersonLock&) = delete;
    PetersonLock(PetersonLock&&) = delete;
    PetersonLock& operator=(const PetersonLock&) = delete;
    PetersonLock& operator=(PetersonLock&&) = delete;

private:
    HcclResult AllocDeviceMem();

    HcclResult WriteSelfFlag(u32 selfFlag);
    HcclResult WriteTurn();
    HcclResult ReadPeerFlag(u32& peerFlag);
    HcclResult ReadTurn(u32& peerTurn);

    static constexpr size_t MIN_SHM_LEN = 32; /* 最小的共享内存大小 */
    enum class Type : int { HOST = 0, DEVICE = 1 };

    static constexpr u32 TURN_FOR_HOST = 1;
    static constexpr u32 TURN_FOR_DEVICE = 0;
    static constexpr u32 FLAG_LOCK = 1;   /* 获取锁 */
    static constexpr u32 FLAG_UNLOCK = 0; /* 释放锁 */

    void MemFence() const
    {
        /* 内存屏障，即阻止编译器重排变量读写，也阻止CPU重排变量读写 */
        std::atomic_thread_fence(std::memory_order_seq_cst);
    }

    void Wait() const
    {
        /* 无事可做暂时释放CPU */
        sched_yield();
    }

    size_t size_ = 0;
    Type type_ = Type::HOST;
    std::string typeName_;
    u64 timeout_;
    DeviceMem devMem_;
    u32 myTurn_;

    /*
     * 这里变量使用volatile修饰，是为了Device直接读写内存数据，而不是CPU cache
     */
    volatile u32* turn_ = nullptr;
    volatile u32* hostFlag_ = nullptr;
    volatile u32* deviceFlag_ = nullptr;
};
/**
 * 使用RAII特性使用PetersonLock，在该对象构造时加锁，析构时释放锁
 * 因为获取锁有可能失败，所以要调用IsLockFailed()去检查，只有加锁成功才能继续往下执行
 */
class PetersonLockGuard {
public:
    explicit PetersonLockGuard(PetersonLock* lock);
    ~PetersonLockGuard();

    bool IsLockFailed() const { return lockFailed_; }

    /* 显式禁用所有copy、move构造函数，因为锁不可复制，不可移动 */
    PetersonLockGuard(const PetersonLockGuard&) = delete;
    PetersonLockGuard(PetersonLockGuard&&) = delete;
    PetersonLockGuard& operator=(const PetersonLockGuard&) = delete;
    PetersonLockGuard& operator=(PetersonLockGuard&&) = delete;

private:
    PetersonLock* lock_ = nullptr;
    bool lockFailed_ = false;
};
} // namespace hccl

#endif
