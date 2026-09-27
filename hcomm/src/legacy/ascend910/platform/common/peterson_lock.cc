/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include "adapter_rts_common.h"
#include "peterson_lock.h"

namespace hccl {
PetersonLock::PetersonLock(u64 timeoutSec)
    : size_(MIN_SHM_LEN),
      type_(Type::HOST),
      typeName_("Host"),
      timeout_(timeoutSec),
      myTurn_(TURN_FOR_HOST)
{}

PetersonLock::PetersonLock(void* devPtr, u64 timeoutSec)
    : size_(MIN_SHM_LEN),
      type_(Type::DEVICE),
      typeName_("Device"),
      timeout_(timeoutSec),
      devMem_(devPtr, size_, false),
      myTurn_(TURN_FOR_DEVICE)
{}

PetersonLock::~PetersonLock() { DeInit(); }

HcclResult PetersonLock::Init()
{
    if (type_ == Type::HOST) {
        if (AllocDeviceMem() != HCCL_SUCCESS) {
            return HCCL_E_INTERNAL;
        }
    }

    auto buffer = reinterpret_cast<u8*>(devMem_.ptr());
    size_t offset = 0;
    turn_ = reinterpret_cast<volatile u32*>(buffer + offset);
    offset += sizeof(u32);

    hostFlag_ = reinterpret_cast<volatile u32*>(buffer + offset);
    offset += sizeof(u32);

    deviceFlag_ = reinterpret_cast<volatile u32*>(buffer + offset);
    offset += sizeof(u32);

    HCCL_INFO(
        "[PetersonLock][Init] type [%s] init success, memSize [%lu Byte] timeout[%lu s]", typeName_.c_str(),
        devMem_.size(), timeout_);
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::AllocDeviceMem()
{
    if (devMem_.ptr() != nullptr || devMem_.size() != 0) {
        HCCL_ERROR("[PetersonLock][AllocDeviceMem] init failed, maybe it's already inited");
        return HCCL_E_INTERNAL;
    }

    CHK_RET(DeviceMem::alloc(devMem_, size_));

    if (hrtMemSet(devMem_.ptr(), devMem_.size(), devMem_.size()) != HCCL_SUCCESS) {
        HCCL_ERROR("[PetersonLock][AllocDeviceMem] memset device memory failed");
        return HCCL_E_INTERNAL;
    }

    HCCL_INFO("[PetersonLock][AllocDeviceMem] Type[%s] alloc memSize[%lu Byte]", typeName_.c_str(), size_);
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::DeInit()
{
    size_ = 0;

    turn_ = nullptr;
    hostFlag_ = nullptr;
    deviceFlag_ = nullptr;
    return HCCL_SUCCESS;
}

u64 PetersonLock::GetDevMemAddr() const { return reinterpret_cast<u64>(devMem_.ptr()); }

HcclResult PetersonLock::Lock()
{
    if (turn_ == nullptr || hostFlag_ == nullptr || deviceFlag_ == nullptr) {
        HCCL_ERROR("[PetersonLock][lock] ptr is nullptr, maybe not call Init()");
        return HCCL_E_INTERNAL;
    }

    HCCL_DEBUG("[PetersonLock][Lock] type [%s] before require the lock", typeName_.c_str());
    auto startTime = std::chrono::steady_clock::now();
    auto timeout = std::chrono::seconds(timeout_);

    /* 更新flag表明自己准备获取锁 */
    CHK_RET(WriteSelfFlag(FLAG_LOCK));

    /* 更新TURN值 */
    CHK_RET(WriteTurn());

    /*
     * 这里判断为真只有一种场景会进入等待，需满足以下两个条件
     *    - 条件1：peer也想要获取锁
     *    - 条件2：self更新的turn等于自己设置的值
     * 通常条件2意味着peer执行的更快，使得自己是覆盖写turn，那么自己就需要等待
     *
     *   （PS:当然这里存在并发场景，即self刚写完turn就又读turn，peer也是刚写完turn就又读turn，
     *   此时两端都满足上述为真，就都进入等待，但是因为turn最终只会有一个值，
     *   所以再循环一遍后最后turn的值就只能是一个了，此时谁最后更新谁就去等待）
     *
     * Q：为什么读取peer为FLAG_UNLOCK时本端一定可以安全获取锁？
     * A：当自己读到peer是释放锁状态时，peer侧有三种场景：
     *     - 场景1：peer不准备获取锁，此时self可以安全获取锁
     *     - 场景2：peer已经释放了锁，此时self可以安全获取锁
     *     - 场景3：peer侧也想获取锁，但是由于并发(self读flag peer写flag)导致peer更新的flag未被self读到
     *  对于场景3，由于严格内存序那么写self与读peer操作一定是串行，此时self虽未及时读到peer的flag，
     *  但是因为自己的flag已经被更新，不会与peer的读有并发，因此一定能被peer读到，那么peer就会获取到
     *  self要获取锁，就会进入前面描述的状态，即等待self释放锁
     *  时序图描述大致如下：
     *             self                          peer
     *             write flag (LOCK)
     *             write turn (HOST)
     *             read peer flag (UNLOCK)       write flag (LOCK)     # 这里并发导致未及时读到最新值
     *             成功获取锁                     write turn (DEVICE)      # 这里自己一定是最后更新的turn
     *                                           read peer flag (LOCK) # 读到的一定是获取锁，因为没有并发问题
     *                                           peer==LOCK && turn==DEVICE 为真，所以等待
     */
    u32 peerFlag;
    u32 turn;
    while (true) {
        if ((std::chrono::steady_clock::now() - startTime) > timeout) {
            HCCL_ERROR("[PetersonLock][Lock] type [%s] get lock timeout [%lu s]", typeName_.c_str(), timeout_);

            /* 重置flag */
            CHK_RET(WriteSelfFlag(FLAG_UNLOCK));
            return HCCL_E_TIMEOUT;
        }

        CHK_RET(ReadPeerFlag(peerFlag));
        CHK_RET(ReadTurn(turn));

        if (peerFlag == FLAG_LOCK && turn == myTurn_) {
            Wait();
        } else {
            HCCL_DEBUG("[PetersonLock][Lock] type [%s] got the lock", typeName_.c_str());
            break;
        }
    }

    HCCL_DEBUG("[PetersonLock][Lock] type [%s] after require the lock", typeName_.c_str());
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::Unlock()
{
    if (deviceFlag_ == nullptr) {
        HCCL_ERROR("[PetersonLock][Unlock] ptr is nullptr, maybe not call Init()");
        return HCCL_E_INTERNAL;
    }

    /* 释放锁 */
    WriteSelfFlag(FLAG_UNLOCK);

    HCCL_DEBUG("[PetersonLock][Unlock] type [%s] release the lock", typeName_.c_str());
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::WriteSelfFlag(u32 selfFlag)
{
    if (type_ == Type::DEVICE) {
        *deviceFlag_ = selfFlag;
    } else {
        u32 hostFlag = selfFlag;
        if (hrtMemSyncCopy(
                const_cast<u32*>(hostFlag_), sizeof(u32), &hostFlag, sizeof(u32),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE)
            != HCCL_SUCCESS) {
            HCCL_INFO("[PetersonLock][WriteSelfFlag] H2D write hostFlag not correct");
            return HCCL_E_INTERNAL;
        }
    }

    MemFence();
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::WriteTurn()
{
    if (type_ == Type::DEVICE) {
        *turn_ = TURN_FOR_DEVICE;
    } else {
        u32 turn = TURN_FOR_HOST;
        if (hrtMemSyncCopy(
                const_cast<u32*>(turn_), sizeof(u32), &turn, sizeof(u32),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE)
            != HCCL_SUCCESS) {
            HCCL_INFO("[PetersonLock][WriteSelfFlag] H2D write turn not correct");
            return HCCL_E_INTERNAL;
        }
    }

    MemFence();
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::ReadPeerFlag(u32& peerFlag)
{
    if (type_ == Type::DEVICE) {
        peerFlag = *hostFlag_;
    } else {
        if (hrtMemSyncCopy(
                &peerFlag, sizeof(u32), const_cast<u32*>(deviceFlag_), sizeof(u32),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST)
            != HCCL_SUCCESS) {
            HCCL_INFO("[PetersonLock][ReadPeerFlag] D2H read device flag not correct");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult PetersonLock::ReadTurn(u32& turn)
{
    if (type_ == Type::DEVICE) {
        turn = *turn_;
    } else {
        if (hrtMemSyncCopy(
                &turn, sizeof(u32), const_cast<u32*>(turn_), sizeof(u32),
                HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST)
            != HCCL_SUCCESS) {
            HCCL_INFO("[PetersonLock][ReadPeerFlag] D2H read turn not correct");
            return HCCL_E_INTERNAL;
        }
    }
    return HCCL_SUCCESS;
}

PetersonLockGuard::PetersonLockGuard(PetersonLock* lock) : lock_(lock), lockFailed_(false)
{
    if (lock_ == nullptr) {
        HCCL_ERROR("[PetersonLockGuard] invalid lock");
        lockFailed_ = true;
        return;
    }

    if (lock_->Lock() != HCCL_SUCCESS) {
        HCCL_ERROR("[PetersonLockGuard] lock failed");
        lockFailed_ = true;
        lock_ = nullptr;
    }
}

PetersonLockGuard::~PetersonLockGuard()
{
    if (lock_) {
        lock_->Unlock();
    }
}
} // namespace hccl
