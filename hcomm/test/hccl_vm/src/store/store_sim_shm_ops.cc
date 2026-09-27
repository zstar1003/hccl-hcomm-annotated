/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "store_sim_shm_ops.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fcntl.h>
#include <string>
#include <sys/mman.h>
#include <sys/shm.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

#include "sim_log.h"

#define SHM_HEAD_SIZE sizeof(ShmHead)

static void* GetShmHead(void* shm) { return (char*)shm - SHM_HEAD_SIZE; }

static size_t GetShmTotalSize(size_t dataSize) { return SHM_HEAD_SIZE + dataSize; }

// 大小端转换函数
static uint32_t Swap32(uint32_t value)
{
    return ((value & 0x000000FF) << 24) | ((value & 0x0000FF00) << 8) | ((value & 0x00FF0000) >> 8)
           | ((value & 0xFF000000) >> 24);
}

static uint64_t Swap64(uint64_t value)
{
    return ((value & 0x00000000000000FFULL) << 56) | ((value & 0x000000000000FF00ULL) << 40)
           | ((value & 0x0000000000FF0000ULL) << 24) | ((value & 0x00000000FF000000ULL) << 8)
           | ((value & 0x000000FF00000000ULL) >> 8) | ((value & 0x0000FF0000000000ULL) >> 24)
           | ((value & 0x00FF000000000000ULL) >> 40) | ((value & 0xFF00000000000000ULL) >> 56);
}

// 检测当前架构是否为大端
static int IsBigEndian()
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};
    return bint.c[0] == 1;
}

// 平台相关的字节序转换
static uint32_t ToLe32(uint32_t value)
{
    if (IsBigEndian()) {
        return Swap32(value);
    }
    return value;
}

static uint64_t ToLe64(uint64_t value)
{
    if (IsBigEndian()) {
        return Swap64(value);
    }
    return value;
}

static uint32_t FromLe32(uint32_t value)
{
    if (IsBigEndian()) {
        return Swap32(value);
    }
    return value;
}

static uint64_t FromLe64(uint64_t value)
{
    if (IsBigEndian()) {
        return Swap64(value);
    }
    return value;
}

void* ShmCreate(const char* name, size_t size)
{
    if (!name || size == 0) {
        HCCL_VM_ERROR("create: name is nullptr or size is 0");
        return nullptr;
    }

    int shmFd = shm_open(name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (shmFd == -1) {
        HCCL_VM_ERROR("create: shm_open failed, name: {} errno:{}", name, errno);
        return nullptr;
    }

    size_t totalSize = GetShmTotalSize(size);
    if (ftruncate(shmFd, totalSize) == -1) {
        HCCL_VM_ERROR("create: ftruncate failed, name: {} errno:{}", name, errno);
        close(shmFd);
        shm_unlink(name);
        return nullptr;
    }

    void* addr = mmap(nullptr, totalSize, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    int err = errno;
    close(shmFd);

    if (addr == MAP_FAILED) {
        shm_unlink(name);
        HCCL_VM_ERROR("create: mmap failed, name: {} errno:{}", name, err);
        return nullptr;
    }

    ShmHead* head = (ShmHead*)addr;
    head->magic = ToLe32(SHM_MAGIC);
    head->version = ToLe32(SHM_VERSION);
    strncpy(head->name, name, sizeof(head->name) - 1);
    head->name[sizeof(head->name) - 1] = '\0';
    head->size = ToLe64(size);
    head->lock = 0;
    head->refCount = 1;
    HCCL_VM_INFO(
        "create name: {}, size: {:d}, ptr: {:p} ref:{:d}", head->name, size, (char*)addr + SHM_HEAD_SIZE,
        (int)head->refCount);
    return (char*)addr + SHM_HEAD_SIZE;
}

void* ShmOpen(const char* name, size_t* size)
{
    if (!name) {
        HCCL_VM_ERROR("open: name is nullptr");
        return nullptr;
    }

    int shmFd = shm_open(name, O_RDWR, 0666);
    if (shmFd == -1) {
        HCCL_VM_ERROR("open: shm_open failed, name: {} errno:{}", name, errno);
        return nullptr;
    }

    struct stat statBuf;
    if (fstat(shmFd, &statBuf) == -1) {
        HCCL_VM_ERROR("open: fstat failed, name: {} errno:{}", name, errno);
        close(shmFd);
        return nullptr;
    }

    void* addr = mmap(nullptr, statBuf.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, shmFd, 0);
    int err = errno;
    close(shmFd);

    if (addr == MAP_FAILED) {
        HCCL_VM_ERROR("open: mmap failed, name: {} errno:{}", name, err);
        return nullptr;
    }

    ShmHead* head = (ShmHead*)addr;
    if (FromLe32(head->magic) != SHM_MAGIC) {
        HCCL_VM_ERROR("open: invalid magic number, name: {}", name);
        munmap(addr, statBuf.st_size);
        return nullptr;
    }

    *size = FromLe64(head->size);
    // 增加引用计数
    uint32_t current_count = FromLe32(head->refCount);
    uint32_t new_count = current_count + 1;
    while (!__sync_bool_compare_and_swap(&head->refCount, ToLe32(current_count), ToLe32(new_count))) {
        current_count = FromLe32(head->refCount);
        new_count = current_count + 1;
    }
    HCCL_VM_INFO("open name: {}, size: {:d}, ptr: {:p}", head->name, *size, (char*)addr + SHM_HEAD_SIZE);
    return (void*)((char*)addr + SHM_HEAD_SIZE);
}

void ShmClose(void* shm)
{
    if (!shm) {
        HCCL_VM_ERROR("close: shm is nullptr");
        return;
    }

    ShmHead* head = (ShmHead*)GetShmHead(shm);
    size_t totalSize = GetShmTotalSize(FromLe64(head->size));

    char tmp[64] = {0};
    strncpy(tmp, head->name, sizeof(tmp) - 1);

    int refCount = __sync_fetch_and_sub(&head->refCount, 1);
    int ref = head->refCount;
    munmap(head, totalSize);
    if (refCount == 1) {
        // 最后一个引用，删除共享内存
        HCCL_VM_INFO("close name: {}, ptr: {:p}", tmp, (char*)head);
        shm_unlink(tmp);
    } else {
        HCCL_VM_INFO("close name: {}, ptr: {:p}, refCount: {} {}", tmp, (char*)head, refCount, ref);
    }
}

int ShmLock(void* shm)
{
    if (!shm) {
        HCCL_VM_ERROR("ShmLock: shm is nullptr");
        return -1;
    }

    ShmHead* head = (ShmHead*)GetShmHead(shm);
    while (1) {
        uint32_t expected = 0;
        uint32_t desired = ToLe32(1);

        if (__sync_bool_compare_and_swap(&head->lock, 0, desired)) {
            break;
        }
        sched_yield();
    }
    return 0;
}

int ShmUnlock(void* shm)
{
    if (!shm) {
        HCCL_VM_ERROR("ShmUnlock: shm is nullptr");
        return -1;
    }

    ShmHead* head = (ShmHead*)GetShmHead(shm);
    __sync_lock_release(&head->lock);
    return 0;
}
