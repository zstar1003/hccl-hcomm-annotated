/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_SHM_OPS_H
#define SIM_SHM_OPS_H

#include <cstdint>
#include <stddef.h>

#define SHM_MAGIC 0x53484D50 // 'SHMP'
#define SHM_VERSION 1

typedef struct __attribute__((packed)) {
    uint32_t magic;             // 魔数
    uint32_t version;           // 版本号
    char name[64];              // 共享内存名称
    uint64_t size;              // 大小
    volatile uint32_t lock;     // 锁(0:未锁定， 1：已锁定)
    volatile uint32_t refCount; // 引用计数,记录所有进程对共享内存的引用
} ShmHead;

/*根据名称创建共享内存，ShmHead引用计数初始化为1*/
void* ShmCreate(const char* name, size_t size);

/*根据名称打开共享内存,ShmHead引用计数加1*/
void* ShmOpen(const char* name, size_t* size);

/*共享内存, ShmHead引用计数减1,引用计数归0则删除共享内存区域*/
void ShmClose(void* shm);

/*锁定指定的内存该内存由ShmCreate/ShmOpen创建，整块锁定，0 成功，-1 失败,内部使用POSIX信号量或者futex/atomic
 * compare-and-swap实现细粒度锁*/
int ShmLock(void* shm);

/*解锁指定的内存该内存由ShmCreate/ShmOpen创建，整块锁定，0 成功，-1 失败，内部使用POSIX信号量或者futex/atomic
 * compare-and-swap实现细粒度锁*/
int ShmUnlock(void* shm);

#endif // SIM_SHM_OPS_H
