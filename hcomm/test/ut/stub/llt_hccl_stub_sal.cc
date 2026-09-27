/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <assert.h> /* for assert  */
#include <errno.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <sys/time.h> /* 获取时间 */
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h>    /* For O_* constants */
#include <math.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <dirent.h>
#include <securec.h>
#include "llt_hccl_stub.h"
using namespace std;

#define __HCCL_SAL_GLOBAL_RES_INCLUDE__

#include "llt_hccl_stub_sal_pub.h"

sal_manage_class sal_manage;

/* 检查SAL是否已经初始化完成，若没有则返回错误 */
#define SAL_ERR_RET_IF_SAL_IS_NOT_ACTIVED(ret)                   \
    do {                                                         \
        if (sal_manage.sal_active_get() != SAL_TRUE) {           \
            HCCL_ERROR("sal is not available before activated"); \
            return ret;                                          \
        }                                                        \
    } while (0)

#define SAL_LOCK() (sal_manage.sal_manage_lock.lock())
#define SAL_UNLOCK() (sal_manage.sal_manage_lock.unlock())

/* 华为安全库函数 */
#if T_DESC("华为安全库函数适配", 1)

#include <securec.h>

/* 华为安全函数返回值转换 */
#define HUAWEI_SECC_RET_CHECK_AND_RETURN(ret) \
    do {                                      \
        switch (ret) {                        \
            case EOK:                         \
                return HCCL_SUCCESS;          \
            case EINVAL:                      \
                return HCCL_E_PARA;           \
            default:                          \
                return HCCL_E_INTERNAL;       \
        }                                     \
    } while (0)
#define HUAWEI_SECC_RET_TRANSFORM(ret) ((ret == EOK) ? HCCL_SUCCESS : ((ret == EINVAL) ? HCCL_E_PARA : HCCL_E_INTERNAL))
#endif

#if T_DESC("信号量及互斥锁适配", 1)
/*
 * 函 数 名  : sal_compute_timeout
 * 功能描述  : 计算超时时间戳, 信号量/互斥锁接口使用.
 * 输入参数  : ts
 *             usec
 * 输出参数  : 无
 * 返 回 值  : static
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_compute_timeout(struct timespec* ts, s32 usec)
{
    s32 sec;
    u32 nsecs;

    /* CLOCK_REALTIME会受ntp干扰, 未来替换
       clock_gettime(CLOCK_MONOTONIC, ts);  */
    (void)clock_gettime(CLOCK_REALTIME, ts);

    /* Add in the delay */
    ts->tv_sec += usec / SAL_SECOND_USEC;

    /* compute new nsecs */
    nsecs = ts->tv_nsec + (usec % SAL_SECOND_USEC) * 1000;

    /* detect and handle rollover */
    if (nsecs < ts->tv_nsec) {
        ts->tv_sec += 1;
        nsecs -= SAL_SECOND_NSEC;
    }

    ts->tv_nsec = nsecs;

    /* Normalize if needed */
    sec = ts->tv_nsec / SAL_SECOND_NSEC;

    if (sec) {
        ts->tv_sec += sec;
        ts->tv_nsec = ts->tv_nsec % SAL_SECOND_NSEC;
    }

    /* indicate that we successfully got the time */
    return HCCL_SUCCESS;
}
s32 sal_vsnprintf(char* strDest, size_t destMaxSize, size_t count, const char* format, va_list argList)
{
    /* 返回值不是错误码,无需转换 */
    CHK_PTR_NULL(strDest);
    CHK_PTR_NULL(format);
    return vsnprintf_s(strDest, destMaxSize, count, format, argList);
}
HcclResult sal_memset(void* dest, size_t destMaxSize, int c, size_t count)
{
    CHK_PTR_NULL(dest);
    s32 ret = memset_s(dest, destMaxSize, c, count);
    if (ret != EOK) {
        HCCL_ERROR(
            "errNo[0x%016llx] In sal_memset, memset_s failed. errorno[%d], params: dest[%p], "
            "destMaxSize[%d], c[%d], count[%d]",
            HCCL_ERROR_CODE(HUAWEI_SECC_RET_TRANSFORM(ret)), ret, dest, destMaxSize, c, count);
    }
    HUAWEI_SECC_RET_CHECK_AND_RETURN(ret);
}
HcclResult sal_strncpy(char* strDest, size_t destMaxSize, const char* strSrc, size_t count)
{
    CHK_PTR_NULL(strDest);
    CHK_PTR_NULL(strSrc);
    s32 ret = strncpy_s(strDest, destMaxSize, strSrc, count);
    if (ret != EOK) {
        HCCL_ERROR(
            "errNo[0x%016llx] In sal_strncpy, strncpy_s failed. errorno[%d], params: strDest[%p], "
            "destMaxSize[%d], strSrc[%p], count[%d]",
            HCCL_ERROR_CODE(HUAWEI_SECC_RET_TRANSFORM(ret)), ret, strDest, destMaxSize, strSrc, count);
    }
    HUAWEI_SECC_RET_CHECK_AND_RETURN(ret);
}
HcclResult sal_memcpy(void* dest, size_t destMaxSize, const void* src, size_t count)
{
    CHK_PTR_NULL(dest);
    CHK_PTR_NULL(src);
    s32 ret = memcpy_s(dest, destMaxSize, src, count);
    if (ret != EOK) {
        HCCL_ERROR(
            "errNo[0x%016llx] In sal_memecpy, memcpy_s failed. errorno[%d], params: dest[%p], "
            "destMaxSize[%d], src[%p], count[%d]",
            HCCL_ERROR_CODE(HUAWEI_SECC_RET_TRANSFORM(ret)), ret, dest, destMaxSize, src, count);
    }
    HUAWEI_SECC_RET_CHECK_AND_RETURN(ret);
}
/*
 * 函 数 名  : sal_mutex_create
 * 功能描述  : 创建互斥锁
 * 输入参数  : desc
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_mutex_t sal_mutex_create(const char* desc, s32 canBeShared)
{
    recursive_mutex_t* rm;
    pthread_mutexattr_t attr;
    char myUniqueId[SAL_UNIQUE_ID_BYTES] = {0};
    char mutexUniqueId[SAL_MUTEX_UNIQUE_ID_BYTES] = {0};
    HcclResult ret;
    s32 sRet = 0;

    if (canBeShared) {
        /* 申请一个本os内唯一的ID */
        ret = SalGetUniqueId(myUniqueId);
        HCCL_RET_NULL_IF_RUN_FAILED(ret);

        sRet = snprintf_s(
            mutexUniqueId, SAL_MUTEX_UNIQUE_ID_BYTES, SAL_MUTEX_UNIQUE_ID_BYTES - 1, "%s%s", SAL_MUTEX_UNIQUE_ID_PREFIX,
            myUniqueId);
        if (sRet == -1) {
            HCCL_ERROR("get mutexUniqueId fail,snprintf_s ERROR");
            return NULL;
        }

        /* 新创建的共享内存默认全0，无需memset */
        rm = (recursive_mutex_t*)sal_share_memory_create(mutexUniqueId, sizeof(recursive_mutex_t));
        HCCL_RET_NULL_IF_PTR_IS_NULL(rm);

        ret = sal_strncpy(rm->rootInfo, SAL_MUTEX_UNIQUE_ID_BYTES, mutexUniqueId, (SAL_MUTEX_UNIQUE_ID_BYTES - 1));
        HCCL_RET_NULL_RELEASE_SHM_IF_RUN_FAILED(ret, rm);
    } else {
        if ((rm = (recursive_mutex_t*)malloc(sizeof(recursive_mutex_t))) == NULL) {
            return NULL;
        }

        (void)sal_memset(rm, sizeof(recursive_mutex_t), 0, sizeof(recursive_mutex_t));
    }

    rm->shared = canBeShared;

    (void)pthread_mutexattr_init(&attr);
    (void)pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_RECURSIVE);  /* 互斥锁支持递归 */
    (void)pthread_mutexattr_setprotocol(&attr, PTHREAD_PRIO_INHERIT); /* PI锁支持优先级继承和反转.  */

    if (canBeShared) {
        /* 支持跨进程共享 */
        (void)pthread_mutexattr_setpshared(&attr, PTHREAD_PROCESS_SHARED);
    }

    (void)pthread_mutex_init(&rm->mutex, &attr);

    /* 刷新引用计数，原子操作避免互斥，先取值，再加一，初始值肯定为0，不再做异常处理 */
    (void)__sync_fetch_and_add(&(rm->ref_cnt), 1);

    return (sal_mutex_t)rm;
}

/*
 * 函 数 名  : sal_mutex_open
 * 功能描述  : 打开另一个进程创建的互斥锁
 * 输入参数  : mutex_unique_id
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月17日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_mutex_t sal_mutex_open(const char* mutexUniqueId)
{
    recursive_mutex_t* rm;
    u32 myRefCnt = 0;

    rm = (recursive_mutex_t*)sal_share_memory_create(mutexUniqueId, sizeof(recursive_mutex_t));
    HCCL_RET_NULL_IF_PTR_IS_NULL(rm);

    /* 刷新引用计数，原子操作避免互斥，先取值，再加一 */
    myRefCnt = __sync_fetch_and_add(&(rm->ref_cnt), 1);

    if (0 == myRefCnt) {
        /*
            计数为0，表示该互斥锁已经被销毁了，此时打开的只是一片空白的共享内存。
            未来考虑增加magic number进行mutex内容有效性校验。
        */
        (void)__sync_fetch_and_sub(&(rm->ref_cnt), 1);
        HCCL_ERROR("share mutex[%s] is invalid", mutexUniqueId);
        sal_share_memory_destroy(rm);
        return NULL;
    }

    return (sal_mutex_t)rm;
}

/*
 * 函 数 名  : sal_mutex_get_unique_id
 * 功能描述  : 返回跨进程Mutex的unique_id
 * 输入参数  : mutex
 *             rootInfo
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月14日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_mutex_get_unique_id(sal_mutex_t mutex, char* uniqueId)
{
    recursive_mutex_t* rm = (recursive_mutex_t*)mutex;
    HcclResult hcclRet;

    CHK_PTR_NULL(mutex);
    CHK_PTR_NULL(uniqueId);

    if (rm->shared) {
        hcclRet = sal_memcpy(uniqueId, SAL_MUTEX_UNIQUE_ID_BYTES, rm->rootInfo, SAL_MUTEX_UNIQUE_ID_BYTES);
        return hcclRet;
    } else {
        HCCL_WARNING("Mutex[%s] did not support share", rm->desc);
        return HCCL_E_UNAVAIL;
    }
}

/*
 * 函 数 名  : sal_mutex_destroy
 * 功能描述  : 销毁互斥锁
 * 输入参数  : mutex
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
void sal_mutex_destroy(sal_mutex_t mutex)
{
    recursive_mutex_t* rm = (recursive_mutex_t*)mutex;
    u32 myRefCnt = 0;

    assert(rm);

    /* 刷新引用计数，原子操作避免互斥，先取值，再减一 */
    myRefCnt = __sync_fetch_and_sub(&(rm->ref_cnt), 1);

    if (1 == myRefCnt) {
        (void)pthread_mutex_destroy(&rm->mutex);
    }

    if (rm->shared) {
        /*  从sal资源管理库删除 */
        sal_share_memory_destroy(rm);
    } else {
        sal_free(rm);
    }
}

/*
 * 函 数 名  : sal_mutex_take
 * 功能描述  : 获取互斥锁
 * 输入参数  : mutex
 *             usec
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_mutex_take(sal_mutex_t mutex, s32 usec)
{
    CHK_PTR_NULL(mutex);
    recursive_mutex_t* rm = (recursive_mutex_t*)mutex;
    s32 err = 0;

    struct timespec ts;

    assert(rm);

    errno = 0;

    if (usec == SAL_MUTEX_FOREVER) {
        do {
            err = pthread_mutex_lock(&rm->mutex);
        } while (err != 0 && errno == EINTR);

        if (err) {
            HCCL_WARNING("SAL: take mutex failed[%d]: %s [%d]", err, strerror(errno), errno);
            /* 屏蔽pclint关于获取锁但是未释放的告警 */
            return HCCL_E_INTERNAL; // lint !e454
        }

        /* 屏蔽pclint关于获取锁但是未释放的告警 */
        return HCCL_SUCCESS; // lint !e454
    } else if (HCCL_SUCCESS == sal_compute_timeout(&ts, usec)) {
        /* Treat EAGAIN as a fatal error on Linux */
        err = pthread_mutex_timedlock(&rm->mutex, &ts);

        if (err) {
            HCCL_WARNING("SAL: take mutex failed[%d]: %s [%d]", err, strerror(errno), errno);
            return HCCL_E_INTERNAL;
        } else {
            return HCCL_SUCCESS;
        }
    }

    return HCCL_E_INTERNAL;
}

/*
 * 函 数 名  : sal_mutex_give
 * 功能描述  : 释放互斥锁
 * 输入参数  : mutex
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_mutex_give(sal_mutex_t mutex)
{
    recursive_mutex_t* rm = (recursive_mutex_t*)mutex;
    s32 err;

    assert(rm);
    /* 屏蔽pclint关于未获取锁但是释放的告警 */
    err = pthread_mutex_unlock(&rm->mutex); // lint !e455

    return err ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

/*
 * 函 数 名  : sal_sem_create
 * 功能描述  : 创建信号量
 * 输入参数  : desc
 *             binary
 *             initial_count
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_sem_t sal_sem_create(const char* desc, s32 binary, s32 initialCount, s32 canBeShared)
{
    wrapped_sem_t* s = NULL;
    char myUniqueId[SAL_UNIQUE_ID_BYTES];
    char semUniqueId[SAL_SEM_UNIQUE_ID_BYTES] = {0};
    HcclResult ret;
    s32 sRet = 0;

    if (canBeShared) {
        /* 申请一个本OS内唯一的ID */
        ret = SalGetUniqueId(myUniqueId);
        HCCL_RET_NULL_IF_RUN_FAILED(ret);

        sRet = snprintf_s(
            semUniqueId, SAL_SEM_UNIQUE_ID_BYTES, SAL_SEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_SEM_UNIQUE_ID_PREFIX,
            myUniqueId);
        if (sRet == -1) {
            HCCL_ERROR("get mutexUniqueId fail,snprintf_s ERROR");
            return NULL;
        }

        /* 新创建的共享内存默认全0，无需memset */
        s = (wrapped_sem_t*)sal_share_memory_create(semUniqueId, sizeof(wrapped_sem_t));
        HCCL_RET_NULL_IF_PTR_IS_NULL(s);

        ret = sal_strncpy(s->rootInfo, SAL_SEM_UNIQUE_ID_BYTES, semUniqueId, (SAL_SEM_UNIQUE_ID_BYTES - 1));
        HCCL_RET_NULL_RELEASE_SHM_IF_RUN_FAILED(ret, s);

        ret = sal_strncpy(s->desc, SAL_SEM_DESC_BYTES, desc, (SAL_SEM_DESC_BYTES - 1));
        HCCL_RET_NULL_RELEASE_SHM_IF_RUN_FAILED(ret, s);
    } else {
        if ((s = (wrapped_sem_t*)sal_malloc(sizeof(wrapped_sem_t))) == NULL) {
            return NULL;
        }

        /*
         * This is needed by some libraries with a bug requiring to zero sem_t before calling sem_init(),
         * even though this it is not required by the function description.
         * Threads using sem_timedwait() to maintain polling interval use 100% CPU if we not set the memory to zero
         * SDK-77724
         */
        ret = sal_memset(s, sizeof(wrapped_sem_t), 0, sizeof(wrapped_sem_t));

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("sal_memset failed, return[%d]", ret);
            sal_free(s);
            // s指针已经释放，pclint误报内存泄露，此处屏蔽
            return NULL; // lint !e429
        }

        ret = sal_strncpy(s->desc, SAL_SEM_DESC_BYTES, desc, (SAL_SEM_DESC_BYTES - 1));

        if (ret != HCCL_SUCCESS) {
            HCCL_ERROR("sal_strncpy[%s] failed[%d]", desc, ret);
            sal_free(s);
            // s指针已经释放，pclint误报内存泄露，此处屏蔽
            return NULL; // lint !e429
        }
    }

    s->shared = canBeShared;

    (void)sem_init(&s->s, canBeShared, initialCount);
    s->binary = binary;

    /* 刷新引用计数，原子操作避免互斥，先取值，再加一，初始值肯定为0，不再做异常处理 */
    (void)__sync_fetch_and_add(&(s->ref_cnt), 1);

    return (sal_sem_t)s;
}
/*
 * 函 数 名  : sal_sem_open
 * 功能描述  : 打开跨进程共享信号量
 * 输入参数  : sem_unique_id
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月17日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_sem_t sal_sem_open(const char* semUniqueId)
{
    wrapped_sem_t* s = NULL;
    u32 myRefCnt = 0;

    s = (wrapped_sem_t*)sal_share_memory_create(semUniqueId, sizeof(wrapped_sem_t));
    HCCL_RET_NULL_IF_PTR_IS_NULL(s);

    /* 刷新引用计数，原子操作避免互斥，先取值，再加一 */
    myRefCnt = __sync_fetch_and_add(&(s->ref_cnt), 1);

    if (0 == myRefCnt) {
        /*
            计数为0，表示该互斥锁已经被销毁了，此时打开的只是一片空白的共享内存。
            未来考虑增加magic number进行sem内容有效性校验。
        */
        (void)__sync_fetch_and_sub(&(s->ref_cnt), 1);
        HCCL_ERROR("share sem ID[%s] is invalid", semUniqueId);
        sal_share_memory_destroy(s);
        return NULL;
    }
    return (sal_sem_t)s;
}

/*
 * 函 数 名  : sal_sem_get_unique_id
 * 功能描述  : 返回跨进程信号量的unique id
 * 输入参数  : sem
 *             rootInfo
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月14日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_sem_get_unique_id(sal_sem_t sem, char* uniqueId)
{
    wrapped_sem_t* s = (wrapped_sem_t*)sem;
    HcclResult ret = HCCL_SUCCESS;

    CHK_PTR_NULL(sem);
    CHK_PTR_NULL(uniqueId);

    if (s->shared) {
        ret = sal_memcpy(uniqueId, SAL_SEM_UNIQUE_ID_BYTES, s->rootInfo, SAL_SEM_UNIQUE_ID_BYTES);
        HCCL_RET_IF_RUN_FAILED(ret);
    } else {
        HCCL_WARNING("Sem [%s] did not support share", s->desc);
        return HCCL_E_UNAVAIL;
    }

    return HCCL_SUCCESS;
}

/*
 * 函 数 名  : sal_sem_destroy
 * 功能描述  : 销毁信号量
 * 输入参数  : sem
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
void sal_sem_destroy(sal_sem_t sem)
{
    u32 myRefCnt = 0;
    wrapped_sem_t* s = (wrapped_sem_t*)sem;

    assert(s);

    /* 刷新引用计数，原子操作避免互斥，先取值，再减一 */
    myRefCnt = __sync_fetch_and_sub(&(s->ref_cnt), 1);

    if (1 == myRefCnt) {
        (void)sem_destroy(&s->s);
    }

    if (s->shared) {
        sal_share_memory_destroy(s);
    } else {
        sal_free(s);
    }
}

/*
 * 函 数 名  : sal_sem_take
 * 功能描述  : 获取信号量
 * 输入参数  : sem
 *             usec
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_sem_take(sal_sem_t sem, s32 usec)
{
    CHK_PTR_NULL(sem);
    wrapped_sem_t* s = (wrapped_sem_t*)sem;
    s32 err = 0;

    struct timespec ts;

    if (usec == SAL_SEM_FOREVER) {
        do {
            err = sem_wait(&s->s);
        } while (err != 0 && errno == EINTR);
    } else if (HCCL_SUCCESS == sal_compute_timeout(&ts, usec)) {
        while (1) {
            if (!sem_timedwait(&s->s, &ts)) {
                err = 0;
                break;
            }

            if (errno != EAGAIN && errno != EINTR) {
                err = errno;
                break;
            }
        }
    }

    return (!err) ? HCCL_SUCCESS : ((ETIMEDOUT == err) ? HCCL_E_TIMEOUT : HCCL_E_INTERNAL);
}

/*
 * 函 数 名  : sal_sem_give
 * 功能描述  : 释放信号量
 * 输入参数  : sem
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年7月26日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_sem_give(sal_sem_t sem)
{
    wrapped_sem_t* s = (wrapped_sem_t*)sem;
    s32 err = 0;
    s32 semVal = 0;

    /* Binary sem only post if sem_val == 0 */
    if (s->binary) {
        /* Post sem on getvalue failure */
        (void)sem_getvalue(&s->s, &semVal);

        if (semVal == 0) {
            err = sem_post(&s->s);
        }
    } else {
        err = sem_post(&s->s);
    }

    return err ? HCCL_E_INTERNAL : HCCL_SUCCESS;
}

#endif

#if T_DESC("线程处理适配", 1)

#ifdef PTHREAD_STACK_MIN
#define SAL_PTHREAD_STACK_SIZE (PTHREAD_STACK_MIN + SAL_PTHREAD_DEFAULT_STACK_SIZE)
#else
#define SAL_PTHREAD_STACK_SIZE (SAL_PTHREAD_DEFAULT_STACK_SIZE)
#endif

/*
 * 函 数 名  : sal_thread_boot
 * 功能描述  : 线程引导函数, 所有新线程从该函数启动
 * 输入参数  : thread_info
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
static void* sal_thread_boot(void* threadInfo)
{
    thread_info_t* ti = (thread_info_t*)threadInfo;
    void* (*f)(void*);
    void* arg;
    void* ret;

    /* 线程信号配置,屏蔽 Control-C 对应的 SIGINT 信号 */
    sigset_t newMask, origMask;
    /* Make sure no child thread catches Control-C */
    (void)sigemptyset(&newMask);
    (void)sigaddset(&newMask, SIGINT);
    (void)sigprocmask(SIG_BLOCK, &newMask, &origMask);

    /* 配置线程 detach 模式, 线程退出时无需父线程善后 */
    (void)pthread_detach(pthread_self());

    /* 设置线程名称 */
    (void)prctl(PR_SET_NAME, ti->name.c_str(), 0, 0, 0);

    /* 设置线程被 cancel(Destroy,强制退出) 时立即退出 */
    (void)pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, NULL);
    (void)pthread_setcancelstate(PTHREAD_CANCEL_ENABLE, NULL);

    f = ti->f;
    arg = ti->arg;

    /* 获取线程句柄 */
    ti->id = pthread_self();

    /* 获取进程pid */
    ti->pid = (long)syscall(SYS_gettid);

    /* 释放同步信号, 线程启动完成, 开始业务处理 */
    (void)sal_sem_give(ti->sem);

    /* 调用业务处理函数(需使用 return 返回) */
    ret = (*f)(arg);

    /* 线程主动退出 */
    sal_thread_exit(ret);

    return NULL;
}

/*
 * 函 数 名  : sal_thread_create
 * 功能描述  : 创建新线程
 * 输入参数  : name
 *             f
 *             arg
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_thread_t sal_thread_create(string name, void* (*f)(void*), void* arg)
{
    /* 避免变量定义时操作sal，在函数最开头判断 */
    SAL_ERR_RET_IF_SAL_IS_NOT_ACTIVED(NULL);

    pthread_attr_t attribs;
    thread_info_t* ti;
    pthread_t id;
    sal_sem_t sem;
    s32 ss = SAL_PTHREAD_STACK_SIZE;
    s32 ret = SAL_OK;

    if (pthread_attr_init(&attribs)) {
        return NULL;
    }

    // 配置线程栈大小 8M.
    (void)pthread_attr_setstacksize(&attribs, ss);

    // 暂时不允许配置实时或者fifo级别的线程, 如有需要可以在线程的业务处理函数中调整优先级.
    // 申请 thread info 结构体
    HCCL_EXECUTE_CMD((ti = new thread_info_t) == NULL, return NULL);
    // 申请 线程同步信号量
    if ((sem = sal_sem_create("threadBoot", 1, 0)) == NULL) {
        delete (ti);
        return NULL;
    }

    // 申请线程名称字符串
    ti->name = name;

    /* 线程信息 */
    ti->f = f;
    ti->arg = arg;
    ti->id = (pthread_t)0;
    ti->ss = ss;
    ti->sem = sem;

    /* 将待启动的线程加入线程列表 */
    SAL_LOCK();
    sal_manage.sal_thread_regist(ti);
    /* 启动线程 */
    ret = pthread_create(&id, &attribs, sal_thread_boot, (void*)ti);

    if (ret) {
        HCCL_ERROR("Create Thread[%s] failed[%d]: [%s] [%d]", name.c_str(), ret, strerror(errno), errno);
        /* 线程启动失败,删除线程信息 */
        sal_manage.sal_thread_unregist(ti);
        SAL_UNLOCK();

        delete ti;
        sal_sem_destroy(sem);
        return NULL;
    }

    SAL_UNLOCK();

    /* 等待线程启动 */
    (void)sal_sem_take(sem, SAL_SEM_FOREVER);
    /* 线程已经开始运行, 可能正在运行,也可能已经运行结束. */
    sal_sem_destroy(sem);

    return ((sal_thread_t)(uintptr_t)id);
}

/*
 * 函 数 名  : sal_thread_destroy
 * 功能描述  : 销毁指定线程, 危险操作, 被销毁的线程可能还持有互斥锁/内存. 尽量让线程主动退出.
               可以考虑使用 pthread_cleanup_push 添加线程退出时执行释放函数.
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_thread_destroy(sal_thread_t thread)
{
    thread_info_t* ti;
    pthread_t id = (pthread_t)(uintptr_t)thread;
    s32 ret = 0;

    ti = NULL;

    SAL_LOCK();

    (void)sal_manage.sal_thread_find(id, &ti);

    if (ti) {
        /* 删除线程信息软表 */
        sal_manage.sal_thread_unregist(ti);
        delete ti;
        SAL_UNLOCK();

        /* 强制线程退出 */
        ret = pthread_cancel(id);

        if (ret) {
            /* 线程不存在时返回成功 */
            if (ESRCH == ret) {
                return HCCL_SUCCESS;
            } else {
                return HCCL_E_INTERNAL;
            }
        }

        /* 等待 50 ms, 确认线程已经退出 */
        SaluSleep(SAL_MILLISECOND_USEC * 50);

        if (sal_thread_is_running(thread)) {
            /* 线程仍然在运行 */
            return HCCL_E_INTERNAL;
        } else {
            /* 线程已经停止 */
            return HCCL_SUCCESS;
        }
    } else {
        SAL_UNLOCK();
    }

    return HCCL_SUCCESS;
}

/*
 * 函 数 名  : sal_thread_is_running_internal
 * 功能描述  : 返回线程是否仍在运行(内部调用,不校验软表)
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
s32 sal_thread_is_running_internal(sal_thread_t thread)
{
    pthread_t id = (pthread_t)(uintptr_t)thread;
    /* 向被检测线程发送 0 信号,验证pid是否有效 */
    s32 ret = pthread_kill(id, 0);

    /* Not a valid thread handle.  */
    if (ESRCH == ret) {
        return SAL_FALSE;
    } else {
        return SAL_TRUE;
    }
}

/*
 * 函 数 名  : sal_thread_is_running
 * 功能描述  : 返回线程是否仍在运行
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
s32 sal_thread_is_running(sal_thread_t thread)
{
    thread_info_t* ti;
    pthread_t id = (pthread_t)(uintptr_t)thread;
    s32 threadIsRunning = SAL_FALSE;

    /* 先搜索SAL维护的线程列表, 针对找到的线程进行确认 */
    ti = NULL;

    SAL_LOCK(); /* 防止途中ti被删除，此处访问出错，加锁 */
    (void)sal_manage.sal_thread_find(id, &ti);

    if (ti) {
        /* 通过底层接口进一步确认线程状态 */
        if (sal_thread_is_running_internal(thread)) {
            threadIsRunning = SAL_TRUE;
        } else {
            /* 线程异常退出,且SAL未捕获退出,用户函数中可能直接使用了底层线程退出接口,清理软表并告警. */
            HCCL_WARNING(
                "Bug: thread[%s] is not running, but task info is remain in mem. Clean it now", ti->name.c_str());
            sal_manage.sal_thread_unregist(ti);
            delete ti;
        }
    }

    SAL_UNLOCK();
    return threadIsRunning;
}

/*
 * 函 数 名  : sal_thread_show
 * 功能描述  : 打印所有子线程信息,调测函数
 * 输入参数  : void
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
HcclResult sal_thread_show(void)
{
    thread_info_t* threadInfo = NULL;
    HcclResult ret = HCCL_SUCCESS;

    SAL_LOCK(); /* 复合操作，加锁 */

    for (u32 pos = 0; pos < sal_manage.len_of_thread_res; pos++) {
        (void)sal_manage.sal_thread_info_pop(pos, &threadInfo);
        HCCL_INFO("Name[%s]\t PID[%08u]", threadInfo->name.c_str(), threadInfo->pid);
    }
    SAL_UNLOCK();
    return ret;
}

/*
 * 函 数 名  : sal_thread_self
 * 功能描述  : 返回当前线程句柄(线程中调用)
 * 输入参数  : void
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
sal_thread_t sal_thread_self(void) { return (sal_thread_t)(uintptr_t)pthread_self(); }

/*
 * 函 数 名  : sal_thread_exit
 * 功能描述  : 主动退出当前线程(线程中调用)
 * 输入参数  : rc
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2017年8月8日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 */
void sal_thread_exit(void* rc)
{
    thread_info_t* ti;
    pthread_t id = pthread_self();

    ti = NULL;

    SAL_LOCK();
    /* 查找线程信息 */
    (void)sal_manage.sal_thread_find(id, &ti);

    /* 销毁线程信息 */
    if (ti) {
        sal_manage.sal_thread_unregist(ti);
        delete ti;
    }
    SAL_UNLOCK(); /* 防止上面的delte后，其他函数再访问thread_info，此处加锁 */

    /* 线程退出 */
    pthread_exit(rc);
}
#endif

#if T_DESC("跨进程处理函数", 1)
/*
 *
 * 函 数 名  : sal_share_memory_create
 * 功能描述  : 创建和映射共享内存区域
 * 输入参数  : rootInfo
 *             mem_size
 * 输出参数  : 无
 * 返 回 值  : void
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月16日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 *
 */
void* sal_share_memory_create(const char* uniqueId, u64 memSize)
{
    s32 ret = SAL_OK;
    HcclResult hcclRet;
    s32 fd = 0;
    share_mem_t* shareMemPtr = NULL;
    u32 realMemSize = offsetof(share_mem_t, user_data) + memSize;
    u32 currRefCnt = 0;

    HCCL_RET_NULL_IF_PTR_IS_NULL(uniqueId);

    if (SAL_SHARE_MEM_UNIQUE_ID_BYTES <= SalStrLen(uniqueId)) {
        HCCL_ERROR("rootInfo len[%d] exceed the limit[%d]", SalStrLen(uniqueId), SAL_SHARE_MEM_UNIQUE_ID_BYTES);
        return NULL;
    }

    HCCL_DEBUG("start create share mem[%s], data size[%d]", uniqueId, memSize);

    /* 打开、创建共享内存对象，默认位于/dev/shm */
    fd = shm_open(uniqueId, O_CREAT | O_RDWR, S_IRUSR | S_IWUSR);

    if (0 > fd) {
        HCCL_ERROR("shm_open [%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
        return NULL;
    }

    /* 调整共享内存大小，新扩充的空间会被填0 */
    ret = ftruncate(fd, realMemSize);

    if (0 > ret) {
        HCCL_ERROR("ftruncate file[%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
        return NULL;
    }

    /* 将共享内存映射到当前进程的虚拟地址空间 */
    shareMemPtr = (share_mem_t*)mmap(NULL, realMemSize, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);

    if (MAP_FAILED == shareMemPtr) {
        HCCL_ERROR("mmap for[%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
        return NULL;
    }

    /* 刷新引用计数，原子操作避免互斥，先取值，再加一 */
    currRefCnt = __sync_fetch_and_add(&(shareMemPtr->ref_cnt), 1);

    if (0 == currRefCnt) {
        /* 第一个创建共享内存的进程 */
        shareMemPtr->mem_size = realMemSize;
    }

    /* 关闭共享内存对象句柄，已经创建的共享内存对象不会被释放 */
    ret = close(fd);

    if (0 > ret) {
        HCCL_ERROR("close file[%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
        /* 取消虚拟地址映射 */
        (void)munmap(shareMemPtr, realMemSize);
        return NULL;
    }

    /* 共享内存管理数据初始化 */
    hcclRet = sal_strncpy(
        shareMemPtr->rootInfo, SAL_SHARE_MEM_UNIQUE_ID_BYTES, uniqueId, (SAL_SHARE_MEM_UNIQUE_ID_BYTES - 1));

    if (HCCL_SUCCESS != hcclRet) {
        HCCL_ERROR("sal_strncpy[%s] failed[%d]", uniqueId, hcclRet);
        /* 取消虚拟地址映射 */
        (void)munmap(shareMemPtr, realMemSize);
        return NULL;
    }

    /* 返回用户内存起始地址 */
    return shareMemPtr->user_data;
}

/*
 *
 * 函 数 名  : sal_share_memory_destroy
 * 功能描述  : 销毁共享内存区域
 * 输入参数  : ptr
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月16日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 *
 */
void sal_share_memory_destroy(void* ptr)
{
    HCCL_RET_VOID_IF_PTR_IS_NULL(ptr);

    s32 ret = SAL_OK;
    HcclResult hcclRet;
    share_mem_t* shareMemPtr = (share_mem_t*)((char*)ptr - offsetof(share_mem_t, user_data));
    u32 realMemSize = shareMemPtr->mem_size;
    char uniqueId[SAL_SHARE_MEM_UNIQUE_ID_BYTES];
    u32 currRefCnt = 0;

    HCCL_DEBUG(
        "start destroy share mem: id[%s], ref_cnt[%d], size[%d]", shareMemPtr->rootInfo, shareMemPtr->ref_cnt,
        shareMemPtr->mem_size);

    /* 缓存 rootInfo, unmap后share_mem_ptr不可用 */
    hcclRet = sal_strncpy(
        uniqueId, SAL_SHARE_MEM_UNIQUE_ID_BYTES, shareMemPtr->rootInfo, (SAL_SHARE_MEM_UNIQUE_ID_BYTES - 1));

    if (HCCL_SUCCESS != hcclRet) {
        HCCL_ERROR("invalid rootInfo[%s] failed[%d]", uniqueId, hcclRet);
        return;
    }

    /* 刷新引用计数，原子操作避免互斥，先取值，再减一 */
    currRefCnt = __sync_fetch_and_sub(&(shareMemPtr->ref_cnt), 1);

    /* 取消虚拟地址映射 */
    ret = munmap(shareMemPtr, realMemSize);

    if (0 > ret) {
        HCCL_ERROR("munmap [%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
        return;
    }

    if (1 == currRefCnt) {
        /*
            最后一个销毁的进程负责销毁共享内存对象

            例外场景:
            当A进程执行销毁动作 将 ref_cnt
           减小为0，但是尚未来得及unlink的时候，B进程打开该共享内存对象，将ref_cnt增加为1；
            A进程后续会先执行unlink操作，销毁共享内存对象(只销毁句柄，已经映射的物理内存不受影响)成功。
            B进程执行销毁动作时，会再次unlink，但是句柄已经被A进程销毁，此时B进程unlink会报文件不存在。

            出现重复unlik场景时，直接忽略，对功能无影响。
        */
        ret = shm_unlink(uniqueId);

        if ((0 > ret) && (ENOENT != errno)) {
            HCCL_ERROR("shm_unlink [%s] failed[%d]: [%s] [%d]", uniqueId, ret, strerror(errno), errno);
            return;
        }
    }

    HCCL_DEBUG("destroy share mem[%s] success", uniqueId);

    return;
    // ptr 指针已经 unmap，共享内存也已销毁，pclint误报，此处屏蔽。
} // lint !e429

#endif

// Add for event execute in order by l on 2018-02-11 Above
#if T_DESC("sal资源管理", 1)
/*
 * 函 数 名  : sal_init_active_get
 * 功能描述  : sal资源管理库有效无效状态获取
 * 输入参数  : void
 * 输出参数  : void
 * 返 回 值  : 0：无效  1：有效
 * 其它说明  :用
 * 修改历史      :
 *  1.日    期   : 2018年6月27日
 *    作    者   : ligang  00442453
 *    修改内容   : 新生成函数
 *
 */
s32 sal_manage_class::sal_active_get(void) const { return sal_actived; }

/*
 *
 * 函 数 名  : sal_thread_regist
 * 功能描述  : 注册线程到sal资源管理库.
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  : HcclResult
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年6月16日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *
 */
HcclResult sal_manage_class::sal_thread_regist(const thread_info_t* thread)
{
    CHK_PTR_NULL(thread);

    sal_thread_list_node_t* node = (sal_thread_list_node_t*)(new (std::nothrow) sal_thread_list_node_t);
    HCCL_RET_MEMORY_ERR_IF_PTR_IS_NULL(node);

    node->thread_info = (thread_info_t*)thread;

    sal_thread_list.push_back(node);
    len_of_thread_res++;

    HCCL_DEBUG("thread Name[%s]\t PID[%08u] is registered", thread->name.c_str(), thread->pid);
    return HCCL_SUCCESS;
}

/*
 *
 * 函 数 名  : sal_thread_unregist
 * 功能描述  : 从sal资源管理库销毁线程.
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  : HcclResult
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年6月16日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *
 */

void sal_manage_class::sal_thread_unregist(const thread_info_t* thread)
{
    HCCL_RET_VOID_IF_PTR_IS_NULL(thread);

    for (auto it = sal_thread_list.begin(); it != sal_thread_list.end(); ++it) {
        if (thread == (*it)->thread_info) {
            len_of_thread_res--;
            HCCL_DEBUG(
                "thread Name[%s]\t PID[%08u] is unregistered", (*it)->thread_info->name.c_str(),
                (*it)->thread_info->pid);

            delete (*it);
            *it = NULL;
            sal_thread_list.erase(it);
            break;
        }
    }
}

/*
 *
 * 函 数 名  : sal_thread_unregist
 * 功能描述  : sla资源管理库中是否存在指定的线程
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  : HcclResult
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年6月16日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *
 */
HcclResult sal_manage_class::sal_thread_find(pthread_t id, thread_info_t** ti)
{
    HcclResult ret = HCCL_E_PARA;

    for (auto it = sal_thread_list.begin(); it != sal_thread_list.end(); ++it) {
        if ((*it)->thread_info->id == id) {
            *ti = (*it)->thread_info;
            ret = HCCL_SUCCESS;
            break;
        }
    }

    return ret;
}
/*
 *
 * 函 数 名  : sal_thread_info_pop
 * 功能描述  : 从sal资源管理库总弹出指定位置的thread信息
 * 输入参数  : thread
 * 输出参数  : 无
 * 返 回 值  : HcclResult
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年6月16日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *
 */
HcclResult sal_manage_class::sal_thread_info_pop(u32 pos, thread_info_t** ti)
{
    u32 listLoopCnt = 0;
    HcclResult ret = HCCL_SUCCESS;

    if (pos >= len_of_thread_res) {
        HCCL_ERROR("wrong thread pos, thread pos :%d,sal thread list length:%d", pos, len_of_thread_res);
        ret = HCCL_E_PARA;
    } else {
        for (auto it = sal_thread_list.begin(); it != sal_thread_list.end(); ++it) {
            if (listLoopCnt == pos) {
                *ti = (*it)->thread_info;
                break;
            }

            listLoopCnt++;
        }
    }

    return ret;
}
/*
 *
 * 函 数 名  : sal_manage_class
 * 功能描述  : sal资源管理库初始化
 * 输入参数  : void
 * 输出参数  : void
 * 返 回 值  : void
 * 其它说明  :
              注册进程退出时的回调函数
              接管SIGINT信号和SIGTERM信号(普通kill),触发exit操作,走正常的全局资源回收流程,
              通过 __run_exit_handlers 依次调用析构函数及通过atexit等注册的回调函数
              全局对象 sequence_excute 构造函数中调用
 * 修改历史      :
 *  1.日    期   : 2018年6月19日
 *    作    者   : ligang  00442453
 *    修改内容   : 新生成函数
 *
 *
 */
sal_manage_class::sal_manage_class()
{
    len_of_thread_res = 0;
    sal_actived = SAL_TRUE;
}
/*
 *
 * 函 数 名  : sal_manage_class
 * 功能描述  : sal资源管理库销毁
 * 输入参数  : void
 * 输出参数  : void
 * 返 回 值  : void
 * 其它说明  :
              SAL资源回收
              局对象 sequence_excute 析构函数中调用
 * 修改历史      :
 *  1.日    期   : 2018年6月19日
 *    作    者   : ligang  00442453
 *    修改内容   : 新生成函数
 *
 *
 */
sal_manage_class::~sal_manage_class()
{
    if (sal_active_get() == SAL_TRUE) {
        /*  关闭sal, 禁止其他线程操作sal资源,active 为0, 禁止申请需回收的资源. */
        sal_actived = SAL_FALSE;
        len_of_thread_res = 0;
        for (auto* ptr : sal_thread_list) {
            delete ptr; // 释放每个对象
        }
        sal_thread_list.clear(); // 清空列表（避免野指针）
    }
}

pid_t drvDeviceGetBarePid(void) { return getpid(); }

/*
 *
 * 函 数 名  : SalGetUniqueId
 * 功能描述  : 返回跨进程唯一标识符(字符串)，适用于跨进程场景
 * 输入参数  : out
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年3月13日
 *    作    者   : xiaoshizhong
 *    修改内容   : 新生成函数
 *
 *
 */
HcclResult SalGetUniqueId(char* salUniqueId, int maxLen)
{
    static volatile u32 myCounter = 0; // 静态变量保证每次获取到不同的计数。
    CHK_PTR_NULL(salUniqueId);

    u32 myPid = drvDeviceGetBarePid();                        // 当前进程id
    u32 currentCounter = __sync_fetch_and_add(&myCounter, 1); // 本次获取的唯一计数
    u32 currentTime = SalGetSysTime();                        // 本次获取的唯一计数
    s32 hcclRet = sprintf_s(salUniqueId, maxLen, "%08x-%08x-%08x", myPid, currentTime, currentCounter);
    if (hcclRet == -1) {
        HCCL_ERROR(
            "In get unique id, printf failed.uniqueId[%s], "
            "dest max size[%d] mypid[0x%08x] current time[0x%08x] current counter[0x%08x]",
            salUniqueId, maxLen, myPid, currentTime, currentCounter);
    }

    return HCCL_SUCCESS;
}

HcclResult rt_stream_synchronize(HcclRtStream stream)
{
    // 参数有效性检查
    CHK_PTR_NULL(stream);
    aclrtStream rtStream = stream;

    aclError ret = aclrtSynchronizeStream(rtStream);
    HCCL_DEBUG("Call aclrtSynchronizeStream, return value[%d], para: rt_stream[%p].", ret, rtStream);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "errNo[0x%016llx] rt stream synchronize fail. return[%d], "
            "para: rt_stream[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, rtStream),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

void* sal_malloc(u32 size)
{
    if (size == 0) {
        HCCL_ERROR("errNo[0x%016llx] sal malloc fail, size[%u], return NULL", HCCL_ERROR_CODE(HCCL_E_MEMORY), size);
        return NULL;
    }

    return malloc(size);
}

void sal_free(void* ptr)
{
    if (ptr) {
        free(ptr);
    }
}

// u32 IpStr2Num(const std::string &strIp) {
//     u32 numericIp;
//     HcclResult ret = SalInetPton(strIp.c_str(), &numericIp);
//     if (ret != HCCL_SUCCESS) {
//         HCCL_ERROR("input string ip[%s] is invalid", strIp.c_str());
//         return INVALID_IPV4_ADDR;
//     } else {
//         return numericIp;
//     }
// }

// std::string IpNum2Str(const u32 numIp) {
//     std::string strBuff;
//     HcclResult ret = TransformIpNum2Str(numIp, strBuff);
//     if (ret != HCCL_SUCCESS) {
//         HCCL_ERROR("input numeric ip[%u] is invalid", numIp);
//         return "";
//     } else {
//         return strBuff;
//     }
// }

#endif
