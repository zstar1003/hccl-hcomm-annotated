/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __LLT_SAL_PUB_H__
#define __LLT_SAL_PUB_H__

#include <chrono>
#include <stdarg.h>
#include <string>
#include <exception>
#include <sys/socket.h>
#include <vector>
#include <climits>
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <semaphore.h>
#include <pthread.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <sys/ioctl.h>
#include <list>
#include <vector>
#include <mutex>
#include "comm.h"
#include "mmpa_api.h"

// 获取服务器IP所需头文件
/* 释放share mem句柄,并将指针设置为NULL */
#define HCCL_RELEASE_SHM_AND_SET_NULL(shm) \
    do {                                   \
        if (shm) {                         \
            sal_share_memory_destroy(shm); \
            shm = NULL;                    \
        }                                  \
    } while (0)

/* 释放mutex句柄,并将指针设置为NULL */
#define HCCL_RELEASE_MUTEX_AND_SET_NULL(mutex) \
    do {                                       \
        if (mutex) {                           \
            sal_mutex_destroy(mutex);          \
            mutex = NULL;                      \
        }                                      \
    } while (0)

/* 释放sem句柄,并将指针设置为NULL */
#define HCCL_RELEASE_SEM_AND_SET_NULL(sem) \
    do {                                   \
        if (sem) {                         \
            sal_sem_destroy(sem);          \
            sem = NULL;                    \
        }                                  \
    } while (0)

/* 执行cmd函数,并检查函数返回值. 若函数执行错误, 则释放MUTEX，记录日志, 并返回错误码 */
#define HCCL_RET_AND_RELEASE_MUTEX_IF_RUN_FAILED(cmd, mutex)                                    \
    do {                                                                                        \
        HcclResult __ret__ = cmd;                                                               \
        if (__ret__ != HCCL_SUCCESS) {                                                          \
            HCCL_RELEASE_MUTEX_AND_SET_NULL(mutex);                                             \
            HCCL_ERROR("run [%s] failed, free ptr [%s] and return[%d]", #cmd, #mutex, __ret__); \
            return __ret__;                                                                     \
        }                                                                                       \
    } while (0)

/* 判断cmd的值，若cmd不成功，则记录信息，释放share mem句柄,并返回NULL */
#define HCCL_RET_NULL_RELEASE_SHM_IF_RUN_FAILED(cmd, shm)                                                \
    do {                                                                                                 \
        HcclResult __ret__ = cmd;                                                                        \
        if (__ret__ != HCCL_SUCCESS) {                                                                   \
            HCCL_RELEASE_SHM_AND_SET_NULL(shm);                                                          \
            HCCL_ERROR("run [%s] is failed, share_memory_destroy [%s] return[%d]", #cmd, #shm, __ret__); \
            return NULL;                                                                                 \
        }                                                                                                \
    } while (0)

/* 释放指针指向的host内存,并将指针设置为NULL */
#define HCCL_RELEASE_PTR_AND_SET_NULL(ptr) \
    do {                                   \
        if (ptr) {                         \
            sal_free(ptr);                 \
            ptr = NULL;                    \
        }                                  \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回内存异常错误 */
#define HCCL_RET_MEMORY_ERR_IF_PTR_IS_NULL(ptr)                         \
    do {                                                                \
        if (NULL == ptr) {                                              \
            HCCL_ERROR("ptr [%s] is NULL, return HCCL_E_MEMORY", #ptr); \
            return HCCL_E_MEMORY;                                       \
        }                                                               \
    } while (0)
/* 检查指针, 若指针为NULL, 则记录日志, 并return */
#define HCCL_RET_VOID_IF_PTR_IS_NULL(ptr)                 \
    do {                                                  \
        if (NULL == ptr) {                                \
            HCCL_ERROR("ptr [%s] is NULL, return", #ptr); \
            return;                                       \
        }                                                 \
    } while (0)
/* 检查指针, 若指针为NULL, 则记录日志, 并返回内部异常错误 */
#define HCCL_RET_INTERNAL_ERR_IF_PTR_IS_NULL(ptr)                         \
    do {                                                                  \
        if (NULL == ptr) {                                                \
            HCCL_ERROR("ptr [%s] is NULL, return HCCL_E_INTERNAL", #ptr); \
            return HCCL_E_INTERNAL;                                       \
        }                                                                 \
    } while (0)

/* 检查指针, 若指针为NULL, 则记录日志, 并返回NULL */
#define HCCL_RET_NULL_IF_PTR_IS_NULL(ptr)            \
    do {                                             \
        if (NULL == ptr) {                           \
            HCCL_ERROR("Pointer[%s] is NULL", #ptr); \
            return NULL;                             \
        }                                            \
    } while (0)

/* 执行cmd函数,并检查函数返回值. 若函数执行错误, 则记录日志, 并返回错误码 */
#define HCCL_RET_IF_RUN_FAILED(cmd)                                   \
    do {                                                              \
        HcclResult __ret__ = cmd;                                     \
        if (__ret__ != HCCL_SUCCESS) {                                \
            HCCL_ERROR("run [%s] failed, return[%d]", #cmd, __ret__); \
            return __ret__;                                           \
        }                                                             \
    } while (0)

/* 执行cmd函数,并检查函数返回值. 若函数执行错误, 则记录日志, 并返回指定的错误码 */
#define HCCL_RET_VAVLE_IF_RUN_FAILED(cmd, ret)                         \
    do {                                                               \
        s32 __ret__ = cmd;                                             \
        if (__ret__) {                                                 \
            HCCL_ERROR("failed: run [%s], return[%d]", #cmd, __ret__); \
            return ret;                                                \
        }                                                              \
    } while (0)

#define HCCL_EXECUTE_CMD(condition, cmd) \
    do {                                 \
        if (condition) {                 \
            cmd;                         \
        }                                \
    } while (0)

/* 执行cmd函数,并检查函数返回值. 若函数执行错误, 则记录日志, 并返回NULL */
#define HCCL_RET_NULL_IF_RUN_FAILED(cmd)                              \
    do {                                                              \
        HcclResult __ret__ = cmd;                                     \
        if (__ret__ != HCCL_SUCCESS) {                                \
            HCCL_ERROR("run [%s] failed, return[%d]", #cmd, __ret__); \
            return NULL;                                              \
        }                                                             \
    } while (0)

#ifndef T_DESC
#define T_DESC(_msg, _y) (_y)
#endif

#if T_DESC("公共常量及宏", 1)
/* 公共模块函数返回值定义,跟业务层同步  */
enum {
    SAL_OK = HCCL_SUCCESS,
    SAL_E_ERROR = HCCL_E_INTERNAL,
    SAL_E_MEMORY = HCCL_E_MEMORY,
    SAL_E_PARA = HCCL_E_PARA,
    SAL_E_NOT_FOUND = HCCL_E_NOT_FOUND,
    SAL_E_TIMEOUT = HCCL_E_TIMEOUT,
    SAL_E_UNAVAIL = HCCL_E_UNAVAIL
};

enum { SAL_FALSE = 0, SAL_TRUE = 1 };

enum { SAL_DISABLE = 0, SAL_ENABLE = 1 };
#endif

#if T_DESC("异常处理类定义", 1)
class sal_except : public std::exception {
protected:
    s32 ret;         // 错误码
    std::string msg; // 错误信息
public:
    /* 构造函数 */
    explicit sal_except(const std::string& msg, s32 ret) : ret(ret), msg(msg) {}
    explicit sal_except(const std::string& msg) : ret(0), msg(msg) {}
    explicit sal_except(s32 ret) : ret(ret), msg("") {}

    /* 析构函数 */
    ~sal_except() throw() {}

    /* 返回出错信息 */
    virtual const char* what() const throw() { return msg.c_str(); }
    /* 返回错误代码 */
    virtual s32 get_ret() const throw() { return ret; }
};
#endif

#if T_DESC("跨进程处理函数", 1)
#define SAL_SHARE_MEM_EXTRA_ID_BYTES 79
#define SAL_SHARE_MEM_UNIQUE_ID_BYTES (((SAL_UNIQUE_ID_BYTES + SAL_SHARE_MEM_EXTRA_ID_BYTES) * 4 + 3) / 4)
extern void* sal_share_memory_create(const char* uniqueId, u64 memSize);
extern void sal_share_memory_destroy(void* ptr);

#define SAL_DMEM_UNIQUE_ID_BYTES SAL_SHARE_MEM_UNIQUE_ID_BYTES
#define SAL_DMEM_NAME_MAX_BYTES (SAL_UNIQUE_ID_BYTES + 69)
#define SAL_DMEM_UNIQUE_ID_PREFIX "hccl-dmem-"
#define DEV_MEM_SHARE_NUM_MAX 1000
#define IPC_SET_NAME_COUNT_MAX 1000
#endif

#if T_DESC("信号量及互斥锁适配", 1)
/* 信号量接口  */
typedef struct sal_sem_s {
    char SalOpaqueType;
}* sal_sem_t;

#define SAL_UNIQUE_ID_BYTES (27)
#define SAL_SEM_UNIQUE_ID_BYTES (SAL_UNIQUE_ID_BYTES + 9)
#define SAL_SEM_UNIQUE_ID_PREFIX "hccl-sem-"
#define SAL_SEM_FOREVER (-1)

/* 创建信号量  */
extern sal_sem_t sal_sem_create(const char* desc, s32 binary, s32 initialCount, s32 canBeShared = 0);
/* 获取跨进程信号量的唯一id */
extern HcclResult sal_sem_get_unique_id(sal_sem_t sem, char* uniqueId);
/* 打开另外一个进程创建的信号量 */
extern sal_sem_t sal_sem_open(const char* semUniqueId);
/* 销毁信号量  */
extern void sal_sem_destroy(sal_sem_t sem);
/* 获取信号量  */
extern HcclResult sal_sem_take(sal_sem_t sem, s32 usec);
/* 释放信号量  */
extern HcclResult sal_sem_give(sal_sem_t sem);

/* 互斥锁接口  */
typedef struct sal_mutex_s {
    char MutexOpaqueType;
}* sal_mutex_t;

#define SAL_MUTEX_UNIQUE_ID_BYTES (SAL_UNIQUE_ID_BYTES + 11)
#define SAL_MUTEX_UNIQUE_ID_PREFIX "hccl-mutex-"
#define SAL_MUTEX_FOREVER (-1)

/* 创建互斥锁, 支持递归, 优先级继承.  */
extern sal_mutex_t sal_mutex_create(const char* desc, s32 canBeShared = 0);
/* 获取跨进程互斥锁的唯一id */
extern HcclResult sal_mutex_get_unique_id(sal_mutex_t mutex, char* uniqueId);
/* 打开另外一个进程创建的互斥锁 */
extern sal_mutex_t sal_mutex_open(const char* mutexUniqueId);
/* 销毁互斥锁  */
extern void sal_mutex_destroy(sal_mutex_t mutex);
/* 获取互斥锁  */
extern HcclResult sal_mutex_take(sal_mutex_t mutex, s32 usec);
/* 释放互斥锁  */
extern HcclResult sal_mutex_give(sal_mutex_t mutex);
#endif

#if T_DESC("线程处理适配", 1)

typedef struct sal_thread_s {
    char thread_opaque_type;
}* sal_thread_t;

/* 创建线程 */
extern sal_thread_t sal_thread_create(std::string name, void* (*f)(void*), void* arg);
/* 销毁线程(危险, 线程被强制停止后, 线程持有的互斥锁/内存等资源可能无法释放) */
extern HcclResult sal_thread_destroy(sal_thread_t thread);
/* 检查线程是否在运行 */
extern s32 sal_thread_is_running(sal_thread_t thread);
/* 打印当前所有正在运行的线程 */
HcclResult sal_thread_show(void);
/* 返回当前线程句柄 */
sal_thread_t sal_thread_self(void);
/* 退出当前线程 */
void sal_thread_exit(void* rc);

#endif

#if T_DESC("sal资源管理", 1)
extern HcclResult sal_etaction_regist(void (*etaction)(void*), void* etpara);
extern void sal_etaction_unregist(void (*etaction)(void*), const void* etpara);
#endif

#if T_DESC("信号量及互斥锁适配", 1)

#define SAL_MUTEX_DESC_BYTES (24)
#define SAL_SEM_DESC_BYTES (24)

extern HcclResult sal_compute_timeout(struct timespec* ts, s32 usec);
/*
 * recursive_mutex_t
 *
 *   This is an abstract type built on the POSIX mutex that allows a
 *   mutex to be taken recursively by the same thread without deadlock.
 *
 *   The Linux version of pthreads supports recursive mutexes
 *   (a non-portable extension to posix). In this case, we
 *   use the Linux support instead of our own.
 */
typedef struct recursive_mutex_s {
    pthread_mutex_t mutex;                    // 底层互斥锁实现
    u32 shared;                               // 是否允许跨进程使用
    u32 ref_cnt;                              // 互斥锁被打开次数
    char rootInfo[SAL_MUTEX_UNIQUE_ID_BYTES]; // 跨进程共享时的唯一句柄
    char desc[SAL_MUTEX_DESC_BYTES];          // 互斥锁描述信息
} recursive_mutex_t;

/* semaphore  */
/*
 * Wrapper class to hold additional info
 * along with the semaphore.
 */
typedef struct {
    sem_t s;                                // 底层信号量实现
    s32 shared;                             // 是否允许跨进程使用
    u32 ref_cnt;                            // 信号量被打开次数
    char rootInfo[SAL_SEM_UNIQUE_ID_BYTES]; // 跨进程共享时的唯一句柄
    char desc[SAL_SEM_DESC_BYTES];          // 信号量描述信息
    s32 binary;
} wrapped_sem_t;

#endif

#if T_DESC("线程处理适配", 1)

/* 新线程默认栈大小,暂定8M */
#define SAL_PTHREAD_DEFAULT_STACK_SIZE (1024 * 1024 * 8)

typedef struct tag_thread_info {
    void* (*f)(void*); // 线程的业务处理函数.
    void* arg;         // 线程业务处理函数的参数.

    std::string name; // 线程名称
    pthread_t id;     // 线程句柄
    pid_t pid;        // 线程PID
    s32 ss;           // 线程栈大小
    sal_sem_t sem;    // 线程同步信号量
} thread_info_t;

#endif

#if T_DESC("跨进程处理函数", 1)

/* 跨进程共享内存信息 */
typedef struct tag_share_mem_s {
    char rootInfo[SAL_SHARE_MEM_UNIQUE_ID_BYTES];
    u32 mem_size;
    u32 ref_cnt;
    void* relate_ptr[IPC_SET_NAME_COUNT_MAX];
    u32 relate_ptr_cnt = 0;
    u32 open_ref_cnt = 0;
    char user_data[1];
} share_mem_t;

#endif

#if T_DESC("时间处理接口适配", 1)
/* 常用时间定义
   基础单位 us  */
#define SAL_MILLISECOND_USEC 1000            // 1ms等于1000us
#define SAL_SECOND_USEC 1000000              // 1s等于1000000us
#define SAL_MINUTE_USEC 60 * SAL_SECOND_USEC // 1min等于60s
#define SAL_HOUR_USEC 60 * SAL_MINUTE_USEC   // 1h等于60min

/* 基础单位 ms  */
#define SAL_SECOND_MSEC 1000                 // 1s等于1000ms
#define SAL_MINUTE_MSEC 60 * SAL_SECOND_MSEC // 1min等于60s

/* 基础单位 ns  */
#define SAL_SECOND_NSEC 1000 * 1000 // 1s等于1000*1000纳秒

#define LLT_SOCKET_SLEEP_MILLISECONDS 1
#define LLT_ONE_HUNDRED_MICROSECOND_OF_USLEEP 100
#define LLT_ONE_MILLISECOND_OF_USLEEP 1000
#endif

#if T_DESC("日志处理适配", 1)

/* 信息直接打印到终端  */
#define SAL_PRINT(...)       \
    do {                     \
        printf(__VA_ARGS__); \
    } while (0)

// 获取当前时间字符串
HcclResult sal_get_current_time(char* timeStr, u32 len);

#endif

#if T_DESC("sal资源管理", 1)
typedef struct tag_sal_thread_list_node_s {
    thread_info_t* thread_info;
} sal_thread_list_node_t;
/* sal管理信息, 进程退出时销毁 */
class sal_manage_class {
private:
    std::list<sal_thread_list_node_t*> sal_thread_list; // 通过链表来管理 sal 创建的所有线程.
    s32 sal_actived; // 状态管理, active状态允许共享资源操作,destroy后不允许操作.
    // 删除拷贝构造函数和拷贝赋值函数, 禁止使用
    sal_manage_class(const sal_manage_class&) = delete;
    sal_manage_class& operator=(const sal_manage_class&) = delete;

public:
    std::mutex sal_manage_lock;
    u32 len_of_thread_res; // 各类资源管理链表的长度
    sal_manage_class();
    ~sal_manage_class();
    s32 sal_active_get(void) const;
    HcclResult sal_thread_regist(const thread_info_t* thread);
    void sal_thread_unregist(const thread_info_t* thread);
    HcclResult sal_thread_find(pthread_t id, thread_info_t** ti);
    HcclResult sal_thread_info_pop(u32 pos, thread_info_t** ti);
};
extern s32 sal_vsnprintf(char* strDest, size_t destMaxSize, size_t count, const char* format, va_list argList);
extern HcclResult sal_memset(void* dest, size_t destMaxSize, int c, size_t count);
extern HcclResult sal_memcpy(void* dest, size_t destMaxSize, const void* src, size_t count);
extern HcclResult sal_strncpy(char* strDest, size_t destMaxSize, const char* strSrc, size_t count);

#endif

HcclResult SalGetUniqueId(char* salUniqueId, int maxLen);
HcclResult rt_stream_synchronize(void* stream);
void* sal_malloc(u32 size); // 主机侧申请堆内存接口
void sal_free(void* ptr);   // 主机侧释放堆内存接口

// u32 IpStr2Num(const std::string &strIp);
// std::string IpNum2Str(const u32 numIp);

#endif // __LLT_SAL_PUB_H__
