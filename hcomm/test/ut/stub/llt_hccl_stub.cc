/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <driver/ascend_hal.h>
#include "rt_external.h"
#include <hccl/base.h>
#include <hccl/hccl_types.h>
#include <securec.h>
#include <unistd.h>
#include <signal.h>
#include <syscall.h>
#include <sys/prctl.h>
#include <syslog.h>
#include <sys/mman.h>
#include <sys/stat.h> /* For mode constants */
#include <fcntl.h>    /* For O_* constants */
#include <arpa/inet.h>
#include <time.h>
#include <map>

#include <stdlib.h>
#include <set>
#include <mutex>
#include <utility>

#include "llt_hccl_stub.h"
#include "llt_hccl_stub_gdr.h"
#include "llt_hccl_stub_profiling_plugin.h"
#include "llt_hccl_stub_fp16.h"

#include "adapter_rts.h"
#include "adapter_hal.h"
#include "hcomm_res_defs.h"
#include "hcomm_diag.h"

#include "runtime/rt_error_codes.h"
#include "mmpa_api.h"

#include "hccl_socket.h"
#include "hccl_socket_manager.h"
#include "dlog_pub.h"
#include "acl/acl_rt.h"
#include "adapter_tdt.h"
#include "acl/acl_rt.h"
#include "urma_api.h"

/*----------------------------------------------*
 * 外部变量说明                                 *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 外部函数原型说明                             *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 内部函数原型说明                             *
 *----------------------------------------------*/
/*----------------------------------------------*
 * 全局变量                                     *
 *----------------------------------------------*/
/*记录当前操作的设备*/
__thread int32_t current_dev;
__thread int32_t current_die;
__thread rtDeviceMode current_devMode;

/*记录当前操作的设备*/
u32 gBoardId;
uint32_t gDevPhyId;
u32 gIsVM = 0;                        // 当前是否为虚拟机
static s32 chip_type_stub[256] = {0}; /*最大为16，下面不再做判断*/

/*----------------------------------------------*
 * 模块级变量                                   *
 *----------------------------------------------*/

/*----------------------------------------------*
 * 常量定义                                     *
 *----------------------------------------------*/
constexpr u64 GIGABYTE_TO_BYTE = 1024ULL * 1024ULL * 1024ULL;

/*----------------------------------------------*
 * 宏定义                                       *
 *----------------------------------------------*/

static s32 GetLogLevelFromEnv()
{
    const char* env = getenv("HCCL_UT_LOG_LEVEL");
    if (env != nullptr) {
        int val = atoi(env);
        if (val >= 0 && val <= 3) {
            return val;
        }
    }
    return DLOG_ERROR;
}
static s32 stub_log_level = GetLogLevelFromEnv();
static u32 FailureDeviceId = 0xFFFFFFFF;
static tasktype_e FailureTaskType = TASK_TYPE_RESERVED;
static std::mutex taskFailCallbackMapMutex;
static std::mutex taskAbortCallbackMapMutex;
static std::mutex isExecutedMutex;
static std::map<string, rtTaskFailCallback> taskFailCallbackMap;
static std::map<string, aclrtDeviceTaskAbortCallback> taskAbortCallbackMap;
s32 log_level_get_stub() { return stub_log_level; }
void log_level_set_stub(s32 log_level) { stub_log_level = log_level; }
string get_log_str_from_type_stub(s32 type)
{
    string str = "";
    switch (type) {
        case DLOG_DEBUG:
            str = "[DEBUG]";
            break;
        case DLOG_INFO:
            str = "[INFO]";
            break;
        case DLOG_WARN:
            str = "[WARNING]";
            break;
        case DLOG_ERROR:
            str = "[ERROR]";
            break;
        // case DLOG_EVENT:
        //     str = "[EVENT]";
        //     break;
        default:
            break;
    }
    return str;
}
using DevicePlaneInfo_t = struct DevicePlaneInfo {
    s32 devicePhyId; // 服务器内device唯一标识
    s32 planeId;
    bool operator==(const s32& devicePhyId) { return (this->devicePhyId == devicePhyId); }
};
std::vector<DevicePlaneInfo_t> DevicePlaneList;
// 记录日志时获取当前时间字符串
HcclResult sal_get_current_time(char* timeStr, u32 len)
{
    struct timeval tv;
    time_t tmpt;
    struct tm* now;

    if (timeStr == NULL) {
        return HCCL_E_PARA;
    }

    if (0 > gettimeofday(&tv, NULL)) {
        return HCCL_E_INTERNAL;
    }

    tmpt = (time_t)tv.tv_sec;
    now = localtime(&tmpt);
    if (now == NULL) {
        return HCCL_E_INTERNAL;
    }

    int iLen = snprintf_s(
        timeStr, len, len, "%04d-%02d-%02d %02d:%0d:%02d.%06u", now->tm_year + TIME_FROM_1900, now->tm_mon + 1,
        now->tm_mday, now->tm_hour, now->tm_min, now->tm_sec, (u32)tv.tv_usec);
    if (iLen == -1) {
        HCCL_WARNING(
            "Print time failed[%d]."
            "params: time[%s], len[%u], time_format:%04d-%02d-%02d %02d:%02d:%02d.%06u",
            iLen, timeStr, len, now->tm_year + TIME_FROM_1900, now->tm_mon + 1, now->tm_mday, now->tm_hour, now->tm_min,
            now->tm_sec, (u32)tv.tv_usec);
    }

    return HCCL_SUCCESS;
}

/*
 *****************************************************************************
 * 函 数 名  : sal_dlog_printf_stub
 * 功能描述  : 输出错误log
 * 输入参数  : int module_id, const char *fmt, ...)
 * 输出参数  : 无
 * 返 回 值  : void
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   :  2018年6月25日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
void sal_dlog_printf_stub(int level, char* log_buffer)
{
    char time_stamp[LOG_TIME_STAMP_SIZE] = {0}; /* 缓存时间戳  */

    /* 获取时间标签 */
    (void)sal_memset(time_stamp, LOG_TIME_STAMP_SIZE, 0, sizeof(time_stamp));
    (void)sal_get_current_time(time_stamp, LOG_TIME_STAMP_SIZE);

    string str_type = get_log_str_from_type_stub(level);
    printf("[%-26s][pid:%u][tid:%u]%s  %s\n", time_stamp, SalGetPid(), SalGetTid(), str_type.c_str(), log_buffer);
}
/*
 *****************************************************************************
 * 函 数 名  : Dlog
 * 功能描述  : 输出错误log
 * 输入参数  : int moduleId, int level, const char *fmt, ...
 * 输出参数  : 无
 * 返 回 值  : void
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   :  2018年6月25日
 *    作    者   : ligang 00442453
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
void DlogRecord(int moduleId, int level, const char* fmt, ...)
{
    if (level < stub_log_level) {
        return;
    }

    char stack_log_buffer[LOG_TMPBUF_SIZE]; /* 优先使用栈中的buffer, 小而快  */
    va_list arg;
    (void)va_start(arg, fmt); // lint !e530
    (void)sal_memset(stack_log_buffer, LOG_TMPBUF_SIZE, 0, sizeof(stack_log_buffer));
    /*
        C库标准的vsnprintf()函数在字符串超出缓存长度后返回需要的缓存空间.
        公司的安全函数库包装后的vsnprintf_s()在字符串超出缓存长度后返回值为-1, 无法根据返回值重新申请堆内存.
    */
    sal_vsnprintf(stack_log_buffer, sizeof(stack_log_buffer), (sizeof(stack_log_buffer) - 1), fmt, arg);
    va_end(arg);

    sal_dlog_printf_stub(level, stack_log_buffer);
}

void DlogInner(int moduleId, int level, const char* fmt, ...)
{
    if (level < stub_log_level) {
        return;
    }

    char stack_log_buffer[LOG_TMPBUF_SIZE]; /* 优先使用栈中的buffer, 小而快  */
    va_list arg;
    (void)va_start(arg, fmt); // lint !e530
    (void)sal_memset(stack_log_buffer, LOG_TMPBUF_SIZE, 0, sizeof(stack_log_buffer));
    /*
        C库标准的vsnprintf()函数在字符串超出缓存长度后返回需要的缓存空间.
        公司的安全函数库包装后的vsnprintf_s()在字符串超出缓存长度后返回值为-1, 无法根据返回值重新申请堆内存.
    */
    sal_vsnprintf(stack_log_buffer, sizeof(stack_log_buffer), (sizeof(stack_log_buffer) - 1), fmt, arg);
    va_end(arg);

    sal_dlog_printf_stub(level, stack_log_buffer);
}

int CheckLogLevel(int moduleId, int level) { return 1; }

int dlog_getlevel(int moduleId, int* enableEvent) { return 0; }

int dlog_setlevel(int moduleId, int level, int enableEvent) { return 0; }

rtError_t rtStreamCreate(rtStream_t* stream, int32_t priority)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream;
    s32 device_id;
    aclrtGetDevice(&device_id);
    if ((rtstream = new (std::nothrow) stream_class(device_id)) == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    // 记录当前设备信息
    rtstream->current_dev = current_dev;

    *stream = (rtStream_t)rtstream;
    // Mod for optimize runtime Stub by l on 2018-01-11 Above
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return ACL_ERROR_RT_PARAM_INVALID for error input priority
 */
aclError aclrtCreateStream(aclrtStream* stream)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream;
    s32 device_id;
    aclrtGetDevice(&device_id);
    if ((rtstream = new (std::nothrow) stream_class(device_id)) == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    // 记录当前设备信息
    rtstream->current_dev = current_dev;

    *stream = (rtStream_t)rtstream;
    // Mod for optimize runtime Stub by l on 2018-01-11 Above
    return ACL_SUCCESS;
}
/**
 * @ingroup dvrt_base
 * @brief register callback for fail task
 * @param [in] uniName unique register name, can't be null
 * @param [in] callback fail task callback function
 * @param [out] NA
 * @return RT_ERROR_NONE for ok
 */
rtError_t rtRegTaskFailCallbackByModule(const char_t* moduleName, rtTaskFailCallback callback)
{
    string tmpStr = string(moduleName);
    std::unique_lock<std::mutex> lock(taskFailCallbackMapMutex);
    taskFailCallbackMap.clear();
    taskFailCallbackMap.insert({tmpStr, callback});
    return ACL_SUCCESS;
}
aclError aclrtSetDeviceTaskAbortCallback(const char_t* moduleName, aclrtDeviceTaskAbortCallback callback, void* args)
{
    string tmpStr(moduleName);
    void* p = nullptr;
    std::unique_lock<std::mutex> lock(taskAbortCallbackMapMutex);
    taskAbortCallbackMap.clear();
    taskAbortCallbackMap.insert({tmpStr, callback});
    return RT_ERROR_NONE;
}

aclError aclrtRegDeviceStateCallback(const char* regName, aclrtDeviceStateCallback callback, void* args)
{
    return ACL_SUCCESS;
}

aclError aclrtGetLogicDevIdByPhyDevId(const int32_t phyDevId, int32_t* const logicDevId) { return RT_ERROR_NONE; }

rtError_t rtNotifyResetAll() { return RT_ERROR_NONE; }
rtError_t TaskFailCallbackClean()
{
    taskFailCallbackMap.clear();
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_stream
 * @brief inquire max stream count and max task count per stream
 * @param [in] streamType   Stream Type
 * @param [in] MaxStrCount   Max stream count
 * @param [in] MaxTaskCount   max task count per stream
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_VALUE for error input
 */
rtError_t rtGetMaxStreamAndTask(uint32_t streamType, uint32_t* maxStrCount, uint32_t* maxTaskCount)
{
    *maxStrCount = 1024;
    *maxTaskCount = 1024;
    return RT_ERROR_NONE;
}
/**
 * @ingroup dvrt_stream
 * @brief create stream instance
 * @param [in|out] stream   created stream
 * @param [in] priority   stream priority
 * @param [in] flags  stream op flags
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 * @return ACL_ERROR_RT_PARAM_INVALID for error input priority
 */
aclError aclrtCreateStreamWithConfig(aclrtStream* stream, uint32_t priority, uint32_t flag)
{
    return aclrtCreateStream(stream);
}

aclError aclrtSetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAttribute(aclrtStream stream, aclrtStreamAttr stmAttrType, aclrtStreamAttrValue* value)
{
    return ACL_SUCCESS;
}

aclError aclrtGetStreamAvailableNum(uint32_t* streamCount) { return ACL_SUCCESS; }

std::mutex g_tidStreamMapMutex;
std::map<uint64_t, std::vector<void*>> g_tidStreamMap;

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stream   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
rtError_t rtStreamDestroy(rtStream_t stream)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;
    if (NULL != rtstream) {
        delete rtstream;
        rtstream = NULL;
    }
    std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
    g_tidStreamMap.clear();
    lock.unlock();

    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return RT_ERROR_NONE;
}

/*add ipc stub optimize*/
std::map<void*, u32> g_ipcBasePtrMap;
std::mutex g_ipcMtx;
void rtIpcBasePtrAdd(void* ptr, u32 size)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    g_ipcBasePtrMap[ptr] = size;
    return;
}
void rtIpcBasePtrErase(void* ptr)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    g_ipcBasePtrMap.erase(ptr);
    return;
}

void* rtIpcBasePtrLookup(const void* ptr)
{
    std::lock_guard<std::mutex> lock(g_ipcMtx);
    for (const auto& pair : g_ipcBasePtrMap) {
        if ((ptr >= pair.first) && (static_cast<const u8*>(ptr) < static_cast<u8*>(pair.first) + pair.second)) {
            return pair.first;
        }
    }
    return nullptr;
}
std::map<void*, u32> g_ipcOpenBasePtrMap;
std::mutex g_openIpcMtx;
void rtIpcOpenBasePtrAdd(void* ptr, u32 size)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    g_ipcOpenBasePtrMap[ptr] = size;
    return;
}
void rtIpcOpenBasePtrErase(void* ptr)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    g_ipcOpenBasePtrMap.erase(ptr);
    return;
}

void* rtIpcOpenBasePtrLookup(const void* ptr)
{
    std::lock_guard<std::mutex> lock(g_openIpcMtx);
    for (const auto& pair : g_ipcOpenBasePtrMap) {
        if ((ptr >= pair.first) && (static_cast<const u8*>(ptr) < static_cast<u8*>(pair.first) + pair.second)) {
            return pair.first;
        }
    }
    return nullptr;
}

/**
 * @ingroup dvrt_stream
 * @brief destroy stream instance.
 * @param [in] stream   the stream to destroy
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
aclError aclrtDestroyStream(aclrtStream stream)
{
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;
    if (NULL != rtstream) {
        delete rtstream;
        rtstream = NULL;
    }
    std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
    g_tidStreamMap.clear();
    lock.unlock();

    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return ACL_SUCCESS;
}

aclError aclrtDestroyStreamForce(aclrtStream stream) { return aclrtDestroyStream(stream); }

/**
 * @ingroup dvrt_stream
 * @brief get stream id from a stream handle
 * @param [in] stream   stream handle
 * @param [in] streamId   stream id
 * @return RT_ERROR_NONE for complete
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream handle
 */
aclError aclrtStreamGetId(aclrtStream stream, int32_t* streamId)
{
    static int32_t stream_id_counter = 0;
    static std::map<rtStream_t, int32_t> streamMap;
    static std::mutex mapMutex;

    if (streamId == nullptr || stream == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    stream_class* rtstream = (stream_class*)stream;
    *streamId = rtstream->get_stream_id();

    return ACL_SUCCESS;
}

/* 记录task_info*/
thread_local task_info_t task_info;

/**
 * @ingroup dvrt_base
 * @brief get current thread last stream id and task id
 * @param [out] stream id and task id
 * @param [in] null
 * @return RT_ERROR_NONE for ok
 * @return ACL_ERROR_RT_PARAM_INVALID for input null ptr
 */
rtError_t rtGetTaskIdAndStreamID(uint32_t* taskId, uint32_t* streamId)
{
    *streamId = task_info.streamId;
    *taskId = task_info.taskId;
    return RT_ERROR_NONE;
}

// Add for optimize runtime Stub by l on 2018-01-11 Below
/**
 * @ingroup dvrt_stream
 * @brief wait stream to be complete
 * @param [in] stream   stream to wait
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_RESOURCE_HANDLE for error input stream or event handle
 */
aclError aclrtSynchronizeStream(aclrtStream stream)
{
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;
    rtstream->stream_synchronize();
    return ACL_SUCCESS;
}

aclError aclrtSynchronizeStreamWithTimeout(aclrtStream stream, int32_t timeout)
{
    return aclrtSynchronizeStream(stream);
}

static u32 modelCallBackStub = 0; // 0:host网卡 1：HVD
aclError aclrtActiveStream(aclrtStream activeStream, aclrtStream stream) { return ACL_SUCCESS; }

int rtModelFake = 0;
aclError aclmdlRICaptureGetInfo(aclrtStream stream, aclmdlRICaptureStatus* status, aclmdlRI* modelRI)
{
    *modelRI = &rtModelFake;
    return ACL_SUCCESS;
}

aclError aclmdlRICaptureThreadExchangeMode(aclmdlRICaptureMode* mode) { return ACL_SUCCESS; }

aclError aclrtBinaryLoadFromData(
    const void* data, size_t length, const aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    static int i = 0;
    *binHandle = &i;
    return ACL_SUCCESS;
}

aclError aclrtGetFunctionAddr(aclrtFuncHandle funcHandle, void** aicAddr, void** aivAddr)
{
    static int i = 0;
    *aivAddr = &i;
    return ACL_SUCCESS;
}

rtError_t rtStreamAddToModel(rtStream_t stm, rtModel_t captureMdl)
{
    captureMdl = &rtModelFake;
    return RT_ERROR_NONE;
}

void SetRtCallbackModleStub(u32 model) { modelCallBackStub = model; }

aclError aclrtLaunchCallback(aclrtCallback fn, void* userData, aclrtCallbackBlockType blockType, aclrtStream stream)
{
    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_CALLBACK_FUNC;
    stream_task.stream_para.callbackTask.func = fn;
    stream_task.stream_para.callbackTask.para = userData;
    stream_task.stream_para.callbackTask.isExecuted = 0;
    stream_task.stream_para.callbackTask.isBlock = (blockType == ACL_CALLBACK_BLOCK);
    rtstream->push_task(&stream_task);
    return ACL_SUCCESS;
}

aclError aclrtSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    if (modelCallBackStub != 0) {
        return ACL_SUCCESS;
    }
    std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
    g_tidStreamMap[threadId].push_back(stream);
    lock.unlock();
    return ACL_SUCCESS;
}

aclError aclrtProcessReport(int32_t timeout)
{
    if (modelCallBackStub != 0) {
        return ACL_SUCCESS;
    }
    uint64_t threadId = pthread_self();
    auto iter = g_tidStreamMap.find(threadId);
    if (iter == g_tidStreamMap.end()) {
        HCCL_ERROR("this thread id[%llu] has not been registered", threadId);
        return 1;
    }
    for (auto stream : g_tidStreamMap[threadId]) {
        stream_class* rtstream = (stream_class*)stream;
        if (rtstream->stream_task_list.front().task_type == TASK_TYPE_CALLBACK_FUNC
            && rtstream->stream_task_list.front().stream_para.callbackTask.isExecuted == 0) {
            rtstream->ExecuteCallbackFunc();
        } else {
            SaluSleep(timeout * 10);
        }
    }
    return ACL_SUCCESS;
}

aclError aclrtUnSubscribeReport(uint64_t threadId, aclrtStream stream)
{
    size_t tmpVectorSize = g_tidStreamMap[threadId].size();
    for (size_t i = 0; i < tmpVectorSize; ++i) {
        if (g_tidStreamMap[threadId][i] == stream) {
            std::unique_lock<std::mutex> lock(g_tidStreamMapMutex);
            g_tidStreamMap[threadId].erase(g_tidStreamMap[threadId].begin() + i);
            lock.unlock();
            HCCL_INFO("llt rt UnSubscribe Report success thread[%llu], stream[%p]", threadId, stream);
            return ACL_SUCCESS;
        }
    }
    HCCL_INFO(
        "llt rt UnSubscribe Report success thread[%llu] stream[%p] tmpVectorSize[%d]", threadId, stream, tmpVectorSize);
    return ACL_SUCCESS;
}

aclError aclrtStreamWaitEvent(aclrtStream stream, aclrtEvent event) { return ACL_SUCCESS; }

aclError aclrtResetEvent(aclrtEvent event, aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtCreateEventWithFlag(aclrtEvent* event, uint32_t flag) { return aclrtCreateEvent(event); }

rtError_t rtModelBindStream(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtModelUnbindStream(rtModel_t model, rtStream_t stream) { return RT_ERROR_NONE; }

rtError_t rtModelExecute(rtModel_t model, rtStream_t stream, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtModelCreate(rtModel_t* model, uint32_t flag)
{
    *model = (rtModel_t)1;
    return RT_ERROR_NONE;
}
aclError aclrtGetDeviceCount(uint32_t* count);

/*****************************************************************************
 函 数 名  : aclrtGetDeviceCount
 功能描述  : 获取设备数量
 输入参数  : uint32_t *count
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

*****************************************************************************/
aclError aclrtGetDeviceCount(uint32_t* count)
{
    /*打桩函数先默认设备上芯片数量为8*/
    *count = 8;

    return ACL_SUCCESS;
}

rtError_t rtSetDeviceV2(int32_t device, rtDeviceMode deviceMode)
{
    current_dev = device;
    current_devMode = deviceMode;
    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : aclrtSetDevice
 功能描述  : 设置当前操作的设备
 输入参数  : int32_t device
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtSetDevice(int32_t deviceId)
{
    current_dev = deviceId;
    return ACL_SUCCESS;
}

rtError_t rtSetDie(int32_t die)
{
    current_die = die;

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : aclrtResetDevice
 功能描述  : 设置当前操作的设备
 输入参数  : int32_t deviceId
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2019年05月09日
    作    者   : w00500539
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtResetDevice(int32_t deviceId)
{
    current_dev = deviceId;

    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtGetDevice
 功能描述  : 查询当前操作的设备
 输入参数  : int32_t *device
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtGetDevice(int32_t* device)
{
    *device = current_dev;
    return ACL_SUCCESS;
}

rtError_t rtGetDie(int32_t* die)
{
    *die = current_die;
    return RT_ERROR_NONE;
}

aclError aclrtGetCurrentContext(aclrtContext* ctx)
{
    void* tmp;
    *ctx = tmp;
    return ACL_SUCCESS;
}

aclError aclrtSetCurrentContext(aclrtContext ctx) { return ACL_SUCCESS; }

aclError aclrtGetDevicesTopo(uint32_t devId, uint32_t otherDevId, uint64_t* value)
{
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {
        if ((devId / 8) != (otherDevId / 8)) {
            *value = ACL_RT_DEVS_TOPOLOGY_PIX; // PXI
        } else {
            *value = ACL_RT_DEVS_TOPOLOGY_HCCS; // HCCS
        }
    }
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
        // 0-1 2-3 4-5 6-7
        int32_t diff = static_cast<int32_t>(devId) - static_cast<int32_t>(otherDevId);
        if ((abs(diff) == 1) && ((devId + otherDevId) % 4 == 1)) {
            *value = ACL_RT_DEVS_TOPOLOGY_SIO; // SIO
        } else {
            *value = ACL_RT_DEVS_TOPOLOGY_HCCS_SW; // HCCS_SW
        }
        return ACL_SUCCESS;
    }

    if ((gBoardId == 0x1E || gIsVM == 1) || (devId / 4) != (otherDevId / 4)) // 若当前为标卡/虚拟机/非同一clustor
    {
        *value = ACL_RT_DEVS_TOPOLOGY_PIX; // PXI
    } else {
        *value = ACL_RT_DEVS_TOPOLOGY_HCCS; // HCCS
    }

    return ACL_SUCCESS;
}

aclError aclrtGetDeviceSatMode(aclrtFloatOverflowMode* floatOverflowMode)
{
    *floatOverflowMode = ACL_RT_OVERFLOW_MODE_INFNAN;
    return ACL_SUCCESS;
}

aclError aclrtSetDeviceSatMode(aclrtFloatOverflowMode mode) { return ACL_SUCCESS; }

/*****************************************************************************
 函 数 名  : aclrtMemcpyAsync
 功能描述  : 异步内存拷贝
 输入参数  : void *dst
             void *src
             uint64_t count
             aclrtMemcpyKind kind
             aclrtStream stream
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError
aclrtMemcpyAsync(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind, aclrtStream stream)
{
    /*拷贝类型检测*/
    if ((kind < ACL_MEMCPY_HOST_TO_HOST) || (kind > ACL_MEMCPY_DEVICE_TO_DEVICE)) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if ((!dst) || (!src)) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /*桩函数中不管kind，都统一使用memcpy完成*/
    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    // 将asynchronous Memcpy任务，压入任务队列
    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_MEMCPY;
    stream_task.stream_para.memcpystruct.dst = dst;
    stream_task.stream_para.memcpystruct.src = (void*)src;
    stream_task.stream_para.memcpystruct.count = count;

    rtstream->push_task(&stream_task);
    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return ACL_SUCCESS;
}

aclError aclrtMemcpyAsyncWithOffset(
    void** dst, size_t destMax, size_t dstDataOffset, const void** src, size_t cnt, size_t srcDataOffset,
    aclrtMemcpyKind kind, aclrtStream stm)
{
    std::cout << "aclrtMemcpyAsyncWithOffset rtStream_t: " << stm << std::endl;
    if ((!dst) || (!src) || (kind != ACL_MEMCPY_INNER_DEVICE_TO_DEVICE)) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    stream_class* rtstream = nullptr;
    rtstream = (stream_class*)stm;

    std::cout << "aclrtMemcpyAsyncWithOffset stream_class*: " << rtstream << std::endl;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_MEMCPY;
    stream_task.stream_para.memcpystruct.dst = *dst;
    stream_task.stream_para.memcpystruct.src = (void*)(*src);
    stream_task.stream_para.memcpystruct.count = cnt;

    rtstream->push_task(&stream_task);

    return RT_ERROR_NONE;
}

aclError aclrtDevicePeerAccessStatus(int32_t deviceId, int32_t peerDeviceId, int32_t* status)
{
    *status = 1;
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtCreateEvent
 功能描述  : 创建事件
 输入参数  : aclrtEvent *event
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtCreateEvent(aclrtEvent* event)
{
    /*
    1、原来的event机制需要使用者确保先下发record然后再下发wait。新的event机制没有这种限制，与notify流程类似。故打桩直接使用notify机制
    2、event为软件资源，与device无关。为了复用notify已有机制，不关注device id默认填0。*/
    return aclrtCreateNotify((aclrtNotify*)event, 0UL);
}

aclError aclrtCreateEventExWithFlag(aclrtEvent* event, uint32_t flag)
{
    return aclrtCreateNotify((aclrtNotify*)event, 0UL);
}

aclError aclrtGetEventId(aclrtEvent event, uint32_t* eventId)
{
    *eventId = 0;
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtDestroyEvent
 功能描述  : 销毁事件
 输入参数  : rtEvent_t event
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t aclrtDestroyEvent(rtEvent_t event) { return aclrtDestroyNotify((rtNotify_t)event); }

/*****************************************************************************
 函 数 名  : aclrtRecordEvent
 功能描述  : 记录事件
 输入参数  : aclrtEvent event
             aclrtStream stream
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtRecordEvent(aclrtEvent event, aclrtStream stream)
{
    /*异步下发record后返回*/
    return aclrtRecordNotify((aclrtNotify)event, stream);
}

/*阻塞查询event是否被record*/
aclError aclrtQueryEventStatus(aclrtEvent event, aclrtEventRecordedStatus* status)
{
    rt_notify_t* ipc_notify = (rt_notify_t*)event;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    HCCL_DEBUG("wait : device[%d], notify_id[%llu]", notify_shm->device_id, ipc_notify->notify_id);

    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT; // 20s
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 1, 0)) {
        SaluSleep(1000);

        timeout_cnt--;
        if (timeout_cnt <= 0) {
            HCCL_ERROR(
                "wait timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id], notify_shm->device_id, ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    *status = ACL_EVENT_RECORDED_STATUS_COMPLETE;
    return ACL_SUCCESS;
}

aclError aclrtNotifyBatchReset(aclrtNotify* notifies, size_t num) { return ACL_SUCCESS; }

rtError_t rtNotifyGetAddrOffset(rtNotify_t notify, uint64_t* devAddrOffset)
{
    if (notify == nullptr || devAddrOffset == nullptr) {
        HCCL_ERROR("parameter error : notify[%p], devAddrOffset[%p]", notify, devAddrOffset);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_notify_t* ipc_notify = (rt_notify_t*)notify;
    if (nullptr == ipc_notify->ipc_notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", ipc_notify->ipc_notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    *devAddrOffset = (u64)&ipc_notify->ipc_notify_shm->record_cnt[ipc_notify->notify_id];
    return RT_ERROR_NONE;
}

aclError aclrtStreamStop(aclrtStream stream) { return ACL_SUCCESS; }

aclError aclrtStreamAbort(aclrtStream stream) { return ACL_SUCCESS; }

s32 sal_close_name_map(void* name_map)
{
    /* 通过共享内存名称在本进程关闭name_map共享设备内存 */
    (void)sal_share_memory_destroy((void*)name_map);

    return SAL_OK;
}

s32 sal_create_name_map(const char* name, rt_name_map_stub_t** name_map)
{
    s32 ret;

    /* 名字长度不能超标 */
    if (SalStrLen(name) > SAL_DMEM_NAME_MAX_BYTES) {
        HCCL_ERROR("length of name is [%d], out of range", SalStrLen(name));
        return SAL_E_PARA;
    }

    /* 将名字添加前缀，防止该名字在别的共享内存已经使用了 */
    char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES] = {0};
    ret = snprintf_s(
        mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, name);
    if (ret == -1) {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return SAL_E_MEMORY;
    }

    // 创建或打开name_map映射表
    rt_name_map_stub_t* name_map_ptr
        = (rt_name_map_stub_t*)sal_share_memory_create(mapped_name, (sizeof(rt_name_map_stub_t)));

    if (name_map_ptr == NULL) {
        HCCL_ERROR("create share memory %s failed", mapped_name);
        return SAL_E_PARA;
    }

    *name_map = name_map_ptr;

    return SAL_OK;
}

/*****************************************************************************
 函 数 名  : aclrtMallocWithCfg & aclrtMallocForTaskScheduler
 功能描述  : 设备内存申请
 输入参数  : void **devPtr
             uint64_t size
             aclrtMemMallocPolicy policy
             aclrtMallocConfig *cfg
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数
  2.日    期   : 2018年6月12日
    作    者   : liubanglan
    修改内容   : 桩函数中rtMalloc申请共享内存，以便rtIpcSetMemoryName等跨进程操作

*****************************************************************************/

aclError aclrtMallocForTaskScheduler(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    return aclrtMallocWithCfg(devPtr, size, policy, cfg);
}

aclError aclrtMalloc(void** devPtr, size_t size, aclrtMemMallocPolicy policy)
{
    if (devPtr == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (policy > ACL_MEM_ACCESS_USER_SPACE_READONLY) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    *devPtr = malloc(size);
    return (*devPtr == nullptr) ? ACL_ERROR_RT_MEMORY_ALLOCATION : ACL_ERROR_NONE;
}

aclError aclrtMallocWithCfg(void** devPtr, size_t size, aclrtMemMallocPolicy policy, aclrtMallocConfig* cfg)
{
    char my_unique_id[SAL_UNIQUE_ID_BYTES];
    char mem_name[SAL_DMEM_UNIQUE_ID_BYTES];

    if (!devPtr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (policy > ACL_MEM_ACCESS_USER_SPACE_READONLY) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (cfg == nullptr || cfg->numAttrs == 0 || cfg->attrs == nullptr || cfg->attrs->attr != ACL_RT_MEM_ATTR_MODULE_ID
        || cfg->attrs->value.moduleId != HCCL) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* 通过uniqueid的方式，申请桩函数设备内存的名字 */
    s32 ret = SalGetUniqueId(my_unique_id);

    if (ret != SAL_OK) {
        HCCL_ERROR("Generate sal_unique_id failed[%d]", ret);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* UniqueID添加HCCL-mem-的前缀 */
    ret = snprintf_s(
        mem_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX,
        my_unique_id);

    if (ret == -1) {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return ACL_ERROR_RT_CONTEXT_NULL;
    }

    /* 封装的共享内存结构自带了管理信息,返回值是管理信息后的有效存储空间 */
    void* shm_buf = (void*)sal_share_memory_create(mem_name, size);

    if (shm_buf == nullptr) {
        HCCL_ERROR("sal_share_memory_create failed, device_memory_share_info is NULL");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    HCCL_DEBUG("aclrtMallocWithCfg sal_share_memory_create[%s] OK", mem_name);

    *devPtr = shm_buf;
    rtIpcBasePtrAdd(shm_buf, size);

    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtFree
 功能描述  : 设备内存释放
 输入参数  : void *devPtr
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数
  2.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 桩函数中rtFree释放共享内存，以便rtIpcSetMemoryName等跨进程操作

*****************************************************************************/
aclError aclrtFree(void* devPtr)
{
    if (!devPtr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /* 通过共享内存管理信息中的关联指针找到name map */
    share_mem_t* shm_head = (share_mem_t*)((char*)devPtr - offsetof(share_mem_t, user_data));
    for (int i = 0; i < shm_head->relate_ptr_cnt; i++) {
        rt_name_map_stub_t* name_map_ptr
            = (rt_name_map_stub_t*)shm_head->relate_ptr[i]; // name map的地址作为本块共享内存的关联指针
        /* 关闭该段内存的name map */
        if (nullptr != name_map_ptr) {
            HCCL_DEBUG("rtfree devPtr[%p] name_map_ptr[%p], ", devPtr, name_map_ptr);
            (void)sal_close_name_map(name_map_ptr);
            shm_head->relate_ptr[i] = nullptr;
        }
    }
    shm_head->relate_ptr_cnt = 0;
    /* destroy devPtr指向的共享内存 */
    (void)sal_share_memory_destroy(devPtr);

    rtIpcBasePtrErase(devPtr);

    return ACL_SUCCESS;
}

/*
        creat和destroy share memory中的映射表
process A                                            process B
rtIpcSetMemoryName   ----    create map
                             create map       ----       rtIpcOpenMemory
                             destroy map     ----       rtIpcOpenMemory
aclrtFree               ----    destroy map

*/

/*****************************************************************************
 函 数 名  : aclrtMallocHostWithCfg
 功能描述  :
 输入参数  : void** hostPtr
            uint64_t size
            aclrtMallocConfig cfg
 输出参数  : 无
 返 回 值  :
 调用函数  :
 被调函数  :

 修改历史  :
      1. 2018年7月18日      创建

*****************************************************************************/
aclError aclrtMallocHostWithCfg(void** hostPtr, uint64_t size, aclrtMallocConfig* cfg)
{
    void* buf = NULL;

    if (!hostPtr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    if (cfg == nullptr || cfg->attrs == nullptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    uint16_t moduleId = 0;
    for (uint32_t i = 0U; i < cfg->numAttrs; i++) {
        aclrtMallocAttribute* attr = &(cfg->attrs[i]);
        if (attr->attr == ACL_RT_MEM_ATTR_MODULE_ID) {
            moduleId = attr->value.moduleId;
            break;
        }
    }
    if (moduleId != HCCL) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    buf = malloc(size);

    if (buf == nullptr) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    *hostPtr = buf;

    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtFreeHost
 功能描述  :
 输入参数  : void* hostPtr
 输出参数  : 无
 返 回 值  :
 调用函数  :
 被调函数  :

 修改历史  :
      1. 2018年7月18日      创建

*****************************************************************************/
aclError aclrtFreeHost(void* hostPtr)
{
    if (!hostPtr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    free(hostPtr);

    return ACL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
/*****************************************************************************
 函 数 名  : rtIpcSetMemoryName
 功能描述  : 给之前申请的设备内存命名，使其能通过名字在进程间共享
 输入参数  : void* ptr
             const char* name
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 新生成函数

*****************************************************************************/
std::mutex ipcMemptrMapLock; // 写内存时，互斥锁
std::mutex ipcMemWhiteListLock;

extern char g_shm_name[64]; // 共享内存名，基于用文件名设定

#define IPC_SHM_MEM_NUM_MAX 3600    // 最大IPC MEM 数量
#define IPC_SHM_NOTIFY_NUM_MAX 3600 // 最大IPC NOTIFY 共享内存数量
#define IPC_SHM_PID_NUM_MAX 48      // 每个Name最大支撑连接进程数

/* IPC memory 共享内存储格式*/
typedef struct IpcShmNode {
    char ipcName[HCCL_IPC_MEM_NAME_LEN];
    int pid[IPC_SHM_PID_NUM_MAX];
} IpcShmNode_S;

/* IPC memory 白名单的存储格式*/
typedef struct IpcShmWriteList {
    IpcShmNode_S memNode[IPC_SHM_MEM_NUM_MAX];
    int ref_cnt; // 引用计数,用防多进程打开时同步
} IpcShmWriteList_S;

/* IPC notify 共享内存储格式*/
typedef struct IpcNotifyNode {
    char ipcName[HCCL_IPC_MEM_NAME_LEN];
    int pid;
} IpcNotifyNode_S;

/* IPC notify 白名单的存储格式*/
typedef struct IpcNotifyWriteList {
    IpcNotifyNode_S notifyNode[IPC_SHM_NOTIFY_NUM_MAX];
    int ref_cnt; // 引用计数,用防多进程打开时同步
} IpcNotifyWriteList_S;

/* 销毁IPC memory 的共享内存，用于失败时销毁*/
rtError_t DestroyIpcMemShm()
{
    // 无法通过此时的ptr来获取name, 一次性全部清除
    std::unique_lock<std::mutex> lock(ipcMemWhiteListLock);

    // 采用共享内存方式存储连接信息
    char ipcMemShmName[100] = {0};
    (void)snprintf_s(ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_MEM_", g_shm_name);
    void* shm_ptr = (void*)sal_share_memory_create(ipcMemShmName, sizeof(IpcShmWriteList_S));

    if (NULL == shm_ptr) {
        HCCL_ERROR("sal_share_memory_create failed, device_memory_share_info is NULL");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else {
        // 该共享内存多次打开，但是只有一个地方释放。因此得先将
        // 引用计数改成1，避免出现内存泄露
        share_mem_t* shm_mem_ptr = (share_mem_t*)((char*)shm_ptr - offsetof(share_mem_t, user_data));
        shm_mem_ptr->ref_cnt = 1;
        // 销毁此共享内存
        memset(shm_ptr, 0, sizeof(IpcShmWriteList_S));
        sal_share_memory_destroy(shm_ptr);
    }
    return RT_ERROR_NONE;
}

u32 g_IpcNameCount = 0;
void rtIpcGetName(char* name, u32 nameLen)
{
    const u32 ipcNameLen = 41;  // ipc 名字长度默认为41位
    const u32 randByteLen = 24; // ipc 随机字符取24个byte

    // 获取24位随机值
    char randLen[25] = {0};
    RAND_bytes(randLen, randByteLen); // 24位随机值
    // 获取8 位tgid
    s32 tgid = SalGetTid();

    // 获取8 位引用计数
    __sync_fetch_and_add(&g_IpcNameCount, 1);

    (void)snprintf_s(name, nameLen, nameLen - 1, "%08x%08x%s", g_IpcNameCount, tgid, randLen);
    return;
}

aclError aclrtIpcMemGetExportKey(void* ptr, size_t byteCount, char* name, size_t nameLen, uint64_t flag)
{
    (void)flag;

    // 创建或打开映射表
    rt_name_map_stub_t* name_map_ptr;
    rtIpcGetName(name, nameLen);

    void* baseptr = rtIpcBasePtrLookup(ptr);
    if (baseptr == nullptr) {
        HCCL_ERROR("rtIpcBasePtrLookup failed ret[%p]", ptr);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    s32 ret = sal_create_name_map(name, &name_map_ptr);

    if (ret != SAL_OK) {
        HCCL_ERROR("sal_open_shm_name_map failed ret[%d]", ret);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    if (IsUseRealPortAndName()) {
        /* 映射后的名字会在name前添加前缀 */
        char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES] = {0};
        (void)snprintf_s(
            mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX,
            name);
        /* 将映射的相关信息放入name_map */
        (void)sal_strncpy(name_map_ptr->mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, mapped_name, SAL_DMEM_UNIQUE_ID_BYTES);

        name_map_ptr->valid_flag = 1;
        return RT_ERROR_NONE;
    }

    /* 在name map中记录共享内存的地址、原名称、映射名称、内存大小，供查找使用 */
    share_mem_t* shm_head = (share_mem_t*)((char*)baseptr - offsetof(share_mem_t, user_data));
    if (shm_head->relate_ptr_cnt >= IPC_SET_NAME_COUNT_MAX) {
        HCCL_ERROR("shm_head->relate_ptr_cnt over max ");
        (void)sal_close_name_map(name_map_ptr);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    shm_head->relate_ptr[shm_head->relate_ptr_cnt] = name_map_ptr; // 本块共享内存的关联指针记录name map的地址
    __sync_fetch_and_add(&(shm_head->relate_ptr_cnt), 1);

    HCCL_DEBUG("set ptr[%p] name_map_ptr[%p] baseptr[%p]", ptr, name_map_ptr, baseptr, shm_head->relate_ptr_cnt);

    /* 映射后的名字会在name前添加前缀 */
    char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES] = {0};
    (void)snprintf_s(
        mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, name);
    /* 将映射的相关信息放入name_map */
    (void)sal_strncpy(name_map_ptr->mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, mapped_name, SAL_DMEM_UNIQUE_ID_BYTES);

    (void)sal_strncpy(
        name_map_ptr->shm_real_name, SAL_DMEM_UNIQUE_ID_BYTES, shm_head->rootInfo, SAL_DMEM_UNIQUE_ID_BYTES);

    name_map_ptr->offset = (char*)ptr - (char*)baseptr;
    name_map_ptr->mem_size = shm_head->mem_size;
    name_map_ptr->valid_flag = 1;

    HCCL_DEBUG(
        "set memory ptr[%p], name[%s] --> namp_name[%s] relate_ptr[%p]", ptr, name_map_ptr->shm_real_name,
        name_map_ptr->mapped_name, shm_head->relate_ptr);

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : rtIpcCloseMemory
 功能描述  : 关闭通过rtIpcOpenMemory打开的共享内存
 输入参数  : void* ptr
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtIpcCloseMemory(const void* ptr)
{
    void* baseptr = rtIpcOpenBasePtrLookup(ptr);
    if (baseptr == nullptr) {
        HCCL_ERROR("rtIpcOpenBasePtrLookup failed ret[%p]", ptr);
        return RT_ERROR_NONE;
    }
    share_mem_t* shm_head = (share_mem_t*)((char*)baseptr - offsetof(share_mem_t, user_data));
    u32 openCnt = __sync_fetch_and_sub(&(shm_head->open_ref_cnt), 1);
    if (openCnt == 1) {
        /* 通过共享内存名称在本进程关闭桩函数共享设备内存 */
        sal_share_memory_destroy((void*)baseptr);

        /* 销毁IPC memory 白名单相关的共享内存*/
        DestroyIpcMemShm();

        rtIpcOpenBasePtrErase(baseptr);
    }

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : rtIpcDestroyMemoryName
 功能描述  : 销毁共享名称
 输入参数  : void* ptr
             const char* name
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 新生成函数

*****************************************************************************/
std::unordered_map<std::string, void*> g_ipcName2Ptr;
std::mutex ipcNameMapLock;
aclError aclrtIpcMemClose(const char* key)
{
    // 暂时不改现有rtIpcSetMemoryName 实现，此处不销毁，在close 时销毁
    if (!key) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    std::unique_lock<std::mutex> lock(ipcNameMapLock);
    auto iter = g_ipcName2Ptr.find(key);
    if (iter == g_ipcName2Ptr.end()) {
        HCCL_ERROR("[aclrtIpcMemClose]key[%s] is not found, g_ipcName2Ptr size[%zu]", key, g_ipcName2Ptr.size());
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    void* ptr = iter->second;
    if (rtIpcCloseMemory(ptr) != RT_ERROR_NONE) {
        HCCL_ERROR("[aclrtIpcMemClose]rtIpcCloseMemory failed, ptr[%p], key[%s]", ptr, key);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    g_ipcName2Ptr.erase(iter);
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : aclrtIpcMemSetImportPid
 功能描述  : 设置IpcMem的白名单
 输入参数  : void* ptr
                           : pid
                           : num
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2019年8月13日
    作    者   : z00382765
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtIpcMemSetImportPid(const char* key, int32_t* pid, size_t num)
{
    if (key == nullptr) {
        HCCL_ERROR("parameter error : key[%p], pid[%p], num[%d]", key, pid, num);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    // 此处有并发，需加锁
    std::unique_lock<std::mutex> lock(ipcMemWhiteListLock);

    int i = 0;
    int j = 0;

    char ipcMemShmName[100] = {0};
    (void)snprintf_s(ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_MEM_", g_shm_name);
    IpcShmWriteList_S* shm_buf = (IpcShmWriteList_S*)sal_share_memory_create(ipcMemShmName, sizeof(IpcShmWriteList_S));

    /*采用引用计数对进程间互斥*/
    while (!__sync_bool_compare_and_swap(&(shm_buf->ref_cnt), 0, 1))
        ;

    char ipcMemCheck[HCCL_IPC_MEM_NAME_LEN] = {0};

    for (i = 0; i < IPC_SHM_MEM_NUM_MAX; i++) {
        // 判断当前数组是否为空
        if (!strcmp(shm_buf->memNode[i].ipcName, ipcMemCheck)) {
            // 当前未存储Name  与PID 映射关系, 则保存当前的PID
            sal_strncpy(shm_buf->memNode[i].ipcName, HCCL_IPC_MEM_NAME_LEN, key, HCCL_IPC_MEM_NAME_LEN);
            for (j = 0; j < IPC_SHM_PID_NUM_MAX; j++) {
                if (shm_buf->memNode[i].pid[j] == 0) {
                    shm_buf->memNode[i].pid[j] = pid[0];
                    break;
                }
            }
            if (j == IPC_SHM_PID_NUM_MAX) {
                HCCL_ERROR("IPC_SHM_PID_NUM_MAX0 size is not enough");
                DestroyIpcMemShm();
            }
            break;
        } else if (!strcmp(shm_buf->memNode[i].ipcName, key)) {
            for (j = 0; j < IPC_SHM_PID_NUM_MAX; j++) {
                if (shm_buf->memNode[i].pid[j] == pid[0]) {
                    break;
                } else if (shm_buf->memNode[i].pid[j] == 0) {
                    shm_buf->memNode[i].pid[j] = pid[0];
                    break;
                }
            }
            if (j == IPC_SHM_PID_NUM_MAX) {
                HCCL_ERROR("IPC_SHM_PID_NUM_MAX1 size is not enough");
                DestroyIpcMemShm();
            }
            break;
        }
    }
    /*释放进程间互斥锁*/
    while (!__sync_bool_compare_and_swap(&(shm_buf->ref_cnt), 1, 0))
        ;

    if (i == IPC_SHM_MEM_NUM_MAX) {
        HCCL_ERROR("IPC_SHM_PID_NUM_MAX size is not enough");
        DestroyIpcMemShm();
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    HCCL_DEBUG("aclrtIpcMemSetImportPid key[%s] ,pid[%d]", key, pid[0]);
    return ACL_SUCCESS;
}

aclError aclrtIpcMemImportPidInterServer(const char* name, aclrtServerPid* serverPids, size_t num)
{
    const aclrtServerPid& rtServerPid = *serverPids;
    return aclrtIpcMemSetImportPid(name, rtServerPid.pid, rtServerPid.num);
}

/*****************************************************************************
 函 数 名  : aclrtIpcMemImportByKey
 功能描述  : 打开其他进程或者线程中命名的设备内存
 输入参数  : void** ptr
             char* name
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年6月26日
    作    者   : liubanglan
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtIpcMemImportByKey(void** ptr, const char* name, uint64_t flag)
{
    (void)flag;

    /* 较验该指针是否在白名单内*/
    char ipcMemShmName[100] = {0};
    (void)snprintf_s(ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_MEM_", g_shm_name);
    IpcShmWriteList_S* shm_buf = (IpcShmWriteList_S*)sal_share_memory_create(ipcMemShmName, sizeof(IpcShmWriteList_S));
    int i = 0;
    int j = 0;
    for (i = 0; i < IPC_SHM_MEM_NUM_MAX; i++) {
        if (!strcmp(shm_buf->memNode[i].ipcName, name)) {
            s32 pid = 0;
            SalGetBareTgid(&pid); // 当前进程id
            for (j = 0; j < IPC_SHM_PID_NUM_MAX; j++) {
                if (shm_buf->memNode[i].pid[j] == pid) {
                    break;
                }
            }
            if (j == IPC_SHM_PID_NUM_MAX) {
                HCCL_ERROR("aclrtIpcMemImportByKey error , can't find pid[%d]", pid);
                DestroyIpcMemShm();
                return ACL_ERROR_RT_PARAM_INVALID;
            }

            break;
        }
    }

    if (i == IPC_SHM_MEM_NUM_MAX) {
        HCCL_ERROR("aclrtIpcMemImportByKey error , can't find name[%s]", name);
        DestroyIpcMemShm();
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    // 创建或打开映射表
    rt_name_map_stub_t* name_map_ptr;
    s32 ret = sal_create_name_map(name, &name_map_ptr);

    if (ret != SAL_OK) {
        HCCL_ERROR("sal_open_shm_name_map failed ret[%d]", ret);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    /* 如果此时打开的name_map内没有内容，说明没有执行过rtIpcSetMemoryName*/
    if (0 == name_map_ptr->valid_flag) {
        HCCL_ERROR("name[%s] not set, open failed", name);
        (void)sal_close_name_map(name_map_ptr);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    /* 映射后的名字会在name前添加前缀 */
    char mapped_name[SAL_DMEM_UNIQUE_ID_BYTES] = {0};
    (void)snprintf_s(
        mapped_name, SAL_DMEM_UNIQUE_ID_BYTES, SAL_DMEM_UNIQUE_ID_BYTES - 1, "%s%s", SAL_DMEM_UNIQUE_ID_PREFIX, name);

    /* 比较映射后的名字和name_map中的名字是否能匹配 */
    if (0 == strcmp(mapped_name, name_map_ptr->mapped_name)) {
        HCCL_DEBUG(
            "open memory name[%s] match name_map -> shm_real_name[%s], mem_size(%d), realport[%d]", name,
            name_map_ptr->shm_real_name, name_map_ptr->mem_size, IsUseRealPortAndName());

        /* 通过原名称打开这段被共享的内存，并返回地址 */
        *ptr = sal_share_memory_create(
            IsUseRealPortAndName() ? name : name_map_ptr->shm_real_name, name_map_ptr->mem_size);

        if (NULL == *ptr) {
            HCCL_ERROR("create share memory %s failed", name_map_ptr->shm_real_name);
            return ACL_ERROR_RT_MEMORY_ALLOCATION;
        }

        rtIpcOpenBasePtrAdd(*ptr, name_map_ptr->mem_size);

        share_mem_t* shm_head = (share_mem_t*)((char*)(*ptr) - offsetof(share_mem_t, user_data));

        __sync_fetch_and_add(&(shm_head->open_ref_cnt), 1);

        *ptr += name_map_ptr->offset;
        /* 销毁本操作打开的name map */
        (void)sal_close_name_map(name_map_ptr);

        return RT_ERROR_NONE;
    } else {
        HCCL_ERROR("name[%s] was not match name_map[%s]", name, name_map_ptr->mapped_name);

        /* 销毁本操作打开的name map */
        (void)sal_close_name_map(name_map_ptr);
        // return RT_ERROR_INVALID_RESOURCE_HANDLE;
        return ACL_ERROR_RT_CONTEXT_NULL;
    }
}

/*****************************************************************************
 函 数 名  : 获取页表大小
 功能描述  : aclrtPointerGetAttributes 获取指针属性
 输入参数  : void* ptr
 输出参数  : 无
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年6月26日
    作    者   : l00382765
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtPointerGetAttributes(const void* ptr, aclrtPtrAttributes* attributes)
{
    // 桩函数固定反回2M的页表大小
    attributes->pageSize = 1;

    return ACL_SUCCESS;
}

aclError aclrtHostRegister(void* ptr, uint64_t size, aclrtHostRegisterType type, void** devPtr)
{
    *devPtr = ptr;
    return ACL_SUCCESS;
}

aclError aclrtHostUnregister(void* ptr) { return ACL_SUCCESS; }

aclError aclrtGetOpTimeOutInterval(uint64_t* interval) { return ACL_SUCCESS; }

rtError_t rtMemPrefetchToDevice(void* ptr, uint64_t size, int32_t advise) { return RT_ERROR_NONE; }

#ifdef __cplusplus
} // extern "C"
#endif

bool Adx::AdumpIsDumpEnable(Adx::DumpType type)
{
    if (type == Adx::DumpType::OP_OVERFLOW) {
        return false;
    } else {
        return true;
    }
}

void Adx::AdumpPrintWorkSpace(
    const void* workSpaceAddr, const size_t dumpWorkSpaceSize, rtStream_t stream, const char* opType, bool enableSync)
{
    return;
}

/*****************************************************************************
 函 数 名  : rtMemAllocManaged
 功能描述  : alloc managed memory
 输入参数  : void **ptr
             uint64_t size
             uint32_t flag
 输出参数  : void **ptr
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年06月26日
    作    者   :
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtMemAllocManaged(void** ptr, uint64_t size, uint32_t flag)
{
    void* buf = NULL;

    if (!ptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    buf = malloc(size);

    if (NULL == buf) {
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    *ptr = buf;

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : rtMemFreeManaged
 功能描述  : 设备内存释放
 输入参数  : void **ptr
 输出参数  : 无
 返 回 值  : rtError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年06月26日
    作    者   :
    修改内容   : 新生成函数

*****************************************************************************/
rtError_t rtMemFreeManaged(void* ptr)
{
    if (!ptr) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    free(ptr);

    return RT_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : set_chip_peer_stub
 功能描述  : 设置芯片peer类型
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : -
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/
static bool chip_peer_stub[16][16] = {0}; /*最大为16，下面不再做判断*/
void set_chip_peer_stub(s32 devId1, s32 devId2, bool can_peer)
{
    if ((devId1 >= 0) && (devId2 >= 0) && (devId1 < 16) && (devId2 < 16)) {
        chip_peer_stub[devId1][devId2] = can_peer;
    } else {
        HCCL_ERROR("device id is illegal");
    }
}
/*****************************************************************************
 函 数 名  : drvMemCanPeer
 功能描述  : 获取设备的PCIE编号
 输入参数  : DVdevice device1
             DVdevice device2
 输出参数  : int* canPeer
 返 回 值  : DVresult
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年07月8日
    作    者   : mali
    修改内容   : 新生成函数

*****************************************************************************/
DVresult drvMemCanPeer(int* canPeer, DVdevice device1, DVdevice device2)
{
    if (NULL == canPeer) {
        HCCL_ERROR("parameter: canPeer is NULL");
        return DRV_ERROR_INVALID_VALUE;
    }

    // 暂时返回0
    if ((device1 >= 0) && (device2 >= 0) && (device1 < 16) && (device2 < 16)) {
        *canPeer = chip_peer_stub[device1][device2];
    } else {
        HCCL_ERROR("device id is illegal");
        return DRV_ERROR_INVALID_VALUE;
    }
    return DRV_ERROR_NONE;
}

#if 0
pid_t drvDeviceGetBarePid(void)
{
    return getpid();
}
#endif
bool g_isCommonPidMode = false;

void rtSetCommonPidMode(bool state) { g_isCommonPidMode = state; }

aclError aclrtDeviceGetBareTgid(s32* pid)
{
    if (g_isCommonPidMode == false) {
        *pid = syscall(SYS_gettid);
    } else {
        *pid = getpid();
    }

    return RT_ERROR_NONE;
}

rtError_t rtDeviceGetBareTgid(u32* pid)
{
    if (g_isCommonPidMode == false) {
        *pid = syscall(SYS_gettid);
    } else {
        *pid = getpid();
    }

    return RT_ERROR_NONE;
}

aclError aclrtReserveMemAddress(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    return ACL_SUCCESS;
    ;
}

aclError aclrtReserveMemAddressNoUCMemory(void** virPtr, size_t size, size_t alignment, void* expectPtr, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)alignment;
    (void)expectPtr;
    (void)flags;
    return ACL_SUCCESS;
    ;
}

aclError aclrtReleaseMemAddress(void* virPtr)
{
    (void)virPtr;
    return ACL_SUCCESS;
}

aclError aclrtFreePhysical(aclrtDrvMemHandle handle)
{
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMallocPhysical(aclrtDrvMemHandle* handle, size_t size, const aclrtPhysicalMemProp* prop, uint64_t flags)
{
    (void)handle;
    (void)size;
    (void)prop;
    (void)flags;
    return ACL_SUCCESS;
}

aclError
aclrtMemGetAllocationGranularity(aclrtPhysicalMemProp* prop, aclrtMemGranularityOptions option, size_t* granularity)
{
    (void)prop;
    (void)option;
    if (granularity != nullptr) {
        *granularity = 2097152;
    }
    return ACL_SUCCESS;
}

aclError aclrtMemRetainAllocationHandle(void* virPtr, aclrtDrvMemHandle* handle)
{
    (void)virPtr;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMapMem(void* virPtr, size_t size, size_t offset, aclrtDrvMemHandle handle, uint64_t flags)
{
    (void)virPtr;
    (void)size;
    (void)offset;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtUnmapMem(void* devPtr)
{
    (void)devPtr;
    return ACL_SUCCESS;
}

aclError aclrtMemExportToShareableHandle(
    aclrtDrvMemHandle handle, aclrtMemHandleType handleType, uint64_t flags, uint64_t* shareableHandle)
{
    (void)handle;
    (void)handleType;
    (void)flags;
    (void)shareableHandle;
    return ACL_SUCCESS;
}

aclError aclrtMemImportFromShareableHandle(uint64_t shareableHandle, int32_t deviceId, aclrtDrvMemHandle* handle)
{
    (void)shareableHandle;
    (void)deviceId;
    (void)handle;
    return ACL_SUCCESS;
}

aclError aclrtMemSetPidToShareableHandle(uint64_t shareableHandle, int32_t* pid, size_t pidNum)
{
    (void)shareableHandle;
    (void)pid;
    (void)pidNum;
    return ACL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : drvRegisterExitHandler
 功能描述  : 注册异常退出处理函数
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442456
    修改内容   : 新生成函数

*****************************************************************************/
drvError_t drvRegisterExitHandler(void (*handler)(int signum))
{
    /* 进程异常退出时的处理回调函数,接管系统信号*/
    struct sigaction sa_exit;     /*包含信号处理动作的结构体*/
    sa_exit.sa_handler = handler; /*指定信号处理函数*/
    sigemptyset(&sa_exit.sa_mask);
    sigaction(SIGINT, &sa_exit, NULL);  /* 注册SIGINT信号 */
    sigaction(SIGTERM, &sa_exit, NULL); /* 注册SIGINT信号, 对应 普通kill */
    return DRV_ERROR_NONE;
}

int setDevPhyId(uint32_t devIndex)
{
    gDevPhyId = devIndex;
    return DRV_ERROR_NONE;
}

aclError aclrtGetLogicDevIdByUserDevId(int32_t userDevId, int32_t* const logicDevId)
{
    *logicDevId = userDevId;
    return ACL_SUCCESS;
}

aclError aclrtGetPhyDevIdByLogicDevId(int32_t logicDevId, int32_t* const phyDevId)
{
    if (gBoardId == 0x2000) {
        *phyDevId = logicDevId * 2;
        return ACL_SUCCESS;
    }

    *phyDevId = logicDevId;
    if (gDevPhyId) {
        *phyDevId = static_cast<int32_t>(gDevPhyId);
    }
    return ACL_SUCCESS;
}

aclError aclrtGetPhyDevIdByUserDevId(int32_t userDevId, int32_t* const phyDevId)
{
    return aclrtGetPhyDevIdByLogicDevId(userDevId, phyDevId);
}

aclError aclrtGetUserDevIdByPhyDevId(const int32_t phyDevId, int32_t* const userDevId)
{
    if (gBoardId == 0x2000) {
        *userDevId = phyDevId / 2;
        return ACL_SUCCESS;
    }
    *userDevId = phyDevId;
    if (gDevPhyId) {
        *userDevId = static_cast<int32_t>(gDevPhyId);
    }
    return ACL_SUCCESS;
}

rtError_t rtsGetLogicDevIdByPhyDevId(int32_t phyDevId, int32_t* const logicDevId)
{
    if (gBoardId == 0x2000) {
        *logicDevId = phyDevId / 2;
        return ACL_SUCCESS;
    }
    *logicDevId = phyDevId;
    if (gDevPhyId) {
        *logicDevId = static_cast<int32_t>(gDevPhyId);
    }
    return ACL_SUCCESS;
}

rtError_t rtGetSocVersion(char* chipVer, u32 maxLen)
{
    if (chipVer == NULL) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    sal_memcpy(chipVer, sizeof("Ascend910"), "Ascend910", sizeof("Ascend910"));

    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {
        sal_memcpy(chipVer, sizeof("Ascend910B1"), "Ascend910B1", sizeof("Ascend910B1"));
    } else if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
        sal_memcpy(chipVer, sizeof("Ascend910_9391"), "Ascend910_9391", sizeof("Ascend910_9391"));
    } else if (gBoardId == 0x0000) {
        sal_memcpy(chipVer, sizeof("Ascend910"), "Ascend910", sizeof("Ascend910"));
    } else if (gBoardId == 0x2000) { // 临时定义的 board id
        sal_memcpy(chipVer, sizeof("Ascend310P3"), "Ascend310P3", sizeof("Ascend310P3"));
    }
    return RT_ERROR_NONE;
}

int dsmi_get_device_ip_address(
    int device_id, int port_type, int port_id, ip_addr_t* ip_address, ip_addr_t* mask_address)
{
    *(u32*)(ip_address->u_addr.ip4) = 0x400007f;
    return DRV_ERROR_NONE;
}

int set_board_id(unsigned int board_id)
{
    gBoardId = board_id;
    return DRV_ERROR_NONE;
}
int set_VM(unsigned int VMModel)
{
    gIsVM = VMModel;
    return DRV_ERROR_NONE;
}
int dsmi_get_board_id(int device_id, unsigned int* board_id)
{
    if (board_id == nullptr) {
        return DRV_ERROR_INVALID_VALUE;
    }
    *board_id = 0;
    if (gBoardId) {
        *board_id = gBoardId;
    }
    return DRV_ERROR_NONE;
}

/*****************************************************************************
 函 数 名  : set_chip_type_stub
 功能描述  : 设置芯片类型
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/

void set_chip_type_stub(s32 devId, s32 chipType)
{
    if ((devId >= 0) && (chipType < static_cast<s32>(DevType::DEV_TYPE_COUNT)) && (devId < 256)) {
        chip_type_stub[devId] = chipType;
    } else {
        HCCL_ERROR("device id is illegal");
    }
}
/*****************************************************************************
 函 数 名  : set_devphyid_to_sdId_stub
 功能描述  : 将devicePhyId赋值给sdid
 输入参数  :  rankList
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :
 被调函数  :

*****************************************************************************/
void set_devphyid_to_sdId_stub(std::vector<hccl::RankInfo_t>& rankList)
{
    for (auto it = 0; it < rankList.size(); ++it) {
        rankList[it].superDeviceId = rankList[it].deviceInfo.devicePhyId;
    }
    return;
}

HcclResult
stub_hrtRaGetSingleSocketVnicIpInfo(u32 phy_id, DeviceIdType deviceIdType, u32 deviceId, hccl::HcclIpAddress& vnicIP)
{
    HcclIpAddress addr(deviceId);
    vnicIP = addr;
    return HCCL_SUCCESS;
}

/*****************************************************************************
 函 数 名  : dsmi_get_chip_info
 功能描述  : 获取芯片信息
 输入参数  :  handler
 输出参数  : 无
 返 回 值  : drvError_t
 调用函数  :git log
 被调函数  :

 修改历史      :
  1.日    期   : 2018年8月7日
    作    者   : l00442453
    修改内容   : 新生成函数

*****************************************************************************/
namespace cce {
/*****************************************************************************
 函 数 名  : ccVectorReduce
 功能描述  : 规约操作
 输入参数  : const void *src1
             const void *src2
             uint32_t count
             const ccDataType_t datatype
             const HcclReduceOp op
             rtStream_t stream
             void *dst
 输出参数  : 无
 返 回 值  : ccStatus_t
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2017年12月7日
    作    者   : p00335137
    修改内容   : 新生成函数

*****************************************************************************/
ccStatus_t ccVectorReduce(
    const void* src1, const void* src2, uint64_t count, const ccDataType_t datatype, const ccReduceOp_t op,
    rtStream_t stream, const void* dst)
{
    if (!src1 || !src2 || !dst) {
        return cce::CC_STATUS_RESERVED;
    }

    if (datatype >= cce::CC_DATA_RESERVED) {
        return cce::CC_STATUS_RESERVED;
    }

    if (op >= CCE_RED_OP_RESERVED) {
        return cce::CC_STATUS_RESERVED;
    }

    // Mod for optimize runtime Stub by l on 2018-01-11 Below
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;

    // 将vector reduce任务，压入任务队列
    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_REDUCE;
    stream_task.stream_para.reducestruct.src1 = (void*)src1;
    stream_task.stream_para.reducestruct.src2 = (void*)src2;
    stream_task.stream_para.reducestruct.count_reduce = count;
    stream_task.stream_para.reducestruct.datatype = datatype;
    stream_task.stream_para.reducestruct.op = op;
    stream_task.stream_para.reducestruct.dst_reduce = (void*)dst;

    rtstream->push_task(&stream_task);
    // Mod for optimize runtime Stub by l on 2018-01-11 Above

    return CC_STATUS_SUCCESS;
}

/*****************************************************************************
     函 数 名  : cceSysInit
     功能描述  : cce init
     输入参数  : 无
     输出参数  : 无
     返 回 值  : void
     调用函数  :
     被调函数  :

     修改历史      :
      1.日    期   : 2019年05月16日
        作    者   : w00500539
        修改内容   : 新生成函数

    *****************************************************************************/
void cceSysInit() { return; }

} // namespace cce

#if 1 // Cloud迭代1新增桩函数
/*****************************************************************************
 函 数 名  : aclrtReduceAsync
 功能描述  : 跨V80的inline reduce操作, 仅V80支持
 输入参数  : void *dst  :本地数据输入地址
             void *src  :远端数据输入地址
             u64 count  :数据字节数
             aclrtReduceKind kind    :规约类型
             aclDataType type      :数据类型
 输出参数  : void *dst  :结果数据输出地址
 返 回 值  : aclError
 调用函数  :
 被调函数  :

 修改历史      :
  1.日    期   : 2018年9月5日
    作    者   : mali
    修改内容   : 新生成函数

*****************************************************************************/
aclError aclrtReduceAsync(
    void* dst, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type, aclrtStream stream,
    void* reserve)
{
    cce::ccStatus_t cce_status;
    cce::ccReduceOp_t op;
    s32 data_unit_size = 0;

    switch (kind) {
        case ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM:
            op = cce::CCE_RED_OP_SUM;
            break;

        default: /*inline reduce 目前只支持sum*/
        {
            HCCL_ERROR("Not support the vector reduce red_op[%d].", kind);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    cce::ccDataType_t ccDataType;
    switch (type) {
        case ACL_FLOAT:
            data_unit_size = sizeof(float);
            ccDataType = cce::CC_DATA_FLOAT;
            break;
        case ACL_FLOAT16:
            data_unit_size = 2; // sizeof(fp16)
            ccDataType = cce::CC_DATA_HALF;
            break;
        case ACL_INT16:
            ccDataType = cce::CC_DATA_INT16;
            data_unit_size = 2; // sizeof(int16)
            break;
        case ACL_INT32:
            ccDataType = cce::CC_DATA_INT32;
            data_unit_size = 4; // sizeof(int16)
            break;
        case ACL_INT8:
            if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)
                || chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
                ccDataType = cce::CC_DATA_INT8;
                data_unit_size = 1; // sizeof(int8)
                break;
            }
        default: {
            HCCL_ERROR("Not support the vector reduce type[%d].", type);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    uint32_t data_cnt = count / data_unit_size;

    cce_status = cce::ccVectorReduce(src, dst, data_cnt, ccDataType, op, stream, dst);

    if (cce::CC_STATUS_SUCCESS != cce_status) {
        HCCL_ERROR("cce::ccVectorReduce run error, ret = %d", cce_status);
        return ACL_ERROR_RT_DRV_INTERNAL_ERROR;
    }

    return ACL_SUCCESS;
}

/**
 * @ingroup rt_stars
 * @brief general ctrl if
 * @param [in] ctl              ctl input
 * @param [in] num              ctl input num
 * @param [in] type             ctl type
 * @return RT_ERROR_NONE for ok, others failed
 */
rtError_t rtGeneralCtrl(uintptr_t* ctrl, uint32_t num, uint32_t type) { return RT_ERROR_NONE; }

rtError_t rtMemHostRegister(void* ptr, u64 size, uint32_t flags)
{
    // Nothing to do

    return RT_ERROR_NONE;
}

aclError aclrtMemset(void* devPtr, size_t maxCount, int32_t value, size_t count)
{
    if (maxCount == 321) {
        return -1;
    }
    return ACL_SUCCESS;
}

rtError_t rtNotifyGetPhyInfo(rtNotify_t notify, uint32_t* phyDevId, uint32_t* tsId)
{
    *phyDevId = 1;
    *tsId = 3;
    return RT_ERROR_NONE;
}

rtError_t rtNotifyGetPhyInfoExt(rtNotify_t notify, rtNotifyPhyInfo* notifyInfo)
{
    notifyInfo->phyId = 1;
    notifyInfo->tsId = 3;
    notifyInfo->flag = 0;
    return RT_ERROR_NONE;
}

rtError_t rtAicpuKernelLaunchExWithArgs(
    uint32_t kernelType, const char* opName, uint32_t numBlocks, const rtAicpuArgsEx_t* argsInfo, rtSmDesc_t* smDesc,
    rtStream_t stream, uint32_t flags)
{
    return RT_ERROR_NONE;
}

rtError_t rtStreamGetSqid(const rtStream_t stream, uint32_t* sqId)
{
    if (stream == NULL || sqId == NULL) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    *sqId = 122;
    return RT_ERROR_NONE;
}

std::map<string, std::set<int>> ipcNotifyWhiteList;
std::mutex ipcNotifyWhiteListLock;

/* 销毁IPC notify 白名单相关 的共享内存*/
rtError_t DestroyIpcNotifyShm()
{
    char ipcMemShmName[100] = {0};
    (void)snprintf_s(
        ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_NOTIFY_", g_shm_name);
    void* shm_ptr = (void*)sal_share_memory_create(ipcMemShmName, sizeof(IpcNotifyWriteList_S));

    if (NULL == shm_ptr) {
        HCCL_ERROR("sal_share_memory_create failed, device_memory_share_info is NULL");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    } else {
        // 该共享内存多次打开，但是只有一个地方释放。因此得先将
        // 引用计数改成1，避免出现内存泄露
        share_mem_t* shm_mem_ptr = (share_mem_t*)((char*)shm_ptr - offsetof(share_mem_t, user_data));
        shm_mem_ptr->ref_cnt = 1;
        // 销毁此共享内存
        memset(shm_ptr, 0, sizeof(IpcNotifyWriteList_S));
        sal_share_memory_destroy(shm_ptr);
    }
    return RT_ERROR_NONE;
}

class event_deque {
public:
    explicit event_deque() {}
    virtual ~event_deque()
    {
        for (auto index : eventQueue) {
            if (index != nullptr) {
                aclrtDestroyEvent(index);
            }
        }
    }
    std::deque<rtNotify_t> eventQueue;
};
event_deque g_eventQueue;
std::mutex g_eventQueueMutex;
aclError aclrtCreateNotify(aclrtNotify* notify, uint64_t flag)
{
    int32_t device;
    aclError rtRet = aclrtGetDevice(&device);
    if (device < 0 || notify == nullptr) {
        HCCL_ERROR("parameter invalid : dev_id[%d], notify[%p]", device, notify);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (rtRet != ACL_SUCCESS) {
        HCCL_ERROR("Get device failed[%d]", rtRet);
        return rtRet;
    }

    char ipc_notify_shm_name[NOTIFY_SHM_NAME_LEN] = {0};
    s32 ret = snprintf_s(
        ipc_notify_shm_name, NOTIFY_SHM_NAME_LEN, NOTIFY_SHM_NAME_LEN - 1, "%s-%d-%d", "hccl-notify-stub", getpid(),
        device);
    if (ret == -1) {
        HCCL_ERROR("snprintf_s failed[%d]", ret);
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    rt_notify_t* ipc_notify = (rt_notify_t*)malloc(sizeof(rt_notify_t));
    if (ipc_notify == nullptr) {
        HCCL_ERROR("ipc_notify allocate failed");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    memset(ipc_notify, 0, sizeof(rt_notify_t));
    rt_shm_notify_t* notify_shm =

        (rt_shm_notify_t*)sal_share_memory_create(ipc_notify_shm_name, sizeof(rt_shm_notify_t));
    if (notify_shm == nullptr) {
        HCCL_ERROR("notify_shm allocate failed");

        /*notify结构赋值*/
        free(ipc_notify);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    u64 cnt = __sync_fetch_and_add(&(notify_shm->ref_cnt), 1);
    if (cnt == 0) {
        memcpy(notify_shm->ipc_notify_shm_name, ipc_notify_shm_name, NOTIFY_SHM_NAME_LEN);
        notify_shm->name_flag = 1;
        notify_shm->device_id = device;
    } else {
        while (notify_shm->name_flag == 0) {
            HCCL_ERROR("waiting name flag set...");
            SaluSleep(1000);
        }
    }

    u64 notify_id = 0;
    for (; notify_id < NOTIFY_MAX; notify_id++) {
        if (__sync_bool_compare_and_swap(&(notify_shm->occupied_flag[notify_id]), 0, 1)) {
            break;
        }
    }
    if (notify_id >= NOTIFY_MAX) {
        HCCL_ERROR("no free notify_id");
        sal_share_memory_destroy(notify_shm);
        free(ipc_notify);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    notify_shm->device_id = device;
    notify_shm->record_cnt[notify_id] = 0;
    memcpy(notify_shm->ipc_notify_shm_name, ipc_notify_shm_name, NOTIFY_SHM_NAME_LEN);

    ipc_notify->notify_id = notify_id;
    ipc_notify->ipc_notify_shm = (rt_shm_notify_t*)notify_shm;
    HCCL_DEBUG("notify_id[%llu], device[%d]", notify_id, device);
    *notify = ipc_notify;
    HCCL_DEBUG("aclrtCreateNotify: create notify[%p]", ipc_notify);
    std::unique_lock<std::mutex> lock(g_eventQueueMutex);
    g_eventQueue.eventQueue.push_back(*notify);
    return ACL_SUCCESS;
}

aclError aclrtDestroyNotify(aclrtNotify notify)
{
    if (nullptr == notify) {
        HCCL_WARNING("notify is null");
        return RT_ERROR_NONE;
    }
    HCCL_INFO("aclrtDestroyNotify: destroy notify[%p]", notify);

    std::unique_lock<std::mutex> lock(g_eventQueueMutex);
    auto iter = std::find(g_eventQueue.eventQueue.begin(), g_eventQueue.eventQueue.end(), notify);

    rt_notify_t* ipc_notify = (rt_notify_t*)notify;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;

    if (ipc_notify->notify_id >= NOTIFY_MAX) {
        HCCL_ERROR("notify id[%llu] is overflow", ipc_notify->notify_id);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (notify_shm->record_cnt[ipc_notify->notify_id] != 0) {
        HCCL_WARNING(
            "wait for record before destroy,notify_id[%llu], device[%d]", ipc_notify->notify_id, notify_shm->device_id);
        SaluSleep(100000);
    }
    __sync_bool_compare_and_swap(&(notify_shm->occupied_flag[ipc_notify->notify_id]), 1, 0);

    if (nullptr != ipc_notify->ipc_name_shm) {
        __sync_fetch_and_sub(&(notify_shm->ref_cnt), 1);
        sal_share_memory_destroy(ipc_notify->ipc_name_shm);
    }
    if (nullptr != ipc_notify->ipc_notify_shm) {
        sal_share_memory_destroy(ipc_notify->ipc_notify_shm);
    }
    free(ipc_notify);

    // 销毁IPC notify 的共享内存
    DestroyIpcNotifyShm();

    if (iter != g_eventQueue.eventQueue.end()) {
        *iter = nullptr;
    }
    return RT_ERROR_NONE;
}

aclError aclrtRecordNotify(aclrtNotify notify, aclrtStream stream)
{
    if (nullptr == notify || nullptr == stream) {
        HCCL_ERROR("parameter error : notify[%p], stream[%p]", notify, stream);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    // 压入任务队列
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_NOTIFY_RECORD;
    stream_task.stream_para.notify = (rt_notify_t*)notify;
    rtstream->push_task(&stream_task);

    return ACL_SUCCESS;
}

aclError aclrtWaitAndResetNotify(aclrtNotify notify, aclrtStream stream, uint32_t timeout)
{
    if (nullptr == notify || nullptr == stream || !timeout) {
        HCCL_ERROR("parameter error : notify[%p], stream[%p], timeOut[%u ms]", notify, stream, timeout);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    // 压入任务队列
    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_NOTIFY_WAIT;
    stream_task.stream_para.notify = (rt_notify_t*)notify;
    rtstream->push_task(&stream_task);

    return ACL_SUCCESS;
}
#define RT_INFO_TYPE_PHY_CHIP_ID 18
#define INFO_TYPE_SDID 26
#define INFO_TYPE_SERVER_ID 27
#define INFO_TYPE_SUPPER_POD_ID 29

aclError aclrtGetDeviceInfo(uint32_t deviceId, aclrtDevAttr attr, int64_t* value)
{
    if (attr == ACL_DEV_ATTR_PHY_CHIP_ID) {
        *value = deviceId;
    } else if (attr == ACL_DEV_ATTR_AICORE_CORE_NUM || attr == ACL_DEV_ATTR_VECTOR_CORE_NUM) {
        *value = 32;
    } else if (
        attr == ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID || attr == ACL_DEV_ATTR_SUPER_POD_SERVER_ID
        || attr == ACL_DEV_ATTR_SUPER_POD_ID) {
        *value = 1;
    } else if (attr = ACL_DEV_ATTR_SMP_ID) {
        *value = 0;
    } else {
        return 1;
    }
    return ACL_SUCCESS;
}

rtError_t rtGetNotifyAddress(rtNotify_t notify, uint64_t* const notifyAddres)
{
    if (notify == nullptr) {
        HCCL_ERROR("Notify pointer is nullptr, please check your test case!");
        return RT_ERROR_NONE;
    }
    if (notifyAddres == nullptr) {
        HCCL_ERROR("notifyAddres pointer is nullptr, please check your test case!");
        return RT_ERROR_NONE;
    }
    rt_notify_t* inner_notify = (rt_notify_t*)notify;
    *notifyAddres = (inner_notify->notify_id) << 1;
    return RT_ERROR_NONE;
}

aclError aclrtGetNotifyId(aclrtNotify notify, uint32_t* notifyId)
{
    rt_notify_t* inner_notify = (rt_notify_t*)notify;
    if (inner_notify == nullptr) {
        HCCL_ERROR("inner_notify pointer is nullptr, please check your test case!");
        return ACL_SUCCESS;
    }
    *notifyId = inner_notify->notify_id;
    return ACL_SUCCESS;
    if (inner_notify->ipc_name_shm == nullptr) {
        HCCL_ERROR("ipc_name_shm pointer is nullptr, please check your test case!");
        return ACL_SUCCESS;
    }

    *notifyId = inner_notify->ipc_name_shm->notify_id;
    return ACL_SUCCESS;
}

aclError aclrtNotifyGetExportKey(aclrtNotify notify, char* name, size_t len, uint64_t flag)
{
    (void)flag;
    if (nullptr == notify || nullptr == name) {
        HCCL_ERROR("parameter error : notify[%p], name[%p]", notify, name);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    (void)rtIpcGetName(name, len);

    rt_shm_ipc_name_t* ipc_name_shm = (rt_shm_ipc_name_t*)sal_share_memory_create(name, sizeof(rt_shm_ipc_name_t));
    if (ipc_name_shm == nullptr) {
        HCCL_ERROR("ipc_name_shm allocate failed");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    rt_notify_t* ipc_notify = (rt_notify_t*)notify;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;

    /*记录用户name创建的共享内存区地址*/
    memcpy(ipc_name_shm->ipc_notify_shm_name, notify_shm->ipc_notify_shm_name, NOTIFY_SHM_NAME_LEN);
    ipc_name_shm->notify_id = ipc_notify->notify_id;

    // 记录用户name创建的共享内存区地址到notify_shm

    /*把rtNotifyCreate创建的notify信息保存到用户传入的name所在的共享内存区*/
    ipc_notify->ipc_name_shm = ipc_name_shm;

    return ACL_SUCCESS;
}

aclError aclrtNotifySetImportPid(aclrtNotify notify, int32_t* pid, size_t num)
{
    rt_notify_t* ipcNotify = static_cast<rt_notify_t*>(notify);
    const char* name = ipcNotify->ipc_notify_shm->ipc_notify_shm_name;
    if (name == NULL) {
        HCCL_ERROR("parameter error : name[%p], pid[%p], num[%d]", name, pid, num);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    // 此处有并发，需加锁
    std::unique_lock<std::mutex> lock(ipcNotifyWhiteListLock);

    int i = 0;
    int j = 0;
    char ipcMemShmName[100] = {0};
    (void)snprintf_s(
        ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_NOTIFY_", g_shm_name);
    IpcNotifyWriteList_S* shm_buf
        = (IpcNotifyWriteList_S*)sal_share_memory_create(ipcMemShmName, sizeof(IpcNotifyWriteList_S));

    /*采用引用计数对进程间互斥*/
    while (!__sync_bool_compare_and_swap(&(shm_buf->ref_cnt), 0, 1))
        ;

    char ipcNotifyCheck[HCCL_IPC_MEM_NAME_LEN] = {0};
    for (i = 0; i < IPC_SHM_NOTIFY_NUM_MAX; i++) {
        // 判断当前数组是否为空
        if (!strcmp(shm_buf->notifyNode[i].ipcName, ipcNotifyCheck)) {
            // 当前未存储Name  与PID 映射关系, 则保存当前的PID
            sal_strncpy(shm_buf->notifyNode[i].ipcName, HCCL_IPC_MEM_NAME_LEN, name, HCCL_IPC_MEM_NAME_LEN);
            shm_buf->notifyNode[i].pid = pid[0];
            HCCL_DEBUG("aclrtNotifySetImportPid name[%s] ,pid[%d]", name, pid[0]);
            break;
        }
    }
    HCCL_DEBUG("after aclrtNotifySetImportPid i = %d", i);
    /*释放进程间互斥锁*/
    while (!__sync_bool_compare_and_swap(&(shm_buf->ref_cnt), 1, 0))
        ;

    if (i == IPC_SHM_NOTIFY_NUM_MAX) {
        HCCL_ERROR("IPC_SHM_NOTIFY_NUM_MAX size is not enough");
        DestroyIpcNotifyShm();
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    return ACL_SUCCESS;
}

aclError aclrtNotifySetImportPidInterServer(aclrtNotify notify, aclrtServerPid* serverPids, size_t num)
{
    const aclrtServerPid& serverPid = *serverPids;
    return aclrtNotifySetImportPid(notify, serverPid.pid, serverPid.num);
}

aclError aclrtNotifyImportByKey(aclrtNotify* notify, const char* name, uint64_t flag)
{
    if (nullptr == notify || nullptr == name) {
        HCCL_ERROR("parameter error : notify[%p], name[%p]", notify, name);
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    // open notify之前，先进行白名单较验
    char ipcMemShmName[100] = {0};
    (void)snprintf_s(
        ipcMemShmName, sizeof(ipcMemShmName), sizeof(ipcMemShmName) - 1, "%s%s", "IPC_NOTIFY_", g_shm_name);
    IpcNotifyWriteList_S* shm_buf
        = (IpcNotifyWriteList_S*)sal_share_memory_create(ipcMemShmName, sizeof(IpcNotifyWriteList_S));
    int i = 0;
    for (i = 0; i < IPC_SHM_NOTIFY_NUM_MAX; i++) {
        // 根据name 进行索引，先较对name，再较对pid
        if (!strcmp(shm_buf->notifyNode[i].ipcName, name)) {
            HCCL_INFO("notifyNode[%d], name[%s] pid[%d]", i, name, shm_buf->notifyNode[i].pid);
            s32 pid = 0;
            SalGetBareTgid(&pid); // 当前进程id
            if (shm_buf->notifyNode[i].pid == pid) {
                break;
            }
        }
    }
    // 未找到NAME
    if (i == IPC_SHM_NOTIFY_NUM_MAX) {
        HCCL_ERROR("aclrtNotifyImportByKey error , can't find name[%s]", name);
        DestroyIpcNotifyShm();
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_notify_t* ipc_notify = (rt_notify_t*)malloc(sizeof(rt_notify_t));
    if (ipc_notify == nullptr) {
        HCCL_ERROR("ipc_notify allocate failed");
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }
    memset(ipc_notify, 0, sizeof(rt_notify_t));

    rt_shm_ipc_name_t* ipc_name_shm = (rt_shm_ipc_name_t*)sal_share_memory_create(name, sizeof(rt_shm_ipc_name_t));
    if (nullptr == ipc_name_shm) {
        HCCL_ERROR("ipc_name_shm allocate failed");
        free(ipc_notify);
        return ACL_ERROR_RT_MEMORY_ALLOCATION;
    }

    rt_shm_notify_t* notify_shm
        = (rt_shm_notify_t*)sal_share_memory_create(ipc_name_shm->ipc_notify_shm_name, sizeof(rt_shm_notify_t));
    if (nullptr == notify_shm) {
        HCCL_ERROR("ipc_notify_shm allocate failed");
        sal_share_memory_destroy(ipc_name_shm);
        free(ipc_notify);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    /*记录创建或者打开的共享内存区地址*/
    ipc_notify->ipc_notify_shm = notify_shm;
    ipc_notify->ipc_name_shm = nullptr;
    ipc_notify->notify_id = ipc_name_shm->notify_id;

    sal_share_memory_destroy(ipc_name_shm);
    *notify = (rt_notify_t*)ipc_notify;

    HCCL_DEBUG("notify_id[%llu], device[%d]", ipc_notify->notify_id, notify_shm->device_id);
    return ACL_SUCCESS;
}

rtError_t rtNotifyGetAddr(rtNotify_t notify, uint64_t* host_PA, uint64_t* device_PA)
{
    if (notify == nullptr || host_PA == nullptr || device_PA == nullptr) {
        HCCL_ERROR("parameter error : notify[%p], host_PA[%p], device_PA[%p]", notify, host_PA, device_PA);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_notify_t* ipc_notify = (rt_notify_t*)notify;
    if (nullptr == ipc_notify->ipc_notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", ipc_notify->ipc_notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    *device_PA = (u64)&ipc_notify->ipc_notify_shm->record_cnt[ipc_notify->notify_id];
    *host_PA = (u64)&ipc_notify->ipc_notify_shm->record_cnt[ipc_notify->notify_id];

    return RT_ERROR_NONE;
}
#endif

aclError aclrtMemcpy(void* dst, size_t destMax, const void* src, size_t count, aclrtMemcpyKind kind)
{
    aclError ret;

    ret = (aclError)memcpy_s(dst, count, src, count);

    if (ret) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return ACL_SUCCESS;
}

rtError_t rtRDMASend_stub(u32 wqe_index, struct cn_info* cn, rtStream_t stream)
{
    if (cn == nullptr) {
        HCCL_ERROR("parameter error : notify[%p]", cn);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    stream_class* rtstream = NULL;
    rtstream = (stream_class*)stream;

    stream_task_t stream_task;
    stream_task.task_type = TASK_TYPE_RDMA_SEND;
    stream_task.stream_para.rdmasend.wqe_index = wqe_index;
    stream_task.stream_para.rdmasend.cn = cn;
    rtstream->push_task(&stream_task);

    return RT_ERROR_NONE;
}

rtError_t rtMetadataRegister(void* handle, const char* metadata) { return RT_ERROR_NONE; }

// Add for optimize runtime Stub by l on 2018-01-11 Below
/*
 *****************************************************************************
 * 函 数 名  : threadfun
 * 功能描述  : 线程函数实现
 * 输入参数  : p
 * 输出参数  : 无
 * 返 回 值  : void
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
void* threadfun(void* p)
{
    s32 iRet = SAL_OK;
    thread_class* pcthread = (thread_class*)p; // 管理本任务的对象.

    iRet = pcthread->update_thread_state(THREAD_STATE_WORKING);

    if (iRet) {
        HCCL_ERROR("Thread Update State To WORKING failed[%d]", iRet);
        return NULL;
    }

    iRet = pcthread->thread_handler();

    if (iRet) {
        HCCL_WARNING("[STUB] Thread Handler Return Failed[%d]", iRet);
    }

    iRet = pcthread->update_thread_state(THREAD_STATE_STOPED);

    if (iRet) {
        HCCL_WARNING("[STUB] Thread Update State To STOP failed[%d]", iRet);
    }

    return NULL;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.thread_class
 * 功能描述  : 构造函数, 填充默认值.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
thread_class::thread_class()
{
    // HCCL_INFO("Creat a new ThreadClass.");
    uithread_update_interval = THREAD_DEFAULT_UPDATE_INTERVAL; // 100ms
    uithreadstate = THREAD_STATE_STOPED;
    threadfd = NULL;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.~thread_class
 * 功能描述  : 析构函数, 释放必要资源.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
thread_class::~thread_class()
{
    // HCCL_INFO("Destroy an old ThreadClass.");
    try {
        (void)stop_thread(); // 停止任务
    } catch (...) {
        HCCL_ERROR("exception.");
    }
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.set_new_interval
 * 功能描述  : 设定任务间隔
 * 输入参数  : uiNewInterval
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::set_new_interval(u32 uinewinterval)
{
    s32 iRet = SAL_OK;

    // HCCL_INFO("[STUB] thread_class::set_new_interval:Set Thread Update Interval to [%d]us", uiNewInterval);
    uithread_update_interval = uinewinterval;

    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.get_current_interval
 * 功能描述  : 查询当前任务间隔
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : u32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
u32 thread_class::get_current_interval() { return uithread_update_interval; }

/*
 *****************************************************************************
 * 函 数 名  : thread_class.start_thread
 * 功能描述  : 启动任务(带线程名).
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::start_thread(string thread_name)
{
    s32 iRet = SAL_OK;

    if (threadfd) // 先停止任务.
    {
        (void)stop_thread();
    }

    iRet = update_thread_state(THREAD_STATE_INITALING); // 恢复任务默认状态

    if (iRet) {
        HCCL_ERROR("Thread Update State failed[%d]", iRet);
        threadfd = 0;
        return iRet;
    }

    iRet = pre_start_handler();

    if (iRet) {
        HCCL_ERROR("Pre Start Handler failed[%d]", iRet);
        threadfd = 0;
        return SAL_E_ERROR;
    }

    // 启动任务
    threadfd = sal_thread_create(thread_name, threadfun, this);

    if (NULL == threadfd) // 任务启动失败
    {
        HCCL_ERROR("Create Thread failed");
        threadfd = NULL;
        return SAL_E_ERROR;
    }

    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.start_thread
 * 功能描述  : 启动任务(使用默认线程名).
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::start_thread() { return start_thread("ThreadClass"); }

/*
 *****************************************************************************
 * 函 数 名  : thread_class.stop_thread
 * 功能描述  : 停止任务.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::stop_thread()
{
    u32 uiInterval = get_current_interval() / 10 + THREAD_UPDATE_MIN; // 计算检查周期
    u32 uiStopCounter = THREAD_STOP_COUNTER;
    s32 iRet = SAL_OK;

    if (threadfd) {
        // HCCL_INFO("Stop Thread. Waiting handler's exit. Check Every [%d]us", uiInterval);

        iRet = pre_stop_handler(); // 通知Handler主动退出.

        if (iRet) {
            HCCL_WARNING("Pre Stop Handler failed[%d]", iRet);
        }

        // 等待任务主动停止
        for (uiStopCounter = THREAD_STOP_COUNTER; (THREAD_STATE_STOPED != uithreadstate) && uiStopCounter;
             uiStopCounter--) {
            SaluSleep(uiInterval);
        }

        if (THREAD_STATE_STOPED != uithreadstate) // 任务没有主动停止,尝试强制退出.
        {
            HCCL_WARNING("Stop Thread failed after [%d]us, Force Stop...", uiInterval * THREAD_STOP_COUNTER);

            iRet = sal_thread_destroy(threadfd);

            if (SAL_OK != iRet) {
                HCCL_ERROR("Force Stop Thread failed[%d]", iRet);

                iRet = SAL_E_ERROR;
            }

            iRet = update_thread_state(THREAD_STATE_STOPED); // 恢复任务默认状态

            if (iRet) {
                HCCL_ERROR("Thread Update State failed[%d]", iRet);

                threadfd = 0;
            }

            HCCL_WARNING("Force Stop Thread Success");
        } else {
            // HCCL_INFO("Stop Thread Success. Counter cost [%d]", THREAD_STOP_COUNTER - uiStopCounter);
        }
    }

    threadfd = 0;
    return iRet;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.update_thread_state
 * 功能描述  : 刷新任务状态
 * 输入参数  : uiNewState
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::update_thread_state(u32 uinewstate)
{
    if (THREAD_STATE_MAX <= uinewstate) {
        HCCL_ERROR("Thread update to State [%d] failed", uinewstate);

        return SAL_E_PARA;
    }

    uithreadstate = uinewstate;

    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.thread_handler
 * 功能描述  : Thread回调函数.
               pre_stop_handler()执行后, thread_handler()需要在
GetCurrentInterval() us内主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::thread_handler()
{
    while (get_current_interval()) {
        HCCL_WARNING("Thread Handler use default func");

        SaluSleep(get_current_interval());
    }

    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.pre_start_handler
 * 功能描述  : Thread即将启动, 通知ThreadHandler做好准备.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::pre_start_handler()
{
    HCCL_WARNING("Pre Start Handler use default func");
    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : thread_class.pre_stop_handler
 * 功能描述  : Thread即将停止, 通知ThreadHandler主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 thread_class::pre_stop_handler()
{
    HCCL_WARNING("Pre Stop Handler use default func");
    (void)set_new_interval(0);
    return SAL_OK;
}

std::atomic<s32> stream_class::streamIdCounter_(0);
std::atomic<u32> stream_class::taskIdCounter_(0);

std::map<rtStream_t, int32_t> stream_class::streamMap_;
std::mutex stream_class::mapMutex_;
std::map<s32, atomic_ptr_t> stream_class::refCountMap_;
// std::map<s32, std::unique_ptr<Msprof::Engine::Reporter> > stream_class::reporterMapRuntime_;
// std::map<s32, std::unique_ptr<Msprof::Engine::Reporter> > stream_class::reporterMapHWTS_;
std::array<std::string, 8> stream_class::lineFeed_ = {"", "", "", "", "", "", "", ""};

/*
 *****************************************************************************
 * 函 数 名  : stream_class.stream_class
 * 功能描述  : 构造函数, 填充默认值.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
stream_class::stream_class(s32 device_id) : deviceId_(device_id), streamId_(-1), stream_enabled_(true)
{
    // 生成stream id, 并添加到容器
    streamId_ = streamIdCounter_.fetch_add(1);
    task_info.streamId = streamId_;
    {
        std::unique_lock<std::mutex> lock(mapMutex_);
        streamMap_[(rtStream_t)this] = streamId_; // 构造函数中使用this当Key是可行的
    }

    // 初始化reportData_
    std::string postFixRuntime("runtime-");
    // postFixRuntime += std::to_string(SalGetPid());
    // postFixRuntime += "-";
    postFixRuntime += std::to_string(deviceId_);
    memcpy(dataRuntime_.tag, postFixRuntime.c_str(), postFixRuntime.size() + 1);
    dataRuntime_.deviceId = deviceId_;

    std::string postFixHWTS("HWTS-");
    // postFixHWTS += std::to_string(SalGetPid());
    // postFixHWTS += "-";
    postFixHWTS += std::to_string(deviceId_);
    memcpy(dataHWTS_.tag, postFixHWTS.c_str(), postFixHWTS.size() + 1);
    dataHWTS_.deviceId = deviceId_;

    // device id增加stream的计数
    {
        std::unique_lock<std::mutex> lock(mapMutex_);

        if (refCountMap_.find(device_id) == refCountMap_.end()) {
            refCountMap_[device_id] = atomic_ptr_t(new std::atomic<u32>(0));

            HCCL_DEBUG("new device found[%d]", device_id);

            // // 生成Runtime/HWTS reporter
            // std::unique_ptr<Msprof::Engine::ReporterStub> reporterRuntime = \
            //     std::unique_ptr<Msprof::Engine::ReporterStub>(new Msprof::Engine::ReporterStub());
            // // reporterMapRuntime_[device_id] = std::move(reporterRuntime);

            // std::unique_ptr<Msprof::Engine::ReporterStub> reporterHWTS = \
            //     std::unique_ptr<Msprof::Engine::ReporterStub>(new Msprof::Engine::ReporterStub());
            // reporterMapHWTS_[device_id] = std::move(reporterHWTS);

            // HWTS日志的文件头
            std::string file_content("{\r\"traceEvents\": [\r");
            dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
            dataHWTS_.dataLen = file_content.size();
            // reporterMapHWTS_[device_id]->Report(&dataHWTS_);
        }
        u32 streamCount = refCountMap_[device_id]->fetch_add(1);
        HCCL_DEBUG("streamCount = %u, deviceId_[%d]", streamCount, deviceId_);
    }

    stream_task_lock = sal_mutex_create("stream_task_lock");
    stream_task_list.clear(); // 清空任务队列

    thread_trigger = sal_sem_create("stream_thread", SAL_FALSE, 0);
    stream_task_done = sal_sem_create("stream_task_done", SAL_FALSE, 0);
    (void)start_thread("StreamThread");
}

/*
 *****************************************************************************
 * 函 数 名  : stream_class.~stream_class
 * 功能描述  : 析构函数, 释放必要资源.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  :
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
stream_class::~stream_class()
{
    try {
        // 退出信息记录, 方便问题定位
        HCCL_DEBUG("Stream[%p] Destroy Now", this);
        // 停止任务
        (void)stop_thread();

        if (thread_trigger) {
            sal_sem_destroy(thread_trigger);
        }

        if (stream_task_done) {
            sal_sem_destroy(stream_task_done);
        }

        // device id增加stream的计数, 最后一个stream销毁时将文件Flush
        {
            std::unique_lock<std::mutex> lock(mapMutex_);
            u32 streamCount = refCountMap_[deviceId_]->fetch_sub(1);
            HCCL_DEBUG("streamCount = %u, deviceId_[%d]", streamCount, deviceId_);
            if (streamCount == 1) {
                // Runtime的task信息文件Flush
                // reporterMapRuntime_[deviceId_]->Flush();

                // HWTS的task信息文件Flush
                std::string file_content;
                file_content += "\r]\r}";
                dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
                dataHWTS_.dataLen = file_content.size();
                // reporterMapHWTS_[deviceId_]->Report(&dataHWTS_);
                // reporterMapHWTS_[deviceId_]->Flush();

                // Map里删除对应的deviceId
                refCountMap_.erase(deviceId_);
                // reporterMapHWTS_.erase(deviceId_);
                // reporterMapRuntime_.erase(deviceId_);
            }

            streamMap_.erase((rtStream_t)this);
        }

        // 清空软表
        stream_task_list.clear();

        // 销毁互斥锁
        if (stream_task_lock) {
            sal_mutex_destroy(stream_task_lock);
        }

        stream_task_lock = NULL;

    } catch (...) {
        HCCL_ERROR("exception.");
    }
}

void stream_class::HWTSLog(const stream_task_t& task, u64 ts_start, u64 duration)
{
    std::string content;
    content += "{ \"pid\":";
    content += std::to_string(streamId_);
    content += ", \"ts\":";
    content += std::to_string((double)ts_start / 1000.0);
    content += ", \"dur\":";
    content += std::to_string((double)duration / 1000.0);
    content += ", \"name\":";
    content += std::to_string(task.task_id);
    content += ", \"args\":{ ";
    // content += "task type\":";
    // content += std::to_string(task.task_type);
    content += "\"us\":";
    content += std::to_string((double)duration / 1000.0);
    content += " } }";

    {
        std::unique_lock<std::mutex> lock(mapMutex_);

        std::string file_content("");
        file_content += lineFeed_[deviceId_];
        file_content += content;
        dataHWTS_.data = (unsigned char*)(const_cast<char*>(file_content.c_str()));
        dataHWTS_.dataLen = file_content.size();
        // reporterMapHWTS_[deviceId_]->Report(&dataHWTS_);
        lineFeed_[deviceId_] = ",\r";
    }
}

u64 stream_class::TimestampNanosecond()
{
    // 此时间戳获取方式需要与runtime保持一致

    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (u64)(ts.tv_sec * 1000000000 + ts.tv_nsec);
}

/*
*****************************************************************************
* 函 数 名  : stream_class.push_task
* 功能描述  : stream list pushback API
* 输入参数  : module
*             eLogType
*             fmt
*             args
* 输出参数  : 无
* 返 回 值  : s32
* 其它说明  :
*
* 修改历史      :
*  1.日    期   : 2018年1月11日
*    作    者   : lifuning
*    修改内容   : 新生成函数
*
*****************************************************************************
*/
s32 stream_class::push_task(stream_task_t* stream_task)
{
    if (stream_task == nullptr) {
        HCCL_ERROR("stream_task is NULL");
        return SAL_E_ERROR;
    }

    // add task_id for this task
    stream_task->task_id = taskIdCounter_.fetch_add(1);
    task_info.taskId = stream_task->task_id;
    task_info.streamId = streamId_;
    // 将task信息写入到文件, runtime profiling桩函数
    rtProfTaskTrack_t taskTrackData;
    taskTrackData.head.rserved = 0x1020304; // hccl_perf_tool用于防错
    taskTrackData.timeStamp = TimestampNanosecond();
    taskTrackData.streamId = streamId_;
    taskTrackData.taskType = (u16)stream_task->task_type;
    taskTrackData.taskId = (u16)stream_task->task_id;
    taskTrackData.deviceId = deviceId_;
    dataRuntime_.data = (unsigned char*)&taskTrackData;
    dataRuntime_.dataLen = sizeof(rtProfTaskTrack_t);

    {
        std::unique_lock<std::mutex> lock(mapMutex_);
        // reporterMapRuntime_[deviceId_]->Report(&dataRuntime_);
    }

    MSG_LOCK();
    stream_task_list.push_back(*stream_task);
    MSG_UNLOCK();

    trigger_thread();

    return SAL_OK;
}

s32 stream_class::get_stream_id() const { return streamId_; }

s32 stream_class::get_device_id() const { return deviceId_; }

void stream_class::trigger_thread()
{
    if (thread_trigger) {
        (void)sal_sem_give(thread_trigger);
    } else {
        HCCL_ERROR("thread_trigger is NULL");
    }

    return;
}

void stream_class::set_stream_enabled(bool enabled) { stream_enabled_ = enabled; }

bool IsFailureTask(u32 deviceId, tasktype_e task_type)
{
    return (FailureTaskType != TASK_TYPE_RESERVED && FailureDeviceId == deviceId && FailureTaskType == task_type);
}
void ClearFailureTask()
{
    FailureDeviceId = 0xFFFFFFFF;
    FailureTaskType = TASK_TYPE_RESERVED;
}
/*
 *****************************************************************************
 * 函 数 名  : stream_class.thread_handler
 * 功能描述  : 任务队列主处理函数
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::thread_handler()
{
    u32 streamId = get_stream_id();
    u32 deviceId = get_device_id();
    while (get_current_interval()) {
        // 释放 CPU, 周期性唤醒任务
        (void)sal_sem_take(thread_trigger, get_current_interval());
        // 检查 stream_task_list 是否为空
        if (!stream_task_list.empty()) {
            // HCCL_DEBUG("stream_task_list size is [%d].", stream_task_list.size());

            // 处理stream_task_list
            do {
                list<stream_task_t>::iterator iter;

                MSG_LOCK();

                for (iter = stream_task_list.begin(); iter != stream_task_list.end();) {
                    MSG_UNLOCK();
                    if (IsFailureTask(deviceId, iter->task_type)) {
                        for (auto it : taskFailCallbackMap) {
                            rtExceptionInfo tmpInfo;
                            tmpInfo.taskid = iter->task_id;
                            tmpInfo.streamid = streamId;
                            tmpInfo.tid = 0;
                            tmpInfo.deviceid = deviceId;
                            tmpInfo.retcode = 1;
                            if (it.second != nullptr) {
                                it.second(&tmpInfo);
                            }
                        }
                        ClearFailureTask();
                    }

                    switch (iter->task_type) {
                        case TASK_TYPE_MEMCPY: {
                            u64 ts_start = TimestampNanosecond();
                            rtError_t ret = memcpy_async(
                                iter->stream_para.memcpystruct.dst, iter->stream_para.memcpystruct.src,
                                iter->stream_para.memcpystruct.count);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task MemcpyAsync failed, ret[%d], para:dst[%d] src[%d] count[%d].", ret,
                                    iter->stream_para.memcpystruct.dst, iter->stream_para.memcpystruct.src,
                                    iter->stream_para.memcpystruct.count);
                            }

                            break;
                        }

                        case TASK_TYPE_RECORD: {
                            rtError_t ret = event_record(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task event record failed, ret[%d], para:event[%p],sem[%p].", ret,
                                    iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_MULTIDEV_RECORD: {
                            rtError_t ret = event_multidev_record(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task event multidev record failed, ret[%d], para:event[%p],sem[%p].", ret,
                                    iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_WAIT: {
                            rtError_t ret = event_wait(&(iter->stream_para.event));

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task event wait failed, ret[%d], para:event[%p],sem[%p].", ret,
                                    iter->stream_para.event.event_handler, iter->stream_para.event.sem);
                            }

                            break;
                        }

                        case TASK_TYPE_REDUCE: {
                            u64 ts_start = TimestampNanosecond();
                            cce::ccStatus_t ret = vector_reduce(
                                iter->stream_para.reducestruct.src1, iter->stream_para.reducestruct.src2,
                                iter->stream_para.reducestruct.count_reduce, iter->stream_para.reducestruct.datatype,
                                iter->stream_para.reducestruct.op, iter->stream_para.reducestruct.dst_reduce);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (cce::CC_STATUS_SUCCESS != ret) {
                                HCCL_ERROR(
                                    "Task vector reduce failed, ret[%d], para:src1[%d] src2[%d] count_reduce[%d] "
                                    "datatype[%d] op[%d] dst_reduce[%d].",
                                    ret, iter->stream_para.reducestruct.src1, iter->stream_para.reducestruct.src2,
                                    iter->stream_para.reducestruct.count_reduce,
                                    iter->stream_para.reducestruct.datatype, iter->stream_para.reducestruct.op,
                                    iter->stream_para.reducestruct.dst_reduce);
                            }

                            break;
                        }

                        case TASK_TYPE_USLEEP: {
                            rtError_t ret = stream_usleep(iter->stream_para.usec);

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task MemcpyAsync failed, ret[%d], para:usec[%d].", ret, iter->stream_para.usec);
                            }

                            break;
                        }

                        case TASK_TYPE_NOTIFY_RECORD: {
                            u64 ts_start = TimestampNanosecond();

                            rtError_t ret = notify_record(iter->stream_para.notify);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR("Task NotyfiRecordAsync failed, ret[%d]", ret);
                            }

                            break;
                        }

                        case TASK_TYPE_NOTIFY_WAIT: {
                            u64 ts_start = TimestampNanosecond();

                            rtError_t ret = notify_wait(iter->stream_para.notify);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR("Task NotyfiWaitAsync failed, ret[%d]", ret);
                            }

                            break;
                        }

                        case TASK_TYPE_RDMA_SEND: {
                            u64 ts_start = TimestampNanosecond();
                            rtError_t ret
                                = rdma_send(iter->stream_para.rdmasend.wqe_index, iter->stream_para.rdmasend.cn);
                            u64 duration = TimestampNanosecond() - ts_start;
                            HWTSLog(*iter, ts_start, duration);

                            if (RT_ERROR_NONE != ret) {
                                HCCL_ERROR(
                                    "Task RdmaSend Async failed, ret[%d], wqe_index[%d]", ret,
                                    iter->stream_para.rdmasend.wqe_index);
                            }

                            break;
                        }

                        case TASK_TYPE_CALLBACK_FUNC: {
                            while (iter->stream_para.callbackTask.isBlock) {
                                if (iter->stream_para.callbackTask.isExecuted) {
                                    break;
                                }
                            }
                            break;
                        }

                        default: {
                            HCCL_DEBUG("Not support the task type [%d].", iter->task_type);
                            break;
                        }
                    }

                    MSG_LOCK();
                    iter = stream_task_list.erase(iter);
                    MSG_UNLOCK();

                    (void)sal_sem_give(stream_task_done);
                    MSG_LOCK();
                }

                MSG_UNLOCK();

            } while (!stream_task_list.empty());
        }
    }

    return SAL_OK;
}

/*
 *****************************************************************************
 * 函 数 名  : stream_class.pre_stop_handler
 * 功能描述  : StopThread触发, 通知ThreadHandler主动退出.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::pre_stop_handler()
{
    s32 ret = SAL_OK;
    // 唤醒日志线程, 让日志线程将所有日志写入文件.
    trigger_thread();
    ret = set_new_interval(0);
    // 唤醒日志线程, 让其快速主动退出
    trigger_thread();
    return ret;
}

/*
 *****************************************************************************
 * 函 数 名  : stream_class.pre_start_handler
 * 功能描述  : StartThread触发, 通知ThreadHandler即将被调用.
 * 输入参数  : 无
 * 输出参数  : 无
 * 返 回 值  : s32
 * 其它说明  :
 *
 * 修改历史      :
 *  1.日    期   : 2018年1月11日
 *    作    者   : lifuning
 *    修改内容   : 新生成函数
 *
 *****************************************************************************
 */
s32 stream_class::pre_start_handler()
{
    u32 new_interval = THREAD_UPDATE_MIN;

    return set_new_interval(new_interval);
}

rtError_t stream_class::stream_synchronize()
{
    if (!stream_task_list.empty()) {
        do {
            (void)sal_sem_take(stream_task_done, get_current_interval());

        } while (!stream_task_list.empty());
    }
    return RT_ERROR_NONE;
}

rtError_t stream_class::stream_usleep(u32 usec)
{
    s32 iRet = 0;
    iRet = usleep(usec);

    if (iRet) {
        HCCL_ERROR("Sleep: usleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::event_record(rtEvent_t event)
{
    s32 err = -1;

    if (!event) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_give(rtevent->sem);

    if (err) {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::event_multidev_record(rtEvent_t event)
{
    s32 err = -1;

    if (!event) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_give(rtevent->sem);

    if (err) {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::event_wait(rtEvent_t event)
{
    s32 err = -1;
    s32 usec = SAL_SEM_FOREVER; // 定义10ms等不到event，认为出错，避免挂死

    if (!event) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    rt_event_stub_t* rtevent = NULL;
    rtevent = (rt_event_stub_t*)event;

    if (!rtevent->sem) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    err = sal_sem_take(rtevent->sem, usec);

    if (err) {
        return ACL_ERROR_RT_INTERNAL_ERROR;
    }

    if (rtevent->sem) {
        HCCL_DEBUG(" ++ event[%p] wait sem[%p] destroy!", rtevent->event_handler, rtevent->sem);
        sal_sem_destroy(rtevent->sem);
        rtevent->sem = NULL;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::memcpy_async(void* dst, void* src, uint64_t count)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }

    rtError_t ret;

    ret = (rtError_t)memcpy_s(dst, count, src, count);

    if (ret) {
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::notify_record(rt_notify_t* notify)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }
    rt_notify_t* ipc_notify = notify;
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    HCCL_DEBUG("record : device[%d], notify_id[%llu]", notify_shm->device_id, ipc_notify->notify_id);

    // wait notify, 通过shm的原子操作实现
    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT;
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 0, 1)) {
        SaluSleep(1000);

        timeout_cnt--;
        if (timeout_cnt <= 0) {
            HCCL_ERROR(
                "record timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id], notify_shm->device_id, ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::notify_wait(rt_notify_t* notify)
{
    if (stream_enabled_ == false) {
        return RT_ERROR_NONE;
    }
    rt_notify_t* ipc_notify = notify;
    HCCL_DEBUG("notify_wait: notify[%p]", ipc_notify);
    rt_shm_notify_t* notify_shm = (rt_shm_notify_t*)ipc_notify->ipc_notify_shm;
    if (nullptr == notify_shm) {
        HCCL_ERROR("parameter error : notify_shm[%p]", notify_shm);
        return ACL_ERROR_RT_PARAM_INVALID;
    }

    void* tmpPtr = &(notify_shm->record_cnt[ipc_notify->notify_id]);
    HCCL_INFO("wait : device[%d], notify_id[%llu] tmpPtr[%p]", notify_shm->device_id, ipc_notify->notify_id, tmpPtr);

    s32 timeout_cnt = NOTIFY_TIMEOUT_CNT;
    while (!__sync_bool_compare_and_swap(&(notify_shm->record_cnt[ipc_notify->notify_id]), 1, 0)) {
        SaluSleep(1000);

        timeout_cnt--;
        if (timeout_cnt <= 0) {
            HCCL_ERROR(
                "wait timeout : record_cnt[%d], device_id[%d], notify_id[%llu]",
                notify_shm->record_cnt[ipc_notify->notify_id], notify_shm->device_id, ipc_notify->notify_id);
            return ACL_ERROR_RT_PARAM_INVALID;
        }
    }

    return RT_ERROR_NONE;
}

rtError_t stream_class::rdma_send(u32 wqe_index, void* cnn)
{
    struct cn_info* con_info = (struct cn_info*)cnn;
    struct SendWr* wqe = &(con_info->qp.send_mr_mgr.wq[wqe_index]);

    while ((con_info->qp.local_qp_msg_ptr->cnt != con_info->qp.remote_qp_msg_ptr->rsp_cnt)
               && (con_info->qp.send_mr_mgr.wqe_set[wqe_index])
           || (!(con_info->qp.send_mr_mgr.wqe_set[wqe_index]))) {
        // 上个命令处理完再处理本次的*/
        // HCCL_INFO("waiting for previous cmd OK...");
        SaluSleep(10000);
    }

    con_info->qp.local_qp_msg_ptr->cmd = QP_CMD_WRITE_DATA;
    con_info->qp.local_qp_msg_ptr->msg.write_info.dst_addr = (void*)wqe->dstAddr;
    con_info->qp.local_qp_msg_ptr->msg.write_info.len = wqe->bufList->len;
    con_info->qp.local_qp_msg_ptr->msg.write_info.op = wqe->op;
    HCCL_INFO(
        "wqe_index:%d qp.local_qp_msg_ptr.dstAddr[0x%0x] qp write_info.len[%d] wqe_op[%d] local[0x%0x] "
        "wqe_dstAddr[0x%0x]",
        wqe_index, (u64)(con_info->qp.local_qp_msg_ptr->msg.write_info.dst_addr),
        con_info->qp.local_qp_msg_ptr->msg.write_info.len, wqe->op, wqe->bufList->addr, wqe->dstAddr);
    HcclResult ret;
    if (wqe->op == 0) {
        HCCL_INFO(
            "data[0][%f], wqe->bufList->addr[%f]", (float)con_info->qp.local_qp_msg_ptr->msg.write_info.data[0],
            *((float*)(wqe->bufList->addr)));
        ret = sal_memcpy(
            &(con_info->qp.local_qp_msg_ptr->msg.write_info.data[0]), QP_MSG_MAX_SIZE, (void*)wqe->bufList->addr,
            wqe->bufList->len);
        HCCL_INFO(
            "data[0][%f], wqe->bufList->addr[%f]", (float)con_info->qp.local_qp_msg_ptr->msg.write_info.data[0],
            *((float*)(wqe->bufList->addr)));
    } else if (wqe->op == 4) {
        void* ptrTmp = (void*)wqe->bufList->addr;
        HCCL_INFO(
            "[TMP] ptrTmp[%p]  notifyData[%d] Length[%d]", ptrTmp,
            con_info->qp.local_qp_msg_ptr->msg.write_info.data[0], wqe->bufList->len);
        ret = sal_memcpy(
            (void*)wqe->bufList->addr, QP_MSG_MAX_SIZE, (con_info->qp.local_qp_msg_ptr->msg.write_info.dst_addr),
            wqe->bufList->len);
    } else {
        HCCL_ERROR("rdma send: sal memcpy error");
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("rdma send: sal memcpy error");
        return ACL_ERROR_RT_PARAM_INVALID;
    }
    con_info->qp.local_qp_msg_ptr->cnt++;
    wqe->bufList->len = 0;
    wqe->bufList->addr = 0;
    con_info->qp.send_mr_mgr.wqe_set[wqe_index] = false; /*本wqe已发送，故标记为空*/

    return RT_ERROR_NONE;
}

void stream_class::ExecuteCallbackFunc()
{
    if (stream_task_list.empty()) {
        return;
    }
    stream_task_t& task = stream_task_list.front();
    std::unique_lock<std::mutex> lock(isExecutedMutex);
    if (task.task_type == TASK_TYPE_CALLBACK_FUNC) {
        task.stream_para.callbackTask.func(task.stream_para.callbackTask.para);
        task.stream_para.callbackTask.isExecuted = 1;
    }
    isExecutedMutex.unlock();
}

template <typename T>
cce::ccStatus_t stream_class::reduce_op(T src1, T src2, T* dst, const cce::ccReduceOp_t op)
{
    switch (op) {
        case cce::CCE_RED_OP_SUM:
            *dst = src1 + src2;
            break;

        case cce::CCE_RED_OP_PROD:
            *dst = src1 * src2;
            break;

        case cce::CCE_RED_OP_MAX:
            *dst = (src1 > src2) ? src1 : src2;
            break;

        case cce::CCE_RED_OP_Min:
            *dst = (src1 < src2) ? src1 : src2;
            break;

        default:
            return cce::CC_STATUS_NOT_SUPPORTED;
    }

    return cce::CC_STATUS_SUCCESS;
}

cce::ccStatus_t stream_class::vector_reduce(
    const void* src1, const void* src2, uint32_t count, const cce::ccDataType_t datatype, const cce::ccReduceOp_t op,
    void* dst)
{
    int loop;
    void* input1 = (void*)src1;
    void* input2 = (void*)src2;
    float *x1, *x2, *x;
    char *y1, *y2, *y;
    u16 *z1, *z2, *z;
    s32 *m1, *m2, *m;
    s16 *n1, *n2, *n;
    if (!src1 || !src2 || !dst) {
        return cce::CC_STATUS_RESERVED;
    }

    if (datatype >= cce::CC_DATA_RESERVED) {
        return cce::CC_STATUS_RESERVED;
    }

    if (op >= cce::CCE_RED_OP_RESERVED) {
        return cce::CC_STATUS_RESERVED;
    }

    // float16在桩函数中使用int16代替
    if (datatype == cce::CC_DATA_HALF) {
        z1 = (u16*)input1;
        z2 = (u16*)input2;
        z = (u16*)dst;
    }

    if (datatype == cce::CC_DATA_FLOAT) {
        x1 = (float*)input1;
        x2 = (float*)input2;
        x = (float*)dst;
    }

    if (datatype == cce::CC_DATA_INT8) {
        y1 = (char*)input1;
        y2 = (char*)input2;
        y = (char*)dst;
    }

    if (datatype == cce::CC_DATA_INT16) {
        n1 = (s16*)input1;
        n2 = (s16*)input2;
        n = (s16*)dst;
    }

    if (datatype == cce::CC_DATA_INT32) {
        m1 = (s32*)input1;
        m2 = (s32*)input2;
        m = (s32*)dst;
    }

    for (loop = 0; loop < count; loop++) {
        if (datatype == cce::CC_DATA_HALF) {
            if (op == cce::CCE_RED_OP_SUM) {
                float f1 = fp16_ieee_to_fp32_value(*z1);
                float f2 = fp16_ieee_to_fp32_value(*z2);
                float f = (u32)f1 + (u32)f2;
                *z = fp16_ieee_from_fp32_value(f);
            } else {
                (void)reduce_op(*z1, *z2, z, op);
            }
            z1++;
            z2++;
            z++;
        }

        if (datatype == cce::CC_DATA_FLOAT) {
            (void)reduce_op(*x1, *x2, x, op);
            x1++;
            x2++;
            x++;
        }

        if (datatype == cce::CC_DATA_INT8) {
            (void)reduce_op(*y1, *y2, y, op);
            y1++;
            y2++;
            y++;
        }

        if (datatype == cce::CC_DATA_INT16) {
            (void)reduce_op(*n1, *n2, n, op);
            n1++;
            n2++;
            n++;
        }

        if (datatype == cce::CC_DATA_INT32) {
            (void)reduce_op(*m1, *m2, m, op);
            m1++;
            m2++;
            m++;
        }
    }

    return cce::CC_STATUS_SUCCESS;
}
// Add for optimize runtime Stub by l on 2018-01-11 Above

#ifdef __cplusplus
extern "C" {
#endif

#if 1
u32 __inet_addr_stub(const char* ip)
{
    if (!strcmp(ip, "192.168.1.62")) /*ibv_exp_server_ip*/
    {
        return htonl(
            0x7F000002); /*GDR的桩函数基于IP创建的共享内存，如果只有以下分支则UT/ST跑的时候，共享内存会冲突，故此处特殊处理下*/
    } else if (!strcmp(ip, "192.168.11.100")) /*ibv_exp_server_ip*/
    {
        return htonl(0x7F000003);
    } else {
        u32 ipv4_segment;
        u32 ipv4_addr = 0;
        u32 step = 0;
        u32 i = 0;
        u32 val[3] = {0};
        u32 val_cnt = 0;

        while (ip[i] != '\0') {
            if (ip[i] == '.') {
                if (3 == val_cnt) {
                    ipv4_segment = val[0] * 100 + val[1] * 10 + val[2];
                    ipv4_addr |= (ipv4_segment << (24 - (8 * step)));
                } else if (2 == val_cnt) {
                    ipv4_segment = val[0] * 10 + val[1];
                    ipv4_addr |= (ipv4_segment << (24 - (8 * step)));
                } else if (1 == val_cnt) {
                    ipv4_segment = val[0];
                } else {
                    HCCL_ERROR("Unknown IPv4 address");
                    return 0;
                }

                val_cnt = 0;
                step++;
            } else {
                val[val_cnt] = (u32)(ip[i] - '0');
                val_cnt++;
            }

            i++;
        }

        if (3 == val_cnt) {
            ipv4_segment = val[0] * 100 + val[1] * 10 + val[2];
            ipv4_addr |= (ipv4_segment << (24 - (8 * step)));
        } else if (2 == val_cnt) {
            ipv4_segment = val[0] * 10 + val[1];
            ipv4_addr |= (ipv4_segment << (24 - (8 * step)));
        } else if (1 == val_cnt) {
            ipv4_segment = val[0];
        } else {
            HCCL_ERROR("Unknown IPv4 address");
            return 0;
        }

        return htonl(ipv4_addr); // 127.0.0.1
    }
}

// 桩函数作为强符号在llt链接时替换交付代码中的弱符号
strong_alias(__inet_addr_stub, inet_addr);
#else
// 取消对inet_addr的打桩
#endif

#if 1
s32 __inet_pton_stub(int af, const char* ip, void* addrptr)
{
    *(u32*)addrptr = htonl(0x7F000001);
    return 1;
}

// 桩函数作为强符号在llt链接时替换交付代码中的弱符号
// strong_alias(__inet_pton_stub, inet_pton);
#else
// 取消对inet_addr的打桩
#endif

void* __WorkSpaceMemAllocStub(std::string tag, u64 size)
{
    void* ptr = NULL;
    HcclResult ret = hrtMalloc(&ptr, size);
    CHK_PRT_RET(ret, HCCL_ERROR("rt_malloc fail, tag[%s], size[%llu], ret[%d]", tag.c_str(), size, ret), NULL);
    return ptr;
}

strong_alias(__WorkSpaceMemAllocStub, WorkSpaceMemAlloc);

#ifdef __cplusplus
} // extern "C"
#endif

// 获取随机值
int RAND_bytes(char* buf, int num)
{
    int i;
    unsigned seed; // Random generator seed
    CHK_PTR_NULL(buf);
    seed = time(0);

    srand(seed);

    for (i = 0; i < num; i++) {
        buf[i] = (rand() % 25) + 65; // 产生1~15 的随机数
    }
    return 1;
}

static const s32 size_table[HCCL_DATA_TYPE_RESERVED] = {1, 2, 4, 2, 4}; // HCCL_DATA_TYPE_RESERVED
s32 get_data_size(const HcclDataType dataType)
{
    if (dataType < HCCL_DATA_TYPE_RESERVED) {
        return size_table[dataType];
    } else {
        HCCL_ERROR("data type[%d] out of range[%d, %d]", dataType, HCCL_DATA_TYPE_INT8, HCCL_DATA_TYPE_RESERVED - 1);
        return 0;
    }
}

/* 底层 runtime api 返回值转换成 HcclResult 格式 */

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __rt_get_dev_ip(s32 chipType, s32 devId, u32* ipAddr)
{
    CHK_PTR_NULL(ipAddr);
    std::string devIpStr;

    if (chipType == 0) { /* mini */
        /* HCCL用于节点内交换数据的socket通过host侧的环回IP完成 */
        /* IP地址的分配规则为：device0 - 127.0.0.1, device1 - 127.0.0.2, device3 - 127.0.0.3, device3 - 127.0.0.4 */
        devIpStr = "127.0.0." + std::to_string(1 + (devId % 8)); // （魔鬼数字解释）1,8与硬件资源数量相关
        *ipAddr = inet_addr(devIpStr.c_str());
    } else if (chipType == 1) { /* cloud */
        /* HCCL用于节点内交换数据的socket通过dev侧的虚拟IP完成 */
        /* 192.168.1 +（device devid 0 -3）.199-（host devid 0-7） */
        unsigned int board_id;
        dsmi_get_board_id((int)devId, &board_id);
        if ((board_id & 0xFFFFFFF0) == 0x10) // 判定当前为标卡场景910
        {
            devIpStr = "192.168." + std::to_string(1) + "." + std::to_string(199 - devId);
        } else {
            devIpStr = "192.168." + std::to_string(1 + (devId % 4)) + "."
                       + std::to_string(199 - devId); // 魔鬼数字资源数1,4,199
        }
        *ipAddr = inet_addr(devIpStr.c_str());
    } else {
        HCCL_ERROR("get unknown chip type[%d] dev:[%d]", chipType, devId);
        return HCCL_E_NOT_SUPPORT;
    }
    return HCCL_SUCCESS;
}
weak_alias(__rt_get_dev_ip, rt_get_dev_ip);

int32_t QueryPartitionMapPsId(uint64_t key, uint32_t* psId)
{
    *psId = 0;
    return 0;
}

int32_t InitPartitionMap(uint32_t partitionNum, uint32_t psNum, const uint32_t psId[]) { return 0; }

int32_t GetBatchPsIds(uint64_t* keys, uint32_t* psIds[], uint32_t num)
{
    for (int i = 0; i < num; i++) {
        (*psIds)[i] = 0;
    }
    return 0;
}

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle);

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName);

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api);

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length);

uint64_t MsprofGetHashId(const char* hashInfo, size_t length);

uint64_t MsprofSysCycleTime();

HcclResult AtraceSubmit(int32_t handle, const void* buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void AtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

HcclResult UtraceSubmit(int32_t handle, const void* buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    return HCCL_SUCCESS;
}

void UtraceDestroy(int32_t handle)
{
    (void)(handle);
    return;
}

int32_t AtraceCreateWithAttr(int32_t tracerType, const char* objName, const TraceAttr* attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceCreateWithAttr(int32_t tracerType, const char* objName, const TraceAttr* attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    return 0;
}

int32_t UtraceSetGlobalAttr(const TraceGlobalAttr* attr)
{
    (void)(attr);
    return 0;
}

int32_t AtraceSetGlobalAttr(const TraceGlobalAttr* attr)
{
    (void)(attr);
    return 0;
}

int32_t AtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

int32_t UtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    return 0;
}

int ibv_ext_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp)
{
    return 0;
}

int stub_ibv_ext_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp)
{
    return ibv_ext_post_send(qp, wr, bad_wr, ext_attr, ext_resp);
}

int stub_ibv_exp_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct wr_exp_rsp* exp_rsp)
{
    return -12;
}

urma_status_t stub_urma_post_jetty_send_wr(urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad_wr)
{
    return URMA_SUCCESS;
}

int stub_urma_poll_jfc(urma_jfc_t* jfc, int cr_cnt, urma_cr_t* cr)
{
    cr->status = URMA_CR_SUCCESS;
    return 1;
}

namespace hccl {
    std::map<std::string, void*> dlRaFuntionPtrMap = {
        {"RaQpCreate", (void*)&RaQpCreate},
        {"RaGetQpContext", (void*)&RaGetQpContext},
        {"RaGetTsqpDepth", (void*)&RaGetTsqpDepth},
        {"RaSetTsqpDepth", (void*)&RaSetTsqpDepth},
        {"RaQpDestroy", (void*)&RaQpDestroy},
        {"RaQpConnectAsync", (void*)&RaQpConnectAsync},
        {"RaGetQpStatus", (void*)&RaGetQpStatus},
        {"RaDeinit", (void*)&RaDeinit},
        {"RaGetNotifyBaseAddr", (void*)&RaGetNotifyBaseAddr},
        {"RaGetSockets", (void*)&RaGetSockets},
        {"RaInit", (void*)&RaInit},
        {"RaIsFirstUsed", (void*)&ra_is_first_used},
        {"RaIsLastUsed", (void*)&ra_is_last_used},
        {"RaMrDereg", (void*)&RaMrDereg},
        {"RaMrReg", (void*)&RaMrReg},
        {"RaRegisterMr", (void*)&RaRegisterMr},
        {"RaDeregisterMr", (void*)&RaDeregisterMr},
        {"RaRdevDeinit", (void*)&RaRdevDeinit},
        {"RaRdevInit", (void*)&RaRdevInit},
        {"RaRdevInitV2", (void*)&RaRdevInitV2},
        {"RaRdevInitWithBackup", (void*)&RaRdevInitWithBackup},
        {"RaSendWr", (void*)&RaSendWr},
        {"RaSendWrlist", (void*)&RaSendWrlist},
        {"RaSendWrlistExt", (void*)&RaSendWrlistExt},
        {"RaSocketBatchClose", (void*)&RaSocketBatchClose},
        {"RaSocketBatchConnect", (void*)&RaSocketBatchConnect},
        {"RaSocketBatchAbort", (void*)&RaSocketBatchAbort},
        {"RaSocketDeinit", (void*)&RaSocketDeinit},
        {"RaSocketInit", (void*)&RaSocketInit},
        {"RaSocketInitV1", (void*)&RaSocketInitV1},
        {"RaSocketListenStart", (void*)&RaSocketListenStart},
        {"RaSocketListenStop", (void*)&RaSocketListenStop},
        {"RaSocketRecv", (void*)&RaSocketRecv},
        {"RaSocketSend", (void*)&RaSocketSend},
        {"RaSocketSetWhiteListStatus", (void*)&RaSocketSetWhiteListStatus},
        {"RaSocketGetWhiteListStatus", (void*)&RaSocketGetWhiteListStatus},
        {"RaSocketWhiteListAdd", (void*)&RaSocketWhiteListAdd},
        {"RaSocketWhiteListDel", (void*)&RaSocketWhiteListDel},
        {"RaGetIfnum", (void*)&RaGetIfnum},
        {"RaGetIfaddrs", (void*)&RaGetIfaddrs},
        {"RaGetInterfaceVersion", (void*)&RaGetInterfaceVersion},
        {"RaEpollCtlAdd", (void*)&RaEpollCtlAdd},
        {"RaEpollCtlMod", (void*)&RaEpollCtlMod},
        {"RaEpollCtlDel", (void*)&RaEpollCtlDel},
        {"RaSetTcpRecvCallback", (void*)&RaSetTcpRecvCallback},
        {"RaCqCreate", (void*)&RaCqCreate},
        {"RaCqDestroy", (void*)&RaCqDestroy},
        {"RaNormalQpCreate", (void*)&RaNormalQpCreate},
        {"RaNormalQpDestroy", (void*)&RaNormalQpDestroy},
        {"RaSetQpAttrQos", (void*)&RaSetQpAttrQos},
        {"RaSetQpAttrTimeout", (void*)&RaSetQpAttrTimeout},
        {"RaSetQpAttrRetryCnt", (void*)&RaSetQpAttrRetryCnt},
        {"RaCreateCompChannel", (void*)&RaCreateCompChannel},
        {"RaDestroyCompChannel", (void*)&RaDestroyCompChannel},
        {"RaGetCqeErrInfo", (void*)&RaGetCqeErrInfo},
        {"RaRdevGetCqeErrInfoList", (void*)&RaRdevGetCqeErrInfoList},
        {"RaGetQpAttr", (void*)&RaGetQpAttr},
        {"RaCreateSrq", (void*)&RaCreateSrq},
        {"RaDestroySrq", (void*)&RaDestroySrq},
        {"RaQpCreateWithAttrs", (void*)&RaQpCreateWithAttrs},
        {"RaAiQpCreate", (void*)&RaAiQpCreate},
        {"RaSendWrV2", (void*)&RaSendWrV2},
        {"RaSendNormalWrlist", (void*)&RaSendNormalWrlist},
        {"RaPollCq", (void*)&RaPollCq},
        {"RaRecvWrlist", (void*)&RaRecvWrlist},
        {"RaSocketGetVnicIpInfos", (void*)&RaSocketGetVnicIpInfos},
        {"RaRdevGetSupportLite", (void*)&RaRdevGetSupportLite},
        {"RaCreateEventHandle", (void*)&RaCreateEventHandle},
        {"RaCtlEventHandle", (void*)&RaCtlEventHandle},
        {"RaWaitEventHandle", (void*)&RaWaitEventHandle},
        {"RaDestroyEventHandle", (void*)&RaDestroyEventHandle},
        {"RaQpBatchModify", (void*)&RaQpBatchModify},
        {"RaGetNotifyMrInfo", (void*)&RaGetNotifyMrInfo},
        {"RaTypicalQpCreate", (void*)&RaTypicalQpCreate},
        {"RaTypicalQpModify", (void*)&RaTypicalQpModify},
        {"RaTypicalSendWr", (void*)&RaTypicalSendWr},
        {"RaRdevGetPortStatus", (void*)&RaRdevGetPortStatus},
        {"RaSocketAcceptCreditAdd", (void*)&RaSocketAcceptCreditAdd},
        {"RaRemapMr", (void*)&RaRemapMr},
        {"RaTlvInit", (void*)&RaTlvInit},
        {"RaTlvDeinit", (void*)&RaTlvDeinit},
        {"RaTlvRequest", (void*)&RaTlvRequest},
        {"RaGetTlsEnable", (void*)&RaGetTlsEnable},
        {"RaSaveSnapshot", (void*)&RaSaveSnapshot},
        {"RaRestoreSnapshot", (void*)&RaRestoreSnapshot},
        {"RaQpDestroyWithoutCQ", (void*)&RaQpDestroyWithoutCQ},
        {"RaSendWrVerbs", (void*)&RaSendWrVerbs},
        {"RaRecvWrVerbs", (void*)&RaRecvWrVerbs},
        {"RaPollTypicalCq", (void*)&RaPollTypicalCq},
        {"RaTypicalCqCreate", (void*)&RaTypicalCqCreate},
        {"RaTypicalCqDestroy", (void*)&RaTypicalCqDestroy},
        {"RaQpCreateWithCQWithAttrs", (void*)&RaQpCreateWithCQWithAttrs},
    };

    std::map<std::string, void*> dlTdtFuntionPtrMap = {
        {"TsdOpen", (void*)&TsdOpen},
        {"TsdProcessOpen", (void*)&TsdProcessOpen},
        {"ProcessCloseSubProcList", (void*)&ProcessCloseSubProcList},
        {"TsdCapabilityGet", (void*)&TsdCapabilityGet},
    };

    std::map<std::string, void*> dlHalFuntionPtrMap = {
        {"halEschedSubmitEvent", (void*)&halEschedSubmitEvent},
        {"halEschedAttachDevice", (void*)&halEschedAttachDevice},
        {"halEschedDettachDevice", (void*)&halEschedDettachDevice},
        {"halEschedCreateGrp", (void*)&halEschedCreateGrp},
        {"halEschedCreateGrpEx", (void*)&halEschedCreateGrpEx},
        {"halEschedSubscribeEvent", (void*)&halEschedSubscribeEvent},
        {"halEschedWaitEvent", (void*)&halEschedWaitEvent},
        {"halEschedRegisterAckFunc", (void*)&halEschedRegisterAckFunc},
        {"halGetAPIVersion", (void*)&halGetAPIVersion},
        {"drvMemcpy", (void*)&drvMemcpy},
        {"drvDeviceGetBareTgid", (void*)&drvDeviceGetBareTgid},
        {"halGrpQuery", (void*)&halGrpQuery},
        {"drvGetDevNum", (void*)&drvGetDevNum},
        {"halGetDeviceInfo", (void*)&halGetDeviceInfo},
        {"halEschedQueryInfo", (void*)&halEschedQueryInfo},
        {"drvGetPlatformInfo", (void*)&drvGetPlatformInfo},
        {"halGetChipInfo", (void*)&halGetChipInfo},
        {"halBindCgroup", (void*)&halBindCgroup},
        {"drvDeviceGetPhyIdByIndex", (void*)&drvDeviceGetPhyIdByIndex},
        {"halHostRegister", (void*)&halHostRegister},
        {"halHostUnregister", (void*)&halHostUnregister},
        {"halHostUnregisterEx", (void*)&halHostUnregisterEx},
        {"halMemCtl", (void*)&halMemCtl},
        {"halSensorNodeRegister", (void*)&halSensorNodeRegister},
        {"halSensorNodeUnregister", (void*)&halSensorNodeUnregister},
        {"halSensorNodeUpdateState", (void*)&halSensorNodeUpdateState},
        {"halSdmaCopy", (void*)&halSdmaCopy},
        {"drvQueryProcessHostPid", (void*)&drvQueryProcessHostPid},
        {"drvGetDevIDByLocalDevID", (void*)&drvGetDevIDByLocalDevID},
    };
    std::map<std::string, void*> dlIbvFuntionPtrMap
        = {{"ibv_get_cq_event", (void*)&ibv_get_cq_event_stub},
           {"ibv_ack_cq_events", (void*)&ibv_ack_cq_events_stub},
           {"ibv_query_qp", (void*)&ibv_query_qp_stub}};
    std::map<std::string, void*> dlHddsFuntionPtrMap
        = {{"QueryPartitionMapPsId", (void*)&QueryPartitionMapPsId},
           {"InitPartitionMap", (void*)&InitPartitionMap},
           {"GetBatchPsIds", (void*)&GetBatchPsIds}};

    std::map<std::string, void*> dlProfFuntionPtrMap
        = {{"MsprofRegisterCallback", (void*)&MsprofRegisterCallback},
           {"MsprofRegTypeInfo", (void*)&MsprofRegTypeInfo},
           {"MsprofReportApi", (void*)&MsprofReportApi},
           {"MsprofReportCompactInfo", (void*)&MsprofReportCompactInfo},
           {"MsprofReportAdditionalInfo", (void*)&MsprofReportAdditionalInfo},
           {"MsprofStr2Id", (void*)&MsprofStr2Id},
           {"MsprofSysCycleTime", (void*)&MsprofSysCycleTime}};

    std::map<std::string, void*> dlAtraceFuntionPtrMap
        = {{"AtraceDestroy", (void*)&AtraceDestroy},
           {"AtraceSubmit", (void*)&AtraceSubmit},
           {"AtraceCreateWithAttr", (void*)&AtraceCreateWithAttr},
           {"AtraceSetGlobalAttr", (void*)&AtraceSetGlobalAttr},
           {"AtraceSave", (void*)&AtraceSave}};

    std::map<std::string, void*> dlUtraceFuntionPtrMap
        = {{"UtraceDestroy", (void*)&UtraceDestroy},
           {"UtraceSubmit", (void*)&UtraceSubmit},
           {"UtraceCreateWithAttr", (void*)&UtraceCreateWithAttr},
           {"UtraceSetGlobalAttr", (void*)&UtraceSetGlobalAttr},
           {"UtraceSave", (void*)&UtraceSave}};

    std::map<std::string, void*> dlrdmaFuntionPtrMap = {
        {"ibv_ext_post_send", (void*)&stub_ibv_ext_post_send}, {"ibv_exp_post_send", (void*)&stub_ibv_exp_post_send}};

    std::map<std::string, void*> dlurmaFuntionPtrMap
        = {{"urma_post_jetty_send_wr", (void*)&stub_urma_post_jetty_send_wr},
           {"urma_poll_jfc", (void*)&stub_urma_poll_jfc}};

    std::map<std::string, void*> dlAclFuntionPtrMap;

    static int dlRaHandle;
    static int dlTdtHandle;
    static int dlHalHandle;
    static int dlIbvHandle;
    static int dlHddsHandle;
    static int dlProfHandle;
    static int dlAtraceHandle;
    static int dlUtraceHandle;
    static int dlHnsRdmav17Handle;
    static int dlHnsRdmav25Handle;
    static int dlHrn0Rdmav17Handle;
    static int dlUrmaHandle;
    static int dlAclHandle;
    void* __HcclDlopenSub(const char* libName, int mode)
    {
        HCCL_INFO("run dlopen(const char*[%s], int[%d])", libName, mode);
        std::string LibName(libName);
        if (LibName == "libra.so") {
            return &dlRaHandle;
        } else if (LibName == "libtsdclient.so") {
            return &dlTdtHandle;
        } else if (LibName == "libascend_hal.so") {
            return &dlHalHandle;
        } else if (LibName == "libibverbs.so") {
            return &dlIbvHandle;
        } else if (LibName == "libhdds_base.so") {
            return &dlHddsHandle;
        } else if (LibName == "libprofapi.so") {
            return &dlProfHandle;
        } else if (LibName == "libascend_trace.so") {
            return &dlAtraceHandle;
        } else if (LibName == "libutrace.so") {
            return &dlUtraceHandle;
        } else if (LibName == "libhns-rdmav17.so") {
            return &dlHnsRdmav17Handle;
        } else if (LibName == "libhns-rdmav25.so") {
            return &dlHnsRdmav25Handle;
        } else if (LibName == "libhrn0-rdmav17.so") {
            return &dlHrn0Rdmav17Handle;
        } else if (LibName == "liburma.so.0") {
            return &dlUrmaHandle;
        } else if (LibName == "libascendcl.so") {
            return &dlAclHandle;
        }

        return nullptr;
    }

    HcclResult __hrtOpenNetServiceSub(rtNetServiceOpenArgs* openArgs) { return HCCL_SUCCESS; }

    HcclResult __hrtCloseNetServiceSub() { return HCCL_SUCCESS; }

    int __HcclDlcloseSub(void* handle)
    {
        HCCL_INFO("run dlclose");
        handle = nullptr;
        return 0;
    }

    void* __HcclDlsymSub(void* handle, const char* funcName)
    {
        std::string tempName(funcName);
        if (handle == &dlRaHandle) {
            return dlRaFuntionPtrMap[tempName];
        } else if (handle == &dlTdtHandle) {
            return dlTdtFuntionPtrMap[tempName];
        } else if (handle == &dlHalHandle) {
            return dlHalFuntionPtrMap[tempName];
        } else if (handle == &dlIbvHandle) {
            return dlIbvFuntionPtrMap[tempName];
        } else if (handle == &dlHddsHandle) {
            return dlHddsFuntionPtrMap[tempName];
        } else if (handle == &dlProfHandle) {
            return dlProfFuntionPtrMap[tempName];
        } else if (handle == &dlAtraceHandle) {
            return dlAtraceFuntionPtrMap[tempName];
        } else if (handle == &dlUtraceHandle) {
            return dlUtraceFuntionPtrMap[tempName];
        } else if (handle == &dlHnsRdmav17Handle) {
            return dlrdmaFuntionPtrMap[tempName];
        } else if (handle == &dlHnsRdmav25Handle) {
            return dlrdmaFuntionPtrMap[tempName];
        } else if (handle == &dlHrn0Rdmav17Handle) {
            return dlrdmaFuntionPtrMap[tempName];
        } else if (handle == &dlUrmaHandle) {
            return dlurmaFuntionPtrMap[tempName];
        } else if (handle == &dlAclHandle) {
            return dlAclFuntionPtrMap[tempName];
        }
        return nullptr;
    }
    strong_alias(__HcclDlopenSub, HcclDlopen);
    strong_alias(__HcclDlcloseSub, HcclDlclose);
    strong_alias(__HcclDlsymSub, HcclDlsym);
    strong_alias(__hrtOpenNetServiceSub, hrtOpenNetService);
    strong_alias(__hrtCloseNetServiceSub, hrtCloseNetService);
} // namespace hccl

HcclResult __hrtGetDeviceTypeStub(DevType& devType)
{
#ifndef HCCD
    std::string socName;
    CHK_RET(hrtGetSocVer(socName));
    auto iter = SOC_VER_CONVERT.find(socName);
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), socName.c_str());
        return HCCL_E_RUNTIME;
    }
    devType = iter->second;
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDeviceType]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDeviceTypeStub, hrtGetDeviceTypeStub);

HcclResult __hrtGetDeviceStub(s32* deviceLogicId)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(deviceLogicId);

    DevType deviceType;
    CHK_RET(hrtGetDeviceTypeStub(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        *deviceLogicId = 0;
        return HCCL_SUCCESS;
    }
    rtError_t ret = 0;
    ret = aclrtGetDevice(deviceLogicId);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_WARNING(
            "[Get][Device]errNo[0x%016llx] rtGet device fail, "
            "please make sure that device is set. return[%d], para:deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *deviceLogicId),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevice]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDeviceStub, hrtGetDevice);

HcclResult __hrtGetDevicePhyIdByIndexStub(u32 deviceLogicId, u32& devicePhyId, bool isRefresh)
{
#ifndef HCCD
    DevType deviceType;
    CHK_RET(hrtGetDeviceTypeStub(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        devicePhyId = 0;
        return HCCL_SUCCESS;
    }

    s32 logicDevId = static_cast<s32>(deviceLogicId);
    s32 phyDevId;
    aclError ret = aclrtGetPhyDevIdByLogicDevId(logicDevId, &phyDevId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DevicePhyId]errNo[0x%016llx] rtGet device PhyId by index failed, return[%d], "
            "para: devIndex[%d], phyId[%d]",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, logicDevId, phyDevId);
        return HCCL_E_RUNTIME;
    }
    devicePhyId = static_cast<u32>(phyDevId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevicePhyIdByIndex]The helper does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
strong_alias(__hrtGetDevicePhyIdByIndexStub, hrtGetDevicePhyIdByIndex);

enum callHccl {
    HcclOpKernelAddCounter,
    HcclOpKernelClearCounter,
    HcclOpKernelLogStr,
    HcclOpKernelCmpInt8,
    HcclOpKernelCmpInt32,
    HcclOpKernelCmpFp16,
    HcclOpKernelCmpFp32,
    HcclOpKernelLogInt8,
    HcclOpKernelLogInt32,
    HcclOpKernelLogFp16,
    HcclOpKernelLogFp32,
    HcclOpKernelLogVariable,
};

#ifdef __cplusplus
} // extern "C"
#endif
HcclResult SetDevicePlaneId(u32 devicePhyId, u32 planeId)
{
    auto it = std::find(DevicePlaneList.begin(), DevicePlaneList.end(), devicePhyId);
    if (it == DevicePlaneList.end()) {
        DevicePlaneInfo_t tmpPlaneInfo;
        tmpPlaneInfo.devicePhyId = devicePhyId;
        tmpPlaneInfo.planeId = planeId;
        DevicePlaneList.push_back(tmpPlaneInfo);
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}
HcclResult GetDevicePlaneId(u32 devicePhyId, u32& planeId)
{
    auto it = std::find(DevicePlaneList.begin(), DevicePlaneList.end(), devicePhyId);
    if (it != DevicePlaneList.end()) {
        planeId = it->planeId;
        return HCCL_SUCCESS;
    }
    return HCCL_E_PARA;
}
void ClearDevicePlaneId()
{
    DevicePlaneList.clear();
    return;
}
void FailureInjectStub(u32 deviceId, tasktype_e taskType)
{
    FailureDeviceId = deviceId;
    FailureTaskType = taskType;
}
void FailureClear()
{
    FailureDeviceId = 0xFFFFFFFF;
    FailureTaskType = TASK_TYPE_RESERVED;
}

bool g_isUseRealPortAndName = false;
void UseRealPortAndName(bool isUse) { g_isUseRealPortAndName = isUse; }
bool IsUseRealPortAndName() { return g_isUseRealPortAndName; }

aclError aclrtCreateContext(aclrtContext* ctx, int32_t deviceId)
{
    static int rtCtx = 0;
    *ctx = &rtCtx;
    return ACL_SUCCESS;
}

aclError aclrtDestroyContext(aclrtContext ctx) { return ACL_SUCCESS; }

uint32_t GetCPUNum() { return 1; }

rtError_t rtStreamGetCqid(const rtStream_t stm, uint32_t* cqId, uint32_t* logicCqId)
{
    static uint32_t i = 0U;
    *logicCqId = i++;
    return RT_ERROR_NONE;
}

void ParallelFor(int64_t total, int64_t perUnitSize, const std::function<void(int64_t, int64_t)>& work) { return; }

aclError aclrtCtxGetFloatOverflowAddr(void** overflowAddr)
{
    *overflowAddr = (void*)0x1;
    return ACL_SUCCESS;
}
rtError_t rtGetDevArgsAddr(rtStream_t stm, rtArgsEx_t* argsInfo, void** devArgsAddr, void** argsHandle)
{
    return RT_ERROR_NONE;
}
HcclIpAddress invalidIp;

HcclResult GetSocketRole(std::vector<u32>& userRanks, u32 srcRank, u32 destRank, HcclSocketRole& role)
{
    /* 获取两个userrank在列表中的位置，用特定的规则决定这对连接中的servre、client 角色 */
    u32 srcPos = INVALID_UINT;
    for (u32 loop = 0; loop < userRanks.size(); loop++) {
        if (userRanks[loop] == srcRank) {
            srcPos = loop;
            break;
        }
    }

    if (srcPos == INVALID_UINT) {
        HCCL_ERROR(
            "[GetSocketRole]errNo[0x%016llx] srcRank is not found in user rank list",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), srcRank);
        return HCCL_E_INTERNAL;
    }

    u32 destPos = INVALID_UINT;
    for (u32 loop = 0; loop < userRanks.size(); loop++) {
        if (userRanks[loop] == destRank) {
            destPos = loop;
            break;
        }
    }
    if (destPos == INVALID_UINT) {
        HCCL_ERROR(
            "[GetSocketRole]errNo[0x%016llx] destPos is not found in user rank list",
            HCCL_ERROR_CODE(HCCL_E_TCP_TRANSFER), destPos);
        return HCCL_E_INTERNAL;
    }

    u32 gap = 0;
    if (destPos > srcPos) {
        /* 如果destPos > srcPos，判断gap是奇数还是偶数，如果是奇数则做server, 反之作为client */
        gap = destPos - srcPos;
        /* 数字1和2用于判断奇偶 */
        role = (((gap % 2) == 1) ? HcclSocketRole::SOCKET_ROLE_SERVER : HcclSocketRole::SOCKET_ROLE_CLIENT);
    } else if (destPos < srcPos) {
        /* 如果destPos < srcPos，角色分配与大于时相反 */
        gap = srcPos - destPos;
        /* 数字1和2用于判断奇偶 */
        role = (((gap % 2) == 1) ? HcclSocketRole::SOCKET_ROLE_CLIENT : HcclSocketRole::SOCKET_ROLE_SERVER);
    } else {
        /* 相同user rank不建链 */
        role = HcclSocketRole::SOCKET_ROLE_RESERVED;
    }
    return HCCL_SUCCESS;
}

HcclResult GetIntraRankIPInfo(
    u32 localUserRank, std::vector<u32> userRanks, std::vector<s32> deviceIds, HcclIpAddress& localIPs,
    std::map<u32, HcclRankLinkInfo>& dstServerMap, std::map<u32, HcclRankLinkInfo>& dstClientMap)
{
    auto devIter = deviceIds.begin();
    auto rankIter = userRanks.begin();
    for (; devIter != deviceIds.end(); devIter++, rankIter++) {
        HcclSocketRole localRole;
        HcclResult ret = GetSocketRole(userRanks, localUserRank, *rankIter, localRole);
        if (ret != HCCL_SUCCESS) {
            return ret;
        }
        // Rank devicePhyId 作为地址
        HcclRankLinkInfo linkInfo{};
        linkInfo.userRank = *rankIter;
        linkInfo.devicePhyId = *devIter;
        HcclIpAddress ipAddress(*devIter);
        linkInfo.ip = ipAddress;
        linkInfo.socketsPerLink = 1;

        if (localRole == HcclSocketRole::SOCKET_ROLE_CLIENT) {
            dstServerMap.insert(std::make_pair(*rankIter, linkInfo));
        } else if (localRole == HcclSocketRole::SOCKET_ROLE_SERVER) {
            dstClientMap.insert(std::make_pair(*rankIter, linkInfo));
        } else {
            // 当前上层逻辑，保证 userRank_(当前 Rank) 在 userRanks 中
            localIPs = linkInfo.ip;
        }
    }
    return HCCL_SUCCESS;
}

HcclResult ConstructNetDevCtx(
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, NICDeployment nicDeploy, s32 deviceLogicId, u32 devicePhyId,
    NicType nicType, HcclIpAddress localIp)
{
    CHK_RET(HcclNetInit(nicDeploy, devicePhyId, deviceLogicId, false));

    HcclNetDevCtx portCtx;
    CHK_RET(HcclNetOpenDev(&portCtx, nicType, devicePhyId, deviceLogicId, localIp));

    netDevCtxMap.insert(std::make_pair(localIp, portCtx));

    return HCCL_SUCCESS;
}

void DeConstructNetDevCtx(
    std::map<HcclIpAddress, HcclNetDevCtx>& netDevCtxMap, NICDeployment nicDeploy, s32 deviceLogicId, u32 devicePhyId)
{
    for (auto& iter : netDevCtxMap) {
        HcclNetCloseDev(iter.second);
    }

    HcclNetDeInit(nicDeploy, devicePhyId, deviceLogicId);
}

HcclResult CreateIntraExchanger(
    const std::string& commTag, HcclNetDevCtx portCtx, s32 deviceLogicId, u32 devicePhyId, u32 localUserRank,
    const u32 userRankSize, std::vector<s32> deviceIds, std::vector<u32> userRanks, bool isSupportReuse,
    IntraExchanger& exchanger)
{
    HcclResult ret;
    std::shared_ptr<HcclSocketManager> socketManager = nullptr;
    socketManager.reset(new (std::nothrow) HcclSocketManager(
        NICDeployment::NIC_DEPLOYMENT_DEVICE, deviceLogicId, devicePhyId, localUserRank));
    CHK_PTR_NULL(socketManager);

    HcclIpAddress localIPs;
    std::map<u32, HcclRankLinkInfo> dstServerMap;
    std::map<u32, HcclRankLinkInfo> dstClientMap;
    CHK_RET(GetIntraRankIPInfo(localUserRank, userRanks, deviceIds, localIPs, dstServerMap, dstClientMap));

    CHK_RET(socketManager->ServerInit(portCtx, 16666));

    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> serverSocketsMap;
    std::map<u32, std::vector<std::shared_ptr<HcclSocket>>> clientSocketsMap;
    ret = socketManager->CreateSockets(
        commTag, false, portCtx, dstServerMap, dstClientMap, serverSocketsMap, clientSocketsMap, isSupportReuse);
    CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("sync create connections Failed, ret[%u]", ret), ret);

    exchanger.socketsMap.insert(serverSocketsMap.begin(), serverSocketsMap.end());
    exchanger.socketsMap.insert(clientSocketsMap.begin(), clientSocketsMap.end());
    exchanger.socketManager = socketManager;

    socketManager->ServerDeInit(portCtx, 16666);

    return HCCL_SUCCESS;
}

/**
 * @ingroup dvrt_mem
 * @brief set the attribute of shared memory
 * @param [in] name   identification name
 * @param [in] type   shared memory mapping type
 * @param [in] attr   shared memory attribute
 * @return RT_ERROR_NONE for ok
 * @return RT_ERROR_INVALID_VALUE for error input
 * @return RT_ERROR_DRV_ERR for driver error
 */
aclError aclrtIpcMemSetAttr(const char* key, aclrtIpcMemAttrType type, uint64_t attr) { return ACL_SUCCESS; }

aclError aclrtBinaryUnLoad(aclrtBinHandle binHandle) { return ACL_SUCCESS; }

aclError aclrtKernelArgsAppend(aclrtArgsHandle argsHandle, void* param, size_t paramSize, aclrtParamHandle* paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsAppendPlaceHolder(aclrtArgsHandle argsHandle, aclrtParamHandle* paramHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsGetPlaceHolderBuffer(
    aclrtArgsHandle argsHandle, aclrtParamHandle paramHandle, size_t dataSize, void** bufferAddr)
{
    return ACL_SUCCESS;
}

aclError aclrtKernelArgsFinalize(aclrtArgsHandle argsHandle) { return ACL_SUCCESS; }

aclError aclrtKernelArgsInit(aclrtFuncHandle funcHandle, aclrtArgsHandle* argsHandle) { return ACL_SUCCESS; }

aclError aclrtBinaryLoadFromFile(const char* binPath, aclrtBinaryLoadOptions* options, aclrtBinHandle* binHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithConfig(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg,
    aclrtArgsHandle argsHandle, void* reserve)
{
    return ACL_SUCCESS;
}

aclError aclrtBinaryGetFunction(const aclrtBinHandle binHandle, const char* kernelName, aclrtFuncHandle* funcHandle)
{
    return ACL_SUCCESS;
}

aclError aclrtGetDeviceResLimit(int32_t deviceId, aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtGetResInCurrentThread(aclrtDevResLimitType type, uint32_t* value)
{
    *value = 48;
    return ACL_SUCCESS;
}

aclError aclrtLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    return ACL_SUCCESS;
}

aclError aclrtSnapShotCallbackRegister(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback, void* args)
{
    return ACL_SUCCESS;
}

aclError aclrtSnapShotCallbackUnregister(aclrtSnapShotStage stage, aclrtSnapShotCallBack callback)
{
    return ACL_SUCCESS;
}

aclError aclmdlRIBindStream(aclmdlRI modelRI, aclrtStream stream, uint32_t flag) { return ACL_SUCCESS; }

aclError aclmdlRIUnbindStream(aclmdlRI modelRI, aclrtStream stream) { return ACL_SUCCESS; }

const char* aclrtGetSocName()
{
    if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910B)) {
        return "Ascend910B1";
    } else if (chip_type_stub[0] == static_cast<s32>(DevType::DEV_TYPE_910_93)) {
        return "Ascend910_9391";
    } else if (gBoardId == 0x0000) {
        return "Ascend910";
    } else if (gBoardId == 0x2000) { // 临时定义的 board id
        return "Ascend310P3";
    }
    return "Ascend910";
}

extern "C" ACL_FUNC_VISIBILITY aclError aclsysGetVersionStr(char* pkgNname, char* versionStr)
{
    sal_memcpy(versionStr, sizeof("8.5.0"), "8.5.0", sizeof("8.5.0"));
    return ACL_SUCCESS;
}

extern "C" ACL_FUNC_VISIBILITY aclError aclsysGetVersionNum(char* pkgNname, int32_t* versionNum)
{
    *versionNum = 80500;
    return ACL_SUCCESS;
}

rtError_t rtModelGetId(rtModel_t mdl, uint32_t* modelId) { return RT_ERROR_NONE; }

rtError_t rtGetPhyDeviceInfo(uint32_t phyId, int32_t moduleType, int32_t infoType, int64_t* val)
{
    return RT_ERROR_NONE;
}

rtError_t rtGetPairDevicesInfo(uint32_t devId, uint32_t otherDevId, int32_t infoType, int64_t* val)
{
    return RT_ERROR_NONE;
}

rtError_t rtEnableP2P(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t flag) { return RT_ERROR_NONE; }

rtError_t rtDisableP2P(uint32_t devIdDes, uint32_t phyIdSrc) { return RT_ERROR_NONE; }

rtError_t rtGetP2PStatus(uint32_t devIdDes, uint32_t phyIdSrc, uint32_t* status)
{
    *status = 1;
    return RT_ERROR_NONE;
}

rtError_t aclrtMemExportToShareableHandleV2(
    aclrtDrvMemHandle handle, uint64_t flags, aclrtMemSharedHandleType shareType, void* shareableHandle)
{
    return RT_ERROR_NONE;
}

rtError_t aclrtMemSetPidToShareableHandleV2(
    void* shareableHandle, aclrtMemSharedHandleType shareType, int32_t* pid, size_t pidNum)
{
    return RT_ERROR_NONE;
}

rtError_t aclrtMemImportFromShareableHandleV2(
    void* shareableHandle, aclrtMemSharedHandleType shareType, uint64_t flags, aclrtDrvMemHandle* handle)
{
    return RT_ERROR_NONE;
}

aclError aclrtMemGetAddressRange(void* ptr, void** baseUserVa, size_t* baseVaSize)
{
    (void)ptr;
    (void)baseUserVa;
    (void)baseVaSize;
    return ACL_SUCCESS;
}

aclError aclrtGetMemInfo(aclrtMemAttr attr, size_t* free, size_t* total)
{
    *total = 64 * GIGABYTE_TO_BYTE;
    *free = 50 * GIGABYTE_TO_BYTE;
    return ACL_SUCCESS;
}

aclError aclmdlRIDestroyRegisterCallback(aclmdlRI modelRI, aclrtCallback func, void* ptr) { return ACL_SUCCESS; }

namespace hccl {
HcclResult LoadBinaryFromFile(
    const char* binPath, aclrtBinaryLoadOptionType optionType, uint32_t cpuKernelMode, aclrtBinHandle& binHandle)
{
    return HCCL_SUCCESS;
}

HcclResult GetCustomKernelFilePath(std::string& binaryPath)
{
    binaryPath = "./";
    return HCCL_SUCCESS;
}

} // namespace hccl

aclError aclrtCacheLastTaskExtendInfo(const char* tag, size_t tagLen) { return ACL_SUCCESS; }

namespace Hccl {
int HcclCheckLogLevel(int logLevel) { return 1; }

void CallDlogMemError(int level, std::string file, int line)
{
    (void)level;
    (void)file;
    (void)line;
}

void CallDlogPrintError(int level, std::string file, int line)
{
    (void)level;
    (void)file;
    (void)line;
}

void CallDlog(int level, int sysCallBack, const char* buffer, std::string file, int line)
{
    (void)level;
    (void)sysCallBack;
    (void)buffer;
    (void)file;
    (void)line;
}
} // namespace Hccl

HcclResult HcommProfilingReportDeviceOp(const char* groupname)
{
    (void)groupname;
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelStartTask(uint64_t thread, const char* groupname)
{
    (void)thread;
    (void)groupname;
    return HCCL_SUCCESS;
}

HcclResult HcommProfilingReportKernelEndTask(uint64_t thread, const char* groupname)
{
    (void)thread;
    (void)groupname;
    return HCCL_SUCCESS;
}
