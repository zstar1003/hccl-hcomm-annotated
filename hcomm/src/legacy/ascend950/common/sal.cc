/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sal.h"
#include <unistd.h>
#include "log.h"
#include <unistd.h>
#include <sys/time.h> /* 获取时间 */

namespace Hccl {

void SaluSleep(u32 usec)
{
    /* usleep()可能会因为进程收到信号(比如alarm)而提前返回EINTR, 后续优化  */
    s32 iRet = usleep(usec);
    if (iRet != 0) {
        HCCL_WARNING("Sleep: usleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
    }
}

void SalSleep(u32 sec)
{
    /* sleep()可能会因为进程收到信号(比如alarm)而提前返回EINTR, 后续优化  */
    s32 iRet = sleep(sec);
    if (iRet != 0) {
        HCCL_WARNING("Sleep: sleep failed[%d]: %s [%d]", iRet, strerror(errno), errno);
    }
}

std::string SalGetEnv(const char* name)
{
    if (name == nullptr || getenv(name) == nullptr) {
        return "EmptyString";
    }

    return getenv(name);
}

constexpr u32 INVALID_UINT = 0xFFFFFFFF;
// 字串符转换成无符号整型
HcclResult SalStrToULong(const std::string str, int base, u32& val)
{
    try {
        u64 tmp = std::stoull(str, nullptr, base);
        if (tmp > INVALID_UINT) {
            HCCL_ERROR("[Transform][StrToULong]stoul out of range, str[%s] base[%d] val[%llu]", str.c_str(), base, tmp);
            return HCCL_E_PARA;
        } else {
            val = static_cast<u32>(tmp);
        }
    } catch (std::invalid_argument&) {
        HCCL_ERROR("[Transform][StrToULong]stoull invalid argument, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    } catch (std::out_of_range&) {
        HCCL_ERROR("[Transform][StrToULong]stoull out of range, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    } catch (...) {
        HCCL_ERROR("[Transform][StrToULong]stoull catch error, str[%s] base[%d] val[%u]", str.c_str(), base, val);
        return HCCL_E_PARA;
    }
    return HCCL_SUCCESS;
}

u64 SalGetCurrentTimestamp()
{
    u64 timestamp;
    struct timeval tv {};
    int ret = gettimeofday(&tv, nullptr);
    if (ret != 0) {
        HCCL_ERROR("[Get][GetCurrentTimestamp]get timestamp fail, return[%d].", ret);
    }
    timestamp = tv.tv_sec * 1000000 + tv.tv_usec; // 1000000: 单位转换 秒 -> 微秒
    return timestamp;
}

u64 GetCurAicpuTimestamp()
{
    struct timespec timestamp;
    (void)clock_gettime(1, &timestamp);
    return static_cast<u64>((timestamp.tv_sec * 1000000000U) + (timestamp.tv_nsec));
}

// 返回当前线程ID
s32 SalGetTid() { return syscall(SYS_gettid); }

void SetThreadName(const std::string& threadStr)
{
    // 线程名应限制在15个字符内，防止被截断
    s32 sRet = pthread_setname_np(pthread_self(), threadStr.c_str());
    CHK_PRT_CONT(sRet != 0, HCCL_WARNING("err[%d] link[%s] nameSet failed.", sRet, threadStr.c_str()));
}

} // namespace Hccl
