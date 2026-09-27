/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

// 日志染色: 模块 tag (须在 include sim_log.h 之前)
#define HCCL_VM_MODULE "RTS_STUB"

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <unistd.h>
#include <vector>

#include "adapter_rts_common.h"
#include "ascend_hal.h"
#include "atrace_pub.h"
#include "sim_log.h"
#include "hccp.h"
#include "hccp_async.h"
#include "hccp_ping.h"
#include "hccp_tlv.h"
#include "rts_device.h"

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

#if !defined(weak_alias)
#define weak_alias(name, aliasname) _weak_alias(name, aliasname)
#define _weak_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((weak, alias(#name)));
#endif

#if !defined(strong_alias)
#define strong_alias(name, aliasname) _strong_alias(name, aliasname)
#define _strong_alias(name, aliasname) extern __typeof(name) aliasname __attribute__((alias(#name)));
#endif

int RaIsFirstUsed(int insId)
{
    (void)insId;
    HCCL_VM_WARN("is empty");
    return 0;
}

int RaIsLastUsed(int insId)
{
    (void)insId;
    HCCL_VM_WARN("is empty");
    return 0;
}

int ibv_get_cq_event_stub(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context)
{
    (void)cq;
    (void)cq_context;
    HCCL_VM_WARN("is empty");
    if (!channel) {
        HCCL_VM_ERROR("channel is null");
        return -1;
    }
    return 0;
}

void ibv_ack_cq_events_stub(struct ibv_cq* cq, unsigned int nevents)
{
    (void)cq;
    (void)nevents;
    HCCL_VM_WARN("is empty");
    return;
}

void ibv_query_qp_stub(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)
{
    (void)qp;
    (void)attr;
    (void)attr_mask;
    (void)init_attr;
    HCCL_VM_WARN("is empty");
    return;
}

int32_t UtraceCreateWithAttr(int32_t tracerType, const char* objName, const TraceAttr* attr)
{
    (void)(tracerType);
    (void)(objName);
    (void)(attr);
    HCCL_VM_WARN("is empty");
    return 0;
}

HcclResult UtraceSubmit(int32_t handle, const void* buffer, uint32_t bufSize)
{
    (void)(handle);
    (void)(buffer);
    (void)(bufSize);
    HCCL_VM_WARN("is empty");
    return HCCL_SUCCESS;
}

int32_t UtraceSetGlobalAttr(const TraceGlobalAttr* attr)
{
    (void)(attr);
    HCCL_VM_WARN("is empty");
    return 0;
}

int32_t UtraceSave(TracerType tracerType, bool syncFlag)
{
    (void)(tracerType);
    (void)(syncFlag);
    HCCL_VM_WARN("is empty");
    return 0;
}

void UtraceDestroy(int32_t handle)
{
    (void)(handle);
    HCCL_VM_WARN("is empty");
    return;
}

int ibv_exp_post_send_stub(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp)
{
    (void)qp;
    (void)wr;
    (void)bad_wr;
    (void)ext_attr;
    (void)ext_resp;
    HCCL_VM_WARN("is empty");
    return 0;
}

int ibv_ext_post_send_stub(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp)
{
    (void)qp;
    (void)wr;
    (void)bad_wr;
    (void)ext_attr;
    (void)ext_resp;
    HCCL_VM_WARN("is empty");
    return 0;
}

HcclResult hrtTsdCapabilityGet(uint32_t deviceLogicId, int32_t type, uint64_t ptr)
{
    HCCL_VM_INFO("deviceLogicId:[{}] type:[{}]], ptr:[{}]", deviceLogicId, type, ptr);
    return (HcclResult)0;
}

HcclResult
hrtGetDeviceInfo(u32 deviceId, HcclRtDeviceModuleType hcclModuleType, HcclRtDeviceInfoType hcclInfoType, s64& val)
{
    HCCL_VM_INFO(
        "deviceId:[{}] hcclModuleType:[{}], HcclRtDeviceInfoType:[{}]", deviceId, (int)hcclModuleType,
        (int)hcclInfoType);
    return (HcclResult)0;
}

namespace hccl {
    void* __HcclDlopenSub(const char* libName, int mode)
    {
        void* handle = dlopen(libName, mode);
        if (handle == nullptr) {
            HCCL_VM_ERROR("dlopen {} failed: {}", libName, dlerror());
            return nullptr;
        }
        return handle;
    }

    void* __HcclDlsymSub(void* handle, const char* funcName)
    {
        void* addr = dlsym(RTLD_DEFAULT, funcName);
        if (addr == nullptr) {
            HCCL_VM_WARN("rtld default not find {}: [{}]", funcName, dlerror());
            addr = dlsym(handle, funcName);
            if (addr == nullptr) {
                HCCL_VM_WARN("not find {}: [{}]", funcName, dlerror());
                return nullptr;
            }
            return addr;
        }
        return addr;
    }

    HcclResult __hrtOpenNetServiceSub(rtNetServiceOpenArgs* openArgs)
    {
        (void)openArgs;
        return HCCL_SUCCESS;
    }

    HcclResult __hrtCloseNetServiceSub() { return HCCL_SUCCESS; }

    int __HcclDlcloseSub(void* handle)
    {
        if (handle == nullptr) {
            return 0;
        }
        return dlclose(handle);
    }

    strong_alias(__HcclDlopenSub, HcclDlopen);
    strong_alias(__HcclDlcloseSub, HcclDlclose);
    strong_alias(__HcclDlsymSub, HcclDlsym);
    strong_alias(__hrtOpenNetServiceSub, hrtOpenNetService);
    strong_alias(__hrtCloseNetServiceSub, hrtCloseNetService);
} // namespace hccl

#ifdef __cplusplus
}
#endif // __cplusplus
