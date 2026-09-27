/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_map>
#include <dlog_pub.h>
#include <securec.h>

#include "driver/ascend_hal.h"
#include "externalinput_pub.h"
#include "log.h"
#include "sal_pub.h"
#include "dlrts_function.h"
#include "adapter_error_manager.h"
#include "adapter_hal.h"
#include "device_capacity.h"
#include "config_plf_log.h"
#include "adapter_rts.h"

using namespace hccl;
using namespace std;

#define CHK_RT_RET(call)                              \
    do {                                              \
        s32 ret = call;                               \
        if (ret != 0) {                               \
            HCCL_ERROR("call trace: ret -> %d", ret); \
            return HCCL_E_RUNTIME;                    \
        }                                             \
    } while (0)

#define REPLACE_NOTIFY_WITH_EVENT(notify, event) \
    do {                                         \
        s32 result = 0;                          \
        if (result != 0) {                       \
            CHK_RT_RET(event);                   \
        } else {                                 \
            CHK_RT_RET(notify);                  \
        }                                        \
    } while (0)

inline s32 GetDeviceLogicalId()
{
    s32 deviceId = 0;
    if (hrtGetDevice(&deviceId) != HCCL_SUCCESS) {
        deviceId = -1;
    }
    return deviceId;
}

constexpr char RTS_SO_NAME[] = "libruntime.so";
constexpr u32 H2D_COPY_FLAG = 0; // hccl中都需要runtime做args copy默认都设置为0
DlRtsFunction<RTS_SO_NAME> g_dlRts;

constexpr char ACL_SO_NAME[] = "libascendcl.so";
DlRtsFunction<ACL_SO_NAME> g_dlAcl;

constexpr char ACL_RT_SO_NAME[] = "libacl_rt.so";
DlRtsFunction<ACL_RT_SO_NAME> g_dlAclrt;

namespace {
constexpr char RT_MEMCPY[] = "rtMemcpy";
constexpr char RT_POINTER_GET_ATTR[] = "rtPointerGetAttributes";
constexpr char RT_IPC_CLOSE_MEM[] = "rtIpcCloseMemory";
constexpr char ACL_RT_DESTORY_MEM_NAME[] = "aclrtIpcMemClose";
constexpr char ACL_RT_GET_LOGICID_BY_PHYID[] = "rtsGetLogicDevIdByPhyDevId";
constexpr char RT_RDMA_DB_SEND[] = "rtRDMADBSend";
constexpr char ACL_RT_MALLOC_WITH_CFG[] = "aclrtMallocWithCfg";
constexpr char ACL_GET_SOC_NAME[] = "aclrtGetSocName";
constexpr char RT_GET_PAIR_PHY_DEVICES_INFO[] = "rtGetPairPhyDevicesInfo";
constexpr char ACL_RT_CREATE_CONTEXT[] = "aclrtCreateContext";
constexpr char ACL_RT_DESTROY_CONTEXT[] = "aclrtDestroyContext";
constexpr char ACL_RT_GET_CURRENT_CONTEXT[] = "aclrtGetCurrentContext";
constexpr char ACL_RT_SET_CURRENT_CONTEXT[] = "aclrtSetCurrentContext";
constexpr char ACL_RT_SET_DEVICE[] = "aclrtSetDevice";
constexpr char ACL_RT_RESET_DEVICE[] = "aclrtResetDevice";
constexpr char ACL_RT_FREE[] = "aclrtFree";
constexpr char ACL_RT_FREE_HOST[] = "aclrtFreeHost";
constexpr char ACL_RT_MALLOC_HOST_WITH_CFG[] = "aclrtMallocHostWithCfg";
constexpr char ACL_RT_IPC_MEM_SET_IMPORT_PID[] = "aclrtIpcMemSetImportPid";
constexpr char ACL_RT_IPC_MEM_IMPORT_BY_KEY[] = "aclrtIpcMemImportByKey";
constexpr char ACL_RT_IPC_MEM_GET_EXPORT_KEY[] = "aclrtIpcMemGetExportKey";
constexpr char ACL_RT_MEMCPY[] = "aclrtMemcpy";
constexpr char ACL_RT_POINTER_GET_ATTRIBUTES[] = "aclrtPointerGetAttributes";
constexpr char ACL_RT_CACHE_LAST_TASK_EXTEND_INFO[] = "aclrtCacheLastTaskExtendInfo";
} // namespace
u32 g_stubDeviceId = 0;
static std::unordered_map<s32, s64> g_deviceChipIdMap; // 记录 devLogID 和 chipID 的关系，避免重复查询
bool g_workModeAicpu = false;                          // AI场景下aicpu工作模式
DevType g_localDeviceType = DevType::DEV_TYPE_COUNT;
s32 g_localDeviceLogicId = INVALID_INT;
aclrtFloatOverflowMode g_deviceSatMode = ACL_RT_OVERFLOW_MODE_UNDEF;
namespace {
static thread_local s32 g_deviceLogicId = INVALID_INT;
static thread_local u32 g_devicePhyId = INVALID_UINT;
static thread_local DevType g_deviceType = DevType::DEV_TYPE_COUNT;
} // namespace

// 根据soc name判断是否支持 hccl v2流程，使用全局变量减少重复调用aclrt接口
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
enum class HcclV2SupportStatus {
    UNKNOWN,      // 获取soc name失败
    SUPPORTED,    // 支持v2
    NOT_SUPPORTED // 不支持v2
};

static thread_local HcclV2SupportStatus g_socV2SupportStatus = HcclV2SupportStatus::UNKNOWN;
#endif

HcclResult hrtGetHcclV2Support([[maybe_unused]] bool* isSupport)
{
#if (!defined(HCCD)) && (!defined(CCL_KERNEL_AICPU))
    CHK_PTR_NULL(isSupport);
    if (LIKELY(g_socV2SupportStatus != HcclV2SupportStatus::UNKNOWN)) {
        *isSupport = g_socV2SupportStatus == HcclV2SupportStatus::SUPPORTED;
        return HcclResult::HCCL_SUCCESS;
    }

    const char* socNamePtr = aclrtGetSocName();
    CHK_PTR_NULL(socNamePtr);
    if (strstr(socNamePtr, "Ascend950") != nullptr) {
        g_socV2SupportStatus = HcclV2SupportStatus::SUPPORTED;
        *isSupport = true;
        return HcclResult::HCCL_SUCCESS;
    }
    if (strstr(socNamePtr, "Ascend960") != nullptr || strstr(socNamePtr, "ascend960") != nullptr) {
        g_socV2SupportStatus = HcclV2SupportStatus::SUPPORTED;
        *isSupport = true;
        return HcclResult::HCCL_SUCCESS;
    }

    g_socV2SupportStatus = HcclV2SupportStatus::NOT_SUPPORTED;
    *isSupport = false;
    return HcclResult::HCCL_SUCCESS;

#endif
    HCCL_WARNING("[%s] Does not support this interface.", __func__);
    return HCCL_E_NOT_SUPPORT;
}

#if T_DESC("Device管理", true)
HcclResult hrtThreadExchangeCaptureMode(aclmdlRICaptureMode* mode);

HcclResult hrtGetDeviceCount([[maybe_unused]] u32* count)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(count);

    aclError ret = aclrtGetDeviceCount(count);

    HCCL_DEBUG("Call rtGetDeviceCount, return value[%d], para: count[%u].", ret, *count);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[hrtGetDeviceCount]errNo[0x%016llx] aclGet device count fail, "
            "return[%d], para:count[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *count),
        HCCL_E_RUNTIME);

    if (*count == 0) {
        return HCCL_SUCCESS;
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        *count = 0;
        return HCCL_SUCCESS;
    }

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDeviceCount]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
};

/* 设定当前线程操作的目标设备编号 */
HcclResult hrtSetDevice(s32 deviceLogicId)
{
#ifndef HCCD
    aclError ret = aclrtSetDevice(deviceLogicId);
    HCCL_DEBUG(
        "Call aclrtSetDevice, return value[%d], para: device_id[%d], g_deviceLogicId = %d.", ret, deviceLogicId,
        g_deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][Device]errNo[0x%016llx] rtSet device fail, return[%d], "
            "para:deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
        HCCL_E_RUNTIME);
    g_deviceLogicId = INVALID_INT;
    return HCCL_SUCCESS;
#else
    static auto funcPtr = (aclError(*)(int32_t))g_dlAcl.Handle<ACL_RT_SET_DEVICE>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(deviceLogicId);
    HCCL_DEBUG(
        "Call aclrtSetDevice, return value[%d], para: device_id[%d], g_deviceLogicId = %d.", ret, deviceLogicId,
        g_deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][Device]errNo[0x%016llx] rtSet device fail, return[%d], "
            "para:deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#endif
}

/* 释放当前线程操作的目标设备编号,释放前必须先释放设备资源 */
HcclResult hrtResetDevice(s32 deviceLogicId)
{
#ifndef HCCD
    aclError ret = aclrtResetDevice(deviceLogicId);

    HCCL_DEBUG("Call aclrtResetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Reset][Device]errNo[0x%016llx] rtReset device fail, return[%d], "
            "para: deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    static auto funcPtr = (aclError(*)(int32_t))g_dlAcl.Handle<ACL_RT_RESET_DEVICE>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(deviceLogicId);

    HCCL_DEBUG("Call aclrtResetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Reset][Device]errNo[0x%016llx] rtReset device fail, return[%d], "
            "para: deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#endif
}

HcclResult stubSetDevice(u32 deviceLogicId)
{
    g_stubDeviceId = deviceLogicId;
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceRefresh([[maybe_unused]] s32* deviceLogicId)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(deviceLogicId);

    aclError ret = 0;
    ret = aclrtGetDevice(deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][DeviceRefresh]errNo[0x%016llx] rtGet device fail, "
            "please make sure that device is set. return[%d], para:deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *deviceLogicId),
        HCCL_E_RUNTIME);
    g_deviceLogicId = *deviceLogicId;
    HCCL_INFO("[hrtGetDeviceRefresh]deviceLogicId[%d]", *deviceLogicId);
    return HCCL_SUCCESS;
#else
    HCCL_WARNING("[hrtGetDeviceRefresh]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtSetlocalDevice([[maybe_unused]] s32 deviceLogicId)
{
#ifdef HCCD
    g_localDeviceLogicId = deviceLogicId;
#endif
    return HCCL_SUCCESS;
}

/* 查询当前线程目前操作的目标设备编号 */
#ifdef __cplusplus
extern "C" {
#endif
HcclResult __hrtGetDevice(s32* deviceLogicId)
{
    // 参数有效性检查
    CHK_PTR_NULL(deviceLogicId);
#ifndef HCCD

    if (LIKELY(g_deviceLogicId != INVALID_INT)) {
        *deviceLogicId = g_deviceLogicId;
        return HCCL_SUCCESS;
    }
    aclError ret = 0;
    ret = aclrtGetDevice(deviceLogicId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_WARNING(
            "[Get][Device]errNo[0x%016llx] rtGet device fail, "
            "please make sure that device is set. return[%d], para:deviceLogicId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *deviceLogicId),
        HCCL_E_RUNTIME);
    g_deviceLogicId = *deviceLogicId;
    HCCL_INFO("[hrtGetDevice]deviceLogicId[%d]", *deviceLogicId);
    return HCCL_SUCCESS;
#else
    if (g_workModeAicpu) {
        HCCL_DEBUG("[hrtGetDevice]Device logicId = %d.", g_localDeviceLogicId);
        *deviceLogicId = g_localDeviceLogicId;
        return HCCL_SUCCESS;
    }
    *deviceLogicId = 0;
    HCCL_WARNING("[hrtGetDevice] Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
weak_alias(__hrtGetDevice, hrtGetDevice);
#ifdef __cplusplus
} // extern "C"
#endif

HcclResult hrtCtxCreate(aclrtContext* createCtx, uint32_t flags, int32_t devId)
{
    CHK_PTR_NULL(createCtx);
#ifndef HCCD

    // acl 接口的 flag 默认是 rtCtxMode_t::RT_CTX_NORMAL_MODE
    aclError ret = aclrtCreateContext(createCtx, devId);
#else
    static auto funcPtr = (aclError(*)(aclrtContext*, int32_t))g_dlAcl.Handle<ACL_RT_CREATE_CONTEXT>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(createCtx, devId);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS && ret != ACL_ERROR_RT_CONTEXT_NULL,
        HCCL_ERROR(
            "[Get][Device]errNo[0x%016llx] aclrtCreateContext fail, return[%d], para:flags[%u], devId[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, flags, devId),
        HCCL_E_RUNTIME);
    HCCL_INFO("[hrtCtxCreate]deviceLogicId[%d]", devId);
    return HCCL_SUCCESS;
}

HcclResult hrtCtxDestroy(aclrtContext destroyCtx)
{
    CHK_PTR_NULL(destroyCtx);
#ifndef HCCD
    aclError ret = aclrtDestroyContext(destroyCtx);
#else
    static auto funcPtr = (aclError(*)(aclrtContext))g_dlAcl.Handle<ACL_RT_DESTROY_CONTEXT>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(destroyCtx);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS && ret != ACL_ERROR_RT_CONTEXT_NULL,
        HCCL_ERROR(
            "[Get][Device]errNo[0x%016llx] aclrtDestroyContext fail, return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult hrtCtxGetCurrent(HcclRtContext* ctx)
{
    CHK_PTR_NULL(ctx);
#ifndef HCCD
    aclError ret = aclrtGetCurrentContext(ctx);
#else
    static auto funcPtr = (aclError(*)(aclrtContext*))g_dlAcl.Handle<ACL_RT_GET_CURRENT_CONTEXT>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(ctx);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS && ret != ACL_ERROR_RT_CONTEXT_NULL,
        HCCL_ERROR(
            "[Get][Device]errNo[0x%016llx] aclrtGetCurrentContext fail, "
            "please make sure that DIE is set. return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    if (ret == ACL_ERROR_RT_CONTEXT_NULL) {
        HCCL_INFO("[Get][Device] curCtx is nullptr!");
        *ctx = nullptr;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtCtxSetCurrent(HcclRtContext ctx)
{
    CHK_PTR_NULL(ctx);
#ifndef HCCD
    aclError ret = aclrtSetCurrentContext(ctx);
#else
    static auto funcPtr = (aclError(*)(aclrtContext))g_dlAcl.Handle<ACL_RT_SET_CURRENT_CONTEXT>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(ctx);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][Device]errNo[0x%016llx] aclrtSetCurrentContext fail, "
            "please make sure that DIE is set. return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    HCCL_INFO("[hrtCtxSetCurrent] success");
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
extern "C" {
#endif
HcclResult __hrtGetDevicePhyIdByIndex(
    [[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] u32& devicePhyId, [[maybe_unused]] bool isRefresh)
{
#ifndef HCCD
    if (LIKELY(g_devicePhyId != INVALID_UINT) && (!isRefresh)) {
        devicePhyId = g_devicePhyId;
        return HCCL_SUCCESS;
    }

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        devicePhyId = 0;
        g_devicePhyId = 0;
        return HCCL_SUCCESS;
    }

    s32 logicDevId = static_cast<s32>(deviceLogicId);
    s32 phyDevId = 0;
    aclError ret = aclrtGetPhyDevIdByLogicDevId(logicDevId, &phyDevId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DevicePhyId]errNo[0x%016llx] rtGet device PhyId by index failed, return[%d], "
            "para: devIndex[%d], phyId[%d]",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, logicDevId, phyDevId);
        return HCCL_E_RUNTIME;
    }
    devicePhyId = static_cast<u32>(phyDevId);
    g_devicePhyId = devicePhyId;
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevicePhyIdByIndex]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
weak_alias(__hrtGetDevicePhyIdByIndex, hrtGetDevicePhyIdByIndex);
#ifdef __cplusplus
} // extern "C"
#endif

HcclResult hrtGetDeviceIndexByPhyId(u32 devicePhyId, u32& deviceLogicId)
{
    s32 phyDevId = static_cast<s32>(devicePhyId);
    s32 logicDevId = 0;
#ifndef HCCD
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        deviceLogicId = 0;
        return HCCL_SUCCESS;
    }

    aclError ret = aclrtGetLogicDevIdByPhyDevId(phyDevId, &logicDevId);
#else
    static auto funcPtr = (rtError_t(*)(int32_t, int32_t* const))g_dlAclrt.Handle<ACL_RT_GET_LOGICID_BY_PHYID>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(phyDevId, &logicDevId);
#endif
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[Get][DeviceIndex]errNo[0x%016llx] rtGet device logicid by PhyId failed, return[%d], "
            "para: phyId[%d], devIndex[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, phyDevId, logicDevId);
        return HCCL_E_RUNTIME;
    }
    deviceLogicId = static_cast<u32>(logicDevId);
    return HCCL_SUCCESS;
};

HcclResult hrtGetLogicDevIdByUserDevId(s32 userDevId, s32& logicDevId)
{
#ifndef HCCD
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        logicDevId = 0;
        return HCCL_SUCCESS;
    }

    aclError ret = aclrtGetLogicDevIdByUserDevId(userDevId, &logicDevId);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "aclrtGetLogicDevIdByUserDevId failed, return[%d], "
            "para: userDevId[%d], logicDevId[%d]",
            ret, userDevId, logicDevId);
        return HCCL_E_RUNTIME;
    }
#else
    logicDevId = userDevId;
    HCCL_WARNING("[aclrtGetLogicDevIdByUserDevId]Does not support this interface.");
#endif
    return HCCL_SUCCESS;
};

HcclResult hrtGetPhyDeviceInfo(
    [[maybe_unused]] u32 devicePhysicId, [[maybe_unused]] s32 moduleType, [[maybe_unused]] s32 infoType,
    [[maybe_unused]] s64& value)
{
#ifndef HCCD
    rtError_t rtRet = rtGetPhyDeviceInfo(devicePhysicId, moduleType, infoType, reinterpret_cast<int64_t*>(&value));
    HCCL_INFO(
        "[hrtGetPhyDeviceInfo] Call rtGetPhyDeviceInfo, ret[%d], para: devicePhysicId[%u], moduleType[%d], "
        "infoType[%d], value[%lld]",
        rtRet, devicePhysicId, moduleType, infoType, value);
    CHK_PRT_RET(
        rtRet != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][DeviceInfo]errNo[0x%016llx] rt get pair devices "
            "info failed, return[%d], para:devicePhysicId[%u], moduleType[%d], infoType[%d], value[%lld].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), rtRet, devicePhysicId, moduleType, infoType, value),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetPhyDeviceInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetSocVer([[maybe_unused]] std::string& socName)
{
#ifndef HCCD
    const char* socNamePtr = aclrtGetSocName();
    CHK_PRT_RET(
        (socNamePtr == nullptr),
        HCCL_ERROR("[Get][SocVer]errNo[0x%016llx] aclrtGet socName failed", HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
        HCCL_E_RUNTIME);

    socName = socNamePtr;
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetSocVer]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

// 获取芯片类型
#ifdef __cplusplus
extern "C" {
#endif
HcclResult hrtGetDeviceTypeBySocVersion(std::string& socVersion, DevType& devType)
{
    devType = DevType::DEV_TYPE_COUNT;
    if (socVersion == "Ascend310B1") {
        HCCL_WARNING("hrtGetDeviceTypeBySocVersion Ascend310B1 not support! please check usage");
    }

    if (socVersion.find("Ascend950") != std::string::npos) {
        devType = DevType::DEV_TYPE_950;
        return HCCL_SUCCESS;
    }

    if (socVersion.find("Ascend960") != std::string::npos || socVersion.find("ascend960") != std::string::npos) {
        devType = DevType::DEV_TYPE_960;
        return HCCL_SUCCESS;
    }

    auto iter = SOC_VER_CONVERT.find(socVersion);
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), socVersion.c_str());
        return HCCL_E_RUNTIME;
    }
    devType = iter->second;
    return HCCL_SUCCESS;
}
HcclResult hrtSetWorkModeAicpu([[maybe_unused]] bool workModeAicpu)
{
#ifdef HCCD
    g_workModeAicpu = workModeAicpu;
#endif
    HCCL_INFO("[Set][hrtSetWorkModeAicpu]work mode aicpu[%u]", g_workModeAicpu);
    return HCCL_SUCCESS;
}

HcclResult hrtSetlocalDeviceType([[maybe_unused]] DevType devType)
{
#ifdef HCCD
    g_localDeviceType = devType;
#endif
    return HCCL_SUCCESS;
}

HcclResult __hrtGetDeviceType(DevType& devType)
{
    if (LIKELY((g_deviceType != DevType::DEV_TYPE_COUNT))) {
        devType = g_deviceType;
        return HCCL_SUCCESS;
    }

    std::string socName;
#ifndef HCCD
    CHK_RET(hrtGetSocVer(socName));
#else
    if (g_workModeAicpu) {
        devType = g_localDeviceType;
        return HCCL_SUCCESS;
    }

    static auto funcPtr = (const char* (*)())g_dlAcl.Handle<ACL_GET_SOC_NAME>();
    CHK_PTR_NULL(funcPtr);
    const char* socNamePtr = funcPtr();
    CHK_PRT_RET(
        (socNamePtr == nullptr),
        HCCL_ERROR("[hrtGetDeviceType]errNo[0x%016llx] aclrtGet socName failed", HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
        HCCL_E_RUNTIME);
    socName = socNamePtr;
#endif
    //  根据芯片版本号获取芯片类型
    HCCL_DEBUG("[hrtGetDeviceType]socName = %s.", socName.c_str());
    if (socName.find("Ascend950") != std::string::npos) {
        devType = DevType::DEV_TYPE_950;
        g_deviceType = devType;
        HCCL_DEBUG("[hrtGetDeviceType]DeviceType = %d.", static_cast<s32>(g_deviceType));
        return HCCL_SUCCESS;
    }

    if (socName.find("Ascend960") != std::string::npos || socName.find("ascend960") != std::string::npos) {
        devType = DevType::DEV_TYPE_960;
        g_deviceType = devType;
        HCCL_DEBUG("[hrtGetDeviceType]DeviceType = %d.", static_cast<s32>(g_deviceType));
        return HCCL_SUCCESS;
    }

    auto iter = SOC_VER_CONVERT.find(socName);
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get illegal chipver, chip_ver[%s].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), socName.c_str());
        return HCCL_E_RUNTIME;
    }
    devType = iter->second;
    g_deviceType = devType;
    return HCCL_SUCCESS;
}
weak_alias(__hrtGetDeviceType, hrtGetDeviceType);
#ifdef __cplusplus
} // extern "C"
#endif

#endif

#if T_DESC("DeviceMemory管理", true)

HcclResult HrtDevFree(void* devPtr)
{
    CHK_PTR_NULL(devPtr);
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(PLF_RES, "Free DevMem para: deviceId[%d] devPtr[%p]", deviceId, devPtr);

    static auto funcPtr = (aclError(*)(void*))g_dlAcl.Handle<ACL_RT_FREE>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(devPtr);

    HCCL_DEBUG("Call aclrtFree, ret[%d], devPtr[%p]", ret, devPtr);
    CHK_PRT_RET(
        (ret != ACL_SUCCESS),
        HCCL_ERROR(
            "[Free][Mem]errNo[0x%016llx] aclrtFree failed, return[%d]"
            ", para: devPtr[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, devPtr),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult HrtDevMalloc(void** devPtr, u64 size)
{
    CHK_PTR_NULL(devPtr);

    static auto funcPtr = (rtError_t(*)(
        void**, size_t, aclrtMemMallocPolicy, aclrtMallocConfig*))g_dlAcl.Handle<ACL_RT_MALLOC_WITH_CFG>();
    CHK_PTR_NULL(funcPtr);

    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

    aclError ret = funcPtr(devPtr, size, ACL_MEM_TYPE_HIGH_BAND_WIDTH, &cfg);

    CHK_PRT_RET(
        (ret != ACL_SUCCESS || *devPtr == nullptr),
        HCCL_ERROR(
            "[Malloc][Mem]errNo[0x%016llx] "
            "aclrtMallocWithCfg failed, return[%d], para: devPtr[%p], size[%llu Byte].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *devPtr, size),
        HCCL_E_RUNTIME);

    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(PLF_RES, "Malloc DevMem para: deviceId[%d] devPtr[%p] size[%llu Byte]", deviceId, *devPtr, size);
    return HCCL_SUCCESS;
}

HcclResult hrtMalloc(void** devPtr, u64 size, [[maybe_unused]] bool level2Address)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(devPtr);

    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    aclError ret = ACL_SUCCESS;
    s32 policy = 0;
    bool isTsMem = false;
    if (Is310PDevice()) {
        if (devType == DevType::DEV_TYPE_310P3 || devType == DevType::DEV_TYPE_310P1) {
            if (level2Address) { // 310P二级地址刷新时申请内存类型为：RT_MEMORY_TS
                isTsMem = true;
            } else {
                policy = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH);
            }
        } else {
            policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH);
        }
    } else {
        if (devType == DevType::DEV_TYPE_310P3) {
            if (level2Address) { // 310P二级地址刷新时申请内存类型为：RT_MEMORY_TS
                isTsMem = true;
            } else {
                policy
                    = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_NORMAL_ONLY_P2P);
            }
        } else if (devType == DevType::DEV_TYPE_310P1) {
            policy = static_cast<int>(ACL_MEM_TYPE_LOW_BAND_WIDTH);
        } else {
            policy = static_cast<int>(ACL_MEM_TYPE_HIGH_BAND_WIDTH) | static_cast<int>(ACL_MEM_MALLOC_HUGE_FIRST);
        }
    }

    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};

    if (UNLIKELY(isTsMem)) {
        ret = aclrtMallocForTaskScheduler(devPtr, size, ACL_MEM_MALLOC_HUGE_FIRST, &cfg);
    } else {
        ret = aclrtMallocWithCfg(devPtr, size, static_cast<aclrtMemMallocPolicy>(policy), &cfg);
    }
    RPT_ENV_ERR(
        ret == ACL_ERROR_RT_MEMORY_ALLOCATION, "EI0011", std::vector<std::string>({"memory_size"}),
        std::vector<std::string>({std::to_string(size)}));

    CHK_PRT_RET(
        ret == ACL_ERROR_RT_MEMORY_ALLOCATION,
        HCCL_ERROR(
            "[Malloc][Mem] rtMalloc failed, "
            "Reason: out of memory, return[%d], para: devPtrAddr[%p], size[%llu Byte].",
            ret, *devPtr, size),
        HCCL_E_OOM);

    RPT_ENV_ERR(
        (ret != ACL_SUCCESS), "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
        std::vector<std::string>({"DeviceMemory", std::string("Malloc, size:") + std::to_string(size) + " bytes"}));

    CHK_PRT_RET(
        (ret != ACL_SUCCESS),
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rtMalloc failed, "
            "return[%d], para: devPtrAddr[%p], size[%llu Byte].",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret,
            *devPtr, size),
        HCCL_E_RUNTIME);
    PLF_CONFIG_INFO(
        PLF_RES,
        "Malloc DevMem para: deviceId[%d] devPtr[%p] size[%llu Byte] "
        "level2Address[%u]",
        deviceId, *devPtr, size, level2Address);
    return HCCL_SUCCESS;
#else
    // 参数有效性检查
    CHK_PTR_NULL(devPtr);

    *devPtr = malloc(size);
    CHK_PTR_NULL(*devPtr);
    PLF_CONFIG_INFO(PLF_RES, "Malloc para: devPtr[%p] size[%llu Byte]", *devPtr, size);
    return HCCL_SUCCESS;
#endif
}

HcclResult hrtMemSyncCopy(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] const void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] HcclRtMemcpyKind kind)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemSyncCopy] count is zero"), HCCL_SUCCESS);

    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemSyncCopy] hrtThreadExchangeCaptureMode return [%d]", hcclRet));

    aclrtMemcpyKind rtKind;
    CHK_RET(MemcpyKindTranslate(kind, &rtKind));
    aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);
    HCCL_DEBUG(
        "Call aclrtMemcpy, return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d]", ret,
        dst, destMax, src, count, rtKind);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SyncCopy][Mem]errNo[0x%016llx] aclrtMemcpy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, src, count, rtKind),
        HCCL_E_RUNTIME);

    hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemSyncCopy] hrtThreadExchangeCaptureMode return [%d]", hcclRet));
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemSyncCopy]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMemSyncCopyEx(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] const void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] HcclRtMemcpyKind kind)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemSyncCopyEx] count is zero"), HCCL_SUCCESS);

    aclrtMemcpyKind rtKind;
    CHK_RET(MemcpyKindTranslate(kind, &rtKind));
    aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);

    HCCL_DEBUG(
        "[hrtMemSyncCopyEx] Call aclrtMemcpy, ret[%d], dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], "
        "rtKind[%d]",
        ret, dst, destMax, src, count, rtKind);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[SyncCopy][Mem]errNo[0x%016llx] aclrtMemcpy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, src, count, rtKind),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemSyncCopyEx]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtFree(void* devPtr)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(devPtr);
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(PLF_RES, "Free DevMem para: deviceId[%d] dev_ptr[%p].", deviceId, devPtr);

    aclError ret = aclrtFree(devPtr);
    HCCL_DEBUG("Call aclrtFree, return value[%d], para: dev_ptr[%p].", ret, devPtr);
    CHK_PRT_RET(
        (ret != ACL_SUCCESS),
        HCCL_ERROR(
            "[Free][Mem]errNo[0x%016llx] aclrtFree failed, "
            "return[%d], para: devPtrAddr[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, devPtr),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    CHK_PTR_NULL(devPtr);
    PLF_CONFIG_DEBUG(PLF_RES, "Free DevMem para: dev_ptr[%p].", devPtr);
    free(devPtr);

    return HCCL_SUCCESS;
#endif
}

HcclResult hrtMemcpyAddrAsync(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] uint64_t destOffset,
    [[maybe_unused]] const void* src, [[maybe_unused]] uint64_t count, [[maybe_unused]] uint64_t srcOffset,
    [[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(stream);

    if (count == 0) {
        HCCL_WARNING("Call hrtMemcpyAddrAsync, count [%d]", count);
        return HCCL_SUCCESS;
    }

    aclError ret = aclrtMemcpyAsyncWithOffset(
        reinterpret_cast<void**>(dst), destMax, destOffset, reinterpret_cast<const void**>(const_cast<void*>(src)),
        count, srcOffset, ACL_MEMCPY_INNER_DEVICE_TO_DEVICE, stream);

    HCCL_DEBUG(
        "Call hrtMemcpyAddrAsync, return value[%d], dstAddr[%p], destMax[%llu], destOffset[%llu], "
        "srcAddr[%p], count[%llu], srcOffset[%llu]",
        ret, dst, destMax, destOffset, src, count, srcOffset);

    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING("hrtMemcpyAddrAsync is not supported.", ret);
        return HCCL_E_NOT_SUPPORT;
    }

    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], destOffset[%llu], srcAddr[%p], count[%llu], "
            "srcOffset[%llu], "
            "stream[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, destOffset, src, count, srcOffset, stream),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemcpyAddrAsync]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult MemcpyKindTranslate(HcclRtMemcpyKind kind, aclrtMemcpyKind* rtKind)
{
    switch (kind) {
        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_HOST: {
            *rtKind = ACL_MEMCPY_HOST_TO_HOST;
            return HCCL_SUCCESS;
        }

        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_HOST_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_HOST_TO_DEVICE;
            return HCCL_SUCCESS;
        }

        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_HOST: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_HOST;
            return HCCL_SUCCESS;
        }

        case HcclRtMemcpyKind::HCCL_RT_MEMCPY_KIND_DEVICE_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            return HCCL_SUCCESS;
        }

        default: {
            HCCL_ERROR("[MemcpyKindTranslate]Not support the memory copy type[%d].", kind);
            return HCCL_E_PARA;
        }
    }
}

HcclResult hrtMemAsyncCopy(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] const void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] HcclRtMemcpyKind kind, [[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemAsyncCopy] count is zero"), HCCL_SUCCESS);

    aclrtMemcpyKind rtKind;
    CHK_RET(MemcpyKindTranslate(kind, &rtKind));

    aclError ret = aclrtMemcpyAsync(dst, destMax, src, count, rtKind, stream);
    HCCL_DEBUG(
        "Call aclrtMemcpyAsync, return value[%d], para: dstAddr[%p], destMax[%llu], "
        "srcAddr[%p], count[%llu], rtKind[%d]",
        ret, dst, destMax, src, count, rtKind);

    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d], stream[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, src, count, rtKind, stream),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemAsyncCopy]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMemAsyncCopyWithoutCheckKind(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] const void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] HcclRtMemcpyKind kind, [[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PTR_NULL(stream);
    CHK_PRT_RET(count == 0, HCCL_WARNING("[hrtMemAsyncCopyWithoutCheckKind] count is zero"), HCCL_SUCCESS);

    aclrtMemcpyKind rtKind;
    CHK_RET(MemcpyKindTranslate(kind, &rtKind));
    aclError ret = aclrtMemcpyAsync(dst, destMax, src, count, rtKind, stream);

    HCCL_DEBUG(
        "Call rtsMemcpyAsync, return value[%d], dstAddr[%p], destMax[%llu],"
        "srcAddr[%p], count[%llu]",
        ret, dst, destMax, src, count);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d], stream[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, src, count, rtKind, stream),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemAsyncCopyWithoutCheckKind]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetPairDeviceLinkTypeRaw(u32 phyDevId, u32 otherPhyDevId, s32 infoType, s64* pValue)
{
    static auto funcPtr = (rtGetPairPhyDevicesInfoPtr)g_dlRts.Handle<RT_GET_PAIR_PHY_DEVICES_INFO>();
    if (funcPtr != nullptr) {
        return hrtGetPairPhyDevicesInfo(phyDevId, otherPhyDevId, infoType, pValue, funcPtr);
    } else {
        return hrtGetPairDevicesInfo(phyDevId, otherPhyDevId, infoType, pValue);
    }
}

HcclResult hrtGetPairDevicesInfo(
    [[maybe_unused]] u32 phyDevId, [[maybe_unused]] u32 otherPhyDevId, [[maybe_unused]] s32 infoType,
    [[maybe_unused]] s64* pValue)
{
#ifndef HCCD
    CHK_PTR_NULL(pValue);
    u32 logicIdLocal = 0;
    u32 logicIdDest = 0;
    CHK_RET(hrtGetDeviceIndexByPhyId(phyDevId, logicIdLocal));

    CHK_RET(hrtGetDeviceIndexByPhyId(otherPhyDevId, logicIdDest));

    aclError ret = aclrtGetDevicesTopo(logicIdLocal, logicIdDest, reinterpret_cast<uint64_t*>(pValue));
    HCCL_DEBUG(
        "aclrt get pair devices info, return[%d], "
        "para: phyDevId[%u], otherPhyDevId[%u], logicIdLocal[%u], logicIdDest[%u], value[%lld].",
        ret, phyDevId, otherPhyDevId, logicIdLocal, logicIdDest, *pValue);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][PairPhyDevicesInfo]errNo[0x%016llx] rt get pair devices info "
            "failed, return[%d], para: phyDevId[%u], otherPhyDevId[%u], logicIdLocal[%u], logicIdDest[%u], "
            "value[%lld].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, phyDevId, otherPhyDevId, logicIdLocal, logicIdDest, *pValue),
        HCCL_E_RUNTIME);

    // 当前 aclrtGetDevicesTopo 仅支持返回一种链路类型，将来可能会同时按照二进制位排列返回所支持的多种类型
    // 按二进制位进行比较
    if ((*pValue & ACL_RT_DEVS_TOPOLOGY_HCCS) != 0) {
        *pValue = TOPOLOGY_HCCS;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_PIX) != 0) {
        *pValue = TOPOLOGY_PIX;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_PIB) != 0) {
        *pValue = TOPOLOGY_PIB;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_PHB) != 0) {
        *pValue = TOPOLOGY_PHB;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_SYS) != 0) {
        *pValue = TOPOLOGY_SYS;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_SIO) != 0) {
        *pValue = TOPOLOGY_SIO;
    } else if ((*pValue & ACL_RT_DEVS_TOPOLOGY_HCCS_SW) != 0) {
        *pValue = TOPOLOGY_HCCS_SW;
    } else {
        HCCL_ERROR(
            "aclrt get pair devices info failed, unknown linkType[%lld], "
            "para: phyDevId[%u], otherPhyDevId[%u], logicIdLocal[%u], logicIdDest[%u].",
            *pValue, phyDevId, otherPhyDevId, logicIdLocal, logicIdDest);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetPairDevicesInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetPairPhyDevicesInfo(
    [[maybe_unused]] u32 phyDevId, [[maybe_unused]] u32 otherPhyDevId, [[maybe_unused]] s32 infoType,
    [[maybe_unused]] s64* pValue, [[maybe_unused]] rtGetPairPhyDevicesInfoPtr funcPtr)
{
#ifndef HCCD
    CHK_PTR_NULL(pValue);
    rtError_t ret = funcPtr(phyDevId, otherPhyDevId, infoType, reinterpret_cast<int64_t*>(pValue));
    HCCL_DEBUG(
        "rt get pair devices info, return[%d], para: phyDevId[%u], otherPhyDevId[%u], infoType[%d], value[%lld]", ret,
        phyDevId, otherPhyDevId, infoType, *pValue);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][PairPhyDevicesInfo]errNo[0x%016llx] rt get pair devices "
            "info failed, return[%d], para: phyDevId[%u], otherPhyDevId[%u], infoType[%d], value[%lld].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, phyDevId, otherPhyDevId, infoType, *pValue),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetPairPhyDevicesInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetPairDeviceLinkType(u32 phyDevId, u32 otherPhyDevId, LinkTypeInServer& linkType)
{
    if (Is310PDevice()) {
        linkType = LinkTypeInServer::HCCS_TYPE;
        return HCCL_SUCCESS;
    }

    s64 linkTypeRaw = 0;
#ifndef HCCD
    CHK_RET(hrtGetPairDeviceLinkTypeRaw(phyDevId, otherPhyDevId, 0, &linkTypeRaw));
#else
    HCCL_ERROR("[hrtGetPairDeviceLinkTypeRaw]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
    HCCL_INFO(
        "[hrtGetPairDeviceLinkTypeRaw]phyDevId[%u] otherPhyDevId[%u] linkTypeRaw[%d]", phyDevId, otherPhyDevId,
        linkTypeRaw);

    // 若当前为标卡/虚拟机device间通过HCCS直接互联：HCCS_TYPE，device间通过HCCS交换芯片互联：TOPOLOGY_HCCS_SW
    // Ascend910_93* die间为SIO_TYPE
    // 其他情况为PXI_TYPE

    switch (linkTypeRaw) {
        case TOPOLOGY_HCCS:
            linkType = LinkTypeInServer::HCCS_TYPE;
            break;

        case TOPOLOGY_HCCS_SW:
            linkType = LinkTypeInServer::HCCS_SW_TYPE;
            break;

        case TOPOLOGY_SIO:
            linkType = LinkTypeInServer::SIO_TYPE;
            break;

        default:
            linkType = LinkTypeInServer::PXI_TYPE;
    }

    return HCCL_SUCCESS;
}

HcclResult hrtGetPairDevicePhyId([[maybe_unused]] u32 localDevPhyId, [[maybe_unused]] u32& pairDevPhyId)
{
#ifndef HCCD
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    CHK_PRT_RET(
        deviceType != DevType::DEV_TYPE_910_93,
        HCCL_ERROR(
            "[hrtGetPairDevicePhyId] is not supported on device type[%d]. Please check device type.", deviceType),
        HCCL_E_NOT_SUPPORT);

    // 奇数die对应的phyid - 1，偶数die对应的phyid + 1
    int offset = static_cast<s32>(localDevPhyId) % 2 == 0 ? 1 : -1;

    // 通过本端的phyId获取同一个chip上的另一个die的PhyId
    pairDevPhyId = localDevPhyId + offset;
    LinkTypeInServer linkType = LinkTypeInServer::RESERVED_LINK_TYPE;
    CHK_RET(hrtGetPairDeviceLinkType(localDevPhyId, pairDevPhyId, linkType));
    if (linkType != LinkTypeInServer::SIO_TYPE) {
        pairDevPhyId = localDevPhyId - offset;
        CHK_RET(hrtGetPairDeviceLinkType(localDevPhyId, pairDevPhyId, linkType));
        CHK_PRT_RET(
            linkType != LinkTypeInServer::SIO_TYPE,
            HCCL_ERROR(
                "[hrtGetPairDevicePhyId] neither of the two neighbour device is a pair device of dev[%u].",
                localDevPhyId),
            HCCL_E_NOT_SUPPORT);
    }
    HCCL_DEBUG(
        "[hrtGetPairDevicePhyId]GetPairDevicePhyId success, phyDevId[%u], pairDevPhyId[%u]", localDevPhyId,
        pairDevPhyId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetPairDevicePhyId]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

// 获取指针的属性，主要是页表大小，单位Byte
HcclResult hrtGetPointAttr(HcclRtPointAttr ptrAttr, const void* ptr)
{
    // 参数有效性检查
    CHK_PTR_NULL(ptrAttr);
    CHK_PTR_NULL(ptr);

#ifndef HCCD
    aclError ret = aclrtPointerGetAttributes(ptr, reinterpret_cast<aclrtPtrAttributes*>(ptrAttr));
#else
    static auto funcPtr
        = (aclError(*)(const void* ptr, aclrtPtrAttributes* attributes))g_dlAcl.Handle<ACL_RT_POINTER_GET_ATTRIBUTES>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(ptr, reinterpret_cast<aclrtPtrAttributes*>(ptrAttr));
#endif
    HCCL_DEBUG("Call aclrtPointerGetAttributes, return value[%d], para: ptr[%p].", ret, ptr);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][PointAttr]errNo[0x%016llx] rt get point attr failed, "
            "return[%d], para: ptrAttrAddr[%p], ptrAddr[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, ptrAttr, ptr),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult hrtIpcSetMemoryName(void* ptr, u8* name, u64 ptrMaxLen, u32 nameMaxLen)
{
    // 参数有效性检查
    CHK_PTR_NULL(ptr);
    CHK_PTR_NULL(name);
#ifndef HCCD
    // flag 预留字段填 0
    aclError rtRet = aclrtIpcMemGetExportKey(ptr, ptrMaxLen, reinterpret_cast<char*>(name), nameMaxLen, 0UL);
#else
    static auto funcPtr
        = (aclError(*)(void*, size_t, char*, size_t, uint64_t))g_dlAcl.Handle<ACL_RT_IPC_MEM_GET_EXPORT_KEY>();
    CHK_PTR_NULL(funcPtr);
    aclError rtRet = funcPtr(ptr, ptrMaxLen, reinterpret_cast<char*>(name), nameMaxLen, 0UL);
#endif
    HCCL_INFO(
        "Call aclrtIpcMemGetExportKey, return value[%d], para: ptr[%p], name[%s], byteCount[%llu], nameLen[%u]", rtRet,
        ptr, name, ptrMaxLen, nameMaxLen);
    CHK_PRT_RET(
        rtRet != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][IpcMemoryName]errNo[0x%016llx] rtSet Ipc Memory Name, "
            "return[%d], para: ptr[%p] byteCount[%llu]. Possible Cause: The memory addr or size is not aligned with "
            "pagesize.",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), rtRet, ptr, ptrMaxLen),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult hrtIpcDestroyMemoryName(const u8* name)
{
    // 参数有效性检查
    CHK_PTR_NULL(name);

#ifndef HCCD
    aclError ret = aclrtIpcMemClose(reinterpret_cast<const char*>(name));
#else
    static auto funcPtr = (aclError(*)(const char*))g_dlAcl.Handle<ACL_RT_DESTORY_MEM_NAME>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(reinterpret_cast<const char*>(name));
#endif
    HCCL_INFO("Call aclrtIpcMemClose, return[%d], para: name[%s]", ret, name);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Destroy][IpcMemoryName]errNo[0x%016llx] "
            "rtDestroy Ipc memory name fail. return[%d], para: name[%s]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, name),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult hrtIpcSetMemoryAttr(
    [[maybe_unused]] const u8* name, [[maybe_unused]] aclrtIpcMemAttrType type, [[maybe_unused]] u64 attr)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(name);
    rtError_t ret = aclrtIpcMemSetAttr(reinterpret_cast<const char*>(name), type, attr);
    HCCL_INFO("Call aclrtIpcMemSetAttr, return[%d], para: name: %s, type: %u, attr: %llu", ret, name, type, attr);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[%s]errNo[0x%016llx] "
            "aclrtIpcMemSetAttr fail. return[%d], para: name: %s, type: %u, attr: %llu",
            __func__, HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, name, type, attr),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("Call aclrtIpcMemSetAttr not support");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtIpcOpenMemory(void** ptr, const u8* name)
{
    CHK_PTR_NULL(ptr);
    CHK_PTR_NULL(name);
#ifndef HCCD
    // flag 预留字段填 0
    aclError ret = aclrtIpcMemImportByKey(ptr, reinterpret_cast<const char*>(name), 0UL);
#else
    static auto funcPtr = (aclError(*)(void**, const char*, uint64_t))g_dlAcl.Handle<ACL_RT_IPC_MEM_IMPORT_BY_KEY>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(ptr, reinterpret_cast<const char*>(name), 0UL);
#endif
    RPT_CALL_ERR(ret != ACL_SUCCESS, "aclrtIpcMemImportByKey failed. return[%d], ptr[%p], name[%s]", ret, ptr, name);
    CHK_PRT_RET(
        ret == ACL_ERROR_RT_MEMORY_ALLOCATION,
        HCCL_ERROR(
            "[Open][IpcMemory]errNo[0x%016llx] "
            "rtIpc memory allocation error. return[%d], para: ptr[%p], name[%s]",
            HCCL_ERROR_CODE(HCCL_E_MEMORY), ret, *ptr, name),
        HCCL_E_MEMORY);

    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Open][IpcMemory]errNo[0x%016llx] "
            "rtOpen ipc memory fail. return[%d], para: ptr[%p], name[%s]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *ptr, name),
        HCCL_E_RUNTIME);

    HCCL_INFO("Call aclrtIpcMemImportByKey, return value[%d], para: ptr[%p], name[%s].", ret, *ptr, name);

    return HCCL_SUCCESS;
}

HcclResult hrtIpcSetMemoryPid(const u8* name, int pid[], int num)
{
    CHK_PTR_NULL(name);
    CHK_PTR_NULL(pid);
#ifndef HCCD
    aclError ret = aclrtIpcMemSetImportPid(reinterpret_cast<const char*>(name), pid, static_cast<size_t>(num));
#else
    static auto funcPtr = (aclError(*)(const char*, int32_t*, size_t))g_dlAcl.Handle<ACL_RT_IPC_MEM_SET_IMPORT_PID>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(reinterpret_cast<const char*>(name), pid, static_cast<size_t>(num));
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][IpcMemoryPid]errNo[0x%016llx] rtSet ipc memory pid fail. return[%d], num[%d], name[%s]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, num, name),
        HCCL_E_RUNTIME);
    HCCL_INFO("Call aclrtIpcMemSetImportPid, return value[%d], num[%d], name[%s].", ret, num, name);

    return HCCL_SUCCESS;
}

HcclResult hrtSetIpcMemorySuperPodPid(
    [[maybe_unused]] const u8* name, [[maybe_unused]] s32 peerSdid, [[maybe_unused]] s32 peerPid[],
    [[maybe_unused]] s32 pidNum)
{
#ifndef HCCD
    CHK_PTR_NULL(name);
    CHK_PTR_NULL(peerPid);

    // 批量设置共享内存的 SDID
    aclrtServerPid serverPid = {};
    serverPid.sdid = static_cast<u32>(peerSdid); // 白名单 Server Device Id，整网设备编号，配置在 BIOS 中
    serverPid.pid = peerPid;                     // Host 侧进程 ID 白名单数组
    serverPid.num = static_cast<size_t>(pidNum); // pid 数组长度
    aclError ret = aclrtIpcMemImportPidInterServer(reinterpret_cast<const char*>(name), &serverPid, 1U);
    if (ret != ACL_SUCCESS) {
        std::string pidStr;
        for (int i = 0; i < pidNum; ++i) {
            pidStr += std::to_string(*(peerPid + i)) + " ";
        }
        HCCL_ERROR(
            "[Set][IpcMemorySuperPodPid]errNo[0x%016llx] "
            "rtSet ipc memory pid fail. return[%d], name[%s], peerSdid[%016llx], peerPid[%s], pidNum[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, name, peerSdid, pidStr.c_str(), pidNum);
        return HCCL_E_RUNTIME;
    }
    HCCL_INFO(
        "Call aclrtIpcMemImportPidInterServer, return value[%d], name[%s], peerSdid[%016llx], "
        "peerPid[%d], pidNum[%d].",
        ret, name, peerSdid, *peerPid, pidNum);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtSetIpcMemorySuperPodPid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtDevMemAlignWithPage(void*& ptr, u64& size)
{
    // 参数有效性检查
    CHK_PTR_NULL(ptr);

    // 获取页表属性大小
    HcclRtPointAttr ptrAttr = nullptr;
    HcclResult ret = hrtMallocHost(&ptrAttr, sizeof(aclrtPtrAttributes));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[hrtDevMemAlignWithPage]runtime malloc host fail. return[%d]", ret),
        HCCL_E_INTERNAL);

    ret = hrtGetPointAttr(ptrAttr, ptr);
    if (ret != HCCL_SUCCESS) {
        ret = hrtFreeHost(ptrAttr);
        ptrAttr = nullptr;
        CHK_PRT_RET(
            ret != HCCL_SUCCESS, HCCL_ERROR("[hrtDevMemAlignWithPage]runtime free host fail.return[%d]", ret),
            HCCL_E_INTERNAL);
        HCCL_ERROR("[hrtDevMemAlignWithPage]runtime get point attr fail. return[%d]", ret);
        return HCCL_E_INTERNAL;
    }
    int32_t pageSize = (reinterpret_cast<aclrtPtrAttributes*>(ptrAttr))->pageSize;
    CHK_PRT_RET(
        pageSize < 0,
        {
            ret = hrtFreeHost(ptrAttr);
            ptrAttr = nullptr;
            HCCL_ERROR("[hrtDevMemAlignWithPage]ptr[%p] pageSize[%d] is invalid", ptr, pageSize);
        },
        HCCL_E_DRV);
    HCCL_INFO("[hrtDevMemAlignWithPage]get pageSize[%d]", pageSize);
    ret = hrtFreeHost(ptrAttr);
    ptrAttr = nullptr;
    CHK_PRT_RET(
        ret != HCCL_SUCCESS, HCCL_ERROR("[hrtDevMemAlignWithPage]runtime free host fail. return[%d]", ret),
        HCCL_E_INTERNAL);

    // 按照page_size = 0当做没有跨进程映射对齐要求
    if (pageSize == 0) {
        u64 offset = 0;
        size = size + offset;
        return HCCL_SUCCESS;
    }
    // 按页表大小对齐指针
    u64 tmpPtr = reinterpret_cast<u64>(ptr);
    ptr = reinterpret_cast<void*>((reinterpret_cast<u64>(ptr)) & (~(static_cast<u64>(pageSize) - 1)));
    u64 offset = tmpPtr - reinterpret_cast<u64>(ptr);

    // 计算size值
    size = size + offset;

    return HCCL_SUCCESS;
}

#endif

#if T_DESC("host memory管理", true)

HcclResult hrtMallocHost(void** hostPtr, u64 size)
{
    // 参数有效性检查
    CHK_PTR_NULL(hostPtr);
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
#ifndef HCCD
    aclError ret = aclrtMallocHostWithCfg(hostPtr, size, &cfg);
#else
    static auto funcPtr
        = (aclError(*)(void**, uint64_t, aclrtMallocConfig*))g_dlAcl.Handle<ACL_RT_MALLOC_HOST_WITH_CFG>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(hostPtr, size, &cfg);
#endif
    RPT_ENV_ERR(
        (ret != ACL_SUCCESS), "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
        std::vector<std::string>({"HostMemory", std::string("MallocHost, size:") + std::to_string(size) + " bytes"}));

    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rt malloc host fail. return[%d], "
            "para: hostPtr[%p], size[%llu Byte].",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret,
            *hostPtr, size),
        HCCL_E_RUNTIME);
    PLF_CONFIG_DEBUG(PLF_RES, "Malloc HostMem para: hostPtr[%p], size[%llu Byte]", *hostPtr, size);
    return HCCL_SUCCESS;
}

HcclResult hrtFreeHost(void* hostPtr)
{
    // 参数有效性检查
    CHK_PTR_NULL(hostPtr);
    PLF_CONFIG_DEBUG(PLF_RES, "Free HostMem para: hostPtr[%p].", hostPtr);
#ifndef HCCD
    aclError ret = aclrtFreeHost(hostPtr);
#else
    static auto funcPtr = (aclError(*)(void*))g_dlAcl.Handle<ACL_RT_FREE_HOST>();
    CHK_PTR_NULL(funcPtr);
    aclError ret = funcPtr(hostPtr);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Free][Host]errNo[0x%016llx] rt free host fail. return[%d], "
            "para: hostPtr[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, hostPtr),
        HCCL_E_RUNTIME);
    HCCL_DEBUG("Call aclrtFreeHost, return value[%d].", ret);
    return HCCL_SUCCESS;
}

#endif

#if T_DESC("stream管理", true)

HcclResult hrtStreamActive([[maybe_unused]] HcclRtStream activeStream, [[maybe_unused]] HcclRtStream stream)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(activeStream);
    CHK_PTR_NULL(stream);

    aclrtStream rtActiveStream = activeStream;
    aclrtStream rtStream = stream;
    aclError ret = aclrtActiveStream(rtActiveStream, rtStream);
    HCCL_DEBUG("Call aclrtActiveStream, return value[%d].", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Activate][Stream]errNo[0x%016llx] "
            "rt stream active fail. return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamActive]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetStreamId([[maybe_unused]] HcclRtStream stream, [[maybe_unused]] s32& streamId)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(stream);

    aclrtStream aclStream = stream;
    aclError ret = aclrtStreamGetId(aclStream, &streamId);

    HCCL_DEBUG("Call aclrtStreamGetId, return value[%d] streamId[%d].", ret, streamId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][StreamId]errNo[0x%016llx] "
            "rt get stream ID fail. return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetStreamId]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetTaskIdAndStreamID([[maybe_unused]] u32& taskId, [[maybe_unused]] u32& streamId)
{
#ifndef HCCD
    rtError_t ret = rtGetTaskIdAndStreamID(&taskId, &streamId);
    HCCL_DEBUG("Call rtGetTaskIdAndStreamId, return value[%d], para: taskId[%u], streamId[%u].", ret, taskId, streamId);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][TaskIdAndStreamID]errNo[0x%016llx] "
            "rt get task ID and stream ID fail. return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetTaskIdAndStreamID]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

#endif

#if T_DESC("event 同步机制", true)
HcclResult hrtEventCreate([[maybe_unused]] aclrtEvent* event)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    aclError ret = aclrtCreateEvent(event);
    RPT_ENV_ERR(
        ret != ACL_SUCCESS, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
        std::vector<std::string>({"event", "null"}));

    HCCL_DEBUG("Call aclrtCreateEvent, return value[%d], para: event[%p]", ret, *event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[%s][%s]errNo[0x%016llx] rt event create, return[%d], "
            "event[%p]",
            LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret,
            event),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtEventCreate]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventDestroy([[maybe_unused]] HcclRtEvent event)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    aclError ret = aclrtDestroyEvent(event);
    HCCL_DEBUG("Call aclrtDestroyEvent, return value[%d], para: event[%p]", ret, event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Destroy][Event]errNo[0x%016llx] rt event destroy, return[%d], "
            "event[%p]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, event),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtEventDestroy]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventRecord([[maybe_unused]] aclrtEvent event, [[maybe_unused]] aclrtStream stream)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    CHK_PTR_NULL(stream);
    aclError ret = aclrtRecordEvent(event, stream);
    HCCL_DEBUG("Call aclrtRecordEvent, return value[%d], para: event[%p]", ret, event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Record][Event]errNo[0x%016llx] rt event record, return[%d], "
            "event[%p]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, event),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtEventRecord]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamWaitEvent([[maybe_unused]] aclrtStream stream, [[maybe_unused]] aclrtEvent event)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    CHK_PTR_NULL(stream);
    aclError ret = aclrtStreamWaitEvent(stream, event);
    HCCL_DEBUG("Call aclrtStreamWaitEvent, return value[%d], para: event[%p]", ret, event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "errNo[0x%016llx] rt stream wait event, return[%d], event[%p], ", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret,
            event),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamWaitEvent]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventQuery([[maybe_unused]] aclrtEvent event)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    aclrtEventRecordedStatus status = ACL_EVENT_RECORDED_STATUS_NOT_READY;
    aclError ret = aclrtQueryEventStatus(event, &status);
    HCCL_DEBUG("Call aclrtQueryEventStatus, return value[%d], status[%d], para: event[%p]", ret, status, event);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtQueryEventStatus] query event status failed, event:%p, return value[%d], status[%d].", event, ret,
            status),
        HCCL_E_RUNTIME);
    return (status == ACL_EVENT_RECORDED_STATUS_COMPLETE) ? HCCL_SUCCESS : HCCL_E_RUNTIME;
#else
    HCCL_ERROR("[hrtEventQuery]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
#endif

#if T_DESC("RDMA异步", true)
// 进行单元测试时对改函数打桩可完成条件测试
bool CompareDevType(DevType left, DevType right) { return left == right; }

HcclResult hrtGetNotifySize([[maybe_unused]] u32& notifySize)
{
#ifndef HCCD
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910) {
        notifySize = NOTIFY_SIZE_BYTES_910A;
    } else if (deviceType == DevType::DEV_TYPE_910B || deviceType == DevType::DEV_TYPE_910_93) {
        notifySize = NOTIFY_SIZE_BYTES_910B;
    } else {
        notifySize = NOTIFY_SIZE_BYTES_910A;
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetNotifySize]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyGetOffset([[maybe_unused]] HcclRtNotify notify, [[maybe_unused]] u64& offset)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    auto getEventOffsetFuncPtr = [](HcclRtNotify notify, u64& offset) -> s32 {
        u32 eventId = 0;
        aclError ret = aclrtGetEventId(notify, &eventId);
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR("[aclrtGetEventId] get event id failed, event:%p, return value[%d].", notify, ret),
            HCCL_E_RUNTIME);
        offset = eventId * 0x8;
        HCCL_INFO("event id[%u] get offset[%llu]", eventId, offset);
        return HCCL_SUCCESS;
    };

    REPLACE_NOTIFY_WITH_EVENT(
        rtNotifyGetAddrOffset(notify, reinterpret_cast<uint64_t*>(&offset)), getEventOffsetFuncPtr(notify, offset));
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyGetOffset]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

// hrtNotifyCreate 要求当前线程设置过 setDevice
HcclResult hrtNotifyCreate([[maybe_unused]] s32 deviceId, [[maybe_unused]] aclrtNotify* notify)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    auto creatEventFuncPtr = [](rtNotify_t* notify) -> s32 {
        CHK_RT_RET(aclrtCreateEvent(notify));
        aclrtStream stream = nullptr;
        aclError ret = aclrtCreateStream(&stream);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[hrtNotifyCreate] aclrtCreateStream fail, ret[%d], destroy event.", ret);
            aclrtDestroyEvent(*notify);
            return ret;
        }
        ret = aclrtRecordEvent(*notify, stream);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[hrtNotifyCreate] aclrtRecordEvent fail, ret[%d], destroy stream and event.", ret);
            aclrtDestroyStream(stream);
            aclrtDestroyEvent(*notify);
            return ret;
        }
        ret = aclrtResetEvent(*notify, stream);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[hrtNotifyCreate] aclrtResetEvent fail, ret[%d], destroy stream and event.", ret);
            aclrtDestroyStream(stream);
            aclrtDestroyEvent(*notify);
            return ret;
        }
        ret = aclrtSynchronizeStream(stream);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[hrtNotifyCreate] aclrtSynchronizeStream fail, ret[%d], destroy stream and event.", ret);
            aclrtDestroyStream(stream);
            aclrtDestroyEvent(*notify);
            return ret;
        }
        ret = aclrtDestroyStream(stream);
        if (ret != ACL_SUCCESS) {
            HCCL_ERROR("[hrtNotifyCreate] aclrtDestroyStream fail, ret[%d], destroy event.", ret);
            aclrtDestroyEvent(*notify);
            return ret;
        }
        return 0;
    }; // 使用event替换notify后需要获取event id，get event id 前必须先record，因此在创建event时执行一把record

    // aclrtCreateNotify 中通过 aclrtGetDevice 获取 deviceId，所以要求当前线程设置过 setDevice
    // flag 预留字段填 0
    REPLACE_NOTIFY_WITH_EVENT(aclrtCreateNotify(notify, ACL_NOTIFY_DEFAULT), creatEventFuncPtr(notify));
    HCCL_DEBUG("[hrtNotifyCreate] deviceId[%d]", deviceId);

    u32 notifyId = 0;
    REPLACE_NOTIFY_WITH_EVENT(aclrtGetNotifyId(*notify, &notifyId), aclrtGetEventId(*notify, &notifyId));
    PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyCreate]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyDestroy([[maybe_unused]] rtNotify_t notify)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);

    u32 notifyId = 0;
    REPLACE_NOTIFY_WITH_EVENT(aclrtGetNotifyId(notify, &notifyId), aclrtGetEventId(notify, &notifyId));
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_INFO(PLF_RES, "Destroy Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    HCCL_INFO("Destroy Notify begin, para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    REPLACE_NOTIFY_WITH_EVENT(aclrtDestroyNotify(notify), aclrtDestroyEvent(notify));
    HCCL_INFO("Destroy Notify end, para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyDestroy]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyRecord([[maybe_unused]] rtNotify_t notify, [[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    u32 streamId = 0;
    u32 taskId = 0;
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(stream);
    REPLACE_NOTIFY_WITH_EVENT(aclrtRecordNotify(notify, stream), aclrtRecordEvent(notify, stream));
    CHK_RT_RET(rtGetTaskIdAndStreamID(&taskId, &streamId));
    HCCL_INFO("hrtNotifyRecord notify[%p] taskId[%u] streamId[%u]", notify, taskId, streamId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyRecord]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyWaitWithTimeOut(
    [[maybe_unused]] rtNotify_t notify, [[maybe_unused]] rtStream_t stream, [[maybe_unused]] uint32_t timeOut)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(stream);

    u32 streamId = 0;
    u32 taskId = 0;
    auto eventWaitFuncPtr = [](rtNotify_t notify, rtStream_t stream) -> s32 {
        CHK_RT_RET(aclrtStreamWaitEvent(stream, notify));
        CHK_RT_RET(aclrtResetEvent(notify, stream));
        return 0;
    };
    REPLACE_NOTIFY_WITH_EVENT(aclrtWaitAndResetNotify(notify, stream, timeOut), eventWaitFuncPtr(stream, notify));
    CHK_RT_RET(rtGetTaskIdAndStreamID(&taskId, &streamId));
    HCCL_INFO("hrtNotifyWaitWithTimeOut notify[%p] taskId[%u] streamId[%u]", notify, taskId, streamId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyWaitWithTimeOut]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyReset([[maybe_unused]] aclrtNotify notify)
{
#ifndef HCCD
    HCCL_INFO("hrtNotifyReset notify[%p]", notify);
    CHK_PTR_NULL(notify);
    CHK_RT_RET(aclrtNotifyBatchReset(&notify, 1));
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyRecord]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
#endif

#if T_DESC("EnableP2P", true)

HcclResult hrtEnableP2P([[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] u32 devicePhyId)
{
#ifndef HCCD
    rtError_t ret = rtEnableP2P(deviceLogicId, devicePhyId, 0);

    HCCL_INFO("rt enableP2P deviceLogicId[%u] and devicePhyId[%u] fail[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Enable][P2P]errNo[0x%016llx] rt enableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtEnableP2P]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtDisableP2P([[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] u32 devicePhyId)
{
#ifndef HCCD
    rtError_t ret = rtDisableP2P(deviceLogicId, devicePhyId);

    HCCL_INFO("rt disableP2P deviceLogicId[%u] and devicePhyId[%u] fail[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Disable][P2P]errNo[0x%016llx] rt disableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtDisableP2P]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult
hrtGetP2PStatus([[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] u32 devicePhyId, [[maybe_unused]] uint32_t* status)
{
#ifndef HCCD
    rtError_t ret = rtGetP2PStatus(deviceLogicId, devicePhyId, status);

    HCCL_DEBUG(
        "rt getp2pstatus deviceLogicId[%u] and devicePhyId[%u] fail[%d], status[%u]", deviceLogicId, devicePhyId, ret,
        *status);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][P2PStatus]errNo[0x%016llx]Call rtGetP2PStatus failed, "
            "ret[%d], deviceLogicId[%u], devicePhyId[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId, devicePhyId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetP2PStatus]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
#endif

s32 GetMsTimeFromExecTimeout(s32 execTimeOut)
{
    s64 timeOutMs = 0;
    timeOutMs = (execTimeOut + HCCL_EXEC_TIME_OUT_OFFSET_S) * TIME_S_TO_MS;
    timeOutMs = (timeOutMs > 0x7FFFFFFF) ? 0x7FFFFFFF : timeOutMs;
    return static_cast<s32>(timeOutMs & (0x7FFFFFFF));
}

HcclResult hcclStreamSynchronize([[maybe_unused]] HcclRtStream stream, [[maybe_unused]] s32 execTimeOut)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);
    aclError ret = aclrtSynchronizeStreamWithTimeout(stream, GetMsTimeFromExecTimeout(execTimeOut));
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Synchronize][Stream]errNo[0x%016llx] rt "
            "streamsynchronizewithtimeout fail. return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hcclStreamSynchronize]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult
hrtTaskAbortHandleCallback([[maybe_unused]] aclrtDeviceTaskAbortCallback callback, [[maybe_unused]] void* args)
{
#ifndef HCCD
    aclError ret = aclrtSetDeviceTaskAbortCallback("HCCL", callback, args);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Reg][TaskAbortCallBack]errNo[0x%016llx] rt reg taskabortcallback "
            "fail. return[%d], para: callback[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, callback),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtTaskAbortHandleCallback]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult PrintMemoryAttr([[maybe_unused]] const void* memAddr)
{
#ifndef HCCD
    if (LIKELY(!HcclCheckLogLevel(HCCL_LOG_INFO))) {
        return HCCL_SUCCESS;
    }

    HcclResult ret = HCCL_SUCCESS;
    aclrtPtrAttributes memAttr;
    CHK_PTR_NULL(memAddr);
    s32 sRet = memset_s(&memAttr, sizeof(aclrtPtrAttributes), 0, sizeof(aclrtPtrAttributes));
    CHK_PRT_RET(
        sRet != EOK,
        HCCL_ERROR(
            "[Print][MemoryAttr]errNo[0x%016llx]memory set 0 failed for memAttr. "
            "params: dest[%p], destMaxSize[%zu], count[%zu]",
            HCOM_ERROR_CODE(HCCL_E_MEMORY), &memAttr, sizeof(aclrtPtrAttributes), sizeof(aclrtPtrAttributes)),
        HCCL_E_MEMORY);
    ret = hrtGetPointAttr(&memAttr, memAddr);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR("[Print][MemoryAttr]errNo[0x%016llx] runtime get memory attr failed", HCOM_ERROR_CODE(ret)), ret);
    HCCL_INFO("memory attributes: address[%p], page size[%u]", memAddr, memAttr.pageSize);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[PrintMemoryAttr]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
HcclResult hrtRegTaskFailCallbackByModule([[maybe_unused]] rtTaskFailCallback callback)
{
#ifndef HCCD
    rtError_t ret = rtRegTaskFailCallbackByModule("HCCL", callback);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Reg][TaskFailCallback]errNo[0x%016llx] rt reg taskFailCallback "
            "fail. return[%d], para: callback[%p].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, callback),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtRegTaskFailCallbackByModule]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetStreamAvailableNum([[maybe_unused]] u32& maxStrCount)
{
#ifndef HCCD
    aclError ret = aclrtGetStreamAvailableNum(&maxStrCount);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][StreamAvailableNum]errNo[0x%016llx] aclrtGetStreamAvailableNum "
            "fail. return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetStreamAvailableNum]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtSubscribeReport([[maybe_unused]] u64 threadId, [[maybe_unused]] rtStream_t& stream)
{
#ifndef HCCD
    aclError ret = aclrtSubscribeReport(threadId, stream);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[HostNic][RegStream]errNo[0x%016llx] aclrtSubscribeReport fail,"
            "return[%d], para: threadId[%llu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, threadId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtSubscribeReport]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
HcclResult hrtUnSubscribeReport([[maybe_unused]] uint64_t threadId, [[maybe_unused]] aclrtStream& stream)
{
#ifndef HCCD
    aclError ret = aclrtUnSubscribeReport(threadId, stream);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Close][HostNicThread]errNo[0x%016llx] aclrtUnSubscribeReport fail,"
            "return[%d], para: threadId[%llu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, threadId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtUnSubscribeReport]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
HcclResult hrtProcessReport([[maybe_unused]] s32 timeout)
{
#ifndef HCCD
    aclError ret = aclrtProcessReport(timeout);
    if (ret != ACL_SUCCESS) {
        HCCL_INFO("aclrtProcessReport timeout, return[%d]", ret);
        return HCCL_E_TIMEOUT;
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtProcessReport]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtDeviceGetBareTgid(s32* pid)
{
    CHK_PTR_NULL(pid);
#ifndef HCCD
    aclError ret = aclrtDeviceGetBareTgid(pid);
    HCCL_DEBUG("Call rtDeviceGetBareTgid, return value[%d], rtGet pid[%u].", ret, *pid);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Get][BareTgid]errNo[0x%016llx] rtGet pid fail, "
            "return[%d], rtGet pid[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *pid),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    if (g_workModeAicpu) {
        // aicpu不使用PID。
        *pid = 0;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[hrtDeviceGetBareTgid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtIpcOpenNotify([[maybe_unused]] aclrtNotify* notify, [[maybe_unused]] const u8* name)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(name);

    aclError ret = aclrtNotifyImportByKey(notify, reinterpret_cast<const char*>(name), ACL_NOTIFY_DEFAULT);
    HCCL_INFO("Call aclrtNotifyImportByKey, return value[%d] para: notify[%p], name[%s].", ret, *notify, name);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[rt][IpcOpenNotify]errNo[0x%016llx] rt ipc notify open fail,"
            "return[%d]. para: notify[%p], name[%s]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, *notify, name),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtIpcOpenNotify]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult
hrtSetIpcNotifyPid([[maybe_unused]] aclrtNotify notify, [[maybe_unused]] int32_t pid[], [[maybe_unused]] int num)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(pid);

    aclError ret = aclrtNotifySetImportPid(notify, pid, num);
    HCCL_DEBUG("Call aclrtNotifySetImportPid, return value[%d]. Params: num[%d].", ret, num);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][IpcNotifyPid]errNo[0x%016llx] "
            "rtSet ipc Notify pid fail. return[%d], num[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, num),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtSetIpcNotifyPid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtSetIpcNotifySuperPodPid(
    [[maybe_unused]] rtNotify_t notify, [[maybe_unused]] s32 peerSdid, [[maybe_unused]] s32 peerPid[],
    [[maybe_unused]] s32 pidNum)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(peerPid);

    // 批量设置共享 Notify 的 SDID
    aclrtServerPid serverPid = {};
    serverPid.sdid = static_cast<u32>(peerSdid); // 白名单 Server Device Id，整网设备编号，配置在 BIOS 中
    serverPid.pid = peerPid;                     // 白名单进程 ID 数组
    serverPid.num = static_cast<size_t>(pidNum); // pid 数组长度
    aclError ret = aclrtNotifySetImportPidInterServer(notify, &serverPid, 1U); // 设置 1 组 SDID
    if (ret != ACL_SUCCESS) {
        std::string pidStr;
        for (int i = 0; i < pidNum; ++i) {
            pidStr += std::to_string(*(peerPid + i)) + " ";
        }
        HCCL_ERROR(
            "[Set][IpcNotifyPid]errNo[0x%016llx] "
            "rtSet ipc Notify pid fail. return[%d], notify[%p], peerSdid[%016llx], peerPid[%s], pidNum[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, notify, peerSdid, pidStr.c_str(), pidNum);
        return HCCL_E_RUNTIME;
    }
    HCCL_DEBUG(
        "Call aclrtNotifySetImportPidInterServer, return value[%d]. "
        "Params: notify[%p], peerSdid[%016llx], peerPid[%d], pidNum[%d].",
        ret, notify, peerSdid, *peerPid, pidNum);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtSetIpcNotifySuperPodPid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtCtxGetOverflowAddr([[maybe_unused]] void** overflowAddr)
{
#ifndef HCCD
    CHK_PTR_NULL(overflowAddr);

    aclError ret = aclrtCtxGetFloatOverflowAddr(overflowAddr);
    HCCL_DEBUG("Call aclrtCtxGetFloatOverflowAddr, return value[%d].", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[rt][aclrtCtxGetFloatOverflowAddr]errNo[0x%016llx] "
            "rtCtx get overflow addr fail. return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtCtxGetOverflowAddr]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtReduceAsync(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] const void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] aclrtReduceKind kind, [[maybe_unused]] aclDataType type,
    [[maybe_unused]] aclrtStream stream)
{
#ifndef HCCD
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    // reserve 预留字段填 nullptr
    aclError ret = aclrtReduceAsync(dst, src, count, kind, type, stream, nullptr);
    HCCL_DEBUG(
        "Call aclrtReduceAsync, return value[%d]. para: dst[%p] destMax[%llu] src[%p] count[%llu] rtReduceOp[%d]"
        " runtimeDataType[%d].",
        ret, dst, destMax, src, count, kind, type);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[rt][ReduceAsync]errNo[0x%016llx] rt Reduce async fail,"
            "return[%d]. para: dst[%p] destMax[%llu] src[%p] count[%llu] rtReduceOp[%d] runtimeDataType[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind, type),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtReduceAsync]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtCallbackLaunch(
    [[maybe_unused]] aclrtCallback callBackFunc, [[maybe_unused]] void* fnData, [[maybe_unused]] aclrtStream stream,
    [[maybe_unused]] bool isBlock)
{
#ifndef HCCD
    CHK_PTR_NULL(fnData);
    aclrtCallbackBlockType blockType = isBlock ? ACL_CALLBACK_BLOCK : ACL_CALLBACK_NO_BLOCK;
    aclError ret = aclrtLaunchCallback(callBackFunc, fnData, blockType, stream);
    HCCL_DEBUG("Call aclrtLaunchCallback, return value[%d].", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[rt][CallbackLaunch]callback launch failed, ret[%d], isBlock[%d]", ret, isBlock), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtCallbackLaunch]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtRDMASend([[maybe_unused]] u32 qpn, [[maybe_unused]] u32 wqe_index, [[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    rtError_t ret = rtRDMASend(qpn, wqe_index, stream);
    HCCL_DEBUG("Call rtRDMASend, return value[%d]. Params: qpn[%u] wqe_index[%u].", ret, qpn, wqe_index);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rt][RdmaSend]errNo[0x%016llx] rt rdma send fail, "
            "return[%d]. para: qpn[%u] wqe_index[%u].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, qpn, wqe_index),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtRDMASend]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult
hrtIpcSetNotifyName([[maybe_unused]] aclrtNotify notify, [[maybe_unused]] u8* name, [[maybe_unused]] uint32_t len)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(name);

    aclError ret = aclrtNotifyGetExportKey(notify, reinterpret_cast<char*>(name), len, 0UL);
    HCCL_INFO("[hrtIpcSetNotifyName] Call aclrtNotifyGetExportKey, name[%s] return value[%d].", name, ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Set][IPCNotify]errNo[0x%016llx] IPC set notify name[%s] fail. "
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), name, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtIpcSetNotifyName]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamDestroy([[maybe_unused]] rtStream_t stream)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);
    s32 streamId = 0;
    CHK_RET(hrtGetStreamId(stream, streamId));
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(PLF_RES, "Destroy Stream para: deviceId[%d] streamId[%d]", deviceId, streamId);
    HCCL_INFO("Destroy Stream begin, para: deviceId[%d] streamId[%d]", deviceId, streamId);
    aclError ret = aclrtDestroyStreamForce(stream);
    HCCL_INFO("Call aclrtDestroyStreamForce, return value[%d].", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Stream][Destroy]errNo[0x%016llx] rt stream Destroy fail, "
            "return[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamDestroy]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamCreate([[maybe_unused]] aclrtStream* stream)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);

    aclError ret = aclrtCreateStream(stream);
    HCCL_DEBUG("Call aclrtCreateStream, return value[%d].", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Stream][Create]errNo[0x%016llx] aclrtCreateStream error, "
            "rtRet[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);

    s32 streamId = 0;
    HcclResult hcclRet = hrtGetStreamId(*stream, streamId);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[hrtStreamCreate] hrtGetStreamId fail, ret[%d], destroy stream.", ret);
        aclrtDestroyStream(*stream);
        return hcclRet;
    }
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(PLF_RES, "Create Stream para: deviceId[%d] streamId[%d]", deviceId, streamId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamCreate]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamCreateWithFlags(
    [[maybe_unused]] aclrtStream* stream, [[maybe_unused]] int32_t priority, [[maybe_unused]] uint32_t flags)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);

    aclError ret = aclrtCreateStreamWithConfig(stream, static_cast<uint32_t>(priority), flags);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Stream][CreateWithFlags]errNo[0x%016llx] aclrtCreateStreamWithConfig "
            "error, rtRet[%d], flags[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, flags),
        HCCL_E_RUNTIME);

    s32 streamId = 0;
    HcclResult hcclRet = hrtGetStreamId(*stream, streamId);
    if (hcclRet != HCCL_SUCCESS) {
        HCCL_ERROR("[hrtStreamCreateWithFlags] hrtGetStreamId fail, ret[%d], destroy stream.", ret);
        aclrtDestroyStream(*stream);
        return hcclRet;
    }
    s32 deviceId = GetDeviceLogicalId();
    PLF_CONFIG_DEBUG(
        PLF_RES, "Create Stream para: deviceId[%d] streamId[%d] priority[%d] flags[%u]", deviceId, streamId, priority,
        flags);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamCreateWithFlags]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamSetMode([[maybe_unused]] HcclRtStream stream, [[maybe_unused]] const uint64_t stmMode)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);
    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(stream, &streamId);
    HCCL_DEBUG("Call aclrtStreamGetId, return value[%d].", ret);

    aclrtStreamAttrValue value;
    value.failureMode = stmMode;
    ret = aclrtSetStreamAttribute(stream, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    HCCL_DEBUG("Call aclrtSetStreamAttribute return value[%d]. stmMode[%llu], streamId[%d].", ret, stmMode, streamId);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_WARNING(
            "[Stream][SetMode]errNo[0x%016llx] aclrtSetStreamAttribute fail, "
            "nothing changed. rtRet[%d], stmMode[%llu]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, stmMode),
        HCCL_SUCCESS);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamSetMode]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamGetMode([[maybe_unused]] HcclRtStream const stream, [[maybe_unused]] uint64_t* const stmMode)
{
#ifndef HCCD
    CHK_PTR_NULL(stream);

    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(stream, &streamId);
    HCCL_DEBUG("Call aclrtStreamGetId, return value[%d].", ret);

    aclrtStreamAttrValue value;
    ret = aclrtGetStreamAttribute(stream, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[Stream][GetMode]errNo[0x%016llx] aclrtGetStreamAttribute error, "
            "rtRet[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    *stmMode = value.failureMode;
    HCCL_DEBUG("Call aclrtGetStreamAttribute return value[%d]. stmMode[%llu], streamId[%d].", ret, *stmMode, streamId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtStreamGetMode]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, HcclRtMemcpyKind kind)
{
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(
        count > destMax, HCCL_ERROR("[hrtMemcpy] count[%llu] exceeds destMax[%llu].", count, destMax), HCCL_E_PARA);
    aclrtMemcpyKind rtKind;
    CHK_RET(MemcpyKindTranslate(kind, &rtKind));
#ifndef HCCD
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemcpy] hrtThreadExchangeCaptureMode return [%d]", hcclRet));

    aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);
    HCCL_DEBUG(
        "Call aclrtMemcpy, return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], rtKind[%d]", ret,
        dst, destMax, src, count, rtKind);

    hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemcpy] hrtThreadExchangeCaptureMode return [%d]", hcclRet));
#else
    aclError ret = ACL_SUCCESS;
    static auto funcPtr
        = (aclError(*)(void*, size_t, const void*, size_t, aclrtMemcpyKind))g_dlAcl.Handle<ACL_RT_MEMCPY>();
    CHK_PTR_NULL(funcPtr);
    ret = funcPtr(dst, destMax, src, count, rtKind);
#endif
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtMemcpy] rt data memcpy host to device failed,"
            "size[%llu Byte] src[%p]",
            destMax, src),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult hrtRDMADBSend(uint32_t dbindex, uint64_t dbinfo, rtStream_t stream)
{
#ifndef HCCD
    // stream为空时使用rts内部set device后的默认stream
    rtError_t ret = rtRDMADBSend(dbindex, dbinfo, stream);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rtRDMADBSend]errNo[0x%016llx] rt rdma send fail, "
            "return[%d]. para: dbindex[%u] dbinfo[%llu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dbindex, dbinfo),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    static auto funcPtr = (rtError_t(*)(uint32_t, uint64_t, rtStream_t))g_dlRts.Handle<RT_RDMA_DB_SEND>();
    CHK_PTR_NULL(funcPtr);
    rtError_t ret = funcPtr(dbindex, dbinfo, stream);
    HCCL_DEBUG("Call rtRDMADBSend, return value[%d]", ret);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR("rtRDMADBSend errNo[0x%016llx] RDMADBSend fail, return[%d]", HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#endif
}

HcclResult hrtSetLocalDeviceSatMode([[maybe_unused]] aclrtFloatOverflowMode floatOverflowMode)
{
#ifdef HCCD
    g_deviceSatMode = floatOverflowMode;
#endif
    return HCCL_SUCCESS;
}

HcclResult hrtGetDeviceSatMode(aclrtFloatOverflowMode* floatOverflowMode)
{
    CHK_PTR_NULL(floatOverflowMode);
#ifndef HCCD
    aclError ret = aclrtGetDeviceSatMode(floatOverflowMode);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[hrtGetDeviceSatMode]rt get dev satmode failed,"), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    if (g_workModeAicpu) {
        *floatOverflowMode = g_deviceSatMode;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[hrtGetDeviceSatMode]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyGetAddr([[maybe_unused]] HcclRtNotify signal, [[maybe_unused]] u64* notifyAddr)
{
#ifndef HCCD
    uint64_t* const addr = reinterpret_cast<uint64_t*>(notifyAddr);
    rtError_t ret = rtGetNotifyAddress(signal, addr);
    HCCL_DEBUG("Call rtGetNotifyAddress, signal[%p], notifyAddr[%016llx]", signal, *notifyAddr);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[rtGetNotifyAddress]rt get notify address failed."), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyGetAddr]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetNotifyID([[maybe_unused]] HcclRtNotify signal, [[maybe_unused]] u32* notifyID)
{
#ifndef HCCD
    CHK_PTR_NULL(signal);
    CHK_PTR_NULL(notifyID);
    aclError ret = aclrtGetNotifyId(signal, notifyID);
    HCCL_DEBUG("Call aclrtGetNotifyId signal:%p, notifyID:%u.", signal, *notifyID);
    CHK_PRT_RET(ret != ACL_SUCCESS, HCCL_ERROR("[hrtGetNotifyID]rt get notify id failed."), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetNotifyID]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetDeviceInfo(
    [[maybe_unused]] u32 deviceId, [[maybe_unused]] HcclRtDeviceModuleType hcclModuleType,
    [[maybe_unused]] HcclRtDeviceInfoType hcclInfoType, [[maybe_unused]] s64& val)
{
#ifndef HCCD
    static const std::map<HcclRtDeviceInfoType, aclrtDevAttr> systemInfoTypeMap = {
        {HcclRtDeviceInfoType::HCCL_INFO_TYPE_PHY_CHIP_ID, ACL_DEV_ATTR_PHY_CHIP_ID},
        {HcclRtDeviceInfoType::HCCL_INFO_TYPE_SDID, ACL_DEV_ATTR_SUPER_POD_DEVIDE_ID},
        {HcclRtDeviceInfoType::HCCL_INFO_TYPE_SERVER_ID, ACL_DEV_ATTR_SUPER_POD_SERVER_ID},
        {HcclRtDeviceInfoType::HCCL_INFO_TYPE_SUPER_POD_ID, ACL_DEV_ATTR_SUPER_POD_ID},
        {HcclRtDeviceInfoType::HCCL_INFO_TYPE_CUST_OP_ENHANCE, ACL_DEV_ATTR_CUST_OP_PRIVILEGE},
    };
    CHK_PRT_RET(
        hcclModuleType != HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
        HCCL_ERROR("[hrtGetDeviceInfo]Unsupported moduleType[%d].", hcclModuleType), HCCL_E_NOT_SUPPORT);

    auto it = systemInfoTypeMap.find(hcclInfoType);
    CHK_PRT_RET(
        it == systemInfoTypeMap.end(),
        HCCL_ERROR("[hrtGetDeviceInfo]Unsupported infoType[%d] for moduleType=SYSTEM.", hcclInfoType),
        HCCL_E_NOT_SUPPORT);
    aclrtDevAttr attr = it->second;

    aclError ret = aclrtGetDeviceInfo(deviceId, attr, reinterpret_cast<int64_t*>(&val));
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[hrtGetDeviceInfo]rt get device info failed. ret[%d], attr[%d], val[%ld]", ret, attr, val),
        HCCL_E_RUNTIME);
    HCCL_DEBUG("Call aclrtGetDeviceInfo, ret[%d], attr[%d], val[%ld]", ret, attr, val);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDeviceInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetRdmaDoorbellAddr([[maybe_unused]] u32 dbIndex, [[maybe_unused]] u64& dbAddr)
{
#ifndef HCCD
    s32 devLogID = 0;
    s64 chipID = 0;
    static std::mutex devChipIdMapSpinMutex; // devLogID 和 chipID 关系表的读写互斥锁

    CHK_RET(hrtGetDevice(&devLogID));

    // 读写表前先加锁
    std::unique_lock<std::mutex> lockDevChipIdMap(devChipIdMapSpinMutex);

    if (g_deviceChipIdMap.find(devLogID) != g_deviceChipIdMap.end()) {
        // 若已有记录，则直接获取chipID
        chipID = g_deviceChipIdMap[devLogID];
    } else {
        CHK_RET(hrtGetDeviceInfo(
            devLogID, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_PHY_CHIP_ID, chipID));
        g_deviceChipIdMap[devLogID] = chipID;
    }
    // 解锁
    lockDevChipIdMap.unlock();

    u64 roceBaseAddr = 0;
    u64 roceVfDbCfg0Reg = 0;
    u64 chipAddrOffset = 0;
    u64 dieAddrOffset = 0;
    u32 dbDieIdMask = 0;
    u32 dbDieIdShift = 0;

    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType == DevType::DEV_TYPE_910_93) {
        HCCL_DEBUG(
            "[hrtGetRdmaDoorbellAddr] The roceBaseAddr and dieAddrOffset has changed, when deviceType is 910_93.");
        // 910_93 HCCS_SW 组网
        roceBaseAddr = 0x202000000000ULL;
        roceVfDbCfg0Reg = 0x230ULL;
        chipAddrOffset = 0x20000000000ULL;
        dieAddrOffset = 0x10000000000ULL;
        dbDieIdMask = 0x00ff0000;
        dbDieIdShift = 16; // 16 is dbDieIdShift
    } else {
        roceBaseAddr = 0x2000000000ULL;
        roceVfDbCfg0Reg = 0x230ULL;
        chipAddrOffset = 0x80000000000ULL;
        dieAddrOffset = 0x10000000000ULL;
        dbDieIdMask = 0x00ff0000;
        dbDieIdShift = 16; // 16 is dbDieIdShift
    }

    dbAddr = roceBaseAddr + roceVfDbCfg0Reg + chipAddrOffset * chipID
             + dieAddrOffset * ((dbIndex & dbDieIdMask) >> dbDieIdShift);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetRdmaDoorbellAddr]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetDevArgsAddr(
    [[maybe_unused]] rtStream_t stm, [[maybe_unused]] rtArgsEx_t* argsInfo, [[maybe_unused]] void** devArgsAddr,
    [[maybe_unused]] void** argsHandle)
{
#ifndef HCCD
    rtError_t ret = rtGetDevArgsAddr(stm, argsInfo, devArgsAddr, argsHandle);
    CHK_PRT_RET(ret != RT_ERROR_NONE, HCCL_ERROR("[rtGetDevArgsAddr]rtKernel Get Addr failed."), HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetDevArgsAddr]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtMemSet([[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] uint64_t count)
{
#ifndef HCCD
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemSet] hrtThreadExchangeCaptureMode return [%d]", hcclRet));
    aclError ret = aclrtMemset(dst, destMax, 0, count);

    HCCL_DEBUG("Call aclrtMemset, return value[%d]", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async set failed, "
            "return[%d], para: dstAddr[%p], destMax[%llu], count[%llu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, dst, destMax, count),
        HCCL_E_RUNTIME);
    hcclRet = hrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_ERROR("[hrtMemSet] hrtThreadExchangeCaptureMode return [%d]", hcclRet));
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemSet]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtEventCreateWithFlag([[maybe_unused]] aclrtEvent* evt)
{
#ifndef HCCD
    CHK_PTR_NULL(evt);

    aclError ret = aclrtCreateEventWithFlag(evt, ACL_EVENT_DEVICE_USE_ONLY);
    HCCL_DEBUG("Call aclrtCreateEventWithFlag, return value[%d] evt[%p].", ret, evt);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtCreateEventWithFlag]rtEvent Create WithFlags failed "
            "value[%d] evt[%p].",
            ret, evt),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[aclrtCreateEventWithFlag]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyCreateWithFlag([[maybe_unused]] int32_t deviceId, [[maybe_unused]] aclrtNotify* notify)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);

    aclError ret = aclrtCreateNotify(notify, ACL_NOTIFY_DEVICE_USE_ONLY);
    HCCL_DEBUG("Call aclrtCreateNotify deviceId:[%d], return value[%d].", deviceId, ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtCreateNotify]rtNotify Create WithFlags failed "
            "deviceId:%d, notify:%p, return value[%d].",
            deviceId, notify, ret),
        HCCL_E_RUNTIME);

    u32 notifyId = 0;
    CHK_RET(hrtGetNotifyID(*notify, &notifyId));
    PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[aclrtCreateNotify]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtIpcOpenNotifyWithFlag(
    [[maybe_unused]] rtNotify_t* notify, [[maybe_unused]] const u8* name, [[maybe_unused]] uint32_t flags)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(name);

    rtError_t ret = aclrtNotifyImportByKey(notify, reinterpret_cast<const char*>(name), static_cast<uint64_t>(flags));
    HCCL_DEBUG(
        "Call aclrtNotifyImportByKey notify:[%p], name:[%s], flags[%u], return value[%d].", notify, name, flags, ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtNotifyImportByKey]rtIpc OpenNotify WithFlags failed "
            "notify:%p, name:%s, flags[%u], return value[%d].",
            notify, name, flags, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[aclrtNotifyImportByKey]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyImportByKey([[maybe_unused]] rtNotify_t* notify, [[maybe_unused]] const u8* name)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(name);

    constexpr u64 notifyFlag = 0;
    auto ret = aclrtNotifyImportByKey(notify, reinterpret_cast<const char*>(name), notifyFlag);
    HCCL_DEBUG(
        "Call aclrtNotifyImportByKey notify:[%p], name:[%s], flags[%u], return value[%d].", notify, name, notifyFlag,
        ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[hrtNotifyImportByKey]aclrtIpc OpenNotify WithFlags failed "
            "notify:%p, name:%s, return value[%d].",
            notify, name, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtNotifyImportByKey]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyGetPhyInfo(
    [[maybe_unused]] rtNotify_t notify, [[maybe_unused]] uint32_t* phyDevId, [[maybe_unused]] uint32_t* tsId)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(phyDevId);
    CHK_PTR_NULL(tsId);

    rtError_t ret = rtNotifyGetPhyInfo(notify, phyDevId, tsId);
    HCCL_DEBUG(
        "Call rtNotifyGetPhyInfo notify:%p, phyDevId:%u, tsId:%u, return value[%d].", notify, *phyDevId, *tsId, ret);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rtNotifyGetPhyInfo]rtNotify GetPhy failed notify:%p, phyDevId:%u, "
            "tsId:%u, return value[%d].",
            notify, *phyDevId, *tsId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[rtNotifyGetPhyInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtNotifyGetPhyInfoExt([[maybe_unused]] rtNotify_t notify, [[maybe_unused]] rtNotifyPhyInfo* notifyInfo)
{
#ifndef HCCD
    CHK_PTR_NULL(notify);
    CHK_PTR_NULL(notifyInfo);

    rtError_t ret = rtNotifyGetPhyInfoExt(notify, notifyInfo);
    HCCL_DEBUG(
        "Call rtNotifyGetPhyInfoExt notify:%p, phyId:%u, tsId:%u, idType:%u, notifyid:[%u], flag:[%u], "
        "return value[%d].",
        notify, notifyInfo->phyId, notifyInfo->tsId, notifyInfo->idType, notifyInfo->shrId, notifyInfo->flag, ret);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "Call rtNotifyGetPhyInfoExt notify:%p, phyId:%u, tsId:%u, "
            "idType:%u, notifyid:[%u], flag:[%u], return value[%d].",
            notify, notifyInfo->phyId, notifyInfo->tsId, notifyInfo->idType, notifyInfo->shrId, notifyInfo->flag, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[rtNotifyGetPhyInfoExt]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetEventID([[maybe_unused]] rtEvent_t event, [[maybe_unused]] uint32_t* eventId)
{
#ifndef HCCD
    CHK_PTR_NULL(event);
    CHK_PTR_NULL(eventId);

    aclError ret = aclrtGetEventId(event, eventId);
    HCCL_DEBUG("Call aclrtGetEventId event:%p, eventId:%u, return value[%d].", event, *eventId, ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[aclrtGetEventId]rtGet EventID failed event:%p, eventId:%u "
            "return value[%d].",
            event, *eventId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetEventID]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamGetSqid([[maybe_unused]] const rtStream_t stm, [[maybe_unused]] uint32_t* sqId)
{
#ifndef HCCD
    CHK_PTR_NULL(stm);
    CHK_PTR_NULL(sqId);

    rtError_t ret = rtStreamGetSqid(stm, sqId);
    HCCL_DEBUG("Call rtStreamGetSqid stm:%p, sqId:%u value[%d].", stm, *sqId, ret);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rtStreamGetSqid]rtStream Get Sqid failed stm:%p, "
            "sqId:%u value[%d].",
            stm, *sqId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[rtStreamGetSqid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtStreamGetCqid(
    [[maybe_unused]] const rtStream_t stm, [[maybe_unused]] uint32_t* cqId, [[maybe_unused]] uint32_t* logicCqId)
{
#ifndef HCCD
    CHK_PTR_NULL(stm);
    CHK_PTR_NULL(cqId);
    CHK_PTR_NULL(logicCqId);

    rtError_t ret = rtStreamGetCqid(stm, cqId, logicCqId);
    HCCL_DEBUG("Call rtStreamGetCqid stm:%p, cqId:%u, logicCqId:%u value[%d].", stm, *cqId, *logicCqId, ret);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[rtStreamGetCqid]rtStream Get logicCqId failed stm:%p, "
            "cqId:%u logicCqId:%u value[%d].",
            stm, *cqId, *logicCqId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[rtStreamGetCqid]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

// 要求设置过 SetDevice
HcclResult hrtResourceClean()
{
#ifndef HCCD
    // rts 接口内部通过 GetDevice 获得 deviceId
    rtError_t ret = rtNotifyResetAll();
    HCCL_DEBUG("Call aclrtNotifyBatchReset, ret: %d", ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS, HCCL_ERROR("[hrtResourceClean]aclrtNotifyBatchReset failed, return value[%d].", ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtResourceClean]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtGetHccsPortNum([[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] s32& num)
{
#ifndef HCCD
    constexpr s32 HCCS_PORT_NUM_UNKNOWN = -1;
    constexpr s32 HCCS_PORT_NUM_910_93_UNKNOWN = -1;
    constexpr s32 HCCS_PORT_NUM_910_93_6 = 6;
    constexpr s32 HCCS_PORT_NUM_910_93_7 = 7;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V1_2_4_4 = 0x1c;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V1_4_8_8 = 0x1d;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V2_4_8_7 = 0x18;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V2_2_8_7 = 0x19;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V3_4_8_7 = 0x14;
    constexpr int64_t MAINBORDID_910_93_PRODUCT_V3_2_8_7 = 0x15;
    DevType deviceType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(deviceType));
    if (deviceType != DevType::DEV_TYPE_910_93) {
        num = HCCS_PORT_NUM_UNKNOWN;
        HCCL_WARNING("[hrtGetHccsPortNum]deviceType: %d, does not support this interface.", deviceType);
        return HCCL_E_NOT_SUPPORT;
    }
    int64_t val = 0;
    aclError ret = aclrtGetDeviceInfo(deviceLogicId, ACL_DEV_ATTR_MAINBOARD_ID, &val);
    HCCL_DEBUG("Call aclrtGetDeviceInfo deviceLogicId:%u, val: %lld, ret:%d.", deviceLogicId, val, ret);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR(
            "[rtGetHccsPortNum]rtGet Hccs Port Num failed deviceLogicId:%u, "
            "return value[%d].",
            deviceLogicId, ret),
        HCCL_E_RUNTIME);
    switch (val) {
        case MAINBORDID_910_93_PRODUCT_V1_2_4_4:
            num = HCCS_PORT_NUM_910_93_6;
            break;
        case MAINBORDID_910_93_PRODUCT_V1_4_8_8:
            num = HCCS_PORT_NUM_910_93_6;
            break;
        case MAINBORDID_910_93_PRODUCT_V2_4_8_7:
            num = HCCS_PORT_NUM_910_93_7;
            break;
        case MAINBORDID_910_93_PRODUCT_V2_2_8_7:
            num = HCCS_PORT_NUM_910_93_7;
            break;
        case MAINBORDID_910_93_PRODUCT_V3_4_8_7:
            num = HCCS_PORT_NUM_910_93_7;
            break;
        case MAINBORDID_910_93_PRODUCT_V3_2_8_7:
            num = HCCS_PORT_NUM_910_93_7;
            break;
        default:
            num = HCCS_PORT_NUM_910_93_UNKNOWN;
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtGetHccsPortNum]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtThreadExchangeCaptureMode([[maybe_unused]] aclmdlRICaptureMode* mode)
{
#ifndef HCCD
    HCCL_DEBUG("Call aclmdlRICaptureThreadExchangeMode mode after: %d", *mode);
    aclError ret = aclmdlRICaptureThreadExchangeMode(mode);
    HCCL_DEBUG("Call aclmdlRICaptureThreadExchangeMode mode before: %d, ret: %d", *mode, ret);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_INFO("[hrtThreadExchangeCaptureMode]aclmdlRICaptureThreadExchangeMode not support!");
        return HCCL_E_NOT_SUPPORT;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[hrtThreadExchangeCaptureMode]aclmdlRICaptureThreadExchangeMode "
                "failed mode:%d, return value[%d].",
                *mode, ret),
            HCCL_E_RUNTIME);
    }
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtThreadExchangeCaptureMode]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtCacheLastTaskExtendInfo([[maybe_unused]] const char* tag, [[maybe_unused]] size_t tagLen)
{
#ifndef HCCD
    auto funcPtr = (aclError(*)(const char*, size_t))g_dlAcl.Handle<ACL_RT_CACHE_LAST_TASK_EXTEND_INFO>();
    if (funcPtr == nullptr) {
        HCCL_INFO("[hrtCacheLastTaskExtendInfo] aclrtCacheLastTaskExtendInfo not supported.");
        return HCCL_E_NOT_SUPPORT;
    }
    aclError ret = funcPtr(tag, tagLen);
    CHK_PRT_RET(
        ret != ACL_SUCCESS,
        HCCL_ERROR("[hrtCacheLastTaskExtendInfo] call aclrtCacheLastTaskExtendInfo fail, return[%d]", ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtCacheLastTaskExtendInfo]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

__attribute__((constructor)) void CallBackInitRts()
{
    g_deviceType = DevType::DEV_TYPE_COUNT;
    g_deviceLogicId = INVALID_INT;
    g_devicePhyId = INVALID_UINT;
    HCCL_RUN_INFO(
        "[adapter_rts.cc][CallBackInitRts] g_deviceType [%d] g_deviceLogicId [%d] g_devicePhyId [%d]", g_deviceType,
        g_deviceLogicId, g_devicePhyId);
}

HcclResult hrtMemcpyEx(
    [[maybe_unused]] void* dst, [[maybe_unused]] uint64_t destMax, [[maybe_unused]] void* src,
    [[maybe_unused]] uint64_t count, [[maybe_unused]] rtMemcpyKind_t kind)
{
#ifndef HCCD
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);
    CHK_PRT_RET(
        count > destMax, HCCL_ERROR("[hrtMemcpyEx] count[%llu] exceeds destMax[%llu].", count, destMax), HCCL_E_PARA);
    rtError_t ret = rtMemcpyEx(dst, destMax, src, count, kind);
    HCCL_DEBUG(
        "[hrtMemcpyEx] Call rtMemcpyEx, ret[%d], dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], "
        "rtKind[%d]",
        ret, dst, destMax, src, count, kind);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[hrtMemcpyEx] rtMemcpyEx failed, return[%d], dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], "
            "rtKind[%d].",
            ret, dst, destMax, src, count, kind),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[hrtMemcpyEx] Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
