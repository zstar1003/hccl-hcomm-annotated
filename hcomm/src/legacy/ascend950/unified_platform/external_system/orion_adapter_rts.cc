/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cast_utils.h"
#include "orion_adapter_rts.h"
#include "hccl_common_v2.h"
#include "runtime_api_exception.h"
#include "exception_util.h"
#include "invalid_params_exception.h"
#include "log.h"
#include "config_plf_log_v2.h"
#include "acl/acl_rt.h"
#include "driver/ascend_hal.h"
#include "not_support_exception.h"
#include "adapter_error_manager_pub.h"
#include "dlrts_function_v2.h"

using namespace std;
namespace Hccl {

HcclResult HrtThreadExchangeCaptureMode(aclmdlRICaptureMode* mode);
constexpr u32 TOKEN_ID_RIGHT_SHIF = 8; // URMA_TOKEN_ID_RIGHT_SHIF，因URMA配置原因需要右移8位
namespace {
    constexpr char RT_SET_XPU_DEVICE[] = "rtSetXpuDevice";
    constexpr char RT_RESET_XPU_DEVICE[] = "rtResetXpuDevice";
} // namespace
const std::unordered_map<std::string, DevType> SOC_VER_CONVERT{
    {"Ascend310P1", DevType::DEV_TYPE_V51_310_P1},
    {"Ascend310P3", DevType::DEV_TYPE_V51_310_P3},
    {"Ascend910", DevType::DEV_TYPE_910A},
    {"Ascend910A", DevType::DEV_TYPE_910A},
    {"Ascend910B", DevType::DEV_TYPE_910A},
    {"Ascend910ProA", DevType::DEV_TYPE_910A},
    {"Ascend910ProB", DevType::DEV_TYPE_910A},
    {"Ascend910PremiumA", DevType::DEV_TYPE_910A},
    {"Ascend910B1", DevType::DEV_TYPE_910A2},
    {"Ascend910B2", DevType::DEV_TYPE_910A2},
    {"Ascend910B3", DevType::DEV_TYPE_910A2},
    {"Ascend910B4", DevType::DEV_TYPE_910A2},
    {"Ascend910B4-1", DevType::DEV_TYPE_910A2},
    {"Ascend910_9391", DevType::DEV_TYPE_910A3},
    {"Ascend910_9381", DevType::DEV_TYPE_910A3},
    {"Ascend910_9392", DevType::DEV_TYPE_910A3},
    {"Ascend910_9382", DevType::DEV_TYPE_910A3},
    {"Ascend910_9372", DevType::DEV_TYPE_910A3},
    {"Ascend910_9362", DevType::DEV_TYPE_910A3},
    {"Ascend910_9363", DevType::DEV_TYPE_910A3},
    {"nosoc", DevType::DEV_TYPE_NOSOC}};

// 添加编译宏，防止返回82类型芯片造成已有UT失效
DevType HrtGetDeviceType()
{
    std::string targetChipVerStr;
    HrtGetSocVer(targetChipVerStr);

    HCCL_INFO("[HrtGetDeviceType]targetChipVerStr = %s.", targetChipVerStr.c_str());
    if (targetChipVerStr.find("Ascend950") != std::string::npos) {
        HCCL_INFO("[HrtGetDeviceType]DeviceType = DevType::DEV_TYPE_950.");
        return DevType::DEV_TYPE_950;
    }

    if (targetChipVerStr.find("Ascend960") != std::string::npos
        || targetChipVerStr.find("ascend960") != std::string::npos) {
        HCCL_INFO("[HrtGetDeviceType]DeviceType = DevType::DEV_TYPE_960.");
        return DevType::DEV_TYPE_960;
    }

    auto iter = SOC_VER_CONVERT.find(targetChipVerStr);
    if (iter == SOC_VER_CONVERT.end()) {
        string msg = StringFormat(
            "[Get][DeviceType]errNo[0x%016llx] rtGetSocVersion get "
            "illegal chipver, chip_ver[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), targetChipVerStr.c_str());
        MACRO_THROW(RuntimeApiException, msg);
    }
    return iter->second;
}

DevId HrtGetDevicePhyIdByUserDevId(s32 userDevId)
{
    DevType deviceType = HrtGetDeviceType();
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        return 0;
    }

    s32 devicePhyId = 0;
    aclError ret = aclrtGetPhyDevIdByUserDevId(userDevId, &devicePhyId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][DevicePhyId]errNo[0x%016llx] rtGet device PhyId by "
            "userDevId failed. return[%d], "
            "para: userDevId[%d], phyId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_DRV), ret, userDevId, devicePhyId);
        MACRO_THROW(RuntimeApiException, msg);
    }
    HCCL_INFO("[HrtGetDevicePhyIdByUserDevId]userDevId=%d, devicePhyId=%d.", userDevId, devicePhyId);
    return static_cast<DevId>(devicePhyId);
}

HcclResult HrtGetUserDevIdByPhyDevId(s32 devicePhyId, s32& userDevId)
{
    DevType deviceType = HrtGetDeviceType();
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        userDevId = 0;
        return HCCL_SUCCESS;
    }

    s32 tmpUserDevId = 0;
    aclError ret = aclrtGetUserDevIdByPhyDevId(devicePhyId, &tmpUserDevId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][UserDevId]errNo[0x%016llx] rtGet userDevId by PhyDevId failed. "
            "return[%d], para: phyId[%d], userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, devicePhyId, tmpUserDevId);
        HCCL_ERROR("%s", msg.c_str());
        return HCCL_E_RUNTIME;
    }
    userDevId = tmpUserDevId;
    HCCL_INFO("[HrtGetUserDevIdByPhyDevId]devicePhyId=%d, userDevId=%d.", devicePhyId, userDevId);
    return HCCL_SUCCESS;
}

s32 HrtDeviceGetBareTgid()
{
    s32 pid = 0;
    aclError ret = aclrtDeviceGetBareTgid(&pid);
    HCCL_INFO("Call rtDeviceGetBareTgid, return value[%d], rtGet pid[%d].", ret, pid);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][BareTgid]errNo[0x%016llx] rtGet pid fail. "
            "return[%d], rtGet pid[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, pid);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return pid;
}

void HrtGetSocVer(std::string& socName)
{
    const char* socNamePtr = aclrtGetSocName();
    if (socNamePtr == nullptr) {
        string msg = StringFormat(
            "[Get][SocVer]errNo[0x%016llx] rtGet deviceVer failed.", HCCL_ERROR_CODE((HcclResult::HCCL_E_RUNTIME)));
        MACRO_THROW(RuntimeApiException, msg);
    }
    socName = socNamePtr;
}

s32 HrtGetDevice()
{
    s32 userDevId = 0;
    aclError ret = aclrtGetDevice(&userDevId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][Device]errNo[0x%016llx] rtGet device fail, "
            "please make sure that device is set. return[%d], para:userDevId[%d]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, userDevId);
        MACRO_THROW(RuntimeApiException, msg);
    }
    HCCL_INFO("[HrtGetDevice]userDevId=%d.", userDevId);
    return userDevId;
}

void HrtSetDevice(s32 deviceLogicId)
{
    aclError ret = aclrtSetDevice(deviceLogicId);
    HCCL_INFO("Call rtSetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Set][Device]errNo[0x%016llx] rtSet device fail. "
            "return[%d], para:userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtResetDevice(s32 deviceLogicId)
{
    aclError ret = aclrtResetDevice(deviceLogicId);
    HCCL_INFO("Call aclrtResetDevice, return value[%d], para: device_id[%d].", ret, deviceLogicId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Reset][Device]errNo[0x%016llx] rtReset device fail. "
            "return[%d], para: userDevId[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

u32 HrtGetDeviceCount()
{
    u32 count = 0;
    aclError ret = aclrtGetDeviceCount(&count);
    HCCL_INFO("Call rtGetDeviceCount, return value[%d], para: count[%u].", ret, count);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][DeviceCount]errNo[0x%016llx] rtGet device count fail. "
            "return[%d], para:count[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, count);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return count;
}

HcclResult HrtGetLogicDevIdByUserDevId(s32 userDevId, s32& logicDevId)
{
    DevType deviceType = HrtGetDeviceType();
    if (deviceType == DevType::DEV_TYPE_NOSOC) {
        logicDevId = 0;
        return HCCL_SUCCESS;
    }

    aclError aclRet = aclrtGetLogicDevIdByUserDevId(userDevId, &logicDevId); // userDevId 转 logicDevId
    if (aclRet != ACL_SUCCESS) {
        HCCL_ERROR("aclrtGetLogicDevIdByUserDevId failed, userDevId: %d, ret: %d", userDevId, aclRet);
        return HCCL_E_RUNTIME;
    }

    return HCCL_SUCCESS;
}

constexpr char RTS_SO_NAME[] = "libruntime.so";
DlRtsFunctionV2<RTS_SO_NAME> g_dlRts;
HcclResult HrtResetXpuDevice(uint32_t devType, const uint32_t devId)
{
    static auto funcPtr = ReinterpretAs<rtError_t (*)(uint32_t, const uint32_t)>(g_dlRts.Handle<RT_RESET_XPU_DEVICE>());
    CHK_PTR_NULL(funcPtr);
    rtError_t ret = funcPtr(devType, devId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] reset xpu device failed, devType[%u], devId[%u], return[%d].", __func__, devType, devId, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtSetXpuDevice(uint32_t devType, const uint32_t devId)
{
    static auto funcPtr = ReinterpretAs<rtError_t (*)(uint32_t, const uint32_t)>(g_dlRts.Handle<RT_SET_XPU_DEVICE>());
    CHK_PTR_NULL(funcPtr);
    rtError_t ret = funcPtr(devType, devId);
    if (ret != RT_ERROR_NONE) {
        HCCL_ERROR("[%s] set xpu device failed, devType[%u], devId[%u], return[%d].", __func__, devType, devId, ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

s32 HrtGetStreamId(aclrtStream ptr)
{
    s32 streamId;
    aclError ret = aclrtStreamGetId(ptr, &streamId);
    HCCL_INFO("Call aclrtStreamGetId, ptr[%p] return value[%d] streamId[%d].", ptr, ret, streamId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][StreamId]errNo[0x%016llx]. "
            "rt get stream ID fail. ptr[%p], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }

    return streamId;
}

u64 HrtStreamGetMode(HcclRtStream const ptr)
{
    if (ptr == nullptr) {
        throw RuntimeApiException(StringFormat("ptr is null, call aclrtGetStreamAttribute failed, ptr=%p", ptr));
    }
    u64 stmMode = 0;
    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(ptr, &streamId);
    HCCL_DEBUG("[HrtStreamGetMode] ptr[%p], ret[%d].", ptr, ret);
    aclrtStreamAttrValue value;
    ret = aclrtGetStreamAttribute(ptr, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    stmMode = value.failureMode;
    HCCL_INFO("Call rtStreamGetMode return value[%d]. stmMode[%llu].", ret, stmMode);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Stream][GetMode]errNo[0x%016llx] rtStreamGetMode error. "
            "ptr[%p], stmMode[%llu], streamId[%d], rtRet[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, stmMode, streamId, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return static_cast<u64>(stmMode);
}

void HrtStreamSetMode(HcclRtStream streamPtr, const uint64_t stmMode)
{
    if (streamPtr == nullptr) {
        throw RuntimeApiException(StringFormat("ptr is null, call aclrtSetStreamAttribute failed, ptr=%p", streamPtr));
    }
    s32 streamId = -1;
    aclError ret = aclrtStreamGetId(streamPtr, &streamId);
    HCCL_DEBUG("Call aclrtStreamGetId, return value[%d].", ret);
    aclrtStreamAttrValue value;
    value.failureMode = stmMode;
    ret = aclrtSetStreamAttribute(streamPtr, ACL_STREAM_ATTR_FAILURE_MODE, &value);
    HCCL_INFO("[HrtStreamSetMode]streamPtr[%p], stmMode[%llu], ret[%d].", streamPtr, stmMode, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Stream][SetMode]errNo[0x%016llx] rtStreamSetMode error. "
            "streamPtr[%p], rtRet[%d], stmMode[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), streamPtr, ret, stmMode);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

HcclResult HrtGetDeviceInfo(uint32_t deviceLogicId, int32_t moduleType, aclrtDevAttr infoType, int64_t& val)
{
    if (moduleType != DEV_MODULE_TYPE::MODULE_TYPE_SYSTEM) {
        THROW<NotSupportException>(StringFormat("[hrtGetDeviceInfo]Unsupported moduleType[%d].", moduleType));
    }
    aclError ret = aclrtGetDeviceInfo(deviceLogicId, infoType, ReinterpretAs<int64_t*>(&val));
    HCCL_INFO(
        "[HrtGetDeviceInfo]deviceLogicId[%u], moduleType[%d], infoType[%d], return[%d], val[%lld].", deviceLogicId,
        moduleType, infoType, ret, val);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR(
            "[HrtGetDeviceInfo]errNo[0x%016llx] rt get device info failed, "
            "deviceLogicId=%u, moduleType=%d, infoType=%d",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), deviceLogicId, moduleType, infoType);
        return HcclResult::HCCL_E_RUNTIME;
    }
    return HcclResult::HCCL_SUCCESS;
}

constexpr uint64_t POD_MAINBOARD = 0x0;
constexpr uint64_t A_K_SERVER_MAINBOARD = 0x1;
constexpr uint64_t A_X_SERVER_MAINBOARD = 0x2;
constexpr uint64_t PCIE_STD_MAINBOARD = 0x3;
constexpr uint64_t RSV1_MAINBOARD = 0x4;
constexpr uint64_t RSV2_MAINBOARD = 0x5;
constexpr uint64_t EQUIP_MAINBOARD = 0x6;
constexpr uint64_t EVB_MAINBOARD = 0x7;

const std::unordered_map<uint64_t, HcclMainboardId> rtMainboardIdToHcclMainboardId
    = {{POD_MAINBOARD, HcclMainboardId::MAINBOARD_POD},
       {A_K_SERVER_MAINBOARD, HcclMainboardId::MAINBOARD_A_K_SERVER},
       {A_X_SERVER_MAINBOARD, HcclMainboardId::MAINBOARD_A_X_SERVER},
       {PCIE_STD_MAINBOARD, HcclMainboardId::MAINBOARD_PCIE_STD},
       {RSV1_MAINBOARD, HcclMainboardId::MAINBOARD_RSV},
       {RSV2_MAINBOARD, HcclMainboardId::MAINBOARD_RSV},
       {EQUIP_MAINBOARD, HcclMainboardId::MAINBOARD_EQUIPMENT},
       {EVB_MAINBOARD, HcclMainboardId::MAINBOARD_EVB}};

/*
 * 获取Mainboard ID 5-7位，输出整机形态枚举值
 * Mainboard ID描述说明
 * Mainboard ID采用了16bit，区分形态，主从，以及端口配置
 * bit[7:5] 区分整机形态(当前POD和EVB没有区分A+X或A+K)
 * {
 *  000: 天成 POD
 *  001: A+K Server
 *  010: A+X Server
 *  011: PCIE标卡
 *  100-101: RSV
 *  110: 装备
 *  111: EVB
 * }
 * bit[4:1] 整机形态细分
 * {
 *  0000-1111
 * }
 * bit[0] 主从或池化
 * {
 *  0: 主从（NPU作为某个Host的从设备，Host主控）
 *  1: 池化（NPU作为资源池，其它Host对等访问）
 * }
 */
HcclResult HrtGetMainboardId(uint32_t deviceLogicId, HcclMainboardId& hcclMainboardId)
{
    constexpr int32_t moduleType = DEV_MODULE_TYPE::MODULE_TYPE_SYSTEM;
    constexpr aclrtDevAttr infoType = aclrtDevAttr::ACL_DEV_ATTR_MAINBOARD_ID;
    constexpr uint64_t BITS_5 = 5;
    constexpr uint64_t MASK_7 = 0x7;
    int64_t val = 0;
    CHK_RET(HrtGetDeviceInfo(deviceLogicId, moduleType, infoType, val));
    HCCL_INFO("[HrtGetMainboardId] deviceLogicId[%u] val[%lld].", deviceLogicId, val);
    CHK_PRT_RET(val < 0, HCCL_ERROR("[HrtGetMainboardId]val[%lld] < 0", val), HCCL_E_RUNTIME);
    uint64_t mainboardId = (static_cast<uint64_t>(val) >> BITS_5) & MASK_7; // 提取val的5-7位，判断整机形态
    auto it = rtMainboardIdToHcclMainboardId.find(mainboardId);
    if (it != rtMainboardIdToHcclMainboardId.end()) {
        hcclMainboardId = it->second;
    } else {
        hcclMainboardId = HcclMainboardId::MAINBOARD_OTHERS;
    }
    HCCL_INFO(
        "[HrtGetMainboardId] deviceLogicId[%u] mainboardId[%llu] hcclMainboardId[%s].", deviceLogicId, mainboardId,
        hcclMainboardId.Describe().c_str());
    return HcclResult::HCCL_SUCCESS;
}

aclrtStream HrtStreamCreateWithFlags(uint32_t priority, uint32_t flag)
{
    aclrtStream ptr = nullptr;
    aclError ret = aclrtCreateStreamWithConfig(&ptr, priority, flag);
    PLF_CONFIG_INFO(PLF_RES, "priority[%u], flags[%u], ptr[%p], ret[%d].", priority, flag, ptr, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Stream][CreateWithFlags]errNo[0x%016llx] rtStreamCreate error, "
            "rtRet[%d], flags[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, flag);
        MACRO_THROW(RuntimeApiException, msg);
    }

    return ptr;
}

void HrtStreamDestroy(aclrtStream ptr)
{
    HcclUs startut = TIME_NOW();
    if (GetPlfDebugConfigValue() & PLF_RES) {
        s32 deviceId = HrtGetDevice();
        s32 streamId = HrtGetStreamId(ptr);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Stream para: deviceId[%d] streamId[%d]", deviceId, streamId);
    }
    aclError ret = aclrtDestroyStreamForce(ptr);
    HCCL_INFO(
        "[HrtStreamDestroy] ptr[%p], ret[%d], take time [%lld]us.", ptr, ret,
        DURATION_US(TIME_NOW() - startut).count());
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Stream][Destroy]errNo[0x%016llx] rt stream Destroy fail. "
            "return[%d], ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtStreamActive(aclrtStream activeStream, aclrtStream stream)
{
    aclError ret = aclrtActiveStream(activeStream, stream);
    HCCL_INFO("[HrtStreamActive] activeStream[%p], stream[%p], ret[%d].", activeStream, stream, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Activate][Stream]errNo[0x%016llx] "
            "rt stream active fail. return[%d], active_stream=%p, stream=%p.",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, activeStream, stream);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

inline s32 GetMsTimeFromExecTimeout()
{
    constexpr s32 HCCL_EXEC_TIME_OUT_OFFSET_S = 5; // 避免与notifywait timeout时间冲突，增加5s的偏移值
    constexpr u32 TIME_S_TO_MS = 1000;
    s32 execTimeOut = 3000; // 从环境变量中获取超时时间
    s64 timeOutMs = (execTimeOut + HCCL_EXEC_TIME_OUT_OFFSET_S) * TIME_S_TO_MS;
    timeOutMs = (timeOutMs > 0x7FFFFFFF) ? 0x7FFFFFFF : timeOutMs;
    return static_cast<s32>(static_cast<u64>(timeOutMs) & (0x7FFFFFFF));
}

void HcclStreamSynchronize(HcclRtStream ptr)
{
    if (ptr == nullptr) {
        string msg = StringFormat("ptr is null, call aclrtSynchronizeStreamWithTimeout failed, ptr=%p", ptr);
        MACRO_THROW(RuntimeApiException, msg);
    }
    HCCL_INFO("[HcclStreamSynchronize] ptr[%p].", ptr);
    s32 timeout = GetMsTimeFromExecTimeout();
    aclError ret = aclrtSynchronizeStreamWithTimeout(ptr, timeout);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Synchronize][Stream]errNo[0x%016llx] rt "
            "streamsynchronizewithtimeout fail. return[%d], stream[%p], timeout[%d ms].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, timeout);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

/*
 *RT_MEMORY_TS：aclrtMallocForTaskScheduler
 *RT_MEMORY_DDR：ACL_MEM_TYPE_LOW_BAND_WIDTH
 *RT_MEMORY_HBM：ACL_MEM_TYPE_HIGH_BAND_WIDTH
 */
void* HrtMalloc(u64 size, aclrtMemType_t memType)
{
    aclError ret = ACL_SUCCESS;
    void* devPtr = nullptr;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    ret = aclrtMallocWithCfg(&devPtr, size, static_cast<aclrtMemMallocPolicy>(memType), &cfg);
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(
            PLF_RES, "ret[%d] size[%llu] memType[%d] devPtr[%p] deviceId[%d]", ret, size, memType, devPtr,
            HrtGetDevice());
    }
    if (ret == ACL_ERROR_RT_MEMORY_ALLOCATION) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"DeviceMemory", std::string("MallocWithCfg, size:") + std::to_string(size) + " bytes"}));
        string msg = StringFormat(
            "[Malloc][Mem]errNo[0x%016llx] aclrtMallocWithCfg failed, "
            "Reason: out of memory, return[%d], para: devPtrAddr[%p], size[%llu]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, devPtr, size);
        HCCL_ERROR("[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), msg.c_str());
        MACRO_THROW(RuntimeApiException, msg);
    }
    if (ret != ACL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"DeviceMemory", std::string("MallocWithCfg, size:") + std::to_string(size) + " bytes"}));
        string msg = StringFormat(
            "[Malloc][Mem]errNo[0x%016llx] aclrtMallocWithCfg failed, "
            "return[%d], para: devPtrAddr[%p], size[%llu]",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, devPtr, size);
        HCCL_ERROR("[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), msg.c_str());
        MACRO_THROW(RuntimeApiException, msg);
    }
    return devPtr;
}

void HrtFree(void* devPtr)
{
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(PLF_RES, "Free DevMem para: deviceId[%d] devPtr[%p]", HrtGetDevice(), devPtr);
    }
    aclError ret = aclrtFree(devPtr);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "[Free][Mem]errNo[0x%016llx] aclrtFree failed. "
            "return[%d], para: devPtrAddr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, devPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

HcclResult MemcpyKindTranslate(rtMemcpyKind_t kind, aclrtMemcpyKind* rtKind)
{
    switch (kind) {
        case rtMemcpyKind_t::RT_MEMCPY_HOST_TO_HOST: {
            *rtKind = ACL_MEMCPY_HOST_TO_HOST;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_HOST_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_HOST_TO_DEVICE;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_HOST: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_HOST;
            return HCCL_SUCCESS;
        }
        case rtMemcpyKind_t::RT_MEMCPY_DEVICE_TO_DEVICE: {
            *rtKind = ACL_MEMCPY_DEVICE_TO_DEVICE;
            return HCCL_SUCCESS;
        }
        default: {
            HCCL_ERROR("[MemcpyKindTranslate]Not support the memory copy type[%d].", kind);
            return HCCL_E_PARA;
        }
    }
}

void HrtMemcpy(void* dst, uint64_t destMax, const void* src, uint64_t count, rtMemcpyKind_t kind)
{
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemcpy] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    aclrtMemcpyKind rtKind = ACL_MEMCPY_DEFAULT;
    hcclRet = MemcpyKindTranslate(kind, &rtKind);
    aclError ret = aclrtMemcpy(dst, destMax, src, count, rtKind);
    HCCL_INFO("[HrtMemcpy] dst[%p], destMax[%llu], src[%p], count[%llu], ret[%d].", dst, destMax, src, count, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[SyncCopy][Mem]errNo[0x%016llx] rtMemcpy failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], kind[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind);
        MACRO_THROW(RuntimeApiException, msg);
    }
    hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemcpy] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
}

void HrtMemset(void* dst, uint64_t destMax, uint64_t count)
{
    HCCL_INFO("[HrtMemset] dst[%p], destMax[%llu], count[%llu].", dst, destMax, count);
    aclmdlRICaptureMode mode = aclmdlRICaptureMode::ACL_MODEL_RI_CAPTURE_MODE_RELAXED;
    HcclResult hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemSet] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
    aclError ret = aclrtMemset(dst, destMax, 0, count);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[SyncSet][Mem]errNo[0x%016llx] aclrtMemset failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], count[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, count);
        MACRO_THROW(RuntimeApiException, msg);
    }
    hcclRet = HrtThreadExchangeCaptureMode(&mode);
    CHK_PRT_CONT(
        hcclRet != HCCL_SUCCESS && hcclRet != HCCL_E_NOT_SUPPORT,
        HCCL_WARNING("[hrtMemSet] HrtThreadExchangeCaptureMode return [%d].", hcclRet));
}

void HrtIpcSetMemoryName(void* ptr, char_t* name, u64 ptrMaxLen, u32 nameMaxLen)
{
    aclError ret = aclrtIpcMemGetExportKey(ptr, ptrMaxLen, name, nameMaxLen, 1UL);
    HCCL_INFO(
        "Call aclrtIpcMemGetExportKey, return value[%d], para: ptr[%p], name[%s], byteCount[%llu], nameLen[%u]", ret,
        ptr, name, ptrMaxLen, nameMaxLen);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Set][IpcMemoryName]errNo[0x%016llx] rtSet Ipc Memory Name. "
            "return[%d], para: ptr[%p], byteCount[%llu], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, ptrMaxLen, name);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtIpcDestroyMemoryName(const char_t* name)
{
    aclError ret = aclrtIpcMemClose(ReinterpretAs<const char*>(name));
    HCCL_INFO("Call aclrtIpcMemClose, return[%d], para: name[%s]", ret, ReinterpretAs<const char*>(name));
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Destroy][IpcMemoryName]errNo[0x%016llx] "
            "rtDestroy Ipc memory name fail. return[%d], para: name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, name);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void* HrtIpcOpenMemory(const char_t* name)
{
    HCCL_INFO("[HrtIpcOpenMemory] name[%s].", name);
    void* ptr = nullptr;
    aclError ret = aclrtIpcMemImportByKey(&ptr, name, 0UL);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Open][IpcMemory]errNo[0x%016llx] "
            "rtOpen ipc memory fail. return[%d], para: ptr[%p], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, name);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return ptr;
}

void HrtIpcCloseMemory(const void* ptr)
{
    aclError ret = aclrtIpcMemClose(ReinterpretAs<const char*>(ptr));
    HCCL_INFO("Call aclrtIpcMemClose, return[%d], para: name[%s].", ret, ReinterpretAs<const char*>(ptr));
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Close][IpcMemory]errNo[0x%016llx] "
            "rtClose ipc memory failed. return[%d], para: ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtIpcSetMemoryPid(const char_t* name, int pid)
{
    aclError ret = aclrtIpcMemSetImportPid(name, &pid, 1);
    HCCL_INFO("Call aclrtIpcMemSetImportPid, return value[%d], pid[%d], name[%s].", ret, pid, name);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Set][IpcMemoryPid]errNo[0x%016llx] "
            "rtSet ipc memory pid fail. return[%d], pid[%d], name[%s].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, pid, name);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

aclrtPtrAttributes HrtPointerGetAttributes(const void* ptr)
{
    HCCL_INFO("[HrtPointerGetAttributes] ptr[%p].", ptr);
    aclrtPtrAttributes ptrAttr;
    aclError ret = aclrtPointerGetAttributes(ptr, ReinterpretAs<aclrtPtrAttributes*>(&ptrAttr));
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Get][PointAttr]errNo[0x%016llx] rt get point attr failed, "
            "return[%d], para: ptrAddr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return ptrAttr;
}

void PrintMemoryAttr(const void* memAddr)
{
    if (LIKELY(!CheckInfoLogLevel())) {
        return;
    }
    aclrtPtrAttributes memAttr = HrtPointerGetAttributes(memAddr);
    HCCL_INFO(
        "[PrintMemoryAttr] address[%p], page size[%u], type[%d]", memAddr, memAttr.pageSize, memAttr.location.type);
}

void HrtDevMemAlignWithPage(void* ptr, u64 size, void*& ipcPtr, u64& ipcSize, u64& ipcOff)
{
    aclrtPtrAttributes memAttr = HrtPointerGetAttributes(ptr);

    HCCL_INFO(
        "[HrtDevMemAlignWithPage] ptr[%p], size[%llu], ipcPtr[%p], ipcSize[%llu], ipcOff[%llu], pageSize[%u].", ptr,
        size, ipcPtr, ipcSize, ipcOff, memAttr.pageSize);
    if (memAttr.pageSize == 0) {
        ipcPtr = ptr;
        ipcSize = size;
        ipcOff = 0;
        return;
    }

    u64 tmpPtr = ReinterpretAs<u64>(ptr);
    ipcPtr = ReinterpretAs<void*>((ReinterpretAs<u64>(ptr)) & (~(static_cast<u64>(memAttr.pageSize) - 1)));
    ipcOff = tmpPtr - ReinterpretAs<u64>(ipcPtr);
    ipcSize = size + ipcOff;
}

void* HrtMallocHost(u64 size)
{
    void* hostPtr = nullptr;
    aclrtMallocAttrValue moduleIdValue;
    moduleIdValue.moduleId = HCCL;
    aclrtMallocAttribute attrs{.attr = ACL_RT_MEM_ATTR_MODULE_ID, .value = moduleIdValue};
    aclrtMallocConfig cfg{.attrs = &attrs, .numAttrs = 1};
    aclError ret = aclrtMallocHostWithCfg(&hostPtr, size, &cfg);
    PLF_CONFIG_INFO(PLF_RES, "Malloc HostMem para: return value[%d] hostPtr[%p] size[%llu]", ret, hostPtr, size);
    if (ret != ACL_SUCCESS) {
        RPT_INPUT_ERR(
            true, "EI0007", std::vector<std::string>({"resource_type", "resource_info"}),
            std::vector<std::string>(
                {"HostMemory", std::string("MallocHostWithCfg, size:") + std::to_string(size) + " bytes"}));
        string msg = StringFormat(
            "[Malloc][Host]errNo[0x%016llx] rt malloc host fail. return[%d], "
            "para: size[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, size);
        HCCL_ERROR("[%s][%s] %s", LOG_KEYWORDS_INIT_GROUP.c_str(), LOG_KEYWORDS_RESOURCE.c_str(), msg.c_str());
        MACRO_THROW(RuntimeApiException, msg);
    }
    return hostPtr;
}

void HrtFreeHost(void* hostPtr)
{
    PLF_CONFIG_INFO(PLF_RES, "Free HostMem para: hostPtr[%p]", hostPtr);
    aclError ret = aclrtFreeHost(hostPtr);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Free][Host]errNo[0x%016llx] rt free host fail. return[%d], "
            "para: hostPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, hostPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

aclrtNotify HrtNotifyCreate(s32 deviceLogicId)
{
    aclrtNotify ptr = nullptr;
    // aclrtCreateNotify 中通过 aclrtGetDevice 获取 deviceId，所以要求当前线程设置过 setDevice
    aclError ret = aclrtCreateNotify(&ptr, ACL_NOTIFY_DEFAULT);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Notify][Create]errNo[0x%016llx] rtNotifyCreate error. "
            "return[%d], deviceId[%d], ptr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, deviceLogicId, ptr);
        MACRO_THROW(RuntimeApiException, msg);
    }
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", HrtGetDevice(), HrtGetNotifyID(ptr));
    }
    return ptr;
}

void HrtNotifyDestroy(RtNotify_t ptr)
{
    HcclUs startut = TIME_NOW();
    if (GetPlfDebugConfigValue() & PLF_RES) {
        s32 deviceId = HrtGetDevice();
        u32 notifyId = HrtGetNotifyID(ptr);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    aclError ret = aclrtDestroyNotify(ptr);
    HCCL_INFO(
        "[HrtNotifyDestroy] ptr[%p], ret[%d], take time [%lld]us.", ptr, ret,
        DURATION_US(TIME_NOW() - startut).count());
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Notify][Destroy]errNo[0x%016llx] aclrtDestroyNotify error. "
            "ptr[%p], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtIpcSetNotifyName(RtNotify_t ptr, char_t* name, uint32_t len)
{
    aclError ret = aclrtNotifyGetExportKey(ptr, name, len, 2UL);
    HCCL_INFO("[HrtIpcSetNotifyName] ptr[%p], name[%s], len[%u], ret[%d].", ptr, name, len, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Set][IPCNotify]errNo[0x%016llx] IPC set notify name fail. "
            "ptr[%p], name[%s], len[%u], return[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ptr, name, len, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

u32 HrtGetNotifyID(RtNotify_t notifyHandle)
{
    u32 notifyID = 0;
    aclError ret = aclrtGetNotifyId(notifyHandle, &notifyID);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[HrtGetNotifyID]rt get notify id failed. "
            "notifyHandle[%p], notifyID[%u], return[%d].",
            notifyHandle, notifyID, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
    HCCL_INFO("[HrtGetNotifyID] notifyHandle[%p], notifyID[%u].", notifyHandle, notifyID);
    return notifyID;
}

u64 HrtNotifyGetAddr(RtNotify_t notifyHandle)
{
    HCCL_INFO("[HrtNotifyGetAddr] notifyHandle[%p].", notifyHandle);
    uint64_t addr;
    rtError_t ret = rtGetNotifyAddress(notifyHandle, &addr);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "[rtGetNotifyAddress]rt get notify address failed. "
            "notifyHandle[%p], addr[%llu], return[%d].",
            notifyHandle, addr, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return addr;
}

void HrtSetIpcNotifyPid(aclrtNotify notify, int32_t pid)
{
    aclError ret = aclrtNotifySetImportPid(notify, &pid, 1);
    HCCL_INFO("[HrtSetIpcNotifyPid] notify[%p], pid[%d], ret[%d].", notify, pid, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[Set][IpcNotifyPid]errNo[0x%016llx] "
            "rtSet ipc Notify pid fail. notify[%p], return[%d], pid[%d].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), notify, ret, pid);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

RtNotify_t HrtIpcOpenNotify(const char_t* name)
{
    uint64_t flags = 0;
    aclrtNotify* notify = nullptr;
    aclError ret = aclrtNotifyImportByKey(notify, name, static_cast<uint64_t>(flags));
    HCCL_INFO("[HrtIpcOpenNotify] ret[%d], para: notify[%p], name[%s].", ret, notify, name);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[rt][IpcOpenNotify]errNo[0x%016llx] rt ipc notify open fail. "
            "return[%d]. para: notify[%p], name[%s], flags[%llu].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, notify, name, flags);
        MACRO_THROW(RuntimeApiException, msg);
    }
    return notify;
}

u32 HrtNotifyGetOffset(RtNotify_t ptr)
{
    uint32_t offset = 0;
    aclError ret = aclrtGetNotifyId(ptr, &offset);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[rt][NotifyGetOffset]errNo[0x%016llx] rt ipc notify open fail. "
            "return[%d], ptr[%p], offset[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, ptr, offset);
        MACRO_THROW(RuntimeApiException, msg);
    }
    HCCL_INFO("[HrtNotifyGetOffset] ptr[%p], offset[%u].", ptr, offset);
    return offset;
}

void HrtNotifyWaitWithTimeOut(RtNotify_t notifyPtr, aclrtStream streamPtr, uint32_t timeOut)
{
    aclError ret = aclrtWaitAndResetNotify(notifyPtr, streamPtr, timeOut);
    HCCL_INFO(
        "[HrtNotifyWaitWithTimeOut] notifyPtr[%p], streamPtr[%p], timeOut[%u ms], ret[%d].", notifyPtr, streamPtr,
        timeOut, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "call aclrtWaitAndResetNotify failed. notifyPtr=[%p], streamPtr=[%p], timeout=[%u ms], return=[%d].",
            notifyPtr, streamPtr, timeOut, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtNotifyRecord(RtNotify_t notifyPtr, aclrtStream streamPtr)
{
    aclError ret = aclrtRecordNotify(notifyPtr, streamPtr);
    HCCL_INFO("[HrtNotifyRecord] notifyPtr[%p], streamPtr[%p], ret[%d].", notifyPtr, streamPtr, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "call HrtNotifyRecord failed. notifyPtr=%p, streamPtr=%p, return=%d.", notifyPtr, streamPtr, ret);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtMemAsyncCopy(
    void* dst, uint64_t destMax, const void* src, uint64_t count, aclrtMemcpyKind kind, aclrtStream streamPtr)
{
    aclError ret = aclrtMemcpyAsync(dst, destMax, src, count, kind, streamPtr);
    HCCL_DEBUG(
        "[HrtMemAsyncCopy] ret[%d], para: dstAddr[%p], destMax[%llu], "
        "srcAddr[%p], count[%llu], rtKind[%d], streamPtr[%p].",
        ret, dst, destMax, src, count, kind, streamPtr);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[AsyncCopy][Mem]errNo[0x%016llx] rt memory async copy failed. "
            "return[%d], para: dstAddr[%p], destMax[%llu], srcAddr[%p], count[%llu], kind[%d], stream[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind, streamPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtReduceAsync(
    void* dst, uint64_t destMax, const void* src, uint64_t count, aclrtReduceKind kind, aclDataType type,
    aclrtStream streamPtr)
{
    // reserve 预留字段填 nullptr
    aclError ret = aclrtReduceAsync(dst, src, count, kind, type, streamPtr, nullptr);
    HCCL_INFO(
        "Call rtReduceAsync, return value[%d], para: dst[%p], destMax[%llu], src[%p], count[%llu], rtReduceOp[%d], "
        "rtDataType[%d], streamPtr[%p].",
        ret, dst, destMax, src, count, kind, type, streamPtr);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "[rt][ReduceAsync]errNo[0x%016llx] rt reduce async fail. "
            "return[%d], para: dst[%p], destMax[%llu], src[%p], count[%llu], rtReduceOp[%d], rtDataType[%d], "
            "streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dst, destMax, src, count, kind, type, streamPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtRDMASend(u32 qpn, u32 wqeIndex, aclrtStream streamPtr)
{
    rtError_t ret = rtRDMASend(qpn, wqeIndex, streamPtr);
    HCCL_INFO(
        "Call rtRDMASend, return value[%d]. Params: qpn[%u], wqeIndex[%u], streamPtr[%p].", ret, qpn, wqeIndex,
        streamPtr);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "[rt][RdmaSend]errNo[0x%016llx] rt rdma send fail. "
            "return[%d]. para: qpn[%u], wqeIndex[%u], streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, qpn, wqeIndex, streamPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtRDMADBSend(uint32_t dbindex, uint64_t dbinfo, aclrtStream streamPtr)
{
    HCCL_INFO("[HrtRDMADBSend] dbindex[%u], dbinfo[%llu], streamPtr[%p].", dbindex, dbinfo, streamPtr);
    rtError_t ret = rtRDMADBSend(dbindex, dbinfo, streamPtr);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "[rtRDMADBSend]errNo[0x%016llx] rt rdma send fail. "
            "return[%d]. para: dbindex[%u], dbinfo[%llu], streamPtr[%p].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, dbindex, dbinfo, streamPtr);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtGetTaskIdAndStreamID(u32& taskId, u32& streamId)
{
    rtError_t ret = rtGetTaskIdAndStreamID(&taskId, &streamId);
    HCCL_INFO("[HrtGetTaskIdAndStreamID] ret[%d], para: taskId[%u], streamId[%u].", ret, taskId, streamId);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "[Get][TaskIdAndStreamID]errNo[0x%016llx] "
            "rt get task ID and stream ID fail. return[%d], taskId[%u], streamId[%u].",
            HCCL_ERROR_CODE(HcclResult::HCCL_E_RUNTIME), ret, taskId, streamId);
        MACRO_THROW(RuntimeApiException, msg);
    }
}

void HrtUbDbSend([[maybe_unused]] const HrtUbDbInfo& info, [[maybe_unused]] aclrtStream streamPtr)
{
    THROW<NotSupportException>(StringFormat("Unsupported rtUbDbSend"));
}

void HrtUbDirectSend([[maybe_unused]] const HrtUbWqeInfo& info, [[maybe_unused]] aclrtStream streamPtr)
{
    THROW<NotSupportException>(StringFormat("Unsupported rtUbDirectSend"));
}

aclrtCntNotify HrtCntNotifyCreate(u32 deviceId)
{
    aclrtCntNotify handle;
    aclError ret = aclrtCntNotifyCreate(&handle, RT_NOTIFY_FLAG_DEFAULT);
    HCCL_INFO("Call aclrtCntNotifyCreate, return value[%d] devId[%u].", ret, deviceId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call aclrtCntNotifyCreate failed");
        THROW<RuntimeApiException>(msg);
    }
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(
            PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", HrtGetDevice(), HrtGetCntNotifyId(handle));
    }
    return handle;
}

u32 HrtGetCntNotifyId(const aclrtCntNotify inCntNotify)
{
    u32 notifyId = 0; // 待接口rtGetCntNotifyId(inCntNotify, notifyId)上库，目前打桩;
    aclError ret = aclrtCntNotifyGetId(inCntNotify, &notifyId);
    HCCL_INFO("[HrtGetCntNotifyId] ret[%d], inCntNotify[%p], notifyId[%u]", ret, inCntNotify, notifyId);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "Call rtGetCntNotifyId failed. return[%d], inCntNotify[%p], notifyId[%u].", ret, inCntNotify, notifyId);
        THROW<RuntimeApiException>(msg);
    }
    return notifyId;
}

void HrtCntNotifyDestroy(const aclrtCntNotify inCntNotify)
{
    if (GetPlfDebugConfigValue() & PLF_RES) {
        s32 deviceId = HrtGetDevice();
        u32 notifyId = HrtGetCntNotifyId(inCntNotify);
        PLF_CONFIG_INFO(PLF_RES, "Destroy Notify para: deviceId[%d] notifyId[%u]", deviceId, notifyId);
    }
    aclError ret = aclrtCntNotifyDestroy(inCntNotify);
    HCCL_INFO("[HrtCntNotifyDestroy] ret[%d], inCntNotify[%p].", ret, inCntNotify);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call aclrtCntNotifyDestroy failed");
        THROW<RuntimeApiException>(msg);
    }
}

const std::map<HrtCntNotifyRecordMode, aclrtCntNotifyRecordMode> HRT_CNT_NOTIFY_RECORD_MODE_MAP
    = {{HrtCntNotifyRecordMode::WRITE_BIT, aclrtCntNotifyRecordMode::ACL_RT_CNT_NOTIFY_RECORD_BIT_OR_MODE},
       {HrtCntNotifyRecordMode::STORE, aclrtCntNotifyRecordMode::ACL_RT_CNT_NOTIFY_RECORD_SET_VALUE_MODE}};
void HrtCntNotifyRecord(
    const aclrtCntNotify inCntNotify, const aclrtStream streamPtr, HrtCntNotifyRecordMode mode, u32 value)
{
    aclrtCntNotifyRecordInfo recordInfo{};
    recordInfo.mode = HRT_CNT_NOTIFY_RECORD_MODE_MAP.at(mode);
    recordInfo.value = value;
    aclError ret = aclrtCntNotifyRecord(inCntNotify, streamPtr, &recordInfo);
    HCCL_INFO(
        "[HrtCntNotifyRecord] inCntNotify[%p], streamPtr[%p], mode[%d], value[%u], ret[%d].", inCntNotify, streamPtr,
        mode, value, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call aclrtCntNotifyRecord failed");
        THROW<RuntimeApiException>(msg);
    }
}
// david接口 包间接口
const std::map<HrtCntNotifyWaitMode, aclrtCntNotifyWaitMode> HRT_CNT_NOTIFY_WAIT_MODE_MAP
    = {{HrtCntNotifyWaitMode::EQUAL, aclrtCntNotifyWaitMode::ACL_RT_CNT_NOTIFY_WAIT_EQUAL_MODE},
       {HrtCntNotifyWaitMode::BITMAP, aclrtCntNotifyWaitMode::ACL_RT_CNT_NOTIFY_WAIT_EQUAL_WITH_BITMASK_MODE}};
void HrtCntNotifyWaitWithTimeOut(
    const aclrtCntNotify inCntNotify, const aclrtStream streamPtr, HrtCntNotifyWaitMode mode, u32 value, u32 timeout,
    bool isClear)
{
    aclrtCntNotifyWaitInfo waitInfo{};
    waitInfo.mode = HRT_CNT_NOTIFY_WAIT_MODE_MAP.at(mode);
    waitInfo.value = value;
    waitInfo.isClear = isClear;
    waitInfo.timeout = timeout;
    aclError ret = aclrtCntNotifyWaitWithTimeout(inCntNotify, streamPtr, &waitInfo);
    HCCL_INFO(
        "[HrtCntNotifyWaitWithTimeOut] inCntNotify[%p], streamPtr[%p], mode[%d], "
        "value[%u], timeout[%u ms], isClear[%d], ret[%d].",
        inCntNotify, streamPtr, mode, value, timeout, isClear, ret);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "Call rtCntNotifyWaitWithTimeout failed. return[%d], inCntNotify[%p], "
            "streamPtr[%p], mode[%d], value[%u], timeout[%u ms], isClear[%d].",
            ret, inCntNotify, streamPtr, mode, value, timeout, isClear);
        THROW<RuntimeApiException>(msg);
    }
}

aclrtNotify HrtNotifyCreateWithFlag(u32 devId, u32 flag)
{
    aclrtNotify ptr = nullptr;
    aclError ret = aclrtCreateNotify(&ptr, flag);
    HCCL_INFO("Call HrtNotifyCreateWithFlag, return value[%d], flag[%u] devid[%u].", ret, flag, devId);
    if (ret != ACL_SUCCESS) {
        string msg
            = StringFormat("Call rtNotifyCreateWithFlag failed. return[%d], devId[%u], flag[%u].", ret, devId, flag);
        THROW<RuntimeApiException>(msg);
    }
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(PLF_RES, "Create Notify para: deviceId[%d] notifyId[%u]", HrtGetDevice(), HrtGetNotifyID(ptr));
    }
    return ptr;
}

RtNotify_t HrtIpcOpenNotifyWithFlag(const char_t* name, uint32_t flags)
{
    HCCL_INFO("[HrtIpcOpenNotifyWithFlag] name[%s], flags[%u].", name, flags);
    RtNotify_t ptr = nullptr;
    aclError ret = aclrtNotifyImportByKey(&ptr, name, static_cast<uint64_t>(flags));
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "Call rtIpcOpenNotifyWithFlag failed. return[%d], name[%p], flags[%u], ptr[%p].", ret, name, flags, ptr);
        THROW<RuntimeApiException>(msg);
    }
    return ptr;
}

// 兜底extern形式
void HrtAicpuLaunchKernelWithHostArgs(
    aclrtFuncHandle funcHandle, uint32_t numBlocks, aclrtStream stream, aclrtLaunchKernelCfg* cfg, void* hostArgs,
    size_t argsSize, aclrtPlaceHolderInfo* placeHolderArray, size_t placeHolderNum)
{
    rtError_t ret = aclrtLaunchKernelWithHostArgs(
        funcHandle, numBlocks, stream, cfg, hostArgs, argsSize, placeHolderArray, placeHolderNum);
    if (ret != RT_ERROR_NONE) {
        THROW<RuntimeApiException>(StringFormat("Call aclrtLaunchKernelWithHostArgs failed, with ret[%d]", ret));
    }
}

void HrtRegTaskExceptionCallbackByModule(aclrtExceptionInfoCallback callback)
{
    HCCL_INFO("[HrtRegTaskExceptionCallbackByModule] callback[%p].", callback);
    aclError ret = aclrtExceptionInfoCallbackRegister(callback);
    if (ret != ACL_SUCCESS) {
        string msg
            = StringFormat("Call aclrtExceptionInfoCallbackRegister failed. return[%d], callback[%p].", ret, callback);
        THROW<RuntimeApiException>(msg);
    }
}

void HrtUnregTaskExceptionCallbackByModule(aclrtExceptionInfoCallback callback)
{
    HCCL_INFO("[HrtUnregTaskExceptionCallbackByModule] callback[%p].", callback);
    aclError ret = aclrtExceptionInfoCallbackUnregister(callback);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "Call aclrtExceptionInfoCallbackUnregister failed. return[%d], callback[%p].", ret, callback);
        THROW<RuntimeApiException>(msg);
    }
}

u32 HrtStreamGetSqId(const aclrtStream ptr)
{
    HCCL_INFO("[HrtStreamGetSqId] ptr[%p].", ptr);
    u32 sqId;
    rtError_t ret = rtStreamGetSqid(ptr, &sqId);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat("Call rtStreamGetSqid failed. return[%d], ptr[%p], sqId[%u].", ret, ptr, sqId);
        THROW<RuntimeApiException>(msg);
    }
    return sqId;
}

u32 HrtStreamGetCqId(const aclrtStream ptr)
{
    HCCL_INFO("[HrtStreamGetCqId] ptr[%p].", ptr);
    u32 cqId;
    u32 logicCqId;
    rtError_t ret = rtStreamGetCqid(ptr, &cqId, &logicCqId);
    if (ret != RT_ERROR_NONE) {
        THROW<RuntimeApiException>(StringFormat("Call rtStreamGetCqid failed, with ret[%d]", ret));
    }
    return logicCqId;
}

void HrtCcuLaunch(rtCcuTaskInfo_t& taskInfo, aclrtStream const streamPtr)
{
    HCCL_INFO("[HrtCcuLaunch] taskInfo[%p], streamPtr[%p].", &taskInfo, streamPtr);
    auto ret = rtCCULaunch(&taskInfo, streamPtr);
    if (ret != RT_ERROR_NONE) {
        std::string msg;
        if (ret == ACL_ERROR_RT_STREAM_TASK_FULL) {
            // task 队列满
            msg = StringFormat(
                "[%s] rtCCULaunch failed because the task queue on the stream is full. "
                "ret[%d], param: taskInfo[%p], streamPtr[%p]. "
                "Possible causes: 1) too many tasks have been submitted to the stream; "
                "2) the HCCL buffer is configured too small.",
                __func__, ret, &taskInfo, streamPtr);
        } else {
            msg = StringFormat(
                "Call rtCCULaunch failed. return[%d], taskInfo[%p], streamPtr[%p].", ret, &taskInfo, streamPtr);
        }
        THROW<RuntimeApiException>(msg);
    }
}

void HrtUbDevQueryInfo(rtUbDevQueryCmd cmd, void* devInfo)
{
    HCCL_INFO("[HrtUbDevQueryInfo] cmd[%d], devInfo[%p].", cmd, devInfo);
    auto ret = rtUbDevQueryInfo(cmd, devInfo);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat("Call rtUbDevQueryInfo failed. return[%d], cmd[%d], devInfo[%p].", ret, cmd, devInfo);
        THROW<RuntimeApiException>(msg);
    }
    if (cmd == QUERY_PROCESS_TOKEN) {
        rtMemUbTokenInfo* info = static_cast<rtMemUbTokenInfo*>(devInfo);
        info->tokenId = info->tokenId >> TOKEN_ID_RIGHT_SHIF;
    }
}
// pair<tokendId, tokenValue>
std::pair<u32, u32> HrtUbDevQueryToken(u64 addr, u64 size)
{
    HCCL_INFO("[HrtUbDevQueryToken] addr[%llu], size[%llu].", addr, size);
    rtMemUbTokenInfo info;
    info.va = addr;
    info.size = size;
    auto ret = rtUbDevQueryInfo(QUERY_PROCESS_TOKEN, &info);
    if (ret != RT_ERROR_NONE) {
        HCCL_WARNING("query(va=0x%llx, size=0x%llx) token failed, ret=%d", addr, size, ret);
        return std::make_pair(0, 0);
    }

    return {info.tokenId >> TOKEN_ID_RIGHT_SHIF, info.tokenValue};
}

const std::map<HrtDevResProcType, rtDevResProcType_t> HRT_DEV_RES_PROC_TYPE_MAP
    = {{HrtDevResProcType::PROCESS_CP1, RT_PROCESS_CP1}, {HrtDevResProcType::PROCESS_HCCP, RT_PROCESS_HCCP}};

const std::map<HrtDevResType, rtDevResType_t> HRT_DEV_RES_TYPE_MAP
    = {{HrtDevResType::RES_TYPE_STARS_NOTIFY_RECORD, RT_RES_TYPE_STARS_NOTIFY_RECORD},
       {HrtDevResType::RES_TYPE_CCU_CKE, RT_RES_TYPE_CCU_CKE},
       {HrtDevResType::RES_TYPE_CCU_XN, RT_RES_TYPE_CCU_XN},
       {HrtDevResType::RES_TYPE_STARS_CNT_NOTIFY_BIT_WR, RT_RES_TYPE_STARS_CNT_NOTIFY_BIT_WR}};
HrtDevResAddrInfo HrtGetDevResAddress(const HrtDevResInfo& devResInfo)
{
    HCCL_INFO(
        "[HrtGetDevResAddress] devResInfo.dieId[%u], devResInfo.procType[%u], "
        "devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
        devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
    rtDevResInfo resInfo;
    resInfo.dieId = devResInfo.dieId;
    resInfo.procType = HRT_DEV_RES_PROC_TYPE_MAP.at(devResInfo.procType);
    resInfo.resType = HRT_DEV_RES_TYPE_MAP.at(devResInfo.resType);
    resInfo.flag = devResInfo.flag;
    resInfo.resId = devResInfo.resId;

    uint64_t addr = 0;
    u32 len = 0;
    rtDevResAddrInfo addrInfo;
    addrInfo.resAddress = &addr;
    addrInfo.len = &len;
    auto ret = rtGetDevResAddress(&resInfo, &addrInfo);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "Call rtGetDevResAddress failed. return[%d], devResInfo.dieId[%u], "
            "devResInfo.procType[%u], devResInfo.resType[%u], devResInfo.flag[%u], "
            "devResInfo.resId[%u].",
            ret, devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
        THROW<RuntimeApiException>(msg);
    }
    HrtDevResAddrInfo devResAddrInfo;
    devResAddrInfo.address = addr;
    devResAddrInfo.len = len;
    HCCL_INFO("devResAddrInfo.address[%llu], devResAddrInfo.len[%u].", devResAddrInfo.address, devResAddrInfo.len);
    return devResAddrInfo;
}

void HrtReleaseDevResAddress(const HrtDevResInfo& devResInfo)
{
    HCCL_INFO(
        "[HrtReleaseDevResAddress] devResInfo.dieId[%u], devResInfo.procType[%u], "
        "devResInfo.resType[%u], devResInfo.flag[%u], devResInfo.resId[%u].",
        devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
    rtDevResInfo resInfo;
    resInfo.dieId = devResInfo.dieId;
    resInfo.procType = HRT_DEV_RES_PROC_TYPE_MAP.at(devResInfo.procType);
    resInfo.resType = HRT_DEV_RES_TYPE_MAP.at(devResInfo.resType);
    resInfo.flag = devResInfo.flag;
    resInfo.resId = devResInfo.resId;

    rtError_t ret = rtReleaseDevResAddress(&resInfo);
    if (ret != RT_ERROR_NONE) {
        string msg = StringFormat(
            "Call rtReleaseDevResAddress failed. return[%d], devResInfo.dieId[%u], "
            "devResInfo.procType[%u],devResInfo.resType[%u], devResInfo.flag[%u], "
            "devResInfo.resId[%u].",
            ret, devResInfo.dieId, devResInfo.procType, devResInfo.resType, devResInfo.flag, devResInfo.resId);
        THROW<RuntimeApiException>(msg);
    }
}

aclrtEvent HrtEventCreateWithFlag(u32 flag)
{
    HCCL_INFO("[HrtEventCreateWithFlag] flag[%u].", flag);
    aclrtEvent ptr = nullptr;
    aclError ret = aclrtCreateEventWithFlag(&ptr, flag);
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(PLF_RES, "Create Event para: deviceId[%d] event[%p]", HrtGetDevice(), ptr);
    }
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call rtEventCreateWithFlag failed. return[%d], flag[%u], ptr[%p].", ret, flag, ptr);
        THROW<RuntimeApiException>(msg);
    }
    return ptr;
}

void HrtEventDestroy(RtEvent_t eventPtr)
{
    if (GetPlfDebugConfigValue() & PLF_RES) {
        PLF_CONFIG_INFO(PLF_RES, "Destroy Event para: deviceId[%d] event[%p]", HrtGetDevice(), eventPtr);
    }
    aclError ret = aclrtDestroyEvent(eventPtr);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call aclrtDestroyEvent failed. return[%d], eventPtr[%p].", ret, eventPtr);
        THROW<RuntimeApiException>(msg);
    }
}

void HrtEventRecord(RtEvent_t eventPtr, aclrtStream streamPtr)
{
    HCCL_INFO("[HrtEventRecord] eventPtr[%p], streamPtr[%p].", eventPtr, streamPtr);
    aclError ret = aclrtRecordEvent(eventPtr, streamPtr);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat(
            "Call aclrtRecordEvent failed. return[%d], eventPtr[%p], streamPtr[%p].", ret, eventPtr, streamPtr);
        THROW<RuntimeApiException>(msg);
    }
}

const std::map<aclrtEventWaitStatus, HrtEventStatus> HRT_EVENT_STATUS_MAP{
    {ACL_EVENT_WAIT_STATUS_NOT_READY, HrtEventStatus::EVENT_INIT},
    {ACL_EVENT_WAIT_STATUS_COMPLETE, HrtEventStatus::EVENT_RECORDED},
};

HrtEventStatus HrtEventQueryStatus(RtEvent_t eventPtr)
{
    HCCL_INFO("[HrtEventQueryStatus] eventPtr[%p].", eventPtr);
    aclrtEventWaitStatus status = ACL_EVENT_WAIT_STATUS_NOT_READY;
    aclError ret = aclrtQueryEventWaitStatus(eventPtr, &status);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("Call aclrtQueryEventWaitStatus failed. return[%d], eventPtr[%p].", ret, eventPtr);
        THROW<RuntimeApiException>(msg);
    }
    if (HRT_EVENT_STATUS_MAP.find(status) == HRT_EVENT_STATUS_MAP.end()) {
        THROW<InvalidParamsException>(
            StringFormat("event status[%u] not in HRT_EVENT_STATUS_MAP.", static_cast<u32>(status)));
    }
    return HRT_EVENT_STATUS_MAP.at(status);
}

void HrtWriteValue([[maybe_unused]] u64 addr, [[maybe_unused]] u32 piVal, [[maybe_unused]] const aclrtStream streamPtr)
{
    THROW<NotSupportException>(StringFormat("Unsupported rtWriteValue"));
}

void HrtDeviceAbortRegCallBack(aclrtDeviceTaskAbortCallback callback, void* args, const std::string& name)
{
    aclError ret = aclrtSetDeviceTaskAbortCallback(name.c_str(), callback, args);
    if (ret != ACL_SUCCESS) {
        string msg = StringFormat("call rtSetTaskAbortCallBack failed. ret=[%d].", ret);
        THROW<RuntimeApiException>(msg);
    }
}

HcclResult HrtThreadExchangeCaptureMode(aclmdlRICaptureMode* mode)
{
    HCCL_INFO("[HrtThreadExchangeCaptureMode] mode[%p].", mode);
    aclError ret = aclmdlRICaptureThreadExchangeMode(mode);
    if (ret == ACL_ERROR_RT_FEATURE_NOT_SUPPORT) {
        HCCL_WARNING(
            "[HrtThreadExchangeCaptureMode] rtThreadExchangeCaptureMode not support!, ret=%d, mode=%p.", ret, mode);
        return HCCL_E_NOT_SUPPORT;
    } else {
        CHK_PRT_RET(
            ret != ACL_SUCCESS,
            HCCL_ERROR(
                "[HrtThreadExchangeCaptureMode]rtThreadExchangeCaptureMode "
                "failed mode:%d, return value[%d].",
                *mode, ret),
            HCCL_E_RUNTIME);
    }
    return HCCL_SUCCESS;
}

HcclResult HrtMemPrefetchToDevice(void* devPtr, uint64_t len)
{
    CHK_PRT_RET(
        aclrtMemP2PMap == nullptr,
        HCCL_ERROR("aclrtMemP2PMap is nullptr, "
                   "Does not support this interface."),
        HCCL_E_RUNTIME);
    aclError ret = aclrtMemP2PMap(devPtr, static_cast<size_t>(len), HrtGetDevice(), 0);
    HCCL_INFO("[HrtMemPrefetchToDevice] devPtr[%p], len[%llu], ret[%d].", devPtr, len, ret);
    if (ret != ACL_SUCCESS) {
        HCCL_ERROR("aclrtMemP2PMap fail ret = %d", ret);
        return HCCL_E_RUNTIME;
    }
    return HCCL_SUCCESS;
}

HcclResult HrtEnableP2P(u32 deviceLogicId, u32 devicePhyId)
{
    rtError_t ret = rtEnableP2P(deviceLogicId, devicePhyId, 0);

    HCCL_INFO("rt enableP2P deviceLogicId[%u] and devicePhyId[%u] ret[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Enable][P2P]errNo[0x%016llx] rt enableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);

    return HCCL_SUCCESS;
}

HcclResult HrtDisableP2P(u32 deviceLogicId, u32 devicePhyId)
{
    rtError_t ret = rtDisableP2P(deviceLogicId, devicePhyId);

    HCCL_INFO("rt disableP2P deviceLogicId[%u] and devicePhyId[%u] ret[%d]", deviceLogicId, devicePhyId, ret);

    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Disable][P2P]errNo[0x%016llx] rt disableP2P deviceLogicId[%u] and "
            "devicePhyId[%u] fail[%d]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), deviceLogicId, devicePhyId, ret),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}

HcclResult HrtGetP2PStatus(u32 deviceLogicId, u32 devicePhyId, uint32_t* status)
{
    rtError_t ret = rtGetP2PStatus(deviceLogicId, devicePhyId, status);

    HCCL_DEBUG(
        "rt getp2pstatus deviceLogicId[%u] and devicePhyId[%u] ret[%d], status[%u]", deviceLogicId, devicePhyId, ret,
        *status);
    CHK_PRT_RET(
        ret != RT_ERROR_NONE,
        HCCL_ERROR(
            "[Get][P2PStatus]errNo[0x%016llx]Call rtGetP2PStatus failed, "
            "ret[%d], deviceLogicId[%u], devicePhyId[%u]",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, deviceLogicId, devicePhyId),
        HCCL_E_RUNTIME);
    return HCCL_SUCCESS;
}
} // namespace Hccl
