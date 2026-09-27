/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <unordered_set>
#include <mutex>
#include "log.h"
#include "adapter_rts.h"
#include "adapter_hccp.h"
#include "dtype_common.h"
#include "externalinput.h"
#include "network_manager_pub.h"
#include "adapter_hal.h"
#include "dlhal_function.h"
#include "device_capacity.h"

constexpr u32 INLINEREDUCE_ALIGN_BYTES_910A = 128;
constexpr u32 INLINEREDUCE_ALIGN_BYTES_310P = 2;
constexpr u32 INLINEREDUCE_ALIGN_BYTES_910_93 = 1; // A2和A3场景的inline reduce不受地址对齐限制
constexpr float BANDWIDTH_HCCS_910A = 10.0f;
constexpr float BANDWIDTH_HCCS_910B = 18.3f;
constexpr float BANDWIDTH_RDMA_910A = 12.5f * 0.8;
constexpr float BANDWIDTH_RDMA_910B = 25.0f * 0.8;
constexpr float BANDWIDTH_HBM_910_93 = 650.0f * 0.9;
constexpr float BANDWIDTH_SIO_910_93 = 240.0f * 0.85;

// 常用带宽值   GB/s
constexpr float BANDWIDTH_PCIE_GEN3 = 16.0f * 0.85;
constexpr float BANDWIDTH_PCIE_GEN4 = 32.0f * 0.85;
constexpr float BANDWIDTH_PCIE_GEN5 = 64.0f * 0.85;

// 非超节点模式的server id配置, 通过hrtGetDeviceInfo接口查询到的server id统一为0x3FF
constexpr s64 INVALID_SUPERPOD_SERVERID = 0x3FF;

namespace hccl {
#ifdef ASCEND_310P_DEVICE
bool g_is310PDevice = true;
#else
bool g_is310PDevice = false;
#endif

bool IsSupportAIVCopy(HcclDataType dataType)
{
    return (
        dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_UINT16
        || dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_UINT32
        || dataType == HCCL_DATA_TYPE_INT8 || dataType == HCCL_DATA_TYPE_UINT8 || dataType == HCCL_DATA_TYPE_BFP16
        || dataType == HCCL_DATA_TYPE_INT64 || dataType == HCCL_DATA_TYPE_UINT64 || dataType == HCCL_DATA_TYPE_FP64);
}

bool IsSupportAIVReduce(HcclDataType dataType, HcclReduceOp op)
{
    bool checkDataType
        = (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT8
           || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);

    return checkDataType && checkReduceType;
}

bool IsAddressAlign(const void* inputPtr, const void* outputPtr, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_910A)
                   == (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_910A);
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_910_93)
                   == (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_910_93);
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_310P1:
            return (reinterpret_cast<intptr_t>(inputPtr) % INLINEREDUCE_ALIGN_BYTES_310P)
                   == (reinterpret_cast<intptr_t>(outputPtr) % INLINEREDUCE_ALIGN_BYTES_310P);
        default:
            HCCL_WARNING(
                "device type[%d] is out of range, valid range[0, %d]", static_cast<s32>(devType),
                static_cast<s32>(DevType::DEV_TYPE_COUNT) - 1);
            return false;
    }
}

bool IsDataTypeSupport(HcclDataType dataType, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (
                dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT8
                || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32
                || dataType == HCCL_DATA_TYPE_BFP16);
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P1:
            return dataType == HCCL_DATA_TYPE_FP32;
        case DevType::DEV_TYPE_310P3:
            return (
                dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_FP16);
        default:
            HCCL_WARNING(
                "device type[%d] is out of range, valid range[0, %d]", static_cast<s32>(devType),
                static_cast<s32>(DevType::DEV_TYPE_COUNT) - 1);
            return false;
    }
}

bool IsRedOpSupport(HcclReduceOp op, DevType devType)
{
    switch (devType) {
        case DevType::DEV_TYPE_910B:
        case DevType::DEV_TYPE_910_93:
            return (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);
        case DevType::DEV_TYPE_910:
        case DevType::DEV_TYPE_310P3:
        case DevType::DEV_TYPE_310P1:
            return op == HCCL_REDUCE_SUM;
        default:
            HCCL_WARNING(
                "device type[%d] is out of range, valid range[0, %d]", static_cast<s32>(devType),
                static_cast<s32>(DevType::DEV_TYPE_COUNT) - 1);
            return false;
    }
}

bool IsSupportSDMAReduce(const void* inputPtr, const void* outputPtr, HcclDataType dataType, HcclReduceOp op)
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    return IsAddressAlign(inputPtr, outputPtr, devType) && IsDataTypeSupport(dataType, devType)
           && IsRedOpSupport(op, devType);
}

bool IsSupportRDMAReduce(HcclDataType dataType, HcclReduceOp op)
{
    bool checkDataType
        = (dataType == HCCL_DATA_TYPE_FP32 || dataType == HCCL_DATA_TYPE_FP16 || dataType == HCCL_DATA_TYPE_INT8
           || dataType == HCCL_DATA_TYPE_INT16 || dataType == HCCL_DATA_TYPE_INT32 || dataType == HCCL_DATA_TYPE_BFP16);
    bool checkReduceType = (op == HCCL_REDUCE_SUM || op == HCCL_REDUCE_MAX || op == HCCL_REDUCE_MIN);
    bool isInfNanMode = IsOverFlowInfNanMode();
    return checkDataType && checkReduceType && isInfNanMode;
}

HcclResult GetBandWidthPerNPU(u32 level, u32 userRankSize, u32 deviceNumPerAggregation, float& bandWidth)
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    // 处理 level=1、910B 的特殊条件
    if (level == 1 && (devType == DevType::DEV_TYPE_910B || devType == DevType::DEV_TYPE_910_93)
        && userRankSize == deviceNumPerAggregation * 2) // 2: 910B 16p形态 单server场景
    {
        bandWidth = BANDWIDTH_PCIE_GEN5;
        return HCCL_SUCCESS;
    }
    // 其余情况查表
    static const std::map<std::pair<u32, DevType>, float> bwTable = {
        // level 0
        {{0, DevType::DEV_TYPE_310P3}, BANDWIDTH_PCIE_GEN3},
        {{0, DevType::DEV_TYPE_910}, BANDWIDTH_HCCS_910A},
        {{0, DevType::DEV_TYPE_910B}, BANDWIDTH_HCCS_910B},
        {{0, DevType::DEV_TYPE_910_93}, BANDWIDTH_HCCS_910B},
        // level 1
        {{1, DevType::DEV_TYPE_910}, BANDWIDTH_RDMA_910A},
        {{1, DevType::DEV_TYPE_910B}, BANDWIDTH_RDMA_910B},
        {{1, DevType::DEV_TYPE_910_93}, BANDWIDTH_RDMA_910B},
        // level 2
        {{2, DevType::DEV_TYPE_910_93}, BANDWIDTH_HBM_910_93},
        // level 3
        {{3, DevType::DEV_TYPE_910_93}, BANDWIDTH_SIO_910_93},
    };
    auto key = std::make_pair(level, devType);
    auto it = bwTable.find(key);
    if (it != bwTable.end()) {
        bandWidth = it->second;
        return HCCL_SUCCESS;
    }
    HCCL_ERROR("[Get][BandWidthPerNPU] Failed, deviceType[%d] Bandwidth Level[%u]", devType, level);
    return HCCL_E_NOT_SUPPORT;
}

HcclResult CheckDeviceType(const DevType deviceType)
{
    if ((deviceType >= DevType::DEV_TYPE_COUNT) || (deviceType < DevType::DEV_TYPE_910)) {
        HCCL_ERROR(
            "[Check][DeviceType]errNo[0x%016llx] device Type[%d] out of range[%d, %d]", HCCL_ERROR_CODE(HCCL_E_PARA),
            deviceType, DevType::DEV_TYPE_910, DevType::DEV_TYPE_NOSOC);
        return HCCL_E_PARA;
    }

    return HCCL_SUCCESS;
}

bool IsOverFlowInfNanMode()
{
    aclrtFloatOverflowMode floatOverflowMode = ACL_RT_OVERFLOW_MODE_UNDEF;
    HcclResult ret = hrtGetDeviceSatMode(&floatOverflowMode);
    if (ret != HCCL_SUCCESS) {
        HCCL_WARNING("[impl][IsOverFlowInfNanMode] GetDeviceSatMode failed");
    }
    return (!GetExternalInputHcclDumpDebug()) && (floatOverflowMode == ACL_RT_OVERFLOW_MODE_INFNAN);
}

bool Is310PDevice() { return g_is310PDevice; }

bool IsUseSdidForDeviceId(const u32 superDeviceId)
{
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    if (devType == DevType::DEV_TYPE_910_93 && superDeviceId != INVALID_UINT) {
        return true;
    }
    return false;
}

HcclResult IsSuperPodMode(bool& useSuperPodMode)
{
    useSuperPodMode = false;
    DevType devType = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType));
    s64 serverId = INVALID_SUPERPOD_SERVERID;
    if (devType == DevType::DEV_TYPE_910_93) {
        s32 deviceLogicId = INVALID_INT;
        HcclResult ret = hrtGetDevice(&deviceLogicId);
        CHK_PRT_RET(ret != HCCL_SUCCESS, HCCL_ERROR("[IsSuperPodMode]Get device id fail"), ret);
        CHK_RET(hrtGetDeviceInfo(
            deviceLogicId, HcclRtDeviceModuleType::HCCL_RT_MODULE_TYPE_SYSTEM,
            HcclRtDeviceInfoType::HCCL_INFO_TYPE_SERVER_ID, serverId));
        useSuperPodMode = (serverId != INVALID_SUPERPOD_SERVERID);
        if (!useSuperPodMode) {
            HCCL_WARNING(
                "If server Id is not configured, the network port may need to be "
                "kept up to ensure normal service running, server id[%016llx]",
                serverId);
        }
    }
    HCCL_INFO("[IsSuperPodMode]ret[%d], devType[%d], serverId[%016llx]", useSuperPodMode, devType, serverId);
    return HCCL_SUCCESS;
}

HcclResult GetMaxDevNum(u32& MaxDevNum)
{
    // 静态缓存最大设备数
    static u32 cachedMaxDevNum;
    static std::once_flag initFlag; // 用于确保初始化只执行一次
    static std::mutex initMutex;    // 用于保护call_once调用

    // 使用mutex保护call_once调用
    std::lock_guard<std::mutex> lock(initMutex);

    // 使用call_once确保初始化逻辑只执行一次
    std::call_once(initFlag, [&]() {
        DevType devType = DevType::DEV_TYPE_COUNT;
        HcclResult result = hrtGetDeviceType(devType);
        if (result != HCCL_SUCCESS) {
            HCCL_ERROR("[GetMaxDevNum] [hrtGetDeviceType] get device type failed");
        }
        switch (devType) {
            case DevType::DEV_TYPE_310P3:
                cachedMaxDevNum = MAX_DEVICE_NUM_THIRTY_TWO;
                break;
            case DevType::DEV_TYPE_950:
            case DevType::DEV_TYPE_960:
                cachedMaxDevNum = MAX_DEVICE_NUM_SIXTY_FIVE;
                break;
            default:
                cachedMaxDevNum = MAX_DEVICE_NUM_SIXTEEN;
                break;
        }
    });

    // 直接读取缓存值
    MaxDevNum = cachedMaxDevNum;
    HCCL_DEBUG("[GetMaxDevNum] MaxDevNum[%u]", MaxDevNum);
    return HCCL_SUCCESS;
}

#ifndef OPEN_HCCL_TEST
HcclResult IsSupportAicpuNormalQP(const u32& devicePhyId, bool& isSupportNormalQP)
{
    u32 aiQpCreateVersion = 0;
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, AI_QP_CREATE, &aiQpCreateVersion);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[IsSupportAicpuNormalQP] ret[%d]"
            "devicePhyId[%u] not support",
            ret, devicePhyId),
        HCCL_E_NETWORK);
    if (aiQpCreateVersion >= AI_NORMAL_QP_CREATE_VERSION) {
        isSupportNormalQP = true;
    } else {
        isSupportNormalQP = false;
    }
    HCCL_DEBUG("IsSupportAicpuNormalQP devicePhyId[%u], isSupportNormalQP[%d].", devicePhyId, isSupportNormalQP);
    return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_HCCL_TEST
HcclResult IsSupportAIVNormalQP(const u32& devicePhyId, bool& isSupport)
{
    u32 version = 0;
    HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, AI_QP_CREATE_WITH_ATTRS, &version);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[hrtRaGetInterfaceVersion] ret[%d]"
            "devicePhyId[%u] not support",
            ret, devicePhyId),
        HCCL_E_NETWORK);

    if (version >= AI_QP_CREATE_WITH_ATTRS_VERSION) {
        isSupport = true;
    } else {
        isSupport = false;
    }
    HCCL_INFO("IsSupportAIVNormalQP devicePhyId[%u], isSupport[%d].", devicePhyId, isSupport);
    return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_HCCL_TEST
bool IsSupportRDMALite(const s32 deviceLogicId)
{
    return NetworkManager::GetInstance(deviceLogicId).GetRdmaLiteStatus();
}
HcclResult IsSupportHccsAndSio(bool& flag)
{
    flag = false;
    size_t outputLen = 0;
    supportFeaturePara inputPara = {};
    supportFeaturePara outputPara = {};
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));
    s32 logicDevId = 0;
    // 调用驱动接口前需将userDevId转换为logicDevId
    CHK_RET(hrtGetLogicDevIdByUserDevId(deviceId, logicDevId));
    inputPara.support_feature = CTRL_SUPPORT_SHMEM_MAP_EXBUS_MASK;
    inputPara.devid = static_cast<unsigned int>(logicDevId);
    CHK_RET(hrtHalMemCtl(CTRL_TYPE_SUPPORT_FEATURE, &inputPara, sizeof(supportFeaturePara), &outputPara, &outputLen));

    if ((outputPara.support_feature & CTRL_SUPPORT_SHMEM_MAP_EXBUS_MASK) != 0) {
        flag = true;
    }
    HCCL_INFO("[IsSupportHccsAndSio] isSupportHccsAndSio %d", flag);
    return HCCL_SUCCESS;
}
#endif

#ifndef OPEN_HCCL_TEST
HcclResult GetMemBlockNum([[maybe_unused]] const u32 devicePhyId, [[maybe_unused]] u32& memBlockNum)
{
#ifndef CCL_KERNEL_AICPU
    u32 info = 0;
    CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    CHK_RET(hrtDrvGetPlatformInfo(&info));
    if (info == 0) { // 在device侧
        std::string chipName;
        if (hrtHalGetChipInfo(devicePhyId, chipName) == HCCL_SUCCESS) {
            if (chipName.find(SOC_NAME_910B) != std::string::npos) {
                // 共享内存池目前不支持动态扩容；910B场景需要的内存池较大，但是申请太大，会导致310P上内存不足，通过硬件区分。
                memBlockNum = MEM_BLOCK_NUM_BIGER;
            }
        }
    }
#endif
    return HCCL_SUCCESS;
}
#endif
// 获取算子最大超时时间
u32 GetNotifyMaxWaitTime()
{
    static bool init = false;
    static uint32_t notifyMaxWaitTime = NOTIFY_MAX_WAIT_TIME;
    if (UNLIKELY(!init)) {
        DevType deviceType = DevType::DEV_TYPE_COUNT;
        if (hrtGetDeviceType(deviceType) == HCCL_SUCCESS) {
            notifyMaxWaitTime = (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) ?
                                    NOTIFY_MAX_WAIT_TIME_910_93 :
                                    NOTIFY_MAX_WAIT_TIME;
            init = true;
        }
    }

    HCCL_INFO("[GetNotifyMaxWaitTime] notifyMaxWaitTime is %us", notifyMaxWaitTime);
    return notifyMaxWaitTime;
}

HcclResult IsSupportAtomicWrite(DevType deviceType, u32 devicePhyId, bool& isSupportAtomicWrite)
{
    if (deviceType == DevType::DEV_TYPE_910_93 || deviceType == DevType::DEV_TYPE_910B) {
        u32 version = 0;
        HcclResult ret = hrtRaGetInterfaceVersion(devicePhyId, RA_RS_GET_ROCE_API, &version);
        CHK_PRT_RET(
            ret != HCCL_SUCCESS,
            HCCL_ERROR("%s call hrtRaGetInterfaceVersion ret[%d] devicePhyId[%u]", __func__, ret, devicePhyId),
            HCCL_E_NETWORK);
        isSupportAtomicWrite = (version >= RA_RS_ATOMIC_WRITE_VERSION);
        HCCL_INFO(
            "%s deviceType[%d] devicePhyId[%u], version[%u], isSupportAtomicWrite[%d]", __func__, deviceType,
            devicePhyId, version, isSupportAtomicWrite);
    } else {
        isSupportAtomicWrite = false;
        HCCL_INFO("%s deviceType[%d] not support", __func__, deviceType);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
