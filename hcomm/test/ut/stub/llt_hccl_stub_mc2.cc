/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_hccl_sqcq.h"

#include "common/aicpu_kfc_def.h"
#include "llt_hccl_stub_mc2.h"
#include <thread>
#include "hcomm_task_struct.h"
#include "type_def.h"

using mc2Funcs = void (*)(void*);
extern "C" {
drvError_t __attribute__((weak)) halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info);
drvError_t __attribute__((weak)) halResourceIdCheck(struct drvResIdKey* info);
drvError_t __attribute__((weak)) halResourceIdInfoGet(struct drvResIdKey* key, drvResIdProcType type, uint64_t* value);
drvError_t __attribute__((weak)) halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info);
drvError_t __attribute__((weak)) halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info);
drvError_t __attribute__((weak))
halTsdrvCtl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize);
drvError_t __attribute__((weak))
halResAddrMap(unsigned int devId, struct res_addr_info* res_info, unsigned long* va, unsigned int* len);
drvError_t __attribute__((weak)) drvGetLocalDevIDByHostDevID(uint32_t host_udevid, uint32_t* localDevid);
drvError_t __attribute__((weak)) drvMemSmmuQuery(uint32_t localDevid, uint32_t* SSID);
int32_t __attribute__((weak)) StartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);
int32_t __attribute__((weak)) AicpuCreateCtrlThread(uint32_t type, mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);
};

DevType g_stubDevType = DevType::DEV_TYPE_COUNT;
namespace {
uint8_t sqBuffer[64 * 32 * 64];
}
drvError_t halCqReportRecv(uint32_t devId, struct halReportRecvInfo* info)
{
    info->report_cqe_num = 0U;
    return drvError_t(0);
}

drvError_t halResourceIdCheck(struct drvResIdKey* info) { return drvError_t(0); }

drvError_t halResourceIdRestore(struct drvResIdKey* key) { return drvError_t(0); }

drvError_t halResourceIdInfoGet(struct drvResIdKey* key, drvResIdProcType type, uint64_t* value)
{
    return drvError_t(0);
}

drvError_t halResAddrMap(unsigned int devId, struct res_addr_info* res_info, unsigned long* va, unsigned int* len)
{
    return drvError_t(0);
}

drvError_t halTsdrvCtl(uint32_t devId, int type, void* param, size_t paramSize, void* out, size_t* outSize)
{
    ts_ctrl_msg_body_t* out_msg = (ts_ctrl_msg_body_t*)out;
    out_msg->u.query_task_ack_info.status = APP_ABORT_STATUS_INVALID;
    return DRV_ERROR_NONE;
}

drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info)
{
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    auto queryinfo = *info;
    int head = 1;
    switch (queryinfo.prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = head;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = 4096;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = head; // head == tail 代表执行结束
            return DRV_ERROR_NONE;
        };
        case DRV_SQCQ_PROP_SQ_BASE: {
            uint8_t* buffer = sqBuffer;
            info->value[0] = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;
            info->value[1] = reinterpret_cast<uintptr_t>(buffer) >> 32;
        }
        default:
            return DRV_ERROR_NONE;
    }
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info) { return drvError_t(0); }

drvError_t drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t* local_dev_id)
{
    local_dev_id = &host_dev_id;
    return drvError_t(0);
}

drvError_t drvMemSmmuQuery(uint32_t local_devid, uint32_t* SSID) { return drvError_t(0); }

drvError_t drvQueryProcessHostPid(
    int pid, unsigned int* chip_id, unsigned int* vfid, unsigned int* host_pid, unsigned int* cp_type)
{
    return drvError_t(0);
}

drvError_t drvGetDevIDByLocalDevID(unsigned int pid, unsigned int* chip_id) { return drvError_t(0); }

#define PLAT_COMBINE(arch, chip, ver) (((arch) << 16U) | ((chip) << 8U) | (ver))
#define PLAT_GET_ARCH(type) (((type) >> 16U) & 0xffffU)
#define PLAT_GET_CHIP(type) (((type) >> 8U) & 0xffU)

enum rtPGVersion_t {
    RT_VER_NA = 0,     /* Ascend910B4 */
    RT_VER_BIN1,       /* Ascend910B1 */
    RT_VER_BIN2,       /* Ascend910B2 */
    RT_VER_BIN3,       /* Ascend910B3 */
    RT_VER_BIN4,       /* reserved is same as driver */
    RT_VER_BIN8 = 8,   /* Ascend910B2C */
    RT_VER_BIN10 = 10, /* Ascend910B4_1 */
    RT_VER_END
};

typedef enum tagRtVersion {
    VER_BEGIN = 0,
    VER_NA = VER_BEGIN,
    VER_ES = 1,
    VER_CS = 2,
    VER_SD3403 = 3,
    VER_LITE = 4,
    VER_310M1 = 5,
    VER_END = 6,
} rtVersion_t;

typedef enum tagRtArchType {
    ARCH_BEGIN = 0,
    ARCH_V100 = ARCH_BEGIN,
    ARCH_V200 = 1,
    ARCH_V300 = 2,
    ARCH_C100 = 3,  /* Ascend910 */
    ARCH_C220 = 4,  /* Ascend910B & Ascend910_93 */
    ARCH_M100 = 5,  /* Ascend310 */
    ARCH_M200 = 6,  /* Ascend310P */
    ARCH_M201 = 7,  /* BS9SX1A */
    ARCH_T300 = 8,  /* Tiny */
    ARCH_N350 = 9,  /* Nano */
    ARCH_M300 = 10, /* Ascend310B */
    ARCH_M310 = 11, /* */
    ARCH_S200 = 12, /* Hi3796CV300ES & TsnsE */
    ARCH_S202 = 13, /* Hi3796CV300CS & OPTG & SD3403 &TsnsC */
    ARCH_M510 = 14, /* MC62CM12A */
    ARCH_END,
} rtArchType_t;

typedef enum tagRtChipType {
    CHIP_BEGIN = 0,
    CHIP_MINI = CHIP_BEGIN,
    CHIP_CLOUD = 1,
    CHIP_MDC = 2,
    CHIP_LHISI = 3,
    CHIP_DC = 4,
    CHIP_CLOUD_V2 = 5,
    CHIP_NO_DEVICE = 6,
    CHIP_MINI_V3 = 7,
    CHIP_5612 = 8, /* 1910b tiny */
    CHIP_NANO = 9,
    CHIP_1636 = 10,
    CHIP_AS31XM1 = 11,
    CHIP_610LITE = 12,
    CHIP_CLOUD_V3 = 13, // drive used, runtime not used
    CHIP_BS9SX1A = 14,  /* BS9SX1A */
    CHIP_DAVID = 15,
    CHIP_SOLOMON = 16,
    CHIP_MC62CM12A = 17, /* MC62CM12A */
    CHIP_END
} rtChipType_t;

enum devdrvHardwareVersion {
    DEVDRV_PLATFORM_ASCEND310P = PLAT_COMBINE(ARCH_V200, CHIP_DC, RT_VER_NA),
    DEVDRV_PLATFORM_CLOUD_V2 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_NA),
    DEVDRV_PLATFORM_CLOUD_V2_B1 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_BIN1),
    DEVDRV_PLATFORM_CLOUD_V2_B2 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_BIN2),
    DEVDRV_PLATFORM_CLOUD_V2_B3 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_BIN3),
    DEVDRV_PLATFORM_CLOUD_V2_B4 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_NA),
    DEVDRV_PLATFORM_CLOUD_V2_B2C = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, RT_VER_BIN8),
    DEVDRV_PLATFORM_CLOUD_V3 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD_V2, VER_NA),
    DEVDRV_PLATFORM_CLOUD_V1 = PLAT_COMBINE(ARCH_V100, CHIP_CLOUD, VER_NA),
    DEVDRV_PLATFORM_END
};

drvError_t StubhalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    if (g_stubDevType == DevType::DEV_TYPE_910B) { // 910B
        *value = DEVDRV_PLATFORM_CLOUD_V2_B1;
    } else if (g_stubDevType == DevType::DEV_TYPE_310P1 || g_stubDevType == DevType::DEV_TYPE_310P3) {
        *value = DEVDRV_PLATFORM_ASCEND310P;
    } else if (g_stubDevType == DevType::DEV_TYPE_910_93) {
        *value = DEVDRV_PLATFORM_CLOUD_V3;
    } else if (g_stubDevType == DevType::DEV_TYPE_910) {
        *value = DEVDRV_PLATFORM_CLOUD_V1;
    } else {
        return drvError_t(DRV_ERROR_NOT_SUPPORT);
    }
    return drvError_t(0);
}

int32_t StartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2)
{
    std::thread thread_tmp = std::thread(f1, static_cast<void*>(p1));
    sleep(1);
    f2(p2);
    if (thread_tmp.joinable()) {
        thread_tmp.join();
    }
    return 0U;
}

int32_t AicpuCreateCtrlThread(uint32_t type, mc2Funcs f1, void* p1, mc2Funcs f2, void* p2)
{
    std::thread thread_tmp = std::thread(f1, static_cast<void*>(p1));
    sleep(1);
    f2(p2);
    if (thread_tmp.joinable()) {
        thread_tmp.join();
    }
    return 0U;
}

int32_t AdprofCheckFeatureIsOn(uint64_t feature) { return 0; }

int32_t AdprofReportAdditionalInfo(uint32_t agingFlag, const void* data, uint32_t length) { return 0; }

int32_t AdprofReportBatchAdditionalInfo(uint32_t nonPersistantFlag, const void* data, uint32_t length) { return 0; }

uint64_t AdprofGetHashId(const char* hashInfo, size_t length)
{
    return std::hash<std::string>{}(std::string(hashInfo));
}

uint64_t MsprofStr2Id(const char* hashInfo, size_t length) { return std::hash<std::string>{}(std::string(hashInfo)); }

uint32_t AicpuGetStreamId() { return 0; }

uint64_t AicpuGetTaskId() { return 0; }

namespace aicpu {
status_t GetTaskAndStreamId(uint64_t& taskId, uint32_t& streamId)
{
    taskId = 0;
    streamId = 0;
    return AICPU_ERROR_NONE;
}

} // namespace aicpu
