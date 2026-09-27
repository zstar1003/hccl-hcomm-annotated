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
#define HCCL_VM_MODULE "STUB"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

#include "ascend_hal.h"
#include "device_sqe_parse_stub.h"
#include "hccl_device_pub.h"
#include "sim_models.h"
#include "db_sim_runner_common.h"
#include "sim_log.h"

using namespace HcclSim;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

drvError_t halEschedAttachDevice(unsigned int devId)
{
    (void)devId;
    HCCL_VM_INFO("AttachDevice is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halEschedDettachDevice(unsigned int devId)
{
    (void)devId;
    HCCL_VM_INFO("DettachDevice is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halEschedCreateGrpEx(uint32_t devId, struct esched_grp_para* grpPara, unsigned int* grpId)
{
    (void)devId;
    (void)grpPara;
    (void)grpId;
    HCCL_VM_INFO("CreateGrpEx is empty.");
    return DRV_ERROR_NONE;
}

drvError_t
halEschedSubscribeEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, unsigned long long eventBitmap)
{
    (void)devId;
    (void)grpId;
    (void)threadId;
    (void)eventBitmap;
    HCCL_VM_INFO("SubscribeEvent is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halEschedQueryInfo(
    unsigned int devId, ESCHED_QUERY_TYPE type, struct esched_input_info* inPut, struct esched_output_info* outPut)
{
    (void)devId;
    (void)type;
    (void)inPut;
    (void)outPut;
    HCCL_VM_INFO("QueryInfo is empty.");
    return DRV_ERROR_NONE;
}

drvError_t
halEschedWaitEvent(unsigned int devId, unsigned int grpId, unsigned int threadId, int timeout, struct event_info* event)
{
    (void)devId;
    (void)grpId;
    (void)threadId;
    (void)timeout;
    (void)event;
    HCCL_VM_INFO("WaitEvent is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halEschedRegisterAckFunc(
    unsigned int grpId, EVENT_ID eventId,
    void (*ackFunc)(unsigned int devId, unsigned int subevent_id, char* msg, unsigned int msgLen))
{
    (void)grpId;
    (void)eventId;
    (void)ackFunc;
    HCCL_VM_INFO("RegisterAckFunc is empty.");
    return DRV_ERROR_NONE;
}

drvError_t drvGetDevNum(uint32_t* num_dev)
{
    *num_dev = 1;
    HCCL_VM_INFO("drvGetDevNum:{}", *num_dev);
    return DRV_ERROR_NONE;
}

drvError_t halGetChipInfo(unsigned int devId, halChipInfo* chipInfo)
{
    sim::Device locDevice{};
    if (sim::GetDeviceByPhysicalId(devId, locDevice) != ACL_SUCCESS) {
        HCCL_VM_ERROR("halGetChipInfo get device by physic id {} failed.", devId);
        return DRV_ERROR_INNER_ERR;
    }
    strncpy((char*)chipInfo->name, locDevice.soc_version, MAX_CHIP_NAME - 1);
    HCCL_VM_INFO("GetChipInfo is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halHostRegister(void* src_ptr, UINT64 size, UINT32 flag, UINT32 devid, void** dst_ptr)
{
    (void)src_ptr;
    (void)size;
    (void)flag;
    (void)devid;
    (void)dst_ptr;
    HCCL_VM_INFO("HostRegister is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halHostUnregister(void* src_ptr, UINT32 devid)
{
    (void)src_ptr;
    (void)devid;
    HCCL_VM_INFO("HostUnregister is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halMemCtl(int type, void* param_value, size_t param_value_size, void* out_value, size_t* out_size_ret)
{
    (void)type;
    (void)param_value;
    (void)param_value_size;
    (void)out_value;
    (void)out_size_ret;
    HCCL_VM_INFO("MemCtl is empty.");
    return DRV_ERROR_NONE;
}

drvError_t drvGetPlatformInfo(uint32_t* info)
{
    *info = 1;
    HCCL_VM_INFO("drvGetPlatformInfo:{}", *info);
    return DRV_ERROR_NONE;
}

drvError_t drvQueryProcessHostPid(
    int pid, unsigned int* chip_id, unsigned int* vfid, unsigned int* host_pid, unsigned int* cp_type)
{
    (void)pid;
    (void)chip_id;
    (void)vfid;
    (void)host_pid;
    (void)cp_type;
    HCCL_VM_INFO("QueryProcessHostPid is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    (void)devId;
    if (moduleType == (int32_t)MODULE_TYPE_SYSTEM && infoType == (int32_t)INFO_TYPE_VERSION) {
        int64_t hardwareVersion = 0xf00;
    } else if ((moduleType == (int32_t)MODULE_TYPE_SYSTEM) && (infoType == (int32_t)INFO_TYPE_CORE_NUM)) {
        *value = 1;
    } else {
        *value = 0;
    }
    // *value = hardwareVersion;
    return DRV_ERROR_NONE;
}

drvError_t halEschedSubmitEvent(unsigned int devId, struct event_summary* event)
{
    (void)devId;
    (void)event;
    HCCL_VM_INFO("SubmitEvent is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halBindCgroup(BIND_CGROUP_TYPE bindType)
{
    (void)bindType;
    HCCL_VM_INFO("BindCgroup is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halGetAPIVersion(int* halAPIVersion)
{
    (void)halAPIVersion;
    HCCL_VM_INFO("GetAPIVersion is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeRegister(uint32_t devId, struct halSensorNodeCfg* cfg, uint64_t* handle)
{
    (void)devId;
    (void)cfg;
    (void)handle;
    HCCL_VM_INFO("RegisterSensorNode is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUnregister(uint32_t devId, uint64_t handle)
{
    (void)devId;
    (void)handle;
    HCCL_VM_INFO("UnregisterSensorNode is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halSensorNodeUpdateState(uint32_t devId, uint64_t handle, int val, halGeneralEventType_t assertion)
{
    (void)devId;
    (void)handle;
    (void)val;
    (void)assertion;
    HCCL_VM_INFO("UpdateSensorNodeState is empty.");
    return DRV_ERROR_NONE;
}

drvError_t drvDeviceGetPhyIdByIndex(uint32_t devIndex, uint32_t* phyId)
{
    (void)devIndex;
    (void)phyId;
    return DRV_ERROR_NONE;
}

drvError_t drvMemcpy(DVdeviceptr dst, size_t dest_max, DVdeviceptr src, size_t byte_count)
{
    (void)dst;
    (void)dest_max;
    (void)src;
    (void)byte_count;
    return DRV_ERROR_NONE;
}

int halGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen)
{
    (void)cmd;
    (void)inBuff;
    (void)inLen;
    (void)outBuff;
    (void)outLen;
    HCCL_VM_INFO("GrpQuery is empty.");
    return 0;
}

pid_t drvDeviceGetBareTgid(void)
{
    pid_t pid = getpid();
    HCCL_VM_INFO("drvDeviceGetBareTgid] pid:{}", pid);
    return pid;
}

drvError_t halResourceIdCheck(struct drvResIdKey* info)
{
    (void)info;
    HCCL_VM_INFO("ResourceIdCheck is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halSqCqQuery(uint32_t devId, struct halSqCqQueryInfo* info)
{
    (void)devId;
    HCCL_VM_INFO("SqCqQuery is empty.");
    if (info == nullptr) {
        return DRV_ERROR_INNER_ERR;
    }
    int head = GetSqTail(info->value[0]);
    switch (info->prop) {
        case DRV_SQCQ_PROP_SQ_HEAD: {
            info->value[0] = head;
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_DEPTH: {
            info->value[0] = HCCL_SQE_MAX_CNT; // 2048
            return DRV_ERROR_NONE;
        }
        case DRV_SQCQ_PROP_SQ_TAIL: {
            info->value[0] = head; // 代表sqBuffer一直为空
            return DRV_ERROR_NONE;
        };
        case DRV_SQCQ_PROP_SQ_BASE: {
            uint8_t* buffer = nullptr;
            GetSqBufferAddr(&buffer);
            info->value[0] = reinterpret_cast<uintptr_t>(buffer) & 0xFFFFFFFF;
            info->value[1] = reinterpret_cast<uintptr_t>(buffer) >> 32;
        }
        default:
            return DRV_ERROR_NONE;
    }
    return DRV_ERROR_NONE;
}

drvError_t halSqCqConfig(uint32_t devId, struct halSqCqConfigInfo* info)
{
    HCCL_VM_INFO("devId[{}] is empty.", devId);
    if (info->prop == DRV_SQCQ_PROP_SQ_TAIL) {
        ParseA5SqeFromSqBuffer(devId, info);
    }
    return DRV_ERROR_NONE;
}

drvError_t halShrIdInfoGet(const char* name, struct shrIdGetInfo* info)
{
    (void)name;
    (void)info;
    HCCL_VM_INFO("is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halTsdrvCtl(uint32_t devId, int cmd, void* param, size_t paramSize, void* out, size_t* outSize)
{
    (void)devId;
    (void)cmd;
    (void)param;
    (void)paramSize;
    (void)out;
    (void)outSize;
    HCCL_VM_INFO("is empty.");
    return DRV_ERROR_NONE;
}

drvError_t halResourceIdRestore(struct drvResIdKey* info)
{
    (void)info;
    HCCL_VM_INFO("is empty.");
    return DRV_ERROR_NONE;
}

drvError_t drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t* local_dev_id)
{
    *local_dev_id = host_dev_id;
    HCCL_VM_INFO("host_dev_id:{}, local_dev_id:{}", host_dev_id, *local_dev_id);
    return DRV_ERROR_NONE;
}

#ifdef __cplusplus
}
#endif
