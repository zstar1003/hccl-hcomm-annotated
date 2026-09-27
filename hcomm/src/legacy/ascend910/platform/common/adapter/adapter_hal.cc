/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <thread>
#include <adapter_hal.h>
#include "dlhal_function.h"
#include "log.h"
#include "sal_pub.h"
#include "aicpu_schedule/aicpu_mc2_maintenance_thread.h"

using namespace hccl;

int (*g_grpIdCallback)(int tag, int* grpId, int* devId) = nullptr;
#ifdef __cplusplus
extern "C" {
#endif

constexpr u32 MEMORY_PAGE_SIZE = 4096;
constexpr u32 PRE_FETCH_THREADS_NUM = 24;
constexpr u32 PRE_FETCH_MEMORY_THRESHOLD = 268435456; // 考虑到多线程预访问也有开销，只有在内存大于256MB时才启动预访问

drvError_t __attribute__((weak))
halResourceIdRestore(struct drvResIdKey* info); // custom进程notify资源同步，在调用halResourceIdCheck前调用

HcclResult HcclSetGrpIdCallback(int (*grpIdCallback)(int tag, int* grpId, int* devId))
{
    if (grpIdCallback == nullptr) {
        HCCL_ERROR("para grpIdCallback is nullptr");
        return HCCL_E_PARA;
    }

    g_grpIdCallback = grpIdCallback;
    HCCL_DEBUG("New grpIdCallback was set, grpIdCallback[%p]", grpIdCallback);
    return HCCL_SUCCESS;
}

#ifdef __cplusplus
}
#endif

s32 hrtGetgrpId(int& groupId, int& devId)
{
    if (g_grpIdCallback == nullptr) {
        groupId = HCCL_ESCHED_GROUP_ID;
        devId = HCCL_RESERVED_DEV_ID;

        HCCL_DEBUG("g_grpIdCallback is nullptr, event.grp_id[%d] = HCCL_ESCHED_GROUP_ID", groupId);
    } else {
        HCCL_DEBUG("g_grpIdCallback is not nullptr");
        s32 ret = g_grpIdCallback(0, &groupId, &devId);
        if (ret != 0) {
            HCCL_ERROR("g_grpIdCallback failed, ret[%d]", ret);
            return ret;
        }

        HCCL_DEBUG("g_grpIdCallback is not nullptr, g_grpIdCallback(0) event.grp_id[%d], devId[%d]", groupId, devId);
    }

    return 0;
}

HcclResult hrtHalSubmitEvent(u32 devId, u32 eventId, u32 groupId)
{
    struct event_summary event = {};
    event.pid = SalGetPid();
    event.grp_id = groupId;
    event.event_id = static_cast<EVENT_ID>(eventId);
    event.subevent_id = 0;
    event.msg_len = 0;
    event.msg = nullptr;
    event.dst_engine = ACPU_DEVICE;
    event.policy = ONLY;
    for (u32 i = 0; i < EVENT_SUMMARY_RSV; i++) {
        event.rsv[i] = 0;
    }

    HCCL_DEBUG("SubmitEvent: event id:%u, pid:%d, grp id:%u, dev id:%u", eventId, event.pid, event.grp_id, devId);

    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedSubmitEvent(devId, &event);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[Submit][Event]errNo[0x%016llx] halEschedSubmitEvent fail,"
            "return[%d], para: deviceLogicId[%u] group[%u] eventId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId, event.grp_id, eventId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedAttachDevice(unsigned int devId)
{
    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedAttachDevice(devId);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[Attach][Device]errNo[0x%016llx] hrtHalEschedAttachDevice fail,"
            "return[%d], para: deviceLogicId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedDettachDevice(unsigned int devId)
{
    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedDettachDevice(devId);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[Detach][Device]errNo[0x%016llx] hrtHalEschedDettachDevice fail,"
            "return[%d], para: deviceLogicId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalGetAPIVersion(int& apiVersion)
{
    drvError_t ret = DlHalFunction::GetInstance().dlHalGetAPIVersion(&apiVersion);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[Get][APIVersion]errNo[0x%016llx] dlHalGetAPIVersion fail,"
            "return[%d], para: apiVersion[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, apiVersion),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedCreateGrp(unsigned int devId, unsigned int grpId, GROUP_TYPE type)
{
    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedCreateGrp(devId, grpId, type);
    CHK_PRT_RET(
        ret == DRV_ERROR_GROUP_EXIST,
        HCCL_INFO("hal event group is exist, devId[%u] grpId[%u] group type[%u]", devId, grpId, type), HCCL_SUCCESS);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalEschedCreateGrp]errNo[0x%016llx] hrtHalEschedCreateGrp fail,"
            "return[%d], para: devId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedCreateGrpEx(
    [[maybe_unused]] unsigned int devId, [[maybe_unused]] unsigned int* grpId, [[maybe_unused]] unsigned int threadNum,
    [[maybe_unused]] GROUP_TYPE type)
{
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedSubscribeEvent(
    [[maybe_unused]] unsigned int devId, [[maybe_unused]] unsigned int grpId, [[maybe_unused]] unsigned int threadId,
    [[maybe_unused]] unsigned long long eventBitmap)
{
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedWaitEvent(
    [[maybe_unused]] unsigned int devId, [[maybe_unused]] unsigned int grpId, [[maybe_unused]] unsigned int threadId,
    [[maybe_unused]] int timeout, [[maybe_unused]] struct event_info* event)
{
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedRegisterAckFunc(
    unsigned int eventId, void (*ackFunc)(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen))
{
    CHK_PTR_NULL(ackFunc);
    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedRegisterAckFunc(HCCL_ESCHED_GROUP_ID, eventId, ackFunc);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalEschedRegisterAckFunc fail,"
            "return[%d], para: eventId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, eventId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalEschedRegisterAckFuncWithGrpid(
    unsigned int grpid, unsigned int eventId,
    void (*ackFunc)(unsigned int devId, unsigned int subeventId, u8* msg, unsigned int msgLen))
{
    CHK_PTR_NULL(ackFunc);
    drvError_t ret = DlHalFunction::GetInstance().dlHalEschedRegisterAckFunc(grpid, eventId, ackFunc);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalEschedRegisterAckFunc fail,"
            "return[%d], para: eventId[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, eventId),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalDrvOpenIpcNotify([[maybe_unused]] const char* name, [[maybe_unused]] struct drvIpcNotifyInfo* notify)
{
    return HCCL_SUCCESS;
}

HcclResult hrtHalDrvCloseIpcNotify([[maybe_unused]] const char* name, [[maybe_unused]] struct drvIpcNotifyInfo* notify)
{
    return HCCL_SUCCESS;
}

HcclResult hrtHalDrvIpcNotifyRecord([[maybe_unused]] const char* name) { return HCCL_SUCCESS; }

HcclResult HrtHalDrvQueryProcessHostPid(
    int pid, unsigned int* chipId, unsigned int* vfid, unsigned int* hostPid, unsigned int* cpType)
{
    if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    }

    CHK_PTR_NULL(hostPid);
    // 和底软确认，chipId、vfid、hostPid、cpType不需要校验空指针，如果传入空指针表示当前不获取该值
    drvError_t ret = DlHalFunction::GetInstance().dlHalDrvQueryProcessHostPid(pid, chipId, vfid, hostPid, cpType);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] HrtHalDrvQueryProcessHostPid fail,"
            "return[%d], para: pid[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, pid),
        HCCL_E_DRV);
    HCCL_INFO("HrtHalDrvQueryProcessHostPid pid[%d] hostPid[%u]", pid, *hostPid);
    return HCCL_SUCCESS;
}

HcclResult HrtHalDrvGetDevIDByLocalDevID(uint32_t localDevId, uint32_t* devId)
{
    if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    }

    CHK_PTR_NULL(devId);
    drvError_t ret = DlHalFunction::GetInstance().dlHalDrvGetDevIDByLocalDevID(localDevId, devId);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] HrtHalDrvGetDevIDByLocalDevID fail,"
            "return[%d], para: localDevId[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, localDevId),
        HCCL_E_DRV);
    HCCL_INFO("HrtHalDrvGetDevIDByLocalDevID localDevId[%d] devId[%u]", localDevId, *devId);
    return HCCL_SUCCESS;
}

HcclResult hrtDrvMemCpy(void* dst, uint64_t destMax, const void* src, uint64_t count)
{
    // 参数有效性检查
    CHK_PTR_NULL(dst);
    CHK_PTR_NULL(src);

    uint64_t dstAddr = reinterpret_cast<uintptr_t>(dst);
    uint64_t srcAddr = reinterpret_cast<uintptr_t>(const_cast<void*>(src));
    drvError_t ret = DlHalFunction::GetInstance().dlDrvMemCpy(dstAddr, destMax, srcAddr, count);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtDrvMemCpy fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtDrvGetDevNum(uint32_t* num_dev)
{
    // 参数有效性检查
    CHK_PTR_NULL(num_dev);
    drvError_t ret = DlHalFunction::GetInstance().dlHalDrvGetDevNum(num_dev);
    if (ret == DRV_ERROR_NOT_SUPPORT) {
        // 通用服务器不支持该接口，默认num_dev为0，返回成功
        *num_dev = 0;
        return HCCL_SUCCESS;
    }
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtDrvGetDevNum fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalGetDeviceInfo(uint32_t devId, int32_t moduleType, int32_t infoType, int64_t* value)
{
    // 参数有效性检查
    CHK_PTR_NULL(value);
    if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    }
    HCCL_INFO("Entry-hrtHalGetDeviceInfo");
    drvError_t ret = DlHalFunction::GetInstance().dlHalGetDeviceInfo(devId, moduleType, infoType, value);
    CHK_PRT_RET(
        ret == DRV_ERROR_NOT_SUPPORT,
        HCCL_WARNING(
            "hrtHalGetDeviceInfo not support, "
            "return[%d].",
            HCCL_ERROR_CODE(DRV_ERROR_NOT_SUPPORT), ret),
        HCCL_E_NOT_SUPPORT);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalGetDeviceInfo fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

void hrtDrvDeviceGetBareTgid(s32& pid)
{
    pid = DlHalFunction::GetInstance().dlDrvDeviceGetBareTgid();

    return;
}

HcclResult hrtHalGrpQuery(GroupQueryCmdType cmd, void* inBuff, unsigned int inLen, void* outBuff, unsigned int* outLen)
{
    CHK_PTR_NULL(inBuff);
    CHK_PTR_NULL(outBuff);
    CHK_PTR_NULL(outLen);
    drvError_t ret = DlHalFunction::GetInstance().dlHalGrpQuery(cmd, inBuff, inLen, outBuff, outLen);
    CHK_PRT_RET(ret == DRV_ERROR_NOT_SUPPORT, HCCL_WARNING("dlHalGrpQuery is not support."), HCCL_SUCCESS);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalGrpQuery fail,"
            "return[%d], para: cmd[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, cmd),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtEschedQueryInfo(
    [[maybe_unused]] unsigned int devId, [[maybe_unused]] ESCHED_QUERY_TYPE type,
    [[maybe_unused]] struct esched_input_info* inPut, [[maybe_unused]] struct esched_output_info* outPut)
{
    return HCCL_SUCCESS;
}

HcclResult hrtDrvGetPlatformInfo(uint32_t* info)
{
    // 参数有效性检查
    CHK_PTR_NULL(info);
    CHK_SMART_PTR_NULL(DlHalFunction::GetInstance().dlHalDrvGetPlatformInfo);
    HCCL_INFO("Entry-hrtDrvGetPlatformInfo");
    drvError_t ret = DlHalFunction::GetInstance().dlHalDrvGetPlatformInfo(info);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtDrvGetPlatformInfo fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalGetChipInfo(uint32_t devId, std::string& chipName)
{
    // 参数有效性检查
    CHK_SMART_PTR_NULL(DlHalFunction::GetInstance().dlHalGetChipInfo);
    halChipInfo chipInfo = {};
    drvError_t ret = DlHalFunction::GetInstance().dlHalGetChipInfo(devId, &chipInfo);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalGetChipInfo fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    chipName
        = std::string(reinterpret_cast<char*>(chipInfo.type)) + std::string(reinterpret_cast<char*>(chipInfo.name));
    HCCL_INFO("hrtHalGetChipInfo succ chipName[%s]", chipName.c_str());
    return HCCL_SUCCESS;
}

HcclResult hrtHalBindCgroup(BIND_CGROUP_TYPE bindType)
{
    if (DlHalFunction::GetInstance().dlHalBindCgroup == nullptr) {
        HCCL_ERROR("dlHalBindCgroup is nullptr, can not use dlHalBindCgroup");
        return HCCL_E_DRV;
    }
    drvError_t ret = DlHalFunction::GetInstance().dlHalBindCgroup(bindType);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "errNo[0x%016llx] hrtHalBindCgroup fail,"
            "return[%d].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtDrvDeviceGetPhyIdByIndex([[maybe_unused]] u32 deviceLogicId, [[maybe_unused]] u32& devicePhyId)
{
    return HCCL_SUCCESS;
}

void MemoryPreFetchImpl(u64 start, u64 end, u32 pageSize)
{
    // 多线程并发对待映射的内存进行逐页表的预访问
    volatile uint64_t* ptr = nullptr;
    for (u64 i = start; i < end; i += pageSize) {
        ptr = reinterpret_cast<volatile uint64_t*>(i);
        *ptr;
    }
}

HcclResult MemoryPreFetch(u64 size, void* hostPtr)
{
    if (size >= PRE_FETCH_MEMORY_THRESHOLD) {
        std::vector<std::unique_ptr<std::thread>> threads(PRE_FETCH_THREADS_NUM);
        u64 chunkSize = size / PRE_FETCH_THREADS_NUM;
        u64 hostPtrUpperLimit = size + reinterpret_cast<uintptr_t>(hostPtr);
        for (unsigned int i = 0; i < PRE_FETCH_THREADS_NUM; ++i) {
            u64 start = i * chunkSize + reinterpret_cast<uintptr_t>(hostPtr);
            u64 end = (start + chunkSize < hostPtrUpperLimit) ? start + chunkSize : hostPtrUpperLimit;
            threads[i].reset(new (std::nothrow) std::thread(&MemoryPreFetchImpl, start, end, MEMORY_PAGE_SIZE));
            CHK_PRT_RET(
                !threads[i], HCCL_ERROR("[MemoryPreFetch]threads[%d] threads reset failed.", i), HCCL_E_INTERNAL);
        }

        for (auto& thread : threads) {
            thread->join();
        }
    }
    return HCCL_SUCCESS;
}

HcclResult hrtHalHostRegister(void* hostPtr, u64 size, u32 flag, u32 devid, void*& devPtr)
{
    CHK_PTR_NULL(hostPtr);
    if (flag == HOST_MEM_MAP_DEV_PCIE_TH || flag == HOST_MEM_MAP_DEV) {
        CHK_RET(MemoryPreFetch(size, hostPtr));
    }

    if (DlHalFunction::GetInstance().dlHalHostRegister == nullptr) {
        HCCL_ERROR("dlHalHostRegister is nullptr, can not use dlHalHostRegister");
        return HCCL_E_DRV;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalHostRegister(hostPtr, size, flag, devid, &devPtr);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalHostRegister]errNo[0x%016llx]"
            "hrtHalHostRegister fail, return[%d], para: size[%llu] flag[%u] devid[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, size, flag, devid),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtHalHostUnregister(void* hostPtr, u32 devid)
{
    CHK_PTR_NULL(hostPtr);
    if (DlHalFunction::GetInstance().dlHalHostUnregister == nullptr) {
        HCCL_ERROR("dlHalHostUnregister is nullptr, can not use dlHalHostUnregister");
        return HCCL_E_DRV;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalHostUnregister(hostPtr, devid);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalHostUnregister]errNo[0x%016llx]"
            "hrtHalHostUnregister fail, return[%d], para: devid[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devid),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
}

HcclResult hrtHalHostUnregisterEx(void* hostPtr, u32 devid, u32 flag)
{
    CHK_PTR_NULL(hostPtr);
    if (DlHalFunction::GetInstance().dlHalHostUnregisterEx == nullptr) {
        HCCL_ERROR("dlHalHostUnregisterEx is nullptr, can not use dlHalHostUnregisterEx");
        return HCCL_E_DRV;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalHostUnregisterEx(hostPtr, devid, flag);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalHostUnregisterEx]errNo[0x%016llx]"
            "hrtHalHostUnregisterEx fail, return[%d], para: devid[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devid),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
}

HcclResult hrtHalMemCtl(int type, void* input, size_t inputSize, void* output, size_t* outputSize)
{
    CHK_PTR_NULL(input);
    CHK_PTR_NULL(output);
    CHK_PTR_NULL(outputSize);
    if (DlHalFunction::GetInstance().dlHalMemCtl == nullptr) {
        HCCL_ERROR("dlHalMemCtl is nullptr, can not use dlHalMemCtl");
        return HCCL_E_DRV;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalMemCtl(type, input, inputSize, output, outputSize);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[dlHalMemCtl]errNo[0x%016llx]"
            "dlHalMemCtl fail, return[%d], para: type[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, type),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
}

HcclResult hrtHalSensorNodeRegister([[maybe_unused]] uint32_t devId, [[maybe_unused]] uint64_t* handle)
{
#ifdef CCL_KERNEL
    CHK_PTR_NULL(handle);
    if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    }
    if (DlHalFunction::GetInstance().dlHalSensorNodeRegister == nullptr) {
        HCCL_ERROR("dlHalSensorNodeRegister is nullptr, can not use dlHalSensorNodeRegister");
        return HCCL_E_DRV;
    }

    constexpr uint32_t HCCL_ASSERT_EVENT_MASK = 0xE00; // 当前仅使能 bit 9-B
    constexpr uint32_t HCCL_DEASSERT_EVENT_MASK = 0x0; // 0x09/0x0A/0x0B 使能为通知事件
    // 构造 Sensor Node 注册参数
    halSensorNodeCfg cfg = {};
    cfg.NodeType = HAL_DMS_DEV_TYPE_HCCP;
    cfg.SensorType = SAFTY_STATE_SENSOR_TYTPE;
    cfg.AssertEventMask = HCCL_ASSERT_EVENT_MASK;
    cfg.DeassertEventMask = HCCL_DEASSERT_EVENT_MASK;

    auto const sRet = snprintf_s(cfg.name, sizeof(cfg.name), sizeof(cfg.name) - 1U, "hccl_%d", getpid());
    if (sRet <= 0) {
        HCCL_ERROR("[CannErrorReporter][ConstructSensorNodeCfg] sprintf_s name err, ret[%d].", sRet);
        return HCCL_E_INTERNAL;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalSensorNodeRegister(devId, &cfg, handle);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalSensorNodeRegister]errNo[0x%016llx]"
            "hrtHalSensorNodeRegister fail, return[%d], para: devid[%u], nodeName[%s], nodeType[%u], sensorType[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId, cfg.name, cfg.NodeType, cfg.SensorType),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[halSensorNodeRegister]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtHalSensorNodeUnregister([[maybe_unused]] uint32_t devId, [[maybe_unused]] uint64_t handle)
{
#ifdef CCL_KERNEL
    if (DlHalFunction::GetInstance().dlHalSensorNodeUnregister == nullptr) {
        HCCL_ERROR("dlHalSensorNodeUnregister is nullptr, can not use dlHalSensorNodeUnregister");
        return HCCL_E_DRV;
    }

    drvError_t ret = DlHalFunction::GetInstance().dlHalSensorNodeUnregister(devId, handle);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtHalSensorNodeUnregister]errNo[0x%016llx]"
            "hrtHalSensorNodeUnregister fail, return[%d], para: devid[%u], handle[%llu].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId, handle),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[halSensorNodeUnregister]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}
halGeneralEventType_t TranslateEventType(HcclGeneralEventType assertion)
{
    switch (assertion) {
        case HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_RESUME:
            return GENERAL_EVENT_TYPE_RESUME;

        case HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_OCCUR:
            return GENERAL_EVENT_TYPE_OCCUR;

        case HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_ONE_TIME:
            return GENERAL_EVENT_TYPE_ONE_TIME;

        case HcclGeneralEventType::HCCL_GENERAL_EVENT_TYPE_MAX:
            return GENERAL_EVENT_TYPE_MAX;

        default: {
            HCCL_ERROR("[TranslateEventType]Not support the General Event type[%d].", assertion);
            return GENERAL_EVENT_TYPE_MAX;
        }
    }
}

HcclResult hrtHalSensorNodeUpdateState(
    [[maybe_unused]] uint32_t devId, [[maybe_unused]] uint64_t handle, [[maybe_unused]] int val,
    [[maybe_unused]] HcclGeneralEventType assertion)
{
#ifdef CCL_KERNEL
    if (DlHalFunction::GetInstance().dlHalSensorNodeUpdateState == nullptr) {
        HCCL_ERROR("dlHalSensorNodeUpdateState is nullptr, can not use dlHalSensorNodeUpdateState");
        return HCCL_E_DRV;
    }

    halGeneralEventType_t type = TranslateEventType(assertion);
    drvError_t ret = DlHalFunction::GetInstance().dlHalSensorNodeUpdateState(devId, handle, val, type);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_RUN_WARNING(
            "[hrtHalSensorNodeUpdateState]errNo[0x%016llx]"
            "hrtHalSensorNodeUpdateState fail, return[%d], para: devid[%u], handle[%llu], val[%d], assertion[%u].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, devId, handle, val, assertion),
        HCCL_E_DRV);

    return HCCL_SUCCESS;
#else
    HCCL_ERROR("[halSensorNodeUpdateState]Does not support this interface.");
    return HCCL_E_NOT_SUPPORT;
#endif
}

HcclResult hrtHalGetDeviceType(const uint32_t devId, DevType& devType)
{
    if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
        CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
    }

    std::string chipName;
    HcclResult ret = hrtHalGetChipInfo(devId, chipName);
    if (ret != HCCL_SUCCESS) {
        HCCL_ERROR("hrtHalGetChipInfo failed, ret[%d], devId[%u]", ret, devId);
        devType = DevType::DEV_TYPE_COUNT;
        return ret;
    }
    HCCL_INFO("[Get][DeviceType]Chip name[%s].", chipName.c_str());

    if (chipName.find("Ascend950") != std::string::npos) {
        devType = DevType::DEV_TYPE_950;
        return HCCL_SUCCESS;
    }

    if (chipName.find("Ascend960") != std::string::npos || chipName.find("ascend960") != std::string::npos) {
        devType = DevType::DEV_TYPE_960;
        return HCCL_SUCCESS;
    }

    auto iter = SOC_VER_CONVERT.find(chipName);
    if (iter == SOC_VER_CONVERT.end()) {
        HCCL_ERROR(
            "[Get][DeviceType]errNo[0x%016llx] hrtHalGetChipInfo get illegal chipver, chipName[%s].",
            HCCL_ERROR_CODE(HCCL_E_DRV), chipName.c_str());
        devType = DevType::DEV_TYPE_COUNT;
        return HCCL_E_DRV;
    }
    devType = iter->second;

    HCCL_INFO("[Get][DeviceType]deviceId[%u] get devType[%d] success!", devId, devType);
    return HCCL_SUCCESS;
}

HcclResult GetRunSideIsDevice(bool& isDeviceSide)
{
    static bool deviceSide = false;
    static bool init = false;
    if (UNLIKELY(!init)) {
        if (UNLIKELY(!hccl::DlHalFunction::GetInstance().DlHalFunctionIsInit())) {
            CHK_RET(DlHalFunction::GetInstance().DlHalFunctionInit());
        }
        u32 info = 0;
        CHK_RET(hrtDrvGetPlatformInfo(&info));
        deviceSide = info == 0 ? true : false;
        init = true;
    }

    isDeviceSide = deviceSide;
    return HCCL_SUCCESS;
}

HcclResult CheckRunSideIsDevice()
{
    static bool isDeviceSide = false;
    static bool init = false;
    if (UNLIKELY(!init)) {
        CHK_RET(GetRunSideIsDevice(isDeviceSide));
        init = true;
    }
    if (UNLIKELY(!isDeviceSide)) {
        HCCL_ERROR("currently running on host");
        return HCCL_E_INTERNAL;
    }
    return HCCL_SUCCESS;
}
extern "C" {
drvError_t __attribute__((weak)) drvGetLocalDevIDByHostDevID(uint32_t host_dev_id, uint32_t* local_dev_id);
drvError_t __attribute__((weak)) drvMemSmmuQuery(DVdevice device, uint32_t* SSID);
int32_t __attribute__((weak)) StartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);
int32_t __attribute__((weak)) AicpuCreateCtrlThread(uint32_t type, mc2Funcs f1, void* p1, mc2Funcs f2, void* p2);
};

HcclResult hrtDrvGetLocalDevIDByHostDevID(u32 hostUdevid, u32* localDevid)
{
    CHK_PTR_NULL(drvGetLocalDevIDByHostDevID);
    CHK_PTR_NULL(localDevid);

    drvError_t ret = drvGetLocalDevIDByHostDevID(hostUdevid, localDevid);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtDrvGetLocalDevIDByHostDevID]errNo[0x%016llx]"
            "hrtDrvGetLocalDevIDByHostDevID fail, return[%d], para: hostUdevid[%u] localDevid[%p].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, hostUdevid, localDevid),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

HcclResult hrtDrvMemSmmuQuery(uint32_t localDevid, uint32_t* SSID)
{
    CHK_PTR_NULL(drvMemSmmuQuery);
    CHK_PTR_NULL(SSID);

    drvError_t ret = drvMemSmmuQuery(localDevid, SSID);
    CHK_PRT_RET(
        ret != DRV_ERROR_NONE,
        HCCL_ERROR(
            "[hrtDrvMemSmmuQuery]errNo[0x%016llx]"
            "hrtDrvMemSmmuQuery fail, return[%d], para: localDevid[%u] SSID[%p].",
            HCCL_ERROR_CODE(HCCL_E_DRV), ret, localDevid, SSID),
        HCCL_E_DRV);
    return HCCL_SUCCESS;
}

bool IsSupportStartMC2MaintenanceThread()
{
    return StartMC2MaintenanceThread != nullptr || AicpuCreateCtrlThread != nullptr;
}

HcclResult hrtHalStartMC2MaintenanceThread(mc2Funcs f1, void* p1, mc2Funcs f2, void* p2)
{
    CHK_PTR_NULL(f1);
    CHK_PTR_NULL(p1);
    CHK_PTR_NULL(f2);
    CHK_PTR_NULL(p2);

    int ret = 0;

    if (AicpuCreateCtrlThread != nullptr) {
        HCCL_INFO("[hrtHalStartMC2MaintenanceThread] AicpuCreateCtrlThread");
        ret = AicpuCreateCtrlThread(THREAD_TYPE_HCOM, f1, p1, f2, p2);
        CHK_PRT_RET(
            (ret != 0 && ret != AICPU_SCHEDULE_THREAD_ALREADY_EXISTS),
            HCCL_ERROR(
                "[AicpuCreateCtrlThread]errNo[0x%016llx]"
                "AicpuCreateCtrlThread fail, return[%d], para: mc2Funcs f1[%p] p1[%p], mc2Funcs f2[%p] p2[%p].",
                HCCL_ERROR_CODE(HCCL_E_DRV), ret, f1, p1, f2, p2),
            HCCL_E_DRV);
    } else if (StartMC2MaintenanceThread != nullptr) {
        HCCL_INFO("[hrtHalStartMC2MaintenanceThread] StartMC2MaintenanceThread");
        ret = StartMC2MaintenanceThread(f1, p1, f2, p2);
        CHK_PRT_RET(
            (ret != 0 && ret != AICPU_SCHEDULE_THREAD_ALREADY_EXISTS),
            HCCL_ERROR(
                "[StartMC2MaintenanceThread]errNo[0x%016llx]"
                "StartMC2MaintenanceThread fail, return[%d], para: mc2Funcs f1[%p] p1[%p], mc2Funcs f2[%p] p2[%p].",
                HCCL_ERROR_CODE(HCCL_E_DRV), ret, f1, p1, f2, p2),
            HCCL_E_DRV);
    } else {
        HCCL_ERROR(
            "[hrtHalStartMC2MaintenanceThread] AicpuCreateCtrlThread and StartMC2MaintenanceThread are both nullptr");
        return HCCL_E_DRV;
    }
    return HCCL_SUCCESS;
}

HcclResult hrtHalResourceIdRestore(u32 devId, u32 tsId, drvIdType_t resType, u32 resId, u32 flag)
{
    // 兼容老的12包，找不到符号时不报错，返回HCCL_E_NOT_SUPPORT由上层处理是否报错
    CHK_PRT_RET(
        halResourceIdRestore == nullptr,
        HCCL_WARNING("hrtHalResourceIdRestore not support, halResourceIdRestore is nullptr"), HCCL_E_NOT_SUPPORT);

    drvResIdKey resInfo = {};
    resInfo.ruDevId = devId;
    resInfo.tsId = tsId;
    resInfo.resType = resType;
    resInfo.resId = resId;
    resInfo.flag = flag;

    int checkResult = halResourceIdRestore(&resInfo);
    if (checkResult != 0) {
        HCCL_ERROR(
            "[drv api]res restore failed, result:%d, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u", checkResult,
            resInfo.resType, resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag);
        return HCCL_E_DRV;
    }

    HCCL_INFO(
        "res restore success, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u", resInfo.resType, resInfo.resId,
        resInfo.tsId, resInfo.ruDevId, resInfo.flag);
    return HCCL_SUCCESS;
}
