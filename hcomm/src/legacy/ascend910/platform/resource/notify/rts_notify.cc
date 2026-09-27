/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rts_notify.h"
#include "sal_pub.h"

namespace hccl {

constexpr u32 CHIP_ID = 32;

RtsNotify::RtsNotify(NotifyType notifyType) : NotifyBase(notifyType) {}

RtsNotify::RtsNotify(NotifyType notifyType, HcclNotifyInfo notifyInfo) : NotifyBase(notifyType, notifyInfo) {}

RtsNotify::RtsNotify(NotifyType notifyType, const HcclSignalInfo& notifyInfo) : NotifyBase(notifyType)
{
    (void)SetNotifyData(notifyInfo);
}

RtsNotify::~RtsNotify() { (void)Destroy(); }

HcclResult RtsNotify::Open()
{
    HCCL_DEBUG(
        "[RtsNotify][Open]remote withIpc[%d], notify type[%d], ipcName[%s].", notifyInfo_.ipcNotify.withIpc, notifyType,
        notifyInfo_.ipcNotify.ipcName);
    if (notifyInfo_.ipcNotify.withIpc) {
        if (notifyType == NotifyType::RUNTIME_NOTIFY) {
            CHK_RET(hrtIpcOpenNotify(&notifyPtr, notifyInfo_.ipcNotify.ipcName));
        } else {
            CHK_RET(hrtIpcOpenNotifyWithFlag(&notifyPtr, notifyInfo_.ipcNotify.ipcName, ACL_NOTIFY_DEVICE_USE_ONLY));
        }
    } else {
        notifyPtr = notifyInfo_.ipcNotify.ptr;
    }
    HCCL_DEBUG("[RtsNotify][Open]notifyPtr[%p], ipcNotify[%p].", notifyPtr, notifyInfo_.ipcNotify.ptr);

    CHK_PRT_RET(
        notifyPtr == nullptr,
        HCCL_ERROR(
            "[RtsNotify][Open]errNo[0x%016llx] Notify open failed. "
            "notify is nullptr",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
        HCCL_E_RUNTIME);

    inchip = false;
    isLocal = false;
    CHK_RET(UpdateNotifyInfo());
    CHK_RET(hrtNotifyGetAddr(notifyPtr, &address));

    return HCCL_SUCCESS;
}

HcclResult RtsNotify::Close() { return Destroy(); }

HcclResult RtsNotify::Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut)
{
    CHK_PTR_NULL(dispatcher);
    return reinterpret_cast<DispatcherPub*>(dispatcher)
        ->SignalWait(
            notifyPtr, stream, INVALID_VALUE_RANKID, INVALID_VALUE_RANKID, stage, inchip, INVALID_UINT, timeOut);
}

HcclResult RtsNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage)
{
    CHK_PTR_NULL(dispatcher);
    return reinterpret_cast<DispatcherPub*>(dispatcher)
        ->SignalRecord(notifyPtr, stream, INVALID_VALUE_RANKID, notifyInfo_.ipcNotify.offset, stage, inchip, address);
}

HcclResult
RtsNotify::Wait(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 timeOut, u32 userRank, u32 remoteUserRank)
{
    CHK_PTR_NULL(dispatcher);
    return reinterpret_cast<DispatcherPub*>(dispatcher)
        ->SignalWait(notifyPtr, stream, userRank, remoteUserRank, stage, inchip, INVALID_UINT, timeOut);
}

HcclResult RtsNotify::Post(Stream& stream, HcclDispatcher dispatcher, s32 stage, u32 remoteUserRank)
{
    CHK_PTR_NULL(dispatcher);
    return reinterpret_cast<DispatcherPub*>(dispatcher)
        ->SignalRecord(notifyPtr, stream, remoteUserRank, notifyInfo_.ipcNotify.offset, stage, inchip, address);
}

HcclResult RtsNotify::Post(Stream& stream)
{
    CHK_RET(hrtNotifyRecord(notifyPtr, stream.ptr()));
    return HCCL_SUCCESS;
}

HcclResult RtsNotify::Wait(Stream& stream, u32 timeOut)
{
    CHK_RET(hrtNotifyWaitWithTimeOut(notifyPtr, stream.ptr(), timeOut));
    return HCCL_SUCCESS;
}

HcclResult RtsNotify::SetIpc()
{
    SecIpcName_t ipcName;
    HcclResult ret = hrtIpcSetNotifyName(notifyPtr, reinterpret_cast<u8*>(ipcName.ipcName), sizeof(ipcName.ipcName));
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[SetIpc][hrtIpcSetNotifyName]errNo[0x%016llx] "
            "IPC set notify name fail. return[%d] name len=[%zu].",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME), ret, sizeof(ipcName.ipcName)),
        HCCL_E_RUNTIME);
    if (memcpy_s(
            notifyInfo_.ipcNotify.ipcName, HCCL_IPC_MEM_NAME_LEN, reinterpret_cast<char*>(ipcName.ipcName),
            sizeof(ipcName.ipcName))
        != EOK) {
        HCCL_ERROR("ipcName:%s, size:%u", ipcName.ipcName, sizeof(ipcName.ipcName));
        return HCCL_E_MEMORY;
    };
    HCCL_DEBUG("[RtsNotify][SetIpc]ipcName:%s, size:%u.", ipcName.ipcName, sizeof(ipcName.ipcName));
    CHK_RET(hrtNotifyGetAddr(notifyPtr, &address));
    CHK_RET(UpdateNotifyInfo());

    return HCCL_SUCCESS;
}

HcclResult RtsNotify::Grant(s64 recvId)
{
    // 设置notify 的白名单
    inchip = false;
    s32 pid = static_cast<s32>((recvId & 0x00000000FFFFFFFF));
    s32 localPid = 0;
    CHK_RET(SalGetBareTgid(&localPid)); // 当前进程id

    // 多进程操作多卡场景，notify pool用pid区分notify
    s32 sdid = static_cast<s32>((recvId & 0xFFFFFFFF00000000) >> 32);
    HCCL_DEBUG(
        "[RtsNotify][Grant]remote sdid[%016llx], remote pid[%d], local pid[%d], withIpc[%d].", sdid, pid, localPid,
        notifyInfo_.ipcNotify.withIpc);

    // 单进程多线程操作多卡场景，notify pool用rankId区分notify
    if (pid == localPid && sdid == INVALID_INT) {
        if (notifyType == NotifyType::RUNTIME_NOTIFY_MC2) {
            notifyInfo_.ipcNotify.withIpc = true;
        }
        return HCCL_SUCCESS;
    }

    notifyInfo_.ipcNotify.withIpc = true;

    if (sdid != INVALID_INT) {
        // recvId由s32的sdid和pid拼接而成, 高32位是sdid, 低32位是pid
        CHK_RET(hrtSetIpcNotifySuperPodPid(notifyPtr, sdid, &pid, IPC_NOTIFY_PID_ARRAY_SIZE));
    } else {
        CHK_RET(hrtSetIpcNotifyPid(notifyPtr, &pid, IPC_NOTIFY_PID_ARRAY_SIZE));
    }
    return HCCL_SUCCESS;
}

HcclResult RtsNotify::Alloc()
{
    s32 deviceId = 0;
    CHK_RET(hrtGetDevice(&deviceId));

    if (notifyType == NotifyType::RUNTIME_NOTIFY) {
        CHK_RET(hrtNotifyCreate(deviceId, &notifyPtr));
        CHK_RET(hrtGetNotifyID(notifyPtr, &id));
    } else {
        CHK_RET(hrtNotifyCreateWithFlag(deviceId, &notifyPtr));
    }
    CHK_PRT_RET(
        notifyPtr == nullptr,
        HCCL_ERROR(
            "[RtsNotify][Alloc]errNo[0x%016llx] Notify create failed. "
            "notify is nullptr",
            HCCL_ERROR_CODE(HCCL_E_RUNTIME)),
        HCCL_E_RUNTIME);
    if (notifyType == NotifyType::RUNTIME_NOTIFY_MC2) {
        CHK_RET(UpdateNotifyInfo());
    }
    CHK_RET(hrtNotifyGetOffset(notifyPtr, notifyInfo_.ipcNotify.offset));
    notifyInfo_.ipcNotify.ptr = notifyPtr;
    return HCCL_SUCCESS;
}

HcclResult RtsNotify::Destroy()
{
    // 本卡notify直接释放，非本卡判断且非单进程多线程场景直接释放
    if (notifyPtr != nullptr && (isLocal || notifyInfo_.ipcNotify.withIpc)) {
        CHK_RET(hrtNotifyDestroy(notifyPtr));
    }
    notifyPtr = nullptr;
    return HCCL_SUCCESS;
}

HcclResult RtsNotify::UpdateNotifyInfo()
{
    CHK_RET(hrtGetNotifyID(notifyPtr, &id));

    DevType devType_ = DevType::DEV_TYPE_COUNT;
    CHK_RET(hrtGetDeviceType(devType_));
    if (devType_ == DevType::DEV_TYPE_950 || devType_ == DevType::DEV_TYPE_960) {
        s32 deviceLogicId;
        CHK_RET(hrtGetDevice(&deviceLogicId));
        CHK_RET(hrtGetDevicePhyIdByIndex(static_cast<uint32_t>(deviceLogicId), devId));
        return HCCL_SUCCESS;
    }

    CHK_RET(hrtNotifyGetPhyInfo(notifyPtr, &devId, &tsId));

    rtNotifyPhyInfo notifyInfo;
    CHK_RET(hrtNotifyGetPhyInfoExt(notifyPtr, &notifyInfo));
    flag = notifyInfo.flag;

    return HCCL_SUCCESS;
}

extern "C" {
drvError_t __attribute__((weak)) halResourceIdCheck(struct drvResIdKey* info);
drvError_t __attribute__((weak)) halResourceIdInfoGet(struct drvResIdKey* key, drvResIdProcType type, uint64_t* value);
};

#ifdef CCL_KERNEL
static HcclResult GetResVerifyDevId(bool isLocalSignal, u32& resDevId)
{
    if (!isLocalSignal) {
        return HCCL_SUCCESS;
    }
    u32 localDevId = 0;
    u32 hostPid = 0;
    CHK_RET(HrtHalDrvQueryProcessHostPid(getpid(), &localDevId, nullptr, &hostPid, nullptr));
    if (localDevId >= CHIP_ID) {
        CHK_PRT(HrtHalDrvGetDevIDByLocalDevID(localDevId, &resDevId));
    }
    return HCCL_SUCCESS;
}

static HcclResult RestoreResourceId(drvResIdKey& resInfo)
{
    static bool init = false;
    if (!init) {
        CHK_PRT_RET(
            halResourceIdCheck == nullptr,
            HCCL_ERROR("halResourceIdCheck is nullptr, "
                       "Does not support this interface."),
            HCCL_E_DRV);
        CHK_PRT_RET(
            halResourceIdInfoGet == nullptr,
            HCCL_ERROR("halResourceIdInfoGet is nullptr, "
                       "Does not support this interface."),
            HCCL_E_DRV);
        init = true;
    }

    HcclResult ret
        = hrtHalResourceIdRestore(resInfo.ruDevId, resInfo.tsId, resInfo.resType, resInfo.resId, resInfo.flag);
    if (ret != HCCL_SUCCESS && ret != HCCL_E_NOT_SUPPORT) {
        HCCL_ERROR(
            "[drv api]res restore failed, result:%d, resType:%d, resId:%u, tsId:%d, ruDevId:%d, flag:%d", ret,
            resInfo.resType, resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag);
        return HCCL_E_DRV;
    }
    HCCL_DEBUG(
        "res restore end, ret:%d, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u", ret, resInfo.resType,
        resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag);
    return HCCL_SUCCESS;
}

static HcclResult CheckAndGetResourceAddr(drvResIdKey& resInfo, u64& address)
{
    int checkResult = halResourceIdCheck(&resInfo);
    if (checkResult != 0) {
        HCCL_ERROR(
            "[drv api]res check failed, result:%d, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u", checkResult,
            resInfo.resType, resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag);
        return HCCL_E_DRV;
    }
    HCCL_DEBUG(
        "res check success, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u", resInfo.resType, resInfo.resId,
        resInfo.tsId, resInfo.ruDevId, resInfo.flag);

    checkResult = halResourceIdInfoGet(&resInfo, TRS_RES_ID_ADDR, reinterpret_cast<uint64_t*>(&address));
    if (checkResult != 0) {
        HCCL_ERROR(
            "[drv api]res get addr failed, result:%d, resType:%d, resId:%u, tsId:%d, ruDevId:%u, flag:%u", checkResult,
            resInfo.resType, resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag);
        return HCCL_E_DRV;
    }
    HCCL_DEBUG(
        "res get write value success, resType:%d, resId:%u, tsId:%u, ruDevId:%u, flag:%u, addr:0x%lx", resInfo.resType,
        resInfo.resId, resInfo.tsId, resInfo.ruDevId, resInfo.flag, address);
    return HCCL_SUCCESS;
}
#endif

HcclResult RtsNotify::InitAndVerifySingleSignal(bool isLocalSignal)
{
    (void)isLocalSignal;
#ifdef CCL_KERNEL
    if (id == INVALID_UINT) {
        // 无效值不做校验
        HCCL_DEBUG("[%s]resId[%u] is invalid, need not check", __func__, id);
        return HCCL_SUCCESS;
    }

    u32 tmpDevId = devId;
    CHK_RET(GetResVerifyDevId(isLocalSignal, tmpDevId));

    drvResIdKey resInfo = {};
    resInfo.ruDevId = tmpDevId;
    resInfo.tsId = tsId;
    resInfo.resType = DRV_NOTIFY_ID;
    resInfo.resId = static_cast<uint32_t>(id);
    resInfo.flag = flag;
    resInfo.rsv[0] = 0; // 0 is reserved array idx
    resInfo.rsv[1] = 0; // 1 is reserved array idx
    resInfo.rsv[2] = 0; // 2 is reserved array idx

    CHK_RET(RestoreResourceId(resInfo));
    CHK_RET(CheckAndGetResourceAddr(resInfo, address));
#endif

    return HCCL_SUCCESS;
}
} // namespace hccl
