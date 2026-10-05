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
#include "aicpu_channel_process.h"
#include "dev_aicpu_ts_channel_mgr.h"
#include "aicpu_res_package_helper.h"
#include "../channel.h"
#include "aicpu_ts_channel_helper.h"
#include "ub_transport_lite_impl.h"
#include "roce_transport_lite_impl.h"
#include "p2p_transport_lite_impl.h"
#include "aicpu_task_cache_manager.h"

#include "adapter_rts_common.h"
#include "log.h"

#include <securec.h>

#include <cstdint>
#include <vector>

std::mutex AicpuChannelProcess::mutex_;
std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>> AicpuChannelProcess::transportMap_;

namespace {
// 创建 lite transport 并登记进 transportMap_（handle 即 transport 指针，primitives 层按指针解引用）。
// ROCE 无额外配置走这里；UB 需在入表前设 cache callback，故单独内联。
// 注意：本函数不加锁，调用方必须已持有 AicpuChannelProcess::mutex_。
template <typename T>
HcclResult CreateAndInsertTransport(
    std::vector<char>& uniqueId, ChannelHandle& handle,
    std::unordered_map<ChannelHandle, std::unique_ptr<Hccl::BaseTransportLiteImpl>>& transportMap)
{
    std::unique_ptr<T> impl;
    EXCEPTION_CATCH(impl = std::make_unique<T>(uniqueId), return HCCL_E_PTR);
    CHK_SMART_PTR_NULL(impl);
    handle = ReinterpretAs<uint64_t>(impl.get());
    transportMap.insert({handle, std::move(impl)});
    return HCCL_SUCCESS;
}
} // namespace

// [中文导读] [AllReduce逐行 S51] AicpuChannelProcess::ParsePackData的接口声明：当前对象句柄；这些参数属于本函数调用边界。
HcclResult AicpuChannelProcess::ParsePackData(std::vector<char>& data, ChannelHandle& handle)
// [中文导读] [AllReduce逐行 S52] 进入AicpuChannelProcess::ParsePackData函数体：按打包TransportType创建 UB/RoCE/P2P设备传输对象，UB分支绑定缓存回调并输出对象句柄。
{
    // [中文导读] [AllReduce逐行 S53] 记录AicpuChannelProcess::ParsePackData的调试诊断；日志本身不执行传输。
    HCCL_DEBUG("[HcclCommAicpu][%s] data: ptr[%p], size[%u]", __func__, data.data(), data.size());
    // [中文导读] [AllReduce逐行 S54] 调用binaryStream，使用资源标识序列化/反序列化流；对象涉及资源标识序列化/反序列化流。
    Hccl::BinaryStream binaryStream(data);

    // [中文导读] [AllReduce逐行 S56] 准备待恢复的transport序列化标识的局部存储/结构描述，初始化方式以本行声明为准。
    std::vector<char> transpUniqueId;
    // [中文导读] [AllReduce逐行 S57] 从通道资源包反序列化transport唯一标识字节数组，随后用此标识恢复设备传输对象。
    binaryStream >> transpUniqueId;

    // [中文导读] [AllReduce逐行 S59] 调用binaryStreamForType，使用待恢复的transport序列化标识；对象涉及待恢复的transport序列化标识。
    Hccl::BinaryStream binaryStreamForType(transpUniqueId);
    // [中文导读] [AllReduce逐行 S60] 准备序列化transport类型的局部存储/结构描述，初始化方式以本行声明为准。
    u32 transType;
    // [中文导读] [AllReduce逐行 S61] 从transport唯一标识开头读取传输类型，用于选择UB/UBoE、RoCE或P2P具体设备对象。
    binaryStreamForType >> transType;
    // [中文导读] [AllReduce逐行 S62] 记录AicpuChannelProcess::ParsePackData的状态/性能诊断，字段包含序列化transport类型；日志本身不执行传输。
    HCCL_INFO("[CollCommAicpu][ParsePackData] transType[%u]", transType);
    // [中文导读] [AllReduce逐行 S63] 仅当`(transType == Hccl::TransportType::UB || transType == Hccl::TransportType::UBoE)`（序列化transport类型）成立时进入此分支。
    if (transType == Hccl::TransportType::UB || transType == Hccl::TransportType::UBoE) {
        // [中文导读] [AllReduce逐行 S64] 准备`std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransportLiteImpl`的局部存储/结构描述，初始化方式以本行声明为准。
        std::unique_ptr<Hccl::UbTransportLiteImpl> ubTransportLiteImpl;
        // [中文导读] [AllReduce逐行 S65] 在异常捕获边界执行后续表达式；异常按后续处理语句转换成HCCL状态或提前返回。
        EXCEPTION_CATCH(
            // [中文导读] [AllReduce逐行 S66] 为前述多行表达式补入`(ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)), return HCCL_E_PTR)`（待恢复的transport序列化标识）；本行是参数/结构化初始化续行。
            (ubTransportLiteImpl = std::make_unique<Hccl::UbTransportLiteImpl>(transpUniqueId)), return HCCL_E_PTR);
        // [中文导读] [AllReduce逐行 S67] 检查`ubTransportLiteImpl`不是空对象；宏命中失败条件时立即返回对应指针错误。
        CHK_SMART_PTR_NULL(ubTransportLiteImpl);
        // [中文导读] [AllReduce逐行 S68] 绑定设备任务缓存需求判断回调；返回非成功时由检查宏立即向上传递。
        CHK_RET(ubTransportLiteImpl->SetNeedCacheTaskCallback(hcomm::AicpuTaskCacheManager::NeedCacheTask));
        // [中文导读] [AllReduce逐行 S69] 绑定设备任务缓存WQE数组保存回调；返回非成功时由检查宏立即向上传递。
        CHK_RET(ubTransportLiteImpl->SetAddWqeArrayCallback(hcomm::AicpuTaskCacheManager::AddWqeArray));
        // [中文导读] [AllReduce逐行 S70] 把具体UbTransportLiteImpl对象地址编码为设备ChannelHandle；C原语随后以BaseTransportLiteImpl基类指针进行虚派发。
        handle = ReinterpretAs<uint64_t>(ubTransportLiteImpl.get());
        // [中文导读] [AllReduce逐行 S71] 将UB对象所有权移入transportMap_，以刚输出的设备句柄为键保持其生命周期。
        transportMap_.insert({handle, std::move(ubTransportLiteImpl)});
    // [中文导读] [AllReduce逐行 S72] 仅当`(transType == Hccl::TransportType::ROCE)`（序列化transport类型）成立时进入此分支。
    } else if (transType == Hccl::TransportType::ROCE) {
        // [中文导读] [AllReduce逐行 S73] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。
        CHK_RET(CreateAndInsertTransport<Hccl::RoceTransportLiteImpl>(transpUniqueId, handle, transportMap_));
    // [中文导读] [AllReduce逐行 S74] 仅当`(transType == Hccl::TransportType::P2P)`（序列化transport类型）成立时进入此分支。
    } else if (transType == Hccl::TransportType::P2P) {
        // [中文导读] [AllReduce逐行 S75] 执行本行包裹的资源/任务调用；返回非成功时由检查宏立即向上传递。
        CHK_RET(CreateAndInsertTransport<Hccl::P2PTransportLiteImpl>(transpUniqueId, handle, transportMap_));
    // [中文导读] [AllReduce逐行 S76] 进入上一个条件未命中的替代路径，按本函数的资源/设备/协议分流继续处理。
    } else {
        // [中文导读] [AllReduce逐行 S77] 记录AicpuChannelProcess::ParsePackData的错误诊断，字段包含序列化transport类型；日志本身不执行传输。
        HCCL_ERROR("[AicpuChannelProcess][%s] transType[%u] is invalid", __func__, transType);
        // [中文导读] [AllReduce逐行 S78] 返回HCCL_E_PARA，表示参数不满足此分支要求；此路径停止本函数的后续处理。
        return HCCL_E_PARA;
    // [中文导读] [AllReduce逐行 S79] 结束`if (transType == Hccl::TransportType::UB || transType == Hccl::TransportType::UBoE)`（序列化transport类型）分支/循环；控制流返回外层。
    }

    // [中文导读] [AllReduce逐行 S81] 当前路径返回成功状态；仅说明本函数处理/任务组织成功，完成语义由其具体调用职责决定。
    return HCCL_SUCCESS;
// [中文导读] [AllReduce逐行 S82] 结束AicpuChannelProcess::ParsePackData函数体；控制流返回外层。
}

HcclResult AicpuChannelProcess::InitUrmaChannel(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[HcclCommAicpu][%s] commParam->uniqueIdAddr[%p], commParam->uniqueIdSize[%u]", __func__,
        commParam->uniqueIdAddr, commParam->uniqueIdSize);

    u8* currentSrcAddr = ReinterpretAs<u8*>(commParam->uniqueIdAddr);
    u32* addSize = ReinterpretAs<u32*>(commParam->channelSizeAddr);
    for (u32 index = 0; index < commParam->listNum; index++) {
        std::vector<char> data(*addSize);

        CHK_SAFETY_FUNC_RET(memcpy_s(data.data(), data.size(), currentSrcAddr, *addSize));
        currentSrcAddr += *addSize;
        addSize++;
        Hccl::AicpuResPackageHelper helper;
        auto dataVec = helper.ParsePackedData(data);

        Hccl::AicpuResMgrType resType = Hccl::AicpuResMgrType::STREAM;
        if (static_cast<u32>(resType) >= dataVec.size()) {
            HCCL_ERROR("[HcclCommAicpu][%s] fail, resType[%d], dataVec size[%u]", __func__, resType, dataVec.size());
            return HCCL_E_PARA;
        }
        ChannelHandle channelHandle;
        CHK_RET(ParsePackData(dataVec[resType].data, channelHandle));

        if (commParam->ctxList != nullptr) {
            // ctx模式：device侧填充abiHeader + deviceChannel
            auto** ctxList = ReinterpretAs<HcommAicpuChannelCtx**>(commParam->ctxList);
            ctxList[index]->abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
            ctxList[index]->abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
            ctxList[index]->abiHeader.size = sizeof(HcommAicpuChannelCtx);
            ctxList[index]->deviceChannel = ReinterpretAs<void*>(channelHandle);
        } else {
            ChannelHandle* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);
            channelList[index] = channelHandle;
        }
        HCCL_INFO(
            "[HcclCommAicpu][%s] index[%u], currentSrcAddr[%p], channelSizeAddr[%p], channelHandle[0x%llx]", __func__,
            index, currentSrcAddr, commParam->channelSizeAddr, channelHandle);
    }

    return HCCL_SUCCESS;
}

HcclResult AicpuChannelProcess::AicpuChannelInit(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[AicpuChannelProcess][%s] commParam->channelList[%p], commParam->listNum[%u], commParam->uniqueIdAddr[%p], "
        "commParam->uniqueIdSize[%u]",
        __func__, commParam->channelList, commParam->listNum, commParam->uniqueIdAddr, commParam->uniqueIdSize);

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commParam->deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commParam->deviceType)));

    std::lock_guard<std::mutex> addLock(mutex_);

    HcclResult ret = InitUrmaChannel(commParam);
    CHK_PRT_RET(
        ret != HCCL_SUCCESS,
        HCCL_ERROR(
            "[AicpuChannelProcess][AicpuChannelInit]errNo[0x%016llx] Failed to init channels", HCCL_ERROR_CODE(ret)),
        ret);

    HCCL_INFO("[AicpuChannelProcess][%s] aicpuTask End.", __func__);
    return HCCL_SUCCESS;
}

namespace {

void RollbackDestroy(DevAicpuTsChannelMgr& mgr, const std::vector<ChannelHandle>& rollback)
{
    for (const auto& h : rollback) {
        if (mgr.DestroyChannel(h)) {
            HCCL_DEBUG("[AicpuChannelProcess][%s] rollback destroyed handle[0x%llx]", __func__, h);
        } else {
            HCCL_WARNING("[AicpuChannelProcess][%s] rollback failed to destroy handle[0x%llx]", __func__, h);
        }
    }
}

HcclResult CreateSingleHcommChannel(
    DevAicpuTsChannelMgr& mgr, void* dp, u64 sz, const HcommDeviceInfo& deviceInfo, hcomm::HcommChannelKind kind,
    HcommChannelRes* commParam, u32 index, ChannelHandle* channelList, std::vector<ChannelHandle>& rollback)
{
    DevAicpuTsChannel* channel = mgr.GetOrCreateAicpuTsChannel(kind);
    if (channel == nullptr) {
        HCCL_ERROR(
            "[AicpuChannelProcess][%s] index[%u] unsupported kind[%u]", __func__, index, static_cast<uint32_t>(kind));
        RollbackDestroy(mgr, rollback);
        return HCCL_E_NOT_SUPPORT;
    }
    CHK_PTR_NULL(dp);
    ChannelHandle h{};
    HcclResult pret = channel->Create(dp, sz, deviceInfo, h);
    if (pret != HCCL_SUCCESS) {
        HCCL_ERROR("[AicpuChannelProcess][%s] parse fail at index[%u]", __func__, index);
        RollbackDestroy(mgr, rollback);
        return pret;
    }
    if (commParam->ctxList != nullptr) {
        auto** ctxList = ReinterpretAs<HcommAicpuChannelCtx**>(commParam->ctxList);
        ctxList[index]->abiHeader.version = HCOMM_AICPU_CHANNEL_CTX_VERSION;
        ctxList[index]->abiHeader.magicWord = HCOMM_AICPU_CHANNEL_CTX_MAGIC_WORD;
        ctxList[index]->abiHeader.size = sizeof(HcommAicpuChannelCtx);
        ctxList[index]->deviceChannel = ReinterpretAs<void*>(h);
    } else {
        channelList[index] = h;
    }
    rollback.push_back(h);
    return HCCL_SUCCESS;
}

} // namespace

HcclResult AicpuChannelProcess::InitHcommChannelRes(HcommChannelRes* commParam)
{
    CHK_PTR_NULL(commParam);
    HCCL_INFO(
        "[AicpuChannelProcess][%s] channelList[%p], listNum[%u]", __func__, commParam->channelList, commParam->listNum);

    CHK_PTR_NULL(commParam->channelList);
    CHK_PTR_NULL(commParam->channelDataListAddr);
    CHK_PTR_NULL(commParam->channelDataSizeListAddr);
    CHK_PTR_NULL(commParam->channelTypeListAddr);

    CHK_RET(hrtSetWorkModeAicpu(true));
    CHK_RET(hrtSetlocalDevice(commParam->deviceInfo.deviceLogicId));
    CHK_RET(hrtSetlocalDeviceType(static_cast<DevType>(commParam->deviceInfo.deviceType)));

    void** dataList = ReinterpretAs<void**>(commParam->channelDataListAddr);
    auto* sizeList = ReinterpretAs<u64*>(commParam->channelDataSizeListAddr);
    auto* typeList = ReinterpretAs<u32*>(commParam->channelTypeListAddr);
    auto* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);

    auto& mgr = DevAicpuTsChannelMgr::Instance();
    std::vector<ChannelHandle> rollback;
    rollback.reserve(commParam->listNum);

    std::lock_guard<std::mutex> addLock(mutex_);
    for (u32 index = 0; index < commParam->listNum; ++index) {
        hcomm::HcommChannelKind kind = static_cast<hcomm::HcommChannelKind>(typeList[index]);
        CHK_RET(CreateSingleHcommChannel(
            mgr, dataList[index], sizeList[index], commParam->deviceInfo, kind, commParam, index, channelList,
            rollback));
        HCCL_INFO(
            "[AicpuChannelProcess][%s] index[%u] channelHandle[0x%llx]", __func__, index,
            commParam->ctxList != nullptr ? 0 : channelList[index]);
    }

    HCCL_INFO("[AicpuChannelProcess][%s] aicpu_task End.", __func__);
    return HCCL_SUCCESS;
}

HcclResult AicpuChannelProcess::AicpuChannelDestroy(HcclChannelUrmaRes* commParam)
{
    HCCL_INFO(
        "[AicpuChannelProcess][%s] commParam->channelList[%p], commParam->listNum[%u]", __func__,
        commParam->channelList, commParam->listNum);

    auto& mgr = DevAicpuTsChannelMgr::Instance();
    std::lock_guard<std::mutex> addLock(mutex_);

    ChannelHandle* channelList = ReinterpretAs<ChannelHandle*>(commParam->channelList);
    for (u32 index = 0; index < commParam->listNum; ++index) {
        ChannelHandle handle = channelList[index];

        auto it = transportMap_.find(handle);
        if (it != transportMap_.end()) {
            transportMap_.erase(it);
            HCCL_DEBUG("[AicpuChannelProcess][%s] destroyed lite transport handle[0x%llx]", __func__, handle);
            continue;
        }

        if (mgr.DestroyChannel(handle)) {
            HCCL_DEBUG("[AicpuChannelProcess][%s] destroyed hcomm res handle[0x%llx]", __func__, handle);
            continue;
        }

        HCCL_WARNING(
            "[AicpuChannelProcess][%s] handle[0x%llx] not found in ub/hcomm maps, maybe already destroyed?", __func__,
            handle);
    }

    HCCL_INFO("[AicpuChannelProcess][%s] aicpu_task End.", __func__);
    return HCCL_SUCCESS;
}
