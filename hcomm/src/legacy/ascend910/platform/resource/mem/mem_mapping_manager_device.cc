/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mem_mapping_manager.h"
#include "private_types.h"
#include "adapter_hal.h"
#include "adapter_rts.h"
#include "dlhal_function.h"

namespace hccl {
HcclResult MemMappingManager::MapMem(s32 deviceLogicID, void* addr, u64 size, void*& devVA)
{
    if (IsRequireMapping(addr, size, devVA)) {
        DevType devType;
        CHK_RET(hrtHalGetDeviceType(deviceLogicID, devType));
        drvRegisterTpye registerTpye = HOST_MEM_MAP_DEV;
        if ((devType == DevType::DEV_TYPE_910B) || (devType == DevType::DEV_TYPE_910_93)) {
            // 910B环境传参要特殊处理
            registerTpye = HOST_MEM_MAP_DEV_PCIE_TH;
            HCCL_INFO(
                "[MemMappingManager][MapMem]hrtHalHostRegister addr[%p], size[%llu], flag[%u], devId[%u]", addr, size,
                HOST_MEM_MAP_DEV_PCIE_TH, deviceLogicID);
            CHK_RET(hrtHalHostRegister(addr, size, HOST_MEM_MAP_DEV_PCIE_TH, deviceLogicID, devVA));
        } else {
            CHK_RET(hrtHalHostRegister(addr, size, HOST_MEM_MAP_DEV, deviceLogicID, devVA));
        }
        HostMappingKey hostMappingKey(reinterpret_cast<u64>(addr), size);
        mappedHostToDevMap_[hostMappingKey].devVA = devVA;
        mappedHostToDevMap_[hostMappingKey].ref.Ref();
        mappedHostToDevMap_[hostMappingKey].registerTpye = registerTpye;
    }
    return HCCL_SUCCESS;
}

// 先去map找内存，找到后引用计数--，减到0后做解映射，从map移除
HcclResult MemMappingManager::ReleaseDevVA(s32 deviceLogicID, void* addr, u64 size)
{
    std::unique_lock<std::mutex> lockMapping(mappedHostToDevMutex_);
    DevType devType;
    CHK_RET(hrtHalGetDeviceType(deviceLogicID, devType));
    u64 userAddr = reinterpret_cast<u64>(addr);
    auto iter = SearchMappingMap(userAddr, size);
    CHK_PRT_RET(
        (iter == mappedHostToDevMap_.end()),
        HCCL_ERROR("[MemMappingManager][ReleaseDevVA]the memory dereged isn't been reged"), HCCL_E_PARA);
    if (iter->second.ref.Unref() == 0) {
        // 解除内存映射，注册与解注册的 flag 保持一致
        CHK_RET(hrtHalHostUnregisterEx(addr, deviceLogicID, iter->second.registerTpye));
        mappedHostToDevMap_.erase(iter->first);
        HCCL_INFO("[MemMappingManager][ReleaseDevVA]addr[%p], size[%llu] unregister success.", addr, size);
    }
    return HCCL_SUCCESS;
}
} // namespace hccl
