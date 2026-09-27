/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_INNER_NET_DEVICE_MANAGER_H
#define HCCLV2_INNER_NET_DEVICE_MANAGER_H

#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include "port.h"
#include "orion_adapter_hccp.h"
#include "ip_address.h"
#include "inner_net_dev.h"
#include "net_device.h"

namespace Hccl {
class InnerNetDevManager {
public:
    // 获取单例实例（线程安全）
    static InnerNetDevManager& GetInstance();

    // 删除拷贝构造和赋值运算符，防止实例复制
    InnerNetDevManager(const InnerNetDevManager&) = delete;
    InnerNetDevManager& operator=(const InnerNetDevManager&) = delete;

    // 添加设备（增加引用计数）
    HcclResult AddDevice(const NetDevInfo& info, HcclNetDevice*& device);

    // 删除设备（减少引用计数，计数为0时删除）
    HcclResult DeleteDevice(HcclNetDevice* device);

    // 删除设备（减少引用计数，计数为0时删除）
    HcclResult RemoveDevice(const NetDevInfo& info);

private:
    // 私有构造函数和析构函数
    InnerNetDevManager() = default;
    ~InnerNetDevManager() = default;

    std::unordered_map<NetDevInfo, unique_ptr<InnerNetDev>> netDevMap_;
    std::unordered_map<NetDevInfo, uint32_t> netDevCnt_;

    mutable std::shared_mutex mtx_{};
};

} // namespace Hccl

#endif // HCCLV2_INNER_NET_DEVICE_MANAGER_H
