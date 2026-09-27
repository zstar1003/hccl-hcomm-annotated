/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef EXCHANGE_DATA_FORMAT_H
#define EXCHANGE_DATA_FORMAT_H

#include <cstdint>
#include <cstring>
#include "hccl_common.h"
#include "hccp_common.h"

namespace hcomm {

// ========== RoCE 混合模式（Cross-Mode）常量定义 ==========
constexpr uint32_t EXCHANGE_DATA_MAGIC = 0x48434C52; // "HCLR"
constexpr uint16_t ROCE_CAPABILITY_VERSION = 1;
constexpr uint16_t ROCE_MIN_SUPPORTED_VERSION = 1;
constexpr uint32_t INVALID_DPU_NOTIFY_ID = 0xFFFFFFFF;

// ========== 枚举类型定义 ==========

// 通信协议栈类型
enum class CommStackType : uint8_t {
    COMM_STACK_HOST_CPU_ROCE = 0,     // Host CPU RoCE (ibverbs)
    COMM_STACK_TRANSPORT_IBVERBS = 1, // NPU RoCE (HCCP)
    COMM_STACK_UNKNOWN = 255
};

// 同步模式
enum class SyncMode : uint8_t {
    SYNC_MODE_WRITE_IMM = 0,    // Write With Immediate (原生模式)
    SYNC_MODE_WRITE_NOTIFY = 1, // Write + Notify (混合模式)
    SYNC_MODE_UNKNOWN = 255
};

// ========== 能力协商结构 ==========
// 注意：使用 packed 属性确保跨模块二进制兼容
struct RoCECapability {
    uint32_t magic;       // 魔数：0x48434C52 ("HCLR")
    uint16_t version;     // 版本号
    uint16_t totalLength; // 总长度（包括头部和变长数据）

    uint8_t nodeType;        // 节点类型（参考 HcclDeviceType）
    CommStackType commStack; // 通信协议栈类型
    SyncMode syncMode;       // 支持的同步模式
    uint8_t padding;

    NICDeployment nicDeploy; // NIC 部署位置

    uint8_t reserved[4]; // 预留字段，用于未来扩展

    // 序列化（直接内存拷贝）
    void Serialize(uint8_t* buffer, size_t& len) const
    {
        len = sizeof(RoCECapability);
        (void)memcpy_s(buffer, len, this, len);
    }

    // 反序列化（直接内存拷贝）
    bool Deserialize(const uint8_t* buffer, size_t len)
    {
        if (len < sizeof(RoCECapability)) {
            return false;
        }
        if (memcpy_s(this, sizeof(RoCECapability), buffer, sizeof(RoCECapability)) < 0) {
            return false;
        }
        return true;
    }

    // 快速检查魔数
    static bool CheckMagic(const uint8_t* buffer, size_t len)
    {
        if (len < sizeof(uint32_t)) {
            return false;
        }
        uint32_t magic;
        if (memcpy_s(&magic, sizeof(magic), buffer, sizeof(magic)) < 0) {
            return false;
        }
        return magic == EXCHANGE_DATA_MAGIC;
    }

    // 校验字段有效性
    bool Validate() const
    {
        if (magic != EXCHANGE_DATA_MAGIC) {
            return false;
        }
        if (version < ROCE_MIN_SUPPORTED_VERSION) {
            return false;
        }
        // 校验枚举值范围
        if (nicDeploy >= NICDeployment::NIC_DEPLOYMENT_RESERVED) {
            return false;
        }
        if (static_cast<uint8_t>(commStack) > 1) {
            return false;
        }
        if (static_cast<uint8_t>(syncMode) > 1) {
            return false;
        }
        return true;
    }

    // 初始化默认值
    void InitDefaults()
    {
        magic = EXCHANGE_DATA_MAGIC;
        version = ROCE_CAPABILITY_VERSION;
        totalLength = sizeof(RoCECapability);
        nodeType = 0;
        nicDeploy = NICDeployment::NIC_DEPLOYMENT_HOST;
        commStack = CommStackType::COMM_STACK_HOST_CPU_ROCE;
        syncMode = SyncMode::SYNC_MODE_WRITE_IMM;
        (void)memset_s(reserved, sizeof(reserved), 0, sizeof(reserved));
    }
} __attribute__((packed));

struct ExchangeDataFmt {
    uint32_t qpn;
    uint32_t psn;
    uint8_t gid[HCCP_GID_RAW_LEN];
    uint8_t gidIdx;
    uint8_t padding[3];
};

} // namespace hcomm

#endif // EXCHANGE_DATA_FORMAT_H
