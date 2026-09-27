/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMMON_DEFS_H
#define HCCL_COMMON_DEFS_H

#include <array>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#define RT_CCU_SQE_ARGS_LEN (13U)

namespace HcclSim {
constexpr uint32_t MOVE_TOW_BYTES = 16;
constexpr uint32_t MOVE_THREE_BYTES = 24;
constexpr uint32_t SOCKET_VNIC_IP_INFOS_INTERFACE = 55;
constexpr uint32_t GET_NOTIFY_BA = 14;
constexpr uint32_t MOVE_20_BITS = 20;
constexpr uint32_t MOVE_16_BITS = 16;

constexpr int DIE_NUM = 2;
constexpr uint32_t BYTE_NUM_4K = 4096;

constexpr uint32_t CCU_INSTRUCTION_NUM = 32 * 1024;
constexpr uint32_t CCU_INSTRUCTION_SIZE = 32;

constexpr uint16_t CCU_RESOURCE_GSA_MAX = 3072;
constexpr uint16_t CCU_RESOURCE_XN_MAX = 3072;
constexpr uint16_t CCU_RESOURCE_CKE_MAX = 1024;
constexpr uint16_t CCU_RESOURCE_LOOP_ENGINE_MAX = 200;
constexpr uint16_t CCU_RESOURCE_MEM_SLICE_4K = 4096;

constexpr uint16_t CCU_RESOURCE_MS_NUM = 1536;
constexpr uint32_t CCU_RESOURCE_MS_SIZE = CCU_RESOURCE_MS_NUM * BYTE_NUM_4K;

constexpr uint32_t URMA_EID_LEN = 16;
using SimEid = std::array<uint8_t, URMA_EID_LEN>;

constexpr uint8_t DEVICE_WAIT = 0;
constexpr uint8_t DEVICE_RUN = 1;

constexpr uint32_t AICPU_JETTY_NUM_MAX = 40960; // 暂定扩大至40960个

typedef struct {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint32_t instStartId;
    uint32_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
} CcuTaskParam;

enum ProtocolType {
    SIM_PROTOCOL_RESERVED = -1,
    SIM_PROTOCOL_HCCS = 0,
    SIM_PROTOCOL_ROCE = 1,
    SIM_PROTOCOL_PCIE = 2,
    SIM_PROTOCOL_SIO = 3,
    SIM_PROTOCOL_UB_CTP = 4,
    SIM_PROTOCOL_UBC_TP = 5,
    SIM_PROTOCOL_UB_MEM = 6
};

enum HcclVmResult {
    // 0 ~ 4095: HCCL业务错误码
    HCCL_SIM_SUCCESS = 0,              /**< success */
    HCCL_SIM_E_PARA = 1,               /**< parameter error */
    HCCL_SIM_E_PTR = 2,                /**< empty pointer */
    HCCL_SIM_E_MEMORY = 3,             /**< memory error */
    HCCL_SIM_E_INTERNAL = 4,           /**< internal error */
    HCCL_SIM_E_NOT_SUPPORT = 5,        /**< not support feature */
    HCCL_SIM_E_NOT_FOUND = 6,          /**< not found specific resource */
    HCCL_SIM_E_UNAVAIL = 7,            /**< resource unavailable */
    HCCL_SIM_E_SYSCALL = 8,            /**< call system interface error */
    HCCL_SIM_E_TIMEOUT = 9,            /**< timeout */
    HCCL_SIM_E_OPEN_FILE_FAILURE = 10, /**< open file fail */
    HCCL_SIM_E_TCP_CONNECT = 11,       /**< tcp connect fail */
    HCCL_SIM_E_ROCE_CONNECT = 12,      /**< roce connect fail */
    HCCL_SIM_E_TCP_TRANSFER = 13,      /**< tcp transfer fail */
    HCCL_SIM_E_ROCE_TRANSFER = 14,     /**< roce transfer fail */
    HCCL_SIM_E_RUNTIME = 15,           /**< call runtime api fail */
    HCCL_SIM_E_DRV = 16,               /**< call driver api fail */
    HCCL_SIM_E_PROFILING = 17,         /**< call profiling api fail */
    HCCL_SIM_E_CCE = 18,               /**< call cce api fail */
    HCCL_SIM_E_NETWORK = 19,           /**< call network api fail */
    HCCL_SIM_E_AGAIN = 20,             /**< try again */
    HCCL_SIM_E_REMOTE = 21,            /**< error cqe */
    HCCL_SIM_E_SUSPENDING = 22,        /**< error communicator suspending */
    HCCL_SIM_E_OPRETRY_FAIL = 23,      /**< retry constraint */
    HCCL_SIM_E_IN_STATUS = 1041,       /**< The error information is in the status. */
    HCCL_SIM_E_RESERVED,               /**< reserved */
    HCCL_SIM_E_SKIP,                   /**< skip */

    // 4096 ~ 8191: HCCL_SIM错误码
    // 4096 ~ 5119: HCCL_SIM Host错误码
    HCCL_SIM_HOST_ERROR_CMD = 4096,
    HCCL_SIM_HOST_SUCCESS_CMD = 4097,
    HCCL_SIM_HOST_ERROR_PARSE_CMD = 4098,
    HCCL_SIM_HOST_ERROR_RESOURCE = 4099,
    HCCL_SIM_HOST_EXIT = 4100,
    HCCL_SIM_HOST_ERROR_EXE = 4101,
    HCCL_SIM_HOST_RESERVED = 5119,

    // 5120 ~ 6143: HCCL_SIM Porxy错误码
    HCCL_SIM_PROXY_ERROR_CMD = 5120,
    HCCL_SIM_PROXY_RESERVED = 6143,

    // 6144 ~ 7167: HCCL_SIM SHM错误码
    HCCL_SIM_SHM_ERROR_CMD = 6144,
    HCCL_SIM_SHM_TASK_COLLECTION_ERROR,
    HCCL_SIM_SHM_NOT_INIT,
    HCCL_SIM_SHM_BUFFER_OUT_OF_MEMORY,
    HCCL_SIM_SHM_PROXY_OUT_OF_RANGE,
    HCCL_SIM_SHM_OBJ_NOT_FOUND,
    HCCL_SIM_SHM_FAIL,

    // 8192 ~ 9215: HCCL_SIM VIRTUAL_RUNTIME错误码
    HCCL_SIM_VRT_ERROR_CMD = 8192,
    HCCL_SIM_VRT_HOLD_CMD = 8193,
    HCCL_SIM_VRT_CONTINUE_CMD = 8194,
    HCCL_SIM_VRT_RESERVED = 9215
};

enum HcclVmMode { CHECKER = 0, RUNNER = 1 };

enum CcuVersion { CCU_V1 = 0, CCU_V2 = 1 };

} // namespace HcclSim

// AIV communication information buffer layout shared by the proxy, runner and checker.
namespace AivCommInfoLayout {
constexpr uint64_t SIZE_BYTES = 65ULL * 1024ULL * 1024ULL;
constexpr uint64_t GM_IN_TABLE_OFFSET = 0;
constexpr uint64_t GM_OUT_TABLE_OFFSET = 16ULL * 1024ULL;
constexpr uint64_t TAG_OFFSET = 512ULL * 1024ULL;
constexpr uint64_t FLAG1_OFFSET = 1ULL * 1024ULL * 1024ULL;
constexpr uint64_t FLAG2_OFFSET = 5ULL * 1024ULL * 1024ULL;
constexpr uint64_t BASE_FLAG_OFFSET = 9ULL * 1024ULL * 1024ULL;
constexpr uint64_t EMPTY_CLEAR_OFFSET = 10ULL * 1024ULL * 1024ULL;
constexpr uint64_t PING_OFFSET = 18ULL * 1024ULL * 1024ULL;
constexpr uint64_t PONG_OFFSET = 34ULL * 1024ULL * 1024ULL;
constexpr uint64_t SYNC_CELL_BYTES = 128ULL;
constexpr uint32_t FIXED_TAG = 1;
} // namespace AivCommInfoLayout

using PhyDeviceId = uint32_t;
using ServerMeta = std::vector<PhyDeviceId>;
using SuperPodMeta = std::vector<ServerMeta>;
using TopoMeta = std::vector<SuperPodMeta>;

static inline uint32_t ShmGetPhyDeviceTotalCount(const TopoMeta& topo_meta)
{
    uint32_t total_count = 0;

    // 第一层：遍历所有 SuperPod（超级节点）
    for (const auto& super_pod : topo_meta) {
        // 第二层：遍历每个 SuperPod 中的所有 Server（服务器）
        for (const auto& server : super_pod) {
            // 第三层：累加每个 Server 中的 PhyDeviceId 数量
            total_count += static_cast<uint32_t>(server.size());
        }
    }
    return total_count;
}

enum class HccLTaskMetaType : char {
    NOTIFY_WAIT,
    NOTIFY_RECORD,
    REDUCE,
    MEM_CPY,
    CCU_GRAPH,
    AIV_GRAPH,
    EVENT_WAIT,
    EVENT_RECORD
};

typedef enum {
    SUCCESS = 0,
    FAILURE = 1,
    PENDING = 2,
} HccLTaskStatus;

union HcclTaskCid {
    struct {
        uint64_t commId : 16;
        uint64_t rankId : 16;
        uint64_t index : 32;
    } field;
    uint64_t value;
};

#pragma pack(push, 1)

typedef struct {
    uint64_t count;
    uint16_t dataType;
    uint32_t reduceOpType;
} OpInfoV1;

typedef struct {
    uint64_t sendCount;
    uint16_t sendDataType;
    uint64_t recvCount;
    uint16_t recvDataType;
    uint16_t extInfo;
} OpInfoV2;

typedef struct {
    uint16_t opType;
    uint16_t dataType;
    uint16_t reduceType;
    union {
        OpInfoV1 opV1;
        OpInfoV2 opV2;
    };
} OpDetails;

typedef struct {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t len;
    uint8_t protocol;
} TransMemTask;

typedef struct {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commId;
    uint64_t streamId;
    uint64_t inputAddr;
    uint64_t inputSize;
    uint64_t outputAddr;
    uint64_t outputSize;
    uint64_t cclAddr;
    uint64_t cclSize;
} OpStartTask;

typedef struct {
    uint32_t rankId;
    uint32_t rankSize;
    uint64_t commId;
    uint64_t streamId;
} OpSyncTask;

typedef struct {
    uint32_t srcRankId;
    uint64_t srcOffset;
    uint32_t dstRankId;
    uint64_t dstOffset;
    uint64_t dataCount;
    uint8_t dataType; // HcclDataType from hccl_types.h
    uint8_t reduceOp; // HcclReduceOp from hccl_types.h
    uint8_t protocol;
    uint8_t resv;
} ReduceTask;

typedef struct {
    uint32_t srcRankId;
    uint64_t notifyId;
    uint32_t dstRankId;
    uint8_t notifyCount;
    uint8_t protocol;
} NotifyTask;

typedef struct {
    uint8_t dieId;
    uint8_t missionId;
    uint16_t timeout;
    uint16_t instStartId;
    uint16_t instCnt;
    uint32_t key;
    uint32_t argSize;
    uint64_t args[RT_CCU_SQE_ARGS_LEN];
} CcuTask;

typedef struct {
    uint64_t launchIdx;
} AivTask;

typedef struct HcclTaskMetaData {
    HccLTaskMetaType taskType;
    uint16_t commId;
    uint32_t rankId;
    uint64_t streamId;
    uint32_t jettyId;
    uint8_t rmEid[16];
    union {
        TransMemTask transMem;
        ReduceTask reduce;
        NotifyTask notify;
        CcuTask ccu;
        AivTask aiv;
        OpStartTask opStartTask;
        OpSyncTask opSyncTask;
    } taskData;
    HcclTaskMetaData()
    {
        jettyId = UINT32_MAX;
        memset(rmEid, 0, sizeof(rmEid));
    }
} HcclTaskMetaData;

typedef struct {
    uint64_t taskCid;
    uint16_t dispatchId;
} HcclTaskReq;

typedef struct {
    uint64_t taskCid;
    uint16_t status;
} HcclTaskRsp;

#pragma pack(pop)

enum BufferType { INPUT = 0, OUTPUT, CCL, RESERVED };

class DataSlice {
public:
    DataSlice() : type_(BufferType::INPUT), offset_(0), size_(0) {}
    DataSlice(BufferType type, uint64_t offset, uint64_t size) : type_(type), offset_(offset), size_(size) {}
    std::string Describe() const
    {
        return "BufferType: " + std::to_string(static_cast<int>(type_)) + ", Offset: " + std::to_string(offset_)
               + ", Size: " + std::to_string(size_);
    }
    inline BufferType GetType() const { return type_; }
    inline uint64_t GetOffset() const { return offset_; }
    inline uint64_t GetSize() const { return size_; }
    void SetBufferType(const BufferType type) { type_ = type; }
    void SetOffset(uint64_t offset) { offset_ = offset; }
    void SetSize(uint64_t size) { size_ = size; }

private:
    BufferType type_;
    uint64_t offset_;
    uint64_t size_;
};

/**
 * @brief 获取当前编译目标架构字符串
 *
 * 返回值为编译期常量, 用于拼接 lib/<arch>/ 目录下的动态库路径.
 * 与运行时 uname() 不同, 编译期宏不受 QEMU 用户态仿真影响:
 *   - Host 侧 (hccl-vm):   本地 gcc 编译,  返回 "x86_64"
 *   - Device 侧 (device):  交叉编译 aarch64, 返回 "aarch64"
 *
 * @return "x86_64" 或 "aarch64"
 */
inline std::string GetArchStr()
{
#if defined(__aarch64__)
    return "aarch64";
#else
    return "x86_64";
#endif
}

#endif
