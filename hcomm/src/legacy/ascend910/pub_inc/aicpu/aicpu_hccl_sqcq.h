/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __AICPU_HCCL_SQCQ_H__
#define __AICPU_HCCL_SQCQ_H__

#include <memory>
#include <vector>
#include "ascend_hal.h"
#include "log.h"
#include "rt_external.h"
#include "acl/acl_rt.h"
#include "driver/ascend_hal_define.h"

namespace dfx {
enum class CqeStatus : int64_t {
    kDefault = 0,
    kCqeException,
    kCqeTimeOut,
    kCqeInnerError,
    kCqeUnknown,
};

const uint64_t kCreditTimeInvalid = 255U;
const uint64_t kCreditTimeDefault = 240U; // 240对应到rts的credit字段就代表960s, 硬件同步时间
const uint64_t kTimeOutTimeInvalid = 0U;
const uint64_t kKfcTimeOut = 960U;       // 960s
const uint64_t kSqFullWaitTimeOut = 60U; // 60s, rtsq full等待时间要大于超时代答时间
const uint64_t kPrintSqInterval = 30U;   // 算子执行阶段打印sqe状态的间隔，单位s

struct DfxTimeOutConfig {
    uint64_t sqeTimeOutTimeOut; // 软件同步
    uint64_t sqeCreditTimeOut;  // 硬件同步
    bool useCredit;             // 是否使用硬件同步
    uint64_t sqeWaitTimeOut;    // kfc自己的超时
    uint64_t sqFullWaitTimeOut; // sq满的时候等待的超时
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "sqeTimeOutTimeOut: " << sqeTimeOutTimeOut << " s";
        ss << " sqeCreditTimeOut: " << sqeCreditTimeOut << " s";
        ss << " useCredit: " << useCredit;
        ss << " sqeWaitTimeOut: " << sqeWaitTimeOut << " s";
        ss << " sqFullWaitTimeOut: " << sqFullWaitTimeOut << " s";
        return ss.str();
    }
};

struct CqeQueryInput {
    uint32_t devId;
    uint16_t streamId;
    uint32_t sqId;
    uint32_t cqId;
    uint32_t type;
    uint8_t* cqeAddr;
    std::string ToString() const
    {
        std::stringstream ss;
        ss << "devId: " << devId;
        ss << " streamId: " << streamId;
        ss << " sqId: " << sqId;
        ss << " cqId: " << cqId;
        ss << " type: " << type;
        return ss.str();
    }
};
} // namespace dfx
namespace ts {
enum ts_app_abort_status {
    APP_ABORT_TERMINATE_FAIL = 0x0U,
    APP_ABORT_INIT,
    APP_ABORT_KILL_FINISH,
    APP_ABORT_TERMINATE_FINISH,
    APP_ABORT_STATUS_INVALID,
};
}
using rtError_t = int32_t;
constexpr int32_t RT_ERROR_FEATURE_NOT_SUPPORT = 0x0711000d;

constexpr uint32_t UINT32_BIT_NUM = 32U;
constexpr uint32_t MASK_32_BIT = 0xFFFFFFFFU;
constexpr uint32_t MASK_17_BIT = 0x0001FFFFU;

constexpr uint16_t RT_TASK_TYPE_FLIP = 98;
constexpr uint8_t RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT = dfx::kCreditTimeInvalid;
constexpr uint8_t RT_STARS_DEFAULT_KERNEL_CREDIT = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT - 1U;
constexpr uint8_t RT_STARS_EXIST_ERROR = 0x3FU;
constexpr uint8_t RT_STARS_EXIST_WARNING = 0x40U;
constexpr uint16_t RT_SYNC_TASK_FLAG = 0x8000U;
constexpr u64 RT_GET_HEAD_CYCLE_NUM = 100000000U;
constexpr uint8_t RT_STARS_AICPU_MODEL_KERNEL_CREDIT = 18U;
constexpr uint8_t RT_STARS_AICPU_DEFAULT_TIMEOUT = 28U;
constexpr uint8_t RT_STARS_TASK_KERNEL_CREDIT_SCALE_UINT8 = 4U;

constexpr u64 RT_CHIP_ADDR_OFFSET = 0x80000000000ULL;
constexpr u64 RT_DIE_ADDR_OFFSET = 0x10000000000ULL;

constexpr u64 RT_STARS_PCIE_BASE_ADDR = 0x400004008000ULL;
constexpr u64 RT_PCIE_LOCAL_DEV_OFFSET = 36ULL;
constexpr u64 RT_PCIE_REMOTE_DEV_OFFSET = 32ULL;

constexpr u64 STARS_NOTIFY_BASE_ADDR = 0x100000ULL;
constexpr u64 STARS_NOTIFY_OFFSET = 0x4ULL;
constexpr uint32_t STARS_NOTIFY_NUM_OF_SINGLE_TABLE = 512U;
constexpr u64 STARS_NOTIFY_TABLE_OFFSET = 0x10000ULL;

constexpr u64 RT_STARS_BASE_ADDR = 0x06a0000000ULL;
constexpr u64 STARS_EVENT_BASE_ADDR = 0x200000ULL;
constexpr u64 STARS_EVENT_OFFSET = 0x4ULL;
constexpr uint32_t STARS_EVENT_NUM_OF_SINGLE_TABLE = 4096U;
constexpr u64 STARS_EVENT_TABLE_OFFSET = 0x10000ULL;

constexpr u32 RT_SDMA_COMPERR = 0x9; // A3 sdma error类型为0x9时，表示写拷贝发生超时代答，或者数据搬移时地址译码错误
constexpr u32 RT_SDMA_COMPDATAERR = 0xa; // A3 sdma error类型为0xa时，表示读拷贝发生超时代答，或者读HBM返回ERROR
constexpr u32 RT_SDMA_DATAERR = 0x8;                 // A3 sdma error类型为0x8时，表示读HBM返回ERROR
constexpr uint16_t TS_ERROR_RETRY_CONSTRAINT = 1000; // 重执行失败，约束是算子不一致或者inplace算子不支持
constexpr uint16_t TS_ERROR_AICPU_SDMA = 1001; // AICPU算子失败，sqe类型为SDMA是，上报给aicpu的框架错误码

constexpr u32 AC_SQE_REV_MAX_CNT = 32U; // 910B平台下调用halCqReportRecv接口最大能够接收的个数
constexpr uint32_t MAX_REPORT_CNT = 256U;

enum SqeType : uint8_t {
    SQE_TYPE_DEFAULT = 0,
    NOTIFY_SQE,
    WRITE_VALUE_SQE,
    EVENT_SQE,
    MEMCPY_ASYNC_SQE,
    CCORE_WAIT_START_SQE,
    CCORE_WRITE_VALUE_SQE,
    NOTIFY_SQE_V2,
    WRITE_VALUE_SQE_V2,
    EVENT_SQE_V2,
    MEMCPY_ASYNC_SQE_V2,
    RDMA_DB_SEND_SQE,
    FLIP_PLACEHOLDER_SQE,
    MEMCPY_ASYNC_SQE_V3,
    CACHE_MEMCPY_PLACEHOLDER_SQE,
    CACHE_NOTIFY_PLACEHOLDER_SQE,
    CACHE_WRITE_VALUE_PLACEHOLDER_SQE,
    CACHE_MEMCPY_RECORD_PLACEHOLDER_SQE
};

constexpr aclDataType DT_MAP_TABLE[HCCL_DATA_TYPE_RESERVED + 1] = {
    ACL_INT8,         /* HCCL_DATA_TYPE_INT8 = 0 */
    ACL_INT16,        /* HCCL_DATA_TYPE_INT16 = 1 */
    ACL_INT32,        /* HCCL_DATA_TYPE_INT32 = 2 */
    ACL_FLOAT16,      /* HCCL_DATA_TYPE_FP16 = 3 */
    ACL_FLOAT,        /* HCCL_DATA_TYPE_FP32 = 4 */
    ACL_DT_UNDEFINED, /* HCCL_DATA_TYPE_INT64 = 5 */
    ACL_DT_UNDEFINED, /* HCCL_DATA_TYPE_UINT64 = 6 */
    ACL_UINT8,        /* HCCL_DATA_TYPE_UINT8 = 7 */
    ACL_UINT16,       /* HCCL_DATA_TYPE_UINT16 = 8 */
    ACL_UINT32,       /* HCCL_DATA_TYPE_UINT32 = 9 */
    ACL_DT_UNDEFINED, /* HCCL_DATA_TYPE_FP64 = 10 */
    ACL_BF16,         /* HCCL_DATA_TYPE_BFP16 = 11 */
    ACL_DT_UNDEFINED  /* HCCL_DATA_TYPE_RESERVED */
};

constexpr aclrtReduceKind ACL_RT_MEMCPY_INVALID = static_cast<aclrtReduceKind>(-1);

constexpr aclrtReduceKind RK_MAP_TABLE[HCCL_REDUCE_RESERVED + 1] = {
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_SUM, /* HCCL_REDUCE_SUM = 0 */
    ACL_RT_MEMCPY_INVALID,            /* HCCL_REDUCE_PROD = 1 */
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MAX, /* HCCL_REDUCE_MAX = 2 */
    ACL_RT_MEMCPY_SDMA_AUTOMATIC_MIN, /* HCCL_REDUCE_MIN = 3 */
    ACL_RT_MEMCPY_INVALID,            /* HCCL_REDUCE_RESERVED = 4 */
};

enum aicpuNotifySqeType { NOTIFY_RECORD = 0, NOTIFY_WAIT = 1 };

enum rtStarsWriteValueSubType {
    RT_STARS_WRITE_VALUE_SUB_TYPE_DEFAULT = 0,
    RT_STARS_WRITE_VALUE_SUB_TYPE_EVENT_RESET = 1,
    RT_STARS_WRITE_VALUE_SUB_TYPE_RDMA_DB_SEND = 2,
    RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE = 3,
    RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_PCIE = 4,
    RT_STARS_WRITE_VALUE_SUB_TYPE_MAX,
};

enum rtStarsSqeType {
    RT_STARS_SQE_TYPE_FFTS = 0,          // FFTS
    RT_STARS_SQE_TYPE_AICPU = 1,         // AICPU
    RT_STARS_SQE_TYPE_PLACE_HOLDER = 3,  // PLACE_HOLDER
    RT_STARS_SQE_TYPE_EVENT_RECORD = 4,  // EVENT_RECORD
    RT_STARS_SQE_TYPE_EVENT_WAIT = 5,    // EVENT_WAIT
    RT_STARS_SQE_TYPE_NOTIFY_RECORD = 6, // NOTIFY_RECORD
    RT_STARS_SQE_TYPE_NOTIFY_WAIT = 7,   // NOTIFY_WAIT
    RT_STARS_SQE_TYPE_WRITE_VALUE = 8,   // for EVENT_RESET task
    RT_STARS_SQE_TYPE_SDMA = 11,         // SDMA
    RT_STARS_SQE_TYPE_VPC = 12,          // VPC
    RT_STARS_SQE_TYPE_JPEGE = 13,        // JPEGE
    RT_STARS_SQE_TYPE_JPEGD = 14,        // JPEGD
    RT_STARS_SQE_TYPE_DSA = 15,
    RT_STARS_SQE_TYPE_ROCCE = 16,    // RoCCE
    RT_STARS_SQE_TYPE_PCIE_DMA = 17, // PCIE_DMA
    RT_STARS_SQE_TYPE_RESV = 18,     // reserve
    RT_STARS_SQE_TYPE_CDQM = 19,     // CDQM
    RT_STARS_SQE_TYPE_COND = 20,     // condition
    RT_STARS_SQE_TYPE_END = 21,
    RT_STARS_SQE_TYPE_INVALID = 63,   // STARS_SQE_TYPE_INVALID
    RT_STARS_SQE_TYPE_VIR_TYPE = 0xFF // DVPP virtual SQE TYPE
};

/* stars send interrupt direction */
enum rtStarsSqeIntDirType {
    RT_STARS_SQE_INT_DIR_NO = 0,         // send no interrupt
    RT_STARS_SQE_INT_DIR_TO_TSCPU = 1,   // to tscpu
    RT_STARS_SQE_INT_DIR_TO_CTRLCPU = 2, // to ctrlcpu
    RT_STARS_SQE_INT_DIR_TO_HOST = 3,    // to host
    RT_STARS_SQE_INT_DIR_END = 4
};

enum rtStarsMemcpyAsyncOperationKind {
    RT_STARS_MEMCPY_ASYNC_OP_KIND_CPY = 0x00,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_ADD = 0x01,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_MAX = 0x02,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_MIN = 0x03,
    RT_STARS_MEMCPY_ASYNC_OP_KIND_EQUAL = 0x04
};

enum rtStarsMemcpyAsyncDataType {
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT8 = 0x00,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT16 = 0x10,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_INT32 = 0x20,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT8 = 0x30,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT16 = 0x40,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_UINT32 = 0x50,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP16 = 0x60,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_FP32 = 0x70,
    RT_STARS_MEMCPY_ASYNC_DATA_TYPE_BFP16 = 0x80,
    RT_STARS_MEMCPY_ASYNC_OP_RESERVED = 0xf0
};

enum rtStarsWriteValueSizeType {
    RT_STARS_WRITE_VALUE_SIZE_TYPE_8BIT = 0,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_16BIT = 1,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT = 2,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_64BIT = 3,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_128BIT = 4,
    RT_STARS_WRITE_VALUE_SIZE_TYPE_256BIT = 5
};

enum rtStarsCondIsaRegister_t {
    RT_STARS_COND_ISA_REGISTER_R0 = 0, // R0 is always zero, can't be destination register
    RT_STARS_COND_ISA_REGISTER_R1 = 1,
    RT_STARS_COND_ISA_REGISTER_R2 = 2,
    RT_STARS_COND_ISA_REGISTER_R3 = 3,
    RT_STARS_COND_ISA_REGISTER_R4 = 4,
    RT_STARS_COND_ISA_REGISTER_R5 = 5
};

// enum for isa op LOAD func3
enum rtStarsCondIsaLoadFunc3_t { RT_STARS_COND_ISA_LOAD_FUNC3_LDR = 0B011 };

// enum for isa op LWI func3
enum rtStarsCondIsaLwiFunc3_t { RT_STARS_COND_ISA_LWI_FUNC3_LHWI = 0B000, RT_STARS_COND_ISA_LWI_FUNC3_LLWI = 0B001 };

// enum for isa op Branch func3
enum rtStarsCondIsaBranchFunc3_t {
    RT_STARS_COND_ISA_BRANCH_FUNC3_BEQ = 0B000,
    RT_STARS_COND_ISA_BRANCH_FUNC3_BNE = 0B001,
    RT_STARS_COND_ISA_BRANCH_FUNC3_BLT = 0B100,
    RT_STARS_COND_ISA_BRANCH_FUNC3_BGE = 0B101,
    RT_STARS_COND_ISA_BRANCH_FUNC3_BLTU = 0B110,
    RT_STARS_COND_ISA_BRANCH_FUNC3_BGEU = 0B111
};

enum rtStarsCondIsaOpCode_t {
    RT_STARS_COND_ISA_OP_CODE_OP_IMM = 0B0010011,                     // Integer Register-immd Instructions
    RT_STARS_COND_ISA_OP_CODE_NOP = RT_STARS_COND_ISA_OP_CODE_OP_IMM, // NOP is using OP_IMM ADDI R0,R0,0
    RT_STARS_COND_ISA_OP_CODE_OP = 0B0110011,                         // Integer Register-Register Operations
    RT_STARS_COND_ISA_OP_CODE_LWI = 0B1011011,                        // load immd
    RT_STARS_COND_ISA_OP_CODE_BRANCH = 0B1100011,                     // Conditional stream-jump
    RT_STARS_COND_ISA_OP_CODE_LOOP = 0B1111011,                       // LOOP
    RT_STARS_COND_ISA_OP_CODE_STREAM = 0B0101011,                     // STREAM
    RT_STARS_COND_ISA_OP_CODE_LOAD_IMM = 0B0000111,                   // LOAD immd
    RT_STARS_COND_ISA_OP_CODE_LOAD = 0B0000011,                       // Load
    RT_STARS_COND_ISA_OP_CODE_STORE = 0B0100111,                      // Store
    RT_STARS_COND_ISA_OP_CODE_FUNC_CALL = 0B1101011,                  // FUNC_CALL
    RT_STARS_COND_ISA_OP_CODE_SYSTEM = 0B1110011                      // CSR
};

// enum for isa op Op Imm func3
enum rtStarsCondIsaOpImmFunc3_t {
    RT_STARS_COND_ISA_OP_IMM_FUNC3_ADDI = 0B000,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_NOP = RT_STARS_COND_ISA_OP_IMM_FUNC3_ADDI, // NOP is using OP_IMM ADDI R0,R0,0
    RT_STARS_COND_ISA_OP_IMM_FUNC3_SLLI = 0B001,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_SLTI = 0B010,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_SLTIU = 0B011,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_XORI = 0B100,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_SRLI = 0B101,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_ORI = 0B110,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_ANDI = 0B111,
    RT_STARS_COND_ISA_OP_IMM_FUNC3_SRAI = 0B101 // diff with SRLI by func7
};

// enum for isa op store func3
enum rtStarsCondIsaStoreFunc3_t {
    RT_STARS_COND_ISA_STORE_FUNC3_SB = 0B000,
    RT_STARS_COND_ISA_STORE_FUNC3_SH = 0B001,
    RT_STARS_COND_ISA_STORE_FUNC3_SW = 0B010,
    RT_STARS_COND_ISA_STORE_FUNC3_SD = 0B011,
};

#define TOPOLOGY_HCCS 0
#define TOPOLOGY_PIX 1
#define TOPOLOGY_PIB 2
#define TOPOLOGY_PHB 3
#define TOPOLOGY_SYS 4

#pragma pack(push)
#pragma pack(1)

struct rtStarsWriteValueSqe_t {
    rtStarsSqeHeader_t header;

    uint32_t res3;

    uint32_t res4 : 16;
    uint32_t kernel_credit : 8;
    uint32_t res5 : 8;

    uint32_t write_addr_low;

    uint32_t write_addr_high : 17;
    uint32_t res6 : 3;
    uint32_t awsize : 3;
    uint32_t snoop : 1;
    uint32_t awcache : 4;
    uint32_t awprot : 3;
    uint32_t va : 1; // 1 /* 1: virtual address; 0: phy addr */

    uint32_t res7; // eventId for event reset task
    uint32_t sub_type;

    uint32_t write_value_part0;
    uint32_t write_value_part1;
    uint32_t rdmaWrLenth; // rdmaWr len
    uint32_t rdmaType;    // notify / payload
    uint32_t write_value_part4;
    uint32_t write_value_part5;
    uint32_t write_value_part6;
    uint32_t write_value_part7;
};

struct rtStarsEventSqe_t {
    rtStarsSqeHeader_t header;
    uint16_t eventId;
    uint16_t res2;

    uint16_t res3;
    uint8_t kernel_credit;
    uint8_t res4;
    uint32_t exe_result;
    uint32_t timeout;
    uint32_t res5[10];
};

struct rtStarsMemcpyAsyncSqe_t {
    rtStarsSqeHeader_t header;

    uint32_t res3;
    /* *******12 bytes********* */

    uint16_t res4;
    uint8_t kernel_credit;
    uint8_t ptr_mode : 1;
    uint8_t res5 : 7;
    /* *******16 bytes********* */

    uint32_t opcode : 8;
    uint32_t ie2 : 1;
    uint32_t sssv : 1;
    uint32_t dssv : 1;
    uint32_t sns : 1;
    uint32_t dns : 1;
    uint32_t qos : 4;
    uint32_t sro : 1;
    uint32_t dro : 1;
    uint32_t partid : 8;
    uint32_t mpam : 1;
    uint32_t res6 : 4;
    /* *******20 bytes********* */

    uint16_t src_streamid;
    uint16_t src_sub_streamid;
    uint16_t dst_streamid;
    uint16_t dst_sub_streamid;
    /* *******28 bytes********* */

    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    uint8_t linkType;
    uint8_t resvered[3];
    uint32_t reslast[3];
};

struct rtFlipTaskTag_t {
    uint16_t flipNumReport;
    uint8_t reserved[46];
};

// 用于alltoallv算子小数据量下展开时cache中的cache-memcpy placeholder (for zero-len memcpy SQE)
struct rtCacheMemcpyTaskTag_t {
    // 注意: 不需要维护length, 因为每次cache hit时, length会根据send/recv info动态计算得到

    // 第一次展开时保留src/dst addr, cache miss后处理时, 用于UpdateRefreshAddrInfoForAlltoallv
    // 后续cache hit时只需要保留hccl buffer内的src/dst addr
    // 如果是user memory内的src/dst addr, 则根据send/recv info动态计算得到, 不需要在placeholder中维护
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    // 始终需要维护
    // LocalCopy/PrepareIntraData: linkType使用默认参数LINK_ONCHIP
    // RemoteCopy: linkType使用srcRank-localRank之间的LINK对应的linkType
    uint8_t kernel_credit;
    uint8_t linkType;
    uint32_t qos;

    uint8_t reserved[26];
};

struct rtCacheNotifyTaskTag_t {
    uint8_t is_wait; // NotifyWait or NotifyRecord
    // Only for NotifyWait
    uint8_t kernel_credit;
    uint32_t timeout;
    // Shared by NotifyWait and NotifyRecord
    uint32_t notify_id;
    uint8_t reserved[38];
};

struct rtCacheWriteValueTaskTag_t {
    // For WriteValueRecord
    uint32_t write_addr_low;
    uint32_t write_addr_high;
    uint8_t reserved[40];
};

struct rtCacheMemcpyRecordTaskTag_t {
    // 20 bytes
    uint32_t length;
    uint32_t src_addr_low;
    uint32_t src_addr_high;
    uint32_t dst_addr_low;
    uint32_t dst_addr_high;

    // 4 bytes
    uint32_t opcode : 8;
    uint32_t partid : 8;
    uint32_t res : 16;

    // 2 bytes
    uint8_t kernel_credit;
    uint8_t linkType;

    // 4 bytes
    uint32_t qos;

    uint8_t reserved[18];
};

struct rtStarsPlaceHolderSqe_t {
    rtStarsSqeHeader_t header;

    uint32_t res1;
    uint16_t res2;
    uint8_t kernel_credit;
    uint8_t res3;

    /* The struct in the union must be 48 bytes */
    union {
        rtFlipTaskTag_t flip_task_info;
        rtCacheMemcpyTaskTag_t cache_memcpy_task_info;
        rtCacheNotifyTaskTag_t cache_notify_task_info;
        rtCacheWriteValueTaskTag_t cache_write_value_task_info;
        rtCacheMemcpyRecordTaskTag_t cache_memcpy_record_task_info;
        uint32_t resv[12];
    } u;
};

struct rtStarsCondOpLHWI_t {
    uint32_t opCode : 7;
    uint32_t rd : 3;
    uint32_t reserved0 : 2; // reserved
    uint32_t func3 : 3;
    uint32_t reserved1 : 2; // reserved
    uint32_t immd : 15;
};

// LWI LLWI: RT_STARS_COND_ISA_OP_CODE_LWI
struct rtStarsCondOpLLWI_t {
    uint32_t opCode : 7;
    uint32_t rd : 3;
    uint32_t reserved0 : 2; // reserved
    uint32_t func3 : 3;
    uint32_t immdHigh : 17;
    uint32_t immdLow : 32;
};

// Branch: RT_STARS_COND_ISA_OP_CODE_BRANCH
struct rtStarsCondOpBranch_t {
    uint32_t opCode : 7;
    uint32_t jumpInstrOffset : 4;
    uint32_t rsvd : 1;
    uint32_t func3 : 3;
    uint32_t rs1 : 3;
    uint32_t rsvd1 : 2;
    uint32_t rs2 : 3;
    uint32_t rsvd2 : 2;
    uint32_t rsvd3 : 7;
};

struct rtStarsCondOpLoad_t {
    uint32_t opCode : 7;
    uint32_t rd : 3;
    uint32_t reserved0 : 2;
    uint32_t func3 : 3;
    uint32_t rs1 : 3;
    uint32_t reserved1 : 2;
    uint32_t immd : 12;
};

// enum for isa op load imm func3
enum rtStarsCondIsaLoadImmFunc3_t {
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LB = 0B000,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LH = 0B001,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LW = 0B010,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LD = 0B011,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LBU = 0B100,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LHU = 0B101,
    RT_STARS_COND_ISA_LOAD_IMM_FUNC3_LWU = 0B110
};

struct rtStarsCondOpLoadImm_t {
    uint32_t opCode : 7;
    uint32_t rd : 3;
    uint32_t reserved : 2;
    uint32_t func3 : 3;
    uint32_t immdAddrHigh : 17;
    uint32_t immdAddrLow;
};

struct rtStarsCondOpImm_t {
    uint32_t opCode : 7;
    uint32_t rd : 3;
    uint32_t reserved0 : 2; // reserved
    uint32_t func3 : 3;
    uint32_t rs1 : 3;
    uint32_t reserved1 : 2; // reserved
    uint32_t immd : 12;
};

// store register data to ddr, RT_STARS_COND_ISA_OP_CODE_STORE
struct rtStarsCondOpStore_t {
    uint32_t opCode : 7;
    uint32_t immdLow : 5;
    uint32_t func3 : 3;
    uint32_t rs1 : 3;
    uint32_t reserved1 : 2;
    uint32_t rs2 : 3;
    uint32_t reserved2 : 2;
    uint32_t immdHigh : 7;
};

// nop is using op-imm ADDI
using rtStarsCondOpNop_t = rtStarsCondOpImm_t;
struct rtStarsCondOpClear_t {
    rtStarsCondOpLLWI_t llwi1;
    rtStarsCondOpLHWI_t lhwi1; // load wait address as the immediate to R1
    rtStarsCondOpStore_t sw;   // the last turn clear write_value
    rtStarsCondOpNop_t nop[3];
};
struct rtStarsCcoreWaitStartSqe_t {
    rtStarsSqeHeader_t sqeHeader;

    uint32_t reserved0;
    uint16_t reserved1;
    uint8_t kernel_credit;
    uint8_t reserved2 : 7;
    uint8_t csc : 1;

    rtStarsCondOpLoadImm_t ldrImm1; // load current turn as the immediate to R3
    rtStarsCondOpLoadImm_t ldrImm2; // load wait value, to R2
    rtStarsCondOpBranch_t beq;      // if waitvalue == 0, goto read R2
    union {
        rtStarsCondOpClear_t clear;
        rtStarsCondOpNop_t nop[7];
    };
};

struct rtStarsCcoreWriteValueSqe_t {
    rtStarsSqeHeader_t sqeHeader;

    uint32_t reserved0;
    uint16_t reserved1;
    uint8_t kernel_credit;
    uint8_t reserved2 : 7;
    uint8_t csc : 1;

    rtStarsCondOpLoadImm_t ldrImm;
    rtStarsCondOpLLWI_t llwi1;
    rtStarsCondOpLHWI_t lhwi1;
    rtStarsCondOpStore_t sw;
    rtStarsCondOpNop_t nop[6];
};
struct rtLogicCqReport_t {
    volatile uint16_t streamId;
    volatile uint16_t taskId;
    volatile uint32_t errorCode; // cqe acc_status/sq_sw_status
    volatile uint8_t errorType;  // bit0 ~ bit5 cqe stars_defined_err_code, bit 6 cqe warning bit
    volatile uint8_t sqeType;
    volatile uint16_t sqId;
    volatile uint16_t sqHead;
    volatile uint16_t matchFlag : 1;
    volatile uint16_t dropFlag : 1;
    volatile uint16_t errorBit : 1;
    volatile uint16_t accError : 1;
    volatile uint16_t reserved0 : 12;
    union {
        volatile uint64_t timeStamp;
        volatile uint16_t sqeIndex;
    } u1;
    /* Union description:
     * Internal: enque_timestamp temporarily used as dfx
     * External: reserved1
     */
    union {
        volatile uint64_t enqueTimeStamp;
        volatile uint64_t reserved1;
    } u2;
};

const std::vector<std::string> StarsCqeErrorDesc
    = {"task exception",          "task trap",          "task timeout", "sqe error",
       "resource conflict error", "sq sw status error", "warning"};
#pragma pack(pop)

inline void
AddOneWriteValueRecordSqe(uint16_t streamId, uint16_t taskId, u64 notifyWRAddr, rtStarsWriteValueSqe_t* const sqe)
{
    sqe->header.type = RT_STARS_SQE_TYPE_WRITE_VALUE;
    sqe->header.rtStreamId = streamId;
    sqe->header.taskId = taskId;
    sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;
    sqe->awsize = RT_STARS_WRITE_VALUE_SIZE_TYPE_32BIT;
    sqe->write_value_part0 = 1U;
    sqe->sub_type = RT_STARS_WRITE_VALUE_SUB_TYPE_NOTIFY_RECORD_IPC_NO_PCIE;
    sqe->write_addr_low = static_cast<uint32_t>(notifyWRAddr & MASK_32_BIT);
    sqe->write_addr_high = static_cast<uint32_t>((notifyWRAddr >> UINT32_BIT_NUM) & MASK_17_BIT);
    HCCL_DEBUG("[SQE] write value: writePtr=0x%lx, streamId=%u, taskId=%u.", notifyWRAddr, streamId, taskId);
}

inline void AddOneEventRecordSqe(uint16_t streamId, int32_t eventId, uint16_t taskId, rtStarsEventSqe_t* const ev_sqe)
{
    ev_sqe->header.type = RT_STARS_SQE_TYPE_EVENT_RECORD;
    ev_sqe->header.wrCqe = 1U; // 1: set wrCqe
    ev_sqe->kernel_credit = RT_STARS_DEFAULT_KERNEL_CREDIT;

    // eventRecordTaskInfo->waitCqflag  是否为同步task
    if (false) {
        streamId |= RT_SYNC_TASK_FLAG;
    }

    ev_sqe->header.rtStreamId = streamId;
    ev_sqe->eventId = static_cast<uint16_t>(eventId);
    ev_sqe->header.taskId = taskId;
    HCCL_DEBUG("[SQE] event record: eventId=%d, streamId=%u, taskId=%u.", eventId, streamId, taskId);
}

inline void AddOneEventWaitSqe(uint16_t streamId, int32_t eventId, uint16_t taskId, rtStarsEventSqe_t* const ev_sqe)
{
    ev_sqe->header.type = RT_STARS_SQE_TYPE_EVENT_WAIT;
    ev_sqe->kernel_credit = RT_STARS_NEVER_TIMEOUT_KERNEL_CREDIT;

    ev_sqe->header.rtStreamId = streamId;
    ev_sqe->eventId = eventId;
    ev_sqe->header.taskId = taskId;
    HCCL_DEBUG("[SQE] event wait: eventId=%d, streamId=%u, taskId=%u", eventId, streamId, taskId);
}

extern void AddOneWaitStartSqe(
    uint16_t streamId, uint16_t taskId, u64 waitAddr, u64 curTurnCntAddr, bool last,
    rtStarsCcoreWaitStartSqe_t* const sqe, uint8_t* sqeType);
extern void AddOneWriteValueStartSqe(
    uint16_t streamId, uint16_t taskId, u64 writeAddr, u64 valueAddr, rtStarsCcoreWriteValueSqe_t* const sqe,
    uint8_t* sqeType);

extern HcclResult QuerySqStatus(uint32_t devId, uint32_t sqId, uint32_t& sqHead, uint32_t& sqTail);
extern HcclResult QuerySqStatusByType(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t& outVal);
extern HcclResult ConfigSqStatusByType(uint32_t devId, uint32_t sqId, drvSqCqPropType_t type, uint32_t value);
extern HcclResult QuerySqBaseAddr(uint32_t devId, uint32_t sqId, u64& outVal);
using CqeStatus = dfx::CqeStatus;
using CqeQueryInput = dfx::CqeQueryInput;
extern CqeStatus CqReportRecv(const CqeQueryInput& cqeQueryInput, rtLogicCqReport_t& cqeException);
extern void PrintTaskException(const rtLogicCqReport_t& reportOfOne);
extern HcclResult StreamsKill(const uint32_t devId);
extern HcclResult DeviceQuery(const uint32_t devId, const uint32_t step, const uint32_t timeout);
namespace hccl_plf {
HcclResult SendTaskExceptionByMBox(
    const u32 localDeviceId, const u32 notifyId, const u32 tsId, const s32 userStreamId, const u32 cqeErrCode);
}
#endif // __AICPU_HCCL_SQCQ_H__
