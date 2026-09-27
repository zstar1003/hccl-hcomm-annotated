/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef SIM_MODEL_DEFS_H
#define SIM_MODEL_DEFS_H

#include <cstdint>
#include <functional>

#include "sim_common_defs.h"

using namespace HcclSim;

namespace sim {
typedef struct {
    uint64_t id; // PK
    uint64_t pod_id;
    uint32_t server_id;
    char version[16];
} Server;

typedef struct {
    uint64_t id;  // PK
    uint8_t mode; // 0=normal,1=check-only
} RunModeConfig;

typedef struct {
    uint64_t id; // PK
    uint64_t server_id;
    char ip_addr[40]; // ip地址
    uint8_t arch;
} Host;

typedef struct {
    uint64_t id;      // PK
    uint64_t host_id; // FK
    uint64_t pid;
    uint64_t thread_id;
    uint64_t timeout_config_ms;
    uint64_t current_ctx_id; // FK
} Runner;

typedef struct {
    uint64_t id;        // PK
    uint32_t server_id; // FK
    uint32_t user_id{0xFFFF};
    uint32_t logic_id{0xFFFF};
    uint32_t physical_id;
    uint32_t super_device_id;
    uint32_t overflow_mode;
    char soc_version[16];
    uint32_t status; // 0 可用， 1不可用
} Device;

enum TsDevType { TS_DEV_TYPE_SCALAR = 0, TS_DEV_TYPE_CPU = 1, TS_DEV_TYPE_CCU = 2 };

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint8_t type;       // 0:Scalar, 1:CCU, 2:CPU
} TaskSchedulerDevice;

enum ComputeDieType { COMPUTE_TYPE_CUBE = 0, COMPUTE_TYPE_VECTOR = 1, COMPUTE_TYPE_HYBRID = 2 };

typedef struct {
    uint64_t id;    // PK
    uint64_t ts_id; // FK
    uint8_t type;   // 0:Cube 1:Vector 2: HybridCompute
} ComputeDie;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint8_t overflow_status;
    uint8_t synchronize_strategy;
    uint8_t synchronize_timeout;
    uint8_t capability_mask;
    uint8_t run_by_host;
    uint8_t ts_core;
    uint8_t online_status;
} DeviceStatus;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    uint64_t die_id;    // FK
    char name[128];     // 对应topo文件中的port_id: 0/0 ~ 0/8
    uint8_t status;     // 0: 未使用 1: 已使用
} Port;

typedef struct {
    uint64_t id;            // PK
    uint64_t device_id;     // FK
    uint64_t resource_addr; // ccu资源地址，区分不同die的资源：die0: 0x123123123; die1: 0x456456456
    uint8_t die_id;
    uint8_t status;
} Ccu;

typedef struct {
    uint64_t id;                                                    // PK
    uint64_t ccu_id;                                                // FK
    uint8_t instr_space[CCU_INSTRUCTION_NUM][CCU_INSTRUCTION_SIZE]; // 指令空间1M: 32K个指令
    uint16_t instr_cnt{0};
    uint16_t state{0};
    // uint64_t xn[CCU_RESOURCE_XN_MAX];  // xn寄存器
    // uint64_t gsa[CCU_RESOURCE_GSA_MAX]; // gsa寄存器
    // uint16_t cke[CCU_RESOURCE_CKE_MAX]; // cke寄存器
    // char ms[CCU_RESOURCE_MS_SIZE]; // ms寄存器
} CcuResource; // todo: 后续可能改为关系表，数据保存在共享内存

typedef struct {
    uint64_t id;         // PK
    uint64_t src_dev_id; // FK
    uint64_t dst_dev_id; // FK
    uint8_t link_type;
    uint8_t access_by_remote;
} DeviceConnection;

typedef struct {
    uint64_t id; // PK
    uint32_t device_id;
    uint16_t func_id;
    uint8_t die_id;
    uint16_t type; // 地址类型: 0 - EID, 1 - IPV4, 2 - IPV6
    uint8_t eid[16];
    char ip_addr[64];
    uint8_t status{0};
} EndPoint;

typedef struct {
    uint64_t id; // PK
    uint64_t local_enpoint_id;
    uint64_t remote_enpoint_id;
    uint8_t tp_type;
} EndPointPair;

typedef struct {
    uint64_t id; // PK
    uint64_t port_id;
    uint64_t endpoint_id;
    uint8_t net_layer;
} EndPointPortMapping;

typedef struct {
    uint64_t id; // PK
    uint64_t local_endpoint_id;
    uint64_t remote_endpoint_id;
    uint8_t net_layer;    // 网络层级: 0/1/2/3
    uint8_t type;         // 0: peer2peer, 1: peer2net
    uint8_t protocols[8]; // 0: UB_CTP, 1: UB_MEM, 2: UB_TP, ...
} Link;

typedef struct {
    uint64_t id; // PK
    uint16_t channel_id;
    uint64_t local_endpoint_id;
    uint64_t remote_endpoint_id;
    uint8_t protocol;
    uint32_t jetty_start;
    uint32_t jetty_num;
} CcuChannel;

typedef struct {
    uint64_t id; // PK
    uint64_t run_id;
    uint64_t device_id;
    uint8_t is_default;
    uint32_t ref_cnt;
    uint64_t float_overflow_addr;
    uint8_t capture_mode;
} Context;

typedef struct {
    uint64_t id;     // PK
    uint64_t ctx_id; // 上下文ID (FK)
    uint64_t sq_base_addr;
    uint8_t is_primary_default;
    uint8_t is_other_default;
    uint8_t priority;
    uint8_t schedule_strategy;
    uint8_t failure_mode;
    uint32_t user_tag;
    uint8_t overflow_switch;
    uint8_t activated;
    uint8_t capture_status;
    uint8_t task_complete_status;
} Stream;

typedef struct {
    uint64_t id; // PK
    uint64_t stream_id;
    uint64_t cid;
    uint64_t seq_number;
    uint8_t type;
} Task;

typedef struct {
    uint64_t id; // PK
    uint64_t excute_time_ms;
    uint64_t finish_time_ms;
    uint32_t op_timeout_s;
} EventSyncTask;

typedef struct {
    uint64_t id; // PK
    uint64_t create_ctx_id;
    uint64_t device_notify_seq;
    uint8_t value;
} Notify;

typedef struct {
    uint64_t id; // PK
    uint64_t notify_id;
    uint8_t name_or_key[16];
    uint8_t create_pid;
} IpcNotify;

typedef struct {
    uint64_t id;     // PK
    uint64_t ipc_id; // FK
    uint64_t visitor_pid;
} IpcNotifyVistorList;

typedef struct {
    uint64_t id;        // PK
    uint64_t notify_id; // FK
} NotifyRecordTask;

typedef struct {
    uint64_t id;        // PK
    uint64_t notify_id; // FK
} NotifyWaitTask;

typedef struct {
    uint64_t id;            // PK
    uint64_t create_ctx_id; // FK
    uint64_t event_flag;
    uint64_t device_res_seq;
    uint64_t created_time;
    uint8_t status;
} Event;

typedef struct {
    uint64_t id;
    uint64_t device_id;
    char name[64];
    uint64_t size;
    uint8_t type;
    uint64_t ref_count;
    uint8_t is_freed; // 0: using, 1: freed
} PhyMemBlock;

enum VirMemType { VIR_MEM_TYPE_HOST, VIR_MEM_TYPE_DEV };

typedef struct {
    uint64_t id;
    uint64_t start_ptr;      // 按照卡分配的虚拟编址的地址
    uint64_t dev_mapped_ptr; // device进程打开共享内存后的地址
    uint8_t is_dev_access;   // 0: dev不可直接访问, 1: dev可以直接访问
    uint64_t size;
    uint64_t ctx_id;
    uint64_t device_id; // Device表主键
    uint64_t rank_id;   // 通信域RankId
    uint64_t phy_mem_id;
    uint64_t owner_pid; // host查找根据pid
    uint8_t src_type;   // 0: host, 1: device
    uint8_t policy;
    uint8_t is_freed; // 0: using, 1: freed
} VirtualMemBlock;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint64_t offset;
    uint8_t create_pid;
} IpcMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} IpcMemWhiteList;

typedef struct {
    uint64_t id;
    uint64_t vir_mem_id;
    uint64_t phy_mem_id;
    uint8_t create_pid;
} FdMemRecord;

typedef struct {
    uint64_t id;
    uint64_t name_or_key;
    uint64_t pid;
    uint8_t create_pid;
} FdMemWhiteList;

typedef struct {
    uint64_t id;
    uint32_t device_id;
    uint8_t role;         // 0:server,1:client
    uint8_t state;        // 0: inited 1:listened
    uint64_t endpoint_id; // ip id
} RaSocket;

typedef struct {
    uint64_t id;
    uint64_t server_id; // FK
    uint64_t client_id; // FK
    uint32_t ref_cnt;
    uint32_t port;
    uint64_t tag_hash;
    uint8_t buf_status; // 0: buffer pending, 1: buffer ready
} RaSocketPair;

// todo: 数据建模？
typedef struct {
    uint64_t id;      // PK
    uint32_t rank_id; // todo: 后续可能需要跟device关联
    uint64_t base_addr;
    uint8_t buf_type;
    uint8_t reserved;
    uint64_t size;
    uint64_t global_offset;
} MemoryLayout;

// ReduceScatterV AllGatherV使用
struct VDataDesTagInner {
    uint16_t dataType;   // 数据类型
    uint32_t count{0};   // rank size
    uint64_t displs[64]; // 每个rank的数据在sendBuf中的偏移量（单位为dataType）
    uint64_t counts[64]; // 每个rank在sendBuf中的数据size，第i个元素表示需要向rank i发送/接受的数据量
};

struct All2AllDataDesTagInner {
    uint16_t sendType;              // 发送数据的数据类型
    uint16_t recvType;              // 接收数据的数据类型
    uint64_t sendCount;             // 发送数据量 (All2All)
    uint64_t recvCount;             // 接收数据量 (All2All)
    uint32_t count{0};              // count = rankSize * rankSize
    uint64_t sendCountMatrix[4096]; // (All2AllVC) sendCountMatrix[i * ranksize + j] 代表rank i发送到rank j的count参数
};

enum SimOpExpansionMode {
    SIM_OP_EXPANSION_MODE_CCU = 0,
    SIM_OP_EXPANSION_MODE_AICPU = 1,
    SIM_OP_EXPANSION_MODE_AIV = 2,
    SIM_OP_EXPANSION_MODE_RESERVED = 255
};

typedef struct {
    uint64_t id; // PK
    uint32_t rank_id;
    uint32_t src_rank;
    uint32_t dst_rank;
    uint32_t root;
    uint32_t rank_size;
    uint16_t chip_type;
    uint16_t op_type;
    uint16_t reduce_op;
    uint16_t data_type;
    uint64_t data_count;
    uint8_t op_expansion_mode;
    uint64_t ccu0_resource_base_addr;
    uint64_t ccu1_resource_base_addr;
    VDataDesTagInner vDataDes;
    All2AllDataDesTagInner all2AllDataDes;
} SimModelData;

typedef struct {
    uint64_t id; // PK
    uint32_t rank_id;
    uint64_t device_id; // FK
    uint16_t state{0};
} Rank;

typedef struct {
    uint64_t id; // PK
    uint32_t device_id;
    uint8_t mac_addr[14];
    uint8_t state;
    uint64_t endpoint_id; // FK local EID
} RaDevice;

typedef struct {
    uint64_t id;        // PK
    uint64_t ra_dev_id; // FK
    uint64_t send_cq_handle;
    uint64_t recv_cq_handle;
    uint32_t qp_num;
    uint8_t type;
    uint8_t state; // 0: RESET, 1: INIT, 2: RTR, 3:RTS
    uint64_t taJettyId;
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t peer_qp_id;
    uint32_t peer_qpn;
    uint32_t perr_lid;
    uint64_t pid;
} RaQP;

typedef struct {
    uint64_t id;        // PK
    uint64_t ra_dev_id; // FK
    uint32_t cqn;
    uint32_t size;
} RaCQ;

typedef struct {
    uint64_t id; // PK
    uint64_t cq_handle;
    uint32_t wr_id;
    uint8_t status; // 0: success, 1: flush_err
} RaCQE;

typedef struct {
    uint64_t id; // PK
    uint64_t local_key;
    uint64_t remote_key;
    uint64_t vptr_id;
    uint64_t length;
    uint64_t addr;
} RaMR;

typedef struct {
    uint64_t id;        // PK
    uint64_t device_id; // FK
    int mode;
    uint64_t endpoint_id; // FK local EID
    uint32_t eidIndex;
    uint64_t max_jetty_num;
    uint64_t max_jfc_num;
} RaContext;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint32_t chann_id;
    uint32_t mode;
} RaChan;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint32_t token_id;
    uint64_t ref_count;
} RaTokenId;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint8_t tp_type;
    uint64_t tpn;
    uint64_t speed;
    uint8_t status;
} RaTp;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t remote_key;
    uint64_t target_seg_handle;
    uint64_t remote_eid;
} RaRmem;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t addr;
    uint64_t size;
    uint64_t mem_key;
    uint64_t token_id;
} RaLmem;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t send_cq_handle;
    uint64_t recv_cq_handle;
    uint32_t sqDepth;
    uint64_t sqBuffer; // jetty下发wqe对应的buffer地址
    uint32_t rqDepth;
    uint8_t type;
    uint32_t jetty_id;
    uint32_t dieId;
    uint8_t state; // 0: RESET, 1: INIT, 2: RTR, 3:RTS
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t peer_jetty_handle;
    uint64_t pid;
} RaJetty;

typedef struct {
    uint64_t id;         // PK
    uint64_t ctx_handle; // FK
    uint64_t jfc_id;
    uint64_t depth;
    uint32_t mode; // jetty mode 0: URMA, 2: CCU, 3: Normal
    uint64_t policy;
} RaJfc;

typedef struct {
    uint64_t id;         // PK
    uint64_t jfc_handle; // FK
    uint64_t user_ctx;
    uint64_t status;
    uint32_t opcode;
    uint64_t byte_len;
} RaCr;

typedef struct {
    uint64_t id; // PK
    uint32_t physical_id;
} RaTlv;
} // namespace sim
#endif
