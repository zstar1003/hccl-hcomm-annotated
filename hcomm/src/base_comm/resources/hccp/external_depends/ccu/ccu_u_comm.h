/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_U_COMM_H
#define CCU_U_COMM_H

#include <stdbool.h>
#include <stdint.h>

#define CCU_DATA_TYPE_UNION_ARRAY_SIZE 8
#define CCU_RESOURCE_PATH_LEN_MAX 512
#define CCU_DATA_MAX_SIZE 0x800 /* CCU data max size is 2048 */
#define BYTE8 8
#define BYTE32 32
#define BYTE64 64
#define CCU_PROD_MASK 0x000FFFFF
#define CCU_PROD_SHIFT 8

#ifdef CCU_CONFIG_LLT
#define STATIC
#else
#define STATIC static
#endif

enum ccu_array_data_index {
    CCU_ARRAY_DATA_0 = 0,
    CCU_ARRAY_DATA_1,
    CCU_ARRAY_DATA_2,
    CCU_ARRAY_DATA_3,
    CCU_ARRAY_DATA_4,
    CCU_ARRAY_DATA_5,
    CCU_ARRAY_DATA_6,
    CCU_ARRAY_DATA_7,
};

/* opcode definition */
typedef enum ccu_u_opcode {
    CCU_U_OP_GET_VERSION = 0, /* 获取所有die的版本号，未启用的die版本号为非法值 */
    CCU_U_OP_K_MIN = 10,      /* 定义需要向内核发送请求的操作最小值 */

    CCU_U_OP_GET_BASIC_INFO = 11,               /* 获取基础信息 */
    CCU_U_OP_GET_MISSION_TIMEOUT_DURATION = 12, /* 获取任务超时时间 */
    CCU_U_OP_GET_LOOP_TIMEOUT_DURATION = 13,    /* 获取循环超时时间 */
    CCU_U_OP_GET_MSID = 14,                     /* 获取MSID */
    CCU_U_OP_GET_DIE_WORKING = 15,              /* 获取该dieId是否工作 */

    CCU_U_OP_SET_MISSION_TIMEOUT_DURATION = 51, /* 设置任务超时时间 */
    CCU_U_OP_SET_LOOP_TIMEOUT_DURATION = 52,    /* 设置循环超时时间 */
    CCU_U_OP_SET_MSID_TOKEN = 53,               /* 设置MSID和Token相关值 */
    CCU_U_OP_SET_TASKKILL = 54,                 /* 启动taskkill任务 */
    CCU_U_OP_CLEAN_TASKKILL_STATE = 55,         /* 清除taskkill任务 */
    CCU_U_OP_CLEAN_TIF_TABLE = 56,              /* 清除TIF表项 */

    CCU_U_OP_K_MAX = 100, /* 定义需要向内核发送请求的操作最大值 */

    CCU_U_OP_GET_CHN_JETTY_MAP = 101, /* 获取Channel与Jetty的映射关系 */
    CCU_U_OP_GET_HIGH_PERF_XN = 102,  /* Get high-performance XN register range for 0.5RTT feature */

    CCU_U_OP_SET_CHN_JETTY_MAP = 126,  /* 设置Channel与Jetty的映射关系 */
    CCU_U_OP_SET_TIF_SPLIT_SIZE = 127, /* Set count unit for the 0.5RTT feature */
    CCU_U_OP_SET_XN_TOTAL_CNT = 128,   /* Set compare register for 0.5RTT feature */
    CCU_U_OP_SET_TRANM_NTF_HVAL = 129, /* Set high 32bits of notify value in Transmem for 0.5RTT feature */
    CCU_U_OP_CONFIG_HW_MAX = 150,      /* 定义操作CCU硬件寄存器的最大值 */

    /* 以下为操作CCU映射到用户态资源空间的操作码 */
    CCU_U_OP_IN_RS_MIN = 200, /* 定义一个在RS空间操作的最小值 */

    CCU_U_OP_GET_INSTRUCTION = 201,  /* 获取INS指令 */
    CCU_U_OP_GET_GSA = 202,          /* 获取GSA数据 */
    CCU_U_OP_GET_XN = 203,           /* 获取XN数据 */
    CCU_U_OP_GET_CKE = 204,          /* 获取CKE数据 */
    CCU_U_OP_GET_PFE = 205,          /* 获取PFE数据 */
    CCU_U_OP_GET_CHANNEL = 206,      /* 获取Channel数据 */
    CCU_U_OP_GET_JETTY_CTX = 207,    /* 获取Jetty_ctx数据 */
    CCU_U_OP_GET_MISSION_CTX = 208,  /* 获取Mission_ctx数据 */
    CCU_U_OP_GET_LOOP_CTX = 209,     /* 获取Loop_ctx数据 */
    CCU_U_OP_GET_MEMORY_SLICE = 210, /* 获取memory slice数据 */
    CCU_U_OP_GET_LOOP_CKE_CTX = 211, /* 获取LOOP_CKE_CTX数据 */
    CCU_U_OP_GET_MISSION_SQE = 212,  /* 获取MISSION_SQE数据 */
    CCU_U_OP_GET_CQE_BLOCK0 = 213,   /* 获取CQE_BLOCK0数据 */
    CCU_U_OP_GET_CQE_BLOCK1 = 214,   /* 获取CQE_BLOCK1数据 */
    CCU_U_OP_GET_CQE_BLOCK2 = 215,   /* 获取CQE_BLOCK2数据 */
    CCU_U_OP_GET_WQEBB = 216,        /* 获取WQEBB数据 */
    CCU_U_OP_GET_MS_BLOCK0 = 217,    /* 获取MS_BLOCK0数据 */
    CCU_U_OP_GET_MS_BLOCK1 = 218,    /* 获取MS_BLOCK1数据 */
    CCU_U_OP_GET_MS_BLOCK2 = 219,    /* 获取MS_BLOCK2数据 */
    CCU_U_OP_GET_MS_BLOCK3 = 220,    /* 获取MS_BLOCK3数据 */

    CCU_U_OP_SET_INSTRUCTION = 251,  /* 设置INS指令 */
    CCU_U_OP_SET_GSA = 252,          /* 设置GSA数据 */
    CCU_U_OP_SET_XN = 253,           /* 设置XN数据 */
    CCU_U_OP_SET_CKE = 254,          /* 设置CKE数据 根据芯片约束set前需要先清零 */
    CCU_U_OP_SET_PFE = 255,          /* 设置PFE数据 */
    CCU_U_OP_SET_CHANNEL = 256,      /* 设置Channel数据 */
    CCU_U_OP_SET_JETTY_CTX = 257,    /* 设置Jetty_ctx数据 */
    CCU_U_OP_SET_LOOP_CKE_CTX = 258, /* 设置LOOP_CKE_CTX数据 */

    CCU_U_OP_IN_RS_MAX = 300, /* 定义一个在RS空间操作的最大值 */
} ccu_u_opcode_t;

struct ccu_region {
    ccu_u_opcode_t ccu_u_op;       /* ccu u opcode */
    unsigned int ccu_region_saddr; /* base offset start addr */
    unsigned int ccu_region_size;  /* region total size */
    unsigned int ccu_entry_size;   /* entry size (per Byte) */
    void *ccu_va_udie0;            /* virtual address get by mmap func */
    void *ccu_va_udie1;            /* virtual address get by mmap func */
};

struct ccu_data_byte8 {
    char raw[BYTE8];
};

struct ccu_data_byte32 {
    char raw[BYTE32];
};

struct ccu_data_byte64 {
    char raw[BYTE64];
};

struct ccu_data_caps {
    unsigned int lqc_ccu_cap0;
    unsigned int lqc_ccu_cap1;
    unsigned int lqc_ccu_cap2;
    unsigned int lqc_ccu_cap3;
    unsigned int lqc_ccu_cap4;
};

struct ccu_baseinfo {
    // set from hccl
    unsigned int ms_id;
    unsigned int token_id;
    unsigned int token_value;

    // get from ccu drv
    unsigned int token_valid;

    // get from chip
    unsigned int missionKey;
    void *resourceAddr;
    struct ccu_data_caps caps;
    unsigned int chn_jetty_map;
};

/* 设置Instruction指令的专用数据结构 */
struct ccu_insinfo {
    unsigned long resourceAddr;
};

struct ccu_dieinfo {
    unsigned int enable_flag;
};

enum ccu_version_e {
    CCU_V1 = 0,
    CCU_V2 = 1,
    CCU_VERSION_MAX,
};

struct ccu_tif_split_size {
    unsigned int split_pkt_unit;
    unsigned int tp_split_size;
    unsigned int ctp_split_size;
};

struct ccu_high_perf_xn {
    unsigned int start_id;
    unsigned int xn_size;
};

struct ccu_xn_total_cnt {
    unsigned int cnt_index;
    unsigned int total_value;
    unsigned int total_addr;
    unsigned int flag_from_addr;
    unsigned int flag_to_addr;
};

struct ccu_mission_kill_info {
    bool is_single_mission; // 0:kill all; 1:single mission kill
    unsigned int mission_id;
};

union ccu_data_type_union {
    struct ccu_data_byte8 byte8;
    struct ccu_data_byte32 byte32;
    struct ccu_data_byte64 byte64;
    struct ccu_baseinfo baseinfo;
    /* ccu_insinfo 只支持set时，传入一个元素，不支持循环配置 */
    struct ccu_insinfo insinfo;
    struct ccu_dieinfo dieinfo;
    enum ccu_version_e version;
    struct ccu_tif_split_size tif_split_info;
    struct ccu_high_perf_xn high_perf_xn;
    struct ccu_xn_total_cnt xn_total_cnt;
    unsigned int tranm_ntf_hval;
    struct ccu_mission_kill_info mission_info;
};

struct ccu_data {
    unsigned int udie_idx;
    unsigned int data_array_len; /* 数据的总长度 (data_array_size * 每个元素的大小，以Byte为单位) 的值 */
    unsigned int data_array_size;                                         /* data_array里总共有多少个元素 */
    union ccu_data_type_union data_array[CCU_DATA_TYPE_UNION_ARRAY_SIZE]; /* 不同类型的数据，通过联合体来存储 */
};

union ccu_data_union {
    char raw[CCU_DATA_MAX_SIZE]; /* 对外呈现是一个字符数组，内部转换成对应类型ccu_data */
    struct ccu_data data_info;
};

struct channel_info_in {
    union ccu_data_union data;
    unsigned int offset_start; /* 对应需要操作的元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小 */
    ccu_u_opcode_t op;
};

struct channel_info_out {
    union ccu_data_union data; /* 对外呈现是一个字符数组，内部转换成对应类型ccu_data */
    unsigned int offset_next; /* 操作后返回下一个元素的idx位置，位置用正整数代替，使用者不需要关心元素的实际大小 */
    int op_ret;
};

struct ccu_version {
    unsigned int prod_version;
    unsigned int arch_version;
};

/*
 * ccu_u_info 用户态内部数据结构体,用于存放用户态运行时的内部数据
 * 包括部分全局变量，历史信息
 */
struct ccu_u_info {
    /* set at init proc, and will never change */
    unsigned int uent_num;     /* ccu uent_num */
    unsigned int ccu_flag;     /* ccu exist flag */
    unsigned int eid;          /* ccu eid */
    unsigned int ms_id;        /* ccu ms id */
    unsigned int missionKey;   /* mission sqe secure key */
    void *resourceAddr;        /* ccu resource addr va */
    struct ccu_data_caps caps; /* ccu caps, get from ccu regs */
    struct ccu_version version;
};

#define MEM_BITMAP_NUM 64U

#define CCU_MEMTYPE_INVALID ((uint64_t)0ULL)
#define CCU_MEMTYPE_INS ((uint64_t)1ULL << 0)
#define CCU_MEMTYPE_GSA ((uint64_t)1ULL << 1)
#define CCU_MEMTYPE_XN ((uint64_t)1ULL << 2)
#define CCU_MEMTYPE_CKE ((uint64_t)1ULL << 3)
#define CCU_MEMTYPE_LOOP_CKE ((uint64_t)1ULL << 4)
#define CCU_MEMTYPE_PFE ((uint64_t)1ULL << 5)
#define CCU_MEMTYPE_CHN ((uint64_t)1ULL << 6)
#define CCU_MEMTYPE_JETTY_CTX ((uint64_t)1ULL << 7)
#define CCU_MEMTYPE_MISSION_CTX ((uint64_t)1ULL << 8)
#define CCU_MEMTYPE_LOOP_CTX ((uint64_t)1ULL << 9)
#define CCU_MEMTYPE_MISSION_SQE ((uint64_t)1ULL << 10)
#define CCU_MEMTYPE_CQE_BLOCK0 ((uint64_t)1ULL << 11)
#define CCU_MEMTYPE_CQE_BLOCK1 ((uint64_t)1ULL << 12)
#define CCU_MEMTYPE_CQE_BLOCK2 ((uint64_t)1ULL << 13)
#define CCU_MEMTYPE_WQEBB ((uint64_t)1ULL << 14)

#define CCU_MEMTYPE_MS_BLOCK0 ((uint64_t)1ULL << 32) // CCUA0
#define CCU_MEMTYPE_MS_BLOCK1 ((uint64_t)1ULL << 33) // CCUA1
#define CCU_MEMTYPE_MS_BLOCK2 ((uint64_t)1ULL << 34) // CCUA2
#define CCU_MEMTYPE_MS_BLOCK3 ((uint64_t)1ULL << 35) // CCUA3

struct ccu_memtype_map {
    uint64_t memtype;
    ccu_u_opcode_t ccu_u_op;
};

struct ccu_mem_info {
    unsigned long long mem_va;
    unsigned int mem_size;
    unsigned int resv[1U];
};

bool is_ccu_attached(unsigned int die_id);

#endif /* CCU_U_COMM_H */
