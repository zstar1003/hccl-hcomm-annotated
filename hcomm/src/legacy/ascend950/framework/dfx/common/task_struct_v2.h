/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef TS_TASK_STRUCT_V2_H
#define TS_TASK_STRUCT_V2_H

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif /* __cplusplus */

enum tag_ts_to_aicpu_msg_cmd_type {
    TS_AICPU_MSG_VERSION = 0,            /* 0 aicpu msg version */
    TS_AICPU_MODEL_OPERATE = 1,          /* 1 model operate */
    TS_AICPU_TASK_REPORT = 2,            /* 2 aic task report */
    TS_AICPU_ACTIVE_STREAM = 3,          /* 3 aicpu active stream */
    TS_AICPU_RECORD = 4,                 /* 4 aicpu notify */
    TS_AICPU_NORMAL_DATADUMP_REPORT = 5, /* 5 normal data dump report */
    TS_AICPU_DEBUG_DATADUMP_REPORT = 6,  /* 6 debug datadump report */
    TS_AICPU_DATADUMP_INFO_LOAD = 7,     /* 7 datadump info load */
    TS_AICPU_TIMEOUT_CONFIG = 8,         /* 8 aicpu timeout config */
    TS_AICPU_INFO_LOAD = 9,              /* 9 aicpu info load for tiling key sink */
    TS_AIC_ERROR_REPORT = 10,            /* 10 aic task err report */
    TS_INVALID_AICPU_CMD                 /* invalid flag */
};

typedef struct tag_ts_aicpu_msg_version {
    volatile uint16_t magic_num;
    volatile uint16_t version;
} ts_aicpu_msg_version_t;

typedef struct tag_ts_aicpu_model_operate_msg {
    uint64_t arg_ptr;
    uint16_t stream_id;
    uint16_t model_id;
    uint8_t cmd_type;
    uint8_t reserved[3];
} ts_aicpu_model_operate_msg_t;

typedef struct tag_ts_to_aicpu_task_report_msg {
    uint16_t model_id;
    uint16_t stream_id;
    uint32_t task_id;
    uint16_t result_code;
    uint16_t reserve;
} ts_to_aicpu_task_report_msg_t;

typedef struct tag_ts_aicpu_active_stream {
    volatile uint16_t stream_id;
    volatile uint8_t reserved[6];
    volatile uint64_t aicpu_stamp;
} ts_aicpu_active_stream_t;

typedef struct tag_ts_aicpu_record_msg {
    uint32_t record_id;
    uint8_t record_type;
    uint8_t reserved;
    uint16_t ret_code;      // using ts_error_t
    uint32_t fault_task_id; // using report error of operator
} ts_aicpu_record_msg_t;

typedef struct tag_ts_to_aicpu_normal_datadump_msg {
    uint32_t dump_task_id;
    uint16_t dump_stream_id;
    uint8_t is_model : 1;
    uint8_t rsv : 7;
    uint8_t dump_type;
} ts_to_aicpu_normal_datadump_msg_t;

typedef struct tag_ts_to_aicpu_debug_datadump_msg {
    uint32_t dump_task_id;
    uint32_t debug_dump_task_id;
    uint16_t dump_stream_id;
    uint8_t is_model : 1;
    uint8_t rsv : 7;
    uint8_t dump_type;
} ts_to_aicpu_debug_datadump_msg_t;

typedef struct ts_to_aicpu_datadump_info_load_msg {
    uint64_t dumpinfoPtr;
    uint32_t length;
    uint32_t task_id;
    uint16_t stream_id;
    uint16_t reserve;
} ts_to_aicpu_datadump_info_load_msg_t;

typedef struct tag_ts_to_aicpu_info_load_msg {
    uint64_t aicpu_info_ptr;
    uint32_t length;
    uint32_t task_id;
    uint16_t stream_id;
    uint16_t reserve;
} ts_to_aicpu_info_load_msg_t;

typedef struct tag_ts_aicpu_response_msg {
    uint32_t task_id;
    uint16_t stream_id;
    uint16_t result_code;
    uint8_t reserved; /* for normal/debug dump, and info load */
    uint8_t rsv[3];
} ts_aicpu_response_msg_t;

typedef struct tag_ts_to_aicpu_aic_err_msg {
    uint16_t result_code;
    uint16_t aic_bitmap_num;
    uint16_t aiv_bitmap_num;
    uint8_t bitmap[26];
} ts_to_aicpu_aic_err_msg_t;

typedef struct tag_ts_to_aicpu_timeout_config {
    volatile uint32_t op_wait_timeout_en : 1;
    volatile uint32_t op_execute_timeout_en : 1;
    volatile uint32_t rsv : 30;
    volatile uint32_t op_wait_timeout;
    volatile uint32_t op_execute_timeout;
} ts_to_aicpu_timeout_config_t;

typedef struct tag_ts_aicpu_msg_info {
    uint32_t pid;
    uint8_t cmd_type;
    uint8_t vf_id;
    uint8_t tid;
    uint8_t ts_id;
    union {
        ts_aicpu_msg_version_t aicpu_msg_version;
        ts_aicpu_model_operate_msg_t aicpu_model_operate;
        ts_to_aicpu_task_report_msg_t ts_to_aicpu_task_report;
        ts_aicpu_active_stream_t aicpu_active_stream;
        ts_aicpu_record_msg_t aicpu_record;
        ts_to_aicpu_normal_datadump_msg_t ts_to_aicpu_normal_datadump;
        ts_to_aicpu_debug_datadump_msg_t ts_to_aicpu_debug_datadump;
        ts_to_aicpu_datadump_info_load_msg_t ts_to_aicpu_datadump_info_load;
        ts_to_aicpu_timeout_config_t ts_to_aicpu_timeout_cfg;
        ts_to_aicpu_info_load_msg_t ts_to_aicpu_info_load;
        ts_aicpu_response_msg_t aicpu_resp;
        ts_to_aicpu_aic_err_msg_t aic_err_msg;
    } u;
} ts_aicpu_msg_info_t;

typedef struct tag_stream_snapshot {
    uint16_t stream_id;
    uint16_t task_id;
    uint16_t sq_id : 12;
    uint16_t sq_fsm : 4;
    uint16_t acsq_id : 8;
    uint16_t acsq_fsm : 6;
    uint16_t is_swap_in : 1;
    uint16_t rsv : 1;
} ts_stream_snapshot_t;

// ts_to_aicpu_task_report_t  resultcode
#define TASK_REPORT_RESULT_CODE_FAIL (1U)

typedef enum tag_ts_aicpu_record_type_v2 {
    AICPU_MSG_EVENT_RECORD_V2 = 1, /* 1 aicpu event record */
    AICPU_MSG_NOTIFY_RECORD_V2     /* 2 aicpu notify record */
} ts_aicpu_record_type_t_v2;

#ifdef __cplusplus
}
#endif /* __cplusplus */
#endif /* TS_TASK_STRUCT_V2_H */
