/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef IB_VERBS_H
#define IB_VERBS_H

#include <pthread.h>

#include <infiniband/driver.h>

#define INIT __attribute__((constructor))

#define PFX "libibverbs: "

struct ibv_abi_compat_v2 {
    struct ibv_comp_channel channel;
    pthread_mutex_t in_use;
};

extern const struct verbs_context_ops verbs_dummy_ops;

int ibverbs_get_device_list(struct list_head* list);
int ibverbs_init(void);
void ibverbs_device_put(struct ibv_device* dev);
void ibverbs_device_hold(struct ibv_device* dev);

struct verbs_ex_private {
    uint32_t driver_id;
    bool use_ioctl_write;
    struct verbs_context_ops ops;
};

#define IBV_INIT_CMD(cmd, size, opcode)                  \
    do {                                                 \
        (cmd)->hdr.command = IB_USER_VERBS_CMD_##opcode; \
        (cmd)->hdr.in_words = (size) / 4;                \
        (cmd)->hdr.out_words = 0;                        \
    } while (0)

#define IBV_INIT_CMD_RESP(cmd, size, opcode, out, outsize) \
    do {                                                   \
        (cmd)->hdr.command = IB_USER_VERBS_CMD_##opcode;   \
        (cmd)->hdr.in_words = (size) / 4;                  \
        (cmd)->hdr.out_words = (outsize) / 4;              \
        (cmd)->response = (uintptr_t)(out);                \
    } while (0)

static inline uint32_t _cmd_ex(uint32_t cmd)
{
    return (IB_USER_VERBS_CMD_FLAG_EXTENDED << IB_USER_VERBS_CMD_FLAGS_SHIFT) | cmd;
}

#define IBV_INIT_CMD_RESP_EX_V(cmd, cmd_size, size, opcode, out, resp_size, outsize) \
    do {                                                                             \
        size_t c_size = cmd_size - sizeof(struct ex_hdr);                            \
        (cmd)->hdr.hdr.command = _cmd_ex(IB_USER_VERBS_EX_CMD_##opcode);             \
        (cmd)->hdr.hdr.in_words = ((c_size) / 8);                                    \
        (cmd)->hdr.hdr.out_words = ((resp_size) / 8);                                \
        (cmd)->hdr.ex_hdr.provider_in_words = (((size) - (cmd_size)) / 8);           \
        (cmd)->hdr.ex_hdr.provider_out_words = (((outsize) - (resp_size)) / 8);      \
        (cmd)->hdr.ex_hdr.response = (uintptr_t)(out);                               \
        (cmd)->hdr.ex_hdr.cmd_hdr_reserved = 0;                                      \
    } while (0)

#define IBV_INIT_CMD_RESP_EX_VCMD(cmd, cmd_size, size, opcode, out, outsize) \
    IBV_INIT_CMD_RESP_EX_V(cmd, cmd_size, size, opcode, out, sizeof(*(out)), outsize)

#define IBV_INIT_CMD_RESP_EX(cmd, size, opcode, out, outsize) \
    IBV_INIT_CMD_RESP_EX_V(cmd, sizeof(*(cmd)), size, opcode, out, sizeof(*(out)), outsize)

#define IBV_INIT_CMD_EX(cmd, size, opcode) IBV_INIT_CMD_RESP_EX_V(cmd, sizeof(*(cmd)), size, opcode, NULL, 0, 0)

/**
 * ibv_modify_qp - Modify a queue pair.
 */
int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask);

/**
 * ibv_reg_mr - Register a memory region
 */
struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access);

/**
 * ibv_query_gid - Get a GID table entry
 */
int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid);

int ibv_query_gid_type(
    struct ibv_context* context, uint8_t port_num, unsigned int index, enum ibv_gid_type_sysfs* type);
/**
 * ibv_create_cq - Create a completion queue
 * @context - Context CQ will be attached to
 * @cqe - Minimum number of entries required for CQ
 * @cq_context - Consumer-supplied context returned for completion events
 * @channel - Completion channel where completion events will be queued.
 *     May be NULL if completion events will not be used.
 * @comp_vector - Completion vector used to signal completion events.
 *     Must be >= 0 and < context->num_comp_vectors.
 */
struct ibv_cq* ibv_create_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector);

/**
 * ibv_destroy_cq - Destroy a completion queue
 */
int ibv_destroy_cq(struct ibv_cq* cq);

/**
 * ibv_alloc_pd - Allocate a protection domain
 */
struct ibv_pd* ibv_alloc_pd(struct ibv_context* context);

/**
 * ibv_create_qp - Create a queue pair.
 */
struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr);

/**
 * ibv_query_qp - Returns the attribute list and current values for the
 *   specified QP.
 * @qp: The QP to query.
 * @attr: The attributes of the specified QP.
 * @attr_mask: A bit-mask used to select specific attributes to query.
 * @init_attr: Additional attributes of the selected QP.
 *
 * The qp_attr_mask may be used to limit the query to gathering only the
 * selected attributes.
 */
int ibv_query_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr);

/**
 * ibv_destroy_qp - Destroy a queue pair.
 */
int ibv_destroy_qp(struct ibv_qp* qp);

/**
 * ibv_dealloc_pd - Free a protection domain
 */
int ibv_dealloc_pd(struct ibv_pd* pd);

/**
 * ibv_get_device_list - Get list of IB devices currently available
 * @num_devices: optional.  if non-NULL, set to the number of devices
 * returned in the array.
 *
 * Return a NULL-terminated array of IB devices.  The array can be
 * released with ibv_free_device_list().
 */
struct ibv_device** ibv_get_device_list(int* num_devices);

/**
 * ibv_open_device - Initialize device for use
 */
struct ibv_context* ibv_open_device(struct ibv_device* device);

/**
 * ibv_dereg_mr - Deregister a memory region
 */
int ibv_dereg_mr(struct ibv_mr* mr);

/**
 * ibv_query_port - Get port properties
 */
int ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr);

/**
 * ibv_create_ah - Create an address handle.
 */
struct ibv_ah* ibv_create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr);

/**
 * ibv_destroy_ah - Destroy an address handle.
 */
int ibv_destroy_ah(struct ibv_ah* ah);

struct ibv_comp_channel* ibv_create_comp_channel(struct ibv_context* context);
int ibv_destroy_comp_channel(struct ibv_comp_channel* channel);

int ibv_get_cq_event(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context);
void ibv_ack_cq_events(struct ibv_cq* cq, unsigned int nevents);

int ibv_poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc);

int ibv_poll_cq_stub(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc);

int ibv_req_notify_cq(struct ibv_cq* cq, int solicited_only);

int ibv_get_cq_event_stub(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context);

const char* ibv_wc_status_str(enum ibv_wc_status status);

const char* ibv_get_device_name_stub2(struct ibv_device* device);

struct ibv_device** ibv_get_device_list_stub2(int* num_devices);

#endif /* IB_VERBS_H */
