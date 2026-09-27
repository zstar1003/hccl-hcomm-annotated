/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <config.h>

#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <errno.h>
#include <alloca.h>
#include <string.h>

#include <infiniband/verbs.h>
#include <infiniband/driver.h>
#include "ibverbs.h"
#include "rs_inner.h"
#include "verbs_exp.h"

static int testcase_return = 0;

static void copy_query_dev_fields(
    struct ibv_device_attr* device_attr, struct ib_uverbs_query_device_resp* resp, uint64_t* raw_fw_ver)
{
    *raw_fw_ver = 0x100;
    device_attr->node_guid = 3;
    device_attr->sys_image_guid = 3;
    device_attr->max_mr_size = 10;
    device_attr->page_size_cap = 100;
    device_attr->max_qp = 1000;
    device_attr->max_qp_wr = 1000;
    device_attr->max_sge = 100;
    device_attr->max_sge_rd = 100;
    device_attr->max_cq = 100;
    device_attr->max_cqe = 100;
    device_attr->max_mr = 100;
    device_attr->max_pd = 100;
    device_attr->max_qp_rd_atom = 100;
    device_attr->max_ee_rd_atom = 100;
    device_attr->max_res_rd_atom = 100;
    device_attr->max_qp_init_rd_atom = 100;
    device_attr->max_ee_init_rd_atom = 100;
    device_attr->atomic_cap = 100;
    device_attr->max_ee = 100;
    device_attr->max_rdd = 100;
    device_attr->max_mw = 100;
    device_attr->max_raw_ipv6_qp = 100;
    device_attr->max_raw_ethy_qp = 100;
    device_attr->max_mcast_grp = 100;
    device_attr->max_mcast_qp_attach = 100;
    device_attr->max_total_mcast_qp_attach = 100;
    device_attr->max_ah = 100;
    device_attr->max_fmr = 100;
    device_attr->max_map_per_fmr = 100;
    device_attr->max_srq = 100;
    device_attr->max_srq_wr = 100;
    device_attr->max_srq_sge = 100;
    device_attr->max_pkeys = 100;
    device_attr->local_ca_ack_delay = 100;
    device_attr->phys_port_cnt = 100;
}

int ibv_cmd_query_device(
    struct ibv_context* context, struct ibv_device_attr* device_attr, uint64_t* raw_fw_ver,
    struct ibv_query_device* cmd, size_t cmd_size)
{
    memset(device_attr->fw_ver, 0, sizeof device_attr->fw_ver);
    copy_query_dev_fields(device_attr, NULL, raw_fw_ver);

    return testcase_return;
}

int ibv_cmd_query_device_ex(
    struct ibv_context* context, const struct ibv_query_device_ex_input* input, struct ibv_device_attr_ex* attr,
    size_t attr_size, uint64_t* raw_fw_ver, struct ibv_query_device_ex* cmd, size_t cmd_core_size, size_t cmd_size,
    struct ib_uverbs_ex_query_device_resp* resp, size_t resp_core_size, size_t resp_size)
{
    return 0;
}

int ibv_cmd_query_device_any(
    struct ibv_context* context, const struct ibv_query_device_ex_input* input, struct ibv_device_attr_ex* attr,
    size_t attr_size, struct ib_uverbs_ex_query_device_resp* resp, size_t* resp_size)
{
    return 0;
}

int ibv_cmd_query_port(
    struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr, struct ibv_query_port* cmd,
    size_t cmd_size)
{
    return 0;
}

int ibv_cmd_alloc_pd(
    struct ibv_context* context, struct ibv_pd* pd, struct ibv_alloc_pd* cmd, size_t cmd_size,
    struct ib_uverbs_alloc_pd_resp* resp, size_t resp_size)
{
    pd->handle = 0;
    pd->context = context;
    return 0;
}

int ibv_cmd_dealloc_pd(struct ibv_pd* pd) { return 0; }

int ibv_cmd_open_xrcd(
    struct ibv_context* context, struct verbs_xrcd* xrcd, int vxrcd_size, struct ibv_xrcd_init_attr* attr,
    struct ibv_open_xrcd* cmd, size_t cmd_size, struct ib_uverbs_open_xrcd_resp* resp, size_t resp_size)
{
    return 0;
}

int ibv_cmd_close_xrcd(struct verbs_xrcd* xrcd) { return 0; }

int ibv_cmd_reg_mr(
    struct ibv_pd* pd, void* addr, size_t length, uint64_t hca_va, int access, struct ibv_mr* mr,
    struct ibv_reg_mr* cmd, size_t cmd_size, struct ib_uverbs_reg_mr_resp* resp, size_t resp_size)
{
    return 0;
}

int ibv_cmd_rereg_mr(
    struct ibv_mr* mr, uint32_t flags, void* addr, size_t length, uint64_t hca_va, int access, struct ibv_pd* pd,
    struct ibv_rereg_mr* cmd, size_t cmd_sz, struct ib_uverbs_rereg_mr_resp* resp, size_t resp_sz)
{
    return 0;
}

int ibv_cmd_dereg_mr(struct ibv_mr* mr) { return 0; }

int ibv_cmd_alloc_mw(
    struct ibv_pd* pd, enum ibv_mw_type type, struct ibv_mw* mw, struct ibv_alloc_mw* cmd, size_t cmd_size,
    struct ib_uverbs_alloc_mw_resp* resp, size_t resp_size)
{
    return 0;
}

int ibv_cmd_dealloc_mw(struct ibv_mw* mw) { return 0; }

int ibv_cmd_create_cq(
    struct ibv_context* context, int cqe, struct ibv_comp_channel* channel, int comp_vector, struct ibv_cq* cq,
    struct ibv_create_cq* cmd, size_t cmd_size, struct ib_uverbs_create_cq_resp* resp, size_t resp_size)
{
    cq->handle = 0;
    cq->cqe = 0;
    cq->context = context;
    return 0;
}

int ibv_cmd_create_cq_ex(
    struct ibv_context* context, struct ibv_cq_init_attr_ex* cqAttr, struct ibv_cq_ex* cq, struct ibv_create_cq_ex* cmd,
    size_t cmd_core_size, size_t cmd_size, struct ib_uverbs_ex_create_cq_resp* resp, size_t resp_core_size,
    size_t resp_size)
{
    return 0;
}

int ibv_cmd_poll_cq(struct ibv_cq* ibcq, int ne, struct ibv_wc* wc) { return 0; }

int ibv_cmd_req_notify_cq(struct ibv_cq* ibcq, int solicited_only) { return 0; }

int ibv_cmd_resize_cq(
    struct ibv_cq* cq, int cqe, struct ibv_resize_cq* cmd, size_t cmd_size, struct ib_uverbs_resize_cq_resp* resp,
    size_t resp_size)
{
    return 0;
}

int ibv_cmd_destroy_cq(struct ibv_cq* cq) { return 0; }

int ibv_cmd_create_srq(
    struct ibv_pd* pd, struct ibv_srq* srq, struct ibv_srq_init_attr* attr, struct ibv_create_srq* cmd, size_t cmd_size,
    struct ib_uverbs_create_srq_resp* resp, size_t resp_size)
{
    srq->handle = 0;
    srq->context = pd->context;
    return 0;
}

int ibv_cmd_create_srq_ex(
    struct ibv_context* context, struct verbs_srq* srq, int vsrq_sz, struct ibv_srq_init_attr_ex* attr_ex,
    struct ibv_create_xsrq* cmd, size_t cmd_size, struct ib_uverbs_create_srq_resp* resp, size_t resp_size)
{
    return 0;
}

static int ibv_cmd_modify_srq_v3(
    struct ibv_srq* srq, struct ibv_srq_attr* srq_attr, int srq_attr_mask, struct ibv_modify_srq* new_cmd,
    size_t new_cmd_size)
{
    return 0;
}

int ibv_cmd_modify_srq(
    struct ibv_srq* srq, struct ibv_srq_attr* srq_attr, int srq_attr_mask, struct ibv_modify_srq* cmd, size_t cmd_size)
{
    return 0;
}

int ibv_cmd_query_srq(struct ibv_srq* srq, struct ibv_srq_attr* srq_attr, struct ibv_query_srq* cmd, size_t cmd_size)
{
    return 0;
}

int ibv_cmd_destroy_srq(struct ibv_srq* srq) { return 0; }

static int create_qp_ex_common(
    struct verbs_qp* qp, struct ibv_qp_init_attr_ex* qpAttr, struct verbs_xrcd* vxrcd, struct ibv_create_qp_common* cmd)
{
    return 0;
}

static void create_qp_handle_resp_common(
    struct ibv_context* context, struct verbs_qp* qp, struct ibv_qp_init_attr_ex* qpAttr,
    struct ib_uverbs_create_qp_resp* resp, struct verbs_xrcd* vxrcd, int vqp_sz)
{}

enum {
    CREATE_QP_EX2_SUP_CREATE_FLAGS = IBV_QP_CREATE_BLOCK_SELF_MCAST_LB | IBV_QP_CREATE_SCATTER_FCS
                                     | IBV_QP_CREATE_CVLAN_STRIPPING | IBV_QP_CREATE_SOURCE_QPN
                                     | IBV_QP_CREATE_PCI_WRITE_END_PADDING,
};

int ibv_cmd_create_qp_ex2(
    struct ibv_context* context, struct verbs_qp* qp, int vqp_sz, struct ibv_qp_init_attr_ex* qpAttr,
    struct ibv_create_qp_ex* cmd, size_t cmd_core_size, size_t cmd_size, struct ib_uverbs_ex_create_qp_resp* resp,
    size_t resp_core_size, size_t resp_size)
{
    return 0;
}

int ibv_cmd_create_qp_ex(
    struct ibv_context* context, struct verbs_qp* qp, int vqp_sz, struct ibv_qp_init_attr_ex* attr_ex,
    struct ibv_create_qp* cmd, size_t cmd_size, struct ib_uverbs_create_qp_resp* resp, size_t resp_size)
{
    return 0;
}

int ibv_cmd_create_qp(
    struct ibv_pd* pd, struct ibv_qp* qp, struct ibv_qp_init_attr* attr, struct ibv_create_qp* cmd, size_t cmd_size,
    struct ib_uverbs_create_qp_resp* resp, size_t resp_size)
{
    qp->handle = 0;
    qp->qp_num = 0x100;
    qp->context = pd->context;
    return 0;
}

int ibv_cmd_open_qp(
    struct ibv_context* context, struct verbs_qp* qp, int vqp_sz, struct ibv_qp_open_attr* attr,
    struct ibv_open_qp* cmd, size_t cmd_size, struct ib_uverbs_create_qp_resp* resp, size_t resp_size)
{
    return 0;
}

int ibv_cmd_query_qp(
    struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr,
    struct ibv_query_qp* cmd, size_t cmd_size)
{
    return 0;
}

static void
copy_modify_qp_fields(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ib_uverbs_modify_qp* cmd)
{}

int ibv_cmd_modify_qp(
    struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_modify_qp* cmd, size_t cmd_size)
{
    return 0;
}

int ibv_cmd_modify_qp_ex(
    struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_modify_qp_ex* cmd, size_t cmd_core_size,
    size_t cmd_size, struct ib_uverbs_ex_modify_qp_resp* resp, size_t resp_core_size, size_t resp_size)
{
    return 0;
}

int ibv_cmd_post_send(struct ibv_qp* ibqp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) { return 0; }

int ibv_cmd_post_recv(struct ibv_qp* ibqp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) { return 0; }

int ibv_cmd_post_srq_recv(struct ibv_srq* srq, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) { return 0; }

int ibv_cmd_create_ah(
    struct ibv_pd* pd, struct ibv_ah* ah, struct ibv_ah_attr* attr, struct ib_uverbs_create_ah_resp* resp,
    size_t resp_size)
{
    return 0;
}

int ibv_cmd_destroy_ah(struct ibv_ah* ah) { return 0; }

int ibv_cmd_destroy_qp(struct ibv_qp* qp) { return 0; }

int ibv_cmd_attach_mcast(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid) { return 0; }

int ibv_cmd_detach_mcast(struct ibv_qp* qp, const union ibv_gid* gid, uint16_t lid) { return 0; }

static int buffer_is_zero(char* addr, ssize_t size) { return addr[0] == 0 && !memcmp(addr, addr + 1, size - 1); }

static int get_filters_size(
    struct ibv_flow_spec* ib_spec, struct ibv_kern_spec* kern_spec, int* ib_filter_size, int* kern_filter_size,
    enum ibv_flow_spec_type type)
{
    return 0;
}

static int ib_spec_to_kern_spec(struct ibv_flow_spec* ib_spec, struct ibv_kern_spec* kern_spec) { return 0; }

int ibv_cmd_create_flow(struct ibv_qp* qp, struct ibv_flow* flow_id, struct ibv_flow_attr* flow_attr) { return 0; }

int ibv_cmd_destroy_flow(struct ibv_flow* flow_id) { return 0; }

int ibv_cmd_create_wq(
    struct ibv_context* context, struct ibv_wq_init_attr* wq_init_attr, struct ibv_wq* wq, struct ibv_create_wq* cmd,
    size_t cmd_core_size, size_t cmd_size, struct ib_uverbs_ex_create_wq_resp* resp, size_t resp_core_size,
    size_t resp_size)
{
    return 0;
}

int ibv_cmd_modify_wq(
    struct ibv_wq* wq, struct ibv_wq_attr* attr, struct ibv_modify_wq* cmd, size_t cmd_core_size, size_t cmd_size)
{
    return 0;
}

int ibv_cmd_destroy_wq(struct ibv_wq* wq) { return 0; }

int ibv_cmd_create_rwq_ind_table(
    struct ibv_context* context, struct ibv_rwq_ind_table_init_attr* init_attr, struct ibv_rwq_ind_table* rwq_ind_table,
    struct ibv_create_rwq_ind_table* cmd, size_t cmd_core_size, size_t cmd_size,
    struct ib_uverbs_ex_create_rwq_ind_table_resp* resp, size_t resp_core_size, size_t resp_size)
{
    return 0;
}

int ibv_cmd_destroy_rwq_ind_table(struct ibv_rwq_ind_table* rwq_ind_table) { return 0; }

int ibv_cmd_modify_cq(struct ibv_cq* cq, struct ibv_modify_cq_attr* attr, struct ibv_modify_cq* cmd, size_t cmd_size)
{
    return 0;
}

int ibv_dofork_range(void* base, size_t size) { return 0; }

int ibv_dontfork_range(void* base, size_t size) { return 0; }

#ifndef __HNS_RS_VERBS_STUB__
/**
 * ibv_modify_qp - Modify a queue pair.
 */
int ibv_modify_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask)
{
    if (attr_mask & IBV_QP_STATE) {
        qp->state = attr->qp_state;
    }

    return 0;
}

/**
 * ibv_reg_mr - Register a memory region
 */
struct ibv_mr* ibv_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access)
{
    struct ibv_mr* mr;

    mr = malloc(sizeof(struct ibv_mr));
    if (NULL == mr) {
        return NULL;
    }
    memset(mr, 0, sizeof(struct ibv_mr));

    mr->addr = addr;
    mr->length = length;
    mr->lkey = 0x5a5a;
    mr->rkey = 0xa5a5;
    mr->pd = pd;
    mr->context = pd->context;

    return mr;
}

/**
 * ibv_query_gid - Get a GID table entry
 */
int ibv_query_gid(struct ibv_context* context, uint8_t port_num, int index, union ibv_gid* gid)
{
    unsigned int gid_v4[4];

    gid_v4[0] = 0;
    gid_v4[1] = 0;
    gid_v4[2] = htonl(0x0000FFFF);
    gid_v4[3] = inet_addr("127.0.0.1");

    memcpy(gid->raw, &gid_v4, sizeof(gid->raw));

    return 0;
}

int ibv_query_gid_type(struct ibv_context* context, uint8_t port_num, unsigned int index, enum ibv_gid_type_sysfs* type)
{
    if (index % 2 == 1) {
        *type = IBV_GID_TYPE_SYSFS_ROCE_V2;
    } else {
        *type = IBV_GID_TYPE_SYSFS_IB_ROCE_V1;
    }
    return 0;
}

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
struct ibv_cq*
ibv_create_cq(struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector)
{
    struct ibv_cq* cq;

    cq = malloc(sizeof(struct ibv_cq));
    if (NULL == cq) {
        return NULL;
    }
    memset(cq, 0, sizeof(struct ibv_cq));

    cq->channel = channel;
    cq->cqe = cqe;
    cq->context = context;
    cq->cq_context = cq_context;

    return cq;
}

/**
 * ibv_destroy_cq - Destroy a completion queue
 */
int ibv_destroy_cq(struct ibv_cq* cq)
{
    free(cq);

    return 0;
}

/**
 * ibv_alloc_pd - Allocate a protection domain
 */
struct ibv_pd* ibv_alloc_pd(struct ibv_context* context)
{
    struct ibv_pd* pd;

    pd = malloc(sizeof(struct ibv_pd));
    if (NULL == pd) {
        return NULL;
    }
    memset(pd, 0, sizeof(struct ibv_pd));

    pd->context = context;

    return pd;
}

/**
 * ibv_create_qp - Create a queue pair.
 */
struct ibv_qp* ibv_create_qp(struct ibv_pd* pd, struct ibv_qp_init_attr* qp_init_attr)
{
    static int qpn = 0;
    struct ibv_qp* qp;

    qp = malloc(sizeof(struct ibv_qp));
    if (NULL == qp) {
        return NULL;
    }
    memset(qp, 0, sizeof(struct ibv_qp));

    qp->pd = pd;
    qp->state = IBV_QPS_RESET;
    qp->qp_type = qp_init_attr->qp_type;
    qp->send_cq = qp_init_attr->send_cq;
    qp->recv_cq = qp_init_attr->recv_cq;
    qp->context = pd->context;

    qp->qp_num = qpn;
    qpn++;

    return qp;
}

struct ibv_qp* ibv_exp_create_qp(
    struct ibv_pd* pd, struct ibv_exp_qp_init_attr* qp_init_attr, struct rdma_lite_device_qp_attr* qp_resp)
{
    return ibv_create_qp(pd, &(qp_init_attr->attr));
}

int ibv_exp_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct wr_exp_rsp* exp_rsp)
{
    return ibv_post_send(qp, wr, bad_wr);
}

struct ibv_mr*
ibv_exp_reg_mr(struct ibv_pd* pd, void* addr, size_t length, int access, struct roce_process_sign roce_sign)
{
    struct ibv_mr* mr;

    mr = malloc(sizeof(struct ibv_mr));
    if (NULL == mr) {
        return NULL;
    }
    memset(mr, 0, sizeof(struct ibv_mr));

    mr->addr = addr;
    mr->length = length;
    mr->lkey = 0xa2a2;
    mr->rkey = 0x2a2a;
    mr->pd = pd;
    mr->context = pd->context;

    return mr;
}

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
int ibv_query_qp(struct ibv_qp* qp, struct ibv_qp_attr* attr, int attr_mask, struct ibv_qp_init_attr* init_attr)
{
    return 0;
}

/**
 * ibv_destroy_qp - Destroy a queue pair.
 */
int ibv_destroy_qp(struct ibv_qp* qp)
{
    free(qp);

    return 0;
}

/**
 * ibv_dealloc_pd - Free a protection domain
 */
int ibv_dealloc_pd(struct ibv_pd* pd)
{
    free(pd);

    return 0;
}

int rs_post_send_stub(struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr) { return 0; }

int rs_post_recv_stub(struct ibv_qp* qp, struct ibv_recv_wr* wr, struct ibv_recv_wr** bad_wr) { return 0; }

int ibv_close_device(struct ibv_context* context)
{
    if (context && context->device)
        free(context->device);

    if (context)
        free(context);

    return 0;
}

struct ibv_context* ibv_open_device(struct ibv_device* device)
{
    static int num = 0;
    struct ibv_context* ctx;
    struct ibv_device* dev;

    ctx = malloc(sizeof(struct ibv_context));
    if (NULL == ctx) {
        return NULL;
    }
    memset(ctx, 0, sizeof(struct ibv_context));

    dev = malloc(sizeof(struct ibv_device));
    if (NULL == dev) {
        return NULL;
    }
    memset(dev, 0, sizeof(struct ibv_device));

    ctx->device = dev;
    ctx->ops.post_send = rs_post_send_stub;
    ctx->ops.post_recv = rs_post_recv_stub;

    return ctx;
}

static struct ibv_device* tc_dev[2] = {(void*)0x123, (void*)0x456};

struct ibv_device** ibv_get_device_list(int* num_devices)
{
    *num_devices = 2;

    return tc_dev;
}

void ibv_free_device_list(struct ibv_device** list) { return; }

/**
 * ibv_dereg_mr - Deregister a memory region
 */
int ibv_dereg_mr(struct ibv_mr* mr)
{
    free(mr);

    return 0;
}

int ibv_query_device(struct ibv_context* context, struct ibv_device_attr* device_attr) { return 0; }

#undef ibv_query_port
/**
 * ibv_query_port - Get port properties
 */
int ibv_query_port(struct ibv_context* context, uint8_t port_num, struct ibv_port_attr* port_attr)
{
    port_attr->gid_tbl_len = 3;
    port_attr->state = IBV_PORT_ACTIVE;
    return 0;
}

struct ibv_comp_channel* ibv_create_comp_channel(struct ibv_context* context)
{
    struct ibv_comp_channel* comp_channel;

    if (!context)
        return NULL;

    comp_channel = malloc(sizeof(struct ibv_comp_channel));
    if (NULL == comp_channel) {
        return NULL;
    }
    memset(comp_channel, 0, sizeof(struct ibv_comp_channel));
#define ERR_FD 1000
    comp_channel->fd = ERR_FD;

    return comp_channel;
}
int ibv_destroy_comp_channel(struct ibv_comp_channel* channel)
{
    if (channel) {
        free(channel);
    }
    return 0;
}

extern struct RsQpCb* qpCbAb2;
int ibv_get_cq_event(struct ibv_comp_channel* channel, struct ibv_cq** cq, void** cq_context)
{
    if (!channel)
        return -1;

    *cq = qpCbAb2->ibSendCq;
    return 0;
}

void ibv_ack_cq_events(struct ibv_cq* cq, unsigned int nevents) {}

int ibv_poll_cq(struct ibv_cq* cq, int num_entries, struct ibv_wc* wc)
{
    wc[0].status = IBV_WC_SUCCESS;
    wc[1].status = IBV_WC_TM_ERR;
    wc[0].wr_id = 0;
    wc[1].wr_id = 0;

    return 2;
}

int ibv_req_notify_cq(struct ibv_cq* cq, int solicited_only)
{
    if (!cq)
        return -1;
    return 0;
}

extern char* sTmp;
const char* ibv_wc_status_str(enum ibv_wc_status status) { return sTmp; }

const char* ibv_get_device_name(struct ibv_device* device) { return "hns_0"; };

const char* ibv_get_device_name_stub2(struct ibv_device* device) { return "hrn0_0"; };

int ibv_exp_query_notify(struct ibv_context* context, unsigned long long* notify_va, unsigned long long* size)
{
    return 0;
}

int ibv_ext_post_send(
    struct ibv_qp* qp, struct ibv_send_wr* wr, struct ibv_send_wr** bad_wr, struct ibv_post_send_ext_attr* ext_attr,
    struct ibv_post_send_ext_resp* ext_resp)
{
    return 0;
}

struct ibv_cq* ibv_create_ext_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector,
    int partid)
{
    return 0;
}

struct ibv_srq* ibv_create_srq(struct ibv_pd* pd, struct ibv_srq_init_attr* srq_init_attr)
{
    struct ibv_srq* srq;

    srq = malloc(sizeof(struct ibv_srq));
    if (NULL == srq) {
        return NULL;
    }
    memset(srq, 0, sizeof(struct ibv_srq));

    srq->context = pd->context;
    srq->srq_context = srq;

    return srq;
}

int ibv_destroy_srq(struct ibv_srq* srq)
{
    free(srq);
    return 0;
}

struct ibv_cq* ibv_exp_create_cq(
    struct ibv_context* context, int cqe, void* cq_context, struct ibv_comp_channel* channel, int comp_vector,
    struct rdma_lite_device_cq_init_attr* attr, struct rdma_lite_device_cq_attr* cq_resp)
{
    struct ibv_cq* cq = malloc(sizeof(struct ibv_cq));
    if (NULL == cq) {
        return NULL;
    }

    return cq;
}

int ibv_exp_query_device(struct ibv_context* context, struct dev_cap_info* cap) { return 0; }

int ibv_exp_set_dev_id(struct ibv_context* context, unsigned int dev_id) { return 0; }

struct ibv_ah* ibv_create_ah(struct ibv_pd* pd, struct ibv_ah_attr* attr)
{
    struct ibv_ah* ah;

    if (pd == NULL || attr == NULL) {
        return NULL;
    }

    ah = malloc(sizeof(struct ibv_ah));
    if (NULL == ah) {
        return NULL;
    }
    memset(ah, 0, sizeof(struct ibv_ah));

    ah->pd = pd;
    ah->context = pd->context;

    return ah;
}

int ibv_destroy_ah(struct ibv_ah* ah)
{
    if (ah == NULL) {
        return -1;
    }

    free(ah);
    ah = NULL;

    return 0;
}

#endif
