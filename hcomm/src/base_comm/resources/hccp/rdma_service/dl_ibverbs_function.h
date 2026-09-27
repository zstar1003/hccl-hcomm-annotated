/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DL_IBVERBS_FUNCTION_H
#define DL_IBVERBS_FUNCTION_H

#include <ccan/list.h>
#include <infiniband/driver.h>
#include <infiniband/verbs.h>
#include "hccp_dl.h"
#include "hns_roce_u_cmd.h"
#include "verbs_exp.h"
#include "rdma_lite_common.h"

enum SoType {
    SO_TYPE_EXP,
    SO_TYPE_EXT,
    SO_TYPE_INVALID,
};

struct RsIbverbsOps {
    void (*rsIbvFreeDeviceList)(struct ibv_device **list);
    void (*rsIbvAckCqEvents)(struct ibv_cq *cq, unsigned int nevents);
    const char *(*rsIbvGetDeviceName)(struct ibv_device *device);
    const char *(*rsIbvWcStatusStr)(enum ibv_wc_status status);
    int (*rsIbvQueryGidType)(struct ibv_context *context, uint8_t portNum, unsigned int index,
        enum ibv_gid_type_sysfs *type);
    int (*rsIbvDeregMr)(struct ibv_mr *mr);
    int (*rsIbvQueryQp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask, struct ibv_qp_init_attr *initAttr);
    int (*rsIbvDestroyQp)(struct ibv_qp *qp);
    int (*rsIbvGetCqEvent)(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cqContext);
    int (*rsIbvDestroyCq)(struct ibv_cq *cq);
    int (*rsIbvModifyQp)(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask);
    int (*rsIbvQueryDevice)(struct ibv_context *context, struct ibv_device_attr *deviceAttr);
    int (*rsIbvQueryPort)(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr *portAttr);
    int (*rsIbvQueryGid)(struct ibv_context *context, uint8_t portNum, int index, union ibv_gid *gid);
    int (*rsIbvCloseDevice)(struct ibv_context *context);
    int (*rsIbvDeallocPd)(struct ibv_pd *pd);
    int (*rsIbvDestroyCompChannel)(struct ibv_comp_channel *channel);
    struct ibv_context *(*rsIbvOpenDevice)(struct ibv_device *device);
    struct ibv_pd *(*rsIbvAllocPd)(struct ibv_context *context);
    struct ibv_device **(*rsIbvGetDeviceList)(int *numDevices);
    struct ibv_cq *(*rsIbvCreateCq)(struct ibv_context *context, int cqe, void *cqContext,
        struct ibv_comp_channel *channel, int compVector);
    struct ibv_mr *(*rsIbvRegMr)(struct ibv_pd *pd, void *addr, size_t length, int access);
    struct ibv_qp *(*rsIbvCreateQp)(struct ibv_pd *pd, struct ibv_qp_init_attr *qpInitAttr);
    struct ibv_comp_channel *(*rsIbvCreateCompChannel)(struct ibv_context *context);
    struct ibv_srq *(*rsIbvCreateSrq)(struct ibv_pd *pd, struct ibv_srq_init_attr *srqInitAttr);
    int (*rsIbvDestroySrq)(struct ibv_srq *);
    struct ibv_ah *(*rsIbvCreateAh)(struct ibv_pd *pd, struct ibv_ah_attr *attr);
    int (*rsIbvDestroyAh)(struct ibv_ah *ah);
};

struct RsRoceUserOps {
    int (*rsRoceSetTsqpDepth)(const char *devName, unsigned int rdevIndex, unsigned int tempDepth, unsigned int *qpNum,
        unsigned int *sqDepth);
    int (*rsRoceGetTsqpDepth)(const char *devName, unsigned int rdevIndex, unsigned int *tempDepth, unsigned int *qpNum,
        unsigned int *sqDepth);
    int (*rsRoceGetRoceDevData)(const char *devName, struct roce_dev_data *rdevData);
    int (*rsIbvExpQueryNotify)(struct ibv_context *context, unsigned long long *notifyVa, unsigned long long *size);
    int (*rsIbvExpPostSend)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
        struct wr_exp_rsp *expRsp);
    int (*rsIbvExtPostSend)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
        struct ibv_post_send_ext_attr *extAttr, struct ibv_post_send_ext_resp *extResp);
    struct ibv_mr *(*rsIbvExpRegMr)(struct ibv_pd *pd, void *addr, size_t length, int access,
        struct roce_process_sign roceSign);
    struct ibv_qp *(*rsIbvExpCreateQp)(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *qpInitAttr,
        struct rdma_lite_device_qp_attr *qpResp);
    struct ibv_cq *(*rsIbvExpCreateCq)(struct ibv_context *context, int cqe, void *cqContext,
        struct ibv_comp_channel *channel, int compVector, struct rdma_lite_device_cq_init_attr *attr,
        struct rdma_lite_device_cq_attr *cqResp);
    int (*rsIbvExpQueryDevice)(struct ibv_context *context, struct dev_cap_info *cap);
    int (*rsRoceInitMemPool)(const struct roce_mem_cq_qp_attr *memAttr, struct rdma_lite_device_mem_attr *memData,
        unsigned int devId);
    int (*rsRoceDeinitMemPool)(unsigned int memIdx);
    int (*rsRoceQueryQpc)(struct ibv_qp *qp, struct hns_roce_qpc_attr_val *attrVal, unsigned int attrMask);
    struct ibv_ah *(*rsIbvExpCreateAh)(struct ibv_pd *pd, struct ibv_exp_ah_attr *attrx);
    int (*rsRoceMmapAiDbReg)(struct ibv_context *context, unsigned int tgid);
    int (*rsRoceUnmmapAiDbReg)(struct ibv_context *context);
    int (*rsRoceGetCqDataPlaneInfo)(struct ibv_cq *cq, struct hns_roce_cq_data_plane_info *info);
    int (*rsRoceGetQpDataPlaneInfo)(struct ibv_qp *qp, struct hns_roce_qp_data_plane_info *info);
    int (*rsRoceRemapMr)(struct ibv_mr *mr, struct hns_roce_mr_remap_info info[], unsigned int num);
    unsigned int (*rsRoceGetApiVersion)(void);
};

struct RsHrnOps {
    int (*rsRoceSetQpLbValue)(struct ibv_qp *qp, int lbValue);
    int (*rsRoceGetQpLbValue)(struct ibv_qp *qp, int *lbValue);
    int (*rsRoceGetQpNum)(struct ibv_context *context, int *qpNum);
};

struct ibv_mr *RsIbvExpRegMr(struct ibv_pd *pd, void *addr, size_t length, int access,
    struct roce_process_sign roceSign);
int RsIbvExpQueryNotify(struct ibv_context *context, unsigned long long *notifyVa, unsigned long long *size);
int RsIbvExpPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr, struct wr_exp_rsp *expRsp);
int RsIbvExtPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
    struct ibv_post_send_ext_attr *extAttr, struct ibv_post_send_ext_resp *extResp);
struct ibv_qp *RsIbvExpCreateQp(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *qpInitAttr,
    struct rdma_lite_device_qp_attr *qpResp);
int RsRoceSetTsqpDepth(const char *devName, unsigned int rdevIndex, unsigned int tempDepth, unsigned int *qpNum,
    unsigned int *sqDepth);
int RsRoceGetTsqpDepth(const char *devName, unsigned int rdevIndex, unsigned int *tempDepth, unsigned int *qpNum,
    unsigned int *sqDepth);
int RsRoceGetRoceDevData(const char *devName, struct roce_dev_data *rdevData);

DL_ATTRI_VISI_DEF void RsApiDeinit(void);
DL_ATTRI_VISI_DEF int RsApiInit(void);
void RsIbvFreeDeviceList(struct ibv_device **list);
void RsIbvAckCqEvents(struct ibv_cq *cq, unsigned int nevents);
const char *RsIbvGetDeviceName(struct ibv_device *device);
const char *RsIbvWcStatusStr(enum ibv_wc_status status);
int RsIbvQueryGidType(struct ibv_context *context, uint8_t portNum, unsigned int index, enum ibv_gid_type_sysfs *type);
int RsIbvDeregMr(struct ibv_mr *mr);
int RsIbvPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr);
int RsIbvPostRecv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **badWr);
int RsIbvQueryQp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask, struct ibv_qp_init_attr *initAttr);
int RsIbvDestroyQp(struct ibv_qp *qp);
int RsIbvGetCqEvent(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cqContext);
int RsIbvPollCq(struct ibv_cq *cq, int numEntries, struct ibv_wc *wc);
int RsIbvReqNotifyCq(struct ibv_cq *cq, int solicitedOnly);
int RsIbvDestroyCq(struct ibv_cq *cq);
int RsIbvModifyQp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask);
int RsIbvQueryDevice(struct ibv_context *context, struct ibv_device_attr *deviceAttr);
int RsIbvQueryPort(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr *portAttr);
int RsIbvQueryGid(struct ibv_context *context, uint8_t portNum, int index, union ibv_gid *gid);
int RsIbvCloseDevice(struct ibv_context *context);
int RsIbvDeallocPd(struct ibv_pd *pd);
int RsIbvDestroyCompChannel(struct ibv_comp_channel *channel);
struct ibv_context *RsIbvOpenDevice(struct ibv_device *device);
struct ibv_pd *RsIbvAllocPd(struct ibv_context *context);
struct ibv_device **RsIbvGetDeviceList(int *numDevices);
struct ibv_cq *RsIbvCreateCq(struct ibv_context *context, int cqe, void *cqContext, struct ibv_comp_channel *channel,
    int compVector);
struct ibv_mr *RsIbvRegMr(struct ibv_pd *pd, void *addr, size_t length, int access);
struct ibv_qp *RsIbvCreateQp(struct ibv_pd *pd, struct ibv_qp_init_attr *qpInitAttr);
struct ibv_comp_channel *RsIbvCreateCompChannel(struct ibv_context *context);
struct ibv_srq *RsIbvCreateSrq(struct ibv_pd *pd, struct ibv_srq_init_attr *srqInitAttr);
int RsIbvDestroySrq(struct ibv_srq *srq);
struct ibv_ah *RsIbvCreateAh(struct ibv_pd *pd, struct ibv_ah_attr *attr);
int RsIbvDestroyAh(struct ibv_ah *ah);
struct ibv_cq *RsIbvExpCreateCq(struct ibv_context *context, int cqe, void *cqContext, struct ibv_comp_channel *channel,
    int compVector, struct rdma_lite_device_cq_init_attr *attr, struct rdma_lite_device_cq_attr *cqResp);
int RsIbvExpQueryDevice(struct ibv_context *context, struct dev_cap_info *cap);
int RsRoceInitMemPool(const struct roce_mem_cq_qp_attr *memAttr, struct rdma_lite_device_mem_attr *memData,
    unsigned int devId);
int RsRoceDeinitMemPool(unsigned int memIdx);
int RsRoceQueryQpc(struct ibv_qp *qp, struct hns_roce_qpc_attr_val *attrVal, unsigned int attrMask);
struct ibv_ah *RsIbvExpCreateAh(struct ibv_pd *pd, struct ibv_exp_ah_attr *attrx);
int RsRoceMmapAiDbReg(struct ibv_context *context, unsigned int tgid);
int RsRoceUnmmapAiDbReg(struct ibv_context *context);
int RsRoceGetCqDataPlaneInfo(struct ibv_cq *cq, struct hns_roce_cq_data_plane_info *info);
int RsRoceGetQpDataPlaneInfo(struct ibv_qp *qp, struct hns_roce_qp_data_plane_info *info);
int RsRoceRemapMr(struct ibv_mr *mr, struct hns_roce_mr_remap_info info[], unsigned int num);
unsigned int RsRoceGetApiVersion(void);
int RsRoceSetQpLbValue(struct ibv_qp *qp, int lbValue);
int RsRoceGetQpLbValue(struct ibv_qp *qp, int *lbValue);
int RsRoceGetQpNum(struct ibv_context *context, int *qpNum);
#endif // DL_IBVERBS_FUNCTION_H
