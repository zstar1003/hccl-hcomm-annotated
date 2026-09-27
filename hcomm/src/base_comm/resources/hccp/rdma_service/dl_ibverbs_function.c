/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "config_log.h"
#include <pthread.h>
#include "ra_rs_err.h"
#include "dl_ibv_extend_function.h"
#include "dl_ibverbs_function.h"

static pthread_mutex_t gRoceUserApiLock = PTHREAD_MUTEX_INITIALIZER;
static int gRoceUserApiRefcnt = 0;
void *gIbverbsApiHandle = NULL;
void *gRoceUserApiHandle = NULL;
void *gHrnApiHandle = NULL;
#ifndef CA_CONFIG_LLT
struct RsIbverbsOps gIbverbsOps;
struct RsRoceUserOps gRoceUserOps;
struct RsHrnOps gHrnOps;
#else
struct RsIbverbsOps gIbverbsOps = {
    .rsIbvFreeDeviceList = ibv_free_device_list,
    .rsIbvAckCqEvents = ibv_ack_cq_events,
    .rsIbvGetDeviceName = ibv_get_device_name,
    .rsIbvWcStatusStr = ibv_wc_status_str,
    .rsIbvQueryGidType = ibv_query_gid_type,
    .rsIbvDeregMr = ibv_dereg_mr,
    .rsIbvQueryQp = ibv_query_qp,
    .rsIbvDestroyQp = ibv_destroy_qp,
    .rsIbvGetCqEvent = ibv_get_cq_event,
    .rsIbvDestroyCq = ibv_destroy_cq,
    .rsIbvModifyQp = ibv_modify_qp,
    .rsIbvQueryDevice = ibv_query_device,
    .rsIbvQueryPort = ibv_query_port,
    .rsIbvQueryGid = ibv_query_gid,
    .rsIbvCloseDevice = ibv_close_device,
    .rsIbvDeallocPd = ibv_dealloc_pd,
    .rsIbvDestroyCompChannel = ibv_destroy_comp_channel,
    .rsIbvOpenDevice = ibv_open_device,
    .rsIbvAllocPd = ibv_alloc_pd,
    .rsIbvGetDeviceList = ibv_get_device_list,
    .rsIbvCreateCq = ibv_create_cq,
    .rsIbvRegMr = ibv_reg_mr,
    .rsIbvCreateQp = ibv_create_qp,
    .rsIbvCreateCompChannel = ibv_create_comp_channel,
    .rsIbvCreateSrq = ibv_create_srq,
    .rsIbvDestroySrq = ibv_destroy_srq,
    .rsIbvCreateAh = ibv_create_ah,
    .rsIbvDestroyAh = ibv_destroy_ah,
};

struct RsRoceUserOps gRoceUserOps = {
    .rsRoceSetTsqpDepth = roce_set_tsqp_depth,
    .rsRoceGetTsqpDepth = roce_get_tsqp_depth,
    .rsRoceGetRoceDevData = roce_get_roce_dev_data,
    .rsIbvExpQueryNotify = ibv_exp_query_notify,
    .rsIbvExpPostSend = ibv_exp_post_send,
    .rsIbvExpRegMr = ibv_exp_reg_mr,
    .rsIbvExpCreateQp = ibv_exp_create_qp,
    .rsIbvExtPostSend = ibv_ext_post_send,
    .rsIbvExpCreateCq = ibv_exp_create_cq,
    .rsIbvExpQueryDevice = ibv_exp_query_device,
    .rsRoceInitMemPool = roce_init_mem_pool,
    .rsRoceDeinitMemPool = roce_deinit_mem_pool,
    .rsRoceQueryQpc = roce_query_qpc,
    .rsIbvExpCreateAh = ibv_exp_create_ah,
    .rsRoceMmapAiDbReg = roce_mmap_ai_db_reg,
    .rsRoceUnmmapAiDbReg = roce_unmmap_ai_db_reg,
    .rsRoceGetCqDataPlaneInfo = roce_get_cq_data_plane_info,
    .rsRoceGetQpDataPlaneInfo = roce_get_qp_data_plane_info,
    .rsRoceRemapMr = roce_remap_mr,
    .rsRoceGetApiVersion = roce_get_api_version,
};

struct RsHrnOps gHrnOps = {
    .rsRoceSetQpLbValue = roce_set_qp_lb_value,
    .rsRoceGetQpLbValue = roce_get_qp_lb_value,
    .rsRoceGetQpNum = roce_get_qp_num,
};
#endif

STATIC int RsContextOpsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gIbverbsOps.rsIbvQueryDevice = (int (*)(struct ibv_context *, struct ibv_device_attr *))HccpDlsym(gIbverbsApiHandle,
        "ibv_query_device");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvQueryDevice, "ibv_query_device");

    gIbverbsOps.rsIbvQueryPort = (int (*)(struct ibv_context *, uint8_t,
        struct ibv_port_attr *))HccpDlsym(gIbverbsApiHandle, "ibv_query_port");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvQueryPort, "ibv_query_port");

    gIbverbsOps.rsIbvQueryGid = (int (*)(struct ibv_context *, uint8_t, int,
        union ibv_gid *))HccpDlsym(gIbverbsApiHandle, "ibv_query_gid");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvQueryGid, "ibv_query_gid");

    gIbverbsOps.rsIbvAllocPd = (struct ibv_pd * (*)(struct ibv_context *)) HccpDlsym(gIbverbsApiHandle, "ibv_alloc_pd");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvAllocPd, "ibv_alloc_pd");

    gIbverbsOps.rsIbvDeallocPd = (int (*)(struct ibv_pd *))HccpDlsym(gIbverbsApiHandle, "ibv_dealloc_pd");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDeallocPd, "ibv_dealloc_pd");

    gIbverbsOps.rsIbvCreateCq = (struct ibv_cq * (*)(struct ibv_context *, int, void *, struct ibv_comp_channel *, int))
        HccpDlsym(gIbverbsApiHandle, "ibv_create_cq");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCreateCq, "ibv_create_cq");

    gIbverbsOps.rsIbvDestroyCq = (int (*)(struct ibv_cq *))HccpDlsym(gIbverbsApiHandle, "ibv_destroy_cq");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDestroyCq, "ibv_destroy_cq");

    gIbverbsOps.rsIbvCreateCompChannel = (struct ibv_comp_channel * (*)(struct ibv_context *))
        HccpDlsym(gIbverbsApiHandle, "ibv_create_comp_channel");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCreateCompChannel, "ibv_create_comp_channel");

    gIbverbsOps.rsIbvDestroyCompChannel = (int (*)(struct ibv_comp_channel *))HccpDlsym(gIbverbsApiHandle,
        "ibv_destroy_comp_channel");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDestroyCompChannel, "ibv_destroy_comp_channel");

    gIbverbsOps.rsIbvCreateSrq = (struct ibv_srq * (*)(struct ibv_pd * pd, struct ibv_srq_init_attr * srqInitAttr))
        HccpDlsym(gIbverbsApiHandle, "ibv_create_srq");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCreateSrq, "ibv_create_srq");

    gIbverbsOps.rsIbvDestroySrq = (int (*)(struct ibv_srq *))HccpDlsym(gIbverbsApiHandle, "ibv_destroy_srq");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDestroySrq, "ibv_destroy_srq");

    gIbverbsOps.rsIbvQueryGidType = (int (*)(struct ibv_context *, uint8_t, unsigned int,
        enum ibv_gid_type_sysfs *))HccpDlsym(gIbverbsApiHandle, "ibv_query_gid_type");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvQueryGidType, "ibv_query_gid_type");

    gIbverbsOps.rsIbvCreateAh = (struct ibv_ah * (*)(struct ibv_pd *, struct ibv_ah_attr *))
        HccpDlsym(gIbverbsApiHandle, "ibv_create_ah");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCreateAh, "ibv_create_ah");

    gIbverbsOps.rsIbvDestroyAh = (int (*)(struct ibv_ah *))HccpDlsym(gIbverbsApiHandle, "ibv_destroy_ah");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDestroyAh, "ibv_destroy_ah");
#endif
    return 0;
}

STATIC int RsQpOpsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gIbverbsOps.rsIbvQueryQp = (int (*)(struct ibv_qp *, struct ibv_qp_attr *, int,
        struct ibv_qp_init_attr *))HccpDlsym(gIbverbsApiHandle, "ibv_query_qp");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvQueryQp, "ibv_query_qp");

    gIbverbsOps.rsIbvGetCqEvent = (int (*)(struct ibv_comp_channel *, struct ibv_cq **,
        void **))HccpDlsym(gIbverbsApiHandle, "ibv_get_cq_event");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvGetCqEvent, "ibv_get_cq_event");

    gIbverbsOps.rsIbvAckCqEvents = (void (*)(struct ibv_cq *, unsigned int))HccpDlsym(gIbverbsApiHandle,
        "ibv_ack_cq_events");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvAckCqEvents, "ibv_ack_cq_events");

    gIbverbsOps.rsIbvModifyQp = (int (*)(struct ibv_qp *, struct ibv_qp_attr *, int))HccpDlsym(gIbverbsApiHandle,
        "ibv_modify_qp");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvModifyQp, "ibv_modify_qp");
#endif
    return 0;
}

STATIC int RsPdOpsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gIbverbsOps.rsIbvRegMr = (struct ibv_mr * (*)(struct ibv_pd *, void *, size_t, int))
        HccpDlsym(gIbverbsApiHandle, "ibv_reg_mr");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvRegMr, "ibv_reg_mr");

    gIbverbsOps.rsIbvDeregMr = (int (*)(struct ibv_mr *))HccpDlsym(gIbverbsApiHandle, "ibv_dereg_mr");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDeregMr, "ibv_dereg_mr");

    gIbverbsOps.rsIbvCreateQp = (struct ibv_qp * (*)(struct ibv_pd *, struct ibv_qp_init_attr *))
        HccpDlsym(gIbverbsApiHandle, "ibv_create_qp");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCreateQp, "ibv_create_qp");

    gIbverbsOps.rsIbvDestroyQp = (int (*)(struct ibv_qp *))HccpDlsym(gIbverbsApiHandle, "ibv_destroy_qp");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvDestroyQp, "ibv_destroy_qp");
#endif
    return 0;
}

STATIC int RsDeviceOpsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gIbverbsOps.rsIbvGetDeviceList = (struct ibv_device * *(*)(int *))
        HccpDlsym(gIbverbsApiHandle, "ibv_get_device_list");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvGetDeviceList, "ibv_get_device_list");

    gIbverbsOps.rsIbvFreeDeviceList = (void (*)(struct ibv_device **))HccpDlsym(gIbverbsApiHandle,
        "ibv_free_device_list");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvFreeDeviceList, "ibv_free_device_list");

    gIbverbsOps.rsIbvGetDeviceName = (const char *(*)(struct ibv_device *))HccpDlsym(gIbverbsApiHandle,
        "ibv_get_device_name");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvGetDeviceName, "ibv_get_device_name");

    gIbverbsOps.rsIbvCloseDevice = (int (*)(struct ibv_context *))HccpDlsym(gIbverbsApiHandle, "ibv_close_device");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvCloseDevice, "ibv_close_device");

    gIbverbsOps.rsIbvOpenDevice = (struct ibv_context * (*)(struct ibv_device *))
        HccpDlsym(gIbverbsApiHandle, "ibv_open_device");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvOpenDevice, "ibv_open_device");

    gIbverbsOps.rsIbvWcStatusStr = (const char *(*)(enum ibv_wc_status))HccpDlsym(gIbverbsApiHandle,
        "ibv_wc_status_str");
    DL_API_RET_IS_NULL_CHECK(gIbverbsOps.rsIbvWcStatusStr, "ibv_wc_status_str");
#endif
    return 0;
}

STATIC int RsOpenIbverbsSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gIbverbsApiHandle == NULL) {
        gIbverbsApiHandle = HccpDlopen("libibverbs.so", RTLD_NOW);
        if (gIbverbsApiHandle != NULL) {
            return 0;
        }

        gIbverbsApiHandle = HccpDlopen("libibverbs.so.1", RTLD_NOW);
        if (gIbverbsApiHandle != 0) {
            return 0;
        }
        return -EINVAL;
    } else {
        hccp_run_info("ibverbs_api dlopen again!");
    }
#endif
    return 0;
}

STATIC void RsCloseIbverbsSo(void)
{
    if (gIbverbsApiHandle != NULL) {
        (void)HccpDlclose(gIbverbsApiHandle);
        gIbverbsApiHandle = NULL;
    }
    return;
}

STATIC int RsIbverbsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    int ret;

    ret = RsOpenIbverbsSo();
    CHK_PRT_RETURN(ret,
        hccp_err("HccpDlopen[libibverbs.so or libibverbs.so.1] failed! ret=[%d], "
                 "Please check network adapter driver has been installed",
            ret),
        ret);

    ret = RsContextOpsApiInit();
    if (ret) {
        hccp_err("[rs_context_ops_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseIbverbsSo();
        return ret;
    }

    ret = RsQpOpsApiInit();
    if (ret) {
        hccp_err("[rs_qp_ops_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseIbverbsSo();
        return ret;
    }

    ret = RsPdOpsApiInit();
    if (ret) {
        hccp_err("[rs_pd_ops_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseIbverbsSo();
        return ret;
    }

    ret = RsDeviceOpsApiInit();
    if (ret) {
        hccp_err("[rs_device_ops_api_init]HccpDlopen failed! ret=[%d]", ret);
        RsCloseIbverbsSo();
        return ret;
    }
#endif
    return 0;
}

#ifdef CUSTOM_INTERFACE
STATIC int RsOpenRoceUserSo(enum SoType *type)
{
    pthread_mutex_lock(&gRoceUserApiLock);
#ifndef CA_CONFIG_LLT
    if (gRoceUserApiHandle == NULL) {
        gRoceUserApiHandle = HccpDlopen("libhns-rdmav17.so", RTLD_NOW);
        if (gRoceUserApiHandle != NULL) {
            *type = SO_TYPE_EXP;
            goto out;
        }

        gRoceUserApiHandle = HccpDlopen("libhns-rdmav25.so", RTLD_NOW);
        if (gRoceUserApiHandle != NULL) {
            *type = SO_TYPE_EXT;
            goto out;
        }

        gRoceUserApiHandle = HccpDlopen("libhrn0-rdmav17.so", RTLD_NOW);
        if (gRoceUserApiHandle != NULL) {
            *type = SO_TYPE_EXP;
            goto out;
        }

        pthread_mutex_unlock(&gRoceUserApiLock);
        return -EINVAL;
    } else {
        hccp_run_info("dlopen roce_user_api again, gRoceUserApiRefcnt:%d", gRoceUserApiRefcnt + 1);
    }

out:
#endif
    gRoceUserApiRefcnt++;
    pthread_mutex_unlock(&gRoceUserApiLock);
    return 0;
}
#endif

STATIC void RsCloseRoceUserSo(void)
{
    pthread_mutex_lock(&gRoceUserApiLock);
#ifndef CA_CONFIG_LLT
    if (gRoceUserApiHandle != NULL) {
        gRoceUserApiRefcnt--;
        if (gRoceUserApiRefcnt > 0) {
            goto out;
        }

        hccp_run_info("dlclose roce_user_api, gRoceUserApiRefcnt:%d", gRoceUserApiRefcnt);
        (void)HccpDlclose(gRoceUserApiHandle);
        gRoceUserApiHandle = NULL;
        gRoceUserApiRefcnt = 0;
    }
out:
#endif
    pthread_mutex_unlock(&gRoceUserApiLock);
    return;
}

#ifdef CUSTOM_INTERFACE
STATIC int RsRoceUserIbvApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gRoceUserOps.rsIbvExpCreateQp = (struct ibv_qp * (*)(struct ibv_pd * pd, struct ibv_exp_qp_init_attr * qpInitAttr,
                                                         struct rdma_lite_device_qp_attr * qpResp))
        HccpDlsym(gRoceUserApiHandle, "ibv_exp_create_qp");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpCreateQp, "ibv_exp_create_qp");
    gRoceUserOps.rsIbvExpRegMr = (struct ibv_mr * (*)(struct ibv_pd * pd, void *addr, size_t length, int access,
                                                      struct roce_process_sign roceSign))
        HccpDlsym(gRoceUserApiHandle, "ibv_exp_reg_mr");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpRegMr, "ibv_exp_reg_mr");
    gRoceUserOps.rsIbvExpQueryNotify = (int (*)(struct ibv_context *context, unsigned long long *notifyVa,
        unsigned long long *size))HccpDlsym(gRoceUserApiHandle, "ibv_exp_query_notify");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpQueryNotify, "ibv_exp_query_notify");
    gRoceUserOps.rsIbvExpPostSend = (int (*)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
        struct wr_exp_rsp *expRsp))HccpDlsym(gRoceUserApiHandle, "ibv_exp_post_send");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpPostSend, "ibv_exp_post_send");
    gRoceUserOps.rsIbvExpCreateCq = (struct ibv_cq * (*)(struct ibv_context *, int, void *, struct ibv_comp_channel *,
                                                         int, struct rdma_lite_device_cq_init_attr *,
                                                         struct rdma_lite_device_cq_attr *))
        HccpDlsym(gRoceUserApiHandle, "ibv_exp_create_cq");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpCreateCq, "ibv_exp_create_cq");
    gRoceUserOps.rsIbvExpQueryDevice = (int (*)(struct ibv_context *,
        struct dev_cap_info *))HccpDlsym(gRoceUserApiHandle, "ibv_exp_query_device");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpQueryDevice, "ibv_exp_query_device");
    gRoceUserOps.rsIbvExpCreateAh = (struct ibv_ah * (*)(struct ibv_pd * pd, struct ibv_exp_ah_attr * attrx))
        HccpDlsym(gRoceUserApiHandle, "ibv_exp_create_ah");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsIbvExpCreateAh, "ibv_exp_create_ah");
#endif
    return 0;
}
#endif

#ifdef CUSTOM_INTERFACE
STATIC int RsRoceUserDrvApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gRoceUserOps.rsRoceGetRoceDevData = (int (*)(const char *devName,
        struct roce_dev_data *rdevData))HccpDlsym(gRoceUserApiHandle, "roce_get_roce_dev_data");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceGetRoceDevData, "roce_get_roce_dev_data");
    gRoceUserOps.rsRoceGetTsqpDepth = (int (*)(const char *devName, unsigned int rdevIndex, unsigned int *tempDepth,
        unsigned int *qpNum, unsigned int *sqDepth))HccpDlsym(gRoceUserApiHandle, "roce_get_tsqp_depth");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceGetTsqpDepth, "roce_get_tsqp_depth");
    gRoceUserOps.rsRoceSetTsqpDepth = (int (*)(const char *devName, unsigned int rdevIndex, unsigned int tempDepth,
        unsigned int *qpNum, unsigned int *sqDepth))HccpDlsym(gRoceUserApiHandle, "roce_set_tsqp_depth");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceSetTsqpDepth, "roce_set_tsqp_depth");
    gRoceUserOps.rsRoceInitMemPool = (int (*)(const struct roce_mem_cq_qp_attr *, struct rdma_lite_device_mem_attr *,
        unsigned int))HccpDlsym(gRoceUserApiHandle, "roce_init_mem_pool");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceInitMemPool, "roce_init_mem_pool");
    gRoceUserOps.rsRoceDeinitMemPool = (int (*)(unsigned int))HccpDlsym(gRoceUserApiHandle, "roce_deinit_mem_pool");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceDeinitMemPool, "roce_deinit_mem_pool");
    gRoceUserOps.rsRoceQueryQpc = (int (*)(struct ibv_qp *qp, struct hns_roce_qpc_attr_val *attrVal,
        unsigned int attrMask))HccpDlsym(gRoceUserApiHandle, "roce_query_qpc");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceQueryQpc, "roce_query_qpc");
    gRoceUserOps.rsRoceMmapAiDbReg = (int (*)(struct ibv_context *context,
        unsigned int tgid))HccpDlsym(gRoceUserApiHandle, "roce_mmap_ai_db_reg");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceMmapAiDbReg, "roce_mmap_ai_db_reg");
    gRoceUserOps.rsRoceUnmmapAiDbReg = (int (*)(struct ibv_context *context))HccpDlsym(gRoceUserApiHandle,
        "roce_unmmap_ai_db_reg");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceUnmmapAiDbReg, "roce_unmmap_ai_db_reg");
    gRoceUserOps.rsRoceGetCqDataPlaneInfo = (int (*)(struct ibv_cq *cq,
        struct hns_roce_cq_data_plane_info *info))HccpDlsym(gRoceUserApiHandle, "roce_get_cq_data_plane_info");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceGetCqDataPlaneInfo, "roce_get_cq_data_plane_info");
    gRoceUserOps.rsRoceGetQpDataPlaneInfo = (int (*)(struct ibv_qp *qp,
        struct hns_roce_qp_data_plane_info *info))HccpDlsym(gRoceUserApiHandle, "roce_get_qp_data_plane_info");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceGetQpDataPlaneInfo, "roce_get_qp_data_plane_info");
    gRoceUserOps.rsRoceRemapMr = (int (*)(struct ibv_mr *mr, struct hns_roce_mr_remap_info info[],
        unsigned int num))HccpDlsym(gRoceUserApiHandle, "roce_remap_mr");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceRemapMr, "roce_remap_mr");
    gRoceUserOps.rsRoceGetApiVersion = (unsigned int (*)(void))HccpDlsym(gRoceUserApiHandle, "roce_get_api_version");
    DL_API_RET_IS_NULL_CHECK(gRoceUserOps.rsRoceGetApiVersion, "roce_get_api_version");
#endif
    return 0;
}
#endif

#ifdef CUSTOM_INTERFACE
STATIC int RsRoceUserApiInit(void)
{
    enum SoType type = SO_TYPE_INVALID;
    int ret = 0;

    ret = RsOpenRoceUserSo(&type);
    CHK_PRT_RETURN(ret != 0,
        hccp_err("HccpDlopen[libhns-rdmav17.so or libhns-rdmav25.so or libhrn0-rdmav17.so] "
                 "failed! ret=[%d]. Please check network adapter driver has been installed.",
            ret),
        ret);

#ifndef CA_CONFIG_LLT
    ret = RsRoceUserIbvApiInit();
    if (ret != 0) {
        hccp_err("rs_roce_user_ibv_api_init failed! ret=[%d]", ret);
        goto close_roce_user_so;
    }

    ret = RsRoceUserDrvApiInit();
    if (ret != 0) {
        hccp_err("rs_roce_user_drv_api_init failed! ret=[%d]", ret);
        goto close_roce_user_so;
    }

    if (type == SO_TYPE_EXT) {
        gRoceUserOps.rsIbvExtPostSend = (int (*)(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
            struct ibv_post_send_ext_attr *extAttr,
            struct ibv_post_send_ext_resp *extResp))HccpDlsym(gRoceUserApiHandle, "ibv_ext_post_send");
        if (gRoceUserOps.rsIbvExtPostSend == NULL) {
            ret = -EINVAL;
            hccp_err("ibv_ext_post_send is null");
            goto close_roce_user_so;
        }
    }
    return 0;

close_roce_user_so:
    RsCloseRoceUserSo();
#endif
    return ret;
}
#endif

STATIC int RsHrnIbvApiInit(void)
{
#ifndef CA_CONFIG_LLT
    gHrnOps.rsRoceSetQpLbValue = (int (*)(struct ibv_qp *qp, int lbValue))HccpDlsym(gHrnApiHandle,
        "roce_set_qp_lb_value");
    gHrnOps.rsRoceGetQpLbValue = (int (*)(struct ibv_qp *qp, int *lbValue))HccpDlsym(gHrnApiHandle,
        "roce_get_qp_lb_value");
    gHrnOps.rsRoceGetQpNum = (int (*)(struct ibv_context *context, int *qpNum))HccpDlsym(gHrnApiHandle,
        "roce_get_qp_num");
#endif
    return 0;
}

STATIC int RsOpenHrnSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gHrnApiHandle == NULL) {
        gHrnApiHandle = HccpDlopen("libhrn5-rdmav34.so", RTLD_NOW);
        if (gHrnApiHandle != NULL) {
            return 0;
        }
        gHrnApiHandle = HccpDlopen("libhrn5.so.1", RTLD_NOW);
        if (gHrnApiHandle != NULL) {
            return 0;
        }
        return -EINVAL;
    } else {
        hccp_run_info("HrnApi dlopen again!");
    }
#endif
    return 0;
}

STATIC void RsCloseHrnSo(void)
{
#ifndef CA_CONFIG_LLT
    if (gHrnApiHandle != NULL) {
        (void)HccpDlclose(gHrnApiHandle);
        gHrnApiHandle = NULL;
    }
#endif
    return;
}

int RsHrnApiInit(void)
{
#ifndef CA_CONFIG_LLT
    int ret = 0;

    ret = RsOpenHrnSo();
    if (ret != 0) {
        hccp_warn_others("HccpDlopen[libhrn5-rdmav34.so or libhrn5.so.1] doesn't exist!");
        return 0;
    }

    ret = RsHrnIbvApiInit();
    if (ret != 0) {
        hccp_err("RsHrnIbvApiInit failed! ret:%d", ret);
        RsCloseHrnSo();
        return ret;
    }
#endif
    return 0;
}

DL_ATTRI_VISI_DEF int RsApiInit(void)
{
#ifndef CA_CONFIG_LLT
    int ret;
    ret = RsIbverbsApiInit();
    CHK_PRT_RETURN(ret, hccp_err("rs_ibverbs_api_init failed! ret=[%d]", ret), ret);
    ret = RsHrnApiInit();
    if (ret != 0) {
        hccp_err("RsHrnApiInit failed! ret=[%d]", ret);
        RsCloseIbverbsSo();
        return ret;
    }
    ret = RsIbvExtendApiInit();
    if (ret != 0) {
        hccp_err("RsIbvExtendApiInit failed! ret=[%d]", ret);
        RsCloseHrnSo();
        RsCloseIbverbsSo();
        return ret;
    }
#ifdef CUSTOM_INTERFACE
    ret = RsRoceUserApiInit();
    if (ret != 0) {
        hccp_err("rs_roce_user_api_init failed! ret=[%d]", ret);
        RsCloseHrnSo();
        RsCloseIbverbsSo();
        RsIbvExtendApiDeinit();
        return ret;
    }
#endif
#endif
    return 0;
}

DL_ATTRI_VISI_DEF void RsApiDeinit(void)
{
    RsCloseIbverbsSo();
    RsCloseHrnSo();
    RsIbvExtendApiDeinit();
    RsCloseRoceUserSo();
    return;
}

struct ibv_mr *RsIbvExpRegMr(struct ibv_pd *pd, void *addr, size_t length, int access,
    struct roce_process_sign roceSign)
{
    if (gRoceUserOps.rsIbvExpRegMr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_reg_mr is null");
        return NULL;
#endif
    }
    return gRoceUserOps.rsIbvExpRegMr(pd, addr, length, access, roceSign);
}

struct ibv_qp *RsIbvExpCreateQp(struct ibv_pd *pd, struct ibv_exp_qp_init_attr *qpInitAttr,
    struct rdma_lite_device_qp_attr *qpResp)
{
    if (gRoceUserOps.rsIbvExpCreateQp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_create_qp is null");
        return NULL;
#endif
    }
    return gRoceUserOps.rsIbvExpCreateQp(pd, qpInitAttr, qpResp);
}

int RsRoceSetTsqpDepth(const char *devName, unsigned int rdevIndex, unsigned int tempDepth, unsigned int *qpNum,
    unsigned int *sqDepth)
{
    if (gRoceUserOps.rsRoceSetTsqpDepth == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_set_tsqp_depth is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceSetTsqpDepth(devName, rdevIndex, tempDepth, qpNum, sqDepth);
}

int RsRoceGetTsqpDepth(const char *devName, unsigned int rdevIndex, unsigned int *tempDepth, unsigned int *qpNum,
    unsigned int *sqDepth)
{
    if (gRoceUserOps.rsRoceGetTsqpDepth == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_get_tsqp_depth is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceGetTsqpDepth(devName, rdevIndex, tempDepth, qpNum, sqDepth);
}

int RsRoceGetRoceDevData(const char *devName, struct roce_dev_data *rdevData)
{
    if (gRoceUserOps.rsRoceGetRoceDevData == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_get_roce_dev_data is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceGetRoceDevData(devName, rdevData);
}

int RsIbvExpQueryNotify(struct ibv_context *context, unsigned long long *notifyVa, unsigned long long *size)
{
    if (gRoceUserOps.rsIbvExpQueryNotify == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_query_notify is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsIbvExpQueryNotify(context, notifyVa, size);
}
int RsIbvExpPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr, struct wr_exp_rsp *expRsp)
{
    if (gRoceUserOps.rsIbvExpPostSend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_post_send is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsIbvExpPostSend(qp, wr, badWr, expRsp);
}

int RsIbvExtPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr,
    struct ibv_post_send_ext_attr *extAttr, struct ibv_post_send_ext_resp *extResp)
{
    if (gRoceUserOps.rsIbvExtPostSend == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_ext_post_send is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsIbvExtPostSend(qp, wr, badWr, extAttr, extResp);
}

void RsIbvFreeDeviceList(struct ibv_device **list)
{
    if (gIbverbsOps.rsIbvFreeDeviceList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_free_device_list is null");
        return;
#endif
    }
    gIbverbsOps.rsIbvFreeDeviceList(list);
}

void RsIbvAckCqEvents(struct ibv_cq *cq, unsigned int nevents)
{
    if (gIbverbsOps.rsIbvAckCqEvents == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_ack_cq_events is null");
        return;
#endif
    }
    gIbverbsOps.rsIbvAckCqEvents(cq, nevents);
}

const char *RsIbvGetDeviceName(struct ibv_device *device)
{
    if (gIbverbsOps.rsIbvGetDeviceName == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_get_device_name is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvGetDeviceName(device);
}

const char *RsIbvWcStatusStr(enum ibv_wc_status status)
{
    if (gIbverbsOps.rsIbvWcStatusStr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_wc_status_str is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvWcStatusStr(status);
}

int RsIbvQueryGidType(struct ibv_context *context, uint8_t portNum, unsigned int index, enum ibv_gid_type_sysfs *type)
{
    if (gIbverbsOps.rsIbvQueryGidType == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_query_gid_type is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvQueryGidType(context, portNum, index, type);
}

int RsIbvDeregMr(struct ibv_mr *mr)
{
    if (gIbverbsOps.rsIbvDeregMr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_dereg_mr is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDeregMr(mr);
}

int RsIbvPostSend(struct ibv_qp *qp, struct ibv_send_wr *wr, struct ibv_send_wr **badWr)
{
    return ibv_post_send(qp, wr, badWr);
}

int RsIbvPostRecv(struct ibv_qp *qp, struct ibv_recv_wr *wr, struct ibv_recv_wr **badWr)
{
    return ibv_post_recv(qp, wr, badWr);
}

int RsIbvQueryQp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask, struct ibv_qp_init_attr *initAttr)
{
    if (gIbverbsOps.rsIbvQueryQp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_query_qp is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvQueryQp(qp, attr, attrMask, initAttr);
}

int RsIbvDestroyQp(struct ibv_qp *qp)
{
    if (gIbverbsOps.rsIbvDestroyQp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_destroy_qp is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDestroyQp(qp);
}

int RsIbvGetCqEvent(struct ibv_comp_channel *channel, struct ibv_cq **cq, void **cqContext)
{
    if (gIbverbsOps.rsIbvGetCqEvent == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_get_cq_event is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvGetCqEvent(channel, cq, cqContext);
}

int RsIbvPollCq(struct ibv_cq *cq, int numEntries, struct ibv_wc *wc)
{
    return ibv_poll_cq(cq, numEntries, wc);
}

int RsIbvReqNotifyCq(struct ibv_cq *cq, int solicitedOnly)
{
    return ibv_req_notify_cq(cq, solicitedOnly);
}

int RsIbvDestroyCq(struct ibv_cq *cq)
{
    if (gIbverbsOps.rsIbvDestroyCq == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_destroy_cq is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDestroyCq(cq);
}

int RsIbvModifyQp(struct ibv_qp *qp, struct ibv_qp_attr *attr, int attrMask)
{
    if (gIbverbsOps.rsIbvModifyQp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_modify_qp is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvModifyQp(qp, attr, attrMask);
}

int RsIbvQueryDevice(struct ibv_context *context, struct ibv_device_attr *deviceAttr)
{
    if (gIbverbsOps.rsIbvQueryDevice == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_query_device is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvQueryDevice(context, deviceAttr);
}

int RsIbvQueryPort(struct ibv_context *context, uint8_t portNum, struct ibv_port_attr *portAttr)
{
    if (gIbverbsOps.rsIbvQueryPort == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_query_port is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvQueryPort(context, portNum, portAttr);
}

int RsIbvQueryGid(struct ibv_context *context, uint8_t portNum, int index, union ibv_gid *gid)
{
    if (gIbverbsOps.rsIbvQueryGid == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_query_gid is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvQueryGid(context, portNum, index, gid);
}

int RsIbvCloseDevice(struct ibv_context *context)
{
    if (gIbverbsOps.rsIbvCloseDevice == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_close_device is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvCloseDevice(context);
}

int RsIbvDeallocPd(struct ibv_pd *pd)
{
    if (gIbverbsOps.rsIbvDeallocPd == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_dealloc_pd is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDeallocPd(pd);
}

int RsIbvDestroyCompChannel(struct ibv_comp_channel *channel)
{
    if (gIbverbsOps.rsIbvDestroyCompChannel == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_destroy_comp_channel is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDestroyCompChannel(channel);
}

struct ibv_context *RsIbvOpenDevice(struct ibv_device *device)
{
    if (gIbverbsOps.rsIbvOpenDevice == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_open_device is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvOpenDevice(device);
}

struct ibv_pd *RsIbvAllocPd(struct ibv_context *context)
{
    if (gIbverbsOps.rsIbvAllocPd == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_alloc_pd is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvAllocPd(context);
}

struct ibv_device **RsIbvGetDeviceList(int *numDevices)
{
    if (gIbverbsOps.rsIbvGetDeviceList == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_get_device_list is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvGetDeviceList(numDevices);
}

struct ibv_cq *RsIbvCreateCq(struct ibv_context *context, int cqe, void *cqContext, struct ibv_comp_channel *channel,
    int compVector)
{
    if (gIbverbsOps.rsIbvCreateCq == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_create_cq is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvCreateCq(context, cqe, cqContext, channel, compVector);
}

struct ibv_mr *RsIbvRegMr(struct ibv_pd *pd, void *addr, size_t length, int access)
{
    if (gIbverbsOps.rsIbvRegMr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_reg_mr is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvRegMr(pd, addr, length, access);
}

struct ibv_qp *RsIbvCreateQp(struct ibv_pd *pd, struct ibv_qp_init_attr *qpInitAttr)
{
    if (gIbverbsOps.rsIbvCreateQp == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_create_qp is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvCreateQp(pd, qpInitAttr);
}

struct ibv_comp_channel *RsIbvCreateCompChannel(struct ibv_context *context)
{
    if (gIbverbsOps.rsIbvCreateCompChannel == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_create_comp_channel is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvCreateCompChannel(context);
}

struct ibv_srq *RsIbvCreateSrq(struct ibv_pd *pd, struct ibv_srq_init_attr *srqInitAttr)
{
    if (gIbverbsOps.rsIbvCreateSrq == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_create_srq is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvCreateSrq(pd, srqInitAttr);
}

int RsIbvDestroySrq(struct ibv_srq *srq)
{
    if (gIbverbsOps.rsIbvDestroySrq == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_destroy_srq is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDestroySrq(srq);
}

struct ibv_ah *RsIbvCreateAh(struct ibv_pd *pd, struct ibv_ah_attr *attr)
{
    if (gIbverbsOps.rsIbvCreateAh == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_create_ah is null");
        return NULL;
#endif
    }
    return gIbverbsOps.rsIbvCreateAh(pd, attr);
}

int RsIbvDestroyAh(struct ibv_ah *ah)
{
    if (gIbverbsOps.rsIbvDestroyAh == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_destroy_ah is null");
        return -EINVAL;
#endif
    }
    return gIbverbsOps.rsIbvDestroyAh(ah);
}

struct ibv_cq *RsIbvExpCreateCq(struct ibv_context *context, int cqe, void *cqContext, struct ibv_comp_channel *channel,
    int compVector, struct rdma_lite_device_cq_init_attr *attr, struct rdma_lite_device_cq_attr *cqResp)
{
    if (gRoceUserOps.rsIbvExpCreateCq == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_create_cq is null");
        return NULL;
#endif
    }
    return gRoceUserOps.rsIbvExpCreateCq(context, cqe, cqContext, channel, compVector, attr, cqResp);
}

int RsIbvExpQueryDevice(struct ibv_context *context, struct dev_cap_info *cap)
{
    if (gRoceUserOps.rsIbvExpQueryDevice == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_query_device is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsIbvExpQueryDevice(context, cap);
}

int RsRoceInitMemPool(const struct roce_mem_cq_qp_attr *memAttr, struct rdma_lite_device_mem_attr *memData,
    unsigned int devId)
{
    if (gRoceUserOps.rsRoceInitMemPool == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_init_mem_pool is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceInitMemPool(memAttr, memData, devId);
}

int RsRoceDeinitMemPool(unsigned int memIdx)
{
    if (gRoceUserOps.rsRoceDeinitMemPool == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_deinit_mem_pool is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceDeinitMemPool(memIdx);
}

int RsRoceQueryQpc(struct ibv_qp *qp, struct hns_roce_qpc_attr_val *attrVal, unsigned int attrMask)
{
    if (gRoceUserOps.rsRoceQueryQpc == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_query_qpc is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceQueryQpc(qp, attrVal, attrMask);
}

struct ibv_ah *RsIbvExpCreateAh(struct ibv_pd *pd, struct ibv_exp_ah_attr *attrx)
{
    if (gRoceUserOps.rsIbvExpCreateAh == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_ibv_exp_create_ah is null");
        return NULL;
#endif
    }
    return gRoceUserOps.rsIbvExpCreateAh(pd, attrx);
}

int RsRoceMmapAiDbReg(struct ibv_context *context, unsigned int tgid)
{
    if (gRoceUserOps.rsRoceMmapAiDbReg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_mmap_ai_db_reg is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceMmapAiDbReg(context, tgid);
}

int RsRoceUnmmapAiDbReg(struct ibv_context *context)
{
    if (gRoceUserOps.rsRoceUnmmapAiDbReg == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_unmmap_ai_db_reg is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceUnmmapAiDbReg(context);
}

int RsRoceGetCqDataPlaneInfo(struct ibv_cq *cq, struct hns_roce_cq_data_plane_info *info)
{
    if (gRoceUserOps.rsRoceGetCqDataPlaneInfo == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_get_cq_data_plane_info is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceGetCqDataPlaneInfo(cq, info);
}

int RsRoceGetQpDataPlaneInfo(struct ibv_qp *qp, struct hns_roce_qp_data_plane_info *info)
{
    if (gRoceUserOps.rsRoceGetQpDataPlaneInfo == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_get_qp_data_plane_info is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceGetQpDataPlaneInfo(qp, info);
}

int RsRoceRemapMr(struct ibv_mr *mr, struct hns_roce_mr_remap_info info[], unsigned int num)
{
    if (gRoceUserOps.rsRoceRemapMr == NULL) {
#ifndef CA_CONFIG_LLT
        hccp_err("rs_roce_remap_mr is null");
        return -EINVAL;
#endif
    }
    return gRoceUserOps.rsRoceRemapMr(mr, info, num);
}

unsigned int RsRoceGetApiVersion(void)
{
    if (gRoceUserOps.rsRoceGetApiVersion != NULL) {
        return gRoceUserOps.rsRoceGetApiVersion();
    }

    return 0;
}

int RsRoceSetQpLbValue(struct ibv_qp *qp, int lbValue)
{
    if (gHrnOps.rsRoceSetQpLbValue == NULL) {
        hccp_run_warn("rsRoceSetQpLbValue is null");
        return -ENOTSUPP;
    }
    return gHrnOps.rsRoceSetQpLbValue(qp, lbValue);
}

int RsRoceGetQpLbValue(struct ibv_qp *qp, int *lbValue)
{
    CHK_PRT_RETURN(lbValue == NULL, hccp_err("param error, lbValue is NULL"), -EINVAL);

    if (gHrnOps.rsRoceGetQpLbValue == NULL) {
        *lbValue = 0;
        return 0;
    }
    return gHrnOps.rsRoceGetQpLbValue(qp, lbValue);
}

int RsRoceGetQpNum(struct ibv_context *context, int *qpNum)
{
    CHK_PRT_RETURN(qpNum == NULL, hccp_err("param error, qpNum is NULL"), -EINVAL);

    if (gHrnOps.rsRoceGetQpNum == NULL) {
        *qpNum = 0;
        return 0;
    }
    return gHrnOps.rsRoceGetQpNum(context, qpNum);
}
