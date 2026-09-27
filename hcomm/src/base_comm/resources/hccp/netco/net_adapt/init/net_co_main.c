/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define BKF_LOG_HND ((co)->log)
#define BKF_MOD_NAME ((co)->name)

#include <stdlib.h>
#include <sys/types.h>
#include <unistd.h>
#include "net_co_main_data.h"
#include "net_co_mem.h"
#include "net_co_xmap.h"
#include "net_co_disp.h"
#include "net_co_log_cnt.h"
#include "net_co_log.h"
#include "net_co_pfm.h"
#include "net_co_mux.h"
#include "net_co_tmr.h"
#include "net_co_disp_out.h"
#include "net_co_job.h"
#include "net_co_sys_log.h"
#include "net_co_in.h"
#include "net_co_out.h"
#include "securec.h"
#include "net_co_main.h"

#define CFG_BUILD_DEBUG 1
#ifdef __cplusplus
extern "C" {
#endif
#define NET_CO_NAME "netCo"
#define NET_CO_CID (0x12345)
#ifdef CFG_BUILD_DEBUG
#define NET_CO_DBG_ON VOS_TRUE
#define NET_CO_PFM_ON VOS_TRUE
#else
#define NET_CO_DBG_ON VOS_FALSE
#define NET_CO_PFM_ON VOS_FALSE
#endif

STATIC NetCo *NetCoDataNew(NetCoInitArg *arg)
{
    uint32_t len = sizeof(NetCo);
    NetCo *co = malloc(len);
    if (co == VOS_NULL) {
        return VOS_NULL;
    }

    (void)memset_s(co, len, 0, len);
    co->argInit = *arg;
    co->name = NET_CO_NAME;
    co->dbgOn = NET_CO_DBG_ON;
    co->selfCid = NET_CO_CID;
    co->pfmOn = NET_CO_PFM_ON;

    return co;
}
STATIC void NetCoDataDelete(NetCo *co)
{
    free(co);
}
STATIC const BkfModVTbl g_NetCoModVTbl[] = {
    {(F_BKF_DO)NetCoMemInit, (F_BKF_DOV)NetCoMemUninit},
    {(F_BKF_DO)NetCoXMapInit, (F_BKF_DOV)NetCoXMapUninit},
    {(F_BKF_DO)NetCoDispInit, (F_BKF_DOV)NetCoDispUninit},
    {(F_BKF_DO)NetCoLogCntInit, (F_BKF_DOV)NetCoLogCntUninit},
    {(F_BKF_DO)NetCoLogInit, (F_BKF_DOV)NetCoLogUninit},
    {(F_BKF_DO)NetCoPfmInit, (F_BKF_DOV)NetCoPfmUninit},
    {(F_BKF_DO)NetCoMuxInit, (F_BKF_DOV)NetCoMuxUninit},
    {(F_BKF_DO)NetCoTmrInit, (F_BKF_DOV)NetCoTmrUninit},
    {(F_BKF_DO)NetCoDispOutInit, (F_BKF_DOV)NetCoDispOutUninit},
    {(F_BKF_DO)NetCoJobInit, (F_BKF_DOV)NetCoJobUninit},
    {(F_BKF_DO)NetCoSysLogInit, (F_BKF_DOV)NetCoSysLogUninit},

    {(F_BKF_DO)NetCoInInit, (F_BKF_DOV)NetCoInUninit},
    {(F_BKF_DO)NetCoOutInit, (F_BKF_DOV)NetCoOutUninit},
};
NetCo *NetCoInit(NetCoInitArg *arg)
{
    if (arg == VOS_NULL) {
        return VOS_NULL;
    }
    NetCo *co = NetCoDataNew(arg);
    if (co == VOS_NULL) {
        return VOS_NULL;
    }
    uint32_t ret = BkfModsInit(g_NetCoModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoModVTbl));
    if (ret != BKF_OK) {
        BKF_LOG_ERROR(BKF_LOG_HND, "ret(%u), ng\n", ret);
        NetCoUninit(co);
        return VOS_NULL;
    }

    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "co(%#x), init ok, pid(%d)\n", BKF_MASK_ADDR(co), getpid());
    NetCoDispOutRegedCmd(co);
    return co;
}

void NetCoUninit(NetCo *co)
{
    if (co == VOS_NULL) {
        return;
    }
    NET_CO_LOG_MOD_DO(BKF_LOG_HND, "co(%#x), pid(%d)\n", BKF_MASK_ADDR(co), getpid());

    BkfModsUninit(g_NetCoModVTbl, co, BKF_GET_ARY_COUNT(g_NetCoModVTbl));
    NetCoDataDelete(co);
}

uint32_t NetCoFdEvtDispatch(NetCo *co, int fd, uint32_t curEvents)
{
    if (co == VOS_NULL) {
        return BKF_ERR;
    }

    void *cookie;
    uint32_t interestedEvents;
    F_WFW_MUX_FD_PROC proc = WfwMuxGetFdProc(co->mux, fd, &cookie, &interestedEvents);
    if (proc == VOS_NULL) {
        return BKF_ERR;
    }

    uint32_t events = curEvents & interestedEvents;
    BKF_LOG_DEBUG(BKF_LOG_HND, "fd(%d)/events(%#x, %#x), pid(%d)\n", fd, curEvents, events, getpid());
    proc(fd, events, cookie);
    return NET_CO_PROCED;
}

#ifdef __cplusplus
}
#endif
