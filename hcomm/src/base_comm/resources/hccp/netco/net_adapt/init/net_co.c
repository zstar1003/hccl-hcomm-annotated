/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <stdio.h>
#include "net_co_main.h"
#include "net_co_out_tbl.h"
#include "net_vo_tbl_adj.h"
#include "net_vo_tbl_comm_info.h"
#include "net_vo_tbl_oper.h"
#include "net_vo_tbl_rank.h"
#include "net_vo_tbl_rank_dist.h"
#include "net_vo_tbl_root_rank.h"
#include "net_slog.h"
#include "net_co.h"

__attribute__((visibility("default"))) void *NetCoInitFactory(int epollfd, NetCoIpPortArg ipPortArg)
{
    NetCoInitArg arg = {0};
    int ret;
    NetCoOpenSlogSo();
    NetCoSlogApiInit();
    arg.epFd = epollfd;
    // 打桩接口, 后面通过ip获取接口进行处理
    ret = (int)BkfUrlLeTcpSet(&arg.inConnSelfUrl, ipPortArg.localIp, ipPortArg.localNetPort);
    ret = (int)BkfUrlLeTcpSet(&arg.inConnNetUrl, ipPortArg.gatewayIp, ipPortArg.gatewayPort);
    ret = (int)BkfUrlLeTcpSet(&arg.outConnLsnUrl, ipPortArg.localIp, ipPortArg.listenPort);
    nslb_dbg("NetCoInitFactory NetCo init. ret %d", ret);
    nslb_dbg("NetCoInitFactory local ip info: (0x%x) port(%d).", arg.inConnSelfUrl.ip.addrH, arg.inConnSelfUrl.ip.port);
    nslb_dbg("NetCoInitFactory remote ip info: (0x%x) port(%d).", arg.inConnNetUrl.ip.addrH, arg.inConnNetUrl.ip.port);
    nslb_dbg("NetCoInitFactory listen ip info: (0x%x) port(%d).", arg.outConnLsnUrl.ip.addrH,
        arg.outConnLsnUrl.ip.port);
    NetCo *netco_handle = NetCoInit(&arg);
    nslb_dbg("NetCoInitFactory NetCo init handle:%p epFd:%x", netco_handle, arg.epFd);
    return netco_handle;
}

__attribute__((visibility("default"))) void NetCoDestruct(void *co)
{
    NetCoCloseSlogSo();
    NetCoUninit(co);
}

__attribute__((visibility("default"))) unsigned int NetCoFdEventDispatch(void *co, int fd, unsigned int curEvents)
{
    return (int)NetCoFdEvtDispatch(co, fd, (uint32_t)curEvents);
}

__attribute__((visibility("default"))) int NetCoTblAddUpd(void *netcoHandle, unsigned int type, char *data,
    unsigned int data_len)
{
    int ret = 0;

    if ((data == NULL) || data_len == 0) {
        nslb_err("data is NULL or data_len(%u) invalid", data_len);
        return -1;
    }

    nslb_info("type(%u).", type);
    if (netcoHandle == NULL) {
        nslb_err("netco_handle is NULL");
        return -1;
    }

    switch (type) {
        case TYPE_TBL_COMM_INFO:
            ret = (int)NetCoTblCommInfoAddUpd(netcoHandle, (NetTblCommInfo *)data, (uint32_t)data_len);
            break;
        case TYPE_TBL_OPER:
            ret = (int)NetCoTblOperAddUpd(netcoHandle, (NetTblOperator *)data, (uint32_t)data_len);
            break;
        case TYPE_TBL_ADJ:
            ret = (int)NetCoTblAdjAddUpd(netcoHandle, (NetTblAdjacency *)data, (uint32_t)data_len);
            break;
        case TYPE_TBL_RANK:
            ret = (int)NetCoTblRankAddUpd(netcoHandle, (NetTblGlobalRank *)data, (uint32_t)data_len);
            break;
        case TYPE_TBL_RANK_DIST:
            ret = (int)NetCoTblRankDistAddUpd(netcoHandle, (NetTblRankDistributeInfo *)data, (uint32_t)data_len);
            break;
        case TYPE_TBL_ROOT_RANK:
            ret = (int)NetCoTblRootRankAddUpd(netcoHandle, (NetTblRootRank *)data, (uint32_t)data_len);
            break;
        default:
            break;
    };

    if (ret != 0) {
        nslb_err("type(%u), ret(%d).", type, ret);
    }
    return ret;
}