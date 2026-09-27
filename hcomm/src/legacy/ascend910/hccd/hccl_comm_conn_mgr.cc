/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_comm_conn_mgr.h"

using namespace std;

namespace hccl {
HcclCommConnMgr& HcclCommConnMgr::GetInstance()
{
    static HcclCommConnMgr connMgr;
    HCCL_INFO("HcclCommConnMgr::GetInstance connMgr[%p]", &connMgr);
    return connMgr;
}

HcclCommConnMgr::HcclCommConnMgr() {}

HcclCommConnMgr::~HcclCommConnMgr() { (void)UninitRa(); }

HcclResult HcclCommConnMgr::InitRa()
{
    if (raInited_) {
        HCCL_DEBUG("InitRa has been already inited");
        return HCCL_SUCCESS;
    }

    CHK_RET(DlRaFunction::GetInstance().DlRaFunctionInit());

    raConfig_.phyId = defaultDevId_; // 暂缺获取物理id的手段
    raConfig_.nicPosition = static_cast<u32>(NICDeployment::NIC_DEPLOYMENT_HOST);
    CHK_RET(HrtRaInit(&raConfig_));

    raInited_ = true;
    HCCL_INFO("Ra has been inited successfully");

    return HCCL_SUCCESS;
}

HcclResult HcclCommConnMgr::UninitRa()
{
    if (!raInited_) {
        HCCL_DEBUG("InitRa has been already uninited");
        return HCCL_SUCCESS;
    }

    CHK_RET(HrtRaDeInit(&raConfig_));
    raInited_ = false;
    HCCL_INFO("Ra has been uninited successfully");

    return HCCL_SUCCESS;
}

HcclResult HcclCommConnMgr::AddAndGetCommConn(HcclCommConn*& commConn)
{
    commConn = new (nothrow) HcclCommConn();
    CHK_PTR_NULL(commConn);

    lock_guard<mutex> lock(commConnMtx_);
    auto result = commConnSet_.insert(commConn);
    if (result.second && commConnSet_.size() == COMM_CONN_NUM_ONE) {
        CHK_RET(InitRa());
        CHK_RET(InitExternalInput());
    }
    HCCL_INFO("AddAndGetCommConn commConnSet_ size[%u]", commConnSet_.size());

    return HCCL_SUCCESS;
}

HcclResult HcclCommConnMgr::AddCommConn(HcclCommConn*& commConn)
{
    CHK_PTR_NULL(commConn);

    lock_guard<mutex> lock(commConnMtx_);
    commConnSet_.insert(commConn);

    return HCCL_SUCCESS;
}

bool HcclCommConnMgr::IsExceedMaxLinkNum(u32 role)
{
    lock_guard<mutex> lock(commConnMtx_);
    if (role == SERVER_ROLE_SOCKET) {
        // server侧有自身的一个comm, 因此判断最大通信连接数的时候要加1
        return commConnSet_.size() >= (MAX_CONN_LINK_NUM + 1);
    }
    return commConnSet_.size() > MAX_CONN_LINK_NUM;
}

bool HcclCommConnMgr::IsExistCommConn(HcclAddr& connectAddr)
{
    std::unique_lock<std::mutex> lock(connectCommMapMtx_);
    return connectCommMap_.find(connectAddr.info.tcp.ipv4Addr) != connectCommMap_.end();
}

void HcclCommConnMgr::InsertConnectCommMap(HcclAddr& connectAddr, HcclConn& conn)
{
    std::unique_lock<std::mutex> lock(connectCommMapMtx_);
    connectCommMap_.insert(std::make_pair(connectAddr.info.tcp.ipv4Addr, conn));
}

void HcclCommConnMgr::DeleteConnectCommMap(HcclAddr& connectAddr)
{
    std::unique_lock<std::mutex> lock(connectCommMapMtx_);
    connectCommMap_.erase(connectAddr.info.tcp.ipv4Addr);
}

HcclResult HcclCommConnMgr::DelCommConn(HcclCommConn* commConn)
{
    {
        lock_guard<mutex> lock(commConnMtx_);
        auto result = commConnSet_.erase(commConn);
        if (result == 0) {
            HCCL_ERROR("commConn is not found in commConnSet");
            return HCCL_E_NOT_FOUND;
        }
        HCCL_INFO("DelCommConn commConnSet_ size[%u]", commConnSet_.size());
    }

    // 对外接口已经统一校验过commConn
    delete commConn;
    commConn = nullptr;

    // commConnSet_.size() == 0的时候说明P侧进程业务完成要退出
    lock_guard<mutex> lock(commConnMtx_);
    if (commConnSet_.size() == 0) {
        (void)UninitRa();
    }
    return HCCL_SUCCESS;
}

} // namespace hccl
