/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_COMM_CONN_MGR_H
#define HCCL_COMM_CONN_MGR_H

#include <unordered_set>
#include <unordered_map>
#include <mutex>
#include "comm.h"
#include "hccl_comm_conn.h"

namespace hccl {

class HcclCommConnMgr {
public:
    static HcclCommConnMgr& GetInstance();

    HcclResult AddAndGetCommConn(HcclCommConn*& commConn);

    HcclResult AddCommConn(HcclCommConn*& commConn);

    HcclResult DelCommConn(HcclCommConn* commConn);
    bool IsExceedMaxLinkNum(u32 role);
    bool IsExistCommConn(HcclAddr& connectAddr);
    void InsertConnectCommMap(HcclAddr& connectAddr, HcclConn& conn);
    void DeleteConnectCommMap(HcclAddr& connectAddr);

private:
    HcclCommConnMgr();
    ~HcclCommConnMgr();
    HcclResult InitRa();
    HcclResult UninitRa();

    static constexpr u32 COMM_CONN_NUM_ONE = 1;
    std::unordered_set<HcclCommConn*> commConnSet_{};
    mutable std::mutex commConnMtx_{};
    s32 defaultDevId_{0};
    bool raInited_{false};
    RaInitConfig raConfig_{DEFAULT_INIT_PHY_ID, DEFAULT_INIT_NIC_POS, DEFAULT_HDC_TYPE, false};

    std::mutex connectCommMapMtx_{};
    std::map<uint32_t, HcclConn> connectCommMap_{};
};
} // namespace hccl

#endif
