/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "hccl_st_test_case.h"
#include "rts_stub/rts_stub.h"
#include "ranktable/stub_rank_table.h"
#include <thread>
#include <iostream>
#include <vector>
#include <signal.h>
#include <execinfo.h>
#include <cstring>

using namespace std;
using namespace Hccl;

void SignalHandler(int sig)
{
    printf("received signal %d !!!\n", sig);
    constexpr int BACKTRACE_DEPTH = 15;
    void* array[BACKTRACE_DEPTH];
    char** callBackStrings;
    int size = backtrace(array, BACKTRACE_DEPTH);
    callBackStrings = backtrace_symbols(array, size);
    for (auto i = 0; i < size; ++i) {
        cout << callBackStrings << endl;
    }
    free(callBackStrings);
}

CommParams CreateCommParams(ThreadContext* ctx)
{
    Hccl::CommParams commParams;
    commParams.commId = ctx->commId;
    commParams.myRank = ctx->myRank;
    commParams.rankSize = ctx->situation.GetRankSize();
    commParams.devType = ctx->situation.GetDevType();
    return commParams;
}

CollOpParams CreateCollOpParams(ThreadContext* ctx)
{
    Hccl::CollOpParams collOpParams;
    collOpParams.opType = ctx->situation.GetOpType();
    collOpParams.dataType = ctx->situation.GetDataType();
    collOpParams.reduceOp = ctx->situation.GetReduceOp();
    collOpParams.dstRank = ctx->situation.GetDstRank();
    collOpParams.count = ctx->situation.GetCount();
    collOpParams.root = ctx->situation.GetRoot();
    collOpParams.staticAddr = ctx->situation.GetStaticAddr();
    collOpParams.staticShape = ctx->situation.GetStaticShape();
    collOpParams.sendBuf = ctx->sendBuf;
    collOpParams.recvBuf = ctx->recvBuf;
    return collOpParams;
}

void HcclStTestCase::InternalProcess(ThreadContext* ctx)
{
    try {
        SetCurrentThreadContext(ctx);

        cout << "Start Internal process at rank:  " << ctx->myRank << endl;
        PrepareCtx(ctx);

        // 1. 设置要使用的device
        aclrtSetDevice(ctx->myRank);

        // 2. 创建通讯域并初始化
        auto commParams = CreateCommParams(ctx);
        auto* communicator = new Hccl::HcclCommunicator(commParams);
        auto initRes = communicator->Init("ranktable.json");
        if (initRes != HcclResult::HCCL_SUCCESS) {
            cout << "communicator init failed!" << endl;
            return;
        }

        // 3. 准备集合通信用到的入参
        Hccl::CollOpParams collOpParams = CreateCollOpParams(ctx);
        aclrtStream stream = nullptr;
        aclrtCreateStreamWithConfig(&stream, 0U, 0U);

        // 4. 执行集合通信
        auto res = communicator->LoadOpbasedCollOp(collOpParams, stream);
        if (res != HcclResult::HCCL_SUCCESS) {
            cout << "Rank: " << ctx->myRank << " run failed!" << endl;
        } else {
            cout << "Rank: " << ctx->myRank << " run success!" << endl;
        }

        // 5. 等待集合通信任务完成, 在主线程内完成等待
        // communicator 也由主线程删除
        ctx->comm = communicator;
    } catch (...) {
        cout << "Rank: " << ctx->myRank << "Some exception occurs and was catched by test fwk." << endl;
    }
}

void HcclStTestCase::SetEnv()
{
    for (const auto& cfg : situation.GetEnv()) {
        setenv(cfg.first.c_str(), cfg.second.c_str(), 1);
    }
}

void HcclStTestCase::UnsetEnv()
{
    for (const auto& cfg : situation.GetEnv()) {
        unsetenv(cfg.first.c_str());
    }
}

void RankTableFileCreate(FakeClusterType cluster)
{
    if (cluster == FakeClusterType::CLUSTER_1_SERVER_8_DEV) { // 单机8卡
        GenRankTableFile(RankTable1Ser8Dev);
    } else if (cluster == FakeClusterType::CLUSTER_1_SERVER_2_DEV) { // 单机2卡
        GenRankTableFile(RankTable1Ser2Dev);
    } else {
        std::cout << "ST only support 1 Server now" << std::endl;
        throw std::exception();
    }
}

void RankTableFileDestroy() { DelRankTableFile(); }

void HcclStTestCase::Start()
{
    std::cout << "===== st test case " << testcaseName << " ===== start =====" << std::endl;
    RankTableFileCreate(situation.GetClusterType());
    SetEnv();
    EnvConfig::GetInstance(); // 强制解析环境变量，后续主代码仓适配环境变量接口后，可去除

    signal(SIGSEGV, SignalHandler);

    vector<vector<thread>> threads;

    int myRank = 0;
    for (int serverIndex = 0; serverIndex < situation.GetServerNum(); ++serverIndex) {
        threads.emplace_back();
        for (int deviceIndex = 0; deviceIndex < situation.GetDeviceNum(); ++deviceIndex) {
            auto* ctx = new ThreadContext();
            ctx->commId = "st-fwk";
            ctx->situation = situation;
            ctx->serverId = serverIndex;
            ctx->deviceId = deviceIndex;
            ctx->myRank = myRank++;
            contexts.push_back(ctx);
            std::thread container(&HcclStTestCase::InternalProcess, this, ctx);
            threads[serverIndex].push_back(std::move(container));
        }
    }

    for (int serverIndex = 0; serverIndex < situation.GetServerNum(); ++serverIndex) {
        for (int deviceIndex = 0; deviceIndex < situation.GetDeviceNum(); ++deviceIndex) {
            threads[serverIndex][deviceIndex].join();
        }
    }

    // 等待mock starts启动调度SQE
    int dummyStream = 0;
    aclrtSynchronizeStreamWithTimeout(&dummyStream, 0);

    UnsetEnv();
    RankTableFileDestroy();
    std::cout << "===== st test case " << testcaseName << " ===== finish =====" << std::endl << std::endl;
}

bool HcclStTestCase::Verify()
{
    for (auto ctx : contexts) {
        if (!VerifyCtx(ctx)) {
            return false;
        }
        delete ctx;
    }
    return true;
}

void HcclStTestCase::InitSituationEnv()
{
    situation.SetEnv("PRIM_QUEUE_GEN_NAME", "AllReduceRing");
    // 为提升覆盖率，多解析一些环境变量。后续用例增多后可逐渐去除
    situation.SetEnv("HCCL_IF_IP", "10.10.10.1");
    situation.SetEnv("HCCL_IF_BASE_PORT", "50000");
    situation.SetEnv("HCCL_SOCKET_IFNAME", "^=eth0,endvnic");
    situation.SetEnv("HCCL_WHITELIST_DISABLE", "0");
    situation.SetEnv("HCCL_NPU_NET_PROTOCOL", "RDMA");
    situation.SetEnv("HCCL_SOCKET_FAMILY", "AF_INET6");
    situation.SetEnv("HCCL_CONNECT_TIMEOUT", "200");
    situation.SetEnv("HCCL_EXEC_TIMEOUT", "1800");
    situation.SetEnv("HCCL_RDMA_TC", "100");
    situation.SetEnv("HCCL_RDMA_SL", "3");
    situation.SetEnv("HCCL_RDMA_TIMEOUT", "6");
    situation.SetEnv("HCCL_RDMA_RETRY_CNT", "5");
    situation.SetEnv("HCCL_INTRA_PCIE_ENABLE", "1");
    situation.SetEnv("HCCL_INTRA_ROCE_ENABLE", "0");
    situation.SetEnv("HCCL_INTER_HCCS_DISABLE", "FALSE");
    situation.SetEnv("PRIM_QUEUE_GEN_NAME", "AllReduceRing");
    situation.SetEnv("HCCL_ALGO", "level0:NA;level1:ring");
    situation.SetEnv("HCCL_BUFFSIZE", "200");
    situation.SetEnv("HCCL_OP_EXPANSION_MODE", "AI_CPU");
    situation.SetEnv("HCCL_DETERMINISTIC", "false");
    situation.SetEnv("HCCL_DIAGNOSE_ENABLE", "1");
    situation.SetEnv("HCCL_ENTRY_LOG_ENABLE", "1");
    situation.SetEnv("PROFILING_MODE", "true");
    situation.SetEnv(
        "PROFILING_OPTIONS", "{\"output\":\"/tmp/"
                             "profiling\",\"training_trace\":\"on\",\"task_trace\":\"on\",\"fp_point\":"
                             "\"\",\"bp_point\":\"\",\"aic_metrics\":\"PipeUtilization\"}");
    situation.SetEnv("LD_LIBRARY_PATH", "/temp:/runtime");
    situation.SetEnv("HCCL_DETOUR", "detour:0");
}
