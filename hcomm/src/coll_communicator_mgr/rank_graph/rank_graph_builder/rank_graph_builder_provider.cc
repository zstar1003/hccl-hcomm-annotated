/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "rank_graph_builder_bridge.h"

#include <exception>
#include <new>

#include "hccl_exception.h"
#include "log.h"
#include "rank_graph_builder.h"

namespace Hccl {
namespace {

    // RankGraph 的实现和分配均位于 hcomm，跨 SO 持有时也必须回到 hcomm 执行析构。
    void DestroyRankGraphImpl(RankGraph* rankGraph) { delete rankGraph; }

    HcclResult AdoptRankGraphImpl(std::unique_ptr<RankGraph> rankGraph, std::shared_ptr<RankGraph>& sharedRankGraph)
    {
        if (rankGraph == nullptr) {
            HCCL_ERROR("[%s] input rankGraph is nullptr.", __func__);
            return HCCL_E_PTR;
        }

        RankGraph* rawRankGraph = rankGraph.release();
        try {
            // shared_ptr 构造失败时会自行调用 deleter，禁止在 catch 中再次释放 rawRankGraph。
            std::shared_ptr<RankGraph> adoptedRankGraph(rawRankGraph, DestroyRankGraphImpl);
            sharedRankGraph = std::move(adoptedRankGraph);
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[%s] create shared RankGraph failed: %s", __func__, e.what());
            return HCCL_E_MEMORY;
        }
        return HCCL_SUCCESS;
    }

    HcclResult FillBuildResult(
        RankGraphBuilder& rankGraphBuilder, std::unique_ptr<RankGraph> rankGraph, RankGraphBuildResult& result)
    {
        // 在栈上 builder 析构前转移完整结果，保证 RankGraph、RankTableInfo 与 TopoInfo 来自同一次构建。
        std::unique_ptr<RankTableInfo> rankTableInfo = rankGraphBuilder.GetRankTableInfo();
        std::shared_ptr<TopoInfo> topoInfo = rankGraphBuilder.GetTopoInfo();
        if (rankGraph == nullptr || rankTableInfo == nullptr || topoInfo == nullptr) {
            HCCL_ERROR("[%s] rank graph build result is invalid.", __func__);
            return HCCL_E_INTERNAL;
        }

        CHK_RET(AdoptRankGraphImpl(std::move(rankGraph), result.rankGraph));
        result.rankTableInfo = std::move(*rankTableInfo);
        result.topoInfo = *topoInfo;
        return HCCL_SUCCESS;
    }

    // BuildRankGraph 是三个 provider 入口共享的构建骨架，BuildFunc 表示各入口不同的构建动作：
    template <typename BuildFunc>
    HcclResult BuildRankGraph(BuildFunc&& buildFunc, RankGraphBuildResult& result)
    {
        RankGraphBuilder rankGraphBuilder;
        return FillBuildResult(rankGraphBuilder, buildFunc(rankGraphBuilder), result);
    }

    template <typename BuildFunc>
    HcclResult RunBuild(const char* funcName, BuildFunc&& buildFunc, RankGraphBuildResult& result)
    {
        try {
            return BuildRankGraph(buildFunc, result);
        } catch (const HcclException& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return e.GetErrorCode();
        } catch (const std::bad_alloc& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return HCCL_E_MEMORY;
        } catch (const std::exception& e) {
            HCCL_ERROR("[%s] failed: %s", funcName, e.what());
            return HCCL_E_INTERNAL;
        } catch (...) {
            HCCL_ERROR("[%s] failed: unknown exception", funcName);
            return HCCL_E_INTERNAL;
        }
    }

    // 三个窄入口分别承接字符串初始化、RootInfoDetect 结果初始化和故障恢复，
    HcclResult BuildFromStringImpl(
        const std::string& rankTable, const std::string& topoPath, RankId myRank, RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result);
    }

    HcclResult BuildFromRankTableImpl(
        const RankTableInfo& rankTable, const std::string& topoPath, RankId myRank, RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.Build(rankTable, topoPath, myRank);
            },
            result);
    }

    HcclResult RecoverBuildImpl(
        const RankTableInfo& rankTable, const TopoInfo& topoInfo, RankId myRank, RankGraphBuildResult& result)
    {
        return RunBuild(
            __func__,
            [&](RankGraphBuilder& rankGraphBuilder) {
                return rankGraphBuilder.RecoverBuild(rankTable, topoInfo, myRank);
            },
            result);
    }

    // 将 bridge 的四个字段绑定到本文件中的 provider 实现；调用函数指针等价于调用对应的 Impl 函数。
    const RankGraphBuilderBridge RANK_GRAPH_BUILDER_BRIDGE = {
        BuildFromStringImpl,
        BuildFromRankTableImpl,
        RecoverBuildImpl,
        AdoptRankGraphImpl,
    };

    // hcomm 装载时向 hccl_v2 发布窄回调接口，legacy 调用方无需链接 RankGraphBuilder 的具体实现。
    struct RankGraphBuilderBridgeRegistrar {
        RankGraphBuilderBridgeRegistrar()
        {
            HcclResult ret = RegisterRankGraphBuilderBridge(RANK_GRAPH_BUILDER_BRIDGE);
            if (ret != HCCL_SUCCESS) {
                HCCL_ERROR("[%s] register RankGraphBuilder bridge failed, ret[%d].", __func__, static_cast<int>(ret));
            }
        }
    };

    RankGraphBuilderBridgeRegistrar g_rankGraphBuilderBridgeRegistrar;

} // namespace
} // namespace Hccl
