/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_OPRETRY_CONNECTION_H
#define HCCL_OPRETRY_CONNECTION_H

#include <map>
#include <mutex>
#include <thread>
#include <memory>
#include <atomic>
#include "hccl_socket.h"
#include "hccl_network_pub.h"
#include "universal_concurrent_map.h"
#include "hccl_op_retry_pub.h"

namespace hccl {
const std::string OP_RETRY_CONN_SOCKET_TAG = "OP_RETRY_CONN_SOCKET_TAG";
const u32 OP_RETRY_CONN_PORT_MAX_RANGE = 15;

class OpRetryConnection {
public:
    OpRetryConnection();
    ~OpRetryConnection();

    using OpRetryConnectionPtr = std::shared_ptr<OpRetryConnection>;

    /* 配置是否开启该建链功能，如果不开启该功能部分接口会直接返回 */
    static void SetOpRetryConnEnable(bool enable);
    static bool IsOpRetryConnEnable();

    /* 创建并初始化对应全局静态资源实例*/
    static HcclResult Init(
        const std::string& group, u32 rankSize, const OpRetryServerInfo& serverInfo, const OpRetryAgentInfo& agentInfo,
        u32 rootRank = 0);

    /*
     * 供创建全局实例，以group为Key
     * forceNew表示如果已有存在则释放，创建新的实例，用户一般使用默认配置false即可
     */
    static HcclResult GetInstance(const std::string& group, OpRetryConnectionPtr& conn, bool forceNew = false);
    static HcclResult DelInstance(const std::string& group);

    HcclResult Init(
        u32 rankId, u32 rankSize, const HcclIpAddress& serverIp, u32 serverPort, s32 serverDevId,
        const HcclIpAddress& localIp, u32 rootRank = 0);
    HcclResult DeInit();

    bool IsRoot() const /* 判断自己是否为Root节点 */ { return rankId_ == rootRank_; }

    HcclResult GetAgentSocket(std::shared_ptr<HcclSocket>& sock);
    HcclResult GetServerSockets(std::map<u32, std::shared_ptr<HcclSocket>>& socks);

private:
    void SetGroup(const std::string& group) { group_ = group; }

    const std::string& GetTag()
    {
        if (tag_.empty()) {
            tag_ = OP_RETRY_CONN_SOCKET_TAG + "_" + group_ + "_" + std::to_string(serverPort_);
        }

        return tag_;
    }

    HcclResult InitHcclNet();
    HcclResult LoadHostWhiteList(const std::string& whiteListFile);

    HcclResult StartListen();
    HcclResult StopListen();
    HcclResult Accept();
    HcclResult WaitAcceptFinish(); /* 阻塞等待accept完成 */
    HcclResult RecvMetaInfo(std::shared_ptr<HcclSocket>& peerSocket);
    HcclResult SendAckInfo(std::shared_ptr<HcclSocket>& peerSocket);
    void RunAccept(); /* 线程入口，异步接收所有连接 */

    HcclResult Connect();
    HcclResult SendMetaInfo();
    HcclResult RecvAckInfo();

    HcclResult GetHostSocketWhiteList();   /* 从文件中解析白名单 */
    HcclResult AddListenSocketWhiteList(); /* 转换白名单到listen socket中 */

    static u32 GetServerPort();

    /* 公共数据结构 */
    static std::mutex lock_;
    static UniversalConcurrentMap<std::string, OpRetryConnectionPtr>* instance_;
    static bool enable_;

    s32 deviceLogicalID_{INVALID_INT};
    u32 devicePhysicID_{INVALID_UINT};
    u32 rankId_{INVALID_UINT};
    u32 rankSize_{0};
    u32 rootRank_{0};
    HcclIpAddress serverIp_;
    u32 serverPort_;
    HcclIpAddress localIp_;
    bool hcclNetInit_{false};
    std::string group_;
    std::string tag_;

    /* Server侧的数据结构 */
    std::shared_ptr<HcclSocket> listenSocket_{nullptr};
    std::map<u32, std::shared_ptr<HcclSocket>> connectionSockets_;
    HcclNetDevCtx serverNetCtx_{nullptr};
    bool enableWhitelist_{false};
    std::vector<HcclIpAddress> whitelist_;
    std::vector<SocketWlistInfo> wlistInfosVec_;
    std::atomic<bool> acceptFinished_{false};
    std::atomic<bool> backgroudThreadStop_{false};
    std::thread backgroudThread_;

    /* Client侧的数据结构 */
    std::shared_ptr<HcclSocket> socket_{nullptr};
    HcclNetDevCtx clientNetCtx_{nullptr};
};
} // namespace hccl

#endif
