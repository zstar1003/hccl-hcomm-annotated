/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <vector>
#include <string>
#include <map>
#include <future>

#include "proxy_call.hpp"
#include "topo/control_topo.h"

/*
    用于并行proxy_call
*/
#define proxy_call_paralled(device_list, ct, func, ...) \
    proxy_call_paralled_internal(device_list, ct, func, #func, ##__VA_ARGS__)

inline int get_or_create_client_index(
    const std::string& dst, const std::string& host, std::map<std::string, int>& dst_to_index,
    std::vector<std::shared_ptr<rpc::client>>& client_list)
{
    if (dst_to_index.find(dst) == dst_to_index.end()) {
        dst_to_index[dst] = client_list.size();
        client_list.emplace_back(make_rpc_client_shared(host));
        client_list.back()->set_timeout(RPC_DEFAULT_TIMEOUT);
    }
    return dst_to_index[dst];
}

template <typename R, typename... Args, typename... VecArgs>
std::enable_if_t<!std::is_void_v<R>, std::vector<R>> proxy_call_paralled_internal(
    const std::vector<std::string>& device_list, ControlTopo* ct, R (*func)(Args...), const std::string& func_name,
    VecArgs&&... args)
{
    int n = device_list.size();
    std::map<std::string, int> dst_to_index;
    std::vector<std::shared_ptr<rpc::client>> client_list;

    std::vector<std::future<RPCLIB_MSGPACK::object_handle>> future_res_rpc(n);
    std::vector<R> results(n);
    for (int i = 0; i < n; i++) {
        auto& dst = device_list[i];
        auto hosts = ct->route_to_device(dst);
        std::string host;
        try {
            // 获取当前主机并移除
            host = hosts.back();
            hosts.pop_back();
            const int client_index = get_or_create_client_index(dst, host, dst_to_index, client_list);

            // 构造代理函数名 "proxy_call<R>"
            std::string proxy_func_name = get_proxy_func_name<R, Args...>();

            if (hosts.empty()) {
                // 直接调用目标函数
                future_res_rpc[i] = client_list[client_index]->async_call(
                    func_name, std::forward<Args>(static_cast<Args>(args[i]))...);
            } else {
                // 递归调用代理函数
                future_res_rpc[i] = client_list[client_index]->async_call(
                    proxy_func_name, hosts, func_name, std::forward<Args>(static_cast<Args>(args[i]))...);
            }
        } catch (rpc::rpc_error& e) {
            auto error_info = "[" + host + "]" + e.what();
            rpc::this_handler().respond_error(error_info);
            throw std::runtime_error("[proxy]" + error_info);
        }
    }

    for (int i = 0; i < n; i++) {
        results[i] = future_res_rpc[i].get().get().as<R>();
    }

    return results;
}

template <typename R, typename... Args, typename... VecArgs>
std::enable_if_t<std::is_void_v<R>, void> proxy_call_paralled_internal(
    const std::vector<std::string>& device_list, ControlTopo* ct, R (*func)(Args...), const std::string& func_name,
    VecArgs&&... args)
{
    int n = device_list.size();
    std::vector<std::future<RPCLIB_MSGPACK::object_handle>> future_res_rpc(n);
    std::map<int, std::future<void>> future_res; // 使用future<void>
    std::vector<int> status_results(n, 0);       // 返回状态码向量，0表示成功
    std::map<std::string, int> dst_to_index;
    std::vector<std::shared_ptr<rpc::client>> client_list;

    for (int i = 0; i < n; i++) {
        auto& dst = device_list[i];
        auto hosts = ct->route_to_device(dst);
        std::string host;
        try {
            // 获取当前主机并移除
            host = hosts.back();
            hosts.pop_back();
            const int client_index = get_or_create_client_index(dst, host, dst_to_index, client_list);

            std::string proxy_func_name = get_proxy_func_name<void, Args...>();

            if (hosts.empty()) {
                // 直接调用目标函数
                future_res_rpc[i] = client_list[client_index]->async_call(
                    func_name, std::forward<Args>(static_cast<Args>(args[i]))...);
            } else {
                // 递归调用代理函数
                future_res_rpc[i] = client_list[client_index]->async_call(
                    proxy_func_name, hosts, func_name, std::forward<Args>(static_cast<Args>(args[i]))...);
            }
        } catch (rpc::rpc_error& e) {
            auto error_info = "[" + host + "]" + e.what();
            rpc::this_handler().respond_error(error_info);
            throw std::runtime_error("[proxy]" + error_info);
        }
    }

    // 处理RPC调用结果
    for (int i = 0; i < n; i++) {
        try {
            auto handle = future_res_rpc[i].get();
            // 对于void函数，我们不尝试转换返回值
            // 只是确保调用完成
            status_results[i] = 0; // 成功
        } catch (...) {
            status_results[i] = -1; // 失败
        }
    }
}
