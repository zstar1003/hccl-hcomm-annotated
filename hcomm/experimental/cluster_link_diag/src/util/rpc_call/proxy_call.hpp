/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include <vector>
#include <string>
#include <algorithm>
#include <cctype>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <typeinfo>
#include <sstream>
#include "rpc/client.h"
#include "rpc/server.h"
#include "rpc/rpc_error.h"
#include "rpc/this_handler.h"

#include "rpc_info.h"

#include <iostream>

inline bool rpc_host_is_port(const std::string& host)
{
    return !host.empty() && std::all_of(host.begin(), host.end(), [](unsigned char c) {
        return std::isdigit(c);
    });
}

inline int parse_rpc_port(const std::string& host)
{
    try {
        size_t parsed = 0;
        int port = std::stoi(host, &parsed);
        if (parsed != host.size() || port <= 0 || port > 65535) {
            throw std::out_of_range("port");
        }
        return port;
    } catch (const std::exception&) {
        throw std::invalid_argument("[RPC_PROXY] invalid localhost port: " + host);
    }
}

inline rpc::client make_rpc_client(const std::string& host)
{
    if (host.empty()) {
        throw std::invalid_argument("[RPC_PROXY] empty host");
    }
    if (rpc_host_is_port(host)) {
        return rpc::client("localhost", parse_rpc_port(host));
    }
    return rpc::client(host, RPC_DEFAULT_PORT);
}

inline std::shared_ptr<rpc::client> make_rpc_client_shared(const std::string& host)
{
    if (rpc_host_is_port(host)) {
        return std::make_shared<rpc::client>("localhost", parse_rpc_port(host));
    }
    if (host.empty()) {
        throw std::invalid_argument("[RPC_PROXY] empty host");
    }
    return std::make_shared<rpc::client>(host, RPC_DEFAULT_PORT);
}
/*
    get_proxy_func_name: 通过R,Args...获取代理函数名
*/
// 类型名称获取函数（基础版本）
template <typename T>
std::string type_name()
{
    return typeid(T).name();
}

// 主模板函数
template <typename R, typename... Args>
std::string get_proxy_func_name()
{
    std::ostringstream oss;
    oss << "proxy_call<";

    // 添加返回类型
    oss << type_name<R>();

    // 添加参数类型（如果有）
    if constexpr (sizeof...(Args) > 0) {
        oss << ",";

        // 使用折叠表达式展开参数包
        std::vector<std::string> arg_names = {type_name<Args>()...};
        for (size_t i = 0; i < arg_names.size(); ++i) {
            if (i > 0)
                oss << ",";
            oss << arg_names[i];
        }
    }

    oss << ">";
    return oss.str();
}

/*
    proxy_call: 代理调用函数
*/
// 内部函数模板
template <typename R, typename... Args>
R proxy_call_internal(std::vector<std::string> hosts, const std::string& func_name, Args... args)
{
    std::future<RPCLIB_MSGPACK::object_handle> fut_res;
    RPCLIB_MSGPACK::object_handle res;

    std::string host;
    try {
        // 获取当前主机并移除
        host = hosts.back();
        hosts.pop_back();

        rpc::client c = make_rpc_client(host);
        c.set_timeout(RPC_DEFAULT_TIMEOUT);
        // 构造代理函数名 "proxy_call<R>"
        std::string proxy_func_name = get_proxy_func_name<R, Args...>();
        if (hosts.empty()) {
            // 直接调用目标函数
            fut_res = c.async_call(func_name, std::forward<Args>(args)...);
#ifdef RPC_PROXY_DEBUG
            std::cout << "[RPC_PROXY]" << host << "->" << func_name << std::endl;
#endif
        } else {
            // 递归调用代理函数
            fut_res = c.async_call(proxy_func_name, hosts, func_name, std::forward<Args>(args)...);
#ifdef RPC_PROXY_DEBUG
            std::cout << "[RPC_PROXY]" << host << "->" << proxy_func_name << std::endl;
#endif
        }
        res = fut_res.get();
    } catch (rpc::rpc_error& e) {
        auto error_info = "[" + host + "]" + e.what();
        rpc::this_handler().respond_error(error_info);
        throw std::runtime_error("[RPC_PROXY]" + error_info);
    }
    if constexpr (std::is_void_v<R>) {
        int status;
        try {
            auto handle = res.get();
            status = 0;
        } catch (...) {
            status = -1;
        }
        return;
    } else {
        // 非void返回类型：转换并返回值
        return res.get().as<R>();
    }
}

// 宏定义用于简化调用，自动获取函数名和返回类型
#define proxy_call(proxy_hosts, func, ...) proxy_call_outernal(proxy_hosts, func, #func, ##__VA_ARGS__)
// 外部函数模板
template <typename R, typename... Args, typename... input_args>
R proxy_call_outernal(
    std::vector<std::string> hosts, R (*func)(Args...), const std::string& func_name, input_args&&... args)
{
    if (hosts.empty()) {
        if constexpr (std::is_void_v<R>) {
            func(std::forward<Args>(static_cast<Args>(args))...);
            return;
        } else {
            return func(std::forward<Args>(static_cast<Args>(args))...);
        }
    } else {
        if constexpr (std::is_void_v<R>) {
            proxy_call_internal<R, Args...>(hosts, func_name, std::forward<Args>(static_cast<Args>(args))...);
            return;
        } else {
            return proxy_call_internal<R, Args...>(hosts, func_name, std::forward<Args>(static_cast<Args>(args))...);
        }
    }
}

/*
    proxy_bind: 代理绑定函数
*/
#define proxy_bind(server, func) proxy_bind_internal(server, func, #func)

static std::set<std::string> bound_functions;
template <typename R, typename... Args>
void proxy_bind_internal(rpc::server& server, R (*func)(Args...), const std::string& func_name)
{
    std::string proxy_func_name = get_proxy_func_name<R, Args...>();
    // 如果server没有绑定该函数，则绑定该函数
    if (bound_functions.find(proxy_func_name) == bound_functions.end()) {
        server.bind(proxy_func_name, proxy_call_internal<R, Args...>);
        bound_functions.insert(proxy_func_name);
    }
#ifdef RPC_CALL_DEBUG
    auto func_lambda = [func, func_name](Args... args) -> R {
        std::cout << "[RPC_CALL]" << func_name << std::endl;
        return func(std::forward<Args>(args)...);
    };
    server.bind(func_name, func_lambda);
    std::cout << "[RPC_BIND]" << func_name << std::endl;
#else
    server.bind(func_name, func);
#endif
}
