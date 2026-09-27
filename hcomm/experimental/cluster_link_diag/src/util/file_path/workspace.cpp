/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <workspace.h>

#include <filesystem>
#include <chrono>

// 获取可执行文件绝对路径
std::string get_exec_path()
{
    namespace fs = std::filesystem;
#if defined(_WIN32) || defined(_WIN64)
    char path[MAX_PATH];
    GetModuleFileNameA(nullptr, path, MAX_PATH);
    return std::string(path);
#elif defined(__APPLE__)
    char path[PATH_MAX];
    uint32_t size = sizeof(path);
    if (_NSGetExecutablePath(path, &size) == 0) {
        return std::string(path);
    }
    return "";
#else
    return fs::canonical("/proc/self/exe").string();
#endif
}

// 根据工程名截取工程路径
std::string get_project_path(const std::string& projectName)
{
    namespace fs = std::filesystem;
    std::string exePath = get_exec_path();
    fs::path path(exePath);

    static std::string project_path;
    if (!project_path.empty()) {
        return project_path;
    }

    // 从可执行文件路径向上遍历目录
    while (!path.empty()) {
        auto filename = path.filename().string();
        if (filename.find(projectName) != filename.npos) {
            project_path = path.string();
            return project_path;
        }
        path = path.parent_path();
    }

    // 未找到工程目录时返回空字符串
    return "";
}

bool make_abs_path(std::string& path)
{
    if (path.front() == '.') {
        path = get_project_path() + path.substr(1);
        return true;
    }
    return false;
}

// 根据工程名截取工程路径
std::string get_project_path() { return get_project_path(PROJECT_NAME); }

// 获取输出路径,如果输出目录不存在则创建
std::string get_output_path()
{
    static bool created_path = false;
    static std::string timestamp;
    static std::string output_path;
    if (!created_path) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);

        // 线程安全的时间转换
        std::tm local_time_buf;
#ifdef _WIN32
        localtime_s(&local_time_buf, &now_time);
#else
        localtime_r(&now_time, &local_time_buf);
#endif

        // 使用 strftime 手动格式化
        char buffer[80];
        strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", &local_time_buf);
        timestamp = buffer;
        output_path = get_project_path() + "/output/" + timestamp.substr(0, 23);
        std::filesystem::create_directories(output_path);
        created_path = true;
    }
    return output_path;
}
