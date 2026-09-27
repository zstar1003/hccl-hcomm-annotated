/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <atomic>
#include <mutex>
#include "mmpa_api.h"
#include "llt_hccl_stub_profiling_plugin.h"
#include "toolchain/prof_api.h"
using namespace hccl;

int32_t Report(uint32_t moduleId, uint32_t type, void* data, uint32_t len) { return 0; }

rtError_t rtProfRegisterCtrlCallback(uint32_t modelid, rtProfCtrlHandle callback)
{
    // 对应id模块保存callback，之后rts通过这个callback向该模块传递业务开关和reporter函数指针
    if (modelid == HCCL) {
        // 传递 reporter 函数指针
        rtError_t ret = callback(RT_PROF_CTRL_REPORTER, reinterpret_cast<void*>(Report), 0);

        struct rtProfCommandHandle handle;
        handle.type = PROF_COMMANDHANDLE_TYPE_START;
        // 传递 start 业务开关
        ret = callback(RT_PROF_CTRL_SWITCH, static_cast<void*>(&handle), 0);
    }
    return RT_ERROR_NONE;
}

uint64_t MsprofGetHashId(const char* hashInfo, size_t length) { return 1; }

int32_t MsprofRegTypeInfo(uint16_t level, uint32_t typeId, const char* typeName) { return 0; }

uint64_t MsprofSysCycleTime() { return 1; }

int32_t MsprofRegisterCallback(uint32_t moduleId, ProfCommandHandle handle) { return 0; }

int32_t MsprofReportApi(uint32_t agingFlag, const MsprofApi* api) { return 0; }

int32_t MsprofReportAdditionalInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

int32_t MsprofReportCompactInfo(uint32_t agingFlag, const VOID_PTR data, uint32_t length) { return 0; }

namespace prof_stub {
namespace {
    static std::map<std::string, std::unique_ptr<std::ofstream>> ofile_;
    static std::map<std::string, std::filebuf*> ofileBuff_;
    static uint32_t moduleId_;
    static std::mutex profMutexStub_; // 多环时的线程互斥锁
} // namespace
int ReportDataStub(uint32_t moduleId, const ProfReporterData* data, uint32_t len)
{
    CHK_PRT_RET((moduleId != moduleId_), HCCL_ERROR("moduleId is not inited"), HCCL_E_PARA);
    std::unique_lock<std::mutex> lock(profMutexStub_);

    if (data == nullptr) {
        HCCL_ERROR("invalid data parameter");
        return -1;
    }

    std::string fileName("HCCL-prof-");
    fileName += data->tag;
    fileName += "-";
    fileName += std::to_string(getpid());
    if (ofile_.find(fileName) == ofile_.end()) {
        std::unique_ptr<std::ofstream> ofile
            = std::unique_ptr<std::ofstream>(new std::ofstream(fileName, std::ofstream::out | std::ofstream::binary));

        if (!ofile->is_open()) {
            HCCL_ERROR("oftream file open error");
            return -1;
        }

        std::filebuf* ofileBuf = ofile->rdbuf();

        ofile_[fileName] = std::move(ofile);
        ofileBuff_[fileName] = ofileBuf;
    }

    {
        // std::string ofileContent("");
        // ofileContent += (const char*)(data->data);
        // ofileContent += "\n\r";
        std::streamsize size = ofileBuff_[fileName]->sputn((const char*)data->data, data->dataLen);
        if (size < 0) {
            HCCL_ERROR("sputn error ret[%d]", size);
            return size;
        }

        HCCL_DEBUG("sputn success, size[%d]", size);
    }
    if (remove(fileName.c_str()) != 0) {
        HCCL_WARNING("file[%s] move fail", fileName.c_str());
    }

    return 0;
}

int ReportIntStub(uint32_t moduleId)
{
    moduleId_ = moduleId;
    return HCCL_SUCCESS;
}

int ReportUnintStub(uint32_t moduleId)
{
    CHK_PRT_RET((moduleId != moduleId_), HCCL_ERROR("moduleId is not inited"), HCCL_E_PARA);
    // 全部的ofstream关闭
    for (auto& ofileBuff : ofileBuff_) {
        ofileBuff.second->close();
        ofileBuff.second->pubsync();
    }

    for (auto& ofile : ofile_) {
        ofile.second->close();
    }
    return HCCL_SUCCESS;
}
} // namespace prof_stub

/*
 * 描述:判断是否是目录
 * 参数: fileName -- 文件路径名
 * 返回值:执行成功返回EN_OK(是目录), 执行错误返回EN_ERROR(不是目录), 入参检查错误返回EN_INVALID_PARAM
 */
INT32 mmIsDir(const CHAR* fileName)
{
    if (fileName == NULL) {
        return EN_INVALID_PARAM;
    }
    struct stat fileStat;
    (void)memset_s(&fileStat, sizeof(fileStat), 0, sizeof(fileStat)); /* unsafe_function_ignore: memset */
    INT32 ret = lstat(fileName, &fileStat);
    if (ret < MMPA_ZERO) {
        return EN_ERROR;
    }

    if (!S_ISDIR(fileStat.st_mode)) {
        return EN_ERROR;
    }
    return EN_OK;
}
