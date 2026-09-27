/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef MEM_NAME_REPOSITORY_PUB_H
#define MEM_NAME_REPOSITORY_PUB_H

#include <map>
#include <memory>
#include <string>
#include <mutex>
#include <securec.h>
#include "log.h"
#include "hccl_common.h"

namespace hccl {
/* IPC 相关 */
constexpr u32 HCCL_IPC_MEM_NAME_LEN = 65;

using SecIpcName_t = struct tagSecIpcName {
    u8 ipcName[HCCL_IPC_MEM_NAME_LEN] = {0};
    // 析构时,自动将内存数据清零
    ~tagSecIpcName()
    {
        s32 ret = memset_s(ipcName, HCCL_IPC_MEM_NAME_LEN, 0, HCCL_IPC_MEM_NAME_LEN);
        if (ret != EOK) {
            HCCL_ERROR(
                "[Destroy][TagSecIpcName]errNo[0x%016llx] In SecIpcName, memset_s failed. "
                "errorno[%d]",
                HCCL_ERROR_CODE(HCCL_E_SYSCALL), ret);
        }
    }
};

struct IpcMemInfo {
    inline bool operator<(const IpcMemInfo& that) const
    {
        std::string strThat = std::to_string(reinterpret_cast<uintptr_t>(that.ptr)) + std::to_string(that.size)
                              + std::to_string(that.isSioToHccs);
        std::string strThis = std::to_string(reinterpret_cast<uintptr_t>(this->ptr)) + std::to_string(this->size)
                              + std::to_string(this->isSioToHccs);
        return (strThis < strThat);
    }
    void* ptr;
    u64 size;
    bool isSioToHccs; // sio和hccs并发时，同一块内存需要多次set ipc，通过标记区分
};

class MemNameRepository {
public:
    ~MemNameRepository();
    static MemNameRepository* GetInstance(s32 deviceLogicID);
    // 设置一个ipc mem, 并且返回ipc 名字
    HcclResult SetIpcMem(void* ptr, u64 size, u8* name, u32 nameLen, u64& offset, bool isSioToHccs = false);

    HcclResult SetIpcMem(void* ptr, u64 size, u8* name, u32 nameLen);

    HcclResult SetIpcMem(
        void* ptr, u64 size, u8* name, u32 nameLen, u64& offset, s32 pid, s32 sdid = INVALID_INT,
        bool isSioToHccs = false);

    // 抽离SetIpcMem的公共逻辑，在SetNameMap中查找MemName,若未找到则插入
    HcclResult FindIpcMem(IpcMemInfo& ipcMemInfo, u8* name, u32 nameLen);
    // 打开ipc nem
    HcclResult
    OpenIpcMem(void** ptr, u64 size, const u8* name, u32 nameLen, u64 offset, bool& isOpened, bool isSioToHccs = false);

    // 关闭ipc mem
    void CloseIpcMem(const u8* name);

    // 销毁ipc mem
    void DestroyIpcMem(void* ptr, u64 size, bool isSioToHccs = false);

    // 清空map
    void ClearMemNameRepository();

    HcclResult SetDeviceUnavailable(bool unavailable);

private:
    std::map<IpcMemInfo, SecIpcName_t> setNameMap_; // 记录用于input/output和mem name对应关系，由link模块填充
    std::map<IpcMemInfo, SecIpcName_t> openedNameMap_; // 记录已打开的mem name 与ptr对应关系
    std::map<IpcMemInfo, Referenced> setNameMapRef_;   // 记录用于input/output和mem name对应关系的引用计数
    std::map<IpcMemInfo, Referenced> openedNameMapRef_; // 记录已打开的mem name 与ptr对应关系的引用计数
    std::map<IpcMemInfo, IpcMemInfo> alignPtrMap_;      // 记录根据页表对齐前后的ipc mem对应关系
    std::mutex memMutex_;
    bool unavailable_{false};

    // 清空map不加锁
    void ClearMemNameRepositoryImpl();
};
} // namespace hccl

#endif /* __MEM_NAME_REPOSITRY_PUB_H__ */
