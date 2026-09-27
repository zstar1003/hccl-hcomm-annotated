/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCLV2_IPC_LOCAL_NOTIFY_H
#define HCCLV2_IPC_LOCAL_NOTIFY_H

#include "local_notify_v2.h"

namespace Hccl {

class IpcLocalNotify : public BaseLocalNotify {
public:
    explicit IpcLocalNotify(bool devUsed = false);

    IpcLocalNotify(const IpcLocalNotify& that) = delete;

    IpcLocalNotify& operator=(const IpcLocalNotify& that) = delete;

    void Wait(const Stream& stream, u32 timeout) const override;

    void Post(const Stream& stream) const override;

    std::unique_ptr<Serializable> GetExchangeDto() override;

    string Describe() const override;

    void Grant(u32 pid) const;

private:
    char_t ipcName[RTS_IPC_MEM_NAME_LEN]{0};
};

} // namespace Hccl
#endif // !HCCLV2_IPC_LOCAL_NOTIFY_H
