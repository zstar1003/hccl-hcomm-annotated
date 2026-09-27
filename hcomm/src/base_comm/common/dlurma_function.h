/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SRC_DLURMAFUNCTION_H
#define HCCL_SRC_DLURMAFUNCTION_H

#include <functional>
#include <mutex>
#include <hccl/hccl_types.h>
#include "hccl/base.h"
#include "urma_api.h"

namespace hcomm {
class DlUrmaFunction {
public:
    virtual ~DlUrmaFunction();
    static DlUrmaFunction& GetInstance();
    HcclResult DlUrmaFunctionInit();
    std::function<urma_status_t(urma_jetty_t* jetty, urma_jfs_wr_t* wr, urma_jfs_wr_t** bad_wr)> dlUrmaPostJettySendWr
        = nullptr;
    std::function<int(urma_jfc_t* jfc, int cr_cnt, urma_cr_t* cr)> dlUrmaPollJfc = nullptr;

    bool DlUrmaFunctionIsInit()
    {
        std::lock_guard<std::mutex> lock(handleMutex_);
        return (handle_ != nullptr);
    }

protected:
private:
    void* handle_{nullptr};
    std::mutex handleMutex_;
    DlUrmaFunction(const DlUrmaFunction&);
    DlUrmaFunction& operator=(const DlUrmaFunction&);
    DlUrmaFunction();
    HcclResult DlUrmaFunctionApiInit();
};
} // namespace hcomm

#endif // HCCL_SRC_DLURMAFUNCTION_H
