/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef CCU_RESPACK_MGR_H
#define CCU_RESPACK_MGR_H

#include "ccu/ccu_res_pack_legacy.h"

namespace Hccl {

class CcuResPackMgr {
public:
    void PrepareAlloc(u32 size);
    void Confirm();
    void Fallback();
    CcuResPack& GetCcuResPack(u32 idx);

private:
    vector<CcuResPack> resPacks;
    u32 unConfirmedNum{0};
};

} // namespace Hccl

#endif // CCU_RESPACK_MGR_H
