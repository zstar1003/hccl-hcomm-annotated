/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_CCU_TRANS_GROUP_H
#define HCCL_CCU_TRANS_GROUP_H

#include <vector>
#include "ccu_transport.h"
#include "ccu_device_manager.h"

namespace Hccl {

MAKE_ENUM(TransportGrpStatus, INIT, FAIL)

class CcuTransportGroup {
public:
    explicit CcuTransportGroup(const vector<CcuTransport*>& transports, u32 cntCkeNum);
    CcuTransportGroup() = delete; // 禁用默认构造函数
    virtual ~CcuTransportGroup();
    void Destroy();
    TransportGrpStatus GetGrpStatus() const;
    HcclResult GetCntCkeId(u32 index, u32& cntCkeId) const;
    bool CheckTransports(const vector<CcuTransport*>& transports);
    HcclResult CheckTransportCntCke();
    const vector<CcuTransport*>& GetTransports() const;

private:
    vector<CcuTransport*> transportsGrp;
    vector<u32> cntCkesGroup;
    u32 cntCkesGroupDieId{0};
    u32 cntCkeNumTransportGroupUse{0};
    vector<CcuResHandle> cntResHandleTransportGroupUse;
    vector<ResInfo> ckeInfoTransportGroupUse;
    bool isDestroyed{false};
    TransportGrpStatus grpStatus;
    int32_t devLogicId{0};
};

} // namespace Hccl

#endif // HCCL_CCU_TRANS_GROUP_H
