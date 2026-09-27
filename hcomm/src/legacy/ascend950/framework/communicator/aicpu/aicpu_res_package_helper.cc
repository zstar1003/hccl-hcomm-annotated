/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "aicpu_res_package_helper.h"
#include "binary_stream.h"
#include "log.h"
namespace Hccl {

BinaryStream& operator<<(BinaryStream& binaryStream, const ModuleData& m)
{
    binaryStream << m.name;
    binaryStream << m.data;

    HCCL_INFO("BinaryStream: packed Data name %s", m.name);

    return binaryStream;
}

BinaryStream& operator>>(BinaryStream& binaryStream, ModuleData& m)
{
    binaryStream >> m.name;
    binaryStream >> m.data;

    HCCL_INFO("BinaryStream: unpacked Data name %s", m.name);

    return binaryStream;
}

std::vector<char> AicpuResPackageHelper::GetPackedData(std::vector<ModuleData>& dataVec) const
{
    std::vector<char> result;
    BinaryStream binaryStream;
    binaryStream << dataVec;
    binaryStream.Dump(result);

    return result;
}

std::vector<ModuleData> AicpuResPackageHelper::ParsePackedData(std::vector<char>& data) const
{
    std::vector<ModuleData> result;
    BinaryStream binaryStream(data);
    binaryStream >> result;

    return result;
}

} // namespace Hccl
