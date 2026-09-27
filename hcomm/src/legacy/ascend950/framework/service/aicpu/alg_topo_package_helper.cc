/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "alg_topo_package_helper.h"
#include "binary_stream.h"

namespace Hccl {

template <typename U, typename V>
BinaryStream& operator<<(BinaryStream& binaryStream, const std::map<U, V>& m)
{
    size_t mapSize = m.size();
    binaryStream << mapSize;
    for (const auto& mapPair : m) {
        binaryStream << mapPair.first;
        binaryStream << mapPair.second;
    }
    return binaryStream;
}

template <typename U, typename V>
BinaryStream& operator>>(BinaryStream& binaryStream, std::map<U, V>& m)
{
    size_t mapSize;
    binaryStream >> mapSize;
    for (u32 i = 0; i < mapSize; ++i) {
        U k;
        V v;
        binaryStream >> k;
        binaryStream >> v;
        m.emplace(k, v);
    }
    return binaryStream;
}

std::vector<char> AlgTopoPackageHelper::GetPackedData(const AlgTopoInfo& algTopo) const
{
    std::vector<char> result;
    BinaryStream binaryStream;

    binaryStream << algTopo.virtRanks;
    binaryStream << algTopo.virtRankMap;
    binaryStream << algTopo.vTopo;

    binaryStream.Dump(result);
    return result;
}

AlgTopoInfo AlgTopoPackageHelper::GetAlgTopoInfo(std::vector<char>& packedData) const
{
    AlgTopoInfo algTopo;
    BinaryStream binaryStream(packedData);

    binaryStream >> algTopo.virtRanks;
    binaryStream >> algTopo.virtRankMap;
    binaryStream >> algTopo.vTopo;

    return algTopo;
}

} // namespace Hccl
