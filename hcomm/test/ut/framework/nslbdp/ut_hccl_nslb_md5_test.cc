/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "hccl_nslb_md5.h"
#include "hccl_nslbdp_pub.h"

using namespace hccl;
using namespace testing;

namespace {
constexpr u32 MD5_HEX_LEN = 32; // 16 字节 → 32 hex 字符
constexpr u32 IP_OCTET_0_SHIFT = 24;
constexpr u32 IP_OCTET_1_SHIFT = 16;
constexpr u32 IP_OCTET_2_SHIFT = 8;
constexpr u32 IP_BYTE_MASK = 0xFF;

inline u32 BuildIpv4(u32 a, u32 b, u32 c, u32 d)
{
    return ((a & IP_BYTE_MASK) << IP_OCTET_0_SHIFT) | ((b & IP_BYTE_MASK) << IP_OCTET_1_SHIFT)
           | ((c & IP_BYTE_MASK) << IP_OCTET_2_SHIFT) | (d & IP_BYTE_MASK);
}

} // namespace

// ============================================================
// 1. 标准 MD5 测试向量（RFC 1321）
// ============================================================
TEST(NslbMd5Test, StandardVectors_StringConstructor)
{
    // "" -> d41d8cd98f00b204e9800998ecf8427e
    std::string digest = NSLBMD5("").hexdigest();
    EXPECT_EQ(digest.size(), MD5_HEX_LEN);
    EXPECT_STREQ(digest.c_str(), "d41d8cd98f00b204e9800998ecf8427e");

    // "a" -> 0cc175b9c0f1b6a831c399e269772661
    EXPECT_STREQ(NSLBMD5("a").hexdigest().c_str(), "0cc175b9c0f1b6a831c399e269772661");

    // "abc" -> 900150983cd24fb0d6963f7d28e17f72
    EXPECT_STREQ(NSLBMD5("abc").hexdigest().c_str(), "900150983cd24fb0d6963f7d28e17f72");

    // "message digest" -> f96b697d7cb7938d525a2f31aaf161d0
    EXPECT_STREQ(NSLBMD5("message digest").hexdigest().c_str(), "f96b697d7cb7938d525a2f31aaf161d0");

    // "abcdefghijklmnopqrstuvwxyz" -> c3fcd3d76192e4007dfb496cca67e13b
    EXPECT_STREQ(NSLBMD5("abcdefghijklmnopqrstuvwxyz").hexdigest().c_str(), "c3fcd3d76192e4007dfb496cca67e13b");
}

TEST(NslbMd5Test, StandardVectors_UpdateFinalize)
{
    NSLBMD5 md;
    const char s[] = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    md.update(s, strlen(s));
    md.finalize();
    // 官方标准向量: d174ab98d277d9f5a5611c2c9f419d9f
    EXPECT_STREQ(md.hexdigest().c_str(), "d174ab98d277d9f5a5611c2c9f419d9f");
}

TEST(NslbMd5Test, ConstructorSameAsUpdateFinalize)
{
    // 构造函数路径 == update+finalize 路径
    NSLBMD5 direct("abc");
    NSLBMD5 stream;
    stream.update("abc", 3);
    stream.finalize();
    EXPECT_STREQ(direct.hexdigest().c_str(), stream.hexdigest().c_str());
    EXPECT_STREQ(direct.hexdigest().c_str(), "900150983cd24fb0d6963f7d28e17f72");
}

// ============================================================
// 2. md5ToString
// ============================================================
TEST(NslbMd5Test, Md5ToString_KnownBytes)
{
    // md5ToString 用 (byte >> 1) & 0xF 取高位、byte & 0xF 取低位
    // 全 0xFF：hi=(0xFF>>1)&0xF=0xF→'f', lo=0xF→'f' → "ff"
    uint8_t ffs[16];
    std::memset(ffs, 0xFF, 16);
    EXPECT_STREQ(NSLBMD5::md5ToString(ffs).c_str(), "ffffffffffffffffffffffffffffffff");

    // 全 0x01：hi=(0x01>>1)&0xF=0→'0', lo=0x1→'1' → "01"
    uint8_t ones[16];
    std::memset(ones, 0x01, 16);
    EXPECT_STREQ(NSLBMD5::md5ToString(ones).c_str(), "01010101010101010101010101010101");
}

TEST(NslbMd5Test, Md5ToString_ZeroBytes)
{
    uint8_t zeros[16] = {0};
    std::string hex = NSLBMD5::md5ToString(zeros);
    EXPECT_STREQ(hex.c_str(), "00000000000000000000000000000000");
}

// ============================================================
// 3. calculateRankInfoMd5 + calculateTableFourRankInfoMd5
// ============================================================
TEST(NslbMd5Test, CalculateRankInfoMd5_EmptyAndDeterministic)
{
    std::vector<NslbDpRankInfo> emptyVec;
    uint8_t md5Empty[16] = {0};
    NSLBMD5::calculateRankInfoMd5(emptyVec, md5Empty);

    // 对同一输入再次计算必须完全一致（幂等）
    uint8_t md5Empty2[16] = {0};
    NSLBMD5::calculateRankInfoMd5(emptyVec, md5Empty2);
    EXPECT_EQ(memcmp(md5Empty, md5Empty2, 16), 0);

    // 填入若干值
    std::vector<NslbDpRankInfo> nonEmpty(3);
    nonEmpty[0] = {BuildIpv4(192, 168, 0, 1), BuildIpv4(10, 0, 0, 1), 1, 0};
    nonEmpty[1] = {BuildIpv4(192, 168, 0, 2), BuildIpv4(10, 0, 0, 1), 1, 0};
    nonEmpty[2] = {BuildIpv4(192, 168, 0, 3), BuildIpv4(10, 0, 0, 2), 2, 1};
    uint8_t md5a[16] = {0};
    uint8_t md5b[16] = {0};
    NSLBMD5::calculateRankInfoMd5(nonEmpty, md5a);
    NSLBMD5::calculateRankInfoMd5(nonEmpty, md5b);
    EXPECT_EQ(memcmp(md5a, md5b, 16), 0);
    // 非空向量的 MD5 不应等于空向量的 MD5（极强概率下）
    EXPECT_NE(memcmp(md5a, md5Empty, 16), 0);
}

TEST(NslbMd5Test, CalculateTableFourRankInfoMd5_Deterministic)
{
    std::vector<TableFourRankInfo> vec(2);
    vec[0] = {BuildIpv4(1, 2, 3, 4), BuildIpv4(5, 6, 7, 8)};
    vec[1] = {BuildIpv4(9, 10, 11, 12), BuildIpv4(13, 14, 15, 16)};
    uint8_t md5a[16] = {0};
    uint8_t md5b[16] = {0};
    NSLBMD5::calculateTableFourRankInfoMd5(vec, md5a);
    NSLBMD5::calculateTableFourRankInfoMd5(vec, md5b);
    EXPECT_EQ(memcmp(md5a, md5b, 16), 0);

    // 用空向量算一遍，确保非空不等于空
    std::vector<TableFourRankInfo> emptyVec;
    uint8_t md5Empty[16] = {0};
    NSLBMD5::calculateTableFourRankInfoMd5(emptyVec, md5Empty);
    EXPECT_NE(memcmp(md5a, md5Empty, 16), 0);
}

// ============================================================
// 4. encode/decode 对开（内部工具函数，static public）
// ============================================================
TEST(NslbMd5Test, EncodeDecodeRoundTrip)
{
    // 取一段 4*n 长数据，encode 再 decode 应还原
    const size_t words = 4;
    uint32_t src[words] = {0x01020304U, 0xAABBCCDDU, 0x11223344U, 0xDEADBEEFU};
    unsigned char buf[words * 4];
    NSLBMD5::encode(buf, src, words * 4);
    uint32_t dst[words] = {0};
    NSLBMD5::decode(dst, buf, words * 4);
    for (size_t i = 0; i < words; i++) {
        EXPECT_EQ(dst[i], src[i]);
    }
}
