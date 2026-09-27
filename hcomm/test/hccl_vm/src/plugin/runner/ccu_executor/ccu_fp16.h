/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef HCCL_SIM_CCU_FP16_H
#define HCCL_SIM_CCU_FP16_H
#include <cmath>
#include <ostream>
#include <string>

#include "hccl/base.h"
#include "hccl/hccl_types.h"

typedef unsigned short half;
typedef unsigned short ushort;
typedef unsigned int uint;

class FP16 {
public:
    half value; // 存储半精度浮点数的值

    FP16() { value = float_to_half(0.0f); }

    FP16(float x) { value = float_to_half(x); }

    FP16(half x) { value = x; }

    FP16(int x) { value = float_to_half(static_cast<float>(x)); }

    float to_float() const { return half_to_float(value); }

    // fp16 加法运算符重载：返回两个 FP16 对象的和
    FP16 operator+(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return FP16(a + b); // 返回 FP16 对象
    }

    FP16 operator*(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return FP16(a * b); // 返回乘积的 FP16 对象
    }

    FP16 operator-(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return FP16(a - b); // 返回和的 FP16 对象
    }

    FP16& operator+=(const FP16& other)
    {
        float a = this->to_float();
        float b = other.to_float();
        this->value = FP16(a + b).value;
        return *this; // 返回当前对象的引用，以支持链式调用
    }

    // fp16 大于运算符重载
    bool operator>(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return a > b; // 返回比较结果
    }

    // fp16 小于运算符重载
    bool operator<(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return a < b; // 返回比较结果
    }

    // fp16 等于运算符重载
    bool operator==(const FP16& other) const
    {
        float a = this->to_float();
        float b = other.to_float();
        return a == b; // 返回比较结果
    }

    static FP16 fabs(const FP16& val) { return FP16(std::fabs(val.to_float())); }

    friend std::ostream& operator<<(std::ostream& os, const FP16& obj)
    {
        os << obj.to_float(); // 输出对应的 float 值
        return os;
    }

private:
    // 辅助函数：将半精度浮点数转换为单精度浮点数
    static float half_to_float(half x)
    {
        const uint e = (x & 0x7C00) >> 10;
        const uint m = (x & 0x03FF) << 13;
        const uint v = as_uint((float)m) >> 23;

        return as_float(
            ((x & 0x8000) << 16) | ((e != 0) * (((e + 112) << 23) | m))
            | (((e == 0) & (m != 0)) * (((v - 37) << 23) | ((m << (150 - v)) & 0x007FE000))));
    }

    // 辅助函数：将单精度浮点数转换为半精度浮点数
    static half float_to_half(float x)
    {
        suf32 in;
        in.f = x;
        unsigned sign = in.u & 0x80000000;
        in.u ^= sign; // 清除符号位
        ushort w;
        if (in.u >= 0x47800000) {
            w = (ushort)(in.u > 0x7f800000 ? 0x7e00 : 0x7c00); // NaN 或 Infinity
        } else {
            if (in.u < 0x38800000) {
                in.f += 0.5f;                    // 进行舍入
                w = (ushort)(in.u - 0x3f000000); // 对小数部分进行精度截断
            } else {
                unsigned t = in.u + 0xc8000fff; // 进行四舍五入
                w = (ushort)((t + ((in.u >> 13) & 1)) >> 13);
            }
        }
        w = (ushort)(w | (sign >> 16));
        return w;
    }

    // 辅助函数：将（unsigned int）转换为浮点数
    static uint as_uint(const float x) { return *(uint*)&x; }

    // 辅助函数：将无符号整数转换为浮点数
    static float as_float(const uint x) { return *(float*)&x; }

    typedef union suf32 {
        int i;      // 整型
        unsigned u; // 无符号整型
        float f;    // 浮点型
    } suf32;
};

#endif
