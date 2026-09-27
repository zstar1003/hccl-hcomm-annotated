/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef RANK_TABLE_REPORT_MACRO_H
#define RANK_TABLE_REPORT_MACRO_H

#include "../rank_table_info/rank_table_source.h" // RankTableSource 枚举
#include "nlohmann/json.hpp"                      // OBJ.contains / OBJ.begin / OBJ[CONFIG].dump()
#include "exception_util.h"                       // THROW / StringFormat
#include "adapter_error_manager_pub.h"            // RPT_INPUT_ERR

namespace Hccl {

#define TRY_CATCH_THROW_REPORT_ROOTINFO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)             \
    do {                                                                                       \
        try {                                                                                  \
            EXPR;                                                                              \
        } catch (HcclException & e) {                                                          \
            if (!OBJ.contains(CONFIG)) {                                                       \
                std::string keys;                                                              \
                if (OBJ.is_object()) {                                                         \
                    for (auto it = OBJ.begin(); it != OBJ.end(); ++it) {                       \
                        if (!keys.empty())                                                     \
                            keys += ", ";                                                      \
                        keys += it.key();                                                      \
                    }                                                                          \
                }                                                                              \
                RPT_INPUT_ERR(                                                                 \
                    true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}), \
                    std::vector<std::string>({keys, CONFIG, EXPECT}));                         \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));               \
            } else {                                                                           \
                RPT_INPUT_ERR(                                                                 \
                    true, "EI0016", std::vector<std::string>({"value", "variable", "expect"}), \
                    std::vector<std::string>({OBJ[CONFIG].dump(), CONFIG, EXPECT}));           \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));               \
            }                                                                                  \
        } catch (std::exception & e) {                                                         \
            THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                   \
        } catch (...) {                                                                        \
            THROW<EXCEPTION>(StringFormat("Unknown error occurs, %s", MSG.c_str()));           \
        }                                                                                      \
    } while (0)

#define TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)                                 \
    do {                                                                                                       \
        try {                                                                                                  \
            EXPR;                                                                                              \
        } catch (HcclException & e) {                                                                          \
            if (!OBJ.contains(CONFIG)) {                                                                       \
                RPT_INPUT_ERR(                                                                                 \
                    true, "EI0017", std::vector<std::string>({"config"}), std::vector<std::string>({CONFIG})); \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                               \
            } else {                                                                                           \
                RPT_INPUT_ERR(                                                                                 \
                    true, "EI0014", std::vector<std::string>({"value", "variable", "expect"}),                 \
                    std::vector<std::string>({OBJ[CONFIG].dump(), CONFIG, EXPECT}));                           \
                THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                               \
            }                                                                                                  \
        } catch (std::exception & e) {                                                                         \
            THROW<EXCEPTION>(StringFormat("%s, %s", e.what(), MSG.c_str()));                                   \
        } catch (...) {                                                                                        \
            THROW<EXCEPTION>(StringFormat("Unknown error occurs, %s", MSG.c_str()));                           \
        }                                                                                                      \
    } while (0)

#define TRY_CATCH_THROW_REPORT(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT, SOURCE)       \
    do {                                                                                \
        if (SOURCE == RankTableSource::ROOTINFO) {                                      \
            TRY_CATCH_THROW_REPORT_ROOTINFO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT); \
        } else {                                                                        \
            TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT);     \
        }                                                                               \
    } while (0)

#define TRY_CATCH_THROW_REPORT_TOPO(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT) \
    TRY_CATCH_THROW_REPORT_JSON(EXCEPTION, MSG, EXPR, OBJ, CONFIG, EXPECT)

} // namespace Hccl

#endif // RANK_TABLE_REPORT_MACRO_H
