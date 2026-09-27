/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef DB_HCCL_DB_FACTORY_H
#define DB_HCCL_DB_FACTORY_H

#include <functional>
#include <memory>
#include <unordered_map>

#include "db_hccl_db.h"

namespace HcclSim {
namespace DB {
    class HcclDBFactory {
    public:
        using CreatorFunc = std::function<std::unique_ptr<HcclDB>()>;

        static HcclDBFactory& Instance();

        std::unique_ptr<HcclDB> CreateDB(sim::DbType type);

        void RegisterDB(sim::DbType type, CreatorFunc creator);

        bool IsRegistered(sim::DbType type) const;

    private:
        HcclDBFactory();
        ~HcclDBFactory() = default;
        HcclDBFactory(const HcclDBFactory&) = delete;
        HcclDBFactory& operator=(const HcclDBFactory&) = delete;

        void RegisterBuiltInTypes();

        std::unordered_map<sim::DbType, CreatorFunc> m_creators;
    };

    inline HcclDBFactory& HcclDBFactory::Instance()
    {
        static HcclDBFactory instance;
        return instance;
    }

    inline std::unique_ptr<HcclDB> HcclDBFactory::CreateDB(sim::DbType type)
    {
        auto it = m_creators.find(type);
        if (it == m_creators.end()) {
            return nullptr;
        }
        return it->second();
    }

    inline void HcclDBFactory::RegisterDB(sim::DbType type, CreatorFunc creator)
    {
        m_creators[type] = std::move(creator);
    }

    inline bool HcclDBFactory::IsRegistered(sim::DbType type) const
    {
        return m_creators.find(type) != m_creators.end();
    }
} // namespace DB
} // namespace HcclSim

#endif
