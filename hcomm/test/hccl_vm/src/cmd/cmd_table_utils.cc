/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>
#include <sys/wait.h>
#include <thread>
#include <vector>

#include "cmd_base_utils.h"
#include "sim_common_defs.h"
#include "sim_log.h"
#include "db_sim_runner_db.h"

namespace HcclSim {
template <typename T>
static void
PrintTable(const std::string& header, const std::vector<T>& rows, std::function<std::string(const T&)> formatter)
{
    std::cout << header << std::endl;
    for (const auto& row : rows) {
        std::cout << formatter(row) << std::endl;
    }
    std::cout << std::endl;
}

void CmdTableShow(std::string& tableName)
{
    if (tableName == "Device") {
        auto tables = RunnerDB::GetByPred<sim::Device>([](auto&&) {
            return true;
        });
        PrintTable<sim::Device>(
            "| id | server_id | user_id | logic_id | physical_id | super_device_id | overflow_mode | soc_version | "
            "status |",
            tables, [](const sim::Device& d) {
                return "| " + std::to_string(d.id) + " | " + std::to_string(d.server_id) + " | "
                       + std::to_string(d.user_id) + " | " + std::to_string(d.logic_id) + " | "
                       + std::to_string(d.physical_id) + " | " + std::to_string(d.super_device_id) + " | "
                       + std::to_string(d.overflow_mode) + " | " + std::string(d.soc_version) + " | "
                       + std::to_string(d.status) + " |";
            });
    } else if (tableName == "Server") {
        auto tables = RunnerDB::GetByPred<sim::Server>([](auto&&) {
            return true;
        });
        PrintTable<sim::Server>("| id | pod_id | version |", tables, [](const sim::Server& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.pod_id) + " | " + std::string(tmp.version)
                   + " |";
        });
    } else if (tableName == "Host") {
        auto tables = RunnerDB::GetByPred<sim::Host>([](auto&&) {
            return true;
        });
        PrintTable<sim::Host>("| id | server_id | ip | arch |", tables, [](const sim::Host& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | "
                   + std::string(tmp.ip_addr) + " | " + std::to_string(tmp.arch) + " |";
        });
    } else if (tableName == "Runner") {
        auto tables = RunnerDB::GetByPred<sim::Runner>([](auto&&) {
            return true;
        });
        PrintTable<sim::Runner>(
            "| id | host_id | pid | thread_id | timeout_config_ms | current_ctx_id |", tables,
            [](const sim::Runner& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.host_id) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.thread_id) + " | "
                       + std::to_string(tmp.timeout_config_ms) + " | " + std::to_string(tmp.current_ctx_id) + " |";
            });
    } else if (tableName == "TaskSchedulerDevice") {
        auto tables = RunnerDB::GetByPred<sim::TaskSchedulerDevice>([](auto&&) {
            return true;
        });
        PrintTable<sim::TaskSchedulerDevice>(
            "| id | device_id | type|", tables, [](const sim::TaskSchedulerDevice& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.type) + " |";
            });
    } else if (tableName == "ComputeDie") {
        auto tables = RunnerDB::GetByPred<sim::ComputeDie>([](const sim::ComputeDie&) {
            return true;
        });
        PrintTable<sim::ComputeDie>("| id | ts_id | type |", tables, [](const sim::ComputeDie& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ts_id) + " | " + std::to_string(tmp.type)
                   + " |";
        });
    } else if (tableName == "DeviceStatus") {
        auto tables = RunnerDB::GetByPred<sim::DeviceStatus>([](const sim::DeviceStatus&) {
            return true;
        });
        PrintTable<sim::DeviceStatus>(
            "| id | device_id | overflow | sync_strat | sync_timeout | capability | run_by_host | ts_core | online |",
            tables, [](const sim::DeviceStatus& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.overflow_status) + " | " + std::to_string(tmp.synchronize_strategy) + " | "
                       + std::to_string(tmp.synchronize_timeout) + " | " + std::to_string(tmp.capability_mask) + " | "
                       + std::to_string(tmp.run_by_host) + " | " + std::to_string(tmp.ts_core) + " | "
                       + std::to_string(tmp.online_status) + " |";
            });
    } else if (tableName == "Port") {
        auto tables = RunnerDB::GetByPred<sim::Port>([](const sim::Port&) {
            return true;
        });
        PrintTable<sim::Port>("| id | device_id | die_id | name | status |", tables, [](const sim::Port& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                   + std::to_string(tmp.die_id) + " | " + std::string(tmp.name) + " | " + std::to_string(tmp.status)
                   + "|";
        });
    } else if (tableName == "Ccu") {
        auto tables = RunnerDB::GetByPred<sim::Ccu>([](const sim::Ccu&) {
            return true;
        });
        PrintTable<sim::Ccu>("| id | device_id | resource_addr | die_id | status |", tables, [](const sim::Ccu& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                   + std::to_string(tmp.resource_addr) + " | " + std::to_string(tmp.die_id) + " | "
                   + std::to_string(tmp.status) + " |";
        });
    } else if (tableName == "CcuResource") {
        auto tables = RunnerDB::GetByPred<sim::CcuResource>([](const sim::CcuResource&) {
            return true;
        });
        PrintTable<sim::CcuResource>("| id | ccu_id | instr_cnt | state |", tables, [](const sim::CcuResource& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ccu_id) + " | "
                   + std::to_string(tmp.instr_cnt) + " | " + std::to_string(tmp.state) + " | ";
        });
    } else if (tableName == "DeviceConnection") {
        auto tables = RunnerDB::GetByPred<sim::DeviceConnection>([](const sim::DeviceConnection&) {
            return true;
        });
        PrintTable<sim::DeviceConnection>(
            "| id | src_dev_id | dst_dev_id | link_type | access_by_remote |", tables,
            [](const sim::DeviceConnection& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.src_dev_id) + " | "
                       + std::to_string(tmp.dst_dev_id) + " | " + std::to_string(tmp.link_type) + " | "
                       + std::to_string(tmp.access_by_remote) + " |";
            });
    } else if (tableName == "EndPoint") {
        auto tables = RunnerDB::GetByPred<sim::EndPoint>([](const sim::EndPoint&) {
            return true;
        });
        PrintTable<sim::EndPoint>(
            "| id | device_id | func_id | die_id | type | eid | ip_addr |", tables, [](const sim::EndPoint& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.func_id) + " | " + std::to_string(tmp.die_id) + " | "
                       + std::to_string(tmp.type) + " | " + std::string(reinterpret_cast<const char*>(tmp.eid)) + " | "
                       + std::string(tmp.ip_addr) + " |";
            });
    } else if (tableName == "EndPointPair") {
        auto tables = RunnerDB::GetByPred<sim::EndPointPair>([](const sim::EndPointPair&) {
            return true;
        });
        PrintTable<sim::EndPointPair>(
            "| id | local_enpoint_id | remote_enpoint_id | tp_type |", tables, [](const sim::EndPointPair& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_enpoint_id) + " | "
                       + std::to_string(tmp.remote_enpoint_id) + " | " + std::to_string(tmp.tp_type) + " |";
            });
    } else if (tableName == "EndPointPortMapping") {
        auto tables = RunnerDB::GetByPred<sim::EndPointPortMapping>([](const sim::EndPointPortMapping&) {
            return true;
        });
        PrintTable<sim::EndPointPortMapping>(
            "| id | port_id | endpoint_id | net_layer |", tables, [](const sim::EndPointPortMapping& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.port_id) + " | "
                       + std::to_string(tmp.endpoint_id) + " | " + std::to_string(tmp.net_layer) + " |";
            });
    } else if (tableName == "Link") {
        auto tables = RunnerDB::GetByPred<sim::Link>([](const sim::Link&) {
            return true;
        });
        PrintTable<sim::Link>(
            "| id | local_endpoint_id | remote_endpoint_id | net_layer | type | protocols |", tables,
            [](const sim::Link& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_endpoint_id) + " | "
                       + std::to_string(tmp.remote_endpoint_id) + " | " + std::to_string(tmp.net_layer) + " | "
                       + std::to_string(tmp.type) + " | " + std::to_string(tmp.protocols[0]) + " |";
            });
    } else if (tableName == "CcuChannel") {
        auto tables = RunnerDB::GetByPred<sim::CcuChannel>([](const sim::CcuChannel&) {
            return true;
        });
        PrintTable<sim::CcuChannel>(
            "| id | channel_id | local_endpoint_id | remote_endpoint_id | protocol | jetty_start | jetty_num |", tables,
            [](const sim::CcuChannel& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.channel_id) + " | "
                       + std::to_string(tmp.local_endpoint_id) + " | " + std::to_string(tmp.remote_endpoint_id) + " | "
                       + std::to_string(tmp.protocol) + " | " + std::to_string(tmp.jetty_start) + " | "
                       + std::to_string(tmp.jetty_num) + " |";
            });
    } else if (tableName == "Context") {
        auto tables = RunnerDB::GetByPred<sim::Context>([](const sim::Context&) {
            return true;
        });
        PrintTable<sim::Context>(
            "| id | run_id | device_id | is_default | ref_cnt | float_overflow_addr | capture_mode |", tables,
            [](const sim::Context& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.run_id) + " | "
                       + std::to_string(tmp.device_id) + " | " + std::to_string(tmp.is_default) + " | "
                       + std::to_string(tmp.ref_cnt) + " | " + std::to_string(tmp.float_overflow_addr) + " | "
                       + std::to_string(tmp.capture_mode) + " |";
            });
    } else if (tableName == "Stream") {
        auto tables = RunnerDB::GetByPred<sim::Stream>([](const sim::Stream&) {
            return true;
        });
        PrintTable<sim::Stream>(
            "| id | ctx_id | sq_base_addr | is_primary_default | is_other_default | priority | schedule_strategy | "
            "failure_mode | user_tag | overflow_switch | activated | capture_status | task_complete_status |",
            tables, [](const sim::Stream& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_id) + " | "
                       + std::to_string(tmp.sq_base_addr) + " | " + std::to_string(tmp.is_primary_default) + " | "
                       + std::to_string(tmp.is_other_default) + " | " + std::to_string(tmp.priority) + " | "
                       + std::to_string(tmp.schedule_strategy) + " | " + std::to_string(tmp.failure_mode) + " | "
                       + std::to_string(tmp.user_tag) + " | " + std::to_string(tmp.overflow_switch) + " | "
                       + std::to_string(tmp.activated) + " | " + std::to_string(tmp.capture_status) + " | "
                       + std::to_string(tmp.task_complete_status) + " |";
            });
    } else if (tableName == "Task") {
        auto tables = RunnerDB::GetByPred<sim::Task>([](const sim::Task&) {
            return true;
        });
        PrintTable<sim::Task>("| id | stream_id | cid | seq_number | type |", tables, [](const sim::Task& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.stream_id) + " | "
                   + std::to_string(tmp.cid) + " | " + std::to_string(tmp.seq_number) + " | " + std::to_string(tmp.type)
                   + " |";
        });
    } else if (tableName == "EventSyncTask") {
        auto tables = RunnerDB::GetByPred<sim::EventSyncTask>([](const sim::EventSyncTask&) {
            return true;
        });
        PrintTable<sim::EventSyncTask>(
            "| id | excute_time_ms | finish_time_ms | op_timeout_s |", tables, [](const sim::EventSyncTask& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.excute_time_ms) + " | "
                       + std::to_string(tmp.finish_time_ms) + " | " + std::to_string(tmp.op_timeout_s) + " |";
            });
    } else if (tableName == "Notify") {
        auto tables = RunnerDB::GetByPred<sim::Notify>([](const sim::Notify&) {
            return true;
        });
        PrintTable<sim::Notify>(
            "| id | create_ctx_id | device_notify_seq | value |", tables, [](const sim::Notify& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) + " | "
                       + std::to_string(tmp.device_notify_seq) + " | " + std::to_string(tmp.value) + " |";
            });
    } else if (tableName == "IpcNotify") {
        auto tables = RunnerDB::GetByPred<sim::IpcNotify>([](const sim::IpcNotify&) {
            return true;
        });
        PrintTable<sim::IpcNotify>(
            " id | notify_id | name_or_key | create_pid |", tables, [](const sim::IpcNotify& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.notify_id) + " | "
                       + std::string(reinterpret_cast<const char*>(tmp.name_or_key)) + " | "
                       + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "IpcNotifyVistorList") {
        auto tables = RunnerDB::GetByPred<sim::IpcNotifyVistorList>([](const sim::IpcNotifyVistorList&) {
            return true;
        });
        PrintTable<sim::IpcNotifyVistorList>(
            "| ipc_id | visitor_pid |", tables, [](const sim::IpcNotifyVistorList& tmp) {
                return "| " + std::to_string(tmp.ipc_id) + " | " + std::to_string(tmp.visitor_pid) + " |";
            });
    } else if (tableName == "NotifyRecordTask") {
        auto tables = RunnerDB::GetByPred<sim::NotifyRecordTask>([](const sim::NotifyRecordTask&) {
            return true;
        });
        PrintTable<sim::NotifyRecordTask>("| notify_id |", tables, [](const sim::NotifyRecordTask& tmp) {
            return "| " + std::to_string(tmp.notify_id) + " |";
        });
    } else if (tableName == "NotifyWaitTask") {
        auto tables = RunnerDB::GetByPred<sim::NotifyWaitTask>([](const sim::NotifyWaitTask&) {
            return true;
        });
        PrintTable<sim::NotifyWaitTask>("| notify_id |", tables, [](const sim::NotifyWaitTask& tmp) {
            return "| " + std::to_string(tmp.notify_id) + " |";
        });
    } else if (tableName == "Event") {
        auto tables = RunnerDB::GetByPred<sim::Event>([](const sim::Event&) {
            return true;
        });
        PrintTable<sim::Event>(
            "| id | create_ctx_id | event_flag | device_res_seq | created_time | status |", tables,
            [](const sim::Event& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.create_ctx_id) + " | "
                       + std::to_string(tmp.event_flag) + " | " + std::to_string(tmp.device_res_seq) + " | "
                       + std::to_string(tmp.created_time) + " | " + std::to_string(tmp.status) + " |";
            });
    } else if (tableName == "PhyMem") {
        auto tables = RunnerDB::GetByPred<sim::PhyMemBlock>([](const sim::PhyMemBlock&) {
            return true;
        });
        PrintTable<sim::PhyMemBlock>(
            "| id | device_id | name | size | type | ref_count |", tables, [](const sim::PhyMemBlock& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::string(tmp.name) + " | " + std::to_string(tmp.size) + " | " + std::to_string(tmp.type)
                       + " | " + std::to_string(tmp.ref_count) + " |";
            });
    } else if (tableName == "VirMem") {
        auto tables = RunnerDB::GetByPred<sim::VirtualMemBlock>([](const sim::VirtualMemBlock&) {
            return true;
        });
        PrintTable<sim::VirtualMemBlock>(
            "| id | start_ptr | size | ctx_id | phy_mem_id | owner_pid | src_type | policy |", tables,
            [](const sim::VirtualMemBlock& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.start_ptr) + " | "
                       + std::to_string(tmp.size) + " | " + std::to_string(tmp.ctx_id) + " | "
                       + std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.owner_pid) + " | "
                       + std::to_string(tmp.src_type) + " | " + std::to_string(tmp.policy) + " |";
            });
    } else if (tableName == "IpcMemRecord") {
        auto tables = RunnerDB::GetByPred<sim::IpcMemRecord>([](const sim::IpcMemRecord&) {
            return true;
        });
        PrintTable<sim::IpcMemRecord>(
            "| id | vir_mem_id | offset | create_pid |", tables, [](const sim::IpcMemRecord& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | "
                       + std::to_string(tmp.offset) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "IpcMemWhiteList") {
        auto tables = RunnerDB::GetByPred<sim::IpcMemWhiteList>([](const sim::IpcMemWhiteList&) {
            return true;
        });
        PrintTable<sim::IpcMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::IpcMemWhiteList& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemRecord") {
        auto tables = RunnerDB::GetByPred<sim::FdMemRecord>([](const sim::FdMemRecord&) {
            return true;
        });
        PrintTable<sim::FdMemRecord>(
            "| id | vir_mem_id | phy_mem_id | create_pid |", tables, [](const sim::FdMemRecord& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.vir_mem_id) + " | "
                       + std::to_string(tmp.phy_mem_id) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "FdMemWhiteList") {
        auto tables = RunnerDB::GetByPred<sim::FdMemWhiteList>([](const sim::FdMemWhiteList&) {
            return true;
        });
        PrintTable<sim::FdMemWhiteList>(
            "| id | name_or_key | pid | create_pid |", tables, [](const sim::FdMemWhiteList& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.name_or_key) + " | "
                       + std::to_string(tmp.pid) + " | " + std::to_string(tmp.create_pid) + " |";
            });
    } else if (tableName == "RaSocket") {
        auto tables = RunnerDB::GetByPred<sim::RaSocket>([](const sim::RaSocket&) {
            return true;
        });
        PrintTable<sim::RaSocket>(
            "| id | device_id | role | state | endpoint_id |", tables, [](const sim::RaSocket& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.role) + " | " + std::to_string(tmp.state) + " | "
                       + std::to_string(tmp.endpoint_id) + " |";
            });
    } else if (tableName == "RaSocketPair") {
        auto tables = RunnerDB::GetByPred<sim::RaSocketPair>([](const sim::RaSocketPair&) {
            return true;
        });
        PrintTable<sim::RaSocketPair>(
            "| id | server_id | client_id | ref_cnt | port | tag_hash |", tables, [](const sim::RaSocketPair& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.server_id) + " | "
                       + std::to_string(tmp.client_id) + " | " + std::to_string(tmp.ref_cnt) + " | "
                       + std::to_string(tmp.port) + " | " + std::to_string(tmp.tag_hash) + " |";
            });
    } else if (tableName == "MemoryLayout") {
        auto tables = RunnerDB::GetByPred<sim::MemoryLayout>([](const sim::MemoryLayout&) {
            return true;
        });
        PrintTable<sim::MemoryLayout>(
            "| id | rank_id | base_addr | buf_type | reserved | size | global_offset |", tables,
            [](const sim::MemoryLayout& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | "
                       + std::to_string(tmp.base_addr) + " | " + std::to_string(tmp.buf_type) + " | "
                       + std::to_string(tmp.reserved) + " | " + std::to_string(tmp.size) + " | "
                       + std::to_string(tmp.global_offset) + " |";
            });
    } else if (tableName == "SimModelData") {
        auto tables = RunnerDB::GetByPred<sim::SimModelData>([](const sim::SimModelData&) {
            return true;
        });
        PrintTable<sim::SimModelData>(
            "| id | rank_id | src_rank | dst_rank | root | rank_size | chip_type | op_type | reduce_op | data_type | "
            "data_count | op_expansion_mode | ccu0_resource_base_addr | ccu1_resource_base_addr |",
            tables, [](const sim::SimModelData& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | "
                       + std::to_string(tmp.src_rank) + " | " + std::to_string(tmp.dst_rank) + " | "
                       + std::to_string(tmp.root) + " | " + std::to_string(tmp.rank_size) + " | "
                       + std::to_string(tmp.chip_type) + " | " + std::to_string(tmp.op_type) + " | "
                       + std::to_string(tmp.reduce_op) + " | " + std::to_string(tmp.data_type) + " | "
                       + std::to_string(tmp.data_count) + " | " + std::to_string(tmp.op_expansion_mode) + " | "
                       + std::to_string(tmp.ccu0_resource_base_addr) + " | "
                       + std::to_string(tmp.ccu1_resource_base_addr) + " |";
            });
    } else if (tableName == "Rank") {
        auto tables = RunnerDB::GetByPred<sim::Rank>([](const sim::Rank&) {
            return true;
        });
        PrintTable<sim::Rank>("| id | rank_id | device_id |", tables, [](const sim::Rank& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.rank_id) + " | "
                   + std::to_string(tmp.device_id) + " |";
        });
    } else if (tableName == "RaDevice") {
        auto tables = RunnerDB::GetByPred<sim::RaDevice>([](const sim::RaDevice&) {
            return true;
        });
        PrintTable<sim::RaDevice>(
            "| id | device_id | mac_addr | state | endpoint_id |", tables, [](const sim::RaDevice& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::string((char*)tmp.mac_addr) + " | " + std::to_string(tmp.state) + " |"
                       + std::to_string(tmp.endpoint_id) + " |";
            });
    } else if (tableName == "RaQP") {
        auto tables = RunnerDB::GetByPred<sim::RaQP>([](const sim::RaQP&) {
            return true;
        });
        PrintTable<sim::RaQP>(
            "| id | ra_dev_id | send_cq_handle | recv_cq_handle | qp_num | type | state | taJettyId | mode | "
            "peer_qp_id | peer_qpn | perr_lid | pid |",
            tables, [](const sim::RaQP& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ra_dev_id) + " | "
                       + std::to_string(tmp.send_cq_handle) + " | " + std::to_string(tmp.recv_cq_handle) + " | "
                       + std::to_string(tmp.qp_num) + " | " + std::to_string(tmp.type) + " | "
                       + std::to_string(tmp.state) + " | " + std::to_string(tmp.taJettyId) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.peer_qp_id) + " | "
                       + std::to_string(tmp.peer_qpn) + " | " + std::to_string(tmp.perr_lid) + " | "
                       + std::to_string(tmp.pid) + " |";
            });
    } else if (tableName == "RaCQ") {
        auto tables = RunnerDB::GetByPred<sim::RaCQ>([](const sim::RaCQ&) {
            return true;
        });
        PrintTable<sim::RaCQ>("| id | ra_dev_id | cqn | size |", tables, [](const sim::RaCQ& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ra_dev_id) + " | "
                   + std::to_string(tmp.cqn) + " | " + std::to_string(tmp.size) + " |";
        });
    } else if (tableName == "RaCQE") {
        auto tables = RunnerDB::GetByPred<sim::RaCQE>([](const sim::RaCQE&) {
            return true;
        });
        PrintTable<sim::RaCQE>("| id | cq_handle | wr_id | status |", tables, [](const sim::RaCQE& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.cq_handle) + " | "
                   + std::to_string(tmp.wr_id) + " | " + std::to_string(tmp.status) + " |";
        });
    } else if (tableName == "RaMR") {
        auto tables = RunnerDB::GetByPred<sim::RaMR>([](const sim::RaMR&) {
            return true;
        });
        PrintTable<sim::RaMR>(
            "| id | local_key | remote_key | vptr_id | length | addr |", tables, [](const sim::RaMR& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.local_key) + " | "
                       + std::to_string(tmp.remote_key) + " | " + std::to_string(tmp.vptr_id) + " | "
                       + std::to_string(tmp.length) + " | " + std::to_string(tmp.addr) + " |";
            });
    } else if (tableName == "RaContext") {
        auto tables = RunnerDB::GetByPred<sim::RaContext>([](const sim::RaContext&) {
            return true;
        });
        PrintTable<sim::RaContext>(
            "| id | device_id | mode | endpoint_id | eidIndex | max_jetty_num | max_jfc_num |", tables,
            [](const sim::RaContext& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.device_id) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.endpoint_id) + " | "
                       + std::to_string(tmp.eidIndex) + " | " + std::to_string(tmp.max_jetty_num) + " | "
                       + std::to_string(tmp.max_jfc_num) + " |";
            });
    } else if (tableName == "RaChan") {
        auto tables = RunnerDB::GetByPred<sim::RaChan>([](const sim::RaChan&) {
            return true;
        });
        PrintTable<sim::RaChan>("| id | ctx_handle | chann_id | mode |", tables, [](const sim::RaChan& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                   + std::to_string(tmp.chann_id) + " | " + std::to_string(tmp.mode) + " |";
        });
    } else if (tableName == "RaTokenId") {
        auto tables = RunnerDB::GetByPred<sim::RaTokenId>([](const sim::RaTokenId&) {
            return true;
        });
        PrintTable<sim::RaTokenId>("| id | ctx_handle | token_id | ref_count |", tables, [](const sim::RaTokenId& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                   + std::to_string(tmp.token_id) + " | " + std::to_string(tmp.ref_count) + " |";
        });
    } else if (tableName == "RaTp") {
        auto tables = RunnerDB::GetByPred<sim::RaTp>([](const sim::RaTp&) {
            return true;
        });
        PrintTable<sim::RaTp>("| id | ctx_handle | tp_type | tpn | speed | status |", tables, [](const sim::RaTp& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                   + std::to_string(tmp.tp_type) + " | " + std::to_string(tmp.tpn) + " | " + std::to_string(tmp.speed)
                   + " | " + std::to_string(tmp.status) + " |";
        });
    } else if (tableName == "RaRmem") {
        auto tables = RunnerDB::GetByPred<sim::RaRmem>([](const sim::RaRmem&) {
            return true;
        });
        PrintTable<sim::RaRmem>(
            "| id | ctx_handle | remote_key | target_seg_handle | remote_eid |", tables, [](const sim::RaRmem& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.remote_key) + " | " + std::to_string(tmp.target_seg_handle) + " | "
                       + std::to_string(tmp.remote_eid) + " |";
            });
    } else if (tableName == "RaLmem") {
        auto tables = RunnerDB::GetByPred<sim::RaLmem>([](const sim::RaLmem&) {
            return true;
        });
        PrintTable<sim::RaLmem>(
            "| id | ctx_handle | addr | size | mem_key | token_id |", tables, [](const sim::RaLmem& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.addr) + " | " + std::to_string(tmp.size) + " | "
                       + std::to_string(tmp.mem_key) + " | " + std::to_string(tmp.token_id) + " |";
            });
    } else if (tableName == "RaJetty") {
        auto tables = RunnerDB::GetByPred<sim::RaJetty>([](const sim::RaJetty&) {
            return true;
        });
        PrintTable<sim::RaJetty>(
            "| id | ctx_handle | send_cq_handle | recv_cq_handle | sqDepth | rqDepth | type | jetty_id | dieId | state "
            "| mode | peer_jetty_handle | pid |",
            tables, [](const sim::RaJetty& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.send_cq_handle) + " | " + std::to_string(tmp.recv_cq_handle) + " | "
                       + std::to_string(tmp.sqDepth) + " | " + std::to_string(tmp.rqDepth) + " | "
                       + std::to_string(tmp.type) + " | " + std::to_string(tmp.jetty_id) + " | "
                       + std::to_string(tmp.dieId) + " | " + std::to_string(tmp.state) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.peer_jetty_handle) + " | "
                       + std::to_string(tmp.pid) + " |";
            });
    } else if (tableName == "RaJfc") {
        auto tables = RunnerDB::GetByPred<sim::RaJfc>([](const sim::RaJfc&) {
            return true;
        });
        PrintTable<sim::RaJfc>(
            "| id | ctx_handle | jfc_id | depth | mode | policy |", tables, [](const sim::RaJfc& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.ctx_handle) + " | "
                       + std::to_string(tmp.jfc_id) + " | " + std::to_string(tmp.depth) + " | "
                       + std::to_string(tmp.mode) + " | " + std::to_string(tmp.policy) + " |";
            });
    } else if (tableName == "RaCr") {
        auto tables = RunnerDB::GetByPred<sim::RaCr>([](const sim::RaCr&) {
            return true;
        });
        PrintTable<sim::RaCr>(
            "| id | jfc_handle | user_ctx | status | opcode | byte_len |", tables, [](const sim::RaCr& tmp) {
                return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.jfc_handle) + " | "
                       + std::to_string(tmp.user_ctx) + " | " + std::to_string(tmp.status) + " | "
                       + std::to_string(tmp.opcode) + " | " + std::to_string(tmp.byte_len) + " |";
            });
    } else if (tableName == "RaTlv") {
        auto tables = RunnerDB::GetByPred<sim::RaTlv>([](const sim::RaTlv&) {
            return true;
        });
        PrintTable<sim::RaTlv>("| id | physical_id |", tables, [](const sim::RaTlv& tmp) {
            return "| " + std::to_string(tmp.id) + " | " + std::to_string(tmp.physical_id) + " |";
        });
    } else {
        HCCL_VM_ERROR("undefine table {}", tableName);
    }
}

bool CmdTableUpdate(const std::string& table, const uint64_t id, const std::string& column, const std::string& value)
{
    if (table == "Device" && column == "soc_version") {
        RunnerDB::Update<sim::Device>(id, [value](sim::Device& d) {
            memcpy(d.soc_version, value.data(), value.size());
        });
        return true;
    } else {
        HCCL_VM_ERROR("undefine update {} [id={}].{} = \"{}\"", table, id, column, value);
        return false;
    }
}
} // namespace HcclSim
