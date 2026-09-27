/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "base/err_msg.h"
#include <string>
#include <stdio.h>
namespace {

const std::string hcomm_g_msg = R"(
{
    "error_info_list": [
    {
      "errClass": "HCCL Errors",
      "errTitle": "Config_Error_Invalid_Environment_Variable",
      "ErrCode": "EI0001",
      "ErrMessage": "Value %s for environment variable %s is invalid. Expected value: %s.",
      "Arglist": "value,env,expect",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Timeout",
      "ErrCode": "EI0002",
      "ErrMessage": "A timeout occurs when the Notify register waits for execution. Waiting peer rank: %s; task information: %s; communication operator information: %s; communicator: %s.",
      "Arglist": "remote_rankid, base_information, task_information, group_rank_content",
      "suggestion": {
        "Possible Cause": "1. An exception occurs during the execution on some NPUs in the cluster. As a result, collective communication operation failed.\r\n2. The execution speed on some NPU in the cluster is too slow to complete a communication operation within the timeout interval. (The default timeout interval is 1800s. You can set the interval by using HCCL_EXEC_TIMEOUT.)\r\n3. The number of training samples of each NPU is inconsistent.\r\n4. Packet loss or other connectivity problems occur on the communication link.",
        "Solution": "1. If this error is reported on only some ranks, check other ranks for earlier errors and investigate the first reported error.\r\n2. If this error is reported on all ranks, verify that the collective operation type, data count and data type are consistent across all ranks.\r\n3. Check whether the error reporting time difference between ranks exceeds HCCL_EXEC_TIMEOUT (1800s by default). If so, investigate the execution gap between ranks or increase HCCL_EXEC_TIMEOUT if necessary.\r\n4. Check for CQE errors in the plog (grep -rn 'error cqe'). If so, check the network connection status.\r\n5. For detailed troubleshooting guidance, search for the keyword \"EI0002\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Invalid_Argument_Collective_Communication_Operator",
      "ErrCode": "EI0003",
      "ErrMessage": "Failed to verify parameters of operator %s. Value %s for parameter %s is invalid. The expected value is %s.",
      "Arglist": "ccl_op,value,parameter,expect",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "File_Operation_Error_Parse",
      "ErrCode": "EI0004",
      "ErrMessage": "Failed to parse the ranktable file %s. Reason: %s",
      "Arglist": "ranktable_path, error_reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors ",
      "errTitle": "Invalid_Argument",
      "ErrCode": "EI0005",
      "ErrMessage": "The parameters of operator %s are inconsistent between ranks, parameter %s is %s on the local rank and %s on the remote rank.",
      "Arglist": "ccl_op,para_name,local_para,remote_para",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "For detailed troubleshooting guidance, search for the keyword \"EI0005\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Get_Socket",
      "ErrCode": "EI0006",
      "ErrMessage": "Getting socket times out. Reason: %s",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Resource_Error",
      "ErrCode": "EI0007",
      "ErrMessage": "Failed to allocate resource %s with info %s. Reason: Resources are exhausted.",
      "Arglist": "resource_type, resource_info",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Package_Error_Incorrect_HCCL_Version",
      "ErrCode": "EI0008",
      "ErrMessage": "The %s versions are inconsistent. The local %s, while the remote %s.",
      "Arglist": "inconsistent_package, local_version, remote_version",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Install the same version. The local Toolkit (cann-hcomm) and ops (cann-hccl) versions must be consistent with the corresponding remote versions."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Initialize_Transport",
      "ErrCode": "EI0009",
      "ErrMessage": "Device %s transport init error. Reason: %s.",
      "Arglist": "device_id,reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Use the following hccn_tool commands to check whether the port link is down. (The scope of i represents the number of NPUs of each node. 8 is used as an example.)\n1. Check whether the optical module is in position: for i in {0..7}; do hccn_tool -i $i -optical -g; done | grep present.\n2. Check whether the IP address is configured: for i in {0..7}; do hccn_tool -i $i -ip -g; done.\n3. Check whether the switch is connected: for i in {0..7}; do hccn_tool -i $i -lldp -g; done."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_P2P",
      "ErrCode": "EI0010",
      "ErrMessage": "P2P communication failed. Reason: %s",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Ensure that the NPU card is normal and enter environment variable 'export HCCL_INTRA_ROCE_ENABLE=1'."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Resource_Error_Insufficient_Device_Memory",
      "ErrCode": "EI0011",
      "ErrMessage": "Failed to allocate %s bytes of NPU memory.",
      "Arglist": "memory_size",
      "suggestion": {
        "Possible Cause": "Allocation failure due to insufficient NPU memory.",
        "Solution": "Stop unnecessary processes and ensure the required memory is available. For detailed troubleshooting guidance, search for the keyword \"EI0011\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Execution_Error_SDMA",
      "ErrCode": "EI0012",
      "ErrMessage": "SDMA memory copy task exception occurred. Remote rank: %s. Base information: %s. Task information: %s. Communicator information: %s.",
      "Arglist": "remote_rankid, base_information, task_information, group_rank_content",
      "suggestion": {
        "Possible Cause": "1. Network connection exception occurred during the SDMA task execution.\n2. The peer process exits abnormally.\n3. The input or output memory address is not allocated, the actual allocated size is smaller than the input data size, or the memory is freed before the operator execution is complete.",
        "Solution": "1. Check whether the network link is abnormal during the execution.\n2. Check whether a process in the cluster exits before an error is reported. If yes, locate the cause of the process exit.\n3. Check whether the input/output memory size is correct and whether the memory or communicator is released prematurely.\n4. For detailed troubleshooting guidance, search for the keyword \"EI0012\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Execution_Error_ROCE_CQE",
      "ErrCode": "EI0013",
      "ErrMessage": "An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.",
      "Arglist": "localServerId,localDeviceId,localDeviceIp,remoteServerId,remoteDeviceId,remoteDeviceIp",
      "suggestion": {
        "Possible Cause": "1. The network between two devices is abnormal. For example, the network port is intermittently disconnected. \r\n2. The peer process exits abnormally in advance. As a result, the local end cannot receive the response from the peer end.",
        "Solution": "1. Check whether the network devices between the two ends are abnormal. \r\n2. Check whether the peer process exits first. If yes, check the cause of the process exit. \r\n3. For detailed troubleshooting guidance, search for the keyword \"EI0013\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Config_Error_Ranktable",
      "ErrCode": "EI0014",
      "ErrMessage": "Value %s for ranktable variable %s is invalid. Expected value: %s.",
      "Arglist": "value, variable, expect",
      "suggestion": {
        "Possible Cause": "Failed to verify the content of the ranktable file, possibly due to inconsistency between the file content and the actual device information.",
        "Solution": "Try again with a valid cluster configuration in the ranktable file. Ensure that the configuration matches the operating environment."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Ranktable_Detect",
      "ErrCode": "EI0015",
      "ErrMessage": "Failed to collect cluster information of the communicator based on rootInfo detection. Reason: %s.",
      "Arglist": "error_reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution":"1. Check whether all ranks in the communicator have delivered the communicator creation interface. \r\n2. Check the connectivity between the host networks of all nodes and the server node. \r\n3. Check whether the HCCL_SOCKET_IFNAME environment variable of all nodes is correctly configured. \r\n4. Increase the timeout by configuring the HCCL_CONNECT_TIMEOUT environment variable. \r\n5. For detailed troubleshooting guidance, search for the keyword \"EI0015\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Config_Error",
      "ErrCode": "EI0016",
      "ErrMessage": "Value %s for config %s is invalid. Expected value: %s.",
      "Arglist": "value, variable, expect",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Config_Error_Ranktable",
      "ErrCode": "EI0017",
      "ErrMessage": "Config %s is missing in the ranktable file.",
      "Arglist": "config",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "N/A"
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Execution_Error_UB_CQE",
      "ErrCode": "EI0018",
      "ErrMessage": "An error CQE occurred during operator execution. Local information: server %s, device ID %s, device IP %s. Peer information: server %s, device ID %s, device IP %s.",
      "Arglist": "localServerId, localDeviceId, localDeviceIp, remoteServerId, remoteDeviceId, remoteDeviceIp",
      "suggestion": {
        "Possible Cause": "1. The network between two devices is abnormal. For example, the network port is intermittently disconnected. \r\n2. The peer process exits unexpectedly in advance. As a result, the local end cannot receive the response from the peer end. \r\n3.The hardware of the HBM or UB chip processing module of either device is abnormal.",
        "Solution": "1. Check whether the network devices between the two ends are abnormal. Generally, packet loss occurs due to intermittent disconnection of the port. If the ping test fails, check whether the port is linkdown or the network configuration is incorrect.\r\n2. Check whether the peer process exits first. If yes, check the reason why the process exit.\r\n3.Use the RAS fault check mechanism to check whether the hardware of the HBM or UB chip processing module of either device is abnormal.If the hardware is abnormal, contact Huawei technical support."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Bind_IP_Port",
      "ErrCode": "EI0019",
      "ErrMessage": "Failed to enable listening for the host network adapter socket. Reason: %s",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "1. Check whether this port has been occupied by another process. If yes, you can make adjustment using the environment variable HCCL_IF_BASE_PORT and use sysctl -w net.ipv4.ip_local_reserved_ports=****-**** to adjust the scope of reserved ports. \r\n2. Check whether the service process is started multiple times on a device during this service. \r\n3. For detailed troubleshooting guidance, search for the keyword \"EI0019\" on https://www.hiascend.com/document/."
      }
    },
    {
      "errClass": "HCCL Errors",
      "errTitle": "Communication_Error_Bind_IP_Port",
      "ErrCode": "EI0020",
      "ErrMessage": "Failed to enable listening for the NPU network adapter socket. Reason: %s",
      "Arglist": "reason",
      "suggestion": {
        "Possible Cause": "N/A",
        "Solution": "Check whether the single-card multi-process scenario is used. If yes, configure the port number using the environment variable HCCL_NPU_SOCKET_PORT_RANGE."
      }
    }
  ]
}
)";
}

REG_FORMAT_ERROR_MSG(hcomm_g_msg.c_str(), hcomm_g_msg.size());
