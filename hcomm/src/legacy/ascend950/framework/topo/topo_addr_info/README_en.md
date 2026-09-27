# Topo Address Info

## Introduction

In the Ascend 950 chip generation, the hyperplane uses the Unified Bus for networking. Multiple different topology networking methods are used across different product forms. This module is used to discover the endpoint addresses of each edge under different topologies.

## Networking Introduction

Two main types of networking are used: MESH and CLOS. For related information, refer to the paper: https://arxiv.org/abs/2503.20377

### MESH Networking

Each NPU has a direct physical link to every other NPU, so there is a pair of independent communication addresses. For example, if there are 8 NPUs on the same NPU board, there are 8 * 7 / 2 = 28 physical paths, that is, 28 pairs of communication addresses.

### CLOS Networking

Any two NPUs communicate through a switch chip. Therefore, one NPU requires one address. In common networking scenarios, due to reliability and other reasons, CLOS networking can usually be divided into multiple planes, with each plane corresponding to one address. For example, in a liquid-cooled POD, NPUs use two independent logical ports to connect to two separate network planes.

## Networking Planning

### Network Layer Description

In the 950 chip generation, the network is divided into multiple layers based on communication quality and range.

| Network Layer | Description |
|:-------| :-----------|
| 0 | Highest communication quality and lowest latency. Mostly MESH networking, mainly the fullmesh network within the same NPU board and the box-level network in POD form. |
| 1 | Second-highest communication quality and medium latency. CLOS networking with a larger communication range, at the super node level, still within the scale-up range. |
| 2 | Lowest communication quality. CLOS networking with the entire cluster as the communication range. Mainly scale-out networks such as ROCE or UBOE. |

### Network Address Description

| Network Layer | Description |
|:-------| :-----------|
| 0 | Because MESH networking is the primary method, there are multiple pairs of communication addresses. In topo_addr_info, this is expressed as the address of each port on each NPU. The address type is EID. |
| 1 | Fill in the address based on the networking plane. In multi-plane networking, the number of addresses is the same as the number of planes. Traffic is distributed among different planes during collective communication. The address type is EID. |
| 2 | The address planning is the same as layer 1. The address type is IP address. |
