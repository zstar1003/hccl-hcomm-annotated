# HCOMM

English | [简体中文](./README.md)

## 🔥 Latest News

- [2025/11/30] The HCOMM project is now open source.

## 🚀 Overview

HCOMM (Huawei Communication) is the underlying communication library of HCCL. It provides communication domain and communication resource management capabilities.

HCOMM offers standardized communication programming interfaces with the following key features:

- Supports multiple communication engines on Ascend devices, fully leveraging hardware capabilities.
- Supports multiple communication protocols, including PCIe, HCCS, RDMA, and more.
- Decouples the communication platform from communication operator development, enabling independent development, building, and deployment of communication operators.

<img src="./docs/en/build/figures/architecture.png" alt="hccl-architecture" style="width: 65%;  height:65%;" />

The HCOMM communication library adopts a layered and decoupled design approach, dividing communication capabilities into a control plane and a data plane.

- Control plane: Provides topology information query and communication resource management.
- Data plane: Provides data transfer and computation capabilities such as local operations, inter-operator synchronization, and communication operations.

The control plane provides communication resources, and the data plane provides methods for operating on resources. The communication programming interfaces allow communication operator developers to focus on business innovation without concerning themselves with the complex implementation details at the chip level.

## 🔍 Directory Structure

The key directories of this project are as follows:

```text
├── src                                  # Source code directory
│   ├── base_comm                        # Basic communication layer
│   │   ├── common                       # Common basic functionality for the basic communication layer
│   │   ├── primitives                   # Basic communication primitives
│   │   └── resources                    # Basic communication resources
│   ├── coll_communicator_mgr            # Collective communication domain management
│   │   ├── api_c_adpt                   # C interface adaptation
│   │   ├── common                       # Common basic functionality for the collective communication layer
│   │   ├── communicator                 # Communication domain
│   │   ├── config_mgr                   # Configuration management
│   │   ├── dfx                          # Diagnostics and tracing
│   │   ├── rank_graph                   # Topology management
│   │   └── resource_mgr                 # Resource management
│   └── legacy                           # Historical version compatibility directory
│       ├── ascend910                    # A2 and A3 compatibility code
│       │   ├── algorithm                # Communication algorithm source code
│       │   ├── common                   # Common basic functionality
│       │   ├── framework                # Communication framework source code
│       │   ├── hccd                     # Inter-process point-to-point communication
│       │   ├── platform                 # Communication platform source code
│       │   └── pub_inc                  # Platform interface header files
│       └── ascend950                    # A5 legacy flow compatibility code
│           ├── common                   # Common basic components
│           ├── framework                # Framework core implementation
│           ├── include                  # Public interface header files
│           ├── interface                # Interface adaptation layer
│           ├── local_build              # Local build tools
│           ├── service                  # Service layer
│           └── unified_platform         # Unified platform layer
├── python                               # Python package
├── include                              # External header files
├── pkg_inc                              # Inter-package interface header files
├── experimental                         # Community-contributed experimental code directory
├── test                                 # Test code directory
│   ├── ut                               # Unit test code directory
│   └── st                               # System test code directory
├── docs                                 # Documentation directory
├── examples                             # Sample code directory
└── build.sh                             # Build script
```

## 📝 Version Compatibility

The source code of this project is released alongside the CANN software version. For the mapping between CANN software versions and this project's tags, refer to the corresponding version descriptions in the [release repository](https://gitcode.com/cann/release-management). To ensure a smooth custom source code development experience, select a compatible CANN version and GitCode tag. Using the master branch may pose a version mismatch risk.

## ⚡️ Quick Start

To quickly build and experience this project, refer to the following simple guides.

- [Source Code Build](./docs/en/build/build.md): Learn how to compile and install this project, and perform basic test verification.
- [Sample Execution](./examples/README_en.md): Follow detailed sample code and step-by-step instructions for a quick experience.

## 📖 Learning Tutorials

HCCL provides usage guides, communication operator development guides, technical articles, and training videos. For details, see the [HCCL References](./docs/README_en.md). HCCL also offers QuickStart guides, common FAQs, and other wiki resources. For details, see the [WIKI](https://gitcode.com/cann/hcomm/wiki).

## 📝 Related Information

- [Contribution Guide](CONTRIBUTING_en.md)
- [Security Statement](SECURITY_en.md)
- [License](LICENSE)
