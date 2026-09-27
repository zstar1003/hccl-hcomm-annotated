# HCOMM Agent Rules

This file serves as the main entry point for AI Agent governance of the HCOMM repository. It is intended for AI programming tools that support the AGENTS.md standard. If a subdirectory has its own `AGENTS.md`, the nearer file supplements this one. In case of conflicts, follow the user's explicit requirements first, and then the rules in the nearest subdirectory.

> This file provides only the **essential hard constraints and entry points**. Detailed content is progressively disclosed through links. Before modifying code, read the architecture constraints in Section 3.

## 1. Repository Positioning

HCOMM (Huawei Communication) is the underlying communication library of HCCL. It provides communication domain and communication resource management capabilities and serves as the base layer of the CANN collective communication stack. HCCL consists of two parts: the HCCL operator library (`cann/hccl`) and the HCOMM communication base library (this repository, `cann/hcomm`). The two repositories are decoupled through dynamic loading via `dlsym`. For details, see the [`README.md`](./README_en.md) and the [Architecture Overview (Chinese)](./docs/zh/architecture/architecture-brief.md).

## 2. Directory Structure

```text
src/
├── base_comm/                 # Basic communication layer (L3, lowest layer, must not have reverse dependencies on upper layers)
├── coll_communicator_mgr/     # Collective communication domain management HCCM (L2, depends on base_comm)
└── legacy/                    # Historical compatibility directory, does not accept new features
include/      # External header files (hccl/, ccu/, hcomm_primitives.h, and so on)
pkg_inc/      # Inter-package interface header files (hccl/, hcomm/, not guaranteed to be stable externally)
test/         # ut / st / stub
docs/         # Documentation
build.sh      # One-click build script
```

For details on the target directory structure and layer responsibilities, see [Section 3.2 of the Architecture Overview (Chinese)](./docs/zh/architecture/architecture-brief.md).

## 3. Software Architecture and Architecture Constraints (Core)

> **Authoritative Architecture Source (Chinese)**: [`docs/zh/architecture/architecture-brief.md`](./docs/zh/architecture/architecture-brief.md). Before modifying `src/`, `include/`, or `pkg_inc/`, read Section 3 - Software Layering Logic and the Software Architecture Constraint Description at the end.

### Layers

| Software Layer | Repository Location |
|----|--------|
| HCCL collective communication operators | `cann/hccl` |
| HCOMM collective communication domain management (HCCM) | This repository `src/coll_communicator_mgr` |
| HCOMM basic communication | This repository `src/base_comm` |

Dependency direction is top-down: `coll_comm_ops` (hccl) → `coll_communicator_mgr` → `base_comm`.

### Architecture Constraints (Hard, Must Not Violate)

| Constraint | AI Agent Behavior Requirements |
|------|------------------|
| **Layer dependency direction**: Upper layers depend on lower layers. Lower layers must not have reverse dependencies on upper layers (`base_comm` ↛ `coll_communicator_mgr`; both ↛ `coll_comm_ops`) | Lower layers must not `#include` upper-layer header files or call upper-layer interfaces. When adding new classes or functions, first determine the layer and only call within the same layer or a lower layer. |
| **Control plane and data plane separation**: Resource management and topology query (control plane) and data transfer or synchronization (data plane) interfaces evolve independently and are not coupled. | Do not introduce strong control plane coupling in data plane primitives. The control plane must not depend on specific data plane operator implementations. |
| **HCCL and HCOMM decoupling**: HCCL dynamically loads HCOMM interfaces through `dlsym`. The two repositories compile and version independently. | HCOMM must not `#include` HCCL private headers. Do not introduce compile-time hard dependencies on `cann/hccl`. Cross-repository calls go through symbol tables and `dlsym`. |
| **Legacy does not continuously evolve**: `legacy/` is for historical compatibility only and does not accept new features. | **Do not add new functions, operators, or interfaces in `src/legacy/`**. Legacy is for bug fixes and compatibility maintenance only. Place new features in `base_comm/` or `coll_communicator_mgr/`. |

### External API Layers

| Layer | Header Files | Audience |
|------|------|------|
| L2-comm | `include/hccl/hccl_comm.h` | AI framework layer (communication domain creation) |
| L2-res-rank_graph | `include/hccl/hccl_res.h`, `include/hccl/hccl_channel.h`, `include/hccl/hccl_rank_graph.h` | Operator developers (topology query, resource acquisition) |
| L3-prim | `include/hcomm_primitives.h` | Operator or communication library developers (data transfer and synchronization) |
| L3-res | `include/hcomm_res.h`, `include/hcomm_res_defs.h`, `include/hcomm_channel.h` | Communication library developers (device, channel, and memory resources) |
| CCU | `include/ccu/` (`ccu_primitives.hpp`, `ccu_res.h`, `ccu_launch.h`) | CCU operator developers |

Changes to `include/` must be backward compatible. `pkg_inc/` is for inter-package use between HCOMM, HCCL, GE, and so on, and is not externally stable. For the complete API layer relationships, see [Section 3.3 of the Architecture Overview (Chinese)](./docs/zh/architecture/architecture-brief.md).

## 4. Build and Test

```bash
bash build.sh --pkg            # Build the host package (default)
bash build.sh --pkg --full     # Build the host and device full package
bash build.sh -u               # Build and run UT (equivalent to --ut)
bash build.sh -s               # Build and run ST
bash build.sh --ut --noexec    # Build UT without running test cases (--noexec must be used with --ut or --st)
bash build.sh -j64             # Parallel compilation
```

Output: `build_out/cann-hcomm_<version>_linux-<arch>.run`. For a complete list of options, environment setup, and on-board testing, see [`docs/en/build/build.md`](./docs/en/build/build.md). Before pushing, verify locally with `--pkg` + `--ut --noexec` + `--st --noexec`.

## 5. Coding Standards

- Naming: Classes and functions use PascalCase; member variables use `camelCase_` (lower camelCase with a trailing underscore); constants and macros use `UPPER_SNAKE_CASE`.
- Style: Follow the `.clang-format` in the root directory (120 columns, 4 spaces, pointer right-aligned, K&R braces). Use C++14.
- Static warnings: Code must pass CANN static check requirements (verified during the CI codecheck stage) and compile without warnings.
- pre-commit: clang-format v18.1.8 + OAT compliance check. New source files must include the CANN-2.0 license header.

References: [CANN Coding Standards](https://gitcode.com/cann/community/tree/master/contributor/coding-standards), [CANN CI Guide](https://gitcode.com/cann/community/blob/master/contributor/repository/ci-guide.md), [pre-commit Guide](./docs/en/build/pre-commit-guide.md), `.clang-format`, `OAT.xml`.

## 6. Contribution Process

- Simple issues: Issue → Claim → PR → Committer Review → `/lgtm` + `/approve` merge.
- New features: Requirement Issue → SIG Decision → `docs/en/rfcs/` RFC Review → Implementation (including UT and ST) → Review and Merge.
- All PRs must be associated with an Issue. Fill in the description according to `.gitcode/PULL_REQUEST_TEMPLATE.en-US.md`.

For details, see [`CONTRIBUTING.md`](./CONTRIBUTING_en.md).

## 7. Agent Work Principles

- Prioritize small and reviewable changes. Avoid large-scale refactoring unless the user explicitly requests it.
- Before editing, locate the file and describe the plan in 3 to 6 statements.
- When unsure about APIs, configurations, paths, or facts, search the repository or verify. Do not fabricate information.
- Before modifying `src/`, check against the architecture constraints in Section 3: Does it violate layer dependencies? Does it add new features to `legacy/`? Does it break control plane and data plane separation? Does it introduce compile-time hard dependencies on `cann/hccl`?
- Never write keys, tokens, passwords, private keys, `.env` values, or credentials into code, logs, or replies.
- Unless the user requests it, do not add telemetry, analytics reporting, or additional network calls.
- For behavioral changes, supplement or update tests under the project's existing test framework. Prioritize running the fastest relevant verification.
- When renaming or moving directories in `src/`, simultaneously check `CMakeLists.txt`, test include paths, `#include` relative paths, `classify_rule.yaml`, `blacklist.txt`, and clean the build directory before re-verifying.
- Destructive commands, `git commit`, and `git push` require explicit user approval.
- By default, use Chinese for explanations. Keep output concise, specific, and reproducible.

---

*Architecture constraints are based on [`docs/zh/architecture/architecture-brief.md (Chinese)`](./docs/zh/architecture/architecture-brief.md). The contribution process follows [`CONTRIBUTING_en.md`](./CONTRIBUTING_en.md). Personal temporary preferences go in `AGENTS.local.md` (gitignored).*
