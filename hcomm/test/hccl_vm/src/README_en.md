# TABLE CMD

[toc]

## 1 Core Features

### 1.1 Start the Simulation Environment

#### 1.1.1 View All Current Table Names

```bash
(hvm)$> hccl-vm table show all
all
Server
Host
Runner
Device
...
(hvm)$>
```

#### 1.1.2 View the Content of a Specified Table

```bash
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend950 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |

(hvm)$>
```

#### 1.1.3 Update Specified Content in a Specified Row of a Specified Table (Currently Only Device soc_version Modification Is Supported)

```bash
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend950 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |
(hvm)$> hccl-vm table update device 1 soc_version Ascend951
update device 1 soc_version Ascend951
Updating device [id=1].soc_version = "Ascend951"
(hvm)$> hccl-vm table show Device
| id | server_id | logic_id | physical_id | overflow_mode | soc_version | status |
| 1 | 1 | 0 | 0 | 0 | Ascend951 | 0 |
| 2 | 1 | 1 | 1 | 0 | Ascend950 | 0 |
| 3 | 1 | 2 | 2 | 0 | Ascend950 | 0 |
| 4 | 1 | 3 | 3 | 0 | Ascend950 | 0 |

(hvm)$>
```
