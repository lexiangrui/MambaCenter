# GitHub 提交范围

提交模型源码（mambacenter/、baseline/CenterNet 补丁）、CUDA/C++ 源文件、数据转换、训练与评测入口（tools/）、实验配置（configs/）、测试、文档、Python 依赖说明、第三方许可证及 results/comparison.json。

## 本地文件

以下内容由根 .gitignore 排除，只保留在使用者机器上：

- 虚拟环境、已安装依赖、CUDA 二进制、构建产物和缓存。
- data/、runs/、checkpoint、预测、原始 history、日志和实验快照。
- .env、*.local.*、config.local/、SSH/密钥、IDE 设置及 AGENTS.md。
- 根与 mamba-yolo 下的 slurm/ 作业脚本（含分区、节点和路径设置）。

公开 results 只保留 comparison.json 与 Mamba-YOLO 源码兼容补丁记录，不含个人存储路径或作业编号。
服务器配置放入 .env、config.local/ 或个人 slurm/，不要写回公开代码。

## 已有 Git 仓库

.gitignore 只影响未跟踪文件。若服务器配置曾被跟踪，先用 `git ls-files -ci --exclude-standard` 检查，再对确认的文件使用 `git rm --cached`（目录加 -r），只移除索引、保留工作区。提交前检查 git status 与 git diff --cached。
