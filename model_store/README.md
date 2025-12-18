# model_store/

模型管理的缓存与配置集中在这里，`models/` 目录仅放真正的模型文件/文件夹。

- `downloads/`：默认模型下载的 zip 缓存。
- `cache/`：推理/导出使用的临时缓存。
- `sources.yaml|json`：默认模型来源清单（UI 读取）。
- `nnunet_settings.json`：nnU-Net 命令配置（UI 设置页写入）。

实际的模型文件位于 `models/<model_id>/...`（或 `models/*.pth`），不随 Git 提交。
