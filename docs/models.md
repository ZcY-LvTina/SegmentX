# 模型与 nnU-Net 集成说明

## 目录结构

- `models/<model_id>/manifest.yaml|json`：模型清单，包含 id/name/type(nnunet/sam/...)/capabilities/input_format/version/source/entry/labels(可选) 等字段。
- `models/<model_id>/payload/`：对应模型的权重或 nnU-Net 结果目录。
- `model_store/downloads/`：默认模型 zip 的缓存，不进 Git。
- `model_store/cache/`：推理/导出的临时目录。
- `model_store/sources.yaml|json`：默认模型来源清单；UI 会读取并展示“安装默认模型”。未安装 pyyaml 时可改用 JSON。
- `model_store/nnunet_settings.json`：nnU-Net 命令配置（通过 UI 设置页修改）。
- `model_store/README.md`：存放模型仓库/缓存说明。

## manifest 约定

```yaml
id: brain3d
name: Brain Seg 3D
type: nnunet            # nnunet/sam/sam2...
capabilities: [3d]
input_format: nifti     # nifti/dicom/png_stack/video_frames/image
version: 1.0.0
source: default:sources.yaml
entry: payload          # 相对于模型目录的路径，指向权重/结果
labels: ["bg", "brain"] # 可选
```

缺字段或 payload 丢失会在 UI 中标红但不会导致崩溃。

## 默认模型一键安装

1. 在 `model_store/sources.yaml` 填写 `{model_id,name,type,url,sha256,capabilities,input_format,labels(可选)}`。
2. UI 左侧“安装默认模型”按钮 -> 选择一个来源 -> 后台下载到 `model_store/downloads/<model_id>.zip`。
3. 如 zip 内含 manifest，则直接安装；否则若为 nnU-Net 结果 zip，会自动生成 manifest 并放到 `models/<model_id>/payload/`。
4. 下载/校验/安装进度会显示在状态栏；失败会提示原因（网络、权限、sha256 不一致等）。

## 导入 nnU-Net 原生结果

- 使用 UI 的“导入原生nnU-Net”选择 zip 或目录即可，内部会识别 `plans.json/nnUNetPlans.json` 等特征，拷贝到 `models/<model_id>/payload/` 并生成 manifest。
- 也可调用 `ModelRegistry.import_nnunet_native(path, model_id, meta)` 直接注入。

## nnU-Net 环境配置

- 顶部模型管理面板提供“nnU-Net设置”，可填写 `nnUNetv2_predict` 与 `nnUNetv2_train` 的可执行路径，保存到 `model_store/nnunet_settings.json`。
- 未检测到命令时会提示如何安装/配置，不会让 UI 崩溃。

## 推理/训练流程（UI）

- “当前模型”下拉来自 ModelRegistry，可在 SAM/nnU-Net 之间切换。选择 nnU-Net 后可点击“nnU-Net推理”，选择 PNG 序列文件夹 -> 自动转成伪 3D NIfTI，调用 `nnUNetv2_predict --model_path <payload>`，输出体数据并将中间切片作为遮罩图层显示。
- 侧边栏“nnU-Net训练”仅暴露 dataset id、2D/3D 配置、设备与 fast/dev 选项，其余走 nnU-Net 默认。训练完成后自动打包并注册为可选模型。
- 训练入口由 `TrainRunner` 封装，启动 `nnUNetv2_train` 默认流程，训练完成后自动打包成 SegmentX 模型包进入列表。

## 数据导入/导出

- `src/segmentx/data/volume_loader.py`：从 PNG 序列按自然排序加载为 `Volume3D`（无 spacing 时默认 1,1,1 并标记 unknown）。
- `src/segmentx/data/exporters/nnunet_exporter.py`：将 Volume3D 导出为 nnU-Net 友好的 `.nii.gz`（依赖 nibabel），推理走 `imagesTs/`，训练走 `imagesTr/labelsTr` 并生成 dataset.json。
- 推理/训练缓存默认写入 `model_store/cache/nnunet/...`，不会污染 Git。

## 自检（最小）

`scripts/self_check.py`（依赖 numpy、Pillow）会覆盖：

- registry 扫描/安装/移除（使用临时 manifest.json + payload）；
- png-stack 自然排序是否正确；
- `import_nnunet_native` 自动生成 manifest 是否可用。
