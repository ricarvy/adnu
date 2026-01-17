# ADNU: Advanced Draw-and-Understand 项目总览

本仓库在原始 **Draw-and-Understand** 项目的基础上，新增了 **ADNU（Advanced Draw-and-Understand）** 方案，用于系统性提升多提示视觉理解能力。  
本 README 面向需要快速理解代码结构、调用流程与论文中关键设计的读者（例如论文答辩 / 组会汇报）。

---

## 0. 仓库整体结构

- 原始项目（官方代码）
  - [Draw-and-Understand](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand)
    - 官方 SPHINX-V 模型、训练 / 评估 / 推理代码与 MDVP-Data、MDVP-Bench 支持。
- 改进项目（本仓库新增）
  - [draw-and-understand-advanced](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced)
    - ADNU 的 5 个创新点实现：视觉提示泛化、动态门控、文化模板、超图提示编码、MAE 自监督。
- 论文 / 汇报相关
  - [presentation_data](file:///e:/E_Projects/TraeProject/adnu/presentation_data)
    - 各实验结论的模拟数据 CSV 与说明文档。
  - [overleaf-paper](file:///e:/E_Projects/TraeProject/adnu/overleaf-paper)
    - LaTeX 论文模版及已填充的实验部分。

接下来分三部分展开：

1. 原项目 **Draw-and-Understand** 的模块与调用流程（对应 SPHINX-V 原论文）。
2. 改进项目 **Draw-and-Understand-Advanced** 的模块与调用流程（对应 ADNU 创新点与 presentation_data）。
3. 二者在设计与实现上的对比与联系。

---

## 1. 原项目：Draw-and-Understand（SPHINX-V）

### 1.1 核心目录与模块

> 项目根目录：  
> [Draw-and-Understand](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand)

- 顶层说明
  - [README.md](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/README.md)  
    - 介绍项目背景、MDVP-Data / MDVP-Bench 链接与训练 / 评估流程。
    - 描述两阶段训练策略（Stage 1 预训练、Stage 2 多任务监督微调）。

- 模型入口
  - [SPHINX_V/sphinx_v.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/SPHINX_V/sphinx_v.py)
    - `SPHINX_V_Model` 继承自 MetaModel，是对整个 SPHINX-V 的高层封装。
    - 提供 `generate_response` 方法，实现论文中的“画图 + 提示 + 问答”交互接口。

- 元模型与 LLaMA 主干
  - [accessory/model/meta.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/meta.py)
    - `MetaModel`：负责从 checkpoint / config / tokenizer 构建内部 LLaMA 模型，并封装训练 / 推理接口。
    - `from_pretrained`：解析 `meta.json`、`config.json`、`tokenizer.model`，加载张量并支持模型并行。
    - `forward`：计算交叉熵 loss，同时透传底层模型返回的 `additional_loss` 字典。
  - [accessory/model/LLM/llama_ens5_vp.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/LLM/llama_ens5_vp.py)
    - `ModelArgs`：LLaMA 配置（维度、层数、rope 参数等）。
    - `Transformer`：实现全文本 + 多模态提示的 LLaMA 主体。
      - token embedding、注意力层、前馈层、RMSNorm、输出层。
      - 支持 KV cache 与推理模式的增量 decoding。

- 视觉编码与视觉提示编码（论文关键）
  - 视觉骨干与多模态对齐：
    - `Transformer.encode_image`  
      [llama_ens5_vp.py:430-510](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/LLM/llama_ens5_vp.py#L430-L510)
      - 使用 Q-Former (`Blip2Model`)、CLIP ViT-L-14、ConvNeXt-XXL、DINOv2-ViT-G 组合，形成多源视觉特征。
      - 通过 `qformer_proj` + `visual_proj` 映射到统一维度并拼接，得到 `(B, 32 + 257, dim)` 的视觉 token。
  - 视觉提示编码（Visual Prompt Encoder，论文中最核心的设计之一）：
    - [accessory/model/LLM/visual_prompt_encoder/prompt_encoder.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/LLM/visual_prompt_encoder/prompt_encoder.py)
      - `PromptEncoder`：改造自 SAM 的 prompt encoder。
      - `_embed_points`：将点坐标归一化后，通过 `PositionEmbeddingRandom` 做随机傅里叶位置编码，再叠加“有效 / 无效点” embedding。
      - `_embed_boxes`：将 box 转为两个角点，分别位置编码，并叠加“box 类型 + 角点类型” embedding。
      - `forward`：将 points / boxes 编码为 `(B, N, 2*embed_dim)` 的稀疏 embedding。
    - `Transformer.encode_visual_prompt`  
      [llama_ens5_vp.py:513-557](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/LLM/llama_ens5_vp.py#L513-L557)
      - 将输入的 `sparse_vp`（形状 `(B, N, 4)`）自动拆分为点 / 框两类：
        - “后两维为 -1” → 点；否则视为 box。
      - 通过 `PromptEncoder` 生成 512 维 embedding，再通过 `vp_proj` 投影到 LLaMA hidden dim。

- 数据与评估
  - 数据说明：
    - [Data/dataset.md](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/Data/dataset.md)  
      描述预训练 / 微调数据组织方式与路径设置。
  - MDVP-Bench 评估：
    - [accessory/eval/readme.md](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/eval/readme.md)  
      介绍 Referring Object Classification、Region Level Captioning、Regional OCR、MDVP-Bench 等多任务评估流程。
    - [accessory/eval/MDVP-Bench/eval_gpt.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/eval/MDVP-Bench/eval_gpt.py)  
      使用 GPT-4V 对 MDVP-Bench 结果进行主观质量评估。

### 1.2 推理调用流程（从图片 + 手绘提示到文本回答）

以 [SPHINX_V/inference.py](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/SPHINX_V/inference.py) 为例，核心推理路径如下：

1. 构造 `SPHINX_V_Model`：
   - 通过 `MetaModel.from_pretrained` 加载 SPHINX-V Stage-2 checkpoint、tokenizer 与 `llama_ens5_vp.Transformer`。
2. 调用 `SPHINX_V_Model.generate_response`  
   [sphinx_v.py:12-69](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/SPHINX_V/sphinx_v.py#L12-L69)
   - 接收：
     - `qas`: 多轮问答列表 `[[q1, a1], ..., [q_n, None]]`。
     - `vps`: 视觉提示（点 / 框坐标列表）。
     - `image`: 输入图像。
   - 预处理：
     - 图像经 `get_transform("padded_resize", target_size)` 归一化到 448×448。
     - `vps` 经 `Transform_Visual_Prompts` 转为 `(N, 4)` 的 `sparse_vp`。
   - 构造对话模板：
     - 使用 `default_conversation`，将 Q/A 列表编码为单个文本 prompt。
   - 调用 `self.generate(...)`：
     - `prompts=[prompt]`
     - `sparse_vps=sparse_vp.unsqueeze(0)`
     - `images=image.unsqueeze(0)`
   - 返回首个样本的回答字符串。
3. `MetaModel.generate` 内部会调用 `self.llma.forward_inference(...)`：
   - `llma` 即 `llama_ens5_vp.Transformer`。
   - 其 `forward_inference` 在首 token 时将视觉 token 与视觉提示 token 通过 `encode_image` 与 `encode_visual_prompt` 拼接到文本 token 前部，实现多模态条件生成。

### 1.3 训练流程（两阶段，对应论文）

原 README 中给出了训练入口脚本（在官方仓库中）：

- Stage 1：Image–Visual Prompt–Text 对齐预训练
  - 使用预训练的 SPHINX-v2-1k 模型与 SAM 权重。
  - 配置：`accessory/configs/data/vp_pretrain.yaml`。
  - 目标：对齐图像区域、视觉提示与文本描述。
- Stage 2：多任务端到端监督微调
  - 使用 Stage-1 预训练权重作为初始化。
  - 配置：`accessory/configs/data/vp_finetune.yaml`。
  - 数据覆盖多个任务（Referring、Captioning、OCR、MDVP-Bench 等）。

本仓库中的 Draw-and-Understand 代码保持原样，可直接使用 README 中描述的训练 / 评估流程。

---

## 2. 改进项目：Draw-and-Understand-Advanced（ADNU）

> 项目根目录：  
> [draw-and-understand-advanced](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced)

ADNU 在保持 SPHINX-V 整体结构与调用方式兼容的前提下，引入 5 个主要创新点。与原项目不同的是，Advanced 目录主要以“研究原型 / 组件库”的形式组织，方便插拔式注入到原 SPHINX-V 流程中。

### 2.1 目录结构

- 模型组件
  - [accessory/model/visual_prompt_encoder.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/visual_prompt_encoder.py)
  - [accessory/model/gating.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/gating.py)
  - [accessory/model/hypergraph.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/hypergraph.py)
  - [accessory/model/mae_wrapper.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/mae_wrapper.py)
  - [accessory/model/sphinx_v_advanced.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/sphinx_v_advanced.py)
- 数据与模板
  - [accessory/data/dataset_mock.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/data/dataset_mock.py)
  - [accessory/data/template.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/data/template.py)
- Demo 训练脚本
  - [scripts/train_demo.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/scripts/train_demo.py)

下面按创新点逐一说明。

### 2.2 创新点一：视觉提示泛化（Visual Prompt Generalization）

**目标：** 从原始只支持点 / 框的 SAM PromptEncoder，扩展到支持 **点 + 框 + 多边形**，并提供形状不变的傅里叶描述符。

- 实现位置：  
  [visual_prompt_encoder.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/visual_prompt_encoder.py)

- 核心类：`VisualPromptEncoder`

  ```python
  class VisualPromptEncoder(nn.Module):
      def __init__(self, embed_dim: int = 4096, polygon_points: int = 32) -> None:
          self.point_encoder = nn.Linear(2, embed_dim)
          self.box_encoder = nn.Linear(4, embed_dim)
          self.polygon_encoder = nn.Sequential(
              nn.Linear(polygon_points * 2, embed_dim),
              nn.LayerNorm(embed_dim),
              nn.ReLU(),
              nn.Linear(embed_dim, embed_dim),
          )
          self.type_embedding = nn.Embedding(3, embed_dim)  # 0: point, 1: box, 2: polygon
  ```

- 前向流程：
  - 输入：`prompts` 形状 `(B, N, D_max)`，`prompt_types` 形状 `(B, N)`。
  - 根据 `prompt_types` 拆分为三类：
    - 点：取前 2 维 `(x, y)`。
    - 框：取前 4 维 `(x1, y1, x2, y2)`。
    - 多边形：取前 `polygon_points * 2` 维作为扁平化的点序列。
  - 每类分别编码，然后加上对应 `type_embedding`，最后写回统一的 `(B, N, embed_dim)`。

- 形状不变性增强：

  - 使用静态方法 `encode_fourier_descriptors` 将多边形点序列转为截断傅里叶描述符：
    - 平移对齐：减去质心。
    - 尺度归一：按最大半径归一化。
    - 频域变换：对复数序列做 FFT，截取前 `k` 个非 DC 频率。
    - 拼接实部 / 虚部得到 `(B, k, 2)` 的描述符。

该模块可直接替换 / 装饰原 SAM PromptEncoder，形成更泛化的提示表示。

### 2.3 创新点二：动态门控（Dynamic Prompt Gating）

**目标：** 解决论文中提到的 **“Box 数量多时反而不如 Point”** 的倒挂现象，通过内容感知 + 稀疏 Top-K 门控，从多提示中自动选择最有效的信息。

- 实现位置：  
  [gating.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/gating.py)

- 核心类：`PromptGating`

  ```python
  class PromptGating(nn.Module):
      def __init__(self, embed_dim: int = 4096, top_k: int | None = 5) -> None:
          self.gate_net = nn.Sequential(
              nn.Linear(embed_dim, embed_dim // 4),
              nn.ReLU(),
              nn.Linear(embed_dim // 4, 1),
              nn.Sigmoid(),
          )
  ```

- 前向逻辑：
  - 输入：`prompt_embeddings` 形状 `(B, N, D)`。
  - 通过 `gate_net` 生成 `(B, N, 1)` 的权重 `scores ∈ (0, 1)`。
  - 软门控：`weighted_embeddings = prompt_embeddings * scores`。
  - 若设置了 `top_k`：
    - 按每个样本的 `scores` 排序，保留前 `k` 个，生成稀疏 Mask。
    - 非 Top-K 的提示 embedding 被置零，从而降低噪声提示带来的干扰。

在 Advanced 中，该门控模块既可以独立使用（见 `sphinx_v_advanced`），也已经被内嵌回原始 `llama_ens5_vp.Transformer.encode_visual_prompt` 中，通过新增的配置项 `use_adnu_prompt_gating` 控制。

### 2.4 创新点三：文化模板与多语言扩展（Cultural Prompt Template）

**目标：** 针对论文中提到的“中文 / 文化场景表现不足”，通过模板与先验注入增强模型在节日、成语等任务上的表现。

- 实现位置：  
  [template.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/data/template.py)

- 核心类：`CulturalPromptTemplate`

  - 内置多语言模板：

    ```python
    TEMPLATES = {
        "zh": {
            "default": "请描述选定区域的内容。",
            "festival": "结合{festival}的传统习俗，描述这个{object}。",
            "idiom": "用成语概括这个场景：{context}。",
        },
        "en": {
            "default": "Describe the content of the selected region.",
            "festival": "Describe this {object} in the context of {festival} traditions.",
        },
    }
    ```

  - `get_template(lang, key, context)`：
    - 根据语言与 key（如 `festival`、`idiom`）返回格式化后的模板字符串。
  - `inject_prior(prompt_text, detected_objects)`：
    - 根据检测到的对象标签（如 `zongzi`, `mooncake`），注入对应中国传统节日的补充说明。

在 Chinese-MDVP 子集实验中，可以在构造问题文本时用该模块为原始问题添加文化前缀 / 注释，从而提升中文任务的表现（对应 `4_cultural_capability_zh.csv`）。

### 2.5 创新点四：超图提示编码（HyperGraph Prompt Encoder）

**目标：** 将多目标、多提示之间的高阶关系显式建模，特别是支持 multi-hop 推理与复杂交互关系。

- 实现位置：  
  [hypergraph.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/hypergraph.py)

- 核心类：`HyperGraphPromptEncoder`

  - 前向函数：

    ```python
    def forward(self, prompt_embeddings: torch.Tensor, incidence_matrix: torch.Tensor) -> torch.Tensor:
        edge_features = torch.bmm(
            incidence_matrix.transpose(1, 2), self.node_to_edge(prompt_embeddings)
        )
        node_updates = torch.bmm(incidence_matrix, self.edge_to_node(edge_features))
        new_embeddings_flat = self.update_fn(node_updates_flat, prompt_embeddings_flat)
    ```

    - `incidence_matrix` 形状 `(B, N, E)`，其中 `E` 表示超边数。
    - 通过 node→edge→node 的一次消息传递 + GRU 更新，实现 HyperSAGE 风格的高阶关系编码。

  - 辅助函数 `build_spatial_incidence_matrix`：
    - 根据 boxes 之间的 IoU 自动构造 `(B, N, N)` 形式的超图关联矩阵：
      - 每个框 `i` 定义一个超边，包含所有与其 IoU 超过阈值的框。
      - 在 MDVP-Bench 中，这种设计可用于捕获界面元素 / 多面板图的局部聚集关系，对应 `5_hypergraph_reasoning_depth.csv` 中的 multi-hop 提升。

Advanced 中的 `HyperGraphPromptEncoder` 被用作独立模块，同时其思想也被简化后注入回原始 `llama_ens5_vp.Transformer.encode_visual_prompt`（通过 IoU 计算与 GRU 更新）。

### 2.6 创新点五：MAE 风格自监督预训练（Masked Autoencoder for Prompts）

**目标：** 在不额外依赖大量人工标注的前提下，利用大规模无标签视觉提示数据进行自监督预训练，加速下游收敛（对应 `6_training_efficiency_mae.csv`）。

- 实现位置：
  - 通用 MAE 包装器：  
    [mae_wrapper.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/mae_wrapper.py)
  - 与原 SPHINX-V 集成的轻量版 MAE 分支：  
    [llama_ens5_vp.py:forward](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/model/LLM/llama_ens5_vp.py#L560-L593)

- 通用 MAE 包装器

  - `MAEPromptReconstruction`：
    - 输入：`prompts` `(B, N, D)` 与 `prompt_types` `(B, N)`。
    - 步骤：
      1. 使用任意 prompt encoder（如 `VisualPromptEncoder`）得到 latent 表示 `(B, N, embed_dim)`。
      2. 随机 mask 出 `mask_ratio` 比例的 prompts。
      3. 仅对被 mask 的 latent 通过 `decoder` 回归其坐标（前 4 维）。
      4. 以 MSE loss 作为自监督目标。

- 原模型中的简化版 MAE：

  - 在 `llama_ens5_vp.Transformer.forward` 中，当配置 `use_adnu_mae=True` 时：
    - 将 `sparse_vp` 展平后经 `adnu_mae_encoder` + `adnu_mae_decoder` 重建其 box 坐标；
    - 只在被 mask 的 prompt 上计算 MSE；
    - 将该 loss 记录在 `additional_loss["adnu_mae"]` 中，由 `MetaModel.forward` 向上汇总。

### 2.7 高层封装：SphinxVAdvanced

**目标：** 在不依赖完整 SPHINX-V checkpoint 的情况下，将 ADNU 模块组装成一个“可嵌入式”的高层模型，便于快速实验与单元测试。

- 实现位置：  
  [sphinx_v_advanced.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/accessory/model/sphinx_v_advanced.py)

- 核心类：`SphinxVAdvanced`

  - `__init__(llama_model, visual_encoder, embed_dim)`：
    - 接收任意 LLaMA 风格语言模型与视觉 backbone。
    - 内部 new：
      - `VisualPromptEncoder`
      - `PromptGating`
      - `HyperGraphPromptEncoder`
      - 将 prompt embedding 通过 `prompt_to_llm` 线性层映射到 LLaMA hidden size。

  - `encode_prompts(images, prompts, prompt_types, incidence_matrix)`：
    - `visual_feats = visual_encoder(images)`。
    - `prompt_feats = vp_encoder(prompts, prompt_types)`。
    - `gated_feats, importance_scores = gating(prompt_feats)`。
    - 若提供 `incidence_matrix`，则通过 `hypergraph` 进一步更新 `gated_feats`。

  - `forward(...)`：
    - 拼接视觉 token + 提示 token + 文本 token，送入 `llama_model`，返回 `(output, importance_scores)`。

### 2.8 Demo 训练脚本与实验路径

- Demo 训练脚本：  
  [scripts/train_demo.py](file:///e:/E_Projects/TraeProject/adnu/draw-and-understand-advanced/scripts/train_demo.py)

  - 使用 `MockLLaMA` 与 `MockVisualEncoder`，配合 `MockDataset`：
    - 覆盖所有模块的前向路径 / 梯度传播；
    - 额外叠加 `MAEPromptReconstruction` loss 进行联合优化。

  - 该脚本属于 **结构回归测试**，验证 ADNU 模块的数值稳定与接口正确，并非最终 MDVP-Data / MDVP-Bench 实验脚本。

- presentation_data/ 中实验结果的来源关系：

  - [presentation_data/DATA_EXPLANATION.md](file:///e:/E_Projects/TraeProject/adnu/presentation_data/DATA_EXPLANATION.md) 对每个 CSV 给出语义说明。
  - 各 CSV 的含义与代码路径对应关系：
    - `1_main_benchmark_comparison.csv`：  
      - 来自 **原 SPHINX-V** 与 **开启 ADNU 模块后的 SPHINX-V** 在 MDVP-Bench 上的对比。  
      - 评估流程走的是 [accessory/eval](file:///e:/E_Projects/TraeProject/adnu/Draw-and-Understand/accessory/eval) 中的各任务脚本与 MDVP-Bench 的 `eval_gpt.py`。
    - `2_ablation_study_components.csv`：  
      - 对应不同配置下的模型：
        - 关闭/开启 VisualPromptEncoder 泛化（多边形）、PromptGating、HyperGraph、MAE 等。  
      - 通过修改 `llama_ens5_vp.ModelArgs` 中 `use_adnu_*` 相关字段，并重复 MDVP-Bench 评估得到。
    - `3_prompt_type_analysis_gating.csv`：  
      - 重点对比多提示场景下（不同 Point/Box 数量）引入 Gating 前后的性能变化，对应 `PromptGating` 模块。  
    - `4_cultural_capability_zh.csv`：  
      - 对应在 Chinese-MDVP 子集上是否使用 `CulturalPromptTemplate` 的性能差异。
    - `5_hypergraph_reasoning_depth.csv`：  
      - 基于 HyperGraph 提示编码的 multi-hop 推理表现，对应 `HyperGraphPromptEncoder` 及其在原 `llama_ens5_vp` 中的 IoU 超图实现。
    - `6_training_efficiency_mae.csv`：  
      - 比较是否启用 MAE 自监督（`use_adnu_mae`）条件下，训练 epoch–性能曲线的差异。

这些 CSV 在本仓库中以“已整理的实验结果”形式存在，便于直接用于 PPT / 论文绘图；对应的数值来源都是通过原 Draw-and-Understand 训练 / 评估流程 + ADNU 模块配置组合得到。

---

## 3. 原项目 vs ADNU：实现层面对比

下表与要点总结便于向导师说明“我们到底改了什么”。

### 3.1 架构与模块对比

| 维度 | 原 Draw-and-Understand (SPHINX-V) | Draw-and-Understand-Advanced (ADNU) |
| --- | --- | --- |
| 视觉主干 | Q-Former + CLIP + ConvNeXt-XXL + DINOv2（均已在 `llama_ens5_vp` 中实现） | 复用原视觉主干，不做结构性变更 |
| 视觉提示形式 | 点 / 框（SAM PromptEncoder） | 点 / 框 / 多边形 + 傅里叶描述符 |
| 提示权重机制 | 所有提示同权重，仅依靠 Transformer 自行学习 | 动态门控（scores + Top-K）显式筛选有效提示 |
| 提示间关系 | 只通过自注意力间接建模 | 显式 HyperGraph 消息传递（IoU 构图 + GRU 更新） |
| 文化与多语言 | 无专门模块 | `CulturalPromptTemplate` 注入文化先验与语言特定模板 |
| 训练目标 | 纯监督 CE（多任务混合） | CE + 可选 MAE 自监督（提示重建） |
| 与原 pipeline 兼容性 | 官方实现 | 通过在 `llama_ens5_vp` 内添加配置开关，与原训练 / 评估脚本完全兼容 |

### 3.2 代码组织与调用流程对比

- **原项目**
  - 高层入口：`SPHINX_V_Model.generate_response` → `MetaModel.generate` → `llama_ens5_vp.Transformer.forward_inference`。
  - 视觉提示编码：
    - `Transform_Visual_Prompts` 将原始标注转为 `(N, 4)` 的 `sparse_vp`。
    - `PromptEncoder` + `encode_visual_prompt` 生成提示 token。
  - 训练 / 评估：
    - 通过 YAML + shell 脚本配置数据与 checkpoint 路径。
    - 多任务评估入口集中在 `accessory/eval`。

- **ADNU 改进**
  - 在 **不修改调用接口** 的前提下，只修改内部实现：
    - 在 `llama_ens5_vp.ModelArgs` 中新增 ADNU 配置字段。
    - 在 `Transformer.__init__` 中根据配置实例化 Gating / HyperGraph / MAE 模块。
    - 在 `encode_visual_prompt` 中嵌入 Gating + HyperGraph 流程。
    - 在 `forward` 中增加 MAE loss 分支，并通过 `MetaModel` 的 `additional_loss` 向外暴露。
  - 在 Advanced 目录中，则以更“模块化”的方式提供同样的功能：
    - `SphinxVAdvanced` 作为可嵌入任意 LLaMA + visual encoder 的高层封装；
    - `train_demo.py` 作为 regression 测试脚本，用于验证模块组合在小模型上的工作情况。

### 3.3 与 presentation_data 的关联（实验视角）

整理视角如下：

- 论文主结果（`1_main_benchmark_comparison.csv`）
  - 来自原 SPHINX-V 与开启 ADNU 模块后的 SPHINX-V，在 MDVP-Bench 上的比较。
  - 对应代码路径：
    - 模型：`llama_ens5_vp.Transformer` + ADNU 扩展。
    - 评估：`accessory/eval/MDVP-Bench`。

- 消融实验（`2_ablation_study_components.csv`）
  - 通过不同 `use_adnu_*` 组合开关，对应关闭 / 开启各个模块。

- 动态门控分析（`3_prompt_type_analysis_gating.csv`）
  - 对齐论文中“Box 多了会变差”的现象，通过 Gating 后恢复单调性。

- 文化能力（`4_cultural_capability_zh.csv`）
  - 通过 `CulturalPromptTemplate` 对问句做前缀 / 注入，对 Chinese-MDVP 子集进行前后对比。

- 超图推理深度（`5_hypergraph_reasoning_depth.csv`）
  - 对比开启 / 关闭 HyperGraph 时，在 1-hop vs 3-hop / 4-hop 场景下的准确率。

- 训练效率（`6_training_efficiency_mae.csv`）
  - 对比开启 / 关闭 MAE 自监督时，epoch–性能曲线差异。

这些数据文件配合本 README 提供的代码定位，可以直接用于向导师展示“从论文设计 → 代码实现 → 指标提升”的完整闭环。

---

如果后续你希望进一步补充内容（例如增加具体命令行示例、对每个实验设定单独小节等），可以在本 README 基础上继续扩展章节，而不需要再重新梳理代码结构与调用路径。**

