# Draw-and-Understand Advanced Scaffold

This repository contains the code scaffold for the improved SPHINX-V model, designed to meet SCI Q1 publication standards. It incorporates 5 key innovations based on the ADNU proposal.

## Directory Structure

```
Draw-and-Understand-Advanced/
├── accessory/
│   ├── model/
│   │   ├── sphinx_v_advanced.py    # Main model integrating all innovations
│   │   ├── visual_prompt_encoder.py # Innovation 1: Shape Generalization (Box/Point/Polygon)
│   │   ├── gating.py               # Innovation 2: Dynamic Gating
│   │   ├── hypergraph.py           # Innovation 4: Hyper-Graph Relation Reasoning
│   │   └── mae_wrapper.py          # Innovation 5: Self-supervised Pre-training
│   ├── data/
│   │   └── template.py             # Innovation 3: Cultural-aware Prompt Templates
│   └── ...
├── scripts/                        # Training scripts (placeholders)
└── docs/
```

## Innovation Details

### 1. Visual Prompt Shape Generalization
**File:** `accessory/model/visual_prompt_encoder.py`
**Goal:** Support arbitrary shapes (Mask/Polygon) beyond Box/Point.
**Implementation:** Uses Fourier Descriptors or Bezier curve parameters to encode polygon prompts into the same embedding space.

### 2. Dynamic Gating for Prompt Importance
**File:** `accessory/model/gating.py`
**Goal:** Solve the performance inversion between Box and Point prompts in multi-prompt scenarios.
**Implementation:** A learnable gating network that assigns importance scores to each prompt token, allowing the model to focus on relevant prompts.

### 3. Multilingual Bias Mitigation
**File:** `accessory/data/template.py`
**Goal:** Improve performance on non-English data (e.g., Chinese).
**Implementation:** Cultural-aware prompt templates and injection of cultural priors during inference.

### 4. Multi-prompt/Multi-target Relation Reasoning
**File:** `accessory/model/hypergraph.py`
**Goal:** Explicitly model high-order relations between multiple prompts.
**Implementation:** A Hyper-Graph Neural Network (HyperSAGE) module that updates prompt embeddings based on their relationships (defined by an incidence matrix).

### 5. Self-supervised Pre-training
**File:** `accessory/model/mae_wrapper.py`
**Goal:** Reduce dependency on expensive labeled data.
**Implementation:** An MAE-style wrapper that masks a portion of prompt tokens and trains the model to reconstruct them.

## Usage

1.  **Install Dependencies:** `pip install -r requirements.txt` (Create this based on base repo)
2.  **Train:** Use scripts in `scripts/` folder.
3.  **Integrate:** Copy these files into the original `Draw-and-Understand` repository structure if needed, or use this as a standalone codebase importing the base `accessory` library.
