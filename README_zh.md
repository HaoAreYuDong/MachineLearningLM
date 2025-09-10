# Machine Learning LLM

README.md | README_zh.md

📄 论文: https://arxiv.org/abs/2509.06806

🤗 huggingface: https://huggingface.co/MachineLearningLM

**预训练数据集**

部分数据集已在 Hugging Face 开源。完整数据集托管在 Google Drive：

- 🔹 **warmup数据集**:
  https://drive.google.com/file/d/1OjD0jICy95lOFp52_2hJoO7KzSiFegLH/view?usp=sharing

- 🔹 **完整数据集**:
  https://drive.google.com/file/d/1TYsEMI1WNYDzzE_z83Ah-QAmcoaVHKQA/view?usp=sharing

## evaluate框架

一个用于评估大语言模型在机器学习任务上性能的综合框架，支持传统机器学习模型和深度学习方法，并提供自动化流水线处理。

### 概述

该框架提供端到端的LLM机器学习任务评估能力，包含自动化数据预处理、提示生成、模型推理和综合评估指标。

### 重要说明

1. **特殊字符处理**：由于shell保留字符的原因，TALENT数据集中的CSV文件名可能包含特殊字符（如`(`）。我们建议在处理前预处理这些文件名，仅使用数字、字母和下划线。

2. **文本数据处理**：虽然我们支持文本数据处理，但由于使用逗号（`,`）作为特征分隔符，请替换数据集文本中的任何逗号以避免模型混淆。在我们的评估中，我们使用空格替换逗号。
3. 框架中的XGBoost目前仅用于测试目的。它仅支持二分类，且未包含在论文的实验中。

### 安装

```bash
# 安装Python依赖
pip install -r requirements.txt
```

### 批量处理使用

对于批量处理，您需要提供**输入路径**和**输出路径**参数。框架支持三种执行模式：

#### 步骤1：激活参数
```bash
source ./scripts/parameters.sh
```

#### 步骤2：选择执行模式
根据您偏好的处理模式（顺序、并行或端到端流水线），参考下面的"执行选项"部分查看详细命令。

### 执行选项

#### 选项1：顺序处理
使用`single_process/`目录中的脚本按顺序运行步骤：
```bash
./scripts/single_process/data_prep.sh
./scripts/single_process/prompt_gen.sh  # 仅适用于深度学习
./scripts/single_process/model_pred.sh
./scripts/single_process/evaluation.sh
./scripts/single_process/report.sh
```

#### 选项2：并行处理
使用`multi_process/`目录中的脚本进行加速并行执行：
```bash
./scripts/multi_process/data_prep.sh
./scripts/multi_process/prompt_gen.sh  # 仅适用于深度学习
./scripts/multi_process/model_pred.sh
./scripts/multi_process/evaluation.sh
./scripts/multi_process/report.sh
```

#### 选项3：端到端流水线
使用并行化优化运行完整流水线：
```bash
./scripts/evaluate_pipeline.sh
```

### 单文件处理

对于单个JSONL文件的直接推理，我们支持单文件处理模式。

**重要**：输入文件必须具有`.jsonl`扩展名 - 代码逻辑使用此后缀进行文件类型识别。

文件应包含LLaMA Factory的Alpaca格式的提示，结构如下：
- `instruction`: 任务指令
- `input`: 输入数据
- `output`: 期望输出

#### 本地模型使用示例
```bash
python ./src/evaluation/model_pred/dl_model_pred.py \
  --input_dir ./demo_input.jsonl \
  --output_dir ./demo_output.jsonl \
  --model_name MachineLearningLM/MachineLearningLM-7B-v1
```

#### 云模型使用
对于云模型调用，模型路径必须以`openai::`开头，以便正确解析和执行OpenAI SDK：

```bash
python3 ./src/evaluation/model_pred/dl_model_pred.py \
  --input_dir ./input_demo.jsonl \
  --output_dir ./output_demo.jsonl \
  --model_name openai::gpt-4o-mini \
  --api_key your_own_api_key \
  --base_url your_own_base_url \
  --max_samples 5
```

### 单文件评估

您也可以直接对单个文件进行评估：

```bash
python ./src/evaluation/result_proc/evaluator.py \
  --input_dir ./demo_response.jsonl \
  --output_dir ./output_demo.txt   # 也可以是.jsonl
```

**注意**：我们的评估框架专门设计用于处理由我们的`dl_model_pred`推理流水线生成的结果。请使用我们推理模块的输出作为评估输入以确保兼容性。

### 配置

所有参数都通过`./scripts/parameters.sh`管理。修改此文件以自定义：
- 输入/输出路径
- 模型配置
- 处理参数
- 评估设置

### 功能特点

- **双模型支持**：传统机器学习和深度学习模型
- **灵活处理**：单进程或多进程执行
- **自动化流水线**：端到端工作流自动化
- **单文件支持**：直接对单个JSONL文件进行推理
- **综合评估**：多指标评估框架
- **并行优化**：内置并行化以提高性能

## Tabicl 评估

**这部分代码需要在安装了tabicl和openpyxl库的环境中运行。**

Tabicl的评估代码单独放置在`./src/evaluation/tabicl_evaluate.py`文件中。使用`./scripts/tabicl_evaluate.sh`获取tabicl的评估结果。

使用--datasets指定要评估的数据集，使用--sample_sizes指定样本数量。

如果需要评估多个数据集，请用空格分隔。要评估输入文件夹中的所有CSV文件，请使用**all**。

## 先验数据

MachineLearningLM使用tabicl的代码生成先验数据。

使用`./scripts/generate_data.sh`生成先验数据。它会生成相应的.pt和.csv文件，并将CSV文件中的特征值归一化到0-999的范围，正如我们在论文中所做的那样。

### 参数介绍（参考文件`tabicl\src\tabicl\prior\dataset.py`中的注释）

**数据规模与结构**

| 参数           | 类型 | 描述                       |
| :------------- | :--- | :------------------------- |
| `min_features` | int  | 每个数据集的最小特征数     |
| `max_features` | int  | 每个数据集的最大特征数     |
| `max_classes`  | int  | 目标类的最大数量           |
| `min_seq_len`  | int  | 每个数据集的最小样本数     |
| `max_seq_len`  | int  | 每个数据集的最大样本数     |

**批处理配置**

| 参数                 | 类型 | 描述                             |
| :------------------- | :--- | :------------------------------- |
| `batch_size`         | int  | 每批生成的数据集总数             |
| `batch_size_per_gp`  | int  | 每组数据集的数量（共享特征）     |
| `batch_size_per_subgp` | int  | 每个子组的数据集数量（具有相似因果结构） |

**序列长度控制**

| 参数           | 类型 | 描述                             |
| :------------- | :--- | :------------------------------- |
| `log_seq_len`  | bool | 是否从对数均匀分布中采样序列长度 |
| `seq_len_per_gp` | bool | 是否按组采样序列长度             |
| `replay_small` | bool | 是否偶尔采样较小序列以提高模型鲁棒性 |

**训练-测试分割**

| 参数           | 类型      | 描述                             |
| :------------- | :-------- | :------------------------------- |
| `min_train_size` | int/float | 训练分割的起始位置/比例          |
| `max_train_size` | int/float | 训练分割的结束位置/比例          |

**生成方法**

| 参数       | 类型 | 描述                             |
| :--------- | :--- | :------------------------------- |
| `prior_type` | str  | 先验类型：'mlp_scm', 'tree_scm', 或 'mix_scm' |
| `fixed_hp` | dict | 固定结构配置参数                 |
| `sampled_hp` | dict | 生成过程中采样的参数             |

**计算设置**

| 参数                     | 类型 | 描述                       |
| :----------------------- | :--- | :------------------------- |
| `n_jobs`                 | int  | 并行作业数（-1=使用所有处理器） |
| `num_threads_per_generate` | int  | 每个生成作业的线程数       |
| `device`                 | str  | 计算设备（'cpu'或'cuda'）  |

## 训练

MachineLearningLM使用LLaMA-Factory框架进行训练。

#### 训练环境配置

```bash
cd ./third_party/LLaMA-Factory
pip install -e ".[torch,metrics]" --no-build-isolation
pip install wandb
```

使用`./scripts/train.sh`进行训练。

## 项目结构

```
MachineLearningLM/
├──src/
|   ├──evaluation/
│   │   ├── data_prep/          # 数据预处理和分块工具
│   │   ├── prompt_gen/         # 深度学习模型的提示生成
│   │   ├── model_pred/         # 模型推理（ML和DL预测引擎）
│   │   ├── result_proc/        # 5层评估架构和指标处理
│   │   ├── zero_summary/       # 结果总结和报告生成
│   │   └── tabicl_evaluate.py
│   └──prior_data
│       └── pt_to_csv.py     
├── scripts/
│   ├── single_process/         # 顺序执行shell脚本
│   ├── multi_process/          # 并行执行shell脚本（带_mp后缀）
│   ├── evaluate_parameters.sh  # 全局参数配置
|   ├── evaluate_pipeline.sh    # 自动化流水线
|   ├── generate_data.sh
|   ├── tabicl_evaluate.sh
|   └── train.sh
├── datahub_inputs/
│   ├── data_demo/          # 用于测试的演示数据集
│   └── data_raw/           # 原始输入数据集
├── third_party/
│   ├── tabicl/          
│   └── LLaMA-Factory/   
├── requirements.txt        # 评估框架的Python依赖
├── README.md
├── THIRD_PARTY_NOTICES.md
└── LICENSE
```
