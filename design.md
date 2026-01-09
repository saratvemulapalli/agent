# OpenSearch Solution Architect Agent 设计文档

## 1. 系统概述

本系统是一个基于 LLM 的智能代理，旨在作为 OpenSearch 解决方案架构师，引导用户完成从初始需求分析到最终语义搜索方案落地的全过程。它通过多智能体协作（Multi-Agent Collaboration），结合内部知识库，提供专业的 OpenSearch 语义搜索技术建议和实施方案。

## 2. 系统架构

系统由一个主协调器（Orchestrator）和三个专用子代理（Sub-Agents/Tools）组成。

### 2.1 核心组件

#### 1. Orchestrator (协调器) - `orchestrator.py`
- **角色**: 用户交互的主入口和流程控制中心。
- **职责**:
  - 管理对话状态和上下文。
  - 引导用户提供关键需求信息（文档量、语言、预算、延迟要求、部署偏好等）。
  - 根据对话阶段调用相应的子代理。
  - 最终确认方案并触发执行。
- **模型**: `us.anthropic.claude-sonnet-4-5-20250929-v1:0` (启用 Thinking 模式)。

#### 2. Solution Planning Assistant (方案规划助手) - `solution_planning_assistant.py`
- **角色**: 资深搜索架构专家。
- **职责**:
  - 接收用户需求上下文。
  - 查阅内部知识库 (`read_knowledge_base`)。
  - 生成详细的技术推荐方案，包括检索策略（BM25/向量/混合）、索引变体（HNSW/IVF等）和模型部署选项。
  - 输出结构化的 `<conclusion>` 供协调器解析。

#### 3. OpenSearch QA Assistant (问答助手) - `opensearch_qa_assistant.py`
- **角色**: OpenSearch 语义搜索技术顾问。
- **职责**:
  - 在方案迭代阶段，回答用户关于具体技术细节、权衡或概念的疑问。
  - 基于知识库提供准确、简洁的解答。
  - 帮助用户理解方案并进行微调。

#### 4. Worker Agent (执行代理) - `worker.py`
- **角色**: 自动化实施工程师。
- **职责**:
  - 接收最终确认的技术方案。
  - (目前为占位符) 模拟执行索引创建、模型配置等操作。

### 2.2 辅助工具
- **Data Analysis Tool** (`scripts.tools`): 提供 `get_sample_doc` 工具，获取用户数据样本（Mock或手动输入）以便分析数据结构和内容特征。
- **Knowledge Base Tools** (`scripts.tools`): 提供 `read_knowledge_base` 工具，使代理能够访问 OpenSearch 语义搜索的最佳实践和技术文档。

## 3. 工作流程 (Workflow)

系统遵循以下五个主要阶段：

1.  **数据分析 (Data Analysis)**
    - 协调器**首先**调用 `get_sample_doc` 获取用户数据样本。
    - 分析样本以推断数据结构、语言、内容类型（如日志、电商、文章等）。

2.  **需求澄清 (Clarify Requirements)**
    - 基于数据分析的结果，协调器主动询问用户**剩余的**关键指标：数据规模、预算/成本偏好、延迟要求 (P99)、精度-延迟权衡、模型部署方式等。
    - 避免询问已通过数据分析获知的信息（如语言）。

3.  **方案提案 (Proposal)**
    - 一旦收集到足够信息，协调器调用 `solution_planning_assistant`。
    - 规划助手分析需求，生成初始技术推荐方案。
    - 协调器向用户展示方案。

4.  **方案迭代与精炼 (Refinement)**
    - 用户对方案提出疑问或修改意见。
    - 协调器调用 `opensearch_qa_assistant` 解答疑问，或重新调用规划助手调整方案。
    - 此过程循环进行，直到用户明确确认满意 ("Yes, let's do it")。

5.  **执行 (Execution)**
    - 用户确认后，协调器调用 `worker_agent`。
    - 执行代理根据最终方案完成系统配置（模拟）。
    - 向用户报告完成状态。

## 4. 技术细节
- **框架**: `strands` (Agent Framework)
- **模型服务**: AWS Bedrock
- **核心模型**: Claude 3.5 Sonnet (v2) with Thinking Blocks
- **记忆机制**: 通过 Orchestrator 维护对话历史和状态。
