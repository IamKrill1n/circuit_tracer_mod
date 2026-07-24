# Tóm tắt:
- ADAG (preprint): giống cách hiện tại của mình nhất, analytic 
- Circuit Oracle (ICML 2026 Workshop): query-driven, hỏi gì thì ra một đồ thị con trả lời cho câu hỏi đó. Dùng AI agent khám phá đồ thị.
- Probe Prompting (preprint): gen vài prompt gần giống để mò các đỉnh giống role nhau. 
- Mỗi bài evaluate một kiểu
- Không bài nào đề cập đến vấn đề gộp đỉnh vào thì sẽ mất cạnh mất thông tin như mình.
- Ý tưởng query-driven khá hay và dễ áp vào pipeline của mình, thay SHAP bằng query-driven mask sẽ hợp lý hơn.

# Automation of Analyzing Attribution Graphs: A Comparative Literature Review

## Executive Summary
The emergence of attribution graphs as a primary representation for mechanistic interpretability in large language models (LLMs) has necessitated the development of automated tools to manage their complexity. Traditional manual circuit-tracing methods are labor-intensive and require significant expertise. This report compares three recent advancements in this domain: **arXiv:2511.07002** (Probe Prompting), **arXiv:2604.07615** (ADAG), and the **Circuit Oracle** framework. 

These approaches represent a spectrum of automation strategies, from rule-based pipelines like Probe Prompting that classify features into stable functional roles, to the end-to-end multi-agent interpretation of ADAG, and the query-driven, tool-equipped agents of Circuit Oracle. While all three aim to reduce the manual burden of interpretability, they differ significantly in their tracing backbones, the form of their outputs (ranging from compact supernodes to natural language answers), and their validation strategies (causal interventions vs. proxy task accuracy). This review provides a comparative analysis of their inputs, methodologies, outputs, and experimental validation techniques, highlighting the move toward more modular and task-specific automated interpretability.

## Table of Contents
1. [Introduction](#1-introduction)
2. [Comparative Analysis of Inputs](#2-comparative-analysis-of-inputs)
3. [Comparative Analysis of Outputs](#3-comparative-analysis-of-outputs)
4. [Comparative Analysis of Methodology](#4-comparative-analysis-of-methodology)
5. [Experimental Approach and Validation](#5-experimental-approach-and-validation)
6. [Findings and Comparative Analysis](#6-findings-and-comparative-analysis)
7. [Limitations of the Evidence](#7-limitations-of-the-evidence)
8. [Conclusion](#8-conclusion)
9. [References](#9-references)

## 1. Introduction
Mechanistic interpretability aims to reverse-engineer the internal computations of neural networks into human-understandable components. A key artifact in this process is the attribution graph, which traces the causal influence between interpretable features (often extracted via sparse autoencoders or transcoders). However, as models scale, these graphs can contain thousands of nodes and edges, making manual analysis intractable. 

Recent research has focused on automating the analysis of these graphs. This report examines three contemporary methods:
- **Probe Prompting (arXiv:2511.07002)**: A rule-based pipeline that uses targeted "probe prompts" to identify the functional roles of features and group them into concept-aligned supernodes [1].
- **ADAG (arXiv:2604.07615)**: An end-to-end circuit interpretation pipeline that quantifies feature roles via "attribution profiles" and uses an explainer-simulator framework to generate natural language descriptions [2].
- **Circuit Oracle**: A multi-agent system that treats attribution graphs as data for a frontier LLM equipped with graph-traversal tools, allowing for query-driven analysis of safety-relevant behaviors [3].

## 2. Comparative Analysis of Inputs
The analyzed methods differ in their underlying circuit-tracing backbones and the specific settings used to generate the attribution graphs that serve as their primary inputs.

### 2.1 Tracing Backbones and Substrates
The substrate from which the attribution graphs are derived varies across the three papers:
- **arXiv:2511.07002**: Utilizes the **Neuronpedia API** for attribution graph generation, which relies on Sparse Autoencoder (SAE) features to represent model activations. This substrate provides a sparse representation of neurons, allowing for more interpretable nodes than raw activations [1].
- **arXiv:2604.07615**: Employs **gradient-based attribution** on the **Llama 3.1 8B Instruct** model. It identifies important features and their interaction weights directly to construct the graph, focusing on the model's instruction-following capabilities [2].
- **Circuit Oracle**: Leverages **transcoders** to decompose language model computations into sparse, interpretable features. This approach represents model computation as a directed weighted graph where nodes are transcoder features, token embeddings, or output logits. Transcoders are specifically chosen for their ability to provide causal edges between computational units [3].

### 2.2 Graph Settings and Pruning Strategies
The complexity of the input graphs is managed through various thresholds and pruning strategies:
- **arXiv:2511.07002**: Implements strict thresholds for node inclusion (0.8) and edge influence (0.85). It limits the initial graph to a maximum of 5,000 feature nodes and applies a cumulative influence filter, retaining only nodes whose sum of influence reaches at least 0.95. This reduces circuits from thousands of features down to a more manageable 200–700 features [1].
- **arXiv:2604.07615**: Uses "attribution profiles" to quantify the functional roles of features. These profiles measure input attributions and output logit contributions across a dataset. The methodology also involves "attribution description thresholds" to refine the interpretation process, focusing on overcoming "locality bias" by considering dependencies on preceding tokens [2].
- **Circuit Oracle**: Does not rely on a single global pruning strategy but instead uses a **"query-driven" approach**. The system prompt and task-specific tools allow the agent to traverse the graph dynamically, inspecting "promising nodes" based on the user's specific question rather than committing to a pre-pruned representation. This makes it more flexible for answering diverse safety questions [3].

## 3. Comparative Analysis of Outputs
The final results produced by these automated systems vary in their representation, ranging from structured graph components to free-form natural language.

### 3.1 Smaller Interpreted Graphs and Results
- **arXiv:2511.07002 (Compact Supernodes)**: The primary output is a set of **concept-aligned supernodes**. These supernodes are formed by merging features that share the same functional role and name. The output is structured according to a four-part taxonomy:
  - **SEM-DICT**: Semantic-Dictionary features peaking on the same token.
  - **SEM-CONC**: Semantic-Concept features peaking across related tokens.
  - **REL**: Relationship features encoding context.
  - **SAY-X**: Features promoting specific functional tokens [1].
- **arXiv:2604.07615 (Natural Language Descriptions)**: The output consists of **natural language labels and descriptions** for grouped features. These descriptions are generated by an "explainer" LLM and verified by a "simulator" LLM, providing a semantic narrative of what each cluster of features is doing [2].
- **Circuit Oracle (Query Responses)**: Circuit Oracle produces **natural language answers** to user-specified high-level questions. The output is derived through multi-agent reasoning over the graph's structure and labels. It can also produce task-specific artifacts like shortlists of lemmas or usability scores for jailbroken completions [3].

### 3.2 Form and Representation
- **Taxonomic Representation**: arXiv:2511.07002 uses a fixed taxonomy where each supernode is labeled based on its role and the specific concept it detects (e.g., 'Texas' or 'Say (Austin)') [1]. 
- **Clustered Representation**: arXiv:2604.07615 represents findings as **steerable clusters**. For example, in safety analysis, it identifies clusters responsible for specific behaviors like jailbreaking [2].
- **Static vs. Query-Driven**: While arXiv:2511.07002 and arXiv:2604.07615 produce static descriptions or groupings, Circuit Oracle is **query-driven**, re-deriving feature interpretations based on the task at hand [3].

## 4. Comparative Analysis of Methodology
The core logic for interpreting nodes and grouping them differs between rule-based, clustering, and multi-agent approaches.

### 4.1 Node Interpretation and Role Assignment
- **arXiv:2511.07002 (Rule-Based Probe Prompting)**: Assigns functional roles using a **strict-priority decision tree**. This is based on **Cross-Prompt Activation Signatures (CPAS)**, which aggregate per-probe measures like peak activation, sparsity, z-score, and cosine similarity [1].
- **arXiv:2604.07615 (Attribution Profiles)**: Interprets features by measuring their **input attributions and output logit contributions**. This method specifically seeks to overcome "locality bias" by considering effects on future outputs [2].
- **Circuit Oracle (Multi-Agent Inspection)**: Uses a multi-agent system where an orchestrator dispatches subagents to trace upstream paths and inspect nodes. The oracle uses an off-the-shelf frontier LLM equipped with tools to traverse the directed weighted graph [3].

### 4.2 Grouping and Clustering Strategies
- **arXiv:2511.07002**: Groups features into supernodes based on **shared functional roles and names**. Stability is a key requirement, with features needing at least 60% consistency across probes to be included in a supernode [1].
- **arXiv:2604.07615**: Employs **multi-view spectral clustering** based on attribution profiles. The algorithm aims to group functionally similar features while avoiding imbalanced clusters and preventing the mixing of features with opposing output effects [2].
- **Circuit Oracle**: Uses modular **"skills"** that bundle system prompts with shared graph-traversal tools and task-specific tools. This modular design makes the framework extensible for new safety questions without committing to a global grouping [3].

## 5. Experimental Approach and Validation
Each paper employs a distinct strategy to validate that its automated interpretations are faithful to the model's actual computations.

### 5.1 Validation Strategies
- **arXiv:2511.07002 (Causal Interventions)**: Uses **additive entity-swap interventions**. It tests if suppressing features related to a source entity and amplifying those for a target entity redirects the model's answer. This is conducted across four factual domains (USA, Books, Products, Paintings) [1].
- **arXiv:2604.07615 (Replication and Simulation)**: Validates results by **replicating human-led studies** (e.g., on the Capitals dataset) and using a **simulator LLM** to score the accuracy of descriptions via Pearson correlation coefficients [2].
- **Circuit Oracle (Proxy Tasks and LLM-Judge)**: Evaluates performance on **three safety-relevant proxy tasks**: detecting spurious correlations, eliciting hidden knowledge, and jailbreaking suppression behaviors. Validation is measured through task accuracy and LLM-judge ensemble scores [3].

### 5.2 Metrics and Baselines
- **arXiv:2511.07002**: Reports **Hit%** (target token presence) and **vsMax** (logit margin). It uses matched-random and influence-matched top-K controls to ensure the effect is due to concept-aligned grouping rather than feature count or influence alone [1].
- **arXiv:2604.07615**: Focuses on **Pearson correlation** between predicted and true scores for descriptions. It also uses steering to identify clusters responsible for harmful behaviors [2].
- **Circuit Oracle**: Compares against **task-specific baselines** such as SAE-cosine and transcoder-cosine for spurious features, and diff-in-mean refusal-direction methods for jailbreaking [3].

## 6. Findings and Comparative Analysis
The comparative analysis reveals a clear evolution in the automation of attribution graph analysis:

| Feature | Probe Prompting (arXiv:2511.07002) | ADAG (arXiv:2604.07615) | Circuit Oracle |
| :--- | :--- | :--- | :--- |
| **Automation Strategy** | Rule-based decision tree | Multi-agent explainer-simulator | Multi-agent query-driven |
| **Grouping Method** | Role/Name-based supernodes | Multi-view spectral clustering | Task-specific skill modules |
| **Primary Output** | Concept-aligned supernodes | Natural language descriptions | Natural language answers |
| **Validation** | Causal entity-swap interventions | Human study replication | Safety proxy tasks |
| **Key Metric** | Hit%, vsMax | Pearson correlation | Task accuracy, LLM-judge |

While Probe Prompting focuses on **structured stability** and causal verification, ADAG emphasizes **semantic clarity** through end-to-end descriptions. Circuit Oracle represents a more **flexible, agentic approach** that treats the graph as an interactable database for safety analysis.

## 7. Limitations of the Evidence
Several limitations are highlighted across the sources:
- **Exclusion of Low-Stability Features**: arXiv:2511.07002 excludes features with < 60% stability across probes, which might miss context-dependent features [1].
- **Evaluation on Limited Datasets**: Many experiments are conducted on factual recall (Capitals, Books) or specific safety scenarios (Pills), which may not cover the full complexity of model behaviors [1][2].
- **Model-Specific Biases**: ADAG notes the potential for "locality bias" in prior methods, while Circuit Oracle's performance is intrinsically linked to the reasoning capabilities of the frontier LLM used as the oracle [2][3].

## 8. Conclusion
The automation of attribution graph analysis has advanced from manual expert inspection to sophisticated automated pipelines. **arXiv:2511.07002** provides a robust taxonomy and causal validation for supernodes. **arXiv:2604.07615** automates semantic description through a powerful interpretation pipeline. **Circuit Oracle** offers a query-driven, agentic framework for flexible safety analysis. Together, these methods demonstrate that automated mechanistic interpretability is a viable and increasingly necessary path for understanding and securing large-scale language models.

## 9. References
[1] arXiv:2511.07002, "Interpreting Attribution Graphs via Probe Prompting: Taxonomy, Methodology, and Experimental Approach," 2025.
[2] arXiv:2604.07615, "ADAG: An End-to-End Circuit Interpretation Pipeline," 2026.
[3] "Circuit Oracle: Automating Attribution Graph Analysis via Natural-Language Queries" ICML Workshop 2026 (https://openreview.net/pdf?id=LmkgcUJA8N)
