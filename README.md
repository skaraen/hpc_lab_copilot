# HPC Lab Copilot  
**A Retrieval-Augmented Generative AI System for Understanding, Debugging, and Optimizing HPC Code**

## 1. Prerequisites

### Software Requirements
- Python ≥ 3.9
- Streamlit
- FAISS (CPU version)
- OpenAI Python SDK
- CUDA Toolkit (only required for generating profiler artifacts)
- nvprof (legacy CUDA profiler, typically available on HPC clusters)

### Environment Assumptions
- The Streamlit app runs on a local machine or login node.
- CUDA binaries and profiling are performed separately on GPU nodes (e.g., via SLURM).
- All reasoning is grounded strictly in user-provided artifacts.

## 2. Setup and Running the Code

### Clone and Environment Setup

```bash
git clone https://github.com/skaraen/hpc_lab_copilot
cd hpc_lab_copilot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure OpenAI Access

```bash
export OPENAI_API_KEY="your_api_key_here"
```

### Populate the RAG Corpus

Artifacts are organized as follows:
- data/code_snippets/ : CUDA and C/C++ source files
- data/logs/          : nvprof outputs and SLURM scripts
- data/docs/          : HPC and CUDA reference notes

After adding or modifying artifacts, rebuild the index:

```bash
python -m rag.indexing
```

### Launch the Application

```bash
streamlit run app.py
```

The UI will be accessible at http://localhost:8501

## 3. Overview

This project explores how Generative AI systems can be meaningfully applied to **high-performance computing (HPC)** workflows, where correctness, performance, and reproducibility are critical. Rather than treating large language models as generic code generators, this work frames them as **analysis and reasoning assistants**, grounded in user-provided artifacts such as CUDA code, profiler logs, and job scripts.

The resulting system, **HPC Lab Copilot**, is a Streamlit-based application that combines:
- OpenAI language models for reasoning and explanation,
- Retrieval-Augmented Generation (RAG) for grounding responses in real code and logs,
- attention-style relevance scoring to improve transparency and trust,
- and explicit handling of uncertainty to avoid hallucinated claims.

The project is motivated by real HPC development workflows, including CUDA kernels, SLURM batch scripts, and `nvprof` performance traces.

## 4. Motivation and Design Rationale

HPC development differs fundamentally from casual programming:
- Code is tightly coupled to hardware and runtime environments.
- Incorrect advice can waste expensive compute allocations.
- Performance issues are often subtle and data-dependent.
- Logs and profiler outputs are as important as the code itself.

Pure prompting of LLMs performs poorly in this regime, often producing confident but ungrounded explanations. This project therefore centers on **retrieval and grounding**, treating user-supplied artifacts as first-class inputs rather than optional context.

The guiding principles were:
1. **Ground whenever possible** – analysis should reference actual code and logs.
2. **Be explicit about uncertainty** – say when the system lacks evidence.
3. **Separate conceptual knowledge from grounded claims**.
4. **Favor interpretability over black-box answers**.

## 5. System Components

### 5.1 Frontend (Streamlit)

The Streamlit UI provides a lightweight but expressive interface with multiple interaction modes:
- Code and log analysis
- Research-style RAG queries
- Performance diagnosis using profiler outputs

Key UI features include:
- Adjustable model parameters (temperature, top-k retrieval)
- Visibility into retrieved context
- Clear separation between answers and supporting evidence

The UI is intentionally minimal to keep focus on reasoning rather than presentation.

### 5.2 Language Model Interface

The system uses OpenAI’s text generation and embedding endpoints via a thin abstraction layer. The model is responsible for:
- Explaining code and logs in human-readable terms
- Synthesizing information across retrieved artifacts
- Proposing improvements while respecting scope and uncertainty

Importantly, the model is **not** trusted to invent missing facts. All prompts explicitly instruct it to defer when evidence is insufficient.

### 5.3 Retrieval-Augmented Generation (RAG)

#### Corpus Structure

The knowledge base is divided by artifact type:
```bash
    data/
    ├─ code_snippets/ # CUDA, C/C++, helper code
    ├─ logs/ # nvprof outputs, run logs, sbatch scripts
    └─ docs/ # curated CUDA / HPC notes
```


Each file is chunked, embedded, and indexed using FAISS. Metadata such as file path and artifact type are preserved for transparency.

#### Retrieval Process

For each user query:
1. The query is embedded.
2. Top-k similar chunks are retrieved using cosine similarity.
3. Retrieved chunks are passed to the language model as grounding context.
4. Relevance scores are normalized to act as **attention weights** for visualization.

This allows the system to infer relationships between code, profiler logs, and scripts through semantic similarity rather than hard-coded mappings.

### 5.4 Attention-Style Relevance Scoring

While internal transformer attention is not exposed, the system approximates **interpretability** by:
- assigning normalized relevance scores to retrieved chunks,
- visually emphasizing higher-weight context in the UI.

This helps users understand *why* the model focused on certain artifacts and builds trust in the output.

### 5.5 Profiler Integration (nvprof)

The system supports ingestion and analysis of `nvprof` outputs:
- Kernel timing summaries
- Memory transfer overheads
- API call breakdowns

Profiler logs can be:
- pre-indexed as part of the RAG corpus, or
- pasted ad-hoc for one-off analysis.

In both cases, the system avoids over-interpreting metrics that were not collected.

## 6. Workflow

A typical workflow looks like this:

1. **Artifact ingestion**
   - User adds CUDA code, profiler logs, or notes to the appropriate folder.
   - The RAG index is rebuilt.

2. **Query submission**
   - User asks a question (e.g., performance diagnosis, explanation, comparison).

3. **Retrieval**
   - Relevant code, logs, and documentation are retrieved based on semantic similarity.

4. **Grounded generation**
   - The LLM generates an answer constrained by the retrieved context.

5. **Transparency**
   - Retrieved context and relevance scores are displayed alongside the answer.

6. **User judgment**
   - The user decides how to act on the information, with visibility into evidence and uncertainty.

## 7. Grounded vs Conceptual Reasoning

A deliberate design choice in this project is the separation between:
- **grounded analysis** (based on uploaded artifacts), and
- **conceptual explanations** (based on general HPC knowledge).

Examples:
- Comparing two uploaded CUDA implementations → **grounded**
- Explaining why stencil codes are memory-bound → **conceptual**
- Diagnosing an unloaded codebase → **explicitly refused or generalized**

This distinction is critical to avoiding hallucinations and is treated as a feature rather than a limitation.

## 8. Improvements Achieved Over Naive LLM Usage

Compared to pure prompting, this system achieves:

- **Reduced hallucination**  
  Answers are constrained to retrieved evidence when available.

- **Better performance reasoning**  
  Profiler-driven explanations replace generic CUDA advice.

- **Higher trust and transparency**  
  Users can see what evidence influenced each answer.

- **Scalability across projects**  
  Adding new HPC code automatically extends system capability.

- **Clear failure modes**  
  Low-similarity queries trigger uncertainty instead of fabrication.

## 9. Responsible AI Considerations

Several Responsible AI principles are enforced by design:

- **Evidence-based claims**  
  The system avoids grounding claims without supporting artifacts.

- **Uncertainty signaling**  
  Low retrieval similarity triggers warnings rather than confident answers.

- **No implicit data leakage**  
  Only user-provided artifacts are used for grounding.

- **Separation of authority**  
  The model assists reasoning but does not replace human judgment.

These considerations are particularly important in HPC, where incorrect guidance can have real computational and scientific cost.

## 10. Limitations

- Retrieval quality depends on corpus quality and chunking strategy.
- Semantic similarity is an implicit, not explicit, linkage mechanism.
- Profiler interpretation is limited to metrics actually collected.
- The system does not yet perform automated code correctness checking.

These limitations are explicitly acknowledged in both prompts and UI behavior.

## 11. Future Work

Several natural extensions emerge from this project:

1. **Explicit artifact linking**
   - Group code, logs, and scripts by executable or project metadata.

2. **Structured profiler parsing**
   - Convert profiler outputs into normalized feature representations.

3. **Fine-tuned domain models**
   - Train lightweight models on HPC-specific Q&A or explanations.

4. **Patch-style code suggestions**
   - Emit diffs rather than standalone code blocks.

5. **Cross-application comparisons**
   - Analyze performance patterns across multiple HPC workloads.

6. **Evaluation framework**
   - Quantitative comparison of grounded vs ungrounded responses.

## 12. Conclusion

This project demonstrates that Generative AI becomes significantly more useful for HPC tasks when treated as a **grounded reasoning system** rather than a standalone code generator. By integrating retrieval, interpretability, and explicit uncertainty handling, HPC Lab Copilot moves toward a model of AI assistance that is both practical and responsible in high-stakes technical domains.

The system is intentionally extensible, allowing new codes, logs, and workflows to be incorporated without architectural changes. More broadly, it illustrates how RAG-based systems can bridge the gap between powerful language models and real-world engineering constraints.