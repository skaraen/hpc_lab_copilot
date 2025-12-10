from __future__ import annotations
import textwrap
from typing import List, Dict, Any
import streamlit as st
import html

from config import (
    DEFAULT_TEXT_MODEL,
    DEFAULT_TOP_K,
    MIN_SIMILARITY_THRESHOLD,
)
from llm.openai_client import generate_response
from rag.retrieval import retrieve_chunks_for_query


st.set_page_config(
    page_title="HPC Lab Copilot",
    layout="wide",
)

st.title("HPC Lab Copilot")
st.write(
    "A generative AI + RAG assistant for explaining code/logs and answering questions "
    "grounded in your own research notes and documents."
)

with st.sidebar:
    st.header("Settings")
    model = st.text_input("Model", value=DEFAULT_TEXT_MODEL)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    top_k = st.slider("Top-k retrieval", 1, 10, DEFAULT_TOP_K, 1)
    show_debug = st.checkbox("Show debug info", value=False)

tabs = st.tabs(
    [
        "Code / Log Explainer",
        "Research Notebook RAG",
        "Job & Performance Tuner",
    ]
)

with tabs[0]:
    st.subheader("Explain my code or logs")

    st.write(
        "Paste CUDA/C/C++/Python code, SLURM scripts, or error logs. "
        "The assistant will explain and try to diagnose issues using RAG on your local corpus."
    )

    code_input = st.text_area(
        "Code / Logs",
        height=250,
        placeholder="Paste your kernel, SLURM script, or error log here...",
    )

    question = st.text_input(
        "What do you want to know?",
        value="Explain what this code/log is doing and suggest any obvious fixes or improvements.",
    )

    if st.button("Analyze", key="analyze_code_logs"):
        if not code_input.strip():
            st.warning("Please paste some code or logs.")
        else:
            query = question + "\n\nSNIPPET:\n" + textwrap.shorten(
                code_input, width=512, placeholder="..."
            )

            with st.spinner("Retrieving relevant context and generating answer..."):
                retrieved = retrieve_chunks_for_query(query, k=top_k)

                if retrieved:
                    best_score = max(c["score"] for c in retrieved)
                else:
                    best_score = 0.0

                context_chunks = [c for c in retrieved if c["attention_weight"] >= 0.2]

                system_prompt = (
                    "You are an expert tutor in high-performance computing, CUDA, and "
                    "cluster job scheduling. The user has provided some code or logs. "
                    "Explain clearly, avoid hallucinating APIs or flags that don't exist, "
                    "and if context seems insufficient, say so explicitly.\n\n"
                    "Structure your answer in these sections:\n"
                    "1. High-level explanation\n"
                    "2. Potential issues / bugs\n"
                    "3. Performance / readability suggestions\n"
                    "4. Which CHUNKs you used (if any)\n"
                )

                user_prompt = (
                    "Here is the code/log snippet:\n\n"
                    f"```text\n{code_input}\n```\n\n"
                    f"And here is my question:\n{question}\n"
                )

                answer = generate_response(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    context_chunks=context_chunks,
                    model=model,
                    temperature=temperature,
                )

            st.markdown("### Answer")
            st.markdown(answer)

            if retrieved:
                st.markdown("### Retrieved context (RAG)")

                for i, c in enumerate(retrieved):
                    weight = c["attention_weight"]
                    alpha = 0.15 + 0.5 * weight 
                    bg = f"rgba(255, 215, 0, {alpha:.2f})"

                    chunk_text = html.escape(c["text"])

                    st.markdown(
                        f"""
                        <div style="border-radius: 6px; padding: 8px; margin-bottom: 6px; background-color: {bg}">
                        <strong>CHUNK {i+1}</strong><br>
                        <em>source:</em> {c['source_path']}<br>
                        <em>type:</em> {c['type']} | <em>raw score:</em> {c['score']:.3f} | <em>attention:</em> {weight:.2f}
                        <pre style="white-space: pre-wrap;">{chunk_text}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            if best_score < MIN_SIMILARITY_THRESHOLD:
                st.warning(
                    "Low similarity between your query and retrieved documents. "
                    "The answer may be less reliable; double-check with documentation."
                )

            if show_debug:
                st.json(
                    {
                        "num_retrieved": len(retrieved),
                        "best_score": best_score,
                    }
                )

with tabs[1]:
    st.subheader("Ask questions about my research notes & papers")

    st.write(
        "Ask questions about papers, notes, or docs you've ingested into the RAG index "
        "(place files in `data/docs`, `data/code_snippets`, or `data/logs` and rebuild the index)."
    )

    query = st.text_input(
        "Question",
        placeholder="e.g., Compare WFQ and FIFO scheduling for fairness in HPC workloads.",
        key="notebook_query",
    )

    if st.button("Ask", key="ask_notebook"):
        if not query.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Retrieving and answering..."):
                retrieved = retrieve_chunks_for_query(query, k=top_k)
                context_chunks = retrieved

                system_prompt = (
                    "You are a research assistant. Answer the user's question using ONLY "
                    "the provided chunks when possible. When you rely on a chunk, cite its CHUNK "
                    "ID in brackets, like [CHUNK 1]. If information is missing, say so."
                )

                answer = generate_response(
                    system_prompt=system_prompt,
                    user_prompt=query,
                    context_chunks=context_chunks,
                    model=model,
                    temperature=temperature,
                )

        st.markdown("### Answer")
        st.markdown(answer)

        if retrieved:
            st.markdown("### Retrieved context (RAG)")
            for i, c in enumerate(retrieved):
                weight = c["attention_weight"]
                alpha = 0.12 + 0.5 * weight
                bg = f"rgba(173, 216, 230, {alpha:.2f})"

                chunk_text = html.escape(c["text"])

                st.markdown(
                    f"""
                    <div style="border-radius: 6px; padding: 8px; margin-bottom: 6px; background-color: {bg}">
                    <strong>CHUNK {i+1}</strong><br>
                    <em>source:</em> {c['source_path']}<br>
                    <em>type:</em> {c['type']} | <em>raw score:</em> {c['score']:.3f} | <em>attention:</em> {weight:.2f}
                    <pre style="white-space: pre-wrap;">{chunk_text}</pre>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
    
with tabs[2]:
    st.subheader("Job & Performance Tuner")

    st.write(
        "Paste an nvprof output (or similar profiler log) and optionally your SLURM "
        "script. The assistant will summarize performance, identify bottlenecks, and "
        "suggest improvements, using your existing RAG corpus (CUDA files, scripts, "
        "and logs) as context."
    )

    col1, col2 = st.columns(2)

    with col1:
        nvprof_text = st.text_area(
            "nvprof output",
            height=220,
            placeholder="Paste the contents of nvprof_brute_output.txt or nvprof_bh_output.txt here...",
        )

    with col2:
        sbatch_text = st.text_area(
            "SLURM / run script (optional)",
            height=220,
            placeholder="Paste run_nbody.sh or your sbatch script here (optional)...",
        )

    perf_question = st.text_input(
        "What do you want to know?",
        value=(
            "Summarize the performance profile, identify the bottleneck kernels, "
            "and suggest concrete code or configuration changes to improve runtime."
        ),
    )

    if st.button("Analyze performance", key="analyze_perf"):
        if not nvprof_text.strip():
            st.warning("Please paste an nvprof log.")
        else:
            import textwrap as _tw

            query_pieces = [perf_question]
            if nvprof_text.strip():
                query_pieces.append(
                    "NVPROF SUMMARY:\n" + _tw.shorten(nvprof_text, width=800, placeholder="...")
                )
            if sbatch_text.strip():
                query_pieces.append(
                    "SBATCH SCRIPT SNIPPET:\n" + _tw.shorten(sbatch_text, width=400, placeholder="...")
                )

            full_query = "\n\n".join(query_pieces)

            with st.spinner("Retrieving context and generating performance analysis..."):
                retrieved_perf = retrieve_chunks_for_query(full_query, k=top_k)
                context_chunks = retrieved_perf

                system_prompt = (
                    "You are an expert in CUDA and high-performance computing. The user has provided an "
                    "nvprof log and possibly a SLURM or run script. Your job is to:\n"
                    "1. Summarize the overall performance profile.\n"
                    "2. Identify the main bottlenecks (kernels, cudaMemcpy, etc.).\n"
                    "3. Propose specific improvements:\n"
                    "   - kernel-level (e.g., memory access patterns, block/grid sizes),\n"
                    "   - job-level (e.g., sbatch flags, GPU usage),\n"
                    "   - algorithm-level when evident.\n"
                    "4. Where helpful, connect your suggestions to any relevant context CHUNKs from the corpus.\n"
                    "If information from nvprof is insufficient to support a claim, say that explicitly rather "
                    "than guessing.\n"
                    "Cite CHUNK IDs when referencing retrieved docs (e.g., [CHUNK 2])."
                )

            user_prompt = (
                "Here is the profiler output:\n\n"
                f"```text\n{nvprof_text}\n```\n\n"
                "Here is the (optional) SLURM/run script:\n\n"
                f"```bash\n{sbatch_text or '(none provided)'}\n```\n\n"
                f"My question is:\n{perf_question}\n"
            )

            perf_answer = generate_response(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                context_chunks=context_chunks,
                model=model,
                temperature=temperature,
            )

            st.markdown("### Performance analysis")
            st.markdown(perf_answer)

            if retrieved_perf:
                st.markdown("### Retrieved context (RAG)")
                for i, c in enumerate(retrieved_perf):
                    weight = c["attention_weight"]
                    alpha = 0.12 + 0.5 * weight
                    bg = f"rgba(144, 238, 144, {alpha:.2f})"

                    chunk_text = html.escape(c["text"])

                    st.markdown(
                        f"""
                        <div style="border-radius: 6px; padding: 8px; margin-bottom: 6px; background-color: {bg}">
                        <strong>CHUNK {i+1}</strong><br>
                        <em>source:</em> {c['source_path']}<br>
                        <em>type:</em> {c['type']} | <em>raw score:</em> {c['score']:.3f} | <em>attention:</em> {weight:.2f}
                        <pre style="white-space: pre-wrap;">{chunk_text}</pre>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

