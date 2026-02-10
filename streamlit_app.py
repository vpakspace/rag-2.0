"""Streamlit UI for RAG 2.0 pipeline."""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st

from ui.i18n import get_translator

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="RAG 2.0", page_icon="🔍", layout="wide")

# ---------------------------------------------------------------------------
# Language selector (sidebar, persisted in session_state)
# ---------------------------------------------------------------------------
if "lang" not in st.session_state:
    st.session_state.lang = "en"

if "use_gpu" not in st.session_state:
    st.session_state.use_gpu = False

with st.sidebar:
    lang = st.selectbox(
        "Language / Язык",
        options=["en", "ru"],
        index=["en", "ru"].index(st.session_state.lang),
        key="lang_selector",
    )
    st.session_state.lang = lang

    st.divider()
    st.session_state.use_gpu = st.toggle(
        "GPU Acceleration (Docling)",
        value=st.session_state.use_gpu,
        help="Use CUDA GPU for PDF processing via Docling",
    )

t = get_translator(st.session_state.lang)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------
@st.cache_resource
def get_store():
    from storage.vector_store import VectorStore

    store = VectorStore()
    store.init_index()
    return store


@st.cache_resource
def get_openai_client():
    from openai import OpenAI
    from core.config import settings

    return OpenAI(api_key=settings.openai_api_key)


@st.cache_resource
def get_retriever():
    return _build_retriever()


def _build_retriever():
    from retrieval.retriever import Retriever

    return Retriever(get_store(), get_openai_client())


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title(t("app_title"))
st.caption(t("app_subtitle"))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_ingest, tab_search, tab_bench, tab_settings = st.tabs(
    [t("tab_ingest"), t("tab_search"), t("tab_benchmark"), t("tab_settings")]
)

# ===== Tab 1: Ingest =====
with tab_ingest:
    st.header(t("ingest_header"))
    st.markdown(t("ingest_supported"))

    source_mode = st.radio(
        t("ingest_upload"),
        options=["upload", "path"],
        format_func=lambda x: t("ingest_source_upload") if x == "upload" else t("ingest_source_path"),
        horizontal=True,
        label_visibility="collapsed",
    )

    file_path_to_ingest: str | None = None
    cleanup_temp = False

    if source_mode == "upload":
        uploaded = st.file_uploader(
            t("ingest_upload"),
            type=["txt", "pdf", "docx", "pptx", "xlsx", "html"],
        )
    else:
        uploaded = None
        file_path_input = st.text_input(
            t("ingest_path_input"),
            placeholder=t("ingest_path_placeholder"),
        )
        if file_path_input:
            p = Path(file_path_input.strip())
            if p.is_file():
                file_path_to_ingest = str(p)
            else:
                st.warning(t("ingest_path_not_found", path=file_path_input))

    can_ingest = (source_mode == "upload" and uploaded is not None) or (
        source_mode == "path" and file_path_to_ingest is not None
    )

    skip_enrichment = st.checkbox(t("ingest_skip_enrichment"))

    if st.button(t("ingest_button"), disabled=not can_ingest):
        from ingestion.chunker import chunk_text
        from ingestion.enricher import embed_chunks, enrich_chunks
        from ingestion.loader import load_file

        store = get_store()

        # Resolve file path
        if source_mode == "upload" and uploaded is not None:
            suffix = Path(uploaded.name).suffix
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                file_path_to_ingest = tmp.name
            cleanup_temp = True

        if file_path_to_ingest is not None:
            try:
                progress = st.empty()
                status_text = st.empty()

                # Step 1: Load
                progress.progress(0.1, text=t("ingest_loading"))
                text = load_file(file_path_to_ingest, use_gpu=st.session_state.use_gpu)
                status_text.info(t("ingest_chars_loaded", chars=len(text)))

                # Step 2: Chunk
                progress.progress(0.3, text=t("ingest_chunking"))
                chunks = chunk_text(text)
                status_text.info(t("ingest_chunks_created", count=len(chunks)))

                # Step 3: Enrich
                if not skip_enrichment:
                    progress.progress(0.5, text=t("ingest_enriching"))
                    chunks = enrich_chunks(chunks)

                # Step 4: Embed
                progress.progress(0.7, text=t("ingest_embedding"))
                chunks = embed_chunks(chunks)

                # Step 5: Store
                progress.progress(0.9, text=t("ingest_storing"))
                count = store.add_chunks(chunks)

                progress.progress(1.0, text="Done!")
                total = store.count()
                st.success(t("ingest_success", chunks=count, total=total))

            except Exception as e:
                st.error(t("error", msg=str(e)))
            finally:
                if cleanup_temp:
                    Path(file_path_to_ingest).unlink(missing_ok=True)


# ===== Tab 2: Search & Q&A =====
with tab_search:
    st.header(t("search_header"))

    question = st.text_input(
        t("search_input"),
        placeholder=t("search_placeholder"),
    )
    agent_mode = st.toggle(t("search_agent_mode"), value=False)

    if st.button(t("search_button"), disabled=not question):
        with st.spinner(t("search_thinking")):
            try:
                store = get_store()
                client = get_openai_client()
                retriever = get_retriever()

                if agent_mode:
                    from agent.rag_agent import RAGAgent

                    agent = RAGAgent(retriever, store, client)
                    result = agent.run(question)
                else:
                    from generation.reflector import reflect_and_answer

                    result = reflect_and_answer(question, retriever, client)

                # Display answer
                st.subheader(t("search_answer"))
                st.markdown(result.answer)

                # Confidence bar
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.progress(
                        min(result.confidence, 1.0),
                        text=f"{t('search_confidence')}: {result.confidence:.2f}",
                    )
                with col2:
                    st.metric(
                        label="Retries",
                        value=result.retries,
                    )

                # Sources
                if result.sources:
                    with st.expander(
                        t("search_sources", count=len(result.sources[:5]))
                    ):
                        for i, src in enumerate(result.sources[:5], 1):
                            preview = src.chunk.content[:300].replace("\n", " ")
                            st.markdown(
                                f"**{i}.** [{t('search_source_score', score=src.score)}] "
                                f"{preview}..."
                            )
                            st.divider()

            except Exception as e:
                st.error(t("error", msg=str(e)))


# ===== Tab 3: Benchmark =====
with tab_bench:
    st.header(t("bench_header"))

    bench_agent = st.toggle(t("bench_agent_mode"), value=False, key="bench_agent")

    if st.button(t("bench_run")):
        from evaluation.benchmark import judge_answer, load_questions

        store = get_store()
        client = get_openai_client()
        retriever = get_retriever()

        questions = load_questions()
        total = len(questions)

        progress_bar = st.progress(0, text=t("bench_running", current=0, total=total))
        results_table = []
        correct_count = 0
        total_confidence = 0.0

        for idx, q in enumerate(questions):
            progress_bar.progress(
                (idx + 1) / total,
                text=t("bench_running", current=idx + 1, total=total),
            )

            question_text = q["question"]

            try:
                if bench_agent:
                    from agent.rag_agent import RAGAgent

                    agent = RAGAgent(retriever, store, client)
                    qa_result = agent.run(question_text)
                else:
                    from generation.reflector import reflect_and_answer

                    qa_result = reflect_and_answer(question_text, retriever, client)

                judgment = judge_answer(
                    question_text,
                    q["expected_answer"],
                    qa_result.answer,
                    q.get("key_facts", []),
                    client,
                )

                status = "PASS" if judgment["correct"] else "FAIL"
                if judgment["correct"]:
                    correct_count += 1
                total_confidence += qa_result.confidence

                results_table.append(
                    {
                        t("bench_col_q"): q["id"],
                        t("bench_col_status"): status,
                        t("bench_col_confidence"): f"{qa_result.confidence:.2f}",
                        t("bench_col_retries"): qa_result.retries,
                        t("bench_col_question"): question_text[:60],
                    }
                )
            except Exception as e:
                results_table.append(
                    {
                        t("bench_col_q"): q["id"],
                        t("bench_col_status"): "ERROR",
                        t("bench_col_confidence"): "—",
                        t("bench_col_retries"): "—",
                        t("bench_col_question"): question_text[:60],
                    }
                )

        # Summary
        accuracy = correct_count / total if total else 0.0
        avg_conf = total_confidence / total if total else 0.0

        st.subheader(t("bench_results"))

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{correct_count}/{total} ({accuracy:.0%})")
        col2.metric("Avg Confidence", f"{avg_conf:.2f}")
        col3.metric("Improvement vs Baseline", f"{accuracy - 0.4:+.0%}")

        st.dataframe(results_table, use_container_width=True)

        st.info(t("bench_baseline"))


# ===== Tab 4: Settings & Stats =====
with tab_settings:
    st.header(t("settings_header"))

    # Current settings
    st.subheader(t("settings_current"))

    from core.config import settings

    settings_dict = {
        "LLM Model": settings.llm_model,
        "Embedding Model": settings.embedding_model,
        "Embedding Dimensions": settings.embedding_dimensions,
        "Chunk Size": settings.chunk_size,
        "Chunk Overlap": settings.chunk_overlap,
        "Top-K Retrieval": settings.top_k_retrieval,
        "Top-K Rerank": settings.top_k_rerank,
        "Rerank Method": settings.rerank_method,
        "Max Retries": settings.max_retries,
        "Relevance Threshold": settings.relevance_threshold,
        "Neo4j URI": settings.neo4j_uri,
    }
    st.json(settings_dict)

    # Store stats
    st.subheader(t("settings_store_stats"))
    try:
        store = get_store()
        chunk_count = store.count()
        st.metric(t("settings_total_chunks", count=chunk_count), value=chunk_count)
    except Exception as e:
        st.error(t("error", msg=str(e)))

    # Clear database
    st.subheader(t("settings_clear_db"))
    confirm = st.text_input(t("settings_clear_confirm"), key="clear_confirm")
    if st.button(t("settings_clear_button"), disabled=confirm != "DELETE"):
        try:
            store = get_store()
            deleted = store.delete_all()
            st.success(t("settings_cleared", count=deleted))
        except Exception as e:
            st.error(t("error", msg=str(e)))

    # Re-initialize
    st.subheader(t("settings_reinit"))
    if st.button(t("settings_reinit_button")):
        st.cache_resource.clear()
        st.success(t("settings_reinit_done"))
