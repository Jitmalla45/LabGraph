import base64
import html
import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from webapp.utils.graph_view import graph_html, graph_stats
from webapp.utils.pipeline_bridge import answer_session_query, run_pipeline_from_uploads


WEB_OUTPUT_DIR = Path("webapp") / "outputs"
PIPELINE_STEPS = [
    "Uploading Files",
    "Loading Models",
    "Generating Scene Graph",
    "Updating Scene Graph",
    "Selecting Landmarks",
    "Evaluating Dataset",
    "Preparing QA System",
]

st.set_page_config(
    page_title="LabGraph",
    page_icon="LG",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
:root {
  --lg-bg: #f8fafc;
  --lg-card: #ffffff;
  --lg-line: #e2e8f0;
  --lg-text: #0f172a;
  --lg-muted: #475569;
  --lg-accent: #2563eb;
  --lg-accent-2: #06b6d4;
  --lg-success: #22c55e;
  --lg-warning: #f59e0b;
}
.stApp {
  background:
    linear-gradient(180deg, rgba(248, 250, 252, 0.98) 0%, rgba(241, 245, 249, 0.94) 100%),
    #f8fafc;
  color: var(--lg-text);
}
.block-container {
  max-width: 1500px;
  padding-top: 1.5rem;
  padding-bottom: 2.6rem;
}
[data-testid="stSidebar"], header[data-testid="stHeader"] {
  background: transparent;
}
h1, h2, h3, h4, p, label, span, div {
  letter-spacing: 0;
}
h1, h2, h3, h4, .stMarkdown, .stText, label {
  color: var(--lg-text);
}
.stMarkdown p, .stCaptionContainer, .stCodeBlock, .stJson {
  color: var(--lg-muted);
}
.stButton button, [data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] {
  min-height: 3rem;
  border-radius: 14px !important;
  font-size: 1.05rem !important;
  font-weight: 800 !important;
  box-shadow: 0 12px 26px rgba(37, 99, 235, 0.12);
}
.stTextInput input, .stSelectbox div[data-baseweb="select"] > div {
  min-height: 3.2rem;
  border-radius: 14px;
  font-size: 1.08rem;
}
.stFileUploader {
  font-size: 1.02rem;
}
.hero {
  position: relative;
  min-height: 340px;
  border: 1px solid rgba(37, 99, 235, 0.18);
  border-radius: 26px;
  padding: 58px 48px 54px;
  overflow: hidden;
  background:
    linear-gradient(135deg, rgba(255, 255, 255, 0.98), rgba(239, 246, 255, 0.94)),
    linear-gradient(90deg, rgba(37, 99, 235, 0.08), rgba(6, 182, 212, 0.08));
  box-shadow: 0 24px 70px rgba(15, 23, 42, 0.10);
  animation: fadeIn 520ms ease-out;
  text-align: center;
}
.hero::after {
  content: "";
  position: absolute;
  inset: -46% -14% auto auto;
  width: 560px;
  height: 560px;
  border-radius: 999px;
  border: 1px solid rgba(37, 99, 235, 0.16);
  background: conic-gradient(from 180deg, rgba(37, 99, 235, 0.13), rgba(6, 182, 212, 0.10), transparent);
  animation: orbit 11s linear infinite;
}
.hero-kicker {
  color: var(--lg-accent);
  font-size: 1rem;
  font-weight: 900;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
.hero h1 {
  margin: 14px 0 10px;
  font-size: clamp(4.1rem, 8vw, 7.8rem);
  line-height: 0.9;
  color: #0f172a;
}
.hero h2 {
  margin: 0 auto;
  max-width: 1000px;
  font-size: clamp(1.7rem, 3vw, 3rem);
  color: #1d4ed8;
  font-weight: 850;
}
.hero p {
  max-width: 900px;
  margin: 22px auto 0;
  color: var(--lg-muted);
  font-size: 1.24rem;
  line-height: 1.65;
}
.section-title {
  margin: 46px 0 16px;
  padding-top: 8px;
  border-top: 1px solid rgba(226, 232, 240, 0.8);
}
.section-title span {
  color: var(--lg-accent);
  font-size: 0.95rem;
  text-transform: uppercase;
  font-weight: 900;
}
.section-title h3 {
  margin: 6px 0 0;
  color: #0f172a;
  font-size: 2.15rem;
  font-weight: 900;
}
[data-testid="stVerticalBlockBorderWrapper"] {
  border: 1px solid var(--lg-line);
  border-radius: 22px;
  background: #ffffff;
  box-shadow: 0 18px 48px rgba(15, 23, 42, 0.08);
}
.metric-row {
  display: grid;
  grid-template-columns: repeat(4, minmax(0, 1fr));
  gap: 16px;
  margin: 8px 0 20px;
}
.metric-row:has(.metric:nth-child(3):last-child) {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}
.metric {
  border: 1px solid var(--lg-line);
  border-radius: 18px;
  padding: 19px 20px;
  background: linear-gradient(180deg, #ffffff, #f8fbff);
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.07);
  transition: transform 160ms ease, border-color 160ms ease;
}
.metric:hover {
  transform: translateY(-3px);
  border-color: rgba(37, 99, 235, 0.42);
}
.metric span {
  display: block;
  color: var(--lg-muted);
  font-size: 1.02rem;
  font-weight: 750;
}
.metric strong {
  display: block;
  color: #0f172a;
  font-size: 2.45rem;
  line-height: 1.1;
  margin-top: 4px;
}
.stage-grid {
  display: grid;
  grid-template-columns: repeat(7, minmax(0, 1fr));
  gap: 12px;
}
.stage-card {
  min-height: 108px;
  border: 1px solid #dbeafe;
  border-radius: 18px;
  background: #ffffff;
  padding: 16px;
  overflow: hidden;
  position: relative;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.07);
}
.stage-card::before {
  content: "";
  position: absolute;
  inset: 0;
  background: linear-gradient(90deg, transparent, rgba(37, 99, 235, 0.10), transparent);
  transform: translateX(-100%);
  animation: shimmer 1.7s infinite;
}
.stage-card.done {
  border-color: rgba(34, 197, 94, 0.42);
  background: #f0fdf4;
}
.stage-card.pending::before {
  display: none;
}
.stage-card strong {
  position: relative;
  display: block;
  color: #0f172a;
  font-size: 1.06rem;
  line-height: 1.25;
}
.stage-card span {
  position: relative;
  color: var(--lg-muted);
  font-size: 0.92rem;
  font-weight: 650;
}
.answer {
  border: 1px solid rgba(37, 99, 235, 0.16);
  border-left: 5px solid var(--lg-accent);
  padding: 20px 22px;
  background: #f8fbff;
  border-radius: 18px;
  color: #0f172a;
  font-size: 1.28rem;
  line-height: 1.7;
  box-shadow: 0 12px 28px rgba(37, 99, 235, 0.08);
}
.query {
  color: #475569;
  margin: 0 0 14px;
  font-size: 1.02rem;
}
.strategy {
  color: #2563eb;
  font-weight: 900;
  margin-bottom: 4px;
}
.footer {
  margin-top: 58px;
  padding: 34px 0 18px;
  border-top: 1px solid #e2e8f0;
  color: #475569;
  text-align: center;
  font-size: 1.08rem;
  line-height: 1.7;
}
.footer strong {
  color: #0f172a;
  font-size: 1.35rem;
}
@keyframes shimmer {
  to { transform: translateX(100%); }
}
@keyframes fadeIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
@keyframes orbit {
  to { transform: rotate(360deg); }
}
@media (max-width: 1020px) {
  .stage-grid, .metric-row {
    grid-template-columns: repeat(2, minmax(0, 1fr));
  }
  .hero {
    padding: 28px 24px;
  }
}
</style>
""",
    unsafe_allow_html=True,
)


def section_title(kicker, title):
    st.markdown(
        f"<div class='section-title'><span>{html.escape(kicker)}</span><h3>{html.escape(title)}</h3></div>",
        unsafe_allow_html=True,
    )


def format_number(value):
    if isinstance(value, float):
        return f"{value:.3f}"
    return value


def render_stats(stats):
    st.markdown(
        "<div class='metric-row'>"
        + "".join(
            f"<div class='metric'><span>{html.escape(str(label))}</span><strong>{html.escape(str(format_number(value)))}</strong></div>"
            for label, value in stats.items()
        )
        + "</div>",
        unsafe_allow_html=True,
    )


def aggregate_metric_block(metric_map):
    totals = {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    total_count = 0
    for values in metric_map.values():
        count = values.get("count", 0) or 0
        total_count += count
        for key in totals:
            totals[key] += values.get(key, 0) * count
    if not total_count:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {key: value / total_count for key, value in totals.items()}


def evaluation_metrics(summary):
    if summary.get("updated") or summary.get("original"):
        return summary.get("updated") or summary.get("original") or {}
    if summary.get("stratum_metrics"):
        return aggregate_metric_block(summary["stratum_metrics"])
    updated_strata = summary.get("original_vs_updated_metrics", {}).get("updated")
    if updated_strata:
        return aggregate_metric_block(updated_strata)
    return {}


def render_stage_cards(active_index=None, complete=False):
    cards = []
    for index, label in enumerate(PIPELINE_STEPS):
        if complete or (active_index is not None and index < active_index):
            state = "done"
            status = "Complete"
        elif active_index == index:
            state = "active"
            status = "Running"
        else:
            state = "pending"
            status = "Pending"
        cards.append(
            f"<div class='stage-card {state}'><strong>{html.escape(label)}</strong><span>{status}</span></div>"
        )
    st.markdown("<div class='stage-grid'>" + "".join(cards) + "</div>", unsafe_allow_html=True)


def image_modal_html(path, caption=None, height=360):
    path = Path(path)
    suffix = path.suffix.lower()
    mime = "image/jpeg" if suffix in {".jpg", ".jpeg"} else "image/png"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    safe_caption = html.escape(caption or path.name)
    return f"""
<button class="lg-image-button" type="button" aria-label="Open enlarged image">
  <img src="data:{mime};base64,{encoded}" alt="{safe_caption}" />
</button>
<dialog class="lg-image-dialog">
  <form method="dialog"><button class="lg-close" title="Close">Close</button></form>
  <img src="data:{mime};base64,{encoded}" alt="{safe_caption}" />
  <p>{safe_caption}</p>
</dialog>
<style>
.lg-image-button {{
  width: 100%;
  height: {height}px;
  border: 1px solid #dbeafe;
  border-radius: 18px;
  background: #ffffff;
  padding: 8px;
  cursor: zoom-in;
  box-shadow: 0 14px 32px rgba(15, 23, 42, 0.08);
}}
.lg-image-button img {{
  width: 100%;
  height: 100%;
  object-fit: contain;
}}
.lg-image-dialog {{
  border: 1px solid #dbeafe;
  border-radius: 18px;
  padding: 18px;
  background: #ffffff;
  max-width: 94vw;
  max-height: 94vh;
  box-shadow: 0 26px 80px rgba(15, 23, 42, 0.26);
}}
.lg-image-dialog::backdrop {{
  background: rgba(15, 23, 42, 0.52);
}}
.lg-image-dialog img {{
  max-width: 88vw;
  max-height: 78vh;
  object-fit: contain;
}}
.lg-image-dialog p {{
  margin: 8px 0 0;
  color: #475569;
  font: 15px system-ui, sans-serif;
}}
.lg-close {{
  float: right;
  border: 1px solid #cbd5e1;
  border-radius: 999px;
  background: #ffffff;
  color: #0f172a;
  padding: 7px 12px;
  margin-bottom: 8px;
}}
</style>
<script>
(function() {{
  const script = document.currentScript;
  const dialog = script.previousElementSibling.previousElementSibling;
  const button = dialog.previousElementSibling;
  button.addEventListener("click", () => dialog.showModal());
}})();
</script>
"""


def render_clickable_image(path, caption=None, height=360):
    components.html(image_modal_html(path, caption=caption, height=height), height=height + 16, scrolling=False)


def render_graph_panel(title, graph, image_path, landmarks, changes):
    st.markdown(f"<h4>{html.escape(title)}</h4>", unsafe_allow_html=True)
    render_stats(graph_stats(graph, landmarks=landmarks, changes=changes))
    components.html(graph_html(graph, height=610), height=640, scrolling=False)
    if image_path.exists():
        with st.expander("Expand rendered PNG", expanded=False):
            render_clickable_image(image_path, caption=f"{title} PNG", height=430)


def read_json_if_exists(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path) as handle:
        return json.load(handle)


def render_dataset_evaluation(output_root):
    eval_dir = Path(output_root) / "dataset_evaluation"
    if not eval_dir.exists():
        st.info("Dataset evaluation outputs are not available for this run yet.")
        return
    summary = read_json_if_exists(eval_dir / "evaluation_summary.json") or {}
    metrics = evaluation_metrics(summary)
    if metrics:
        render_stats(
            {
                "Precision": metrics.get("precision", 0),
                "Recall": metrics.get("recall", 0),
                "F1": metrics.get("f1", 0),
            }
        )

    plot_paths = sorted(eval_dir.glob("*.png"))
    if plot_paths:
        plot_cols = st.columns(2)
        for index, plot_path in enumerate(plot_paths):
            with plot_cols[index % 2]:
                st.markdown(f"<h4>{html.escape(plot_path.stem.replace('_', ' ').title())}</h4>", unsafe_allow_html=True)
                render_clickable_image(plot_path, caption=plot_path.name, height=300)

    for table_name in ["qa_results.csv", "query_results.csv"]:
        table_path = eval_dir / table_name
        if table_path.exists():
            with st.expander(f"Evaluation Table · {table_name}", expanded=True):
                st.dataframe(pd.read_csv(table_path), use_container_width=True, height=340)
            break


def reset_session():
    for key in ["pipeline_results", "query_history", "query_input"]:
        st.session_state.pop(key, None)


def clear_query():
    st.session_state.query_input = ""


if "query_history" not in st.session_state:
    st.session_state.query_history = []

st.markdown(
    """
<section class="hero">
  <div class="hero-kicker">Research Demonstration Platform</div>
  <h1>LabGraph</h1>
  <h2>Dynamic Scene Graph Reasoning,<br/>Visual Grounding and Spatial Intelligence</h2>
  <p>
    Interactive framework for scene graph generation, incremental scene updates,
    landmark-centric reasoning, and visual grounding.
  </p>
</section>
""",
    unsafe_allow_html=True,
)

top_cols = st.columns([0.75, 0.25])
top_cols[0].caption(f"Web outputs: `{WEB_OUTPUT_DIR}/<IMAGE_ID>/`")
if top_cols[1].button("Reset Session", use_container_width=True):
    reset_session()
    st.rerun()

section_title("Input", "Upload Section")
with st.container(border=True):
    upload_cols = st.columns(3)
    original_upload = upload_cols[0].file_uploader("Original Image", type=["png", "jpg", "jpeg"])
    annotation_upload = upload_cols[1].file_uploader("Annotation JSON", type=["json"])
    modified_upload = upload_cols[2].file_uploader("Modified Image", type=["png", "jpg", "jpeg"])

    preview_cols = st.columns(3)
    if original_upload:
        preview_cols[0].image(Image.open(original_upload), caption="Original image", use_container_width=True)
    if annotation_upload:
        try:
            annotation_upload.seek(0)
            preview_cols[1].json(json.load(annotation_upload), expanded=False)
        except json.JSONDecodeError:
            preview_cols[1].error("Annotation JSON could not be parsed.")
    if modified_upload:
        preview_cols[2].image(Image.open(modified_upload), caption="Modified image", use_container_width=True)

    ready = all([original_upload, annotation_upload, modified_upload])
    if st.button("Run Pipeline", type="primary", disabled=not ready):
        progress = st.progress(0, text="Uploading Files")
        status = st.status("Running LabGraph pipeline...", expanded=True)
        render_stage_cards(active_index=0)
        try:
            status.write("Uploading Files")
            progress.progress(8, text="Uploading Files")
            status.write("Loading models and executing the existing backend pipeline.")
            progress.progress(18, text="Backend pipeline running")
            st.session_state.pipeline_results = run_pipeline_from_uploads(
                original_upload,
                annotation_upload,
                modified_upload,
            )
            st.session_state.query_history = []
            progress.progress(100, text="Preparing QA System")
            render_stage_cards(complete=True)
            status.update(label="Pipeline complete", state="complete", expanded=False)
        except Exception as exc:
            status.update(label="Pipeline failed", state="error", expanded=True)
            st.exception(exc)

section_title("Pipeline", "Progress Dashboard")
with st.container(border=True):
    render_stage_cards(complete=bool(st.session_state.get("pipeline_results")))

results = st.session_state.get("pipeline_results")
if results:
    output_root = Path(results["output_root"])
    st.success(f"Pipeline outputs saved in {output_root}")

    section_title("Graphs", "Scene Graph Visualization")
    with st.container(border=True):
        graph_cols = st.columns(2)
        with graph_cols[0]:
            with st.container(border=True):
                render_graph_panel(
                    "Original Scene Graph",
                    results["scene_graph"],
                    output_root / "scene_graphs" / "original_scene_graph.png",
                    results["landmarks"],
                    results["changes"],
                )
        with graph_cols[1]:
            with st.container(border=True):
                render_graph_panel(
                    "Updated Scene Graph",
                    results["updated_scene_graph"],
                    output_root / "scene_graphs" / "updated_scene_graph.png",
                    results["landmarks"],
                    results["changes"],
                )

    section_title("Evaluation", "Dataset Evaluation")
    with st.container(border=True):
        render_dataset_evaluation(output_root)

    section_title("Reasoning", "Question Answering Workspace")
    with st.container(border=True):
        qa_cols = st.columns([0.30, 0.70])
        strategy = qa_cols[0].selectbox(
            "Reasoning Strategy",
            ["K-Hop Reasoning", "Visual Grounding", "K-Landmark Reasoning"],
        )
        query = qa_cols[1].text_input(
            "Ask a question about the scene...",
            placeholder="Ask a question about the scene...",
            key="query_input",
        )
        action_cols = st.columns([0.20, 0.18, 0.62])
        generate = action_cols[0].button("Generate Answer", type="primary", disabled=not query.strip())
        action_cols[1].button("Clear Query", on_click=clear_query)

        if generate:
            with st.spinner("Reasoning over cached scene graph data..."):
                response = answer_session_query(results, strategy, query.strip())
            st.session_state.query_history.append(response)

        for item in reversed(st.session_state.get("query_history", [])):
            result = item["result"]
            with st.container(border=True):
                st.markdown(f"<div class='strategy'>{html.escape(result['strategy'])}</div>", unsafe_allow_html=True)
                st.markdown(f"<p class='query'>{html.escape(result['query'])}</p>", unsafe_allow_html=True)
                st.markdown("##### Generated Answer")
                st.markdown(f"<div class='answer'>{html.escape(str(item['answer']))}</div>", unsafe_allow_html=True)
                if result.get("reasoning_path"):
                    st.markdown("##### Reasoning Path")
                    st.json(result["reasoning_path"], expanded=False)
                elif result.get("target") or result.get("landmarks"):
                    st.markdown("##### Reasoning Path")
                    st.json(
                        {
                            key: result.get(key)
                            for key in ["target", "reference", "relation", "landmarks"]
                            if result.get(key) is not None
                        },
                        expanded=False,
                    )
                if item.get("visualization"):
                    st.markdown("##### Visual Grounding")
                    render_clickable_image(item["visualization"], caption="Visual grounding image", height=420)
                st.caption(f"Saved to {item['output_dir']}")
else:
    section_title("Graphs", "Scene Graph Visualization")
    with st.container(border=True):
        st.info("Run the pipeline to activate side-by-side scene graph visualization.")
    section_title("Evaluation", "Dataset Evaluation")
    with st.container(border=True):
        st.info("Evaluation cards and plots will appear after a run.")
    section_title("Reasoning", "Question Answering Workspace")
    with st.container(border=True):
        st.info("QA uses cached graph data after the first pipeline run; it will not rerun graph generation.")

st.markdown(
    """
<div class="footer">
  <div>Developed by</div>
  <strong>Jit Malla</strong><br/>
  Int. MS-PhD Student<br/>
  School of Mathematical &amp; Computational Sciences (SMCS)<br/>
  Indian Association for the Cultivation of Science (IACS)
</div>
""",
    unsafe_allow_html=True,
)
