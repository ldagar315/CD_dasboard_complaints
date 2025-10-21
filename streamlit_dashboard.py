import streamlit as st
import pandas as pd
import dspy
from typing import List, Optional
import os
import chromadb
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import gdown
import zipfile
import datetime
import plotly.express as px
import numpy as np
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
import product_category
import requests
import io
from pathlib import Path
import json

# Google Drive libs
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.errors import HttpError



try:
    from streamlit_theme import st_theme
except ImportError:
    st_theme = None

DATA_URL = "https://docs.google.com/spreadsheets/d/1MSYdK-Z4qjgudUI6Ky3t3U-Qc2Dxx95D/export?format=csv"
FOLDER_ID = "1uxnGomO1D2oJShW67c43GeobVbE1TLKZ"

GOOGLE_AUTH_METHOD = "service_account"

SERVICE_ACCOUNT_FILE = st.secrets.get("google_service_account")

THEME_COLORS = {
    "light": {
        "primary_text": "#1F2A44",
        "secondary_text": "#5C6C83",
        "accent_color": "#FF7633",
        "surface": "#F6F8FC",
        "card_surface": "#FFFFFF",
        "card_border": "#E7ECF3",
        "card_shadow": "0px 4px 12px rgba(31, 42, 68, 0.06)",
        "divider": "rgba(31, 42, 68, 0.1)",
    },
    "dark": {
        "primary_text": "#F3F4F6",
        "secondary_text": "#9CA3AF",
        "accent_color": "#FF8B4D",
        "surface": "#0F172A",
        "card_surface": "#1E293B",
        "card_border": "#26334C",
        "card_shadow": "0px 12px 24px rgba(3, 7, 18, 0.55)",
        "divider": "rgba(148, 163, 184, 0.25)",
    },
}

def build_drive_service():
    method = GOOGLE_AUTH_METHOD
    scopes = ["https://www.googleapis.com/auth/drive"]
    if method == "service_account":
        sa_file = SERVICE_ACCOUNT_FILE
        creds = service_account.Credentials.from_service_account_file(sa_file, scopes=scopes)
    else:
        raise ValueError("Unknown GOOGLE_AUTH_METHOD")
    return build('drive', 'v3', credentials=creds)


def list_files_in_folder(folder_id: str):
    q = f"'{folder_id}' in parents and trashed = false and mimeType != 'application/vnd.google-apps.folder'"
    files, token = [], None
    while True:
        resp = drive_service.files().list(
            q=q,
            fields="nextPageToken, files(id,name,createdTime,modifiedTime,mimeType)",
            pageSize=1000,
            includeItemsFromAllDrives=True,
            supportsAllDrives=True,
            pageToken=token
        ).execute()
        files.extend(resp.get("files", []))
        token = resp.get("nextPageToken")
        if not token: break
    return files


def download_drive_file(file_id: str, dest_path: Path):
    request = drive_service.files().get_media(fileId=file_id)
    fh = io.FileIO(str(dest_path), mode='wb')
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        status, done = downloader.next_chunk()
        if status:
            print(f"Download {file_id}: {int(status.progress() * 100)}%")
    print(f"Downloaded file to {dest_path}")
    return dest_path




def detect_base_theme() -> str:
    """Return 'dark' or 'light' depending on the current browser theme."""
    if not st_theme:
        return "light"
    try:
        theme_info = st_theme()
    except Exception:
        return "light"
    base = theme_info.get("base") if isinstance(theme_info, dict) else None
    return "dark" if base and base.lower().startswith("dark") else "light"


def build_page_style(palette: dict, base: str) -> str:
    """Generate CSS using the palette detected from the browser theme."""
    return f"""
<style>
    :root {{
        color-scheme: {base};
        --primary-text: {palette["primary_text"]};
        --secondary-text: {palette["secondary_text"]};
        --accent-color: {palette["accent_color"]};
        --surface-color: {palette["surface"]};
        --card-surface: {palette["card_surface"]};
        --card-border: {palette["card_border"]};
        --card-shadow: {palette["card_shadow"]};
        --divider-color: {palette["divider"]};
    }}
    body {{
        background-color: var(--surface-color);
    }}
    .main .block-container {{
        padding-top: 2.5rem;
        padding-bottom: 3rem;
        padding-left: 3rem;
        padding-right: 3rem;
        background-color: var(--surface-color);
        transition: background-color 0.3s ease;
    }}
    .dashboard-title {{
        font-size: 2.4rem;
        font-weight: 700;
        color: var(--primary-text);
        margin-bottom: 0.5rem;
    }}
    .dashboard-subtitle {{
        color: var(--secondary-text);
        font-size: 1rem;
        margin-bottom: 2rem;
    }}
    .section-divider {{
        margin: 2.25rem 0 1.75rem 0;
        border: none;
        border-top: 1px solid var(--divider-color);
    }}
    .section-heading {{
        font-size: 1.3rem;
        font-weight: 600;
        color: var(--primary-text);
        margin-bottom: 0.35rem;
    }}
    .section-caption {{
        font-size: 0.9rem;
        color: var(--secondary-text);
        margin-bottom: 1rem;
    }}
    .kpi-card {{
        background-color: var(--card-surface);
        padding: 1rem 1.2rem;
        border-radius: 0.9rem;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
        min-height: 120px;
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }}
    .kpi-label {{
        display: block;
        font-size: 0.75rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--secondary-text);
    }}
    .kpi-value {{
        display: block;
        margin-top: 0.5rem;
        font-size: 1.9rem;
        font-weight: 700;
        color: var(--primary-text);
    }}
    .filter-grid {{
        background-color: var(--card-surface);
        padding: 1.5rem 1.75rem;
        border-radius: 1rem;
        border: 1px solid var(--card-border);
        box-shadow: var(--card-shadow);
    }}
    .stMultiSelect label, .stDateInput label {{
        font-weight: 600 !important;
        color: var(--primary-text) !important;
    }}
    .stMultiSelect div[data-baseweb="select"],
    .stDateInput div[data-baseweb="input"] {{
        border-radius: 0.6rem;
        background-color: var(--card-surface);
        border: 1px solid var(--card-border);
        color: var(--primary-text);
        transition: background-color 0.3s ease, border-color 0.3s ease;
    }}
    .stMultiSelect div[data-baseweb="select"] span,
    .stDateInput div[data-baseweb="input"] span {{
        color: var(--primary-text);
    }}
    .stDataFrame {{
        border-radius: 0.9rem;
        overflow: hidden;
        border: 1px solid var(--card-border);
    }}
    .stButton button {{
        background: var(--accent-color);
        color: #FFFFFF;
        border: none;
        border-radius: 0.6rem;
        padding: 0.55rem 1.2rem;
        font-weight: 600;
        transition: filter 0.2s ease, transform 0.2s ease;
    }}
    .stButton button:hover {{
        filter: brightness(1.05);
        transform: translateY(-1px);
    }}
</style>
"""

COLUMN_NAMES = [
    "complaint_number",
    "city",
    "region",
    "created_date",
    "refund_count_in_15_days",
    "product",
    "concern_type",
    "level_1_classification",
    "level_2_classification",
    "expanded_description",
    "customer_issue",
    "root_cause",
    "resolution_provided_summary",
    "product_category",
]

# check if vector_db folder exists, if not create it
if not os.path.exists("VectorDB"):
    os.makedirs("VectorDB")
# Download Vector DB files
today_date = datetime.date.today()

    
if 'summary_output' not in st.session_state:
    st.session_state.summary_output = ""
if 'tickets' not in st.session_state:
    st.session_state.tickets = []
if 'collection_name' not in st.session_state:
    st.session_state.collection_name = ""
if 'filtered_df' not in st.session_state:
    st.session_state.filtered_df = pd.DataFrame()
if 'df' not in st.session_state:
    st.session_state.df = pd.DataFrame()
if 'vector_db_initialized' not in st.session_state:
    st.session_state.vector_db_initialized = True
cerebras_key = st.secrets.get("CEREBRAS_API_KEY")
if not cerebras_key:
    st.session_state.summary_output = "API key not configured. Check Streamlit Cloud secrets."
    
# Define DSPy signature for summarising tickets
class SummariseTicket(dspy.Signature):
    action: List[str] = dspy.InputField(desc="The list of tickets to be summarised")
    ticket_summary: str = dspy.OutputField(
        desc="""A short concise analytical summary of the tickets provided. 
                Answer in Points, structured format
                Answer in Markdown format"""
    )
    key_insights: str = dspy.OutputField(
        desc="""Key insights from the tickets provided. 
                Answer in Points, structured format
                Answer in Markdown format"""
    )
summarise = dspy.ChainOfThought(SummariseTicket)

class FinalSummary(dspy.Signature):
    segregated_summary: str = dspy.InputField(desc="The combined summaries from multiple models")
    integrated_summary: str = dspy.OutputField(
        desc="""A concise summary of the key insights from the combined summaries provided. 
                Answer in Points, structured format
                Answer in Markdown format"""
    )
final_summarise = dspy.ChainOfThought(FinalSummary)


MODELS = ["openai/gpt-oss-120b", "openai/qwen-3-235b-a22b-instruct-2507", "openai/llama-4-maverick-17b-128e-instruct", "openai/qwen-3-32b", "openai/llama-4-scout-17b-16e-instruct"]

def _chunk_equally(xs, n):
    k = math.ceil(len(xs) / n) if xs else 0
    return [xs[i:i+k] for i in range(0, len(xs), k)] if k else [[] for _ in range(n)]

def _run_model_summary(model_name, tickets_chunk):
    if not tickets_chunk:
        return model_name, ""
    with dspy.context(
        lm=dspy.LM(
            model=model_name,
            api_key=os.getenv("CEREBRAS_API_KEY"),
            api_base="https://api.cerebras.ai/v1",
        ),
        cache=True,
    ):
        s = summarise(action=tickets_chunk)
        text = (
            f"— Summary\n{s.ticket_summary}\n\n"
            f"— Key Insights\n{s.key_insights}\n"
        )
        return model_name, text
    
def final_summary(combined_text):
    with dspy.context(
        lm=dspy.LM(
            model="openai/llama-4-maverick-17b-128e-instruct",
            api_key=os.getenv("CEREBRAS_API_KEY"),
            api_base="https://api.cerebras.ai/v1",
        ),
    ):
        s = final_summarise(segregated_summary=combined_text)
        return s.integrated_summary
def summarise_tickets_second():
    DB_path = "VectorDB"
    complaint_number = st.session_state.filtered_df['complaint_number'].dropna().tolist()
    COLLECTION_NAME = st.session_state.collection_name
    chroma_client = chromadb.PersistentClient(path = DB_path)
    collection_client = chroma_client.get_or_create_collection(name = COLLECTION_NAME)
    # print(collection_client.count())

    try:
        embeddings = collection_client.get(where = {"COMPLAINT_NUMBER": {'$in' : complaint_number}}, include = ["embeddings", "metadatas", "documents"])
        num_components = int(min(len(embeddings['ids']), 100))
        pca = PCA(n_components = num_components)
        embeddings_array = np.array(embeddings['embeddings'])
        embeddings_pca = pca.fit_transform(embeddings_array)
        extreme_indices = set()
        for i in range(num_components):
            component_scores = embeddings_pca[:, i]
            min_idx = np.argmin(component_scores)
            max_idx = np.argmax(component_scores)
            extreme_indices.add(min_idx)
            extreme_indices.add(max_idx)
            #print(f"PC-{i+1}: Min Index={min_idx}, Max Index={max_idx}")

        final_indices = sorted(list(extreme_indices))
        all_ids = embeddings['ids']
        extreme_ids = [all_ids[i] for i in final_indices]
        extreme_documents = collection_client.get(ids = extreme_ids, include = ["documents"])
        # print(f"Extreme Documents : {extreme_documents['documents']}")
        st.session_state.tickets =  extreme_documents['documents']
    except Exception as e:
        print(f"Error during PCA summarization: {str(e)}")
def summarise_ticket():
    #if len(st.session_state.tickets) > 100:
        #summarise_tickets_second()
    tickets = st.session_state.tickets[:100]
    if not tickets:
        st.session_state.summary_output = "No tickets selected for summary."
        return

    # Divide tickets into 5 chunks (as equal as possible)
    chunks = _chunk_equally(tickets, 5)
    # Ensure exactly five buckets for the five models
    if len(chunks) < 5:
        chunks += [[] for _ in range(5 - len(chunks))]
    elif len(chunks) > 5:
        chunks = chunks[:5]

    results = {}
    try:
        with ThreadPoolExecutor(max_workers=5) as ex:
            futures = {
                ex.submit(_run_model_summary, model, chunk): model
                for model, chunk in zip(MODELS, chunks)
            }
            for fut in as_completed(futures):
                model = futures[fut]
                try:
                    m, text = fut.result()
                    results[m] = text
                except Exception as e:
                    results[model] = f"### {model} — Error\n{str(e)}\n"

        # Stable order by MODELS
        combined = "\n\n".join(results.get(m, "") for m in MODELS)

        # One final aggregation call
        final_insights = final_summary(combined)

        st.session_state.summary_output = final_insights
    except Exception as e:
        st.session_state.summary_output = f"Error generating summary: {str(e)}"



@st.cache_data(show_spinner=False)
def load_ticket_dataframe(url: str) -> pd.DataFrame:
    df = pd.read_csv(url, low_memory=False)
    df = df[df["CONCERN AREA NAME"] != "Stop Customer"]
    df = df[df["CONCERN TYPE NAME"] != "Internal"]
    categories = product_category.categories
    product_to_category = {product: cat for cat, products in categories.items() for product in products}
    df["product_category"] = df["product"].map(product_to_category).fillna("")
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").lower())
    df.columns = [c.strip() for c in df.columns]
    df = df[COLUMN_NAMES]
    df["created_date"] = pd.to_datetime(df["created_date"], errors="coerce").dt.date
    return df


def apply_selected_filters(df: pd.DataFrame, selections: dict, exclude: Optional[str] = None) -> pd.DataFrame:
    filtered = df
    date_range = selections.get("created_date")
    if exclude != "created_date" and isinstance(date_range, (list, tuple)) and len(date_range) == 2:
        start_date, end_date = date_range
        if start_date and end_date:
            filtered = filtered[
                (filtered["created_date"] >= start_date) & (filtered["created_date"] <= end_date)
            ]
    for col, selected_values in selections.items():
        if col == "created_date" or col == exclude or not selected_values:
            continue
        filtered = filtered[filtered[col].isin(selected_values)]
    return filtered

def apply_text_filters(df: pd.DataFrame, root_cause_query: str = "", resolution_query: str = "") -> pd.DataFrame:
    # AND across words within each field. Case-insensitive. Ignores NaNs.
    out = df
    if root_cause_query:
        words = root_cause_query.split()
        mask = pd.Series(True, index=out.index)
        for w in words:
            mask &= out["root_cause"].astype(str).str.contains(w, case=False, na=False)
        out = out[mask]
    if resolution_query:
        words = resolution_query.split()
        mask = pd.Series(True, index=out.index)
        for w in words:
            mask &= out["resolution_provided_summary"].astype(str).str.contains(w, case=False, na=False)
        out = out[mask]
    return out

def get_available_options(df: pd.DataFrame, selections: dict, target_col: str) -> List:
    subset = apply_selected_filters(df, selections, exclude=target_col)
    return sorted(subset[target_col].dropna().unique().tolist())


def ensure_multiselect_state(key: str, options: List) -> None:
    current = st.session_state.get(key, [])
    if not isinstance(current, list):
        current = [current]
    valid = [value for value in current if value in options]
    if key not in st.session_state or len(valid) != len(current):
        st.session_state[key] = valid


def render_section_header(title: str, caption: Optional[str] = None) -> None:
    st.markdown(f"<div class='section-heading'>{title}</div>", unsafe_allow_html=True)
    if caption:
        st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)


def render_section_divider() -> None:
    st.markdown("<hr class='section-divider' />", unsafe_allow_html=True)


def render_kpi_cards(df: pd.DataFrame, title: str, caption: Optional[str] = None) -> None:
    render_section_header(title, caption)
    if df.empty:
        st.info("No tickets available for this view yet.")
        return
    metrics = [
        ("Tickets", len(df)),
        ("Unique Products", df["product"].nunique()),
        ("Concern Types", df["concern_type"].nunique()),
        ("Cities", df["city"].nunique()),
    ]
    columns = st.columns(len(metrics))
    for column, (label, value) in zip(columns, metrics):
        column.markdown(
            f"""
            <div class="kpi-card">
                <span class="kpi-label">{label}</span>
                <span class="kpi-value">{value:,}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

def _google_api_key():
    return st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")

def _drive_list(folder_id: str, api_key: str):
    url = (
        "https://www.googleapis.com/drive/v3/files"
        f"?q='{folder_id}'+in+parents"
        "&fields=files(id,name,mimeType)"
        f"&key={api_key}"
    )
    r = requests.get(url, timeout=30); r.raise_for_status()
    return r.json().get("files", []) or []

def _drive_get_shortcut_target(file_id: str, api_key: str):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?fields=id,name,mimeType,shortcutDetails(targetId,targetMimeType)&key={api_key}"
    r = requests.get(url, timeout=30); r.raise_for_status()
    j = r.json()
    sd = j.get("shortcutDetails")
    if sd and sd.get("targetId"):
        return sd["targetId"], j.get("name"), sd.get("targetMimeType")
    return file_id, j.get("name"), j.get("mimeType")

def _drive_download(file_id: str, out_path: str, api_key: str):
    url = f"https://www.googleapis.com/drive/v3/files/{file_id}?alt=media&key={api_key}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        with open(out_path, "wb") as f:
            for chunk in resp.iter_content(1024 * 256):
                if chunk:
                    f.write(chunk)
    return out_path

if __name__ == "__main__":

    base_theme = detect_base_theme()
    palette = THEME_COLORS.get(base_theme, THEME_COLORS["light"])
    st.set_page_config(
    page_title="Customer Support Insights",
    page_icon="🧊",
    layout="wide",
    menu_items={
        'Report a bug': "mailto:lakshaydagar@countrydelight.in",
        'About': "This app is built using Streamlit and DSPy to provide insights into customer support tickets."
    }
    )
    st.markdown(build_page_style(palette, base_theme), unsafe_allow_html=True)
    logo_path = "Frame 6.png"
    if os.path.exists(logo_path):
        st.image(logo_path, width=220)
    st.markdown(
        "<div class='dashboard-title'>Customer Support Insights</div>",
        unsafe_allow_html=True,
    )
    st.markdown("##### Monitoring Customer Support Tickets, and Draw Insights")

    st.session_state.df = load_ticket_dataframe(DATA_URL)
    df = st.session_state.df
    if not st.session_state.vector_db_initialized:
            latest_date_in_data = df["created_date"].max()
            if pd.notna(latest_date_in_data) and today_date > latest_date_in_data:
                drive_service = build_drive_service()
                if not drive_service:
                    st.error("Missing drive_service for Google Drive access.")
                else:
                    files = list_files_in_folder(FOLDER_ID)
                    if len(files) != 1:
                        st.error(f"Expected exactly 1 file in folder, found {len(files)}.")
                    else:
                        fid, fname, mtype = files[0]["id"], files[0]["name"], files[0]["mimeType"]
                        # Resolve Google Drive shortcut if needed
                        if mtype == "application/vnd.google-apps.shortcut":
                            fid, fname, mtype = _drive_get_shortcut_target(fid, api_key)

                        # Enforce ZIP
                        if not (fname.lower().endswith(".zip") or mtype == "application/zip"):
                            st.error(f"File is not a ZIP (name={fname}, mimeType={mtype}).")
                        else:
                            zip_path = download_drive_file(fid, os.path.join("downloads", fname))
                            with zipfile.ZipFile(zip_path, "r") as zf:
                                zf.extractall("VectorDB")
                            st.success(f"Dataset Loading Complete!", icon="✅")

            st.session_state.vector_db_initialized = True

    render_kpi_cards(
        df,
        "Tickets At A Glance",
        "Aggregated counts across the full dataset.",
    )

    render_section_divider()
    render_section_header(
        "Overall Ticket Distribution",
        "Understand how tickets break down before applying any filters.",
    )

    col_1, col_2, col_3 = st.columns(3)
    with col_1:
        st.markdown("**Concern Type Mix**")
        concern_counts = df['concern_type'].value_counts().reset_index()
        concern_counts.columns = ['Concern Type', 'Count']
        fig_concern_type = px.pie(concern_counts, values='Count', names='Concern Type')
        st.plotly_chart(fig_concern_type)
    with col_2:
        st.markdown("**Tickets by Region**")
        region_counts = df['region'].value_counts().reset_index()
        region_counts.columns = ['Region', 'Count']
        vc = df['region'].value_counts(dropna=False).rename_axis('Region').reset_index(name='Count')

        total = vc['Count'].sum()
        thresh = 0.01 * total  # 1%
        small = vc['Count'] < thresh
        region_counts = pd.concat(
            [vc[~small], pd.DataFrame([{'Region':'Other','Count': vc.loc[small,'Count'].sum()}])]
            , ignore_index=True
        ) if small.any() else vc
        fig_region = px.pie(region_counts, values='Count', names='Region')
        st.plotly_chart(fig_region)
    with col_3:
        st.markdown("**Level 1 Categories**")
        level1_counts = df['level_1_classification'].value_counts().reset_index()
        level1_counts.columns = ['Level 1 Category', 'Count']
        fig_level1 = px.pie(level1_counts, values='Count', names='Level 1 Category')
        st.plotly_chart(fig_level1)

    col_4, col_5, col_6 = st.columns(3)
    with col_4:
        st.markdown("**Level 2 Focus Areas**")
        level2_counts = df['level_2_classification'].value_counts().head(10)
        st.bar_chart(level2_counts, horizontal=True, x_label='Count', y_label='Issue', width = 'content', sort="-count", color = "#FF7633")
    with col_5:
        st.markdown("**Top Products**")
        product_counts = df['product'].value_counts().head(10)
        st.bar_chart(product_counts, horizontal=True, x_label='Count', y_label='Product', width = 'content', sort="-count", color = "#337EFF")
    with col_6:
        st.markdown("**Tickets by City**")
        city_counts = df['city'].value_counts().head(10)
        st.bar_chart(city_counts, horizontal=True, x_label='Count', y_label='City', width = 'content', sort="-count", color = "#FFEB33")

    render_section_divider()
    st.markdown("# Filter Tickets")
    render_section_header(
        "Combine filters to narrow the analysis. Options update based on your selections.",
    )

    min_val = df['created_date'].min()
    max_val = df['created_date'].max()
    if pd.isna(min_val) or pd.isna(max_val):
        min_val = max_val = today_date
    date_column, _ = st.columns([1.2, 1.8])
    with date_column:
        date_range_selection = st.date_input(
            label="Date Range",
            value=(min_val, max_val),
            min_value=min_val,
            max_value=max_val,
            key="filter_created_date",
        )
    if isinstance(date_range_selection, (list, tuple)) and len(date_range_selection) == 2:
        start_date, end_date = date_range_selection
    else:
        start_date = end_date = date_range_selection
    selected_filters = {"created_date": (start_date, end_date)}

    filter_config = [
        ("concern_type", "Concern Type", "filter_concern_type"),
        ("region", "Region", "filter_region"),
        ("level_1_classification", "Level 1 Classification", "filter_level_1"),
        ("level_2_classification", "Level 2 Classification", "filter_level_2"),
        ("product", "Product", "filter_product"),
        ("city", "City", "filter_city"),
    ]

    filter_rows = [st.columns(3), st.columns(3)]
    for index, (column_name, label, key) in enumerate(filter_config):
        row = index // 3
        column_index = index % 3
        with filter_rows[row][column_index]:
            options = get_available_options(df, selected_filters, column_name)
            ensure_multiselect_state(key, options)
            selected_filters[column_name] = st.multiselect(label, options, key=key)
    col_31, col_32, col_33 = st.columns(3)
    with col_31:
        product_categories = sorted(df['product_category'].dropna().unique().tolist())
        ensure_multiselect_state("filter_product_category", product_categories)
        selected_filters["product_category"] = st.multiselect("Product Category", product_categories, key="filter_product_category")
    
    with col_32:
        root_cause_filter = st.text_input("Filter by Root Cause", key="root_cause_filter")
    with col_33:
        resolution_filter = st.text_input("Filter by Resolution Provided", key="resolution_filter")

# 3) Apply text filters on the already-filtered df
    st.session_state.filtered_df = apply_text_filters(
        df,
        root_cause_query=root_cause_filter,
        resolution_query=resolution_filter,
    )

    st.session_state.filtered_df = apply_selected_filters(st.session_state.filtered_df, selected_filters).copy()
    filtered_df = st.session_state.filtered_df

    render_section_divider()
    render_kpi_cards(
        filtered_df,
        "Filtered Ticket Snapshot",
        "Metrics based on the active filters above.",
    )
    st.markdown("##")
    render_section_header(
        "Filtered Ticket Distribution"
    )

    if filtered_df.empty:
        st.info("No charts to display for the current filters. Adjust selections above to explore more data.")
    else:
        col_21, col_22, col_23 = st.columns(3)
        with col_21:
            st.markdown("**Concern Type Mix (Filtered)**")
            concern_filtered = filtered_df['concern_type'].value_counts().reset_index()
            concern_filtered.columns = ['Concern Type', 'Count']
            fig_concern_type_filter = px.pie(concern_filtered, values='Count', names='Concern Type')
            st.plotly_chart(fig_concern_type_filter, key="pie_concern_type_filtered")
        with col_22:
            st.markdown("**Tickets by Region (Filtered)**")
            region_filtered = filtered_df['region'].value_counts().reset_index()
            region_filtered.columns = ['Region', 'Count']
            vc = filtered_df['region'].value_counts(dropna=False).rename_axis('Region').reset_index(name='Count')
            total = vc['Count'].sum()
            thresh = 0.01 * total  # 1%
            small = vc['Count'] < thresh
            region_counts = pd.concat(
                [vc[~small], pd.DataFrame([{'Region':'Other','Count': vc.loc[small,'Count'].sum()}])]
                , ignore_index=True
            ) if small.any() else vc
            fig_region_filter = px.pie(region_counts, values='Count', names='Region')
            st.plotly_chart(fig_region_filter, key="pie_region_filtered")
        with col_23:
            st.markdown("**Level 1 Categories (Filtered)**")
            level1_filtered = filtered_df['level_1_classification'].value_counts().reset_index()
            level1_filtered.columns = ['Level 1 Category', 'Count']
            fig_level1_filter = px.pie(level1_filtered, values='Count', names='Level 1 Category')
            st.plotly_chart(fig_level1_filter, key="pie_level1_filtered")

        col_24, col_25, col_26 = st.columns(3)
        with col_24:
            st.markdown("**Level 2 Focus Areas (Filtered)**")
            level2_filtered = filtered_df['level_2_classification'].value_counts().head(10)
            st.bar_chart(level2_filtered, horizontal=True, x_label='Count', y_label='Issue', width = 'content', sort='-count', color = "#FFCF33")
        with col_25:
            st.markdown("**Top Products (Filtered)**")
            product_filtered = filtered_df['product'].value_counts().head(10)
            st.bar_chart(product_filtered, horizontal=True, x_label='Count', y_label='Product', width = 'content', sort='-count', color = "#FF337A")
        with col_26:
            st.markdown("**Tickets by City (Filtered)**")
            city_filtered = filtered_df['city'].value_counts().head(10)
            st.bar_chart(city_filtered, horizontal=True, x_label='Count', y_label='City', width = 'content', sort='-count', color = "#336DFF")



    render_section_divider()
    filtered_count = len(filtered_df)
    if filtered_count:
        caption = f"{filtered_count:,} tickets match the current filters."
    else:
        caption = "No tickets match the current filters. Adjust your selections above to see results."  
    st.markdown('### Ticket Details')
    st.write(caption)

    if filtered_df.empty:
        st.info("Try broadening the filters to explore more tickets.")
    else:
        st.dataframe(filtered_df, width = 'content')

    render_section_divider()
    render_section_header(
        "Ticket Summaries",
        "Qualtitative summaries of filtered tickets to know key insights.",
    )
    summary_config = {
        "Full Ticket Narrative": ("expanded_description", "Expanded_Description_Collection"),
        "Customer Issue": ("customer_issue", "Customer_Issue_Collection"),
        "Resolution Provided": ("resolution_provided_summary", "Resolution_Provided_Collection"),
        "Root Cause": ("root_cause", "Root_Cause_Collection"),
    }
    user_selection = st.segmented_control(
        label="Select narrative focus",
        options=list(summary_config.keys()),
        key="action_control",
        default= "Full Ticket Narrative",
    )
    column_name, collection_name = summary_config[user_selection]
    st.session_state.tickets = filtered_df[column_name].dropna().tolist()
    st.session_state.collection_name = collection_name
    st.button('Generate Summary', on_click=summarise_ticket, type="primary")
    st.markdown(st.session_state.summary_output or "*No summary generated yet.*")
    st.divider()
    st.markdown("### About This Dashboard")
    st.write(
        """
        This app was built to analyse customer support tickets, helping teams identify key issues and areas for improvement.
        
        Currently the dashboard shows metrics from last 15 days of customer support tickets.
        We deliberately kept the some tickets out of the analysis to ensure that insights are drawn from relevant tickets only.
        The excluded tickets are which request for stopping the service ("because they subscribed and service was not available to their location") and internal tickets which are raised by the Customer Support Team to communicate among themselves.

        For any questions or feedback, please contact [Lakshay Dagar](mailto:lakshaydagar@countrydelight.in) or [Nishant Rajpoot](mailto:nishantrajpoot@countrydelight.in).
        """)
    st.divider()
    st.write("© 2025 Country Delight")
    st.write("Built with ❤️ by Digital Innovations Team | Country Delight")

