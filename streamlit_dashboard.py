import streamlit as st
import pandas as pd
import dspy
from typing import List, Literal
import os
import numpy as np
import chromadb
from sklearn.decomposition import PCA
from sentence_transformers import SentenceTransformer
import uuid
import gdown
import zipfile
import datetime

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
cerebras_key = st.secrets.get("CEREBRAS_API_KEY")
if not cerebras_key:
    st.session_state.summary_output = "API key not configured. Check Streamlit Cloud secrets."

def summarise_tickets_second():
    DB_path = "VectorDB"
    complaint_number = st.session_state.filtered_df['complaint_number'].dropna().tolist()
    COLLECTION_NAME = st.session_state.collection_name
    chroma_client = chromadb.PersistentClient(path = DB_path)
    collection_client = chroma_client.get_or_create_collection(name = COLLECTION_NAME)
    # print(collection_client.count())

    try:
        embeddings = collection_client.get(where = {"COMPLAINT_NUMBER": {'$in' : complaint_number}}, include = ["embeddings", "metadatas", "documents"])
        num_components = int(min(len(embeddings['ids']), 25))
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
# DSPy context for summarise_ticket
def summarise_ticket():
    summarise_tickets_second()
    tickets = st.session_state.tickets[:20]  # Limit to first 20 tickets
    if not tickets:
        st.session_state.summary_output = "No tickets selected for summary."
        return
    
    with dspy.context(
        lm=dspy.LM(
            model="openai/gpt-oss-120b",
            api_key= os.getenv("CEREBRAS_API_KEY"),
            api_base="https://api.cerebras.ai/v1"
        ),
        cache=True
    ):
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
        try:
            summary = summarise(action=tickets)
            summary_temp = f"### :orange[Summary]:\n{summary.ticket_summary}\n\n### :orange[Key Insights]:\n{summary.key_insights}"
            st.session_state.summary_output = summary_temp
        except Exception as e:
            st.session_state.summary_output = f"Error generating summary: {str(e)}"


if __name__ == "__main__":

    st.set_page_config(layout="wide")

    # Load your actual DataFrame here. Replace with your loading logic.
    # Example: df = pd.read_excel('path_to_your_file.xlsx')
    # For demonstration, using a sample based on your schema.
    columns_sidebar = [
        'city', 'region', 'created_date', 'refund_count_in_15_days',]
    columns_main = ['product', 'concern_type', 'level_1_classification', 'level_2_classification']

    df = pd.read_csv('https://docs.google.com/spreadsheets/d/1MSYdK-Z4qjgudUI6Ky3t3U-Qc2Dxx95D/export?format=csv')
    df = df[df['CONCERN AREA NAME'] != 'Stop Customer']
    df = df[df['CONCERN TYPE NAME'] != 'Internal']
    df = df.rename(columns=lambda x: x.strip().replace(" ", "_").lower())
    df.columns = [c.strip() for c in df.columns]
    # Schema context
    column_names = [
        'customer_id', 'city', 'region', 'created_date', 'refund_count_in_15_days', 'product',
        'concern_type', 'level_1_classification', 'level_2_classification', 'expanded_description',
        'customer_issue', 'root_cause', 'resolution_provided_summary']
    st.session_state.df = df[column_names]
    latest_date_in_data = st.session_state.df['created_date'].max()
    if today_date > latest_date_in_data:
        url = 'https://drive.google.com/drive/folders/1uxnGomO1D2oJShW67c43GeobVbE1TLKZ' 
        gdown.download_folder(url)
        zip_file_path = 'MASTER CHROMADB READ ONLY\\master_chromadb.zip'
        with zipfile.ZipFile(zip_file_path, 'r') as zf:
            extract_to_directory = 'VectorDB'
            zf.extractall(extract_to_directory)
        
    

    # Streamlit Dashboard
    st.image("Frame 6.png")
    st.title('Customer Support Tickets Dashboard')

    st.write("Select filters from the dropdowns below to narrow down the data. 'None' means no filter on that column.")

    # Create a sidebar for filters
    with st.sidebar:
        st.header('Filters')
        filters = {}
        for col in columns_sidebar:
            # Get unique values, handle NaN/None, and sort
            if col == 'created_date':
                st.session_state.df[col] = pd.to_datetime(st.session_state.df[col], errors='coerce').dt.date
                min_val = st.session_state.df[col].min()
                max_val = st.session_state.df[col].max()
                filters[col] = st.date_input('Date', value=(min_val, max_val), min_value=min_val, max_value=max_val)
            else:
                unique_values = sorted(st.session_state.df[col].dropna().unique().tolist())
                options = ['None'] + unique_values
                filters[col] = st.multiselect(f'{col}', options, default=['None'])

    @st.fragment
    def filter_dataframe():
        col_1, col_2, col_3, col_4 = st.columns(4)
        with col_1:
            st.markdown("### Ticket Category")
            unique_values = sorted(st.session_state.df['concern_type'].dropna().unique().tolist())
            for value in unique_values:
                count = len(st.session_state.df[st.session_state.df['concern_type'] == value])
                st.write(f"{value}: {count}")
            options = ['None'] + unique_values
            st.divider()
            filters['concern_type'] = st.multiselect('Apply Filter on Ticket Type', options, default=['None'])
        with col_2:
            st.markdown("### Top 5 Products")
            top_products = st.session_state.df['product'].value_counts().head(5)
            for product, count in top_products.items():
                st.markdown(f"{product}: {count}")
            unique_values = sorted(st.session_state.df['product'].dropna().unique().tolist())
            options = ['None'] + unique_values
            st.markdown('---')
            filters['product'] = st.multiselect('Apply Filter on Product', options, default=['None'])
        with col_3:
            st.markdown("### Level-1 Ticket Categories")
            top_cities = st.session_state.df['level_1_classification'].value_counts().head(3)
            st.markdown(':blue-background[Top 3 high level concerns overall]')
            for city, count in top_cities.items():
                st.markdown(f":red[{city}]: {count}")
            unique_values = sorted(st.session_state.df['level_1_classification'].dropna().unique().tolist())
            options = ['None'] + unique_values
            st.markdown('---')
            filters['level_1_classification'] = st.multiselect('Apply Filter on Level 1 Classification', options, default=['None'])
        with col_4:
            st.markdown('### Level-2 Ticket Categories')
            st.markdown(':blue-background[Top 3 low level concerns overall]')
            top_issues = st.session_state.df['level_2_classification'].value_counts().head(3)
            for issue, count in top_issues.items():
                st.markdown(f":red[{issue}]: {count}")
            options = ['None'] + sorted(st.session_state.df['level_2_classification'].dropna().unique().tolist())
            st.markdown('---')
            filters['level_2_classification'] = st.multiselect('Apply Filter on Level 2 Classification', options, default=['None'])
        
        # Apply filters to the DataFrame
        st.session_state.filtered_df = st.session_state.df.copy()
        for col, value in filters.items():
            if value != ['None'] and value:  # Skip if only 'None' or empty
                if col == 'created_date':
                    # Handle date range
                    start_date, end_date = value
                    st.session_state.filtered_df = st.session_state.filtered_df[(st.session_state.filtered_df[col] >= start_date) & (st.session_state.filtered_df[col] <= end_date)]
                else:
                    # Use isin() for multiselect values
                    st.session_state.filtered_df = st.session_state.filtered_df[st.session_state.filtered_df[col].astype(str).isin(value)]

    # Display the filtered DataFrame
    st.subheader('Filtered DataFrame')
    st.write(f"Total rows after filtering: {len(st.session_state.filtered_df)}")
    if st.session_state.filtered_df.empty:
        st.write("No data matches the filters. Try adjusting them.")
    else:
        with st.status("Loading filtered data...", expanded=False) as status:
            st.dataframe(st.session_state.filtered_df, width= 'stretch')
            status.update(label = "Data loaded.", expanded=True)

    @st.fragment
    def summary():
        st.markdown("---")
        st.markdown("## Ticket Summary")
        tickets = []  
        user_selection = st.segmented_control(label="Provide Summary for", options=["Entire Ticket", "Why did the customer reach out ?", "Resolution Provided","Root Causes"], key="action_control")
        if user_selection == "Entire Ticket":
            st.session_state.tickets = st.session_state.filtered_df['expanded_description'].dropna().tolist()
            st.session_state.collection_name = "Expanded_Description_Collection"
        elif user_selection == "Why did the customer reach out ?":
            st.session_state.tickets = st.session_state.filtered_df['customer_issue'].dropna().tolist()
            st.session_state.collection_name = "Customer_Issue_Collection"
        elif user_selection == "Resolution Provided":
            st.session_state.tickets = st.session_state.filtered_df['resolution_provided_summary'].dropna().tolist()
            st.session_state.collection_name = "Resolution_Provided_Collection"
        elif user_selection == "Root Causes":
            st.session_state.tickets = st.session_state.filtered_df['root_cause'].dropna().tolist()
            st.session_state.collection_name = "Root_Cause_Collection"
        st.button('Summarise', on_click = summarise_ticket)
        with st.status("Preparing summary...", expanded=False) as status_summary:
            status_summary.update(label = "Generating summary...", expanded=False)
            st.markdown(st.session_state.summary_output)
            status_summary.update(label = "Summary ready.", expanded=True)
    filter_dataframe()
    summary()

