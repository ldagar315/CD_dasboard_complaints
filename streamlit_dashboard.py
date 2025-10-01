import streamlit as st
import pandas as pd
import dspy
from typing import List, Literal


if 'summary_output' not in st.session_state:
    st.session_state.summary_output = ""
if 'tickets' not in st.session_state:
    st.session_state.tickets = []

# DSPy context for summarise_ticket
def summarise_ticket():
    tickets = st.session_state.tickets[:20]  # Limit to first 20 tickets
    if not tickets:
        st.session_state.summary_output = "No tickets selected for summary."
        return
    with dspy.context(
        lm=dspy.LM(
            model="openai/gpt-oss-120b",
            api_key="csk-5prhjrydy285t945r2y3jpr2nhceecr6m3kpeprfeh9v55wk",
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

    df = pd.read_excel("for_schema_excel_sheet_v2.xlsx")

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
                df[col] = pd.to_datetime(df[col], errors='coerce').dt.date
                min_val = df[col].min()
                max_val = df[col].max()
                filters[col] = st.date_input('Date', value=(min_val, max_val), min_value=min_val, max_value=max_val)
            else:
                unique_values = sorted(df[col].dropna().unique().tolist())
                options = ['None'] + unique_values
                filters[col] = st.multiselect(f'{col}', options, default=['None'])

    col_1, col_2, col_3, col_4 = st.columns(4)
    with col_1:
        st.markdown("### Ticket Category")
        unique_values = sorted(df['concern_type'].dropna().unique().tolist())
        for value in unique_values:
            count = len(df[df['concern_type'] == value])
            st.write(f"{value}: {count}")
        options = ['None'] + unique_values
        st.divider()
        filters['concern_type'] = st.multiselect('Apply Filter on Ticket Type', options, default=['None'])
    with col_2:
        st.markdown("### Top 5 Products")
        top_products = df['product'].value_counts().head(5)
        for product, count in top_products.items():
            st.markdown(f"{product}: {count}")
        unique_values = sorted(df['product'].dropna().unique().tolist())
        options = ['None'] + unique_values
        st.markdown('---')
        filters['product'] = st.multiselect('Apply Filter on Product', options, default=['None'])
    with col_3:
        st.markdown("### Level-1 Ticket Categories")
        top_cities = df['level_1_classification'].value_counts().head(3)
        st.markdown(':blue-background[Top 3 high level concerns overall]')
        for city, count in top_cities.items():
            st.markdown(f":red[{city}]: {count}")
        unique_values = sorted(df['level_1_classification'].dropna().unique().tolist())
        options = ['None'] + unique_values
        st.markdown('---')
        filters['level_1_classification'] = st.multiselect('Apply Filter on Level 1 Classification', options, default=['None'])
    with col_4:
        st.markdown('### Level-2 Ticket Categories')
        st.markdown(':blue-background[Top 3 low level concerns overall]')
        top_issues = df['level_2_classification'].value_counts().head(3)
        for issue, count in top_issues.items():
            st.markdown(f":red[{issue}]: {count}")
        options = ['None'] + sorted(df['level_2_classification'].dropna().unique().tolist())
        st.markdown('---')
        filters['level_2_classification'] = st.multiselect('Apply Filter on Level 2 Classification', options, default=['None'])
        


    # Apply filters to the DataFrame
    filtered_df = df.copy()
    for col, value in filters.items():
        if value != ['None'] and value:  # Skip if only 'None' or empty
            if col == 'created_date':
                # Handle date range
                start_date, end_date = value
                filtered_df = filtered_df[(filtered_df[col] >= start_date) & (filtered_df[col] <= end_date)]
            else:
                # Use isin() for multiselect values
                filtered_df = filtered_df[filtered_df[col].astype(str).isin(value)]

    # Display the filtered DataFrame
    st.subheader('Filtered DataFrame')
    st.write(f"Total rows after filtering: {len(filtered_df)}")
    if filtered_df.empty:
        st.write("No data matches the filters. Try adjusting them.")
    else:
        st.dataframe(filtered_df, width= 'stretch')
    
    st.markdown("---")
    st.markdown("## Ticket Summary")
    tickets = []  
    user_selection = st.segmented_control(label="Provide Summary for", options=["Entire Ticket", "Why did the customer reach out ?", "Resolution Provided","Root Causes"], key="action_control")
    if user_selection == "Entire Ticket":
        st.session_state.tickets = filtered_df['expanded_description'].dropna().tolist()
    elif user_selection == "Why did the customer reach out ?":
        st.session_state.tickets = filtered_df['customer_issue'].dropna().tolist()
    elif user_selection == "Resolution Provided":
        st.session_state.tickets = filtered_df['resolution_provided_summary'].dropna().tolist()
    elif user_selection == "Root Causes":
        st.session_state.tickets = filtered_df['root_cause'].dropna().tolist()
    st.button('Summarise', on_click = summarise_ticket)
    st.markdown('''
        :red[Streamlit] :orange[can] :green[write] :blue[text] :violet[in]
        :gray[pretty] :rainbow[colors] and :blue-background[highlight] text.''')
    st.markdown(st.session_state.summary_output)
