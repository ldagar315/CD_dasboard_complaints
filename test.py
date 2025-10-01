from agno.agent import Agent
from agno.models.cerebras import Cerebras
import dspy
import pandas as pd
import duckdb
from pydantic import BaseModel, Field
from agno.tools.reasoning import ReasoningTools
from agno.tools.duckdb import DuckDbTools
from agno.db.sqlite import SqliteDb
from rich.pretty import pprint
from agno.knowledge import Knowledge
from agno.tools.knowledge import KnowledgeTools
import system_prompt
from agno.tools.sql import SQLTools
import weave
import asyncio
import os
from phoenix.otel import register
from agno.models.groq import Groq
from agno.models.google import Gemini

# Set environment variables for Arize Phoenix
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
os.environ['WANDB_API_KEY'] = '4bc35b8e443c90c5bfb2d89bd1fb274d107ef6d7'
#os.environ['PHOENIX_CLIENT_HEADERS'] = "ak-15fa9c3e-6c0c-40f9-923e-030a7f5edc1f-vKi92gpwYGZVH7NMbq_P5nryqil_OPqF"


tracer_provider = register(
    project_name="agno-complaints-dashboard",  # Default is 'default'
    auto_instrument=True,  # Automatically use the installed OpenInference instrumentation
)

###############################################################################
# Load Data
###############################################################################
input_file = "C:\\Users\\cd-user\\Downloads\\processed_output_2025-09-24_09-40-38.xlsx"
df = pd.read_excel(input_file)
df = df.rename(columns=lambda x: x.strip().replace(" ", "_").lower())
df.columns = [c.strip() for c in df.columns]

# Schema context
column_names = [
    'customer_id', 'city', 'region', 'primary_route',
    'primary_franchise', 'closed_by', 'created_by', 'created_date',
    'complaint_source', 'refund_count_in_15_days', 'product',
    'concern_type', 'level_1_classification', 'level_2_classification'
]

df = df[column_names]

city_names = df['city'].unique().tolist()
region_names = df['region'].unique().tolist()
concern_values = df['concern_type'].unique().tolist()
level_1_classification_values = df['level_1_classification'].unique().tolist()
level_2_classification_values = df['level_2_classification'].unique().tolist()
#product_values = df['product'].unique().tolist()

###############################################################################
# DSPy Configuration
###############################################################################
dspy.configure(
    lm=dspy.LM(
        model="openai/gpt-oss-120b",
        api_key="csk-5prhjrydy285t945r2y3jpr2nhceecr6m3kpeprfeh9v55wk",
        api_base="https://api.cerebras.ai/v1"
    ),
    cache=True
)

db = SqliteDb(db_file="my_database.db")

knowledge = Knowledge(
    name="My Knowledge Base",
    contents_db=db  # This enables content tracking!
)

knowledge_tools = KnowledgeTools(
    knowledge=knowledge,
    enable_think=True,
    enable_search=True,
    enable_analyze=True,
    add_few_shot=True,
)

###############################################################################
# Output format + Signatures
###############################################################################
class SQL_output_format(BaseModel):
    answer: dict = Field(
        description="Dictionary of form: {SQL_query: 'query'}"
    )

Generate_SQL_instructions = f"""
    You are a meticulous and accurate SQL code generator.
    Your task is to convert a natural language query into a SQL query that can be run on a pandas DataFrame named 'df'.

    ### SCHEMA CONTEXT
    TABLE NAME: df
    COLUMN NAMES: {column_names}

    ### COLUMN VALUE EXAMPLES
    - The 'city' column contains values like: {city_names}

    ### STRICT RULES
    1. Use ONLY the provided column names.
    2. Do not invent new columns.
    3. Output ONLY the final SQL query.
    4. Always search for %LIKE% for every column value. Never use exact match (=).
"""

class GenerateSQL(dspy.Signature):
    instruction: str = dspy.InputField(desc="Instructions for the SQL generator")
    columns_names: list = dspy.InputField(desc="Available column names")
    city_names: list = dspy.InputField(desc="Available values in 'city'")
    region_names: list = dspy.InputField(desc="Available values in 'region'")
    concern_values: list = dspy.InputField(desc="Available values in 'concern_type'")
    level_1_classification_values: list = dspy.InputField(desc="Available values in 'level_1_classification'")
    level_2_classification_values: list = dspy.InputField(desc="Available values in 'level_2_classification'")
    query: str = dspy.InputField(desc="Natural language query to convert")
    SQL_query: SQL_output_format = dspy.OutputField(desc="Final SQL query")
    
rephrasing_instructions = """
    Decompose a complex, analytical query into its first, simple, factual data-retrieval command.

    Your goal is to strip away all analytical language (like 'why', 'reason', 'summarize', 'cause') and transform the user's question into a pure command to fetch raw data.
    
    ---
    Follow these instructions strictly:
    1.  Analyze the user's complex query.
    2.  Identify the core subject and the filters (e.g., 'incomplete', 'Hyderabad region', concern_type, level_1_classification).
    3.  Rephrase the query into a simple, direct question that ONLY asks to retrieve that raw data.
    4.  **Do NOT generate SQL code.**
    5.  **Do NOT attempt to answer the complex query.** Your only output is the rephrased question.
    
    ##STRICT RULE:
    YOU WILL BE GIVEN WITH 'REASON FOR X IN Y HAVING Z IN W', 'ROOT CAUSE FOR X IN Y HAVING Z IN W', YOUR TASK IS TO CONVERT IT INTO 'LIST ALL X IN Y HAVING Z IN W GROUP BY A RANK B....'

    ---
    Here are some examples:

    Original Query: What were the root causes for incomplete deliveries?
    Rephrased Query: Find all entries with incomplete deliveries.

    Original Query: what is the reason behind complaints in Hyderabad region?
    Rephrased Query: List all complaints in hyderabad region.

    Original Query: Summarize the problems faced by customer in dwarka area of delhi ncr region.
    Rephrased Query: List all problems faced by customer in dwarka area of delhi ncr region.

    Original Query: what is the reason for complaints in delhi ncr region having Partial / Missing Items at Delivery as concern.
    Rephrased Query: List all complaints from the Delhi NCR region having Partial / Missing Items at Delivery as concern.
"""

class RephrasedQuery(dspy.Signature):
    instruction: str = dspy.InputField(desc="The simplification instructions")
    query: str = dspy.InputField(desc="The user's original query")
    rephrased_query: str = dspy.OutputField(desc="Simplified factual query")

def query_simplifier(query: str) -> str:
    rephrasor = dspy.ChainOfThought(RephrasedQuery)
    rephrased = rephrasor(
        instruction=rephrasing_instructions,
        query=query
    )
    return rephrased.rephrased_query

###############################################################################
# Tool Function
###############################################################################
def complaints_sql_tool(query: str) -> dict:
    """
    Tool: Takes a natural language query about complaints data,
    simplifies it if needed, generates SQL, executes on DuckDB, and returns a DataFrame.
    """
    try:
        # Step 1: Simplify the query
        simplified_query = query_simplifier(query)
        print(f"Original Query: {query}")
        print(f"Simplified Query: {simplified_query}")

        # Step 2: Generate SQL
        predictor = dspy.ChainOfThought(GenerateSQL)
        prediction = predictor(
            instruction=Generate_SQL_instructions,
            columns_names=column_names,
            city_names=city_names,
            region_names=region_names,
            concern_values=concern_values,
            level_1_classification_values=level_1_classification_values,
            level_2_classification_values=level_2_classification_values,
            query=simplified_query
        )

        sql_query = prediction.SQL_query.answer["SQL_query"]
        print(f"Generated SQL: {sql_query}")

        # Step 3: Run SQL on DuckDB
        result_df = duckdb.query(sql_query).to_df()
        result = [result_df.to_dict(orient="list")]
        
        return result

    except Exception as e:
        print(f"Error in complaints_sql_tool: {e}")
        result = [pd.DataFrame({"error": [str(e)]}).to_dict(orient="list")]
        return result


agent = Agent(
    #model=Cerebras(id="llama-4-maverick-17b-128e-instruct", api_key="csk-5prhjrydy285t945r2y3jpr2nhceecr6m3kpeprfeh9v55wk", max_completion_tokens= 32000),
    model = Gemini(id="gemini-2.5-flash", api_key="AIzaSyBDrcGqmpaiLgzZ-flFoHoBctHVCP7gZ1I"),
    reasoning_model= Groq(id = "deepseek-r1-distill-llama-70b", api_key="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"),
    system_message= system_prompt.system_prompt,
    description = "You are a helpful data analyst assistant of Country Delight. You have access to customer support tickets data. Use SQL to answer questions about the data. If you don't know the answer, just say you don't know. Do not make up an answer.",
    tools = [DuckDbTools(db_path="my_database.db", include_tools =["inspect_query",'run_query','summarize_table','full_text_search','create_fts_index']), ReasoningTools(add_instructions=True)],
    #knowledge_retriever= complaints_sql_tool,
    markdown=True,
    debug_mode=True,
    debug_level=2,
    instructions= "Follow the system prompt carefully.",
    retries= 3,
)

questions_list = [ 
 "How many complaints did we receive today ?",
 "What information are the user asking us most about ?",
 "What are the top 3 main causes of complaints overall?",
 "How many complaints came in for Cow Milk ? ",
 "What are the most common issues reported for Paneer in Delhi-NCR ?",
 "Are customers asking questions about Bananas ?",
 "List all feedback related to Dahi in North ?",
 "Which city had the highest number of complaints ?",
 "What are the customers asking us to do in Delhi?",
 "In South region, what were the top resolutions provided for complaints for product shortage?",
 "Give me a breakdown of issues in Gurugram by product",
 "At what time do most complaints come ?",
 "Can you compare the complaints in morning vs evening ?",
 "Any feedbacks lately?",
 "Tell me why customers are upset about Cow Milk",
 "Give me details about bread tickets",
 "Show issues for ‘Paneer’ in Chennai",
 "How many complaints did we get for _Avocado Smoothie_?",
 "Any tickets from Houston ?",
 "What was the resolution for yogurt complaints on 2022-01-01?",
 "Which product has the most complaints overall?",
 "Across all regions, what are the most common customer issues?",
 "How many total complaints vs requests vs feedback have we had so far ?",
 "Summarize root causes for all complaints in tabular form"]

"""for question in questions_list:
    print("\n\n")
    print("###" * 10)
    print(f"Question: {question}")
    print("###" * 10)
    agent.print_response(question)"""
    
"""@weave.op()
def run():
    return agent.print_response( "In South region, what were the top resolutions provided for complaints for product shortage?")

run()"""

agent.print_response("Give me a breakdown of issues in Gurgoan by product", 
                     show_full_reasoning=True,
                     stream_intermediate_steps=True)
# Print the response in the terminal
"""run_response = agent.print_response( "How many complaints came in for Cow Milk ?")
# Print metrics per message
if run_response.messages:
    for message in run_response.messages:
        if message.role == "assistant":
            if message.content:
                print(f"Message: {message.content}")
            elif message.tool_calls:
                print(f"Tool calls: {message.tool_calls}")
            print("---" * 5, "Metrics", "---" * 5)
            pprint(message.metrics.to_dict())
            print("---" * 20)

# Print the aggregated metrics for the whole run
print("---" * 5, "Run Metrics", "---" * 5)
pprint(run_response.metrics.to_dict())
# Print the aggregated metrics for the whole session
print("---" * 5, "Session Metrics", "---" * 5)
pprint(agent.get_session_metrics().to_dict())"""
