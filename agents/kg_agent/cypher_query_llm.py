from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from proxy_setting import set_proxy
from llm_config import *

set_proxy()
load_dotenv()

graph = get_graph_db()

llm = get_gemini_llm(temperature=0.0)

examples = [
    {
        "question": "Phương pháp điều trị cho bệnh u lympho sau phúc mạc là gì?",
        "query": "MATCH (d:Disease {name: 'u lympho sau phúc mạc'})-[:TREATED_BY]-(t:Treatment) RETURN t;",
    },
    {
        "question": "Nguyên nhân của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
        "query": "MATCH (d:Disease {name: 'chảy máu khoảng cách sau phúc mạc'}) RETURN d;",
    },
    {
        "question": "Triệu chứng của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
        "query": "MATCH (d:Disease {name: 'chảy máu khoảng cách sau phúc mạc'})-[:HAS_SYMPTOM]-(s:Symptom) RETURN s;",
    },
    {
        "question": "Những bệnh lý nào có thể xuất hiện khi có triệu chứng khóc và đau?",
        "query": "MATCH (s:Symptom)-[:HAS_SYMPTOM]-(d:Disease) WHERE s.symptoms CONTAINS 'khóc' AND s.symptoms CONTAINS 'đau' RETURN d;",
    },
    {
        "question": "Có những loại thuốc phổ biến nào để điều trị bệnh chảy máu khoảng cách sau phúc mạc?",
        "query": "MATCH (d:Disease {name: 'chảy máu khoảng cách sau phúc mạc'})-[:PRESCRIBED]-(m:Medication) RETURN m;",
    },
    {
        "question": "Người bệnh u lympho sau phúc mạc nên ăn thực phẩm gì?",
        "query": "MATCH (d:Disease {name: 'u lympho sau phúc mạc'})-[:HAS_ADVICE]-(a:Advice) RETURN a;",
    },
    {
        "question": "Bệnh u lympho sau phúc mạc có thể liên quan đến những bệnh nào khác?",
        "query": "MATCH (d:Disease {name: 'u lympho sau phúc mạc'})-[:ASSOCIATED_WITH]->(d2:Disease) RETURN d2;",
    }
]

example_prompt = PromptTemplate.from_template(
    "User input: {question}\nCypher query: {query}", 
    template_format = "jinja2"
)

prefix_prompt = """ 
    I have a knowledge graph for Vietnamese traditional medicine, where each node represents a disease "Disease", "Treatment", "Symptom", "Medication", "Advice". Each node can have the following properties:

    1. Disease
        - name
        - description
        - category
        - cause

    2. Treatment
        - disease_name
        - method
        - department
        - success_rate

    3. Symptom
        - disease_name
        - symptoms
        - diagnosis
        - risk_group

    4. Medication
        - disease_name
        - common_drugs
        - drug_info
        - recommended_drugs

    5. Advice
        - disease_name
        - foods_to_eat
        - foods_to_avoid
        - recommended_meals
        - prevention

    Relationships:
    - (Disease)-[:TREATED_BY]-(Treatment)
    - (Disease)-[:HAS_SYMPTOM]-(Symptom)
    - (Disease)-[:PRESCRIBED]-(Medication)
    - (Disease)-[:HAS_ADVICE]-(Advice)
    - (Disease)-[:ASSOCIATED_WITH]->(Disease)

    You are a Neo4j Cypher expert. Given an input question, create a syntactically correct Cypher query to run. Each query is limit 5 records.
    Below are a number of examples of questions and their corresponding Cypher queries:
"""
prompt = FewShotPromptTemplate( 
    examples = examples, 
    example_prompt = example_prompt,
    prefix = prefix_prompt,
    suffix = "User input: {question}\nCypher query: ",
    input_variables = ["question"],
)

gemini_chain = GraphCypherQAChain.from_llm( 
    llm = llm, 
    graph = graph, 
    verbose = True, 
    cypher_prompt = prompt,
    allow_dangerous_requests = True, 
    return_direct = True
)

def retrieve_context_from_kg(question: str):
    return gemini_chain.invoke(question)
