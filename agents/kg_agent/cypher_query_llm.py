from dotenv import load_dotenv
from langchain_neo4j import GraphCypherQAChain
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from pathlib import Path 
import sys 
sys.path.append(str(Path(__file__).parent.parent))
from proxy_setting import set_proxy
from llm_config import *
from prompt import cypher_query

set_proxy()
load_dotenv()

graph = get_graph_db()

llm = get_gemini_llm(temperature=0.0)

examples = [
    {
        "question": "Phương pháp điều trị cho bệnh u lympho sau phúc mạc là gì?",
        "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'u lympho sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 5;",
    },
    {
        "question": "Nguyên nhân của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
        "query": "MATCH (d:Disease) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN d LIMIT 5;",
    },
    {
        "question": "Triệu chứng của bệnh chảy máu khoảng cách sau phúc mạc là gì?",
        "query": "MATCH (d:Disease)-[:HAS_SYMPTOM]-(s:Symptom) WHERE d.name CONTAINS 'chảy máu khoảng cách sau phúc mạc' AND d.description IS NOT NULL RETURN s LIMIT 5;",
    },
    {
        "question": "Những bệnh lý nào có thể xuất hiện khi có triệu chứng khóc và đau?",
        "query": "MATCH (s:Symptom) WHERE s.symptoms CONTAINS 'khóc' AND s.symptoms CONTAINS 'đau' MATCH (s)-[:HAS_SYMPTOM]-(d:Disease) WHERE d.description IS NOT NULL RETURN d LIMIT 5;",
    },
    {
        "question": "Có những loại thuốc phổ biến nào để điều trị bệnh chảy máu khoảng cách sau phúc mạc?",
        "query": "MATCH (m:Medication) WHERE m.disease_name CONTAINS 'chảy máu khoảng cách sau phúc mạc' RETURN m LIMIT 5;",
    },
    {
        "question": "Người bệnh u lympho sau phúc mạc nên ăn thực phẩm gì?",
        "query": "MATCH (a:Advice) WHERE a.disease_name CONTAINS 'u lympho sau phúc mạc' RETURN a LIMIT 5;",
    },
    {
        "question": "Bệnh u lympho sau phúc mạc có thể liên quan đến những bệnh nào khác?",
        "query": "MATCH (d:Disease)-[:ASSOCIATED_WITH]->(d2:Disease) WHERE d.name CONTAINS 'u lympho sau phúc mạc' AND exists(d2.description) RETURN d2 LIMIT 5;",
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
        - embedding

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

    You are a Neo4j Cypher expert. Given an input question, create a syntactically correct Cypher query to run.
    - Always match disease or symptom names using `CONTAINS` instead of exact equality.
    - Always change the entity to lowercase.
    - If the matched node label is `Disease`, ensure it has a non-null `description` property (`d.description IS NOT NULL`). For other labels you can ignore this condition.
    - If the question is not asked to return symptoms and disease is one of the return list, then only disease nodes will be returned.
    - After filtering, limit the results to 30 records.
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
    cypher_prompt = prompt,
    allow_dangerous_requests = True,
    return_direct = True
)

def retrieve_context_from_kg(question: str):
    return gemini_chain.invoke(question)
