from py2neo import Graph, Node, Relationship
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_config import *

URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "Hung31102004")

embedding_model = get_fpt_vietnamese_embedding()

def to_lowercase(text):
    if isinstance(text, str):
        return text.lower()  # Convert to lowercase
    else:
        return text

def parse_list_field(field_value):
    """Parse string representation of list into actual list and convert to lowercase."""
    if not field_value or field_value == "Không có thông tin":
        return []
    
    if isinstance(field_value, str):
        # Try to parse string representation of list
        if field_value.startswith('[') and field_value.endswith(']'):
            try:
                # Use ast.literal_eval to safely parse the string list
                import ast
                parsed_list = ast.literal_eval(field_value)
                if isinstance(parsed_list, list):
                    return [item.lower().strip() if isinstance(item, str) else item for item in parsed_list]
            except (ValueError, SyntaxError):
                # If parsing fails, split by comma and clean up
                items = field_value.strip('[]').split(',')
                return [item.strip().strip('\'"').lower() for item in items if item.strip()]
        else:
            # Single item, not a list
            return [field_value.lower().strip()]
    elif isinstance(field_value, list):
        return [item.lower().strip() if isinstance(item, str) else item for item in field_value]
    else:
        return [str(field_value).lower()]

def truncate_text(text, max_tokens=2000):
    if not text or not isinstance(text, str):
        return ""

    max_chars = max_tokens * 3.5  # 2000 tokens ≈ 7000 characters
    
    if len(text) <= max_chars:
        return text
    
    # Truncate at character limit first
    truncated = text[:int(max_chars)]
    
    # 1. Try to find sentence boundary (. ! ?)
    sentence_endings = ['.', '!', '?']
    best_sentence_end = -1
    for ending in sentence_endings:
        pos = truncated.rfind(ending)
        if pos > best_sentence_end:
            best_sentence_end = pos
    
    # 2. Try to find clause boundary (, ; :)
    clause_endings = [',', ';', ':']
    best_clause_end = -1
    for ending in clause_endings:
        pos = truncated.rfind(ending)
        if pos > best_clause_end:
            best_clause_end = pos
    
    # 3. Find word boundary (space)
    last_space = truncated.rfind(' ')
    
    if best_sentence_end > len(truncated) * 0.8:
        truncated = truncated[:best_sentence_end + 1]
    # Use clause boundary if it's not too far back (within last 30%)
    elif best_clause_end > len(truncated) * 0.7:
        truncated = truncated[:best_clause_end + 1]
    # Otherwise use word boundary
    elif last_space > 0:
        truncated = truncated[:last_space]
    
    return truncated.strip()

def create_disease_embedding(name, description):
    try:
        # Combine name and description with clear structure
        if description and description.strip():
            combined_text = f"Tên bệnh: {name}\nMô tả: {description}"
        else:
            combined_text = f"Tên bệnh: {name}"
        
        # Truncate if too long (use 2000 tokens to leave buffer)
        combined_text = truncate_text(combined_text, max_tokens=2000)
        
        if not combined_text.strip():
            print(f"Empty text after truncation for disease: {name}")
            return None
            
        # Create embedding using Vietnamese_Embedding model
        embedding = embedding_model.embed_query(combined_text)
        
        # Validate embedding
        if embedding and len(embedding) > 0:
            return embedding
        else:
            print(f"Empty embedding generated for disease: {name}")
            return None
            
    except Exception as e:
        print(f"Error creating embedding for {name}: {e}")
        return None
    
def clear_graph(graph_instance):
    query = """
    MATCH (n)
    DETACH DELETE n
    """
    graph_instance.run(query)
    print("Graph has been cleared...")

def create_unique_constraints(graph_instance):
    """Create unique constraints to prevent duplicates"""
    constraints = [
        "CREATE CONSTRAINT disease_name_unique IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
        "CREATE CONSTRAINT treatment_disease_unique IF NOT EXISTS FOR (t:Treatment) REQUIRE t.disease_name IS UNIQUE", 
        "CREATE CONSTRAINT symptom_disease_unique IF NOT EXISTS FOR (s:Symptom) REQUIRE s.disease_name IS UNIQUE",
        "CREATE CONSTRAINT medication_disease_unique IF NOT EXISTS FOR (m:Medication) REQUIRE m.disease_name IS UNIQUE",
        "CREATE CONSTRAINT advice_disease_unique IF NOT EXISTS FOR (a:Advice) REQUIRE a.disease_name IS UNIQUE"
    ]
    
    for constraint in constraints:
        try:
            graph_instance.run(constraint)
            print(f"Created constraint: {constraint.split('FOR')[1].split('REQUIRE')[0].strip()}")
        except Exception as e:
            print(f"Constraint may already exist: {e}")

def check_node_exists(graph, associated_disease):
    disease_name = to_lowercase(associated_disease)
    query = """
    MATCH (n:Disease {name: $disease_name})
    RETURN COUNT(n) > 0 AS node_exists
    """
    result = graph.run(query, disease_name=disease_name).data()
    return result[0]["node_exists"] if result else False

def process_row(row, graph_instance, df_data):
    try:
        disease_name = row['tên_bệnh']
        disease_description = row['mô_tả_bệnh']
        disease_category = row['loại_bệnh']
        disease_prevention = row['cách_phòng_tránh']
        disease_cause = row['nguyên_nhân']
        disease_symptom = row['triệu_chứng']
        people_easy_get = row['đối_tượng_dễ_mắc_bệnh']
        associated_disease = row['bệnh_đi_kèm']
        cure_method = row['phương_pháp']
        cure_department = row['khoa_điều_trị']
        cure_probability = row['tỉ_lệ_chữa_khỏi']
        check_method = row['kiểm_tra']
        nutrition_do_eat = row['nên_ăn_thực_phẩm_chứa']
        nutrition_not_eat = row['không_nên_ăn_thực_phẩm_chứa']
        nutrition_recommend_meal = row['đề_xuất_món_ăn']
        drug_recommend = row['đề_xuất_thuốc']
        drug_common = row['thuốc_phổ_biến']
        drug_detail = row['thông_tin_thuốc']
    except KeyError as e:
        print(f"Missing column: {e}")
        return

    # Initialize disease_node as None
    disease_node = None

    if disease_name and disease_description and disease_category and disease_cause:
        # Create embedding for disease
        disease_embedding = create_disease_embedding(disease_name, disease_description)
        
        # Create disease node using py2neo Node and merge (compatible approach)
        disease_node = Node("Disease", 
                           name=to_lowercase(disease_name),
                           description=to_lowercase(disease_description),
                           category=to_lowercase(disease_category),
                           cause=to_lowercase(disease_cause),
                           embedding=disease_embedding)
        
        graph_instance.merge(disease_node, "Disease", "name")
        

    if cure_method and cure_department and cure_probability:
        # Create treatment node and relationship using MERGE query
        treatment_query = """
        MERGE (t:Treatment {disease_name: $disease_name})
        ON CREATE SET t.method = $method, t.department = $department, t.success_rate = $success_rate
        ON MATCH SET t.method = $method, t.department = $department, t.success_rate = $success_rate
        WITH t
        MATCH (d:Disease {name: $disease_name})
        MERGE (d)-[:TREATED_BY]->(t)
        """
        graph_instance.run(treatment_query,
                         disease_name=to_lowercase(disease_name),
                         method=to_lowercase(cure_method),
                         department=to_lowercase(cure_department),
                         success_rate=to_lowercase(cure_probability))

    if disease_symptom and check_method and people_easy_get and disease_node:
        # Create symptom node and relationship
        symptoms_list = parse_list_field(disease_symptom)
        diagnosis_list = parse_list_field(check_method)
        symptom_node = Node("Symptom", 
                           disease_name=to_lowercase(disease_name), 
                           symptoms=symptoms_list, 
                           diagnosis=diagnosis_list, 
                           risk_group=to_lowercase(people_easy_get))
        graph_instance.merge(symptom_node, "Symptom", "disease_name")
        has_rela = Relationship(disease_node, "HAS_SYMPTOM", symptom_node)
        graph_instance.create(has_rela)

    if drug_recommend and drug_common and drug_detail and disease_node:
        # Create medication node and relationship
        recommended_drugs_list = parse_list_field(drug_recommend)
        common_drugs_list = parse_list_field(drug_common)
        medication_node = Node("Medication", 
                              disease_name=to_lowercase(disease_name), 
                              common_drugs=common_drugs_list, 
                              drug_info=to_lowercase(drug_detail), 
                              recommended_drugs=recommended_drugs_list)
        graph_instance.merge(medication_node, "Medication", "disease_name")
        prescribed_rela = Relationship(disease_node, "PRESCRIBED", medication_node)
        graph_instance.create(prescribed_rela)

    if nutrition_do_eat and nutrition_not_eat and nutrition_recommend_meal and disease_prevention and disease_node:
        # Create nutrition node and relationship
        foods_to_eat_list = parse_list_field(nutrition_do_eat)
        foods_to_avoid_list = parse_list_field(nutrition_not_eat) 
        recommended_meals_list = parse_list_field(nutrition_recommend_meal)
        nutrition_node = Node("Advice", 
                             disease_name=to_lowercase(disease_name), 
                             foods_to_eat=foods_to_eat_list, 
                             recommended_meals=recommended_meals_list, 
                             foods_to_avoid=foods_to_avoid_list, 
                             prevention=to_lowercase(disease_prevention))
        graph_instance.merge(nutrition_node, "Advice", "disease_name")
        treated_rela = Relationship(disease_node, "HAS_ADVICE", nutrition_node)
        graph_instance.create(treated_rela)

    if associated_disease and disease_node:
        if isinstance(associated_disease, str) and not '[' in associated_disease:
            if check_node_exists(graph_instance, to_lowercase(associated_disease)):
                return
            associated_disease_description = None
            associated_disease_category = None
            associated_disease_cause = None
            if not df_data[df_data['tên_bệnh'].str.lower() == associated_disease.lower()].empty:
                associated_row = df_data[df_data['tên_bệnh'].str.lower() == associated_disease.lower()].iloc[0]
                associated_disease_description = associated_row['mô_tả_bệnh']
                associated_disease_category = associated_row['loại_bệnh']
                associated_disease_cause = associated_row['nguyên_nhân']
            # Create embedding for associated disease
            associated_embedding = create_disease_embedding(associated_disease, associated_disease_description)
            associated_disease_node = Node("Disease", name=to_lowercase(associated_disease), description=to_lowercase(associated_disease_description), category=to_lowercase(associated_disease_category), cause=to_lowercase(associated_disease_cause), embedding=associated_embedding)
            graph_instance.merge(associated_disease_node, "Disease", "name")
            has_associated_rela = Relationship(disease_node, "ASSOCIATED_WITH", associated_disease_node)
            graph_instance.create(has_associated_rela)
            return

        try:
            associated_disease = associated_disease.replace("[", "").replace("]", "")  # Remove square brackets if present
            associated_disease = [item.strip() for item in associated_disease.split(',')]

            if isinstance(associated_disease, list):
                for associated_disease_name in associated_disease:
                    associated_disease_row = df_data[df_data["tên_bệnh"] == to_lowercase(associated_disease_name)]
                    if not associated_disease_row.empty:
                        associated_disease_info = associated_disease_row.iloc[0]
                        (
                            associated_disease_name, associated_disease_description, associated_disease_category,
                            _, associated_disease_cause, _, _, _, _, _, _, _, _, _, _, _, _, _
                        ) = associated_disease_info
                    else:
                        associated_disease_description = None
                        associated_disease_category = None
                        associated_disease_cause = None
                    if check_node_exists(graph_instance, to_lowercase(associated_disease_name)):
                        continue
                    # Create embedding for associated disease
                    associated_embedding = create_disease_embedding(associated_disease_name, associated_disease_description)
                    associated_disease_node = Node("Disease", name=to_lowercase(associated_disease_name), description=to_lowercase(associated_disease_description), category=to_lowercase(associated_disease_category), cause=to_lowercase(associated_disease_cause), embedding=associated_embedding)
                    graph_instance.merge(associated_disease_node, "Disease", "name")
                    has_associated_rela = Relationship(disease_node, "ASSOCIATED_WITH", associated_disease_node)
                    graph_instance.create(has_associated_rela)
        except Exception as e:
            print(f"Error processing associated disease: {e}")

if __name__ == "__main__": 
    graph = Graph(URI, auth=AUTH)
    
    clear_graph(graph)
    create_unique_constraints(graph)
    df_cn = pd.read_csv(r'data_translated.csv', encoding="utf-8")    
    num_workers = 8    
    # Process each row in parallel using ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(process_row, row, graph, df_cn) for index, row in df_cn.iterrows()]
        
        completed = 0
        for future in as_completed(futures):
            try:
                future.result()  # Retrieve and handle exceptions if any
                completed += 1
                if completed % 50 == 0:
                    print(f"Processed {completed}/{len(df_cn)} rows")
            except Exception as e:
                print(f"Error processing row: {e}")