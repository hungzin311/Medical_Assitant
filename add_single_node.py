from py2neo import Graph, Node, Relationship
import pandas as pd
import ast
from pathlib import Path
import sys
from utils.proxy_setting import set_proxy
# Add parent directory to path to import utils
sys.path.append(str(Path(__file__).parent))
from utils.llm_config import get_embedding

# Neo4j connection settings
URI = "neo4j://127.0.0.1:7687"
AUTH = ("neo4j", "Hung31102004")

# Initialize embedding model
set_proxy()

embedding_model = get_embedding()

def to_lowercase(text):
    """Convert text to lowercase if it's a string."""
    if isinstance(text, str):
        return text.lower()
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

def create_disease_embedding(disease_name, disease_description):
    try:
        combined_text = f"{disease_name} {disease_description}"
        embedding = embedding_model.aembed_query(combined_text)
        return embedding
    except Exception as e:
        print(f"❌ Lỗi tạo embedding: {e}")
        print("⚠️  Tiếp tục mà không có embedding...")
        return []

def add_single_node_to_kg(row_index, csv_file_path="data/csv/data_translated.csv"):
    """
    Thêm một node duy nhất vào Knowledge Graph từ một dòng cụ thể trong CSV.
    
    Args:
        row_index (int): Index của dòng trong CSV (0-based, không tính header)
        csv_file_path (str): Đường dẫn đến file CSV
    """
    
    # Connect to Neo4j
    try:
        graph = Graph(URI, auth=AUTH)
        print("✅ Kết nối Neo4j thành công!")
    except Exception as e:
        print(f"❌ Lỗi kết nối Neo4j: {e}")
        return False
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_file_path, encoding="utf-8")
        print(f"✅ Đọc file CSV thành công! Tổng số dòng: {len(df)}")
    except Exception as e:
        print(f"❌ Lỗi đọc file CSV: {e}")
        return False
    
    # Check if row_index is valid
    if row_index < 0 or row_index >= len(df):
        print(f"❌ Index {row_index} không hợp lệ. File có {len(df)} dòng (0-{len(df)-1})")
        return False
    
    # Get the specific row
    row = df.iloc[row_index]
    
    print(f"\n📋 Thông tin dòng {row_index}:")
    print(f"Tên bệnh: {row['tên_bệnh']}")
    print(f"Mô tả: {row['mô_tả_bệnh'][:100]}...")
    
    try:
        # Extract data from row
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
        print(f"❌ Thiếu cột: {e}")
        return False

    disease_node = None

    # Create Disease node - Relaxed conditions
    if disease_name and disease_description:
        print("🔄 Tạo Disease node...")
        print(f"📋 Thông tin bệnh:")
        
        # Create embedding for disease
        disease_embedding = create_disease_embedding(disease_name, disease_symptom)
        
        # Create disease node with available data - Match create_kg.py structure
        disease_node = Node("Disease", 
                           name=to_lowercase(disease_name),
                           description=to_lowercase(disease_description),
                           category=to_lowercase(disease_category) if disease_category else "không rõ",
                           cause=to_lowercase(disease_cause) if disease_cause else "không rõ",
                           embedding=disease_embedding)
        
        # Check if node already exists
        existing_query = """
        MATCH (d:Disease {name: $name})
        RETURN COUNT(d) as count
        """
        result = graph.run(existing_query, name=to_lowercase(disease_name)).data()
        
        if result and result[0]['count'] > 0:
            print(f"⚠️  Disease node '{disease_name}' đã tồn tại!")
            overwrite = input("Bạn có muốn ghi đè? (y/n): ").lower().strip()
            if overwrite != 'y':
                print("❌ Hủy bỏ thao tác.")
                return False
        
        graph.merge(disease_node, "Disease", "name")
        print("✅ Tạo Disease node thành công!")
    else:
        print("❌ Thiếu thông tin cơ bản (tên bệnh hoặc mô tả)")
        return False

    # Create Treatment node and relationship - Match create_kg.py structure
    if cure_method and cure_department and cure_probability and disease_node:
        print("🔄 Tạo Treatment node...")
        
        # Use Cypher query like in create_kg.py
        treatment_query = """
        MERGE (t:Treatment {disease_name: $disease_name})
        ON CREATE SET t.method = $method, t.department = $department, t.success_rate = $success_rate
        ON MATCH SET t.method = $method, t.department = $department, t.success_rate = $success_rate
        WITH t
        MATCH (d:Disease {name: $disease_name})
        MERGE (d)-[:TREATED_BY]->(t)
        """
        graph.run(treatment_query,
                 disease_name=to_lowercase(disease_name),
                 method=to_lowercase(cure_method),
                 department=to_lowercase(cure_department),
                 success_rate=to_lowercase(cure_probability))
        print("✅ Tạo Treatment node và relationship thành công!")

    # Create Medication node and relationship - Relaxed conditions
    if (drug_recommend or drug_common or drug_detail) and disease_node:
        print("🔄 Tạo Medication node...")
        
        recommended_drugs_list = parse_list_field(drug_recommend) if drug_recommend else []
        common_drugs_list = parse_list_field(drug_common) if drug_common else []
        
        medication_node = Node("Medication", 
                              disease_name=to_lowercase(disease_name), 
                              common_drugs=common_drugs_list, 
                              drug_info=to_lowercase(drug_detail) if drug_detail else "không có thông tin", 
                              recommended_drugs=recommended_drugs_list)
        
        graph.merge(medication_node, "Medication", "disease_name")
        prescribed_rela = Relationship(disease_node, "PRESCRIBED", medication_node)
        graph.create(prescribed_rela)
        print("✅ Tạo Medication node và relationship thành công!")

    # Create Advice node and relationship - Relaxed conditions
    if (nutrition_do_eat or nutrition_not_eat or nutrition_recommend_meal or disease_prevention) and disease_node:
        print("🔄 Tạo Advice node...")
        
        foods_to_eat_list = parse_list_field(nutrition_do_eat) if nutrition_do_eat else []
        foods_to_avoid_list = parse_list_field(nutrition_not_eat) if nutrition_not_eat else []
        recommended_meals_list = parse_list_field(nutrition_recommend_meal) if nutrition_recommend_meal else []
        
        nutrition_node = Node("Advice", 
                             disease_name=to_lowercase(disease_name), 
                             foods_to_eat=foods_to_eat_list, 
                             recommended_meals=recommended_meals_list, 
                             foods_to_avoid=foods_to_avoid_list, 
                             prevention=to_lowercase(disease_prevention) if disease_prevention else "không có thông tin")
        
        graph.merge(nutrition_node, "Advice", "disease_name")
        treated_rela = Relationship(disease_node, "HAS_ADVICE", nutrition_node)
        graph.create(treated_rela)
        print("✅ Tạo Advice node và relationship thành công!")

    # Create Symptom node and relationship - Following create_kg.py logic
    if disease_symptom and check_method and people_easy_get and disease_node:
        print("🔄 Tạo Symptom node...")
        
        symptoms_list = parse_list_field(disease_symptom)
        diagnosis_list = parse_list_field(check_method)
        
        symptom_node = Node("Symptom", 
                           disease_name=to_lowercase(disease_name), 
                           symptoms=symptoms_list, 
                           diagnosis=diagnosis_list, 
                           risk_group=to_lowercase(people_easy_get))
        
        graph.merge(symptom_node, "Symptom", "disease_name")
        has_symptom_rela = Relationship(disease_node, "HAS_SYMPTOM", symptom_node)
        graph.create(has_symptom_rela)
        print("✅ Tạo Symptom node và relationship thành công!")
    elif disease_symptom and disease_node:
        # Relaxed condition - create symptom node with available data
        print("🔄 Tạo Symptom node (điều kiện nới lỏng)...")
        
        symptoms_list = parse_list_field(disease_symptom)
        diagnosis_list = parse_list_field(check_method) if check_method else []
        
        symptom_node = Node("Symptom", 
                           disease_name=to_lowercase(disease_name), 
                           symptoms=symptoms_list, 
                           diagnosis=diagnosis_list, 
                           risk_group=to_lowercase(people_easy_get) if people_easy_get else "không rõ")
        
        graph.merge(symptom_node, "Symptom", "disease_name")
        has_symptom_rela = Relationship(disease_node, "HAS_SYMPTOM", symptom_node)
        graph.create(has_symptom_rela)
        print("✅ Tạo Symptom node và relationship thành công!")

    # Handle associated diseases
    if associated_disease and disease_node and associated_disease != "Không có thông tin":
        print("🔄 Xử lý bệnh đi kèm...")
        
        try:
            associated_diseases_list = parse_list_field(associated_disease)
            
            for assoc_disease in associated_diseases_list:
                if assoc_disease and assoc_disease.strip():
                    assoc_disease_clean = to_lowercase(assoc_disease.strip())
                    
                    # Check if associated disease node exists
                    check_query = """
                    MATCH (d:Disease {name: $name})
                    RETURN d
                    """
                    existing_assoc = graph.run(check_query, name=assoc_disease_clean).data()
                    
                    if existing_assoc:
                        # Create relationship with existing node
                        create_rel_query = """
                        MATCH (d1:Disease {name: $disease1}), (d2:Disease {name: $disease2})
                        MERGE (d1)-[:ASSOCIATED_WITH]->(d2)
                        """
                        graph.run(create_rel_query, 
                                 disease1=to_lowercase(disease_name), 
                                 disease2=assoc_disease_clean)
                        print(f"✅ Tạo relationship với bệnh đi kèm: {assoc_disease}")
                    else:
                        print(f"⚠️  Bệnh đi kèm '{assoc_disease}' chưa tồn tại trong database")
                        
        except Exception as e:
            print(f"❌ Lỗi xử lý bệnh đi kèm: {e}")

    print(f"\n🎉 Hoàn thành thêm node cho bệnh: {disease_name}")
    return True

def main():
    """Main function để chạy script."""
    print("=" * 60)
    print("🏥 SCRIPT THÊM NODE DUY NHẤT VÀO KNOWLEDGE GRAPH")
    print("=" * 60)
    
    # Get row index from user
    try:
        row_index = int(input("Nhập index của dòng muốn thêm (0-based): "))
    except ValueError:
        print("❌ Vui lòng nhập một số nguyên hợp lệ!")
        return
    
    # Confirm action
    confirm = input(f"Bạn có chắc muốn thêm dòng {row_index} vào Knowledge Graph? (y/n): ").lower().strip()
    if confirm != 'y':
        print("❌ Hủy bỏ thao tác.")
        return
    
    # Add the node
    success = add_single_node_to_kg(row_index)
    
    if success:
        print("\n✅ Thêm node thành công!")
    else:
        print("\n❌ Có lỗi xảy ra khi thêm node!")

if __name__ == "__main__":
    main()
