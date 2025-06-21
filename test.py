import os 
import json 
if __name__ == "__main__":
    file_dir = os.getcwd()
    file_path = os.path.join(file_dir, 'data', 'jsonl_vimed', 'medicine1.jsonl')
    with open(file_path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f] 
    print(data[0])