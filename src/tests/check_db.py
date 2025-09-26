#check_db.py
import chromadb
from config import Config
import sys
import os

# 절대경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def check_vector_db_status():
    """
    ChromaDB의 상태를 확인하고 지정된 컬렉션의 데이터 존재 여부를 검사
    """
    print("--- ChromaDB 상태 검사를 시작합니다 ---")
    
    try:
        print(f"DB 경로: '{Config.VECTOR_DB_PATH}'")
        client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        
        # 모든 컬렉션 목록을 가져온다.
        collections = client.list_collections()
        if not collections:
            print("\n[오류] DB에 컬렉션이 전혀 존재하지 않습니다.")
            print("데이터 임베딩 스크립트(main_vs.py)가 정상적으로 실행되었는지 확인해주세요.")
            return

        print(f"\n존재하는 컬렉션 목록: {[c.name for c in collections]}")
        
        # 설정 파일에 지정된 컬렉션을 가져온다.
        collection_name = Config.RFP_COLLECTION
        print(f"확인하려는 컬렉션: '{collection_name}'")
        
        collection = client.get_collection(name=collection_name)
        
        # 컬렉션의 데이터 수를 확인한다.
        count = collection.count()
        
        if count == 0:
            print(f"\n[오류] '{collection_name}' 컬렉션이 비어있습니다.")
            print("데이터 임베딩 과정에서 오류가 발생했을 수 있습니다.")
        else:
            print(f"\n[성공] '{collection_name}' 컬렉션에서 {count}개의 문서를 찾았습니다.")
            
            # 샘플 데이터를 확인한다.
            print("\n--- 데이터 샘플 (첫 5개) ---")
            sample_data = collection.peek(limit=200)
            for i, metadata in enumerate(sample_data.get('metadatas', [])):
                print(f"{i+1}. project_title: {metadata.get('project_title', '제목 없음')}")

    except ValueError as e:
        if "does not exist" in str(e):
            print(f"\n[오류] '{collection_name}' 컬렉션을 찾을 수 없습니다.")
            print("config.py의 'RFP_COLLECTION' 이름이 올바른지, 또는 데이터 임베딩 시 사용한 이름과 일치하는지 확인해주세요.")
        else:
            print(f"\n[알 수 없는 오류] {e}")
    except Exception as e:
        print(f"\n[오류] DB 연결 또는 데이터 조회 중 예외가 발생했습니다: {e}")
        print("DB 경로가 올바른지, 파일 권한에 문제가 없는지 확인해주세요.")

if __name__ == "__main__":
    check_vector_db_status()