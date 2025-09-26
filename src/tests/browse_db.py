#browse_db.py
import sys
import os
import chromadb
import pprint

# 절대경로 설정
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from config import Config

def browse_vector_db(limit: int = 10):
    """
    ChromaDB에 저장된 데이터를 직접 조회하는 코드.
    
    Args:
        limit (int): 조회할 데이터의 최대 개수.
    """
    print("--- ChromaDB 데이터 탐색을 시작합니다 ---")
    
    try:
        # DB 클라이언트에 연결
        client = chromadb.PersistentClient(path=Config.VECTOR_DB_PATH)
        collection_name = Config.RFP_COLLECTION
        print(f"'{collection_name}' 컬렉션에 연결합니다...")
        collection = client.get_collection(name=collection_name)
        
        # collection.get() 메서드를 사용하여 데이터를 조회
        # documents, metadatas, ids 등을 모두 가져올 수 있다.
        data = collection.get(
            limit=limit,
            include=["metadatas", "documents"]
        )
        
        print(f"\n--- 총 {len(data['ids'])}개의 데이터 조회 결과 ---")
        
        # 예쁘게 출력하기 위해 pprint 사용
        pp = pprint.PrettyPrinter(indent=4)
        
        # 각 문서를 순회하며 상세 정보 출력
        for i in range(len(data['ids'])):
            doc_id = data['ids'][i]
            metadata = data['metadatas'][i]
            document_content = data['documents'][i]
            
            print(f"\n[{i+1}] ------------------------------------")
            print(f"  - ID: {doc_id}")
            print( "  - METADATA:")
            pp.pprint(metadata)
            print( "  - DOCUMENT (첫 100글자):")
            print(f"    '{document_content[:100]}...'")
            print("-----------------------------------------")

    except Exception as e:
        print(f"\n[오류] 데이터베이스 조회 중 오류가 발생했습니다: {e}")

if __name__ == "__main__":
    # 여기서 숫자를 바꾸면 조회하는 데이터 개수를 조절할 수 있습니다.
    browse_vector_db(limit=50)