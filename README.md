# rfp_rag_project_team4

## Baseline 코드입니다. 

## 2025/09/17

파일 전처리하는 preprocessing/extractor.py 미완성 상태

rag/baseline_rag.py chain으로 연결 안된 상태

vectorstore/build_index.py, query_index.py 에서 langchain 관련 오류 발생 가능(Linux 환경으로 변경 후 최대한 빠르게 수정하겠습니다. )

OpenAI API Key는 .env에 저장

config.py에서 경로 및 하이퍼파라미터 일괄적으로 관리


### 환경설정 ###
```
python3 -m venv venv
source .venv/bin/activate
pip install -r requirements.txt
pip freeze > requirements.txt
export OPENAI_API_KEY='your_openai_api_key'
python src/main.py --query "안녕하세요"
```