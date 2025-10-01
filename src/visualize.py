# visualize.py
import graphviz
from chain_router import ChainRouter

def visualize_chain_router(router: ChainRouter, output_file="chain_router", view=True):
    """
    ChainRouter 구조를 Graphviz로 시각화
    """
    dot = graphviz.Digraph(comment="ChainRouter Visualization", format="png")

    # --- 주요 노드 추가 ---
    dot.node("input", "User Input")
    dot.node("rephrasing", "Rephrasing Chain")
    dot.node("deterministic_router", "Deterministic Router")
    dot.node("llm_route_chain", "LLM Route Chain")
    dot.node("routing_branch", "Routing Branch")

    dot.node("metadata_search_chain", "Metadata Search Chain")
    dot.node("summarization_chain", "Summarization Chain")
    dot.node("comparison_chain", "Comparison Chain")
    dot.node("recommendation_chain", "Recommendation Chain")
    dot.node("default_qa_chain", "Default QA Chain")

    # --- 데이터 흐름 엣지 ---
    dot.edge("input", "rephrasing", label="history")
    dot.edge("input", "deterministic_router", label="input")
    
    dot.edge("deterministic_router", "routing_branch", label="classification")
    dot.edge("rephrasing", "llm_route_chain", label="rephrased_question")
    dot.edge("llm_route_chain", "routing_branch", label="llm_classification")

    # --- 라우팅 브랜치 연결 ---
    dot.edge("routing_branch", "metadata_search_chain", label="metadata_search")
    dot.edge("routing_branch", "summarization_chain", label="summarization")
    dot.edge("routing_branch", "comparison_chain", label="comparison")
    dot.edge("routing_branch", "recommendation_chain", label="recommendation")
    dot.edge("routing_branch", "default_qa_chain", label="default_qa")

    # --- 파일 생성 및 시각화 ---
    dot.render(output_file, view=False)
    print(f"✅ ChainRouter 구조 시각화 완료: {output_file}.png")

if __name__ == "__main__":
    # 예시: ChainRouter 인스턴스를 만들어서 시각화
    from chain_router import ChainRouter
    # 임의 LLM, retriever 등은 None 처리 (실제 시각화에는 영향 없음)
    router = ChainRouter(llm=None, retriever=None, self_query_retriever=None, vectorstore=None, tracer=None)
    visualize_chain_router(router)
