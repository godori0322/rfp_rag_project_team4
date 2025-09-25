from typing import Dict, List, Optional
from langchain_core.messages import BaseMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.outputs import LLMResult
from langchain_core.runnables import RunnableConfig
import networkx as nx
import matplotlib.pyplot as plt

class RAGCallbackHandler(BaseCallbackHandler):
    def __init__(self):
        self.graph = nx.DiGraph()
        
    def on_retriever_start(self, query: str, **kwargs):
        self.graph.add_node("query", label=query, type="query")
        
    def on_retriever_end(self, documents: List[str], **kwargs):
        for i, doc in enumerate(documents):
            doc_id = f"doc_{i}"
            self.graph.add_node(doc_id, label=doc[:100] + "...", type="document")
            self.graph.add_edge("query", doc_id, label="retrieves")
            
    def on_llm_start(self, context: str, **kwargs):
        self.graph.add_node("context", label=context, type="context")
        
    def on_llm_end(self, response: LLMResult, **kwargs):
        self.graph.add_node("response", label=str(response), type="response")
        self.graph.add_edge("context", "response", label="generates")
        
    def visualize(self, output_path: Optional[str] = None):
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(self.graph)
        
        node_colors = {
            'query': 'lightblue',
            'document': 'lightgreen',
            'context': 'lightyellow',
            'response': 'lightcoral'
        }
        
        for node_type in node_colors:
            nodes = [n for n, attr in self.graph.nodes(data=True) 
                    if attr.get('type') == node_type]
            if nodes:
                nx.draw_networkx_nodes(self.graph, pos, 
                                     nodelist=nodes,
                                     node_color=node_colors[node_type],
                                     node_size=2000)
        
        nx.draw_networkx_edges(self.graph, pos)
        nx.draw_networkx_labels(self.graph, pos, 
                              labels=nx.get_node_attributes(self.graph, 'label'))
        
        if output_path:
            plt.savefig(output_path)
        plt.show()
        
    def get_graph(self):
        return self.graph