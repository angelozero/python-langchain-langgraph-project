# import operator

from rich import print
from typing import Annotated, TypedDict
from langgraph.graph import StateGraph, add_messages

# def reducer(a: list[str], b: list[str]) -> list[str]:
#     return a + b


# 1 = Definindo um Estado
class StateSimple(TypedDict):

    # lista para armazenamento de todos os nos
    # nodes_path: Annotated[list[str], reducer]
    # nodes_path: Annotated[list[str], lambda a, b: a + b]
    # nodes_path: Annotated[list[str], operator.add]
    nodes_path: Annotated[list[str], add_messages]


# 3 - Definir Nodes (nos)
def node_a(state: StateSimple) -> StateSimple:
    print(f"\n>> node A simple state: {state}")
    return {"nodes_path": ["A"]}


def node_b(state: StateSimple) -> StateSimple:
    print(f">> node B simple state: {state}")
    return {"nodes_path": ["B"]}


# 4 - Definindo o builder do Grafo
builder = StateGraph(StateSimple)

# 5 - Adicionando os nos aoo builder
builder.add_node("A", node_a)
builder.add_node("B", node_b)

# 6 - Conectando as arestas entre os nodes
builder.add_edge("__start__", "A")
builder.add_edge("A", "B")
builder.add_edge("B", "__end__")

# 7 - Compilar o grafo
graph = builder.compile()

# 8 - Criando a imagem do grapho
# graph.get_graph().draw_mermaid_png(output_file_path='graph.png')

# 9 - Recuperando o resultado
response = graph.invoke({"nodes_path": []})

# 10 - Resultado
print()
print(response)
print()
