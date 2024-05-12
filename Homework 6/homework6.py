import networkx as nx
import matplotlib.pyplot as plt
from collections import deque
import heapq

def create_graph():
    G = nx.Graph()
    nodes = ["Центральна станція", "Північний автовокзал", "Південний залізничний вокзал", "Східний ринок", "Західний парк", "Мерія"]
    G.add_nodes_from(nodes)
    edges = [
        ("Центральна станція", "Північний автовокзал", 5),
        ("Центральна станція", "Південний залізничний вокзал", 8),
        ("Північний автовокзал", "Східний ринок", 4),
        ("Східний ринок", "Південний залізничний вокзал", 7),
        ("Південний залізничний вокзал", "Західний парк", 10),
        ("Західний парк", "Мерія", 3),
        ("Мерія", "Центральна станція", 6),
        ("Східний ринок", "Західний парк", 5)
    ]
    G.add_weighted_edges_from(edges)
    return G

def visualize_graph(G):
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', node_size=2000, font_size=10, font_color='black')
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
    plt.title("Транспортна мережа Броварів")
    plt.show()

def bfs_path(graph, start, goal):
    queue = deque([[start]])
    visited = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node in visited:
            continue
        visited.add(node)
        if node == goal:
            return path
        for neighbor in graph.neighbors(node):
            new_path = list(path)
            new_path.append(neighbor)
            queue.append(new_path)
    return None

def dfs_path(graph, start, goal, path=None, visited=None):
    if visited is None:
        visited = set()
    if path is None:
        path = [start]
    if start == goal:
        return path
    visited.add(start)
    for neighbor in graph.neighbors(start):
        if neighbor not in visited:
            new_path = list(path)
            new_path.append(neighbor)
            result = dfs_path(graph, neighbor, goal, new_path, visited)
            if result:
                return result
    return None

def dijkstra(graph, source):
    min_heap = [] 
    heapq.heappush(min_heap, (0, source)) 
    shortest_paths = {vertex: float('infinity') for vertex in graph.nodes}
    shortest_paths[source] = 0
    predecessor = {vertex: None for vertex in graph.nodes}
    
    while min_heap:
        current_distance, current_vertex = heapq.heappop(min_heap)
        if current_distance > shortest_paths[current_vertex]:
            continue
        for neighbor, data in graph[current_vertex].items():
            weight = data['weight']
            distance = current_distance + weight
            if distance < shortest_paths[neighbor]:
                shortest_paths[neighbor] = distance
                predecessor[neighbor] = current_vertex
                heapq.heappush(min_heap, (distance, neighbor))
    return shortest_paths, predecessor

if __name__ == "__main__":
    G_fictional = create_graph()
    visualize_graph(G_fictional)
    start_point = "Центральна станція"
    end_point = "Мерія"
    bfs_result = bfs_path(G_fictional, start_point, end_point)
    dfs_result = dfs_path(G_fictional, start_point, end_point)
    shortest_paths_info, predecessors = dijkstra(G_fictional, start_point)
    print("Результат BFS:", bfs_result)
    print("Результат DFS:", dfs_result)
    print("Найкоротші шляхи за алгоритмом Дейкстри:", shortest_paths_info)

