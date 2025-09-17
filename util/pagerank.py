import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix

def build_graph(anchor):
    G = nx.Graph()
    sparse_anchor = coo_matrix(anchor)
    
    for user, item in zip(sparse_anchor.row, sparse_anchor.col):
        user_node = f'user_{user}'
        item_node = f'item_{item}'
        G.add_node(user_node, bipartite=0)
        G.add_node(item_node, bipartite=1)
        G.add_edge(user_node, item_node)
    
    # print(G.edges)
    return G
    
def PageRank(G):
    pagerank = nx.pagerank(G, alpha=0.95,  max_iter=1000)
    item_pagerank = {node: rank for node, rank in pagerank.items() if node.startswith('item')}
    return item_pagerank

# def PageRank(G, alpha=0.85, max_iter=10, tol=1.0e-6):
#     pagerank = {node: 1 for node in G.nodes()}
    
#     out_degree_user = {f'user_{user}': sum(1 for neighbor in G.neighbors(f'user_{user}') if neighbor.startswith('item')) for user in range(num_users)}
    
#     for _ in range(max_iter):
#         diff = 0 
#         pagerank_new = pagerank.copy()
        
#         for node in G.nodes():
#             if node.startswith('item'):
#                 rank_sum = sum(pagerank[neighbor] / out_degree_user[neighbor] for neighbor in G.neighbors(node))
#                 pagerank_new[node] = (1 - alpha) + alpha * rank_sum
#                 diff += abs(pagerank[node] - pagerank_new[node])
        
#         pagerank = pagerank_new
        
#         if diff < tol:
#             break
    
#     return pagerank

def normalize(data):
    item_pr_list = [value for key,value in data.items()]
    min_val = np.min(item_pr_list)
    max_val = np.max(item_pr_list)
    normalized_data = (item_pr_list - min_val) / (max_val - min_val)
    mapped_data = 1 - (1 - normalized_data) * 0.9
    for key, value in zip(data.keys(), mapped_data):
        data[key] = value
    new_data = {}
    for key, value in data.items():
        new_key = int(key.split('_')[1])
        new_data[new_key] = value
    return new_data
