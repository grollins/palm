import numpy
import networkx as nx

def make_graph_from_route_collection(route_collection):
    G = DiGraph()
    for r_id, r in route_collection.iter_routes():
        start_id = r['start_state']
        end_id = r['end_state']
        G.add_edge(start_id, end_id, route_id=r_id)
    return G


class DiGraph(nx.DiGraph):
    """docstring for DiGraph"""
    def __init__(self):
        super(DiGraph, self).__init__()

    def iter_neighbors(self, start_id):
        for n in self.neighbors_iter(start_id):
            yield n

    def iter_successors(self, start_id, depth):
        visited = set()
        visited_routes = []
        visited.add(start_id)
        stack = [ (start_id, iter(self[start_id])) ]
        while stack:
            parent, children = stack[-1]
            if len(stack) <= depth:
                try:
                    child = next(children)
                    route_id = self[parent][child]['route_id']
                    if route_id in visited_routes:
                        pass
                    else:
                        yield route_id
                        visited_routes.append(route_id)
                    if child not in visited:
                        visited.add(child)
                        stack.append( (child, iter(self[child])) )
                except StopIteration:
                    stack.pop()
            else:
                stack.pop()
