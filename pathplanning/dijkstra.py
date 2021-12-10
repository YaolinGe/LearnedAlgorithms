import sys

class Graph(object):

    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        make a symmetrical graph, from A to B, and from B to A
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}
        graph.update(init_graph)
        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value
        return graph

    def get_nodes(self):
        return self.nodes

    def get_outgoing_edges(self, node):
        # connections = list(self.graph[node].keys())
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        # print(connections)
        return connections

    def value(self, node1, node2):
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    shortest_path = {}
    previous_nodes = {}

    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    shortest_path[start_node] = 0

    while unvisited_nodes:
        print("unvisited_ndoes:  ", unvisited_nodes)

        current_min_node = None
        for node in unvisited_nodes:
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node
        print("Before ++++++++++++++")
        print("shortest_path: ", shortest_path)
        print("current_min_node: ", current_min_node)



        neighbours = graph.get_outgoing_edges(current_min_node)

        for neighbour in neighbours:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbour)
            if tentative_value < shortest_path[neighbour]:
                shortest_path[neighbour] = tentative_value
                previous_nodes[neighbour] = current_min_node
        print("After ++++++++++++++")
        print("shortest_path: ", shortest_path)
        print("current_min_node: ", current_min_node)


        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path

def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node
    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    path.append(start_node)
    print("We found the following best path with a value of {}. ".format(shortest_path[target_node]))
    print(" -> ".join(reversed(path)))

nodes = ["Reykjavik", "Oslo", "Moscow", "London", "Rome", "Berlin", "Belgrade", "Athens"]
init_graph = {}
for node in nodes:
    init_graph[node] = {}

init_graph["Reykjavik"]["Oslo"] = 5
init_graph["Reykjavik"]["London"] = 4
# init_graph["Oslo"]["Berlin"] = 1
# init_graph["Oslo"]["Moscow"] = 3
# init_graph["Moscow"]["Belgrade"] = 5
# init_graph["Moscow"]["Athens"] = 4
# init_graph["Rome"]["Berlin"] = 2
# init_graph["Rome"]["Athens"] = 2

print("Init graph: ", init_graph)
graph = Graph(nodes, init_graph)
print("Cons symmetric graph: ", graph.graph)
previous_nodes, shortest_path = dijkstra_algorithm(graph = graph, start_node="Reykjavik")
# print_result(previous_nodes, shortest_path, start_node="Reykjavik", target_node="Berlin")


#%% Implement dijkstra on my own
import numpy as np

graph = {1:{2:2}, 2:{1:2, 3:3}, 3:{2:3, 4:5, 5:1, 6:3, 8:1}, 4:{3:5, 5:2}, 5:{3:1, 4:2}, 6:{3:3, 7:4}, 7:{6:4}, 8:{3:1}}


class node:
    def __init__(self, label, neighbours):
        self.name = label
        self.neighbour = neighbours[0]
        self.range = neighbours[1]

class pathplanner:
    def __init__(self):
        print("Hello world")




def dijkstra(start, end, graph):
    print("hello, dijkstra")
    print("start: ", start)
    print("end: ", end)
    print("graph: ", graph)
    path = []
    counter = 0
    value = []
    for i in range(len(graph)):
        value.append(np.inf)
    print(value)
    ind_now = list(graph.keys()).index(start)
    print(ind_now)
    value[ind_now] = 0
    unvisited = list(graph.keys())
    print("unvisited: ", unvisited)
    unvisited = [i for i in unvisited if i != start]
    print("unvisited: ", unvisited)
    print("value: ", value)

    while True:
        print("counter: ", counter)
        print("unvisited: ", unvisited)
        ind_min = np.where(value == value)
        neighbours = list(graph[ind_now].keys())
        for neighbour in neighbours:
            if value[neighbour] > value[ind_now] + graph[ind_now][neighbour]:
                value[neighbour] = value[ind_now] + graph[ind_now][neighbour]
                ind_now = neighbour

        if counter > 10:
            break
        counter = counter + 1

if __name__ == "__main__":
    start = 1
    end = 4
    dijkstra(start, end, graph)


#%%

import sys


class Graph(object):
    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(nodes, init_graph)

    def construct_graph(self, nodes, init_graph):
        '''
        This method makes sure that the graph is symmetrical. In other words, if there's a path from node A to B with a value V, there needs to be a path from node B to node A with a value V.
        '''
        graph = {}
        for node in nodes:
            graph[node] = {}

        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph

    def get_nodes(self):
        "Returns the nodes of the graph."
        return self.nodes

    def get_outgoing_edges(self, node):
        "Returns the neighbors of a node."
        connections = []
        for out_node in self.nodes:
            if self.graph[node].get(out_node, False) != False:
                connections.append(out_node)
        return connections

    def value(self, node1, node2):
        "Returns the value of an edge between two nodes."
        return self.graph[node1][node2]


def dijkstra_algorithm(graph, start_node):
    unvisited_nodes = list(graph.get_nodes())

    # We'll use this dict to save the cost of visiting each node and update it as we move along the graph
    shortest_path = {}

    # We'll use this dict to save the shortest known path to a node found so far
    previous_nodes = {}

    # We'll use max_value to initialize the "infinity" value of the unvisited nodes
    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    # However, we initialize the starting node's value with 0
    shortest_path[start_node] = 0

    # The algorithm executes until we visit all nodes
    while unvisited_nodes:
        # The code block below finds the node with the lowest score
        current_min_node = None
        for node in unvisited_nodes:  # Iterate over the nodes
            if current_min_node == None:
                current_min_node = node
            elif shortest_path[node] < shortest_path[current_min_node]:
                current_min_node = node

        # The code block below retrieves the current node's neighbors and updates their distances
        neighbors = graph.get_outgoing_edges(current_min_node)
        for neighbor in neighbors:
            tentative_value = shortest_path[current_min_node] + graph.value(current_min_node, neighbor)
            if tentative_value < shortest_path[neighbor]:
                shortest_path[neighbor] = tentative_value
                # We also update the best path to the current node
                previous_nodes[neighbor] = current_min_node

        # After visiting its neighbors, we mark the node as "visited"
        unvisited_nodes.remove(current_min_node)

    return previous_nodes, shortest_path


def print_result(previous_nodes, shortest_path, start_node, target_node):
    path = []
    node = target_node

    while node != start_node:
        path.append(node)
        node = previous_nodes[node]

    # Add the start node manually
    path.append(start_node)

    print("We found the following best path with a value of {}.".format(shortest_path[target_node]))
    print(" -> ".join(reversed(path)))


nodes = ["Reykjavik", "Oslo", "Moscow", "London", "Rome", "Berlin", "Belgrade", "Athens"]

init_graph = {}
for node in nodes:
    init_graph[node] = {}

init_graph["Reykjavik"]["Oslo"] = 5
init_graph["Reykjavik"]["London"] = 4
init_graph["Oslo"]["Berlin"] = 1
init_graph["Oslo"]["Moscow"] = 3
init_graph["Moscow"]["Belgrade"] = 5
init_graph["Moscow"]["Athens"] = 4
init_graph["Athens"]["Belgrade"] = 1
init_graph["Rome"]["Berlin"] = 2
init_graph["Rome"]["Athens"] = 2

graph = Graph(nodes, init_graph)
previous_nodes, shortest_path = dijkstra_algorithm(graph=graph, start_node="London")

print_result(previous_nodes, shortest_path, start_node="London", target_node="Belgrade")




