import sys


class Graph:

    def __init__(self, nodes, init_graph):
        self.nodes = nodes
        self.graph = self.construct_graph(init_graph)

    def construct_graph(self, init_graph):
        graph = {}
        for node in self.nodes:
            graph[node] = {}
        graph.update(init_graph)

        for node, edges in graph.items():
            for adjacent_node, value in edges.items():
                # print(adjacent_node, value)
                if graph[adjacent_node].get(node, False) == False:
                    graph[adjacent_node][node] = value

        return graph


    def get_nodes(self):
        return self.nodes

    def get_value(self, node1, node2):
        return self.graph[node1][node2]

    def get_neighbours(self, node):
        return self.graph[node].keys()


def dijkstra(start_node, graph):
    unvisited_nodes = list(graph.get_nodes())

    shortest_path = {}
    previous_nodes = {}

    max_value = sys.maxsize
    for node in unvisited_nodes:
        shortest_path[node] = max_value
    shortest_path[start_node] = 0

    while unvisited_nodes:

        min_node = None
        for node in unvisited_nodes:
            if min_node == None:
                min_node = node
            elif shortest_path[node] < shortest_path[min_node]:
                min_node = node

        neighbours = graph.get_neighbours(min_node)
        for neighbour in neighbours:
            temp_value = shortest_path[min_node] + graph.get_value(min_node, neighbour)
            if temp_value < shortest_path[neighbour]:
                shortest_path[neighbour] = temp_value
                previous_nodes[neighbour] = min_node

        unvisited_nodes.remove(min_node)

    return shortest_path, previous_nodes

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
init_graph["Rome"]["Berlin"] = 2
init_graph["Rome"]["Athens"] = 2

graph = Graph(nodes, init_graph)
shortestpath, previous_nodes = dijkstra(start_node="Rome", graph = graph)

def print_result(start_node, end_node, shortest_path, previous_nodes):
    node = end_node
    path = []

    while node != start_node:
        print(node)
        path.append(node)
        node = previous_nodes[node]

    path.append(start_node)
    print("->".join(reversed(path)))

print_result("Reykjavik", "Belgrade", shortestpath, previous_nodes)

#%%

# importing "itertools" for iterator operations
import itertools

# initializing list
li = [2, 4, 6, 7, 8, 10, 20]

# storing list in iterator
iti = iter(li)


# using tee() to make a list of iterators
# makes list of 3 iterators having same values.
it = itertools.tee(iti, 3)
# print(list(it[1]))

# printing the values of iterators
print("The iterators are : ")
for i in range(0, 3):
    print(list(it[i]))

print(it)
print(list(it[0]))

#%%

marks = [65, 71, 68, 74, 61]

# convert list to iterator
iterator_marks = iter(marks)
print(iterator_marks)

# the next element is the first element
marks_1 = next(iterator_marks)
print(marks_1)

# find the next element which is the second element
marks_2 = next(iterator_marks)
print(marks_2)

# Output: 65
#         71
#%%
languages = ['Java', 'Python', 'JavaScript', "test"]
versions = [14, 3, 6, 6]

result = zip(languages, versions)
print(list(result))

# Output: [('Java', 14), ('Python', 3), ('JavaScript', 6)]
import numpy as np
t1 = np.random.rand(3, 1)
t2 = np.random.rand(3, 1)
t = zip(t1, t2)
print(t1, t2)
# print(t)
# print(list(t))

for key, value in t:
    print(key, value)














    # def construct_graph(self, init_graph):
    #     print("hello world")
    #     graph = {}
    #     for node in self.nodes:
    #         graph[node] = {}
    #     graph.update(init_graph)
    #     for node, edges in graph.items():
    #         # print(node, edges)
    #         for adjacent_node, value in edges.items():
    #             if graph[adjacent_node].get(node, False) == False:
    #                 graph[adjacent_node][node] = value
    #     return graph



# nodes = ["shanghai", "beijing", "xian"]
# init_graph = {}
# for node in nodes:
#     init_graph[node] = {}
#
# init_graph["beijing"]["shanghai"] = 1
# init_graph["beijing"]["xian"] = 2
# init_graph["xian"]["shanghai"] = 3
#
#
# print(init_graph)
# graph = Graph(nodes, init_graph)
# print(graph.graph)

