import numpy as np
import pandas as pd
import itertools
from itertools import *
import random


### DAG-RELATED HELP FUNCTIONS ###

def generate_dags(vertices, edges):
    
    # generate every DAG with 'vertices' number of vertices and 'edges' number of edges
    
    def generate_dags_iter(vertices, edges, base):
        if edges == 0:
            graph = base.copy()
            graph[graph == -1] = 0
            return [graph]
        ans = []
        for index, value in np.ndenumerate(base):
            if value == -1:
                graph = base.copy()
                graph[index] = 1
                if is_dag(graph):
                    ans += generate_dags_iter(vertices, edges-1, graph)
        return ans 
    
    def remove_duplicates(dags):
        if dags == []:
            return []
        head, *tail = dags
        tail = remove_duplicates(tail)
        if any([duplicate(head, dag) for dag in tail]):
            return tail
        else:
            return [head] + tail
        
    def duplicate(graph1, graph2):
        for i, x in np.ndenumerate(graph1):
            if graph2[i] != x:
                return False
        return True
    
    ans = generate_dags_iter(vertices, edges, empty_dag(vertices))
    ans = remove_duplicates(ans)
    
    return ans

def is_dag(graph):
    
    # returns True if 'graph' is a DAG (directed acyclic graph)
    
    path = set()
    def visit(vertex):
        path.add(vertex)
        for child in children(graph,vertex):
            if child in path or visit(child):
                return True
        path.remove(vertex)
        return False
    return not any(visit(v) for v in graph[0])

def empty_dag(vertices):
    
    # returns a dag with all vertices unknown
    
    dag = np.full((vertices,vertices),-1,int) # -1 is used to mean "unknown"
    np.fill_diagonal(dag,0)
    return dag


### GRAPH-RELATED HELP FUNCTIONS ###

def children(graph,vertex):
    
    # returns the children of vertex in graph
    
    ans = []
    for i in range(len(graph[0])):
        if graph[vertex][i] == 1:
            ans.append(i)
    return ans

def parents(graph,vertex):
    
    # returns the parents of vertex in graph
    
    ans = []
    for i in range(len(graph[0])):
        if graph[i][vertex] == 1:
            ans.append(i)
    return ans

def path(graph, start, goals):
    
    # returns True if 'graph' has a path from 'start' to some vertex in 'goals'
    
    reachable = set()
    frontier = [start]
    
    while frontier != []:
        v = frontier.pop()
        reachable.add(v)
        for i in children(graph, v):
            if i not in reachable:
                frontier.append(i)
                
    if intersection(goals, list(reachable)) != []:
        return True
    else:
        return False


### MISC HELP FUNCTIONS ###

def intersection(lst1, lst2): 
    
    # returns the intersection between two lists
    
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3

def powerset(iterable):
    
    # returns the powerset of 'iterable'
    # eg [1,2,3] => [(,),(1,),(2,),(3,),(1,2),(1,3),(2,3),(1,2,3)]
    
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


### CAUSATION-RELATED HELP FUNCTIONS ###

def d_connected(x0, x1, graph, condition_set):
    
    # compute whether 'x0' and 'x1' are d-connected in 'graph' conditional on 'condition_set'
    # see https://arxiv.org/pdf/1305.5506.pdf
    # http://bayes.cs.ucla.edu/BOOK-2K/d-sep.html
    
    unblocked = []
    for v in graph[0]:
        if path(graph,v,condition_set):
            unblocked.append(v)
    
    reachable = set()
    frontier = [x0]
    frontier_collision_warning = []
    
    while not (frontier == [] and frontier_collision_warning == []):

        if frontier != []:
            v = frontier.pop()
            reachable.add(v)
            for i in children(graph,v):
                if i not in condition_set and i not in reachable:
                    if i in unblocked:
                        frontier.append(i)
                    else:
                        frontier_collision_warning.append(i)
            for i in parents(graph,v):
                if i not in condition_set and i not in reachable:
                    frontier.append(i)            
        else:
            v = frontier_collision_warning.pop()
            reachable.add(v)
            for i in children(graph,v):
                if i not in condition_set and i not in reachable:
                    if i in unblocked:
                        frontier.append(i)
                    else:
                        frontier_collision_warning.append(i)
                       
    return x1 in reachable

def compatible(graph, data):
    
    # returns True if the data is compatible with the causal structure described by the graph
    
    features = list(range(len(data.columns)-1))
    for condition_set in powerset(features):
        feature_set = []
        for x in features:
            if x not in condition_set:
                feature_set.append(x)
        for x0, x1 in itertools.combinations(feature_set,2):
            if d_connected(x0, x1, graph, condition_set) == independent(x0, x1, data, condition_set):
                #if d_connected(x0, x1, graph, condition_set):
                #    print('\"{}\" and \"{}\" are d-connected but independent under {}'
                #          .format(data.columns[x0],data.columns[x1],condition_set))
                #else:
                #    print('\"{}\" and \"{}\" are dependent but not d-connected under {}'
                #          .format(data.columns[x0],data.columns[x1],condition_set))
                return False
    return True


### PROBABILITY-RELATED HELP FUNCTIONS ###

def prob(var_name, var_val, data, con={}):
    
    # P(var_name=var_val|con) in data
    
    if len(con) == 0:
        #computes P = (var_name = var_val)
        h = data.loc[data[var_name] == var_val].sum(axis = 0, skipna = True)['instances']
        p = h/data.sum(axis=0,skipna=True)['instances'] 
        
        #data.loc[data[var_name] == var_val]
        #len(data.index)
        
    else:
        #computes P = (var_name = var_val / conditions)
        #find index of rows that meets each condition
        common = []
        for key, value in con.items():
            dfnew = data.loc[data[key].isin([value])]
            common.append(dfnew.index.values)
            
        #find index of rows that meet all of the conditions
        index = common[0]
        for ind in common[1:]:
            index = np.intersect1d(index, ind)
            
        #find prob of var == var_val
        data = data.iloc[index]
        #if len(data.index) == 0:
        #    print(con)
        #    raise ValueError('sample to small, zero division')
            
        h = data.loc[data[var_name] == var_val].sum(axis = 0, skipna = True)['instances']
        p = h/data.sum(axis=0,skipna=True)['instances'] 
            
        # h = data.loc[data[var_name] == var_val]
        # p = len(h.index)/len(data.index)
        
    return p

def bigen(n):
    x = []
    for _ in range(n):
        x.append(random.randint(0,1))
    return x

def independent(x0, x1, data, con):
    
    # returns True if x0⊥x1|con in data

    x0 = data.columns[x0]
    x1 = data.columns[x1]

    for set_to_1 in powerset(con):
        con = {}
        for c in con:
            if c in con:
                con[data.columns[c]] = 1
            else:
                con[data.columns[c]] = 0

        con[x1]=0
        p1 = prob(x0, 1, data, con)
        con[x1]=1
        p2 = prob(x0, 1, data, con)
        
        if abs(p1-p2) > 0.01:
            return False
    return True


### MAIN ###

def generate_causal_models(data):
    
    # generates a set of causal models that fit the data
    # it assumes that the causal structure is likely to be simple, and returns the set of all causal graphs
    # that have as many edges as the smallest causal graph that fits the data
    # the models are provided as a list of causal graphs in the form of adjacency matrices
    
    vertices = len(data.columns)-1
    max_edges = int(0.5*vertices*(vertices-1))
    
    for edges in range(max_edges+1):
        
        print('edges: {}'.format(edges))
        
        graphs = generate_dags(vertices, edges)
        graphs = list(filter(lambda graph: compatible(graph,data), graphs))
        if graphs != []:
            return graphs
    return []


### EXAMPLE ###

# example data taken from: https://www.lesswrong.com/posts/hzuSDMx7pd2uxFc5w/causal-diagrams-and-causal-models 

data = pd.DataFrame(np.array([[1, 1, 1, 1119], 
                              [1, 1, 0, 16104],
                              [1, 0, 1, 11121],
                              [1, 0, 0, 60032],
                              [0, 1, 1, 18102],
                              [0, 1, 0, 132111],
                              [0, 0, 1, 29120],
                              [0, 0, 0, 155033]]),
                    columns=['overweight','exercise','internet','instances'])

gs = generate_causal_models(data)
print()
for g in gs:
    print(g)





