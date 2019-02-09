#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import itertools
from itertools import *


# In[2]:


def generate_dags(vertices, edges):
    return generate_dags_iter(vertices, edges, empty_dag(vertices))

def generate_dags_iter(vertices, edges, base):
    if edges == 0:
        return [base]
    ans = []
    for index, value in np.ndenumerate(base):
        if value == -1:
            graph = base.copy()
            graph[index] = 1
            if is_dag(graph):
                ans += generate_dags_iter(vertices, edges-1, graph)
    return ans    

def empty_dag(vertices):
    dag = np.full((vertices,vertices),-1,int)
    np.fill_diagonal(dag, 0)
    return dag

def children(graph,vertex):
    ans = []
    for i in graph[0]:
        if graph[vertex,i] == 1:
            ans.append(i)
    return ans

def parents(graph,vertex):
    ans = []
    for i in graph[0]:
        if graph[i,vertex]:
            ans.append(i)
    return ans

def is_dag(graph):
    path = set()
    def visit(vertex):
        path.add(vertex)
        for child in children(graph,vertex):
            if child in path or visit(child):
                return True
        path.remove(vertex)
        return False
    return not any(visit(v) for v in graph[0])


# In[25]:


def d_connected(x0, x1, graph, condition_set):
    
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
                if i not in reachable:
                    if i in unblocked:
                        frontier.append(i)
                    else:
                        frontier_collision_warning.append(i)
            for i in parents(graph,v):
                if i not in reachable:
                    frontier.append(i)            
        else:
            v = frontier_collision_warning.pop()
            reachable.add(v)
            for i in children(graph,v):
                if i not in reachable:
                    if i in unblocked:
                        frontier.append(i)
                    else:
                        frontier_collision_warning.append(i)
                        
    if x1 in reachable:
        return True
    else:
        return False
    
def path(graph, start, goals):
    
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
                
def intersection(lst1, lst2): 
    lst3 = [value for value in lst1 if value in lst2] 
    return lst3    
        
def independent(x0, x1, data, condition_set):
    return np.random.choice([True,False])


# In[26]:


def generate_causal_models(data):
    
    vertices = len(data.columns)-1
    max_edges = int(0.5*vertices*(vertices-1))
    
    for edges in range(max_edges+1):
        print(edges)
        graphs = generate_dags(vertices, edges)
        graphs = list(filter(lambda graph: compatible(graph,data), graphs))
        if graphs != []:
            return graphs
    return []

def compatible(graph, data):
    features = list(range(len(data.columns)))
    for condition_set in powerset(features):
        feature_set = []
        for x in features:
            if x not in condition_set:
                feature_set.append(x)
        for x0, x1 in itertools.combinations(feature_set,2):
            if d_connected(x0, x1, graph, condition_set) != independent(x0, x1, data, condition_set):
                return False
    return True

def powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))


# In[27]:


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

for g in gs:
    print(g)


# In[ ]:


print(list(range(len(['overweight', 'exercise', 'internet', 'instances']))))


# In[ ]:





# In[ ]:




