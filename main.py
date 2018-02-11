import networkx as nx
import sys
import random
import math
from numpy import linalg as LAD
import numpy as np
import heapq
import copy
import matplotlib.pyplot as plt


# Katz Index
def Katz_Index (matrix, nodes_num):
    w, v = LAD.eig(matrix)
    # max(w) - najwieksza eigenvalue
    beta = (1 / max(w)) - 0.1
    idn = np.identity(nodes_num)
    # LAD.inv - odwrotnosc macierzy
    S_KATZ = LAD.inv((idn - beta * matrix)) - idn
    return S_KATZ

# Leicht-Holme-Newman Index
def LHT_Index (graph, matrix, nodes_num, m, phi):
    w, v = LAD.eig(matrix)
    eig = max(w)
    # laplacian_matrix <==> L = D - A
    degree_matrix = nx.laplacian_matrix(graph) + matrix
    idn = np.identity(nodes_num)
    if(LAD.det(degree_matrix) == 0):
        print("\nLHT_Index:\nPrzy próbie obliczania odwrotności macierzy, napotkano wyznacznik równy 0.")
    else:
        deg_inv = LAD.inv(degree_matrix)
        phi_mul = (phi * matrix) / eig
        # nie jestem pewien znaczenia wspolczynnika m - zostawiam go jako param
        S_LHT = 2 * m * eig * deg_inv * (LAD.inv(idn - (phi_mul))) * deg_inv
        return S_LHT

# Average Commute Time
def ACT_Index (graph, nodes_num):
    lap = np.zeros(nodes_num) + nx.laplacian_matrix(graph)
    # L+ <- pseudo inverse of laplacian matrix
    pseudo_inv = LAD.pinv(lap)
    s = (nodes_num, nodes_num)
    S_ACT = np.zeros(s)
    for x in range(0, nodes_num):
        for y in range(0, nodes_num):
            num = 0
            if(x != y):
                denominator = (pseudo_inv[x, x] + pseudo_inv[y, y] - 2 * pseudo_inv[x, y])
                if(denominator == 0):
                    num = 0
                else:
                    num = 1 / denominator
            S_ACT[x, y] = num
    return S_ACT

#Cosine based on L+
def Cosine_Index (graph, nodes_num):
    lap = np.zeros(nodes_num) + nx.laplacian_matrix(graph)
    # L+ <- pseudo inverse of laplacian matrix
    pseudo_inv = LAD.pinv(lap)
    s = (nodes_num, nodes_num)
    S_Cosine = np.zeros(s)
    for x in range(0, nodes_num):
        for y in range(0, nodes_num):
            inner_sqrt = pseudo_inv[x, x] * pseudo_inv[y, y]
            if(inner_sqrt):
                num = 0
            else:
                denominator = (math.sqrt(inner_sqrt))
                if(denominator == 0):
                    num = 0
                else:
                    num = pseudo_inv[x, y] / denominator
            S_Cosine [x, y] = num
    return S_Cosine

# Random Walk with Restart
def RWR_Index (graph, nodes_num, c):
    # google matrix == transition matrix z papera
    P = nx.google_matrix(graph)
    idn = np.identity(nodes_num)
    trans = np.transpose(P)
    core = (1 - c) * LAD.inv((idn - c * trans))
    vector = np.zeros(nodes_num)
    qs = []
    vector[0] = 1
    new_v = copy.copy(vector)
    # vector <==> "baza" wektora
    qs.append(core.dot(new_v))

    for i in range(1, nodes_num):
        vector[i - 1] = 0
        vector[i] = 1
        new_v = copy.copy(vector)
        qs.append(core.dot(new_v))

    s = (nodes_num, nodes_num)
    S_RWR = np.zeros(s)
    for x in range(0, nodes_num):
        for y in range(0, nodes_num):
            S_RWR[x, y] = qs[x].tolist()[0][y] + qs[y].tolist()[0][x]
    return S_RWR


def MatrixForest_Index (graph, nodes_num, alpha):
    lap = np.zeros(nodes_num) + nx.laplacian_matrix(graph)
    idn = np.identity(nodes_num)
    S_MFI = LAD.inv(idn + alpha * lap)
    return S_MFI

#################################
#### Quasi-Local Indices
#################################

def LocalPath_Index(matrix, epsilon, n):
    mat_pow = matrix * matrix
    eps_pow = 1
    result = mat_pow
    for i in range(0, n-1):
        mat_pow = mat_pow * matrix
        eps_pow = eps_pow * epsilon
        result = result + eps_pow * mat_pow
    return result

# Local Random Walk
def LRW_Index (matrix, graph, n):
    P = nx.google_matrix(graph)
    trans = np.transpose(P)
    pi = []
    vector = np.zeros(nodes_num)
    vector[0] = 1
    new_v = copy.copy(vector)
    pi.append(new_v)

    for i in range(1, nodes_num):
        vector[i-1] = 0
        vector[i] = 1
        new_v = copy.copy(vector)
        pi.append(new_v)

    for i in range(0, n):
        for x in range(0, nodes_num):
            pi[x] = pi[x].dot(trans)

    s = (nodes_num, nodes_num)
    S_LRW = np.zeros(s)
    degree_matrix = nx.laplacian_matrix(graph) + matrix
    for x in range(0, nodes_num):
        q_x = degree_matrix[x, x] / len(graph.edges)
        for y in range(0, nodes_num):
            q_y = degree_matrix[y, y] / len(graph.edges)
            S_LRW[x, y] = q_x  * pi[x].tolist()[0][y] +q_y  * pi[y].tolist()[0][x]
    return S_LRW

##################################
#### Rank functions
##################################
def AUC (n, preds, graph, removed, nodes):
    # dopelnienie oryginalnego grafu
    nodes = [int(i) for i in nodes]
    compl = nx.complement(graph)
    missing = list(compl.edges())
    # krawedzie nie istniejace
    random.shuffle(missing)
    count = 0
    # zgodnie z def AUC z danego papera losuje i porownuje prawd.
    for i in range(0, n):
        x = random.randint(0, len(missing) - 1)
        fst = int(missing[x][0])
        snd = int(missing[x][1])
        # similarity danej krawedzi
        prob = preds[nodes.index(fst), nodes.index(snd)]
        # removed to krawedzie faktycznie istniejace grafie,
        # ale usuniete w celu testow
        x = random.randint(0, len(removed) - 1)
        fst = int(removed[x][0])
        snd = int(removed[x][1])
        chance = preds[nodes.index(fst), nodes.index(snd)]
        if (chance - prob > 0):
            count = count + 1
        elif(chance - prob == 0):
            count = count + 0.5
    return count / n


def precision(n, preds, correct, nodes, edges):
    tops = []
    already = []
    heapq.heapify(tops)
    nodes = list(nodes)
    for x in range(0, len(nodes)):
        for y in range(0, len(nodes)):
            #print(str(x) + ", " + str(y))
            xi = nodes[x]
            yi = nodes[y]
            if ((xi,yi) in edges or xi == yi or (yi,xi) in already):
                pass
            else:
                heapq.heappush(tops, (preds[x,y], (xi,yi)))
                already.append((xi,yi))

    pars = []
    rank = heapq.nlargest(n, tops)
    for i in range(0, n):
        num, (a,b) = rank[i]
        pars.append((a,b))

    count = 0
    for i in range(0, len(correct)):
        a = correct[i][0]
        b = correct[i][1]
        if((a,b) in pars or (b,a) in pars):
            count = count + 1
    print(count)
    print(heapq.nlargest(n, tops))


#########################
### Głowny program
#########################

# Wczytywanie opisu grafu z pliku
# dany graf jest w postaci: node node
# każda krawędź w osobnym wierzu
with open(str(sys.argv[1])) as f:
    lines = f.readlines()

myList = [line.strip().split() for line in lines]

# Mieszamy dane krawedzie, tak zeby wykorzystac
# ja potem w podziale na zbiory testowe/treningowe.
random.shuffle(myList)

removed = [] # removed - krawedzie usuniete, sluzace do testow

orginal = nx.Graph()

# Tworzenie grafu z opisu z pliku
for i in range(0, len(myList)):
    orginal.add_edge(myList[i][0], myList[i][1])

# Podzial na zbiory "uczacy" i testowy
training_set = math.ceil(len(myList) * 0.7)
test_set = len(myList) - training_set


g = nx.Graph()
#g.add_edges_from(myList)
for i in range (0, test_set):
    #g.remove_edge(myList[i][0], myList[i][1])
    removed.append((myList[i][0], myList[i][1]))
    g.add_node(myList[i][0])
    g.add_node(myList[i][1])

for i in range(test_set, len(myList)):
    g.add_edge(myList[i][0], myList[i][1])



# macierz incydencji
M = nx.to_numpy_matrix(g)
# liczba wierzcholkow w grafie
nodes_num = len(g.nodes())


###################
# Global
###################
print(M)
s_katz = Katz_Index(M, nodes_num)
auc = AUC(5, s_katz, orginal, removed, g.nodes())
precision(5, s_katz,removed, g.nodes(), g.edges())
s_lht = LHT_Index(g, M, nodes_num, 1, 0.5)
s_act = ACT_Index(g, nodes_num)
s_cosine = Cosine_Index(g, nodes_num)
s_mfi = MatrixForest_Index (g, nodes_num, 0.5)
s_rwr = RWR_Index(g, nodes_num, 0.8)

###################
# Quasi-Local
###################
s_lp = LocalPath_Index(M, 0.5, 3)
s_lrw = LRW_Index(M, g, 10)


# poglądowe rysowanie grafu
nx.draw(orginal, None, node_size=100, alpha=0.5, node_color="blue", with_labels=True)
plt.show()

nx.draw(g, None, node_size=100, alpha=0.5, node_color="blue", with_labels=True)
plt.show()
