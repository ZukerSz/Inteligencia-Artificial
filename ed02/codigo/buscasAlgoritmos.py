import time
import pandas as pd
import psutil
import os
from heapq import heappush, heappop
from collections import deque

# carrega o arquivo CSV
caminho_arquivo = "ed02-puzzle8.csv"
df = pd.read_csv(caminho_arquivo)

# defini o estado inicial e objetivo
estados_iniciais = [tuple(linha) for linha in df.values]
estado_objetivo = (1, 2, 3, 4, 5, 6, 7, 8, 0)

# movimentos e movimentos inválidos
movimentos = {
    'cima': -3,
    'baixo': 3,
    'esquerda': -1,
    'direita': 1
}

movimentos_invalidos = {
    'cima': [0, 1, 2],
    'baixo': [6, 7, 8],
    'esquerda': [0, 3, 6],
    'direita': [2, 5, 8]
}

# funcões
def mover(estado, direcao):
    indice_zero = estado.index(0)
    if indice_zero in movimentos_invalidos[direcao]:
        return None
    novo_indice = indice_zero + movimentos[direcao]
    lista = list(estado)
    lista[indice_zero], lista[novo_indice] = lista[novo_indice], lista[indice_zero]
    return tuple(lista)

def vizinhos(estado):
    return [novo for direcao in movimentos if (novo := mover(estado, direcao))]

def heuristica_manhattan(estado):
    distancia = 0
    for i, val in enumerate(estado):
        if val == 0:
            continue
        pos_objetivo = estado_objetivo.index(val)
        distancia += abs(i // 3 - pos_objetivo // 3) + abs(i % 3 - pos_objetivo % 3)
    return distancia

def reconstruir_caminho(origem, atual):
    caminho = []
    while atual in origem:
        atual = origem[atual]
        caminho.append(atual)
    return caminho[::-1]

# ALGORITMOS DE BUSCAS

# busca largura
def busca_largura(inicial):
    inicio = time.time()
    fila = deque([inicial])
    origem = {}
    visitados = set()
    visitados.add(inicial)
    mem_inicial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # memória inicial em MB

    while fila:
        atual = fila.popleft()
        if atual == estado_objetivo:
            tempo = time.time() - inicio
            mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # memória final em MB
            memoria_usada = mem_final - mem_inicial
            return len(reconstruir_caminho(origem, atual)), len(visitados), tempo, memoria_usada
        for vizinho in vizinhos(atual):
            if vizinho not in visitados:
                visitados.add(vizinho)
                fila.append(vizinho)
                origem[vizinho] = atual
    return None

# busca em profundidade
def busca_profundidade(inicial, profundidade_maxima=50):
    inicio = time.time()
    pilha = [(inicial, 0)]
    origem = {}
    visitados = set()
    mem_inicial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  

    while pilha:
        atual, profundidade = pilha.pop()
        if profundidade > profundidade_maxima:
            continue
        if atual == estado_objetivo:
            tempo = time.time() - inicio
            mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  
            memoria_usada = mem_final - mem_inicial
            return len(reconstruir_caminho(origem, atual)), len(visitados), tempo, memoria_usada
        if atual not in visitados:
            visitados.add(atual)
            for vizinho in vizinhos(atual):
                if vizinho not in visitados:
                    pilha.append((vizinho, profundidade + 1))
                    origem[vizinho] = atual
    return None

# busca gulosa
def busca_gulosa(inicial, heuristica):
    inicio = time.time()
    fila = []
    heappush(fila, (heuristica(inicial), inicial))
    origem = {}
    visitados = set()
    mem_inicial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 

    while fila:
        _, atual = heappop(fila)
        if atual == estado_objetivo:
            tempo = time.time() - inicio
            mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024) 
            memoria_usada = mem_final - mem_inicial
            return len(reconstruir_caminho(origem, atual)), len(visitados), tempo, memoria_usada
        visitados.add(atual)
        for vizinho in vizinhos(atual):
            if vizinho not in visitados:
                heappush(fila, (heuristica(vizinho), vizinho))
                origem[vizinho] = atual
    return None

# algoritmo A*
def busca_a_estrela(inicial, heuristica):
    inicio = time.time()
    fila = []
    custo_g = {inicial: 0}
    heappush(fila, (heuristica(inicial), inicial))
    origem = {}
    visitados = set()
    mem_inicial = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # memória inicial em MB

    while fila:
        _, atual = heappop(fila)
        if atual == estado_objetivo:
            tempo = time.time() - inicio
            mem_final = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # memória final em MB
            memoria_usada = mem_final - mem_inicial
            return len(reconstruir_caminho(origem, atual)), len(visitados), tempo, memoria_usada
        visitados.add(atual)
        for vizinho in vizinhos(atual):
            custo_tentativo = custo_g[atual] + 1
            if vizinho not in custo_g or custo_tentativo < custo_g[vizinho]:
                custo_g[vizinho] = custo_tentativo
                prioridade = custo_tentativo + heuristica(vizinho)
                heappush(fila, (prioridade, vizinho))
                origem[vizinho] = atual
    return None

# executa os algoritmos para cada instância
resultados = []

for i, estado in enumerate(estados_iniciais):
    print(f"Executando instância {i + 1}...")

    bfs_resultado = busca_largura(estado)
    dfs_resultado = busca_profundidade(estado)
    gulosa_resultado = busca_gulosa(estado, heuristica_manhattan)
    a_star_resultado = busca_a_estrela(estado, heuristica_manhattan)

    resultado = {
        "Instância": i + 1,
        " Movimentos(BFS)": bfs_resultado[0] if bfs_resultado else "Não encontrado",
        " Tempo(BFS)": f"{bfs_resultado[2]:.4f}s" if bfs_resultado else "Não encontrado",
        " Memória(BFS)": f"{bfs_resultado[3]:.2f} MB" if bfs_resultado else "Não encontrado",
        
        " Movimentos(DFS)": dfs_resultado[0] if dfs_resultado else "Não encontrado",
        " Tempo(DFS)": f"{dfs_resultado[2]:.4f}s" if dfs_resultado else "Não encontrado",
        " Memória(DFS)": f"{dfs_resultado[3]:.2f} MB" if dfs_resultado else "Não encontrado",
        
        " Movimentos(Gulosa)": gulosa_resultado[0] if gulosa_resultado else "Não encontrado",
        " Tempo(Gulosa)": f"{gulosa_resultado[2]:.4f}s" if gulosa_resultado else "Não encontrado",
        " Memória(Gulosa)": f"{gulosa_resultado[3]:.2f} MB" if gulosa_resultado else "Não encontrado",
        
        " Movimentos(A*)": a_star_resultado[0] if a_star_resultado else "Não encontrado",
        " Tempo(A*)": f"{a_star_resultado[2]:.4f}s" if a_star_resultado else "Não encontrado",
        " Memória(A*)": f"{a_star_resultado[3]:.2f} MB" if a_star_resultado else "Não encontrado",
    }

    resultados.append(resultado)

# mostra a tabela de resultados
df_resultados = pd.DataFrame(resultados)

# mostra os resultados das instâncias 1 a 10
print("\nResultados das instâncias 1 a 10:")
print(df_resultados.iloc[:10])


