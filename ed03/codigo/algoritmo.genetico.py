
import random
import time
import csv
import glob
import os
import re

def carregar_instancias_de_csvs(pasta="."):
    instancias = {}
    caminho_arquivos = os.path.join(pasta, 'knapsack_*.csv')

    def chave_ordenacao_natural(s):
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

    arquivos_ordenados = sorted(glob.glob(caminho_arquivos), key=chave_ordenacao_natural)

    if not arquivos_ordenados:
        print("Nenhum arquivo no formato 'knapsack_*.csv' foi encontrado.")
        return None

    print("--- Carregando Instâncias do Problema ---")
    for caminho_arquivo in arquivos_ordenados:
        try:
            with open(caminho_arquivo, mode='r', encoding='utf-8') as f:
                linhas = f.readlines()
                linha_capacidade_str = linhas[-1].strip()
                capacidade = int(linha_capacidade_str.split(',')[1])
                itens = []
                for linha in linhas[1:-1]:
                    partes = linha.strip().split(',')
                    peso = int(partes[1])
                    valor = int(partes[2])
                    itens.append((peso, valor))

                nome_base = os.path.basename(caminho_arquivo)
                numero_instancia = re.search(r'\d+', nome_base).group()
                nome_instancia = f"Knapsack {numero_instancia}"
                
                instancias[nome_instancia] = {'capacidade': capacidade, 'itens': itens}
        except Exception as e:
            print(f"ERRO ao processar o arquivo {caminho_arquivo}: {e}")

    print(f"\nCarga concluída. {len(instancias)} instâncias prontas para o teste:")
    for nome in instancias.keys():
        print(f"  - {nome}")
    print("-" * 45)
    time.sleep(1)
    
    return instancias

def calcular_fitness(individuo, itens, capacidade):
    peso_total, valor_total = 0, 0
    for i, gene in enumerate(individuo):
        if gene == 1:
            peso_total += itens[i][0]
            valor_total += itens[i][1]
    
    return valor_total if peso_total <= capacidade else 0

def inicializar_populacao(tamanho_pop, num_itens, modo='aleatoria', itens=None, capacidade=None):
    if modo == 'heuristica' and itens and capacidade:
        populacao = []
        densidade = sorted([(i, v / p if p > 0 else 0) for i, (p, v) in enumerate(itens)], key=lambda x: x[1], reverse=True)
        
        for _ in range(tamanho_pop):
            individuo = [0] * num_itens
            peso_atual = 0
            for i, _ in densidade:
                if random.random() > 0.3 and peso_atual + itens[i][0] <= capacidade:
                    individuo[i] = 1
                    peso_atual += itens[i][0]
            for _ in range(int(num_itens * 0.1)):
                idx = random.randint(0, num_itens - 1)
                if individuo[idx] == 0 and peso_atual + itens[idx][0] <= capacidade:
                    individuo[idx] = 1
                    peso_atual += itens[idx][0]
            populacao.append(individuo)
        return populacao

    return [[random.randint(0, 1) for _ in range(num_itens)] for _ in range(tamanho_pop)]

def selecionar_pais(populacao, fitness_valores):
    indices_participantes = random.sample(range(len(populacao)), k=3)
    melhor_indice = max(indices_participantes, key=lambda i: fitness_valores[i])
    
    indices_participantes.remove(melhor_indice)
    segundo_melhor_indice = max(indices_participantes, key=lambda i: fitness_valores[i])
    
    return populacao[melhor_indice], populacao[segundo_melhor_indice]

def crossover(pai1, pai2, tipo='ponto_unico'):
    tamanho = len(pai1)
    if tamanho < 2: return pai1[:], pai2[:]
    filho1, filho2 = pai1[:], pai2[:]
    
    if tipo == 'ponto_unico':
        ponto = random.randint(1, tamanho - 1)
        filho1 = pai1[:ponto] + pai2[ponto:]
        filho2 = pai2[:ponto] + pai1[ponto:]
    elif tipo == 'dois_pontos' and tamanho > 2:
        p1, p2 = sorted(random.sample(range(1, tamanho), 2))
        filho1 = pai1[:p1] + pai2[p1:p2] + pai1[p2:]
        filho2 = pai2[:p1] + pai1[p1:p2] + pai2[p2:]
    elif tipo == 'uniforme':
        for i in range(tamanho):
            if random.random() < 0.5:
                filho1[i], filho2[i] = pai2[i], pai1[i]
            
    return filho1, filho2

def mutacao(individuo, taxa_mutacao):
    for i in range(len(individuo)):
        if random.random() < taxa_mutacao:
            individuo[i] = 1 - individuo[i]
    return individuo

def algoritmo_genetico(itens, capacidade, config):
    tempo_inicio = time.time()
    tamanho_pop = config['tamanho_pop']
    num_itens = len(itens)
    populacao = inicializar_populacao(tamanho_pop, num_itens, config['init_pop'], itens, capacidade)
    melhor_solucao_geral = None
    melhor_fitness_geral = 0
    convergencia_count = 0
    
    for geracao in range(config['num_geracoes']):
        fitness_valores = [calcular_fitness(ind, itens, capacidade) for ind in populacao]
        nova_populacao = []
        
        idx_melhor_geracao = max(range(len(fitness_valores)), key=fitness_valores.__getitem__)
        melhor_da_geracao = populacao[idx_melhor_geracao]
        fitness_melhor_da_geracao = fitness_valores[idx_melhor_geracao]
        nova_populacao.append(melhor_da_geracao)
        
        if fitness_melhor_da_geracao > melhor_fitness_geral:
            melhor_fitness_geral = fitness_melhor_da_geracao
            melhor_solucao_geral = melhor_da_geracao
            convergencia_count = 0
        else:
            convergencia_count += 1
        
        if config['criterio_parada'] == 'convergencia' and convergencia_count >= config['paciencia']:
            break

        while len(nova_populacao) < tamanho_pop:
            pai1, pai2 = selecionar_pais(populacao, fitness_valores)
            filho1, filho2 = crossover(pai1, pai2, config['crossover'])
            nova_populacao.append(mutacao(filho1, config['taxa_mutacao']))
            if len(nova_populacao) < tamanho_pop:
                nova_populacao.append(mutacao(filho2, config['taxa_mutacao']))
        
        populacao = nova_populacao
        
    tempo_fim = time.time()
    
    return {
        "melhor_valor": melhor_fitness_geral,
        "tempo_execucao": tempo_fim - tempo_inicio,
    }

def rodar_experimentos(instancias):
    configuracoes = {
        'Tipo de Crossover': {
            'Ponto Único': {'crossover': 'ponto_unico'},
            'Dois Pontos': {'crossover': 'dois_pontos'},
            'Uniforme': {'crossover': 'uniforme'}
        },
        'Taxa de Mutação': {
            'Baixa (1%)': {'taxa_mutacao': 0.01},
            'Média (5%)': {'taxa_mutacao': 0.05},
            'Alta (10%)': {'taxa_mutacao': 0.10}
        },
        'Método de Inicialização': {
            'Aleatória': {'init_pop': 'aleatoria'},
            'Heurística': {'init_pop': 'heuristica'}
        },
        'Critério de Parada': {
            'Gerações Fixas': {'criterio_parada': 'geracoes_fixas'},
            'Convergência': {'criterio_parada': 'convergencia'}
        }
    }
    
    base_config = {
        'tamanho_pop': 100,
        'num_geracoes': 200,
        'taxa_mutacao': 0.05,
        'crossover': 'ponto_unico',
        'init_pop': 'aleatoria',
        'criterio_parada': 'geracoes_fixas',
        'paciencia': 20
    }
    
    print("\n\n>>> Iniciando testes <<<\n")
    
    for nome_grupo, configs_grupo in configuracoes.items():
        print(f"\n{'='*60}")
        print(f"   AVALIANDO O IMPACTO DE: {nome_grupo.upper()}")
        print(f"{'='*60}")
        
        for nome_config, partial_config in configs_grupo.items():
            config_atual = base_config.copy()
            config_atual.update(partial_config)
            
            print(f"\n   [*] Configuração em teste: '{nome_config}'")
            
            for nome_instancia, dados_instancia in instancias.items():
                resultado = algoritmo_genetico(
                    dados_instancia['itens'], 
                    dados_instancia['capacidade'], 
                    config_atual
                )
                print(f"         - {nome_instancia}: \tValor = {resultado['melhor_valor']}, \tTempo = {resultado['tempo_execucao']:.4f}s")
        
        time.sleep(1)

if __name__ == '__main__':
    random.seed(42)
    
    instancias_carregadas = carregar_instancias_de_csvs()
    
    if instancias_carregadas:
        rodar_experimentos(instancias_carregadas)
        print("\n\n--- Bateria de testes finalizada! ---")