import heapq
from turtle import pos
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import time
import numpy as np
from ipywidgets import interact, fixed, IntSlider, FloatSlider, Dropdown, Button, VBox, HBox, Output
import IPython.display as display
import plotly.graph_objects as go
import plotly.express as px
from ipywidgets import interact, FloatSlider, IntSlider

class Entrega:
    def __init__(self, minuto, destino, bonus):
        self.minuto = int(minuto)
        self.destino = destino
        self.bonus = int(bonus)

    def __repr__(self):
        return f"{self.destino}@{self.minuto}(+{self.bonus})"

class Conexao:
    def __init__(self, origem, destino, tempo):
        self.origem = origem
        self.destino = destino
        self.tempo = tempo

    def __repr__(self):
        return f"Conexao(origem={self.origem}, destino={self.destino}, tempo={self.tempo})"

def ler_conexoes(caminho_arquivo: str):
    """
    Lê a matriz de conexões a partir de um arquivo .txt e cria instâncias da classe Conexao.
    """
    # Lê o arquivo CSV como um DataFrame
    matriz = pd.read_csv(caminho_arquivo, delimiter=',', header=None, names=['origem', 'destino', 'tempo'])
    
    # Cria instâncias da classe Conexao para cada linha no DataFrame
    conexoes = [Conexao(row['origem'], row['destino'], row['tempo']) for _, row in matriz.iterrows()]

    return conexoes

def ler_entregas(caminho_arquivo: str):
    """
    Lê a lista de entregas a partir de um arquivo .txt e cria instâncias da classe Entrega.
    """
    # Lê o arquivo CSV como um DataFrame
    entregas_df = pd.read_csv(caminho_arquivo, delimiter=',', header=None, names=['minuto', 'destino', 'bonus'])
    
    # Ordena as entregas pelo 'minuto'
    entregas_df = entregas_df.sort_values(by='minuto')
    
    # Cria instâncias da classe Entrega para cada linha no DataFrame
    entregas = [Entrega(row['minuto'], row['destino'], row['bonus']) for _, row in entregas_df.iterrows()]
    
    return entregas

def criar_grafo(conexoes):
    """
    Cria um grafo direcionado a partir de uma lista de objetos Conexao, incluindo arestas reversas.
    """
    grafo = nx.DiGraph()

    # Itera sobre as conexões
    for c in conexoes:
        origem = c.origem
        destino = c.destino
        peso = c.tempo
        
        # Adiciona a aresta original (origem -> destino)
        grafo.add_edge(origem, destino, weight=peso)

        # Adiciona a aresta reversa (destino -> origem) com o mesmo peso
        grafo.add_edge(destino, origem, weight=peso)

    return grafo

def dijkstra_basico(grafo, inicio, entregas):
    """
    Algoritmo básico para o Leilão de Entregas Versão 1.
    Agora é guloso em relação ao bônus acumulado.
    """
    pq = []  # Fila de prioridade para os caminhos
    heapq.heappush(pq, (0, 0, inicio, []))  # (-bônus_acumulado, tempo_atual, posição_atual, caminho)

    melhor_lucro = 0
    melhor_caminho = []

    while pq:
        # Desempilhar o estado atual
        lucro_neg, tempo_atual, posicao_atual, caminho = heapq.heappop(pq)
        lucro_atual = -lucro_neg  # Transformar o lucro de volta para positivo

        # Atualiza o melhor lucro e caminho se necessário
        if lucro_atual > melhor_lucro:
            melhor_lucro = lucro_atual
            melhor_caminho = caminho

        # Verifica as entregas disponíveis
        for entrega in entregas:
            horario_saida = entrega.minuto  # Horário programado para saída
            destino = entrega.destino
            bonus = entrega.bonus

            # Ignora destinos já entregues
            if destino in [c[0] for c in caminho]:
                continue

            # Ignora entregas com horário de saída já passado
            if horario_saida < tempo_atual:
                continue

            try:
                # Tempo de espera até o horário de saída
                tempo_espera = max(0, horario_saida - tempo_atual)
                
                # Tempo para chegar ao destino
                tempo_ida = nx.shortest_path_length(grafo, source=posicao_atual, target=destino, weight='weight')
                
                # Tempo de chegada ao destino (após esperar e viajar)
                tempo_chegada = tempo_atual + tempo_espera + tempo_ida

                # Verifica se há caminho de retorno
                try:
                    tempo_volta = nx.shortest_path_length(grafo, source=destino, target=inicio, weight='weight')
                    
                    # Tempo total após completar a entrega e voltar
                    tempo_final = tempo_chegada + tempo_volta
                    
                    # Novo lucro após fazer esta entrega
                    novo_lucro = lucro_atual + bonus
                    
                    # Adiciona ao heap com novo bônus, tempo e caminho
                    novo_caminho = caminho + [(destino, bonus, tempo_final)]
                    
                    # Ordem: -bônus, tempo, posição, caminho
                    # Ordenar por bônus faz com que o algoritmo seja guloso em relação ao bônus
                    heapq.heappush(pq, (-novo_lucro, tempo_final, inicio, novo_caminho))

                except nx.NetworkXNoPath:
                    # Se não houver caminho de volta, ignora a entrega
                    continue

            except nx.NetworkXNoPath:
                # Se não houver caminho até o destino, ignora a entrega
                continue

    return melhor_lucro, melhor_caminho

def a_star_otimizado(grafo, entregas, inicio='A'):
    # Calcular caminhos mais curtos entre todos os pares de nós
    caminhos_minimos = dict(nx.all_pairs_dijkstra(grafo))
    
    # Fila de prioridade para A*: (f, -bônus_atual, tempo_atual, posição, entregas_feitas, entregas_restantes)
    fila = []
    
    # Iniciar com todas as entregas disponíveis, sem bônus acumulado, no tempo 0
    entregas_restantes = tuple(sorted([(e.minuto, e.destino, e.bonus) for e in entregas]))
    heapq.heappush(fila, (0, 0, 0, inicio, (), entregas_restantes))
    
    # Conjunto para estados visitados
    visitados = set()
    
    # Melhor solução encontrada
    melhor_bonus = 0
    melhor_caminho = []
    
    while fila:
        # Desempacotar o estado atual
        _, bonus_neg, tempo_atual, posicao, entregas_feitas, entregas_restantes = heapq.heappop(fila)
        bonus_atual = -bonus_neg
        
        # Verificar se já encontramos uma solução melhor
        if bonus_atual > melhor_bonus:
            melhor_bonus = bonus_atual
            melhor_caminho = list(entregas_feitas)
        
        # Verificar se este estado já foi visitado
        estado = (tempo_atual, posicao, entregas_feitas)
        if estado in visitados:
            continue
        visitados.add(estado)
        
        # Explorar todas as possíveis próximas entregas
        for i, (minuto_saida, destino, bonus) in enumerate(entregas_restantes):
            # Verificar se podemos fazer esta entrega agora
            if minuto_saida < tempo_atual:
                continue  # Já passou o horário de saída
            
            # Verificar se há caminho até o destino
            if destino not in caminhos_minimos[posicao][0]:
                continue
            
            # Tempo de espera até o horário de saída
            tempo_espera = max(0, minuto_saida - tempo_atual)
            
            # Tempo para chegar ao destino
            tempo_ida = caminhos_minimos[posicao][0][destino]
            
            # Tempo de chegada ao destino
            tempo_chegada = tempo_atual + tempo_espera + tempo_ida
            
            # Tempo para voltar ao início
            tempo_volta = caminhos_minimos[destino][0][inicio]
            
            # Tempo total após completar a entrega
            tempo_final = tempo_chegada + tempo_volta
            
            # Novo bônus acumulado
            novo_bonus = bonus_atual + bonus
            
            # Novas entregas feitas
            novas_entregas_feitas = entregas_feitas + ((destino, bonus, tempo_final),)
            
            # Remover esta entrega da lista de restantes
            novas_entregas_restantes = entregas_restantes[:i] + entregas_restantes[i+1:]
            
            # Calcular heurística: máximo bônus possível das entregas restantes
            # Considerar apenas entregas que ainda podem ser feitas
            h = sum(b for m, _, b in novas_entregas_restantes if m > tempo_final)
            
            # Função de avaliação f(n) = -g(n) + h(n)
            # Maximizar o bônus (negativo para priorizar maior bônus) e minimizar o tempo
            f = -(novo_bonus) + tempo_final + h
            
            # Adicionar à fila de prioridade
            heapq.heappush(fila, (f, -novo_bonus, tempo_final, inicio, novas_entregas_feitas, novas_entregas_restantes))
    
    return melhor_bonus, melhor_caminho

def visualizar_grafo(grafo, caminho=None, entregas=None):
    """
    Visualiza o grafo de conexões com opção de destacar um caminho e exibir o bônus em cada nó.
    """
    plt.figure(figsize=(10, 8))
    
    # Posicionar os nós
    pos = nx.spring_layout(grafo, seed=42)  # seed para reprodutibilidade
    
    # Desenhar nós e arestas
    nx.draw_networkx_nodes(grafo, pos, node_size=900, node_color='skyblue')
    
    # Destacar o nó inicial 'A'
    nx.draw_networkx_nodes(grafo, pos, nodelist=['A'], node_size=900, node_color='red')
    
    # Desenhar todas as arestas
    nx.draw_networkx_edges(grafo, pos, width=1.0, alpha=0.5)
    
    # Desenhar pesos das arestas
    edge_labels = {(u, v): d['weight'] for u, v, d in grafo.edges(data=True)}
    nx.draw_networkx_edge_labels(grafo, pos, edge_labels=edge_labels)
    
    # Adicionar os bônus como rótulos nos nós (mantendo as letras)
    if entregas:
        # Criar um dicionário de rótulos com "letra (bônus)"
        bonus_labels = {entrega.destino: f"{entrega.destino} ({entrega.bonus})" for entrega in entregas}
        # Adicionar o nó inicial 'A' com bônus 0 (ou outro valor, se necessário)
        bonus_labels['A'] = "A (0)"
        nx.draw_networkx_labels(grafo, pos, labels=bonus_labels, font_color='black', font_size=10)
    else:
        # Adicionar rótulos padrão (apenas os nomes dos nós)
        nx.draw_networkx_labels(grafo, pos)
    
    # Se um caminho for fornecido, destacar as arestas do caminho
    if caminho:
        # Extrair os nós do caminho
        nodes_path = ['A']  # Começar do ponto inicial A
        for destino, _, _ in caminho:
            # Calcular o caminho mais curto para o destino
            subcaminho = nx.shortest_path(grafo, source=nodes_path[-1], target=destino, weight='weight')
            nodes_path.extend(subcaminho[1:])  # Adicionar o subcaminho, ignorando o nó inicial duplicado
            nodes_path.append('A')  # Retornar para A após cada entrega
        
        # Criar as arestas a partir dos nós do caminho
        path_edges = [(nodes_path[i], nodes_path[i+1]) for i in range(len(nodes_path)-1)]
        nx.draw_networkx_edges(grafo, pos, edgelist=path_edges, width=3, alpha=0.8, edge_color='red')
    
    plt.title('Grafo de Conexões com Bônus nos Nós')
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def main():
    # Ler dados
    conexoes = ler_conexoes('conexoes.txt')
    entregas = ler_entregas('entregas.txt')

    # Criar o grafo a partir das conexões
    grafo = criar_grafo(conexoes)

    # Executar algoritmos e medir tempos
    start_time = time.perf_counter()
    bonus_basico, caminho_basico = dijkstra_basico(grafo, 'A', entregas)
    tempo_basico = (time.perf_counter() - start_time) * 1000

    start_time = time.perf_counter()
    bonus_otimizado, caminho_otimizado = a_star_otimizado(grafo, entregas)
    tempo_otimizado = (time.perf_counter() - start_time) * 1000

    # Mostrar resultados no console
    print(f"\n💰 Simulação básica")
    # print(f"💰 Bônus total acumulado: {bonus_basico}")

    print(f"⏱️ Tempo de execução: {tempo_basico:.6f} ms")
    print("📦 Melhor sequência de entregas (A -> entrega -> A):")
    print("Entrega: ", caminho_basico)
    for destino, bonus, tempo in caminho_basico:
        print(f" - {destino}: bônus {bonus}, entregue até {tempo} minutos")

    print(f"\n💰 Simulação otimizada")
    print(f"💰 Bônus total acumulado: {bonus_otimizado}")
    print(f"⏱️ Tempo de execução: {tempo_otimizado:.6f} ms")
    print("📦 Melhor sequência de entregas (A -> entrega -> A):")
    print("Entrega: ", caminho_otimizado)
    for destino, bonus, tempo in caminho_otimizado:
        print(f" - {destino}: bônus {bonus}, entregue até {tempo} minutos")

    # Criar gráficos comparativos
    plt.figure(figsize=(12, 5))
    
    # Gráfico de bônus
    plt.subplot(1, 2, 1)
    algoritmos = ['Básico', 'Otimizado']
    bonus = [bonus_basico, bonus_otimizado]
    plt.bar(algoritmos, bonus, color=['blue', 'green'])
    plt.title('Comparação de Bônus')
    plt.ylabel('Bônus Total')
    
    # Gráfico de minutos gastos na entrega
    plt.subplot(1, 2, 2)
    minutos_gastos = [sum([tempo for _, _, tempo in caminho_basico]), sum([tempo for _, _, tempo in caminho_otimizado])]
    plt.bar(algoritmos, minutos_gastos, color=['blue', 'green'])
    plt.title('Comparação de Minutos Gastos na Entrega')
    plt.ylabel('Minutos Totais')
    plt.tight_layout()
    plt.savefig('comparacao_alg.png')
    plt.show()

    # Visualizar grafo com o melhor caminho
    visualizar_grafo(grafo, caminho_otimizado, entregas)

if __name__ == '__main__':
    main()
