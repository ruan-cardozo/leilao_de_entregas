import heapq
import networkx as nx
import pandas as pd
import matplotlib.pyplot as plt

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
    grafo = nx.Graph()

    # Itera sobre as conexões
    for c in conexoes:
        origem = c.origem
        destino = c.destino
        peso = c.tempo
        
        # Adiciona a aresta original (origem -> destino)
        grafo.add_edge(origem, destino, weight=peso)

    return grafo

def dijkstra_basico(grafo, inicio, entregas):
    """
    Algoritmo básico para o Leilão de Entregas Versão 1.
    Considera o horário programado para saída.
    """
    pq = []  # Fila de prioridade para os caminhos
    heapq.heappush(pq, (0, 0, inicio, []))  # (tempo_atual, lucro_atual, posição_atual, caminho)

    melhor_lucro = 0
    melhor_caminho = []

    while pq:
        tempo_atual, lucro_atual, posicao_atual, caminho = heapq.heappop(pq)

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
                    
                    # Adiciona ao heap com novo tempo, lucro e caminho
                    novo_caminho = caminho + [(destino, bonus, tempo_final)]
                    
                    # Ordem: tempo, lucro, posição, caminho
                    # Ordenar por tempo faz com que o algoritmo seja guloso em relação ao tempo
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
            
            # Função de avaliação f(n) = -g(n) - h(n)
            # Como queremos maximizar o bônus, usamos o negativo
            f = -(novo_bonus + h)
            
            # Adicionar à fila de prioridade
            heapq.heappush(fila, (f, -novo_bonus, tempo_final, inicio, novas_entregas_feitas, novas_entregas_restantes))
    
    return melhor_bonus, melhor_caminho

def plotar_grafico_tempo_lucro(tempos, lucros):
    """
    Plota um gráfico do lucro em relação ao tempo de execução.
    """
    plt.plot(tempos, lucros, marker='o')  # Adicionei marcadores para melhor visualização
    plt.title('Comparação de Lucro e Tempo')
    plt.xlabel('Tempo de Execução (min)')
    plt.ylabel('Lucro (Bônus)')
    plt.grid(True)  # Adiciona uma grade para facilitar a leitura
    plt.show()

def main():
    # Ler dados
    conexoes = ler_conexoes('conexoes.txt')
    entregas = ler_entregas('entregas.txt')

    # Criar o grafo a partir das conexões
    grafo = criar_grafo(conexoes)

    # Rodar simulações
    # Tem que ajustar para considerar a ida e volta + minutagem de saída do ponto A
    bonus, caminho = dijkstra_basico(grafo,'A',entregas)
    print(f"\n💰 Simulação básica")
    print(f"\n💰 Bônus total acumulado: {bonus}")
    print("📦 Melhor sequência de entregas (A -> entrega -> A):")
    for destino, bonus, tempo in caminho:
        print(caminho)
        print(f" - {destino}: bônus {bonus}, entregue até {tempo} minutos")

    bonus, caminho = a_star_otimizado(grafo, entregas)
    print(f"\n💰 Simulação otimizada")
    print(f"\n💰 Bônus total acumulado: {bonus}")
    print("📦 Melhor sequência de entregas (A -> entrega -> A):")
    for destino, bonus, tempo in caminho:
        print(f" - {destino}: bônus {bonus}, entregue até {tempo} minutos")

    #Exibir gráficos comparando os lucros
    print("\nGerando gráfico de comparação...")
    plotar_grafico_tempo_lucro(
         ['Simulação Básica', 'Simulação Otimizada'],
         [resultado_basico['lucro_total'], resultado_otimizado['lucro_total']]
    )

if __name__ == '__main__':
    main()
