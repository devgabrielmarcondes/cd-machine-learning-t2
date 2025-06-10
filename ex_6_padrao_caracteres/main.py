from linear_algebra.operations import dot
import math, random

def sigmoid(t):
    return 1 / (1 + math.exp(-t))

def neuronio_MCP (pesos, entradas):
    uk = dot(pesos, entradas)
    return sigmoid(uk)

def feed_forward(rede, vetor_entrada):
    """
    recebe uma rede neural (representada como uma lista de listas de listas de pesos)
    e retorna a saída da propagação direta da entrada
    """

    vetor_saida = []
    
    for ponteiro in rede:
        entrada_com_bias = vetor_entrada + [1]
        saida = [neuronio_MCP(neuronio, entrada_com_bias) for neuronio in ponteiro]
        vetor_saida.append(saida)
        vetor_entrada = saida

    return vetor_saida

alpha = 0.1

def backpropagation(rede_neural, vetor_entrada, vetor_saida):
    saidas_intermediarias, saidas_neuronios = feed_forward(rede_neural, vetor_entrada)

    deltas_saida = [ saida * (1 - saida) * (saida - vetor_saida[i]) * alpha for i, saida in enumerate(saidas_neuronios)]

    for i, neuronio_saida in enumerate(rede_neural[-1]):
        for j, saida_intermediaria in enumerate(saidas_intermediarias+ [1]):
            neuronio_saida[j] -= deltas_saida[i] * saida_intermediaria

    deltas_intermediarios = [ saida_intermediaria * (1 - saida_intermediaria) * dot(deltas_saida, [n[i] for n in rede_neural[-1]]) for i, saida_intermediaria in enumerate(saidas_intermediarias)]

    for i, neuronio_intermediario in enumerate(rede_neural[0]):
        for j, entrada in enumerate(vetor_entrada + [1]):
            neuronio_intermediario[j] -= deltas_intermediarios[i] * entrada

if __name__ == "__main__":
    lista_treino = [
        """
        11111
        1...1
        11111
        1...1
        1...1
        """,
        """	
        11111
        1....
        11111
        1....
        11111
        """,
        """
        ..1..
        ..1..
        ..1..
        ..1..
        ..1..
        """,
        """
        11111
        1...1
        1...1
        1...1   
        11111
        """,
        """
        1...1
        1...1
        1...1
        1...1
        11111
        """
    ]

def make_digit(dados_treino):
    """
    Recebe uma string com o padrão de um dígito e retorna uma lista de 25 valores
    representando o padrão do dígito, onde 1 representa um pixel aceso e 0 um pixel apagado.
    """
    return [1 if c == '1' else 0 for row in dados_treino.split("\n") for c in row.strip()]

caracteres_numericos = list(map(make_digit, lista_treino))
print("\n vetores de entrada: INPUTS \n ", caracteres_numericos)

saidas = [[1 if i == j else 0 for i in range(len(caracteres_numericos))] for j in range(len(caracteres_numericos))]

print("\n saídas esperadas: OUTPUTS \n", saidas)

random.seed(0)
dimensao_entrada = 25
neuronios_ocultos = 5
dimensao_saida = 5

camada_intermediaria = [[random.random() for _ in range(dimensao_entrada + 1)] for _ in range(neuronios_ocultos)]

camada_saida = [[random.random() for _ in range(neuronios_ocultos + 1)] for _ in range(dimensao_saida)]

rede_neural = [camada_intermediaria, camada_saida]

# 1000 Ciclos de treinamento
for _ in range(75000):
    for vetor_entrada, vetor_saida in zip(caracteres_numericos, saidas):
        backpropagation(rede_neural, vetor_entrada, vetor_saida)

def predict(vogais):
    return feed_forward(rede_neural, vogais)[-1]

for i, vogais in enumerate(caracteres_numericos):
    saidas_rede = predict(vogais)
    print(f"Entrada: {vogais}, Saída prevista: {saidas_rede}")


# Teste com vogais com novo padrão
vogais_arredondadas = [
    """
    .111.
    1...1
    11111
    1...1
    1...1
    """,  # A arredondado
    
    """
    .111.
    1....
    11111
    1....
    .111.
    """,  # E arredondado
    
    """
    ..1..
    ..1..
    ..1..
    ..1..
    .111.
    """,  # I arredondado
    
    """
    .111.
    1...1
    1...1
    1...1
    .111.
    """,  # O arredondado
    
    """
    1...1
    1...1
    1...1
    .1.1.
    ..1..
    """   # U arredondado
]

print("\n=== TESTE COM PADRÕES ARREDONDADOS ===")
vogais_arredondadas_vetores = list(map(make_digit, vogais_arredondadas))
vogais_nomes = ['A', 'E', 'I', 'O', 'U']

for i, vogal in enumerate(vogais_arredondadas_vetores):
    saidas_rede = predict(vogal)
    print(f"Vogal arredondada {i+1}: {saidas_rede}")
    print(f"Classe predita: {vogais_nomes[saidas_rede.index(max(saidas_rede))]}")
    print("-" * 50)

# Visualização dos padrões arredondados
print("\n=== VISUALIZAÇÃO DOS PADRÕES ARREDONDADOS ===")
for i, (nome, padrao) in enumerate(zip(vogais_nomes, vogais_arredondadas)):
    print(f"\nVogal {nome} (arredondada):")
    for linha in padrao.strip().split('\n'):
        print(linha.strip())