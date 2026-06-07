import os
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import json

# Importe os seus métodos aqui (ajuste os caminhos se necessário)
from multidimensionais.fletcher_reeves import fletcher_reeves
from multidimensionais.bfgs import bfgs
from unidimensionais.wolfe import wolfe

def executar_e_salvar_grafico(func_str, x0, nome_metodo, prefixo_arquivo):
    """
    Executa a otimização e salva um gráfico duplo (Contorno + Decaimento) como PNG.
    """
    # 1. Configura a Função Simbólica garantindo a ordem x1, x2
    x1, x2 = sp.symbols('x1 x2')
    f_sym = sp.sympify(func_str)
    symbols = [x1, x2] # Força a ordem correta para o NumPy

    f_num = sp.lambdify(symbols, f_sym, 'numpy')
    grad_sym = [sp.diff(f_sym, var) for var in symbols]
    grad_num = sp.lambdify(symbols, grad_sym, 'numpy')

    def f_eval(x_arr): return float(f_num(*x_arr))
    def grad_eval(x_arr): return np.array(grad_num(*x_arr), dtype=float)

    # 2. Configura a Busca Linear (Wolfe é a mais recomendada para esses métodos)
    def calcular_alpha(phi_func, x_k, d_k):
        def phi_prime(alpha):
            g_val = grad_eval(x_k + alpha * d_k)
            return float(np.dot(g_val, d_k))
        res = wolfe(phi_func, phi_prime, alpha_init=1.0)
        return res['alpha']

    # 3. Roda a Otimização
    tol = 1e-5
    max_iter = 500
    
    if nome_metodo == "BFGS":
        res = bfgs(f_eval, grad_eval, x0, calcular_alpha, tol, max_iter)
    else:
        res = fletcher_reeves(f_eval, grad_eval, x0, calcular_alpha, tol, max_iter)

    historico = np.array(res['historico'])
    valores_f = [f_eval(p) for p in historico]

    # 4. Desenha o Gráfico
    fig = plt.figure(figsize=(12, 5))

    # --- Lado Direito: Decaimento ---
    ax2 = fig.add_subplot(122)
    ax2.set_title(f"Decaimento - {nome_metodo}")
    ax2.set_xlabel("Iteração (k)")
    ax2.set_ylabel("f(x)")
    
    ax2.set_xlim(0, max(1, len(historico) - 1))
    margem_y = (max(valores_f) - min(valores_f)) * 0.1 if max(valores_f) != min(valores_f) else 1.0
    ax2.set_ylim(min(valores_f) - margem_y, max(valores_f) + margem_y)
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    ax2.plot(range(len(historico)), valores_f, 'b-', lw=2, marker='o', markersize=4)

    # --- Lado Esquerdo: Curvas de Nível ---
    ax1 = fig.add_subplot(121)
    # Aumenta a margem para vermos bem as curvas em volta do caminho
    x_min, x_max = np.min(historico[:, 0]) - 2, np.max(historico[:, 0]) + 2
    y_min, y_max = np.min(historico[:, 1]) - 2, np.max(historico[:, 1]) + 2
    
    X, Y = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    Z = f_num(X, Y)

    ax1.contour(X, Y, Z, levels=40, cmap='viridis')
    
    # Plota a linha vermelha do histórico
    ax1.plot(historico[:, 0], historico[:, 1], 'r-', lw=2)
    # Marca Início (Verde) e Fim (Estrela Azul)
    ax1.plot(historico[0, 0], historico[0, 1], 'go', markersize=8, label='Início')
    ax1.plot(historico[-1, 0], historico[-1, 1], 'b*', markersize=12, label='Fim')
    
    ax1.set_xlim(x_min, x_max)
    ax1.set_ylim(y_min, y_max)
    ax1.set_xlabel('x1')
    ax1.set_ylabel('x2')
    ax1.set_title(f"Trajetória - {nome_metodo}\n(Iterações: {res['iteracoes']})")
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend()

    # Salva a imagem
    fig.subplots_adjust(left=0.1, right=0.95, bottom=0.15, top=0.85, wspace=0.3)
    caminho_arquivo = f"{prefixo_arquivo}_{nome_metodo}.png"
    plt.savefig(caminho_arquivo, dpi=150)
    plt.close(fig)
    print(f"  -> Salvo: {caminho_arquivo} (Iterações: {res['iteracoes']}, Convergiu: {res['convergiu']})")
    
    return {
        "metodo": nome_metodo,
        "funcao": func_str,
        "ponto_inicial": x0,
        "ponto_otimo": res['x_otimo'].tolist(), # Converte array do numpy para lista
        "iteracoes": res['iteracoes'],
        "convergiu": res['convergiu'],
        "historico_f": valores_f # O decaimento passo a passo
    }


if __name__ == '__main__':
    # Cria a pasta para organizar as imagens
    os.makedirs("resultados_imagens", exist_ok=True)

    # Dicionário com os dados exatos do seu trabalho
    testes = [
        {
            'nome': 'func1',
            'expr': "(x1-2)**4 + (x1-2*x2)**2",
            'pontos': [[0, 3], [-1, -1], [10, -10]] # [10, -10]: Teste de distância extrema
        },
        {
            'nome': 'func2_rosenbrock',
            'expr': "100 * (x2 - x1**2)**2 + (1 - x1)**2",
            'pontos': [[-5, 5], [10, 10], [0, 0]] # [0, 0]: Início do vale plano
        },
        {
            'nome': 'func3_trig',
            'expr': "(x1**2 + x2**2 + x1*x2)**2 + (sin(x1)**2) + (cos(x2)**2)",
            'pontos': [[3, 1], [2, -2], [0, 0]] # [0, 0]: Ponto perigoso para funções trigonométricas
        },
        {
            'nome': 'func4_polinomio',
            'expr': "1.41 * x1**4 - 12.76 * x1**3 + 39.91 * x1**2 - 51.93 * x1 + 24.37 + (x2 - 3.9)**2",
            'pontos': [[0, 0], [5, 5], [-5, 0]] # [-5, 0]: Força a busca a cair em outro mínimo local
        }
    ]

    print("Iniciando geração de gráficos para o relatório...")

    # Lista para guardar os dados para a IA
    dados_completos = []
    
    for teste in testes:
        print(f"\nProcessando {teste['nome']}...")
        for i, pt in enumerate(teste['pontos']):
            prefixo = f"resultados_imagens/{teste['nome']}_pt{i+1}"
            
            try:
                # Salva os retornos nas variáveis
                dado_bfgs = executar_e_salvar_grafico(teste['expr'], pt, "BFGS", prefixo)
                dado_fr = executar_e_salvar_grafico(teste['expr'], pt, "Fletcher-Reeves", prefixo)
                
                # Adiciona na lista geral
                dados_completos.append(dado_bfgs)
                dados_completos.append(dado_fr)
                
            except Exception as e:
                print(f"Erro ao processar o ponto {pt}: {e}")
                
    # --- SALVA O ARQUIVO JSON PARA A IA ---
    caminho_json = "resultados_imagens/dados_experimentos.json"
    with open(caminho_json, 'w', encoding='utf-8') as f:
        json.dump(dados_completos, f, indent=4, ensure_ascii=False)
        
    print(f"\nTodas as imagens e o arquivo '{caminho_json}' foram gerados!")