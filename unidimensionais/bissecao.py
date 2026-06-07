import numpy as np

def bissecao(phi, a, b, tol=1e-6, max_iter=100):
    """
    Método da Bisseção sem derivadas (Divisão do Intervalo pela Metade).
    Avalia pontos intermediários y e z para determinar a direção de descida.
    """
    alpha_a = a
    alpha_b = b
    historico = []
    
    for k in range(max_iter):
        # 1. Ponto central do intervalo
        alpha_c = (alpha_a + alpha_b) / 2
        phi_c = phi(alpha_c)
        
        # 2. Avalia o ponto y (meio do caminho entre a e c)
        alpha_y = (alpha_a + alpha_c) / 2
        phi_y = phi(alpha_y)
        
        # Variáveis auxiliares para o registro
        alpha_z = None
        phi_z = None
        
        # 3. Lógica de eliminação de região ditada pelo professor
        if phi_y < phi_c:
            # A função já desceu na primeira metade, o mínimo está entre 'a' e 'c'
            novo_a = alpha_a
            novo_b = alpha_c
        else:
            # Se não desceu em y, precisamos testar z (meio do caminho entre c e b)
            alpha_z = (alpha_c + alpha_b) / 2
            phi_z = phi(alpha_z)
            
            if phi_z < phi_c:
                # A função desceu na segunda metade, o mínimo está entre 'c' e 'b'
                novo_a = alpha_c
                novo_b = alpha_b
            else:
                # O mínimo está concentrado no meio do intervalo, entre y e z
                novo_a = alpha_y
                novo_b = alpha_z
                
        # 4. Registra a iteração (incluindo phi_a e phi_b para compatibilidade com sua UI)
        historico.append({
            'iter': k + 1,
            'a': alpha_a,
            'b': alpha_b,
            'c': alpha_c,
            'y': alpha_y,
            'z': alpha_z if alpha_z is not None else float('nan'),
            'phi_a': phi(alpha_a),
            'phi_b': phi(alpha_b),
            'phi_c': phi_c,
            'intervalo': abs(alpha_b - alpha_a)
        })
        
        # 5. Atualiza o intervalo para a próxima iteração
        alpha_a = novo_a
        alpha_b = novo_b
        
        # 6. Critério de Parada
        if abs(alpha_b - alpha_a) < tol:
            alpha_otimo = (alpha_a + alpha_b) / 2
            return {
                'alpha': alpha_otimo,
                'phi_alpha': phi(alpha_otimo),
                'iteracoes': k + 1,
                'historico': historico,
                'convergiu': True
            }
            
    # Se estourar o limite de iterações
    alpha_otimo = (alpha_a + alpha_b) / 2
    return {
        'alpha': alpha_otimo,
        'phi_alpha': phi(alpha_otimo),
        'iteracoes': max_iter,
        'historico': historico,
        'convergiu': False
    }

# Teste com a sua função parametrizada
if __name__ == "__main__":
    # Função do seu teste: 22α² - 10α + 2
    def phi_exemplo(alpha):
        return 22 * (alpha**2) - 10 * alpha + 2
    
    resultado = bissecao(phi_exemplo, a=0, b=3, tol=1e-4)
    
    print(f"α* = {resultado['alpha']:.8f}")
    print(f"φ(α*) = {resultado['phi_alpha']:.8f}")
    print(f"Iterações: {resultado['iteracoes']}")