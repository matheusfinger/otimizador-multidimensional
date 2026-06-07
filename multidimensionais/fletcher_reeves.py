import numpy as np

def fletcher_reeves(f, grad, x0, calcular_alpha_func, tol=1e-5, max_iter=100):
    """
    Método do Gradiente Conjugado (Fletcher-Reeves).
    calcular_alpha_func: função que recebe phi(alpha) e retorna o alpha ótimo
    """
    x = np.array(x0, dtype=float)
    g = np.array(grad(x))
    d = -g
    
    historico = [x.copy()]
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return {'x_otimo': x, 'iteracoes': k, 'historico': historico, 'convergiu': True}
            
        # 1. Cria a função unidimensional puramente numérica (muito mais rápido que SymPy)
        def phi(alpha):
            return f(x + alpha * d)
            
        # 2. Chama o método que o usuário escolheu na interface (Bisseção, Newton, etc)
        alpha = calcular_alpha_func(phi, x, d)
            
        # Atualiza o ponto
        x_new = x + alpha * d
        historico.append(x_new.copy())
        
        g_new = np.array(grad(x_new))
        
        # Beta FR
        beta_fr = np.dot(g_new, g_new) / np.dot(g, g)
        
        d = -g_new + beta_fr * d
        x = x_new
        g = g_new
        
    return {'x_otimo': x, 'iteracoes': max_iter, 'historico': historico, 'convergiu': False}