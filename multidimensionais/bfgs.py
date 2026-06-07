import numpy as np

def bfgs(f, grad, x0, calcular_alpha_func, tol=1e-5, max_iter=100):
    """Método Quase-Newton (BFGS)."""
    x = np.array(x0, dtype=float)
    n = len(x)
    H = np.eye(n)
    g = np.array(grad(x))
    
    historico = [x.copy()]
    
    for k in range(max_iter):
        if np.linalg.norm(g) < tol:
            return {'x_otimo': x, 'iteracoes': k, 'historico': historico, 'convergiu': True}
            
        d = -np.dot(H, g)
        
        # 1. Cria a função unidimensional
        def phi(alpha):
            return f(x + alpha * d)
            
        # 2. Executa o método escolhido pelo usuário para achar o passo
        alpha = calcular_alpha_func(phi, x, d)
            
        x_new = x + alpha * d
        historico.append(x_new.copy())
        
        g_new = np.array(grad(x_new))
        s = x_new - x
        y = g_new - g
        
        rho_bfgs = np.dot(y, s)
        if rho_bfgs > 1e-10:
            I = np.eye(n)
            A = I - np.outer(s, y) / rho_bfgs
            B = I - np.outer(y, s) / rho_bfgs
            C = np.outer(s, s) / rho_bfgs
            H = np.dot(A, np.dot(H, B)) + C
            
        x = x_new
        g = g_new
        
    return {'x_otimo': x, 'iteracoes': max_iter, 'historico': historico, 'convergiu': False}