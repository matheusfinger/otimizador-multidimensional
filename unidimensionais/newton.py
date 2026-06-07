import sympy as sp

def newton(phi_sym, alpha0, tol=1e-6, max_iter=100):
    """
    Método de Newton para busca unidimensional usando SymPy.
    Encontra α* que minimiza φ(α) usando derivadas primeira e segunda simbólicas.
    
    Args:
        phi_sym: expressão simbólica φ(α) do SymPy
        alpha0: chute inicial (float)
        tol: tolerância para o critério de parada
        max_iter: número máximo de iterações
    
    Returns:
        dicionário com resultados
    """
    # Detecta o símbolo da expressão automaticamente
    free_syms = list(phi_sym.free_symbols)
    if len(free_syms) == 0:
        # Função constante: φ'=0, já está no mínimo
        return {
            'alpha': alpha0,
            'phi_alpha': float(phi_sym),
            'iteracoes': 0,
            'historico': [],
            'convergiu': True,
            'aviso': 'Função constante'
        }
    if len(free_syms) > 1:
        raise ValueError(f"phi_sym deve ter 1 variável livre, encontradas: {free_syms}")
    alpha_sym = free_syms[0]

    # Calcula derivadas simbólicas
    phi_prime_sym = sp.diff(phi_sym, alpha_sym)      # φ'(α)
    phi_double_prime_sym = sp.diff(phi_sym, alpha_sym, 2)  # φ''(α)
    
    # Converte para funções numéricas
    phi_func = sp.lambdify(alpha_sym, phi_sym, modules=['numpy'])
    phi_prime_func = sp.lambdify(alpha_sym, phi_prime_sym, modules=['numpy'])
    phi_double_prime_func = sp.lambdify(alpha_sym, phi_double_prime_sym, modules=['numpy'])
    
    alpha = alpha0
    historico = []
    
    for k in range(max_iter):
        # Avalia as funções numericamente
        phi_val = float(phi_func(alpha))
        phi_prime_val = float(phi_prime_func(alpha))
        phi_double_prime_val = float(phi_double_prime_func(alpha))
        
        # Registra iteração
        historico.append({
            'iter': k + 1,
            'alpha': alpha,
            'phi_alpha': phi_val,
            'phi_prime': phi_prime_val,
            'phi_double_prime': phi_double_prime_val
        })
        
        # Critério de parada: gradiente próximo de zero
        if abs(phi_prime_val) < tol:
            return {
                'alpha': alpha,
                'phi_alpha': phi_val,
                'iteracoes': k + 1,
                'historico': historico,
                'convergiu': True
            }
        
        # Evita divisão por zero
        if abs(phi_double_prime_val) < 1e-12:
            # Se φ''(α) = 0, não é possível usar Newton
            # Retorna com aviso
            return {
                'alpha': alpha,
                'phi_alpha': phi_val,
                'iteracoes': k + 1,
                'historico': historico,
                'convergiu': False,
                'aviso': 'Derivada segunda igual a zero'
            }
        
        # Atualização de Newton
        alpha_novo = alpha - phi_prime_val / phi_double_prime_val
        
        # Critério de parada por mudança pequena em α
        if abs(alpha_novo - alpha) < tol:
            alpha = alpha_novo
            phi_val = float(phi_func(alpha))
            return {
                'alpha': alpha,
                'phi_alpha': phi_val,
                'iteracoes': k + 1,
                'historico': historico,
                'convergiu': True
            }
        
        alpha = alpha_novo
    
    # Se estourar o limite de iterações
    return {
        'alpha': alpha,
        'phi_alpha': float(phi_func(alpha)),
        'iteracoes': max_iter,
        'historico': historico,
        'convergiu': False,
        'aviso': 'Número máximo de iterações atingido'
    }