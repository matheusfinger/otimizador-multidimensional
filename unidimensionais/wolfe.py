import sympy as sp
import numpy as np


def _interp_cubica(alpha_lo, alpha_hi, phi_lo, phi_hi, phi_prime_lo, phi_prime_hi):
    """Interpolação cúbica de Hermite entre (α_lo, φ_lo, φ'_lo) e (α_hi, φ_hi, φ'_hi)."""
    d1 = phi_prime_lo + phi_prime_hi - 3 * (phi_lo - phi_hi) / (alpha_lo - alpha_hi)
    d2 = d1 ** 2 - phi_prime_lo * phi_prime_hi
    if d2 < 0:
        return (alpha_lo + alpha_hi) / 2
    raiz = np.sqrt(d2)
    if alpha_lo < alpha_hi:
        return alpha_hi - (alpha_hi - alpha_lo) * ((phi_prime_hi + raiz - d1) / (phi_prime_hi - phi_prime_lo + 2 * raiz))
    return alpha_lo - (alpha_lo - alpha_hi) * ((phi_prime_lo + raiz - d1) / (phi_prime_lo - phi_prime_hi + 2 * raiz))


def _zoom(phi, phi_prime, alpha_lo, alpha_hi, phi_0, phi_prime_0, c1, c2, tol, max_iter, forte=True):
    """Fase de zoom: refina α no intervalo [α_lo, α_hi] até satisfazer Wolfe."""
    historico = []
    phi_lo = phi(alpha_lo)
    phi_prime_lo = phi_prime(alpha_lo)

    for k in range(max_iter):
        if abs(alpha_hi - alpha_lo) < tol:
            alpha = (alpha_lo + alpha_hi) / 2
            return alpha, historico, True

        alpha_j = _interp_cubica(alpha_lo, alpha_hi, phi_lo, phi(alpha_hi), phi_prime_lo, phi_prime(alpha_hi))
        alpha_j = max(alpha_lo + tol, min(alpha_j, alpha_hi - tol))

        phi_j = phi(alpha_j)
        armijo_violada = phi_j > phi_0 + c1 * alpha_j * phi_prime_0
        nao_decrescente = phi_j >= phi_lo

        historico.append({
            'fase': 'zoom',
            'iter': k + 1,
            'alpha_lo': alpha_lo,
            'alpha_hi': alpha_hi,
            'alpha': alpha_j,
            'phi_alpha': phi_j,
            'phi_prime': phi_prime(alpha_j),
            'armijo': not armijo_violada,
            'curvatura': False,
        })

        if armijo_violada or nao_decrescente:
            alpha_hi = alpha_j
        else:
            phi_prime_j = phi_prime(alpha_j)
            curvatura_ok = (
                abs(phi_prime_j) <= -c2 * phi_prime_0
                if forte
                else phi_prime_j >= c2 * phi_prime_0
            )
            historico[-1]['curvatura'] = curvatura_ok

            if curvatura_ok:
                return alpha_j, historico, True

            if phi_prime_j * (alpha_hi - alpha_lo) >= 0:
                alpha_hi = alpha_lo

            alpha_lo = alpha_j
            phi_lo = phi_j
            phi_prime_lo = phi_prime_j

    alpha = (alpha_lo + alpha_hi) / 2
    return alpha, historico, False


def wolfe(phi, phi_prime, alpha_init=1.0, phi_0=None, phi_prime_0=None,
          c1=1e-4, c2=0.9, alpha_max=10.0, tol=1e-6, max_iter=100, forte=True):
    """
    Busca unidimensional por linha com condições de Wolfe.

    Condição de Armijo (suficiente decréscimo):
        φ(α) ≤ φ(0) + c1 · α · φ'(0)

    Condição de curvatura fraca:
        φ'(α) ≥ c2 · φ'(0)

    Condição de curvatura forte:
        |φ'(α)| ≤ c2 · |φ'(0)|

    Args:
        phi: função φ(α) -> float
        phi_prime: função φ'(α) -> float
        alpha_init: passo inicial
        phi_0: φ(0); calculado se None
        phi_prime_0: φ'(0); calculado se None
        c1: parâmetro de Armijo (0 < c1 < c2 < 1)
        c2: parâmetro de curvatura
        alpha_max: limite superior para expansão do passo
        tol: tolerância na fase zoom
        max_iter: iterações máximas na expansão
        forte: True para Wolfe forte, False para Wolfe fraco

    Returns:
        dicionário com α*, histórico, flags de convergência e verificação das condições
    """
    if not (0 < c1 < c2 < 1):
        raise ValueError("Exija 0 < c1 < c2 < 1 para as condições de Wolfe.")

    phi_0 = float(phi(0)) if phi_0 is None else float(phi_0)
    phi_prime_0 = float(phi_prime(0)) if phi_prime_0 is None else float(phi_prime_0)

    if phi_prime_0 >= 0:
        return {
            'alpha': 0.0,
            'phi_alpha': phi_0,
            'phi_prime_alpha': phi_prime_0,
            'iteracoes': 0,
            'historico': [],
            'convergiu': False,
            'aviso': "φ'(0) ≥ 0: a direção não é de descida. Escolha outra direção d.",
            'condicoes': {'armijo': True, 'curvatura': False, 'descida': False},
        }

    historico = []
    alpha = alpha_init
    alpha_prev = 0.0
    phi_prev = phi_0

    for k in range(max_iter):
        if alpha > alpha_max:
            alpha = alpha_max

        phi_alpha = float(phi(alpha))
        phi_prime_alpha = float(phi_prime(alpha))

        armijo_ok = phi_alpha <= phi_0 + c1 * alpha * phi_prime_0
        if forte:
            curvatura_ok = abs(phi_prime_alpha) <= -c2 * phi_prime_0
        else:
            curvatura_ok = phi_prime_alpha >= c2 * phi_prime_0

        historico.append({
            'fase': 'expansao',
            'iter': k + 1,
            'alpha': alpha,
            'alpha_prev': alpha_prev,
            'phi_alpha': phi_alpha,
            'phi_prime': phi_prime_alpha,
            'armijo': armijo_ok,
            'curvatura': curvatura_ok,
        })

        if armijo_ok and curvatura_ok:
            return _montar_resultado(
                alpha, phi_alpha, phi_prime_alpha, historico, True,
                phi_0, phi_prime_0, c1, c2, forte
            )

        if (not armijo_ok) or (phi_alpha >= phi_prev and k > 0):
            alpha_star, hist_zoom, conv = _zoom(
                phi, phi_prime, alpha_prev, alpha, phi_0, phi_prime_0,
                c1, c2, tol, max_iter, forte
            )
            historico.extend(hist_zoom)
            return _montar_resultado(
                alpha_star, float(phi(alpha_star)), float(phi_prime(alpha_star)),
                historico, conv, phi_0, phi_prime_0, c1, c2, forte
            )

        if phi_prime_alpha >= 0:
            alpha_star, hist_zoom, conv = _zoom(
                phi, phi_prime, alpha_prev, alpha, phi_0, phi_prime_0,
                c1, c2, tol, max_iter, forte
            )
            historico.extend(hist_zoom)
            return _montar_resultado(
                alpha_star, float(phi(alpha_star)), float(phi_prime(alpha_star)),
                historico, conv, phi_0, phi_prime_0, c1, c2, forte
            )

        alpha_prev = alpha
        phi_prev = phi_alpha
        alpha = min(2.0 * alpha, alpha_max)

    alpha_star = alpha_prev if alpha_prev > 0 else alpha
    return _montar_resultado(
        alpha_star, float(phi(alpha_star)), float(phi_prime(alpha_star)),
        historico, False, phi_0, phi_prime_0, c1, c2, forte,
        aviso='Número máximo de iterações na expansão atingido'
    )


def _montar_resultado(alpha, phi_alpha, phi_prime_alpha, historico, convergiu,
                      phi_0, phi_prime_0, c1, c2, forte, aviso=None):
    limite_armijo = phi_0 + c1 * alpha * phi_prime_0
    armijo_ok = phi_alpha <= limite_armijo + 1e-12
    if forte:
        curvatura_ok = abs(phi_prime_alpha) <= -c2 * phi_prime_0 + 1e-12
    else:
        curvatura_ok = phi_prime_alpha >= c2 * phi_prime_0 - 1e-12

    resultado = {
        'alpha': alpha,
        'phi_alpha': phi_alpha,
        'phi_prime_alpha': phi_prime_alpha,
        'phi_0': phi_0,
        'phi_prime_0': phi_prime_0,
        'iteracoes': len(historico),
        'historico': historico,
        'convergiu': convergiu and armijo_ok and curvatura_ok,
        'condicoes': {
            'armijo': armijo_ok,
            'curvatura': curvatura_ok,
            'descida': phi_prime_0 < 0,
            'forte': forte,
            'limite_armijo': limite_armijo,
        },
        'parametros': {'c1': c1, 'c2': c2},
    }
    if aviso:
        resultado['aviso'] = aviso
    return resultado


def criar_phi_e_derivada(phi_sym):
    """
    A partir de φ(α) simbólica (SymPy), retorna funções numéricas φ e φ'.
    """
    alpha_sym = sp.Symbol('α', real=True)
    phi_prime_sym = sp.diff(phi_sym, alpha_sym)

    phi_func = sp.lambdify(alpha_sym, phi_sym, modules=['numpy'])
    phi_prime_func = sp.lambdify(alpha_sym, phi_prime_sym, modules=['numpy'])

    def phi(alpha):
        return float(phi_func(alpha))

    def phi_prime(alpha):
        return float(phi_prime_func(alpha))

    return phi, phi_prime, phi_prime_sym


if __name__ == "__main__":
    # Exemplo: φ(α) = 22α² - 10α + 2, mínimo em α = 10/44
    alpha = sp.Symbol('α', real=True)
    phi_sym = 22 * alpha ** 2 - 10 * alpha + 2
    phi, phi_p, _ = criar_phi_e_derivada(phi_sym)

    res = wolfe(phi, phi_p, alpha_init=1.0, c1=1e-4, c2=0.9)
    print(f"α* = {res['alpha']:.8f}")
    print(f"φ(α*) = {res['phi_alpha']:.8f}")
    print(f"Armijo: {res['condicoes']['armijo']}, Curvatura: {res['condicoes']['curvatura']}")