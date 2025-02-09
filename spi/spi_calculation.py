import numpy as np
from scipy.stats import gamma, norm
from scipy.optimize import minimize

def neg_log_likelihood(params, precip_pos_norm):
    """Calcula a verossimilhança negativa para a distribuição gama"""
    a, scale = params
    if a <= 0 or scale <= 0:
        return np.inf  # Evita valores inválidos
    return -np.sum(gamma.logpdf(precip_pos_norm, a, 0, scale))

def calculate_spi_gamma(precip_series):
    """Calcula SPI com ajustes robustos para casos extremos"""
    precip = np.array(precip_series)
    valid = ~np.isnan(precip)

    if np.sum(valid) < 30:
        return np.full_like(precip, np.nan)

    precip_valid = precip[valid]
    zeros = precip_valid == 0
    q = np.mean(zeros)

    if q == 1:
        return np.full_like(precip, -3.0)

    precip_pos = precip_valid[~zeros]
    if len(precip_pos) < 2:
        return np.full_like(precip, np.nan)

    max_precip = np.max(precip_pos)
    precip_pos_norm = precip_pos / max_precip

    try:
        result = minimize(
            neg_log_likelihood,
            x0=[1.0, 1.0],
            bounds=[(1e-6, None), (1e-6, None)],
            args=(precip_pos_norm,)  # Passa os dados normalizados como argumento extra
        )
        a, scale = result.x
        scale *= max_precip  # Desfaz a normalização

        cdf_pos = gamma.cdf(precip_pos, a, 0, scale)
        cdf_full = np.zeros_like(precip_valid)
        cdf_full[~zeros] = q + (1 - q) * cdf_pos
        cdf_full[zeros] = q

        spi = np.full_like(precip, np.nan)
        spi[valid] = norm.ppf(cdf_full)

    except Exception as e:
        print(f"Erro no cálculo do SPI: {e}")
        spi = np.full_like(precip, np.nan)

    return spi