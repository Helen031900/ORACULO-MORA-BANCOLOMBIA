"""
Núcleo del modelo predictivo — Desafío IA 2030 · Bancolombia / Grupo Cibest

¿Qué predice?  El indicador de cartera vencida a más de 30 días de la cartera de
CONSUMO (ICV30 consumo, en %), trimestre a trimestre.

Modelo (regresión lineal, mínimos cuadrados):
    ICV_t = a + b · ICV_{t-1} + c · ΔTasa_{t-3}
donde ΔTasa_{t-3} = cambio de la tasa de política monetaria del Banco de la República
en los 2 trimestres que terminan 3 trimestres antes de t.
Idea de negocio: la mora tiene inercia (b) y reacciona, con retraso, a las subidas
o bajadas de tasas (c), porque encarecen las cuotas de los créditos y enfrían la economía.

Modelo de referencia ("ingenuo"): ICV_t = ICV_{t-1} (el futuro se parece al último dato).
"""
import numpy as np
import pandas as pd

ARCHIVO_DATOS = "datos_cartera_bancolombia.csv"
REZAGO = 3          # trimestres de retraso entre el movimiento de tasas y la mora
VENTANA_CAMBIO = 2  # el cambio de tasa se mide en 2 trimestres (un semestre)
TRIMESTRES_FUTUROS = ["3T26", "4T26", "1T27", "2T27", "3T27", "4T27"]
TASA_3T26_OBSERVADA = 12.25  # decisión BanRep del 30-sep-2026 (ya ocurrió)


def cargar_datos(ruta=ARCHIVO_DATOS):
    return pd.read_csv(ruta).reset_index(drop=True)


def cambio_tasa(tasa, ventana=None):
    ventana = VENTANA_CAMBIO if ventana is None else ventana
    tasa = np.asarray(tasa, dtype=float)
    return np.array([tasa[i] - tasa[max(0, i - ventana)] for i in range(len(tasa))])


def armar_matriz(y, tasa, rezago=REZAGO):
    d = cambio_tasa(tasa)
    X, obj = [], []
    for t in range(max(1, rezago), len(y)):
        X.append([1.0, y[t - 1], d[t - rezago]])
        obj.append(y[t])
    return np.array(X), np.array(obj)


def ajustar(y, tasa, rezago=REZAGO):
    X, obj = armar_matriz(y, tasa, rezago)
    coef, *_ = np.linalg.lstsq(X, obj, rcond=None)
    resid = obj - X @ coef
    r2 = 1 - np.sum(resid ** 2) / np.sum((obj - obj.mean()) ** 2)
    return coef, resid, float(r2)


def pronosticar(coef, y_hist, tasa_total, pasos, rezago=REZAGO, choques=None):
    """Pronóstico recursivo. tasa_total = historia de tasas + senda futura supuesta."""
    d = cambio_tasa(tasa_total)
    y = list(y_hist)
    n0 = len(y_hist)
    for h in range(pasos):
        t = n0 + h
        pred = coef[0] + coef[1] * y[t - 1] + coef[2] * d[t - rezago]
        if choques is not None:
            pred += choques[h]
        y.append(pred)
    return np.array(y[n0:])


def backtest(y, tasa, inicio=13, horizonte=1, rezago=REZAGO):
    """Validación rodante: en cada origen el modelo SOLO ve datos hasta ese momento."""
    filas = []
    for origen in range(inicio, len(y) - horizonte + 1):
        coef, _, _ = ajustar(y[:origen], tasa[:origen], rezago)
        pred = pronosticar(coef, y[:origen], tasa, horizonte, rezago)[-1]
        objetivo = origen + horizonte - 1
        filas.append({"idx": objetivo, "real": y[objetivo], "modelo": pred, "ingenuo": y[origen - 1]})
    return pd.DataFrame(filas)


def metricas(real, pred):
    e = np.asarray(real) - np.asarray(pred)
    return {"MAE": float(np.mean(np.abs(e))),
            "RMSE": float(np.sqrt(np.mean(e ** 2))),
            "MAPE": float(np.mean(np.abs(e) / np.asarray(real)) * 100)}


def senda_tasa(tasa_hist, tasa_4t26, tasa_1t27, cambio_trimestral):
    """Historia + 3T26 observada (12,25%) + supuestos editables hasta 4T27."""
    futura = [TASA_3T26_OBSERVADA, tasa_4t26, tasa_1t27]
    nivel = tasa_1t27
    for _ in range(3):  # 2T27, 3T27, 4T27
        nivel = max(0.0, nivel + cambio_trimestral)
        futura.append(round(nivel, 2))
    return np.concatenate([np.asarray(tasa_hist, dtype=float), futura]), futura


def bandas_bootstrap(coef, resid, y_hist, tasa_total, pasos, n=3000, semilla=7):
    """Rango de incertidumbre (P10–P90) re-muestreando los errores históricos del modelo."""
    rng = np.random.default_rng(semilla)
    caminos = np.empty((n, pasos))
    for i in range(n):
        caminos[i] = pronosticar(coef, y_hist, tasa_total, pasos,
                                 choques=rng.choice(resid, size=pasos, replace=True))
    return np.percentile(caminos, 10, axis=0), np.percentile(caminos, 90, axis=0)


ESCENARIOS = {
    "Conservador": {"tasa_4t26": 12.75, "tasa_1t27": 13.25, "cambio": 0.0, "choque_pp": 0.50,
                    "texto": "La inflación no cede: BanRep sube a 12,75% (4T26) y 13,25% (1T27) y se queda ahí; "
                             "además, un golpe puntual de +0,5 pp a la mora en 1T27 (empleo e ingresos más débiles)."},
    "Base":        {"tasa_4t26": 12.25, "tasa_1t27": 12.25, "cambio": 0.0, "choque_pp": 0.0,
                    "texto": "La tasa se queda en 12,25% hasta 4T27 y no hay choques adicionales."},
    "Optimista":   {"tasa_4t26": 12.25, "tasa_1t27": 11.75, "cambio": -0.50, "choque_pp": -0.20,
                    "texto": "La inflación cede: recortes desde 1T27 (–50 pb por trimestre, hasta 10,25% en 4T27) "
                             "y mejor comportamiento de pago (–0,2 pp puntual en 1T27)."},
}


def correr_escenario(coef, y, tasa, tasa_4t26, tasa_1t27, cambio, choque_pp):
    tasa_total, futura = senda_tasa(tasa, tasa_4t26, tasa_1t27, cambio)
    choques = np.zeros(len(TRIMESTRES_FUTUROS))
    choques[2] = choque_pp  # golpe puntual en 1T27; se arrastra por la inercia de la mora
    pred = pronosticar(coef, y, tasa_total, len(TRIMESTRES_FUTUROS), choques=choques)
    return pred, futura, tasa_total
