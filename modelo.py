"""
Entrena, evalúa y proyecta el modelo. Ejecutar:  python modelo.py
Genera: resultados/metricas.json, resultados/backtest.csv, resultados/pronostico_escenarios.csv,
        resultados/comparacion_especificaciones.csv y gráficos PNG.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import oraculo as o

os.makedirs("resultados", exist_ok=True)
df = o.cargar_datos()
y = df["icv30_consumo_pct"].values
tasa = df["tasa_banrep_pct"].values
trim = list(df["trimestre"])

# 1) Ajuste con toda la historia
coef, resid, r2 = o.ajustar(y, tasa)
print(f"Coeficientes: a={coef[0]:.3f}  b(inercia)={coef[1]:.3f}  c(efecto tasa)={coef[2]:.3f}  R2={r2:.3f}")

# 2) Validación rodante (1T24 a 2T26): 1 y 4 trimestres adelante, vs modelo ingenuo
res = {}
bts = []
for h in (1, 4):
    bt = o.backtest(y, tasa, inicio=13, horizonte=h)
    bt["horizonte"] = h
    bt["trimestre"] = [trim[i] for i in bt["idx"]]
    bts.append(bt)
    res[f"{h}_trimestre"] = {"n_pruebas": int(len(bt)),
                             "modelo": o.metricas(bt.real, bt.modelo),
                             "ingenuo": o.metricas(bt.real, bt.ingenuo)}
pd.concat(bts)[["horizonte", "trimestre", "real", "modelo", "ingenuo"]].round(3).to_csv(
    "resultados/backtest.csv", index=False)

# 3) Transparencia: todas las especificaciones probadas (se eligió la de menor error a 4 trimestres)
filas = []
for ventana in (2, 4):
    for rezago in (1, 2, 3, 4):
        o.VENTANA_CAMBIO = ventana
        b1 = o.backtest(y, tasa, 13, 1, rezago)
        b4 = o.backtest(y, tasa, 13, 4, rezago)
        filas.append({"cambio_tasa_trimestres": ventana, "rezago_trimestres": rezago,
                      "MAE_1T_modelo": o.metricas(b1.real, b1.modelo)["MAE"],
                      "MAE_4T_modelo": o.metricas(b4.real, b4.modelo)["MAE"],
                      "MAE_1T_ingenuo": o.metricas(b1.real, b1.ingenuo)["MAE"],
                      "MAE_4T_ingenuo": o.metricas(b4.real, b4.ingenuo)["MAE"]})
o.VENTANA_CAMBIO = 2
pd.DataFrame(filas).round(3).to_csv("resultados/comparacion_especificaciones.csv", index=False)

# 4) Escenarios + rango de incertidumbre del escenario base
tabla = {"trimestre": o.TRIMESTRES_FUTUROS}
for nombre, p in o.ESCENARIOS.items():
    pred, futura, total = o.correr_escenario(coef, y, tasa, p["tasa_4t26"], p["tasa_1t27"], p["cambio"], p["choque_pp"])
    tabla[f"icv_{nombre.lower()}"] = pred
    tabla[f"tasa_{nombre.lower()}"] = futura
    if nombre == "Base":
        p10, p90 = o.bandas_bootstrap(coef, resid, y, total, len(o.TRIMESTRES_FUTUROS))
        tabla["base_p10"], tabla["base_p90"] = p10, p90
pron = pd.DataFrame(tabla).round(2)
pron.to_csv("resultados/pronostico_escenarios.csv", index=False)
print(pron.to_string(index=False))

json.dump({"coeficientes": {"a": coef[0], "b_inercia": coef[1], "c_efecto_tasa": coef[2]},
           "R2_en_muestra": r2, "observaciones_entrenamiento": int(len(resid)),
           "desviacion_error": float(np.std(resid, ddof=3)),
           "validacion_rodante": res}, open("resultados/metricas.json", "w"), indent=2, ensure_ascii=False)
print(json.dumps(res, indent=2))

# 5) Gráficos
fig, ax = plt.subplots(figsize=(10, 4.8))
x_h = np.arange(len(y)); x_f = np.arange(len(y), len(y) + 6)
ax.plot(x_h, y, "o-", color="#1f3a5f", label="Real (reportado)")
ax.fill_between(x_f, pron.base_p10, pron.base_p90, color="#9bb3cf", alpha=.4, label="Rango P10–P90 (base)")
for nombre, col in (("conservador", "#c0392b"), ("base", "#1f6fb5"), ("optimista", "#2e8b57")):
    ax.plot(np.r_[x_h[-1], x_f], np.r_[y[-1], pron[f"icv_{nombre}"]], "--", color=col, label=nombre.capitalize())
ax.axhline(6.5, color="#e67e22", lw=1, ls=":"); ax.text(1, 6.6, "Alerta amarilla 6,5%", color="#e67e22", fontsize=8)
ax.axhline(7.5, color="#c0392b", lw=1, ls=":"); ax.text(1, 7.6, "Alerta roja 7,5%", color="#c0392b", fontsize=8)
ax.set_xticks(np.r_[x_h, x_f][::2]); ax.set_xticklabels((trim + o.TRIMESTRES_FUTUROS)[::2], rotation=45)
ax.set_ylabel("ICV30 consumo (%)"); ax.set_title("Mora de consumo: historia y escenarios 3T26–4T27")
ax.legend(fontsize=8, ncol=2); fig.tight_layout(); fig.savefig("resultados/escenarios.png", dpi=150)

b1 = bts[0]
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(b1.trimestre, b1.real, "o-", color="#1f3a5f", label="Real")
ax.plot(b1.trimestre, b1.modelo, "s--", color="#1f6fb5", label="Modelo")
ax.plot(b1.trimestre, b1.ingenuo, "^:", color="#999", label="Ingenuo")
ax.set_title("Validación rodante: pronóstico a 1 trimestre (datos que el modelo no vio)")
ax.set_ylabel("ICV30 consumo (%)"); ax.legend(); fig.tight_layout(); fig.savefig("resultados/backtest.png", dpi=150)
print("Listo: carpeta resultados/")
