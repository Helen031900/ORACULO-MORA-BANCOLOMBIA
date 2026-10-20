"""
ORÁCULO DE MORA — Desafío IA 2030 · Bancolombia / Grupo Cibest
Ejecutar localmente:  streamlit run app.py
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

import oraculo as o

st.set_page_config(page_title="Oráculo de Mora · Bancolombia", page_icon="📉", layout="wide")

SALDO_CONSUMO_2T26 = 53_538_432 / 1_000_000   # billones de COP (SEC 6-K 2T26 Grupo Cibest)
COBERTURA_2T26 = 132.94                          # % provisiones / cartera vencida 30 días (2T26)
UMBRAL_AMARILLO, UMBRAL_ROJO = 6.5, 7.5


@st.cache_data
def preparar():
    df = o.cargar_datos()
    y = df["icv30_consumo_pct"].values
    tasa = df["tasa_banrep_pct"].values
    coef, resid, r2 = o.ajustar(y, tasa)
    bt1 = o.backtest(y, tasa, 13, 1)
    bt4 = o.backtest(y, tasa, 13, 4)
    return df, y, tasa, coef, resid, r2, bt1, bt4


df, y, tasa, coef, resid, r2, bt1, bt4 = preparar()
trim_hist = list(df["trimestre"])
FUT = o.TRIMESTRES_FUTUROS

# ---------------- Barra lateral: supuestos editables ----------------
st.sidebar.header("🎛️ Supuestos del escenario")
preset = st.sidebar.radio("Punto de partida", list(o.ESCENARIOS.keys()), index=1, horizontal=True)
p = o.ESCENARIOS[preset]
st.sidebar.caption(p["texto"])
t4 = st.sidebar.slider("Tasa BanRep 4T26 (%)", 9.0, 15.0, p["tasa_4t26"], 0.25, key=f"t4{preset}")
t1 = st.sidebar.slider("Tasa BanRep 1T27 (%)", 8.0, 15.0, p["tasa_1t27"], 0.25, key=f"t1{preset}")
cambio = st.sidebar.slider("Cambio por trimestre desde 2T27 (puntos)", -1.0, 1.0, p["cambio"], 0.25, key=f"c{preset}")
choque = st.sidebar.slider("Golpe extra a la mora en 1T27 (pp)", -1.0, 2.0, p["choque_pp"], 0.1, key=f"s{preset}",
                           help="Algo que el modelo no ve: desempleo, reformas, desastres, cambios de política de crédito.")
st.sidebar.divider()
crec = st.sidebar.slider("Crecimiento trimestral cartera de consumo (%)", -2.0, 4.0, 1.0, 0.5)
cobertura = st.sidebar.slider("Cobertura de provisiones deseada (%)", 100.0, 160.0, COBERTURA_2T26, 1.0)
st.sidebar.caption("3T26 usa la tasa real de 12,25% (decisión BanRep 30-sep-2026).")

pred, senda, tasa_total = o.correr_escenario(coef, y, tasa, t4, t1, cambio, choque)
p10, p90 = o.bandas_bootstrap(coef, resid, y, tasa_total, len(FUT))

# Comparación fija de los 3 escenarios de referencia
ref = {n: o.correr_escenario(coef, y, tasa, e["tasa_4t26"], e["tasa_1t27"], e["cambio"], e["choque_pp"])[0]
       for n, e in o.ESCENARIOS.items()}

# ---------------- Encabezado ----------------
st.title("📉 Oráculo de Mora — Cartera de consumo")
st.markdown("**Pregunta predictiva:** ¿cuál será el indicador de cartera vencida a más de 30 días (ICV30) "
            "de la cartera de consumo de Bancolombia / Grupo Cibest entre 3T26 y 4T27, ahora que el Banco de "
            "la República volvió a subir tasas?  \n*Usuario: Vicepresidencia de Riesgos y comité de crédito de consumo.*")

pico = float(np.max(pred)); idx_pico = int(np.argmax(pred))
saldo = SALDO_CONSUMO_2T26 * (1 + crec / 100) ** np.arange(1, len(FUT) + 1)
vencida = saldo * pred / 100
vencida_hoy = SALDO_CONSUMO_2T26 * y[-1] / 100
extra_prov = (vencida[idx_pico] - vencida_hoy) * cobertura / 100

c1, c2, c3, c4 = st.columns(4)
c1.metric("ICV30 consumo hoy (2T26)", f"{y[-1]:.2f}%")
c2.metric("Pico proyectado", f"{pico:.2f}%", f"{pico - y[-1]:+.2f} pp en {FUT[idx_pico]}", delta_color="inverse")
c3.metric("Cartera vencida en el pico", f"${vencida[idx_pico]:.2f} billones",
          f"{vencida[idx_pico] - vencida_hoy:+.2f} vs hoy", delta_color="inverse")
c4.metric("Provisiones adicionales estimadas", f"${extra_prov:.2f} billones",
          help="(Cartera vencida en el pico – hoy) × cobertura. Es un supuesto ilustrativo, no un cálculo NIIF 9.")

# ---------------- Alertas con reglas explícitas ----------------
if max(senda) > 13.25 or min(senda) < 1.75:
    st.warning("⚠️ Estás usando tasas fuera del rango que el modelo vio en la historia (1,75%–13,25%). "
               "Trata el resultado como exploratorio, no como pronóstico.")
p90_max = float(np.max(p90))
if pico >= UMBRAL_ROJO:
    st.error(f"🔴 ALERTA ROJA: el escenario supera {UMBRAL_ROJO}% (nivel cercano al pico de 2023-2024). "
             "Regla: comité extraordinario, endurecer otorgamiento en segmentos de mayor riesgo y reforzar provisiones.")
elif pico >= UMBRAL_AMARILLO:
    st.warning(f"🟠 ALERTA AMARILLA: el escenario supera {UMBRAL_AMARILLO}%. Regla: revisar políticas de otorgamiento "
               "de consumo, activar programas de normalización/reestructuración temprana y seguimiento mensual.")
elif p90_max >= UMBRAL_AMARILLO:
    st.info(f"🟡 VIGILANCIA: el escenario central está bajo {UMBRAL_AMARILLO}%, pero el rango alto (P90) llega a "
            f"{p90_max:.2f}%. Regla: no expandir agresivamente el crédito de consumo sin revisión trimestral.")
else:
    st.success("🟢 NORMAL: escenario y rango alto por debajo de los umbrales. Mantener política y monitorear.")

# ---------------- Gráfico ----------------
tab1, tab2, tab3, tab4 = st.tabs(["📈 Proyección", "⚖️ Tres futuros", "🔬 ¿Qué tan confiable es?", "🧾 Datos y ética"])
with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=trim_hist, y=y, mode="lines+markers", name="Real reportado", line=dict(color="#1f3a5f")))
    fig.add_trace(go.Scatter(x=FUT + FUT[::-1], y=list(p90) + list(p10[::-1]), fill="toself",
                             fillcolor="rgba(31,111,181,.15)", line=dict(width=0), name="Rango probable P10–P90"))
    fig.add_trace(go.Scatter(x=[trim_hist[-1]] + FUT, y=[y[-1]] + list(pred), mode="lines+markers",
                             name="Tu escenario", line=dict(color="#1f6fb5", dash="dash")))
    fig.add_hline(y=UMBRAL_AMARILLO, line_dash="dot", line_color="#e67e22", annotation_text="Alerta amarilla")
    fig.add_hline(y=UMBRAL_ROJO, line_dash="dot", line_color="#c0392b", annotation_text="Alerta roja")
    fig.update_layout(height=430, yaxis_title="ICV30 consumo (%)", legend=dict(orientation="h", y=-0.2),
                      margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, width="stretch")
    tabla = pd.DataFrame({"Trimestre": FUT, "Tasa BanRep supuesta (%)": senda,
                          "ICV30 proyectado (%)": np.round(pred, 2), "P10 (%)": np.round(p10, 2),
                          "P90 (%)": np.round(p90, 2), "Cartera consumo (billones)": np.round(saldo, 1),
                          "Cartera vencida (billones)": np.round(vencida, 2)})
    st.dataframe(tabla, hide_index=True, width="stretch")
    st.markdown(f"""
**Cómo leerlo, en simple.** El modelo aprendió de 23 trimestres reales dos cosas:
1. **La mora tiene inercia:** cerca de {coef[1]*100:.0f}% del nivel de un trimestre se arrastra al siguiente.
2. **Las tasas pegan con retraso:** por cada punto que sube la tasa de BanRep en un semestre, la mora de consumo
   sube unos **{coef[2]:.2f} pp, tres trimestres después** (las cuotas suben y los hogares se aprietan).

Por eso las alzas de enero–junio de 2026 (de 9,25% a 12%) se reflejan sobre todo en **1T27–2T27**.
""")

with tab2:
    fig2 = go.Figure()
    colores = {"Conservador": "#c0392b", "Base": "#1f6fb5", "Optimista": "#2e8b57"}
    fig2.add_trace(go.Scatter(x=trim_hist[-8:], y=y[-8:], mode="lines+markers", name="Real", line=dict(color="#1f3a5f")))
    for n, serie in ref.items():
        fig2.add_trace(go.Scatter(x=[trim_hist[-1]] + FUT, y=[y[-1]] + list(serie), mode="lines+markers",
                                  name=n, line=dict(color=colores[n], dash="dash")))
    fig2.add_hline(y=UMBRAL_AMARILLO, line_dash="dot", line_color="#e67e22")
    fig2.update_layout(height=380, yaxis_title="ICV30 consumo (%)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig2, width="stretch")
    comp = pd.DataFrame({n: np.round(s, 2) for n, s in ref.items()}, index=FUT)
    comp.loc["Pico"] = [round(float(np.max(s)), 2) for s in ref.values()]
    st.dataframe(comp, width="stretch")
    for n, e in o.ESCENARIOS.items():
        st.markdown(f"**{n}:** {e['texto']}")
    st.info("Una **predicción estadística** es lo que el modelo calcula con los datos (rango P10–P90). "
            "Un **escenario** es una historia de '¿y si…?' construida con supuestos del equipo (tasas futuras, golpes extra).")

with tab3:
    m1 = o.metricas(bt1.real, bt1.modelo); n1 = o.metricas(bt1.real, bt1.ingenuo)
    m4 = o.metricas(bt4.real, bt4.modelo); n4 = o.metricas(bt4.real, bt4.ingenuo)
    st.markdown("**Prueba justa (validación rodante):** se 'retrocede el reloj' a cada trimestre entre 1T24 y 2T26, "
                "el modelo se entrena solo con lo que se sabía en ese momento y se compara su pronóstico con lo que pasó.")
    st.dataframe(pd.DataFrame({
        "Horizonte": ["1 trimestre", "1 trimestre", "4 trimestres", "4 trimestres"],
        "Modelo": ["Oráculo", "Ingenuo (repetir último dato)", "Oráculo", "Ingenuo (repetir último dato)"],
        "MAE (pp)": [m1["MAE"], n1["MAE"], m4["MAE"], n4["MAE"]],
        "RMSE (pp)": [m1["RMSE"], n1["RMSE"], m4["RMSE"], n4["RMSE"]],
        "MAPE (%)": [m1["MAPE"], n1["MAPE"], m4["MAPE"], n4["MAPE"]],
        "Pruebas": [len(bt1), len(bt1), len(bt4), len(bt4)]}).round(2), hide_index=True, width="stretch")
    fig3 = go.Figure()
    t_bt = [trim_hist[i] for i in bt1.idx]
    fig3.add_trace(go.Scatter(x=t_bt, y=bt1.real, mode="lines+markers", name="Real"))
    fig3.add_trace(go.Scatter(x=t_bt, y=bt1.modelo, mode="lines+markers", name="Oráculo", line=dict(dash="dash")))
    fig3.add_trace(go.Scatter(x=t_bt, y=bt1.ingenuo, mode="lines+markers", name="Ingenuo", line=dict(dash="dot")))
    fig3.update_layout(height=320, yaxis_title="ICV30 consumo (%)", margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig3, width="stretch")
    st.markdown(f"""
- R² dentro de la muestra: **{r2:.2f}** (solo {len(resid)} observaciones: el modelo es pequeño a propósito, 3 parámetros).
- A 4 trimestres el modelo **se quedó corto** (subestimó la mora en las 7 pruebas): por eso
  la decisión se toma mirando también el P90, no solo la línea central.
- Se probaron 8 combinaciones de rezago y ventana; se eligió la de menor error a 4 trimestres. Con tan pocos datos,
  esa elección puede estar algo favorecida por el período de prueba: hay que re-validarla cada trimestre.
""")

with tab4:
    st.markdown("**Datos (todos públicos, agregados, sin información personal).**")
    st.dataframe(df, hide_index=True, width="stretch")
    st.download_button("Descargar datos (CSV)", df.to_csv(index=False).encode("utf-8"), "datos_cartera_bancolombia.csv")
    st.markdown("""
**Cuándo NO usar esta predicción (detener o corregir):**
- Si cambia la definición o el perímetro del indicador (p. ej., ventas de filiales, reexpresiones *pro forma*).
- Ante choques que el modelo no conoce: pandemia, alivios masivos, reformas que cambien los ingresos de los hogares.
- Si el dato real se sale del rango P10–P90 dos trimestres seguidos → el modelo deja de usarse hasta re-entrenarlo.
- **Nunca** para aprobar o negar el crédito de una persona: es un modelo del portafolio total, no de individuos.
- Si la tasa sale de lo observado en la historia (por encima de 13,25% o por debajo de 1,75%).

**Supervisión humana:** la Vicepresidencia de Riesgos valida cada proyección antes del comité; cualquier
endurecimiento de políticas requiere análisis de impacto por segmento para no excluir injustamente a grupos de clientes.
""")
st.caption("Proyecto académico · Universidad de San Buenaventura · IA para los Negocios. No es una herramienta oficial de "
           "Bancolombia ni de Grupo Cibest y no constituye recomendación financiera.")
