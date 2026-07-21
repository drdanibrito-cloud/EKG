"""
IA y la muerte súbita cardiaca — Panel Streamlit
Proyecto final · Visualización y Storytelling

Todas las cifras de "estudio" provienen de:
Empirical results reproduced from: Guo et al., "An ECG biomarker for sudden
cardiac death discovered with deep learning", Nature (2026).
DOI: 10.1038/s41586-026-10674-6
(cifras adicionales de cobertura: Scientific American, EurekAlert!)

Las señales de ECG mostradas en la sección "El biomarcador" son SINTÉTICAS,
generadas para fines de storytelling visual y calibradas para reproducir la
morfología descrita en las Fig. 4-5 del estudio (slurring terminal en la
derivación aVL, desviación de eje). No son trazos de pacientes reales: los
datos crudos del estudio (Suecia/EE.UU./Taiwán) no son públicos.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm
import streamlit as st

st.set_page_config(
    page_title="IA y la muerte súbita cardiaca",
    page_icon="\U0001FAC0",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Datos reales publicados en el estudio (constantes, con cita)
# ---------------------------------------------------------------------------
STUDY = {
    "titulo": "An ECG biomarker for sudden cardiac death discovered with deep learning",
    "revista": "Nature (2026)",
    "doi": "10.1038/s41586-026-10674-6",
    "url": "https://www.nature.com/articles/s41586-026-10674-6",
    "n_total_ecgs": 441614,
    "n_total_pacientes": 180000,  # aprox., Region Halland, Suecia (2010-2016)
    "n_lockbox_ecgs": 119541,
    "n_lockbox_pacientes": 35885,
    "auc_suecia": 0.872,
    "auc_suecia_ci": (0.843, 0.899),
    "auc_eeuu": 0.822,
    "auc_eeuu_ci": (0.812, 0.831),
    "auc_taiwan": 0.767,
    "auc_taiwan_ci": (0.706, 0.823),
    "auc_ahaacc": 0.697,
    "auc_seer": 0.655,
    "grupo_alto_riesgo_pct": 2.2,
    "scd_alto_riesgo": 7.0,
    "scd_alto_riesgo_ci": (4.9, 9.5),
    "scd_lvef_reducido": 4.6,
    "overlap_alto_riesgo_con_lvef": 13.9,   # -> 86.1% NO detectado por LVEF
    "scd_ambos_positivos": 10.7,
    "scd_lvef_no_ecg": 3.4,
    "scd_ecg_no_lvef": 6.4,
    "n_eeuu": 251858,
    "n_taiwan_casos": 257,
    "n_taiwan_controles": 4011,
}

TIMELINE = [
    (1895, "Röntgen descubre los rayos X", "Nace la radiografía: ver dentro del cuerpo sin abrir la piel."),
    (1903, "Einthoven registra el primer ECG", "Nace la electrocardiografía y, después, toda la electrofisiología cardiaca."),
    (1986, "Brugada describe un patrón de ECG ligado a muerte súbita", "Ejemplo histórico de correlación humana ECG-riesgo."),
    (2020, "Colapso de Christian Eriksen (Eurocopa)", "Recordatorio de que la muerte súbita golpea sin aviso, incluso en atletas de élite."),
    (2026, "IA descubre un biomarcador nuevo en la derivación aVL", "441,614 ECGs suecos entrenan un modelo que ve lo que 100+ años de cardiología no habían visto."),
]

# ---------------------------------------------------------------------------
# Utilidades: ROC binormal ilustrativa a partir de un AUC publicado
# ---------------------------------------------------------------------------
def roc_from_auc(auc: float, n_points: int = 200):
    """Aproximación binormal (varianzas iguales) de una curva ROC que
    reproduce el AUC dado. Es una reconstrucción ilustrativa: el estudio
    no publica los puntos de la curva ROC completa, solo el AUC."""
    d_prime = np.sqrt(2) * norm.ppf(np.clip(auc, 0.501, 0.999))
    fpr = np.linspace(0, 1, n_points)
    tpr = norm.cdf(d_prime + norm.ppf(np.clip(fpr, 1e-4, 1 - 1e-4)))
    return fpr, np.clip(tpr, 0, 1)


# ---------------------------------------------------------------------------
# Generador de ECG sintético (ilustrativo, no son datos de pacientes reales)
# ---------------------------------------------------------------------------
LEAD_MULT = {
    # (P, Q, R, S, T) multiplicadores relativos "típicos" por derivación,
    # simplificados para fines educativos/visuales, no diagnósticos.
    "I":    (0.10, 0.10, 0.9, 0.2, 0.30),
    "II":   (0.15, 0.10, 1.2, 0.2, 0.35),
    "III":  (0.08, 0.05, 0.5, 0.3, 0.10),
    "aVR":  (-0.08, -0.05, -0.7, -0.1, -0.20),
    "aVL":  (0.06, 0.08, 0.5, 0.3, 0.15),
    "aVF":  (0.10, 0.08, 0.8, 0.2, 0.20),
    "V1":   (0.05, 0.02, 0.3, 0.9, -0.10),
    "V2":   (0.08, 0.02, 0.6, 1.1, 0.25),
    "V3":   (0.10, 0.03, 1.0, 0.7, 0.35),
    "V4":   (0.12, 0.05, 1.3, 0.4, 0.40),
    "V5":   (0.12, 0.06, 1.2, 0.2, 0.40),
    "V6":   (0.10, 0.06, 1.0, 0.1, 0.35),
}
LEADS_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def _gauss(t, mu, sigma, amp):
    return amp * np.exp(-0.5 * ((t - mu) / sigma) ** 2)


def _single_beat(t, p, q, r, s, tw, slur=False):
    p_wave = _gauss(t, -0.20, 0.025, p)
    q_wave = _gauss(t, -0.03, 0.008, -abs(q))
    r_wave = _gauss(t, 0.0, 0.012, r)
    if slur:
        # Reemplaza la S nítida por una deflexión terminal "arrastrada"
        # (slurred), como describe la Fig. 4-5 del estudio para aVL.
        s_wave = _gauss(t, 0.030, 0.018, -abs(s) * 0.35) + _gauss(t, 0.055, 0.022, abs(s) * 0.30)
    else:
        s_wave = _gauss(t, 0.035, 0.010, -abs(s))
    t_wave = _gauss(t, 0.20, 0.045, tw)
    return p_wave + q_wave + r_wave + s_wave + t_wave


@st.cache_data
def generar_ecg_12_derivaciones(riesgo: str = "bajo", fs: int = 500, n_latidos: int = 3, hr_bpm: int = 75):
    """Genera 12 derivaciones sintéticas. riesgo='alto' aplica la morfología
    descrita en el estudio: desviación de eje izquierdo (mayor amplitud en
    I y aVL, menor en III) + slurring terminal en aVL."""
    ciclo = 60.0 / hr_bpm
    dur = n_latidos * ciclo
    t = np.arange(0, dur, 1 / fs)

    señales = {}
    for lead in LEADS_ORDER:
        p, q, r, s, tw = LEAD_MULT[lead]
        if riesgo == "alto":
            if lead == "I":
                r *= 1.35
            elif lead == "aVL":
                r *= 1.45
            elif lead == "III":
                r *= 0.55

        y = np.zeros_like(t)
        for beat_i in range(n_latidos):
            centro = beat_i * ciclo
            local_t = t - centro
            mask = (local_t > -0.30) & (local_t < 0.35)
            slur = riesgo == "alto" and lead == "aVL"
            y[mask] += _single_beat(local_t[mask], p, q, r, s, tw, slur=slur)

        ruido = np.random.default_rng(42).normal(0, 0.004, size=y.shape)
        señales[lead] = y + ruido

    return t, señales


# ---------------------------------------------------------------------------
# Sidebar / navegación
# ---------------------------------------------------------------------------
st.sidebar.title("IA en Medicina de Urgencias")
st.sidebar.caption("Panel de storytelling — muerte súbita cardiaca y deep learning sobre ECG")
seccion = st.sidebar.radio(
    "Ir a la sección:",
    [
        "1. El problema (100 años sin respuesta)",
        "2. El modelo de IA (rendimiento)",
        "3. IA vs. LVEF: ¿redescubre o descubre?",
        "4. El biomarcador en la derivación aVL",
        "5. Validación global (Suecia · EE.UU. · Taiwán)",
        "6. Sube tu ECG (CSV) y calcula el biomarcador",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Fuente principal:** {STUDY['titulo']}, *{STUDY['revista']}*.\n\n"
    f"DOI: [{STUDY['doi']}]({STUDY['url']})"
)

# ---------------------------------------------------------------------------
# Sección 1: Timeline histórico
# ---------------------------------------------------------------------------
if seccion.startswith("1"):
    st.title("Cien años buscando predecir la muerte súbita")
    st.write(
        "Cada avance en la historia de la cardiología generó un mundo entero. "
        "Y sin embargo, hasta hace muy poco, seguíamos sin poder predecir quién "
        "va a sufrir una muerte súbita cardiaca."
    )

    fig = go.Figure()
    years = [x[0] for x in TIMELINE]
    labels = [x[1] for x in TIMELINE]

    fig.add_trace(
        go.Scatter(
            x=years,
            y=[1] * len(years),
            mode="markers",
            marker=dict(size=16, color="#c0392b"),
            hovertext=[x[2] for x in TIMELINE],
            hoverinfo="text",
            showlegend=False,
        )
    )
    # Etiquetas alternadas arriba/abajo para que no se encimen cuando dos
    # eventos caen cerca en el eje de años (ej. 1895/1903, 2020/2026).
    for i, (yr, lab) in enumerate(zip(years, labels)):
        arriba = i % 2 == 0
        fig.add_annotation(
            x=yr,
            y=1,
            text=lab,
            showarrow=True,
            arrowhead=0,
            arrowcolor="#c0392b",
            ax=0,
            ay=-55 if arriba else 55,
            font=dict(size=11, color="#2c3e50"),
            align="center",
        )
    fig.update_yaxes(visible=False, range=[0.3, 1.7])
    fig.update_xaxes(title="Año")
    fig.update_layout(height=380, showlegend=False, margin=dict(t=40, b=20))
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Años desde el primer ECG (Einthoven, 1903)", "120+")
        st.metric("ECGs analizados por el modelo (Suecia, 2010-2016)", f"{STUDY['n_total_ecgs']:,}")
    with col2:
        st.metric("Predictor en uso clínico amplio hoy", "LVEF (ultrasonido)")
        st.metric("Pacientes de alto riesgo sin antecedentes", "41.2%")

    st.info(
        "LVEF (fracción de eyección del ventrículo izquierdo) sigue siendo el único "
        "predictor de uso clínico amplio, no porque sea el mejor posible, sino porque "
        "es barato, estandarizado y fácil de obtener con un ultrasonido."
    )

# ---------------------------------------------------------------------------
# Sección 2: Rendimiento del modelo (AUC)
# ---------------------------------------------------------------------------
elif seccion.startswith("2"):
    st.title("¿Qué tan bien predice el modelo?")
    st.write(
        "El desempeño se mide con AUC (área bajo la curva ROC): 0.5 es azar puro, "
        "1.0 es predicción perfecta. El modelo de IA sobre ECG supera claramente a "
        "las herramientas de riesgo cardiovascular usadas hoy."
    )

    comparacion = pd.DataFrame(
        {
            "Modelo": [
                "IA-ECG (Suecia, hold-out)",
                "Puntaje AHA/ACC (riesgo CV a 10 años)",
                "SEER (ECG deep-learning previo)",
            ],
            "AUC": [STUDY["auc_suecia"], STUDY["auc_ahaacc"], STUDY["auc_seer"]],
        }
    )
    fig_bar = go.Figure(
        go.Bar(
            x=comparacion["AUC"],
            y=comparacion["Modelo"],
            orientation="h",
            marker_color=["#c0392b", "#7f8c8d", "#7f8c8d"],
            text=comparacion["AUC"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
        )
    )
    fig_bar.update_xaxes(range=[0.5, 1.0], title="AUC")
    fig_bar.update_layout(height=320, margin=dict(t=20, b=20))
    st.plotly_chart(fig_bar, use_container_width=True)

    st.subheader("Curva ROC (reconstrucción ilustrativa a partir del AUC)")
    st.caption(
        "El estudio reporta el AUC pero no publica todos los puntos de la curva ROC; "
        "esta curva es una aproximación binormal calibrada para reproducir ese AUC, "
        "solo con fines de visualización."
    )
    fig_roc = go.Figure()
    for nombre, auc, color in [
        ("IA-ECG (Suecia)", STUDY["auc_suecia"], "#c0392b"),
        ("AHA/ACC", STUDY["auc_ahaacc"], "#7f8c8d"),
    ]:
        fpr, tpr = roc_from_auc(auc)
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nombre} (AUC={auc:.3f})", line=dict(color=color, width=3)))
    fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line=dict(dash="dash", color="lightgray"), showlegend=False))
    fig_roc.update_layout(height=420, xaxis_title="1 - Especificidad", yaxis_title="Sensibilidad", margin=dict(t=20, b=20))
    st.plotly_chart(fig_roc, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("AUC Suecia (hold-out)", f"{STUDY['auc_suecia']:.3f}", help=f"IC 95%: {STUDY['auc_suecia_ci']}")
    c2.metric("AUC EE.UU. (zero-shot)", f"{STUDY['auc_eeuu']:.3f}", help=f"IC 95%: {STUDY['auc_eeuu_ci']}")
    c3.metric("AUC Taiwán (zero-shot)", f"{STUDY['auc_taiwan']:.3f}", help=f"IC 95%: {STUDY['auc_taiwan_ci']}")

# ---------------------------------------------------------------------------
# Sección 3: IA vs LVEF
# ---------------------------------------------------------------------------
elif seccion.startswith("3"):
    st.title("¿La IA solo redescubre lo que ya sabíamos, o encuentra algo nuevo?")

    detectado_pct = STUDY["overlap_alto_riesgo_con_lvef"]
    no_detectado_pct = 100 - detectado_pct
    fig_pie = go.Figure(
        go.Pie(
            labels=["Ya detectado por LVEF reducido", "NO detectado por LVEF (nuevo para la medicina)"],
            values=[detectado_pct, no_detectado_pct],
            marker_colors=["#7f8c8d", "#c0392b"],
            hole=0.45,
        )
    )
    fig_pie.update_layout(height=380, margin=dict(t=20, b=20), title="Composición del grupo de alto riesgo identificado por la IA")
    st.plotly_chart(fig_pie, use_container_width=True)
    st.metric(f"Pacientes de alto riesgo nunca antes detectados por la prueba estándar", f"{no_detectado_pct:.1f}%")

    st.subheader("Tasa anual de muerte súbita cardiaca por grupo")
    grupos = pd.DataFrame(
        {
            "Grupo": [
                "LVEF reducido, sin alerta de IA",
                "LVEF reducido (todos)",
                "Alerta de IA (todos)",
                "Alerta de IA, LVEF normal/desconocido",
                "Ambos biomarcadores positivos",
            ],
            "Tasa anual SCD (%)": [
                STUDY["scd_lvef_no_ecg"],
                STUDY["scd_lvef_reducido"],
                STUDY["scd_alto_riesgo"],
                STUDY["scd_ecg_no_lvef"],
                STUDY["scd_ambos_positivos"],
            ],
        }
    )
    fig_bar2 = go.Figure(
        go.Bar(
            x=grupos["Tasa anual SCD (%)"],
            y=grupos["Grupo"],
            orientation="h",
            marker_color=["#95a5a6", "#7f8c8d", "#e67e22", "#c0392b", "#8e44ad"],
            text=grupos["Tasa anual SCD (%)"].map(lambda v: f"{v:.1f}%"),
            textposition="outside",
        )
    )
    fig_bar2.update_layout(height=380, margin=dict(t=20, b=60), xaxis_title="Tasa anual de muerte súbita cardiaca (%)")
    st.plotly_chart(fig_bar2, use_container_width=True)

    st.success(
        "Incluso en pacientes con LVEF normal o desconocido —donde hoy no existe ninguna "
        "forma de estratificar riesgo— el modelo de IA identifica un grupo con más riesgo "
        f"({STUDY['scd_ecg_no_lvef']:.1f}%) que el de los pacientes con LVEF reducido ({STUDY['scd_lvef_reducido']:.1f}%)."
    )

# ---------------------------------------------------------------------------
# Sección 4: Biomarcador en aVL (visor ECG)
# ---------------------------------------------------------------------------
elif seccion.startswith("4"):
    st.title("El patrón que 100 años de cardiología no habían visto")
    st.write(
        "El modelo generativo del estudio transformó un ECG de bajo riesgo en su versión "
        "'de alto riesgo', latido por latido. El resultado: desviación del eje eléctrico "
        "(mayor amplitud en I y aVL, menor en III) y, sobre todo, un **arrastre terminal "
        "(slurring)** nunca antes descrito en la derivación **aVL**."
    )
    st.caption(
        "⚠️ Las señales de abajo son **sintéticas**, generadas para ilustrar la morfología "
        "descrita en las Fig. 4 y 5 del estudio. No son trazos de pacientes reales — esos "
        "datos clínicos no son públicos."
    )

    riesgo = st.radio("Selecciona la morfología a visualizar:", ["Bajo riesgo (típico)", "Alto riesgo (morph del estudio)"], horizontal=True)
    riesgo_key = "alto" if riesgo.startswith("Alto") else "bajo"
    t, señales = generar_ecg_12_derivaciones(riesgo=riesgo_key)

    st.subheader("Derivación aVL (detalle)")
    fig_avl = go.Figure()
    fig_avl.add_trace(go.Scatter(x=t, y=señales["aVL"], mode="lines", line=dict(color="#c0392b" if riesgo_key == "alto" else "#2c3e50", width=2)))
    fig_avl.update_layout(height=280, margin=dict(t=10, b=30), xaxis_title="tiempo (s)", yaxis_title="mV (unidades relativas)")
    st.plotly_chart(fig_avl, use_container_width=True)
    if riesgo_key == "alto":
        st.warning("Observa el arrastre en la parte final del complejo QRS, en vez de la S nítida del trazo de bajo riesgo.")

    st.subheader("12 derivaciones")
    fig_grid = make_subplots(rows=4, cols=3, subplot_titles=LEADS_ORDER, shared_xaxes=True)
    for i, lead in enumerate(LEADS_ORDER):
        r, c = divmod(i, 3)
        color = "#c0392b" if (riesgo_key == "alto" and lead == "aVL") else "#2c3e50"
        fig_grid.add_trace(go.Scatter(x=t, y=señales[lead], mode="lines", line=dict(color=color, width=1.3), showlegend=False), row=r + 1, col=c + 1)
    fig_grid.update_layout(height=700, margin=dict(t=40, b=20))
    fig_grid.update_xaxes(showticklabels=False)
    fig_grid.update_yaxes(showticklabels=False)
    st.plotly_chart(fig_grid, use_container_width=True)

    with st.expander("¿Por qué importa esto? (hipótesis de fibrosis)"):
        st.write(
            "Los autores proponen que este patrón podría reflejar **fibrosis miocárdica difusa y sutil**: "
            "depósito de colágeno eléctricamente inerte entre las células cardiacas, que altera la "
            "conducción eléctrica. En resonancias magnéticas cardiacas de los pacientes de mayor riesgo, "
            "se observó más realce tardío de gadolinio (LGE) difuso, consistente con esta hipótesis."
        )

# ---------------------------------------------------------------------------
# Sección 5: Validación global
# ---------------------------------------------------------------------------
elif seccion.startswith("5"):
    st.title("¿Funciona fuera de Suecia?")
    st.write(
        "El modelo se entrenó exclusivamente con datos suecos y se probó, sin ningún "
        "ajuste adicional (*zero-shot*), en dos poblaciones completamente distintas."
    )

    validacion = pd.DataFrame(
        {
            "País / cohorte": ["Suecia (hold-out)", "EE.UU. (Sharp HealthCare)", "Taiwán (NTUH)"],
            "N ECGs / casos": [
                f"{STUDY['n_lockbox_ecgs']:,} ECGs",
                f"{STUDY['n_eeuu']:,} ECGs",
                f"{STUDY['n_taiwan_casos']} casos + {STUDY['n_taiwan_controles']:,} controles",
            ],
            "AUC": [STUDY["auc_suecia"], STUDY["auc_eeuu"], STUDY["auc_taiwan"]],
            "Desenlace evaluado": [
                "Muerte súbita cardiaca (certificado de defunción)",
                "Fibrilación / taquicardia ventricular (registros clínicos)",
                "Paro cardiaco arrítmico (revisión de caso)",
            ],
        }
    )
    st.dataframe(validacion, use_container_width=True, hide_index=True)

    fig_val = go.Figure(
        go.Bar(
            x=validacion["País / cohorte"],
            y=validacion["AUC"],
            marker_color=["#c0392b", "#e67e22", "#8e44ad"],
            text=validacion["AUC"].map(lambda v: f"{v:.3f}"),
            textposition="outside",
        )
    )
    fig_val.update_yaxes(range=[0.5, 1.0], title="AUC")
    fig_val.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig_val, use_container_width=True)

    st.info(
        "El modelo también se probó contra un 'placebo': paros cardiacos **no** arrítmicos "
        "(por ictus o causas pulmonares) en Taiwán. Ahí el AUC cae a 0.582, casi azar — "
        "evidencia de que el modelo es específico para arritmias, no una alarma general de gravedad."
    )

# ---------------------------------------------------------------------------
# Sección 6: Cálculo del biomarcador de aVL a partir de un CSV
# ---------------------------------------------------------------------------
elif seccion.startswith("6"):
    st.title("Calculadora del biomarcador de aVL")

    st.info(
        "**Esto no es el modelo de IA del estudio** (el paper aclara que el modelo "
        "entrenado no es público, solo accesible con acuerdo con Region Halland por "
        "GDPR). Esta calculadora reproduce **una sola métrica** que sí describen "
        "(Fig. 5): qué tan 'arrastrada' es la caída final del QRS en aVL, entre el "
        "pico de R y el fin del QRS. Es un ejercicio educativo, no una herramienta "
        "clínica."
    )

    st.write(
        "Sube un CSV con una sola columna de voltaje de la derivación aVL "
        "(por ejemplo, exportado de PhysioNet/PTB-XL, o de cualquier ECG digital "
        "que tengas a mano)."
    )

    ejemplo_t, ejemplo_señales = generar_ecg_12_derivaciones(riesgo="alto")
    ejemplo_csv = pd.DataFrame({"aVL": ejemplo_señales["aVL"]}).to_csv(index=False)
    st.download_button(
        "Descargar CSV de ejemplo (aVL sintético, para probar la herramienta)",
        data=ejemplo_csv,
        file_name="ejemplo_aVL.csv",
        mime="text/csv",
    )

    fs = st.number_input("Frecuencia de muestreo del CSV (Hz)", min_value=100, max_value=2000, value=500, step=50)
    archivo = st.file_uploader("Sube tu CSV", type=["csv"])

    if archivo is not None:
        datos = pd.read_csv(archivo)
        columna = datos.columns[0]
        señal = datos[columna].to_numpy(dtype=float)
        t = np.arange(len(señal)) / fs

        st.markdown("#### Señal cargada")
        fig_raw = go.Figure(go.Scatter(x=t, y=señal, mode="lines", line=dict(color="#2c3e50")))
        fig_raw.update_layout(height=280, xaxis_title="tiempo (s)", yaxis_title="amplitud", margin=dict(t=20, b=20))
        st.plotly_chart(fig_raw, use_container_width=True)

        r_idx = int(np.argmax(señal))
        r_t = t[r_idx]
        ventana_ms = st.slider("Ventana tras el pico R considerada como 'fin del QRS' (ms)", 40, 160, 100)
        fin_idx = min(r_idx + int(ventana_ms / 1000 * fs), len(señal) - 1)
        segmento = señal[r_idx:fin_idx]

        if len(segmento) < 4:
            st.error("La ventana quedó muy corta. Sube la frecuencia de muestreo indicada o amplía la ventana.")
        else:
            diffs1 = np.abs(np.diff(segmento))
            diffs2 = np.abs(np.diff(segmento, n=2))

            st.markdown("#### Métrica de aVL (ventana R → fin de QRS, como en la Fig. 5 del estudio)")
            c1, c2 = st.columns(2)
            c1.metric("Media |1ª diferencia|", f"{diffs1.mean():.4f}")
            c2.metric("Media |2ª diferencia|", f"{diffs2.mean():.4f}")

            fig_zoom = go.Figure(go.Scatter(x=t[r_idx:fin_idx], y=segmento, mode="lines+markers", line=dict(color="#c0392b")))
            fig_zoom.update_layout(height=280, title="Ventana analizada (pico R → fin de QRS)", xaxis_title="tiempo (s)", yaxis_title="amplitud", margin=dict(t=40, b=20))
            st.plotly_chart(fig_zoom, use_container_width=True)

            st.caption(
                "El estudio no publica un umbral de corte para esta métrica — solo mostró, "
                "por regresión, que valores más altos se asocian con más riesgo de muerte "
                "súbita. Por eso aquí se muestra el valor calculado, sin clasificarlo en "
                "'alto' o 'bajo riesgo': esa clasificación exige el modelo completo, que no "
                "es público."
            )

# ---------------------------------------------------------------------------
# Footer / fuentes
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Fuente de datos: Guo et al., *An ECG biomarker for sudden cardiac death discovered "
    f"with deep learning*, Nature (2026). DOI: {STUDY['doi']}. "
    "Cobertura adicional: Scientific American, EurekAlert!. "
    "Panel construido para el proyecto final de Visualización y Storytelling."
)
