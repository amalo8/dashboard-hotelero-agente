import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

load_dotenv()

st.set_page_config(page_title="Dashboard Hotelero NLP", page_icon="", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=DM+Sans:wght@400;500&display=swap');

/* Eliminar espacios superiores de Streamlit */
[data-testid="stHeader"]              { display: none !important; }
[data-testid="stAppViewContainer"]    { padding-top: 0 !important; }
[data-testid="stMainBlockContainer"]  { padding-top: 0 !important; }
.block-container                      { padding-top: 0 !important; padding-bottom: 1rem !important; max-width: 100% !important; }
section[data-testid="stMain"]         { padding-top: 0 !important; }

[data-testid="stAppViewContainer"] {
    background-color: #faf9f7;
    font-family: 'DM Sans', sans-serif;
}

/* FRANJA */
.header-band {
    background: linear-gradient(135deg, #2C4A6E 0%, #1a3350 100%);
    padding: 18px 40px 0 40px;
    margin-left: -4rem;
    margin-right: -4rem;
    margin-top: 0;
    margin-bottom: 15px; /* Separación con el selector */
}

.header-top-row {
    display: flex;
    align-items: baseline;
    gap: 18px;
    margin-bottom: 14px;
}

.header-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.55rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: 0.4px;
    margin: 0;
    white-space: nowrap;
}

.header-subtitle {
    font-size: 0.8rem;
    color: #8dafc8;
    margin: 0;
}

/* Las tabs NATIVAS de Streamlit estilizadas para vivir en la franja */
[data-testid="stTabs"] {
    width: calc(100% + 8rem) !important; /* Compensa el padding del contenedor de Streamlit */
    margin-left: -4rem;
    margin-right: -4rem;
    padding-left: 40px;
    padding-right: 40px;
    background: #faf9f7 !important; /* CAMBIO: Fondo de viñetas en color crema, igual que el fondo */
    /*border-bottom: 4px solid #c9a84c;*/ 
}

[data-testid="stTabs"] [role="tablist"] {
    background: transparent !important;
    border-bottom: none !important;
    gap: 0 !important;* CAMBIO CLAVE: Hace que cada pestaña crezca por igual para rellenar el espacio */
    flex-grow: 1 !important;
    text-align: center !important;
}

[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #555555 !important; /* CAMBIO: Color gris oscuro para verse en el fondo crema */
    padding: 9px 28px !important;
    background: transparent !important;
    border: none !important;
    /*border-bottom: 3px solid transparent !important;*/
    border-radius: 0 !important;
    transition: all 0.15s ease !important;
}

[data-testid="stTabs"] [role="tab"]:hover {
    color: #1a3350 !important; /* Hover azul oscuro */
    /*border-bottom-color: rgba(201,168,76,0.35) !important;*/
}

[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1a3350 !important; /* Texto azul oscuro al seleccionar */
    font-weight: 600 !important;
    /*border-bottom: 3px solid #c9a84c !important;*/
    background: transparent !important;
}

/* Quitar la línea roja/indicador por defecto de Streamlit */
[data-testid="stTabs"] [role="tab"] p {
    color: inherit !important;
}

[data-testid="stTabsContent"] {
    padding-top: 12px !important;
}

/* Selector */
[data-testid="stSelectbox"] > div > div {
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
    background-color: #faf9f7 !important;
    max-width: 440px;
}
[data-testid="stSelectbox"] label { display: none !important; }

/* Botones */
[data-testid="stButton"] > button {
    background-color: #2C4A6E !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.87rem !important;
    font-weight: 500 !important;
    padding: 10px 24px !important;
}
[data-testid="stButton"] > button:hover { background-color: #1a3350 !important; }

hr { border-color: #e8e4dc !important; }
[data-testid="stCaptionContainer"] { color: #9ca3af !important; font-size: 0.78rem !important; }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGA DE DATOS
# ==========================================
import gc 

@st.cache_data(max_entries=1)
def cargar_datos():
    dtypes_optimizados = {
        'nombre_del_hotel': 'category'
    }
    df_info = pd.read_csv('datos/procesados/df_hoteles_vlc_info.csv', encoding='utf-8-sig', dtype=dtypes_optimizados)
    df_comentarios = pd.read_csv('datos/procesados/df_comentarios_final_topics.csv', encoding='utf-8-sig', dtype=dtypes_optimizados)
    df_comentarios_con_indices_temporales = pd.read_csv('datos/procesados/df_comentarios_con_indices_temporales.csv', encoding='utf-8-sig', dtype=dtypes_optimizados)
    return df_info, df_comentarios, df_comentarios_con_indices_temporales

df_info, df_comentarios, df_temporales = cargar_datos()
gc.collect() # Limpiamos memoria

lista_hoteles = df_info['nombre_del_hotel'].unique()

# ==========================================
# CABECERA (título + subtítulo)
# ==========================================
st.markdown("""
<div class="header-band">
    <div class="header-top-row">
        <span class="header-title">Cuadro de mandos para la revisión de reseñas hoteleras</span>
        <span class="header-subtitle">Analizador de sentimiento, extracción de tópicos y diagnóstico competitivo mediante IA</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ==========================================
# SELECTOR DE HOTEL (Movido encima de las viñetas/pestañas para que sea visible de forma global)
# ==========================================
hotel_seleccionado = st.selectbox("Hotel", lista_hoteles, label_visibility="collapsed")

# ==========================================
# TABS NATIVAS / VIÑETAS (fondo blanco)
# ==========================================
tab1, tab2 = st.tabs(["Análisis de puntuaciones", "Consultor de estrategias"])

# ==========================================
# FILTRADO
# ==========================================
info_hotel = df_info[df_info['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()].iloc[0]
comentarios_hotel = df_comentarios[df_comentarios['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()]
temporales_hotel = df_temporales[df_temporales['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()]

# ==========================================
# FUNCIONES BASE
# ==========================================
def _consultar_metricas():
    return info_hotel.to_dict()

def _analizar_comentarios():
    def extraer(col):
        todos = []
        for c in comentarios_hotel[col].dropna():
            temas = str(c).split(" - ")
            todos.extend([t for t in temas if t != "Otro Tema / Mixto / Vacío"])
        return pd.Series(todos).value_counts().head(3).index.tolist()
    return {
        "fortalezas": extraer('tema_positivo'),
        "debilidades": extraer('tema_negativo')
    }

def _obtener_evidencia_real(categoria, tipo):
    col_tema = 'tema_positivo' if tipo == 'positivo' else 'tema_negativo'
    col_texto = 'positivo' if tipo == 'positivo' else 'negativo'
    muestras = comentarios_hotel[comentarios_hotel[col_tema].str.contains(categoria, na=False, case=False)]
    if muestras.empty:
        return f"No se han encontrado quejas o alabanzas específicas sobre {categoria}."
    ejemplos = muestras[col_texto].head(6).tolist()
    return "\n".join([f"- {txt}" for txt in ejemplos])

# ------------------------------------------
# PESTAÑA 1: DASHBOARD VISUAL
# ------------------------------------------
with tab1:
    col_izq, col_der = st.columns([0.65, 0.35])

    total_resenas = len(comentarios_hotel)
    resenas_pos = comentarios_hotel['positivo'].notna().sum()
    resenas_neg = comentarios_hotel['negativo'].notna().sum()
    porc_pos = round((resenas_pos / total_resenas) * 100, 1) if total_resenas > 0 else 0
    porc_neg = round((resenas_neg / total_resenas) * 100, 1) if total_resenas > 0 else 0

    with col_izq:
        categorias_base = ['Personal', 'Confort', 'WiFi', 'Instalaciones', 'Calidad/Precio', 'Limpieza', 'Ubicación']
        columnas_notas = ['nota_personal', 'nota_confort', 'nota_wifi', 'nota_instalaciones_servicios', 'nota_calidad_precio', 'nota_limpieza', 'nota_ubicacion']
        
        cat_validas, punt_hotel, med_globales = [], [], []
        for cat, col in zip(categorias_base, columnas_notas):
            nota = info_hotel[col]
            if pd.notna(nota):
                cat_validas.append(cat)
                punt_hotel.append(nota)
                med_globales.append(round(df_info[col].mean(), 1))

        if len(cat_validas) >= 3:
            fig = go.Figure()
            fig.add_trace(go.Scatterpolar(
                r=med_globales + [med_globales[0]], theta=cat_validas + [cat_validas[0]],
                fill='toself', fillcolor='rgba(255, 65, 54, 0.15)',
                line=dict(color='rgba(255, 65, 54, 0.5)', width=1.5), name='Media del Sector'
            ))
            fig.add_trace(go.Scatterpolar(
                r=punt_hotel + [punt_hotel[0]], theta=cat_validas + [cat_validas[0]],
                fill='toself', fillcolor='rgba(31, 119, 180, 0.5)',
                line=dict(color='#1f77b4', width=3), marker=dict(size=8), name=hotel_seleccionado
            ))
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[8, 10], dtick=0.5)),
                showlegend=True, margin=dict(l=20, r=20, t=20, b=20),
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_der:
        st.markdown("""
            <style>
            .bloque { padding: 15px; border-radius: 10px; margin-bottom: 10px; color: #333; text-align: center; }
            .b1 { background-color: #E3F2FD; }
            .b2 { background-color: #F1F8E9; }
            .b3 { background-color: #FFF3E0; }
            .b4 { background-color: #FFEBEE; }
            .titulo-bloque { font-size: 0.9rem; font-weight: bold; margin-bottom: 5px; }
            .valor-bloque { font-size: 1.8rem; font-weight: bold; }
            </style>
        """, unsafe_allow_html=True)

        st.markdown(f'<div class="bloque b1"><div class="titulo-bloque">Nota media</div><div class="valor-bloque">{info_hotel["nota_media_resenas"]}</div><div style="font-size:0.8rem">sobre 10 puntos</div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="bloque b2"><div class="titulo-bloque">Número de reseñas analizadas</div><div class="valor-bloque">{total_resenas}</div></div>', unsafe_allow_html=True)

        sub1, sub2 = st.columns(2)
        with sub1:
            st.markdown(f'<div class="bloque b3"><div class="titulo-bloque">Reseñas con comentarios positivos</div><div class="valor-bloque">{porc_pos}%</div></div>', unsafe_allow_html=True)
        with sub2:
            st.markdown(f'<div class="bloque b4"><div class="titulo-bloque">Reseñas con comentarios negativos</div><div class="valor-bloque">{porc_neg}%</div></div>', unsafe_allow_html=True)

# ------------------------------------------
# PESTAÑA 2: EL AGENTE LANGCHAIN
# ------------------------------------------
with tab2:
    st.subheader(f"Consultor de estrategias para {hotel_seleccionado}")

    @tool
    def comparar_con_media_competencia():
        """
        Compara las notas del hotel seleccionado con la media del resto de hoteles.
        Devuelve un string indicando la categoría que está más por debajo de la media,
        o 'NINGUNA_DEBAJO_MEDIA' si todas están por encima.
        """
        categorias_base = ['Personal', 'Confort', 'WiFi', 'Instalaciones', 'Calidad/Precio', 'Limpieza', 'Ubicación']
        columnas_notas = ['nota_personal', 'nota_confort', 'nota_wifi', 'nota_instalaciones_servicios', 'nota_calidad_precio', 'nota_limpieza', 'nota_ubicacion']
        peor_diferencia = 0
        peor_categoria = "NINGUNA_DEBAJO_MEDIA"
        for cat, col in zip(categorias_base, columnas_notas):
            nota_h = info_hotel[col]
            if pd.notna(nota_h):
                media_sector = df_info[col].mean()
                diferencia = nota_h - media_sector
                if diferencia < peor_diferencia:
                    peor_diferencia = diferencia
                    peor_categoria = cat
        return peor_categoria

    @tool
    def extraer_resenas_categoria(categoria: str):
        """
        Devuelve SOLAMENTE las reseñas negativas asociadas a esa categoría concreta 
        para el hotel seleccionado.
        """
        return _obtener_evidencia_real(categoria, "negativo")

    @tool
    def destacar_puntos_fuertes():
        """Devuelve los puntos fuertes generales del hotel cuando es mejor que la competencia."""
        fortalezas = _analizar_comentarios()['fortalezas']
        if fortalezas:
            return f"Las categorías más alabadas son: {', '.join(fortalezas)}."
        return "El hotel es estable en general, sin picos de excelencia destacables."

    @tool
    def analizar_indices_ponderados_temporales():
        """
        Busca qué categorías tienen los valores más altos y más bajos 
        en sus columnas 'indice_pond_...' para el hotel.
        Devuelve hasta las 3 mejores y las 3 peores.
        """
        columnas_ponderadas = [col for col in temporales_hotel.columns if col.startswith("indice_pond_")]
        if not columnas_ponderadas or temporales_hotel.empty:
            return {"mejores": ["Desconocida"], "peores": ["Desconocida"]}
        medias = temporales_hotel[columnas_ponderadas].mean().sort_values(ascending=False)
        mejores = medias.head(3).index.str.replace("indice_pond_", "").tolist()
        peores = medias.tail(3).index.str.replace("indice_pond_", "").tolist()
        return {"mejores": mejores, "peores": peores}

    @tool
    def extraer_citas_textuales_recientes(categoria: str, tipo: str):
        """
        Busca comentarios recientes sobre la 'categoria' dada.
        'tipo' debe ser 'positivo' o 'negativo'.
        """
        col_tema = 'tema_positivo' if tipo == 'positivo' else 'tema_negativo'
        col_texto = 'positivo' if tipo == 'positivo' else 'negativo'
        if 'dias_pasados' in temporales_hotel.columns:
            df_ordenado = temporales_hotel.sort_values(by='dias_pasados', ascending=True)
        else:
            df_ordenado = temporales_hotel
        muestras = df_ordenado[df_ordenado[col_tema].str.contains(categoria, na=False, case=False)]
        if muestras.empty:
            return f"No se han encontrado comentarios recientes de tipo {tipo} sobre {categoria}."
        ejemplos = muestras[col_texto].dropna().head(4).tolist()
        return "\n".join([f'- "{txt}"' for txt in ejemplos])

    clave = os.getenv("OPENROUTER_API_KEY")
    modelos_prueba = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-2-9b-it:free"
    ]

    col_btn_1, col_btn_2 = st.columns(2)

    with col_btn_1:
        boton_competitivo = st.button("Análisis competitivo", use_container_width=True)

    if boton_competitivo:
        with st.spinner('Analizando posición frente a la competencia...'):
            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(model=m, api_key=clave, base_url="https://openrouter.ai/api/v1", temperature=0.1)
                    tools_comp = [comparar_con_media_competencia, extraer_resenas_categoria, destacar_puntos_fuertes]
                    prompt_comp = ChatPromptTemplate.from_messages([
                        ("system", f"""Eres un Consultor Estratégico Senior. Tu objetivo es realizar un diagnóstico diferencial frente a la competencia del {hotel_seleccionado}.

                            REGLAS DE ORO:
                            1. Cita textualmente. NO edites, NO resumas ni cambies el tono de las reseñas originales. Si tienen errores, los mantienes.
                            2. Si el hotel está por debajo de la media en alguna categoría:
                            - Identifícalas claramente.
                            - Crea una tabla con: | Categoria | Estado | Reseña ilustrativa |
                            - Proporciona una guía de acciones correctoras con 4 o menos puntos estratégicos.
                            3. Si el hotel es líder en todo:
                            - NO te limites a decir "somos líderes".
                            - Destaca las fortalezas analizando el patrón de los comentarios positivos.
                            - Explica técnicamente POR QUÉ estamos por encima (ej: "La ubicación supera la media gracias a la cercanía con X, consolidando un nicho de mercado de negocios").
                            4. NO hables sobre los resultados internos de tus funciones:
                            - si comparar_media_competencia devuelve "NINGUNA_DEBAJO_MEDIA" no lo digas explícitamente, redacta que el hotel tiene un rendimiento sólido y destaca las fortalezas.

                            PROHIBIDO: 
                            - Hablar de tus herramientas o pasos internos.
                            - Hacer listas kilométricas; ve directo a la estrategia y al dato.
                        """),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_tool_calling_agent(llm, tools_comp, prompt_comp)
                    executor = AgentExecutor(agent=agent, tools=tools_comp, verbose=True)
                    respuesta = executor.invoke({"input": "Realiza el análisis competitivo siguiendo estrictamente las reglas de formato."})
                    st.markdown("---")
                    st.markdown("### Informe Competitivo")
                    st.markdown(respuesta["output"])
                    st.caption(f"Modelo utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    continue
            if not exito:
                st.error("Servidores de IA saturados. Reintenta en unos segundos.")

    with col_btn_2:
        boton_actual = st.button("Análisis actual", use_container_width=True)

    if boton_actual:
        with st.spinner('Evaluando tendencias y construyendo informe visual...'):
            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(model=m, api_key=clave, base_url="https://openrouter.ai/api/v1", temperature=0.1)
                    tools_actual = [analizar_indices_ponderados_temporales, extraer_citas_textuales_recientes]
                    prompt_actual = ChatPromptTemplate.from_messages([
                        ("system", f"""Eres un Auditor de Calidad enfocado EXCLUSIVAMENTE en el presente del {hotel_seleccionado}.
                        
                        PROCESO OBLIGATORIO:
                        1. Usa 'analizar_indices_ponderados_temporales' para obtener las mejores y peores categorías (analiza de 1 a 3 por cada lado, es mejor que no llegues a 3, pon 3 SOLO si estan muy a la par las 3 en las puntuaciones, es decir ninguna destaca sobre las otras).
                        2. Usa 'extraer_citas_textuales_recientes' para obtener una cita literal por categoría
                         
                        REGLA DE PRIORIZACIÓN: 
                        No estás obligado a extraer 3 categorías. Analiza los datos y extrae SOLAMENTE las que sean verdaderamente relevantes (pueden ser 1, 2 o hasta 3). Si una categoría no tiene impacto significativo, no la incluyas.
                         
                        REGLA DE ORO SOBRE CITAS:
                            - Debes extraer las citas TEXTUALES. Copia y pega exactamente lo que el usuario escribió.
                            - PROHIBIDO corregir ortografía, gramática, eliminar palabras o cambiar el tono.
                            - Si la reseña original tiene errores, déjalos tal cual. La veracidad es más importante que la estética en este punto.
                            - Si el modelo detecta que ha modificado una sola palabra, el proceso se considera fallido.
                                                    
                        ESTRUCTURA DE SALIDA ESTRICTA (Usa EXACTAMENTE estas etiquetas, sin corchetes ni Markdown extra):
                        
                        FUERTES
                        CAT: [Nombre de la categoría 1]
                        RES: [Explicación directa de por qué es buena, sin usar la palabra "Resumen"]
                        CIT: [Cita literal limpia, sin comillas extra]
                        ===
                        CAT: [Nombre de la categoría 2 si la hay]
                        RES: [Explicación directa del problema, sin usar la palabra "Resumen"]
                        CIT: [Cita literal limpia, sin comillas extra]
                        |||
                        DEBILES
                        CAT: [Nombre de la categoría 3 si la hay]
                        RES: [Explicación directa del problema, sin usar la palabra "Resumen"]
                        CIT: [Cita literal limpia, sin comillas extra]
                        ===
                        
                        """),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    agent = create_tool_calling_agent(llm, tools_actual, prompt_actual)
                    executor = AgentExecutor(agent=agent, tools=tools_actual, verbose=True)
                    respuesta = executor.invoke({"input": "Realiza el análisis siguiendo estrictamente las etiquetas de formato."})
                    texto_ia = respuesta["output"]
                    st.markdown("---")
                    st.markdown("""
                                <style>
                                .col-fuerte { background-color: #eafaf1; padding: 20px; border-radius: 10px; height: 100%; }
                                .col-debil { background-color: #fff9e6; padding: 20px; border-radius: 10px; height: 100%; }
                                .titulo-fuerte { color: #27ae60; font-weight: bold; font-size: 22px; margin-bottom: 25px; text-align: center; display: block; }
                                .titulo-debil { color: #f39c12; font-weight: bold; font-size: 22px; margin-bottom: 25px; text-align: center; display: block; }
                                .cat-fuerte { color: #27ae60; font-weight: bold; font-size: 17px; margin-bottom: 5px; }
                                .cat-debil { color: #f39c12; font-weight: bold; font-size: 17px; margin-bottom: 5px; }
                                .texto-resumen { color: #333333; font-size: 15px; margin-bottom: 10px; }
                                .caja-cita { background-color: #f2f4f4; border-left: 4px solid #bdc3c7; padding: 10px; font-style: italic; color: #555555; margin-bottom: 25px; }
                                </style>
                                """, unsafe_allow_html=True)
                    def parsear_bloque(texto_bloque):
                        items = []
                        bloques_separados = texto_bloque.split("===")
                        for b in bloques_separados:
                            cat = res = cit = ""
                            for linea in b.strip().split('\n'):
                                linea = linea.strip()
                                if linea.startswith("CAT:"): cat = linea.replace("CAT:", "").replace("[", "").replace("]", "").strip()
                                elif linea.startswith("RES:"): res = linea.replace("RES:", "").strip()
                                elif linea.startswith("CIT:"): cit = linea.replace("CIT:", "").replace('"', '').strip()
                            if cat or res or cit:
                                items.append({"cat": cat, "res": res, "cit": cit})
                        return items
                    if "|||" in texto_ia:
                        partes = texto_ia.split("|||")
                        fuertes_data = parsear_bloque(partes[0].replace("FUERTES", ""))
                        debiles_data = parsear_bloque(partes[1].replace("DEBILES", ""))
                        col_fuertes, col_debiles = st.columns(2)
                        with col_fuertes:
                            html_fuertes = '<div class="col-fuerte"><div class="titulo-fuerte">Puntos fuertes recientes</div>'
                            for item in fuertes_data:
                                html_fuertes += f'<div><div class="cat-fuerte">{item["cat"]}</div><div class="texto-resumen">{item["res"]}</div><div class="caja-cita">{item["cit"]}</div></div>'
                            html_fuertes += '</div>'
                            st.markdown(html_fuertes, unsafe_allow_html=True)
                        with col_debiles:
                            html_debiles = '<div class="col-debil"><div class="titulo-debil">Puntos débiles recientes</div>'
                            for item in debiles_data:
                                html_debiles += f'<div><div class="cat-debil">{item["cat"]}</div><div class="texto-resumen">{item["res"]}</div><div class="caja-cita">{item["cit"]}</div></div>'
                            html_debiles += '</div>'
                            st.markdown(html_debiles, unsafe_allow_html=True)
                    else:
                        st.markdown(texto_ia)
                    st.caption(f"Modelo utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    continue
            if not exito:
                st.error("Servidores de IA saturados. Reintenta en unos segundos.")
