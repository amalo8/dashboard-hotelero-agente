import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
from dotenv import load_dotenv

# --- IMPORTS DE LANGCHAIN ---
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_tool_calling_agent

# Cargar las variables ocultas del archivo .env
load_dotenv()

# ==========================================
# 1. CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(page_title="Dashboard Hotelero NLP", page_icon="🏨", layout="wide")
st.title("🏨 Cuadro de Mandos y Agente IA de Calidad")
st.markdown("Analizador inteligente de sentimiento y tópicos con arquitectura LangChain.")

# ==========================================
# 2. CARGA DE DATOS (Con caché)
# ==========================================
@st.cache_data
def cargar_datos():
    df_info = pd.read_csv('datos/procesados/df_hoteles_vlc_info.csv', encoding='utf-8-sig')
    df_comentarios = pd.read_csv('datos/procesados/df_comentarios_final_topics.csv', encoding='utf-8-sig')
    # Se corrige el return para incluir el df_temporales
    df_comentarios_con_indices_temporales = pd.read_csv('datos/procesados/df_comentarios_con_indices_temporales.csv', encoding='latin-1')
    return df_info, df_comentarios, df_comentarios_con_indices_temporales

df_info, df_comentarios, df_temporales = cargar_datos()

# ==========================================
# 3. SELECTOR Y FILTRADO (ROBUSTO)
# ==========================================
lista_hoteles = df_info['nombre_del_hotel'].unique()
hotel_seleccionado = st.selectbox("📌 Selecciona un Hotel para analizar:", lista_hoteles)

# Filtrado ignorando mayúsculas/minúsculas
info_hotel = df_info[df_info['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()].iloc[0]
comentarios_hotel = df_comentarios[df_comentarios['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()]
temporales_hotel = df_temporales[df_temporales['nombre_del_hotel'].str.lower() == hotel_seleccionado.lower()]

# ==========================================
# 4. FUNCIONES BASE (lógica pura)
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
    """Filtra y devuelve ejemplos reales de texto para una categoría específica."""
    col_tema = 'tema_positivo' if tipo == 'positivo' else 'tema_negativo'
    col_texto = 'positivo' if tipo == 'positivo' else 'negativo'
   
    muestras = comentarios_hotel[comentarios_hotel[col_tema].str.contains(categoria, na=False, case=False)]
   
    if muestras.empty:
        return f"No se han encontrado quejas o alabanzas específicas sobre {categoria}."
   
    ejemplos = muestras[col_texto].head(6).tolist()
    return "\n".join([f"- {txt}" for txt in ejemplos])

# ==========================================
# 5. SISTEMA DE PESTAÑAS
# ==========================================
tab1, tab2 = st.tabs(["📊 Análisis de Puntuaciones", "🤖 Consultor IA LangChain"])

# ------------------------------------------
# PESTAÑA 1: DASHBOARD VISUAL
# ------------------------------------------
with tab1:
    col1, col2 = st.columns(2)
    col1.metric(label="⭐ Nota Media Booking", value=f"{info_hotel['nota_media_resenas']} / 10")
    col2.metric(label="📝 Volumen de Reseñas Analizadas", value=len(comentarios_hotel))

    st.markdown("<br>", unsafe_allow_html=True)

    categorias_base = ['Personal', 'Confort', 'WiFi', 'Instalaciones', 'Calidad/Precio', 'Limpieza', 'Ubicación']
    columnas_notas = ['nota_personal', 'nota_confort', 'nota_wifi', 'nota_instalaciones_servicios', 'nota_calidad_precio', 'nota_limpieza', 'nota_ubicacion']
    cat_validas, punt_hotel, med_globales, cat_faltantes = [], [], [], []

    for cat, col in zip(categorias_base, columnas_notas):
        nota = info_hotel[col]
        if pd.isna(nota):
            cat_faltantes.append(cat)
        else:
            cat_validas.append(cat)
            punt_hotel.append(nota)
            med_globales.append(round(df_info[col].mean(), 1))

    if cat_faltantes:
        st.warning(f"⚠️ Categorías sin datos: **{', '.join(cat_faltantes)}**")

    if len(cat_validas) >= 3:
        min_abs = min(min(punt_hotel), min(med_globales))
        inicio_rango = int(min_abs) if min_abs >= 5 else 0
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
        fig.update_layout(polar=dict(radialaxis=dict(visible=True, range=[inicio_rango, 10], dtick=0.5)),
                          showlegend=True, margin=dict(l=80, r=80, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------
# PESTAÑA 2: EL AGENTE LANGCHAIN (2 MODALIDADES)
# ------------------------------------------
with tab2:
    st.subheader(f"🤖 Consultor Estratégico IA: {hotel_seleccionado}")

    # ==========================================
    # HERRAMIENTAS - MODALIDAD 1: COMPETITIVA
    # ==========================================
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
                # Buscamos la mayor diferencia negativa
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

    # ==========================================
    # HERRAMIENTAS - MODALIDAD 2: ACTUAL (Temporal)
    # ==========================================
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
        
        # Hacemos la media y ordenamos de mayor a menor
        medias = temporales_hotel[columnas_ponderadas].mean().sort_values(ascending=False)
        
        # Obtenemos top 3 (mejores) y bottom 3 (peores)
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
        
        # Ordenamos por los comentarios que han ocurrido hace menos tiempo
        if 'dias_pasados' in temporales_hotel.columns:
            df_ordenado = temporales_hotel.sort_values(by='dias_pasados', ascending=True)
        else:
            df_ordenado = temporales_hotel
            
        muestras = df_ordenado[df_ordenado[col_tema].str.contains(categoria, na=False, case=False)]
        
        if muestras.empty:
            return f"No se han encontrado comentarios recientes de tipo {tipo} sobre {categoria}."
            
        ejemplos = muestras[col_texto].dropna().head(4).tolist()
        return "\n".join([f'- "{txt}"' for txt in ejemplos])

    # ==========================================
    # CONFIGURACIÓN DEL LLM
    # ==========================================
    clave = os.getenv("OPENROUTER_API_KEY")
    modelos_prueba = [
        "nvidia/nemotron-3-super-120b-a12b:free",
        "meta-llama/llama-3-8b-instruct:free",
        "google/gemma-2-9b-it:free"
    ]

    col_btn_1, col_btn_2 = st.columns(2)

    # ------------------------------------------
    # BOTÓN 1: MODALIDAD COMPETITIVA
    # ------------------------------------------
    with col_btn_1:
        boton_competitivo = st.button("🏆 Análisis Competitivo", use_container_width=True)
    
    if boton_competitivo:
        with st.spinner('📊 Analizando posición frente a la competencia...'):
            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(model=m, api_key=clave, base_url="https://openrouter.ai/api/v1", temperature=0.1)
                    tools_comp = [comparar_con_media_competencia, extraer_resenas_categoria, destacar_puntos_fuertes]
                    
                    prompt_comp = ChatPromptTemplate.from_messages([
                        ("system", f"""Eres un Consultor Estratégico Hotelero de élite analizando el {hotel_seleccionado}.
                        
                        PROCESO INTERNO (Síguelo, pero NO lo menciones en tu respuesta):
                        1. Usa 'comparar_con_media_competencia' para ver qué categoría está por debajo de la media.
                        2. Si todo está bien, pon un DISCLAIMER breve y usa 'destacar_puntos_fuertes'.
                        3. Si hay una categoría por debajo, usa 'extraer_resenas_categoria' para detectar el problema base.
                        
                        REGLAS ESTRICTAS DE FORMATO Y ESTILO (¡CRÍTICO!):
                        - PROHIBIDO mencionar las herramientas que has usado, los pasos que has seguido o mostrar tu proceso mental. Ve directo a los resultados.
                        - PROHIBIDO crear tablas para el plan de acción. Nada de tablas kilométricas.
                        - Sé extremadamente conciso, directo y ejecutivo. Calidad por encima de cantidad.
                        
                        ESTRUCTURA OBLIGATORIA DEL INFORME FINAL:
                        ### 📉 Área Crítica frente al Sector: [Nombre de la Categoría]
                        *(Nota: Si todas las categorías superan la media, cambia este título a "🏆 Líder del Sector" y explica los puntos fuertes).*
                        
                        **🔍 Causa Raíz Detectada:**
                        [Un único párrafo conciso de 3-4 líneas explicando el problema real basado en las reseñas que has leído].
                        
                        **🚀 Plan de Acción Inmediato:**
                        [Da EXACTAMENTE 3 viñetas (bullet points) con soluciones operativas, reales y muy concretas. Máximo 2 líneas por viñeta].
                        """),
                        ("human", "{input}"),
                        MessagesPlaceholder(variable_name="agent_scratchpad"),
                    ])
                    
                    agent = create_tool_calling_agent(llm, tools_comp, prompt_comp)
                    executor = AgentExecutor(agent=agent, tools=tools_comp, verbose=True)

                    respuesta = executor.invoke({"input": "Realiza el análisis competitivo siguiendo estrictamente las reglas de formato."})
                    
                    st.markdown("---")
                    st.markdown("### 🏆 Informe Competitivo")
                    st.markdown(respuesta["output"])
                    st.caption(f"🧠 Cerebro utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    continue
            if not exito:
                st.error("Servidores de IA saturados. Reintenta en unos segundos.")

    # ------------------------------------------
    # BOTÓN 2: MODALIDAD ACTUAL (TENDENCIAS)
    # ------------------------------------------
    with col_btn_2:
        boton_actual = st.button("⏱️ Análisis Actual (Ponderado)", use_container_width=True)

    if boton_actual:
        with st.spinner('📈 Evaluando tendencias y construyendo informe visual...'):
            exito = False
            for m in modelos_prueba:
                try:
                    llm = ChatOpenAI(model=m, api_key=clave, base_url="https://openrouter.ai/api/v1", temperature=0.1)
                    tools_actual = [analizar_indices_ponderados_temporales, extraer_citas_textuales_recientes]
                    
                    prompt_actual = ChatPromptTemplate.from_messages([
                        ("system", f"""Eres un Auditor de Calidad enfocado EXCLUSIVAMENTE en el presente del {hotel_seleccionado}.
                        
                        PROCESO OBLIGATORIO:
                        1. Usa 'analizar_indices_ponderados_temporales' para obtener las mejores y peores categorías (analiza de 1 a 3 por cada lado).
                        2. Usa 'extraer_citas_textuales_recientes' para obtener una cita literal por categoría.
                         
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
                        RES: [Explicación...]
                        CIT: [Cita...]
                        |||
                        DEBILES
                        CAT: [Nombre de la categoría A]
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
                    
                    # === MAGIA VISUAL: CSS Y HTML PERSONALIZADO ===
                    st.markdown("---")
                    
                    # Estilos CSS inyectados
                    st.markdown("""
                        <style>
                        .col-fuerte { background-color: #eafaf1; padding: 20px; border-radius: 10px; height: 100%; }
                        .col-debil { background-color: #fff9e6; padding: 20px; border-radius: 10px; height: 100%; }
                        
                        .titulo-fuerte { background-color: #27ae60; color: white; font-weight: bold; font-size: 20px; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-bottom: 20px; }
                        .titulo-debil { background-color: #f39c12; color: white; font-weight: bold; font-size: 20px; padding: 6px 12px; border-radius: 6px; display: inline-block; margin-bottom: 20px; }
                        
                        .cat-fuerte { background-color: #2ecc71; color: white; font-weight: bold; font-size: 16px; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }
                        .cat-debil { background-color: #f1c40f; color: white; font-weight: bold; font-size: 16px; padding: 4px 10px; border-radius: 4px; display: inline-block; margin-bottom: 10px; }
                        
                        .texto-resumen { color: #333333; font-size: 15px; margin-bottom: 10px; line-height: 1.5; }
                        
                        .caja-cita {
                            background-color: #f2f4f4;
                            border-left: 4px solid #bdc3c7;
                            padding: 12px 15px;
                            border-radius: 0 6px 6px 0;
                            color: #555555;
                            font-style: italic;
                            position: relative;
                            margin-bottom: 25px;
                            z-index: 1;
                        }
                        /* El dibujo de la comilla de fondo */
                        .caja-cita::before {
                            content: '"';
                            font-size: 60px;
                            color: rgba(0,0,0,0.05);
                            position: absolute;
                            top: -10px;
                            left: 10px;
                            z-index: -1;
                            font-family: Georgia, serif;
                        }
                        </style>
                    """, unsafe_allow_html=True)
                    
                    # Función para extraer los bloques de texto
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

                    # Renderizado en Streamlit
                    if "|||" in texto_ia:
                        partes = texto_ia.split("|||")
                        fuertes_data = parsear_bloque(partes[0].replace("FUERTES", ""))
                        debiles_data = parsear_bloque(partes[1].replace("DEBILES", ""))
                        
                        col_fuertes, col_debiles = st.columns(2)
                        
                        # --- COLUMNA IZQUIERDA (FUERTES) ---
                        with col_fuertes:
                            html_fuertes = '<div class="col-fuerte"><div class="titulo-fuerte">Puntos Fuertes Recientes</div>'
                            for item in fuertes_data:
                                # HTML en una sola línea para evitar que Markdown cree bloques de código
                                html_fuertes += f'<div><div class="cat-fuerte">{item["cat"]}</div><div class="texto-resumen">{item["res"]}</div><div class="caja-cita">{item["cit"]}</div></div>'
                            html_fuertes += '</div>'
                            st.markdown(html_fuertes, unsafe_allow_html=True)
                            
                        # --- COLUMNA DERECHA (DÉBILES) ---
                        with col_debiles:
                            html_debiles = '<div class="col-debil"><div class="titulo-debil">Puntos Débiles Recientes</div>'
                            for item in debiles_data:
                                # HTML en una sola línea para evitar que Markdown cree bloques de código
                                html_debiles += f'<div><div class="cat-debil">{item["cat"]}</div><div class="texto-resumen">{item["res"]}</div><div class="caja-cita">{item["cit"]}</div></div>'
                            html_debiles += '</div>'
                            st.markdown(html_debiles, unsafe_allow_html=True)
                    else:
                        st.markdown(texto_ia)

                    st.caption(f"🧠 Cerebro utilizado: {m}")
                    exito = True
                    break
                except Exception as e:
                    continue
            if not exito:
                st.error("Servidores de IA saturados. Reintenta en unos segundos.")