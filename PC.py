# ==============================
# SISTEMA COMPLETO ONPE 2021
# ==============================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------
# CONFIG
# ------------------------------
URL_DATA = "https://raw.githubusercontent.com/SIC25/analisis-onpe-2021/main/Resultados%20por%20mesa%20de%20la%20Segunda%20Elecci%C3%B3n%20Presidencial%202021.csv"

st.title("Resultados Electorales ONPE 2021")

# ------------------------------
# 1. CARGA DE DATOS (GITHUB)
# ------------------------------
@st.cache_data
def cargar_datos(url):
    df = pd.read_csv(url, sep=';', encoding='latin-1')
    return df

df = cargar_datos(URL_DATA)

# ------------------------------
# 2. LIMPIEZA
# ------------------------------
def limpiar_datos(df):
    df.columns = df.columns.str.strip().str.upper()
    df = df.drop_duplicates()

    columnas_numericas = [
        "VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN", "VOTOS_VI"
    ]

    for col in columnas_numericas:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

    return df

df = limpiar_datos(df)

# ------------------------------
# 3. MOSTRAR DATOS
# ------------------------------
st.subheader("Primeros 10 registros")
st.dataframe(df.head(10), use_container_width=True)

# ------------------------------
# 4. ANÁLISIS
# ------------------------------
st.subheader("Resumen General")

total_mesas = df.shape[0]
ubigeos = df["UBIGEO"].nunique() if "UBIGEO" in df.columns else 0

votos_p1 = df["VOTOS_P1"].sum()
votos_p2 = df["VOTOS_P2"].sum()
votos_blanco = df["VOTOS_VB"].sum()
votos_nulos = df["VOTOS_VN"].sum()
votos_impugnados = df["VOTOS_VI"].sum()
votos_validos = votos_p1 + votos_p2

# Métricas
st.metric("Total de mesas", total_mesas)
st.metric("Ubigeos únicos", ubigeos)

# Detalle votos
st.subheader("Conteo de votos")
st.write(f"Castillo (P1): {votos_p1:,.0f}")
st.write(f"Fujimori (P2): {votos_p2:,.0f}")
st.write(f"Blancos: {votos_blanco:,.0f}")
st.write(f"Nulos: {votos_nulos:,.0f}")
st.write(f"Impugnados: {votos_impugnados:,.0f}")
st.write(f"Votos válidos: {votos_validos:,.0f}")

# ------------------------------
# 5. VISUALIZACIÓN
# ------------------------------

# Gráfico 1: Candidatos
st.subheader("Votos por candidato")

votos = {
    "Castillo": votos_p1,
    "Fujimori": votos_p2
}

fig1, ax1 = plt.subplots()
ax1.bar(votos.keys(), votos.values())
ax1.set_title("Votos por candidato")
ax1.set_xlabel("Candidato")
ax1.set_ylabel("Votos")

st.pyplot(fig1)

# ------------------------------
# FILTRO GLOBAL
# ------------------------------
st.subheader("Filtro por regiones")

if "DEPARTAMENTO" in df.columns:

    regiones = df["DEPARTAMENTO"].unique()

    regiones_seleccionadas = st.multiselect(
        "Selecciona regiones:",
        regiones,
        default=regiones[:5]
    )

    # Filtrar dataset
    df_filtrado = df[df["DEPARTAMENTO"].isin(regiones_seleccionadas)]

    if df_filtrado.empty:
        st.warning("Selecciona al menos una región")
    else:

        # ------------------------------
        # Gráfico 1: Castillo
        # ------------------------------
        st.subheader("Votos por región (Castillo)")

        votos_castillo = df_filtrado.groupby("DEPARTAMENTO")["VOTOS_P1"].sum()

        fig1, ax1 = plt.subplots()
        ax1.pie(
            votos_castillo,
            labels=votos_castillo.index,
            autopct="%1.1f%%"
        )
        ax1.set_title("Distribución de votos de Castillo")

        st.pyplot(fig1)

        # ------------------------------
        # Gráfico 2: Fujimori
        # ------------------------------
        st.subheader("Votos por región (Fujimori)")

        votos_fujimori = df_filtrado.groupby("DEPARTAMENTO")["VOTOS_P2"].sum()

        fig2, ax2 = plt.subplots()
        ax2.pie(
            votos_fujimori,
            labels=votos_fujimori.index,
            autopct="%1.1f%%"
        )
        ax2.set_title("Distribución de votos de Fujimori")

        st.pyplot(fig2)

else:
    st.warning("No se encontró la columna DEPARTAMENTO")

# ------------------------------
# Gráfico 3: Comparativa por región
# ------------------------------
st.subheader("Comparativa de votos por región")

if not df_filtrado.empty:

    # Agrupar datos
    comparativa = df_filtrado.groupby("DEPARTAMENTO")[["VOTOS_P1", "VOTOS_P2"]].sum()

    # Renombrar columnas
    comparativa.columns = ["Castillo", "Fujimori"]

    # Crear gráfico
    fig3, ax3 = plt.subplots()
    comparativa.plot(kind="bar", ax=ax3)

    ax3.set_title("Comparación de votos por región")
    ax3.set_xlabel("Departamento")
    ax3.set_ylabel("Votos")
    ax3.tick_params(axis='x', rotation=90)

    st.pyplot(fig3)

else:
    st.warning("No hay datos para mostrar")

# ==============================
# PARTE 4: MACHINE LEARNING
# ==============================

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans

st.subheader("Modelo de Machine Learning")

# ------------------------------
# 1. PREDICCIÓN DE VOTO (CLASIFICACIÓN)
# ------------------------------
def preparar_datos(df):

    df_ml = df.copy()

    # Variable objetivo: ganador por mesa
    df_ml["GANADOR"] = (df_ml["VOTOS_P1"] > df_ml["VOTOS_P2"]).astype(int)

    X = df_ml[["VOTOS_VB", "VOTOS_VN", "VOTOS_VI"]]
    y = df_ml["GANADOR"]

    return X, y


def entrenar_modelo(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return modelo, accuracy, X_test, y_test, y_pred


# Ejecutar clasificación
X, y = preparar_datos(df)
modelo, accuracy, X_test, y_test, y_pred = entrenar_modelo(X, y)

st.metric("Precisión del modelo (Clasificación)", f"{accuracy:.2%}")


# ------------------------------
# 2. AGRUPAMIENTO (CLUSTERING)
# ------------------------------
st.subheader("Agrupamiento de mesas (Clustering)")

def clustering_mesas(df):

    df_cluster = df.copy()

    X_cluster = df_cluster[["VOTOS_P1", "VOTOS_P2", "VOTOS_VB", "VOTOS_VN"]]

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df_cluster["CLUSTER"] = kmeans.fit_predict(X_cluster)

    return df_cluster


df_cluster = clustering_mesas(df)

# Mostrar distribución de clusters
st.write("Distribución de mesas por cluster:")
st.write(df_cluster["CLUSTER"].value_counts())

st.write("Interpretación:")
st.write("""
- Cluster 0: mesas con mayor tendencia a un candidato
- Cluster 1: mesas mixtas o balanceadas
- Cluster 2: mesas con alta proporción de votos nulos/blancos
""")

# ------------------------------
# PREDICCIÓN DE TENDENCIA DE VOTO
# ------------------------------

def preparar_datos(df):

    df_ml = df.copy()

    # 1 = gana Castillo, 0 = gana Fujimori
    df_ml["GANADOR"] = (df_ml["VOTOS_P1"] > df_ml["VOTOS_P2"]).astype(int)

    X = df_ml[["VOTOS_VB", "VOTOS_VN", "VOTOS_VI"]]
    y = df_ml["GANADOR"]

    return X, y, df_ml


def entrenar_modelo(X, y):

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    modelo = DecisionTreeClassifier(max_depth=5, random_state=42)
    modelo.fit(X_train, y_train)

    # Accuracy train
    acc_train = modelo.score(X_train, y_train)

    # Accuracy test
    acc_test = modelo.score(X_test, y_test)

    return modelo, acc_train, acc_test, X_test, y_test

# ------------------------------
# EJECUCIÓN
# ------------------------------
X, y, df_ml = preparar_datos(df)

modelo, acc_train, acc_test, X_test, y_test = entrenar_modelo(X, y)

st.metric("Accuracy Train", f"{acc_train:.2%}")
st.metric("Accuracy Test", f"{acc_test:.2%}")

gap = acc_train - acc_test

st.write(f"Diferencia (Overfitting): {gap:.2%}")

if gap > 0.15:
    st.warning("⚠️ Sobreajuste fuerte")
elif gap > 0.05:
    st.warning("⚠️ Posible sobreajuste")
else:
    st.success("✅ Modelo bien generalizado")
st.subheader("Predicción de tendencia de voto")

st.metric("Precisión del modelo", f"{accuracy:.2%}")

# ------------------------------
# MOSTRAR PREDICCIONES
# ------------------------------
resultado = pd.DataFrame({
    "Votos Blancos": X_test["VOTOS_VB"],
    "Votos Nulos": X_test["VOTOS_VN"],
    "Votos Impugnados": X_test["VOTOS_VI"],
    "Predicción": y_pred
})

resultado["Predicción"] = resultado["Predicción"].map({
    1: "Castillo",
    0: "Fujimori"
})

st.write("Ejemplo de predicción de tendencia por mesa:")
st.dataframe(resultado.head(10))

# ==============================
# PARTE 5: EVALUACIÓN DEL MODELO
# ==============================

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.subheader("Evaluación del Modelo")

def evaluar_modelo(modelo, X_test, y_test):

    y_pred = modelo.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    st.metric("Accuracy del modelo", f"{acc:.2%}")

    # ------------------------------
    # Reporte de clasificación
    # ------------------------------
    st.write("📋 Reporte de clasificación:")

    report = classification_report(y_test, y_pred, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # ------------------------------
    # Matriz de confusión
    # ------------------------------
    st.write("📉 Matriz de confusión:")

    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

    # ------------------------------
    # Interpretación automática
    # ------------------------------
    st.subheader("Interpretación del modelo")

    if acc > 0.90:
        st.warning("⚠️ Posible sobreajuste (modelo demasiado perfecto)")
    elif acc < 0.60:
        st.warning("⚠️ Posible subajuste (modelo poco preciso)")
    else:
        st.success("✅ Buen desempeño del modelo")

    st.write("""
    ### Limitaciones del modelo en contexto electoral:
    - Solo usa variables de votos nulos, blancos e impugnados
    - No considera factores políticos o geográficos complejos
    - Es un modelo simplificado para análisis exploratorio
    - Puede no capturar completamente el comportamiento real del votante
    """)


# ------------------------------
# EJECUCIÓN
# ------------------------------
evaluar_modelo(modelo, X_test, y_test)