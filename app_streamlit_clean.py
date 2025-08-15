import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Taller 1 ML2", layout="wide")
st.title("Taller 1 - Machine Learning 2")
st.write("Aplicación convertida desde Jupyter Notebook")


import pandas as pd

# URLs de NHANES 2021-2023
demo_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.xpt"

# Leer archivos XPT
demo = pd.read_sas(demo_url)

# Mostrar las variables en filas
variables_df = pd.DataFrame(demo.columns, columns=["Variable"])
st.write(variables_df)



import requests
from io import StringIO  # Importar StringIO

# URL del libro de códigos
url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DEMO_L.htm"
resp = requests.get(url)

# Envolver el texto HTML en un StringIO
html_str = StringIO(resp.text)

# Leer todas las tablas
tables = pd.read_html(html_str)

st.write(len(tables))  # Muestra cuántas tablas encontró


# Filtrar solo los que tienen código 2 en ESTADO DE ENTREVISTA dado ques estos fueron los entrevistados y se les realizó examen MEC
demo = demo[demo["RIDSTATR"] == 2]

# Verificar
st.write(demo["RIDSTATR"].value_counts())

# Filtrar solo personas con 18 años o más
demo= demo[demo["RIDAGEYR"] >= 18]

# Ver tamaño de la nueva base
st.write(demo.shape)

# Mostrar primeras filas
demo.head()


# Filtrar no embarazadas

demo= demo[(demo["RIDEXPRG"] == 2) | (demo["RIDEXPRG"] == 3) | (demo["RIDEXPRG"].isna())]

# Ver tamaño de la nueva base
st.write(demo.shape)

# Mostrar primeras filas
demo.head()

st.write(demo.columns)
demo.shape

# Lista de variables a eliminar del DEMO_L
vars_a_eliminar = [
    "SDDSRVYR", "RIDSTATR", "RIDAGEMN", "RIDRETH1", "RIDEXMON", "RIDEXAGM",
    "DMQMILIZ", "RIDEXPRG", "DMDHRGND", "DMDHRAGZ", "DMDHREDZ",
    "DMDHRMAZ", "DMDHSEDZ", "WTINT2YR", "WTMEC2YR", "SDMVSTRA", "SDMVPSU"
]

# Eliminar columnas del DataFrame demo
demo= demo.drop(columns=vars_a_eliminar)

# Ver primeras filas para verificar
demo.head()


# Renombrar columnas
demo.rename(columns={
    "SEQN": "SEQN",
    "RIAGENDR": "Sexo",
    "RIDAGEYR": "Edad en años",
    "RIDRETH3": "Raza/etnia",
    "DMDBORN4": "Lugar de nacimiento",
    "DMDYRUSR": "Duración en EE. UU.",
    "DMDEDUC2": "Nivel educativo alcanzado",
    "DMDMARTZ": "Estado civil",
    "DMDHHSIZ": "Tamaño del hogar",
    "INDFMPIR": "Índice de pobreza familiar (PIR)"
}, inplace=True)

# Verificar cambios
st.write(demo.head())




import numpy as np

# Recodificar categorías

# Sexo
demo['Sexo'] = demo['Sexo'].map({1: 'Hombre', 2: 'Mujer'})

# Raza/etnia
demo['Raza/etnia'] = demo['Raza/etnia'].map({
    1: 'Mexicano-americano',
    2: 'Otros hispanos',
    3: 'Blanco no hispano',
    4: 'Negro no hispano',
    6: 'Asiático no hispano',
    7: 'Otra raza / multirracial'
})

# Lugar de nacimiento
demo['Lugar de nacimiento'] = demo['Lugar de nacimiento'].map({
    1: 'Nacido en EE.UU.',
    2: 'Otro'
})

# Nivel educativo alcanzado
demo['Nivel educativo alcanzado'] = demo['Nivel educativo alcanzado'].map({
    1: 'Menor a HS',
    2: 'HS/GED',
    3: 'Superior'
})

# Estado civil
demo['Estado civil'] = demo['Estado civil'].map({
    1: 'Casado',
    2: 'Soltero',
    3: 'Divorciado/Separado',
    4: 'Viudo',
    5: 'Unión libre'
})


# Lista de variables categóricas para generar tablas de frecuencia
variables_categoricas = ['Sexo', 'Raza/etnia', 'Lugar de nacimiento',
                         'Nivel educativo alcanzado', 'Estado civil']

# Generar tablas de frecuencia
for col in variables_categoricas:
    st.write(f"--- Tabla de frecuencia: {col} ---\n")

    # Frecuencia absoluta
    tabla_abs = demo[col].value_counts(dropna=False)

    # Frecuencia relativa
    tabla_rel = demo[col].value_counts(normalize=True, dropna=False)

    # Combinar en un solo DataFrame
    tabla_frecuencia = pd.DataFrame({
        'Frecuencia': tabla_abs,
        'Frecuencia relativa': tabla_rel
    })

    st.write(tabla_frecuencia)
    st.write("\n")


demo.shape

body_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BMX_L.xpt"

# Leer el archivo XPT de NHANES
body = pd.read_sas(body_url)

# Mostrar las variables como tabla
variables_body = pd.DataFrame(body.columns, columns=["Variable"])
variables_body.reset_index(drop=True, inplace=True)

# Mostrar tabla
variables_body

# Lista de variables a eliminar del DEMO_L
vars_body_eliminar = [
    "BMDSTATS", "BMIWT", "BMIRECUM", "BMIHEAD", "BMIHT","BMILEG" ,"BMIARML", "BMIARMC", "BMIWAIST", "BMIWAIST", "BMIHIP", "BMDBMIC", "BMXHEAD", "BMXRECUM"
]

# Eliminar columnas del DataFrame demo
body= body.drop(columns=vars_body_eliminar )

# Ver primeras filas para verificar
body.head()


body.rename(columns={
    "SEQN": "SEQN",
    "BMXWT": "Peso",
    "BMXHT": "Estatura",
    "BMXBMI": "Índice de masa corporal",
    "BMXLEG": "Longitud de pierna",
    "BMXARML": "Circunferencia del brazo izquierdo",
    "BMXARMC": "Circunferencia del brazo derecho",
    "BMXWAIST": "Circunferencia de cintura",
    "BMXHIP": "Circunferencia de cadera"
}, inplace=True)

# Verificar cambios
st.write(body.head())


body.shape



#(TCHOL_L) / Colesterol total
tchol_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/TCHOL_L.xpt"
tchol = pd.read_sas(tchol_url)
variables_tchol = pd.DataFrame(tchol.columns, columns=["Variable"]).reset_index(drop=True)
st.write(variables_tchol)

vars_tchol_eliminar = ["WTPH2YR"]

# Eliminar columnas del DataFrame demo
tchol= tchol.drop(columns=vars_tchol_eliminar )

# Ver primeras filas para verificar
tchol.head()

tchol.rename(columns={
    "LBXTC": "Colesterol total",
    "LBDTCSI": "Células T totales"
}, inplace=True)

# Verificar cambios
st.write(tchol.head())






# (CBC_L) / Hemograma completo
cbc_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/CBC_L.xpt"
cbc = pd.read_sas(cbc_url)
variables_cbc = pd.DataFrame(cbc.columns, columns=["Variable"]).reset_index(drop=True)
st.write(variables_cbc)

cbc=cbc.drop(columns=["WTPH2YR"])

# Renombrar columnas de la base cbc
cbc.rename(columns={
    "LBXWBCSI": "Recuento de leucocitos",
    "LBXLYPCT": "Porcentaje de linfocitos",
    "LBXMOPCT": "Porcentaje de monocitos",
    "LBXNEPCT": "Porcentaje de neutrófilos",
    "LBXEOPCT": "Porcentaje de eosinófilos",
    "LBXBAPCT": "Porcentaje de basófilos",
    "LBDLYMNO": "Linfocitos absolutos",
    "LBDMONO": "Monocitos absolutos",
    "LBDNENO": "Neutrófilos absolutos",
    "LBDEONO": "Eosinófilos absolutos",
    "LBDBANO": "Basófilos absolutos",
    "LBXRBCSI": "Glóbulos rojos",
    "LBXHGB": "Hemoglobina",
    "LBXHCT": "Hematocrito",
    "LBXMCVSI": "Volumen corpuscular medio",
    "LBXMC": "Concentración de hemoglobina corpuscular media",
    "LBXMCHSI": "Hemoglobina corpuscular media",
    "LBXRDW": "Ancho de distribución de glóbulos rojos",
    "LBXPLTSI": "Plaquetas",
    "LBXMPSI": "Media plaquetaria",
    "LBXNRBC": "Recuento de reticulocitos"
}, inplace=True)

# Verificar cambios
st.write(cbc.head())



#(INS_L.xpt)  / insulina)
ins_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/INS_L.xpt"
ins = pd.read_sas(ins_url)
variables_ins = pd.DataFrame(ins.columns, columns=["Variable"]).reset_index(drop=True)
st.write(variables_ins)
ins=ins.drop(columns=["LBDINLC"])
ins=ins.drop(columns=["WTSAF2YR"])

# Renombrar columnas en tu DataFrame (por ejemplo df)
ins.rename(columns={
    "LBXIN": "Insulina sérica",
    "LBDINSI": "Insulina"
}, inplace=True)

# Verificar cambios
st.write(ins.head())






# (GLU_L.xpt)  / glucosa en ayunas)
glu_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/GLU_L.xpt"
glu = pd.read_sas(glu_url)
variables_glu = pd.DataFrame(glu.columns, columns=["Variable"]).reset_index(drop=True)
st.write(variables_glu)

glu=glu.drop(columns=["WTSAF2YR"])

# Renombrar columnas en tu DataFrame (por ejemplo df)
glu.rename(columns={
    "LBXGLU": "Glucosa sérica",
    "LBDGLUSI": "Glucosa en ayunas (mg/dL)"
}, inplace=True)

# Verificar cambios
st.write(glu.head())



# (VID_L.xpt)  / vitamina D)
vid_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/VID_L.xpt"
vid = pd.read_sas(vid_url)
variables_vid = pd.DataFrame(vid.columns, columns=["Variable"]).reset_index(drop=True)
st.write(variables_vid)
vid=vid.drop(columns=["WTPH2YR","LBDVIDLC", "LBDVD2LC", "LBDVD3LC" , "LBDVE3LC" ])

# Supongamos que tu DataFrame se llama df
vid.rename(columns={
    "LBXVIDMS": "Vitamina D total sérica",
    "LBDVIDLC": "Vitamina D total (largo plazo)",
    "LBXVD2MS": "Vitamina D2",
    "LBDVD2LC": "Vitamina D2 (largo plazo)",
    "LBXVD3MS": "Vitamina D3",
    "LBDVD3LC": "Vitamina D3 (largo plazo)",
    "LBXVE3MS": "Vitamina E"
}, inplace=True)

# Verificar cambios
st.write(vid.head())


# (BPQ_L.xpt)  / presión arterial y colesterol
bpq_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/BPQ_L.xpt"
bpq = pd.read_sas(bpq_url)
variables_bpq = pd.DataFrame(bpq.columns, columns=["Variable"]).reset_index(drop=True)
# Keep SEQN column for merging
bpq = bpq[['SEQN', "BPQ020"]]

# Renombrar BPQ020 in the bpq base
bpq.rename(columns={
    "BPQ020": "Diagnóstico de hipertensión (Sí/No)"
}, inplace=True)

# Verificar cambios
st.write(bpq.head())

# Frecuencia absoluta y relativa
tabla_bpq020 = pd.DataFrame({
    'Frecuencia': bpq['Diagnóstico de hipertensión (Sí/No)'].value_counts(dropna=False),
    'Frecuencia relativa': bpq['Diagnóstico de hipertensión (Sí/No)'].value_counts(normalize=True, dropna=False)
})

st.write(tabla_bpq020)


# Revisar valores únicos de la variable
st.write(bpq['Diagnóstico de hipertensión (Sí/No)'].unique())

# Sobrescribir la misma variable con Sí/No
bpq['Diagnóstico de hipertensión (Sí/No)'] = bpq['Diagnóstico de hipertensión (Sí/No)'].map({
    1.0: 'Sí',
    2.0: 'No',
    7.0: np.nan,
    9.0: np.nan
})

# Verificar cambios
st.write(bpq['Diagnóstico de hipertensión (Sí/No)'].head(10))

# Generar tabla de frecuencia
tabla = bpq['Diagnóstico de hipertensión (Sí/No)'].value_counts(dropna=False)
tabla_rel = bpq['Diagnóstico de hipertensión (Sí/No)'].value_counts(normalize=True, dropna=False)

tabla_frecuencia = pd.DataFrame({
    'Frecuencia': tabla,
    'Frecuencia relativa': tabla_rel
})

st.write(tabla_frecuencia)

# URL correcta del archivo XPT de NHANES DIQ_L
diq_url = "https://wwwn.cdc.gov/Nchs/Data/Nhanes/Public/2021/DataFiles/DIQ_L.xpt"

# Cargar la base
diq = pd.read_sas(diq_url)

# Select only the DIQ010 and SEQN columns and keep as DataFrame
diq = diq[["SEQN", "DIQ010"]]

# Verificar
st.write(diq.head())
st.write(diq.columns)

# Sobrescribir DIQ010 with the coded version
diq['Diagnóstico de diabetes (Sí/No)'] = diq['DIQ010'].map({
    1.0: 'Sí',
    2.0: 'No',
    7.0: np.nan,  # Rechazado
    9.0: np.nan   # No sabe
})

# Renombrar the column for clarity (optional)
diq.rename(columns={'DIQ010': 'Diagnóstico de diabetes (Original)'}, inplace=True)

# Verificar cambios
st.write(diq.head())

# Unir todas las bases a partir de demo
df = demo.merge(body, on="SEQN", how="left") \
                  .merge(tchol, on="SEQN", how="left") \
                  .merge(ins, on="SEQN", how="left") \
                  .merge(glu, on="SEQN", how="left") \
                  .merge(vid, on="SEQN", how="left") \
                  .merge(bpq, on="SEQN", how="left") \
                  .merge(diq, on="SEQN", how="left")

# Ver tamaño final
st.write("Tamaño final de df", df.shape)

# Elimina las observaciones sin valor en la variable objetivo
df = df.dropna(subset=["Diagnóstico de diabetes (Sí/No)"]).copy()

# (Opcional) Reinicia el índice para evitar huecos
df.reset_index(drop=True, inplace=True)

# Comprobación rápida
st.write("Filas restantes:", df.shape[0])
st.write("Valores únicos objetivo:", df["Diagnóstico de diabetes (Sí/No)"].unique())
st.write("NaN en objetivo:", df["Diagnóstico de diabetes (Sí/No)"].isna().sum())


for col in df.columns:
    st.write(col)



# Contar variables numéricas
num_numericas = df.select_dtypes(include=['int64', 'float64']).shape[1]

# Contar variables categóricas
num_categoricas = df.select_dtypes(include=['object', 'category']).shape[1]

st.write(f"Número de variables numéricas: {num_numericas}")
st.write(f"Número de variables categóricas: {num_categoricas}")


# Seleccionar solo variables categóricas
df_cat = df.select_dtypes(include=['object', 'category'])

# Generar tablas de frecuencia para cada variable categórica
for col in df_cat.columns:
    st.write(f"--- Tabla de frecuencia: {col} ---\n")

    # Frecuencia absoluta
    tabla_abs = df_cat[col].value_counts(dropna=False)

    # Frecuencia relativa
    tabla_rel = df_cat[col].value_counts(normalize=True, dropna=False)

    # Unir en un solo DataFrame
    tabla_frecuencia = pd.DataFrame({
        'Frecuencia': tabla_abs,
        'Frecuencia relativa': tabla_rel
    })

    st.write(tabla_frecuencia)
    st.write("\n")


# Seleccionar todas las variables numéricas
df_num = df.select_dtypes(include=['int64', 'float64'])

# Generar estadística descriptiva
estadisticas = df_num.describe().T  # Transponer para que cada variable sea fila
estadisticas['mediana'] = df_num.median()  # Agregar la mediana explícitamente

# Mostrar resultados
st.write("=== Estadísticas descriptivas para todas las variables numéricas ===\n")
st.write(estadisticas)


# Calcular el porcentaje de valores faltantes por columna
porcentaje_nulos = df.isnull().mean()

# Columnas con más del 80% de valores faltantes
columnas_a_eliminar = porcentaje_nulos[porcentaje_nulos >= 0.8].index
st.write("Columnas que se eliminarán (más del 80% de valores faltantes):")
for col in columnas_a_eliminar:
    st.write("-", col)

# Mantener solo las columnas con menos del 80% de valores faltantes
columnas_a_conservar = porcentaje_nulos[porcentaje_nulos < 0.8].index
df = df[columnas_a_conservar]

# Verificar tamaño final
st.write("\nNúmero de columnas después de eliminar:", df.shape[1])

st.write("Nombres de las variables en df:")
for col in df.columns:
    st.write("-", col)



# --- Parámetros globales ---
VAR_THRESHOLD = 0.80  #  Varianza definida

# Columnas que NO deben entrar en PCA/MCA
ID_COLS = ["SEQN"]
TARGET_COLS = ["DIQ010", "Diagnóstico de diabetes (Sí/No)", "Diagnóstico de diabetes (Original)"]

# Construimos copias de trabajo a partir de los dataframes ya preparados:
X_num = df_num.drop(columns=[c for c in (ID_COLS + TARGET_COLS) if c in df_num.columns], errors="ignore").copy()
X_cat = df_cat.drop(columns=[c for c in (ID_COLS + TARGET_COLS) if c in df_cat.columns], errors="ignore").copy()

st.write("X_num shape:", X_num.shape)
st.write("X_cat shape:", X_cat.shape)


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Pipeline numérico: imputación + escalado
num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Fit-transform de numéricas
Xn_tr = num_pipe.fit_transform(X_num)

# PCA completo (todas las posibles componentes)
pca_full = PCA().fit(Xn_tr)
var_ratio = pca_full.explained_variance_ratio_
var_cum = np.cumsum(var_ratio)

# Nº mínimo de componentes para alcanzar el umbral
k_pca = int(np.searchsorted(var_cum, VAR_THRESHOLD) + 1)

# PCA final con k componentes
pca = PCA(n_components=k_pca).fit(Xn_tr)
Z_num = pca.transform(Xn_tr)

PC_cols = [f"PC{i}" for i in range(1, k_pca+1)]
DF_PCA = pd.DataFrame(Z_num, columns=PC_cols, index=X_num.index)

st.write(f"[PCA] Componentes seleccionados: {k_pca} (umbral={VAR_THRESHOLD:.0%})")
st.write("[PCA] Varianza explicada acumulada (primeras 10):", np.round(var_cum[:10], 4))
st.write("DF_PCA shape:", DF_PCA.shape)


# Definir umbral visual (en este caso 80%)
UMBRAL_VISUAL = 0.80

plt.figure(figsize=(6,4))
plt.plot(range(1, len(var_cum)+1), var_cum, marker="o", label="Varianza acumulada")
plt.axhline(y=UMBRAL_VISUAL, color="red", linestyle="--", label=f"Umbral {UMBRAL_VISUAL*100:.0f}%")
plt.xlabel("Número de componentes")
plt.ylabel("Varianza explicada acumulada")
plt.title("Scree plot - PCA")
plt.grid(True)
plt.legend()
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

# 1) Loadings: filas = variables, columnas = PCs
loadings = pca.components_.T                      # shape: (n_vars, k_pca)
feature_names = list(X_num.columns)               # tras tu limpieza
n_pcs_show = min(10, loadings.shape[1])           # muestra hasta 10 o las que existan

# 2) DataFrame para el heatmap
df_loadings = pd.DataFrame(
    loadings[:, :n_pcs_show],
    index=feature_names,
    columns=[f"PC{i+1}" for i in range(n_pcs_show)]
)

# (Opcional) ordenar variables por influencia total (norma de loadings)
order = np.argsort(np.linalg.norm(df_loadings.values, axis=1))[::-1]
df_loadings = df_loadings.iloc[order]

# 3) Heatmap con escala centrada en 0 y límites simétricos
vmax = np.abs(df_loadings.values).max()
plt.figure(figsize=(12, 8))
sns.heatmap(df_loadings, annot=True, fmt=".2f",
            cmap="coolwarm", center=0, vmin=-vmax, vmax=vmax)
plt.title(f"Heatmap de loadings (primeras {n_pcs_show} PCs)")
plt.xlabel("Componentes principales")
plt.ylabel("Variables")
plt.tight_layout()
plt.show()


import matplotlib.patches as mpatches

# --- Alinear objetivo con X_num / scores_all ---
TARGET_COL = "Diagnóstico de diabetes (Sí/No)"
target_aligned = df.loc[X_num.index, TARGET_COL].astype(str)

# mapa de colores y leyenda
color_map = {'Sí': 'red', 'No': 'blue'}
colors = target_aligned.map(color_map)

legend_patches = [
    mpatches.Patch(color='red',  label='Diabetes: Sí'),
    mpatches.Patch(color='blue', label='Diabetes: No')
]

# Asegurar que el PCA se hizo sobre Xn_tr y que tenemos pca y DF_PCA
scores_all = pca.transform(Xn_tr)  # todas las componentes
feature_names = np.array(X_num.columns)
num_components = DF_PCA.shape[1]   # p.ej., 8

def biplot_pc1_vs_pc(k):
    # Scores en los dos ejes seleccionados
    scores = scores_all[:, [0, k-1]]  # PC1 (0) vs PCk (k-1)

    # Loadings escalados
    loadings = pca.components_[[0, k-1], :].T * np.sqrt(pca.explained_variance_[[0, k-1]])
    scores_max = np.max(np.abs(scores), axis=0)
    load_max = np.max(np.abs(loadings), axis=0)
    scale = 0.6 * scores_max / load_max
    loadings_scaled = loadings * scale

    # Etiquetar solo las variables más “fuertes”
    norms = np.linalg.norm(loadings, axis=1)
    top_idx = np.argsort(norms)[-12:]

    # Plot
    plt.figure(figsize=(7,6))
    plt.scatter(scores[:,0], scores[:,1],
                c=colors, alpha=0.7, edgecolor='k', s=20, label="Muestras")

    # Flechas de loadings
    for i in range(loadings_scaled.shape[0]):
        x, yv = loadings_scaled[i, 0], loadings_scaled[i, 1]
        plt.arrow(0, 0, x, yv, color="tab:orange", alpha=0.9,
                  head_width=0.03*scores_max.max(), length_includes_head=True)

    # Etiquetas de variables (top)
    for i in top_idx:
        x, yv = loadings_scaled[i, 0], loadings_scaled[i, 1]
        plt.text(x*1.07, yv*1.07, feature_names[i], fontsize=9, color="tab:orange")

    plt.axhline(0, color="gray", linewidth=0.8)
    plt.axvline(0, color="gray", linewidth=0.8)
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC{k} ({pca.explained_variance_ratio_[k-1]*100:.1f}%)")
    plt.title(f"Biplot PCA — PC1 vs PC{k}")
    plt.grid(True, linestyle="--", alpha=0.4)

    # --- LEYENDA (cuadrito de convenciones) ---
    plt.legend(handles=legend_patches, title="Diagnóstico", loc="best")

    plt.tight_layout()
    plt.show()

# Generar todos los biplots de PC1 vs PCk
for k in range(2, num_components+1):
    biplot_pc1_vs_pc(k)


# Dataset final con PCs + objetivo para luego concatenar con las dimensiones del MCA
DF_PCA_final = pd.concat([df[["Diagnóstico de diabetes (Sí/No)"]].reset_index(drop=True),
                          DF_PCA.reset_index(drop=True)], axis=1)

DF_PCA_final.to_csv("pca_componentes.csv", index=False)
st.write("Guardado pca_componentes.csv con shape:", DF_PCA_final.shape)


# Construir copias de trabajo a partir de los DF ya preparados
X_num = df_num.drop(columns=[c for c in (ID_COLS + TARGET_COLS) if c in df_num.columns], errors="ignore").copy()
X_cat = df_cat.drop(columns=[c for c in (ID_COLS + TARGET_COLS) if c in df_cat.columns], errors="ignore").copy()

st.write("X_num shape:", X_num.shape)
st.write("X_cat shape:", X_cat.shape)

# Para MCA con la librería mca: imputar y forzar string/categórico
X_cat = X_cat.fillna("Missing").astype(str)



import mca

# Matriz disyuntiva completa (0/1)  <<-- CLAVE
X_disc = pd.get_dummies(X_cat, drop_first=False)   # no dropear nada, o sea no se saca ninguan cat
st.write("Tamaño matriz disyuntiva:", X_disc.shape)

# 3) Ajustar MCA (con Benzécri)
m = mca.MCA(X_disc, benzecri=True)

# 4) Inercia e inercia acumulada (usar atributo correcto: L con mayúscula)
eig = np.array(m.L, dtype=float).ravel()
inertia = eig / eig.sum()
inertia_cum = np.cumsum(inertia)

k_mca = int(np.searchsorted(inertia_cum, VAR_THRESHOLD) + 1)
st.write(f"Dimensiones seleccionadas (MCA): {k_mca} (≥ {VAR_THRESHOLD:.0%})")
st.write("Inercia acumulada (primeras 10):", np.round(inertia_cum[:10], 4))

# 5) Coordenadas de individuos (filas) en las primeras k dimensiones
Fs = m.fs_r(N=k_mca)   # (n_muestras, k_mca)
DF_MCA = pd.DataFrame(Fs, index=X_cat.index,
                      columns=[f"DIM{i}" for i in range(1, k_mca+1)])

# 6) Scree plot en %
plt.figure(figsize=(7,5))
plt.plot(range(1, len(inertia_cum)+1), inertia_cum*100, marker='o', label='Inercia acumulada')
plt.axhline(VAR_THRESHOLD*100, color='red', linestyle='--', label=f'Umbral {VAR_THRESHOLD*100:.0f}%')
plt.xlabel('Dimensiones MCA'); plt.ylabel('% Inercia acumulada'); plt.title('MCA - Inercia acumulada (Benzécri)')
plt.grid(True, ls='--', alpha=0.5); plt.legend(); plt.show()

import matplotlib.patches as mpatches

# --- parámetros / entradas ---
TARGET_COL = "Diagnóstico de diabetes (Sí/No)"
max_plots  = None  # o un entero para limitar cuántos plots (por ejemplo 6)

# Alinear objetivo con el índice de DF_MCA
target = df.loc[DF_MCA.index, TARGET_COL].astype(str)

# Mapeo de colores y leyenda
color_map = {'Sí': 'red', 'No': 'blue'}
colors = target.map(color_map)

legend_patches = [
    mpatches.Patch(color='red',  label='Diabetes: Sí'),
    mpatches.Patch(color='blue', label='Diabetes: No')
]

# Cantidad de dimensiones disponibles en DF_MCA
dims = [c for c in DF_MCA.columns if c.startswith("DIM")]
k_mca = len(dims)

# --- función para un biplot DIM1 vs DIMk ---
def biplot_ind_dim1_vs(dim_k):
    x = DF_MCA["DIM1"].to_numpy()
    y = DF_MCA[f"DIM{dim_k}"].to_numpy()
    plt.figure(figsize=(8,6))
    plt.scatter(x, y, c=colors, alpha=0.7, edgecolor='k', s=18)
    plt.axhline(0, color='gray', lw=0.8)
    plt.axvline(0, color='gray', lw=0.8)
    plt.xlabel("DIM1")
    plt.ylabel(f"DIM{dim_k}")
    plt.title(f"MCA — Biplot de individuos (DIM1 vs DIM{dim_k})")
    plt.grid(True, ls='--', alpha=0.4)
    plt.legend(handles=legend_patches, title="Diagnóstico")
    plt.tight_layout()
    plt.show()

# --- generar todos los biplots: DIM1 vs DIM2..DIM{k_mca} ---
count = 0
for k in range(2, k_mca+1):
    biplot_ind_dim1_vs(k)
    count += 1
    if max_plots is not None and count >= max_plots:
        break


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Asumimos que ya tienes ---
# X_cat: DataFrame solo categóricas
# X_disc: pd.get_dummies(X_cat, drop_first=False)  (si no lo tienes, créalo)
# m:      objeto MCA ya ajustado con benzecri=True
# k_mca:  número de dimensiones retenidas (por tu umbral)

# 1) Masas de las categorías (proporción de 1s en cada dummy)
if 'X_disc' not in globals():
    X_disc = pd.get_dummies(X_cat, drop_first=False)
mass = X_disc.mean(axis=0).values                  # shape (n_categorías,)

# 2) Coordenadas de las categorías en las primeras k dimensiones
G = m.fs_c(N=k_mca)                                # shape (n_categorías, k_mca)

# 3) Autovalores (inercia de cada dimensión)
eig = np.asarray(m.L[:k_mca], dtype=float)         # shape (k_mca,)

# 4) Contribuciones por categoría y dimensión
ctr = (mass[:, None] * (G**2)) / eig[None, :]      # n_cat x k_mca
ctr_pct = ctr / ctr.sum(axis=0, keepdims=True)     # normalizamos a 1 por dimensión

# 5) DataFrame de contribuciones (categorías)
DIM_cols = [f"DIM{i}" for i in range(1, k_mca+1)]
DF_ctr_cat = pd.DataFrame(ctr_pct, index=X_disc.columns, columns=DIM_cols)

# --- Heatmap de las categorías más influyentes en DIM1–DIM2 ---
top_k = 25  # muestra las 25 categorías más influyentes sumando DIM1+DIM2
top_cats = DF_ctr_cat[["DIM1","DIM2"]].sum(axis=1).sort_values(ascending=False).head(top_k).index
plt.figure(figsize=(10, 8))
sns.heatmap(DF_ctr_cat.loc[top_cats, ["DIM1","DIM2"]], annot=True, fmt=".02f",
            cmap="YlGnBu", cbar_kws={'label': 'Contribución (proporción)'})
plt.title("Contribución de categorías a DIM1 y DIM2 (MCA)")
plt.xlabel("Dimensiones del MCA"); plt.ylabel("Categorías (dummies)")
plt.tight_layout(); plt.show()


# Guardar dimensiones MCA
# =========================
# Si DF_MCA no existe, lo reconstruimos desde el objeto 'm' (MCA) y 'k_mca'
if "DF_MCA" not in globals():
    assert "m" in globals() and "k_mca" in globals(), "Faltan 'm' (objeto MCA) o 'k_mca'."
    Fs = m.fs_r(N=k_mca)
    # Usa el índice de tus categóricas (X_cat) o de la matriz disyuntiva (X_disc),
    # según hayas trabajado. Ajusta la siguiente línea si usaste X_disc:
    idx_base = X_cat.index if "X_cat" in globals() else df.index
    DF_MCA = pd.DataFrame(Fs, index=idx_base, columns=[f"DIM{i}" for i in range(1, k_mca+1)])

# Dataset MCA con IDs/objetivo (si existen)
cols_extra = [c for c in ID_COLS if c in df.columns]
if TARGET_COL in df.columns:
    cols_extra += [TARGET_COL]

DF_MCA_out = pd.concat([df[cols_extra].reset_index(drop=True),
                        DF_MCA.reset_index(drop=True)], axis=1)

DF_MCA_out.to_csv("mca_dimensiones.csv", index=False)
st.write("✔ Guardado: mca_dimensiones.csv |", DF_MCA_out.shape)



# Cargar PCs ya guardadas (incluye el objetivo)
DF_PCA_in = DF_PCA_final.copy()  # si está en memoria

# Concatenar con dimensiones del MCA
DF_final = pd.concat([DF_PCA_in.reset_index(drop=True),
                      DF_MCA.reset_index(drop=True)], axis=1)

DF_final.to_csv("tarea1_pca_mca_concat.csv", index=False)
st.write("✔ Guardado: tarea1_pca_mca_concat.csv |", DF_final.shape)


df_final = pd.read_csv("tarea1_pca_mca_concat.csv")
pc_cols  = [c for c in df_final.columns if c.upper().startswith("PC")]
dim_cols = [c for c in df_final.columns if c.upper().startswith("DIM")]
y_cols   = [c for c in df_final.columns if "Diagnóstico de diabetes" in c]

st.write("Filas, Columnas:", df_final.shape)             # -> (6072, 18)
st.write("#PCs:", len(pc_cols), "| #DIMs:", len(dim_cols), "| #Objetivo:", len(y_cols))

st.write(df_final.head())
st.write(df_final.columns.tolist())


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel

#Parámetros + preparación
# ===== Config =====
TARGET_COLS = ["DIQ010", "Diagnóstico de diabetes (Sí/No)", "Diagnóstico de diabetes (Original)"]
TARGET_COL  = "Diagnóstico de diabetes (Sí/No)"   # la que usarás como y
ID_COLS = ["SEQN"]
UMBRAL_ACUM = 0.80                 # 80%
RANDOM_STATE = 42

# 1) Construir X e y SIN ninguna columna objetivo
df = df.dropna(subset=[TARGET_COL]).copy()

drop_cols = set(ID_COLS) | set(TARGET_COLS)

# Separar columnas por tipo según DF previos
num_cols = [c for c in df.select_dtypes(include=["int64","float64","int32","float32"]).columns
            if c not in drop_cols]
cat_cols = [c for c in df.select_dtypes(include=["object","category","bool"]).columns
            if c not in drop_cols]

X = df[num_cols + cat_cols].copy()
y = df[TARGET_COL].map({"No":0, "Sí":1}).astype(int)


# Preprocesador + helper de nombres
def build_preprocessor(num_cols, cat_cols, scale_numeric=True):
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        num_steps.append(("scaler", StandardScaler()))
    num_pipe = Pipeline(num_steps)

    # compatibilidad: sparse_output (>=1.2) vs sparse (<=1.1)
    ohe_kwargs = {"handle_unknown": "ignore", "drop": None}
    try:
        OneHotEncoder(sparse_output=False)
        ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(**ohe_kwargs))
    ])

    return ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

def get_feature_names(pre, num_cols, cat_cols):
    names = list(num_cols)
    oh = pre.named_transformers_["cat"].named_steps["oh"]
    names += oh.get_feature_names_out(cat_cols).tolist()
    return names

def base_col(name):
    for c in cat_cols:
        if name.startswith(c + "_"): return c
    return name

pre_rf = build_preprocessor(num_cols, cat_cols, scale_numeric=False)
rf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
pipe_rf = Pipeline([("pre", pre_rf), ("rf", rf)]).fit(X, y)

feat_names = get_feature_names(pipe_rf.named_steps["pre"], num_cols, cat_cols)
importances = pipe_rf.named_steps["rf"].feature_importances_

# Chequeo anti‑fuga (no debe haber target ni sus dummies)
for t in TARGET_COLS:
    fugas = [f for f in feat_names if f == t or f.startswith(t + "_")]
    assert not fugas, f"¡Fuga detectada! {fugas}"

df_imp_rf = (pd.DataFrame({"feature": feat_names, "importance": importances})
             .sort_values("importance", ascending=False).reset_index(drop=True))
df_imp_rf["cum_importance"] = df_imp_rf["importance"].cumsum() / df_imp_rf["importance"].sum()

sel_rf_feats = df_imp_rf.loc[df_imp_rf["cum_importance"] <= UMBRAL_ACUM, "feature"].tolist()
if not sel_rf_feats: sel_rf_feats = [df_imp_rf.iloc[0]["feature"]]

# Pasar de dummies a columnas originales
orig_sel_rf = sorted(pd.unique([base_col(f) for f in sel_rf_feats]))
DF_sel_RF = df[[c for c in ID_COLS if c in df.columns] + [TARGET_COL] + orig_sel_rf]
DF_sel_RF.to_csv("tarea2_variables_seleccionadas_RF.csv", index=False)

st.write(f"[RF] {len(orig_sel_rf)} variables originales seleccionadas  → tarea2_variables_seleccionadas_RF.csv")


pre_l1 = build_preprocessor(num_cols, cat_cols, scale_numeric=True)
log_l1 = LogisticRegression(penalty="l1", solver="saga", C=1.0, max_iter=4000, n_jobs=-1, random_state=RANDOM_STATE)
pipe_l1 = Pipeline([("pre", pre_l1), ("clf", log_l1)]).fit(X, y)

feat_names_l1 = get_feature_names(pipe_l1.named_steps["pre"], num_cols, cat_cols)
coefs = np.abs(pipe_l1.named_steps["clf"].coef_).ravel()

df_imp_l1 = (pd.DataFrame({"feature": feat_names_l1, "importance": coefs})
             .sort_values("importance", ascending=False).reset_index(drop=True))
df_imp_l1["cum_importance"] = df_imp_l1["importance"].cumsum() / df_imp_l1["importance"].sum()

sel_l1_feats = df_imp_l1.loc[df_imp_l1["cum_importance"] <= UMBRAL_ACUM, "feature"].tolist()
if not sel_l1_feats: sel_l1_feats = [df_imp_l1.iloc[0]["feature"]]

orig_sel_l1 = sorted(pd.unique([base_col(f) for f in sel_l1_feats]))
DF_sel_L1 = df[[c for c in ID_COLS if c in df.columns] + [TARGET_COL] + orig_sel_l1]
DF_sel_L1.to_csv("tarea2_variables_seleccionadas_L1.csv", index=False)

st.write(f"[L1] {len(orig_sel_l1)} variables originales seleccionadas → tarea2_variables_seleccionadas_L1.csv")


from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier

# usa el mismo RANDOM_STATE y TARGET_COL/ID_COLS que definiste arriba

def _make_preprocessor(num_cols, cat_cols):
    num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    # compatibilidad de versiones: sparse_output vs sparse
    ohe_kwargs = {"handle_unknown": "ignore", "drop": None}
    try:
        OneHotEncoder(sparse_output=False); ohe_kwargs["sparse_output"] = False
    except TypeError:
        ohe_kwargs["sparse"] = False
    cat_pipe = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                         ("oh", OneHotEncoder(**ohe_kwargs))])
    return ColumnTransformer([("num", num_pipe, num_cols),
                              ("cat", cat_pipe, cat_cols)])

def eval_auc_df(dfX):
    # y binaria
    y_ = dfX[TARGET_COL].map({"No": 0, "Sí": 1}).astype(int)
    # X sin target ni IDs
    X_ = dfX.drop(columns=[TARGET_COL] + [c for c in ID_COLS if c in dfX.columns], errors="ignore")
    # tipos para este df concreto
    num_cols_eval = X_.select_dtypes(include=["int64","float64","int32","float32"]).columns.tolist()
    cat_cols_eval = X_.select_dtypes(include=["object","category","bool"]).columns.tolist()
    pre = _make_preprocessor(num_cols_eval, cat_cols_eval)
    clf = RandomForestClassifier(n_estimators=400, random_state=RANDOM_STATE, n_jobs=-1)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    return cross_val_score(pipe, X_, y_, cv=cv, scoring="roc_auc").mean()

# ---- comparar RF vs L1 ----
auc_rf = eval_auc_df(DF_sel_RF)
auc_l1 = eval_auc_df(DF_sel_L1)
st.write(f"AUC 5-fold | RF vars: {auc_rf:.4f}")
st.write(f"AUC 5-fold | L1 vars: {auc_l1:.4f}")

DF_sel_final = DF_sel_RF if auc_rf >= auc_l1 else DF_sel_L1
st.write("Ganador:", "RF" if auc_rf >= auc_l1 else "L1")



# valores ya calculados
vals = [auc_rf, auc_l1]
labels = ["RF vars", "L1 vars"]

plt.figure(figsize=(5,4))
bars = plt.bar(labels, vals)
plt.ylim(0, 1)
plt.ylabel("AUC (5-fold)")
plt.title("Comparación de selección de variables")

# anotar valores sobre cada barra
for b, v in zip(bars, vals):
    plt.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.3f}", ha="center", va="bottom")

plt.grid(axis="y", ls="--", alpha=0.4)
plt.tight_layout()
plt.show()


df_pca_mca = pd.read_csv("tarea1_pca_mca_concat.csv")  # de la Tarea 1
auc_pca_mca = eval_auc_df(df_pca_mca)
auc_final   = eval_auc_df(DF_sel_final)

st.write(f"AUC 5-fold | PCA+MCA:        {auc_pca_mca:.4f}")
st.write(f"AUC 5-fold | Sel. variables: {auc_final:.4f}")

# opcional: guarda el dataset ganador para usar después
DF_sel_final.to_csv("tarea2_dataset_ganador.csv", index=False)


import matplotlib.pyplot as plt

# Datos
metodos = ['PCA+MCA', 'Selección L1']
auc_scores = [0.8627, 0.8755]

# Crear gráfica
plt.figure(figsize=(6,5))
bars = plt.bar(metodos, auc_scores, color=['skyblue', 'orange'], edgecolor='black')

# Etiquetas y título
plt.ylabel('AUC promedio (5-fold)')
plt.title('Comparación de AUC: PCA+MCA vs Selección L1')

# Mostrar valores sobre cada barra
for bar, score in zip(bars, auc_scores):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f"{score:.4f}", ha='center', va='bottom', fontsize=10)

plt.ylim(0.85, 0.88)  # Ajustar rango para que la diferencia se note
plt.show()