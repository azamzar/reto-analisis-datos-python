import pandas as pd

# Cargar el archivo CSV
file_path = 'reto_agua.csv'
df = pd.read_csv(file_path)

# Mostrar los primeros 5 datos
print("Primeros 5 datos:")
print(df.head())

# Análisis exploratorio de la estructura y los datos
print("\nInformación del DataFrame:")
print(df.info())

print("\nDescripción del DataFrame:")
print(df.describe(include='all'))

print("\nValores nulos por columna:")
print(df.isnull().sum())

# Responder a las preguntas específicas
# 1. ¿Veis alguna columna que no consideréis necesaria para el modelo?
# La columna 'id' puede no ser necesaria ya que generalmente es solo un identificador único.
print("\nColumnas no necesarias: 'id'")

# 2. ¿Cuántos datos totales hay en el dataset?
total_datos = len(df)
print(f"\nTotal de datos: {total_datos}")

# 3. ¿Hay valores nulos? En ese caso, ¿qué columnas los tienen?
columnas_con_nulos = df.columns[df.isnull().any()].tolist()
print(f"\nColumnas con valores nulos: {columnas_con_nulos}")

# 4. ¿Detectáis alguna columna que tenga datos anómalos? En ese caso, ¿cuáles?
datos_anomalos = df.describe().loc[['min', 'max']]
print(f"\nDatos anómalos:\n{datos_anomalos}")

# Transformar todas las variables objetos en categóricas o numéricas
columns_object = df.loc[:, df.dtypes == object].columns

for col in columns_object:
    df[col] = df[col].astype('category').cat.add_categories(['missing']).fillna('missing')

# Verificar la transformación
print("\nInformación del DataFrame transformado:")
print(df.info())