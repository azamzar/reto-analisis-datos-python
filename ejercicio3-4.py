import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ruta del archivo
file_path = 'reto_agua.csv'
data = pd.read_csv(file_path)

# Eliminar filas con valores nulos en 'status_group'
data = data.dropna(subset=['status_group'])

# Codificar la columna 'status_group'
label_encoder = LabelEncoder()
data['status_group'] = label_encoder.fit_transform(data['status_group'])

# Codificar características categóricas
categorical_features = data.select_dtypes(include=['object']).columns
data_encoded = data.copy()

for feature in categorical_features:
    data_encoded[feature] = LabelEncoder().fit_transform(data_encoded[feature].astype(str))

# Eliminar la columna 'id'
data_encoded = data_encoded.drop(columns=['id'])

# Paso 2: Mapa de Calor de Correlación
# Seleccionar columnas relevantes para la correlación
columns = ['amount_tsh', 'funder', 'gps_height', 'installer', 'longitude', 'latitude', 'num_private', 'basin', 'status_group']
data_subset = data_encoded[columns]

# Crear una matriz de correlación
correlation_matrix = data_subset.corr()

# Visualizar la matriz de correlación con un mapa de calor
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matriz de Correlación')
plt.show()

# Paso 3: Detectar y Eliminar Valores Atípicos
# Visualizar valores atípicos en 'population' y 'gps_height' con diagramas de caja
plt.figure(figsize=(14, 6))

# Diagrama de caja para 'population'
plt.subplot(1, 2, 1)
sns.boxplot(x=data_encoded['population'])
plt.title('Boxplot de Population')

# Diagrama de caja para 'gps_height'
plt.subplot(1, 2, 2)
sns.boxplot(x=data_encoded['gps_height'])
plt.title('Boxplot de GPS Height')

plt.show()

# Definir una función para eliminar valores atípicos basados en IQR
def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

# Eliminar valores atípicos de 'population' y 'gps_height'
data_cleaned = remove_outliers(data_encoded, 'population')
data_cleaned = remove_outliers(data_cleaned, 'gps_height')

# Paso 4: Entrenar el modelo con datos limpios
# Separar características y objetivo
X_cleaned = data_cleaned.drop(columns=['status_group'])
y_cleaned = data_cleaned['status_group']

# Dividir datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_cleaned, y_cleaned, test_size=0.2, random_state=42)

# Selección de características
selector = SelectKBest(score_func=f_classif, k=21)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

# Entrenar el modelo
model = RandomForestClassifier(random_state=42)
model.fit(X_train_selected, y_train)

# Evaluar el modelo
y_train_pred = model.predict(X_train_selected)
accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train, y_train_pred, average='weighted')
recall = recall_score(y_train, y_train_pred, average='weighted')

accuracy, precision, recall

# Paso 5: Búsqueda de Hiperparámetros
from sklearn.model_selection import GridSearchCV

# Definir los hiperparámetros para buscar
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Configurar la búsqueda de hiperparámetros
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)

# Ajustar la búsqueda de hiperparámetros
grid_search.fit(X_train_selected, y_train)

# Mejor modelo encontrado por la búsqueda de hiperparámetros
best_params = grid_search.best_params_

# Entrenar el modelo con los mejores parámetros
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train_selected, y_train)

# Evaluar el mejor modelo
y_train_pred_best = best_model.predict(X_train_selected)
accuracy_best = accuracy_score(y_train, y_train_pred_best)
precision_best = precision_score(y_train, y_train_pred_best, average='weighted')
recall_best = recall_score(y_train, y_train_pred_best, average='weighted')

accuracy_best, precision_best, recall_best
