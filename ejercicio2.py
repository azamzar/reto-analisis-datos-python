import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score

# Cargar el archivo CSV
file_path = 'reto_agua.csv'
df = pd.read_csv(file_path)

# Dividir los datos en variables independientes (X) y target (y)
# Seleccionamos algunas columnas arbitrarias como características
X = df[['management_group', 'source_class', 'quantity_group', 'quality_group']]
y = df['status_group']

# Convertir las variables categóricas en variables dummy
X = pd.get_dummies(X, drop_first=True)
y = pd.get_dummies(y, drop_first=True)

# Dividir los datos en conjuntos de entrenamiento (80%) y prueba (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir los modelos a entrenar
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "Gradient Boosting": GradientBoostingClassifier()
}

# Entrenar y evaluar los modelos
results = {}
for model_name, model in models.items():
    # Entrenar el modelo
    model.fit(X_train, y_train)
    # Evaluar el modelo
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    results[model_name] = {
        "train_score": train_score,
        "test_score": test_score
    }

# Imprimir los resultados
for model_name, metrics in results.items():
    print(f"{model_name}:")
    print(f"  Train Score: {metrics['train_score']}")
    print(f"  Test Score: {metrics['test_score']}\n")

# Evaluar sobreajuste (overfitting) o infraajuste (underfitting)
for model_name, metrics in results.items():
    train_score = metrics['train_score']
    test_score = metrics['test_score']
    if train_score > test_score + 0.1:
        print(f"{model_name} tiene sobreajuste (overfitting)")
    elif train_score < test_score - 0.1:
        print(f"{model_name} tiene infraajuste (underfitting)")
    else:
        print(f"{model_name} tiene buen ajuste (fit)")
