import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Joblib para salvar o modelo

# Carrega o conjunto de treino
train_data = pd.read_csv('/home/usuario/SpaceShipTitanic/data/train.csv')

# Trata valores ausentes em 'Age' e demais colunas do conjunto de treino
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())
train_data['RoomService'] = train_data['RoomService'].fillna(0)
train_data['FoodCourt'] = train_data['FoodCourt'].fillna(0)
train_data['ShoppingMall'] = train_data['ShoppingMall'].fillna(0)
train_data['Spa'] = train_data['Spa'].fillna(0)
train_data['VRDeck'] = train_data['VRDeck'].fillna(0)

# Codificação one-hot para variáveis categóricas
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'CryoSleep', 'Destination'], drop_first=True)

# Defini X (features) e y (target)
X = train_data.drop(['PassengerId', 'Transported', 'Name', 'Cabin'], axis=1, errors='ignore')
y = train_data['Transported'].astype(int)  # Transforma para valores 0 e 1

# Trata valores ausentes remanescentes em X
X = X.fillna(0)

# Dividi os dados em conjuntos de treino e validação
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Inicializa o modelo Random Forest
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Faz as previsões no conjunto de validação
y_pred = model.predict(X_val)

# Avalia o modelo
accuracy = accuracy_score(y_val, y_pred)
print(f'Acurácia no conjunto de validação: {accuracy:.4f}')

# Salvar o modelo treinado
joblib.dump(model, '/home/usuario/SpaceShipTitanic/model/random_forest_model.pkl')  
print("Modelo treinado salvo com sucesso!")


