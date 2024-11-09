import pandas as pd
from sklearn.preprocessing import StandardScaler

# Carrega o conjunto de teste
test_data = pd.read_csv('/home/usuario/SpaceShipTitanic/data/test.csv')

median_age = 27 

# Trata valores ausentes no conjunto de teste
test_data['Age'] = test_data['Age'].fillna(median_age)
test_data['RoomService'] = test_data['RoomService'].fillna(0)
test_data['FoodCourt'] = test_data['FoodCourt'].fillna(0)
test_data['ShoppingMall'] = test_data['ShoppingMall'].fillna(0)
test_data['Spa'] = test_data['Spa'].fillna(0)
test_data['VRDeck'] = test_data['VRDeck'].fillna(0)

# Codificação one-hot para variáveis categóricas no conjunto de teste
test_data = pd.get_dummies(test_data, columns=['HomePlanet', 'CryoSleep', 'Destination'], drop_first=True)

# Adiciona colunas ausentes no conjunto de teste e organiza a ordem das colunas
target_columns = ['PassengerId', 'Cabin', 'Age', 'VIP', 'RoomService', 'FoodCourt', 
                  'ShoppingMall', 'Spa', 'VRDeck', 'Name', 'Transported', 
                  'HomePlanet_Europa', 'HomePlanet_Mars', 'CryoSleep_True', 
                  'Destination_PSO J318.5-22', 'Destination_TRAPPIST-1e']

for col in target_columns:
    if col not in test_data.columns:
        test_data[col] = 0  

# Ordena as colunas no conjunto de teste conforme a ordem do conjunto de treino
test_data = test_data[target_columns]

# Padroniza dados numéricos no conjunto de teste usando um novo scaler
num_cols = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
scaler = StandardScaler()
test_data[num_cols] = scaler.fit_transform(test_data[num_cols])

# Exibir as primeiras linhas para confirmação
test_data.head()