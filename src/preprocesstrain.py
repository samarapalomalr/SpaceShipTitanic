# Pré-processamento de Dados de treinamento                                                                                                                        
import pandas as pd

# Defini a opção para evitar downcasting silencioso
pd.set_option('future.no_silent_downcasting', True)

# Carrega dados
train_data = pd.read_csv('/home/usuario/SpaceShipTitanic/data/train.csv')

# Trata valores ausentes
train_data['Age'] = train_data['Age'].fillna(train_data['Age'].median())  # Preenchendo a idade com a mediana
train_data['HomePlanet'] = train_data['HomePlanet'].fillna(train_data['HomePlanet'].mode()[0])  # Preenchendo o planeta de origem com o valor mais comum
train_data['CryoSleep'] = train_data['CryoSleep'].fillna(train_data['CryoSleep'].mode()[0])  # Preenchendo a animação suspensa com o valor mais comum

# Codificação One-Hot para variáveis categóricas
train_data = pd.get_dummies(train_data, columns=['HomePlanet', 'CryoSleep', 'Destination'], drop_first=True)
 
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
train_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']] = scaler.fit_transform(
    train_data[['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']])

# Exibir as primeiras linhas para verificar o pré-processamento
train_data.head()
