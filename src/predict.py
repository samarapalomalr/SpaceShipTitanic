import pandas as pd
import joblib 
import json 

# Carrega o modelo treinado
model = joblib.load('/home/usuario/SpaceShipTitanic/model/random_forest_model.pkl')

# Carrega o conjunto de teste
test_data = pd.read_csv('/home/usuario/SpaceShipTitanic/data/test.csv')

# Prepara X_test da mesma forma que X no treinamento
X_test = test_data.drop(['PassengerId', 'Name', 'Cabin'], axis=1, errors='ignore')

with open('/home/usuario/SpaceShipTitanic/data/X_columns.json', 'r') as f:
    X_columns = json.load(f)  

X_test = X_test.reindex(columns=X_columns, fill_value=0) 

# Preenche valores ausentes
X_test = X_test.fillna(0)

# Faz previsões com o modelo treinado
predictions = model.predict(X_test)

# Cria o DataFrame de submissão
submission = pd.DataFrame({
    'PassengerId': test_data['PassengerId'],
    'Transported': predictions.astype(bool) 
})

# Salva o arquivo CSV para submissão
submission.to_csv('/home/usuario/SpaceShipTitanic/submissions/submission.csv', index=False)

print("Arquivo de submissão gerado com sucesso!")


