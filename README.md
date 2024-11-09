# Spaceship Titanic - Kaggle Competition

Este repositório contém a solução para a competição "Spaceship Titanic" no Kaggle. O objetivo desta competição é prever quais passageiros foram transportados para uma dimensão alternativa após uma anomalia espaço-temporal.


## Descrição do Problema

O desafio proposto nesta competição é prever a variável `Transported`, que indica se um passageiro foi transportado para outra dimensão. O conjunto de dados possui informações sobre os passageiros, incluindo dados demográficos, informações sobre a viagem e os serviços utilizados a bordo.

### Variáveis principais

- **PassengerId**: ID único de cada passageiro.
- **HomePlanet**: Planeta de origem do passageiro.
- **CryoSleep**: Indica se o passageiro estava em criossono durante a viagem.
- **Cabin**: Localização da cabine do passageiro.
- **Destination**: Destino da viagem do passageiro.
- **Age**: Idade do passageiro.
- **VIP**: Indica se o passageiro era VIP.
- **RoomService, FoodCourt, ShoppingMall, Spa, VRDeck**: Gasto do passageiro em diferentes serviços do navio.
- **Transported**: Variável alvo, indicando se o passageiro foi transportado para outra dimensão.

## Estrutura do Projeto

O projeto é estruturado da seguinte forma:
- **data**: Contém os arquivos de dados.
- **model**: Contém o modelo treinado.
- **notebooks**: Jupyter notebooks para experimentação e visualização de dados.
- **src**: Scripts Python para pré-processamento e construção do modelo.
- **submissions**: contém as previsoes do modelo.
- **requirements.txt**: Dependências do projeto.

