# Desafio - Parte 2

Este repositório contém o código e a documentação para um projeto de predição desenvolvido pela equipe Soldadinhos de Araripe, Time 6 do Mandacaru.dev. Cada membro da equipe foi responsável por implementar um modelo de predição utilizando diferentes algoritmos de Machine Learning, incluindo Rede Neural, SVM (*Support Vector Machine*), Gradient Boosting, XGBoost (*Extreme Gradient Boosting*) e Random Forest. 

*******
## Sumário

 - [Autores](#autores)
 - [Requisitos do Ambiente](#requisitos)
 - [Objetivo do Projeto](#objetivo)
 - [Base de Dados](#dataset)
 - [Modelos Implementados](#modelos)
   - [Rede Neural](#rede-neural)
   - [SVM](#svm)
   - [Gradient Boosting](#gb)
   - [XGBoost](#xgb)
   - [Random Forest](#rf)

*******

<div id = 'autores' />
  
## Autores 

As seguintes pessoas trabalharam no desenvolvimento do projeto:

- [Breno Mota do Nascimento](https://github.com/Escumalha)
- [Francisca Gardênia Silva Estevam](https://github.com/gardeniaestevam)
- [Julyanderson Alves Cavalcanti de Lima](https://github.com/ansderson122)
- [Murillo Inácio da Costa Silva](https://github.com/likotrico)
- [Vinicius Ramon Barros Teles Silva](https://github.com/ViniRamon1)

<div id = 'requisitos' />

## Requisitos do Ambiente

Para a execução do projeto sem nenhum empecilho, é necessário instalar as bibliotecas de cada notebook. Foi definido pela equipe que cada modelo teria um notebook para para cada modelo a fim facilitar a execução e vizualização dos resultados individuais dos modelos e possibilitar um melhor trabalho em equipe da parte analitica.

<div id = 'objetivo' />

## Objetivo do Projeto

O objetivo deste projeto é criar modelos de predição robustos e eficientes para análise de sentimentos. Cada modelo foi desenvolvido por um membro da equipe, utilizando abordagens e algoritmos distintos.

<div id = 'dataset' />

## Base de Dados

A base de dados utilizada é a [Financial Sentiment Analysis](https://www.kaggle.com/datasets/sbhatti/financial-sentiment-analysis) que pode ser encontrada no Kaggle. Certifique-se de fazer o upload do arquivo CSV para o ambiente de trabalho do Google Colab ou diretório do Jupyter Notebook.

A base de dados é destinada para a análise de sentimentos no mercado financeiro e é a combinação de duas bases de dados diferentes, a *FiQA* e a *Financial PhraseBank*. Possui 5841 sentenças, sem valores nulos, sendo composta por 54% de sentenças com sentimentos neutros, 32% com sentimentos positivos e 15% com sentimentos negativos.

<div id = 'prepro' />

## Pré-processamento 

O pré-processamento dos dados é uma etapa crucial em qualquer tarefa de análise de dados. Pois a a qualidade e condição dos dados impactam diretamente nos resultados de algoritmos de aprendizado de máquina. É preciso garantir a qualidade, consistência e relevância dos dados. 

É importante salientar que a interpretação do computador é diferente da interpretação humana e alguns termos e detalhes que podem mudar totalmente a interpretação de uma palavra ou expressão para um ser humano, pode não fazer diferença para o computador. Por exemplo, as palavras 'violeta' e 'Violeta' possuem sentidos diferentes, uma vez que a letra maiúscula no início da palavra indica que é um nome próprio, o ser humano consegue identificar que a segunda palavra pode ser o nome de uma pessoa, de uma marca, uma cidade. Já para o computador, as duas palavras são iguais. Diante disto, nesta etapa, todos os modelos aplicaram as técnicas descritas a seguir:

- Remoção de caracteres especiais;
- Conversão de letras maiúsculas para minúsculas;
- Remoção de stopwords, que são as palavras como artigos, verbos de ligação, advérbios, ou outros tipos de palavras que não interferem na interpretação do computador;
- Tokenização que consiste em transformar a frase em uma lista de palavras;

<div id = 'modelos' />

## Modelos Implementados

A segeuir, tem-se um apanhado de cada modelo desenvolvido no projeto, contendo detalhes sobre a técnica utilizada, parâmetros e resultados obtidos. 

<div id='rede-neural'/>
 
### Rede Neural

Aqui, além das etapas de pré-processamento descritas, também foi aplicada a remoção de caracteres numéricos e a tradução das sentenças para o inglês. As classes de sentimentos foram *dummyficadas* com a ajuda da biblioteca [Keras](https://keras.io/). A *dummyficação* é substituir os rótulos das classes por outros a fim de facilitar a análise dos dados. 

A base de dados foi dividida em 90% para treino e 10% para teste. Foi utilizada a Rede Neural Sequencial disponível no Keras. O modelo foi criado com três camadas, a primeira contendo quatro neurônios e função de ativação *sigmoid*, a segunda camada é uma camada de *drop out* com uma taxa de 40% e a última camada possui três neurônios com função de ativação softmax. O modelo possui 25 épocas, *batch size* de 32. Uma acurácia de 0.6407 foi obtida com esses parâmetros. A figura abaixo mostra um gráfica com os valores de acurácia a cada época. 

![acuracia-rede-neural](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/f775acd5-d900-4915-8f20-f7d741412409)

A figura abaixo mostra os valores de perca para cada época.

![loss-rede-neural](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/ff75d86f-aaaa-4546-b524-0d4e54a17074)

* **Membro Responsável:** Murilo Ínacio

<div id='svm'/>

### SVM 

Nesse modelo, outras técnicas de pré-processamento também foram aplicadas, sendo elas a remoção de caracteres numéricos e a lematização. Essa segunda técnica consiste na conversão das palavras para o seu lema, por exemplo, no processo de lematização a palavra 'pedreira' vira 'pedra'. Em seguida, os rótulos das classes foram *dummyficados* seguindo a tabela a seguir.

Classe   | Rótulo
--------- | ------
Negativo | -1
Neutro | 0
Positivo | 1

No treinamento, a base de dados foi dividida em 60% para treinamento e 40% para teste. Na vetorização, o parâmetro 'ngram_range=(1, 2)' foi utulizado para considerar tanto palavras individuais quanto sequências de duas palavras. No modelo, o parâmetro 'kernel=poly' foi utilizado para criar uma curva polinomial e 'degree=2' para definir que esse polinômio é de grau 2, enquanto o coeficiente independente é de 54.9. Para determinar esse coeficiente, vários testes foram realizados, pois foi entendido que um grande impacto na precisão.

Ao final, uma acurácia de 0,693 foi obtida. A figura abaixo mostra a matriz de confusão obtida.

![matriz-confusão-svm](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/c5102c24-68f6-421d-a843-800062bea000)

A figura abaixo mostra a precisão de cada classe.

![precisao-svm](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/2740becf-31ae-43a6-9855-bb7b67c858f3)

A figura abaixo mostra a f1-score de cada classe.

![precisao-f1](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/dd91ca22-b50d-4dc3-8797-495c9b400a65)


* **Membro Responsável:** Julyanderson

<div id='gb'/>

### Gradient Boosting 

Para este modelo, a base de dados foi dividida em 90% para treino e 10% para teste. A remoção de caracteres numéricos e técnica de lematização também foram aplicadas. As classes foram *dummyficadas* seguindo a tabela a seguir.

Classe   | Rótulo
--------- | ------
Negativo | 0
Neutro | 1
Positivo | 2

Foram obtidas acurácia de 0,68, precisão de 0,65, recall de 0,53, f1 de 0,55 e a figura a seguir mostra a matriz de confusão obtida.

![matriz-gb](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/9d2ba139-13f3-4183-bfdc-b441fc382d7e)

O gráfico de dispersão do modelo pode ser visto na figura abaixo.

![dispersao-gb](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/a490816f-2526-4b4a-bb66-5268938fb37c)

<div id='xgb'/>

### XGboost

O pré-processado utilizado no modelo anteior foi utilizado nesse modelo. Foram obtidas acurácia de 0,68, precisão de 0,57, recall de 0,54, f1 de 0,55 e a figura a seguir mostra a matriz de confusão obtida.

![matriz-xgb](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/c52c85b8-74f4-4e33-8875-6672884b5c3f)

O gráfico de dispersão do modelo pode ser visto na figura abaixo.

![dispersao-xgb](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/2d1c2a64-758c-48c8-a943-4bd9de70fbfc)


* **Membro Responsável:** Vinícius Ramon

<div id='rf'/>

### Random Forest

A técnica de lematização foi aplicada nesse modelo. Além disso, *POS Tagging* também foi aplicado. *POS* é uma sigla para *Part of Speech* e essa técnica é a assossiação cada palavra a sua classe gramatical.

A base de dados foi dividida em 90% para treino e 10% para teste e foi obtido uma acurácia de 0,782, precisão de 0,782 e recall de 0,782. A matriz de confusão obtida pode ser vista na figura abaixo. 

![matriz-rf](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/133ec9cb-74fa-4b72-834a-08dd4e23f34d)

Um gráfico com as *pos taggs* mais importantes para a classificação pode ser visto na figura abaixo.

![top-postag-rf](https://github.com/gardeniaestevam/time6-mandacaru.dev/assets/72508388/2d9b65fa-1d2a-4b88-9bd3-6aa957ee6068)

* **Membro Responsável:** Breno

*******
