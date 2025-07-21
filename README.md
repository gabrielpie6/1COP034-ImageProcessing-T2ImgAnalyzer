# Descrição

A ideia é construir um sistema que permita inserir imagens, então realizar alguns tipos de processamentos nestas imagens para posteriormente aplicar algoritmos de segmentação e detecção, especialmente no contexto de veículos em trânsito.

# Setup

Criação de um ambiente virtual em Python para desenvolvimento da aplicação. Uma forma de minimizar conflitos entre dependências e pacotes externos:
```
python3 -m venv env
```

### 1. Ativação do Environment
Para a utilização do enviroment criado, estando no diretório que contém a pasta ```env```, é necessário realizar a ativação:

#### 1.A. Linux / macOS

```
source env/bin/activate
```

#### 1.B. Windows via CMD

```
env\Scripts\activate
```

#### 1.C. Windows via Powershell

```
.\env\Scripts\Activate.ps1
```


### 2. Instalação
```
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Execução
```
python app.py
```






# Dificuldades e Possı́veis	Melhorias

### 1. Desafios
A metodologia baseada em segmentação por limiar (Otsu) e detecção de bordas (Canny) demonstrou ser funcional, porém inerentemente limitada. Essas técnicas são altamente sensíveis às condições da imagem, como variações de iluminação, contraste, sombras e complexidade do fundo.

O principal desafio encontrado foi a dependência de ajustes de parâmetros. O ajuste de valores como limiares de área, proporção, parâmetros dos algoritmos e parâmetros morfológicos é um processo iterativo e pouco robusto. Uma parametrização que segmenta veículos com sucesso em uma imagem pode falhar completamente em outra, seja por não detectar os alvos ou por classificar incorretamente ruídos e outros elementos do cenário. O que evidencia a baixa capacidade de generalização do método utilizado.

### 2. Possíveis Melhorias:
**Dentro do escopo do trabalho**: A detecção poderia ser aprimorada pela combinação de mais técnicas de segmentação, porém não achamos um modo de fazer isso. Para a classificação, em vez de utilizar apenas a área para classificar, poderíamos treinar um modelo de Machine Learning (como SVM) utilizando características extraídas dos contornos (área, proporção, solidez, etc.), tornando a distinção entre os tipos de veículos mais precisa ou utilizar algoritmos de Deep Learning (uma simples CNN), porém essas abordagens exigiriam a obtenção de dados.

**Fora do escopo do trabalho**: A abordagem mais moderna e eficaz seria a utilização de modelos de Deep Learning State-of-the-Art (SOTA), como o YOLOv8. Esses modelos realizam a detecção e classificação de objetos de forma unificada e são treinados para reconhecer veículos em diversas condições de iluminação, ângulo e oclusão, superando as limitações dos métodos clássicos.