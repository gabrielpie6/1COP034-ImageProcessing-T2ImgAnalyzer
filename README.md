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