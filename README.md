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









# Dados utilizados
Para testar as nossas abordagens utilizamos o conjunto de dados DOTA (Dataset for Object Detection in Aerial Images).

















# Detecção e classificação
A detecção e a classificação de veículos foram implementadas no módulo `processing.py`. Para a detecção dos veículos, utilizamos duas abordagens: uma baseada na limiarização de Otsu e outra na detecção de bordas com o algoritmo de Canny, implementadas com openCV. Ambas as abordagens visam segmentar os veículos do restante da imagem, salva informações dos contornos e por essas informações sobre os contornos (áreas), classifica entre moto, carro e caminhão. Seguem mais detalhes sobre elas:

### **Abordagem 1: Segmentação por Limiarização de Otsu (`segmentAndClassifyVehiclesOtsu`)**


Esta abordagem se baseia na premissa de que a imagem possui um histograma bimodal, ou seja, os pixels dos veículos e do fundo têm níveis de cinza distintos. O método de Otsu calcula automaticamente o limiar ideal para separar esses dois grupos. Em testes com imagens de alto contraste e iluminação uniforme (e.g. `P0002.jpg`, `P0005.jpg`), ela se mostrou uma abordagem eficaz.

No entanto, ela possui limitações significativas para imagens que **não apresentam um contraste claro entre os objetos de interesse e o fundo.**

O método de Otsu pressupõe que o histograma de intensidade da imagem pode ser claramente dividido em dois picos. Essa premissa é violada em imagens como `P0057.jpg` (e outras) pelos seguintes motivos:

1.  **Variações de Iluminação e Sombras**: Sombras fortes projetadas pelos próprios veículos ou por edifícios fazem com que partes de um mesmo objeto tenham valores de intensidade drasticamente distintos. Uma parte de um caminhão sob a luz do sol pode ser muito clara, enquanto outra na sombra pode ser escura.
2.  **Cores Diversas dos Objetos e do Fundo**: Os veículos possuem cores variadas. Na imagem de exemplo, há caminhões com tetos brancos, caminhões com carrocerias escuras. O fundo também não é homogêneo, contendo asfalto (escuro), areia (claro) e telhados de edifícios (claros).
3.  **Similaridade de Intensidade**: Um teto de caminhão branco pode ter a mesma intensidade de cinza que o telhado de um edifício ou a pintura de sinalização no chão. Da mesma forma, um carro preto pode se confundir com o asfalto.



O fluxo do processo é o seguinte:

1.  **Conversão para Escala de Cinza**: A imagem de entrada (BGR) é convertida para escala de cinza, pois a limiarização opera em imagens de um único canal.
    ```python
    # Converte a imagem colorida para escala de cinza
    gray_img = cv2.cvtColor(original_color_img, cv2.COLOR_BGR2GRAY)
    ```

2.  **Limiarização de Otsu**: A função `cv2.threshold` é aplicada com a flag `cv2.THRESH_OTSU`. Ela analisa o histograma da imagem em escala de cinza e determina o valor de limiar ótimo para criar uma imagem binária (preto e branco). Idealmente, os veículos se tornam objetos brancos em um fundo preto.
    ```python
    # cv2.THRESH_OTSU calcula automaticamente o limiar ideal
    ret, otsu_thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    ```

3.  **Operações Morfológicas**: A imagem binária resultante da limiarização raramente é perfeita. Para refinar a segmentação, são aplicadas operações morfológicas:
    * **Abertura (`MORPH_OPEN`)**: Consiste em uma erosão seguida por uma dilatação. Sua principal função é **remover ruídos pequenos** (pixels brancos isolados) da imagem.
    * **Fechamento (`MORPH_CLOSE`)**: Consiste em uma dilatação seguida por uma erosão. É utilizada para **preencher pequenos buracos** dentro dos objetos e conectar componentes próximos.
    ```python
    # kernel_size, open_iter, e close_iter sao parametros da UI
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opening = cv2.morphologyEx(otsu_thresholded_img, cv2.MORPH_OPEN, kernel, iterations=open_iter)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
    segmented_image = closing
    ```

4.  **Detecção e Classificação de Contornos**: Esta etapa é comum a ambas as abordagens e será detalhada mais adiante.

---

### **Abordagem 2: Segmentação por Detecção de Bordas (`segmentAndClassifyByCanny`)**

Esta técnica foca em encontrar as bordas (contornos) dos veículos. A ideia é que as bordas representam uma mudança brusca de intensidade de pixel, o que é característico dos limites de um objeto. Essa técnica não gerou bons resultados.

O fluxo do processo é:

1.  **Conversão para Escala de Cinza**: Similar à primeira abordagem, a imagem é convertida para escala de cinza.

2.  **Detecção de Bordas com Canny**: O algoritmo `cv2.Canny` é aplicado. Ele utiliza um processo de múltiplos estágios para detectar uma vasta gama de bordas em imagens, ao mesmo tempo que reduz o ruído. Os dois limiares (`canny_low_threshold` e `canny_high_threshold`) controlam a sensibilidade da detecção.
    ```python
    # Aplica o Canny para encontrar as bordas
    edges = cv2.Canny(gray_img, canny_low, canny_high)
    ```

3.  **Morfologia para Fechar Contornos**: O resultado do Canny são apenas as linhas das bordas, não objetos preenchidos. Para criar formas sólidas que possam ter sua área medida, aplicamos operações morfológicas:
    * **Dilatação (`dilate`)**: As bordas detectadas são "engrossadas" ou expandidas. Com iterações suficientes, as linhas que formam o contorno de um veículo podem se conectar, formando uma forma fechada.
    * **Fechamento (`MORPH_CLOSE`)**: Em seguida, uma operação de fechamento é aplicada para preencher buracos remanescentes dentro dessas formas recém-criadas.
    ```python
    # Dilata as bordas para conecta-las
    kernel = np.ones((dilate_kernel_size, dilate_kernel_size), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=dilate_iter)
    
    # Aplica um fechamento para preencher buracos
    closing = cv2.morphologyEx(dilated_edges, cv2.MORPH_CLOSE, close_kernel, iterations=2)
    segmented_image = closing
    ```
4.  **Detecção e Classificação de Contornos**: Assim como na primeira abordagem, o próximo passo é analisar os contornos da imagem segmentada.

---

### **Classificação e Contagem (Comum a Ambas as Abordagens)**

Após a etapa de segmentação gerar uma imagem binária (`segmented_image`), o processo para encontrar, filtrar e classificar os objetos é idêntico em ambos os métodos:

1.  **Encontrar Contornos**: A função `cv2.findContours` é usada para detectar os contornos de todos os objetos brancos na imagem segmentada. `RETR_EXTERNAL` garante que apenas os contornos externos sejam capturados.

2.  **Filtragem e Classificação por Área**: O código itera sobre cada contorno encontrado:
    * Calcula-se a área do contorno com `cv2.contourArea()`.
    * O contorno é primeiramente filtrado por uma área mínima e máxima (`area_limiar_moto` e `area_max_geral`). Isso descarta ruídos muito pequenos e áreas muito grandes que provavelmente não são veículos individuais.
    * A classificação é feita através de uma série de comparações sobre a área do objeto com limiares pré-definidos para **caminhão**, **carro** e **moto**.
    ```python
    if area >= area_limiar_caminhao:
        object_type = "Caminhao"
        counts['truck'] += 1
    elif area >= area_limiar_carro:
        object_type = "Carro"
        counts['car'] += 1
    # ... e assim por diante
    ```

3.  **Visualização e Saída**: Para cada objeto classificado, um retângulo delimitador (`bounding box`) e um texto com a classificação, área e proporção são desenhados na cópia da imagem original. Finalmente, a função retorna a imagem anotada, a imagem segmentada e um resumo em texto com a contagem total de cada categoria de veículo.



# Dificuldades e Possı́veis	Melhorias

### 1. Desafios
A metodologia baseada em segmentação por limiar (Otsu) e detecção de bordas (Canny) demonstrou ser funcional, porém inerentemente limitada. Essas técnicas são altamente sensíveis às condições da imagem, como variações de iluminação, contraste, sombras e complexidade do fundo.

O principal desafio encontrado foi a dependência de ajustes de parâmetros. O ajuste de valores como limiares de área, proporção, parâmetros dos algoritmos e parâmetros morfológicos é um processo iterativo e pouco robusto. Uma parametrização que segmenta veículos com sucesso em uma imagem pode falhar completamente em outra, seja por não detectar os alvos ou por classificar incorretamente ruídos e outros elementos do cenário. O que evidencia a baixa capacidade de generalização do método utilizado.

### 2. Possíveis Melhorias:
**Dentro do escopo do trabalho**: A detecção poderia ser aprimorada pela combinação de mais técnicas de segmentação, porém não achamos um modo de fazer isso. Para a classificação, em vez de utilizar apenas a área para classificar, poderíamos treinar um modelo de Machine Learning (como SVM) utilizando características extraídas dos contornos (área, proporção, solidez, etc.), tornando a distinção entre os tipos de veículos mais precisa ou utilizar algoritmos de Deep Learning (uma simples CNN), porém essas abordagens exigiriam a obtenção de dados.

**Fora do escopo do trabalho**: A abordagem mais moderna e eficaz seria a utilização de modelos de Deep Learning State-of-the-Art (SOTA), como o YOLOv8. Esses modelos realizam a detecção e classificação de objetos de forma unificada e são treinados para reconhecer veículos em diversas condições de iluminação, ângulo e oclusão, superando as limitações dos métodos clássicos.