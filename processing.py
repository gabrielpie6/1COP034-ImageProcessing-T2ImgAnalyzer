import cv2
import numpy as np


'''
    Processing class reserved for image analysis tasks:
    - segmentation, morphology, restauration, etc.
'''
class Processing:
    def __init__(self):
        pass
    
    @staticmethod
    def cannyEdgeDetection(img, lowThreshold=100, highThreshold=200, apertureSize=3, L2gradient=False):
        return cv2.Canny(img, lowThreshold, highThreshold, apertureSize=apertureSize, L2gradient=L2gradient)
    
    @staticmethod
    def segmentAndClassifyVehiclesOtsu(
        original_color_img,
        params 
    ):
        """
        Segmenta, classifica e conta veiculos em uma imagem com logica avancada e parametros da UI.
        - original_color_img: A imagem original (BGR) para desenhar os resultados.
        - params: Dicionario contendo todos os limiares e parametros de morfologia.
        Retorna a imagem com as anotacoes e uma lista de strings de log.
        """
        
        logs = []
        # Faz uma copia da imagem original para desenhar sobre ela
        display_img = original_color_img.copy()
        # Converte a imagem colorida para escala de cinza,
        # pois a limiarizacao de Otsu opera em imagens de canal unico
        gray_img = cv2.cvtColor(original_color_img, cv2.COLOR_BGR2GRAY)
        logs.append("INFO: Imagem convertida para escala de cinza.")

        # --- Etapa 1: Limiarizacao e Morfologia Parametrizada ---
        # Aplica a limiarizacao de Otsu
        # cv2.THRESH_BINARY garante que pixels acima do limiar sejam 255 e abaixo sejam 0
        # cv2.THRESH_OTSU calcula automaticamente o limiar ideal
        ret, otsu_thresholded_img = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        logs.append(f"INFO: Limiar de Otsu (invertido) calculado: {ret}")

        kernel_size = params.get('kernel_size', 5)
        open_iter = params.get('open_iter', 1)
        close_iter = params.get('close_iter', 1)

        # kernel_size = 5
        # open_iter = 1
        # close_iter = 1


        # Garante que os parametros sao validos
        if kernel_size % 2 == 0: 
            kernel_size += 1
            
        if open_iter == 0: 
            open_iter = 1
            
        if close_iter == 0: 
            close_iter = 1
        
        # --- Aplica operacoes morfologicas ---
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        # Aplicar operacao de Abertura (erosao seguida de dilatacao)
        # Remove pequenos ruidos
        opening = cv2.morphologyEx(otsu_thresholded_img, cv2.MORPH_OPEN, kernel, iterations=open_iter)

        # Aplicar operacao de Fechamento (dilatacao seguida de erosao)
        # Preenche pequenos buracos dentro dos objetos e conectar partes proximas
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=close_iter)
        segmented_image = closing
        logs.append(f"INFO: Morfologia aplicada (Kernel: {kernel_size}x{kernel_size}, OpenIter: {open_iter}, CloseIter: {close_iter}).")

        # --- Encontra e filtra contornos ---
        # Encontra os contornos na imagem binarizada
        # cv2.RETR_EXTERNAL recupera apenas os contornos externos
        # cv2.CHAIN_APPROX_SIMPLE compacta os pontos redundantes do contorno
        contours, _ = cv2.findContours(segmented_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        logs.append(f"INFO: {len(contours)} contornos brutos encontrados.")


        # --- Classificacao utilizando areas ---
        # Usa um dicionario para as contagens, como na sua funcao mais nova
        counts = {'car': 0, 'motorcycle': 0, 'truck': 0, 'unknown': 0}
        
        # Pega os limiares de area do dicionario de parametros
        area_limiar_moto = params.get('min_area_moto', 200)
        area_limiar_carro = params.get('min_area_carro', 400)
        area_limiar_caminhao = params.get('min_area_caminhao', 800)
        area_max_geral = params.get('max_area_geral', 20000)
        
        logs.append("INFO: Iniciando contagem e classificacao...")
        for contour in contours:
            area = cv2.contourArea(contour)

            # Filtro inicial de ruido
            if area < area_limiar_moto or area > area_max_geral:
                continue

            x, y, w, h = cv2.boundingRect(contour)
            proporcao = float(w) / h if h > 0 else 0
            
            if area >= area_limiar_caminhao:
                object_type, color = "Caminhao", (255, 0, 0) # Azul
                counts['truck'] += 1
            elif area >= area_limiar_carro:
                object_type, color = "Carro", (0, 255, 0) # Verde
                counts['car'] += 1
            elif area >= area_limiar_moto:
                object_type, color = "Moto", (0, 255, 255) # Amarelo
                counts['motorcycle'] += 1
            else:
                object_type, color = 'Desconhecido', (0, 0, 255) # Vermelho
                counts['unknown'] += 1
            
            # Adiciona informacao detalhada na imagem para debug
            info = f"{object_type} A:{int(area)} P:{proporcao:.2f}"
            cv2.rectangle(display_img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(display_img, info, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            logs.append(f"LOG: Objeto detectado: Tipo={object_type}, Area={int(area)}, Proporcao={proporcao:.2f}")

        # --- Etapa 3: Formata o resultado para ser compativel com o app.py ---
        total_veiculos = counts['car'] + counts['motorcycle'] + counts['truck']
        summary = [
            "INFO: Contagem e classificacao concluidas.",
            "----------------- RESULTADO -----------------",
            f"Total de veiculos classificados: {total_veiculos}",
            f"Carros: {counts['car']}",
            f"Motos: {counts['motorcycle']}",
            f"Caminhoes: {counts['truck']}",
            f"Objetos desconhecidos: {counts['unknown']}",
            "-------------------------------------------"
        ]

        # (imagem, lista de strings)
        return display_img, segmented_image, summary + logs