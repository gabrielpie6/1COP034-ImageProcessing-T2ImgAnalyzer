import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from transform  import Transformation
from processing import Processing

vp_width  = 600
vp_height = 400
leftPanel_width = int(vp_width * 0.4)

# Globals for image data and dimensions
pure_img      = None
current_img   = None
current_img_w = 0
current_img_h = 0

result_loading_img    = None
result_transform_img  = None
result_processing_img = None
result_segmented_img = None
display_mode = "single"

# Load image and create static texture
def load_image(sender, app_data, user_data):
    global current_img, current_img_w, current_img_h, pure_img, result_loading_img
    file_path = app_data['file_path_name']
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load: {file_path}")
        return

    # Convert to RGBA
    if img.shape[2] == 3:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    else:
        img_rgba = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)

    pure_img = img_rgba.copy()

    result_loading_img = img_rgba.copy()
    change_image(img_rgba)
    dpg.show_item('loading_bwview_checkbox')
    #
    dpg.show_item('secImgTransformation')
    dpg.show_item('secImgProcessing')
    dpg.show_item('secImgDetectionOtsu')
    dpg.show_item('secImgDetectionCanny')

def change_image(main_img_rgba, segmented_img_rgba=None):
    """ Prepara e atualiza as texturas. Aceita uma segunda imagem opcional. """
    global display_mode
    if main_img_rgba is None: return

    # Prepara dados da imagem principal
    h1, w1, _ = main_img_rgba.shape
    data1 = np.asfarray(main_img_rgba.ravel(), dtype='f') / 255.0
    
    # Cria ou atualiza a textura principal
    if not dpg.does_item_exist('main_tex'):
        dpg.add_dynamic_texture(width=w1, height=h1, default_value=data1, tag='main_tex', parent='texreg')
    else:
        dpg.configure_item('main_tex', width=w1, height=h1, default_value=data1)

    # Se uma segunda imagem foi fornecida, prepara sua textura e muda o modo
    if segmented_img_rgba is not None:
        display_mode = "dual"
        h2, w2, _ = segmented_img_rgba.shape
        data2 = np.asfarray(segmented_img_rgba.ravel(), dtype='f') / 255.0
        if not dpg.does_item_exist('segmented_tex'):
            dpg.add_dynamic_texture(width=w2, height=h2, default_value=data2, tag='segmented_tex', parent='texreg')
        else:
            dpg.configure_item('segmented_tex', width=w2, height=h2, default_value=data2)
    else:
        # Se nenhuma segunda imagem foi fornecida, garante o modo de imagem unica
        display_mode = "single"

    update_image_display()


# Redraw image to fit current drawlist size, centered with white background
def update_image_display():
    """ Desenha uma ou duas imagens, dependendo do modo de exibicao. """
    global display_mode
    
    canvas_w = dpg.get_item_width('RightChild')
    canvas_h = dpg.get_item_height('RightChild')
    if not canvas_w or not canvas_h: return
    
    dpg.delete_item('ImageCanvas', children_only=True)

    if display_mode == "single" and dpg.does_item_exist('main_tex'):
        # --- Logica para desenhar UMA imagem centralizada ---
        tex_w = dpg.get_item_width('main_tex')
        tex_h = dpg.get_item_height('main_tex')
        scale = min(canvas_w / tex_w, canvas_h / tex_h)
        new_w, new_h = int(tex_w * scale), int(tex_h * scale)
        x, y = (canvas_w - new_w) // 2, (canvas_h - new_h) // 2
        dpg.draw_image('main_tex', pmin=[x, y], pmax=[x + new_w, y + new_h], parent='ImageCanvas')

    elif display_mode == "dual" and dpg.does_item_exist('main_tex') and dpg.does_item_exist('segmented_tex'):
        # --- Logica para desenhar DUAS imagens lado a lado ---
        available_w = canvas_w / 2 - 10
        
        # Imagem 1 (Classificada)
        tex_w1, tex_h1 = dpg.get_item_width('main_tex'), dpg.get_item_height('main_tex')
        scale1 = min(available_w / tex_w1, canvas_h / tex_h1)
        new_w1, new_h1 = int(tex_w1 * scale1), int(tex_h1 * scale1)
        offset_x1 = (available_w - new_w1) / 2 + 5
        offset_y1 = (canvas_h - new_h1) / 2
        dpg.draw_image('main_tex', pmin=[offset_x1, offset_y1], pmax=[offset_x1 + new_w1, offset_y1 + new_h1], parent='ImageCanvas')
        dpg.draw_text(pos=[offset_x1, offset_y1 - 20], text="Classified", size=15, color=[255,255,255,255], parent='ImageCanvas')
        
        # Imagem 2 (Segmentada)
        tex_w2, tex_h2 = dpg.get_item_width('segmented_tex'), dpg.get_item_height('segmented_tex')
        scale2 = min(available_w / tex_w2, canvas_h / tex_h2)
        new_w2, new_h2 = int(tex_w2 * scale2), int(tex_h2 * scale2)
        offset_x2 = canvas_w / 2 + (available_w - new_w2) / 2 + 5
        offset_y2 = (canvas_h - new_h2) / 2
        dpg.draw_image('segmented_tex', pmin=[offset_x2, offset_y2], pmax=[offset_x2 + new_w2, offset_y2 + new_h2], parent='ImageCanvas')
        dpg.draw_text(pos=[offset_x2, offset_y2 - 20], text="Segmented", size=15, color=[255,255,255,255], parent='ImageCanvas')

# Callback: when viewport resizes, adjust child sizes and redraw
def on_viewport_resize(sender, app_data):
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    left_w = int(vp_w * 0.3)
    right_w = vp_w - left_w - 40
    height = vp_h - 55
    # height = dpg.get_item_height('LeftChild')

    # Resize left/right containers and drawlist
    dpg.configure_item('LeftChild', width=left_w)
    dpg.configure_item('RightChild', width=right_w, height=height - 23)
    dpg.configure_item('ImageCanvas', width=right_w, height=height - 17 - 25)

    update_image_display()

def change_section(sender, app_data, user_data):
    # print("Section Changed", sender, user_data)
    # Hide all sections
    dpg.hide_item('LoadingSection')
    dpg.hide_item('TransformationSection')
    dpg.hide_item('ProcessingSection')
    dpg.hide_item('OtsuClassificationSection')
    dpg.hide_item('CannyClassificationSection')
    

    if user_data == "LoadingSection":
        if result_loading_img is not None:
            loading_section_update(None, None, None)
            dpg.show_item('LoadingSection')
    elif user_data == "TransformationSection":
        transform_section_update(None, None, None)
        dpg.show_item('TransformationSection')
    elif user_data == "ProcessingSection":
        process_section_update(None, None, None)
        dpg.show_item('ProcessingSection')
    elif user_data == "OtsuClassificationSection":
        otsu_section_update(None, None, None)
        dpg.show_item('OtsuClassificationSection')
    elif user_data == "CannyClassificationSection":
        canny_section_update(None, None, None)
        dpg.show_item('CannyClassificationSection')


    # Show the selected section
    dpg.show_item(user_data)




def loading_section_update(sender, app_data, user_data):
    res = computeLoading()
    change_image(res)

def computeLoading():
    global result_loading_img

    if dpg.get_value('loading_bwview_checkbox'):
        result_img = Transformation.BWConversion(result_loading_img)
    else:
        result_img = result_loading_img.copy()
    
    return result_img







def transform_section_update(sender, app_data, user_data):
    global result_transform_img
    computeTransformation()

    if dpg.get_value('transform_freqview_checkbox'):
        view = Transformation.frequencyDomain(result_transform_img)
    elif dpg.get_value('transform_bandpass_maskview'):
        split   = dpg.get_value('transform_split_slider')
        inner   = dpg.get_value('transform_inner_slider')
        outer   = dpg.get_value('transform_outer_slider')
        fadeIn  = dpg.get_value('transform_fadein_slider')
        fadeOut = dpg.get_value('transform_fadeout_slider')
        img     = Transformation.bandPassMask(result_transform_img.shape[1], result_transform_img.shape[0], split, inner, outer, fadeIn, fadeOut)
        view    = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        view = result_transform_img

    change_image(cv2.cvtColor(view, cv2.COLOR_GRAY2RGBA))

def computeTransformation():
    global result_loading_img, result_transform_img

    result_img = result_loading_img.copy()

    result_img = Transformation.RGBAtoGray(result_img)

    if dpg.get_value('transform_flipH_checkbox'):
        result_img = Transformation.flipHorizontal(result_img)
    if dpg.get_value('transform_flipV_checkbox'):
        result_img = Transformation.flipVertical(result_img)
    if dpg.get_value('transform_histEq_checkbox'):
        result_img = Transformation.histogramEqualization(result_img)

    # Mean Blur
    if dpg.get_value('transform_mean_blur_checkbox'):
        dpg.enable_item('transform_mean_blur')
        ksize = int(dpg.get_value('transform_mean_blur_kernel_slider'))
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd size
        result_img = Transformation.meanBlur(result_img, ksize, ksize)
    else:
        dpg.disable_item('transform_mean_blur')
    
    # Gaussian Blur
    if dpg.get_value('transform_gaussian_blur_checkbox'):
        dpg.enable_item('transform_gaussian_blur')
        ksize = int(dpg.get_value('transform_gaussian_blur_kernel_slider'))
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd size
        sigmaX = dpg.get_value('transform_gaussian_blur_sigmaX_slider')
        sigmaY = dpg.get_value('transform_gaussian_blur_sigmaY_slider')
        result_img = Transformation.gaussianBlur(result_img, ksize, ksize, sigmaX=sigmaX, sigmaY=sigmaY)
    else:
        dpg.disable_item('transform_gaussian_blur')

    # Median Blur
    if dpg.get_value('transform_median_blur_checkbox'):
        dpg.enable_item('transform_median_blur')
        ksize = int(dpg.get_value('transform_median_blur_kernel_slider'))
        ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd size
        result_img = Transformation.medianBlur(result_img, ksize)
    else:
        dpg.disable_item('transform_median_blur')


    if dpg.get_value('transform_spatial_checkbox'):
        dpg.enable_item('transform_spatial_filtering')

        # Laplacian Filter
        if dpg.get_value('transform_laplacian_checkbox'):
            dpg.enable_item('transform_laplacian')
            ksize = int(dpg.get_value('transform_laplacian_kernel_slider'))
            ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd size
            result_img = Transformation.laplacianFilter(result_img, ksize)
        else:
            dpg.disable_item('transform_laplacian')
        
        # Sobel Filters
        if dpg.get_value('transform_sobel_x_checkbox'):
            dpg.enable_item('transform_sobel_x')
            ksize = int(dpg.get_value('transform_sobel_x_kernel_slider'))
            ksize = ksize if ksize % 2 == 1 else ksize + 1  # Ensure odd size
            result_img = Transformation.sobelFilter(result_img, dx=1, dy=0, ksize=ksize)
        else:
            dpg.disable_item('transform_sobel_x')
        if dpg.get_value('transform_sobel_y_checkbox'):
            dpg.enable_item('transform_sobel_y')
            ksize = int(dpg.get_value('transform_sobel_y_kernel_slider'))
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            result_img = Transformation.sobelFilter(result_img, dx=0, dy=1, ksize=ksize)
        else:
            dpg.disable_item('transform_sobel_y')
    else:
        dpg.disable_item('transform_spatial_filtering')


    # Frequency Domain
    if dpg.get_value('transform_freq_checkbox'):
        dpg.enable_item('transform_freq_filtering')

        if dpg.get_value('transform_bandpass_checkbox'):
            dpg.enable_item('transform_freq_bandpass')

            split   = dpg.get_value('transform_split_slider')
            inner   = dpg.get_value('transform_inner_slider')
            outer   = dpg.get_value('transform_outer_slider')
            fadeIn  = dpg.get_value('transform_fadein_slider')
            fadeOut = dpg.get_value('transform_fadeout_slider')
            result_img = Transformation.bandPass(result_img, split, inner, outer, fadeIn, fadeOut)
        else:
            dpg.disable_item('transform_freq_bandpass')
    else:
        dpg.disable_item('transform_freq_filtering')
    


    # Binarization
    if dpg.get_value('transform_binarization_checkbox'):
        dpg.enable_item('transform_binarization')

        # Hide all options initially
        dpg.hide_item('transform_binarization_threshold_slider')
        dpg.hide_item('transform_binarization_adaptive')

        # Apply the selected binarization method
        method = dpg.get_value('transform_binarization_method')
        if method == "Otsu":
            result_img = Transformation.binarizationOtsu(result_img)
        elif method == "Threshold":
            dpg.show_item('transform_binarization_threshold_slider')
            threshold = dpg.get_value('transform_binarization_threshold_slider')
            result_img = Transformation.binarizationThreshold(result_img, threshold)
        elif method == "Adaptive":
            dpg.show_item('transform_binarization_adaptive')
            block_size = int(dpg.get_value('transform_binarization_adaptive_block_slider'))
            block_size = block_size if block_size % 2 == 1 else block_size + 1  # Ensure odd size
            c_value    = dpg.get_value('transform_binarization_adaptive_c_slider')
            result_img = Transformation.binarizationAdaptive(result_img, block_size, c_value)

    else:
        dpg.disable_item('transform_binarization')
    

    result_transform_img = result_img
    return result_img










def otsu_section_update(sender, app_data, user_data):
    pass
    # Ao entrar na secao Otsu, exibe o ultimo resultado ou a imagem original
    # if result_otsu_img is not None:
    #     change_image(result_otsu_img)
    # elif pure_img is not None:
    #     change_image(pure_img)

def run_vehicle_classification_otsu(sender, app_data, user_data):
    global pure_img, result_processing_img
    if pure_img is None:
        dpg.set_value("classification_log_text_otsu", "ERRO: Carregue uma imagem primeiro.")
        return

    # Etapa 1: Coletar todos os parametros da GUI em um unico dicionario.
    # As chaves do dicionario ('min_area_moto', etc.) devem ser exatamente
    # as que a funcao em processing.py espera
    params = {
        'min_area_moto':     dpg.get_value("proc_min_area_moto_drag"),
        'min_area_carro':    dpg.get_value("proc_min_area_carro_drag"),
        'min_area_caminhao': dpg.get_value("proc_min_area_caminhao_drag"),
        'max_area_geral':    dpg.get_value("proc_max_area_geral_drag"),
        'kernel_size':       dpg.get_value("proc_kernel_size_drag"),
        'open_iter':         dpg.get_value("proc_open_iter_drag"),
        'close_iter':        dpg.get_value("proc_close_iter_drag"),
    }

    dpg.set_value("classification_log_text_otsu", "Processando... por favor aguarde.")

    # A imagem BGR e a unica necessaria para essa funcao
    bgr_image = cv2.cvtColor(pure_img, cv2.COLOR_RGBA2BGR)

    # Etapa 2: Chamar a funcao de processamento passando o dicionario de 'params'
    processed_bgr, segmented_gray, logs = Processing.segmentAndClassifyVehiclesOtsu(
        bgr_image,
        params
    )

    # Armazena as duas imagens de resultado
    result_processing_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGBA)
    # Converte a imagem segmentada (cinza) para RGBA para poder ser exibida
    result_segmented_img = cv2.cvtColor(segmented_gray, cv2.COLOR_GRAY2RGBA)
    
    change_image(result_processing_img, result_segmented_img)
    dpg.set_value("classification_log_text_otsu", "\n".join(logs))



def canny_section_update(sender, app_data, user_data):
    pass
    # Ao entrar na secao Canny, exibe o ultimo resultado ou a imagem original
    # if result_canny_img is not None:
    #     change_image(result_canny_img)
    # elif pure_img is not None:
    #     change_image(pure_img)

def run_vehicle_classification_canny(sender, app_data, user_data):
    global pure_img, result_processing_img, result_segmented_img
    if pure_img is None:
        dpg.set_value("classification_log_text_canny", "ERRO: Carregue uma imagem primeiro.")
        return

    # Etapa 1: Coleta TODOS os parametros necessarios para a logica Canny.
    # Inclui os parametros do Canny, da dilatacao E da classificacao final.
    params = {
        # Parametros de classificacao (usados apos a segmentacao)
        'min_area_moto':     dpg.get_value("canny_min_area_moto_drag"),
        'min_area_carro':    dpg.get_value("canny_min_area_carro_drag"),
        'min_area_caminhao': dpg.get_value("canny_min_area_caminhao_drag"),
        'max_area_geral':    dpg.get_value("canny_max_area_geral_drag"),
        'min_aspect_ratio':  dpg.get_value("proc_min_ratio_drag"),
        'max_aspect_ratio':  dpg.get_value("proc_max_ratio_drag"),
        
        # Parametros do Canny (usados para encontrar as bordas)
        'canny_low_threshold':  dpg.get_value("canny_low"),
        'canny_high_threshold': dpg.get_value("canny_high"),

        # Parametros para fechar as bordas (Dilatacao e Fechamento)
        'canny_dilate_kernel': dpg.get_value("canny_dilate_kernel"),
        'canny_dilate_iter':   dpg.get_value("canny_dilate_iter"),
        'kernel_size':         dpg.get_value("proc_kernel_size_drag"),
    }

    dpg.set_value("classification_log_text_canny", "Processando com Canny... por favor aguarde.")

    bgr_image = cv2.cvtColor(pure_img, cv2.COLOR_RGBA2BGR)

    # Etapa 2: Chama a funcao de processamento
    processed_bgr, segmented_gray, logs = Processing.segmentAndClassifyByCanny(
        bgr_image,
        params
    )

    result_processing_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGBA)
    result_segmented_img = cv2.cvtColor(segmented_gray, cv2.COLOR_GRAY2RGBA)
    
    change_image(result_processing_img, result_segmented_img)
    dpg.set_value("classification_log_text_canny", "\n".join(logs))

def process_section_update(sender, app_data, user_data):
    global result_transform_img, result_processing_img

    result_img = computeProcessing()

    if dpg.get_value('processing_canny_checkbox'):
        dpg.enable_item('processing_canny')
        low        = dpg.get_value('processing_canny_low_slider')
        high       = dpg.get_value('processing_canny_high_slider')
        aperture   = dpg.get_value('processing_canny_aperture_slider')
        aperture   = aperture if aperture % 2 == 1 else aperture + 1  # Ensure odd size
        l2gradient = dpg.get_value('processing_canny_l2_checkbox')
        result_processing_img = Processing.cannyEdgeDetection(result_img, lowThreshold=low, highThreshold=high, apertureSize=aperture, L2gradient=l2gradient)
    else:
        dpg.disable_item('processing_canny')

    change_image(cv2.cvtColor(result_processing_img, cv2.COLOR_GRAY2RGBA))

def computeProcessing():
    global result_transform_img, result_processing_img

    result_img = computeTransformation().copy()

    result_processing_img = result_img
    return result_img


















# ---------------
# Setup DearPyGUI
# 
dpg.create_context()

# Hidden registry for textures
with dpg.texture_registry(tag='texreg', show=False):
    pass

# File dialog for PNG
with dpg.file_dialog(directory_selector=False, show=False, callback=load_image, tag='file_dialog'):
    dpg.add_file_extension(".png")



# Primary Window
with dpg.window(tag="Primary Window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Image Loading",        tag="secImgLoading",        callback=change_section, user_data="LoadingSection")
        dpg.add_button(label="Image Transformation", tag="secImgTransformation", callback=change_section, user_data="TransformationSection", show=False)
        dpg.add_button(label="Image Processing",     tag="secImgProcessing",     callback=change_section, user_data="ProcessingSection",     show=False)
        dpg.add_button(label="Vehicle Detection Otsu",     tag="secImgDetectionOtsu",     callback=change_section, user_data="OtsuClassificationSection",     show=False)
        dpg.add_button(label="Vehicle Detection Canny",     tag="secImgDetectionCanny",     callback=change_section, user_data="CannyClassificationSection",     show=False)

    
    with dpg.group(horizontal=True):
        with dpg.child_window(tag="LeftChild", width = leftPanel_width):
            # Loading Section
            with dpg.group(tag="LoadingSection"):
                dpg.add_button(label='Open Image',                                callback = lambda s, a, u: dpg.show_item('file_dialog'))
                dpg.add_checkbox(label='B&W View', tag='loading_bwview_checkbox', callback = loading_section_update, show = False)
            #####

            # Transformation Section
            with dpg.group(tag="TransformationSection", show=False):
                dpg.add_checkbox(label='Flip Horizontal',        tag='transform_flipH_checkbox',  callback=transform_section_update)
                dpg.add_checkbox(label='Flip Vertical',          tag='transform_flipV_checkbox',  callback=transform_section_update)
                dpg.add_checkbox(label='Histogram Equalization', tag='transform_histEq_checkbox', callback=transform_section_update)


                # Blurring
                dpg.add_checkbox(label='Mean Blur', tag='transform_mean_blur_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_mean_blur', enabled=False):
                    dpg.add_slider_int(label='Kernel Size', tag='transform_mean_blur_kernel_slider', default_value=5, min_value=1, max_value=99, callback=transform_section_update)
                #
                dpg.add_checkbox(label='Gaussian Blur', tag='transform_gaussian_blur_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_gaussian_blur', enabled=False):
                    dpg.add_slider_int  (label='Kernel Size', tag='transform_gaussian_blur_kernel_slider', default_value=5, min_value=1, max_value=99, callback=transform_section_update)
                    dpg.add_slider_float(label='Sigma X',     tag='transform_gaussian_blur_sigmaX_slider', default_value=0, min_value=0, max_value=10, callback=transform_section_update)
                    dpg.add_slider_float(label='Sigma Y',     tag='transform_gaussian_blur_sigmaY_slider', default_value=0, min_value=0, max_value=10, callback=transform_section_update)
                #
                dpg.add_checkbox(label='Median Blur', tag='transform_median_blur_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_median_blur', enabled=False):
                    dpg.add_slider_int(label='Kernel Size', tag='transform_median_blur_kernel_slider', default_value=5, min_value=1, max_value=99, callback=transform_section_update)

                # Spatial Filtering
                dpg.add_checkbox(label='Spatial Filtering', tag='transform_spatial_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_spatial_filtering', enabled=False):
                    # Laplacian Filter
                    dpg.add_checkbox(label='Laplacian Filter', tag='transform_laplacian_checkbox', callback=transform_section_update, default_value=False)
                    with dpg.group(tag='transform_laplacian', enabled=False):
                        dpg.add_slider_int(label='Kernel Size', tag='transform_laplacian_kernel_slider', default_value=3, min_value=1, max_value=31, callback=transform_section_update)
                    #
                    # Sobel Filters
                    dpg.add_checkbox(label='Sobel X Filter', tag='transform_sobel_x_checkbox', callback=transform_section_update, default_value=False)
                    with dpg.group(tag='transform_sobel_x', enabled=False):
                        dpg.add_slider_int(label='Kernel Size', tag='transform_sobel_x_kernel_slider', default_value=3, min_value=1, max_value=31, callback=transform_section_update)
                    dpg.add_checkbox(label='Sobel Y Filter', tag='transform_sobel_y_checkbox', callback=transform_section_update, default_value=False)
                    with dpg.group(tag='transform_sobel_y', enabled=False):
                        dpg.add_slider_int(label='Kernel Size', tag='transform_sobel_y_kernel_slider', default_value=3, min_value=1, max_value=31, callback=transform_section_update)


                # Frequency Filtering
                dpg.add_checkbox(label='Frequency Filtering', tag='transform_freq_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_freq_filtering', enabled=False):
                    dpg.add_checkbox(label='Frequency Domain View', tag='transform_freqview_checkbox', callback=transform_section_update, default_value=False)
                    dpg.add_checkbox(label='Band-pass Filter',      tag='transform_bandpass_checkbox', callback=transform_section_update, default_value=False)
                    with dpg.group(tag='transform_freq_bandpass', enabled=False):
                        dpg.add_checkbox(label='Band-pass Mask View', tag='transform_bandpass_maskview', callback=transform_section_update, default_value=False)
                        dpg.add_slider_float(label='Split',    tag='transform_split_slider',   default_value=0.5, min_value=0.0,  max_value=1.0, callback=transform_section_update)
                        dpg.add_slider_float(label='Inner',    tag='transform_inner_slider',   default_value=0.5, min_value=0.0,  max_value=1.0, callback=transform_section_update)
                        dpg.add_slider_float(label='Outer',    tag='transform_outer_slider',   default_value=0.5, min_value=0.0,  max_value=1.0, callback=transform_section_update)
                        dpg.add_slider_float(label='Fade In',  tag='transform_fadein_slider',  default_value=5,   min_value=0.01, max_value=10,  callback=transform_section_update)
                        dpg.add_slider_float(label='Fade Out', tag='transform_fadeout_slider', default_value=5,   min_value=0.01, max_value=10,  callback=transform_section_update)
                

                # Binarization
                dpg.add_checkbox(label='Binarization', tag='transform_binarization_checkbox', callback=transform_section_update, default_value=False)
                with dpg.group(tag='transform_binarization', enabled=False):
                    dpg.add_radio_button(
                        tag='transform_binarization_method',
                        items=["Otsu", "Threshold", "Adaptive"],
                        default_value="Otsu",
                        horizontal=False,
                        callback=transform_section_update
                    )
                    dpg.add_slider_int(label='Threshold Value', tag='transform_binarization_threshold_slider', default_value=127, min_value=0, max_value=255, callback=transform_section_update, show=False)
                    #
                    with dpg.group(tag='transform_binarization_adaptive', show=False):
                        dpg.add_slider_int(label='Adaptive Block Size', tag='transform_binarization_adaptive_block_slider', default_value=11, min_value=3, max_value=99, callback=transform_section_update)
                        dpg.add_slider_float(label='Adaptive C Value', tag='transform_binarization_adaptive_c_slider', default_value=2, min_value=0, max_value=10, callback=transform_section_update)
            #####
                
                        
            

            def on_canny_slider_update(sender, app_data, user_data):
                if sender == 'processing_canny_low_slider':
                    low = dpg.get_value('processing_canny_low_slider')
                    dpg.set_value('processing_canny_high_slider', max(dpg.get_value('processing_canny_high_slider'), low + 1))
                elif sender == 'processing_canny_high_slider':
                    high = dpg.get_value('processing_canny_high_slider')
                    dpg.set_value('processing_canny_low_slider', min(dpg.get_value('processing_canny_low_slider'), high - 1))
                process_section_update(sender, app_data, user_data)
                
            # Processing Section
            with dpg.group(tag="ProcessingSection", show=False):
                dpg.add_checkbox(label='Canny Edge Detection', tag='processing_canny_checkbox', default_value=True, enabled=False)
                with dpg.group(tag='processing_canny', enabled=False):
                    dpg.add_slider_int(label='Low Threshold',  tag='processing_canny_low_slider',      default_value=100,   min_value=0, max_value=254, callback=on_canny_slider_update)
                    dpg.add_slider_int(label='High Threshold', tag='processing_canny_high_slider',     default_value=200,   min_value=1, max_value=255, callback=on_canny_slider_update)
                    dpg.add_slider_int(label='Aperture Size',  tag='processing_canny_aperture_slider', default_value=3,     min_value=3, max_value=7,   callback=process_section_update)
                    dpg.add_checkbox  (label='L2 Gradient',    tag='processing_canny_l2_checkbox',     default_value=False, callback=process_section_update)

                dpg.add_button(label='Segment Image', tag='processing_segment_button', enabled=False)
            
            with dpg.group(tag="OtsuClassificationSection", show=False):
                dpg.add_text("Vehicle Classification Parameters")
                dpg.add_separator()

                dpg.add_text("Area Thresholds (pixels)")
                dpg.add_drag_int(label="Min Area Moto",   tag="proc_min_area_moto_drag",   default_value=500,   min_value=1, max_value=100000, speed=10)
                dpg.add_drag_int(label="Min Area Carro",  tag="proc_min_area_carro_drag",  default_value=2500,  min_value=1, max_value=100000, speed=100)
                dpg.add_drag_int(label="Min Area Caminhao", tag="proc_min_area_caminhao_drag", default_value=8000,  min_value=1, max_value=100000, speed=100)
                dpg.add_drag_int(label="Max Area Geral",  tag="proc_max_area_geral_drag",  default_value=20000, min_value=1, max_value=100000, speed=100)
                
                dpg.add_separator()
                dpg.add_text("Morphology Parameters")
                dpg.add_drag_int(label="Kernel Size", tag="proc_kernel_size_drag", default_value=5, min_value=1, max_value=21, speed=2)
                dpg.add_drag_int(label="Opening Iterations", tag="proc_open_iter_drag", default_value=1, min_value=1, max_value=10)
                dpg.add_drag_int(label="Closing Iterations", tag="proc_close_iter_drag", default_value=1, min_value=1, max_value=10)
                dpg.add_separator()

                dpg.add_button(label='Run Vehicle Classification', callback=run_vehicle_classification_otsu, width=-1, height=40)

                dpg.add_separator()
                dpg.add_text("Results and Logs:")
                with dpg.child_window(tag="classification_log_window_otsu", height=-1, horizontal_scrollbar=True):
                    dpg.add_text("Aguardando analise...", tag="classification_log_text_otsu")
                    
            with dpg.group(tag="CannyClassificationSection", show=False):
                dpg.add_text("Vehicle Classification Parameters")
                dpg.add_separator()

                dpg.add_text("Area Thresholds (pixels)")
                dpg.add_drag_int(label="Min Area Moto",   tag="canny_min_area_moto_drag",   default_value=500,   min_value=1, max_value=100000, speed=10)
                dpg.add_drag_int(label="Min Area Carro",  tag="canny_min_area_carro_drag",  default_value=2500,  min_value=1, max_value=100000, speed=100)
                dpg.add_drag_int(label="Min Area Caminhao", tag="canny_min_area_caminhao_drag", default_value=8000,  min_value=1, max_value=100000, speed=100)
                dpg.add_drag_int(label="Max Area Geral",  tag="canny_max_area_geral_drag",  default_value=20000, min_value=1, max_value=100000, speed=100)

                dpg.add_separator()

                dpg.add_text("Canny Edge Detector Parameters")
                dpg.add_slider_int(label="Canny Low Threshold", tag="canny_low", max_value=255)
                dpg.add_slider_int(label="Canny High Threshold", tag="canny_high", default_value=150, max_value=255)

                dpg.add_separator()
                dpg.add_text("Edge Closing Parameters")
                dpg.add_drag_int(label="Dilation Kernel Size", tag="canny_dilate_kernel", default_value=5)
                dpg.add_drag_int(label="Dilation Iterations", tag="canny_dilate_iter", default_value=2)

                dpg.add_button(label='Run Vehicle Classification', callback=run_vehicle_classification_canny, width=-1, height=40)

                dpg.add_separator()
                dpg.add_text("Results and Logs:")
                with dpg.child_window(tag="classification_log_window_canny", height=-1, horizontal_scrollbar=True):
                    dpg.add_text("Aguardando analise...", tag="classification_log_text_canny")
            #####

        with dpg.child_window(tag="RightChild"):
            dpg.add_drawlist(tag='ImageCanvas', width=100, height=100)
    







dpg.create_viewport(title='Custom Title', width=vp_width, height=vp_height)
dpg.set_viewport_resize_callback(on_viewport_resize)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()