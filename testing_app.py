import dearpygui.dearpygui as dpg
import cv2
import numpy as np

# Certifique-se que os arquivos transform.py e processing.py estao na mesma pasta
from transform  import Transformation
from processing import Processing

vp_width  = 800  # Aumentei a largura padrao para melhor visualizacao
vp_height = 600
leftPanel_width = int(vp_width * 0.35) # Aumentei um pouco para caber os novos controles

# Globals for image data and dimensions
pure_img      = None
current_img   = None
current_img_w = 0
current_img_h = 0

result_loading_img    = None
result_transform_img  = None
result_processing_img = None


# Load image and create static texture
def load_image(sender, app_data, user_data):
    global current_img, current_img_w, current_img_h, pure_img, result_loading_img
    file_path = app_data['file_path_name']
    img = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        print(f"Failed to load: {file_path}")
        return

    # Garante que a imagem tenha 3 ou 4 canais para conversao
    if len(img.shape) < 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

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
    dpg.set_value("classification_log_text", "Imagem carregada. Selecione os parametros e execute a classificacao na aba 'Image Processing'.")

def change_image(imgRGBA):
    if imgRGBA is None:
        return
    global current_img, current_img_w, current_img_h
    current_img = imgRGBA
    current_img_h, current_img_w, _ = imgRGBA.shape

    data = np.asfarray(imgRGBA.ravel(), dtype='f') / 255.0

    # Lida com a atualizacao da textura de forma segura, mesmo com tamanhos diferentes
    if not dpg.does_item_exist('image_tex'):
        dpg.add_dynamic_texture(width=current_img_w, height=current_img_h, default_value=data, tag='image_tex', parent='texreg')
    else:
         # Reconfigura a textura com os novos dados e dimensoes
        dpg.configure_item('image_tex', width=current_img_w, height=current_img_h, default_value=data)

    update_image_display()


# Redraw image to fit current drawlist size, centered with white background
def update_image_display():
    if not dpg.does_item_exist('image_tex') or current_img_w == 0:
        return

    can_w = dpg.get_item_width('RightChild')
    can_h = dpg.get_item_height('RightChild')
    if not can_w or not can_h or can_w <=0 or can_h <=0: return

    scale = min(can_w / current_img_w, can_h / current_img_h)
    new_w = int(current_img_w * scale)
    new_h = int(current_img_h * scale)

    dpg.delete_item('ImageCanvas', children_only=True)
    x = (can_w - new_w) // 2
    y = (can_h - new_h) // 2
    dpg.draw_image('image_tex', pmin=[x, y], pmax=[x + new_w, y + new_h], parent='ImageCanvas')

# Callback: when viewport resizes, adjust child sizes and redraw
def on_viewport_resize(sender, app_data):
    vp_w = dpg.get_viewport_width()
    vp_h = dpg.get_viewport_height()
    left_w = int(vp_w * 0.35)
    right_w = vp_w - left_w - 40
    height = vp_h - 55

    dpg.configure_item('LeftChild', width=left_w)
    dpg.configure_item('RightChild', width=right_w, height=height)
    dpg.configure_item('ImageCanvas', width=right_w, height=height)
    update_image_display()

def change_section(sender, app_data, user_data):
    dpg.hide_item('LoadingSection')
    dpg.hide_item('TransformationSection')
    dpg.hide_item('ProcessingSection')

    # Logica simplificada para exibir a imagem da secao anterior ao trocar
    if user_data == "LoadingSection":
        if result_loading_img is not None: change_image(result_loading_img)
    elif user_data == "TransformationSection":
        if result_transform_img is not None: change_image(cv2.cvtColor(result_transform_img, cv2.COLOR_GRAY2RGBA))
        elif result_loading_img is not None: change_image(result_loading_img)
    elif user_data == "ProcessingSection":
        if result_processing_img is not None: change_image(result_processing_img)
        elif result_transform_img is not None: change_image(cv2.cvtColor(result_transform_img, cv2.COLOR_GRAY2RGBA))
        elif result_loading_img is not None: change_image(result_loading_img)

    dpg.show_item(user_data)

def loading_section_update(sender, app_data, user_data):
    res = computeLoading()
    change_image(res)

def computeLoading():
    global result_loading_img
    if result_loading_img is None: return None

    if dpg.get_value('loading_bwview_checkbox'):
        return Transformation.BWConversion(result_loading_img)
    else:
        return result_loading_img.copy()

def transform_section_update(sender, app_data, user_data):
    global result_transform_img
    result_transform_img = computeTransformation()
    if result_transform_img is not None:
        change_image(cv2.cvtColor(result_transform_img, cv2.COLOR_GRAY2RGBA))

def computeTransformation():
    global result_loading_img
    if result_loading_img is None: return None
    result_img = Transformation.RGBAtoGray(result_loading_img.copy())
    # ... (a logica de transformacao original do usuario iria aqui)
    return result_img

# NOVA FUNCAO PARA EXECUTAR A CLASSIFICACAO
def run_vehicle_classification(sender, app_data, user_data):
    global pure_img, result_processing_img
    if pure_img is None:
        dpg.set_value("classification_log_text", "ERRO: Carregue uma imagem primeiro.")
        return

    # Etapa 1: Coletar todos os parametros da GUI em um unico dicionario.
    # As chaves do dicionario ('min_area_moto', etc.) devem ser exatamente
    # as que a sua funcao em processing.py espera.
    params = {
        'min_area_moto':     dpg.get_value("proc_min_area_moto_drag"),
        'min_area_carro':    dpg.get_value("proc_min_area_carro_drag"),
        'min_area_caminhao': dpg.get_value("proc_min_area_caminhao_drag"),
        'max_area_geral':    dpg.get_value("proc_max_area_geral_drag"),
        'kernel_size':       dpg.get_value("proc_kernel_size_drag"),
        'open_iter':         dpg.get_value("proc_open_iter_drag"),
        'close_iter':        dpg.get_value("proc_close_iter_drag"),
    }

    dpg.set_value("classification_log_text", "Processando... por favor aguarde.")

    # A imagem BGR e a unica necessaria para a nova funcao
    bgr_image = cv2.cvtColor(pure_img, cv2.COLOR_RGBA2BGR)

    # Etapa 2: Chamar a funcao de processamento com a nova assinatura (passando o dicionario 'params')
    processed_bgr, logs = Processing.segmentAndClassifyVehicles(
        bgr_image,
        params
    )

    # O resto da logica permanece o mesmo, pois o formato de retorno foi mantido
    result_processing_img = cv2.cvtColor(processed_bgr, cv2.COLOR_BGR2RGBA)
    change_image(result_processing_img)
    dpg.set_value("classification_log_text", "\n".join(logs))
    

def process_section_update(sender, app_data, user_data):
    pass


# ---------------
# Setup DearPyGUI
#
dpg.create_context()

with dpg.texture_registry(tag='texreg', show=False):
    pass

# Adicionado suporte a JPG/JPEG
with dpg.file_dialog(directory_selector=False, show=False, callback=load_image, tag='file_dialog'):
    dpg.add_file_extension(".png")
    dpg.add_file_extension(".jpg")
    dpg.add_file_extension(".jpeg")

# Primary Window
with dpg.window(tag="Primary Window"):
    with dpg.group(horizontal=True):
        dpg.add_button(label="Image Loading",        tag="secImgLoading",        callback=change_section, user_data="LoadingSection")
        dpg.add_button(label="Image Transformation", tag="secImgTransformation", callback=change_section, user_data="TransformationSection", show=False)
        dpg.add_button(label="Image Processing",     tag="secImgProcessing",     callback=change_section, user_data="ProcessingSection",     show=False)

    with dpg.group(horizontal=True):
        with dpg.child_window(tag="LeftChild", width=leftPanel_width):
            # Loading Section
            with dpg.group(tag="LoadingSection"):
                dpg.add_button(label='Open Image', callback = lambda: dpg.show_item('file_dialog'))
                dpg.add_checkbox(label='B&W View', tag='loading_bwview_checkbox', callback = loading_section_update, show = False)

            # Transformation Section
            with dpg.group(tag="TransformationSection", show=False):
                dpg.add_text("Controles de transformacao iriam aqui.")
                # Exemplo de controle da versao original do usuario
                dpg.add_checkbox(label='Flip Horizontal', tag='transform_flipH_checkbox', callback=transform_section_update)


            # ############# SECAO DE PROCESSAMENTO COM AS NOVAS FUNCOES #############
            with dpg.group(tag="ProcessingSection", show=False):
                dpg.add_text("Vehicle Classification Parameters")
                dpg.add_separator()

                dpg.add_text("Area Thresholds (pixels)")
                dpg.add_drag_int(label="Min Area Moto",   tag="proc_min_area_moto_drag",   default_value=500,   min_value=1, max_value=100000, speed=10)
                dpg.add_drag_int(label="Min Area Carro",  tag="proc_min_area_carro_drag",  default_value=2500,  min_value=1, max_value=100000, speed=100)
                dpg.add_drag_int(label="Min Area Caminhao", tag="proc_min_area_caminhao_drag", default_value=8000,  min_value=1, max_value=100000, speed=100)
                # 'help' removido da linha abaixo
                dpg.add_drag_int(label="Max Area Geral",  tag="proc_max_area_geral_drag",  default_value=20000, min_value=1, max_value=100000, speed=100)
                
                dpg.add_separator()
                dpg.add_text("Morphology Parameters")
                # 'help' removido das 3 linhas abaixo
                dpg.add_drag_int(label="Kernel Size", tag="proc_kernel_size_drag", default_value=5, min_value=1, max_value=21, speed=2)
                dpg.add_drag_int(label="Opening Iterations", tag="proc_open_iter_drag", default_value=1, min_value=1, max_value=10)
                dpg.add_drag_int(label="Closing Iterations", tag="proc_close_iter_drag", default_value=1, min_value=1, max_value=10)
                dpg.add_separator()

                dpg.add_button(label='Run Vehicle Classification', callback=run_vehicle_classification, width=-1, height=40)

                dpg.add_separator()
                dpg.add_text("Results and Logs:")
                with dpg.child_window(tag="classification_log_window", height=-1, horizontal_scrollbar=True):
                    dpg.add_text("Aguardando analise...", tag="classification_log_text")
                    
        with dpg.child_window(tag="RightChild", border=False):
            dpg.add_drawlist(tag='ImageCanvas', width=-1, height=-1)

dpg.create_viewport(title='Vehicle Detection', width=vp_width, height=vp_height, resizable=True)
dpg.set_viewport_resize_callback(on_viewport_resize)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)

# Dispara o on_viewport_resize uma vez no inicio para ajustar o layout
on_viewport_resize(None, None)

dpg.start_dearpygui()
dpg.destroy_context()