import dearpygui.dearpygui as dpg
import cv2
import numpy as np

from transform  import Transformation
from processing import Processing

vp_width  = 600
vp_height = 400
leftPanel_width = int(vp_width * 0.3)

# Globals for image data and dimensions
pure_img      = None
current_img   = None
current_img_w = 0
current_img_h = 0


# Load image and create static texture
def load_image(sender, app_data, user_data):
    global current_img, current_img_w, current_img_h, pure_img
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

    change_image(img_rgba)

def change_image(imgRGBA):
    global current_img, current_img_w, current_img_h
    current_img = imgRGBA
    current_img_h, current_img_w = imgRGBA.shape[:2]

    data = imgRGBA.flatten() / 255.0
    if not dpg.does_item_exist('image_tex'):
        dpg.add_dynamic_texture(
            current_img_w, current_img_h, data,
            tag='image_tex', parent='texreg'
        )
    else:
        dpg.set_value('image_tex', data)

    update_image_display()


# Redraw image to fit current drawlist size, centered with white background
def update_image_display():
    global current_img_w, current_img_h
    if not dpg.does_item_exist('image_tex'):
        return

    can_w = dpg.get_item_width('RightChild')
    can_h = dpg.get_item_height('RightChild')
    # print((can_w, can_h))
    scale = min(can_w / current_img_w, can_h / current_img_h)
    new_w = int(current_img_w * scale)
    new_h = int(current_img_h * scale)

    dpg.delete_item('ImageCanvas', children_only=True)
    # dpg.draw_rectangle([0, 0], [can_w, can_h], color=[0, 0, 0, 0], fill=[255, 255, 255, 255], parent='ImageCanvas')
    x = (can_w - new_w) // 2
    y = (can_h - new_h) // 2
    dpg.draw_image('image_tex', pmin=[x, y], pmax=[x + new_w, y + new_h], parent='ImageCanvas')

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
    dpg.configure_item('RightChild', width=right_w, height=height)
    dpg.configure_item('ImageCanvas', width=right_w, height=height - 17)

    update_image_display()








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
        with dpg.child_window(tag="LeftChild", width = leftPanel_width):
            # dpg.add_text("Hello, world")
            # dpg.add_button(label="Save")
            # dpg.add_input_text(label="string", default_value="Quick brown fox")
            # dpg.add_slider_float(label="float", default_value=0.273, max_value=1)
            dpg.add_button(label='Open Image', callback=lambda s, a, u: dpg.show_item('file_dialog'))
            dpg.add_button(label='B&W View',   callback=lambda s, a, u: change_image(Transformation.BWConversion(pure_img)))
        
        with dpg.child_window(tag="RightChild"):
            dpg.add_drawlist(tag='ImageCanvas', width=100, height=100)








dpg.create_viewport(title='Custom Title', width=vp_width, height=vp_height)
dpg.set_viewport_resize_callback(on_viewport_resize)
dpg.setup_dearpygui()
dpg.show_viewport()
dpg.set_primary_window("Primary Window", True)
dpg.start_dearpygui()
dpg.destroy_context()