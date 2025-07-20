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

def change_image(imgRGBA):
    if imgRGBA is None:
        return
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
    dpg.configure_item('RightChild', width=right_w, height=height - 23)
    dpg.configure_item('ImageCanvas', width=right_w, height=height - 17 - 25)

    update_image_display()

def change_section(sender, app_data, user_data):
    # print("Section Changed", sender, user_data)
    # Hide all sections
    dpg.hide_item('LoadingSection')
    dpg.hide_item('TransformationSection')
    dpg.hide_item('ProcessingSection')
    

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
    

    result_transform_img = result_img
    return result_img




def process_section_update(sender, app_data, user_data):
    computeProcessing()
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
            #####
                
                        
            


            # Processing Section
            with dpg.group(tag="ProcessingSection", show=False):
                dpg.add_button(label='Segment Image', tag='processing_segment_button')
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