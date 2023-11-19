import gradio as gr
import swap_mukham
from collections import OrderedDict
from swap_mukham import global_variables as gv


def update_source_data(face_id, *source_components):
    cmpts = swap_mukham.gui.components
    source_components = list(source_components)

    source_image_files = source_components[1]
    if source_image_files and isinstance(source_image_files, (list, tuple)):
        source_components[1] = [img.name for img in source_image_files if hasattr(img, 'name')]
    else:
        source_components[1] = None

    specific_image_files = source_components[3]
    if specific_image_files and isinstance(specific_image_files, (list, tuple)):
        source_components[3] = [img.name for img in specific_image_files if hasattr(img, 'name')]
    else:
        source_components[3] = None

    cmpts.source_data[face_id] = OrderedDict(zip(gv.SOURCE_DEFAULTS.keys(), source_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_face_control_data(control_id, *face_control_components):
    cmpts = swap_mukham.gui.components
    cmpts.face_control_data[control_id] = OrderedDict(zip(list(gv.FACE_CONTROL_DEFAULTS.keys()), face_control_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_frame_control_data(*frame_control_components):
    cmpts = swap_mukham.gui.components
    cmpts.frame_control_data = OrderedDict(zip(gv.FRAME_CONTROL_DEFAULTS.keys(), frame_control_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_target_selection_data(*target_selection_components):
    cmpts = swap_mukham.gui.components
    cmpts.target_selection_data = OrderedDict(zip(gv.TARGET_SELECTION_DEFAULTS.keys(), target_selection_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_target_data(*target_components):
    cmpts = swap_mukham.gui.components
    cmpts.target_data = OrderedDict(zip(gv.TARGET_DEFAULTS.keys(), target_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_settings_data(*settings_components):
    cmpts = swap_mukham.gui.components
    cmpts.settings_data = OrderedDict(zip(gv.SETTINGS_DEFAULTS.keys(), settings_components))
    swap_mukham.gui.save.write_json(cmpts)


def update_output_data(*output_components):
    cmpts = swap_mukham.gui.components
    cmpts.output_data = OrderedDict(zip(gv.OUTPUT_DEFAULTS.keys(), output_components))
    swap_mukham.gui.save.write_json(cmpts)


def set_on_component_update(component, event):
    cmpts = swap_mukham.gui.components
    method = getattr(component, event, None)

    return method(
        update_source_data,
        [cmpts.face_id, *cmpts.source_components]
    ).then(
        update_face_control_data,
        [cmpts.control_id, *cmpts.face_control_components]
    ).then(
        update_frame_control_data,
        cmpts.frame_control_components
    ).then(
        update_target_data,
        cmpts.target_components
    ).then(
        update_target_selection_data,
        cmpts.target_selection_components,
    ).then(
        update_settings_data,
        cmpts.settings_components
    ).then(
        update_output_data,
        cmpts.output_components
    )
