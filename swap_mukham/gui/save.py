import os
import json
from collections import OrderedDict
from swap_mukham import global_variables as gv

# TODO

def write_json(cmpts, path="last_save.json"):
    if not gv.WRITE_JSON:
        return
    save_data = {
        'source_data':cmpts.source_data,
        'target_data':cmpts.target_data,
        'face_control_data':cmpts.face_control_data,
        'target_selection_data':cmpts.target_selection_data,
        'settings_data':cmpts.settings_data,
        'output_data':cmpts.output_data,
    }
    with open("last_save.json", 'w') as json_file:
        json.dump(save_data, json_file, indent=4)

# TODO

def load_json(path="last_save.json"):
    if not os.path.exists(path):
        print(f"Failed loading {path}")
        return

    with open(path, 'r') as json_file:
        last_saved_data = json.load(json_file)
        last_saved_data = OrderedDict(last_saved_data)

        face_control_data = last_saved_data.get('face_control_data', None)
        source_data = last_saved_data.get('source_data', None)
        target_data = last_saved_data.get('target_data', None)
        settings_data = last_saved_data.get('settings_data', None)
        target_selection_data = last_saved_data.get('target_selection_data', None)
        output_data = last_saved_data.get('output_data', None)

        if settings_data:
            # cmpts.settings_data = settings_data
            gv.SETTINGS_DEFAULTS = settings_data
        if target_data:
            # cmpts.target_data = target_data
            gv.TARGET_DEFAULTS = target_data
        if target_selection_data:
            # cmpts.target_selection_data = target_selection_data
            gv.TARGET_SELECTION_DEFAULTS = target_selection_data
        if face_control_data:
            # cmpts.face_control_data = face_control_data
            gv.CONTROL_DEFAULTS = face_control_data['1']
        if source_data:
            # cmpts.source_data = source_data
            gv.SOURCE_DEFAULTS = source_data['1']
        if output_data:
            # cmpts.output_data = output_data
            gv.OUTPUT_DEFAULTS = output_data

        print(f"Loaded {path}")
