import gradio as gr
import cv2
import os
import numpy as np
from swap_mukham.utils import io as io_util
from swap_mukham.utils import image as image_util
from swap_mukham.utils import video as video_util

class PathList:
    def __init__(self, paths, limit=20):
        self.paths = paths
        self.limit = limit
        self.current_index = 0
        self.current_list = []

    def next(self):
        if self.current_index < len(self.paths):
            start = self.current_index
            end = min(self.current_index + self.limit, len(self.paths))
            paths_to_return = self.paths[start:end]
            self.current_index = end
            self.current_list = paths_to_return
            return paths_to_return
        else:
            return self.current_list

    def prev(self):
        if self.current_index > 0:
            self.current_index = max(0, self.current_index - self.limit * 2)
            return self.next()
        else:
            return self.current_list


def create_multi_image_input(return_gallery_component=False, default_values=None, label="Image", show_label=True):
    with gr.Column():
        kwargs = {}
        if default_values is not None:
            kwargs['value'] = default_values
        images = gr.File(file_count="multiple", file_types=['image',], container=True, label=label, show_label=show_label, elem_id="custom_filelist", **kwargs)
        gallery = gr.Image(label=label, interactive=False, show_label=False, visible=False, container=True)

    def on_images_update(images):
        if not images:
            return gr.File(visible=True, height="auto"), gr.Image(visible=False)
        try:
            images = [cv2.imread(f.name) for f in images[:20]]
            image_grid = image_util.create_image_grid(images, size=256)
            return gr.File(visible=True), gr.Image(visible=True, value=image_grid)
        except (ValueError, AttributeError):
            gr.Warning('Error! Bad image file')
            return gr.File(visible=True, value=None, height="auto"), gr.Image(visible=False, value=None)

    images.change(
        on_images_update,
        [images,],
        [images, gallery],
    )

    if return_gallery_component:
        return images, gallery

    return images


def create_text_and_browse_button(text_name, button_name, mode="directory", show_gallery="NONE", hide=False, **kwargs):
    IMAGE_PATHS = None
    FILE_PATHS = None
    with gr.Column(visible = not hide):
        with gr.Group():
            text_component = gr.Text(label=text_name, **kwargs)
            browse_button = gr.Button(button_name)
        if show_gallery != "NONE":
            with gr.Accordion("Files in same directory", open=False, visible=False) as gallery_accordion:
                gallery = gr.Gallery(
                    show_label=False,
                    columns=[4],
                    rows=[4],
                    height="auto",
                    allow_preview=False,
                )
                with gr.Row():
                    prev_button = gr.Button("Previous", size="md")
                    next_button = gr.Button("Next", size="md")

    def on_browse_button_click(old_path):
        if mode == "directory":
            path = io_util.browse_directory()
        elif mode == "file":
            path = io_util.browse_file()
        if path:
            update_paths(path)
            return path
        return old_path

    def update_paths(directory):
        nonlocal FILE_PATHS
        if not os.path.isdir(directory):
            directory = os.path.dirname(directory)
        if show_gallery == "VIDEO":
            FILE_PATHS = PathList(io_util.get_videos_from_directory(directory))
        elif show_gallery == "IMAGE":
            FILE_PATHS = PathList(io_util.get_images_from_directory(directory))

    def update_gallery(prev):
        nonlocal FILE_PATHS
        if FILE_PATHS is None:
            return gr.Accordion(visible=False), None
        if prev:
            files = FILE_PATHS.prev()
        else:
            files = FILE_PATHS.next()

        thumbnails = None
        thumb_size = (192,192)
        blank = np.zeros((*thumb_size, 3), dtype=np.uint8)

        if show_gallery == "VIDEO":
            thumbnails = []
            for f in files:
                try:
                    thumbnail = video_util.create_video_thumbnail(f,size=thumb_size)[:,:,::-1]
                    thumbnails.append(thumbnail)
                except Exception as e:
                    print(e)
                    thumbnails.append(blank)
        elif show_gallery == "IMAGE":
            thumbnails = []
            for f in files:
                try:
                    img = cv2.imread(f)
                    thumbnail = image_util.center_fit_image(img, thumb_size)[:,:,::-1]
                    thumbnails.append(thumbnail)
                except Exception as e:
                    print(e)
                    thumbnails.append(blank)

        if thumbnails:
            return gr.Accordion(visible=True), thumbnails
        else:
            return gr.Accordion(visible=False), None

    def on_gallery_select(e: gr.SelectData):
        nonlocal FILE_PATHS
        try:
            return FILE_PATHS.current_list[e.index]
        except:
            gr.Warning("Data Missing! Browse again!")
            return gr.Text()

    browse_button_chain = browse_button.click(
        on_browse_button_click,
        [text_component],
        [text_component],
    )

    if show_gallery != "NONE":
        browse_button_chain.then(
            update_gallery,
            [gr.State(False)],
            [gallery_accordion, gallery]
        )

        text_component.submit(
            update_paths,
            [text_component],
        ).then(
            update_gallery,
            [gr.State(False)],
            [gallery_accordion, gallery]
        )

        next_button.click(
            update_gallery,
            [gr.State(False)],
            [gallery_accordion, gallery]
        )

        prev_button.click(
            update_gallery,
            [gr.State(True)],
            [gallery_accordion, gallery]
        )

        gallery.select(
            fn=on_gallery_select,
            outputs = [text_component]
        )

    return text_component