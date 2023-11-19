import os
from collections import namedtuple
import swap_mukham.default_paths as dp
from swap_mukham.models.swapper.dfm import DFM
from swap_mukham.models.swapper.ghost import Ghost
from swap_mukham.models.swapper.blendswap import BlendSwap
from swap_mukham.models.swapper.inswapper import Inswapper
from swap_mukham.models.swapper.simswap import SimSwap, SimSwapUnofficial
from swap_mukham.models.arcface import ArcFace


def load_face_swapper(name, **kwargs):
    for category_name, swapper in dp.EMBEDDING_BASED_SWAPPERS.items():
        if name in swapper.keys():
            if category_name == "inswapper":
                swapper_model = Inswapper(model_file=swapper[name], **kwargs)
                backbone_model = ArcFace(model_file=swapper["backbone"], **kwargs)
                return swapper_model, backbone_model
            elif category_name == "simswap":
                swapper_model = SimSwap(model_file=swapper[name], **kwargs)
                backbone_model = ArcFace(model_file=swapper["backbone"], **kwargs)
                return swapper_model, backbone_model
            elif category_name == "simswap unofficial":
                swapper_model = SimSwapUnofficial(model_file=swapper[name], **kwargs)
                backbone_model = ArcFace(model_file=swapper["backbone"], **kwargs)
                return swapper_model, backbone_model
            elif category_name == "ghost":
                swapper_model = Ghost(model_file=swapper[name], **kwargs)
                backbone_model = ArcFace(model_file=swapper["backbone"], **kwargs)
                return swapper_model, backbone_model
            else:
                raise ValueError(f"Unknown Swapping Model {name}")
