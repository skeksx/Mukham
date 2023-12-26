## Installation


create conda or venv python environment with `Swap-Mukham/requirements_xxxx.txt`.

currently cpu and cuda are supported.
more device support can be added through `Swap-Mukham/swap_mukham/utils/devie.py`.

## Download models

- see [`Swap-Mukham/doc/download_models.md`](https://github.com/harisreedhar/Swap-Mukham/blob/main/doc/download_models.md)

## Running on cpu

`python app.py --local --device cpu`

## Running on cuda

`python app.py --local --device cuda`

## Note

- remove `--local` cli argument to use default image & video components.
- if `cuda` is not detected install pytorch 
