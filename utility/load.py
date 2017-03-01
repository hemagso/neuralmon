import pandas as pd
import csv
import os
import matplotlib.image as mpimg
from matplotlib import gridspec
from skimage import color
import numpy as np
from skimage.util import dtype

def _prepare_rgba_array(arr):
    """Check the shape of the array to be RGBA and convert it to
    floating point representation.
    """
    arr = np.asanyarray(arr)

    if arr.ndim not in [3, 4] or arr.shape[-1] != 4:
        msg = ("the input array must have a shape == (.., ..,[ ..,] 4)), "
               "got {0}".format(arr.shape))
        raise ValueError(msg)

    return dtype.img_as_float(arr)


def rgba2rgb(rgba, background=(1, 1, 1)):
    """RGBA to RGB conversion.
    Parameters
    ----------
    rgba : array_like
        The image in RGBA format, in a 3-D array of shape ``(.., .., 4)``.
    background : array_like
        The color of the background to blend the image with. A tuple
        containing 3 floats between 0 to 1 - the RGB value of the background.
    Returns
    -------
    out : ndarray
        The image in RGB format, in a 3-D array of shape ``(.., .., 3)``.
    Raises
    ------
    ValueError
        If `rgba` is not a 3-D array of shape ``(.., .., 4)``.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Alpha_compositing#Alpha_blending
    Examples
    --------
    >>> from skimage import color
    >>> from skimage import data
    >>> img_rgba = data.logo()
    >>> img_rgb = color.rgba2rgb(img_rgba)
    """
    arr = _prepare_rgba_array(rgba)
    if isinstance(background, tuple) and len(background) != 3:
        raise ValueError('the background must be a tuple with 3 items - the '
                         'RGB color of the background. Got {0} items.'
                         .format(len(background)))

    alpha = arr[..., -1]
    channels = arr[..., :-1]
    out = np.empty_like(channels)

    for ichan in range(channels.shape[-1]):
        out[..., ichan] = np.clip(
            (1 - alpha) * background[ichan] + alpha * channels[..., ichan],
            a_min=0, a_max=1)
    return out

def load_type_dict():
    data_path = "./data"
    types_file = "types.csv"
    types_path = os.path.join(data_path,types_file)
    type_dict = {}
    with open(types_path,"r") as f:
        reader = csv.reader(f)
        next(reader,None) #Skip the header
        for row in reader:
            type_dict[int(row[0])] = {
                "label": row[1],
                "color" : row[4]
            }
    return type_dict

def load_pokemon_dict():
    """Loads the Pokemon Dictionary into memory. This dictionary contains
    information of Pokemon Typing, stored as:
    
    ID: {type_01: value, type_02: value}
    
    """
    data_path = "./data"
    pokemon_file = "pokemon_types.csv"
    pokemon_path = os.path.join(data_path,pokemon_file)
    pokemon_dict = {}
    with open(pokemon_path,"r") as f:
        reader = csv.reader(f)
        next(reader,None) #Skip the header
        for row in reader:
            pkm_id = int(row[0])
            type_slot = int(row[2])
            pkm_type = int(row[1])
            if pkm_id not in pokemon_dict:
                pokemon_dict[pkm_id] = {"type_01": None, "type_02": None}
            if type_slot == 1:
                pokemon_dict[pkm_id]["type_01"] = pkm_type
            elif type_slot == 2:
                pokemon_dict[pkm_id]["type_02"] = pkm_type
            else:
                raise ValueError("Unexpected type slot value")
    return pokemon_dict

def load_dataframe(gen_folders):
    pkm_dict = load_pokemon_dict()
    df_dict = {"id" : [], "gen": [], "type_01": [], "type_02": [], "sprite" : []}
    sprites_folder = "./sprites/pokemon/centered-sprites/"
    for gen, max_pkm in gen_folders.items():
        gen_folder = os.path.join(sprites_folder,gen)
        for pkm_id in range(1,max_pkm+1):
            image_file = "{id}.png".format(id=pkm_id)
            image_path = os.path.join(gen_folder,image_file)
            image = mpimg.imread(image_path)
            if image.shape[2] == 4:
                image = rgba2rgb(image)
            image = color.rgb2hsv(image)
            df_dict["id"].append(pkm_id)
            df_dict["type_01"].append(pkm_dict[pkm_id]["type_01"])
            df_dict["type_02"].append(pkm_dict[pkm_id]["type_02"])
            df_dict["sprite"].append(image)
            df_dict["gen"].append(gen)
    return pd.DataFrame.from_dict(df_dict)