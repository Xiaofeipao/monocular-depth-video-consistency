from pathlib import Path
import cv2
import numpy as np
from ..io import write_image


def colorize(depth, minimum, maximum):
    if not np.isfinite([minimum,maximum]).all() or maximum <= minimum:
        raise ValueError("Visualization range must be finite and increasing")
    valid = np.isfinite(depth) & (depth>0)
    normalized = np.clip((np.where(valid,depth,minimum)-minimum)/(maximum-minimum),0,1)
    colored = cv2.applyColorMap(np.uint8(normalized*255),cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def panel(rgb, depth, minimum, maximum, unit, label="depth", app=None):
    colored = colorize(depth,minimum,maximum)
    images = [rgb,colored] + ([app] if app is not None else [])
    result = np.hstack(images)
    # An explicit fixed numeric scale is part of every output frame.
    footer = np.zeros((60,result.shape[1],3),np.uint8)
    gradient = np.linspace(minimum,maximum,result.shape[1],dtype=np.float32)[None]
    footer[:12] = cv2.resize(colorize(gradient,minimum,maximum),(result.shape[1],12))
    cv2.putText(footer,f"{label} | {minimum:g} - {maximum:g} {unit}",(8,40),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,255,255),1)
    return np.vstack((result,footer))


def render_run(path, records, minimum, maximum, unit, fps=None, label="depth"):
    from ..io import read_bgr
    path = Path(path)
    writer = None
    try:
        for record in records:
            key = record['id']
            image = panel(read_bgr(path/"rgb"/f"{key}.png"),np.load(path/"depth"/f"{key}.npy"),minimum,maximum,unit,label)
            # MPEG-4 encoders require even dimensions.
            h,w = image.shape[:2]
            image = cv2.copyMakeBorder(image,0,h%2,0,w%2,cv2.BORDER_CONSTANT)
            if fps is not None:
                if writer is None:
                    writer = cv2.VideoWriter(str(path/"visualization"/"rgb_depth.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),fps,(image.shape[1],image.shape[0]))
                    if not writer.isOpened():
                        raise OSError("Cannot initialize MP4 encoder")
                writer.write(image)
            else:
                write_image(path/"visualization"/f"{key}.png",image)
    finally:
        if writer:
            writer.release()
