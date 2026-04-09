from typing import Tuple
import math
import cv2
import numpy as np

def get_image_transform_with_border(in_res, out_res, bgr_to_rgb: bool=False):
    """ adds a border to make the input image square, and then resizes it to the output resolution """
    iw, ih = in_res
    interp_method = cv2.INTER_AREA

    # Determine the size of the square
    size = max(iw, ih)
    top = (size - ih) // 2
    bottom = size - ih - top
    left = (size - iw) // 2
    right = size - iw - left

    def transform(img: np.ndarray):
        assert img.shape == (ih, iw, 3)
        # Add border to make the image square
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        # Resize
        img = cv2.resize(img, out_res, interpolation=interp_method)
        if bgr_to_rgb:
            img = img[:, :, ::-1]
        return img
    
    return transform

def get_image_transform(
        input_res: Tuple[int,int]=(1280,720), 
        output_res: Tuple[int,int]=(640,480), 
        bgr_to_rgb: bool=False,
        val: bool=False):

    iw, ih = input_res
    ow, oh = output_res
    rw, rh = None, None
    interp_method = cv2.INTER_AREA

    if (iw/ih) >= (ow/oh):
        # input is wider
        rh = oh
        rw = math.ceil(rh / ih * iw)
        if oh > ih:
            interp_method = cv2.INTER_LINEAR
    else:
        rw = ow
        rh = math.ceil(rw / iw * ih)
        if ow > iw:
            interp_method = cv2.INTER_LINEAR
    
    w_slice_start = (rw - ow) // 2
    w_slice = slice(w_slice_start, w_slice_start + ow)
    h_slice_start = (rh - oh) // 2
    h_slice = slice(h_slice_start, h_slice_start + oh)
    c_slice = slice(None)
    if bgr_to_rgb:
        c_slice = slice(None, None, -1)

    def transform(img: np.ndarray):
        assert img.shape == ((ih,iw,3))
        # resize
        img = cv2.resize(img, (rw, rh), interpolation=interp_method)
        # crop
        img = img[h_slice, w_slice, c_slice]
        return img
    return transform


# Apply mask to image. 
# img: [h,w,3] image
# mask_polygon_vertices: [n,2] mask polygon vertices
def apply_polygon_mask(img: np.ndarray, mask_polygon_vertices: np.ndarray, color: Tuple[int,int,int]=(0,0,0)):
    mask_pts = np.array(mask_polygon_vertices, dtype=np.int32)  # Extract mask points
    mask = np.ones_like(img, dtype=np.uint8) * 255
    cv2.fillPoly(mask, [mask_pts], color)
    # apply the mask to the images
    img_masked = cv2.bitwise_and(img, mask)
    return img_masked