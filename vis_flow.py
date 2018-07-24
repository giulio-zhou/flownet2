import cv2
import h5py
import numpy as np
import skimage.io as skio
import skvideo.io
import sys

def vis(flow):
    hsv = np.zeros(flow.shape + np.array([0, 0, 1]), dtype=np.uint8)
    hsv[..., 1] = 255
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = angle * 180 / (2 * np.pi)
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    rgb_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb_img

flow = h5py.File(sys.argv[1], 'r')['data']
if len(flow) == 1:
    skio.imsave(sys.argv[2], vis(flow[0]))
else:
    vwriter = skvideo.io.FFmpegWriter(
        sys.argv[2], outputdict={'-pix_fmt': 'yuv420p'})
    for i, flow_mat in enumerate(flow):
        vwriter.writeFrame(vis(flow_mat))
