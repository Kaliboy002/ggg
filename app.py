import gradio
from huggingface_hub import Repository
import os

from utils.utils import norm_crop, estimate_norm, inverse_estimate_norm, transform_landmark_points, get_lm
from networks.layers import AdaIN, AdaptiveAttention
from tensorflow_addons.layers import InstanceNormalization
import numpy as np
import cv2
from scipy.ndimage import gaussian_filter

from tensorflow.keras.models import load_model
from options.swap_options import SwapOptions

# .
token = os.environ['model_fetch']

opt = SwapOptions().parse()


retina_repo = Repository(local_dir="retina_model", clone_from="felixrosberg/retinaface_resnet50",
                         private=True, use_auth_token=token, git_user="felixrosberg")
                         
from retina_model.models import *
RetinaFace = load_model("retina_model/retinaface_res50.h5",
                        custom_objects={"FPN": FPN,
                                        "SSH": SSH,
                                        "BboxHead": BboxHead,
                                        "LandmarkHead": LandmarkHead,
                                        "ClassHead": ClassHead})

arc_repo = Repository(local_dir="arcface_model", clone_from="felixrosberg/arcface_tf",
                      private=True, use_auth_token=token)
ArcFace = load_model("arcface_model/arc_res50.h5")

g_repo = Repository(local_dir="g_model_c_hq", clone_from="felixrosberg/affa_config_c_hq",
                    private=True, use_auth_token=token)
G = load_model("g_model_c_hq/generator_t_28.h5", custom_objects={"AdaIN": AdaIN,
                                                         "AdaptiveAttention": AdaptiveAttention,
                                                         "InstanceNormalization": InstanceNormalization})

blend_mask_base = np.zeros(shape=(256, 256, 1))
blend_mask_base[80:250, 32:224] = 1
blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)


def run_inference(target, source, slider, settings):
    try:
        source = np.array(source)
        target = np.array(target)
    
        # Prepare to load video
        if "anonymize" not in settings:
            source_a = RetinaFace(np.expand_dims(source, axis=0)).numpy()[0]
            source_h, source_w, _ = source.shape
            source_lm = get_lm(source_a, source_w, source_h)
            source_aligned = norm_crop(source, source_lm, image_size=256)
            source_z = ArcFace.predict(np.expand_dims(tf.image.resize(source_aligned, [112, 112]) / 255.0, axis=0))
        else:
            source_z = None
    
        # read frame
        im = target
        im_h, im_w, _ = im.shape
        im_shape = (im_w, im_h)
    
        detection_scale = im_w // 640 if im_w > 640 else 1
    
        faces = RetinaFace(np.expand_dims(cv2.resize(im,
                                                     (im_w // detection_scale,
                                                      im_h // detection_scale)), axis=0)).numpy()
    
        total_img = im / 255.0
        for annotation in faces:
            lm_align = np.array([[annotation[4] * im_w, annotation[5] * im_h],
                                 [annotation[6] * im_w, annotation[7] * im_h],
                                 [annotation[8] * im_w, annotation[9] * im_h],
                                 [annotation[10] * im_w, annotation[11] * im_h],
                                 [annotation[12] * im_w, annotation[13] * im_h]],
                                dtype=np.float32)
    
            # align the detected face
            M, pose_index = estimate_norm(lm_align, 256, "arcface", shrink_factor=1.0)
            im_aligned = cv2.warpAffine(im, M, (256, 256), borderValue=0.0)

            if "anonymize" in settings:
                source_z = ArcFace.predict(np.expand_dims(tf.image.resize(im_aligned, [112, 112]) / 255.0, axis=0))
                anon_ratio = int(512 * (slider / 100))
                anon_vector = np.ones(shape=(1, 512))
                anon_vector[:, :anon_ratio] = -1
                np.random.shuffle(anon_vector)
                source_z *= anon_vector
    
            # face swap
            changed_face_cage = G.predict([np.expand_dims((im_aligned - 127.5) / 127.5, axis=0),
                                           source_z])
            changed_face = (changed_face_cage[0] + 1) / 2
    
            # get inverse transformation landmarks
            transformed_lmk = transform_landmark_points(M, lm_align)
    
            # warp image back
            iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
            iim_aligned = cv2.warpAffine(changed_face, iM, im_shape, borderValue=0.0)
    
            # blend swapped face with target image
            blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
            blend_mask = np.expand_dims(blend_mask, axis=-1)
            total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))
    
        if "compare" in settings:
            total_img = np.concatenate((im / 255.0, total_img), axis=1)
    
        total_img = np.clip(total_img, 0, 1)
        total_img *= 255.0
        total_img = total_img.astype('uint8')
    
        return total_img
    except Exception as e:
        print(e)
        return None


description = "Performs subject agnostic identity transfer from a source face to all target faces. \n\n" \
              "Options:\n" \
              "compare returns the target image concatenated with the results.\n" \
              "anonymize will ignore the source image and perform an identity permutation of target faces.\n" \
              "\n" \
              "Note, source image with too high resolution may not work properly!"
examples = [["assets/rick.jpg", "assets/musk.jpg", 80, ["compare"]],
            ["assets/girl_1.png", "assets/girl_0.png", 80, []],
            ["assets/musk.jpg", "assets/musk.jpg", 30, ["anonymize"]]]
article="""
Demo is based of recent research from my Ph.D work. Results expects to be published in the coming months.
"""

iface = gradio.Interface(run_inference,
                         [gradio.inputs.Image(shape=None, label='Target'),
                          gradio.inputs.Image(shape=None, label='Source'),
                          gradio.inputs.Slider(0, 100, default=80, label="Anonymization ratio (%)"),
                          gradio.inputs.CheckboxGroup(["compare", "anonymize"], label='Options')],
                         gradio.outputs.Image(),
                         title="Face Swap",
                         description=description,
                         examples=examples,
                         article=article,
                         theme="dark-huggingface",
                         layout="vertical")
iface.launch()
