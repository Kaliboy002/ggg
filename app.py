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



token = os.environ['model_fetch']

opt = SwapOptions().parse()

retina_repo = Repository(local_dir="retina_model", clone_from="felixrosberg/retinaface_resnet50", use_auth_token=token)

from retina_model.models import *

RetinaFace = load_model("retina_model/retinaface_res50.h5",
                        custom_objects={"FPN": FPN,
                                        "SSH": SSH,
                                        "BboxHead": BboxHead,
                                        "LandmarkHead": LandmarkHead,
                                        "ClassHead": ClassHead})

arc_repo = Repository(local_dir="arcface_model", clone_from="felixrosberg/arcface_tf", use_auth_token=token)
ArcFace = load_model("arcface_model/arc_res50.h5")
ArcFaceE = load_model("arcface_model/arc_res50e.h5")

g_repo = Repository(local_dir="g_model_c_hq", clone_from="felixrosberg/affa_config_c_hq", use_auth_token=token)
G = load_model("g_model_c_hq/generator_t_28.h5", custom_objects={"AdaIN": AdaIN,
                                                                 "AdaptiveAttention": AdaptiveAttention,
                                                                 "InstanceNormalization": InstanceNormalization})

r_repo = Repository(local_dir="reconstruction_attack", clone_from="felixrosberg/reconstruction_attack", use_auth_token=token)
R = load_model("reconstruction_attack/reconstructor_42.h5", custom_objects={"AdaIN": AdaIN,
                                                                            "AdaptiveAttention": AdaptiveAttention,
                                                                            "InstanceNormalization": InstanceNormalization})

permuter_repo = Repository(local_dir="identity_permuter", clone_from="felixrosberg/identitypermuter", use_auth_token=token, git_user="felixrosberg")

from identity_permuter.id_permuter import identity_permuter

IDP = identity_permuter(emb_size=32, min_arg=False)
IDP.load_weights("identity_permuter/id_permuter.h5")

blend_mask_base = np.zeros(shape=(256, 256, 1))
blend_mask_base[80:244, 32:224] = 1
blend_mask_base = gaussian_filter(blend_mask_base, sigma=7)


theme = gr.themes.Monochrome(
    secondary_hue="emerald",
    neutral_hue="teal",
).set(
    body_background_fill='*primary_950',
    body_background_fill_dark='*secondary_950',
    body_text_color='*primary_50',
    body_text_color_dark='*secondary_100',
    body_text_color_subdued='*primary_300',
    body_text_color_subdued_dark='*primary_300',
    background_fill_primary='*primary_600',
    background_fill_primary_dark='*primary_400',
    background_fill_secondary='*primary_950',
    background_fill_secondary_dark='*primary_950',
    border_color_accent='*secondary_600',
    border_color_primary='*secondary_50',
    border_color_primary_dark='*secondary_50',
    color_accent='*secondary_50',
    color_accent_soft='*primary_500',
    color_accent_soft_dark='*primary_500',
    link_text_color='*secondary_950',
    link_text_color_dark='*primary_50',
    link_text_color_active='*primary_50',
    link_text_color_active_dark='*primary_50',
    link_text_color_hover='*primary_50',
    link_text_color_hover_dark='*primary_50',
    link_text_color_visited='*primary_50',
    block_background_fill='*primary_950',
    block_background_fill_dark='*primary_950',
    block_border_color='*secondary_500',
    block_border_color_dark='*secondary_500',
    block_info_text_color='*primary_50',
    block_info_text_color_dark='*primary_50',
    block_label_background_fill='*primary_950',
    block_label_background_fill_dark='*secondary_950',
    block_label_border_color='*secondary_500',
    block_label_border_color_dark='*secondary_500',
    block_label_text_color='*secondary_500',
    block_label_text_color_dark='*secondary_500',
    block_title_background_fill='*primary_950',
    panel_background_fill='*primary_950',
    panel_border_color='*primary_950',
    checkbox_background_color='*primary_950',
    checkbox_background_color_dark='*primary_950',
    checkbox_background_color_focus='*primary_950',
    checkbox_border_color='*secondary_500',
    input_background_fill='*primary_800',
    input_background_fill_focus='*primary_950',
    input_background_fill_hover='*secondary_950',
    input_placeholder_color='*secondary_950',
    slider_color='*primary_950',
    slider_color_dark='*primary_950',
    table_even_background_fill='*primary_800',
    table_odd_background_fill='*primary_600',
    button_primary_background_fill='*primary_800',
    button_primary_background_fill_dark='*primary_800'
)


def run_inference(target, source, slider, adv_slider, settings):
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
            im_aligned = (cv2.warpAffine(im, M, (256, 256), borderValue=0.0) - 127.5) / 127.5

            if "adversarial defense" in settings:
                eps = adv_slider / 200
                X = tf.convert_to_tensor(np.expand_dims(im_aligned, axis=0))
                with tf.GradientTape() as tape:
                    tape.watch(X)

                    X_z = ArcFaceE(tf.image.resize(X * 0.5 + 0.5, [112, 112]))
                    output = R([X, X_z])

                    loss = tf.reduce_mean(tf.abs(0 - output))

                gradient = tf.sign(tape.gradient(loss, X))

                adv_x = X + eps * gradient
                im_aligned = tf.clip_by_value(adv_x, -1, 1)[0]

            if "anonymize" in settings and "reconstruction attack" not in settings:
                """source_z = ArcFace.predict(np.expand_dims(tf.image.resize(im_aligned, [112, 112]) / 255.0, axis=0))
                anon_ratio = int(512 * (slider / 100))
                anon_vector = np.ones(shape=(1, 512))
                anon_vector[:, :anon_ratio] = -1
                np.random.shuffle(anon_vector)
                source_z *= anon_vector"""

                slider_weight = slider / 100

                target_z = ArcFace.predict(np.expand_dims(tf.image.resize(im_aligned, [112, 112]) * 0.5 + 0.5, axis=0))
                source_z = IDP.predict(target_z)

                source_z = slider_weight * source_z + (1 - slider_weight) * target_z

            if "reconstruction attack" in settings:
                source_z = ArcFaceE.predict(np.expand_dims(tf.image.resize(im_aligned, [112, 112]) * 0.5 + 0.5, axis=0))

            # face swap
            if "reconstruction attack" not in settings:
                changed_face_cage = G.predict([np.expand_dims(im_aligned, axis=0),
                                               source_z])
                changed_face = changed_face_cage[0] * 0.5 + 0.5

                # get inverse transformation landmarks
                transformed_lmk = transform_landmark_points(M, lm_align)

                # warp image back
                iM, _ = inverse_estimate_norm(lm_align, transformed_lmk, 256, "arcface", shrink_factor=1.0)
                iim_aligned = cv2.warpAffine(changed_face, iM, im_shape, borderValue=0.0)

                # blend swapped face with target image
                blend_mask = cv2.warpAffine(blend_mask_base, iM, im_shape, borderValue=0.0)
                blend_mask = np.expand_dims(blend_mask, axis=-1)
                total_img = (iim_aligned * blend_mask + total_img * (1 - blend_mask))
            else:
                changed_face_cage = R.predict([np.expand_dims(im_aligned, axis=0),
                                               source_z])
                changed_face = changed_face_cage[0] * 0.5 + 0.5

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
              "Implementation and demo of FaceDancer, accepted to WACV 2023. \n\n" \
              "Pre-print: https://arxiv.org/abs/2210.10473 \n\n" \
              "Code: https://github.com/felixrosberg/FaceDancer \n\n" \
               "\n\n" \
              "Options:\n\n" \
              "-Compare returns the target image concatenated with the results.\n\n" \
              "-Anonymize will ignore the source image and perform an identity permutation of target faces.\n\n" \
              "-Reconstruction attack will attempt to invert the face swap or the anonymization.\n\n" \
              "-Adversarial defense will add a permutation noise that disrupts the reconstruction attack.\n\n" \
              "NOTE: There is no guarantees with the anonymization process currently.\n\n" \
              "NOTE: source image with too high resolution may not work properly!"
examples = [["assets/rick.jpg", "assets/musk.jpg", 100, 10, []],
            ["assets/rick.jpg", "assets/rick.jpg", 100, 10, ["anonymize"]]]
article = """
Demo is based of recent research from my Ph.D work. Results expects to be published in the coming months.
"""

with gr.Blocks(theme=theme) as blk_demo:
        gr.Markdown(value="# Face Dancer")
        with gr.Row():
            with gr.Column():
                with gr.Box():
                    trg_in = gr.Image(type="pil", label='Target').style(height=300)
                    src_in = gr.Image(type="pil", label='Source').style(height=300)
                with gr.Row():
                        b1 = gr.Button("Face Swap")
                with gr.Row():
                        with gr.Accordion("Options", open=False):
                            chk_in = gr.CheckboxGroup(["Compare",
                                                       "Anonymize",
                                                       "Reconstruction Attack",
                                                       "Adversarial Defense"],
                                                      label="Mode",
                                                      info="Anonymize mode? "
                                                           "Apply reconstruction attack? "
                                                           "Apply defense against reconstruction attack?")
                            def_in = gr.Slider(0, 100, value=100,
                                               label='Anonymization ratio (%)')
                            mrg_in = gr.Slider(0, 100, value=100,
                                               label='Adversarial defense ratio (%)')
            with gr.Column():
                with gr.Box():
                    ano_out = gr.Image(type="pil", label='Output').style(height=300)

b1.click(run_inference, inputs=[trg_in, src_in, def_in, mrg_in, chk_in],
                 outputs=[ano_out])

"""iface = gradio.Interface(run_inference,
                         [gradio.Image(shape=None, type="pil", label='Target'),
                          gradio.Image(shape=None, type="pil", label='Source'),
                          gradio.Slider(0, 100, value=100, label="Anonymization ratio (%)"),
                          gradio.Slider(0, 100, value=100, label="Adversarial defense ratio (%)"),
                          gradio.CheckboxGroup(["compare",
                                                       "anonymize",
                                                       "reconstruction attack",
                                                       "adversarial defense"],
                                                      label='Options')],
                         "image",
                         title="Face Swap",
                         description=description,
                         examples=examples,
                         article=article,
                         layout="vertical")"""
blk_demo.launch()
