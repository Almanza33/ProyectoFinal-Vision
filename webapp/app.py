"""
App web (Gradio) para mostrar el modelo de segmentacion de inundaciones SAR.
Carga la U-Net ResNet-34 entrenada y, dado un GeoTIFF Sentinel-1 (bandas VV/VH en dB),
predice la mascara de inundacion y la muestra superpuesta.

Funciona igual en Hugging Face Spaces, Google Colab o local.
Coincide con el pipeline de test.ipynb:
  - modelo: smp.Unet(resnet34, in_channels=3, classes=1, logits)
  - entrada: 3 canales [VV_norm, VH_norm, VH_norm] (z-score con stats de train)
  - NaN -> 0 tras normalizar ; umbral de decision 0.5
"""
import os
import json
import numpy as np
import rasterio
import torch
import gradio as gr
import segmentation_models_pytorch as smp

# ----------------------------------------------------------------------------- config
WEIGHTS_PATH = os.environ.get("WEIGHTS_PATH", "checkpoints/baseline_unet_resnet34.pt")
# Stats de normalizacion del split de entrenamiento (fallback si no hay norm_stats.json)
DEFAULT_NORM = {"mean_vv": -10.3929, "std_vv": 4.0388, "mean_vh": -17.2411, "std_vh": 4.7540}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_norm():
    for p in ["norm_stats.json", "data/sen1floods11/HandLabeled/norm_stats.json"]:
        if os.path.exists(p):
            with open(p) as f:
                return json.load(f)
    return DEFAULT_NORM


NORM = load_norm()


def build_model():
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None,
                     in_channels=3, classes=1, activation=None)
    if os.path.exists(WEIGHTS_PATH):
        ckpt = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        state = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        model.load_state_dict(state)
        info = f"Pesos cargados desde {WEIGHTS_PATH}"
        if isinstance(ckpt, dict) and "val_iou" in ckpt:
            info += f" (epoca {ckpt.get('epoch', '?')}, val IoU={ckpt['val_iou']:.4f})"
    else:
        info = f"ADVERTENCIA: no se encontro {WEIGHTS_PATH}. Sube el checkpoint para predicciones reales."
    model.to(DEVICE).eval()
    return model, info


MODEL, MODEL_INFO = build_model()


# ----------------------------------------------------------------------------- utils
def percentile_clip(x, lo=2, hi=98):
    if np.all(np.isnan(x)):
        return np.zeros_like(x)
    vmin, vmax = np.nanpercentile(x, lo), np.nanpercentile(x, hi)
    c = np.clip((x - vmin) / (vmax - vmin + 1e-10), 0, 1)
    c[np.isnan(x)] = 0
    return c


def to_uint8(x):
    return (percentile_clip(x) * 255).astype(np.uint8)


def pad_to_multiple(arr, m=32):
    _, h, w = arr.shape
    ph, pw = (m - h % m) % m, (m - w % m) % m
    return np.pad(arr, ((0, 0), (0, ph), (0, pw)), mode="reflect"), h, w


def predict(tif_path):
    if tif_path is None:
        return None, None, None, None, "Sube o elige un GeoTIFF Sentinel-1 (2 bandas VV/VH)."
    try:
        with rasterio.open(tif_path) as src:
            if src.count < 2:
                return None, None, None, None, "El archivo debe tener 2 bandas (VV y VH)."
            vv = src.read(1).astype(np.float32)
            vh = src.read(2).astype(np.float32)
    except Exception as e:
        return None, None, None, None, f"No se pudo leer el GeoTIFF: {e}"

    valid = ~np.isnan(vv) & ~np.isnan(vh)
    vvn = (vv - NORM["mean_vv"]) / NORM["std_vv"]
    vhn = (vh - NORM["mean_vh"]) / NORM["std_vh"]
    vvn = np.where(np.isnan(vvn), 0.0, vvn)
    vhn = np.where(np.isnan(vhn), 0.0, vhn)

    img = np.stack([vvn, vhn, vhn], axis=0).astype(np.float32)  # (3,H,W)
    img_p, h0, w0 = pad_to_multiple(img)
    tensor = torch.from_numpy(img_p).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        prob = torch.sigmoid(MODEL(tensor))[0, 0].cpu().numpy()
    prob = prob[:h0, :w0]
    mask = (prob > 0.5) & valid

    vv_img = np.stack([to_uint8(vv)] * 3, axis=-1)
    vh_img = np.stack([to_uint8(vh)] * 3, axis=-1)

    mask_img = np.zeros((*mask.shape, 3), np.uint8)
    mask_img[mask] = [30, 90, 255]
    mask_img[~valid] = [128, 128, 128]

    overlay = vv_img.copy()
    overlay[mask] = (0.40 * vv_img[mask] + 0.60 * np.array([30, 90, 255])).astype(np.uint8)

    pct = 100.0 * mask.sum() / max(int(valid.sum()), 1)
    summary = (f"Pixeles inundados: {int(mask.sum()):,} de {int(valid.sum()):,} validos "
               f"({pct:.1f} %). Azul = inundado, gris = sin dato.")
    return vv_img, vh_img, mask_img, overlay, summary


# ----------------------------------------------------------------------------- UI
example_files = sorted(
    os.path.join("examples", f) for f in os.listdir("examples")
    if f.lower().endswith((".tif", ".tiff"))
) if os.path.isdir("examples") else []

DESC = f"""
# Segmentacion de inundaciones en imagenes Sentinel-1 SAR (U-Net ResNet-34)

Sube un **GeoTIFF de Sentinel-1** con dos bandas (**VV** y **VH**, en dB) o elige uno de los
ejemplos. El modelo predice una **mascara de inundacion** por pixel y la muestra superpuesta
sobre la imagen VV.

*Las imagenes deben ser tiles SAR (no fotos normales). En la carpeta de entrega se incluyen
tiles de prueba listos para usar.*

**Estado del modelo:** {MODEL_INFO}
"""

with gr.Blocks(title="Inundaciones SAR — U-Net") as demo:
    gr.Markdown(DESC)
    with gr.Row():
        with gr.Column(scale=1):
            inp = gr.File(label="GeoTIFF Sentinel-1 (VV/VH)", file_types=[".tif", ".tiff"], type="filepath")
            btn = gr.Button("Predecir inundacion", variant="primary")
            out_txt = gr.Textbox(label="Resumen", interactive=False)
            if example_files:
                gr.Examples(examples=[[f] for f in example_files], inputs=inp)
        with gr.Column(scale=2):
            with gr.Row():
                out_vv = gr.Image(label="VV (dB)")
                out_vh = gr.Image(label="VH (dB)")
            with gr.Row():
                out_mask = gr.Image(label="Mascara predicha")
                out_over = gr.Image(label="Overlay sobre VV")

    btn.click(predict, inputs=inp, outputs=[out_vv, out_vh, out_mask, out_over, out_txt])
    inp.change(predict, inputs=inp, outputs=[out_vv, out_vh, out_mask, out_over, out_txt])

if __name__ == "__main__":
    demo.launch()
