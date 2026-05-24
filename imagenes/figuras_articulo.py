# ============================================================
#  FIGURAS PARA EL ARTÍCULO — pegar estas celdas al final
#  de cualquiera de los notebooks de entrenamiento
#  (testLeeFilter.ipynb, testMedian3x3.ipynb, Base_line.ipynb)
#
#  IMPORTANTE: correr DESPUÉS de haber entrenado los 3 modelos
#  (las 3 configuraciones deben tener su .csv en logs/ y su
#  checkpoint en checkpoints/)
# ============================================================

# ─────────────────────────────────────────────────────────────
#  CELDA 1 — CURVAS DE ENTRENAMIENTO COMPARATIVAS (3 configs)
#  Pegar como celda nueva al final de cualquier notebook
# ─────────────────────────────────────────────────────────────

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from pathlib import Path

ROOT = Path('.')

# Rutas a los 3 CSV de métricas (se crean al entrenar cada notebook)
csv_paths = {
    'Sin filtro':  ROOT / 'logs' / 'metrics_HandLabeled.csv',
    'Mediana 3×3': ROOT / 'logs' / 'metrics_median.csv',
    'Lee':         ROOT / 'logs' / 'metrics_lee.csv',
}

colors = {
    'Sin filtro':  '#E07B00',   # naranja
    'Mediana 3×3': '#2E8B57',   # verde
    'Lee':         '#7B3FA0',   # morado
}

# Épocas de mejor val IoU (para marcar con línea vertical)
best_epochs = {
    'Sin filtro':  20,
    'Mediana 3×3': 40,
    'Lee':         45,
}

# Leer todos los CSV disponibles
dfs = {}
for name, path in csv_paths.items():
    if path.exists():
        dfs[name] = pd.read_csv(path)
    else:
        print(f'[AVISO] No se encontró {path} — entrena ese modelo primero.')

if not dfs:
    raise FileNotFoundError('No se encontró ningún CSV de métricas. Entrena los modelos primero.')

fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
fig.suptitle('Curvas de entrenamiento — comparación de configuraciones de preprocesamiento',
             fontsize=13, fontweight='bold')

# Panel izquierdo: Loss (train + val)
ax = axes[0]
for name, df in dfs.items():
    c = colors[name]
    ax.plot(df['epoch'], df['train_loss'], color=c, linewidth=1.5,
            linestyle='--', alpha=0.6, label=f'{name} (train)')
    ax.plot(df['epoch'], df['val_loss'],   color=c, linewidth=2.0,
            label=f'{name} (val)')
ax.set_title('Loss combinada (BCE + Dice)', fontsize=11)
ax.set_xlabel('Época')
ax.set_ylabel('Loss')
ax.legend(fontsize=7.5, ncol=1)
ax.grid(True, alpha=0.3)

# Panel central: val IoU + val Dice
ax = axes[1]
for name, df in dfs.items():
    c = colors[name]
    ax.plot(df['epoch'], df['val_iou'],  color=c, linewidth=2.0, label=f'{name} — IoU')
    ax.plot(df['epoch'], df['val_dice'], color=c, linewidth=1.5,
            linestyle=':', alpha=0.8, label=f'{name} — Dice')
    # Marcar época del mejor checkpoint
    if name in best_epochs:
        ep = best_epochs[name]
        row = df[df['epoch'] == ep]
        if not row.empty:
            ax.axvline(ep, color=c, linewidth=0.8, linestyle='-.', alpha=0.5)
            ax.scatter([ep], [row['val_iou'].values[0]], color=c, s=60, zorder=5)
ax.set_title('IoU y Dice en validación', fontsize=11)
ax.set_xlabel('Época')
ax.set_ylabel('Métrica')
ax.legend(fontsize=7.5, ncol=1)
ax.grid(True, alpha=0.3)

# Panel derecho: Precisión y Recall en val
ax = axes[2]
for name, df in dfs.items():
    c = colors[name]
    ax.plot(df['epoch'], df['val_prec'], color=c, linewidth=2.0, label=f'{name} — Precisión')
    ax.plot(df['epoch'], df['val_rec'],  color=c, linewidth=1.5,
            linestyle=':', alpha=0.8, label=f'{name} — Recall')
ax.set_title('Precisión y Recall en validación', fontsize=11)
ax.set_xlabel('Época')
ax.set_ylabel('Métrica')
ax.legend(fontsize=7.5, ncol=1)
ax.grid(True, alpha=0.3)

plt.savefig('curvas_entrenamiento_comparativas.png', dpi=200, bbox_inches='tight')
plt.show()
print('Figura guardada: curvas_entrenamiento_comparativas.png')


# ─────────────────────────────────────────────────────────────
#  CELDA 2 — ANÁLISIS CUALITATIVO: CASOS BUENOS Y MALOS
#  Pegar como celda nueva en testLeeFilter.ipynb,
#  DESPUÉS de haber cargado el checkpoint (celda 21)
#
#  Requiere que ya estén definidos en el notebook:
#    model, test_ds, test_pairs, S1_DIR, DEVICE, SEED
#    percentile_clip() — ya definida en celda 24
# ─────────────────────────────────────────────────────────────

import numpy as np
import torch
import rasterio
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from tqdm.notebook import tqdm

# --- 1. Calcular IoU por tile en el test set ---
model.eval()
per_tile_iou = []

with torch.inference_mode():
    for idx in tqdm(range(len(test_ds)), desc='Calculando IoU por tile'):
        sample = test_ds[idx]
        image      = sample['image'].unsqueeze(0).to(DEVICE)       # (1,3,H,W)
        label      = sample['label']                                # (H,W) long
        valid_mask = sample['valid_mask']                           # (H,W) bool

        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            logits = model(image)                                   # (1,1,H,W)

        probs = torch.sigmoid(logits).squeeze().cpu()               # (H,W)
        pred  = (probs >= 0.5).long()

        # IoU solo sobre píxeles válidos, clase inundado (1)
        vm   = valid_mask
        tp   = ((pred == 1) & (label == 1) & vm).sum().item()
        fp   = ((pred == 1) & (label == 0) & vm).sum().item()
        fn   = ((pred == 0) & (label == 1) & vm).sum().item()
        denom = tp + fp + fn
        iou  = tp / denom if denom > 0 else float('nan')

        per_tile_iou.append((idx, iou))

# --- 2. Separar buenos y malos (ignorar tiles sin píxeles inundados) ---
valid_tiles = [(idx, iou) for idx, iou in per_tile_iou if not np.isnan(iou)]
valid_tiles.sort(key=lambda x: x[1])

N = 3  # número de tiles a mostrar por categoría
worst_tiles = valid_tiles[:N]                   # menor IoU
best_tiles  = valid_tiles[-(N):][::-1]          # mayor IoU

print(f'Peores IoU: {[round(v,4) for _,v in worst_tiles]}')
print(f'Mejores IoU: {[round(v,4) for _,v in best_tiles]}')

# --- 3. Función para predecir un tile ---
def predict_tile(idx):
    """Devuelve (vv_raw, label_np, valid_mask_np, pred_np, iou, tile_name)."""
    s1_fname, _ = test_pairs[idx]
    with rasterio.open(S1_DIR / s1_fname) as src:
        vv_raw = src.read(1).astype(np.float32)

    sample     = test_ds[idx]
    image      = sample['image'].unsqueeze(0).to(DEVICE)
    label_np   = sample['label'].numpy()
    valid_np   = sample['valid_mask'].numpy()

    with torch.inference_mode():
        with torch.cuda.amp.autocast(enabled=(DEVICE.type == 'cuda')):
            logits = model(image)
    pred_np = (torch.sigmoid(logits).squeeze().cpu() >= 0.5).numpy().astype(np.uint8)

    _, iou = per_tile_iou[idx]
    tile_name = Path(s1_fname).stem.replace('_S1Hand', '')
    return vv_raw, label_np, valid_np, pred_np, iou, tile_name

# --- 4. Figura: 3 buenos + 3 malos ---
cmap_mask = mcolors.ListedColormap(['lightgray', 'navy'])

fig, axes = plt.subplots(
    2 * N, 4,
    figsize=(14, 4.5 * N),
    constrained_layout=True,
    gridspec_kw={'hspace': 0.35, 'wspace': 0.08}
)
fig.suptitle(
    'Análisis cualitativo — Mejores y peores predicciones (Filtro Lee, umbral 0.5)',
    fontsize=13, fontweight='bold'
)

col_titles = ['SAR (VV banda)', 'Máscara real', 'Predicción', 'Overlay (TP/FP/FN)']

def overlay_colors(label, pred, valid):
    """RGB overlay: TP=verde, FP=rojo, FN=naranja, fondo=gris, nodata=negro."""
    h, w = label.shape
    rgb = np.ones((h, w, 3)) * 0.85       # fondo gris claro
    rgb[~valid] = [0.1, 0.1, 0.1]         # nodata negro
    tp = (pred == 1) & (label == 1) & valid
    fp = (pred == 1) & (label == 0) & valid
    fn = (pred == 0) & (label == 1) & valid
    rgb[tp] = [0.18, 0.63, 0.18]          # verde
    rgb[fp] = [0.85, 0.15, 0.15]          # rojo
    rgb[fn] = [1.00, 0.55, 0.00]          # naranja
    return rgb

for row_idx, (tile_list, group_label) in enumerate(
    [(best_tiles, 'BUENO'), (worst_tiles, 'MALO')]
):
    for col_n, (idx, iou) in enumerate(tile_list):
        r = row_idx * N + col_n
        vv_raw, label_np, valid_np, pred_np, iou_val, tile_name = predict_tile(idx)

        vv_vis   = percentile_clip(vv_raw)
        label_vis = np.where(valid_np, label_np, -1)

        # columna 0: SAR
        axes[r, 0].imshow(vv_vis, cmap='gray', interpolation='nearest')
        axes[r, 0].set_title(
            f'[{group_label}] {tile_name}\nIoU = {iou_val:.4f}',
            fontsize=8, fontweight='bold',
            color='#1b5e20' if group_label == 'BUENO' else '#b71c1c'
        )

        # columna 1: Máscara real
        masked_label = np.ma.masked_where(~valid_np, label_np)
        axes[r, 1].imshow(np.zeros_like(vv_vis), cmap='gray', vmin=0, vmax=1)
        axes[r, 1].imshow(masked_label, cmap=cmap_mask, vmin=0, vmax=1,
                           interpolation='nearest', alpha=0.9)

        # columna 2: Predicción
        axes[r, 2].imshow(vv_vis, cmap='gray', interpolation='nearest', alpha=0.4)
        axes[r, 2].imshow(
            np.ma.masked_where(pred_np == 0, pred_np),
            cmap=mcolors.ListedColormap(['navy']),
            interpolation='nearest', alpha=0.7
        )

        # columna 3: Overlay TP/FP/FN
        axes[r, 3].imshow(overlay_colors(label_np, pred_np, valid_np),
                           interpolation='nearest')

        for ax in axes[r]:
            ax.axis('off')

# Títulos de columna solo en la primera fila
for c, title in enumerate(col_titles):
    axes[0, c].set_title(title, fontsize=9, fontweight='bold', pad=4)

# Leyenda overlay
from matplotlib.patches import Patch
legend_patches = [
    Patch(color='#2ea02e', label='TP (correcto)'),
    Patch(color='#d92424', label='FP (falso positivo)'),
    Patch(color='#ff8c00', label='FN (falso negativo)'),
    Patch(color='lightgray', label='Seco correcto'),
]
fig.legend(handles=legend_patches, loc='lower center',
           ncol=4, fontsize=9, frameon=True,
           bbox_to_anchor=(0.5, -0.02))

plt.savefig('casos_buenos_malos_lee.png', dpi=180, bbox_inches='tight')
plt.show()
print('Figura guardada: casos_buenos_malos_lee.png')
