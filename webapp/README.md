# Demo web — Segmentación de inundaciones SAR (U-Net ResNet-34)

App interactiva para que cualquier persona (p. ej. el profesor) pruebe el modelo: sube un
GeoTIFF Sentinel-1 (bandas VV/VH en dB) o elige un ejemplo, y obtiene la **máscara de
inundación** predicha superpuesta sobre la imagen.

## Archivos
- `app.py` — aplicación Gradio (carga el modelo y hace la inferencia).
- `requirements.txt` — dependencias.
- `checkpoints/baseline_unet_resnet34.pt` — **pesos entrenados** (debes copiarlo aquí; lo genera `test.ipynb`).
- `examples/` — tiles SAR de prueba (`.tif`, 2 bandas) que aparecen como ejemplos clicables.

## Cómo lo usa el profesor (instrucciones de prueba)
1. Abrir la URL de la app.
2. En "GeoTIFF Sentinel-1" elegir uno de los **ejemplos** ya cargados, o subir uno de los
   tiles `.tif` entregados en la carpeta `examples/`.
3. Pulsar **Predecir inundación**.
4. La app muestra cuatro paneles: VV, VH, la máscara predicha y el overlay sobre VV
   (azul = inundado, gris = sin dato), más un resumen con el porcentaje de píxeles inundados.

> Nota: la entrada deben ser tiles SAR de Sentinel-1 (2 bandas VV/VH en dB), no fotos comunes.

## Opción A — Desplegar en Hugging Face Spaces (recomendado, URL pública)
1. Crear cuenta gratuita en https://huggingface.co y un nuevo **Space** con SDK = **Gradio**.
2. Subir al Space: `app.py`, `requirements.txt`, la carpeta `examples/` (con los tiles) y
   `checkpoints/baseline_unet_resnet34.pt`.
3. El Space construye solo y queda accesible en una URL pública para compartir con el profesor.
   (Funciona en CPU; la inferencia de un tile 512×512 es rápida.)

## Opción B — Google Colab (sin instalar nada)
```python
!pip -q install gradio torch segmentation-models-pytorch rasterio numpy
# sube app.py, checkpoints/ y examples/ al entorno de Colab, luego:
!python app.py
```
Gradio imprime un link público temporal (`share=True`) para abrir la app.

## Opción C — Local
```bash
pip install -r requirements.txt
python app.py        # abre http://127.0.0.1:7860
```

## Notas técnicas (coincide con test.ipynb)
- Modelo: `smp.Unet(encoder_name="resnet34", in_channels=3, classes=1)` (logits).
- Entrada: 3 canales `[VV_norm, VH_norm, VH_norm]`, z-score con las estadísticas de train
  (`norm_stats.json` si está; si no, valores por defecto del entrenamiento).
- NaN → 0 tras normalizar; umbral de decisión 0.5.
