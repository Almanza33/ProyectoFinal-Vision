from pathlib import Path
import random
import math
from datetime import datetime

from pystac_client import Client
import rasterio
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from affine import Affine
from tqdm import tqdm


# -----------------------------
# CONFIGURACIÓN
# -----------------------------

OUT_DIR = Path("tiffs_colombia")
OUT_DIR.mkdir(exist_ok=True)

STAC_URL = "https://earth-search.aws.element84.com/v1"
COLLECTION = "sentinel-2-c1-l2a"

DATE_RANGE = "2026-01-01/2026-03-01"
MAX_CLOUD = 30

N_TIFS = 100          # Cambia esto a 500, 1000, etc.
CHIP_PIXELS = 512    # 512 x 512 píxeles
CHIP_KM = 5.12       # 512 px * 10 m = 5.12 km aprox.

# Zonas base dentro de Colombia.
# Se usa jitter aleatorio alrededor de estos puntos para no bajar siempre lo mismo.
COLOMBIA_POINTS = [
    ("bogota", -74.10, 4.65),
    ("medellin", -75.57, 6.25),
    ("cali", -76.53, 3.45),
    ("barranquilla", -74.80, 10.98),
    ("cartagena", -75.50, 10.40),
    ("santa_marta", -74.20, 11.24),
    ("guajira", -72.80, 11.50),
    ("santander", -73.10, 7.10),
    ("eje_cafetero", -75.70, 4.80),
    ("huila", -75.30, 2.90),
    ("pasto", -77.28, 1.20),
    ("putumayo", -76.60, 0.50),
    ("llanos_meta", -73.60, 4.10),
    ("llanos_vichada", -70.80, 5.20),
    ("arauca", -70.75, 7.05),
    ("choco", -76.65, 5.70),
    ("pacifico_valle", -77.10, 3.85),
    ("amazonas_leticia", -69.95, -4.20),
    ("caqueta", -74.00, 1.60),
    ("guaviare", -72.65, 2.57),
]


# -----------------------------
# FUNCIONES
# -----------------------------

def bbox_around_point(lon, lat, size_km):
    """
    Crea un bbox en EPSG:4326 alrededor de un punto.
    """
    half_km = size_km / 2

    delta_lat = half_km / 111.0
    delta_lon = half_km / (111.0 * math.cos(math.radians(lat)))

    return [
        lon - delta_lon,
        lat - delta_lat,
        lon + delta_lon,
        lat + delta_lat,
    ]


def jitter_point(lon, lat, max_jitter_km=40):
    """
    Mueve aleatoriamente el punto hasta max_jitter_km.
    Sirve para generar variedad espacial.
    """
    dx = random.uniform(-max_jitter_km, max_jitter_km)
    dy = random.uniform(-max_jitter_km, max_jitter_km)

    new_lat = lat + dy / 111.0
    new_lon = lon + dx / (111.0 * math.cos(math.radians(lat)))

    return new_lon, new_lat


def s3_to_https(href):
    """
    Convierte rutas s3:// de Earth Search a URL HTTPS pública.
    Esto ayuda mucho en Windows.
    """
    if href.startswith("s3://"):
        no_scheme = href.replace("s3://", "")
        bucket, key = no_scheme.split("/", 1)
        return f"https://{bucket}.s3.us-west-2.amazonaws.com/{key}"
    return href


def search_best_item(catalog, bbox):
    """
    Busca una escena Sentinel-2 que intersecte el bbox.
    """
    search = catalog.search(
        collections=[COLLECTION],
        bbox=bbox,
        datetime=DATE_RANGE,
        query={"eo:cloud_cover": {"lt": MAX_CLOUD}},
        max_items=5,
        sortby=[{"field": "properties.eo:cloud_cover", "direction": "asc"}],
    )

    items = list(search.items())

    if not items:
        return None

    return items[0]


def save_visual_chip(item, bbox_wgs84, out_tif):
    """
    Lee el asset 'visual' RGB de Sentinel-2 y guarda un recorte GeoTIFF.
    """
    if "visual" not in item.assets:
        return False

    href = s3_to_https(item.assets["visual"].href)

    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
        with rasterio.open(href) as src:
            # Convertir bbox de lon/lat al CRS del raster Sentinel-2
            bbox_src = transform_bounds(
                "EPSG:4326",
                src.crs,
                *bbox_wgs84,
                densify_pts=21
            )

            window = from_bounds(*bbox_src, transform=src.transform)

            # Si el recorte queda fuera de la imagen, saltar
            if window.width <= 0 or window.height <= 0:
                return False

            data = src.read(
                indexes=[1, 2, 3],
                window=window,
                out_shape=(3, CHIP_PIXELS, CHIP_PIXELS),
                resampling=Resampling.bilinear,
                boundless=False
            )

            # Evitar chips vacíos
            if data.max() == 0:
                return False

            win_transform = src.window_transform(window)

            # Ajustar transform por el remuestreo a CHIP_PIXELS x CHIP_PIXELS
            scale_x = window.width / CHIP_PIXELS
            scale_y = window.height / CHIP_PIXELS
            out_transform = win_transform * Affine.scale(scale_x, scale_y)

            profile = src.profile.copy()
            profile.update(
                driver="GTiff",
                height=CHIP_PIXELS,
                width=CHIP_PIXELS,
                count=3,
                dtype=data.dtype,
                transform=out_transform,
                compress="lzw",
                tiled=True
            )

            with rasterio.open(out_tif, "w", **profile) as dst:
                dst.write(data)

    return True


# -----------------------------
# DESCARGA
# -----------------------------

def main():
    catalog = Client.open(STAC_URL)

    saved = 0
    attempts = 0
    max_attempts = N_TIFS * 5

    pbar = tqdm(total=N_TIFS, desc="GeoTIFFs guardados")

    while saved < N_TIFS and attempts < max_attempts:
        attempts += 1

        region, lon, lat = random.choice(COLOMBIA_POINTS)
        lon2, lat2 = jitter_point(lon, lat, max_jitter_km=50)

        bbox = bbox_around_point(lon2, lat2, CHIP_KM)

        try:
            item = search_best_item(catalog, bbox)

            if item is None:
                continue

            date_txt = item.datetime.strftime("%Y%m%d")
            cloud = item.properties.get("eo:cloud_cover", None)

            out_tif = OUT_DIR / f"colombia_{saved + 1:04d}_{region}_{date_txt}.tif"

            ok = save_visual_chip(item, bbox, out_tif)

            if ok:
                saved += 1
                pbar.update(1)

                print(
                    f"Guardado: {out_tif.name} | "
                    f"nube={cloud:.1f}% | "
                    f"item={item.id}"
                )

        except Exception as e:
            print(f"Error en intento {attempts}: {e}")
            continue

    pbar.close()

    print(f"\nListo. Guardados {saved} GeoTIFFs en: {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()