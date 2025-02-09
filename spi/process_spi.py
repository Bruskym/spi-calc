import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from spi.spi_calculation import calculate_spi_gamma


def load_shapefile(shapefile_path, reference_crs):
    """Carrega o shapefile e reprojeta se necessário."""
    gdf = gpd.read_file(shapefile_path)
    return gdf.to_crs(reference_crs)


def get_tif_files(tif_directory):
    """Obtém e ordena os arquivos TIF do diretório."""
    tif_files = sorted([os.path.join(tif_directory, f) for f in os.listdir(tif_directory) if f.endswith('.tif')])
    if not tif_files:
        raise FileNotFoundError("Nenhum arquivo .tif encontrado no diretório especificado.")
    return tif_files


def get_raster_metadata(raster_path, geometry):
    """Obtém metadados do primeiro raster e aplica máscara para determinar dimensões."""
    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True, filled=True)

        if len(out_image.shape) == 2:  # Garantir que sempre tenha dimensão de banda
            out_image = out_image[np.newaxis, :, :]

        out_image[out_image == src.nodata] = np.nan
        _, height, width = out_image.shape

        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": out_transform,
            "count": None,  # Será atualizado depois
            "nodata": -9999,
            "dtype": "float32"
        })

    return meta, height, width


def load_precipitation_data(tif_files, geometry, height, width):
    """Carrega e mascara os dados de precipitação de todos os anos em um array 3D."""
    precip_cube = np.full((len(tif_files), height, width), np.nan, dtype=np.float32)

    for i, tif_file in enumerate(tif_files):
        with rasterio.open(tif_file) as src:
            out_image, _ = mask(src, geometry, crop=True, filled=True)

            if len(out_image.shape) == 2:
                out_image = out_image[np.newaxis, :, :]

            out_image[out_image == src.nodata] = np.nan
            precip_cube[i] = out_image[0]

    return precip_cube


def compute_spi(precip_cube):
    """Calcula SPI para cada pixel ao longo do tempo."""
    bands, height, width = precip_cube.shape
    spi_cube = np.full_like(precip_cube, np.nan, dtype=np.float32)

    for row in range(height):
        for col in range(width):
            precip_series = precip_cube[:, row, col]
            spi_cube[:, row, col] = calculate_spi_gamma(precip_series)

    return spi_cube


def save_raster(output_tif, spi_cube, meta):
    """Salva os dados de SPI como um arquivo TIF multi-banda."""
    meta["count"] = spi_cube.shape[0]  # Atualiza o número de bandas

    with rasterio.open(output_tif, 'w', **meta) as dst:
        for i in range(spi_cube.shape[0]):
            dst.write(spi_cube[i], i + 1)

    print(f"Arquivo SPI salvo em {output_tif}")


def process_spi_region(tif_directory, shapefile_path, output_tif):
    """Pipeline principal para calcular SPI a partir dos dados TIF e shapefile."""
    tif_files = get_tif_files(tif_directory)

    with rasterio.open(tif_files[0]) as src:
        gdf = load_shapefile(shapefile_path, src.crs)

    meta, height, width = get_raster_metadata(tif_files[0], gdf.geometry)
    precip_cube = load_precipitation_data(tif_files, gdf.geometry, height, width)
    spi_cube = compute_spi(precip_cube)

    save_raster(output_tif, spi_cube, meta)