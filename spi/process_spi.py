import os
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from rasterio.features import geometry_mask
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

def get_raster_metadata_and_mask(raster_path, geometry):
    """Obtém metadados e máscara binária do shapefile."""
    with rasterio.open(raster_path) as src:
        # Aplicar máscara para obter transformação e dimensões
        out_image, out_transform = mask(src, geometry, crop=True, filled=True)
        
        if len(out_image.shape) == 2:
            out_image = out_image[np.newaxis, :, :]
        
        # Criar máscara binária (1 = dentro do shapefile, 0 = fora)
        binary_mask = geometry_mask(
            geometry,
            out_shape=out_image.shape[1:],
            transform=out_transform,
            invert=True
        )
        
        out_image[out_image == src.nodata] = np.nan
        _, height, width = out_image.shape

        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "transform": out_transform,
            "count": None,
            "nodata": np.nan,
            "dtype": "float32"
        })

    return meta, height, width, binary_mask

def load_precipitation_data(tif_files, geometry, height, width):
    """Carrega e mascara os dados de precipitação."""
    precip_cube = np.full((len(tif_files), height, width), np.nan, dtype=np.float32)

    for i, tif_file in enumerate(tif_files):
        with rasterio.open(tif_file) as src:
            out_image, _ = mask(src, geometry, crop=True, filled=True)
            
            if len(out_image.shape) == 2:
                out_image = out_image[np.newaxis, :, :]
            
            out_image[out_image == src.nodata] = np.nan
            precip_cube[i] = out_image[0]

    return precip_cube

def compute_spi_with_mask(precip_cube, binary_mask):
    """Calcula SPI apenas para pixels dentro do shapefile."""
    bands, height, width = precip_cube.shape
    spi_cube = np.full_like(precip_cube, np.nan, dtype=np.float32)

    for row in range(height):
        for col in range(width):
            if binary_mask[row, col]:  # Processa apenas pixels dentro do shapefile
                precip_series = precip_cube[:, row, col]
                spi_cube[:, row, col] = calculate_spi_gamma(precip_series)

    return spi_cube

def save_raster(output_tif, spi_cube, meta):
    """Salva o raster de SPI."""
    meta["count"] = spi_cube.shape[0]

    with rasterio.open(output_tif, 'w', **meta) as dst:
        for i in range(spi_cube.shape[0]):
            dst.write(spi_cube[i], i + 1)

    print(f"Arquivo SPI salvo em {output_tif}")

def process_spi_region(tif_directory, shapefile_path, output_tif):
    """Pipeline principal com máscara espacial."""
    tif_files = get_tif_files(tif_directory)
    
    with rasterio.open(tif_files[0]) as src:
        gdf = load_shapefile(shapefile_path, src.crs)

    # Obter metadados e máscara binária
    meta, height, width, binary_mask = get_raster_metadata_and_mask(tif_files[0], gdf.geometry)
    
    # Carregar dados e calcular SPI
    precip_cube = load_precipitation_data(tif_files, gdf.geometry, height, width)
    spi_cube = compute_spi_with_mask(precip_cube, binary_mask)
    
    # Salvar resultado
    save_raster(output_tif, spi_cube, meta)