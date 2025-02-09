import os
from pathlib import Path
from spi.process_spi import process_spi_region

if __name__ == "__main__":
    tif_dir = Path(__file__).parent / "data_tif"
    shapefile_dir = Path(__file__).parent / "data_shapefile"
    output_dir = Path(__file__).parent / "tif_output"

    # Buscar qualquer arquivo que termine com .shp
    shapefiles = list(shapefile_dir.glob("*.shp"))

    if not shapefiles:
        raise FileNotFoundError("Nenhum arquivo .shp encontrado no diretório data_shapefile.")
    
    shapefile = shapefiles[0]  # Usa o primeiro arquivo encontrado

    # Criar diretório de saída se não existir
    os.makedirs(output_dir, exist_ok=True)

    output_tif = os.path.join(output_dir, "spi_multibanda.tif")

    process_spi_region(tif_dir, shapefile, output_tif)
