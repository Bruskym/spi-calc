import os
import csv
import rasterio
import pandas as pd
from pathlib import Path

def get_spi_value(tif_path, lon, lat):
    """
    Extrai os valores de SPI para um ponto específico em um arquivo TIF multibanda.
    
    Args:
        tif_path (str): Caminho do arquivo TIF multibanda.
        lon (float): Longitude do ponto.
        lat (float): Latitude do ponto.
    
    Returns:
        list: Valores de SPI para cada banda no ponto especificado.
    """
    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)
        spi_values = src.read()[:, row, col]  # Lê todas as bandas
        return spi_values

# Função para extrair os dados de SPI e salvar em CSV
def extract_spi_to_csv(tif_path, lon, lat, output_file, initial_year=None):
    """
    Extrai valores SPI para uma coordenada específica de um arquivo TIFF 
    e os exporta para um CSV no formato esperado.

    Args:
        tif_path (str): Caminho do arquivo TIFF.
        lon (float): Longitude da coordenada.
        lat (float): Latitude da coordenada.
        output_file (str): Caminho de saída do arquivo CSV.
        initial_year (int, opcional): Ano inicial para nomear as bandas. 
                                     Se não for passado, mantém a nomenclatura padrão.
    """
    import rasterio
    import pandas as pd

    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)

        # Lista para armazenar os valores extraídos
        data = []

        # Iterar por todas as bandas do TIFF
        for band in range(1, src.count + 1):
            # Lê o valor da banda atual para a coordenada especificada
            spi_value = src.read(band)[row, col]

            # Definir o nome da banda baseado no ano inicial, se fornecido
            if initial_year:
                year = initial_year + band - 1
                band_name = f"Ano {year}"
            else:
                band_name = f"Banda {band}"

            # Adiciona uma entrada com o nome da banda e o valor SPI
            data.append({
                "Banda/Ano": band_name,
                "SPI": spi_value,
                "Longitude": lon,
                "Latitude": lat
            })

    # Criar DataFrame com os dados extraídos
    df = pd.DataFrame(data)

    # Exporta o DataFrame para o arquivo CSV
    df.to_csv(output_file, index=False)

    print(f"Dados exportados para {output_file}")
