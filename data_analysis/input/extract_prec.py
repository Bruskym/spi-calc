import os
import csv
import rasterio

from pathlib import Path

def get_precipitation_value(tif_path, lon, lat):
    """
    Extrai o valor de precipitação para um ponto específico em um arquivo TIF.
    
    Args:
        tif_path (str): Caminho do arquivo TIF.
        lon (float): Longitude do ponto.
        lat (float): Latitude do ponto.
    
    Returns:
        float: Valor da precipitação no ponto especificado.
    """
    with rasterio.open(tif_path) as src:
        row, col = src.index(lon, lat)
        return src.read(1)[row, col]


def extract_precipitation_to_csv(tif_directory, lon, lat, output_csv):
    """
    Extrai valores de precipitação de arquivos TIF e salva em um arquivo CSV.
    
    Args:
        tif_directory (str): Diretório contendo os arquivos TIF anuais.
        lon (float): Longitude do ponto.
        lat (float): Latitude do ponto.
        output_csv (str): Caminho para salvar o arquivo CSV.
    """
    data = []
    for filename in sorted(os.listdir(tif_directory)):
        if filename.endswith('.tif'):
            year = int(filename.split('.')[2])
            tif_path = os.path.join(tif_directory, filename)

            precip_value = get_precipitation_value(tif_path, lon, lat)
            data.append((year, lon, lat, precip_value))

    with open(output_csv, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Ano', 'Longitude', 'Latitude', 'Precipitacao'])
        writer.writerows(data)

    print(f"Dados exportados para {output_csv}")
