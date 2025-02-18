import os
from pathlib import Path
import pandas as pd
from data_analysis.extract_spi import extract_spi_to_csv
from data_analysis.plot import plot_spi

if __name__ == "__main__":
    # Definindo o diretório de entrada e saída
    input_dir = Path(__file__).parent / "tif_output"

    tifs = list(input_dir.glob("*.tif"))

    if not tifs:
        raise FileNotFoundError("Nenhum arquivo .tif encontrado no diretório tif_output.")
    
    input_tif = tifs[0]

    output_dir = Path(__file__).parent / "csv_output"
    output_dir.mkdir(exist_ok=True)
    
    output_file = output_dir / "relatorio_spi_coordenada.csv"
    longitude = -36.0348
    latitude = -8.3639

    # Gera o CSV com os dados SPI
    extract_spi_to_csv(input_tif, longitude, latitude, output_file)

    df_spi = pd.read_csv(output_file)
    df_spi = df_spi.drop(columns=["Longitude", "Latitude"])

    if df_spi['Banda/Ano'].str.startswith("Ano ").any():
        df_spi['Banda/Ano'] = df_spi['Banda/Ano'].str.replace("Ano ", "").astype(int)
    else:
        df_spi['Banda/Ano'] = df_spi['Banda/Ano'].astype(str)

    df_spi.set_index("Banda/Ano", inplace=True)

    plot_spi(df_spi)
