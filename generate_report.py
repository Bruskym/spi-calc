import os
from pathlib import Path
import pandas as pd
from data_analysis.extract_spi import extract_spi_to_csv
from data_analysis.input.extract_prec import extract_precipitation_to_csv
from data_analysis.plot import plot_spi

if __name__ == "__main__":
    # Diretórios de entrada e saída
    spi_input_dir = Path(__file__).parent / "tif_output"
    precip_input_dir = Path(__file__).parent / "data_tif"

    # Arquivos SPI
    spi_tifs = list(spi_input_dir.glob("*.tif"))
    if not spi_tifs:
        raise FileNotFoundError("Nenhum arquivo .tif encontrado no diretório tif_output.")
    spi_input_tif = spi_tifs[0]

    # Arquivos de precipitação
    precip_tifs = list(precip_input_dir.glob("*.tif"))
    if not precip_tifs:
        raise FileNotFoundError("Nenhum arquivo .tif encontrado no diretório data_tif.")
    
    # Saída de CSV
    output_dir = Path(__file__).parent / "csv_output"
    output_dir.mkdir(exist_ok=True)
    
    spi_output_file = output_dir / "relatorio_spi_coordenada.csv"
    precip_output_file = output_dir / "relatorio_precip_coordenada.csv"
    
    # Coordenadas
    longitude = -36.0301
    latitude = -8.3664

    # Gera o CSV com os dados SPI
    extract_spi_to_csv(spi_input_tif, longitude, latitude, spi_output_file, 1981)

    # Gera o CSV com os dados de precipitação
    extract_precipitation_to_csv(precip_input_dir, longitude, latitude, precip_output_file)

    # Processamento e plotagem dos dados SPI
    df_spi = pd.read_csv(spi_output_file)
    df_spi = df_spi.drop(columns=["Longitude", "Latitude"])

    if df_spi['Banda/Ano'].str.startswith("Ano ").any():
        df_spi['Banda/Ano'] = df_spi['Banda/Ano'].str.replace("Ano ", "").astype(int)
    else:
        df_spi['Banda/Ano'] = df_spi['Banda/Ano'].astype(str)

    df_spi.set_index("Banda/Ano", inplace=True)

    plot_spi(df_spi)

    # Processa os dados de precipitação (se necessário)
    df_precip = pd.read_csv(precip_output_file)
    print("Dados de precipitação extraídos com sucesso:")
    print(df_precip.head())
