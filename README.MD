# Cálculo de SPI a partir de Dados Raster e Shapefile

## Descrição
Este projeto tem como objetivo calcular o Índice Padronizado de Precipitação (SPI) utilizando dados raster de precipitação e um shapefile de região de interesse. O SPI é calculado para cada pixel da região de recorte com base em uma distribuição gama ajustada aos dados históricos de precipitação.

## Funcionalidades
- Leitura e processamento de arquivos raster (.tif) de precipitação.
- Recorte dos dados raster com base em um shapefile (.shp).
- Cálculo do SPI utilizando distribuição gama.
- Geração de um arquivo raster (.tif) contendo o SPI em formato multibanda.

## Estrutura do Projeto
```
├── data_tif/               # Diretório com os arquivos raster de entrada
├── data_shapefile/         # Diretório com o shapefile da região de interesse
├── tif_output/             # Diretório de saída dos resultados
├── spi/
│   ├── __init__.py
│   ├── process_spi.py      # Módulo principal do processamento
│   ├── spi_calculation.py  # Implementação do cálculo de SPI
├── app.py                 # Script principal de execução
└── README.md               # Documentação do projeto
```

## Instalação
Certifique-se de ter o Python 3.8 ou superior instalado. Instale as dependências necessárias com:
```bash
pip install -r requirements.txt
```

## Uso
Para executar o cálculo do SPI:
```bash
python app.py
```
### Parâmetros de Entrada
- **Arquivos raster (.tif)**: Devem estar localizados no diretório `data_tif/`.
- **Shapefile (.shp)**: Deve estar localizado no diretório `data_shapefile/`. O script automaticamente usa o primeiro `.shp` encontrado.

### Saída
- O resultado do SPI é salvo em `tif_output/spi_multibanda.tif`.

## Explicação do Algoritmo
1. **Leitura dos Dados**: Os arquivos raster são lidos e recortados com base no shapefile.
2. **Cálculo do SPI**:
   - O algoritmo ajusta uma distribuição gama para cada pixel ao longo do tempo.
   - Se a quantidade de valores válidos for insuficiente, o SPI é definido como NaN.
   - Para regiões sem precipitação, o SPI é fixado em -3.0.
3. **Geração do Raster de SPI**: O SPI é salvo como um arquivo raster multibanda.

## Dependências
- `numpy`
- `geopandas`
- `rasterio`
- `scipy`


