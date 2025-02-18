import matplotlib.pyplot as plt
import numpy as np

def plot_spi(df_spi):
    """
    Plota o SPI no estilo visual com áreas preenchidas para valores positivos e negativos.
    
    Args:
        df_spi (DataFrame): DataFrame com o índice do ano/banda e a coluna 'SPI'.
    """
    years = df_spi.index.astype(str)  # Converter os índices para strings
    spi_values = df_spi['SPI']

    # Dividindo os valores positivos e negativos
    positive_spi = np.maximum(spi_values, 0)
    negative_spi = np.minimum(spi_values, 0)

    # Criando a figura
    plt.figure(figsize=(12, 6))

    # Ajuste para garantir que as barras ocupem todo o espaço do ano
    bar_width = 1.0  # Largura da barra
    plt.bar(years, positive_spi, color='blue', label='SPI > 0', width=bar_width, align='center')
    plt.bar(years, negative_spi, color='red', label='SPI < 0', width=bar_width, align='center')

    # Adicionando títulos e rótulos
    plt.title('Índice Padronizado de Precipitação (SPI)', fontsize=14)
    plt.xlabel('Anos/Bandas', fontsize=12)
    plt.ylabel('SPI', fontsize=12)

    # Exibir todos os rótulos no eixo X
    plt.xticks(rotation=45, ha='right')  # Rotaciona os rótulos para melhor visualização

    # Grade e legenda
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend()

    # Melhor layout para evitar cortes
    plt.tight_layout()

    # Exibindo o gráfico
    plt.show()
