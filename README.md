# Temporal Fusion Transformer (TFT)

## Descripción general

Este proyecto implementa un modelo **Temporal Fusion Transformer (TFT)** para predecir movimientos de precios en mercados financieros en **timeframes intradía** (barras de 5, 15 y 60 minutos).  
El objetivo es estimar la distribución futura de los retornos logarítmicos de un activo a distintos horizontes de tiempo y comparar el desempeño del TFT frente a enfoques más simples.

El trabajo principal está contenido en el cuaderno:

- `Proyecto_Final_DL.ipynb`

---

## Objetivos

- **Objetivo general:**  
  Implementar un modelo TFT para predecir movimientos de precios intradía y analizar su capacidad para capturar dependencias temporales complejas.

- **Objetivos específicos:**
  - Construir una serie de tiempo limpia a partir de datos históricos de un activo (acción, ETF o índice).
  - Calcular **retornos logarítmicos** y otros *features* relevantes para el trading intradía.
  - Definir correctamente un `TimeSeriesDataSet` de `pytorch-forecasting` adecuado al problema financiero.
  - Entrenar y evaluar un TFT usando una **pérdida cuantílica**, obteniendo predicciones probabilísticas.
  - Discutir fortalezas, limitaciones y posibles mejoras del modelo.

---

## Datos utilizados

- **Fuente:** APIs financieras como Yahoo Finance y/o Alpha Vantage.
- **Activo:** Un único instrumento financiero (stock, ETF o índice).
- **Periodo:** Historial de precios intradía (ventana de varios meses/años según disponibilidad).
- **Frecuencia:** Barras OHLCV con resoluciones de **5, 15 y 60 minutos**.
- **Variables originales:**
  - `Open`, `High`, `Low`, `Close`, `Volume`
  - Marca de tiempo (`Datetime`) usada como índice de la serie.

---

## Preparación de datos

En el cuaderno se realiza una limpieza profunda para obtener una serie utilizable por el TFT:

1. **Limpieza básica**
   - Conversión de fechas a `DatetimeIndex`.
   - Ordenamiento temporal y eliminación de duplicados.
   - Manejo de huecos de mercado (fines de semana, festivos, horarios fuera de sesión).

2. **Cálculo del retorno logarítmico**
   - Definición del *target* principal:
     - `log_return = log(Close_t / Close_{t-1})`
   - Generación de retornos futuros a distintos horizontes para predicción multi-horizonte, por ejemplo:
     - `target_return_1`, `target_return_3`, `target_return_6`, `target_return_12`, `target_return_24`
     - Cada uno representa el retorno acumulado a 1, 3, 6, 12 y 24 pasos hacia adelante (para barras de 5 minutos).

3. **Features adicionales (opcionales pero útiles)**
   - **Volatilidad intradía**
     - `volatility_5`: desviación estándar móvil de `log_return` en una ventana de 5 barras.
     - `volatility_20`: desviación estándar móvil en una ventana de 20 barras.
   - **Indicador técnico**
     - `RSI_14`: Índice de Fuerza Relativa de 14 periodos sobre el precio de cierre.
   - Estas variables buscan aportar información sobre el nivel de riesgo y el momentum del mercado.

4. **Features temporales (conocidas hacia el futuro)**
   - `hour`: hora del día.
   - `day_of_week`: día de la semana.
   - `is_london_session`: indicador binario de si el punto pertenece a la sesión europea.
   - `is_ny_session`: indicador binario de si el punto pertenece a la sesión estadounidense.
   - Estas variables ayudan al modelo a aprender patrones intradía y semanales.

5. **Tratamiento final**
   - Alineación de todas las series.
   - Eliminación de filas con valores faltantes generados por las transformaciones.

El resultado de este proceso se almacena en un dataframe final, por ejemplo `data_set_listo`.

---

## Definición del conjunto de series temporales (TimeSeriesDataSet)

Se utiliza `TimeSeriesDataSet` de `pytorch-forecasting` para estructurar los datos de entrada del TFT.

Configuración conceptual:

- **Datos de entrada:** `data_set_listo` con índice temporal y columnas de features.
- **`time_idx`:** índice entero que representa el orden temporal de cada observación.
- **`target`:** retorno logarítmico futuro a predecir (por ejemplo `target_return_1` o un horizonte seleccionado).
- **`group_ids`:** identificador de serie. En este caso puede ser un único grupo (por ejemplo, el símbolo del activo).
- **Longitudes:**
  - `max_encoder_length`: número máximo de pasos históricos que ve el modelo (ventana de contexto).
  - `max_prediction_length`: número de pasos futuros que debe predecir el TFT.
- **Variables:**
  - `time_varying_known_reals`: variables numéricas conocidas hacia el futuro (por ejemplo `hour`, `day_of_week`).
  - `time_varying_unknown_reals`: variables numéricas desconocidas en el futuro (por ejemplo `log_return`, volatilidades, RSI).
  - `static_reals` o `static_categoricals`: características que describen al activo y no cambian en el tiempo (por ejemplo tipo de activo, zona horaria).

La pérdida utilizada es típicamente `QuantileLoss`, lo que permite obtener **predicciones de cuantiles** (p. ej. 0.1, 0.5, 0.9) y modelar la distribución de posibles retornos.

---

## Modelo y arquitectura

- **Modelo:** `TemporalFusionTransformer` de `pytorch-forecasting`.
- **Backend:** `PyTorch Lightning`.
- **Características clave del TFT:**
  - Manejo conjunto de variables estáticas, conocidas y desconocidas en el tiempo.
  - Atención temporal para seleccionar automáticamente los momentos históricos más relevantes.
  - Predicción multi-horizonte con cuantiles de probabilidad.

En el cuaderno se ajustan hiperparámetros como:

- Dimensión del *hidden state*.
- Número de capas LSTM en el encoder/decoder.
- Número de cabezas de atención.
- Cuantiles a predecir.
- Tamaño de lote y número de épocas.

---

## Entrenamiento y evaluación

Pasos principales:

1. División temporal del dataset en entrenamiento, validación y prueba.
2. Definición de `TimeSeriesDataSet` para cada partición.
3. Creación de `DataLoaders` para PyTorch.
4. Entrenamiento del TFT con `QuantileLoss`.
5. Evaluación mediante:
   - Pérdida cuantílica en validación y prueba.
   - Métricas adicionales como MAE o RMSE sobre el retorno medio.
   - Gráficas de:
     - Retornos reales vs. predichos.
     - Bandas de cuantiles (incertidumbre) a distintos horizontes.

La interpretación se centra en qué tan bien el modelo captura la dirección del movimiento y la dispersión de la distribución de retornos futuros.

---

## Resultados y discusión

En la sección final del cuaderno se analizan:

- Comportamiento del TFT en distintos horizontes de predicción.
- Situaciones donde el modelo acierta la dirección y la magnitud del movimiento.
- Casos donde el modelo falla, relacionándolos con:
  - Picos de volatilidad.
  - Eventos de mercado inesperados.
- Valor añadido de usar TFT frente a modelos más simples (por ejemplo, modelos lineales o LSTM sin atención).

También se discuten limitaciones:

- Uso de un único activo y una única frecuencia temporal.
- Falta de variables macroeconómicas o de *order book*.
- Coste computacional del TFT frente a modelos más ligeros.

---

## Requisitos

Versión recomendada:

- Python 3.9 o superior.

Bibliotecas principales:

- `pandas`, `numpy`
- `matplotlib` o `plotly`
- `pandas_ta`
- `yfinance` o cliente de Alpha Vantage
- `torch`, `pytorch-lightning`
- `pytorch-forecasting`
- `scikit-learn` (para métricas adicionales)

---

## Cómo usar el cuaderno

1. Clonar o descargar el repositorio del proyecto.
2. Crear un entorno virtual e instalar las dependencias necesarias.
3. Abrir `Proyecto_Final_DL.ipynb` en Jupyter Lab, VS Code o Google Colab.
4. Ejecutar las celdas en orden:
   - Descarga y limpieza de datos.
   - Ingeniería de características.
   - Definición de `TimeSeriesDataSet`.
   - Entrenamiento del TFT.
   - Evaluación y visualización de resultados.

---

## Trabajo futuro

- Incorporar múltiples activos y tratar el problema como una serie multivariante.
- Explorar diferentes horizontes de predicción y funciones de pérdida.
- Comparar empiricamente TFT con otros modelos de *Deep Learning* para series temporales.
- Integrar el modelo en un pipeline de backtesting para evaluar estrategias de trading basadas en las predicciones de cuantiles.

