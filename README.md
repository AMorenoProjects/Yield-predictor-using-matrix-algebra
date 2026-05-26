# 📈 Predictor de Rendimientos del S&P 500 Usando Álgebra Matricial

Este proyecto implementa un **modelo de Regresión Lineal desde cero** utilizando operaciones puras de álgebra lineal y matricial con Numpy y Pandas. En lugar de depender de librerías de alto nivel para machine learning (como `scikit-learn`), el motor matemático de predicción se resuelve de manera exacta mediante la **Ecuación Normal**.

Además, el proyecto contiene una demostración práctica del fenómeno matemático de la **multicolinealidad** y ejecuta un **backtesting vectorizado** para comparar la estrategia predictiva frente al clásico *Buy & Hold* (Comprar y Mantener) en el índice S&P 500 (`SPY`).

---

## 📐 Fundamento Matemático

El núcleo del predictor se basa en resolver el problema de mínimos cuadrados ordinarios (OLS) de manera analítica mediante representaciones matriciales:

### 1. La Ecuación Normal
Para encontrar el vector de coeficientes óptimos $\beta$ que minimiza la suma de los residuos al cuadrado, aplicamos la solución cerrada del cálculo matricial:

$$\beta = (X^T X)^{-1} X^T y$$

Donde:
* **$\beta$**: Es el vector de pesos o coeficientes de tamaño $(D + 1) \times 1$.
* **$X$**: Es la **Matriz de Diseño** de tamaño $N \times (D+1)$. Contiene las características del mercado (lags, medias móviles, volatilidad) y una primera columna de unos ($1$) para modelar el término de intercepto ($\beta_0$).
* **$y$**: Es el vector objetivo de tamaño $N \times 1$, que representa los rendimientos reales observados al día siguiente ($t+1$).

### 2. Proyección y Predicción (Hat Matrix)
Una vez obtenidos los coeficientes óptimos $\beta$, proyectamos las nuevas estimaciones del mercado ($\hat{y}$) aplicando una simple multiplicación de matrices:

$$\hat{y} = X \beta$$

> [!NOTE]
> La inclusión de una columna de unos en la matriz $X$ es crucial. Sin ella, obligaríamos a la recta o hiperplano de regresión a pasar obligatoriamente por el origen $(0,0)$, lo cual sesgaría severamente nuestras predicciones de mercado.

---

## ⚡ Ingeniería de Características (Matriz $X$)

Para alimentar el modelo, calculamos métricas estadísticas sobre el retorno porcentual diario del S&P 500 (`SPY`). El vector de características para cada día $t$ está compuesto por:

| Variable | Nombre en Código | Descripción Matemática |
| :--- | :--- | :--- |
| **Intercepto** | `unos` | Columna de constantes ($1$) para estimar el sesgo de retorno básico ($\beta_0$). |
| **Lag 1** | `Ret_Lag1` | Rendimiento porcentual del mercado del día anterior ($R_{t-1}$). |
| **Lag 2** | `Ret_Lag2` | Rendimiento porcentual del mercado de hace dos días ($R_{t-2}$). |
| **Media Móvil** | `SMA_10_Ret` | Media móvil simple de los rendimientos de los últimos 10 días ($\frac{1}{10} \sum_{i=0}^{9} R_{t-i}$). |
| **Volatilidad** | `Vol_10` | Desviación estándar móvil de los rendimientos de los últimos 10 días ($\sigma_{10}(R)$). |

El objetivo a predecir (`Target`) es el retorno real del día siguiente: $y_t = R_{t+1}$.

---

## ⚠️ Demostración Práctica: El Peligro de la Multicolinealidad

El álgebra lineal impone una restricción estricta: para poder invertir la matriz $(X^T X)$, las columnas de la matriz de diseño $X$ deben ser **linealmente independientes** (la matriz debe ser de rango completo). 

El script `predictor.py` realiza un experimento controlado que fuerza este colapso:
1. Introduce una columna redundante que es exactamente el doble de `Ret_Lag1` ($2 \times R_{t-1}$).
2. Intenta recalcular la ecuación normal.
3. El motor de álgebra lineal de Numpy detecta una **matriz singular** (cuyo determinante es cero y carece de inversa).
4. El programa captura el error `numpy.linalg.LinAlgError` y valida de forma práctica la teoría matemática subyacente.

> [!WARNING]
> En aplicaciones del mundo real, la correlación perfecta o casi perfecta entre variables predictoras (como usar la temperatura en Celsius y en Fahrenheit simultáneamente) provoca inestabilidad numérica extrema o caídas del sistema como la que este script demuestra de forma exitosa.

---

## 📈 Regla de Trading y Backtesting Vectorizado

El script evalúa la utilidad del modelo en un escenario de trading cuantitativo histórico mediante un **backtest vectorizado**:

1. **Generación de Señales**: 
   $$\text{Señal}_t = \begin{cases} 1 & \text{si } \hat{y}_{t} > 0 \quad \text{(Predice retorno positivo: Compra)} \\ 0 & \text{si } \hat{y}_{t} \le 0 \quad \text{(Predice retorno negativo/nulo: Liquidez)} \end{cases}$$
2. **Cálculo de Retornos**: Multiplicamos la señal del día de hoy ($1$ o $0$) por el rendimiento real que el mercado experimenta mañana ($y$).
3. **Efecto Compuesto (Retorno Acumulado)**:
   $$\text{Retorno Acumulado}_t = \prod_{i=1}^{t} \left(1 + \frac{R_{\text{estrategia}, i}}{100}\right) - 1$$

> [!TIP]
> Al evitar el uso de bucles `for` para simular las operaciones día a día, el backtest vectorizado se ejecuta instantáneamente en microsegundos, lo que permite iteraciones ultra-rápidas en el diseño de modelos.

---

## 🚀 Instalación y Uso

### Requisitos Previos
* **Python**: `>=3.12`
* Se recomienda utilizar [uv](https://github.com/astral-sh/uv) como gestor de paquetes y entornos virtuales por su extrema velocidad y consistencia.

### Paso 1: Clonar el Repositorio
```bash
git clone https://github.com/tu-usuario/Yield-predictor-using-matrix-algebra.git
cd Yield-predictor-using-matrix-algebra
```

### Paso 2: Instalación de Dependencias

#### Opción A: Usando `uv` (Recomendado)
Sincroniza el entorno virtual automáticamente a partir de los archivos de configuración:
```bash
uv sync
```

#### Opción B: Usando `pip` clásico
Si prefieres usar `pip`, instala las librerías necesarias directamente en tu entorno:
```bash
pip install numpy pandas yfinance
```

### Paso 3: Ejecución del Script

#### Con `uv`:
```bash
uv run predictor.py
```

#### Con Python estándar:
```bash
python predictor.py
```

---

## 📊 Ejemplo de Salida en Terminal

Al ejecutar el script `predictor.py`, verás un reporte detallado similar a este en tu terminal:

```text
1. Descargando datos del S&P 500 (SPY)...

2. Entrenando el modelo con Álgebra Lineal...

--- PESOS DESCUBIERTOS (Beta) ---
Beta 0 (Intercepto) : 0.0906
Beta 1 (Ret_Lag1)   : 0.1371
Beta 2 (Ret_Lag2)   : 0.0148
Beta 3 (SMA_10)     : -0.2306
Beta 4 (Vol_10)     : -0.0285

3. Comprobando la teoría: Forzando un colapso matricial...
Intentando invertir la matriz singular (X^T * X)...

[!] CRASH EXITOSO: La teoría es correcta.
El motor matricial de Numpy se negó a resolverlo lanzando el error: Singular matrix

4. Ejecutando Backtest Vectorizado...

--- RESULTADOS DEL BACKTEST (2020-2025) ---
Rendimiento Buy & Hold (SPY) : 52.96%
Rendimiento Estrategia       : 23.45%
```

---

## 🛠️ Tecnologías Empleadas

* **[Python 3.12](https://www.python.org/)** - Lenguaje base.
* **[Numpy](https://numpy.org/)** - Computación científica y operaciones de álgebra matricial de alto rendimiento.
* **[Pandas](https://pandas.pydata.org/)** - Estructuración de series temporales e ingeniería de características predictivas.
* **[Yahoo Finance API (yfinance)](https://github.com/ranaroussi/yfinance)** - Descarga automatizada de históricos de precios de mercado.