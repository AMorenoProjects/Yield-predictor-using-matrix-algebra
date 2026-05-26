# Importados las librerías
import numpy as np
import pandas as pd
import yfinance as yf

# Descargamos los datos del sp500
print("1. Descargando datos del S&P 500 (SPY)...")
df = yf.download('SPY', start='2020-01-01', end='2024-01-01', progress=False)

# SOLUCIÓN YFINANCE: 'Close' ya está ajustado. Extraemos el primer nivel del MultiIndex.
df = df['Close'].copy()
df.columns = ['Close']

# Calcular el retorno diario (%)
df['Return'] = df['Close'].pct_change() * 100

# Crear las variables predictivas (La Matriz X)
df['Ret_Lag1'] = df['Return'].shift(1)               
df['Ret_Lag2'] = df['Return'].shift(2)               
df['SMA_10_Ret'] = df['Return'].rolling(10).mean()   
df['Vol_10'] = df['Return'].rolling(10).std()        

# Crear la variable objetivo (El Vector y)
df['Target'] = df['Return'].shift(-1)                

# Limpiar filas con datos faltantes
df = df.dropna()

# Extraer datos puros a arrays de Numpy
X_raw = df[['Ret_Lag1', 'Ret_Lag2', 'SMA_10_Ret', 'Vol_10']].values
y = df['Target'].values

# Columna de 1s para el intercepto
unos = np.ones((X_raw.shape[0], 1))
X = np.hstack((unos, X_raw))


# Motor matemático
print("\n2. Entrenando el modelo con Álgebra Lineal...")

# Aplicamos la fórmula exacta: Beta = (X^T * X)^-1 * X^T * y
# En Python, '@' es el operador de multiplicación de matrices.
X_T = X.T
beta = np.linalg.inv(X_T @ X) @ X_T @ y

print("\n--- PESOS DESCUBIERTOS (Beta) ---")
print(f"Beta 0 (Intercepto) : {beta[0]:.4f}")
print(f"Beta 1 (Ret_Lag1)   : {beta[1]:.4f}")
print(f"Beta 2 (Ret_Lag2)   : {beta[2]:.4f}")
print(f"Beta 3 (SMA_10)     : {beta[3]:.4f}")
print(f"Beta 4 (Vol_10)     : {beta[4]:.4f}")

# Proyectar las predicciones (Hat Matrix en acción: y_pred = X * Beta)
y_pred = X @ beta


# Forzando el error de colinelidad
print("\n3. Comprobando la teoría: Forzando un colapso matricial...")
try:
    # Creamos una columna "basura" que es exactamente el doble de Lag1
    # Esto viola la regla de independencia lineal de las columnas
    columna_clon = X[:, 1] * 2  
    X_corrupto = np.column_stack((X, columna_clon))
    
    print("Intentando invertir la matriz singular (X^T * X)...")
    beta_crash = np.linalg.inv(X_corrupto.T @ X_corrupto) @ X_corrupto.T @ y

except np.linalg.LinAlgError as e:
    print(f"\n[!] CRASH EXITOSO: La teoría es correcta.")
    print(f"El motor matricial de Numpy se negó a resolverlo lanzando el error: {e}")
    

# Backtesting
print("\n4. Ejecutando Backtest Vectorizado...")

# Generación de Señales
# Si el modelo predice que el retorno de mañana será positivo (> 0), compramos (1).
# Si predice que será negativo o cero, nos quedamos en liquidez (0).
senales = np.where(y_pred > 0, 1, 0)

# Cálculo de Retornos de la Estrategia
retornos_estrategia = senales * y

# Empaquetado en Pandas para análisis
df_resultados = pd.DataFrame({
    'Retorno_Mercado': y,
    'Retorno_Estrategia': retornos_estrategia
})

# Cálculo del Retorno Acumulado (Compound Return)
df_resultados['Acumulado_Mercado'] = (1 + df_resultados['Retorno_Mercado'] / 100).cumprod()
df_resultados['Acumulado_Estrategia'] = (1 + df_resultados['Retorno_Estrategia'] / 100).cumprod()

rendimiento_mercado = (df_resultados['Acumulado_Mercado'].iloc[-1] - 1) * 100
rendimiento_estrategia = (df_resultados['Acumulado_Estrategia'].iloc[-1] - 1) * 100

print("\n--- RESULTADOS DEL BACKTEST (2020-2025) ---")
print(f"Rendimiento Buy & Hold (SPY) : {rendimiento_mercado:.2f}%")
print(f"Rendimiento Estrategia       : {rendimiento_estrategia:.2f}%")