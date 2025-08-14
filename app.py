import streamlit as st
import requests, csv, json, urllib
import pandas as pd
import numpy as np
import time
from fake_useragent import UserAgent
from datetime import datetime
import plotly.graph_objects as go
import ccxt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Configuración de la página
st.set_page_config(page_title="BTC Fear & Greed Strategy", layout="wide", page_icon="📈")

# Título y descripción
st.title("📈 Bitcoin Trading Strategy with Fear & Greed Index")
st.markdown("""
Esta aplicación implementa una estrategia de trading para Bitcoin basada en el Índice de Miedo y Codicia (Fear & Greed Index).
""")

# Sidebar con parámetros configurables
with st.sidebar:
    st.header("⚙️ Parámetros de la Estrategia")
    initial_capital = st.number_input("Capital Inicial (USDT)", min_value=100, value=10000, step=1000)
    buy_threshold = st.slider("Umbral de Compra (F&G Index)", min_value=0, max_value=50, value=25)
    sell_threshold = st.slider("Umbral de Venta (F&G Index)", min_value=50, max_value=100, value=75)
    buy_fraction = st.slider("Fracción de compra", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    sell_fraction = st.slider("Fracción de venta", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    
    st.markdown("---")
    st.markdown("ℹ️ **Explicación:**")
    st.markdown("""
    - **Compra:** Cuando el F&G Index < Umbral de Compra
    - **Venta:** Cuando el F&G Index > Umbral de Venta
    - Las fracciones determinan qué porcentaje del capital/BTC se usa en cada operación
    """)

# Función para cargar datos
@st.cache_data(ttl=3600)  # Cache por 1 hora
def load_data():
    # FearGreedIndex
    response = requests.get("https://api.alternative.me/fng/?limit=0&format=json").json()['data']
    fear = pd.DataFrame(response, columns=['timestamp', 'value'])
    fear = fear.set_index('timestamp')  
    fear.index = pd.to_datetime(fear.index, unit='s')  
    fear = fear.rename(columns={"value": "FearGreedIndex"})
    fear['FearGreedIndex'] = fear['FearGreedIndex'].astype(float)
    
    # BTC-USD Data
    exchange = ccxt.binance()
    symbol = 'BTC/USDT'  # BTC contra dólar en Binance
    timeframe = '1d'     # Velas diarias      
    since = exchange.parse8601('2023-01-01T00:00:00Z')  # Fecha de inicio
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=1000)
    data = pd.DataFrame(ohlcv, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume'])
    data['Date'] = pd.to_datetime(data['Timestamp'], unit='ms')
    data = data.set_index('Date')
    data.drop(columns=['Timestamp'], inplace=True)
    
    # Unir datos y manejar NaN
    data_merged = data.join(fear, how='left')
    data_merged['FearGreedIndex'] = data_merged['FearGreedIndex'].ffill()
    return data_merged

# Cargar datos con spinner
with st.spinner('Cargando datos...'):
    df = load_data()

# Generar señales
df['Signal'] = None
df.loc[df['FearGreedIndex'] < buy_threshold, 'Signal'] = 'BUY'  
df.loc[df['FearGreedIndex'] > sell_threshold, 'Signal'] = 'SELL'
df['Signal'] = df['Signal'].shift(1)  # Evitar look-ahead bias

# Función de estrategia DCA
def dca_strategy(df, initial_capital=10000, buy_fraction=0.5, sell_fraction=0.5):
    capital = initial_capital
    btc = 0.0
    fee = 0.001
    operations = []
    portfolio_values = []

    for date, row in df.iterrows():
        price = row['Close']
        signal = row['Signal']

        # Comprar
        if signal == 'BUY' and capital > 0:
            amount_to_invest = capital * buy_fraction
            btc_bought = (amount_to_invest / price) * (1 - fee)
            btc += btc_bought
            capital -= amount_to_invest
            operations.append([date, 'BUY', price, amount_to_invest, btc_bought, capital])

        # Vender
        elif signal == 'SELL' and btc > 0:
            btc_to_sell = btc * sell_fraction
            usdt_obtained = (btc_to_sell * price) * (1 - fee)
            btc -= btc_to_sell
            capital += usdt_obtained
            operations.append([date, 'SELL', price, usdt_obtained, btc_to_sell, capital])

        # Valor de cartera
        total_value = capital + btc * price
        portfolio_values.append([date, total_value])

    ops_df = pd.DataFrame(operations, columns=['Date', 'Action', 'Price', 'USDT_Amount', 'BTC_Amount', 'Capital_Remaining'])
    portfolio_df = pd.DataFrame(portfolio_values, columns=['Date', 'Portfolio_Value']).set_index('Date')
    return ops_df, portfolio_df

# Función para métricas de performance
def performance_metrics(portfolio_df, risk_free_rate=0.01):
    returns = portfolio_df['Portfolio_Value'].pct_change().dropna()
    vol = returns.std() * np.sqrt(252)
    sharpe = (returns.mean() - risk_free_rate/252) / returns.std() * np.sqrt(252)
    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (returns.mean() - risk_free_rate/252) / downside
    cumulative = portfolio_df['Portfolio_Value'].cummax()
    drawdown = (portfolio_df['Portfolio_Value'] - cumulative) / cumulative
    max_dd = drawdown.min()

    return {
        "Final Value": portfolio_df['Portfolio_Value'].iloc[-1],
        "Total Return %": (portfolio_df['Portfolio_Value'].iloc[-1] / portfolio_df['Portfolio_Value'].iloc[0] - 1) * 100,
        "Max Drawdown %": max_dd * 100,
        "Sharpe Ratio": sharpe,
        "Sortino Ratio": sortino,
        "Volatilidad anualizada %": vol * 100
    }

# Ejecutar estrategia
ops_dca, port_dca = dca_strategy(df, initial_capital, buy_fraction, sell_fraction)
btc_bh = initial_capital / df['Close'].iloc[0]
port_bh = pd.DataFrame({
    'Date': df.index,
    'Portfolio_Value': btc_bh * df['Close']
}).set_index('Date')

# Calcular métricas
metrics_dca = performance_metrics(port_dca)
metrics_bh = performance_metrics(port_bh)

# Mostrar resultados
st.subheader("📊 Resultados de la Estrategia")
col1, col2 = st.columns(2)

with col1:
    st.metric("Valor Final Estrategia", f"${metrics_dca['Final Value']:,.2f}")
    st.metric("Retorno Total", f"{metrics_dca['Total Return %']:.2f}%")
    st.metric("Máximo Drawdown", f"{metrics_dca['Max Drawdown %']:.2f}%")

with col2:
    st.metric("Valor Final Buy & Hold", f"${metrics_bh['Final Value']:,.2f}")
    st.metric("Retorno Total", f"{metrics_bh['Total Return %']:.2f}%")
    st.metric("Máximo Drawdown", f"{metrics_bh['Max Drawdown %']:.2f}%")

# Gráfico de performance
st.subheader("📈 Comparación de Performance")
fig_perf = go.Figure()
fig_perf.add_trace(go.Scatter(x=port_dca.index, y=port_dca['Portfolio_Value'], name='DCA Strategy', line=dict(color='blue')))
fig_perf.add_trace(go.Scatter(x=port_bh.index, y=port_bh['Portfolio_Value'], name='Buy & Hold', line=dict(color='green')))
fig_perf.update_layout(
    title="Evolución del Valor de la Cartera",
    xaxis_title="Fecha",
    yaxis_title="Valor (USDT)",
    hovermode="x unified",
    template="plotly_dark"
)
st.plotly_chart(fig_perf, use_container_width=True)

# Gráfico principal
st.subheader("📉 Bitcoin Price vs. Fear & Greed Index")
fig = go.Figure()

# Bitcoin Price (línea negra)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['Close'],
    name='Bitcoin Price (USD)',
    yaxis='y1',
    line=dict(color="#000000", width=2)
))

# Fear & Greed Index (línea naranja)
fig.add_trace(go.Scatter(
    x=df.index,
    y=df['FearGreedIndex'],
    name='Fear & Greed Index',
    yaxis='y2',
    line=dict(color="#FFA500" , width=2, dash='dot')
))

# Áreas de Fear & Greed
fig.add_hrect(y0=0, y1=buy_threshold, fillcolor="red", opacity=0.2, layer="below", line_width=0, yref='y2')
fig.add_hrect(y0=buy_threshold, y1=sell_threshold, fillcolor="yellow", opacity=0.2, layer="below", line_width=0, yref='y2')
fig.add_hrect(y0=sell_threshold, y1=100, fillcolor="green", opacity=0.2, layer="below", line_width=0, yref='y2')

# Señales de compra (verde brillante)
df_buy = df[df['Signal'] == 'BUY']
fig.add_trace(go.Scatter(
    x=df_buy.index,
    y=df_buy['Close'],
    mode='markers',
    marker=dict(
        symbol='arrow-up',
        color='lime',
        size=12,
        line=dict(width=1, color="#000000")
    ),
    name=f'Buy Signal (<{buy_threshold})',
    yaxis='y1'
))

# Señales de venta (rojo)
df_sell = df[df['Signal'] == 'SELL']
fig.add_trace(go.Scatter(
    x=df_sell.index,
    y=df_sell['Close'],
    mode='markers',
    marker=dict(
        symbol='arrow-down',
        color='red',
        size=12,
        line=dict(width=1, color="#000000")
    ),
    name=f'Sell Signal (>{sell_threshold})',
    yaxis='y1'
))

# Configuración del layout
fig.update_layout(
    title="Bitcoin Price vs. Fear & Greed Index (Señales de Trading)",
    xaxis=dict(
        title='Date',
        gridcolor='#aaaaaa',
        linecolor="#000000",
        linewidth=2
    ),
    yaxis=dict(
        title='Bitcoin Price (USD)',
        side='left',
        showgrid=True,
        gridcolor='#aaaaaa',
        linecolor="#000000",
        linewidth=2
    ),
    yaxis2=dict(
        title='Fear & Greed Index',
        overlaying='y',
        side='right',
        range=[0, 100],
        showgrid=False,
        linecolor="#000000",
        linewidth=2
    ),
    legend=dict(
        x=0.01,
        y=0.99,
        bgcolor='rgba(255,255,255,0.5)'
    ),
    plot_bgcolor="#3A3838",
    paper_bgcolor='#3A3838',
    font=dict(color='#000000'),
    height=600,
    margin=dict(l=50, r=50, b=50, t=50, pad=4)
)

st.plotly_chart(fig, use_container_width=True)

# Mostrar operaciones
st.subheader("📋 Historial de Operaciones")
st.dataframe(ops_dca.sort_values('Date', ascending=False), use_container_width=True)

# Mostrar datos crudos
if st.checkbox("Mostrar datos crudos"):
    st.subheader("📦 Datos Crudos")
    st.dataframe(df, use_container_width=True)

# Notas finales
st.markdown("---")
st.markdown("""
**Notas:**
- Los datos se actualizan automáticamente desde Binance y Alternative.me
- Las operaciones tienen una comisión del 0.1%
- Los resultados pasados no garantizan performance futura
""")