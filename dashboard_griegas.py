# dashboard.py
'''''
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from matplotlib.gridspec import GridSpec 
import matplotlib
matplotlib.use('TkAgg')  # O 'Qt5Agg' si tienes Qt instalado
import matplotlib.pyplot as plt
# Importamos nuestras clases y funciones anteriores
from griegas import (
    EuropeanOptionBS,  # La clase principal de opciones
    simulate_market_scenarios_adapted,  # Si ya tienes esta función
    calculate_pnl_attribution , # Si ya tienes esta función
    VolatilitySurface,  # Si ya tienes esta clase
    simulate_market_scenarios_adapted
)
# [Asegúrate de tener las clases EuropeanOptionBS y VolatilitySurface en el mismo directorio]
def create_dashboard(S0=100, K=95, T=0.25, r=0.0, sigma=0.4, option_type="Put", 
                    n_sims=1000, realized_vol=0.6, is_delta_hedged=True):
    """
    Crea un dashboard visual del análisis de opciones usando matplotlib
    """
    # Configuración inicial
    plt.style.use('seaborn-v0_8-darkgrid')

    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig)
    
    # 1. Información de la Opción
    option = EuropeanOptionBS(S0, K, T, r, 0, sigma, option_type)
    
    # Crear tabla de métricas
    metrics_ax = fig.add_subplot(gs[0, 0])
    metrics = [
        ['Métrica', 'Valor'],
        ['Precio', f"{option.price:.4f}"],
        ['Delta', f"{option.delta:.4f}"],
        ['Gamma', f"{option.gamma:.4f}"],
        ['Vega', f"{option.vega:.4f}"],
        ['Theta', f"{option.theta:.4f}"]
    ]
    metrics_ax.axis('tight')
    metrics_ax.axis('off')
    table = metrics_ax.table(cellText=metrics, loc='center', cellLoc='center')
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    metrics_ax.set_title("Métricas de la Opción", pad=20)

    # 2. Perfil de Payoff
    payoff_ax = fig.add_subplot(gs[0, 1])
    S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    payoff = np.array([EuropeanOptionBS(s, K, T, r, 0, sigma, option_type).price 
                      for s in S_range])
    payoff_ax.plot(S_range, payoff)
    payoff_ax.axvline(x=K, color='r', linestyle='--', alpha=0.5)
    payoff_ax.grid(True)
    payoff_ax.set_title("Perfil de Payoff")
    payoff_ax.set_xlabel("Precio del Subyacente")
    payoff_ax.set_ylabel("Valor de la Opción")

    # 3. Superficie de Volatilidad
    vol_surface = VolatilitySurface(S0, r)
    vol_surface.generate_surface(sigma)
    
    vol_ax = fig.add_subplot(gs[0, 2], projection='3d')
    surf = vol_ax.plot_surface(
        vol_surface.K_grid, 
        vol_surface.T_grid, 
        vol_surface.vol_surface,
        cmap='viridis',
        alpha=0.8
    )
    vol_ax.set_xlabel('Moneyness (K/S0)')
    vol_ax.set_ylabel('Time to Maturity')
    vol_ax.set_zlabel('Implied Volatility')
    vol_ax.set_title("Superficie de Volatilidad")
    plt.colorbar(surf, ax=vol_ax)

    # 4. Simulación y Análisis P&L
    days = int(T * 252)
    price_paths, vol_paths = simulate_market_scenarios_adapted(
        S0, K, T, sigma, days, n_sims, realized_vol
    )
    
    pnl_components = calculate_pnl_attribution(
        {'S0': S0, 'K': K, 'T': T, 'r': r, 'q': 0, 'Type': option_type},
        price_paths,
        vol_paths,
        is_delta_hedged
    )

    # 5. Distribución del P&L
    pnl_ax = fig.add_subplot(gs[1, 0])
    sns.histplot(data=pnl_components['total'], kde=True, ax=pnl_ax)
    pnl_ax.set_title('Distribución del P&L Total')
    pnl_ax.set_xlabel('P&L')
    pnl_ax.set_ylabel('Frecuencia')

    # 6. Componentes del P&L
    comp_ax = fig.add_subplot(gs[1, 1])
    df_components = pd.DataFrame(pnl_components).melt()
    sns.boxplot(x='variable', y='value', data=df_components, ax=comp_ax)
    comp_ax.tick_params(axis='x', rotation=45)

    comp_ax.set_title('Componentes del P&L')
    comp_ax.set_xlabel('Componente')
    comp_ax.set_ylabel('Valor') 

    # 7. Paths de Precio
    price_ax = fig.add_subplot(gs[1, 2])
    for i in range(min(10, n_sims)):
        price_ax.plot(price_paths[i,:])
    price_ax.set_title('Paths de Precio')
    price_ax.set_xlabel('Días')
    price_ax.set_ylabel('Precio')

    # 8. Paths de Volatilidad
    vol_path_ax = fig.add_subplot(gs[2, 0])
    for i in range(min(10, n_sims)):
        vol_path_ax.plot(vol_paths[i,:])
    vol_path_ax.set_title('Paths de Volatilidad')
    vol_path_ax.set_xlabel('Días')
    vol_path_ax.set_ylabel('Volatilidad')

    # 9. Estadísticas de P&L
    stats_ax = fig.add_subplot(gs[2, 1])
    stats = {
        'Estadística': ['Media', 'Desv. Std', 'VaR 95%', 'VaR 99%'],
        'Valor': [
            np.mean(pnl_components['total']),
            np.std(pnl_components['total']),
            np.percentile(pnl_components['total'], 5),
            np.percentile(pnl_components['total'], 1)
        ]
    }
    stats_ax.axis('tight')
    stats_ax.axis('off')
    stats_table = stats_ax.table(cellText=[[k, f"{v:.4f}"] for k, v in zip(stats['Estadística'], stats['Valor'])],
                                loc='center', cellLoc='center')
    stats_table.set_fontsize(10)
    stats_table.scale(1.2, 1.5)
    stats_ax.set_title("Estadísticas de P&L", pad=20)

    plt.tight_layout()
    plt.savefig('dashboard_output.png', bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    # Ejecutar el dashboard con parámetros por defecto
    create_dashboard()
'''''
# app.py

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
from griegas import (
    EuropeanOptionBS,
    simulate_market_scenarios_adapted,
    calculate_pnl_attribution,
    VolatilitySurface
)

# Configuración de la página
st.set_page_config(
    page_title="Dashboard de Opciones",
    page_icon="📈",
    layout="wide"
)

# Título principal
st.title('Dashboard de Análisis de Opciones')

# Sidebar para parámetros
st.sidebar.header('Parámetros de la Opción')
S0 = st.sidebar.number_input('Precio Spot (S0)', value=100.0, step=1.0)
K = st.sidebar.number_input('Strike (K)', value=100.0, step=1.0)
T = st.sidebar.number_input('Tiempo a vencimiento (años)', value=0.25, step=0.1)
r = st.sidebar.number_input('Tasa libre de riesgo (%)', value=0.0, step=0.1) / 100
sigma = st.sidebar.number_input('Volatilidad implícita (%)', value=20.0, step=1.0) / 100
option_type = st.sidebar.selectbox('Tipo de Opción', ['Call', 'Put'])

# Parámetros de simulación
st.sidebar.header('Parámetros de Simulación')
n_sims = st.sidebar.slider('Número de simulaciones', 100, 5000, 1000)
realized_vol = st.sidebar.number_input('Volatilidad realizada (%)', value=20.0, step=1.0) / 100
is_delta_hedged = st.sidebar.checkbox('Delta Hedging', value=True)

# Crear instancia de la opción
option = EuropeanOptionBS(S0, K, T, r, 0, sigma, option_type)

# Layout principal
col1, col2 = st.columns(2)

# Columna 1: Métricas y Payoff
with col1:
    st.subheader('Métricas de la Opción')
    metrics_df = pd.DataFrame({
        'Métrica': ['Precio', 'Delta', 'Gamma', 'Vega', 'Theta'],
        'Valor': [
            option.price,
            option.delta,
            option.gamma,
            option.vega,
            option.theta
        ]
    })
    st.dataframe(metrics_df.style.format({'Valor': '{:.4f}'}))

    # Payoff Profile
    st.subheader('Perfil de Payoff')
    S_range = np.linspace(S0 * 0.5, S0 * 1.5, 100)
    payoff = np.array([EuropeanOptionBS(s, K, T, r, 0, sigma, option_type).price for s in S_range])
    
    fig_payoff = px.line(x=S_range, y=payoff)
    fig_payoff.add_vline(x=K, line_dash="dash", line_color="red")
    st.plotly_chart(fig_payoff)

# Columna 2: Superficie de Vol y Simulaciones
with col2:
    st.subheader('Superficie de Volatilidad')
    vol_surface = VolatilitySurface(S0, r)
    vol_surface.generate_surface(sigma)
    
    fig_vol = go.Figure(data=[
        go.Surface(
            x=vol_surface.K_grid,
            y=vol_surface.T_grid,
            z=vol_surface.vol_surface
        )
    ])
    st.plotly_chart(fig_vol)

# Simulaciones y P&L
st.header('Análisis de P&L y Riesgo')

days = int(T * 252)
price_paths, vol_paths = simulate_market_scenarios_adapted(
    S0, K, T, sigma, days, n_sims, realized_vol
)

pnl_components = calculate_pnl_attribution(
    {'S0': S0, 'K': K, 'T': T, 'r': r, 'q': 0, 'Type': option_type},
    price_paths,
    vol_paths,
    is_delta_hedged
)

col3, col4 = st.columns(2)

with col3:
    # Distribución del P&L
    st.subheader('Distribución del P&L')
    fig_pnl = px.histogram(pnl_components['total'], nbins=50)
    st.plotly_chart(fig_pnl)
    
    # Estadísticas
    st.subheader('Estadísticas de P&L')
    stats_df = pd.DataFrame({
        'Estadística': ['Media', 'Desv. Std', 'VaR 95%', 'VaR 99%'],
        'Valor': [
            np.mean(pnl_components['total']),
            np.std(pnl_components['total']),
            np.percentile(pnl_components['total'], 5),
            np.percentile(pnl_components['total'], 1)
        ]
    })
    st.dataframe(stats_df)

with col4:
    # Paths
    st.subheader('Paths de Precio')
    fig_price = go.Figure()
    for i in range(min(10, n_sims)):
        fig_price.add_trace(go.Scatter(y=price_paths[i,:], mode='lines'))
    st.plotly_chart(fig_price)
    
    # Volatility Paths
    st.subheader('Paths de Volatilidad')
    fig_vol_paths = go.Figure()
    for i in range(min(10, n_sims)):
        fig_vol_paths.add_trace(go.Scatter(y=vol_paths[i,:], mode='lines'))
    st.plotly_chart(fig_vol_paths)