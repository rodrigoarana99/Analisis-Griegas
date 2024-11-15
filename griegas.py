#import libraries
'''''
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math
import numpy as np
import pandas as pd
from scipy.stats import norm


#Black-Scholes price and Greeks
class EuropeanOptionBS:

    def __init__(self, S, K, T, r, q, sigma, Type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q        
        self.sigma = sigma
        self.Type = Type
        self.d1 = self.d1()
        self.d2 = self.d2()
        self.price = self.price()
        self.delta = self.delta()
        self.theta = self.theta()
        self.vega = self.vega()
        self.gamma = self.gamma()
        self.volga = self.volga()
        self.vanna = self.vanna()        
        
    def d1(self):
        d1 = (math.log(self.S / self.K) \
                   + (self.r - self.q + .5 * (self.sigma ** 2)) * self.T) \
                    / (self.sigma * self.T ** .5)       
        return d1

    def d2(self):
        d2 = self.d1 - self.sigma * self.T ** .5     
        return d2
    
    def price(self):
        if self.Type == "Call":
            price = self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1) \
            - self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2)
        if self.Type == "Put":
            price = self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2) \
            - self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1)            
        return price
    
    def delta(self):
        if self.Type == "Call":
            delta = math.exp(-self.q * self.T) * norm.cdf(self.d1)
        if self.Type == "Put":
            delta = -math.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return delta
    
    def theta(self):
        if self.Type == "Call":
            theta1 = -math.exp(-self.q * self.T) * \
            (self.S * norm.pdf(self.d1) * self.sigma) / (2 * self.T ** .5)
            theta2 = self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(self.d1)
            theta3 = -self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(self.d2)
            theta = theta1 + theta2 + theta3
        if self.Type == "Put":
            theta1 = -math.exp(-self.q * self.T) * \
            (self.S * norm.pdf(self.d1) * self.sigma) / (2 * self.T ** .5)
            theta2 = -self.q * self.S * math.exp(-self.q * self.T) * norm.cdf(-self.d1)
            theta3 = self.r * self.K * math.exp(-self.r * self.T) * norm.cdf(-self.d2)
            theta =  theta1 + theta2 + theta3
        return theta
    
    def vega(self):
        vega = self.S * math.exp(-self.q * self.T) * self.T** .5 * norm.pdf(self.d1)
        return vega
    
    def gamma(self):
        gamma = math.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * self.T** .5)
        return gamma
    
    def volga(self):
        volga = self.vega / self.sigma * self.d1 * self.d2
        return volga
    
    def vanna(self):
        vanna = -self.vega / (self.S * self.sigma * self.T** .5) * self.d2
        return vanna

#parameters
S0 = 100 # stock price
K = 95 # strike price
r = .0 # risk-free interest rate
q = .0 # dividend
T0 = .25 # time to maturity
sigma0 = .4 # implied volatility BS
Type = "Put"
dt = 1 / 252 # 1 business day

# Market changes between t and t + dt
dS = -S0 * .6 * dt**.5 # realised vol = .6
dsigma = .1
T1 = T0 - dt
S1 = S0 + dS
sigma1 = sigma0 + dsigma
P0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).price
P1 = EuropeanOptionBS(S1, K, T1, r, q, sigma1, Type).price
delta0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).delta
isDeltaHedged = 1 #1 if is delta-hedged, 0 otherwise
dPandL = P1 - P0 - delta0 * dS * isDeltaHedged
print("P&L: " + str(dPandL))
#initial greeks
theta0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).theta
vega0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).vega
gamma0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).gamma
volga0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).volga
vanna0 = EuropeanOptionBS(S0, K, T0, r, q, sigma0, Type).vanna

#P&L attribution
delta_PandL = delta0 * dS * (1 - isDeltaHedged)
theta_PandL = theta0 * dt
vega_PandL = vega0 * dsigma
gamma_PandL = 1 / 2 * gamma0 * dS**2
volga_PandL = 1 / 2 * volga0 * dsigma**2
vanna_PandL = vanna0 * dS * dsigma
unexplained = dPandL - sum([delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL])

y = [delta_PandL, theta_PandL, vega_PandL, gamma_PandL, volga_PandL, vanna_PandL, unexplained]
x = ["delta", "theta", "vega", "gamma", "volga", "vanna","unexplained"]

fig = plt.figure(figsize=(15, 5))
plt.bar(x, y)
plt.title("P&L Decomposition")
plt.show();
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
import seaborn as sns

class EuropeanOptionBS:
    # [Mantenemos la clase original igual]
    def __init__(self, S, K, T, r, q, sigma, Type):
        self.S = S
        self.K = K
        self.T = T
        self.r = r
        self.q = q        
        self.sigma = sigma
        self.Type = Type
        self.d1 = self.d1()
        self.d2 = self.d2()
        self.price = self.price()
        self.delta = self.delta()
        self.theta = self.theta()
        self.vega = self.vega()
        self.gamma = self.gamma()
        self.volga = self.volga()
        self.vanna = self.vanna()        
        
    def d1(self):
        d1 = (np.log(self.S / self.K) \
                   + (self.r - self.q + .5 * (self.sigma ** 2)) * self.T) \
                    / (self.sigma * self.T ** .5)       
        return d1

    def d2(self):
        d2 = self.d1 - self.sigma * self.T ** .5     
        return d2
    
    def price(self):
        if self.Type == "Call":
            price = self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1) \
            - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        if self.Type == "Put":
            price = self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) \
            - self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)            
        return price
    
    def delta(self):
        if self.Type == "Call":
            delta = np.exp(-self.q * self.T) * norm.cdf(self.d1)
        if self.Type == "Put":
            delta = -np.exp(-self.q * self.T) * norm.cdf(-self.d1)
        return delta
    
    def theta(self):
        if self.Type == "Call":
            theta1 = -np.exp(-self.q * self.T) * \
            (self.S * norm.pdf(self.d1) * self.sigma) / (2 * self.T ** .5)
            theta2 = self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(self.d1)
            theta3 = -self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
            theta = theta1 + theta2 + theta3
        if self.Type == "Put":
            theta1 = -np.exp(-self.q * self.T) * \
            (self.S * norm.pdf(self.d1) * self.sigma) / (2 * self.T ** .5)
            theta2 = -self.q * self.S * np.exp(-self.q * self.T) * norm.cdf(-self.d1)
            theta3 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
            theta =  theta1 + theta2 + theta3
        return theta
    
    def vega(self):
        vega = self.S * np.exp(-self.q * self.T) * self.T** .5 * norm.pdf(self.d1)
        return vega
    
    def gamma(self):
        gamma = np.exp(-self.q * self.T) * norm.pdf(self.d1) / (self.S * self.sigma * self.T** .5)
        return gamma
    
    def volga(self):
        volga = self.vega / self.sigma * self.d1 * self.d2
        return volga
    
    def vanna(self):
        vanna = -self.vega / (self.S * self.sigma * self.T** .5) * self.d2
        return vanna

def simulate_market_scenarios(S0, sigma0, days, n_sims, realized_vol, vol_of_vol):
    """Simula escenarios de mercado para el precio y la volatilidad."""
    dt = 1/252
    
    # Simulación de precios
    Z1 = np.random.normal(0, 1, (n_sims, days))
    price_paths = np.zeros((n_sims, days+1))
    price_paths[:,0] = S0
    
    # Simulación de volatilidades
    Z2 = np.random.normal(0, 1, (n_sims, days))
    vol_paths = np.zeros((n_sims, days+1))
    vol_paths[:,0] = sigma0
    
    # Correlación entre precio y volatilidad (efecto leverage)
    rho = -0.7
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2
    
    for t in range(days):
        # Cambio en precio (modelo log-normal)
        dS = price_paths[:,t] * (realized_vol * np.sqrt(dt) * Z1[:,t])
        price_paths[:,t+1] = price_paths[:,t] + dS
        
        # Cambio en volatilidad (modelo normal con límite inferior)
        dvol = vol_of_vol * np.sqrt(dt) * Z2[:,t]
        vol_paths[:,t+1] = np.maximum(0.01, vol_paths[:,t] + dvol)
    
    return price_paths, vol_paths

def calculate_pnl_attribution(option_params, price_paths, vol_paths, is_delta_hedged=True):
    """Calcula la atribución de P&L para cada camino simulado."""
    n_sims, n_steps = price_paths.shape
    dt = 1/252
    
    # Inicializar arrays para almacenar componentes del P&L
    pnl_components = {
        'total': np.zeros(n_sims),
        'delta': np.zeros(n_sims),
        'theta': np.zeros(n_sims),
        'vega': np.zeros(n_sims),
        'gamma': np.zeros(n_sims),
        'volga': np.zeros(n_sims),
        'vanna': np.zeros(n_sims),
        'unexplained': np.zeros(n_sims)
    }
    
    # Calcular P&L para cada camino
    for i in range(n_sims):
        total_pnl = 0
        for t in range(n_steps-1):
            # Calcular griegas en t
            opt = EuropeanOptionBS(
                price_paths[i,t], 
                option_params['K'], 
                option_params['T'] - t*dt,
                option_params['r'], 
                option_params['q'], 
                vol_paths[i,t],
                option_params['Type']
            )
            
            # Cambios en el mercado
            dS = price_paths[i,t+1] - price_paths[i,t]
            dsigma = vol_paths[i,t+1] - vol_paths[i,t]
            
            # Componentes del P&L
            delta_pnl = opt.delta * dS * (1 - is_delta_hedged)
            theta_pnl = opt.theta * dt
            vega_pnl = opt.vega * dsigma
            gamma_pnl = 0.5 * opt.gamma * dS**2
            volga_pnl = 0.5 * opt.volga * dsigma**2
            vanna_pnl = opt.vanna * dS * dsigma
            
            # Acumular componentes
            pnl_components['delta'][i] += delta_pnl
            pnl_components['theta'][i] += theta_pnl
            pnl_components['vega'][i] += vega_pnl
            pnl_components['gamma'][i] += gamma_pnl
            pnl_components['volga'][i] += volga_pnl
            pnl_components['vanna'][i] += vanna_pnl
            
            # P&L total del paso
            step_pnl = sum([delta_pnl, theta_pnl, vega_pnl, gamma_pnl, volga_pnl, vanna_pnl])
            total_pnl += step_pnl
            
        pnl_components['total'][i] = total_pnl
        
    # Calcular P&L no explicado
    pnl_components['unexplained'] = pnl_components['total'] - sum([
        pnl_components[k] for k in ['delta', 'theta', 'vega', 'gamma', 'volga', 'vanna']
    ])
    
    return pnl_components

# Configuración de la simulación
option_params = {
    'S0': 100,
    'K': 95,
    'T': 0.25,
    'r': 0.0,
    'q': 0.0,
    'sigma0': 0.4,
    'Type': "Put"
}

# Parámetros de simulación
n_sims = 1000
days = 63  # ~3 meses
realized_vol = 0.6
vol_of_vol = 0.4

# Realizar simulación
print("Simulando escenarios de mercado...")
price_paths, vol_paths = simulate_market_scenarios(
    option_params['S0'], 
    option_params['sigma0'], 
    days, 
    n_sims, 
    realized_vol, 
    vol_of_vol
)

# Calcular P&L
print("Calculando atribución de P&L...")
pnl_components = calculate_pnl_attribution(option_params, price_paths, vol_paths, is_delta_hedged=True)

# [Mantener todo el código anterior de EuropeanOptionBS]

class VolatilitySurface:
    def __init__(self, S0, r=0.0, q=0.0):
        """
        Inicializa la superficie de volatilidad adaptada a nuestros parámetros originales
        """
        self.S0 = S0  # Precio spot inicial
        self.r = r    # Tasa libre de riesgo
        self.q = q    # Tasa de dividendo
        
    def generate_surface_params(self, min_strike_pct=0.7, max_strike_pct=1.3):
        """
        Genera los parámetros de la superficie adaptados a nuestro rango de análisis
        """
        # Strikes como porcentaje del spot (adaptado a nuestro rango de análisis)
        self.strikes_pct = np.linspace(min_strike_pct, max_strike_pct, 25)
        self.strikes = self.S0 * self.strikes_pct
        
        # Tiempos a vencimiento relevantes para nuestro análisis
        self.tenors = np.array([1/252, 1/12, 2/12, 3/12, 6/12, 9/12, 1.0])
        
        # Crear grids para la superficie
        self.K_grid, self.T_grid = np.meshgrid(self.strikes_pct, self.tenors)
    
    def skewed_vol(self, K_norm, T, base_vol=0.4):
        """
        Genera volatilidad con skew adaptada a nuestros parámetros originales
        base_vol: volatilidad base (usando el 0.4 de nuestro ejemplo original)
        """
        # Parámetros adaptados a nuestro caso
        params = {
            'base_vol': base_vol,  # Volatilidad base del ejemplo original
            'skew': -0.2 * base_vol,  # Skew proporcional a la vol base
            'convexity': 0.5 * base_vol,  # Convexidad proporcional
            'term': 0.1  # Estructura temporal moderada
        }
        
        # Cálculo de la volatilidad con skew
        vol = params['base_vol'] + \
              params['skew'] * (K_norm - 1) + \
              params['convexity'] * (K_norm - 1)**2
        
        # Ajuste temporal
        vol = vol * (1 + params['term'] * np.sqrt(T))
        
        return np.maximum(vol, 0.1 * base_vol)  # Mínimo 10% de la vol base
    
    def generate_surface(self, base_vol=0.4):
        """
        Genera la superficie completa usando la volatilidad base especificada
        """
        self.generate_surface_params()
        self.vol_surface = np.zeros_like(self.K_grid)
        
        for i in range(len(self.tenors)):
            for j in range(len(self.strikes_pct)):
                self.vol_surface[i,j] = self.skewed_vol(
                    self.strikes_pct[j], 
                    self.tenors[i],
                    base_vol
                )
        
        return self.vol_surface
    
    def get_vol(self, K, T, base_vol=0.4):
        """
        Obtiene la volatilidad para un strike y tenor específicos,
        adaptada para usar directamente en EuropeanOptionBS
        """
        K_norm = K/self.S0
        
        # Si K_norm está fuera de rango, usamos el extremo más cercano
        K_norm = np.clip(K_norm, min(self.strikes_pct), max(self.strikes_pct))
        
        # Si T está fuera de rango, usamos el extremo más cercano
        T = np.clip(T, min(self.tenors), max(self.tenors))
        
        points = np.column_stack((self.K_grid.flatten(), self.T_grid.flatten()))
        vol = griddata(
            points,
            self.vol_surface.flatten(),
            np.array([[K_norm, T]]),
            method='cubic'
        )[0]
        
        # Si la interpolación falla, calculamos directamente
        if np.isnan(vol):
            vol = self.skewed_vol(K_norm, T, base_vol)
            
        return vol

def simulate_market_scenarios_adapted(S0, K, T, base_vol, days, n_sims, realized_vol):
    """
    Versión adaptada de la simulación que usa nuestra superficie de volatilidad
    """
    dt = 1/252
    vol_surface = VolatilitySurface(S0)
    vol_surface.generate_surface(base_vol)
    
    # Arrays para paths
    price_paths = np.zeros((n_sims, days+1))
    vol_paths = np.zeros((n_sims, days+1))
    price_paths[:,0] = S0
    
    # Inicializar volatilidades
    for i in range(n_sims):
        vol_paths[i,0] = vol_surface.get_vol(K, T, base_vol)
    
    # Simulación con correlación
    rho = -0.7  # Correlación precio-volatilidad
    for t in range(days):
        Z1 = np.random.normal(0, 1, n_sims)
        Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n_sims)
        
        # Actualizar precios
        dS = price_paths[:,t] * (realized_vol * np.sqrt(dt) * Z1)
        price_paths[:,t+1] = price_paths[:,t] + dS
        
        # Actualizar volatilidades
        remaining_T = T - (t+1)*dt
        if remaining_T > 0:
            for i in range(n_sims):
                vol_paths[i,t+1] = vol_surface.get_vol(
                    K,
                    remaining_T,
                    base_vol
                )
        else:
            vol_paths[:,t+1] = vol_paths[:,t]
    
    return price_paths, vol_paths

# Ejemplo de uso con los parámetros originales:
if __name__ == "__main__":
    # Parámetros originales
    S0 = 100
    K = 95
    T = 0.25
    r = 0.0
    q = 0.0
    base_vol = 0.4
    Type = "Put"
    
    # Crear superficie y mostrar
    vol_surface = VolatilitySurface(S0, r, q)
    vol_surface.generate_surface(base_vol)
    
    # Simular escenarios
    n_sims = 1000
    days = 63  # ~3 meses
    realized_vol = 0.6
    
    price_paths, vol_paths = simulate_market_scenarios_adapted(
        S0, K, T, base_vol, days, n_sims, realized_vol
    )
    
    # Visualización
    plt.figure(figsize=(15, 5))
    
    # Paths de precio
    plt.subplot(1, 2, 1)
    for i in range(10):  # Mostrar primeros 10 paths
        plt.plot(price_paths[i,:])
    plt.title('Paths de Precio')
    plt.xlabel('Días')
    plt.ylabel('Precio')
    
    # Paths de volatilidad
    plt.subplot(1, 2, 2)
    for i in range(10):  # Mostrar primeros 10 paths
        plt.plot(vol_paths[i,:])
    plt.title('Paths de Volatilidad')
    plt.xlabel('Días')
    plt.ylabel('Volatilidad')
    
    plt.tight_layout()
    plt.show()
    
    # Mostrar algunas volatilidades ejemplo
    print("\nEjemplos de volatilidades:")
    test_strikes = [90, 95, 100, 105, 110]
    test_tenors = [0.1, 0.25, 0.5]
    
    print("\nK/S0    T    Volatilidad")
    print("--------------------------")
    for K in test_strikes:
        for T in test_tenors:
            vol = vol_surface.get_vol(K, T, base_vol)
            print(f"{K/S0:.2f}   {T:.2f}   {vol:.4f}")