
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from itertools import product
import warnings
warnings.filterwarnings('ignore')

# Data retrieval: First it will try to import he dataretrieval package. 
#If not successful, it will fall back on synthetic data. This will allow the code to run locally, but an internet connection is required for best results.

try:
    from dataretrieval import nwis
    DATA_AVAILABLE = True
except ImportError:
    DATA_AVAILABLE = False
    print("dataretrieval not installed. Using synthetic data.")

# Metrics: These functions implement all evaluation metrics used in the paper. They are kept simple and vectorized for speed.

def rmse(y_true, y_pred): return np.sqrt(mean_squared_error(y_true, y_pred))
def mae(y_true, y_pred): return mean_absolute_error(y_true, y_pred)
def r_coeff(y_true, y_pred): return r2_score(y_true, y_pred)
def rae(y_true, y_pred): return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
def mape(y_true, y_pred): return 100 * np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8)))
def willmott_ia(y_true, y_pred):
    num = np.sum((y_true - y_pred)**2)
    den = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true)))**2)
    return 1 - num / den
def legate_mccabe(y_true, y_pred):
    return 1 - np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true - np.mean(y_true)))
def performance_index(metrics_dict, all_metrics_list):
    R_min = min(m['R'] for m in all_metrics_list)
    RMSE_max = max(m['RMSE'] for m in all_metrics_list)
    MAE_max = max(m['MAE'] for m in all_metrics_list)
    RAE_max = max(m['RAE'] for m in all_metrics_list)
    MAPE_max = max(m['MAPE'] for m in all_metrics_list)
    E_min = min(m['E'] for m in all_metrics_list)
    IA_min = min(m['IA'] for m in all_metrics_list)
    return (1/7)*(R_min/metrics_dict['R'] + metrics_dict['RMSE']/RMSE_max +
                  metrics_dict['MAE']/MAE_max + metrics_dict['RAE']/RAE_max +
                  metrics_dict['MAPE']/MAPE_max + E_min/metrics_dict['E'] +
                  IA_min/metrics_dict['IA'])

# ANFIS MODEL  Adaptive Neuro-Fuzzy Inference System (Sugeno type).Supports any number of inputs and membership functions per input. The number of rules = (n_mfs)^(n_inputs).

class ANFIS:
    def __init__(self, n_inputs, n_mfs=2):
        self.n_inputs = n_inputs
        self.n_mfs = n_mfs
        self.n_rules = n_mfs ** n_inputs
        # Safety check to prevent memory explosion (like 2^28 rules)
        if self.n_rules > 10000:
            raise MemoryError(f"Too many rules ({self.n_rules}). Reduce n_inputs or n_mfs.")
        # Pre‑compute all combinations of membership indices for each rule
        self.rule_indices = list(product(range(n_mfs), repeat=n_inputs))

    def gaussmf(self, x, c, sigma):
        z = ((x - c) / sigma) ** 2
        z = np.clip(z, 0, 100) #keeps exponent manageable
        return np.exp(-0.5 * z)

    def forward(self, X, params):
        # First part: membership function parameters (centers and widths)
        n_mf_params = 2 * self.n_inputs * self.n_mfs
        mf_params = params[:n_mf_params].reshape(self.n_inputs, self.n_mfs, 2)
        # Second part: consequent parameters (p, q, r for each rule)
        cons_params = params[n_mf_params:].reshape(self.n_rules, self.n_inputs + 1)

        # Compute membership grades for each input and each MF
        mf_vals = []
        for i in range(self.n_inputs):
            x = X[:, i]
            mf_vals_i = np.zeros((X.shape[0], self.n_mfs))
            for j in range(self.n_mfs):
                c = mf_params[i, j, 0]
                sigma = mf_params[i, j, 1]
                mf_vals_i[:, j] = self.gaussmf(x, c, sigma)
            mf_vals.append(mf_vals_i)
        
        # Compute firing strengths (product of degrees for each rule)
        firing = np.ones((X.shape[0], self.n_rules))
        for r, idx_tuple in enumerate(self.rule_indices):
            for i, mf_idx in enumerate(idx_tuple):
                firing[:, r] *= mf_vals[i][:, mf_idx]
        
        # Normalise firing strengths
        firing_sum = np.sum(firing, axis=1, keepdims=True)
        w_norm = firing / (firing_sum + 1e-8)
        
        # Compute weighted output: sum over rules of (normalised firing * linear consequent)
        output = np.zeros(X.shape[0])
        for r in range(self.n_rules):
            linear_part = np.dot(X, cons_params[r, :-1]) + cons_params[r, -1]
            output += w_norm[:, r] * linear_part
        return output

    def get_param_vector(self, X_train, y_train=None):
        """Improved: initialise consequent params with linear regression"""
        n_mf_params = 2 * self.n_inputs * self.n_mfs
        n_cons_params = self.n_rules * (self.n_inputs + 1)
        bounds = []
        
        # Bounds for MF centres and widths
        for i in range(self.n_inputs):
            col_min, col_max = X_train[:, i].min(), X_train[:, i].max()
            for _ in range(self.n_mfs):
                bounds.append((col_min, col_max)) #center
                bounds.append((0.1, (col_max - col_min) / 2)) #width
        
        # Consequent parameter bounds
        for _ in range(n_cons_params):
            bounds.append((-5, 5))  # wider range because y is scaled

        # Initialisation: random for MF params, linear regression for cons params
        init_mf = np.array([np.random.uniform(low, high) for low, high in bounds[:n_mf_params]])
        init_cons = np.zeros(n_cons_params)
        if y_train is not None:
            # Fit a linear regression for each rule? Simplified: use global LR to set bias
            lr = LinearRegression().fit(X_train, y_train)
            # Put intercept and coefficients into first rule (others remain zero)
            # This gives a decent starting point
            cons_per_rule = n_inputs + 1
            init_cons[:cons_per_rule] = [lr.coef_[0], lr.coef_[1], lr.coef_[2], lr.intercept_]  # 3 inputs + bias
        init = np.concatenate([init_mf, init_cons])
        return init, bounds

# A-DEPSO (with restart) 
class ADEPSO:
    def __init__(self, dim, bounds, pop_size=40, max_iter=500, beta=30.0, delta=0.5, mcr0=0.5, mu=0.1):
        self.dim = dim # number of decision variables
        self.bounds = np.array(bounds) # (low, high) for each variable
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.beta = beta  # adaptive mutation parameter
        self.delta = delta # inertia weight factor
        self.mcr = mcr0 # initial crossover rate
        self.mu = mu # adaptation rate for mcr
        self.c1 = self.c2 = 1.5 # PSO acceleration constants

    def optimize(self, fitness_func, n_restarts=3, verbose=True):
        """Run multiple restarts and keep the best solution"""
        best_overall = None
        best_fit_overall = np.inf
        for restart in range(n_restarts):
            if verbose:
                print(f"  Restart {restart+1}/{n_restarts}...")
            best, best_fit = self._run_once(fitness_func, verbose=False)
            if best_fit < best_fit_overall:
                best_overall = best
                best_fit_overall = best_fit
                if verbose:
                    print(f"    New best RMSE = {best_fit:.4f}")
        return best_overall, best_fit_overall

    def _run_once(self, fitness_func, verbose=True):
         # Initialise population within bounds
        X = np.random.uniform(low=self.bounds[:,0], high=self.bounds[:,1],
                              size=(self.pop_size, self.dim))
        fitness = np.array([fitness_func(ind) for ind in X])
        p_best = X.copy() # personal best positions
        p_best_fit = fitness.copy()
        g_best_idx = np.argmin(fitness)
        g_best = X[g_best_idx].copy() # global best position
        g_best_fit = fitness[g_best_idx]
        V = np.zeros_like(X) # velocities for PSO part
        S_Ar = [] # store successful crossover rates

        for l in range(1, self.max_iter+1):
            # Adaptive parameters (Eq.31, 32)
            G = np.sin(self.beta * np.pi * (l/self.max_iter)) * np.exp(-l/self.max_iter) * (0.5 + 0.15*np.random.randn())
            w = self.delta * np.exp(-l/self.max_iter)

            for j in range(self.pop_size):
                #  Mutation: combine DE and PSO (Eq.27-30) 
                idx = np.random.choice(self.pop_size, 3, replace=False)
                a1, a2, a3 = idx[0], idx[1], idx[2]
                X_DE = X[a1] + G * (X[a2] - X[a3]) # DE mutation (Eq.27)
                V[j] = w * V[j] + self.c1*np.random.rand(self.dim)*(p_best[j]-X[j]) + \
                       self.c2*np.random.rand(self.dim)*(g_best-X[j]) # PSO velocity (Eq.28)
                X_PSO = X[j] + V[j] # PSO position (Eq.29)
                rho = np.random.rand()
                X_new = rho * X_PSO + (1-rho) * X_DE # Combine (Eq.30)

                # Adaptive crossover (Eq.33-35)
                Ar = self.mcr + 0.1 * np.random.randn()
                z = np.zeros(self.dim)
                for i in range(self.dim):
                    pa, pb = np.random.rand(), np.random.rand()
                    irand = np.random.randint(0, self.dim)
                    if (pa < Ar and pb < 0.5) or i == irand:
                        z[i] = X_new[i]
                    elif (pa < Ar and pb > 0.5) or i == irand:
                        z[i] = X[j][i]
                    else:
                        z[i] = g_best[i]

                # Refreshing operator - reduce probability to avoid disruption. Low probability (0.1) to avoid destroying good solutions.
                if np.random.rand() < 0.1:  #
                    if np.random.rand() < 0.5:
                        sigma = np.random.randn()
                        z = z + sigma * (2*np.random.randn()*g_best - X[j])
                    else:
                        sigma = np.random.randn()
                        idx_a = np.random.choice(self.pop_size, 2, replace=False)
                        z = g_best + sigma * (X[idx_a[0]] - X[idx_a[1]])
                
                # Clip to bounds and evaluate
                z = np.clip(z, self.bounds[:,0], self.bounds[:,1])
                new_fit = fitness_func(z)
               
                #Selection
                if new_fit <= fitness[j]:
                    X[j] = z
                    fitness[j] = new_fit
                    p_best[j] = z
                    p_best_fit[j] = new_fit
                    if new_fit <= g_best_fit:
                        g_best = z
                        g_best_fit = new_fit
                    S_Ar.append(Ar)
            
            # Update crossover rate
            if len(S_Ar) > 0:
                self.mcr = (1-self.mu)*self.mcr + self.mu * np.mean(S_Ar)
                S_Ar = []
            if verbose and l % 100 == 0:
                print(f"    Iter {l:3d}, Best RMSE = {g_best_fit:.4f}")
        return g_best, g_best_fit

# Baseline models 
class LSSVM:
    def __init__(self, gamma=1992.0): self.gamma = gamma
    def fit(self, X, y):
        n = X.shape[0]
        K = np.dot(X, X.T) + np.eye(n) / self.gamma
        self.alpha = np.linalg.solve(K, y)
        self.X_train = X
    def predict(self, X): return np.dot(X, self.X_train.T) @ self.alpha

class GRNN:
    def __init__(self, sigma=380.0): self.sigma = sigma
    def fit(self, X, y): self.X, self.y = X, y
    def predict(self, X):
        y_pred = np.zeros(len(X))
        for i, x in enumerate(X):
            dist = np.linalg.norm(self.X - x, axis=1)
            weights = np.exp(-dist**2 / (2*self.sigma**2))
            y_pred[i] = np.sum(weights * self.y) / (np.sum(weights) + 1e-8)
        return y_pred

# Simplified features (3 inputs)
def create_simple_features(df):
    data = df[['Q', 'EC']].copy()
    data['EC_lag1'] = data['EC'].shift(1)
    data['EC_lag2'] = data['EC'].shift(2)
    data.dropna(inplace=True)
    X = data[['EC_lag1', 'EC_lag2', 'Q']].values
    y = data['EC'].values
    dates = data.index
    return X, y, dates

# ------------------------- Main execution -------------------------
if __name__ == "__main__":
    # Load data (2004-2024)
    if DATA_AVAILABLE:
        site = '07374000'
        start = '2004-01-01'
        end = '2024-05-19'
        print(f"Downloading daily data from USGS station {site} ({start} to {end})...")
        # Downloading daily discharge and specific counductance
        df_q, _ = nwis.get_dv(sites=site, parameterCd='00060', start=start, end=end)
        df_sc, _ = nwis.get_dv(sites=site, parameterCd='00095', start=start, end=end)
        # Rename columns for clarity
        df_q = df_q[['00060_Mean']].rename(columns={'00060_Mean': 'Q'})
        df_sc = df_sc[['00095_Mean']].rename(columns={'00095_Mean': 'EC'})
        # Merge on date and resample to monthly averages
        combined = pd.merge(df_q, df_sc, left_index=True, right_index=True, how='inner')
        monthly = combined.resample('ME').mean().dropna()
        print(f"Obtained {len(monthly)} monthly records.")
    else:
        print("Using synthetic data.")
        dates = pd.date_range('2004-01-01', periods=240, freq='M')
        np.random.seed(42)
        Q = 50 + 30*np.sin(np.arange(240)*2*np.pi/12) + 5*np.random.randn(240)
        EC = 800 + 200*np.sin(np.arange(240)*2*np.pi/6) + 50*np.random.randn(240) + 0.5*Q
        monthly = pd.DataFrame({'Q': Q, 'EC': EC}, index=dates)

    # Create features and scale both X and y
    X_raw, y_raw, dates_raw = create_simple_features(monthly)
    print(f"After creating lags: {len(y_raw)} samples, input dim = {X_raw.shape[1]}")

    # Split before scaling to avoid data leakage
    split = int(0.7 * len(y_raw))
    X_train, X_test = X_raw[:split], X_raw[split:]
    y_train, y_test = y_raw[:split], y_raw[split:]
    dates_train, dates_test = dates_raw[:split], dates_raw[split:]

    # Scale inputs to zero mean and unit variance
    scaler_X = StandardScaler()
    X_train_scaled = scaler_X.fit_transform(X_train)
    X_test_scaled = scaler_X.transform(X_test)

    # Scale target. CRITICAL for convergence, as raw EC values (200-500) are large
    scaler_y = StandardScaler()
    y_train_scaled = scaler_y.fit_transform(y_train.reshape(-1,1)).ravel()
    y_test_scaled = scaler_y.transform(y_test.reshape(-1,1)).ravel()

    # ANFIS + A-DEPSO
    n_inputs = X_train_scaled.shape[1]   # =3
    n_mfs = 2
    print(f"\nBuilding ANFIS with {n_inputs} inputs, {n_mfs} MFs -> {n_mfs**n_inputs} rules")
    anfis = ANFIS(n_inputs=n_inputs, n_mfs=n_mfs)

    # Initialise parameters (using linear regression on scaled y)
    init_params, bounds = anfis.get_param_vector(X_train_scaled, y_train_scaled)

    def fitness(params):
        y_pred_scaled = anfis.forward(X_train_scaled, params)
        return rmse(y_train_scaled, y_pred_scaled)

    print("\nStarting A-DEPSO optimization (3 restarts, 500 iter each)...")
    opt = ADEPSO(dim=len(init_params), bounds=bounds, pop_size=40, max_iter=500)
    best_params, best_rmse_scaled = opt.optimize(fitness, n_restarts=3)
    print(f"Optimization finished. Best training RMSE (scaled): {best_rmse_scaled:.4f}")

    # Training performance (in original scale)
    y_pred_train_scaled = anfis.forward(X_train_scaled, best_params)
    y_pred_train = scaler_y.inverse_transform(y_pred_train_scaled.reshape(-1,1)).ravel()
    train_r2 = r_coeff(y_train, y_pred_train)
    print(f"Training R² (original scale): {train_r2:.4f}")

    # Test predictions
    y_pred_test_scaled = anfis.forward(X_test_scaled, best_params)
    y_pred_test = scaler_y.inverse_transform(y_pred_test_scaled.reshape(-1,1)).ravel()

    # Metrics for ANFIS (original scale)
    metrics_anfis = {
        'R': r_coeff(y_test, y_pred_test),
        'RMSE': rmse(y_test, y_pred_test),
        'MAE': mae(y_test, y_pred_test),
        'RAE': rae(y_test, y_pred_test),
        'MAPE': mape(y_test, y_pred_test),
        'E': legate_mccabe(y_test, y_pred_test),
        'IA': willmott_ia(y_test, y_pred_test)
    }
    all_metrics = [metrics_anfis]

    print("\n=== ANFIS-A-DEPSO Results (final) ===")
    for k, v in metrics_anfis.items():
        print(f"{k}: {v:.4f}")

    # Baseline models (using scaled X, but original y) 
    # Note: baselines use original y (not scaled) to be fair
    def train_eval_baseline(model, X_train, y_train, X_test, y_test):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        return {
            'R': r_coeff(y_test, y_pred),
            'RMSE': rmse(y_test, y_pred),
            'MAE': mae(y_test, y_pred),
            'RAE': rae(y_test, y_pred),
            'MAPE': mape(y_test, y_pred),
            'E': legate_mccabe(y_test, y_pred),
            'IA': willmott_ia(y_test, y_pred)
        }, y_pred

    lssvm = LSSVM(gamma=1992.0)
    metrics_lssvm, _ = train_eval_baseline(lssvm, X_train_scaled, y_train, X_test_scaled, y_test)
    all_metrics.append(metrics_lssvm)
    print("\nLSSVM Results:")
    for k, v in metrics_lssvm.items(): print(f"{k}: {v:.4f}")

    grnn = GRNN(sigma=380.0)
    metrics_grnn, _ = train_eval_baseline(grnn, X_train_scaled, y_train, X_test_scaled, y_test)
    all_metrics.append(metrics_grnn)
    print("\nGRNN Results:")
    for k, v in metrics_grnn.items(): print(f"{k}: {v:.4f}")

    # Performance Index
    pi_value = performance_index(metrics_anfis, all_metrics)
    print(f"\nPerformance Index (PI) for ANFIS-A-DEPSO: {pi_value:.4f}")

    # Plot with dates
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        plt.figure(figsize=(12, 5))
        plt.plot(dates_test, y_test, label='Observed EC', alpha=0.7, color='blue')
        plt.plot(dates_test, y_pred_test, label='ANFIS-A-DEPSO', alpha=0.7, color='orange')
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=12))
        plt.gcf().autofmt_xdate()
        plt.legend()
        plt.title('EC Prediction - USGS 07374000 (Mississippi at Baton Rouge)')
        plt.xlabel('Date')
        plt.ylabel('Specific Conductance (µS/cm)')
        plt.grid(True)
        plt.show()
    except Exception as e:
        print(f"Could not plot: {e}")
