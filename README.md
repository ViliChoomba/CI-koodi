INSTRUCTIONS FOR THE CODE

This code implements a hybrid ANFIS + A‑DEPSO (Adaptive Neuro‑Fuzzy Inference System + Adaptive Hybrid Differential Evolution – Particle Swarm Optimization) model to predict electrical conductivity (EC) of river water using real USGS data.

For practicality, this implementation omits the wavelet decomposition step used in the original paper, but the core ANFIS‑A‑DEPSO optimisation is identical 

The code replicates the core methodology described in the paper we were assigned.


*Dataset Source:*

Real‑time historical data is downloaded automatically from the USGS National Water Information System using the dataretrieval Python package.

Station: USGS 07374000 – Mississippi River at Baton Rouge, Louisiana.

Parameters and their codes for filtering:
00060 – Daily mean discharge (Q) [cfs]
00095 – Specific conductance (EC) [µS/cm]
Time range: `2004‑01‑01` to `2024‑05‑19` (adjustable in the code).
Aggregation: Daily values are resampled to monthly means.

You can choose a different station by changing the site variable inside the script and putting the correspondent station code.


*Required Libraries & Software*

Software:
Python 3.8 or higher
Any text editor / IDE (VS Code recommended)
Working internet connection

Python Libraries
Install the following packages using `pip` in the terminal:

pip install numpy pandas scikit-learn matplotlib dataretrieval


*Libraries and their purposes:*

numpy
numerical arrays and math ops

pandas
time series handling, data cleaning and merging

scikit-learning
metrics (R2, RMSE, MAE) and scaling (StandardScaler)

matplotlib
plotting observed vs predicted EC

dataretrieval
direct download of USGS water data


*How to Run the Code*

Step 1 – Save the script
Take the code and paste it into VS Code or equivalent.

Step 2 – Install dependencies
Open a terminal and run:
pip install numpy pandas scikit-learn matplotlib dataretrieval

Step 3 – Execute the script
Run the code in VS Code from atop the page of VS Code

Alternatively press F5 or paste the following code into the terminal where filename is the name of the file you have the code saved as:

python filename.py


*Troubleshooting*

Here is the list of errors you might encounter and how to fix:


MemoryError : Reduce n_mfs or number of inputs. The default is 3 inputs + 2 MFs → 8 rules, which is safe.  

ModuleNotFoundError : No module named 'dataretrieval' :Install it: pip install dataretrieval. The script will fall back to synthetic data if not installed. 

Very poor predictions (flat line) : Increase max_iter to 800 or n_restarts to 5. Also check that you are using real USGS data (not synthetic). 

Plot does not appear : Ensure matplotlib is installed. On some headless systems you may need to add plt.show() – already present. 

USGS data download fails : The USGS service may be temporarily unavailable. Wait a few minutes and retry, or use synthetic data by commenting out the real‑data block. 


