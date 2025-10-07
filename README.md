
# Ex.No:04   FIT ARMA MODEL FOR TIME SERIES
# Date: 07-10-25



### AIM:
To implement ARMA model in python.
### ALGORITHM:
1. Import necessary libraries.
2. Set up matplotlib settings for figure size.
3. Define an ARMA(1,1) process with coefficients ar1 and ma1, and generate a sample of 1000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

4. Display the autocorrelation and partial autocorrelation plots for the ARMA(1,1) process using
plot_acf and plot_pacf.
5. Define an ARMA(2,2) process with coefficients ar2 and ma2, and generate a sample of 10000

data points using the ArmaProcess class. Plot the generated time series and set the title and x-
axis limits.

6. Display the autocorrelation and partial autocorrelation plots for the ARMA(2,2) process using
plot_acf and plot_pacf.
### PROGRAM:

    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.arima_process import ArmaProcess
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    data=pd.read_csv(r"C:\Users\admin\OneDrive\Desktop\sem 3\time series\AirPassengers.csv")
    N=1000
    plt.rcParams['figure.figsize'] = [12, 6] #plt.rcParams is a dictionary-like object in Mat
    X=data['#Passengers']
    plt.plot(X)
    plt.title('Original Data')
    plt.show()
    plt.subplot(2, 1, 1)
    plot_acf(X, lags=len(X)/4, ax=plt.gca())
    plt.title('Original Data ACF')
    plt.subplot(2, 1, 2)
    plot_pacf(X, lags=len(X)/4, ax=plt.gca())
    plt.title('Original Data PACF')
    plt.tight_layout()
    plt.show()
    arma11_model = ARIMA(X, order=(1, 0, 1)).fit()
    phi1_arma11 = arma11_model.params['ar.L1']
    theta1_arma11 = arma11_model.params['ma.L1']
    ar1 = np.array([1, -phi1_arma11])
    ma1 = np.array([1, theta1_arma11])
    ARMA_1 = ArmaProcess(ar1, ma1).generate_sample(nsample=N)
    plt.plot(ARMA_1)
    plt.title('Simulated ARMA(1,1) Process')
    plt.xlim([0, 500])
    plt.show()
    plot_acf(ARMA_1)
    plt.show()
    plot_pacf(ARMA_1)
    plt.show()
    arma22_model = ARIMA(X, order=(2, 0, 2)).fit()
    phi1_arma22 = arma22_model.params['ar.L1']
    phi2_arma22 = arma22_model.params['ar.L2']
    theta1_arma22 = arma22_model.params['ma.L1']
    theta2_arma22 = arma22_model.params['ma.L2']
    ar2 = np.array([1, -phi1_arma22, -phi2_arma22])
    ma2 = np.array([1, theta1_arma22, theta2_arma22])
    ARMA_2 = ArmaProcess(ar2, ma2).generate_sample(nsample=N*10)
    plt.plot(ARMA_2)
    plt.title('Simulated ARMA(2,2) Process')
    plt.xlim([0, 500])
    plt.show()
    plot_acf(ARMA_2)
    plt.show()
    plot_pacf(ARMA_2)
    plt.show()



### OUTPUT:

**Original data**

<img width="1316" height="647" alt="Screenshot 2025-09-22 152545" src="https://github.com/user-attachments/assets/da7af8aa-0bf6-4340-8481-031fa76f3c3c" />

**Original data ACF**

<img width="1376" height="351" alt="Screenshot 2025-09-22 152607" src="https://github.com/user-attachments/assets/d6eafc3c-1ddb-448b-9b6c-8ca2adefd1c2" />

**Original data PACF**

<img width="1367" height="334" alt="Screenshot 2025-09-22 152625" src="https://github.com/user-attachments/assets/2e49c048-7fa0-40e8-ac9a-00a320b3ecc2" />

**SIMULATED ARMA(1,1) PROCESS:**

<img width="1332" height="677" alt="Screenshot 2025-09-22 152638" src="https://github.com/user-attachments/assets/65bb3fff-7f29-42d7-9c08-11fd905d695f" />


**Partial Autocorrelation**

<img width="1307" height="688" alt="Screenshot 2025-09-22 152707" src="https://github.com/user-attachments/assets/5782359d-6cb2-4ca2-9b11-efb8cca9549c" />

**Autocorrelation**


<img width="1318" height="677" alt="Screenshot 2025-09-22 152652" src="https://github.com/user-attachments/assets/ff8eff2b-0898-4ce1-b92f-99c8035cf116" />

**SIMULATED ARMA(2,2) PROCESS:**

<img width="1339" height="660" alt="Screenshot 2025-09-22 152723" src="https://github.com/user-attachments/assets/23196b9d-b0e8-41cb-bfd2-19ff25cdcb88" />

**Partial Autocorrelation**


<img width="1319" height="700" alt="Screenshot 2025-09-22 152747" src="https://github.com/user-attachments/assets/2e585ea8-4af8-4bd5-9837-5a8787c2c1b2" />

**Autocorrelation**

<img width="1319" height="660" alt="Screenshot 2025-09-22 152735" src="https://github.com/user-attachments/assets/fc71ee0e-ee7a-4e62-ab2b-be30050c76ee" />

RESULT:
Thus, a python program is created to fir ARMA Model successfully.
