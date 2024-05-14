import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import kpss
import seaborn as sns
import numpy.linalg as LA
from scipy import signal
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

def cal_rolling(col):
    m = []
    v = []
    for i in range(1,col.shape[0]):
        m.append(col.head(i).mean())
        v.append(col.head(i).var())

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot rolling mean in the first subplot
    ax1.plot(m, label='Rolling mean - '+col.name, linestyle='-')
    ax1.set_xlabel('Samples')
    ax1.set_ylabel('Magnitude')
    ax1.set_title('Rolling Mean - '+col.name)
    ax1.legend()

    # Plot rolling variance in the second subplot
    ax2.plot(v, label='Rolling variance - '+col.name, linestyle='-')
    ax2.set_xlabel('Samples')
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Rolling Variance - '+col.name)
    ax2.legend()


    # Show the plot
    plt.tight_layout()
    plt.show()

def ADF_Cal(x):
 result = adfuller(x)
 print("ADF Statistic: %f" %result[0])
 print('p-value: %f' % result[1])
 print('Critical Values:')
 for key, value in result[4].items():
    print('\t%s: %.3f' % (key, value))

def kpss_test(timeseries):
    timeseries = timeseries.fillna(0)
    print('\nKPSS TEST')
    print('NULL Hypothesis: Time series is stationary.')
    print('ALTERNATE Hypothesis: Time series is not stationary.')
    print('Results of KPSS Test:')
    kpsstest = kpss(timeseries, regression='c', nlags="auto")
    kpss_output = pd.Series(kpsstest[0:3], index=['Test Statistic', 'p-value', 'Lags Used'])
    for key, value in kpsstest[3].items():
        kpss_output['Critical Value (%s)' % key] = value
    print(kpss_output)
    if kpss_output[1] < 0.05:
        print('Rejecting the NULL hypothesis with more than 95% confidence interval')
        print('Time series is non-stationary')
    else:
        print('Cannot reject the NULL hypothesis with 95% confidence interval')
        print('Time series is stationary')

def calculate_and_plot_acf(data, max_lag):
    n = len(data)
    acf_values = []

    for k in range(max_lag + 1):
        mean = np.mean(data)
        numerator = np.sum((data[k:] - mean) * (data[:n - k] - mean))
        denominator = np.sum((data - mean) ** 2)
        acf_k = numerator / denominator
        acf_values.append(acf_k)

    # Calculate the confidence interval bounds
    bounds = 1.96 / np.sqrt(max_lag+1)

    # Create a symmetric ACF list by reversing and appending
    acf_values_neg = acf_values[:0:-1] # Avoid duplicate lag 0
    acf_values_total = acf_values_neg + acf_values

    # Create a stem plot for the ACF values
    plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.spines['bottom'].set_color('blue')
    ax.spines['top'].set_color('none')
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('blue')
    plt.stem(range(-max_lag, max_lag + 1), acf_values_total, linefmt='-b', markerfmt='or', basefmt='-b')

    # Create shaded transparent regions for confidence intervals
    plt.fill_between(range(-max_lag, max_lag + 1), -bounds, bounds, color='blue', alpha=0.3,
                     label='Confidence Interval')

    plt.xlabel('Lag')
    plt.ylabel('ACF')
    plt.title('Autocorrelation Function (ACF) for Lags -' + str(max_lag) + ' to ' + str(max_lag))
    plt.grid(True)
    plt.legend()
    plt.show()

    # Return the ACF values as a list
    return acf_values

def cal_autocorr(Y, lags, title, axs=None):  # default value is set to None, i.e. the case when we don't need subplots
    flag = True
    if axs is None:
        axs = plt
        flag = False

    T = len(Y)
    ry = []
    den = 0
    ybar = np.mean(Y)
    for y in Y:  # since denominator is constant for every iteration, we calculate it only once and store it.
        den = den + (y - ybar) ** 2

    for tau in range(lags+1):
        num = 0
        for t in range(tau, T):
            num = num + (Y[t] - ybar) * (Y[t - tau] - ybar)
        ry.append(num / den)

    ryy = ry[::-1]
    Ry = ryy[:-1] + ry  # to make the plot on both sides, reversed the list and added to the original list

    x = np.linspace(-lags, lags, 2 * lags + 1)
    markers, _, _ = axs.stem(x, Ry)
    plt.setp(markers, color='red', marker='o')
    axs.axhspan(-(1.96 / (T ** 0.5)), (1.96 / (T ** 0.5)), alpha=0.2, color='blue')

    if not flag:  # in this case, axs = plt, hence different functions to set xlabel, ylabel, and title
        axs.xlabel('Lags')
        axs.ylabel('Magnitude')
        axs.title(f'Autocorrelation plot of {title}')
        plt.show()
    else:
        axs.set_xlabel('Lags')
        axs.set_ylabel('Magnitude')
        axs.set_title(f'Autocorrelation plot of {title}')
        # in case of axes given i.e. we need subplots, we don't use plt.show() inside this function
        # as it will plot every subplot separately. We need to use plt.show() outside the loop from where
        # the function is called when we need subplots.
    return ry

def calc_val(Ry, J, K):

    den = np.zeros((K, K))

    for k in range(K):
        row = np.zeros(K)
        for i in range(K):
            row[i] = Ry[np.abs(J + k - i)]
        den[k] = row
    # num = den.copy()
    col = np.zeros(K)
    for i in range(K):
        col[i] = Ry[J+i+1]

    num = np.concatenate((den[:, :-1], col.reshape(-1, 1)), axis=1)
    num = np.array(num)
    den = np.array(den)

    if np.linalg.det(den) == 0:
        return np.inf
    if np.abs(np.linalg.det(num)/np.linalg.det(den)) < 0.00001:
        return 0
    return np.linalg.det(num)/np.linalg.det(den)

def cal_gpac(Ry, J=7, K=7):
    gpac_arr = np.zeros((J, K-1))
    for k in range(1, K):
        for j in range(J):
            gpac_arr[j, k-1] = calc_val(Ry, j, k)

    df = pd.DataFrame(gpac_arr, columns=range(1, K), index=range(J))

    plt.figure()
    sns.heatmap(df, annot=True, fmt='0.2f', annot_kws={"size": 6})
    plt.title('Generalized Partial Autocorrelation (GPAC) Table')
    plt.tight_layout()
    plt.show()

    print(df)

def ACF_PACF_Plot(y,lags):
 fig = plt.figure()
 plt.subplot(211)
 plt.title('ACF/PACF of the raw data')
 plot_acf(y, ax=plt.gca(), lags=lags)
 plt.subplot(212)
 plot_pacf(y, ax=plt.gca(), lags=lags)
 fig.tight_layout(pad=3)
 plt.show()

def difference_data(y, order, lag=1):
    if order <= 0:
        return y

    n = len(y)
    diff = [None] * lag  # Initialize with None for missing values
    for i in range(lag, n):
        if None in [y[i - j] for j in range(1, lag + 1)]:
            diff.append(None)
        else:
            diff.append(y[i] - y[i - lag])

    return difference_data(pd.Series(diff), order - 1, lag)

def plot_forecasting_models(ytrain, ytest, yhatTest, title, axs=None):
    if axs is None:
        axs = plt
    x = np.arange(1, len(ytrain)+len(ytest)+1)
    x1 = x[:len(ytrain)]
    x2 = x[len(ytrain):]
    axs.plot(ytrain.index, ytrain, color='r', label='train')
    axs.plot(ytest.index, ytest, color='g', label='test')
    axs.plot(ytest.index, yhatTest, color='b', label='h step')
    # axs.plot(np.arange(len(ytrain)), ytrain, color='r', label='train')
    # axs.plot(np.arange(len(ytrain), len(ytrain)+len(ytest)), ytest, color='g', label='test')
    # axs.plot(np.arange(len(ytrain), len(ytrain)+len(yhatTest)), yhatTest, color='b', label='h step')
    if axs is plt:
        plt.xlabel('Time')
        plt.ylabel('Values')
        plt.title(title)
        plt.legend()
        plt.grid()
        plt.show()
    else:
        axs.set_xlabel('Time')
        axs.set_ylabel('Values')
        axs.set_title(title)
        axs.grid()
        axs.legend()

def cal_error_MSE(y, yhat, skip=0):
    y = np.array(y)
    yhat = np.array(yhat)
    error = []
    error_square = []
    n = len(y)
    for i in range(n):
        if yhat[i] is None:
            error.append(None)
            error_square.append(None)
        else:
            error.append(y[i]-yhat[i])
            error_square.append((y[i]-yhat[i])**2)
    mse = 0
    for i in range(skip, n):
        mse = mse + error_square[i]

    mse = mse/(n-skip)

    return error, error_square, np.round(mse, 2)

def average_forecasting(ytrain, ytest):
    n = len(ytrain)
    yhatTrain = ytrain.copy()
    yhatTest = ytest.copy()
    yhatTrain[0] = None
    for i in range(1, n):
        mean = np.mean(ytrain[0:i])
        yhatTrain[i] = np.round(mean, 2)
        # yhatTrain.append(np.round(mean, 2))

    mean = np.mean(ytrain)
    n = len(ytest)
    for i in range(n):
        yhatTest[i] = np.round(mean, 2)
        # yhatTest.append(np.round(mean, 2))

    return yhatTrain, yhatTest

def Naive_forecasting(xtrain, xtest):
    n = len(xtrain)
    yhatTrain = xtrain.copy()
    yhatTest = xtest.copy()
    yhatTrain[0] = None
    for i in range(1, n):
        yhatTrain[i] = xtrain[i-1]
        # yhatTrain.append(xtrain[i-1])

    yT = xtrain[n-1]
    n = len(xtest)
    for i in range(n):
        yhatTest[i] = yT

    return yhatTrain, yhatTest

def drift_forecasting(xtrain, xtest):
    n = len(xtrain)
    yhatTrain = xtrain.copy()
    yhatTest = xtest.copy()
    yhatTrain[0] = None
    yhatTrain[1] = None
    for i in range(2, n):
        y = xtrain[i-1] + ((xtrain[i-1] - xtrain[0])/(i-1))
        yhatTrain[i] = y

    slope = (xtrain[n-1] - xtrain[0])/(n-1)
    y = xtrain[n-1]
    n = len(xtest)
    for i in range(1, n+1):
        yhat = y + i * slope
        yhatTest[i-1] = yhat

    return yhatTrain, yhatTest

def ses(ytrain, ytest, L0, alpha=0.5):
    n = len(ytrain)
    yhatTrain = ytrain.copy()
    yhatTrain[0] = L0
    for i in range(1, n):
        yhat = alpha * ytrain[i-1] + (1-alpha) * yhatTrain[i-1]
        yhatTrain[i] = yhat

    yhatTest = ytest.copy()
    l0 = alpha * ytrain[n-1] + (1-alpha) * yhatTrain[n-1]

    n = len(ytest)
    for i in range(n):
        yhatTest[i] = l0

    return yhatTrain, yhatTest
#9,0
#17,1

def cal_e(num, den, y):
    system = (num, den, 1)
    _, e = signal.dlsim(system, y)
    return e

def num_den(theta, na, nb):
    theta = theta.ravel()
    num = np.concatenate(([1], theta[:na]))
    den = np.concatenate(([1], theta[na:]))
    max_len = max(len(num), len(den))
    num = np.pad(num, (0, max_len - len(num)), 'constant')
    den = np.pad(den, (0, max_len - len(den)), 'constant')
    return num, den


def cal_gradient_hessian(y, e, theta, na, nb):
    delta = 0.000001
    X = np.empty((len(e), 0))
    for i in range(len(theta)):
        temp_theta = theta.copy()
        temp_theta[i] = temp_theta[i] + delta
        num, den = num_den(temp_theta, na, nb)
        e_new = cal_e(num, den, y)
        x_temp = (e - e_new)/delta
        X = np.hstack((X, x_temp))

    # A = X.T @ X
    # g = X.T @ e
    A = np.dot(X.T, X)
    g = np.dot(X.T, e)
    return A, g
def SSE(theta, y, na, nb):
    num, den = num_den(theta, na, nb)
    e = cal_e(num, den, y)
    return np.dot(e.T, e)

def LM(y, na, nb):
    epoch = 0
    epochs = 50
    theta = np.zeros(na + nb)
    mu = 0.01
    n = len(theta)
    N = len(y)
    mu_max = 1e+20
    sse_array = []
    while epoch < epochs:
        sse_array.append(SSE(theta, y, na, nb).ravel())
        num, den = num_den(theta, na, nb)
        e = cal_e(num, den, y)
        A, g = cal_gradient_hessian(y, e, theta, na, nb)
        del_theta = np.linalg.inv(A + mu*np.identity(A.shape[0])) @ g
        theta_new = theta.reshape(-1, 1) + del_theta
        sse_new = SSE(theta_new.ravel(), y, na, nb)
        sse_old = SSE(theta.ravel(), y, na, nb)
        if sse_new[0][0] < sse_old[0][0]:
            if LA.norm(del_theta) < 1e-3:
                theta_hat = theta_new.copy()
                sse_array.append(SSE(theta_new, y, na, nb).ravel())
                variance_hat = SSE(theta_new.ravel(), y, na, nb)/(N-n)
                covariance_hat = variance_hat * np.linalg.inv(A)
                return theta_hat, variance_hat, covariance_hat, sse_array
            else:
                mu = mu/10
        while SSE(theta_new.ravel(), y, na, nb) >= SSE(theta.ravel(), y, na, nb):
            mu = mu*10
            # theta = theta_new.copy()
            if mu > mu_max:
                print('Error')
                break
            del_theta = np.linalg.inv(A + mu * np.identity(A.shape[0])) @ g
            theta_new = theta.reshape(-1, 1) + del_theta
        epoch += 1
        theta = theta_new.copy()
    return

def cal_Q_value(y, title, lags=5):
    # title = 'Average forecasting train data'
    acf = cal_autocorr(y, lags, title)
    sum_rk = 0
    T = len(y)
    for i in range(1, lags+1):
        sum_rk += acf[i]**2
    Q = T * sum_rk
    # if Q < Q* then white residual
    return Q