import streamlit as st
from scipy import signal
import pandas as pd
import numpy as np
import scipy
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.optimize import curve_fit


def michaelis_menten(x, DC, Vmax, n, Km):
    return DC + Vmax * (x**n) / (x**n + Km**n)


def butterworth_filter(data, cutoff, fs, order=4, btype='low'):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype, analog=False)
    y = filtfilt(b, a, data)
    return y


def butter_highpass(cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=5):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = filtfilt(b, a, data)
    return y


uploaded_acc_iTUG = st.file_uploader(
    "Carregue o arquivo de texto do acelerômetro", type=["txt"],)
if uploaded_acc_iTUG is not None:
    # Allocation of the acceleration data to the variables
    if uploaded_acc_iTUG is not None:
        custom_separator = ';'
        df = pd.read_csv(uploaded_acc_iTUG, sep=custom_separator)
        t = df.iloc[:, 0]
        x = df.iloc[:, 1]
        y = df.iloc[:, 2]
        z = df.iloc[:, 3]
        time = t

        # Pre-processing data: All channels were detrended, normalized to gravity acceleration, and interpolated to 100 Hz
        if np.max(x) > 9 or np.max(y) > 9 or np.max(z) > 9:
            x = signal.detrend(x/9.81)
            y = signal.detrend(y/9.81)
            z = signal.detrend(z/9.81)
        else:
            x = signal.detrend(x)
            y = signal.detrend(y)
            z = signal.detrend(z)
        interpf = scipy.interpolate.interp1d(time, x)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        x_ = interpf(time_)
        t, x = time_/1000, x_
        interpf = scipy.interpolate.interp1d(time, y)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        y_ = interpf(time_)
        t, y = time_/1000, y_
        interpf = scipy.interpolate.interp1d(time, z)
        time_ = np.arange(start=time[0], stop=time[len(time)-1], step=10)
        z_ = interpf(time_)
        t, z = time_/1000, z_

        # Calculating acceleration data norm
        norm_waveform = np.sqrt(x**2+y**2+z**2)

        # Filtering acceleration data norm
        f1 = 0.5
        f2 = 100
        x_2 = butterworth_filter(
            x, f1, f2, order=2, btype='low')
        y_2 = butterworth_filter(
            y, f1, f2, order=2, btype='low')
        z_2 = butterworth_filter(
            z, f1, f2, order=2, btype='low')

        norm_waveform_2 = butterworth_filter(
            norm_waveform, f1, f2, order=2, btype='low')

        lim1 = 100

        m = np.max(norm_waveform_2)
        for index, M in enumerate(norm_waveform_2):
            if M == m:
                peak_index = index
                break

        # modelo do fim
        x_data = t[peak_index:len(t)-lim1]-t[peak_index]
        y_data = norm_waveform_2[peak_index:len(t)-lim1]
        y_data = y_data[::-1]
        # Ajustando a função de Michaelis-Menten aos dados
        # Suposições iniciais para os parâmetros Vmax e Km
        initial_guess = [0.1, 0.3, 9, 3]
        params, covariance = curve_fit(
            michaelis_menten, x_data, y_data, p0=initial_guess)

        # Obtendo os parâmetros ajustados
        DC_fit, Vmax_fit, n_fit, Km_fit = params

        # Gerando curva ajustada
        y_fit = michaelis_menten(x_data, DC_fit, Vmax_fit, n_fit, Km_fit)
        y_fit = y_fit[::-1]
        x_data = x_data+t[peak_index]

        # modelo do começo
        x_data2 = t[lim1:peak_index]-t[lim1]
        y_data2 = norm_waveform_2[lim1:peak_index]
        # Ajustando a função de Michaelis-Menten aos dados
        # Suposições iniciais para os parâmetros Vmax e Km
        initial_guess = [0.1, 0.3, 9, 3]
        params, covariance = curve_fit(
            michaelis_menten, x_data2, y_data2, p0=initial_guess)

        # Obtendo os parâmetros ajustados
        DC_fit2, Vmax_fit2, n_fit2, Km_fit2 = params

        # Gerando curva ajustada
        y_fit2 = michaelis_menten(x_data2, DC_fit2, Vmax_fit2, n_fit2, Km_fit2)

        x_data2 = x_data2+t[lim1]

        for index, i in enumerate(y_fit):
            if i < np.max(y_fit)*0.25:
                t2_10 = index
                break
        for index, i in enumerate(y_fit):
            if i < np.max(y_fit)*0.75:
                t2_90 = index
                break
        for index, i in enumerate(y_fit):
            if i < np.max(y_fit)*0.5:
                t2_50 = index
                break

        for index, i in enumerate(y_fit2):
            if i > np.max(y_fit2)*0.25:
                t1_10 = index
                break
        for index, i in enumerate(y_fit2):
            if i > np.max(y_fit2)*0.75:
                t1_90 = index
                break
        for index, i in enumerate(y_fit2):
            if i > np.max(y_fit2)*0.5:
                t1_50 = index
                break

        plt.figure(figsize=(5, 3))
        plt.plot(t, norm_waveform, 'y')
        plt.plot(t, norm_waveform_2, 'k')
        plt.plot(x_data, y_fit, 'b')
        plt.plot(x_data2, y_fit2, 'r')
        plt.plot([x_data[t2_10], x_data[t2_10]], [0, 0.5], '--k')
        plt.plot([x_data[t2_90], x_data[t2_90]], [0, 0.5], '--k')
        plt.plot([x_data2[t1_10], x_data2[t1_10]], [0, 0.5], '--k')
        plt.plot([x_data2[t1_90], x_data2[t1_90]], [0, 0.5], '--k')
        st.pyplot(plt)

        amplitude_maxima_de_aceleracao = Vmax_fit2
        st.text("Amplitude máxima de aceleração (g) = " +
                str(round(amplitude_maxima_de_aceleracao, 2)))

        amplitude_maxima_de_desaceleracao = Vmax_fit
        st.text("Amplitude máxima de desaceleração (g) = " +
                str(round(amplitude_maxima_de_desaceleracao, 2)))

        tempo_de_aceleracao = x_data2[t1_90]-x_data2[t1_10]
        st.text("Tempo de aceleração (s) = " +
                str(round(tempo_de_aceleracao, 2)))

        tempo_de_desaceleracao = x_data[t2_10]-x_data[t2_90]
        st.text("Tempo de desaceleração (s) = " +
                str(round(tempo_de_desaceleracao, 2)))

        tempo_de_aceleracao_constante = x_data[t2_90]-x_data2[t1_90]
        st.text("Tempo de aceleração constante (s) = " +
                str(round(tempo_de_aceleracao_constante, 2)))

        ganho_de_aceleracao = y_fit2[t1_50]/(x_data2[t1_50]-x_data2[lim1])
        st.text("Ganho de aceleração (g/s) = " +
                str(round(ganho_de_aceleracao, 4)))

        ganho_de_desaceleracao = y_fit[t2_50]/(x_data[t2_50]-t[peak_index])
        st.text("Ganho de desaceleração (g/s) = " +
                str(round(ganho_de_desaceleracao, 4)))

        st.text("Razão de ganhos = " +
                str(round(ganho_de_aceleracao/ganho_de_desaceleracao, 4)))

        st.text("Tempo total de caminhada (s) = " +
                str(round(x_data[t2_10]-x_data2[t1_10], 2)))

        st.text("Velocidade média de caminhada (m/s) = " +
                str(round(4/(x_data[t2_10]-x_data2[t1_10]), 2)))
