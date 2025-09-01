import numpy as np
import scipy.integrate as spi

def integrate(y, dx):
    #  return np.sum(y) * dx
    return spi.trapezoid(y, dx=dx)

def calc_pdf(data):
    """
    Calculate the PDF of the data using the sine transform.

    Parameters
    ----------
    data : np.ndarray
        The data to calculate the PDF of. Can be directly read from a file using np.loadtxt.
        Atomic units are assumed.

    Returns
    -------
    r : np.ndarray
        The r values of the PDF.
    pdf : np.ndarray
        The PDF values.
    V_ee_int : np.ndarray
        The integrand of the electron-electron interaction energy.
    
    """
    q = data[:, 0]
    dq = data[1, 0] - data[0, 0]
    no_points = len(q)
    r = np.linspace(0, 1 / dq, no_points)
    pdf = np.zeros_like(r)
    for i, rr in enumerate(r):
        integrand = data[:, 1] * np.sin(rr * q) * q
        pdf[i] = 1 / np.pi * rr * integrate(integrand, dq)
    integral = integrate(pdf, r[1] - r[0])
    print(f"Integral = {integral:10.8f}")

    V_ee_int = np.zeros_like(pdf)
    V_ee_int[1:] = pdf[1:] / r[1:]
    V_ee = integrate(V_ee_int, r[1] - r[0])
    print(f"V_ee     = {V_ee:10.8f}")
    return r, pdf, V_ee_int, V_ee
