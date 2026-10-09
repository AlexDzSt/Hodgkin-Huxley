"""
Port a Python de hhrun.m (modelo de Hodgkin-Huxley, axón gigante de calamar).

Mantiene la estructura del código MATLAB original:
  - Euler hacia adelante con dt = 0.001 ms
  - V se maneja en escala absoluta (reposo = -65 mV) y las tasas / fuerzas
    impulsoras usan (V + 65), igual que en el original
  - eNa = 115, eK = -12, eL = 10.6, gNa = 120, gK = 36, gL = 0.3

Diferencia respecto al original: la corriente es un pulso I(t) (en lugar de
una constante) para reproducir la figura de la tarea. Al graficar se resta el
offset de 65 mV para que el reposo quede en 0 mV, como en la imagen.
"""
import numpy as np
import matplotlib.pyplot as plt


# ---------- funciones alpha / beta (idénticas a las del .m) ----------
def alpha_m(V):
    x = 2.5 - 0.1 * (V + 65)
    return x / (np.exp(x) - 1) if abs(x) > 1e-9 else 1.0   # límite x -> 0

def beta_m(V):
    return 4 * np.exp(-(V + 65) / 18)

def alpha_h(V):
    return 0.07 * np.exp(-(V + 65) / 20)

def beta_h(V):
    return 1.0 / (np.exp(3.0 - 0.1 * (V + 65)) + 1)

def alpha_n(V):
    x = 0.1 - 0.01 * (V + 65)
    return x / (np.exp(1 - 0.1 * (V + 65)) - 1) if abs(x) > 1e-9 else 0.1  # límite

def beta_n(V):
    return 0.125 * np.exp(-(V + 65) / 80)


# ---------- simulación ----------
def hhrun(I, tspan, v, mi, hi, ni, plot=False, dt=0.001):
    """
    I     : float (corriente constante) o función I(t) -> corriente
    tspan : duración de la simulación [ms]
    v, mi, hi, ni : condiciones iniciales de V, m, h, n
    Devuelve V, m, h, n, t (arrays) y el vector de corriente Iv.
    """
    gNa, eNa = 120.0, 115.0
    gK, eK = 36.0, -12.0
    gL, eL = 0.3, 10.6

    loop = int(np.ceil(tspan / dt))
    t = (np.arange(1, loop + 1)) * dt
    V = np.zeros(loop)
    m = np.zeros(loop)
    h = np.zeros(loop)
    n = np.zeros(loop)
    Iv = np.zeros(loop)

    V[0], m[0], h[0], n[0] = v, mi, hi, ni
    I_fun = I if callable(I) else (lambda _t: I)

    # Euler hacia adelante
    for i in range(loop - 1):
        Iv[i] = I_fun(t[i])
        V[i + 1] = V[i] + dt * (gNa * m[i] ** 3 * h[i] * (eNa - (V[i] + 65))
                                + gK * n[i] ** 4 * (eK - (V[i] + 65))
                                + gL * (eL - (V[i] + 65))
                                + Iv[i])
        m[i + 1] = m[i] + dt * (alpha_m(V[i]) * (1 - m[i]) - beta_m(V[i]) * m[i])
        h[i + 1] = h[i] + dt * (alpha_h(V[i]) * (1 - h[i]) - beta_h(V[i]) * h[i])
        n[i + 1] = n[i] + dt * (alpha_n(V[i]) * (1 - n[i]) - beta_n(V[i]) * n[i])
    Iv[-1] = I_fun(t[-1])

    if plot:
        graficar(t, V, m, h, n, Iv)
    return V, m, h, n, t, Iv


# ---------- gráficas ----------
def graficar(t, V, m, h, n, Iv, offset=65.0, guardar="hhrun.png"):
    fig, axs = plt.subplots(3, 1, figsize=(8, 8), sharex=True)

    axs[0].plot(t, V + offset, "b", lw=2)           # reposo en 0 mV
    axs[0].set_ylabel("v [mV]")
    axs[0].set_ylim(-20, 120)

    axs[1].plot(t, m, "k", lw=2, label="m")
    axs[1].plot(t, n, "b", lw=2, label="n")
    axs[1].plot(t, h, "r", lw=2, label="h")
    axs[1].set_ylabel("act./inact.")
    axs[1].set_ylim(0, 1)
    axs[1].legend(loc="center right")

    axs[2].plot(t, Iv, "b", lw=2)
    axs[2].set_ylabel("I [micro A]")
    axs[2].set_ylim(0, 8)
    axs[2].set_xlabel("t [ms]")

    for ax in axs:
        ax.set_xlim(0, t[-1])
        ax.grid(True, ls=":", color="gray", alpha=0.7)
    plt.tight_layout()
    if guardar:
        fig.savefig(guardar, dpi=150)
    return fig


# ---------- ejecución ----------
if __name__ == "__main__":
    I_AMP = 6.7                 # uA/cm^2
    T_ON, T_OFF = 10.0, 46.0    # ms

    def pulso(t):
        return I_AMP if T_ON <= t < T_OFF else 0.0

    # Condiciones iniciales: estado estacionario en reposo (V = -65 mV)
    V0, m0, h0, n0 = -65.0, 0.0529, 0.5961, 0.3177

    V, m, h, n, t, Iv = hhrun(pulso, 70, V0, m0, h0, n0, plot=True)
    plt.show()