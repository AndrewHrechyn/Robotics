"""
Рівняння руху (Лагранж):
    (M+m)·ẍ + m·l·θ̈·cosθ − m·l·θ̇²·sinθ + b·ẋ = F(t)
    m·l²·θ̈ + m·l·ẍ·cosθ  − m·g·l·sinθ   + c·θ̇ = 0
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation

G = 9.81


def runge_kutta(fnc, a, b, h, y0):

    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    dim = len(y0)

    n = int((b - a) / h) + 1

    result = np.zeros((n, 1 + dim))
    result[0, 0]  = a
    result[0, 1:] = y0

    for i in range(1, n):
        x_prev = result[i - 1, 0]
        y_prev = result[i - 1, 1:]

        k1 = h * fnc(x_prev, y_prev)
        k2 = h * fnc(x_prev + h / 2, y_prev + k1 / 2)
        k3 = h * fnc(x_prev + h / 2, y_prev + k2 / 2)
        k4 = h * fnc(x_prev + h, y_prev + k3)

        dy = 1/6 * (k1 + 2 * k2 + 2 * k3 + k4)

        result[i, 0]  = x_prev + h
        result[i, 1:] = y_prev + dy

    return result


def make_cartpole_rhs(M, m, l, b, c, F_func):
    def rhs(t, state):
        x, dx, th, dth = state
        s, co = np.sin(th), np.cos(th)
        A = np.array([[M + m, m * l * co],
                         [m * l * co, m * l**2  ]])
        rhs_ = np.array([F_func(t) - b * dx + m * l * dth**2 * s,
                         m * G * l * s - c * dth])
        ddx, ddth = np.linalg.solve(A, rhs_)
        return np.array([dx, ddx, dth, ddth])
    return rhs

def simulate(M=1.0, m=0, l=0.8, b=0.1, c=0.01,
             F_func=None,
             x0=0.0, dx0=0.0, th0_deg=10.0, dth0=0.0,
             T=10.0, h=0.005):

    if F_func is None:
        F_func = lambda t: 0.0

    rhs = make_cartpole_rhs(M, m, l, b, c, F_func)
    y0 = np.array([x0, dx0, np.deg2rad(th0_deg), dth0])
    result = runge_kutta(rhs, a=0.0, b=T, h=h, y0=y0)

    t = result[:, 0]
    x = result[:, 1]
    dx = result[:, 2]
    th = result[:, 3]
    dth = result[:, 4]

    return t, x, dx, th, dth

def plot_results(t, x, th, title="", save_path=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle(f"Обернений маятник{('  —  ' + title) if title else ''}",
                 fontsize=12, fontweight='bold')

    ax1.plot(t, x, color='#1a6fc4', lw=1.6)
    ax1.axhline(0, color='#aaa', lw=0.7, ls='--')
    ax1.set_xlabel("t (с)"); ax1.set_ylabel("x (м)")
    ax1.set_title("Позиція візка x(t)")
    ax1.grid(alpha=0.3)

    ax2.plot(t, np.rad2deg(th), color='#c04828', lw=1.6)
    ax2.axhline(0, color='#aaa', lw=0.7, ls='--')
    ax2.set_xlabel("t (с)"); ax2.set_ylabel("θ (°)")
    ax2.set_title("Кут маятника θ(t)")
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Збережено: {save_path}")
    plt.show()

def visualize(t, x, th, l=0.8, title="", save_gif=None):
    skip = max(1, len(t) // 600)
    t_a, x_a, th_a = t[::skip], x[::skip], th[::skip]

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.suptitle(f"Анімація{(': ' + title) if title else ''}",
                 fontsize=11, fontweight='bold')

    x_margin = max(1.2, np.max(np.abs(x)) + l + 0.3)
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.5, l + 0.5)
    ax.set_aspect('equal')
    ax.set_facecolor('#f7f6f2')
    ax.set_xlabel("x (м)"); ax.set_ylabel("y (м)")
    ax.axhline(0, color='#888', lw=2.5, zorder=1)
    ax.fill_between([-x_margin, x_margin], -0.5, 0,
                    color='#ddd', alpha=0.5, zorder=0)

    cart_w, cart_h = 0.35, 0.18
    cart_patch = mpatches.FancyBboxPatch(
        (-cart_w/2, -cart_h), cart_w, cart_h,
        boxstyle="round,pad=0.02",
        fc='#2a5fa8', ec='#1a3f78', lw=1.2, zorder=4)
    ax.add_patch(cart_patch)

    wL = plt.Circle((-cart_w/2 + 0.06, -cart_h - 0.045), 0.045,
                    fc='#555', ec='#333', lw=0.8, zorder=5)
    wR = plt.Circle(( cart_w/2 - 0.06, -cart_h - 0.045), 0.045,
                    fc='#555', ec='#333', lw=0.8, zorder=5)
    ax.add_patch(wL); ax.add_patch(wR)

    rod,  = ax.plot([], [], lw=3, color='#3d3d3a', zorder=3,
                    solid_capstyle='round')
    traj, = ax.plot([], [], lw=0.8, color='#c04828', alpha=0.3, zorder=2)
    bob   = plt.Circle((0, l), 0.07, fc='#c04828', ec='#801f0f',
                        lw=1.2, zorder=6)
    ax.add_patch(bob)
    tx, ty = [], []

    info = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9,
                   va='top',
                   bbox=dict(fc='white', alpha=0.8, ec='#ccc',
                             boxstyle='round,pad=0.3'))

    def init():
        rod.set_data([], []); traj.set_data([], [])
        return cart_patch, rod, bob, wL, wR, traj

    def update(i):
        xi, thi = x_a[i], th_a[i]
        xp = xi + l * np.sin(thi)
        yp = l * np.cos(thi)

        cart_patch.set_x(xi - cart_w / 2)
        wL.center = (xi - cart_w/2 + 0.06, -cart_h - 0.045)
        wR.center = (xi + cart_w/2 - 0.06, -cart_h - 0.045)
        rod.set_data([xi, xp], [0, yp])
        bob.center = (xp, yp)

        tx.append(xp); ty.append(yp)
        traj.set_data(tx[-150:], ty[-150:])
        info.set_text(f"t = {t_a[i]:.2f} s\n"
                      f"x = {xi:.3f} m\n"
                      f"θ = {np.rad2deg(thi):.1f}°")
        return cart_patch, rod, bob, wL, wR, traj, info

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_a),
        init_func=init, blit=True, interval=33)

    if save_gif:
        ani.save(save_gif, writer='pillow', fps=30)
        print(f"  Збережено: {save_gif}")
    plt.tight_layout()
    plt.show()
    return ani

if __name__ == "__main__":

    t, x, dx, th, dth = simulate(th0_deg=10, T=10)

    plot_results(t, x, th, title="θ₀=10°", save_path="plots.png")

    visualize(t, x, th, l=0.8, title="θ₀=10°", save_gif="animation.gif")
