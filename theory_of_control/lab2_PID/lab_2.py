"""
    (M+m)·ẍ + m·l·θ̈·cosθ − m·l·θ̇²·sinθ + b·ẋ = F(t)
    m·l²·θ̈ + m·l·ẍ·cosθ  − m·g·l·sinθ   + c·θ̇ = 0

    e(t)  = θ_ref − θ(t)      (θ_ref = 0)
    F(t)  = Kp·e + Ki·∫e dt + Kd·ė
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
import matplotlib.animation as animation

G = 9.81


def runge_kutta(fnc, a, b, h, y0):
    y0 = np.atleast_1d(np.asarray(y0, dtype=float))
    n = int((b - a) / h) + 1
    dim = len(y0)
    result = np.zeros((n, 1 + dim))
    result[0, 0] = a
    result[0, 1:] = y0

    for i in range(1, n):
        xp = result[i - 1, 0]
        yp = result[i - 1, 1:]
        k1 = h * fnc(xp, yp)
        k2 = h * fnc(xp + h / 2, yp + k1 / 2)
        k3 = h * fnc(xp + h / 2, yp + k2 / 2)
        k4 = h * fnc(xp + h, yp + k3)
        result[i, 0]  = xp + h
        result[i, 1:] = yp + (k1 + 2*k2 + 2*k3 + k4) / 6

    return result



class PIDController:

    def __init__(self, Kp, Ki, Kd,
                 setpoint=0.0,
                 F_max=50.0,
                 integral_limit=10.0):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.F_max = F_max
        self.integral_limit = integral_limit

        self._integral = 0.0
        self._prev_error = None

    def reset(self):
        self._integral = 0.0
        self._prev_error = None

    def compute(self, measurement, dt):
        error = self.setpoint - measurement

        self._integral += error * dt
        self._integral = np.clip(self._integral,
                                  -self.integral_limit, self.integral_limit)

        if self._prev_error is None:
            derivative = 0.0
        else:
            derivative = (error - self._prev_error) / dt
        self._prev_error = error

        F = self.Kp * error + self.Ki * self._integral + self.Kd * derivative
        return float(np.clip(F, -self.F_max, self.F_max))



def make_rhs(M, m, l, b, c, F_func):
    def rhs(t, state):
        x, dx, th, dth = state
        s, co = np.sin(th), np.cos(th)
        A = np.array([[M + m,      m * l * co],
                      [m * l * co, m * l**2  ]])
        rhs_ = np.array([F_func(t) - b * dx - m * l * dth ** 2 * s,
                         m * G * l * s - c * dth])
        ddx, ddth = np.linalg.solve(A, rhs_)
        return np.array([dx, ddx, dth, ddth])
    return rhs



def simulate_pid(M=1.0, m=0.2, l=0.8, b=0.1, c=0.01,
                 Kp=50.0, Ki=1.0, Kd=10.0,
                 th0_deg=15.0, dth0=0.0,
                 x0=0.0, dx0=0.0,
                 T=10.0, h=0.005):

    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0.0)

    n = int(T / h) + 1

    t_arr = np.zeros(n)
    x_arr = np.zeros(n)
    dx_arr = np.zeros(n)
    th_arr = np.zeros(n)
    dth_arr = np.zeros(n)
    F_arr = np.zeros(n)

    state = np.array([x0, dx0, np.deg2rad(th0_deg), dth0])
    t_arr[0] = 0.0
    x_arr[0] = state[0]
    dx_arr[0] = state[1]
    th_arr[0] = state[2]
    dth_arr[0] = state[3]
    F_arr[0] = 0.0

    for i in range(1, n):
        t_cur = t_arr[i - 1]
        F_cur = pid.compute(state[2], h)
        F_arr[i] = F_cur

        def F_func(t):
            return F_cur

        rhs = make_rhs(M, m, l, b, c, F_func)

        k1 = h * rhs(t_cur, state)
        k2 = h * rhs(t_cur + h / 2, state + k1 / 2)
        k3 = h * rhs(t_cur + h / 2, state + k2 / 2)
        k4 = h * rhs(t_cur + h, state + k3)
        state = state + (k1 + 2*k2 + 2*k3 + k4) / 6

        t_arr[i] = t_cur + h
        x_arr[i] = state[0]
        dx_arr[i] = state[1]
        th_arr[i] = state[2]
        dth_arr[i] = state[3]

    return t_arr, x_arr, dx_arr, th_arr, dth_arr, F_arr



def plot_results(t, x, dx, th, dth, F, title="", save_path=None,
                 time_hist=None, theta_hist=None, dtheta_hist=None,
                 x_hist=None, F_hist=None):
    _t = time_hist if time_hist is not None else t
    _th = theta_hist if theta_hist is not None else th
    _dth = dtheta_hist if dtheta_hist is not None else dth
    _x = x_hist if x_hist is not None else x
    _F = F_hist if F_hist is not None else F

    print("Побудова графіків динаміки...")
    fig2, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig2.suptitle('Аналіз динаміки оберненого маятника', fontsize=16)

    axs[0, 0].plot(_t, _th, 'r-', lw=2)
    axs[0, 0].set_title('Кут маятника $\\theta(t)$')
    axs[0, 0].set_xlabel('Час (с)')
    axs[0, 0].set_ylabel('Кут (рад)')
    axs[0, 0].grid(True)
    axs[0, 0].axhline(0, color='black', linestyle='--', lw=1)

    axs[0, 1].plot(_t, _dth, 'm-', lw=2)
    axs[0, 1].set_title('Кутова швидкість $\\dot{\\theta}(t)$')
    axs[0, 1].set_xlabel('Час (с)')
    axs[0, 1].set_ylabel('Швидкість (рад/с)')
    axs[0, 1].grid(True)
    axs[0, 1].axhline(0, color='black', linestyle='--', lw=1)

    axs[1, 0].plot(_t, _x, 'b-', lw=2)
    axs[1, 0].set_title('Положення візка $x(t)$')
    axs[1, 0].set_xlabel('Час (с)')
    axs[1, 0].set_ylabel('Позиція (м)')
    axs[1, 0].grid(True)

    axs[1, 1].plot(_t, _F, 'g-', lw=2)
    axs[1, 1].set_title('Прикладена сила керування $F(t)$')
    axs[1, 1].set_xlabel('Час (с)')
    axs[1, 1].set_ylabel('Сила (Н)')
    axs[1, 1].grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Збережено: {save_path}")
    plt.show()



def visualize(t, x, th, F, l=0.8, title="", save_gif=None):
    skip = max(1, len(t) // 600)
    t_a, x_a, th_a, F_a = t[::skip], x[::skip], th[::skip], F[::skip]

    fig, ax = plt.subplots(figsize=(8, 5))
    fig.suptitle(f"Анімація PID{(': ' + title) if title else ''}",
                 fontsize=11, fontweight='bold')

    x_margin = max(1.5, np.max(np.abs(x)) + l + 0.4)
    ax.set_xlim(-x_margin, x_margin)
    ax.set_ylim(-0.5, l + 0.6)
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

    rod, = ax.plot([], [], lw=3, color='#3d3d3a', zorder=3,
                    solid_capstyle='round')
    traj, = ax.plot([], [], lw=0.8, color='#c04828', alpha=0.3, zorder=2)
    bob = plt.Circle((0, l), 0.07, fc='#c04828', ec='#801f0f',
                        lw=1.2, zorder=6)
    ax.add_patch(bob)

    arrow_artist = [None]
    tx, ty = [], []

    info = ax.text(0.02, 0.97, '', transform=ax.transAxes, fontsize=9,
                   va='top',
                   bbox=dict(fc='white', alpha=0.85, ec='#ccc',
                             boxstyle='round,pad=0.3'))

    def init():
        rod.set_data([], [])
        traj.set_data([], [])
        return cart_patch, rod, bob, wL, wR, traj

    def update(i):
        xi, thi, Fi = x_a[i], th_a[i], F_a[i]
        xp = xi + l * np.sin(thi)
        yp = l  * np.cos(thi)

        cart_patch.set_x(xi - cart_w / 2)
        wL.center = (xi - cart_w/2 + 0.06, -cart_h - 0.045)
        wR.center = (xi + cart_w/2 - 0.06, -cart_h - 0.045)
        rod.set_data([xi, xp], [0, yp])
        bob.center = (xp, yp)

        tx.append(xp); ty.append(yp)
        traj.set_data(tx[-200:], ty[-200:])

        if arrow_artist[0] is not None:
            arrow_artist[0].remove()
        scale = 0.015
        arrow_artist[0] = ax.annotate(
            '', xy=(xi + Fi * scale, -cart_h / 2),
            xytext=(xi, -cart_h / 2),
            arrowprops=dict(arrowstyle='->', color='#2a9d3f', lw=2.0))

        info.set_text(f"t = {t_a[i]:.2f} s\n"
                      f"x = {xi:.3f} m\n"
                      f"θ = {np.rad2deg(thi):.2f}°\n"
                      f"F = {Fi:.1f} N")
        return cart_patch, rod, bob, wL, wR, traj, info

    ani = animation.FuncAnimation(
        fig, update, frames=len(t_a),
        init_func=init, blit=False, interval=33)

    if save_gif:
        ani.save(save_gif, writer='pillow', fps=30)
        print(f"  Збережено: {save_gif}")
    plt.tight_layout()
    plt.show()
    return ani



def visualize_realtime(t, x, th, dth, F, l=0.8, title=""):
    steps = len(t)

    plt.ion()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_aspect('equal')
    ax.set_ylim(-1.5, l + 1.5)
    ax.grid(True)
    ax.set_title(f"Обернений маятник з ПІД-регулятором{('  —  ' + title) if title else ''}")
    ax.plot([-500, 500], [-0.25, -0.25], color='black', lw=2)
    cart = patches.Rectangle((0, -0.25), 2.0, 0.5, fc='blue', ec='black')
    ax.add_patch(cart)
    pendulum, = ax.plot([], [], 'o-', lw=4, color='red', markersize=10)

    for i in range(steps):
        xi = x[i]
        thi = th[i]
        pivot_y = 0.25
        ax.set_xlim(xi - 5, xi + 5)
        cart.set_xy((xi - 1.0, -0.25))
        pendulum.set_data(
            [xi, xi + l * np.sin(thi)],
            [pivot_y, pivot_y + l * np.cos(thi)]
        )
        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.pause(0.001)

    plt.ioff()
    plt.close(fig)




def sensitivity_analysis(base_params, th0_deg=20.0, T=8.0):
    configs = [
        {"label": "Тільки P  (Kp=50)", "Kp": 50, "Ki": 0, "Kd": 0  },
        {"label": "PD  (Kp=50, Kd=10)", "Kp": 50, "Ki": 0, "Kd": 10 },
        {"label": "PID (Kp=50, Ki=1, Kd=10)", "Kp": 50,  "Ki": 1.0, "Kd": 10 },
    ]
    colors = ['#e07b00', '#1a6fc4', '#2a9d3f']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Аналіз чутливості PID  (θ₀ = {th0_deg}°)",
                 fontsize=12, fontweight='bold')

    for cfg, col in zip(configs, colors):
        t, x, _, th, _, F = simulate_pid(
            **base_params,
            Kp=cfg["Kp"], Ki=cfg["Ki"], Kd=cfg["Kd"],
            th0_deg=th0_deg, T=T)
        ax1.plot(t, np.rad2deg(th), label=cfg["label"], color=col, lw=1.6)
        ax2.plot(t, F, label=cfg["label"], color=col, lw=1.4)

    for ax, ylabel, title_ in [
        (ax1, "θ (°)",  "Кут маятника θ(t)"),
        (ax2, "F (Н)", "Прикладена сила F(t)")]:
        ax.axhline(0, color='#aaa', lw=0.8, ls='--')
        ax.set_xlabel("t (с)"); ax.set_ylabel(ylabel)
        ax.set_title(title_)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig("sensitivity.png", dpi=150, bbox_inches='tight')
    print("  Збережено: sensitivity.png")
    plt.show()


def simulate_with_disturbance(M=1.0, m=0.2, l=0.8, b=0.1, c=0.01,
                               Kp=50.0, Ki=1.0, Kd=10.0,
                               th0_deg=10.0,
                               disturbance_time=4.0,
                               disturbance_magnitude=8.0,
                               T=10.0, h=0.005):

    pid = PIDController(Kp=Kp, Ki=Ki, Kd=Kd, setpoint=0.0)
    n = int(T / h) + 1
    state = np.array([0.0, 0.0, np.deg2rad(th0_deg), 0.0])

    t_arr   = np.zeros(n); x_arr  = np.zeros(n)
    dx_arr  = np.zeros(n); th_arr = np.zeros(n)
    dth_arr = np.zeros(n); F_arr  = np.zeros(n)
    t_arr[0] = 0.0; x_arr[0]  = state[0]
    th_arr[0] = state[2]

    for i in range(1, n):
        t_cur = t_arr[i - 1]
        F_pid = pid.compute(state[2], h)
        disturbance = (disturbance_magnitude
                       if disturbance_time <= t_cur < disturbance_time + 0.1
                       else 0.0)
        F_total = F_pid + disturbance
        F_arr[i] = F_total

        def F_func(t): return F_total
        rhs = make_rhs(M, m, l, b, c, F_func)
        k1 = h * rhs(t_cur,         state)
        k2 = h * rhs(t_cur + h/2,   state + k1/2)
        k3 = h * rhs(t_cur + h/2,   state + k2/2)
        k4 = h * rhs(t_cur + h,     state + k3)
        state = state + (k1 + 2*k2 + 2*k3 + k4) / 6

        t_arr[i] = t_cur + h
        x_arr[i] = state[0]; dx_arr[i]  = state[1]
        th_arr[i] = state[2]; dth_arr[i] = state[3]

    return t_arr, x_arr, dx_arr, th_arr, dth_arr, F_arr


if __name__ == "__main__":

    BASE = dict(M=1.0, m=0.2, l=0.8, b=0.1, c=0.01)

    # Сценарій 1: θ = 15
    print("Сценарій 1: θ = 15")
    t, x, dx, th, dth, F = simulate_pid(
        **BASE, Kp=50, Ki=1.0, Kd=10,
        th0_deg=15, T=10)

    plot_results(t, x, dx, th, dth, F,
                 title="θ₀=15°, Kp=50, Ki=1, Kd=10",
                 save_path="plots_pid_15.png")

    # Сценарій 2: θ = 30
    print("Сценарій 2: θ = 30")
    t2, x2, dx2, th2, dth2, F2 = simulate_pid(
        **BASE, Kp=50, Ki=1.0, Kd=10,
        th0_deg=30, T=10)

    plot_results(t2, x2, dx2, th2, dth2, F2,
                 title="θ=30, Kp=50, Ki=1, Kd=10",
                 save_path="plots_pid_30.png")

    print("Аналіз чутливості...")
    sensitivity_analysis(BASE, th0_deg=20, T=8)

    print("Сценарій зі збуренням (поштовх 8 Н у t=4 с)...")
    td, xd, dxd, thd, dthd, Fd = simulate_with_disturbance(
        **BASE, Kp=50, Ki=1.0, Kd=10,
        th0_deg=5,
        disturbance_time=4.0,
        disturbance_magnitude=8.0,
        T=10)

    plot_results(td, xd, dxd, thd, dthd, Fd,
                 title="Збурення 8 Н у t=4 с",
                 save_path="plots_disturbance.png")

    print("\nГотово!")