"""
Equation of motion (Lagrangian):
    (M+m)·ẍ + m·l·θ̈·cosθ − m·l·θ̇²·sinθ + b·ẋ = F(t)
    m·l²·θ̈ + m·l·ẍ·cosθ  − m·g·l·sinθ   + c·θ̇ = 0
"""

"""
State vector: x = [x, x`, θ, θ`]
   ẍ  = -b/M · x` - mg/M · θ + c/(Ml) · θ̇ + 1/M · F
   θ̈  =  b/(Ml) · x` + (M+m)g/(Ml²) · θ - c(M+m)/(Mml²) · θ̇ - 1/(Ml) · F
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

M = 1.0
m = 0.3
l = 0.8 
b = 0.1
c = 0.01
g = 9.81

A = np.array([
    [0,      1,              0,                    0           ],
    [0,   -b/M,          -m*g/M,               c/(M*l)        ],
    [0,      0,              0,                    1           ],
    [0,   b/(M*l),  (M+m)*g/(M*l**2),  -c*(M+m)/(M*m*l**2)   ]
])

B = np.array([[0],
              [1/M],
              [0],
              [-1/(M*l)]])

C = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0]
])

C_ctrl = np.hstack([
    B,
    A @ B,
    A @ A @ B,
    A @ A @ A @ B
])

rank_ctrl = np.linalg.matrix_rank(C_ctrl)

print("--- CONTROLLABILITY ---")
print(f"\nControllability Matrix (4×4):")
print(np.array2string(C_ctrl, precision=4, suppress_small=True))
print(f"\nRank of Controllability Matrix: {rank_ctrl}")
if rank_ctrl == 4:
    print("The system is CONTROLLABLE (rank = n = 4)")
else:
    print("The system is NOT controllable")


O = np.vstack([
    C,
    C @ A,
    C @ A @ A,
    C @ A @ A @ A
])

rank_obs = np.linalg.matrix_rank(O)
 
print("\n--- OBSERVABILITY ---")
print(f"\nObservability Matrix (8×4):")
print(np.array2string(O, precision=4, suppress_small=True))
print(f"\nRank of Observability Matrix: {rank_obs}")
if rank_obs == 4:
    print("The system is OBSERVABLE (rank = n = 4)")
else:
    print("The system is NOT observable")

eigenvalues = np.linalg.eigvals(A)
 

print("  EIGENVALUES AND STABILITY")
print("\nEigenvalues of matrix A:")
for i, lam in enumerate(eigenvalues):
    sign = "+" if lam.imag >= 0 else ""
    re_str = f"Re = {lam.real:+.4f}"
    im_str = f"Im = {lam.imag:+.4f}"
    stable = "stable " if lam.real < 0 else "unstable "
    print(f"  λ{i+1} = {lam.real:.4f}{sign}{lam.imag:.4f}j  |  {re_str},  {im_str}  →  {stable}")
 
all_stable = all(lam.real < 0 for lam in eigenvalues)
print(f"\n{' SYSTEM IS STABLE' if all_stable else ' SYSTEM IS UNSTABLE (contains unstable eigenvalues)'}")


fig = plt.figure(figsize=(8, 7), facecolor='#0f1117')
ax = fig.add_subplot(111, facecolor='#0f1117')
 
ax.axhline(0, color='#444', lw=1.0, zorder=1)
ax.axvline(0, color='#ff4444', lw=1.5, ls='--', zorder=2, label='Stability Boundary (Re=0)')
 
ax.axvspan(ax.get_xlim()[0] if ax.get_xlim()[0] < -3 else -3,
           0, alpha=0.07, color='#00ff88', zorder=0)
 
colors = ['#00e5ff', '#ff6b35', '#a8ff3e', '#ff3ea8']
for i, lam in enumerate(eigenvalues):
    color = colors[i % len(colors)]
    ax.scatter(lam.real, lam.imag,
               s=180, color=color, zorder=5,
               edgecolors='white', linewidths=0.8)
    ax.annotate(f'  λ{i+1} = {lam.real:.3f}{("+" if lam.imag>=0 else "")}{lam.imag:.3f}j',
                xy=(lam.real, lam.imag),
                fontsize=9, color=color,
                fontfamily='monospace',
                va='bottom')
 
ax.set_xlabel('Re(λ)', color='#aaa', fontsize=12)
ax.set_ylabel('Im(λ)', color='#aaa', fontsize=12)
ax.set_title('Eigenvalue Placement in the Complex Plane',
             color='white', fontsize=13, fontweight='bold', pad=15)
ax.tick_params(colors='#888')
for spine in ax.spines.values():
    spine.set_edgecolor('#333')
ax.legend(fontsize=9, facecolor='#1a1d27', edgecolor='#444',
          labelcolor='#ccc')
ax.grid(alpha=0.15, color='#555')
 
