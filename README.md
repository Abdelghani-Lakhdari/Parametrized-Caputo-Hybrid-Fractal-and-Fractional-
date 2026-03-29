import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import time
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from scipy.integrate import quad
from scipy.special import beta
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

# ==========================================
# 1. DEFINITION OF THEOREM 3.1
# ==========================================
def calculate_theorem_3_1(gamma, nu, s, p, r1=0, r2=1):
    q = p / (p - 1)

    def deriv_1(t): return t**(s + 1)
    def deriv_2(t): return (s + 1) * (t**s)

    val_d1_r1 = np.abs(deriv_1(r1))**q
    val_d1_r2 = np.abs(deriv_1(r2))**q
    val_d2_r1 = np.abs(deriv_2(r1))**q
    val_d2_r2 = np.abs(deriv_2(r2))**q

    factor1 = (gamma**2) * ((r2 - r1)**(1 + gamma))
    num_nu   = (nu**(p+1) + (1-nu)**(p+1))
    den_nu   = (2**(p+1)) * (p+1)
    term_nu  = (num_nu / den_nu)**(1/p)
    denom_s  = (2**(s+1)) * (s+1)

    part_A = ( ((2**(s+1)-1)*val_d1_r1 + val_d1_r2) / denom_s )**(1/q)
    part_B = ( (val_d1_r1 + (2**(s+1)-1)*val_d1_r2) / denom_s )**(1/q)
    RHS_Part1 = factor1 * term_nu * (part_A + part_B)

    factor2 = ((1 - gamma) * ((r2 - r1)**(2 - gamma))) / 4

    def integrand1(l): return np.abs(l**(2 - 2*gamma) - (nu/2.0))**p
    def integrand2(l): return np.abs(l**(2 - 2*gamma) - ((2 - nu)/2.0))**p

    int_val1, _ = quad(integrand1, 0, 0.5)
    int_val2, _ = quad(integrand2, 0.5, 1.0)
    term_int = (int_val1**(1/p)) + (int_val2**(1/p))

    part_C = ( ((2**(s+1)-1)*val_d2_r1 + val_d2_r2) / denom_s )**(1/q)
    part_D = ( (val_d2_r1 + (2**(s+1)-1)*val_d2_r2) / denom_s )**(1/q)
    RHS_Part2 = factor2 * term_int * (part_C + part_D)
    RHS_Total = RHS_Part1 + RHS_Part2

    term_1    = (gamma**2 / (s + 2)) * ( (nu / 2) + (1 - nu) * (0.5)**(s + 2) )
    term_2    = ((1 - gamma) / 2)    * ( (nu / 2) + (1 - nu) * (0.5)**(s + 1) )
    val_beta  = beta(s + 2, 2 - 2 * gamma)
    term_3    = - ((1 - gamma)**2 / 2) * ( val_beta + (1 / (3 - 2 * gamma + s)) )
    term_4    = - (gamma**2) / (3 * (s + 3))

    LHS_Exact = np.abs(term_1 + term_2 + term_3 + term_4)
    return -RHS_Total, LHS_Exact, RHS_Total


# ==========================================
# 2. GENERATION OF THE LARGE DATASET (MAX SIZE)
# ==========================================
SAMPLE_SIZES  = [2000, 4000, 6000, 8000, 10000]
MAX_N         = max(SAMPLE_SIZES)
K_FOLDS       = 10
OUTPUT_LABELS = ["Lower Bound (-RHS)", "LHS (Exact)", "Upper Bound (+RHS)"]
COLORS_OUT    = ['#E53935', '#43A047', '#1E88E5']

print("=" * 60)
print(f"  Generating master dataset ({MAX_N} samples)...")
print("=" * 60)

np.random.seed(42)
gamma_all = np.random.uniform(0.1, 0.99, MAX_N)
nu_all    = np.random.uniform(0.0, 1.0,  MAX_N)
s_all     = np.random.uniform(0.1, 1.0,  MAX_N)
p_all     = np.random.uniform(1.2, 4.0,  MAX_N)
q_all     = p_all / (p_all - 1)

X_all = np.column_stack((gamma_all, nu_all, s_all, p_all, q_all))
Y_all = np.array([
    calculate_theorem_3_1(g, n, s, p)
    for g, n, s, p in zip(gamma_all, nu_all, s_all, p_all)
])
print(f"  Master dataset ready: {X_all.shape}\n")


# ==========================================
# 3. SENSITIVITY LOOP — SIZE x 10-CV
# ==========================================

# Result structures
results = {}   # results[n] = dict with all stats

for n_samples in SAMPLE_SIZES:

    print("=" * 60)
    print(f"  >>> SIZE N = {n_samples}  ({K_FOLDS}-Fold CV)")
    print("=" * 60)

    t0 = time.time()

    # Subset of the first n_samples (fixed seed -> reproducible)
    X = X_all[:n_samples]
    Y = Y_all[:n_samples]

    kf = KFold(n_splits=K_FOLDS, shuffle=True, random_state=42)

    fold_r2_global  = []
    fold_r2_per_out = [[], [], []]
    fold_loss_curves = []
    best_fold_score  = -np.inf
    best_Y_test      = None
    best_Y_pred      = None
    best_fold_idx    = -1

    for fold, (train_idx, test_idx) in enumerate(kf.split(X), start=1):

        X_tr_raw, X_te_raw = X[train_idx], X[test_idx]
        Y_tr_raw, Y_te_raw = Y[train_idx], Y[test_idx]

        scaler_X = StandardScaler()
        scaler_Y = StandardScaler()

        X_tr = scaler_X.fit_transform(X_tr_raw)
        X_te = scaler_X.transform(X_te_raw)
        Y_tr = scaler_Y.fit_transform(Y_tr_raw)
        Y_te = scaler_Y.transform(Y_te_raw)

        ann = MLPRegressor(
            hidden_layer_sizes=(64, 64),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=5000,
            random_state=42,
            n_iter_no_change=50,
            tol=1e-5
        )
        ann.fit(X_tr, Y_tr)

        r2_global = ann.score(X_te, Y_te)
        fold_r2_global.append(r2_global)
        fold_loss_curves.append(ann.loss_curve_)

        Y_pred_sc   = ann.predict(X_te)
        Y_pred_orig = scaler_Y.inverse_transform(Y_pred_sc)
        Y_te_orig   = scaler_Y.inverse_transform(Y_te)

        for out_i in range(3):
            ss_res = np.sum((Y_te_orig[:, out_i] - Y_pred_orig[:, out_i])**2)
            ss_tot = np.sum((Y_te_orig[:, out_i] - np.mean(Y_te_orig[:, out_i]))**2)
            r2_i = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fold_r2_per_out[out_i].append(r2_i)

        if r2_global > best_fold_score:
            best_fold_score = r2_global
            best_fold_idx   = fold
            best_Y_test     = Y_te_orig
            best_Y_pred     = Y_pred_orig

        print(f"    Fold {fold:2d}/{K_FOLDS} | R²={r2_global:.5f} | "
              f"[LB]={fold_r2_per_out[0][-1]:.4f} "
              f"[LHS]={fold_r2_per_out[1][-1]:.4f} "
              f"[UB]={fold_r2_per_out[2][-1]:.4f}")

    elapsed = time.time() - t0
    r2_arr  = np.array(fold_r2_global)

    results[n_samples] = {
        'r2_mean'        : r2_arr.mean(),
        'r2_std'         : r2_arr.std(),
        'r2_min'         : r2_arr.min(),
        'r2_max'         : r2_arr.max(),
        'r2_folds'       : r2_arr.tolist(),
        'r2_per_out'     : [np.array(fold_r2_per_out[i]) for i in range(3)],
        'loss_curves'    : fold_loss_curves,
        'best_fold_idx'  : best_fold_idx,
        'best_fold_score': best_fold_score,
        'best_Y_test'    : best_Y_test,
        'best_Y_pred'    : best_Y_pred,
        'elapsed_s'      : elapsed,
    }

    ic_lo = r2_arr.mean() - 1.96 * r2_arr.std()
    ic_hi = r2_arr.mean() + 1.96 * r2_arr.std()
    print(f"\n  → Mean R² = {r2_arr.mean():.5f} ± {r2_arr.std():.5f}"
          f"  95% CI [{ic_lo:.5f}, {ic_hi:.5f}]  ({elapsed:.1f}s)\n")


# ==========================================
# 4. COMPARATIVE SUMMARY
# ==========================================
print("\n" + "=" * 60)
print("  SENSITIVITY SUMMARY — Sample Size vs 10-Fold CV R²")
print("=" * 60)
print(f"  {'N':>7}  {'R² Mean':>9}  {'R² Std':>8}  {'R² Min':>8}  {'R² Max':>8}  {'Time(s)':>9}")
print("  " + "-" * 58)
for n in SAMPLE_SIZES:
    r = results[n]
    print(f"  {n:>7}  {r['r2_mean']:>9.5f}  {r['r2_std']:>8.5f}  "
          f"{r['r2_min']:>8.5f}  {r['r2_max']:>8.5f}  {r['elapsed_s']:>9.1f}")
print()


# ==========================================
# 5. PLOTS
# ==========================================

sizes    = SAMPLE_SIZES
r2_means = [results[n]['r2_mean'] for n in sizes]
r2_stds  = [results[n]['r2_std']  for n in sizes]
r2_mins  = [results[n]['r2_min']  for n in sizes]
r2_maxs  = [results[n]['r2_max']  for n in sizes]

# ── Plot 1 : Main Sensitivity ──────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle("Sensitivity Analysis — Sample Size vs 10-Fold CV Performance",
             fontsize=14, fontweight='bold')

# 1a : Mean R² ± std vs N
ax = axes[0]
ax.errorbar(sizes, r2_means, yerr=r2_stds, fmt='o-', color='#1565C0',
            capsize=6, capthick=2, linewidth=2, markersize=8, label='R² Mean ± Std')
ax.fill_between(sizes,
                [r2_means[i] - r2_stds[i] for i in range(len(sizes))],
                [r2_means[i] + r2_stds[i] for i in range(len(sizes))],
                alpha=0.15, color='#1565C0')
ax.set_title("Mean R² ± Std (10-CV)")
ax.set_xlabel("Sample Size (N)")
ax.set_ylabel("R² Score")
ax.set_xticks(sizes)
ax.set_xticklabels([str(n) for n in sizes], rotation=15)
ax.grid(True, alpha=0.3)
ax.legend()

# 1b : Min / Max / Mean band
ax = axes[1]
ax.plot(sizes, r2_maxs,  's--', color='#43A047', linewidth=1.5, markersize=7, label='R² Max')
ax.plot(sizes, r2_means, 'o-',  color='#1565C0', linewidth=2,   markersize=8, label='R² Mean')
ax.plot(sizes, r2_mins,  '^--', color='#E53935', linewidth=1.5, markersize=7, label='R² Min')
ax.fill_between(sizes, r2_mins, r2_maxs, alpha=0.1, color='grey', label='Range')
ax.set_title("R² Min / Mean / Max by N")
ax.set_xlabel("Sample Size (N)")
ax.set_ylabel("R² Score")
ax.set_xticks(sizes)
ax.set_xticklabels([str(n) for n in sizes], rotation=15)
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# 1c : Std (variability) vs N
ax = axes[2]
ax.plot(sizes, r2_stds, 'D-', color='#F57C00', linewidth=2, markersize=8)
ax.fill_between(sizes, 0, r2_stds, alpha=0.2, color='#F57C00')
ax.set_title("Variability (R² Std) vs N\n(Model Stability)")
ax.set_xlabel("Sample Size (N)")
ax.set_ylabel("Std R²")
ax.set_xticks(sizes)
ax.set_xticklabels([str(n) for n in sizes], rotation=15)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensitivity_main.png", dpi=300, bbox_inches='tight')
print("Plot 1 (Main Sensitivity) saved → sensitivity_main.png")


# ── Plot 2 : R² per output vs N ────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Sensitivity by Output (Lower Bound / LHS / Upper Bound)",
             fontsize=13, fontweight='bold')

for out_i, (lbl, col) in enumerate(zip(OUTPUT_LABELS, COLORS_OUT)):
    ax = axes[out_i]
    means_out = [results[n]['r2_per_out'][out_i].mean() for n in sizes]
    stds_out  = [results[n]['r2_per_out'][out_i].std()  for n in sizes]

    ax.errorbar(sizes, means_out, yerr=stds_out, fmt='o-', color=col,
                capsize=6, capthick=2, linewidth=2, markersize=8)
    ax.fill_between(sizes,
                    [means_out[i] - stds_out[i] for i in range(len(sizes))],
                    [means_out[i] + stds_out[i] for i in range(len(sizes))],
                    alpha=0.15, color=col)
    ax.set_title(f"{lbl}")
    ax.set_xlabel("N")
    ax.set_ylabel("R² Score")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(n) for n in sizes], rotation=15)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("sensitivity_per_output.png", dpi=300, bbox_inches='tight')
print("Plot 2 (Sensitivity per output) saved → sensitivity_per_output.png")


# ── Plot 3 : R² Heatmap (fold x size) ──────────────────────
heat_matrix = np.array([results[n]['r2_folds'] for n in sizes])  # (5, 10)

fig, ax = plt.subplots(figsize=(13, 5))
im = ax.imshow(heat_matrix, aspect='auto', cmap='RdYlGn',
               vmin=heat_matrix.min(), vmax=heat_matrix.max())
ax.set_xticks(range(K_FOLDS))
ax.set_xticklabels([f"Fold {k+1}" for k in range(K_FOLDS)], fontsize=9)
ax.set_yticks(range(len(sizes)))
ax.set_yticklabels([f"N={n}" for n in sizes], fontsize=10)
ax.set_title("R² Heatmap — Sample Size × Fold (10-CV)", fontsize=12, fontweight='bold')

for i in range(len(sizes)):
    for j in range(K_FOLDS):
        val = heat_matrix[i, j]
        color = 'black' if 0.4 < val < 0.95 else 'white'
        ax.text(j, i, f"{val:.3f}", ha='center', va='center', fontsize=8, color=color)

plt.colorbar(im, ax=ax, label="R² Score", fraction=0.03)
plt.tight_layout()
plt.savefig("sensitivity_heatmap.png", dpi=300, bbox_inches='tight')
print("Plot 3 (R² Heatmap) saved → sensitivity_heatmap.png")


# ── Plot 4 : R² Boxplot by size ─────────────────────────────
fig, ax = plt.subplots(figsize=(11, 6))
bp_data = [results[n]['r2_folds'] for n in sizes]
bp = ax.boxplot(bp_data, patch_artist=True,
                medianprops=dict(color='black', linewidth=2.5),
                flierprops=dict(marker='o', markerfacecolor='red', markersize=5))

palette = ['#BBDEFB', '#90CAF9', '#64B5F6', '#42A5F5', '#1E88E5']
for patch, col in zip(bp['boxes'], palette):
    patch.set_facecolor(col)
    patch.set_alpha(0.85)

ax.set_xticklabels([f"N={n}" for n in sizes], fontsize=10)
ax.set_title("R² Distribution (10-CV) by Sample Size", fontsize=12, fontweight='bold')
ax.set_ylabel("R² Score")
ax.grid(True, alpha=0.3, axis='y')

# Annotations: mean ± std
for i, n in enumerate(sizes, start=1):
    m, s = results[n]['r2_mean'], results[n]['r2_std']
    ax.text(i, ax.get_ylim()[0] + 0.005, f"μ={m:.3f}\nσ={s:.4f}",
            ha='center', va='bottom', fontsize=8, color='#1A237E')

plt.tight_layout()
plt.savefig("sensitivity_boxplot.png", dpi=300, bbox_inches='tight')
print("Plot 4 (R² Boxplot by size) saved → sensitivity_boxplot.png")


# ── Plot 5 : Loss curves best fold by size ─────────────────
fig, axes = plt.subplots(1, len(sizes), figsize=(20, 4), sharey=True)
fig.suptitle("Convergence Curves — Best Fold by Sample Size",
             fontsize=12, fontweight='bold')

for ax, n, col in zip(axes, sizes, palette):
    r    = results[n]
    best = r['best_fold_idx'] - 1
    for fi, lc in enumerate(r['loss_curves']):
        is_best = (fi == best)
        ax.plot(lc,
                linewidth=2.0 if is_best else 0.7,
                alpha=1.0    if is_best else 0.25,
                color=col    if is_best else 'grey',
                label=f"Fold {fi+1} ★" if is_best else None)
    ax.set_title(f"N = {n}\nR²={r['best_fold_score']:.4f}", fontsize=9)
    ax.set_xlabel("Iterations", fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

axes[0].set_ylabel("Loss (log)")
plt.tight_layout()
plt.savefig("sensitivity_loss_curves.png", dpi=300, bbox_inches='tight')
print("Plot 5 (Loss curves by size) saved → sensitivity_loss_curves.png")


# ── Plot 6 : Validation prediction — all N ──────────────────
fig, axes = plt.subplots(1, len(sizes), figsize=(22, 5), sharey=False)
fig.suptitle("ANN Validation — Best Fold by Sample Size",
             fontsize=12, fontweight='bold')

for ax, n in zip(axes, sizes):
    r         = results[n]
    Y_te      = r['best_Y_test']
    Y_pr      = r['best_Y_pred']
    sort_idx  = np.argsort(Y_te[:, 2])
    step      = max(1, len(sort_idx) // 100)
    subset    = sort_idx[::step]

    ax.plot(range(len(subset)), Y_pr[subset, 0], 'r--', linewidth=1.2, label='LB')
    ax.plot(range(len(subset)), Y_pr[subset, 1], 'g-',  linewidth=1.8, label='LHS')
    ax.plot(range(len(subset)), Y_pr[subset, 2], 'b--', linewidth=1.2, label='UB')
    ax.set_title(f"N={n}\nR²={r['best_fold_score']:.4f}", fontsize=9)
    ax.set_xlabel("Samples", fontsize=8)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7)

axes[0].set_ylabel("Value")
plt.tight_layout()
plt.savefig("sensitivity_predictions.png", dpi=300, bbox_inches='tight')
print("Plot 6 (Predictions by size) saved → sensitivity_predictions.png")


# ── Plot 7 : Training time vs N ─────────────────────────────
times = [results[n]['elapsed_s'] for n in sizes]
fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar([str(n) for n in sizes], times, color=palette, edgecolor='black', linewidth=0.7)
for bar, t in zip(bars, times):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{t:.1f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title("Total Training Time (10-CV) by Sample Size N", fontsize=12, fontweight='bold')
ax.set_xlabel("Sample Size (N)")
ax.set_ylabel("Time (seconds)")
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("sensitivity_timing.png", dpi=300, bbox_inches='tight')
print("Plot 7 (Timing) saved → sensitivity_timing.png")

print("\n" + "=" * 60)
print("  ✓ Complete Sensitivity Analysis — 7 plots generated")
print("=" * 60)
