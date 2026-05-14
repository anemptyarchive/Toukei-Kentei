
# 区間推定-----------------------------------------------------------------------

# ch3.5.2

# 正規分布
# 母平均の差の信頼区間
# 母分散が未知の場合
# 2標本に対応がある場合


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import multivariate_normal, norm, t
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# %%

# ディレクトリの設定 -------------------------------------------------------------

# ライブラリを読込
from pathlib import Path

# ワークスペースを取得
PROJECT_DIR = Path.cwd()
print(PROJECT_DIR)

# 書き出し先を設定
dir_path  = PROJECT_DIR.as_posix()
dir_path += '/figure/confidence_interval/' # パスを指定
dir_path += 'estimate_mean_difference_paired_samples/' # フォルダを指定
print(dir_path)


# %%

# 共通の設定 ------------------------------------------------------------------

### 母集団の設定 -----

# 母分布のパラメータを指定
mu_pop_d = np.array(
    [10.0, 4.0]
)
sigma2_pop_dd = np.array(
    [[9.0, 1.0], 
     [1.0, 4.0]]
)

# 母平均の差を計算
delta_pop = np.subtract(*mu_pop_d)


# 差の母分布のパラメータを計算
mu_diff     = np.subtract(*mu_pop_d)
sigma2_diff = np.sum(np.diag(sigma2_pop_dd)) - 2.0 * sigma2_pop_dd[0, 1]
sigma_diff  = np.sqrt(sigma2_diff)


# %%

# x軸の範囲を設定
k = 3.0
u = 1.0
x_min = np.min([mu_pop_d[d] - k*np.sqrt(sigma2_pop_dd[d, d]) for d in range(2)]) # 標準偏差の定数倍
x_max = np.max([mu_pop_d[d] + k*np.sqrt(sigma2_pop_dd[d, d]) for d in range(2)]) # 標準偏差の定数倍
#x_min = np.min(x_nd) # 最小値
#x_max = np.max(x_nd) # 最大値
x_min = np.floor(x_min /u)*u # u単位で切り下げ
x_max = np.ceil(x_max /u)*u  # u単位で切り上げ
print('x-axis size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)

# x1,x2軸の格子点を作成
x_1_grid, x_2_grid = np.meshgrid(x_vec, x_vec)

# x1,x2軸の座標を整形
x_arr  = np.stack([x_1_grid.flatten(), x_2_grid.flatten()], axis=1)
x_dims = x_1_grid.shape


# d軸の範囲を設定
d_min = x_min - mu_pop_d[1]
d_max = x_max - mu_pop_d[1]
print('d-axis size:', d_min, d_max)

# d軸の値を作成
d_vec = np.linspace(start=d_min, stop=d_max, num=1001)


# %%

### 集計の設定 -----

# 階級幅を指定
class_x_size = 1.0

# 階級数を計算
class_x_num = ((x_max - x_min) // class_x_size).astype(dtype='int') + 1
print('class size:', class_x_num)

# 階級幅を再設定:(割り切れない場合)
class_x_size = (x_max - x_min) / (class_x_num-1)
print('class num: ', class_x_size)

# 階級値を作成
center_x_vec = np.arange(start=x_min, stop=x_max+class_x_size, step=class_x_size)

# 境界値の範囲を設定
bound_x_min = x_min - 0.5*class_x_size
bound_x_max = x_max + 0.5*class_x_size

# 境界値を作成
bound_x_vec = np.arange(start=bound_x_min, stop=bound_x_max+0.5*class_x_size, step=class_x_size)


# 階級幅を指定
class_d_size = 1.0

# 階級数を計算
class_d_num = ((d_max - d_min) // class_d_size).astype(dtype='int') + 1
print('class size:', class_d_num)

# 階級幅を再設定:(割り切れない場合)
class_d_size = (d_max - d_min) / (class_d_num-1)
print('class num: ', class_d_size)

# 階級値を作成
center_d_vec = np.arange(start=d_min, stop=d_max+class_d_size, step=class_d_size)

# 境界値の範囲を設定
bound_d_min = d_min - 0.5*class_d_size
bound_d_max = d_max + 0.5*class_d_size

# 境界値を作成
bound_d_vec = np.arange(start=bound_d_min, stop=bound_d_max+0.5*class_d_size, step=class_d_size)


# %%

# 母分布の確率密度を計算
pop_dens_vec = multivariate_normal.pdf(
    x=x_arr, mean=mu_pop_d, cov=sigma2_pop_dd
)

# 母分布の確率密度を計算
pop_dens_lt = [
    norm.pdf(
        x=x_vec, loc=mu_pop_d[d], scale=np.sqrt(sigma2_pop_dd[d, d])
    ) for d in range(2)
]

# 差の母分布の確率密度を計算
diff_dens_vec = norm.pdf(x=d_vec, loc=mu_diff, scale=sigma_diff)


# %%

### 信頼区間の設定 -----

# 信頼係数を指定
gamma = 0.95

# 有意水準を計算
alpha = 1.0 - gamma
print('α:', alpha)


# %%

# サンプルサイズの影響 -----------------------------------------------------------

### 乱数の生成 -----

# シードを設定:(ノートとの対応用)
np.random.seed(6)

# サンプルサイズ(フレーム数)を指定
N = 100

# 標本を生成
x_nd = np.random.multivariate_normal(mean=mu_pop_d, cov=sigma2_pop_dd, size=N)

# 対標本の差を計算
d_n = np.subtract(*x_nd.T)


# %%

### 表示範囲の設定 ----

# 2D母分布のp軸の範囲を設定
u = 0.001
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 1D母分布のp軸の範囲を設定
u = 0.25
Px_d_max = np.max(pop_dens_lt)
Px_d_max = np.ceil(Px_d_max /u)*u # u単位で切り上げ
print('p(x_d) size:  ', Px_d_max)

# 差の母分布のp軸の範囲を設定
u = 0.05
Pd_max = diff_dens_vec.max()
Pd_max = np.ceil(Pd_max /u)*u # u単位で切り上げ
print('p(d) size:    ', Pd_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_bar_max = norm.pdf(x=mu_diff, loc=mu_diff, scale=sigma_diff/np.sqrt(N)) # 最終試行の最頻値
Pd_bar_max = np.ceil(Pd_bar_max /u)*u # u単位で切り上げ
print('p(d bar) size:', Pd_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pt_max = t.pdf(x=0.0, df=N-1, loc=0.0, scale=1.0) # 最終試行の最頻値
Pt_max = np.ceil(Pt_max /u)*u # u単位で切り上げ
print('p(t) size:    ', Pt_max)


# %%

### 信頼区間の可視化 -----

# 配色を設定
cmap = plt.get_cmap('tab10') # カラーマップを指定

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=2, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle(
    'Population Mean Difference Confidence Interval\n(paired samples, unknown variance)', 
    fontsize=20
)
axes2x = [ax.twiny() for ax in axes.flatten()[[0, 2, 1, 3, 5]]]
axes2y = [ax.twinx() for ax in [axes[0, 0], axes[0, 1]]]

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes.flatten()]
    [ax2x.cla() for ax2x in axes2x]
    [ax2y.cla() for ax2y in axes2y]

    ##### 乱数の集計 -----

    # 値を設定
    n = i + 1 # サンプル数

    # 標本統計量を計算
    x_bar_d       = np.mean(x_nd[:n], axis=0)
    sigma2_hat_d  = np.var(x_nd[:n], ddof=1, axis=0) if n > 1 else np.tile(np.nan, reps=2)
    sigma_hat_d   = np.sqrt(sigma2_hat_d)
    sigma_hat_12  = np.cov(x_nd[:n, 0], x_nd[:n, 1], ddof=1)[0, 1] if n > 1 else np.nan
    sigma2_hat_dd = np.array(
        [[sigma2_hat_d[0], sigma_hat_12], 
        [sigma_hat_12, sigma2_hat_d[1]]]
    )

    # 差の標本統計量を計算
    d_bar      = np.mean(d_n[:n])
    sigma2_hat = np.var(d_n[:n], ddof=1) if n > 1 else np.nan
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_diff
    sigma_smp = sigma_hat / np.sqrt(n)

    # 標本統計量を標準化
    t_obs = (d_bar - mu_smp) / sigma_smp

    # 中央領域の範囲を計算
    cr_bound_upper = t.ppf(q=1.0-0.5*alpha, df=n-1, loc=0.0, scale=1.0)
    cr_bound_lower = -cr_bound_upper

    # 信頼区間の範囲を計算
    ci_bound_lower = d_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = d_bar + cr_bound_upper * sigma_smp

    ##### 母分布の作図:2次元 -----

    # 密度を集計
    obs_dens_grid, _, _ = np.histogram2d(
        x=x_nd[:n, 0], y=x_nd[:n, 1], 
        bins=(class_x_num, class_x_num), 
        range=[(bound_x_min, bound_x_max), (bound_x_min, bound_x_max)], 
        density=True
    )

    # 母分布のラベルを作成
    pop_2d_param_lbl  = f'$N = {n}$\n'
    tmp_mu_str        = ', '.join([f'{mu:.2f}' for mu in mu_pop_d])
    tmp_sgm2_1_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_pop_dd[0, :]])
    tmp_sgm2_2_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_pop_dd[1, :]])
    pop_2d_param_lbl += f'$\\mu_{{pop}} = ({tmp_mu_str}), '
    pop_2d_param_lbl += f'\\Sigma_{{pop}} = \\binom{{{tmp_sgm2_1_str}}}{{{tmp_sgm2_2_str}}}$\n'
    tmp_x_str         = ', '.join([f'{x:.2f}' for x in x_bar_d])
    tmp_sgm2_1_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_hat_dd[0, :]])
    tmp_sgm2_2_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_hat_dd[1, :]])
    pop_2d_param_lbl += f'$\\bar{{x}} = ({tmp_x_str}), '
    pop_2d_param_lbl += f'\\hat{{\\Sigma}} = \\binom{{{tmp_sgm2_1_str}}}{{{tmp_sgm2_2_str}}}$'

    # 母分布を描画
    ax   = axes[0, 0]
    ax2x = axes2x[0]
    ax2y = axes2y[0]

    ax.pcolormesh(
        center_x_vec, center_x_vec, obs_dens_grid.T, 
        shading='nearest', 
        cmap='viridis', vmin=0.0, vmax=Px_max, alpha=0.5, 
        zorder=8
    ) # 標本
    ax.scatter(
        x=x_nd[:n, 0], y=x_nd[:n, 1], 
        color='black', alpha=0.33, s=25, 
        zorder=9
    ) # 標本
    ax.plot(
        [], [], 
        color='purple', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # (凡例表示用のダミー)
    ax.contour(
        x_1_grid, x_2_grid, pop_dens_vec.reshape(x_dims), 
        cmap='viridis', vmin=0.0, vmax=Px_max, 
        linewidths=1.0, 
        zorder=10
    ) # 母分布
    for idx, mu in enumerate([mu_pop_d[0], x_bar_d[0]]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population means', 'sample means'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for idx, mu in enumerate([mu_pop_d[1], x_bar_d[1]]):
        ax.axhline(
            y=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    ax.plot(
        x_vec, x_vec, 
        color='black', linewidth=1.0, 
        zorder=30
    ) # 恒等関数

    ax.set_xlabel('$x_1$')
    ax2x.set_xticks(
        ticks =[mu_pop_d[0], x_bar_d[0]], 
        labels=['$\\mu_1$', '$\\bar{x}_1$']
    ) # パラメータのラベル
    ax.set_ylabel('$x_2$')
    ax2y.set_yticks(
        ticks =[mu_pop_d[1], x_bar_d[1]], 
        labels=['$\\mu_2$', '$\\bar{x}_2$']
    ) # パラメータのラベル
    ax.set_title(pop_2d_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=x_min, ymax=x_max)   # (目盛の共通化用)
    ax2y.set_ylim(ymin=x_min, ymax=x_max) # (目盛の共通化用)

    ##### 母分布の作図:1次元 -----

    # 密度を集計
    obs_dens_lt = [
        np.histogram(
            a=x_nd[:n, d], bins=class_x_num, range=(bound_x_min, bound_x_max), density=True
        )[0] for d in range(2)
    ]

    # 母分布のラベルを作成
    pop_1d_param_lbl  = f'$\\mu_1 = {mu_pop_d[0]:.2f}, '
    pop_1d_param_lbl += f'\\sigma_1 = {np.sqrt(sigma2_pop_dd[0, 0]):.2f}, '
    pop_1d_param_lbl += '\\bar{x}_1 = '+f'{x_bar_d[0]:.2f}, '
    pop_1d_param_lbl += '\\hat{\\sigma}_1 = '+f'{sigma_hat_d[0]:.2f}$\n'
    pop_1d_param_lbl += f'$\\mu_2 = {mu_pop_d[1]:.2f}, '
    pop_1d_param_lbl += f'\\sigma_2 = {np.sqrt(sigma2_pop_dd[1, 1]):.2f}, '
    pop_1d_param_lbl += '\\bar{x}_2 = '+f'{x_bar_d[1]:.2f}, '
    pop_1d_param_lbl += '\\hat{\\sigma}_2 = '+f'{sigma_hat_d[1]:.2f}$'

    # 母分布を描画
    ax   = axes[1, 0]
    ax2x = axes2x[1]

    for d in range(2):
        ax.bar(
            x=center_x_vec, height=obs_dens_lt[d], 
            width=class_x_size, 
            color=cmap(d), alpha=0.5, 
            zorder=8
        ) # 標本
        ax.scatter(
            x=x_nd[:n, d], y=np.zeros(n), 
            facecolor='none', edgecolor=cmap(d), s=25, #clip_on=False, 
            zorder=9
        ) # 標本
    for d in range(2):
        ax.plot(
            x_vec, pop_dens_lt[d], 
            color='black', linewidth=1.0, 
            zorder=10
        ) # 母分布
        for idx, mu in enumerate([mu_pop_d[d], x_bar_d[d]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                zorder=[20, 21][idx]
            ) # 母・標本平均
    ax.hlines(
        y=0.0, xmin=mu_pop_d[0], xmax=mu_pop_d[1], 
        color='red', linewidth=2.0, #clip_on=False, 
        zorder=12
    ) # 母平均の差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[*mu_pop_d, *x_bar_d], 
        labels=['$\\mu_1$', '$\\mu_2$', '$\\bar{x}_1$', '$\\bar{x}_2$']
    ) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax.set_title(pop_1d_param_lbl, loc='left')
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_d_max, ymax=(1.0+margin_ratio)*Px_d_max) # 表示範囲を固定, 余白を追加

    ##### 対標本の作図 -----

    # 標本の対応を描画
    ax = axes[2, 0]

    for d in range(2):
        ax.scatter(
            x=x_nd[:n, d], y=np.arange(n)+1, 
            facecolor='none', edgecolor=cmap(d), s=25, 
            zorder=10
        ) # 標本
    ax.hlines(
        y=np.arange(n)+1, xmin=x_nd[:n, 0], xmax=x_nd[:n, 1], 
        color='black', linewidth=1.0, linestyle=':', 
        label='sample difference', 
        zorder=11
    ) # 対標本の差
    for d in range(2):
        for idx, mu in enumerate([mu_pop_d[d], x_bar_d[d]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                zorder=[20, 21][idx]
            ) # 母・標本平均

    ax.set_xlabel('$x$')
    ax.set_ylabel('$n$')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=-margin_ratio*N, ymax=(1.0+margin_ratio)*N) # 表示範囲を固定
    ax.invert_yaxis() # 標本番号を昇順に表示

    ##### 差の母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=d_n[:n], bins=class_d_num, range=(bound_d_min, bound_d_max), density=True
    )

    # 差の母分布のラベルを作成
    diff_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}$\n'
    diff_param_lbl += '$\\mu_{diff} = \\mu_1 - \\mu_2 = '+f'{mu_diff:.2f}, '
    diff_param_lbl += '\\sigma_{diff} = \\sigma_1^2 + \\sigma_2^2 - 2 \\sigma_{1,2} = '+f'{sigma_diff:.2f}$\n'
    diff_param_lbl += '$L = \\bar{d}_{obs} - t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    diff_param_lbl += 'U = \\bar{d}_{obs} + t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 差の母分布を描画
    ax   = axes[0, 1]
    ax2x = axes2x[2]
    ax2y = axes2y[1]

    ax.bar(
        x=center_d_vec, height=obs_dens_vec, 
        width=class_d_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # 差の標本
    ax.scatter(
        x=d_n[:n], y=np.zeros(n), 
        facecolor='none', edgecolor='black', alpha=0.33, s=25, #clip_on=False, 
        zorder=9
    ) # 差の標本
    ax.plot(
        d_vec, diff_dens_vec, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 差の母分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean difference', None][idx], 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple', linewidth=2.0, #clip_on=False, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\delta = \\mu_1 - \\mu_2, d = x_1 - x_2$')
    ax2x.set_xticks(
        ticks =[delta_pop, ci_bound_lower, ci_bound_upper], 
        labels=['$\\delta_{pop}$', '$L$', '$U$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(d \\mid \\mu_{diff}, \\sigma_{diff}^2)$')
    ax2y.set_ylabel('$\\frac{N_d}{N}$')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax.set_title(diff_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # (目盛の共通化用), 余白を追加
    dens_vals = ax.get_yticks()            # 密度を取得
    freq_vals = dens_vals * class_d_size*n # 度数に変換
    Nd_max    = Pd_max    * class_d_size*n # 度数軸の範囲を設定
    ax2y.set_yticks(ticks=freq_vals, labels=[f'{y:.1f}' for y in freq_vals]) # 度数軸目盛
    ax2y.set_ylim(ymin=-margin_ratio*Nd_max, ymax=(1.0+margin_ratio)*Nd_max) # (目盛の共通化用)

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\bar{d}_{obs} = '+f'{d_bar:.2f}, '
    smp_param_lbl += '\\hat{\\sigma} = '+f'{sigma_hat:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_1 - \\mu_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\frac{\\hat{\\sigma}}{\\sqrt{N}} = '+f'{sigma_smp:.2f}$'

    # 標本分布を描画
    ax   = axes[1, 1]
    ax2x = axes2x[3]

    ax.scatter(
        x=d_bar, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        zorder=9
    ) # 差の標本平均
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=[None, 'saple mean difference'][idx], 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for delta in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値

    ax.set_xlabel('$\\bar{d} = \\frac{1}{n} \\sum_{i=1}^n d_i$')
    ax2x.set_xticks(
        ticks =[d_bar, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\bar{d}_{obs}$', 
            '$\\bar{d}_{obs} - t_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\bar{d}_{obs} + t_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{d} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_bar_max, ymax=(1.0+margin_ratio)*Pd_bar_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # t軸の範囲を設定
    if n > 1:
        t_min = (d_min - d_bar) / sigma_smp
        t_max = (d_max - d_bar) / sigma_smp
    else: # 自由度の条件を満たさない場合 
        # 標本統計量を計算
        tmp_n = 2
        tmp_d_bar     = np.mean(d_n[:tmp_n])
        tmp_sigma_hat = np.sqrt(np.var(d_n[:tmp_n], ddof=1))
        tmp_sigma_smp = tmp_sigma_hat / np.sqrt(tmp_n)
        t_min = (d_min - tmp_d_bar) / tmp_sigma_smp
        t_max = (d_max - tmp_d_bar) / tmp_sigma_smp

    # t軸の値を作成
    t_vec = np.linspace(start=t_min, stop=t_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = t.pdf(x=t_vec, df=n-1, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_t_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = t.pdf(x=cr_t_vec, df=n-1, loc=0.0, scale=1.0)

    # 標準化分布のラベルを作成
    std_param_lbl  = '$t_{obs} = '+f'{t_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += 't_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += 't_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2, 1]
    ax2x = axes2x[4]

    ax.fill_between(
        x=cr_t_vec, y1=np.zeros_like(cr_t_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=9
    ) # 中央領域
    ax.plot(
        t_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standardized statistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        t_val = (delta - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=t_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for idx, t_val in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=t_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$t = \\frac{\\bar{d} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[t_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$t_{obs}$', '$t_{1-\\frac{\\alpha}{2}}$', '$t_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$t(t \\mid n-1, 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=t_min, xmax=t_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=t_min, xmax=t_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pt_max, ymax=(1.0+margin_ratio)*Pt_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper left', prop={'size': 8}
    )

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'mean_difference_ci_n.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 試行回数の影響 ----------------------------------------------------------------

### シミュレーションの設定 -----

# 試行回数(フレーム数)を指定
iter_num = 100

# サンプルサイズを指定
N = 100


# %%

### 信頼区間の設定 -----

# 中央領域の範囲を計算
cr_bound_upper = t.ppf(q=1.0-0.5*alpha, df=N-1, loc=0.0, scale=1.0)
cr_bound_lower = -cr_bound_upper
print('t_α/2:', cr_bound_lower, cr_bound_upper)


# %%

### 表示範囲の設定 ----

# δ軸の範囲を設定
k = 4.0
u = 0.5
delta_size  = cr_bound_upper * sigma_diff / np.sqrt(N) # 信頼区間の半サイズ
delta_size *= k # 定数倍
delta_size  = np.ceil(delta_size /u)*u # u単位で切り上げ
delta_min   = delta_pop - delta_size
delta_max   = delta_pop + delta_size
print('δ-axis size:', delta_min, delta_max)


# 2D母分布のp軸の範囲を設定
u = 0.001
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 1D母分布のp軸の範囲を設定
u = 0.25
Px_d_max = np.max(pop_dens_lt)
Px_d_max = np.ceil(Px_d_max /u)*u # u単位で切り上げ
print('p(x_d) size:  ', Px_d_max)

# 差の母分布のp軸の範囲を設定
u = 0.05
Pd_max = diff_dens_vec.max()
Pd_max = np.ceil(Pd_max /u)*u # u単位で切り上げ
print('p(d) size:    ', Pd_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_bar_max = norm.pdf(x=mu_diff, loc=mu_diff, scale=sigma_diff/np.sqrt(N)) # 最頻値
Pd_bar_max = np.ceil(Pd_bar_max /u)*u # u単位で切り上げ
print('p(d bar) size:', Pd_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pt_max = t.pdf(x=0.0, df=N-1, loc=0.0, scale=1.0) # 最頻値
Pt_max = np.ceil(Pt_max /u)*u # u単位で切り上げ
print('p(t) size:    ', Pt_max)


# %%

### 信頼区間の可視化 -----

# 配色を設定
cmap = plt.get_cmap('tab10') # カラーマップを指定

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig = plt.figure(
    figsize=(18, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle(
    'Population Mean Difference Confidence Interval (paired samples, unknown variance)', 
    fontsize=20
)

# 図を分割
gs = GridSpec(nrows=3, ncols=3, figure=fig)
ax_00, ax_01 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_10, ax_11 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_20, ax_21 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
ax_a2  = fig.add_subplot(gs[:, 2])
axes   = [ax_00, ax_10, ax_20, ax_01, ax_11, ax_21, ax_a2]
axes2x = [axes[idx].twiny() for idx in [0, 1, 3, 4, 5, 6]]
axes2y = [ax_00.twinx()]

# カウントを初期化
cover_cnt = 0

# 受け皿を初期化
res_lt = []

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(I):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2x.cla() for ax2x in axes2x]
    [ax2y.cla() for ax2y in axes2y]

    # オブジェクトを取得
    global cover_cnt

    # 値を設定
    I = I + 1 # 試行回数

    ##### 乱数の生成 -----

    # 標本を生成
    x_nd = np.random.multivariate_normal(mean=mu_pop_d, cov=sigma2_pop_dd, size=N)

    # 対標本の差を計算
    d_n = np.subtract(*x_nd.T)

    # 標本統計量を計算
    x_bar_d       = np.mean(x_nd, axis=0)
    sigma2_hat_d  = np.var(x_nd, ddof=1, axis=0)
    sigma_hat_d   = np.sqrt(sigma2_hat_d)
    sigma_hat_12  = np.cov(x_nd[:, 0], x_nd[:, 1], ddof=1)[0, 1]
    sigma2_hat_dd = np.array(
        [[sigma2_hat_d[0], sigma_hat_12], 
        [sigma_hat_12, sigma2_hat_d[1]]]
    )

    # 差の標本統計量を計算
    d_bar      = np.mean(d_n)
    sigma2_hat = np.var(d_n, ddof=1)
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_diff
    sigma_smp = sigma_hat / np.sqrt(N)

    # 標本統計量を標準化
    t_obs = (d_bar - mu_smp) / sigma_smp

    # 信頼区間の範囲を計算
    ci_bound_lower = d_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = d_bar + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= delta_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図:2次元 -----

    # 密度を集計
    obs_dens_grid, _, _ = np.histogram2d(
        x=x_nd[:, 0], y=x_nd[:, 1], 
        bins=(class_x_num, class_x_num), 
        range=[(bound_x_min, bound_x_max), (bound_x_min, bound_x_max)], 
        density=True
    )

    # 母分布のラベルを作成
    pop_2d_param_lbl  = f'$i = {I}, N = {N}$\n'
    tmp_mu_str        = ', '.join([f'{mu:.2f}' for mu in mu_pop_d])
    tmp_sgm2_1_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_pop_dd[0, :]])
    tmp_sgm2_2_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_pop_dd[1, :]])
    pop_2d_param_lbl += f'$\\mu_{{pop}} = ({tmp_mu_str}), '
    pop_2d_param_lbl += f'\\Sigma_{{pop}} = \\binom{{{tmp_sgm2_1_str}}}{{{tmp_sgm2_2_str}}}$\n'
    tmp_x_str         = ', '.join([f'{x:.2f}' for x in x_bar_d])
    tmp_sgm2_1_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_hat_dd[0, :]])
    tmp_sgm2_2_str    = ', '.join([f'{sgm2:.2f}' for sgm2 in sigma2_hat_dd[1, :]])
    pop_2d_param_lbl += f'$\\bar{{x}}_i = ({tmp_x_str}), '
    pop_2d_param_lbl += f'\\hat{{\\Sigma}}_i = \\binom{{{tmp_sgm2_1_str}}}{{{tmp_sgm2_2_str}}}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]
    ax2y = axes2y[0]

    ax.pcolormesh(
        center_x_vec, center_x_vec, obs_dens_grid.T, 
        shading='nearest', 
        cmap='viridis', vmin=0.0, vmax=Px_max, alpha=0.5, 
        zorder=8
    ) # 標本
    ax.scatter(
        x=x_nd[:, 0], y=x_nd[:, 1], 
        color='black', alpha=0.33, s=25, 
        zorder=9
    ) # 標本
    ax.plot(
        [], [], 
        color='purple', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # (凡例表示用のダミー)
    ax.contour(
        x_1_grid, x_2_grid, pop_dens_vec.reshape(x_dims), 
        cmap='viridis', vmin=0.0, vmax=Px_max, 
        linewidths=1.0, 
        zorder=10
    ) # 母分布
    for idx, mu in enumerate([mu_pop_d[0], x_bar_d[0]]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population means', 'sample means'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for idx, mu in enumerate([mu_pop_d[1], x_bar_d[1]]):
        ax.axhline(
            y=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    ax.plot(
        x_vec, x_vec, 
        color='black', linewidth=1.0, 
        zorder=30
    ) # 恒等関数

    ax.set_xlabel('$x_1$')
    ax2x.set_xticks(
        ticks =[mu_pop_d[0], x_bar_d[0]], 
        labels=['$\\mu_1$', '$\\bar{x}_{i,1}$']
    ) # パラメータのラベル
    ax.set_ylabel('$x_2$')
    ax2y.set_yticks(
        ticks =[mu_pop_d[1], x_bar_d[1]], 
        labels=['$\\mu_2$', '$\\bar{x}_{i,2}$']
    ) # パラメータのラベル
    ax.set_title(pop_2d_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=x_min, ymax=x_max)   # (目盛の共通化用)
    ax2y.set_ylim(ymin=x_min, ymax=x_max) # (目盛の共通化用)

    ##### 母分布の作図:1次元 -----

    # 密度を集計
    obs_dens_lt = [
        np.histogram(
            a=x_nd[:, d], bins=class_x_num, range=(bound_x_min, bound_x_max), density=True
        )[0] for d in range(2)
    ]

    # 母分布のラベルを作成
    pop_1d_param_lbl  = f'$\\mu_1 = {mu_pop_d[0]:.2f}, '
    pop_1d_param_lbl += f'\\sigma_1 = {np.sqrt(sigma2_pop_dd[0, 0]):.2f}, '
    pop_1d_param_lbl += '\\bar{x}_{i,1} = '+f'{x_bar_d[0]:.2f}, '
    pop_1d_param_lbl += '\\hat{\\sigma}_{i,1} = '+f'{sigma_hat_d[0]:.2f}$\n'
    pop_1d_param_lbl += f'$\\mu_2 = {mu_pop_d[1]:.2f}, '
    pop_1d_param_lbl += f'\\sigma_2 = {np.sqrt(sigma2_pop_dd[1, 1]):.2f}, '
    pop_1d_param_lbl += '\\bar{x}_{i,2} = '+f'{x_bar_d[1]:.2f}, '
    pop_1d_param_lbl += '\\hat{\\sigma}_{i,2} = '+f'{sigma_hat_d[1]:.2f}$'

    # 母分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    for d in range(2):
        ax.bar(
            x=center_x_vec, height=obs_dens_lt[d], 
            width=class_x_size, 
            color=cmap(d), alpha=0.5, 
            zorder=8
        ) # 標本
        ax.scatter(
            x=x_nd[:, d], y=np.zeros(N), 
            facecolor='none', edgecolor=cmap(d), s=25, #clip_on=False, 
            zorder=9
        ) # 標本
    for d in range(2):
        ax.plot(
            x_vec, pop_dens_lt[d], 
            color='black', linewidth=1.0, 
            zorder=10
        ) # 母分布
        for idx, mu in enumerate([mu_pop_d[d], x_bar_d[d]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                zorder=[20, 21][idx]
            ) # 母・標本平均
    ax.hlines(
        y=0.0, xmin=mu_pop_d[0], xmax=mu_pop_d[1], 
        color='red', linewidth=2.0, 
        zorder=12
    ) # 母平均の差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[*mu_pop_d, *x_bar_d], 
        labels=['$\\mu_1$', '$\\mu_2$', '$\\bar{x}_{i,1}$', '$\\bar{x}_{i,2}$']
    ) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax.set_title(pop_1d_param_lbl, loc='left')
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_d_max, ymax=(1.0+margin_ratio)*Px_d_max) # 表示範囲を固定, 余白を追加

    ##### 対標本の作図 -----

    # 標本の対応を描画
    ax = axes[2]

    for d in range(2):
        ax.scatter(
            x=x_nd[:, d], y=np.arange(N)+1, 
            facecolor='none', edgecolor=cmap(d), s=25, 
            zorder=10
        ) # 標本
    ax.hlines(
        y=np.arange(N)+1, xmin=x_nd[:, 0], xmax=x_nd[:, 1], 
        color='black', linewidth=1.0, linestyle=':', 
        label='sample difference', 
        zorder=11
    ) # 対標本の差
    for d in range(2):
        for idx, mu in enumerate([mu_pop_d[d], x_bar_d[d]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                zorder=[20, 21][idx]
            ) # 母・標本平均

    ax.set_xlabel('$x$')
    ax.set_ylabel('$n$')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)
    ax.set_ylim(ymin=-margin_ratio*N, ymax=(1.0+margin_ratio)*N) # 表示範囲を固定
    ax.invert_yaxis() # 標本番号を昇順に表示

    ##### 差の母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=d_n, bins=class_d_num, range=(bound_d_min, bound_d_max), density=True
    )

    # 差の母分布のラベルを作成
    diff_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}$\n'
    diff_param_lbl += '$\\mu_{diff} = \\mu_1 - \\mu_2 = '+f'{mu_diff:.2f}, '
    diff_param_lbl += '\\sigma_{diff} = \\sigma_1^2 + \\sigma_2^2 - 2 \\sigma_{1,2} = '+f'{sigma_diff:.2f}$\n'
    diff_param_lbl += '$L = \\bar{d}_i - t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    diff_param_lbl += 'U = \\bar{d}_i + t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 差の母分布を描画
    ax   = axes[3]
    ax2x = axes2x[2]

    ax.bar(
        x=center_d_vec, height=obs_dens_vec, 
        width=class_d_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # 差の標本
    ax.scatter(
        x=d_n, y=np.zeros(N), 
        facecolor='none', edgecolor='black', alpha=0.5, s=25, #clip_on=False, 
        zorder=9
    ) # 差の標本
    ax.plot(
        d_vec, diff_dens_vec, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 差の母分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean difference', None][idx], 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple' if cover_flg else 'blue', linewidth=2.0, clip_on=False, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\delta = \\mu_1 - \\mu_2, d = x_1 - x_2$')
    ax2x.set_xticks(
        ticks =[delta_pop, ci_bound_lower, ci_bound_upper], 
        labels=['$\\delta_{pop}$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(d \\mid \\mu_{diff}, \\sigma_{diff}^2)$')
    ax.set_title(diff_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # 余白を追加

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\bar{d}_i = '+f'{d_bar:.2f}, '
    smp_param_lbl += '\\hat{\\sigma}_i = '+f'{sigma_hat:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_1 - \\mu_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\frac{\\hat{\\sigma}_i}{\\sqrt{N}} = '+f'{sigma_smp:.2f}$'

    # 標本分布を描画
    ax   = axes[4]
    ax2x = axes2x[3]

    ax.scatter(
        x=d_bar, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        zorder=9
    ) # 差の標本平均
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=[None, 'saple mean difference'][idx], 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for delta in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値

    ax.set_xlabel('$\\bar{d} = \\frac{1}{n} \\sum_{j=1}^n d_j$')
    ax2x.set_xticks(
        ticks =[d_bar, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\bar{d}_i$', 
            '$\\bar{d}_i - t_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\bar{d}_i + t_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{d} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_bar_max, ymax=(1.0+margin_ratio)*Pd_bar_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # t軸の範囲を設定
    t_min = (d_min - d_bar) / sigma_smp
    t_max = (d_max - d_bar) / sigma_smp

    # t軸の値を作成
    t_vec = np.linspace(start=t_min, stop=t_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = t.pdf(x=t_vec, df=N-1, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_t_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = t.pdf(x=cr_t_vec, df=N-1, loc=0.0, scale=1.0)

    # 外側領域を計算
    tail_t_vec    = np.hstack([
        np.linspace(start=t_min, stop=cr_bound_lower, num=251), 
        np.nan, # (塗りつぶしの分割用)
        np.linspace(start=cr_bound_upper, stop=t_max, num=251)
    ])
    tail_dens_vec = t.pdf(x=tail_t_vec, df=N-1, loc=0.0, scale=1.0)

    # 標準化分布のラベルを作成
    std_param_lbl     = '$t_i = '+f'{t_obs:.2f}$\n'
    std_param_lbl    += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl    += 't_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl    += 't_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[5]
    ax2x = axes2x[4]

    ax.fill_between(
        x=cr_t_vec, y1=np.zeros_like(cr_t_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=8
    ) # 中央領域
    ax.fill_between(
        x=tail_t_vec, y1=np.zeros_like(tail_t_vec), y2=tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        label='tail regions', 
        zorder=9
    ) # 外側領域
    ax.plot(
        t_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standardized statistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, delta in enumerate([delta_pop, d_bar]):
        t_val = (delta - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=t_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母平均の差・差の標本平均
    for idx, t_val in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=t_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$t = \\frac{\\bar{d} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[t_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$t_i$', '$t_{1-\\frac{\\alpha}{2}}$', '$t_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$t(t \\mid n-1, 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=t_min, xmax=t_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=t_min, xmax=t_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pt_max, ymax=(1.0+margin_ratio)*Pt_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [2, 3, 0, 1] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper left', prop={'size': 8}
    )

    ##### 信頼区間の作図 -----

    # 新規の推定結果を記録
    res_lt.append(
        ([ci_bound_lower, ci_bound_upper], cover_flg)
    )

    # 推定結果のラベルを作成
    non_cover_cnt = I - cover_cnt
    ci_res_lbl    = f'non-coverage: {non_cover_cnt} / {I} ( {non_cover_cnt/I:.3f} )'
  
    # 信頼区間を描画
    ax  = axes[6]
    ax2x = axes2x[5]

    for idx, delta in enumerate([delta_pop, d_bar]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=[2.0, 1.0][idx], linestyle=['-', '--'][idx], 
            label=['population mean diffecence', 'sample mean difference'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均の差
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    for i in range(I):
        # 過去の推定結果を取得
        [delta_lower, delta_upper], cover_flg = res_lt[i]
        ax.hlines(
            y=i+1, xmin=delta_lower, xmax=delta_upper, 
            color='purple' if cover_flg else 'blue', linewidth=2.0, 
            label='confidence interval' if i+1 == I else None, 
            zorder=30
        ) # 信頼区間
    
    ax.set_xlabel('$\\delta$')
    ax2x.set_xticks(
        ticks =[delta_pop, d_bar, ci_bound_lower, ci_bound_upper], 
        labels=['$\\delta_{pop}$', '$\\bar{d}_i$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(ci_res_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=delta_min, xmax=delta_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=delta_min, xmax=delta_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1) # 表示範囲を固定
    ax.invert_yaxis() # 推定結果を昇順に表示

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=iter_num, interval=500
)

# 動画を書出
anim.save(
    filename=dir_path+'mean_difference_ci_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


