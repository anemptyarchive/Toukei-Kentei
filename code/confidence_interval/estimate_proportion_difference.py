
# 区間推定 ----------------------------------------------------------------------

# ch3.5.4

# 二項分布
# 母比率の差の信頼区間


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
dir_path += 'estimate_proportion_difference/' # フォルダを指定
print(dir_path)


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import binom, norm
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.animation import FuncAnimation


# %%

# 共通の設定 ------------------------------------------------------------------

### 母集団の設定 -----

# 母分布のパラメータを指定
p_pop_lt = [0.6, 0.25]

# 母比率の差を計算
delta_pop = np.subtract(*p_pop_lt)


# %%

### 信頼区間の設定 -----

# 信頼係数を指定
gamma = 0.95

# 有意水準を計算
alpha = 1.0 - gamma
print('α:    ', alpha)

# 中央領域の範囲を計算
cr_bound_upper = norm.ppf(q=1.0-0.5*alpha, loc=0.0, scale=1.0)
cr_bound_lower = -cr_bound_upper
print('z_α/2:', cr_bound_lower, cr_bound_upper)


# %%

# サンプルサイズの影響 -----------------------------------------------------------

### シミュレーションの設定 -----

# 追加サンプルサイズ(フレーム数)を指定
N = 100

# サンプルサイズの初期値を指定
N_1_base = 0
N_2_base = 0


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.05
Px_max = max([max([1.0-p, p]) for p in p_pop_lt]) # 初回試行の最頻値
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_max = norm.pdf(
    x=delta_pop, 
    loc=delta_pop, 
    scale=np.sqrt(
        np.sum([p * (1.0-p) / n for n, p in zip([N_1_base+N, N_2_base+N], p_pop_lt)])
    )
) # 最終試行の最頻値
Pd_max = np.ceil(Pd_max /u)*u # u単位で切り上げ
print('p(d) size:', Pd_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print('p(z) size:', Pz_max)


# %%

### 信頼区間の可視化 -----

# 配色を設定
cmap = plt.get_cmap('tab10') # カラーマップを指定

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Proportion Difference Confidence Interval', fontsize=20)
axes2x = [ax.twiny() for ax in axes]

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2x.cla() for ax2x in axes2x]

    ##### 乱数の生成 -----

    # 値を設定
    N_add = i + 1 # 追加サンプル数
    N_lt  = [N_1_base+N_add, N_2_base+N_add] # サンプル数

    # シードを設定:(ノートとの対応用)
    np.random.seed(17) # (変化の安定化用)

    # 標本を生成
    x_lt = [
        np.random.binomial(
            n=N_lt[idx], p=p_pop_lt[idx], size=1
        )[0] for idx in range(2)
    ]
    #x_lt = [np.floor((n+1) * p) for n, p in zip(N_lt, p_pop_lt)] # 最頻値で代用:(変化の安定化用)

    # 標本統計量を計算
    p_hat_lt = [x / n for n, x in zip(N_lt, x_lt)]

    # 標本分布のパラメータを計算
    mu_smp     = np.subtract(*p_pop_lt)
    sigma2_smp = np.sum([p * (1.0-p) / n for n, p in zip(N_lt, p_hat_lt)])
    sigma_smp  = np.sqrt(sigma2_smp)

    # 標本比率の差を計算
    d_obs = np.subtract(*p_hat_lt)

    # 標本統計量を標準化
    z_obs = (d_obs - mu_smp) / sigma_smp if sigma_smp != 0.0 else np.nan

    # 信頼区間の範囲を計算
    ci_bound_lower = d_obs + cr_bound_lower * sigma_smp
    ci_bound_upper = d_obs + cr_bound_upper * sigma_smp

    ##### 母分布の作図 -----

    # x軸の範囲を設定
    x_margin = 1 # (整数)
    x_min    = -x_margin 
    x_max    = max(N_lt) + x_margin

    # p軸の範囲を設定
    p_min = x_min / max(N_lt)
    p_max = x_max / max(N_lt)

    # x軸の値を作成
    x_vec = np.arange(start=x_min, stop=x_max+1, step=1)

    # 母分布の確率を計算
    pop_prob_lt = [
        binom.pmf(
            k=x_vec, n=N_lt[idx], p=p_pop_lt[idx]
        ) for idx in range(2)
    ]

    # 母分布のラベルを作成
    pop_param_lbl  = f'$N_1 = {N_lt[0]}, '
    pop_param_lbl += f'p_1 = {p_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\hat{p}_1 = '+f'{p_hat_lt[0]:.2f}$\n'
    pop_param_lbl += f'$N_2 = {N_lt[1]}, '
    pop_param_lbl += f'p_2 = {p_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\hat{p}_2 = '+f'{p_hat_lt[1]:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    for pop_idx in range(2):
        ax.plot(
            x_vec/N_lt[pop_idx], pop_prob_lt[pop_idx], 
            color='black', linewidth=1.0, 
            label='population distribution' if pop_idx == 0 else None, 
            zorder=10
        ) # 母分布
        ax.scatter(
            x=x_vec/N_lt[pop_idx], y=pop_prob_lt[pop_idx], 
            color='black', s=30, 
            zorder=10
        ) # 母分布
        ax.scatter(
            x=p_hat_lt[pop_idx], y=0.0, 
            color=cmap(pop_idx), s=100, 
            label='sample' if pop_idx == 0 else None, 
            zorder=15
        ) # 標本比率
        for idx, p, in enumerate([p_pop_lt[pop_idx], p_hat_lt[pop_idx]]):
            ax.axvline(
                x=p, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                label=['population proportions', 'sample proportions'][idx] if pop_idx == 0 else None, 
                zorder=[20, 21][idx]
            ) # 母・標本比率
    for delta in [ci_bound_lower, ci_bound_upper]:
        x = delta + p_pop_lt[1] # 並行移動
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=p_pop_lt[0], xmax=p_pop_lt[1], 
        color='red', linewidth=2.0, 
        zorder=30
    ) # 母比率の差

    ax.set_xlabel('$p, \\hat{p} = \\frac{x}{n}$')
    ax2x.set_xticks(
        ticks =p_pop_lt+[p+1e-10 for p in p_hat_lt]+[ci_bound_lower+p_pop_lt[1], ci_bound_upper+p_pop_lt[1]], 
        labels=['$p_1$', '$p_2$', '$\\hat{p}_1$', '$\\hat{p}_2$', '$L + p_2$', '$U + p_2$']
    ) # パラメータのラベル
    ax.set_ylabel('$Bin(x \\mid n_{pop}, p_{pop})$')
    ax.set_title(pop_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=p_min, xmax=p_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=p_min, xmax=p_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 表示範囲を固定, 余白を追加
    
    # ラベルの装飾を調整(表示順の変更用)
    order = [0, 2, 3, 1] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper left', prop={'size': 8}
    )

    ##### 標本分布の作図 -----

    # d軸の範囲を設定
    d_min = p_min - p_pop_lt[1]
    d_max = p_max - p_pop_lt[1]

    # d軸の値を作成
    d_vec = np.linspace(start=d_min, stop=d_max, num=1001)

    # 標本分布の確率密度を計算
    if sigma_smp != 0.0:
        smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)
    else: # 標準偏差の条件を満たさない場合
        smp_dens_vec = np.tile(np.nan, reps=len(d_vec))

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}, '
    smp_param_lbl += 'd_{obs} = '+f'{d_obs:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = p_1 - p_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\sqrt{\\frac{\\hat{p}_1 (1-\\hat{p}_1)}{N_1} + \\frac{\\hat{p}_2 (1-\\hat{p}_2)}{N_2}} = '+f'{sigma_smp:.2f}$\n'
    smp_param_lbl += '$L = d_{obs} - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    smp_param_lbl += 'U = d_{obs} + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=d_obs, y=0.0, 
        color='#00A968', s=100, 
        zorder=9
    ) # 標本比率の差
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, 0.0, d_obs]):
        ax.axvline(
            x=delta, 
            color=['red', 'red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population proportion difference', None, 'saple mean difference'][idx], 
            zorder=[20, 20, 21][idx]
        ) # 母・標本比率の差
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple', linewidth=2.0, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\delta = p_1 - p_2, d = \\hat{p}_1 - \\hat{p}_2$')
    ax2x.set_xticks(
        ticks =[delta_pop, d_obs, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\delta_{pop}$', 
            '$d_{obs}$', 
            '$d_{obs} - z_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$d_{obs} + z_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(d \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # 表示範囲を固定, 余白を追加

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    if sigma_smp != 0.0:
        z_min = (d_min - d_obs) / sigma_smp
        z_max = (d_max - d_obs) / sigma_smp
    else: #標準偏差の条件を満たさない場合
        tmp_sigma_smp = 0.5 # 最大値
        z_min = (d_min - d_obs) / tmp_sigma_smp
        z_max = (d_max - d_obs) / tmp_sigma_smp

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_z_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = norm.pdf(x=cr_z_vec, loc=0.0, scale=1.0)

    # 標準化分布のラベルを作成
    std_param_lbl  = '$z_{obs} = '+f'{z_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += 'z_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += 'z_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2]
    ax2x = axes2x[2]

    ax.fill_between(
        x=cr_z_vec, y1=np.zeros_like(cr_z_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=9
    ) # 中央領域
    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standardized statistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, delta in enumerate([delta_pop, d_obs]):
        z = (delta - mu_smp) / sigma_smp if sigma_smp != 0.0 else np.nan # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率の差
    for idx, z in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=z, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$z = \\frac{d - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[z_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$z_{obs}$', '$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

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
    filename=dir_path+'proportion_difference_ci_n.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 試行回数の影響 ----------------------------------------------------------------

# 試行回数(フレーム数)を指定
iter_num = 100

# サンプルサイズを指定
N_lt = [20, 15]


# %%

### 変数の設定 -----

# x軸の範囲を設定
x_lower  = 0
x_upper  = max(N_lt)
x_margin = 0.5
x_min    = x_lower - x_margin
x_max    = x_upper + x_margin
print('x-axis size:', x_min, x_max)

# x軸の値を作成
x_vec = np.arange(start=x_lower, stop=x_upper+1, step=1)


# p軸の範囲を設定
p_min = x_min / max(N_lt)
p_max = x_max / max(N_lt)


# d軸の範囲を設定
d_min = p_min - p_pop_lt[1]
d_max = p_max - p_pop_lt[1]

# d軸の値を作成
d_vec = np.linspace(start=d_min, stop=d_max, num=1001)


# %%

### 分布の計算 -----

# 母分布の確率を計算
pop_prob_lt = [
    binom.pmf(
        k=x_vec, n=N_lt[idx], p=p_pop_lt[idx]
    ) for idx in range(2)
]


# %%

### 表示範囲の設定 -----

# δ軸の範囲を設定
k = 4.0
u = 0.5
delta_size  = cr_bound_upper * np.sqrt(np.sum([p * (1.0-p) / n for n, p in zip(N_lt, p_pop_lt)])) # 信頼区間の半サイズ
delta_size *= k # 定数倍
delta_size  = np.ceil(delta_size /u)*u # u単位で切り上げ
delta_min   = delta_pop - delta_size
delta_max   = delta_pop + delta_size
print('δ-axis size:', delta_min, delta_max)


# 母分布のp軸の範囲を設定
u = 0.25
Px_max = np.max(pop_prob_lt)
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_max = norm.pdf(
    x=delta_pop, 
    loc=delta_pop, 
    scale=np.sqrt(
        np.sum([p * (1.0-p) / n for n, p in zip(N_lt, p_pop_lt)])
    )
) # 最頻値
Pd_max = np.ceil(Pd_max /u)*u # u単位で切り上げ
print('p(d) size:', Pd_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print('p(z) size:', Pz_max)


# %%

### 信頼区間の可視化 -----

# 配色を設定
cmap = plt.get_cmap('tab10') # カラーマップを指定

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Proportion Difference Confidence Interval', fontsize=20)

# 図を分割
gs     = GridSpec(nrows=3, ncols=2, figure=fig)
ax_00  = fig.add_subplot(gs[0, 0])
ax_10  = fig.add_subplot(gs[1, 0])
ax_20  = fig.add_subplot(gs[2, 0])
ax_a1  = fig.add_subplot(gs[:, 1])
axes   = [ax_00, ax_10, ax_20, ax_a1]
axes2x = [ax.twiny() for ax in axes]

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

    # オブジェクトを取得
    global cover_cnt

    # 値を設定
    I = I + 1 # 試行回数

    ##### 乱数の生成 -----

    # 標本を生成
    x_lt = [
        np.random.binomial(
            n=N_lt[idx], p=p_pop_lt[idx], size=1
        )[0] for idx in range(2)
    ]

    # 標本統計量を計算
    p_hat_lt = [x / n for n, x in zip(N_lt, x_lt)]

    # 標本分布のパラメータを計算
    mu_smp     = np.subtract(*p_pop_lt)
    sigma2_smp = np.sum([p * (1.0-p) / n for n, p in zip(N_lt, p_hat_lt)])
    sigma_smp  = np.sqrt(sigma2_smp)

    # 標本比率の差を計算
    d_obs = np.subtract(*p_hat_lt)

    # 標本統計量を標準化
    z_obs = (d_obs - mu_smp) / sigma_smp if sigma_smp != 0.0 else np.nan

    # 信頼区間の範囲を計算
    ci_bound_lower = d_obs + cr_bound_lower * sigma_smp
    ci_bound_upper = d_obs + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= delta_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図 -----

    # 母分布のラベルを作成
    pop_param_lbl  = f'$i = {I}$\n'
    pop_param_lbl += f'$N_1 = {N_lt[0]}, '
    pop_param_lbl += f'p_1 = {p_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\hat{p}_{i,1} = '+f'{p_hat_lt[0]:.2f}$\n'
    pop_param_lbl += f'$N_2 = {N_lt[1]}, '
    pop_param_lbl += f'p_2 = {p_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\hat{p}_{i,2} = '+f'{p_hat_lt[1]:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    for pop_idx in range(2):
        ax.plot(
            x_vec/N_lt[pop_idx], pop_prob_lt[pop_idx], 
            color='black', linewidth=1.0, 
            label='population distribution' if pop_idx == 0 else None, 
            zorder=10
        ) # 母分布
        ax.scatter(
            x=x_vec/N_lt[pop_idx], y=pop_prob_lt[pop_idx], 
            color='black', s=30, 
            zorder=10
        ) # 母分布
        ax.scatter(
            x=p_hat_lt[pop_idx], y=0.0, 
            color=cmap(pop_idx), s=100, 
            label='sample' if pop_idx == 0 else None, 
            zorder=15
        ) # 標本比率
        for idx, p, in enumerate([p_pop_lt[pop_idx], p_hat_lt[pop_idx]]):
            ax.axvline(
                x=p, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                label=['population proportions', 'sample proportions'][idx] if pop_idx == 0 else None, 
                zorder=[20, 21][idx]
            ) # 母・標本比率
    for delta in [ci_bound_lower, ci_bound_upper]:
        x = delta + p_pop_lt[1] # 並行移動
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=p_pop_lt[0], xmax=p_pop_lt[1], 
        color='red', linewidth=2.0, 
        zorder=30
    ) # 母比率の差

    ax.set_xlabel('$p, \\hat{p} = \\frac{x}{n}$')
    ax2x.set_xticks(
        ticks =p_pop_lt+[p+1e-10 for p in p_hat_lt]+[ci_bound_lower+p_pop_lt[1], ci_bound_upper+p_pop_lt[1]], 
        labels=['$p_1$', '$p_2$', '$\\hat{p}_{i,1}$', '$\\hat{p}_{i,2}$', '$L_i + p_2$', '$U_i + p_2$']
    ) # パラメータのラベル
    ax.set_ylabel('$Bin(x \\mid n_{pop}, p_{pop})$')
    ax.set_title(pop_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=p_min, xmax=p_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=p_min, xmax=p_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 表示範囲を固定, 余白を追加
    
    # ラベルの装飾を調整(表示順の変更用)
    order = [0, 2, 3, 1] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper left', prop={'size': 8}
    )

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    if sigma_smp != 0.0:
        smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)
    else: # 標準偏差の条件を満たさない場合
        smp_dens_vec = np.tile(np.nan, reps=len(d_vec))

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}, '
    smp_param_lbl += 'd_i = '+f'{d_obs:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = p_1 - p_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\sqrt{\\frac{\\hat{p}_{i,1} (1-\\hat{p}_{i,1})}{N_1} + \\frac{\\hat{p}_{i,2} (1-\\hat{p}_{i,2})}{N_2}} = '+f'{sigma_smp:.2f}$\n'
    smp_param_lbl += '$L_i = d_i - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    smp_param_lbl += 'U_i = d_i + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=d_obs, y=0.0, 
        color='#00A968', s=100, 
        zorder=9
    ) # 標本比率の差
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, 0.0, d_obs]):
        ax.axvline(
            x=delta, 
            color=['red', 'red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population proportion difference', None, 'saple mean difference'][idx], 
            zorder=[20, 20, 21][idx]
        ) # 母・標本比率の差
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple', linewidth=2.0, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\delta = p_1 - p_2, d = \\hat{p}_1 - \\hat{p}_2$')
    ax2x.set_xticks(
        ticks =[delta_pop, d_obs, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\delta_{pop}$', 
            '$d_i$', 
            '$d_i - z_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$d_i + z_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(d \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # 表示範囲を固定, 余白を追加

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    if sigma_smp != 0.0:
        z_min = (d_min - d_obs) / sigma_smp
        z_max = (d_max - d_obs) / sigma_smp
    else: #標準偏差の条件を満たさない場合
        tmp_sigma_smp = 0.5 # 最大値
        z_min = (d_min - d_obs) / tmp_sigma_smp
        z_max = (d_max - d_obs) / tmp_sigma_smp

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_z_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = norm.pdf(x=cr_z_vec, loc=0.0, scale=1.0)

    # 外側領域を計算
    tail_z_vec    = np.hstack([
        np.linspace(start=z_min, stop=cr_bound_lower, num=251), 
        np.nan, # (塗りつぶしの分割用)
        np.linspace(start=cr_bound_upper, stop=z_max, num=251)
    ])
    tail_dens_vec = norm.pdf(x=tail_z_vec, loc=0.0, scale=1.0)

    # 標準化分布のラベルを作成
    std_param_lbl  = '$z_i = '+f'{z_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += 'z_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += 'z_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2]
    ax2x = axes2x[2]

    ax.fill_between(
        x=cr_z_vec, y1=np.zeros_like(cr_z_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=8
    ) # 中央領域
    ax.fill_between(
        x=tail_z_vec, y1=np.zeros_like(tail_z_vec), y2=tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        label='tail regions', 
        zorder=9
    ) # 外側領域
    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standardized statistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, delta in enumerate([delta_pop, d_obs]):
        z = (delta - mu_smp) / sigma_smp if sigma_smp != 0.0 else np.nan # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率の差
    for idx, z in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=z, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$z = \\frac{d - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[z_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$z_i$', '$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 0] # 表示順を指定
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
    ax   = axes[3]
    ax2x = axes2x[3]

    for idx, delta in enumerate([delta_pop, d_obs]):
        ax.axvline(
            x=delta, 
            color=['red', 'black'][idx], linewidth=[2.0, 1.0][idx], linestyle=['-', '--'][idx], 
            label=['population proportion diffecence', 'sample proportion difference'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本比率の差
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
        ticks =[delta_pop, d_obs, ci_bound_lower, ci_bound_upper], 
        labels=['$\\delta_{pop}$', '$d_i$', '$L_i$', '$U_i$']
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
    filename=dir_path+'proportion_difference_ci_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


