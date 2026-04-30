
# 区間推定-----------------------------------------------------------------------

# ch3.5.1

# 正規分布
# 母平均の差の信頼区間
# 母分散が既知の場合


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm
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
dir_path += 'estimate_mean_difference_known_variance/' # フォルダを指定
print(dir_path)


# %%

# 共通の設定 ------------------------------------------------------------------

### 母集団の設定 -----

# 母分布のパラメータを指定
mu_pop_lt     = [10.0, 4.0]
sigma_pop_lt  = [3.0, 2.0]
sigma2_pop_lt = [sgm**2 for sgm in sigma_pop_lt]

# 母平均の差を計算
delta_pop = np.subtract(*mu_pop_lt)


# %%

### 変数の設定 -----

# x軸の範囲を設定
k = 3.0
u = 1.0
x_min = np.min([mu - k*sgm for mu, sgm in zip(mu_pop_lt, sigma_pop_lt)]) # 標準偏差の定数倍
x_max = np.max([mu + k*sgm for mu, sgm in zip(mu_pop_lt, sigma_pop_lt)]) # 標準偏差の定数倍
x_min = np.floor(x_min /u)*u # u単位で切り下げ
x_max = np.ceil(x_max /u)*u  # u単位で切り上げ
print('x-axis size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# d軸の範囲を設定
d_min = x_min - mu_pop_lt[1]
d_max = x_max - mu_pop_lt[1]
print('d-axis size:', d_min, d_max)

# d軸の値を作成
d_vec = np.linspace(start=d_min, stop=d_max, num=1001)


# %%

### 集計の設定 -----

# 階級幅を指定
class_size = 1.0

# 階級数を計算
class_num = ((x_max - x_min) // class_size).astype(dtype='int') + 1
print('class size:', class_size)

# 階級幅を再設定:(割り切れない場合)
class_size = (x_max - x_min) / (class_num-1)
print('class num: ', class_num)

# 階級値を作成
center_vec = np.arange(start=x_min, stop=x_max+class_size, step=class_size)

# 境界値の範囲を設定
bound_min = x_min - 0.5*class_size
bound_max = x_max + 0.5*class_size

# 境界値を作成
bound_vec = np.arange(start=bound_min, stop=bound_max+0.5*class_size, step=class_size)


# %%

### 分布の計算 -----

# 母分布の確率密度を計算
pop_dens_lt = [
    norm.pdf(
        x=x_vec, loc=mu_pop_lt[idx], scale=sigma_pop_lt[idx]
    ) for idx in range(2)
]


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

### 乱数の生成 -----

# シードを設定:(ノートとの対応用)
np.random.seed(320)

# 追加サンプルサイズ(フレーム数)を指定
N = 100

# サンプルサイズの初期値を指定
N_1_base = 0
N_2_base = 0

# 標本を生成
x_lt = [
    np.random.normal(
        loc=mu_pop_lt[idx], scale=sigma_pop_lt[idx], size=[N_1_base+N, N_2_base+N][idx]
    ) for idx in range(2)
]


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.25
Px_max = np.max(pop_dens_lt)
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_max = norm.pdf(
    x    =delta_pop, 
    loc  =delta_pop, 
    scale=np.sqrt(np.sum([sgm2 / n for sgm2, n in zip(sigma2_pop_lt, [N_1_base+N, N_2_base+N])]))
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
fig.suptitle('Population Mean Difference Confidence Interval (known variance)', fontsize=20)
axes2x = [ax.twiny() for ax in axes]
axes2y = [axes[0].twinx()]

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2x.cla() for ax2x in axes2x]
    [ax2y.cla() for ax2y in axes2y]

    ##### 乱数の集計 -----

    # 値を設定
    N_add = i + 1 # 追加サンプル数
    N_lt  = [N_1_base+N_add, N_2_base+N_add] # サンプル数

    # 標本統計量を計算
    x_bar_lt      = [np.mean(x_n[:n]) for x_n, n in zip(x_lt, N_lt)]
    sigma2_hat_lt = [np.var(x_n[:n], ddof=1) if n > 1 else np.nan for x_n, n in zip(x_lt, N_lt)]
    sigma_hat_lt  = [np.sqrt(sgm2) for sgm2 in sigma2_hat_lt]

    # 標本分布のパラメータを計算
    mu_smp     = np.subtract(*mu_pop_lt)
    sigma2_smp = np.sum([sgm2 / n for sgm2, n in zip(sigma2_pop_lt, N_lt)])
    sigma_smp  = np.sqrt(sigma2_smp)

    # 標本平均の差を計算
    d_obs = np.subtract(*x_bar_lt)

    # 標本を標準化
    z_obs = (d_obs - mu_smp) / sigma_smp

    # 信頼区間の範囲を計算
    ci_bound_lower = d_obs - cr_bound_upper * sigma_smp
    ci_bound_upper = d_obs + cr_bound_upper * sigma_smp

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_lt = [
        np.histogram(
            a=x_n[:n], bins=class_num, range=(bound_min, bound_max), density=True
        )[0] for x_n, n in zip(x_lt, N_lt)
    ]

    # 母分布のラベルを作成
    pop_param_lbl  = f'$N_1 = {N_lt[0]}, '
    pop_param_lbl += '\\mu_1 = '+f'{mu_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\sigma_1 = '+f'{sigma_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\bar{x}_1 = '+f'{x_bar_lt[0]:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_1 = '+f'{sigma_hat_lt[0]:.2f}$\n'
    pop_param_lbl += f'$N_2 = {N_lt[1]}, '
    pop_param_lbl += '\\mu_2 = '+f'{mu_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\sigma_2 = '+f'{sigma_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\bar{x}_2 = '+f'{x_bar_lt[1]:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_2 = '+f'{sigma_hat_lt[1]:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]
    ax2y = axes2y[0]

    for pop_idx in range(2):
        n = N_lt[pop_idx]
        ax.bar(
            x=center_vec, height=obs_dens_lt[pop_idx], 
            width=class_size, 
            color=cmap(pop_idx), alpha=0.5, 
            label='sample' if pop_idx == 0 else None, 
            zorder=8
        ) # 標本
        ax.scatter(
            x=x_lt[pop_idx][:n], y=np.zeros(n), 
            facecolor='none', edgecolor=cmap(pop_idx), s=50, #clip_on=False, 
            zorder=9
        ) # 標本
    for pop_idx in range(2):
        ax.plot(
            x_vec, pop_dens_lt[pop_idx], 
            color='black', linewidth=1.0, 
            label='population distribution' if pop_idx == 0 else None, 
            zorder=10
        ) # 母分布
        for idx, mu in enumerate([mu_pop_lt[pop_idx], x_bar_lt[pop_idx]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                label=['population means', 'sample means'][idx] if pop_idx == 0 else None, 
                zorder=[20, 21][idx]
            ) # 母・標本平均
    for delta in [ci_bound_lower, ci_bound_upper]:
        x = delta + mu_pop_lt[1] # 並行移動
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=mu_pop_lt[0], xmax=mu_pop_lt[1], 
        color='red', linewidth=2.0, 
        zorder=30
    ) # 母平均の差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =mu_pop_lt+x_bar_lt+[ci_bound_lower+mu_pop_lt[1], ci_bound_upper+mu_pop_lt[1]], 
        labels=['$\\mu_1$', '$\\mu_2$', '$\\bar{x}_1$', '$\\bar{x}_2$', '$L + \\mu_2$', '$U + \\mu_2$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax2y.set_ylabel('$\\frac{N_x}{N}$')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax.set_title(pop_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # (目盛の共通化用), 余白を追加
    dens_vals = ax.get_yticks()                     # 密度を取得
    freq_vals = dens_vals * class_size*np.max(N_lt) # 度数に変換
    Nx_max    = Px_max    * class_size*np.max(N_lt) # 度数軸の範囲を設定
    ax2y.set_yticks(ticks=freq_vals, labels=[f'{y:.1f}' for y in freq_vals]) # 度数軸目盛
    ax2y.set_ylim(ymin=-margin_ratio*Nx_max, ymax=(1.0+margin_ratio)*Nx_max) # (目盛の共通化用)

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}, '
    smp_param_lbl += 'd_{obs} = '+f'{d_obs:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_1 - \\mu_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} = \\sqrt{\\frac{\\sigma_1}{N_1} + \\frac{\\sigma_2}{N_2}} = '+f'{sigma_smp:.2f}$\n'
    smp_param_lbl += '$L = d_{obs} - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    smp_param_lbl += 'U = d_{obs} + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=d_obs, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        zorder=9
    ) # 標本平均の差
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, 0.0, d_obs]):
        ax.axvline(
            x=delta, 
            color=['red', 'red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean difference', None, 'saple mean difference'][idx], 
            zorder=[20, 20, 21][idx]
        ) # 母・標本平均の差
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

    ax.set_xlabel('$\\delta = \\mu_1 - \\mu_2, d = \\bar{x}_1 - \\bar{x}_2$')
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
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (d_min - d_obs) / sigma_smp
    z_max = (d_max - d_obs) / sigma_smp

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
        label='standard distribution', 
        zorder=10
    ) # 標準化分布
    for idx, d in enumerate([delta_pop, d_obs]):
        z = (d - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均の差
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

    ax.set_xlabel('$z = \\frac{\\bar{x} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[z_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$z_{obs}$', '$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper right', prop={'size': 8}
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
N_lt = [16, 20]


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.25
Px_max = np.max(pop_dens_lt)
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.05
Pd_max = norm.pdf(
    x    =delta_pop, 
    loc  =delta_pop, 
    scale=np.sqrt(np.sum([sgm2 / n for sgm2, n in zip(sigma2_pop_lt, N_lt)]))
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
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Mean Difference Confidence Interval (known variance)', fontsize=20)

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
        np.random.normal(
            loc=mu_pop_lt[idx], scale=sigma_pop_lt[idx], size=N_lt[idx]
        ) for idx in range(2)
    ]

    # 標本統計量を計算
    x_bar_lt      = [np.mean(x_n) for x_n in x_lt]
    sigma2_hat_lt = [np.var(x_n, ddof=1) for x_n in x_lt]
    sigma_hat_lt  = [np.sqrt(sgm2) for sgm2 in sigma2_hat_lt]

    # 標本分布のパラメータを計算
    mu_smp     = np.subtract(*mu_pop_lt)
    sigma2_smp = np.sum([sgm2 / n for sgm2, n in zip(sigma2_pop_lt, N_lt)])
    sigma_smp  = np.sqrt(sigma2_smp)

    # 標本平均の差を計算
    d_obs = np.subtract(*x_bar_lt)

    # 標本を標準化
    z_obs = (d_obs - mu_smp) / sigma_smp

    # 信頼区間の範囲を計算
    ci_bound_lower = d_obs - cr_bound_upper * sigma_smp
    ci_bound_upper = d_obs + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= delta_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_lt = [
        np.histogram(
            a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
        )[0] for x_n in x_lt
    ]

    # 母分布のラベルを作成
    pop_param_lbl  = f'$i = {I}$\n'
    pop_param_lbl += f'$N_1 = {N_lt[0]}, '
    pop_param_lbl += '\\mu_1 = '+f'{mu_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\sigma_1 = '+f'{sigma_pop_lt[0]:.2f}, '
    pop_param_lbl += '\\bar{x}_1 = '+f'{x_bar_lt[0]:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_1 = '+f'{sigma_hat_lt[0]:.2f}$\n'
    pop_param_lbl += f'$N_2 = {N_lt[1]}, '
    pop_param_lbl += '\\mu_2 = '+f'{mu_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\sigma_2 = '+f'{sigma_pop_lt[1]:.2f}, '
    pop_param_lbl += '\\bar{x}_2 = '+f'{x_bar_lt[1]:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_2 = '+f'{sigma_hat_lt[1]:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    for pop_idx in range(2):
        n = N_lt[pop_idx]
        ax.bar(
            x=center_vec, height=obs_dens_lt[pop_idx], 
            width=class_size, 
            color=cmap(pop_idx), alpha=0.5, 
            label='sample' if pop_idx == 0 else None, 
            zorder=8
        ) # 標本
        ax.scatter(
            x=x_lt[pop_idx][:n], y=np.zeros(n), 
            facecolor='none', edgecolor=cmap(pop_idx), s=50, #clip_on=False, 
            zorder=9
        ) # 標本
    for pop_idx in range(2):
        ax.plot(
            x_vec, pop_dens_lt[pop_idx], 
            color='black', linewidth=1.0, 
            label='population distribution' if pop_idx == 0 else None, 
            zorder=10
        ) # 母分布
        for idx, mu in enumerate([mu_pop_lt[pop_idx], x_bar_lt[pop_idx]]):
            ax.axvline(
                x=mu, 
                color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
                label=['population means', 'sample means'][idx] if pop_idx == 0 else None, 
                zorder=[20, 21][idx]
            ) # 母・標本平均
    for delta in [ci_bound_lower, ci_bound_upper]:
        x = delta + mu_pop_lt[1] # 並行移動
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=mu_pop_lt[0], xmax=mu_pop_lt[1], 
        color='red', linewidth=2.0, 
        zorder=30
    ) # 母平均の差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =mu_pop_lt+x_bar_lt+[ci_bound_lower+mu_pop_lt[1], ci_bound_upper+mu_pop_lt[1]], 
        labels=['$\\mu_1$', '$\\mu_2$', '$\\bar{x}_1$', '$\\bar{x}_2$', '$L_i + \\mu_2$', '$U_i + \\mu_2$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax.set_title(pop_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=d_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\delta_{pop} = '+f'{delta_pop:.2f}, '
    smp_param_lbl += 'd_i = '+f'{d_obs:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_1 - \\mu_2 = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} = \\sqrt{\\frac{\\sigma_1}{N_1} + \\frac{\\sigma_2}{N_2}} = '+f'{sigma_smp:.2f}$\n'
    smp_param_lbl += '$L_i = d_i - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    smp_param_lbl += 'U_i = d_i + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=d_obs, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        zorder=9
    ) # 標本平均の差
    ax.plot(
        d_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, delta in enumerate([delta_pop, 0.0, d_obs]):
        ax.axvline(
            x=delta, 
            color=['red', 'red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean difference', None, 'saple mean difference'][idx], 
            zorder=[20, 20, 21][idx]
        ) # 母・標本平均の差
    for idx, delta in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=delta, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple' if cover_flg else 'blue', linewidth=2.0, #clip_on=False, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\delta = \\mu_1 - \\mu_2, d = \\bar{x}_1 - \\bar{x}_2$')
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
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pd_max, ymax=(1.0+margin_ratio)*Pd_max) # 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (d_min - d_obs) / sigma_smp
    z_max = (d_max - d_obs) / sigma_smp

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
        label='standard distribution', 
        zorder=10
    ) # 標準化分布
    for idx, delta in enumerate([delta_pop, d_obs]):
        z = (delta - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均の差
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
    #ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [2, 3, 0, 1] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper right', prop={'size': 8}
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
        ticks =[delta_pop, d_obs, ci_bound_lower, ci_bound_upper], 
        labels=['$\\delta_{pop}$', '$d_i$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(ci_res_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=d_min, xmax=d_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=d_min, xmax=d_max) # (目盛の共通化用)
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


