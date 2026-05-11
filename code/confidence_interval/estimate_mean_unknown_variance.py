
# 区間推定 ----------------------------------------------------------------------

# ch3.4.2

# 正規分布
# 母平均の信頼区間
# 母分散が未知の場合


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm, t
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
dir_path += 'estimate_mean_unknown_variance/' # フォルダを指定
print(dir_path)


# %%

# 共通の設定 ------------------------------------------------------------------

### 母集団の設定 -----

# 母分布のパラメータを指定
mu_pop     = 4.0
sigma_pop  = 2.0
sigma2_pop = sigma_pop**2


# %%

### 変数の設定 -----

# x軸の範囲を設定
k = 3.0
u = 1.0
x_size  = sigma_pop # 標準偏差
x_size *= k # 定数倍
#x_size  = max(x_size, np.abs(x_n-mu_pop).max()) # 標本と比較
x_size  = np.ceil(x_size /u)*u # u単位で切り上げ
x_min   = mu_pop - x_size
x_max   = mu_pop + x_size
print('x-axis size:', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


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
pop_dens_vec = norm.pdf(x=x_vec, loc=mu_pop, scale=sigma_pop)


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
np.random.seed(10)

# サンプルサイズ(フレーム数)を指定
N = 100

# 標本を生成
x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)


# %%

### 表示範囲の設定 ----

# 母分布のp軸の範囲を設定
u = 0.25
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最終試行の最頻値
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print('p(x bar) size:', Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pt_max = t.pdf(x=0.0, df=N-1, loc=0.0, scale=1.0) # 最終試行の最頻値
Pt_max = np.ceil(Pt_max /u)*u # u単位で切り上げ
print('p(t) size:    ', Pt_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Mean Confidence Interval (unknown variance)', fontsize=20)
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
    n = i + 1 # サンプル数

    # 標本統計量を計算
    x_bar      = np.mean(x_n[:n])
    sigma2_hat = np.var(x_n[:n], ddof=1) if n > 1 else np.nan
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_pop
    sigma_smp = sigma_hat / np.sqrt(n)

    # 標本統計量を標準化
    t_obs = (x_bar - mu_smp) / sigma_smp

    # 中央領域の範囲を計算
    cr_bound_upper = t.ppf(q=1.0-0.5*alpha, df=n-1, loc=0.0, scale=1.0)
    cr_bound_lower = -cr_bound_upper

    # 信頼区間の範囲を計算
    ci_bound_lower = x_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = x_bar + cr_bound_upper * sigma_smp

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n[:n], bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 母分布のラベルを作成
    pop_param_lbl  = f'$N = {n}$\n'
    pop_param_lbl += '$\\mu_{pop} = '+f'{mu_pop:.2f}, '
    pop_param_lbl += '\\sigma_{pop} = '+f'{sigma_pop:.2f}$\n'
    pop_param_lbl += '$L = \\bar{x}_{obs} - t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    pop_param_lbl += 'U = \\bar{x}_{obs} + t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]
    ax2y = axes2y[0]

    ax.bar(
        x=center_vec, height=obs_dens_vec, 
        width=class_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # 標本
    ax.scatter(
        x=x_n[:n], y=np.zeros(n), 
        color='black', alpha=0.33, s=25, #clip_on=False, 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    for idx, mu in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean', None][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for idx, mu in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=mu, 
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
    
    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[mu_pop, ci_bound_lower, ci_bound_upper], 
        labels=['$\\mu_{pop}$', '$L$', '$U$']
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
    dens_vals = ax.get_yticks()          # 密度を取得
    freq_vals = dens_vals * class_size*n # 度数に変換
    Nx_max    = Px_max    * class_size*n # 度数軸の範囲を設定
    ax2y.set_yticks(ticks=freq_vals, labels=[f'{y:.1f}' for y in freq_vals]) # 度数軸目盛
    ax2y.set_ylim(ymin=-margin_ratio*Nx_max, ymax=(1.0+margin_ratio)*Nx_max) # (目盛の共通化用)

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\bar{x}_{obs} = '+f'{x_bar:.2f}, '
    smp_param_lbl += '\\hat{\\sigma} = '+f'{sigma_hat:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_{pop} = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\frac{\\hat{\\sigma}}{\\sqrt{N}} = '+f'{sigma_smp:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=x_bar, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        zorder=9
    ) # 標本平均
    ax.plot(
        x_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, mu in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=[None, 'sample mean'][idx], 
        zorder=[20, 21][idx]
        ) # 母・標本平均
    for mu in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    
    ax.set_xlabel('$\\bar{x} = \\frac{1}{n} \\sum_{i=1}^n x_i$')
    ax2x.set_xticks(
        ticks =[x_bar, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\bar{x}_{obs}$', 
            '$\\bar{x}_{obs} - t_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\bar{x}_{obs} + t_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 表示範囲を固定, 余白を追加

    # 目盛ラベルの表示を調整(ラベルが重なる対策用)
    halignments = ['center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # t軸の範囲を設定
    if n > 1:
        t_min = (x_min - x_bar) / sigma_smp
        t_max = (x_max - x_bar) / sigma_smp
    else: # 自由度の条件を満たさない場合 
        # 標本統計量を計算
        tmp_n = 2
        tmp_x_bar     = np.mean(x_n[:tmp_n])
        tmp_sigma_hat = np.sqrt(np.var(x_n[:tmp_n], ddof=1))
        tmp_sigma_smp = tmp_sigma_hat / np.sqrt(tmp_n)
        t_min = (x_min - tmp_x_bar) / tmp_sigma_smp
        t_max = (x_max - tmp_x_bar) / tmp_sigma_smp

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
    ax   = axes[2]
    ax2x = axes2x[2]

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
    for idx, mu in enumerate([mu_pop, x_bar]):
        t_val = (mu - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=t_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
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

    ax.set_xlabel('$t = \\frac{\\bar{x} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[t_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$t_{obs}$', '$t_{1-\\frac{\\alpha}{2}}$', '$t_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$t(t \\mid n-1, 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=t_min, xmax=t_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=t_min, xmax=t_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pt_max, ymax=(1.0+margin_ratio)*Pt_max) # 余白を追加

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
    filename=dir_path+'mean_ci_n.mp4', 
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

### 表示範囲の設定 -----

# μ軸の範囲を設定
k = 3.0
u = 0.5
mu_size  = cr_bound_upper * sigma_pop / np.sqrt(N) # 信頼区間の半サイズ
mu_size *= k # 定数倍
mu_size  = np.ceil(mu_size /u)*u # u単位で切り上げ
mu_min   = mu_pop - mu_size
mu_max   = mu_pop + mu_size
print('μ-axis size:', mu_min, mu_max)


# 母分布のp軸の範囲を設定
u = 0.15
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最頻値
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print('p(x bar) size:', Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pt_max = t.pdf(x=0.0, df=N-1, loc=0.0, scale=1.0) # 最頻値
Pt_max = np.ceil(Pt_max /u)*u # u単位で切り上げ
print('p(t) size:    ', Pt_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Mean Confidence Interval (unknown variance)', fontsize=20)

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
    x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)

    # 標本統計量を計算
    x_bar      = np.mean(x_n)
    sigma2_hat = np.var(x_n, ddof=1)
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_pop
    sigma_smp = sigma_hat / np.sqrt(N)

    # 標本統計量を標準化
    t_obs = (x_bar - mu_smp) / sigma_smp

    # 信頼区間の範囲を計算
    ci_bound_lower = x_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = x_bar + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= mu_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 母分布のラベルを作成
    pop_param_lbl  = f'$i = {I}, N = {N}$\n'
    pop_param_lbl += '$\\mu_{pop} = '+f'{mu_pop:.2f}, '
    pop_param_lbl += '\\sigma_{pop} = '+f'{sigma_pop:.2f}$\n'
    pop_param_lbl += '$L_i = \\bar{x}_i - t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    pop_param_lbl += 'U_i = \\bar{x}_i + t_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    ax.bar(
        x=center_vec, height=obs_dens_vec, 
        width=class_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # 標本
    ax.scatter(
        x=x_n, y=np.zeros(N), 
        color='black', alpha=0.33, s=25, #clip_on=False, 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    for idx, mu in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population mean', None][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for idx, mu in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=mu, 
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

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[mu_pop, ci_bound_lower, ci_bound_upper], 
        labels=['$\\mu_{pop}$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax.set_title(pop_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 余白を追加
    
    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = '$\\bar{x}_i = '+f'{x_bar:.2f}, '
    smp_param_lbl += '\\hat{\\sigma}_i = '+f'{sigma_hat:.2f}$\n'
    smp_param_lbl += '$\\mu_{smp} = \\mu_{pop} = '+f'{mu_smp:.2f}, '
    smp_param_lbl += '\\sigma_{smp} \\approx \\frac{\\hat{\\sigma}_i}{\\sqrt{N}} = '+f'{sigma_smp:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=x_bar, y=0.0, 
        color='#00A968', s=100, clip_on=False, 
        zorder=9
    ) # 標本平均
    ax.plot(
        x_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sampling distribution', 
        zorder=10
    ) # 標本分布
    for idx, mu in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=[None, 'sample mean'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for mu in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    
    ax.set_xlabel('$\\bar{x} = \\frac{1}{n} \\sum_{j=1}^n x_j$')
    ax2x.set_xticks(
        ticks =[x_bar, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$\\bar{x}_i$', 
            '$\\bar{x}_i - t_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\bar{x}_i + t_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 表示範囲を固定, 余白を追加

    # 目盛ラベルの表示を調整(ラベルが重なる対策用)
    halignments = ['center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # t軸の範囲を設定
    t_min = (x_min - x_bar) / sigma_smp
    t_max = (x_max - x_bar) / sigma_smp

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
    std_param_lbl  = '$t_i = '+f'{t_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += 't_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += 't_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2]
    ax2x = axes2x[2]

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
    for idx, mu in enumerate([mu_pop, x_bar]):
        t_val = (mu - mu_smp) / sigma_smp # 標準化
        ax.axvline(
            x=t_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
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
    
    ax.set_xlabel('$z = \\frac{\\bar{x} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[t_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$t_i$', '$t_{1-\\frac{\\alpha}{2}}$', '$t_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$t(t \\mid n-1, 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=t_min, xmax=t_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=t_min, xmax=t_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pt_max, ymax=(1.0+margin_ratio)*Pt_max) # 余白を追加

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
    ax  = axes[3]
    ax2x = axes2x[3]

    for idx, mu in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=mu, 
            color=['red', 'black'][idx], linewidth=[2.0, 1.0][idx], linestyle=['-', '--'][idx], 
            label=['population mean', 'sample mean'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本平均
    for idx, mu in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    for i in range(I):
        # 過去の推定結果を取得
        [mu_lower, mu_upper], cover_flg = res_lt[i]
        ax.hlines(
            y=i+1, xmin=mu_lower, xmax=mu_upper, 
            color='purple' if cover_flg else 'blue', linewidth=2.0, 
            label='confidence interval' if i+1 == I else None, 
            zorder=30
        ) # 信頼区間
    
    ax.set_xlabel('$\\mu$')
    ax2x.set_xticks(
        ticks =[mu_pop, x_bar, ci_bound_lower, ci_bound_upper], 
        labels=['$\\mu_{pop}$', '$\\bar{x}_i$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(ci_res_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=mu_min, xmax=mu_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1) # 表示範囲を固定
    ax.invert_yaxis() # 推定結果を昇順に表示

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=iter_num, interval=500
)

# 動画を書出
anim.save(
    filename=dir_path+'mean_ci_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


