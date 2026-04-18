
# 区間推定 ----------------------------------------------------------------------

# ch3.4.4

# 二項分布
# 母比率の信頼区間


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import binom, norm
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
dir_path += 'estimate_proportion/' # フォルダを指定
print(dir_path)


# %%

# 共通の設定 ------------------------------------------------------------------

### 母集団の設定 -----

# 母分布のパラメータを指定
p_pop = 0.6


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

# 最大サンプルサイズ(フレーム数)を指定
max_N = 100


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.05
Px_max = p_pop # 最頻値の確率密度
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=p_pop, loc=p_pop, scale=np.sqrt(p_pop*(1-p_pop)/max_N)) # 最頻値の確率密度
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print('p(x bar) size:', Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print('p(z) size:    ', Pz_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 凡例の表示位置を設定
legend_loc = 'upper left' if p_pop >= 0.5 else 'upper right'

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Proportion Confidence Interval', fontsize=20)
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
    n = i + 1 # サンプルサイズ

    # シードを設定(変化の安定化用):(ノートとの対応用)
    np.random.seed(10)
    
    # 標本を生成
    x_smp, = np.random.binomial(n=n, p=p_pop, size=1)
    #x_smp = np.floor((n+1) * p_pop) # 最頻値で代用(変化の安定化用)

    # 標本統計量を計算
    p_hat = x_smp / n

    # 標本分布のパラメータを計算
    mu_smp     = p_pop
    sigma2_smp = p_pop * (1.0-p_pop) / n
    sigma_smp  = np.sqrt(sigma2_smp)

    # 信頼区間の範囲を計算
    ci_bound_lower = p_hat - cr_bound_upper * sigma_smp
    ci_bound_upper = p_hat + cr_bound_upper * sigma_smp

    ##### 母分布の作図 -----

    # x軸の範囲を設定
    x_min = -1
    x_max = n + 1

    # x軸の値を作成
    x_vec = np.arange(start=x_min, stop=x_max+1, step=1)

    # 母分布の確率を計算
    pop_prob_vec = binom.pmf(k=x_vec, n=n, p=p_pop)

    # 母分布のラベルを作成
    pop_param_lbl = f'$N = {n}, p_{{pop}} = {p_pop:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    ax.scatter(
        x=x_smp, y=0.0, 
        color='black', alpha=0.5, s=100, #clip_on=False, 
        label='sample value', 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_prob_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    ax.scatter(
        x=x_vec, y=pop_prob_vec, 
        color='black', s=30, 
        zorder=10
    ) # 母分布
    for idx, x in enumerate([n*p_pop, x_smp]):
        ax.axvline(
            x=x, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率
    for p in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=n*p, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[n*p_pop, x_smp+1e-10, n*ci_bound_lower, n*ci_bound_upper], 
        labels=['$N p_{pop}$', '$x_{smp}$', '$N L$', '$N U$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$Bin(x \\mid n, p_{pop})$')
    ax.set_title(pop_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 表示範囲を固定, 余白を追加
    
    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc=legend_loc, prop={'size': 8}
    )

    ##### 標本分布の作図 -----

    # p軸の範囲を設定
    p_min = x_min / n
    p_max = x_max / n

    # p軸の値を作成
    p_vec = np.linspace(start=p_min, stop=p_max, num=1001)
    
    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=p_vec, loc=mu_smp, scale=sigma_smp)

    # 信頼区間のラベルを作成
    ci_param_lbl   = '$\\hat{p} = '+f'{p_hat:.2f}$\n'
    ci_param_lbl  += '$\\mu_{smp} = p_{pop} = '+f'{mu_smp:.2f}, '
    ci_param_lbl  += '\\sigma_{smp} = \\sqrt{\\frac{p_{pop} (1-p_{pop})}{N}} = '+f'{sigma_smp:.2f}$\n'
    ci_param_lbl  += '$L = \\hat{p} - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    ci_param_lbl  += 'U = \\hat{p} + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=p_hat, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        label='sample', 
        zorder=9
    ) # 標本比率
    ax.plot(
        p_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, p in enumerate([p_pop, p_hat]):
        ax.axvline(
            x=p, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population proportion', 'saple proportion'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本比率
    for idx, p in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=p, 
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

    ax.set_xlabel('$\\hat{p} = \\frac{x}{n}$')
    ax2x.set_xticks(
        ticks =[p_pop, p_hat+1e-10, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$p_{pop}$', '$\\hat{p}$', 
            '$\\hat{p} - z_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\hat{p} + z_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\hat{p} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(ci_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=p_min, xmax=p_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=p_min, xmax=p_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 3, 4, 5, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc=legend_loc, prop={'size': 8}
    )

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (p_min - p_hat) / sigma_smp
    z_max = (p_max - p_hat) / sigma_smp

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_z_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = norm.pdf(x=cr_z_vec, loc=0.0, scale=1.0)

    # 標本分布のラベルを作成
    std_param_lbl  = f'$\\alpha = {alpha:.2f}, '
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
    for idx, p in enumerate([p_pop, p_hat]):
        z = (p - p_hat) / sigma_smp # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率
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

    ax.set_xlabel('$z = \\frac{\\hat{p} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[cr_bound_lower, cr_bound_upper], 
        labels=['$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
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
        loc=legend_loc, prop={'size': 8}
    )

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=max_N, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'proportion_ci_n.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%

# 試行回数の影響 ----------------------------------------------------------------

### シミュレーションの設定 -----

# 試行回数(フレーム数)を指定
iter_num = 100

# サンプルサイズを指定
N = 20


# %%

### 変数の設定 -----

# x軸の範囲を設定
x_lower = 0
x_upper = N
x_min   = x_lower - 0.5
x_max   = x_upper + 0.5
print('x-axis size:', x_min, x_max)

# x軸の値を作成
x_vec = np.arange(start=x_lower, stop=x_upper+1, step=1)


# %%

### 分布の計算 -----

# 母分布の確率を計算
pop_prob_vec = binom.pmf(k=x_vec, n=N, p=p_pop)


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.05
Px_max = pop_prob_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:    ', Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=p_pop, loc=p_pop, scale=np.sqrt(p_pop*(1-p_pop)/N)) # 最頻値の確率密度
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print('p(x bar) size:', Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print('p(z) size:    ', Pz_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 凡例の表示位置を設定
legend_loc = 'upper left' if p_pop >= 0.5 else 'upper right'

# 図を初期化
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Proportion Confidence Interval', fontsize=20)

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
    x_smp, = np.random.binomial(n=N, p=p_pop, size=1)

    # 標本統計量を計算
    p_hat = x_smp / N

    # 標本分布のパラメータを計算
    mu_smp     = p_pop
    sigma2_smp = p_pop * (1.0-p_pop) / N
    sigma_smp  = np.sqrt(sigma2_smp)

    # 信頼区間の範囲を計算
    ci_bound_lower = p_hat - cr_bound_upper * sigma_smp
    ci_bound_upper = p_hat + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= p_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図 -----

    # 母分布のラベルを作成
    pop_param_lbl  = f'$i = {I}$\n'
    pop_param_lbl += f'$N = {N}, p_{{pop}} = {p_pop:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    ax.scatter(
        x=x_smp, y=0.0, 
        color='black', alpha=0.5, s=100, #clip_on=False, 
        label='sample value', 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_prob_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    ax.scatter(
        x=x_vec, y=pop_prob_vec, 
        color='black', s=30, 
        zorder=10
    ) # 母分布
    for idx, x in enumerate([N*p_pop, x_smp]):
        ax.axvline(
            x=x, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率
    for p in [ci_bound_lower, ci_bound_upper]:
        ax.axvline(
            x=N*p, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[N*p_pop, x_smp+1e-10, N*ci_bound_lower, N*ci_bound_upper], 
        labels=['$N p_{pop}$', '$x_i$', '$N L_i$', '$N U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$Bin(x \\mid n, p_{pop})$')
    ax.set_title(pop_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 表示範囲を固定, 余白を追加
    
    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc=legend_loc, prop={'size': 8}
    )

    ##### 標本分布の作図 -----

    # p軸の範囲を設定
    p_min = x_min / N
    p_max = x_max / N

    # p軸の値を作成
    p_vec = np.linspace(start=p_min, stop=p_max, num=1001)
    
    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=p_vec, loc=mu_smp, scale=sigma_smp)

    # 信頼区間のラベルを作成
    ci_param_lbl   = '$\\hat{p}_i = '+f'{p_hat:.2f}$\n'
    ci_param_lbl  += '$\\mu_{smp} = p_{pop} = '+f'{mu_smp:.2f}, '
    ci_param_lbl  += '\\sigma_{smp} = \\sqrt{\\frac{p_{pop} (1-p_{pop})}{N}} = '+f'{sigma_smp:.2f}$\n'
    ci_param_lbl  += '$L_i = \\hat{p}_i - z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_lower:.2f}, '
    ci_param_lbl  += 'U_i = \\hat{p}_i + z_{\\frac{\\alpha}{2}} \\sigma_{smp} = '+f'{ci_bound_upper:.2f}$'

    # 標本分布を描画
    ax   = axes[1]
    ax2x = axes2x[1]

    ax.scatter(
        x=p_hat, y=0.0, 
        color='#00A968', s=100, #clip_on=False, 
        label='sample', 
        zorder=9
    ) # 標本比率
    ax.plot(
        p_vec, smp_dens_vec, 
        color='black', linewidth=1.0, 
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, p in enumerate([p_pop, p_hat]):
        ax.axvline(
            x=p, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population proportion', 'saple proportion'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本比率
    for idx, p in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=p, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    ax.hlines(
        y=0.0, xmin=ci_bound_lower, xmax=ci_bound_upper, 
        color='purple' if cover_flg else 'blue', linewidth=2.0, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\hat{p} = \\frac{x}{n}$')
    ax2x.set_xticks(
        ticks =[p_pop, p_hat+1e-10, ci_bound_lower, ci_bound_upper], 
        labels=[
            '$p_{pop}$', '$\\hat{p}_i$', 
            '$\\hat{p}_i - z_{\\frac{\\alpha}{2}} \\sigma_{smp}$', 
            '$\\hat{p}_i + z_{\\frac{\\alpha}{2}} \\sigma_{smp}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\hat{p} \\mid \\mu_{smp}, \\sigma_{smp}^2)$')
    ax.set_title(ci_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=p_min, xmax=p_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=p_min, xmax=p_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 表示範囲を固定, 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'right', 'left'] # 表示位置を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 3, 4, 5, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc=legend_loc, prop={'size': 8}
    )

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (p_min - p_hat) / sigma_smp
    z_max = (p_max - p_hat) / sigma_smp

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

    # 標本分布のラベルを作成
    std_param_lbl  = f'$\\alpha = {alpha:.2f}, '
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
    for idx, p in enumerate([p_pop, p_hat]):
        z = (p - p_hat) / sigma_smp # 標準化
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本比率
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

    ax.set_xlabel('$z = \\frac{\\hat{p} - \\mu_{smp}}{\\sigma_{smp}}$')
    ax2x.set_xticks(
        ticks =[cr_bound_lower, cr_bound_upper], 
        labels=['$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [2, 0, 1] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc=legend_loc, prop={'size': 8}
    )

    ##### 信頼区間の作図 -----

    # 新規の推定結果を記録
    res_lt.append(
        ([ci_bound_lower, ci_bound_upper], cover_flg)
    )

    # 推定結果のラベルを作成
    non_cover_cnt = I - cover_cnt
    res_param_lbl = f'non-coverage: {non_cover_cnt} / {I} ( {non_cover_cnt/I:.3f} )'
    
    # 信頼区間を描画
    ax   = axes[3]
    ax2x = axes2x[3]

    for idx, p in enumerate([p_pop, p_hat]):
        ax.axvline(
            x=p, 
            color=['red', 'black'][idx], linewidth=[2.0, 1.0][idx], linestyle=['-', '--'][idx], 
            label=['population mean', 'sample mean'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本比率
    for idx, p in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=p, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    for i in range(I):
        # 過去の推定結果を取得
        [p_lower, p_upper], cover_flg = res_lt[i]
        ax.hlines(
            y=i+1, xmin=p_lower, xmax=p_upper, 
            color='purple' if cover_flg else 'blue', linewidth=2.0, 
            label='confidence interval' if i+1 == I else None, 
            zorder=30
        ) # 信頼区間
    
    ax.set_xlabel('$p$')
    ax2x.set_xticks(
        ticks =[p_pop, p_hat+1e-10, ci_bound_lower, ci_bound_upper], 
        labels=['$p_{pop}$', '$\\hat{p}_i$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(res_param_lbl, loc='left')
    ax.legend(loc=legend_loc, prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=p_min, xmax=p_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=p_min, xmax=p_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1) # 表示範囲を固定
    ax.invert_yaxis() # 推定結果を昇順に表示

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=iter_num, interval=500
)

# 動画を書出
anim.save(
    filename=dir_path+'proportion_ci_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


