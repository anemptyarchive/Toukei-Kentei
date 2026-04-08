
# 区間推定-----------------------------------------------------------------------

# ch3.4.1
# 正規分布
# 母平均の信頼区間
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
dir_path += '/figure/confidence_interval/estimate_mean_known_variance/'
print(dir_path)


# %%

# 母集団の設定 ------------------------------------------------------------------

### パラメータの設定 -----

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

# サンプルサイズの影響 -----------------------------------------------------------

### 乱数の生成 -----

# シードを設定:(ノートとの対応用)
np.random.seed(10)

# サンプルサイズ(フレーム数)を指定
N = 100

# 標本を生成
x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)


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

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
Px_max = 0.5
print(Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最頻値の確率密度
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print(Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print(Pz_max)


# %%

### 信頼区間の計算の可視化 -----

# 余白を設定
margin_ratio = 0.05

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Mean Confidence Interval (known variance)', fontsize=20)
axes2x = [ax.twiny() for ax in axes]
ax2y   = axes[0].twinx()
# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2x.cla() for ax2x in axes2x]
    ax2y.cla()

    ##### 乱数の集計 -----

    # 値を設定
    n = i + 1 # サンプル数

    # 標本統計量を計算
    x_bar      = np.mean(x_n[:n])
    sigma2_hat = np.var(x_n[:n], ddof=1) if n > 1 else np.nan
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_pop
    sigma_smp = sigma_pop / np.sqrt(n)

    # 信頼区間の範囲を計算
    ci_bound_lower = x_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = x_bar + cr_bound_upper * sigma_smp

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n[:n], bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 信頼区間のラベルを作成
    ci_param_lbl   = f'$N = {n}, '
    ci_param_lbl  += f'\\mu = {mu_pop:.2f}, \\sigma = {sigma_pop:.2f}$\n'
    ci_param_lbl  += f'$L = {ci_bound_lower:.2f}, U = {ci_bound_upper:.2f}$'

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
        x=x_n[:n], y=np.tile(0.0, reps=n), 
        color='black', alpha=0.33, s=25, #clip_on=False, 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    for idx, x in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=x, 
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
        labels=['$\\mu$', '$L$', '$U$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(x \\mid \\mu, \\sigma^2)$')
    ax2y.set_ylabel('$\\frac{N_x}{N}$')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax.set_title(ci_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # (目盛の共通化用)
    dens_vals = ax.get_yticks()          # 密度を取得
    freq_vals = dens_vals * class_size*n # 度数に変換
    Nx_max    = Px_max    * class_size*n # 度数軸の範囲を設定
    ax2y.set_yticks(ticks=freq_vals, labels=[f'{y:.1f}' for y in freq_vals]) # 度数軸目盛
    ax2y.set_ylim(ymin=-margin_ratio*Nx_max, ymax=(1.0+margin_ratio)*Nx_max) # (目盛の共通化用)

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = f'$\\bar{{x}} = {x_bar:.2f}, \\hat{{\\sigma}} = {sigma_hat:.2f}$\n'
    smp_param_lbl += f'$\\mu = {mu_smp:.2f}, \\frac{{\\sigma}}{{\\sqrt{{N}}}} = {sigma_smp:.2f}$'

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
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, x in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=x, 
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
        ticks =[ci_bound_lower, x_bar, ci_bound_upper], 
        labels=[
            '$\\bar{x} - z_{\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{N}}$', 
            '$\\bar{x}$', 
            '$\\bar{x} + z_{\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{N}}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu, \\frac{\\sigma^2}{n})$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 余白を追加

    # 目盛ラベルの表示を調整(ラベルが重なる対策用)
    halignments = ['right', 'center', 'left']
    labels = ax2x.get_xticklabels() # ラベル情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)
    
    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (x_min - x_bar) / sigma_smp
    z_max = (x_max - x_bar) / sigma_smp

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準化分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 中央領域を計算
    cr_z_vec    = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = norm.pdf(x=cr_z_vec, loc=0.0, scale=1.0)

    # 標準化分布のラベルを作成
    std_param_lbl  = f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += f'z_{{1-\\frac{{\\alpha}}{{2}}}} = {cr_bound_lower:.2f}, '
    std_param_lbl += f'z_{{\\frac{{\\alpha}}{{2}}}} = {cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2]
    ax2x = axes2x[2]

    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standard distribution', 
        zorder=10
    ) # 標準化分布
    ax.fill_between(
        x=cr_z_vec, y1=np.zeros_like(cr_z_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=11
    ) # 中央領域
    for idx, z in enumerate([(mu_pop-x_bar)/sigma_smp, 0.0]):
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
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

    ax.set_xlabel('$z = (\\bar{x} - \\mu) \\frac{\\sqrt{n}}{\\sigma}$')
    ax2x.set_xticks(
        ticks =[cr_bound_lower, cr_bound_upper], 
        labels=['$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

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

### パラメータの設定 -----

# 試行回数(フレーム数)を指定
iter_num = 100

# サンプルサイズを指定
N = 100


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

### 表示範囲の設定 -----

# μの表示範囲を設定
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
print(Px_max)

# 標本分布のp軸の範囲を設定
u = 0.01
Px_bar_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最頻値の確率密度
Px_bar_max = np.ceil(Px_bar_max /u)*u # u単位で切り上げ
print(Px_bar_max)

# 標準化分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print(Pz_max)


# %%

### 信頼区間の計算の可視化 -----

# 余白を設定
margin_ratio = 0.05

# 図を初期化
fig = plt.figure(
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Mean Confidence Interval (known variance)', fontsize=20)

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
    I = I + 1 # 試行番号

    ##### 乱数の生成 -----

    # 標本を生成
    x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)

    # 標本統計量を計算
    x_bar      = np.mean(x_n)
    sigma2_hat = np.var(x_n, ddof=1)
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本分布のパラメータを計算
    mu_smp    = mu_pop
    sigma_smp = sigma_pop / np.sqrt(N)

    # 信頼区間の範囲を計算
    ci_bound_lower = x_bar + cr_bound_lower * sigma_smp
    ci_bound_upper = x_bar + cr_bound_upper * sigma_smp

    # 被覆を判定
    cover_flg  = ci_bound_lower <= mu_pop <= ci_bound_upper
    cover_cnt += cover_flg

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 信頼区間のラベルを作成
    ci_param_lbl   = f'$i = {I}, N = {N}$\n'
    ci_param_lbl  += f'$\\mu = {mu_pop:.2f}, \\sigma = {sigma_pop:.2f}$\n'
    ci_param_lbl  += f'$L_i = {ci_bound_lower:.2f}, U_i = {ci_bound_upper:.2f}$'

    # 母分布を描画
    ax   = axes[0]
    ax2x = axes2x[0]

    ax.bar(
        x=center_vec, height=obs_dens_vec, width=class_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # 標本
    ax.scatter(
        x=x_n, y=np.tile(0.0, reps=N), 
        color='black', alpha=0.33, s=25, #clip_on=False, 
        zorder=9
    ) # 標本
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    for idx, x in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=x, 
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
        labels=['$\\mu$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(x \\mid \\mu, \\sigma^2)$')
    ax.set_title(ci_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 余白を追加
    
    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_smp, scale=sigma_smp)

    # 標本分布のラベルを作成
    smp_param_lbl  = f'$\\bar{{x}}_i = {x_bar:.2f}, \\hat{{\\sigma}}_i = {sigma_hat:.2f}$\n'
    smp_param_lbl += f'$\\mu = {mu_smp:.2f}, \\frac{{\\sigma}}{{\\sqrt{{N}}}} = {sigma_smp:.2f}$'

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
        label='sample distribution', 
        zorder=10
    ) # 標本分布
    for idx, x in enumerate([mu_pop, x_bar]):
        ax.axvline(
            x=x, 
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
        ticks =[ci_bound_lower, x_bar, ci_bound_upper], 
        labels=[
            '$\\bar{x}_i - z_{\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{N}}$', 
            '$\\bar{x}_i$', 
            '$\\bar{x}_i + z_{\\frac{\\alpha}{2}} \\frac{\\sigma}{\\sqrt{N}}$'
        ]
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu, \\frac{\\sigma^2}{n})$')
    ax.set_title(smp_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_bar_max, ymax=(1.0+margin_ratio)*Px_bar_max) # 余白を追加

    # 目盛ラベルの表示を調整(ラベルが重なる対策用)
    halignments = ['right', 'center', 'left']
    labels = ax2x.get_xticklabels() # ラベル情報を取得
    for label, ha in zip(labels, halignments):
        label.set_ha(ha)

    ##### 標準化分布の作図 -----

    # z軸の範囲を設定
    z_min = (x_min - x_bar) / sigma_smp
    z_max = (x_max - x_bar) / sigma_smp

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
    std_param_lbl  = f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += f'z_{{1-\\frac{{\\alpha}}{{2}}}} = {cr_bound_lower:.2f}, '
    std_param_lbl += f'z_{{\\frac{{\\alpha}}{{2}}}} = {cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2]
    ax2x = axes2x[2]

    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standard distribution', 
        zorder=10
    ) # 標準化分布
    ax.fill_between(
        x=cr_z_vec, y1=np.zeros_like(cr_z_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=11
    ) # 中央領域
    ax.fill_between(
        x=tail_z_vec, y1=np.zeros_like(tail_z_vec), y2=tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        label='tail regions', 
        zorder=11
    ) # 外側領域
    for idx, z in enumerate([(mu_pop-x_bar)/sigma_smp, 0.0]):
        ax.axvline(
            x=z, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本平均
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
    
    ax.set_xlabel('$z = (\\bar{x} - \\mu) \\frac{\\sqrt{n}}{\\sigma}$')
    ax2x.set_xticks(
        ticks =[cr_bound_lower, cr_bound_upper], 
        labels=['$z_{1-\\frac{\\alpha}{2}}$', '$z_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pz_max, ymax=(1.0+margin_ratio)*Pz_max) # 余白を追加

    ##### 信頼区間の作図 -----

    # 新規の推定結果を記録
    res_lt.append(
        ([ci_bound_lower, ci_bound_upper], cover_flg)
    )

    # 推定結果のラベルを作成
    res_param_lbl = f'non-coverage count: {I-cover_cnt}'
    
    # 信頼区間を描画
    ax   = axes[3]
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
            y=i, xmin=mu_lower, xmax=mu_upper, 
            color='purple' if cover_flg else 'blue', linewidth=2.0, 
            label='confidence interval' if i+1 == I else None, 
            zorder=30
        ) # 信頼区間
    
    ax.set_xlabel('$\\mu$')
    ax2x.set_xticks(
        ticks =[mu_pop, ci_bound_lower, x_bar, ci_bound_upper], 
        labels=['$\\mu$', '$L_i$', '$\\bar{x}_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(res_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=mu_min, xmax=mu_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1)
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


