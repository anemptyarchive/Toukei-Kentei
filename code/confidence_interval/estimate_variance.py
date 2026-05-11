
# 区間推定 ----------------------------------------------------------------------

# ch3.4.3

# 正規分布
# 母分散の信頼区間


# %%

# ライブラリの読込 --------------------------------------------------------------

# ライブラリを読込
import numpy as np
from scipy.stats import norm, chi2
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
dir_path += 'estimate_variance/' # フォルダを指定
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
print('x-axis size: ', x_min, x_max)

# x軸の値を作成
x_vec = np.linspace(start=x_min, stop=x_max, num=1001)


# σ軸の範囲を設定
sigma_min = x_min - mu_pop
sigma_max = x_max - mu_pop
print('σ-axis size: ', sigma_min, sigma_max)

# σ軸の値を作成
sigma_vec = np.linspace(start=sigma_min, stop=sigma_max, num=1001)
sigma_vec = sigma_vec[sigma_vec > 0] # 正の値を抽出
print('σ line size: ', sigma_vec.min(), sigma_vec.max())


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

### 変数の設定 -----

# σ2軸の範囲を設定
k = 4.0
u = 5.0
sigma2_size   = sigma2_pop/(N-1) * np.sqrt(2.0*(N-1)) # 標準偏差
sigma2_size  *= k # 定数倍
sigma2_center = sigma2_pop # 真の値
sigma2_min    = 0.0 # (σ軸との対応用)
sigma2_max    = sigma2_center + sigma2_size
sigma2_max    = np.ceil(sigma2_max /u)*u  # u単位で切り上げ
print('σ2-axis size:', sigma2_min, sigma2_max)

# σ2軸の値を作成
sigma2_vec = np.linspace(start=sigma2_min, stop=sigma2_max, num=1001)[1:] # 正の値を抽出


# χ2軸の範囲を設定
u = 5.0
chi2_min = 0.0
chi2_max = N # 期待値+1
chi2_max = np.ceil(chi2_max /u)*u # u単位で切り上げ
print('χ2-axis size:', chi2_min, chi2_max)

# χ2軸の値を作成
chi2_vec = np.linspace(start=chi2_min, stop=chi2_max, num=1001)


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.25
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標準化分布のp軸の範囲を設定
k = 0.5
u = 0.05
Pchi_max  = chi2.pdf(x=0, df=2) # 初回試行の最頻値
Pchi_max *= k # 定数倍
Pchi_max  = np.ceil(Pchi_max /u)*u # u単位で切り上げ
print('p(χ) size:', Pchi_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=2, 
    figsize=(12, 12), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Variance Confidence Interval', fontsize=20)
axes2x = [ax.twiny() for ax in [axes[0, 0], axes[1, 0], axes[2, 1]]]
axes2y = [ax.twinx() for ax in [axes[0, 0], axes[1, 0]]]

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
    x_bar      = np.mean(x_n[:n])
    sigma2_hat = np.var(x_n[:n], ddof=1) if n > 1 else np.nan
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本統計量をスケール変換
    chi2_obs = (n-1) * sigma2_hat / sigma2_pop

    # 中央領域の範囲を計算
    cr_bound_lower = chi2.ppf(q=0.5*alpha, df=n-1)
    cr_bound_upper = chi2.ppf(q=1.0-0.5*alpha, df=n-1)

    # 信頼区間の範囲を計算
    ci_bound_lower = (n-1) * sigma2_hat / cr_bound_upper
    ci_bound_upper = (n-1) * sigma2_hat / cr_bound_lower

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n[:n], bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 母集団のラベルを作成
    pop_param_lbl  = f'$N = {n}$\n'
    pop_param_lbl += '$\\mu_{pop} = '+f'{mu_pop:.2f}, '
    pop_param_lbl += '\\sigma_{pop} = '+f'{sigma_pop:.2f}$\n'
    pop_param_lbl += '$\\bar{x} = '+f'{x_bar:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_{obs} = '+f'{sigma_hat:.2f}$'

    # 母分布を描画
    ax   = axes[0, 0]
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
    for idx, x in enumerate([mu_pop, mu_pop+sigma_pop, x_bar, x_bar+sigma_hat]):
        ax.axvline(
            x=x, 
            color=['red', 'red', 'black', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 20, 21, 21][idx]
        ) # 母数・標本統計量
    for x in [mu_pop+np.sqrt(ci_bound_lower), mu_pop+np.sqrt(ci_bound_upper)]:
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    sgm_dens = norm.pdf(x=mu_pop+sigma_pop, loc=mu_pop, scale=sigma_pop)
    ax.hlines(
        y=sgm_dens, xmin=mu_pop, xmax=mu_pop+sigma_pop, 
        color='red', linewidth=1.0, 
        label='population sd', 
        zorder=30
    ) # 母標準偏差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[mu_pop, mu_pop+sigma_pop, x_bar, x_bar+sigma_hat], 
        labels=['$\\mu_{pop}$', '$\\mu_{pop} + \\sigma_{pop}$', '$\\bar{x}$', '$\\bar{x} + \\hat{\\sigma}_{obs}$']
    ) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax2y.set_ylabel('$\\frac{N_x}{N}$')
    ax2y.yaxis.set_label_position(position='right') # (ラベルの表示位置が初期化される対策)
    ax.set_title(pop_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # (目盛の共通化用), 余白を追加
    dens_vals = ax.get_yticks()          # 密度を取得
    freq_vals = dens_vals * class_size*n # 度数に変換
    Nx_max    = Px_max    * class_size*n # 度数軸の範囲を設定
    ax2y.set_yticks(ticks=freq_vals, labels=[f'{y:.1f}' for y in freq_vals]) # 度数軸目盛
    ax2y.set_ylim(ymin=-margin_ratio*Nx_max, ymax=(1.0+margin_ratio)*Nx_max) # (目盛の共通化用)

    # ラベルの装飾を調整(重なる対策用)
    halignments = [
        'left' if mu_pop >= x_bar else 'right', 
        'left' if mu_pop+sigma_pop >= x_bar+sigma_hat else 'center', 
        'center', 
        'center' if mu_pop+sigma_pop >= x_bar+sigma_hat else 'left'
    ] # 標準位置を指定
    rotations   = [0, 30, 0, 30] # 表示角度を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha, r in zip(labels, halignments, rotations):
        label.set_ha(ha)
        label.set_rotation(r)

    ##### σ to σ2の作図 -----

    # 信頼区間のラベルを作成
    adapt_param_lbl  = '$\\sigma_{pop}^2 = '+f'{sigma2_pop:.2f}, '
    adapt_param_lbl += '\\hat{\\sigma}_{obs}^2 = '+f'{sigma2_hat:.2f}$\n'
    adapt_param_lbl += '$L = \\frac{(N-1) \\hat{\\sigma}_{obs}^2}{\\chi^2_{\\frac{\\alpha}{2}}} = '+f'{ci_bound_lower:.2f}, '
    adapt_param_lbl += 'U = \\frac{(N-1) \\hat{\\sigma}_{obs}^2}{\\chi^2_{1-\\frac{\\alpha}{2}}} = '+f'{ci_bound_upper:.2f}$'

    # 標準偏差と分散の関係を描画
    ax   = axes[1, 0]
    ax2x = axes2x[1]
    ax2y = axes2y[1]

    ax.scatter(
        x=np.sqrt(sigma2_hat), y=sigma2_hat, 
        color='#00A968', s=100, 
        zorder=9
    ) # 不偏分散
    ax.plot(
        sigma_vec, sigma_vec**2, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 変換曲線
    for idx, sgm in enumerate([sigma_pop, sigma_hat]):
        ax.vlines(
            x=sgm, ymin=sgm**2, ymax=sigma2_max, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population parameters', 'sample parameters'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本標準偏差
        ax.hlines(
            y=sgm**2, xmin=sgm, xmax=sigma_max, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
    for idx, sgm2 in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.vlines(
            x=np.sqrt(sgm2), ymin=sgm2, ymax=sigma2_max, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
        ax.hlines(
            y=sgm2, xmin=np.sqrt(sgm2), xmax=sigma_max, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    tmp_val = (1.0-margin_ratio) * sigma_max # (表示位置の設定用)
    ax.vlines(
        x=tmp_val, ymin=ci_bound_lower, ymax=ci_bound_upper, 
        color='purple', linewidth=2.0, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\sigma$')
    ax2x.set_xticks(
        ticks =[sigma_pop, sigma_hat, np.sqrt(ci_bound_lower), np.sqrt(ci_bound_upper)], 
        labels=['$\\sigma_{pop}$', '$\\hat{\\sigma}_{obs}$', '$\\sqrt{L}$', '$\\sqrt{U}$'], 
        rotation=60
    ) # パラメータのラベル
    ax.set_ylabel('$\\sigma^2$')
    ax2y.set_yticks(
        ticks =[sigma2_pop, sigma2_hat, ci_bound_lower, ci_bound_upper], 
        labels=['$\\sigma_{pop}^2$', '$\\hat{\\sigma}_{obs}^2$', '$L$', '$U$']
    ) # パラメータのラベル
    ax.set_title(adapt_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=sigma_min, xmax=sigma_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=sigma_min, xmax=sigma_max) # (目盛の共通化用)
    ax.set_ylim(ymin=sigma2_min, ymax=sigma2_max)   # (目盛の共通化用)
    ax2y.set_ylim(ymin=sigma2_min, ymax=sigma2_max) # (目盛の共通化用)

    ##### σ2 to χ2の作図 -----

    # 分散と標準化変数の関係を描画
    ax = axes[1, 1]

    ax.plot(
        (n-1)*sigma2_vec/sigma2_pop, sigma2_vec, 
        color='maroon', linewidth=1.0, 
        label='$f(\\hat{\\sigma}^2) = c \\hat{\\sigma}^2$', 
        zorder=10
    ) # 変換曲線
    ax.plot(
        (n-1)*sigma2_hat/sigma2_vec, sigma2_vec, 
        color='indigo', linewidth=1.0, 
        label='$f(\\sigma_{pop}^2) = c \\frac{1}{\\sigma_{pop}^2}$', 
        zorder=11
    ) # 変換曲線
    for idx, sgm2 in enumerate([sigma2_pop, sigma2_hat]):
        chi2_val = (n-1) * sgm2 / sigma2_pop # スケール変換
        ax.hlines(
            y=sgm2, xmin=chi2_min, xmax=chi2_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
        ax.vlines(
            x=chi2_val, ymin=sigma2_min, ymax=sgm2, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 標準化変数
    for idx, chi2_val in enumerate([cr_bound_lower, cr_bound_upper]):
        sgm2 = (n-1) * sigma2_hat / chi2_val # 逆変換
        ax.hlines(
            y=sgm2, xmin=chi2_min, xmax=chi2_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 中央領域の境界値
        ax.vlines(
            x=chi2_val, ymin=sigma2_min, ymax=sgm2, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 中央領域の境界値

    std_def_lbl = '$\\chi^2 = (n-1) \\hat{\\sigma}^2 \\frac{1}{\\sigma_{pop}^2}, '
    std_def_lbl += '\\chi^2 = \\frac{(n-1)}{\\sigma_{pop}^2} \\hat{\\sigma}^2$'
    ax.set_xlabel(std_def_lbl)
    inv_def_lbl  = '$\\sigma_{pop}^2 = (n-1) \\hat{\\sigma}^2 \\frac{1}{\\chi^2}, '
    inv_def_lbl += '\\hat{\\sigma}^2 = \\frac{\\sigma_{pop}^2}{(n-1)} \\chi^2$'
    ax.set_ylabel(inv_def_lbl)
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=chi2_min, xmax=chi2_max)
    ax.set_ylim(ymin=sigma2_min, ymax=sigma2_max) # 表示範囲を固定

    ##### 標準化分布の作図 -----

    # 標準化分布の確率密度を計算
    std_dens_vec = chi2.pdf(x=chi2_vec, df=n-1)

    # 中央領域を計算
    cr_chi2_vec = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = chi2.pdf(x=cr_chi2_vec, df=n-1)

    # 標準化分布のラベルを作成
    std_param_lbl  = '$\\chi^2_{obs} = '+f'{chi2_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += '\\chi^2_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += '\\chi^2_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[2, 1]
    ax2x = axes2x[2]

    ax.fill_between(
        x=cr_chi2_vec, y1=np.zeros_like(cr_chi2_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=9
    ) # 中央領域
    ax.plot(
        chi2_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='scale-transformed\nstatistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, sgm2 in enumerate([sigma2_pop, sigma2_hat]):
        chi2_val = (n-1) * sgm2 / sigma2_pop # スケール変換
        ax.axvline(
            x=chi2_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
    for idx, chi2_val in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=chi2_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$\\chi^2 = \\frac{(n-1) \\hat{\\sigma}^2}{\\sigma_{pop}^2}$')
    ax2x.set_xticks(
        ticks =[chi2_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$\\chi^2_{obs}$', '$\\chi^2_{1-\\frac{\\alpha}{2}}$', '$\\chi^2_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$chi2(\\chi^2 \\mid n-1)$')
    ax.set_title(std_param_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=chi2_min, xmax=chi2_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=chi2_min, xmax=chi2_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pchi_max, ymax=(1.0+margin_ratio)*Pchi_max) # 余白を追加

    # ラベルの装飾を調整(表示順の変更用)
    order = [1, 2, 0] # 表示順を指定
    handles, labels = ax.get_legend_handles_labels() # 凡例情報を取得
    ax.legend(
        handles=[handles[i] for i in order], 
        labels =[labels[i] for i in order], 
        loc='upper right', prop={'size': 8}
    )

    ##### その他の設定 -----

    # 不要なサブプロットを非表示
    axes[0, 1].axis('off')
    axes[2, 0].axis('off')

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N, interval=100
)

# 動画を書出
anim.save(
    filename=dir_path+'variance_ci_n.mp4', 
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

### 変数の設定 -----

# σ2軸の範囲を設定
k = 4.0
u = 5.0
sigma2_size   = sigma2_pop/(N-1) * np.sqrt(2.0*(N-1)) # 標準偏差
sigma2_size  *= k # 定数倍
sigma2_center = sigma2_pop # 真の値
sigma2_min    = 0.0 # (σ軸との対応用)
sigma2_max    = sigma2_center + sigma2_size
sigma2_max    = np.ceil(sigma2_max /u)*u  # u単位で切り上げ
print('σ2-axis size:', sigma2_min, sigma2_max)

# σ2軸の値を作成
sigma2_vec = np.linspace(start=sigma2_min, stop=sigma2_max, num=1001)[1:] # 正の値を抽出


# χ2軸の範囲を設定
k = 4.0
u = 10.0
chi2_size   = np.sqrt(2.0 * (N-1)) # 標準偏差
chi2_size  *= k # 定数倍
chi2_center = N-1 # 期待値
chi2_min    = np.max([0.0, chi2_center-chi2_size]) # 正の値
chi2_max    = chi2_center + chi2_size
chi2_min    = np.floor(chi2_min /u)*u # u単位で切り下げ
chi2_max    = np.ceil(chi2_max /u)*u  # u単位で切り上げ
print('χ2-axis size:', chi2_min, chi2_max)

# χ2軸の値を作成
chi2_vec = np.linspace(start=chi2_min, stop=chi2_max, num=1001)


# %%

### 信頼区間の設定 -----

# 中央領域の範囲を計算
cr_bound_lower = chi2.ppf(q=0.5*alpha, df=N-1)
cr_bound_upper = chi2.ppf(q=1.0-0.5*alpha, df=N-1)
print('χ2_α/2:', cr_bound_lower, cr_bound_upper)


# %%

### 表示範囲の設定 -----

# 母分布のp軸の範囲を設定
u = 0.15
Px_max = pop_dens_vec.max()
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print('p(x) size:', Px_max)

# 標準化分布のp軸の範囲を設定
u = 0.005
Pchi_max = chi2.pdf(x=np.max([0, N-1-2]), df=N-1) # 最頻値
Pchi_max = np.ceil(Pchi_max /u)*u # u単位で切り上げ
print('p(χ) size:', Pchi_max)


# %%

### 信頼区間の可視化 -----

# 余白を指定
margin_ratio = 0.05

# 図を初期化
fig = plt.figure(
    figsize=(15, 12.5), dpi=100, facecolor='white', 
    constrained_layout=True
)
fig.suptitle('Population Variance Confidence Interval', fontsize=20)

# 図を分割
gs = GridSpec(nrows=3, ncols=3, figure=fig)
ax_00, ax_01 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
ax_10, ax_11 = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
ax_20, ax_21 = fig.add_subplot(gs[2, 0]), fig.add_subplot(gs[2, 1])
ax_a2        = fig.add_subplot(gs[:, 2])
axes   = [ax_00, ax_10, ax_11, ax_21, ax_a2]
axes2x = [ax.twiny() for ax in [ax_00, ax_10, ax_21, ax_a2]]
axes2y = [ax_10.twinx()]

# 不要なサブプロットを非表示
ax_01.axis('off')
ax_20.axis('off')

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
    x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)

    # 標本統計量を計算
    x_bar      = np.mean(x_n)
    sigma2_hat = np.var(x_n, ddof=1)
    sigma_hat  = np.sqrt(sigma2_hat)

    # 標本統計量をスケール変換
    chi2_obs = (N-1) * sigma2_hat / sigma2_pop

    # 信頼区間の範囲を計算
    ci_bound_lower = (N-1) * sigma2_hat / cr_bound_upper
    ci_bound_upper = (N-1) * sigma2_hat / cr_bound_lower

    # 被覆を判定
    cover_flg  = ci_bound_lower <= sigma2_pop <= ci_bound_upper
    cover_cnt += cover_flg # 被覆回数

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
    )

    # 母集団のラベルを作成
    pop_param_lbl  = f'$i = {I}, N = {N}$\n'
    pop_param_lbl += '$\\mu_{pop} = '+f'{mu_pop:.2f}, '
    pop_param_lbl += '\\sigma_{pop} = '+f'{sigma_pop:.2f}$\n'
    pop_param_lbl += '$\\bar{x}_i = '+f'{x_bar:.2f}, '
    pop_param_lbl += '\\hat{\\sigma}_i = '+f'{sigma_hat:.2f}$'

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
    for idx, x in enumerate([mu_pop, mu_pop+sigma_pop, x_bar, x_bar+sigma_hat]):
        ax.axvline(
            x=x, 
            color=['red', 'red', 'black', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 20, 21, 21][idx]
        ) # 母数・標本統計量
    for x in [mu_pop+np.sqrt(ci_bound_lower), mu_pop+np.sqrt(ci_bound_upper)]:
        ax.axvline(
            x=x, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    sgm_dens = norm.pdf(x=mu_pop+sigma_pop, loc=mu_pop, scale=sigma_pop)
    ax.hlines(
        y=sgm_dens, xmin=mu_pop, xmax=mu_pop+sigma_pop, 
        color='red', linewidth=1.0, 
        label='population sd', 
        zorder=31
    ) # 母標準偏差

    ax.set_xlabel('$x$')
    ax2x.set_xticks(
        ticks =[mu_pop, mu_pop+sigma_pop, x_bar, x_bar+sigma_hat], 
        labels=['$\\mu_{pop}$', '$\\mu_{pop} + \\sigma_{pop}$', '$\\bar{x}_i$', '$\\bar{x}_i + \\hat{\\sigma}_i$']
    ) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu_{pop}, \\sigma_{pop}^2)$')
    ax.set_title(pop_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Px_max, ymax=(1.0+margin_ratio)*Px_max) # 余白を追加

    # ラベルの装飾を調整(重なる対策用)
    halignments = [
        'left' if mu_pop >= x_bar else 'right', 
        'left' if mu_pop+sigma_pop >= x_bar+sigma_hat else 'center', 
        'center', 
        'center' if mu_pop+sigma_pop >= x_bar+sigma_hat else 'left'
    ] # 標準位置を指定
    rotations   = [0, 30, 0, 30] # 表示角度を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha, r in zip(labels, halignments, rotations):
        label.set_ha(ha)
        label.set_rotation(r)

    ##### σ to σ2の作図 -----

    # 信頼区間のラベルを作成
    adapt_param_lbl  = '$\\sigma_{pop}^2 = '+f'{sigma2_pop:.2f}, '
    adapt_param_lbl += '\\hat{\\sigma}_i^2 = '+f'{sigma2_hat:.2f}$\n'
    adapt_param_lbl += '$L = \\frac{(N-1) \\hat{\\sigma}_i^2}{\\chi^2_{\\frac{\\alpha}{2}}} = '+f'{ci_bound_lower:.2f}, '
    adapt_param_lbl += 'U = \\frac{(N-1) \\hat{\\sigma}_i^2}{\\chi^2_{1-\\frac{\\alpha}{2}}} = '+f'{ci_bound_upper:.2f}$'

    # 標準偏差と分散の関係を描画
    ax   = axes[1]
    ax2x = axes2x[1]
    ax2y = axes2y[0]

    ax.scatter(
        x=np.sqrt(sigma2_hat), y=sigma2_hat, 
        color='#00A968', s=100, 
        zorder=9
    ) # 不偏分散
    ax.plot(
        sigma_vec, sigma_vec**2, 
        color='black', linewidth=1.0, 
        zorder=10
    ) # 変換曲線
    for idx, sgm in enumerate([sigma_pop, sigma_hat]):
        ax.vlines(
            x=sgm, ymin=sgm**2, ymax=sigma2_max, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            label=['population parameters', 'sample parameters'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本標準偏差
        ax.hlines(
            y=sgm**2, xmin=sgm, xmax=sigma_max, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
    for idx, sgm2 in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.vlines(
            x=np.sqrt(sgm2), ymin=sgm2, ymax=sigma2_max, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
        ax.hlines(
            y=sgm2, xmin=np.sqrt(sgm2), xmax=sigma_max, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 信頼区間の境界値
    tmp_val = (1.0-margin_ratio) * sigma_max # (表示位置の設定用)
    ax.vlines(
        x=tmp_val, ymin=ci_bound_lower, ymax=ci_bound_upper, 
        color='purple' if cover_flg else 'blue', linewidth=2.0, 
        label='confidence interval', 
        zorder=30
    ) # 信頼区間

    ax.set_xlabel('$\\sigma$')
    ax2x.set_xticks(
        ticks =[sigma_pop, sigma_hat, np.sqrt(ci_bound_lower), np.sqrt(ci_bound_upper)], 
        labels=['$\\sigma_{pop}$', '$\\hat{\\sigma}_i$', '$\\sqrt{L_i}$', '$\\sqrt{U_i}$']
    ) # パラメータのラベル
    ax.set_ylabel('$\\sigma^2$')
    ax2y.set_yticks(
        ticks =[sigma2_pop, sigma2_hat, ci_bound_lower, ci_bound_upper], 
        labels=['$\\sigma_{pop}^2$', '$\\hat{\\sigma}_i^2$', '$L_i$', '$U_i$'], 
        rotation=60
    ) # パラメータのラベル
    ax.set_title(adapt_param_lbl, loc='left')
    ax.legend(loc='upper left', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min-mu_pop, xmax=x_max-mu_pop)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=x_min-mu_pop, xmax=x_max-mu_pop) # (目盛の共通化用)
    ax.set_ylim(ymin=sigma2_min, ymax=sigma2_max)   # (目盛の共通化用)
    ax2y.set_ylim(ymin=sigma2_min, ymax=sigma2_max) # (目盛の共通化用)

    # ラベルの装飾を調整(重なる対策用)
    halignments = ['center', 'center', 'right', 'left'] # 標準位置を指定
    rotations   = [60, 60, 0, 0] # 表示角度を指定
    labels      = ax2x.get_xticklabels() # 軸情報を取得
    for label, ha, r in zip(labels, halignments, rotations):
        label.set_ha(ha)
        #label.set_rotation(r)

    ##### σ2 to χ2の作図 -----

    # 分散と標準化変数の関係を描画
    ax = axes[2]

    ax.plot(
        (N-1)*sigma2_vec/sigma2_pop, sigma2_vec, 
        color='maroon', linewidth=1.0, 
        label='$f(\\hat{\\sigma}^2) = c \\hat{\\sigma}^2$', 
        zorder=10
    ) # 変換曲線
    ax.plot(
        (N-1)*sigma2_hat/sigma2_vec, sigma2_vec, 
        color='indigo', linewidth=1.0, 
        label='$f(\\sigma_{pop}^2) = c \\frac{1}{\\sigma_{pop}^2}$', 
        zorder=11
    ) # 変換曲線
    for idx, sgm2 in enumerate([sigma2_pop, sigma2_hat]):
        chi2_val = (N-1) * sgm2 / sigma2_pop # スケール変換
        ax.hlines(
            y=sgm2, xmin=chi2_min, xmax=chi2_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
        ax.vlines(
            x=chi2_val, ymin=sigma2_min, ymax=sgm2, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 標準化変数
    for idx, chi2_val in enumerate([cr_bound_lower, cr_bound_upper]):
        sgm2 = (N-1) * sigma2_hat / chi2_val # 逆変換
        ax.hlines(
            y=sgm2, xmin=chi2_min, xmax=chi2_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 中央領域の境界値
        ax.vlines(
            x=chi2_val, ymin=sigma2_min, ymax=sgm2, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=22
        ) # 中央領域の境界値

    std_def_lbl = '$\\chi^2 = (n-1) \\hat{\\sigma}^2 \\frac{1}{\\sigma_{pop}^2}, '
    std_def_lbl += '\\chi^2 = \\frac{(n-1)}{\\sigma_{pop}^2} \\hat{\\sigma}^2$'
    ax.set_xlabel(std_def_lbl)
    inv_def_lbl  = '$\\sigma_{pop}^2 = (n-1) \\hat{\\sigma}^2 \\frac{1}{\\chi^2}, '
    inv_def_lbl += '\\hat{\\sigma}^2 = \\frac{\\sigma_{pop}^2}{(n-1)} \\chi^2$'
    ax.set_ylabel(inv_def_lbl)
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=chi2_min, xmax=chi2_max)
    ax.set_ylim(ymin=sigma2_min, ymax=sigma2_max) # 表示範囲を固定

    ##### 標準化分布の作図 -----

    # 標準化分布の確率密度を計算
    std_dens_vec = chi2.pdf(x=chi2_vec, df=N-1)

    # 中央領域を計算
    cr_chi2_vec = np.linspace(start=cr_bound_lower, stop=cr_bound_upper, num=501)
    cr_dens_vec = chi2.pdf(x=cr_chi2_vec, df=N-1)

    # 外側領域を計算
    tail_chi2_vec    = np.hstack([
        np.linspace(start=chi2_min, stop=cr_bound_lower, num=251), 
        np.nan, # (塗りつぶしの分割用)
        np.linspace(start=cr_bound_upper, stop=chi2_max, num=251)
    ])
    tail_dens_vec = chi2.pdf(x=tail_chi2_vec, df=N-1)

    # 標準化分布のラベルを作成
    std_param_lbl  = '$\\chi^2_i = '+f'{chi2_obs:.2f}$\n'
    std_param_lbl += f'$\\alpha = {alpha:.2f}, '
    std_param_lbl += '\\chi^2_{1-\\frac{\\alpha}{2}} = '+f'{cr_bound_lower:.2f}, '
    std_param_lbl += '\\chi^2_{\\frac{\\alpha}{2}} = '+f'{cr_bound_upper:.2f}$'

    # 標準化分布を描画
    ax   = axes[3]
    ax2x = axes2x[2]

    ax.fill_between(
        x=cr_chi2_vec, y1=np.zeros_like(cr_chi2_vec), y2=cr_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='central region', 
        zorder=8
    ) # 中央領域
    ax.fill_between(
        x=tail_chi2_vec, y1=np.zeros_like(tail_chi2_vec), y2=tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        label='tail regions', 
        zorder=9
    ) # 外側領域
    ax.plot(
        chi2_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='scale-transformed\nstatistic distribution', 
        zorder=10
    ) # 標準化分布
    for idx, sgm2 in enumerate([sigma2_pop, sigma2_hat]):
        chi2_val = (N-1) * sgm2 / sigma2_pop # スケール変換
        ax.axvline(
            x=chi2_val, 
            color=['red', 'black'][idx], linewidth=1.0, linestyle='--', 
            zorder=[20, 21][idx]
        ) # 母・標本分散
    for idx, chi2_val in enumerate([cr_bound_lower, cr_bound_upper]):
        ax.axvline(
            x=chi2_val, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='central bounds' if idx == 0 else None, 
            zorder=22
        ) # 中央領域の境界値
    ax.hlines(
        y=0.0, xmin=cr_bound_lower, xmax=cr_bound_upper, 
        color='purple', linewidth=2.0, 
        zorder=30
    ) # 中央領域

    ax.set_xlabel('$\\chi^2 = \\frac{(n-1) \\hat{\\sigma}^2}{\\sigma_{pop}^2}$')
    ax2x.set_xticks(
        ticks =[chi2_obs, cr_bound_lower, cr_bound_upper], 
        labels=['$\\chi^2_i$', '$\\chi^2_{1-\\frac{\\alpha}{2}}$', '$\\chi^2_{\\frac{\\alpha}{2}}$']
    ) # 中央領域のラベル
    ax.set_ylabel('$chi2(\\chi^2 \\mid n-1)$')
    ax.set_title(std_param_lbl, loc='left')
    #ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=chi2_min, xmax=chi2_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=chi2_min, xmax=chi2_max) # (目盛の共通化用)
    ax.set_ylim(ymin=-margin_ratio*Pchi_max, ymax=(1.0+margin_ratio)*Pchi_max) # 表示範囲を固定, 余白を追加

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
    ax   = axes[4]
    ax2x = axes2x[3]

    for idx, sgm2 in enumerate([sigma2_pop, sigma2_hat]):
        ax.axvline(
            x=sgm2, 
            color=['red', 'black'][idx], linewidth=[2.0, 1.0][idx], linestyle=['-', '--'][idx], 
            label=['population variance', 'sample variance'][idx], 
            zorder=[20, 21][idx]
        ) # 母・標本分散
    for idx, sgm2 in enumerate([ci_bound_lower, ci_bound_upper]):
        ax.axvline(
            x=sgm2, 
            color='black', linewidth=1.0, linestyle='-.', 
            label='confidence bounds' if idx == 0 else None, 
            zorder=22
        ) # 信頼区間の境界値
    for i in range(I):
        # 過去の推定結果を取得
        [sgm2_lower, sgm2_upper], cover_flg = res_lt[i]
        ax.hlines(
            y=i+1, xmin=sgm2_lower, xmax=sgm2_upper, 
            color='purple' if cover_flg else 'blue', linewidth=2.0, 
            label='confidence interval' if i+1 == I else None, 
            zorder=30
        ) # 信頼区間
    
    ax.set_xlabel('$\\sigma^2$')
    ax2x.set_xticks(
        ticks =[sigma2_pop, sigma2_hat, ci_bound_lower, ci_bound_upper], 
        labels=['$\\sigma_{pop}^2$', '$\\hat{\\sigma}_i^2$', '$L_i$', '$U_i$']
    ) # 信頼区間のラベル
    ax.set_ylabel('iteration')
    ax.set_title(ci_res_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=sigma2_min, xmax=sigma2_max)   # (目盛の共通化用)
    ax2x.set_xlim(xmin=sigma2_min, xmax=sigma2_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1) # 表示範囲を固定
    ax.invert_yaxis() # 推定結果を昇順に表示

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N, interval=500
)

# 動画を書出
anim.save(
    filename=dir_path+'variance_ci_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)


# %%


