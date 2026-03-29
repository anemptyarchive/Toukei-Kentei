
# 統計的推定 --------------------------------------------------------------------

# ch3.4.1
# 正規分布
# 母平均の区間推定
# 母分散が未知の場合

# %%

# ライブラリを読込
import numpy as np
from scipy.stats import norm, t
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec


# %%

# 母分布の設定 -----------------------------------------------------------

### パラメータの設定 -----

# 母平均を指定
mu_pop = 4.0

# 母分散を指定
sigma2_pop = 4.0

# 母標準偏差を計算
sigma_pop = np.sqrt(sigma2_pop)


# %%

### 変数の設定 -----

# x軸の範囲を設定
k = 2.0
u = 5.0
x_size = sigma_pop # 標準偏差
x_size = np.ceil(x_size /u)*u  # u単位で切り上げ
x_min  = mu_pop - x_size
x_max  = mu_pop + x_size
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

# サンプルサイズ(フレーム数)を指定
N = 100

# サンプルを生成
x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)


# %%

### 信頼区間の設定 -----

# 信頼水準を指定
conf_level = 0.95

# 有意水準を計算
alpha = 1.0 - conf_level
print('α:    ', alpha)

# 臨界値を計算
z_critical = norm.ppf(1.0-0.5*alpha)
print('z_α/2:', z_critical)


# %%

### 信頼区間の可視化 -----

# 母分布のp軸の範囲を設定
u = 0.5
Px_max = np.histogram(
    a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
)[0].max() # 全サンプルの密度
Px_max = np.ceil(Px_max /u)*u # u単位で切り上げ
print(Px_max)

# 標本分布分布のp軸の範囲を設定
u = 0.01
Pm_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最頻値の確率密度
Pm_max = np.ceil(Pm_max /u)*u # u単位で切り上げ
print(Pm_max)

# 標準正規分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print(Pz_max)

# %%

# 図を初期化
fig, axes = plt.subplots(
    nrows=3, ncols=1, constrained_layout=True, 
    figsize=(9, 9), dpi=100, facecolor='white'
)
fig.suptitle('Population Mean Confidence Interval', fontsize=20)
axes2 = [ax.twiny() for ax in axes] # 第2軸の設定用

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(i):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2.cla() for ax2 in axes2]

    ##### 乱数の集計 -----

    # 値を設定
    n = i + 1 # サンプル数

    # 標本平均を計算
    x_bar = np.mean(x_n[:n])

    # 信頼区間を計算
    mu_lower = x_bar - z_critical * sigma_pop / np.sqrt(n)
    mu_upper = x_bar + z_critical * sigma_pop / np.sqrt(n)

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n[:n], bins=class_num, range=(bound_min, bound_max), density=True
    )

    # ラベルを作成
    pop_lbl = f'$n = {n}, \\mu = {mu_pop:.2f}, \\sigma = {sigma_pop:.2f}$'
    
    # 母分布を描画
    ax  = axes[0]
    ax2 = axes2[0]
    ax.bar(
        x=center_vec, height=obs_dens_vec, width=class_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # サンプル
    ax.scatter(
        x=x_n[:n], y=np.tile(0.0, reps=n), 
        color='orange', alpha=0.33, s=25, clip_on=False, 
        zorder=9
    ) # サンプル
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    ax.axvline(
        x=mu_pop, 
        color='red', linewidth=1.0, linestyle='--', 
        label='population mean', 
        zorder=21
    ) # 母平均
    ax.axvline(
        x=x_bar, 
        color='black', linewidth=1.0, linestyle='--', 
        zorder=22
    ) # 標本平均
    for mu in [mu_lower, mu_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=23
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=mu_lower, xmax=mu_upper, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$x$')
    ax2.set_xticks(ticks=[mu_pop], labels=['$\\mu$']) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu, \\sigma^2)$')
    ax.set_title(pop_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Px_max) # 表示範囲を固定

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_pop, scale=sigma_pop/np.sqrt(n))

    # ラベルを作成
    smp_lbl = f'$\\bar{{x}} = {x_bar:.2f}$'
    
    # 標本分布を描画
    ax  = axes[1]
    ax2 = axes2[1]
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
    ax.axvline(
        x=mu_pop, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=21
    ) # 母平均
    ax.axvline(
        x=x_bar, 
        color='black', linewidth=1.0, linestyle='--', 
        label='sample mean', 
        zorder=22
    ) # 標本平均
    for mu in [mu_lower, mu_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=23
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=mu_lower, xmax=mu_upper, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$\\bar{x} = \\frac{1}{n} \\sum_{i=1}^n x_i$')
    ax2.set_xticks(ticks=[x_bar], labels=['$\\bar{x}$']) # 標本のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu, \\frac{\\sigma^2}{n})$')
    ax.set_title(smp_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Pm_max) # 表示範囲を固定

    ##### 信頼区間の作図 -----

    # z軸の範囲を設定
    z_min = (x_min - x_bar) * np.sqrt(n) / sigma_pop
    z_max = (x_max - x_bar) * np.sqrt(n) / sigma_pop

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準正規分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 信頼区間の領域を計算
    conf_z_vec    = np.linspace(start=-z_critical, stop=z_critical, num=1001)
    conf_dens_vec = norm.pdf(x=conf_z_vec, loc=0.0, scale=1.0)
    
    # ラベルを作成
    std_lbl = f'$\\alpha = {alpha:.2f}$'

    # 標準正規分布を描画
    ax  = axes[2]
    ax2 = axes2[2]
    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standard normal distribution', 
        zorder=10
    ) # 標準正規分布
    ax.fill_between(
        x=conf_z_vec, y1=np.zeros_like(conf_z_vec), y2=conf_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='confidence interval', 
        zorder=11
    ) # 信頼区間
    ax.axvline(
        x=0.0, 
        color='black', linewidth=1.0, linestyle='--', 
        zorder=21
    ) # 標本平均
    for z, lbl in zip([-z_critical, z_critical], ['critical value', '']):
        ax.axvline(
            x=z, 
            color='black', linewidth=1.0, linestyle='-.', 
            label=lbl, 
            zorder=22
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=-z_critical, xmax=z_critical, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$z = (x - \\mu) \\frac{\\sqrt{n}}{\\sigma}$')
    ax2.set_xticks(
        ticks =[-z_critical, z_critical], 
        labels=[f'${-z_critical:.2f}$', f'${z_critical:.2f}$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Pz_max) # 表示範囲を固定

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=N, interval=100
)

# 動画を書出
anim.save(
    filename='../../figure/conf_mean_n.mp4', 
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

# 信頼水準を指定
conf_level = 0.95

# 有意水準を計算
alpha = 1.0 - conf_level
print('α:    ', alpha)

# 臨界値を計算
z_critical = norm.ppf(1.0-0.5*alpha)
print('z_α/2:', z_critical)


# %%

### 信頼区間の可視化 -----

# μの表示範囲を設定
k = 3.0
u = 1.0
mu_size  = z_critical * sigma_pop / np.sqrt(N) # 
mu_size *= k # 定数倍
mu_size  = np.ceil(mu_size /u)*u # u単位で切り上げ
mu_min   = mu_pop - mu_size
mu_max   = mu_pop + mu_size
print(mu_min, mu_max)


# 母分布のp軸の範囲を設定
Px_max = 0.5
print(Px_max)

# 標本分布分布のp軸の範囲を設定
u = 0.01
Pm_max = norm.pdf(x=mu_pop, loc=mu_pop, scale=sigma_pop/np.sqrt(N)) # 最頻値の確率密度
Pm_max = np.ceil(Pm_max /u)*u # u単位で切り上げ
print(Pm_max)

# 標準正規分布のp軸の範囲を設定
u = 0.5
Pz_max = norm.pdf(x=0.0, loc=0.0, scale=1.0) # 最頻値の確率密度
Pz_max = np.ceil(Pz_max /u)*u # u単位で切り上げ
print(Pz_max)


# %%

# 図を初期化
fig = plt.figure(
    figsize=(12, 9), dpi=100, facecolor='white', constrained_layout=True
)
fig.suptitle('Population Mean Confidence Interval', fontsize=20)

# 描画領域の分割を設定
gs = GridSpec(nrows=3, ncols=2, figure=fig)
ax_00 = fig.add_subplot(gs[0, 0])
ax_10 = fig.add_subplot(gs[1, 0])
ax_20 = fig.add_subplot(gs[2, 0])
ax_01 = fig.add_subplot(gs[:, 1])
axes  = [ax_00, ax_10, ax_20, ax_01]
axes2 = [ax.twiny() for ax in axes] # 第2軸の設定用

# カウントを初期化
cover_cnt = 0

# 受け皿を初期化
conf_lt = []

# 初期化処理を定義
def init():
    pass

# 作図処理を定義
def update(I):

    # 前フレームのグラフを初期化
    [ax.cla() for ax in axes]
    [ax2.cla() for ax2 in axes2]

    # 値を設定
    I = I + 1 # 試行番号

    ##### 乱数の生成 -----

    # サンプルを生成
    x_n = np.random.normal(loc=mu_pop, scale=sigma_pop, size=N)

    # 標本平均を計算
    x_bar = np.mean(x_n)

    # 信頼区間を計算
    mu_lower = x_bar - z_critical * sigma_pop / np.sqrt(N)
    mu_upper = x_bar + z_critical * sigma_pop / np.sqrt(N)

    ##### 母分布の作図 -----

    # 密度を集計
    obs_dens_vec, _ = np.histogram(
        a=x_n, bins=class_num, range=(bound_min, bound_max), density=True
    )

    # ラベルを作成
    pop_lbl = f'$Iteration = {I}, n = {N}, \\mu = {mu_pop:.2f}, \\sigma = {sigma_pop:.2f}$'
    
    # 母分布を描画
    ax  = axes[0]
    ax2 = axes2[0]
    ax.bar(
        x=center_vec, height=obs_dens_vec, width=class_size, 
        color='#00A968', alpha=0.5, 
        label='sample', 
        zorder=8
    ) # サンプル
    ax.scatter(
        x=x_n, y=np.tile(0.0, reps=N), 
        color='orange', alpha=0.33, s=25, clip_on=False, 
        zorder=9
    ) # サンプル
    ax.plot(
        x_vec, pop_dens_vec, 
        color='black', linewidth=1.0, 
        label='population distribution', 
        zorder=10
    ) # 母分布
    ax.axvline(
        x=mu_pop, 
        color='red', linewidth=1.0, linestyle='--', 
        label='population mean', 
        zorder=21
    ) # 母平均
    ax.axvline(
        x=x_bar, 
        color='black', linewidth=1.0, linestyle='--', 
        zorder=22
    ) # 標本平均
    for mu in [mu_lower, mu_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=23
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=mu_lower, xmax=mu_upper, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$x$')
    ax2.set_xticks(ticks=[mu_pop], labels=['$\\mu$']) # パラメータのラベル
    ax.set_ylabel('$N(x \\mid \\mu, \\sigma^2)$')
    ax.set_title(pop_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Px_max) # 表示範囲を固定

    ##### 標本分布の作図 -----

    # 標本分布の確率密度を計算
    smp_dens_vec = norm.pdf(x=x_vec, loc=mu_pop, scale=sigma_pop/np.sqrt(N))

    # ラベルを作成
    smp_lbl = f'$\\bar{{x}}_i = {x_bar:.2f}$'
    
    # 標本分布を描画
    ax  = axes[1]
    ax2 = axes2[1]
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
    ax.axvline(
        x=mu_pop, 
        color='red', linewidth=1.0, linestyle='--', 
        zorder=21
    ) # 母平均
    ax.axvline(
        x=x_bar, 
        color='black', linewidth=1.0, linestyle='--', 
        label='sample mean', 
        zorder=22
    ) # 標本平均
    for mu in [mu_lower, mu_upper]:
        ax.axvline(
            x=mu, 
            color='black', linewidth=1.0, linestyle='-.', 
            zorder=23
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=mu_lower, xmax=mu_upper, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$\\bar{x} = \\frac{1}{n} \\sum_{j=1}^n x_j$')
    ax2.set_xticks(ticks=[x_bar], labels=['$\\bar{x}_i$']) # 標本のラベル
    ax.set_ylabel('$N(\\bar{x} \\mid \\mu, \\frac{\\sigma^2}{n})$')
    ax.set_title(smp_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=x_min, xmax=x_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=x_min, xmax=x_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Pm_max) # 表示範囲を固定

    ##### 信頼区間の作図 -----

    # z軸の範囲を設定
    z_min = (x_min - x_bar) * np.sqrt(N) / sigma_pop
    z_max = (x_max - x_bar) * np.sqrt(N) / sigma_pop

    # z軸の値を作成
    z_vec = np.linspace(start=z_min, stop=z_max, num=1001)

    # 標準正規分布の確率密度を計算
    std_dens_vec = norm.pdf(x=z_vec, loc=0.0, scale=1.0)

    # 信頼区間の領域を計算
    conf_z_vec    = np.linspace(start=-z_critical, stop=z_critical, num=1001)
    conf_dens_vec = norm.pdf(x=conf_z_vec, loc=0.0, scale=1.0)

    # 区間外の領域を計算
    left_tail_z_vec     = np.linspace(start=z_min, stop=-z_critical, num=201)
    left_tail_dens_vec  = norm.pdf(x=left_tail_z_vec, loc=0.0, scale=1.0)
    right_tail_z_vec    = np.linspace(start=z_critical, stop=z_max, num=201)
    right_tail_dens_vec = norm.pdf(x=right_tail_z_vec, loc=0.0, scale=1.0)
    
    # ラベルを作成
    std_lbl = f'$\\alpha = {alpha:.2f}$'

    # 標準正規分布を描画
    ax  = axes[2]
    ax2 = axes2[2]
    ax.plot(
        z_vec, std_dens_vec, 
        color='black', linewidth=1.0, 
        label='standard normal distribution', 
        zorder=10
    ) # 標準正規分布
    ax.fill_between(
        x=conf_z_vec, y1=np.zeros_like(conf_z_vec), y2=conf_dens_vec, 
        facecolor='purple', alpha=0.5, 
        label='confidence interval', 
        zorder=11
    ) # 信頼区間
    ax.fill_between(
        x=left_tail_z_vec, y1=np.zeros_like(left_tail_z_vec), y2=left_tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        zorder=11
    ) # 区間外
    ax.fill_between(
        x=right_tail_z_vec, y1=np.zeros_like(right_tail_z_vec), y2=right_tail_dens_vec, 
        facecolor='blue', alpha=0.5, 
        zorder=11
    ) # 区間外
    ax.axvline(
        x=0.0, 
        color='black', linewidth=1.0, linestyle='--', 
        zorder=21
    ) # 標本平均
    for z, lbl in zip([-z_critical, z_critical], ['critical value', '']):
        ax.axvline(
            x=z, 
            color='black', linewidth=1.0, linestyle='-.', 
            label=lbl, 
            zorder=22
        ) # 信頼区間
    ax.hlines(
        y=0.0, xmin=-z_critical, xmax=z_critical, 
        color='purple', linewidth=1.0, linestyle='-.', 
        zorder=30
    ) # 信頼区間
    ax.set_xlabel('$z = (x - \\mu) \\frac{\\sqrt{n}}{\\sigma}$')
    ax2.set_xticks(
        ticks =[-z_critical, z_critical], 
        labels=[f'${-z_critical:.2f}$', f'${z_critical:.2f}$']
    ) # 信頼区間のラベル
    ax.set_ylabel('$N(z \\mid 0, 1)$')
    ax.set_title(std_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=z_min, xmax=z_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=z_min, xmax=z_max) # (目盛の共通化用)
    ax.set_ylim(ymax=Pz_max) # 表示範囲を固定

    ##### 試行結果の作図 -----

    # 被覆を判定
    cover_flg  = mu_lower <= mu_pop < mu_upper
    global cover_cnt
    cover_cnt += cover_flg

    # 今回の結果を記録
    conf_lt.append(([mu_lower, mu_upper], cover_flg))

    # ラベルを作成
    res_lbl = f'non-coverage count: {I-cover_cnt}'
    
    # 信頼区間を描画
    ax  = axes[3]
    ax2 = axes2[3]
    ax.axvline(
        x=mu_pop, 
        color='red', linewidth=1.0, 
        label='population mean', 
        zorder=21
    ) # 母平均
    ax.axvline(
        x=x_bar, 
        color='black', linewidth=1.0, 
        label='sample mean', 
        zorder=22
    ) # 標本平均
    for i in range(I):
        # 過去の結果を取得
        [mu_lower, mu_upper], cover_flg = conf_lt[i]
        ax.hlines(
            y=i, xmin=mu_lower, xmax=mu_upper, 
            color='purple' if cover_flg else 'blue', linewidth=1.0, 
            label='confidence interval' if i == I-1 else None, 
            zorder=30
        ) # 信頼区間
    ax.set_xlabel('$x$')
    ax2.set_xticks(
        ticks =[mu_pop, x_bar], 
        labels=['$\\mu$', '$\\bar{x}_i$']
    ) # パラメータのラベル
    ax.set_ylabel('iteration')
    ax.set_title(res_lbl, loc='left')
    ax.legend(loc='upper right', prop={'size': 8})
    ax.grid()
    ax.set_xlim(xmin=mu_min, xmax=mu_max)  # (目盛の共通化用)
    ax2.set_xlim(xmin=mu_min, xmax=mu_max) # (目盛の共通化用)
    ax.set_ylim(ymin=0, ymax=iter_num+1)
    ax.invert_yaxis() # サンプルを昇順に表示

# 動画を作成
anim = FuncAnimation(
    fig=fig, func=update, init_func=init, 
    frames=iter_num, interval=500
)

# 動画を書出
anim.save(
    filename='../../figure/conf_mean_iter.mp4', 
    progress_callback=lambda i, n: print(f'\rframe: {i+1} / {n}', end='', flush=True)
)

# %%


