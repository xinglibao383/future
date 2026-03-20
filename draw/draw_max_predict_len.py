import matplotlib.pyplot as plt


"""
    \begin{table}[h]
    \centering
    \caption{最大预测长度实验结果}
    \label{tab:prediction_horizon}
    \begin{tblr}{
        colspec={X[c] X[c]},
    }
    \toprule
    预测长度区间 & MPJPE \\
    \midrule
    {[-15, 0)}   & 56.1941 \\
    {[0, 15)}    & 60.9056 \\
    {[15, 30)}   & 65.6229 \\
    {[30, 45)}   & 77.7804 \\
    {[45, 60)}   & 88.2202 \\
    {[60, 75)}   & 94.8827 \\
    {[75, 90)}   & 101.9103 \\
    {[90, 105)}  & 108.3201 \\
    {[105, 120)} & 113.4193 \\
    {[120, 135)} & 119.1812 \\
    {[135, 150)} & 125.6723 \\
    \bottomrule
    \end{tblr}
    \end{table}
"""


def plot_prediction_horizon(save_path):
    plt.rcParams.update({
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12
    })

    x = [
        -7.5,   # [-15, 0)
        7.5,    # [0, 15)
        22.5,
        37.5,
        52.5,
        67.5,
        82.5,
        97.5,
        112.5,
        127.5,
        142.5
    ]

    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    x = [v / 15 for v in x]
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""
    """注意，在这里将x轴单位转换为秒"""

    y = [
        56.1941,
        60.9056,
        65.6229,
        77.7804,
        88.2202,
        94.8827,
        101.9103,
        108.3201,
        113.4193,
        119.1812,
        125.6723
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(
        x,
        y,
        marker='o',
        linewidth=1.5,
        label="MPJPE"
    )
    plt.xlim(-1.5, 10.5)
    plt.ylim(48, 132)
    plt.xlabel("Prediction Time (s)")
    plt.ylabel("MPJPE")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(save_path, dpi=900, bbox_inches='tight')
    plt.close()
    print(f"已保存到: {save_path}")


if __name__ == "__main__":
    plot_prediction_horizon(save_path="/mnt/mydata/yh/liming/workspace/future/draw/max_predict_len.png")