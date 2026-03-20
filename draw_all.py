from draw.draw_baselines import *
from draw.draw_cross_env import *
from draw.draw_cross_subject import *
from draw.draw_max_predict_len import *
from draw.draw_noramlize_pose import *
from draw.draw_normalization_ablation import *


if __name__ == "__main__":
    plot_baselines(save_path="/mnt/mydata/yh/liming/workspace/future/draw/imgs/Baseline Pose Reconstruction MPJPE Comparison.png")
    plot_cross_scene("/mnt/mydata/yh/liming/workspace/future/draw/imgs/cross_env.png")
    plot_cross_subject("/mnt/mydata/yh/liming/workspace/future/draw/imgs/cross_subject.png")
    plot_prediction_horizon(save_path="/mnt/mydata/yh/liming/workspace/future/draw/imgs/max_predict_len.png")
    plot_normalization_ablation("/mnt/mydata/yh/liming/workspace/future/draw/imgs/normalization_ablation.png")