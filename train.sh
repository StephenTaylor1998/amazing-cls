
python train.py configs/plain_net/plain_a_analog.py --work-dir /hy-tmp/work/plain_a_analog/ --gpu-id 0
python train.py configs/plain_net/plain_a_digital.py --work-dir /hy-tmp/work/plain_a_digital/ --gpu-id 1
python train.py configs/plain_net/plain_b_analog.py --work-dir /hy-tmp/work/plain_b_analog/ --gpu-id 2
python train.py configs/plain_net/plain_b_digital.py --work-dir /hy-tmp/work/plain_b_digital/ --gpu-id 3
python train.py configs/plain_net/plain_c_analog.py --work-dir /hy-tmp/work/plain_c_analog/ --gpu-id 4
python train.py configs/plain_net/plain_c_digital.py --work-dir /hy-tmp/work/plain_c_digital/ --gpu-id 5


CUDA_VISIBLE_DEVICES=0 python train.py configs/plain_net/plain_a_analog.py --work-dir /hy-tmp/work/plain_a_analog/
CUDA_VISIBLE_DEVICES=1 python train.py configs/plain_net/plain_a_digital.py --work-dir /hy-tmp/work/plain_a_digital/
CUDA_VISIBLE_DEVICES=2 python train.py configs/plain_net/plain_b_analog.py --work-dir /hy-tmp/work/plain_b_analog/
CUDA_VISIBLE_DEVICES=3 python train.py configs/plain_net/plain_b_digital.py --work-dir /hy-tmp/work/plain_b_digital/
CUDA_VISIBLE_DEVICES=0 python train.py configs/plain_net/plain_c_analog.py --work-dir /hy-tmp/work/plain_c_analog/
CUDA_VISIBLE_DEVICES=1 python train.py configs/plain_net/plain_c_digital.py --work-dir /hy-tmp/work/plain_c_digital/


python test.py configs/plain_net/plain_a_analog.py /hy-tmp/work/plain_a_analog/best_accuracy --metrics accuracy
python test.py configs/plain_net/plain_a_digital.py /hy-tmp/work/plain_a_digital/best_accuracy --metrics accuracy
python test.py configs/plain_net/plain_b_analog.py /hy-tmp/work/plain_b_analog/best_accuracy --metrics accuracy
python test.py configs/plain_net/plain_b_digital.py /hy-tmp/work/plain_b_digital/best_accuracy --metrics accuracy
python test.py configs/plain_net/plain_c_analog.py /hy-tmp/work/plain_c_analog/best_accuracy --metrics accuracy
python test.py configs/plain_net/plain_c_digital.py /hy-tmp/work/plain_c_digital/best_accurac --metrics accuracy

