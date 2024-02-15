
# for detection
CUDA_VISIBLE_DEVICES=0 python detect.py --index 0
CUDA_VISIBLE_DEVICES=1 python detect.py --index 1
CUDA_VISIBLE_DEVICES=2 python detect.py --index 2
CUDA_VISIBLE_DEVICES=3 python detect.py --index 3

CUDA_VISIBLE_DEVICES=0 python detect.py --index 4
CUDA_VISIBLE_DEVICES=1 python detect.py --index 5
CUDA_VISIBLE_DEVICES=2 python detect.py --index 6
CUDA_VISIBLE_DEVICES=3 python detect.py --index 7

# for analyzing the detected results
python detect.py --index 10
python detect.py --index 11

# for merge results.
python detect.py