# CUDA_VISIBLE_DEVICES=0 python main.py --methods GICD_retest --datasets CoCA --save_dir ./Result/Detail --root_dir ../SalMaps

python main.py --methods COEGNET+GICD --datasets CoCA+CoSal2015+CoSOD3k --save_dir ./Result/Detail --root_dir ../SalMaps