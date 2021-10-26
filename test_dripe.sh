
python -u test.py \
--dataroot ./dataset/dripe --phase test --norm batch --batchSize 1 --resize_or_crop no --BP_input_nc 18 --no_flip --how_many 100000 --pairLst ./dataset/dripe_data/dripe-pairs-test.csv --display_id 0 --which_epoch latest --gpu_ids 0,1 --G_n_downsampling 4 --ngf 64 --model PoseStyleNet --results_dir ./results \
--name dripe_APS --which_model_netG APS --dataset_mode keypoint
