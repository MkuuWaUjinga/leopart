# Exemplary call to eval vitb8 fcn head on pvoc
python eval_linear.py --ckpt_path_bb <path_to_ckpt_bb> --ckpt_path_head <path_to_ckpt_head> \
--patch_size 8 --arch vit-base --head_type fcn --batch_size 8 --data_dir /tmp/voc