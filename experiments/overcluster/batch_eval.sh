#!/usr/bin/env bash
python sup_overcluster.py --config_path configs/ade20k-street/dino.yml
python sup_overcluster.py --config_path configs/ade20k-street/exp1.yml
python sup_overcluster.py --config_path configs/ade20k-street/exp4.yml
python sup_overcluster.py --config_path configs/ade20k-street/sup_vit.yml
python sup_overcluster.py --config_path configs/ade20k-street/random_vit.yml
