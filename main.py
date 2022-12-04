"""
Entry point to the Leading Indicator Model.
"""
import argparse
import logging

import CONFIG
from data.parse.input_configs import extract_model_configs
from data.transformations.prep import run_data_transform
from data.extracts.run_data_update import update_model_data
from analytics.run_models import run_all_models


def main():
    logger = logging.getLogger()
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser(description='Model processors.')
    parser.add_argument('date', type=str, help='Date to run analysis for. Must be format YYYY-MM-DD')
    parser.add_argument('process', type=str, choices=CONFIG.CODE_BATCHES, help='Process to run')
    args = parser.parse_args()
    params = extract_model_configs()

    if args.process == 'update_data':
        update_model_data(args.date, params)
    elif args.process == 'run_models':
        x, y, x_new, labels = run_data_transform(params)
        run_all_models(x, y, x_new, labels, params)


if __name__ == "__main__":
    main()
