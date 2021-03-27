__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
from vmaf.routine import run_vmaf_cv, run_vmaf_kfold_cv, prepare_custom_logger

if __name__ == '__main__':

    # ==== Run simple cross validation: one training and one testing dataset ====

    # run_vmaf_cv(
    #     train_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_train_first20_with_mean_dmos.py'),
    #     test_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_test_first20_with_mean_dmos.py'),
    #     param_filepath=VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py'),
    #     output_model_filepath=VmafConfig.workspace_path('model', 'cv_first20_meandmos_forest.pkl'),
    #     logger=prepare_custom_logger('vmaf_cv')
    # )
    # DisplayConfig.show(write_to_dir='D:\\vmaf_logs')


    # for selection
    run_vmaf_cv(
        train_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_train_selection_with_mean_dmos.py'),
        test_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_test_selection_with_mean_dmos.py'),
        param_filepath=VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py'),
        output_model_filepath=VmafConfig.workspace_path('model', 'cv_selection_meandmos_forest.pkl'),
        logger=prepare_custom_logger('vmaf_cv_selection')
    )
    DisplayConfig.show(write_to_dir='D:\\vmaf_logs')


