__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
from vmaf.routine import run_vmaf_cv, run_vmaf_kfold_cv, prepare_custom_logger

if __name__ == '__main__':

    # run_vmaf_cv(
    #     train_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_train_selection_with_mean_dmos.py'),
    #     test_dataset_filepath=VmafConfig.resource_path('sizzle', 'sizzle_dataset_test_selection_with_mean_dmos.py'),
    #     param_filepath=VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py'),
    #     output_model_filepath=VmafConfig.workspace_path('model', 'cv_first20_meandmos_forest.pkl'),
    #     logger=prepare_custom_logger('vmaf_cv')
    # )
    # DisplayConfig.show(write_to_dir='D:\\vmaf_logs')

    # for count in range(2, 21):
    #     run_vmaf_cv(
    #         train_dataset_filepath=VmafConfig.resource_path('sizzle', 'different_count','datasets_train',f'sizzle_dataset_train_n{count}.py'),
    #         test_dataset_filepath=VmafConfig.resource_path('sizzle', 'different_count','datasets_test',f'sizzle_dataset_test_n{count}.py'),
    #         param_filepath=VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py'),
    #         output_model_filepath=VmafConfig.workspace_path('model', 'cv_first20_meandmos_forest.pkl'),
    #         logger=prepare_custom_logger('vmaf_cv'),
    #         selection_count=count,
    #     )
    # DisplayConfig.show(write_to_dir='D:\\vmaf_logs')


    for count in range(1, 21):
        run_vmaf_cv(
            train_dataset_filepath=VmafConfig.resource_path('sizzle', 'different_count_first20','datasets_train',f'sizzle_dataset_train_n{count}.py'),
            test_dataset_filepath=VmafConfig.resource_path('sizzle', 'different_count_first20','datasets_test',f'sizzle_dataset_test_n{count}.py'),
            param_filepath=VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py'),
            output_model_filepath=VmafConfig.workspace_path('model', 'cv_first20_meandmos_forest.pkl'),
            logger=prepare_custom_logger('vmaf_cv'),
            selection_count=count,
        )
    DisplayConfig.show(write_to_dir='D:\\vmaf_logs')


    # ==== Run cross validation across genres (tough test) ====

    # nflx_dataset_path = VmafConfig.resource_path('dataset', 'NFLX_dataset_public.py')
    # contentid_groups = [
    #     [0, 5],  # cartoon: BigBuckBunny, FoxBird
    #     [1],  # CG: BirdsInCage
    #     [2, 6, 7],  # complex: CrowdRun, OldTownCross, Seeking
    #     [3, 4],  # ElFuente: ElFuente1, ElFuente2
    #     [8],  # sports: Tennis
    # ]
    # param_filepath = VmafConfig.resource_path('param', 'vmaf_v3.py')
    #
    # aggregate_method = np.mean
    # # aggregate_method = ListStats.harmonic_mean
    # # aggregate_method = partial(ListStats.lp_norm, p=2.0)
    #
    # run_vmaf_kfold_cv(
    #     dataset_filepath=nflx_dataset_path,
    #     contentid_groups=contentid_groups,
    #     param_filepath=param_filepath,
    #     aggregate_method=aggregate_method,
    # )
    #
    # DisplayConfig.show()
    #
    # print('Done.')
