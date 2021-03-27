__copyright__ = "Copyright 2016-2020, Netflix, Inc."
__license__ = "BSD+Patent"

import numpy as np

from vmaf.config import VmafConfig, DisplayConfig
from vmaf.routine import run_vmaf_cv, run_vmaf_kfold_cv, prepare_custom_logger

if __name__ == '__main__':

    # ==== Run cross validation across genres (tough test) ====

    dataset_path = VmafConfig.resource_path('sizzle', 'sizzle_dataset_full_selection_with_mean_dmos.py')
    contentid_groups = [
        [0, 5],  # cartoon: BigBuckBunny, FoxBird
        [1],  # CG: BirdsInCage
        [2, 6, 7],  # complex: CrowdRun, OldTownCross, Seeking
        [3, 4],  # ElFuente: ElFuente1, ElFuente2
        [8],  # sports: Tennis
    ]
    param_filepath = VmafConfig.resource_path('sizzle', 'sizzle_param_cv.py')

    aggregate_method = np.mean
    # aggregate_method = ListStats.harmonic_mean
    # aggregate_method = partial(ListStats.lp_norm, p=2.0)

    run_vmaf_kfold_cv(
        dataset_filepath=dataset_path,
        contentid_groups=contentid_groups,
        param_filepath=param_filepath,
        aggregate_method=aggregate_method,
        logger=prepare_custom_logger('vmaf_kfold'),
    )

    DisplayConfig.show(write_to_dir='D:\\vmaf_logs')
