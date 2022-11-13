import os

from classifiers import (
    EfficientNetB4,
    EfficientNetB5,
    EfficientNetB6,
)


DATA_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), 
    *((os.path.pardir,)*1),
    'datasets',
    'data'
)

BALANCED_SEGMENTED_DATA_PATH = os.path.join(
    DATA_ROOT,
    'balanced-segmented'
)

# EfficientNetB3.experiment_effnetb3(
#     data_path = BALANCED_SEGMENTED_DATA_PATH
# )

# EfficientNetB4.experiment_effnetb4(
#     data_path = BALANCED_SEGMENTED_DATA_PATH
# )

# EfficientNetB5.experiment_effnetb5(
#     data_path = BALANCED_SEGMENTED_DATA_PATH
# )

EfficientNetB6.experiment_effnetb6(
    data_path = BALANCED_SEGMENTED_DATA_PATH
)