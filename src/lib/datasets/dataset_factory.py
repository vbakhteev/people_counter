from .dataset.jde import JointDataset


def get_dataset(dataset, task):
    if task == 'mot':
        return JointDataset
    else:
        return None
