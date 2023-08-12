def CreateDataLoader(opt):
    from data.custom_dataset_data_loader_test import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader
