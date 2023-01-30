import yaml
import os

"""
Video Action Classification

getConfiguration.py
Utils to read and update CONFIG files

Fully developed by Alejandro López-Cifuentes
"""


def updateConfig(CONFIG, new_key, new_value):
    """
    Function to replace the value for a given key in CONFIG dictionary.
    Args:
        CONFIG: Full configuration dictionary
        new_key: New key to store the value
        new_value: New value

    Returns: Updated config

    Note: It only supports 3 indent levels in configuration files
    """

    for first_level, second_level in CONFIG.items():
        if type(second_level) is dict:
            for key, value in second_level.items():
                if type(value) is dict:
                    for key2, value2 in value.items():
                        if key2 == new_key:
                            if '[' in new_value:
                                CONFIG[first_level][key][new_key] = new_value.strip('][').split(',')
                            else:
                                CONFIG[first_level][key][new_key] = new_value
                else:
                    if key == new_key:
                        if '[' in new_value:
                            CONFIG[first_level][new_key] = new_value.strip('][').split(',')
                        else:
                            CONFIG[first_level][new_key] = new_value
        else:
            if first_level == new_key:
                if '[' in new_value:
                    CONFIG[new_key] = new_value.strip('][').split(',')
                else:
                    CONFIG[new_key] = new_value

    return CONFIG


def getConfiguration(args):
    """
    Function to join different configuration into one single dictionary.
    Args:
        args:

    Returns: Configuration structure

    """

    # default_CONFIG = yaml.safe_load(open(os.path.join('Config', 'config_default.yaml'), 'r'))

    # Check arguments for default configuration
    # if args.Dataset is None:
    #     args.Dataset = default_CONFIG['DEFAULT']['DATASET']
    # if args.Architecture is None:
    #     args.Architecture = default_CONFIG['DEFAULT']['ARCHITECTURE']
    # if args.Training is None:
    #     args.Training = default_CONFIG['DEFAULT']['TRAINING']
    # if args.Distillation is None:
    #     args.Distillation = default_CONFIG['DEFAULT']['DISTILLATION']

    # ----------------------------- #
    #      Dataset Configuration    #
    # ----------------------------- #
    # dataset_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Dataset', 'config_' + args.Dataset + '.yaml'), 'r'))

    # ----------------------------- #
    #   Architecture Configuration  #
    # ----------------------------- #
    # architecture_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Architecture', 'config_' + args.Architecture + '.yaml'), 'r'))

    # ----------------------------- #
    #    Training Configuration     #
    # ----------------------------- #
    # training_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Training', 'config_' + args.Training + '.yaml'), 'r'))

    # ----------------------------- #
    #   Distillation Configuration  #
    # ----------------------------- #
    # distillation_CONFIG = yaml.safe_load(open(os.path.join('Config', 'Distillation', 'config_' + args.Distillation + '.yaml'), 'r'))

    # ----------------------------- #
    #      MAIN Configuration    #
    # ----------------------------- #
    CONFIG = yaml.safe_load(open(os.path.join('config', 'config_' + args.Mode + '.yaml'), 'r'))

    # ----------------------------- #
    #     Configuration Update      #
    # ----------------------------- #
    # In case there is a configuration update, update configuration
    if args.Options is not None:
        NewOptions = args.Options

        for option in NewOptions:
            new_key, new_value = option.split('=')

            # dataset_CONFIG= updateConfig(dataset_CONFIG, new_key, new_value)
            # architecture_CONFIG = updateConfig(architecture_CONFIG, new_key, new_value)
            # training_CONFIG = updateConfig(training_CONFIG, new_key, new_value)
            # distillation_CONFIG = updateConfig(distillation_CONFIG, new_key, new_value)
            CONFIG = updateConfig(CONFIG, new_key, new_value)

    # ----------------------------- #
    #      Full Configuration       #
    # ----------------------------- #
    # CONFIG = dict()
    # CONFIG.update(dataset_CONFIG)
    # CONFIG.update(architecture_CONFIG)
    # CONFIG.update(training_CONFIG)
    # CONFIG.update(distillation_CONFIG)

    return CONFIG # dataset_CONFIG, architecture_CONFIG, training_CONFIG, distillation_CONFIG


def getValidationConfiguration(Model, ResultsPath='./results'):
    """
    Function to join different configuration into one single dictionary.
    Args:
        args:

    Returns: Configuration structure

    """

    Path = os.path.join(ResultsPath, Model)

    # Search for configuration files in model folder
    folder_files = [f for f in os.listdir(Path) if os.path.isfile(os.path.join(Path, f))]

    # Extract only config files
    config_files = [s for s in folder_files if "config_" in s]

    CONFIG = dict()
    for c in config_files:
        CONFIG.update(yaml.safe_load(open(os.path.join(Path, c), 'r')))

    return CONFIG


def getValidationConfigurationWithOptions(args, ResultsPath='./results'):
    """
    Function to join different configuration into one single dictionary.
    Args:
        args:

    Returns: Configuration structure

    """

    Path = os.path.join(ResultsPath, args.Model)

    # Search for configuration files in model folder
    folder_files = [f for f in os.listdir(Path) if os.path.isfile(os.path.join(Path, f))]

    # Extract only config files
    config_files = [s for s in folder_files if "config_" in s]

    CONFIG = dict()
    for c in config_files:
        CONFIG.update(yaml.safe_load(open(os.path.join(Path, c), 'r')))


    # ----------------------------- #
    #     Configuration Update      #
    # ----------------------------- #
    # In case there is a configuration update, update configuration
    if args.Options is not None:
        NewOptions = args.Options

        for option in NewOptions:
            new_key, new_value = option.split('=')

            CONFIG= updateConfig(CONFIG, new_key, new_value)

    return CONFIG
