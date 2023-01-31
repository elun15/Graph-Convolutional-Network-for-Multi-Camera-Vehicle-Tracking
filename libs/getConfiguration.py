import yaml
import os

"""
getConfiguration.py
Utils to read and update CONFIG files

Fully developed by Alejandro López-Cifuentes
adapted by Elena Luna
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

            CONFIG = updateConfig(CONFIG, new_key, new_value)


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
