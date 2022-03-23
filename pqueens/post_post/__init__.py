"""Post post.

Extract data from simulation output.
"""


def from_config_create_post_post(config, driver_name):
    """Create PostPost object from problem description.

    Args:
        config (dict): input json file with problem description
        driver_name (str): Name of driver that is used in this job-submission

    Returns:
        post_post (obj): post_post object
    """
    from .post_post_csv_data import PostPostCsv
    from .post_post_ensight import PostPostEnsight
    from .post_post_ensight_interface import PostPostEnsightInterfaceDiscrepancy

    post_post_dict = {
        'csv': PostPostCsv,
        'ensight': PostPostEnsight,
        'ensight_interface_discrepancy': PostPostEnsightInterfaceDiscrepancy,
    }

    driver_params = config.get(driver_name)
    if not driver_params:
        raise ValueError(
            "No driver parameters found in problem description! "
            f"You specified the driver name '{driver_name}', "
            "which could not be found in the problem description. Abort..."
        )

    post_post_options = driver_params["driver_params"].get('post_post')
    if not post_post_options:
        raise ValueError(
            f"The 'post_post' options were not found in the driver '{driver_name}'! Abort..."
        )

    post_post_version = post_post_options.get('post_post_approach_sel')
    if not post_post_version:
        raise ValueError(
            "The post_post section did not specify a valid 'post_post_approach_sel'! "
            f"Valid options are {post_post_dict.keys()}. Abort..."
        )

    post_post_class = post_post_dict[post_post_version]
    post_post = post_post_class.from_config_create_post_post(config, driver_name)
    return post_post
