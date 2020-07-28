import hashlib
import os


def hash_files():
    """
    Hash all files that are used in the singularity image anc check if some files were changed.
    This is important to keep the singularity image always up to date with the code base

    Returns:
        None

    """
    hashlist = []
    hasher = hashlib.md5()
    # hash all drivers
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "../drivers"
    abs_path = os.path.join(script_dir, rel_path)
    elements = os.listdir(abs_path)
    filenames = [
        os.path.join(abs_path, ele) for _, ele in enumerate(elements) if ele.endswith('.py')
    ]
    for filename in filenames:
        with open(filename, 'rb') as inputfile:
            data = inputfile.read()
            hasher.update(data)
        hashlist.append(hasher.hexdigest())

    # hash mongodb
    rel_path = "../database/mongodb.py"
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash utils
    rel_path = '../utils/injector.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    rel_path = '../utils/run_subprocess.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash setup_remote
    rel_path = '../../setup_remote.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash remote_main
    rel_path = '../remote_main.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    # hash postpost files
    rel_path = '../post_post/post_post.py'
    abs_path = os.path.join(script_dir, rel_path)
    with open(abs_path, 'rb') as inputfile:
        data = inputfile.read()
        hasher.update(data)
    hashlist.append(hasher.hexdigest())

    return hashlist
