# -*- coding: utf-8 -*-
"""
Copyright (c) 2020-2022 INRAE

Permission is hereby granted, free of charge, to any person obtaining a
copy of this software and associated documentation files (the "Software"),
to deal in the Software without restriction, including without limitation
the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the
Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""
"""Various system operations"""
import logging
import zipfile
import pathlib
import os
import sys
import git

# --------------------------------------------------- Constants --------------------------------------------------------


COMPLETE_SUFFIX = ".complete"


# ---------------------------------------------------- Helpers ---------------------------------------------------------


def get_commit_hash():
    """ Return the git hash of the repository """
    repo = git.Repo(os.path.dirname(os.path.realpath(__file__)), search_parent_directories=True)

    try:
        commit_hash = repo.active_branch.name + "_" + repo.head.object.hexsha[0:5]
    except TypeError:
        commit_hash = 'DETACHED_' + repo.head.object.hexsha[0:5]

    return commit_hash


def get_directories(root):
    """
    List all directories in the root directory
    :param root: root directory
    :return: list of directories
    """
    return [pathify(root) + item for item in os.listdir(root)]


def get_files(directory, ext=None):
    """ List the files in directory, and sort
    :param directory: directory of the image
    :param ext: optional, end of filename to be matched
    :return: list of the filepaths
    """
    ret = []
    for root, _, files in os.walk(directory, topdown=False):
        for name in files:
            filename = os.path.join(root, name)
            if ext:
                if filename.lower().endswith(ext.lower()):
                    ret.append(filename)
            else:
                ret.append(filename)
    return ret


def new_bname(filename, suffix):
    """ return a new basename (without path, without extension, + suffix) """
    filename = filename[filename.rfind("/"):]
    filename = filename[:filename.rfind(".")]
    return filename + "_" + suffix


def pathify(pth):
    """ Adds posix separator if needed """
    if not pth.endswith("/"):
        pth += "/"
    return pth


def mkdir(pth):
    """ Create a directory """
    path = pathlib.Path(pth)
    path.mkdir(parents=True, exist_ok=True)


def dirname(filename):
    """ Returns the parent directory of the file """
    return str(pathlib.Path(filename).parent)


def basename(pth):
    """ Returns the basename. Works with files and paths"""
    return str(pathlib.Path(pth).name)


def join(*pthslist):
    """ Returns the join of all paths"""
    return str(pathlib.PurePath(*pthslist))


def list_files_in_zip(filename, endswith=None):
    """ List files in zip archive
    :param filename: path of the zip
    :param endswith: optional, end of filename to be matched
    :return: list of the filepaths
    """
    with zipfile.ZipFile(filename) as zip_file:
        filelist = zip_file.namelist()
    if endswith:
        filelist = [f for f in filelist if f.endswith(endswith)]

    return filelist


def to_vsizip(zipfn, relpth):
    """ Create path from zip file """
    return "/vsizip/{}/{}".format(zipfn, relpth)


def remove_ext_filename(filename):
    """ Remove OTB extended filenames (keep only the part before the "?") """
    if "?" in filename:
        return filename[:filename.rfind("?")]
    return filename


def declare_complete(filename):
    """ Declare that a file has been completed, creating a small file """
    filename = remove_ext_filename(filename)
    filename += COMPLETE_SUFFIX
    with open(filename, "w") as text_file:
        text_file.write("ok")


def file_exists(filename):
    """ Check if file exists """
    my_file = pathlib.Path(filename)
    return my_file.is_file()


def is_complete(filename):
    """ Returns True if a file has been completed """
    filename = remove_ext_filename(filename)
    filename += COMPLETE_SUFFIX
    return file_exists(filename)


def set_env_var(var, value):
    """ Set an environment variable """
    os.environ[var] = value


def get_env_var(var):
    """ Return an environment variable """
    value = os.environ[var]
    if value is None:
        logging.warning("Environment variable %s is not set. Returning value None.", var)
    return value


def basic_logging_init():
    """ basic logging initialization """
    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')


def logging_info(msg, verbose=True):
    """
    Prints log info only if required by `verbose`
    :param msg: message to log
    :param verbose: boolean. Whether to log msg or not. Default True
    :return:
    """
    if verbose:
        logging.info(msg)


def is_dir(filename):
    """ return True if filename is the path to a directory """
    return os.path.isdir(filename)


def terminate():
    """ Ends the running program """
    sys.exit()


def run_and_terminate(main):
    """Run the main function then ends the running program"""
    sys.exit(main(args=sys.argv[1:]))
