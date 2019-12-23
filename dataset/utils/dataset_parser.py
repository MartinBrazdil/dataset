import os
import logging
import argparse


class Base(object):
    def __init__(self, *args, **kwargs):
        pass


class DataParser(Base):
    """
    Dataset base class with argparser, logger, path searching/checking utils.
    All checked file paths and root dir will be automagically registered in args.
    Paths can be also supplied from command line using argparse.

    Example:
    class COCOParser(DataParser):
        def __init__(self, subsets: Set[str], *args, **kwargs):
            super().__init__(*args, **kwargs)

            # Check root dir and store it in args
            self.search_root('/host_home/projects/data/coco')

            # Check paths and store them in args
            self.search_path(['annotations'])
            self.search_path(['instances_val2017', ...],
                             extension='.json', subpath=self.args.annotations)

            # Paths are stored in args and can be used
            self.instances_train2017 = COCO(self.args.instances_train2017)
            ...
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = logging.getLogger(self.__class__.__name__)
        self.log.setLevel(logging.INFO)
        log_handler = logging.StreamHandler()
        log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        log_handler.setFormatter(log_formatter)
        self.log.addHandler(log_handler)

        self.argp = argparse.ArgumentParser()
        self.args = argparse.Namespace()

    def _check_path(self, path, log_level=logging.ERROR):
        if os.path.exists(path):
            self.log.info('Found ' + path)
        else:
            self.log.log(log_level, 'Path not found ' + path)

    def search_root(self, default='.', log_level=logging.ERROR):
        """
        Checks existence of root dir based on provided default path or --root arg.

        Example:
        self.search_root('/host_home/projects/data/coco')

        :param default: Default root path (if there is no --root arg supplied).
        :param log_level: Log level for found/not found messages.
        """
        self.argp.add_argument('--root', default=default, help='Root dir')
        self.argp.parse_known_args(namespace=self.args)
        self.log.info('Searching root --root="{}"'.format(default))
        self._check_path(self.args.root, log_level)

    def search_path(self, files, extension='', subpath='', log_level=logging.ERROR):
        """
        Checks existence of file path in root dir based on provided file paths or arg.

        Example:
        self.search_path(['instances_val2017', ...], extension='.json', subpath=self.args.annotations)

        :param files: List of file paths to check, extension for all files can be supplied in extension arg.
        :param extension: Extension which will be appended to all file paths.
        :param subpath: Subpath in root dir where file paths reside.
        :param log_level: Log level for found/not found messages.
        """
        for file in files:
            file_default = os.path.join(self.args.root, subpath, file + extension)
            file_argname = file.replace('-', '_')
            self.argp.add_argument('--' + file_argname, default=file_default)
            self.argp.parse_known_args(namespace=self.args)
            file_name = getattr(self.args, file_argname)
            self.log.info('Searching file --{}="{}"'.format(file_argname, file_name))
            self._check_path(file_name, log_level=log_level)
