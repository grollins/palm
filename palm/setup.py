from numpy.distutils.misc_util import Configuration

def configuration(parent_package='', top_path=None):
    config = Configuration('palm', parent_package, top_path)
    config.add_subpackage('base')
    config.add_subpackage('network')
    config.add_subpackage('test')
    config.add_data_dir('test/test_data')
    config.add_data_dir('scripts')
    return config

if __name__ == '__main__':
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
