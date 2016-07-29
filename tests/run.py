
import os
import os.path
import string


import collections
import subprocess
import click


basepath = os.path.dirname(os.path.realpath(__file__))

is_rust_file = lambda fname: fname.endswith('.rs')

def get_grassc():
    return os.path.realpath(
        os.path.join(basepath, '..', 'target/debug/grassc'))

def get_sysroot():
    return os.environ['GRASS_SYSROOT']

class OrderedDefaultdict(collections.OrderedDict):
    def __init__(self, *args, **kwargs):
        if not args:
            self.default_factory = None
        else:
            if not (args[0] is None or callable(args[0])):
                raise TypeError('first argument must be callable or None')
            self.default_factory = args[0]
            args = args[1:]
        super(OrderedDefaultdict, self).__init__(*args, **kwargs)

    def __missing__ (self, key):
        if self.default_factory is None:
            raise KeyError(key)
        self[key] = default = self.default_factory()
        return default

class TestSuite(OrderedDefaultdict):

    @classmethod
    def load(cls):
        slice_start = len(basepath) + 1

        self = cls()
        # ignore paths that are directories
        for root, _dirs, files in os.walk(basepath, topdown=False):
            for name in filter(is_rust_file, files):
                path = os.path.join(root, name)
                subtree = path[slice_start:]
                self[subtree].append((subtree, path))

        return self

    def __init__(self):
        OrderedDefaultdict.__init__(self, list)

    def filter_cat(self, queries):
        new_dict = TestSuite()


        for cat, cases in self.iteritems():
            for query in queries:
                if query.lower() in cat.lower():
                    new_dict[cat] = cases
        return new_dict

    def run(self):
        for cat in self.values():
            for name, path in cat:
                run_test(name, path)



def run_test(name, path):
    proc = subprocess.Popen(
        [
            get_grassc(),
            '--sysroot', get_sysroot(),
            '-A', 'warnings',
            path
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    print 'Running', name + ' ',

    while True:
        char = proc.stdout.read(1)
        if not char:
            break
        elif char == '.':
            click.secho('.', fg='green', nl=False)
        else:
            click.secho(char, fg='red', nl=False)
    print



if __name__ == '__main__':
    suite = TestSuite.load()

    suite.run()
