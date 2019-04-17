import os


def recursive_glob(rootdir='.', suffix=''):
    """Performs recursive glob with given suffix and rootdir
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [os.path.join(looproot, filename)
            for looproot, _, filenames in os.walk(rootdir)
            for filename in filenames if filename.endswith(suffix)]


file_list = recursive_glob(rootdir="/home/huijun/Datasets/Cityscapes/leftImg8bit/deepdrive", suffix='.png')
file_list.sort()
print("> ok !!!")
list_save = os.path.join("deepdrive.lst")

with open(list_save, 'w') as f:
    for idx, file_path in enumerate(file_list):
        print("> Processing {}".format(str(idx)))
        save_path = file_path.replace("/home/huijun/Datasets/Cityscapes/", "")

        f.write(save_path + os.linesep)

