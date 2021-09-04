# Funcion para crear directorios, es necesario e abspath?
##### se agregó a utils
import os

def create_dir(child_dir_str):

    try:
        pardir = os.path.abspath('static/')
        original_umask  = os.umask(0o000)
        os.makedirs(pardir, exist_ok=True)
        child_dir = os.path.join(pardir,child_dir_str)
        child_dir = os.path.abspath('static/')
        os.makedirs(child_dir, exist_ok=True)
    finally:
        os.umask(original_umask)
        print('static folder created:', os.path.isdir(pardir))
        print('img folder created: ', os.path.isdir(child_dir))

#create_dir('img')

#######################################################