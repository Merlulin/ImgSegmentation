import os

if __name__ == '__main__':
    root_dir = '../data/test/test/'
    if os.path.exists(root_dir):
        files = os.listdir(root_dir)
        files_number = len(files)
        print(f'test set has images : {files_number}')
        with open('../data/test/test.txt', 'w') as f:
            for file_name in files:
                f.write(file_name + '\n')