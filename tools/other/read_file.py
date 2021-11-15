import os

if __name__ == '__main__':
    folder = r"F:\code\pycharm\PCA\Dataset\Test3_9dvf\DVFs"
    save_file = r"F:\code\vs\win_ck\data\file_name.txt"
    files_name = os.listdir(folder)
    with open(save_file, 'w') as f:
        for file_name in files_name:
            f.write(file_name)
            f.write('\n')
