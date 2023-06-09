import os

dir_path = 'dataset/dop/speaker/output/'  # replace with your directory path

# list all files in directory
files = os.listdir(dir_path)

# sort files alphabetically
files.sort()

for i, file in enumerate(files, start=2000):
    # create new filename
    new_filename = f"{i}.jpg"  # adjust the extension as needed

    # create absolute paths
    old_file_path = os.path.join(dir_path, file)
    new_file_path = os.path.join(dir_path, new_filename)

    # rename file
    os.rename(old_file_path, new_file_path)

print("Files renamed.")
