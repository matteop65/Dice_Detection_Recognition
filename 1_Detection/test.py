import os

# path = '/home/User/Documents/Camera_Images'
# print(os.path.dirname(os.path.abspath(__file__)))

def get_parent_dir(n=1):
    """returns the n-th parent dicrectory of the current
    working directory"""
    current_path = os.path.dirname(os.path.abspath(__file__))
    for _ in range(n):
        current_path = os.path.dirname(current_path)
    return current_path

print(get_parent_dir(1))