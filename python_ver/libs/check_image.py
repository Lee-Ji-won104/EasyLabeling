import os

def is_image(filename, verbose=False):

    data = open(filename,'rb').read(10)

    # check if file is JPG or JPEG
    if data[:3] == b'\xff\xd8\xff':
        if verbose == True:
             print(filename+" is: JPG/JPEG.")
        return True

    # check if file is PNG
    if data[:8] == b'\x89\x50\x4e\x47\x0d\x0a\x1a\x0a':
        if verbose == True:
             print(filename+" is: PNG.")
        return True

    # check if file is GIF
    if data[:6] in [b'\x47\x49\x46\x38\x37\x61', b'\x47\x49\x46\x38\x39\x61']:
        if verbose == True:
             print(filename+" is: GIF.")
        return True

    print("checking images is complete.")
    return False

def start_check(folder):
    # go through all files in desired folder
    for filename in os.listdir(folder):
        # check if file is actually an image file
        if is_image(folder+filename, verbose=False) == False:
            # if the file is not valid, remove it
            os.remove(os. path. join(folder, filename))