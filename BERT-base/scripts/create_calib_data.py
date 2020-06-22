train_file_path='/media/drive/Datasets/squad/train-v1.1.json'

with open (train_file_path, 'rt') as myfile:  # Open file lorem.txt for reading text
    for txt_data in myfile:                 # For each line, read it to a string 
        id=txt_data.find('"title"',101385+12)
        new_line= txt_data[:id-3]+txt_data[-20:]
        import pdb; pdb.set_trace()
calib_path= train_file_path.replace('train','calib')        
calib_file = open(calib_path, "wt")
n = calib_file.write(new_line)
calib_file.close()