class Dataset:
    
    filelist = []
    
    def __init__(self):
        pass
    
    def get_filelist(self, list_file=''):
        if list_file == '':
            return filelist
        else:
            with open(list_file,'r') as file:
                line = file.readline()
                filelist = line.split(',')
            return filelist
        
    def cap_frames(filename):
        cap = cv.VideoCapture(video_dir + filename +'.mp4')
        
        motion = []
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break
            img = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            motion.append(img)
            
        cap.release()


    def load_data(list_file):
        Dataset.get_filelist(list_file)
        
        