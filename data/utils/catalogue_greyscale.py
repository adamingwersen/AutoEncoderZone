import os
import cv2
import numpy as np
import shutil

class ClassifyGreyscale:
    def __init__(self, source, destination, threshold = 0.69):
        self.source_path = source
        self.destination_path = destination
        self.threshold = threshold
        self.greyscale_color = 255

    def _list_source_files(self):
        self.files = [f for f in os.listdir(self.source_path) if os.path.isfile(os.path.join(self.source_path, f))]

    def _get_image(self, idx):
        self.img_file = self.files[idx]
        self.img = cv2.imread(os.path.join(self.source_path, self.img_file), cv2.IMREAD_GRAYSCALE)
        return(self.img)

    def _move_image(self):
        shutil.move(os.path.join(self.source_path, self.img_file), os.path.join(self.destination_path, self.img_file))

    def _is_unicolor(self, img):
        if (np.sum(img == self.greyscale_color)/(np.sum(img >= 0))) > self.threshold:
            return(True)
        else:
            return(False)

    def classify_all(self):
        self._list_source_files()
        for i in range(len(self.files)):
            self._get_image(i)
            if self._is_unicolor(self.img):
                self._move_image()
            else:
                pass

if __name__ == '__main__':
    cgs = ClassifyGreyscale(source='../../images', 
                            destination='../../classes/other',
                            threshold=0.7)
    cgs.classify_all()