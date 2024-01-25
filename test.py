import unittest
#import Model.BandaiDataset 
from Model.BandaiDataset import BandaiDataset
from Model.BandaiDataset import Motion

TEST_VEDIO_DIR = 'test/testfolder/'
TEST_FILE_NAME = 'dataset-1_walk_not-confident_001'
TEST_SET_FRAME = 10

class TestBandaiDataset(unittest.TestCase):
    def setUp(self):
        self.dataset = BandaiDataset()
        
    def test_get_filenames(self):
        self.setUp()
        self.dataset.VIDEO_DIR = 'test/testfolder/'
        outfile = 'test/filenames.txt'
        self.dataset.get_filenames(outfile)

        self.assertEqual(self.dataset.filelist, ["dataset-1_walk_not-confident_001","dataset-1_walk_sad_002"])


if __name__ == '__main__':
    unittest.main()