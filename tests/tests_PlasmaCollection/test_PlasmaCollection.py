import unittest
import yaml
from impact2_engine.PlasmaCollection import PlasmaCollection



class TestSafety(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSafety, self).__init__(*args, **kwargs)

        with open('./impact2_engine/config/plasma_config.yml',
                  mode = 'r', encoding = 'utf-8') as stream:
            self.config = yaml.safe_load(stream)

        self.config['data_path'] = './impact2_engine/data/' + self.config['data_path']


    def test_instance_of_plasma_module(self):
        col = PlasmaCollection(**self.config)

        self.assertTrue(isinstance(col, PlasmaCollection))


    # def test


