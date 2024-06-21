import unittest
import yaml
from impact2_engine.Safety.Safety import Safety



class TestSafety(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestSafety, self).__init__(*args, **kwargs)

        with open('./impact2_engine/config/safety_config.yml',
                  mode = 'r', encoding = 'utf-8') as stream:
            self.config = yaml.safe_load(stream)

        self.config['data_path'] = './impact2_engine/data/' + self.config['data_path']


    def test_instance_of_enrolment_module(self):
        saf = Safety(**self.config)

        self.assertTrue(isinstance(saf, Safety))


    # def test


