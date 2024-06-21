import unittest
import yaml
from impact2_engine.Power.Power import Power



class TestPower(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestPower, self).__init__(*args, **kwargs)

        with open('./impact2_engine/config/power_config.yml',
                  mode = 'r', encoding = 'utf-8') as stream:
            self.config = yaml.safe_load(stream)

        self.config['data_path'] = './impact2_engine/data/' + self.config['data_path']


    def test_instance_of_profile_module(self):
        pwr = Power(**self.config)

        self.assertTrue(isinstance(pwr, Power))


    # def test


