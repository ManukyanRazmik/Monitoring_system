import unittest
import yaml
from impact2_engine.Profile.Profile import Profile



class TestProfile(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestProfile, self).__init__(*args, **kwargs)

        with open('./impact2_engine/config/profile_config.yml',
                  mode = 'r', encoding = 'utf-8') as stream:
            self.config = yaml.safe_load(stream)

        self.config['data_path'] = './impact2_engine/data/' + self.config['data_path']


    def test_instance_of_profile_module(self):
        prof = Profile(**self.config)

        self.assertTrue(isinstance(prof, Profile))


    # def test


