import unittest
import yaml
from impact2_engine.Enrolment.Enrolment import Enrolment



class TestEnrolment(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestEnrolment, self).__init__(*args, **kwargs)

        with open('./impact2_engine/config/config.yaml', 'r') as stream:
            self.config = yaml.safe_load(stream)

        self.config['data_path'] = '../../../derived_data/' + self.config['data_path']


    def test_instance_of_enrolment_module(self):
        enr = Enrolment(**self.config)

        self.assertTrue(isinstance(enr, Enrolment))


    # def test
