'''
Created on January 18th  2018
@author: jbi

'''

import unittest
import mock
from pqueens.main import get_options
from pqueens.main import main as queens_main

class TestQUEENSMain(unittest.TestCase):
    def setUp(self):
        self.options = {"method": {"method_name": "monte_carlo",
                                   "method_options":{"seed" : 42,
                                                     "num_samples" : 20,
                                                     "model" : "model"}}}

        self.args = ['--input', 'dummy_no_real_file.json',
                     '--output_dir', '/dummy/path/',
                     '--debug', 'yes']

    @mock.patch('pqueens.iterators.iterator.Iterator.from_config_create_iterator')
    @mock.patch('pqueens.main.get_options')
    def test_main_function(self, mock_parser, mock_iterator):
        mock_parser.return_value = self.options
        queens_main({"dummy_stuff":{"stuff"}})
        mock_iterator.assert_called_with(self.options)
        mock_iterator.return_value.run.assert_called()

    @mock.patch("json.load", return_value={"dummy": "entry"})
    @mock.patch("builtins.open", create=True)
    @mock.patch('os.path.isdir', return_value=True)
    def test_option_parsing_function(self, mock_isdir, mock_open, mock_json):
        get_options(self.args)
        mock_isdir.assert_called_with('/dummy/path')

    @mock.patch('os.path.isdir', return_value=True)
    def test_option_parsing(self, mock_isdir):
        with self.assertRaises(FileNotFoundError):
            get_options(self.args)

    @mock.patch("json.load", return_value={"dummy": "entry"})
    @mock.patch("builtins.open", create=True)
    def test_option_parsing_no_proper_output_dir(self, mock_open, mock_json):
        args = ['--input', '/helper_example_config.json',
                '--debug', 'yes']
        with self.assertRaises(Exception):
            get_options(args)

    @mock.patch("os.path.realpath", return_value="/dummy/path")
    def test_option_parsing_no_proper_input_file(self, mock_realpath):
        with self.assertRaises(Exception):
            get_options(self.args)

    @mock.patch("json.load", return_value={"dummy": "entry"})
    @mock.patch("builtins.open", create=True)
    @mock.patch('os.path.isdir', return_value=True)
    def test_option_parsing_check_proper_debug_flags(self, mock_isdir, mock_open, mock_json):
        options = get_options(self.args)
        self.assertEqual(options['debug'], True, 'Wrong debug flag')
        self.args[-1] = 'no'
        options = get_options(self.args)
        self.assertEqual(options['debug'], False, 'Wrong debug flag')
        self.args = self.args[:len(self.args)-2]
        options = get_options(self.args)
        self.assertEqual(options['debug'], False, 'Wrong debug flag')
