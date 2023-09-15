# Testing #
## How to run the tests? ##
In short
`python setup.py test`

for more control over the test you can run pytest directly
`pytest`

Every test is marked with a pytest marker. These allow us to group tests together and run every test of a group:
- Unit tests 
  - `pytest -m unit_tests`  
- Integration tests without BACI 
  - `pytest -m integration_tests`
- Integration tests with BACI 
  - `pytest -m integration_tests_baci`
- Integration tests with or without BACI
  - `pytest -m "integration_tests or integration_tests_baci"`
- Cluster tests on LNM clusters (requires LNM clusters access as well as BACI) 
  - `pytest -m lnm_cluster`
- Cluster tests on IMCS clusters (requires IMCS cluster access as well as BACI) 
  - `pytest -m imcs_cluster`   
- Benchmarks (requires BACI)   
  - `pytest -m benchmark`

Note that per default the options defined in  `setup.cfg` under 
`[tool:pytest]` will be added.

##### Some useful options are:  
For verbose output type:  
`pytest -ra -v`

To test for codestyle (PEP8) compatibility type:  
`pytest --codestyle`

You can generate a coverage report with  
`pytest --cov-report=html --cov`   
for a static site that can be found in the `htmlcov` folder.  
Use  
`pytest --cov-report=term --cov`  
for output directly to the shell.

To run tests against the installed package type
`pytest --pyargs model_creator`


## Tools for testing ##
- pytest
- doctest

## Testing philosophy ##
Testing reduces the number of bugs and ensures higher code quality, but it is also a lot of work, therefore:
- focus the tests on central junctions between code sections
- do not write tests to check the validity of assumptions and requirements, use assertions instead
- code coverage can show which parts lack testing, but it does not show completeness or usefulness of testing. Do not take extreme measures only for increasing the coverage, i.e. refactoring code
- keep the test lines of code low. The tests have bugs as well and too many test unnecessarily increase the work in adjusting tests when modifying the code.
- be very careful when adjusting the test results. They are the only reason to test in the first place and should be determined by thinking about the expected outcome, not by running the function to test and copying the output into the expected results
- structure your test according to given-when-then: Given X configuration, when Y is done, then Z should happen
- if there is a bug, write a test to reproduce it. This avoids it reoccuring later on.

## Automated testing / Continous integration ##
The test automation utilizes the pipeline provided by gitlab. It is controlled through the `.gitlab-ci.yml` file, which defines the steps that are taken for a test run.
The CI pipeline is executed on every merge request.

## Examples ##
### pytest ###
Py.test is very flexible and will take any file starting with "test\_" to be part of the test suite. However, to maintain everyones sanity all test files should be placed in the tests folder, retaining the relative path and be named identical to the file containing the tested code, but with the "test\_" prefix.
Furthermore py.test counts every raised exception as a failed test, so you do not have to stick to the provided testing methods, e.g. there is no test array equality, so simply use the `numpy.testing.assert_array_equal`.

In its basic form the tests are written as follows:

    import pytest
    import path.to.module as mymod

    def test_is_earth_round():
        ''' Check if the earth is round. If Pythagoras only knew it was that simple. '''
        #prep testing
        earth = mymod.create_earth('solar system')
        earth.setup_shape()
        #testing
        assert earth.shape == 'round'

Testing for failure:

    def test_planet_detection():
        ''' Test the method to differentiate the planets from the wannabes. '''
        #prep testing
        pluto = mymod.create_dwarf('solar system')
        pluto.setup_shape()
        #testing
        with pytest.raises(Exception) as e_info:
            mymod.assert_is_planet(pluto)

#### fixtures ####
Using fixtures to reduce redundancy in creating data for multiple tests:

    @pytest.fixture
    def lander():
        ''' Create a generic planetary lander object. '''
        return FancyLander(max_speed_kph = 300000)

    @pytest.fixture(scope='module')
    def home_planet():
        ''' Create a planet as a starting base. '''
        return mymod.create_earth('solar system')

    def test_travel_to_mars(home_planet, lander):
        ''' Test traveling to Mars. '''
        #prep
        mars = mymod.create_planet('mars')
        trip = mymod.journey(home_planet, mars, lander)
        #testing
        assert trip.arrived()

These fixtures can also be declared project wide in the `tests/conftest.py` file or package specific in the `tests/package_name/conftest.py` file.

#### mocking ####
The mocking is also done through pytest. It allows arbitrary modules and functions to be mocked from within the test itself. All you need to do is `import pytest` and specify the desired return value for the function call.

    import random

    def test_lottery_win(mocker):
        ''' Test the lottery '''
        mocker.patch.object(random, 'uniform')
        random.uniform.return_value = 14.0

        if random.uniform(0.0, 100.0) == 14.0:
            print('Congratulations! You win one million golden unicorns')
        else:
            print('Bad luck! Try again with a better mocking framework')

Fancier constructions are possible as well, including but not limited to stub, shims, mocking callbacks, spies, etc.

### doctest ###
Doctest allows the integration of tests right in the comment to a class, method or even separately in a tutorial. All you have to do is mimic a regular python shell and your example and a test call will be compared.

    def printMoney(amount, currency):
      ''' This method prints arbitrary amounts of money. It returns the bill count.
          Call it like this:
          >>> prntr = Printer()
          >>> prntr.printMoney(100200, 'dong')
          {1000:100, 500:0, 100:2}
      '''

Note (1): Not yet implemented
