__author__ = "Abhishek Gunjan"
__copyright__ = "Copyright 2022, Neo"
__license__ = "Open Source with limited edit permission"
__version__ = "0.0.1"
__maintainer__ = "Abhishek Gunjan"
__status__ = "Development"

import numpy as np
import matplotlib as plt
import random


class AgentInfo:
    """
    AgentInfo defines necessary information about the agent.
    Assumption is that the agent will always go back to its base city after travelling to multiple service points

    """

    def __init__(self, base_city: str, current_city: str):
        """
        Initialize base city and current city of each agent
        :param base_city:
        :param current_city:
        """
        self.base_city = base_city
        self.current_city = current_city

    def get_base_city(self) -> str:
        """ Return the base city of the agent """
        return self.base_city

    def get_current_city(self) -> str:
        """ Return the current city of the agent """
        return self.current_city


class ServicePointInfo:
    """
    ServicePointInfo contains all the necessary information like number of service points in a city, frequency of travel,
    etc. about the service points in the city

    """

    def __init__(self, billing_city: str, address: str, pincode: int):
        self.billing_city = billing_city
        self.pincode = pincode
        self.address = address

    def get_billing_city(self):
        """ Returns the billing city """
        return self.billing_city

    def get_address(self):
        """ Returns the address of the service point """
        return self.address

    def get_pincode(self):
        """ Returns the pincode of the service points """
        return self.pincode


class EA_TSP:
    # Bank parameters
    atms_number = 50  # ATM quantity
    service_centers = 3  # service centers quantity
    velocity = 100  # 100 / hour
    repair_time = 0  # 0.5 hour
    max_engi = 3  # maximum number of engineers in one service center

    # genetic parameters
    population_size = 50  # population size (even number!)
    generations = 1000  # population's generations
    mut_1_prob = 0.4  # prob of replacing together two atms in combined path
    mut_2_prob = 0.6  # prob of reversing the sublist in combined path
    mut_3_prob = 0.8  # probability of changing the length of paths for engineers
    two_opt_search = False  # better convergence, lower speed for large quantity of atms

    # seed
    np.random.seed(2)
    random.seed(1)
    plt.ion()
    engineers = []

    def __init__(self, service_points: list, agent_loc: dict, ):
        pass
