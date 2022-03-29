__author__ = "Abhishek Gunjan"
__copyright__ = "Copyright 2022, Neo"
__license__ = "Open Source with limited edit permission"
__version__ = "0.0.1"
__maintainer__ = "Abhishek Gunjan"
__status__ = "Development"

import pandas as pd


# file_name = "data/mstp_data.xlsx"

@staticmethod
def fetch_service_points(sp_file_path: str) -> list:
    """
    This method read the service points that need to be visited and returns a list of ServicePoints

    :param sp_file_path:
    :return: list of ServicePointInfo

    """
    dfs = pd.ExcelFile(sp_file_path)
    service_points_data = dfs.parse('Leads')
    """
    [{'Billing City': 'outlier', 'Address': '505, Avinashi - Tiruppur Rd, Ram Nagar, Tiruppur, Tamil Nadu 641602', 'Pin': 641602.0}]
    """
    return service_points_data[['Billing City', 'Address', 'Pin']].to_dict(orient='records')


@staticmethod
def fetch_agent_info(agent_file_path: str) -> list:
    """
    This method returns the list of agent info containing base location and current location of the agents.

    :param agent_file_path:
    :return: list of AgentInfo
    """
    dfs = pd.ExcelFile(agent_file_path)
    agent_info_data = dfs.parse('Agent Availability')

    """
    [{'Base City': 'Hyderabad', 'Start City': 'Hyderabad'}, {'Base City': 'Delhi', 'Start City': 'Delhi'}]e
    """
    return agent_info_data[['Base City', 'Start City']].to_dict(orient='records')


