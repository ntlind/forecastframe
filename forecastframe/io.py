"""
Functions to import or export fframes.
"""

import pickle
import os


def save_fframe(self, name="forecastframe.pkl", path=os.getcwd()):
    """
    Save the ForecastFrame as a pickle.

    Parameters
    ----------
    name : string, default "ForecastFrame.pickle"
        The filename to save to.
    path : Path, default os.getcwd()
        The path you want to save the pickle to.
    """
    location = os.path.join(path, name)
    print(f"Saving to {location}...")
    with open(location, "wb") as file:
        pickle.dump(self, file)


def load_fframe(name="ForecastFrame.pkl", path=os.getcwd() + "\\"):
    """
    Load the ForecastFrame from a pickle.

    Parameters
    ----------
    name : string, default "ForecastFrame.pickle"
        The filename to load from.
    path : Path, default os.getcwd()
        The path you want to load the pickle from.
    """
    with (open(path + name, "rb")) as file:
        fframe = pickle.load(file)

    return fframe
