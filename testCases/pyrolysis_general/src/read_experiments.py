import pandas as pd


class ReadExperiments(object):

    def __init__(self, filename=None, folder=None):
            file = pd.read_csv(folder+"/"+filename)
            self.time = file['time']
            self.temperature = file['temperature']
            self.dRho = file['dRho']
            self.Rho = file['rho']