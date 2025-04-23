class StandardizeData:
    def __init__(self, data,mean,std):
        self.data = data
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        return (self.data-self.mean)/self.std


