import torch

class Distribution:
    def __init__(self):
        self.num = 0
        self.accumulated_avg = 0
        self.accumulated_var = 0
        
    def add(self, num, avg, var):
        new_num = self.num + num
        new_avg = (self.num * self.accumulated_avg + num*avg) / new_num
        new_var = (self.num * (self.accumulated_var + ((new_avg - self.accumulated_avg) ** 2)) + num * (var + ((new_avg - avg) ** 2))) / new_num
        self.accumulated_avg = new_avg
        self.accumulated_var = new_var
        self.num = new_num
        
        
class ImageDistribution(Distribution):
    def __init__(self):
        self.num = 0
        self.accumulated_avg = 0
        self.accumulated_var = 0
        
    def add(self, image: torch.Tensor):
        num = 1
        for pixel in image.shape:
            num *= pixel
        avg = image.mean()
        var = image.var(unbiased=False)  
        
        super().add(num, avg, var) 