# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/fns.py


def set_temperature(start_temperature, step_count, end_temperature, total_step):
    tempering_range = end_temperature - start_temperature
    if tempering_type == 'discrete':

    if self.tempering_type == 'continuous':
        t = self.start_temperature + step_count*(self.end_temperature - self.start_temperature)/total_step
    elif self.tempering_type == 'discrete':
        tempering_interval = total_step//(tempering_step + 1)
        t = self.start_temperature + \
            (step_count//self.tempering_interval)*(self.end_temperature-self.start_temperature)/self.tempering_step
    else:
        t = self.start_temperature
    return t