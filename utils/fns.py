# PyTorch StudioGAN: https://github.com/POSTECH-CVLab/PyTorch-StudioGAN
# The MIT License (MIT)
# See license file or visit https://github.com/POSTECH-CVLab/PyTorch-StudioGAN for details

# utils/fns.py


def set_temperature(tempering_type, start_temperature, end_temperature, step_count, tempering_step, total_step):
    if tempering_type == 'continuous':
        t = start_temperature + step_count*(end_temperature - start_temperature)/total_step
    elif tempering_type == 'discrete':
        tempering_interval = total_step//(tempering_step + 1)
        t = start_temperature + \
            (step_count//tempering_interval)*(end_temperature-start_temperature)/tempering_step
    else:
        t = start_temperature
    return t
