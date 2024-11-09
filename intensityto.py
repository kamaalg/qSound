import random


def intensity_to_color(intensity):
    r, g, b = 0.0, 0.0, 0.0
    if intensity < 0:
        mr = 0.5+-1*intensity/10
        r = random.uniform(0.0, mr)

        mg = 0.5+(1+intensity)/2
        sg = (1+intensity)*4/5
        g = random.uniform(sg, mg)

        mb = .6 + -1*intensity*2/5
        sb = -1*intensity*4/5
        b = random.uniform(sb, mb)
    if intensity > 0:
        mr = intensity/2+0.5
        sr = intensity*4/5
        r = random.uniform(sr, mr)

        mg = 1 - intensity*4/5
        sg = 0.8-intensity*3/5
        g = random.uniform(sg, mg)

        mb = .6-(intensity/3)
        b = random.uniform(0.0, mb)
    return r, g, b


def intensity_to_wave_color(intensity):
    r, g, b = 0.0, 0.0, 0.0
    if intensity < 0:
        r = (0.5+-1*intensity/10)/2

        mg = 0.5+(1+intensity)/2
        sg = (1+intensity)*4/5
        g = (mg+sg)/2

        mb = .6 + -1*intensity*2/5
        sb = -1*intensity*4/5
        b = (sb+mb)/2
    if intensity > 0:
        mr = intensity/2+0.5
        sr = intensity*4/5
        r = (sr+mr)/2

        mg = 1 - intensity*4/5
        sg = 0.8-intensity*3/5
        g = (sg+mg)/2

        b = (.6-(intensity/3))/2

    return r, g, b


def intensity_to_speed(intensity):
    min_v = (intensity+1)/100
    max_v = -min_v
    return min_v, max_v



def intensity_to_radius(intensity):
    min_v = ((intensity-1)/1.5-.3)*4
    max_v = -min_v
    return min_v, max_v
