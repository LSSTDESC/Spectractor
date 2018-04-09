# H-alpha filter
HALPHA_CENTER = 655.9e-6 # center of the filter in mm
HALPHA_WIDTH = 6.4e-6 # width of the filter in mm

# Other filters
FGB37 = {'label':'FGB37','min':300,'max':800}
RG715 = {'label':'RG715','min':690,'max':1100}
HALPHA_FILTER = {'label':'Halfa','min':HALPHA_CENTER-2*HALPHA_WIDTH,'max':HALPHA_CENTER+2*HALPHA_WIDTH}
ZGUNN = {'label':'Z-Gunn','min':800,'max':1100}
FILTERS = [RG715,FGB37,HALPHA_FILTER,ZGUNN]


class Filter():

    def __init__(self,wavelength_min,wavelength_max,label):
        self.min = wavelength_min
        self.max = wavelength_max
        self.label = label

