import os
import scipy
import numpy as np
from scipy.optimize import leastsq
home = os.environ['HOME']
from subprocess import Popen,PIPE, STDOUT, call

class UVspec:
    def __init__(self,home=''):
        if home=='':
            self.home=os.environ['HOME']
        else:
            self.home=home
        self.inp = { }

        
    def write_input(self, fn):
        f = open(fn,'w')
        for key in sorted(self.inp):
            if key=="mol_modify2":
                f.write( "mol_modify" + ' ' + str(self.inp[key]) + '\n')
            else:
                f.write( key + ' ' + str(self.inp[key]) + '\n')
            
        f.close()

    def worker(self,num,input,output):
        """thread worker function"""
        verbose = 0
        self.run(input,output,verbose)
        return
            
    def run(self,inp, out, verbose,path=''):
        if verbose:
            print("Running uvspec with input file: ", inp)
            print("Output to file                : ", out)
        if path != '':
            cmd = path+'bin/uvspec '+  ' < ' + inp  +  ' > ' + out
        else:
            cmd = self.home+'/libRadtran/bin/uvspec '+  ' < ' + inp  +  ' > ' + out
        if verbose:
            print("uvspec cmd: ", cmd)
#        p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)
        p   = Popen(cmd,shell=True,stdout=PIPE)
        p.wait()

def peval(x, p):
    return p[0] + p[1]*x + p[2]*x*x  + p[3]*x**3  # + p[4]*x**4 +p[5]*x**5 +p[6]*x**6 +p[7]*x**7 

def curve_fit(x,y):
    A0=0.16
    A1=0.05
    A2=0.005
    A3=1
    A4=1
    A5=1
    A6=1
    A7=1
    p0 = scipy.array([A0, A1, A2, A3])
#    p0 = array([A0, A1, A2, A3, A4])
#    p0 = array([A0, A1, A2, A3, A4, A5])
#    p0 = array([A0, A1, A2, A3, A4, A5])
#    p0 = array([A0, A1, A2, A3, A4, A5,A6,A7])
    plsq = leastsq(residuals, p0, args=(y, x))
#    print plsq[0]
    return plsq[0]

def residuals(p, y, x):
    err = y-peval(x,p)
    return err

def dod(wvl,data_ref,data_obs):

    #    print "We are here", wvl[0],data_obs[0],data_ref[0]
    #    print wvl
    #    print data_obs
    #    print data_ref
    ratio      = scipy.log(data_obs/data_ref)
    coeffs     = curve_fit(wvl, ratio)
    yb         = peval( wvl, coeffs)
    yr         = ratio - yb

    #    print "We are here", yr[0],ratio[0],coeffs[0],yb[0]
    return yr

def get_vals(fn,option):
    """ Returns the values for option in an input file.

        Usage:

        values = get_vals(input_filename,optionname)

        Input:
           filename    uvspec input file
           optionname  name of uvspec option

        Output:
           values      list of option values

        Author: Arve Kylling
        Date:   2011-05-23
    """
    
    f  = open(fn,'r')
    vals = ''
    for line in f:
        l = line.split()
# This does not work with the new input options.....
#        if ( l[0] == option ):  
#            vals = l[1:len(l)]
#        print l, option
        if option in line:                
            nopts = len(option.split())
            vals = l[nopts:len(l)]
#            print l, option, nopts, vals
            break
    f.close()
    return vals

def change_option(fi,fo,option,val):
    """ Returns the values for option in an input file.

        Usage:

        values = get_vals(input_filename,optionname)

        Input:
           filename    uvspec input file
           optionname  name of uvspec option

        Output:
           values      list of option values

        Author: Arve Kylling
        Date:   2011-05-23
    """
    
    fi  = open(fi,'r')
    val_found=False
    vals = ''
    lines=[]
    for line in fi:
        l = line.split()
        if ( l[0] == option ):
            line = option+' '+str(val)+'\n'
            val_found=True
        lines.append(line)

    if not val_found:
        line = option+' '+str(val)+'\n'
        lines.append(line)

    fi.close()
    fo  = open(fo,'w')
    for l in lines:
        fo.write(l)
    fo.close()

def remove_option(fi,fo,option):
    """ Removes option from input file fi, new input file in fo.

        Usage:

        values = get_vals(input_filename,new_input_filename,optionname)

        Input:
           input_filename      uvspec input file
           new_input_filename  new uvspec input file
           optionname          name of uvspec option

        Author: Arve Kylling
        Date:   2012-02-23
    """
    
    fi  = open(fi,'r')
    lines=[]
    for line in fi:
        l = line.split()
        if ( l[0] != option ):
            lines.append(line)

    fi.close()
    fo  = open(fo,'w')
    for l in lines:
        fo.write(l)
    fo.close()

def run(inp, out, verbose):
    if verbose:
        print("Running uvspec with input file: ", inp)
        print("Output to file                : ", out)
        log=out+'_verbose.txt'
    cmd = home+'/libRadtran/bin/uvspec '+  ' < ' + inp  +  ' > ' + out
    #cmd ='('+ home+'/libRadtran/bin/uvspec '+  ' < ' + inp  +  ' > ' + out +')>&'+log
    #print cmd
    #    p   = Popen(cmd,shell=True,stdin=PIPE,stdout=PIPE)
    p   = call(cmd,shell=True,stdin=PIPE,stdout=PIPE)


def mW2photons(wavelength,radiation):
    # Convert from mw m-2 nm -1 to quanta s-1 cm-2 nm -1
    # h*c = 6.6260E-34*2.9979E+08 = 1.986409E-25 Jm, 1 nm = 1.E-09 m.
    # 1 W = 1000 mW, 1 m^2 = 10000 cm^2
    fact       = (wavelength / 1.9864e-16) / (1000*10000.) # Convert from W m-1 nm-1 to quanta s-1 cm-2 nm-1
    radiation  = fact*radiation
    return radiation

def convert_file(fn,conversion):
    fi  = open(fn,'r')
    for line in fi:
        comment = line.find('#')
        if comment == 0:
            continue
        stuff = map(float,line.split())
        wvl = stuff[0]
        print('{0:12.6f} '.format(wvl))
        for val in stuff[1:len(stuff)]:
            if conversion == 'mW2photons':
                val = mW2photons(wvl,val)
            print('{0:12.6e} '.format(val))

        print(' ') 
            
    fi.close()

def read_rad_spc(fn, nx, ny,nrgb):

    RAD = np.zeros((ny,nx,nrgb))
    STD = np.zeros((ny,nx,nrgb))
    f  = open(fn,'r')

    ir = 0
    it = 0
    for line in f:
        ls = line.split()
        ix = int(ls[1])
        iy = int(ls[2])
        RAD[iy,ix,ir] = float(ls[4])
        if  len(ls) > 5:
            STD[iy,ix,ir] = float(ls[5])
            
        if nrgb == 3 and it >= ny*nx:
            ir = ir + 1
            it = 0
        else:
            it = it + 1

    #    print BT

    f.close()

    return RAD,STD

