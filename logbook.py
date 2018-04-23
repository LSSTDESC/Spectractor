from spectractor import *
import parameters
import csv

class LogBook():

    def __init__(self,logbook="ctiofulllogbook_jun2017_v4.csv"):
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.logbook = logbook
        if not os.path.isfile(logbook):
            self.my_logger.error('CSV logbook file %s not found.' % logbook)
            sys.exit()
        self.csvfile =  open(self.logbook,'rU')
        self.reader = csv.DictReader(self.csvfile, delimiter=';', dialect=csv.excel_tab)

    def search_for_image(self,filename):
        target = None
        xpos = None
        ypos = None
        skip = False
        for row in self.reader:
            if filename == row['file']:
                target = row['object']
                if 'bias' in target or 'flat' in target or 'zero' in target:
                    self.my_logger.error('Fits file %s in logbook %s has flag %s. Skip file.' % (filename,self.logbook,target))
                    skip = True
                    break
                if row['skip']=='skip':
                    self.my_logger.error('Fits file %s in logbook has flag "skip". Skip file.' % (filename))
                    skip = True
                    break
                if row['Obj-posXpix']=='':
                    self.my_logger.error('Fits file %s in logbook %s has no target x position. Skip file.' % (filename,self.logbook))
                    skip = True
                    break
                if row['Obj-posYpix']=='':
                    self.my_logger.warning('Fits file %s in logbook %s has no target y position. Skip file.' % (filename,self.logbook))
                    skip = True
                    break
                if row['Dx']!='':
                    parameters.XWINDOW=int(row['Dx'])
                if row['Dy']!='':
                    parameters.YWINDOW=int(row['Dy'])
                xpos = int(row['Obj-posXpix'])
                ypos = int(row['Obj-posYpix'])
                break
        self.csvfile.seek(0)
        if target is None and skip==False:
            self.my_logger.error('Fits file %s not found in logbook %s.' % (filename,self.logbook))
        return target,xpos,ypos

    def plot_columns_vs_date(self,column_names):
        dates = []
        cols = []
        ncols = len(column_names)
        for icol in range(ncols):
            cols.append([])
        for row in self.reader:
            dates.append(row['date'])
            for icol, col in enumerate(column_names):
                cols[icol].append(float(row[col].replace(',','.')))
        fig, ax = plt.subplots(1,len(column_names),figsize=(5*ncols,8))
        for icol, col in enumerate(column_names):
            ax[icol].plot(dates,cols[icol],'b+')
            ax[icol].set_xlabel('Dates')
            ax[icol].set_ylabel(col)
        fig.tight_layout()
        plt.show()
        
        


if __name__ == "__main__":
    import commands, string, re, time, os
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-v", "--verbose", dest="verbose",action="store_true",
                      help="Enter verbose (print more stuff).",default=False)
    (opts, args) = parser.parse_args()

    parameters.VERBOSE = opts.verbose

    logbook = LogBook()
    logbook.plot_columns_vs_date(['Temperature','seeing','PWV (mm)'])
