from spectractor import *
import csv

class LogBook():

    def __init__(self,logbook="ctiofulllogbook_jun2017_V2.csv"):
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
        ypox = None
        for row in self.reader:
            if filename == row['filename']:
                xpos = int(row['Obj-posXpix'])
                ypos = int(row['Obj-posYpix'])
                target = row['object']
                break
        self.csvfile.seek(0)
        if target is not None:
            return target,xpos,ypos
        else:
            self.my_logger.error('Fits file %s not found in logbook %s.' % (filename,self.logbook))
            sys.exit()

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
