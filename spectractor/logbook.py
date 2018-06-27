import sys
import os
import matplotlib.pyplot as plt
from . import parameters
import csv


# noinspection PyShadowingNames
class LogBook:
    """Class to load and analyse observation logbook csv files."""

    def __init__(self, logbook="ctiofulllogbook_jun2017_v5.csv"):
        """Load and initialise the logbook

        Parameters
        ----------
        logbook: str
            Path to the logbook. Must be a CSV file.

        Examples
        ----------
        >>> f = open('test.csv','w')
        >>> f.write('43;2017-05-30T03:36:50.655;data_29may17;85;PNG321.3+02.8;dia;'+
        ...     'Ron400;1,106;300;12500;1,283;779;2,7;47;13,1;reduc_20170529_085.fits;895;785;;;'+
        ...     'spectre pollués par les étoiles;skip\n')
        >>> f.close()
        >>> logbook = LogBook('test.csv')
        >>> assert logbook.csvfile is not None
        >>> print(logbook.logbook)
        >>> logbook = LogBook('wrong.csv')
        >>> assert logbook.csvfile is None
        """
        self.my_logger = parameters.set_logger(self.__class__.__name__)
        self.logbook = logbook
        if not os.path.isfile(logbook):
            self.my_logger.error('CSV logbook file {} not found.'.format(logbook))
            sys.exit()
        self.csvfile = open(self.logbook, 'rU', encoding='latin-1')
        self.reader = csv.DictReader(self.csvfile, delimiter=';', dialect=csv.excel_tab)

    def search_for_image(self, filename):
        """Look for an image file name in the logbook and load properties:
        * Obj-posXpix and Obj-posYpix: the [x0,y0] guessed pixel position in the image
        * Dx and Dy: the x and y windows in pixel to search for the target; set XWINDOW and YWINDOW variables
            in parameters.py
        * object: the name of the target

        Args:
            filename (str): the fits image file name (not the path, only the file name.)

        Returns:
            target: the name of the target
            xpos: the x position of the target (in pixel)
            ypos: the y position of the target (in pixel)

        Parameters
        ----------
        filename: str
            the fits image file name (not the path, only the file name.)

        Returns
        -------
        target: str
            the name of the target
        xpos: int
            the x position of the target (in pixel)
        ypos: int
            the y position of the target (in pixel)

        Examples
        --------
        >>> logbook = LogBook('ctio_png+qso_jun2017.csv')
        >>> print(logbook.logbook)
        >>> target, xpos, ypos = logbook.search_for_image('reduc_20170529_085.fits')
        >>> assert target is None
        >>> target, xpos, ypos = logbook.search_for_image('reduc_20170603_020.fits')
        >>> assert xpos is 830
        """
        target = None
        xpos = None
        ypos = None
        skip = False
        for row in self.reader:
            if filename == row['file']:
                target = row['object']
                if 'bias' in target or 'flat' in target or 'zero' in target:
                    self.my_logger.error(
                        'Fits file %s in logbook %s has flag %s. Skip file.' % (filename, self.logbook, target))
                    skip = True
                    break
                if row['skip'] == 'skip':
                    self.my_logger.error('Fits file %s in logbook has flag "skip". Skip file.' % filename)
                    skip = True
                    break
                if row['Obj-posXpix'] == '':
                    self.my_logger.error(
                        'Fits file %s in logbook %s has no target x position. Skip file.' % (filename, self.logbook))
                    skip = True
                    break
                if row['Obj-posYpix'] == '':
                    self.my_logger.warning(
                        'Fits file %s in logbook %s has no target y position. Skip file.' % (filename, self.logbook))
                    skip = True
                    break
                if row['Dx'] != '':
                    parameters.XWINDOW = int(row['Dx'])
                    parameters.XWINDOW_ROT = int(row['Dx'])
                if row['Dy'] != '':
                    parameters.YWINDOW = int(row['Dy'])
                    parameters.YWINDOW_ROT = int(row['Dy'])
                xpos = int(row['Obj-posXpix'])
                ypos = int(row['Obj-posYpix'])
                break
        self.csvfile.seek(0)
        if target is None and skip is False:
            self.my_logger.error('Fits file %s not found in logbook %s.' % (filename, self.logbook))
        return target, xpos, ypos

    def plot_columns_vs_date(self, column_names):
        """Plot of the column property with respect to the dates.

        Args:
            column_names: a list of the names of the columns to plot
        """
        dates = []
        cols = []
        ncols = len(column_names)
        for icol in range(ncols):
            cols.append([])
        for row in self.reader:
            dates.append(row['date'])
            for icol, col in enumerate(column_names):
                cols[icol].append(float(row[col].replace(',', '.')))
        fig, ax = plt.subplots(1, len(column_names), figsize=(5 * ncols, 8))
        for icol, col in enumerate(column_names):
            ax[icol].plot(dates, cols[icol], 'b+')
            ax[icol].set_xlabel('Dates')
            ax[icol].set_ylabel(col)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()


if __name__ == "__main__":
    from optparse import OptionParser

    parser = OptionParser()
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="Enter verbose (print more stuff).", default=False)
    (opts, args) = parser.parse_args()

    parameters.VERBOSE = opts.verbose

    logbook = LogBook()
    logbook.plot_columns_vs_date(['Temperature', 'seeing', 'PWV (mm)'])
