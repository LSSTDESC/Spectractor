from spectractor import parameters
from spectractor.config import set_logger

import matplotlib.pyplot as plt
import pandas as pd
import os
import numpy as np


class LogBook:
    """Class to load_image and analyse observation logbook csv files."""

    def __init__(self, logbook="ctiofulllogbook_jun2017_v5.csv"):
        """Load and initialise the logbook

        Parameters
        ----------
        logbook: str
            Path to the logbook. Must be a CSV file.

        Examples
        ----------

        >>> logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
        >>> assert logbook.df is not None
        >>> print(logbook.logbook)
        ./ctiofulllogbook_jun2017_v5.csv
        >>> print(logbook.df['disperser'][:2])
        0    Ron400
        1    Ron400
        Name: disperser, dtype: object
        >>> logbook = LogBook('./log.csv')

        """
        self.my_logger = set_logger(self.__class__.__name__)
        self.logbook = logbook
        if not os.path.isfile(logbook):
            self.my_logger.error('CSV logbook file {} not found.'.format(logbook))
            return
        # self.csvfile = open(self.logbook, 'rU', encoding='latin-1')
        # self.reader = csv.DictReader(self.csvfile, delimiter=';', dialect=csv.excel_tab)
        self.df = pd.read_csv(self.logbook, sep=";", decimal=",", encoding='latin-1', header='infer')
        self.df['date'] = pd.to_datetime(self.df.date)

    def search_for_image(self, filename):
        """
        Look for an image file name in the logbook and load_image properties:

        - Obj-posXpix and Obj-posYpix: the [x0,y0] guessed pixel position in the image
        - Dx and Dy: the x and y windows in pixel to search for the target; set XWINDOW and YWINDOW variables in parameters.py
        - object: the name of the target


        Parameters
        ----------
        filename: str
            the fits image file name (not the path, only the file name.)

        Returns
        -------
        disperser_label: str
            the name of the disperser
        target: str
            the name of the target
        xpos: int
            the x position of the target (in pixel)
        ypos: int
            the y position of the target (in pixel)

        Examples
        --------

        >>> logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
        >>> disperser_label, target, xpos, ypos = logbook.search_for_image("unknown_file.fits")
        >>> print(disperser_label, target, xpos, ypos)
        None None None None
        >>> disperser_label, target, xpos, ypos = logbook.search_for_image("reduc_20170605_028.fits")
        >>> print(disperser_label, target, xpos, ypos)
        HoloPhAg PNG321.0+3.9 814 585
        >>> disperser_label, target, xpos, ypos = logbook.search_for_image("reduc_20170608_119.fits")
        >>> print(disperser_label, target, xpos, ypos)
        None HD205905 None None
        >>> disperser_label, target, xpos, ypos = logbook.search_for_image("reduc_20170630_001.fits")
        >>> print(disperser_label, target, xpos, ypos)
        None bias None None

        """
        disperser_label = None
        target = None
        xpos = None
        ypos = None
        skip = False
        try:
            row = self.df.loc[(self.df['file'] == filename)].iloc[0]
            target = row['object']
            if row['object'] == 'bias' or row['object'] == 'flat' or row['object'] == 'zero':
                self.my_logger.error(
                    'Fits file %s in logbook %s has flag %s. Skip file.' % (filename, self.logbook, target))
                skip = True
            if row['skip'] == 'skip':
                self.my_logger.error('Fits file %s in logbook has flag "skip". Skip file.' % filename)
                skip = True
            if np.isnan(row['Obj-posXpix']):
                self.my_logger.error(
                    'Fits file %s in logbook %s has no target x position. Skip file.' % (filename, self.logbook))
                skip = True
            if np.isnan(row['Obj-posYpix']):
                self.my_logger.error(
                    'Fits file %s in logbook %s has no target y position. Skip file.' % (filename, self.logbook))
                skip = True
            if not np.isnan(row['Dx']):
                parameters.XWINDOW = int(row['Dx'])
                parameters.XWINDOW_ROT = int(row['Dx'])
            if not np.isnan(row['Dy']):
                parameters.YWINDOW = int(row['Dy'])
                parameters.YWINDOW_ROT = int(row['Dy'])
            if not skip:
                xpos = int(row['Obj-posXpix'])
                ypos = int(row['Obj-posYpix'])
                disperser_label = row['disperser']
        except IndexError:
            if target is None and skip is False:
                self.my_logger.error('Fits file %s not found in logbook %s.' % (filename, self.logbook))
        return disperser_label, target, xpos, ypos

    def plot_columns_vs_date(self, column_names):
        """Plot of the column property with respect to the dates.

        Parameters
        ----------
        column_names: list, str
            List of column names to plot versus time from the log book.

        Examples
        --------
        >>> logbook = LogBook('./ctiofulllogbook_jun2017_v5.csv')
        >>> logbook.plot_columns_vs_date(['seeing'])
        >>> logbook.plot_columns_vs_date(['P', 'T'])
        """
        if isinstance(column_names, str):
            column_names = [column_names]
        self.df.plot(x='date', y=column_names)
        if parameters.DISPLAY:
            plt.show()
        if parameters.PdfPages:
            parameters.PdfPages.savefig()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
