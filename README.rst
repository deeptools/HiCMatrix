HiCMatrix
===========

This library implements the central class of HiCExplorer to manage Hi-C interaction matrices. It is separated from the main project to enable Hi-C matrices
in other projects without the dependency on HiCExplorer. Moreover, it enables us to use the already separated pyGenomeTracks (former hicPlotTADs) in HiCExplorer
because mutual dependencies are resolved.

With version 8, we dropped the support for Python 2.

Version 14 introduced the official support for scool file format, used by scHiCExplorer since version 5: https://github.com/joachimwolff/scHiCExplorer and https://schicexplorer.readthedocs.io/en/latest/.

Read support
-------------

- h5
- cool / mcool / scool
- hicpro
- homer

Write support
--------------

- h5
- cool / mcool
- scool
- homer
- ginteractions
- hicpro

Citation:
^^^^^^^^^

Joachim Wolff, Leily Rabbani, Ralf Gilsbach, Gautier Richard, Thomas Manke, Rolf Backofen, Björn A Grüning.
**Galaxy HiCExplorer 3: a web server for reproducible Hi-C, capture Hi-C and single-cell Hi-C data analysis, quality control and visualization, Nucleic Acids Research**, Volume 48, Issue W1, 02 July 2020, Pages W177–W184, https://doi.org/10.1093/nar/gkaa220
