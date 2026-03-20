# FracLine
__Analysis of fractures along 1D scanlines__

This project represents an evolution of [DomStudioFracStat1D](https://github.com/gecos-lab/DomStudioFracStat1D), translated and refactored in order to be run as a [QGis](https://qgis.org/) script.

FOr a discussion on the theory see [Bistacchi A., Mittempergher S., Martinelli M., Storti F., 2020. On a new robust workflow for the statistical and spatial analysis of fracture data collected with scanlines (or the importance of stationarity), Solid Earth, 11, 2535–2547, 2020, doi: 10.5194/se-11-2535-2020](https://se.copernicus.org/articles/11/2535/2020/).

To install, first install the PackageInstallerQgis plugin by BRGM, available in QGis under Plugins > Manage and Install Plugins. This is used to manage required libraries for this and other plugins. Then clone the FracLine folder in your QGis plugins folder, which under Windows is C:\Users<your user name>\AppData\Roaming\QGIS\QGIS3\profiles\default\python\plugins. Then you will find the qAttitude plugin under Plugins > Manage and Install Plugins and you will be able to activate it.