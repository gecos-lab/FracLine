# -*- coding: utf-8 -*-
def classFactory(iface):
    from .FracLine import FracLinePlugin
    return FracLinePlugin(iface)
