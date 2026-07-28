# -*- coding: utf-8 -*-
def classFactory(iface):
    from .fracline_core import FracLinePlugin
    return FracLinePlugin(iface)
