## dictify by Erik Aronesty on StackOverflow
## takes a cElementTree root and returns a dictionary
## https://stackoverflow.com/questions/2148119/how-to-convert-an-xml-string-to-a-dictionary

    ## Usage Example:
    ## --------------
    ## from dictify import dictify
    ## 
    ## tree = cElementTree.parse(xml_path)
    ## root = tree.getroot()
    ## xml_dict = dictify(root)

import xml.etree.ElementTree as ET

from copy import copy

def dictify(r,root=True):
    if root:
        return {r.tag : dictify(r, False)}
    d=copy(r.attrib)
    if r.text:
        d["_text"]=r.text
    for x in r.findall("./*"):
        if x.tag not in d:
            d[x.tag]=[]
        d[x.tag].append(dictify(x,False))
    return d