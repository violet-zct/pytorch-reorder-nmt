#!/usr/bin/env python
# -*- coding: utf-8 -*-

# A Head-Finalization implementation by Sho Hoshino (hoshino@nii.ac.jp)
#
# Hideki Isozaki, Katsuhito Sudoh, Hajime Tsukada, and Kevin Duh
# Head Finalization: a simple reordering rule for SOV languages, WMT 2010
#
# 2013/11/18 Added citation
# 2012/08/29 Fixed some base forms (NUMBER, PREIOD, etc.) are not recognized properly
# 2012/04/13 Fixed a bug which caused the differences in a sentence:
#  John hit the bal but Sam threw the ball .
# 2011? Initial Release

import sys
reload(sys)
sys.setdefaultencoding('utf-8')
import os
from xml.etree.ElementTree import fromstring

def main():
    if len(sys.argv) != 1:
        print "Usage:", sys.argv[0], "<input >output"
        return
    data = sys.stdin.readlines()
    for line in data:
        root = fromstring(line)
        if len(list(root)) < 1:
            print root.text
            continue
        global va0, va1, va2
        va0 = []
        va1 = []
        va2 = []
        head = shead(root, None)
        root, passive = pseudo(root, head)
        gonext(root, passive)
        item = list(root)[0].tail
        if item:
            print item,
        print "\n",

def shead(root, head):
    if head is None:
        head = root.get("head")
    flag = False
    tmp = None
    for child in root:
        if child.get("id") == head:
            head = child.get("head")
            flag = True
        tmp = shead(child, head)
        if tmp:
            return tmp
    if flag:
        return head

def pseudo(root, shead):
    global va0, va2
    passive = 0
    if root.get("cat") == "V":
        arg1 = root.get("arg1")
        arg2 = root.get("arg2")
        if root.get("voice") == "passive":
            passive = 1
            (arg1, arg2) = (arg2, arg1)
        if root.get("id") == shead:
            if arg1 not in va0:
                va0.append(arg1)
        else:
            if arg1 not in va1:
                va1.append(arg1)
        if arg2 not in va2:
            va2.append(arg2)
    for child in root:
        pseudo(child, shead)
    return root, passive

def gonext(parent, passive):
    escape = False
    global va1, va2
    cat = parent.get("cat")
    if cat == "N":
        base = parent.get("base")
        if base and base.count("-") < 2:
            #print base,
            print parent.text,
            escape = True
    if not escape and parent.tag == "tok":
        if parent.text.lower() not in ("a", "an", "the"):
            print parent.text,
        escape = True,
    if not escape:
        data = list(parent)
        if len(data) >= 2:
            if data[-1].get("id") != parent.get("head") and "PN" != cat and parent.get("schema") != "coord_left" and parent.get("schema") != "coord_right":
                data = data[::-1]
        for child in data:
            gonext(child, passive)
    global va0
    if parent.get("id") in va0:
        va0.remove(parent.get("id"))
        print "_va0",
        return
    if parent.get("id") in va1:
        va1.remove(parent.get("id"))
        print "_va1",
        return
    if parent.get("id") in va2:
        va2.remove(parent.get("id"))
        print "_va2",
        return

if __name__ == "__main__":
    main()
