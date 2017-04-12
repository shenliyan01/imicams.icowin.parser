#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 17-4-11 下午11:07
# @Author  : icowin
# @Site    : 
# @File    : tl_chin_parse_txt.py
# @Software: PyCharm
import xml.sax

meta_data = []

class tl_chin_Hanlder(xml.sax.ContentHandler):
    def __init__(self):
        self.ui = ""
        self.ti = ""
        self.cmab = ""
        self.tw = ""
        self.mh = ""
        self.ct = ""

        # 元素开始事件处理

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "REC":
            print("*****tl_chin_rec*****")

            # 元素结束事件处理

    def endElement(self, tag):
        if self.CurrentData == "UI":
            print("ui:", self.ui)
        elif self.CurrentData == "TI":
            print("ti:", self.ti)
        elif self.CurrentData == "CMAB":
            print("cmab:", self.cmab)
        elif self.CurrentData == "TW":
            print("tw:", self.tw)
        elif self.CurrentData == "MH":
            print("mh:", self.mh)
        elif self.CurrentData == "CT":
            print("ct:", self.ct)

        self.CurrentData = ""

        # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "UI":
            self.ui = content
        elif self.CurrentData == "TI":
            self.ti = content
            meta_data.append(content)
        elif self.CurrentData == "CMAB":
            self.cmab = content
            meta_data.append(content)
        elif self.CurrentData == "TW":
            self.tw = content
        elif self.CurrentData == "MH":
            self.mh = content
        elif self.CurrentData == "CT":
            self.ct = content
            # meta_data.append(content)

if (__name__ == "__main__"):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()

    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = tl_chin_Hanlder()
    parser.setContentHandler(Handler)

    parser.parse("tl_chin_list.xml")

    out_put_path = "/home/alxor/workspaces/Imicams.icowin.parser/corpus/imicams_data/tl_chin.txt"
    with open(out_put_path, "wt") as out_file:
        for line in meta_data:
            print(line+'\n')
            out_file.write(line)
            out_file.write('\n')