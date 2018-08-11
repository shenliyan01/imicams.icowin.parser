# !usr/bin/env python
# coding=utf-8

import xml.sax
import mesh_files
import glob

class MovieHandler(xml.sax.ContentHandler):
    def __init__(self):
        self.CurrentData = ""
        self.type = ""
        self.format = ""
        self.year = ""
        self.rating = ""
        self.stars = ""
        self.description = ""

    # 元素开始事件处理
    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "movie":
            print("*****Movie*****")
            title = attributes["title"]
            print("Title:", title)

    # 元素结束事件处理
    def endElement(self, tag):
        if self.CurrentData == "type":
            print("Type:", self.type)
        elif self.CurrentData == "format":
            print("Format:", self.format)
        elif self.CurrentData == "year":
            print("Year:", self.year)
        elif self.CurrentData == "rating":
            print("Rating:", self.rating)
        elif self.CurrentData == "stars":
            print("Stars:", self.stars)
        elif self.CurrentData == "description":
            print("Description:", self.description)
        self.CurrentData = ""

    # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "type":
            self.type = content
        elif self.CurrentData == "format":
            self.format = content
        elif self.CurrentData == "year":
            self.year = content
        elif self.CurrentData == "rating":
            self.rating = content
        elif self.CurrentData == "stars":
            self.stars = content
        elif self.CurrentData == "description":
            self.description = content

# 临时集合
str_tnamezh = []

class MeshHanlder(xml.sax.ContentHandler):
    def __init__(self):
        self.dui = ""
        self.dname = ""
        self.dnamezh = ""
        self.tui = ""
        self.tname = ""
        self.tnamezh = ""

        # 元素开始事件处理

    def startElement(self, tag, attributes):
        self.CurrentData = tag
        if tag == "mesh_term_2015":
            print("*****mesh_term*****")

            # 元素结束事件处理

    def endElement(self, tag):
        if self.CurrentData == "dui":
            print("dui:", self.dui)
        elif self.CurrentData == "dname":
            print("dname:", self.dname)
        elif self.CurrentData == "dnamezh":
            print("dnamezh:", self.dnamezh)
        elif self.CurrentData == "tui":
            print("tui:", self.tui)
        elif self.CurrentData == "tname":
            print("tname:", self.tname)
        elif self.CurrentData == "tnamezh":
            print("tnamezh:", self.tnamezh)

        self.CurrentData = ""

        # 内容事件处理
    def characters(self, content):
        if self.CurrentData == "dui":
            self.dui = content
        elif self.CurrentData == "dname":
            self.dname = content
        elif self.CurrentData == "dnamezh":
            self.dnamezh = content
        elif self.CurrentData == "tui":
            self.tui = content
        elif self.CurrentData == "tname":
            self.tname = content
        elif self.CurrentData == "tnamezh":
            self.tnamezh = content
            str_tnamezh.append(content)

if (__name__ == "__main__"):
    # 创建一个 XMLReader
    parser = xml.sax.make_parser()

    # turn off namepsaces
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)

    # 重写 ContextHandler
    Handler = MeshHanlder()
    parser.setContentHandler(Handler)

    # 用户词典
    dictionary = []
    dictionary_tmp = []

    files = mesh_files.eachFile("/")

    for file in glob.glob("mesh_*.xml"):
        parser.parse(file)

        # 获得医学术语词典
        for line in str_tnamezh:
            if(line.find(",")!=-1):
                terms = line.split(',')
                for term in terms:
                    dictionary_tmp.append(term.strip())
            else:
                dictionary_tmp.append(line.strip())

    # 合并重复术语
    for i in dictionary_tmp:
        if(i not in dictionary):
            dictionary.append(i)



    # 输出到用户词典 userdict.txt
    print("词典数量: %d" % len(dictionary))
    out_put_path = "/home/alxor/workspaces/Imicams.icowin.parser/corpus/mesh_userdict.txt"
    with open(out_put_path, "wt") as out_file_userdict:
        num = 0
        for word in dictionary:
            word = word.strip()
            print("[%s][%s]"%(num,word))
            out_file_userdict.write(word+"\n")
            num += 1


