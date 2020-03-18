import lxml.etree as ET
import os
import sys
import re

def parse_xml(inpath,outpath):
    doc=ET.parse(inpath)
    name=os.path.basename(inpath)
    with open(os.path.join(outpath+'_'+name+'.sentences'),'w',encoding='utf-8') as fparas, \
        open(os.path.join(outpath+'_'+name+'.summary'),'w',encoding='utf-8') as fsum:
        for sent in doc.iter('SENT'):
            text=re.sub(r'\s+',' ',' '.join(sent.itertext()))
            if sent.attrib['ALIGN'] != 'NONE': 
                fsum.write(text)
                fsum.write('\n')
            fparas.write(text)
            fparas.write('\n')

def parse(data_dir, output_dir):
    os.makedirs(data_dir, exist_ok=True)
    for rootpath, dirnames, filenames in os.walk(data_dir):
        for filename in filenames:
            if not filename.endswith('.xml') or filename.startswith('.'):continue
            parse_xml(os.path.join(rootpath,filename), os.path.join(output_dir, os.path.relpath(rootpath, data_dir).replace(os.path.sep,'_')))

def main():
    parse(sys.argv[1], sys.argv[2])

if __name__ =='__main__':main() 