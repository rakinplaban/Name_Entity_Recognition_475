import json
import xmltodict
f=open('data.xml',mode='r',encoding='utf-8')
xmlst = f.read()
f.close()
jsonst = json.dumps(xmltodict.parse(xmlst),ensure_ascii=False,indent = 4)
f = open('data.json',mode='w',encoding='utf-8')
f.write(jsonst)
f.close()
